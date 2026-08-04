"""Contrato de la curva percentil -> alfa esperado anualizado.

Esta función decide cuánto espera ganar la cartera de cada acción, y por tanto qué se compra, se
vende y se rota. Sus modos de fallo son silenciosos: un alfa plano no lanza ningún error, solo deja
la cartera sin capacidad de discriminar (que es exactamente lo que hacía la calibración isotónica
anterior cuando el ranking se invertía).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from module.evaluation.signal_diagnostics import (
    FALLBACK_ANNUAL_ALPHA,
    VENTILES,
    _annualize_excess,
    alpha_curve_points,
    calibrated_alpha_path,
)


def _panel(
    quarters: int = 20, tickers: int = 200, *, slope: float, start: str = "2015-03-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores y targets sintéticos donde el retorno real depende del rank con la pendiente pedida."""
    dates = pd.date_range(start, periods=quarters, freq="QE")
    horizon = pd.DateOffset(months=12)
    rows, targets = [], []
    for date in dates:
        for index in range(tickers):
            rank = (index + 0.5) / tickers
            snapshot = date.date().isoformat()
            rows.append({
                "ticker": f"T{index}", "snapshot_date": snapshot,
                "meta_rank": rank, "is_quarterly": True,
            })
            targets.append({
                "ticker": f"T{index}", "snapshot_date": snapshot,
                "label_end_date": (date + horizon).date().isoformat(),
                "target_available": True,
                "forward_return": 0.0, "forward_benchmark_return": 0.0,
                "forward_excess_return": slope * (rank - 0.5),
            })
    return pd.DataFrame(rows), pd.DataFrame(targets)


def test_expected_alpha_is_unique_per_rank_not_flat_per_bucket() -> None:
    """Cada acción recibe el alfa de su rank exacto: p99 espera más que p88.

    La recta se estima agrupando en ventiles (para que cada media tenga muestra), pero evaluarla en
    el ventil aplanaría el alfa dentro de cada tramo y crearía saltos en la frontera, justo donde la
    cartera decide a quién desplaza.
    """
    scores, targets = _panel(slope=0.20)
    curve = calibrated_alpha_path(scores, targets, horizon_months=12)
    last = curve.loc[curve["snapshot_date"] == curve["snapshot_date"].max()]
    last = last.merge(scores[["ticker", "snapshot_date", "meta_rank"]], on=["ticker", "snapshot_date"])
    last = last.dropna(subset=["expected_excess_return"]).sort_values("meta_rank")

    assert last["expected_excess_return"].nunique() == len(last) > VENTILES
    assert last["expected_excess_return"].is_monotonic_increasing
    correlation = last[["meta_rank", "expected_excess_return"]].corr(method="spearman").iloc[0, 1]
    assert correlation == pytest.approx(1.0)


def test_decreasing_relationship_falls_through_to_the_fallback_line() -> None:
    """Con la relación invertida, ninguna ventana vale y entra la salvaguarda declarada."""
    scores, targets = _panel(slope=-0.20)
    curve = calibrated_alpha_path(scores, targets, horizon_months=12)
    resolved = curve.dropna(subset=["expected_excess_return"])

    assert set(resolved["alpha_curve_window"]) == {"fallback"}
    # La recta impuesta va de -FALLBACK a +FALLBACK; con rank en (0, 1) no llega a tocar los extremos.
    assert resolved["expected_excess_return"].min() > -FALLBACK_ANNUAL_ALPHA
    assert resolved["expected_excess_return"].max() < FALLBACK_ANNUAL_ALPHA
    assert resolved["expected_excess_return"].max() == pytest.approx(FALLBACK_ANNUAL_ALPHA, rel=0.02)


def test_increasing_relationship_uses_the_most_reactive_window() -> None:
    """Si la ventana más corta ya muestra pendiente creciente, la cascada no amplía la evidencia."""
    scores, targets = _panel(slope=0.20)
    curve = calibrated_alpha_path(scores, targets, horizon_months=12)
    resolved = curve.dropna(subset=["expected_excess_return"])

    assert set(resolved["alpha_curve_window"]) == {"horizon"}
    assert (resolved["alpha_curve_slope"] > 0).all()


def test_expected_alpha_is_nan_until_there_are_closed_cohorts() -> None:
    """Sin evidencia cerrada el alfa es NaN, nunca 0.0.

    `0.0` significa "se espera alfa nulo" y activaría ventas durante todo el arranque; `NaN` significa
    "todavía no hay evidencia" y deja que mande la ordenación.
    """
    scores, targets = _panel(quarters=6, slope=0.20)
    curve = calibrated_alpha_path(scores, targets, horizon_months=12)
    first = curve.loc[curve["snapshot_date"] == curve["snapshot_date"].min()]

    assert first["expected_excess_return"].isna().all()
    assert set(first["alpha_curve_window"]) == {"none"}


def test_windows_of_the_cascade_are_fitted_on_different_evidence() -> None:
    """Las tres ventanas miran distinta cantidad de historia; si coincidieran, la cascada sería inútil."""
    scores, targets = _panel(quarters=24, slope=-0.20)
    windows = alpha_curve_points(scores, targets, horizon_months=12)

    assert windows["horizon"]["cohorts"] < windows["era"]["cohorts"] < windows["history"]["cohorts"]
    for name in ("horizon", "era", "history"):
        assert len(windows[name]["points"]) == VENTILES


def test_excess_return_annualizes_by_compounding_the_horizon() -> None:
    """Un horizonte más corto se compone hasta el año; con 12 meses la conversión es la identidad."""
    values = pd.Series([0.05])

    assert _annualize_excess(values, 12).iloc[0] == pytest.approx(0.05)
    assert _annualize_excess(values, 6).iloc[0] == pytest.approx(1.05**2 - 1)
    assert _annualize_excess(values, 3).iloc[0] == pytest.approx(1.05**4 - 1)
    # Una caída peor que -100 % frente al benchmark no puede producir NaN al elevar a una potencia
    # fraccionaria: se recorta antes de componer.
    assert np.isfinite(_annualize_excess(pd.Series([-1.5]), 6).iloc[0])
