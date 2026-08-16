"""Contrato de la sensibilidad a costes: la curva describe el motor o no describe nada.

Toda la familia de ruta congelada descansa sobre una identidad —``drag = turnover × tasa``— y sobre
que recomponer la curva con otra tasa reproduzca la contabilidad real. Si cualquiera de las dos
falla, el equilibrio publicado es ficción: una cifra con aspecto de resultado que no corresponde a
ninguna cartera simulada. Por eso este contrato prueba la aritmética antes que las conclusiones.

El orden entre familias también es un contrato. La resimulada debe aguantar **al menos** tanto como
la congelada, porque al encarecerse el coste la cartera opera menos y se protege sola. Si saliera al
revés habría un error de signo en los umbrales de decisión, y sería invisible en el agregado.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from module.evaluation.backtest import run_backtest
from module.research.cost_sensitivity import (
    BPS,
    _repriced,
    break_even_bps,
    build_cost_sensitivity,
    frozen_path_curve,
    windowed_metrics,
)
from module.studies.catalog import SELECTION_UNTIL_YEAR
from tests.test_backtest_contract import _panel, _settings


def _result(**overrides: Any):
    scores, prices, benchmark = _panel(months=60)
    settings = _settings(**{
        "commission_bps": 5.0, "slippage_bps": 10.0, "target_size": 3,
        "max_cash_weight": 0.25, "exit_expected_alpha_bps": 250.0, **overrides,
    })
    return run_backtest(scores, prices, benchmark, settings), settings


def test_cost_drag_is_exactly_turnover_times_the_rate() -> None:
    """El supuesto que sostiene la forma cerrada entera. Si falla, la curva es ficción."""
    result, settings = _result()
    rate = (settings.commission_bps + settings.slippage_bps) / BPS
    assert np.allclose(
        result.equity["cost_drag"], result.equity["turnover_pct"] * rate, rtol=0, atol=1e-12,
    )


def test_the_closed_form_reproduces_the_engine_at_the_adopted_cost() -> None:
    """Autoconsistencia: evaluada en el coste real, la reconstrucción **es** la curva original.

    Es lo que prueba que la familia congelada no se ha desviado del motor. Se exige igualdad exacta,
    no aproximada: la reconstrucción repite la misma aritmética, así que cualquier diferencia
    delataría una fórmula distinta y no un error de redondeo.
    """
    result, settings = _result()
    rate = (settings.commission_bps + settings.slippage_bps) / BPS
    rebuilt = _repriced(result.equity, rate)
    for column in ("portfolio_value", "period_start_portfolio_value", "cost_drag", "excess_return"):
        assert np.array_equal(rebuilt[column].to_numpy(), result.equity[column].to_numpy())
    metrics = windowed_metrics(rebuilt, settings)["selection"]
    assert metrics["geometric_excess_return"] == result.summary["geometric_excess_return"]
    assert metrics["information_ratio"] == result.summary["information_ratio"]


def test_gross_beats_standard_and_the_excess_falls_monotonically_with_cost() -> None:
    """Más coste, menos exceso, siempre. Una curva no monótona delataría un error de signo."""
    result, settings = _result()
    rows = frozen_path_curve(result.equity, settings, (0.0, 5.0, 15.0, 50.0, 150.0, 400.0))
    excess = [row["selection_geometric_excess_return"] for row in rows]
    assert excess[0] > excess[2], "el escenario bruto debe superar al estándar"
    assert all(before > after for before, after in pairwise(excess))


def test_the_two_windows_are_computed_separately_and_never_mixed() -> None:
    """La era reservada se reporta aparte; mezclarla con la selección inflaría el equilibrio."""
    result, settings = _result()
    rows = frozen_path_curve(result.equity, settings, (0.0, 15.0))
    years = pd.to_datetime(result.equity["snapshot_date"]).dt.year
    # El panel sintético no alcanza la era reservada, así que su bloque debe venir **vacío** en vez
    # de heredar por descuido las cifras de la ventana de selección.
    assert years.max() <= SELECTION_UNTIL_YEAR
    assert rows[0]["selection_n_periods"] == len(result.equity)
    assert not [key for key in rows[0] if key.startswith("confirmation_")]


def test_a_series_reaching_the_reserved_era_reports_both_windows_apart() -> None:
    """Con serie que cruza a 2025, cada ventana trae sus propios periodos y ninguno se comparte."""
    equity = _equity_across_the_reserved_era()
    settings = _settings(commission_bps=5.0, slippage_bps=10.0)
    rows = frozen_path_curve(equity, settings, (0.0, 50.0))
    selection = rows[0]["selection_n_periods"]
    confirmation = rows[0]["confirmation_n_periods"]
    assert selection == 12 and confirmation == 6
    assert selection + confirmation == len(equity)
    assert rows[0]["selection_geometric_excess_return"] != rows[0]["confirmation_geometric_excess_return"]


def _equity_across_the_reserved_era() -> pd.DataFrame:
    """Curva mínima que abarca 2024 y 2025, para poder separar las dos ventanas."""
    dates = pd.date_range("2024-01-31", periods=18, freq=pd.DateOffset(months=1))
    rng = np.random.default_rng(17)
    rows, value, index = [], 100.0, 100.0
    for stamp in dates:
        gross = float(rng.normal(0.008, 0.03))
        market = float(rng.normal(0.005, 0.02))
        turnover = float(rng.uniform(0.1, 0.5))
        drag = turnover * 15 / BPS
        start, start_index = value, index
        value = start * (1 + gross) * (1 - drag)
        index = start_index * (1 + market)
        rows.append({
            "snapshot_date": str(stamp.date()), "period_start_portfolio_value": start,
            "period_start_benchmark_value": start_index, "portfolio_value": value,
            "benchmark_value": index, "portfolio_return": value / start - 1,
            "benchmark_return": market, "excess_return": value / start - 1 - market,
            "turnover_pct": turnover, "gross_return": gross, "cost_drag": drag,
            "cash_weight": 0.0, "invested_weight": 1.0, "positions_value": value,
            "cumulative_costs": 0.0,
        })
    return pd.DataFrame(rows)


def test_break_even_interpolates_inside_the_ladder_and_declares_when_it_cannot() -> None:
    """El equilibrio se interpola donde se ha medido, y fuera de ahí se dice que no se sabe."""
    rows = [
        {"cost_bps": 0.0, "selection_geometric_excess_return": 0.10},
        {"cost_bps": 100.0, "selection_geometric_excess_return": 0.05},
        {"cost_bps": 200.0, "selection_geometric_excess_return": -0.05},
    ]
    crossing = break_even_bps(rows, "selection")
    assert crossing["available"] is True
    assert crossing["bps_per_trade"] == pytest.approx(150.0)
    assert crossing["round_trip_bps"] == pytest.approx(300.0)
    assert crossing["pct_per_trade"] == pytest.approx(1.5)

    beyond = break_even_bps(rows[:2], "selection")
    assert beyond["available"] is False and beyond["beyond_ladder"] is True

    negative = break_even_bps(
        [{"cost_bps": 0.0, "selection_geometric_excess_return": -0.01}], "selection",
    )
    assert negative["available"] is False and negative["never_positive"] is True


def test_resimulated_holds_at_least_as_much_as_the_frozen_path(tmp_path: Path, monkeypatch) -> None:
    """`c** ≥ c*`: al encarecerse el coste la cartera opera menos, así que aguanta más.

    Se resimula de verdad —con el motor, no con un doble—, porque lo que se comprueba es justamente
    el efecto del coste sobre los umbrales de decisión, que un doble no tendría.
    """
    scores, prices, benchmark = _panel(months=60)
    rng = np.random.default_rng(5)
    scores = scores.copy()
    # Señal con algo de valor real: una cartera sin alfa no tiene margen que agotar y el contraste
    # no distinguiría entre las dos familias.
    scores["meta_rank"] = np.clip(rng.normal(0.5, 0.25, len(scores)), 0, 1)
    scores["meta_score"] = scores["meta_rank"]
    scores["expected_excess_return"] = scores["meta_rank"] * 0.20 - 0.05

    evidence, prepared = tmp_path / "evidence", tmp_path / "prepared"
    evidence.mkdir()
    prepared.mkdir()
    settings = _settings(commission_bps=5.0, slippage_bps=10.0, target_size=3,
                         max_cash_weight=0.25, exit_expected_alpha_bps=100.0)
    result = run_backtest(scores, prices, benchmark, settings)
    result.equity.to_parquet(evidence / "equity.parquet")

    ladder = (0.0, 25.0, 100.0, 300.0)
    frozen = frozen_path_curve(result.equity, settings, ladder)
    frozen_crossing = break_even_bps(frozen, "selection")

    def fake_evaluation(values, profile, evidence_dir, retain_dir=None, **kwargs):
        runtime = _settings(
            commission_bps=values["commission_bps"], slippage_bps=values["slippage_bps"],
            target_size=3, max_cash_weight=0.25, exit_expected_alpha_bps=100.0,
        )
        simulated = run_backtest(scores, prices, benchmark, runtime)
        return {"summary": simulated.summary}

    monkeypatch.setattr("module.studies.runner.run_profile_evaluation", fake_evaluation)
    from module.research.cost_sensitivity import resimulated_curve

    resimulated = resimulated_curve({"commission_bps": 5.0, "slippage_bps": 10.0}, evidence, 15.0, ladder)
    resimulated_crossing = break_even_bps(resimulated, "selection")

    if frozen_crossing.get("available") and resimulated_crossing.get("available"):
        assert resimulated_crossing["bps_per_trade"] >= frozen_crossing["bps_per_trade"] - 1e-9
    else:
        # Si la congelada ya no cruza dentro de la escalera, la resimulada tampoco puede cruzar
        # antes: ése es el mismo contrato enunciado sobre el caso degenerado.
        assert not frozen_crossing.get("available") or resimulated_crossing.get("available")


def test_the_diagnostic_writes_only_its_own_artifact(tmp_path: Path) -> None:
    """Los costes no seleccionan nada: el artefacto no puede tocar ninguna decisión persistida."""
    result, _ = _result()
    evidence, output = tmp_path / "evidence", tmp_path / "study"
    evidence.mkdir()
    output.mkdir()
    result.equity.to_parquet(evidence / "equity.parquet")
    for name in ("winner.json", "decisions.json", "portfolio_winner.json"):
        (output / name).write_text('{"intacto": true}', encoding="utf-8")

    configuration = {"commission_bps": 5.0, "slippage_bps": 10.0}
    payload = build_cost_sensitivity(
        evidence, {**_configuration(), **configuration}, resimulate=False,
    )
    (output / "cost_sensitivity.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8",
    )
    for name in ("winner.json", "decisions.json", "portfolio_winner.json"):
        assert json.loads((output / name).read_text(encoding="utf-8")) == {"intacto": True}
    assert payload["caveats"], "las salvedades viajan dentro del artefacto"
    assert payload["break_even_against"] == "benchmark"


def _configuration() -> dict[str, Any]:
    from module.studies.catalog import default_definition

    return {key: spec["baseline"] for key, spec in default_definition().items()}
