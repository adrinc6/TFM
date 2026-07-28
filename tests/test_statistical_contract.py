"""Contrato estadístico: la puerta de selección y el bootstrap por bloques.

Este camino no tenía ninguna cobertura y es el que decide qué configuración gana el Model Study.
Un fallo aquí no rompe nada visible: elige en silencio el modelo equivocado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from module.evaluation.stats import (
    DEFAULT_BLOCK_SIZE,
    block_bootstrap_ci,
    paired_difference_ci,
)
from module.studies.selection import NON_INFERIORITY_MARGIN, choose_candidate


def _cohorts(values: list[float], start: str = "2015-01-30", step_months: int = 1) -> list[dict]:
    dates = pd.date_range(start, periods=len(values), freq=pd.DateOffset(months=step_months))
    return [{"date": str(date.date()), "rank_ic": value} for date, value in zip(dates, values, strict=True)]


def _result(values: list[float], *, key: str, start: str = "2015-01-30", step_months: int = 1) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "evaluation_key": key,
        "summary": {
            "mean_rank_ic": float(array.mean()),
            "rank_ic_positive_fraction": float((array > 0).mean()),
            "rank_ic_std": float(array.std(ddof=1)),
        },
        "eras": [{"era": "2015-2018", "rank_ic": float(array.mean())}],
        "rank_ic_by_cohort": _cohorts(values, start=start, step_months=step_months),
    }


def test_block_bootstrap_widens_with_larger_blocks_on_autocorrelated_series() -> None:
    """El bloque solo importa cuando las cohortes están correlacionadas, que es el caso real.

    Con cohortes independientes el tamaño de bloque es irrelevante y los intervalos coinciden. Con
    un proceso autocorrelacionado —lo que producen etiquetas de 12 meses medidas cada mes— un bloque
    corto subestima la varianza y da intervalos artificialmente estrechos. Por eso el bloque por
    defecto cubre un año completo.
    """
    rng = np.random.default_rng(0)
    shocks = rng.normal(0, 0.1, 240)
    series = np.zeros(240)
    for index in range(1, 240):
        series[index] = 0.85 * series[index - 1] + shocks[index]
    values = pd.Series(series + 0.05)
    narrow = block_bootstrap_ci(values, block_size=2, n_boot=800)
    wide = block_bootstrap_ci(values, block_size=DEFAULT_BLOCK_SIZE, n_boot=800)
    assert (wide["ci_high"] - wide["ci_low"]) > (narrow["ci_high"] - narrow["ci_low"])
    assert narrow["mean"] == wide["mean"]


def test_paired_difference_is_not_applicable_on_disjoint_grids() -> None:
    """Rejillas disjuntas no son un empate: son ausencia de evidencia.

    Barrer `execution_lag_days` desplaza la rejilla de snapshots. Devolver `ci_low = 0.0` hacía que
    la puerta `ci_low > margen_negativo` se cumpliera siempre y todos los candidatos pasaran.
    """
    left = pd.Series({f"2015-{month:02d}": 0.1 for month in range(1, 13)})
    right = pd.Series({f"2020-{month:02d}": 0.1 for month in range(1, 13)})
    result = paired_difference_ci(left, right)
    assert result["applicable"] is False
    assert result["ci_low"] is None
    assert result["n_dates"] == 0


def test_paired_difference_is_applicable_with_enough_common_dates() -> None:
    rng = np.random.default_rng(1)
    index = pd.date_range("2015-01-31", periods=60, freq="ME")
    base = rng.normal(0.05, 0.08, 60)
    left = pd.Series(base + 0.03, index=index)
    right = pd.Series(base, index=index)
    result = paired_difference_ci(left, right, block_size=DEFAULT_BLOCK_SIZE, n_boot=400)
    assert result["applicable"] is True
    assert result["mean_diff"] > 0
    assert result["fraction_a_better"] == 1.0


def test_gate_admits_a_challenger_that_dominates_the_incumbent() -> None:
    """El defecto original: un retador mejor era declarado inferior por una prueba de no inferioridad.

    Se reproduce el caso real de `feature_preset`: el retador gana en la mayoría de las fechas y su
    ventaja media es claramente positiva, pero el límite inferior del intervalo queda apenas por
    debajo del margen. La regla corregida lo admite por dominancia.
    """
    rng = np.random.default_rng(7)
    common = rng.normal(0.0, 0.12, 117)
    incumbent_values = (common + 0.073).tolist()
    challenger_values = (common + 0.099 + rng.normal(0.0, 0.02, 117)).tolist()
    incumbent = {"candidate_id": "base", "value": "core", "result": _result(incumbent_values, key="a")}
    challenger = {"candidate_id": "challenger", "value": "technical", "result": _result(challenger_values, key="b")}
    decision = choose_candidate(incumbent, [incumbent, challenger], "feature_preset")
    assert decision["winner_value"] == "technical"
    rows = {row["candidate_id"]: row for row in decision["candidates"]}
    assert rows["challenger"]["gates"]["paired_dominates_incumbent"] is True
    assert rows["challenger"]["paired_advantage"] > 0


def test_gate_rejects_a_challenger_that_wins_on_average_but_loses_most_dates() -> None:
    """Una ventaja media sostenida por unas pocas fechas extremas no es mejor ordenación."""
    values = [-0.01] * 100 + [3.0] * 17
    incumbent = {"candidate_id": "base", "value": "core", "result": _result([0.0] * 117, key="a")}
    challenger = {"candidate_id": "challenger", "value": "all", "result": _result(values, key="b")}
    decision = choose_candidate(incumbent, [incumbent, challenger], "feature_preset")
    rows = {row["candidate_id"]: row for row in decision["candidates"]}
    assert rows["challenger"]["gates"]["paired_dominates_incumbent"] is False
    assert decision["winner_value"] == "core"


def test_gate_falls_back_to_simplicity_when_the_advantage_is_noise() -> None:
    """Una ventaja por debajo de la tolerancia de empate no debe cambiar el ganador."""
    rng = np.random.default_rng(3)
    common = rng.normal(0.06, 0.10, 117)
    incumbent = {"candidate_id": "base", "value": False, "result": _result(common.tolist(), key="a")}
    challenger = {
        "candidate_id": "challenger", "value": True,
        "result": _result((common + 0.0005).tolist(), key="b"),
    }
    decision = choose_candidate(incumbent, [incumbent, challenger], "market_regime_feature")
    assert decision["selection_rule"] == "tie_simplicity"
    assert decision["winner_value"] is False


def test_incumbent_is_the_only_eligible_candidate_when_pairing_is_impossible() -> None:
    incumbent = {"candidate_id": "base", "value": 45, "result": _result([0.05] * 117, key="a")}
    challenger = {
        "candidate_id": "challenger", "value": 60,
        "result": _result([0.30] * 117, key="b", start="2050-01-30"),
    }
    decision = choose_candidate(incumbent, [incumbent, challenger], "execution_lag_days")
    rows = {row["candidate_id"]: row for row in decision["candidates"]}
    assert rows["challenger"]["gates"]["paired_applicable"] is False
    assert rows["challenger"]["eligible"] is False
    assert decision["winner_value"] == 45


def test_non_inferiority_margin_is_still_honoured_for_similar_candidates() -> None:
    rng = np.random.default_rng(11)
    common = rng.normal(0.06, 0.05, 117)
    incumbent = {"candidate_id": "base", "value": "core", "result": _result(common.tolist(), key="a")}
    worse = {
        "candidate_id": "worse", "value": "fundamental",
        "result": _result((common - 0.06).tolist(), key="b"),
    }
    decision = choose_candidate(incumbent, [incumbent, worse], "feature_preset")
    rows = {row["candidate_id"]: row for row in decision["candidates"]}
    assert rows["worse"]["paired_bootstrap_90"]["ci_low"] <= NON_INFERIORITY_MARGIN
    assert rows["worse"]["eligible"] is False
