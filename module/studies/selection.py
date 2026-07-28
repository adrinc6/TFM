"""Selección secuencial basada únicamente en Rank-IC robusto."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from module.evaluation.stats import DEFAULT_BLOCK_SIZE, paired_difference_ci
from module.studies.catalog import BY_ID


ERA_FLOOR = -0.02
NON_INFERIORITY_MARGIN = -0.01
TIE_TOLERANCE = 0.002
PAIRED_BOOTSTRAP_DRAWS = 2000
PAIRED_CONFIDENCE = 0.90
SELECTION_RULE = (
    "Un candidato es elegible si (a) tiene observaciones suficientes, (b) ninguna era disponible "
    f"cae por debajo de {ERA_FLOOR}, y (c) supera la puerta pareada: o bien domina al incumbent "
    "(diferencia media positiva y mejor en más de la mitad de las fechas emparejadas), o bien es no "
    f"inferior (límite inferior del IC al {PAIRED_CONFIDENCE:.0%} por encima de "
    f"{NON_INFERIORITY_MARGIN}). Si el emparejamiento no es aplicable —rejillas de snapshots "
    "disjuntas— no hay evidencia pareada y solo el incumbent es elegible. El orden entre elegibles "
    "lo fija la diferencia pareada contra el incumbent, no la mediana entre eras."
)


def choose_candidate(
    incumbent: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    variable_id: str,
) -> dict[str, Any]:
    """Devuelve una decisión auditable sin consultar ninguna métrica económica."""
    assessed = [_assess(incumbent, candidate, variable_id) for candidate in candidates]
    eligible = [item for item in assessed if item["eligible"]]
    if not eligible:
        winner = next(item for item in assessed if item["is_incumbent"])
        rule = "incumbent_no_eligible_challenger"
    else:
        ranked = sorted(eligible, key=_sort_key)
        best = ranked[0]
        incumbent_row = next(item for item in assessed if item["is_incumbent"])
        if best["candidate_id"] != incumbent_row["candidate_id"] and _tied(best, incumbent_row):
            winner, rule = _simpler(best, incumbent_row, variable_id), "tie_simplicity"
        else:
            winner, rule = best, "robust_rank_ic"
    return {
        "variable_id": variable_id,
        "winner_candidate_id": winner["candidate_id"],
        "winner_value": winner["value"],
        "selection_rule": rule,
        "selection_metric": "rank_ic_only",
        "selection_rule_text": SELECTION_RULE,
        "era_floor": ERA_FLOOR,
        "non_inferiority_margin": NON_INFERIORITY_MARGIN,
        "paired_bootstrap_draws": PAIRED_BOOTSTRAP_DRAWS,
        "paired_confidence": PAIRED_CONFIDENCE,
        "known_stress_excluded": True,
        "candidates": assessed,
    }


def _assess(
    incumbent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    variable_id: str,
) -> dict[str, Any]:
    result = candidate["result"]
    summary = result["summary"]
    eras = result.get("eras", [])
    era_values = [row.get("rank_ic") for row in eras if row.get("rank_ic") is not None]
    observations = len(result.get("rank_ic_by_cohort", []))
    is_incumbent = candidate["candidate_id"] == incumbent["candidate_id"]
    paired = _paired(candidate["result"], incumbent["result"])
    enough = observations >= 3
    above_floor = bool(era_values and min(era_values) >= ERA_FLOOR)
    dominates, non_inferior = _paired_gates(paired, is_incumbent=is_incumbent)
    eligible = bool(enough and era_values and above_floor and (dominates or non_inferior))
    return {
        "candidate_id": candidate["candidate_id"],
        "value": candidate["value"],
        "is_incumbent": is_incumbent,
        "eligible": eligible,
        "gates": {
            "observations_sufficient": enough,
            "all_available_eras_above_floor": above_floor,
            "paired_dominates_incumbent": dominates,
            "paired_bootstrap_non_inferior": non_inferior,
            "paired_applicable": bool(paired.get("applicable", False)),
        },
        "paired_bootstrap_90": paired,
        "paired_advantage": _paired_advantage(paired, is_incumbent=is_incumbent),
        "median_era_rank_ic": _finite(np.median(era_values)) if era_values else None,
        "mean_rank_ic": summary.get("mean_rank_ic"),
        "positive_fraction": summary.get("rank_ic_positive_fraction"),
        "rank_ic_std": summary.get("rank_ic_std"),
        "observations": observations,
        "value_complexity_rank": _complexity(variable_id, candidate["value"]),
        "reason": _reason(eligible, era_values, paired, observations, is_incumbent=is_incumbent),
    }


def _paired_gates(paired: Mapping[str, Any], *, is_incumbent: bool) -> tuple[bool, bool]:
    """Dominancia y no inferioridad frente al incumbent.

    El incumbent se compara consigo mismo, así que no hay diferencia que medir: es no inferior por
    definición y nunca se domina a sí mismo. Para el resto, `dominates` exige que la ventaja media
    sea positiva **y** que gane en más de la mitad de las fechas emparejadas: una ventaja media
    positiva sostenida por unas pocas fechas extremas no es evidencia de mejor ordenación.

    Cuando el emparejamiento no es aplicable (rejillas disjuntas) ninguna de las dos puertas se
    abre: sin fechas comunes no hay evidencia, y tratarlo como empate dejaría pasar a todos los
    candidatos automáticamente.
    """
    if is_incumbent:
        return False, True
    if not paired.get("applicable", False):
        return False, False
    mean_diff = paired.get("mean_diff")
    fraction = paired.get("fraction_a_better")
    ci_low = paired.get("ci_low")
    dominates = bool(
        mean_diff is not None and fraction is not None
        and mean_diff > 0 and fraction > 0.5
    )
    non_inferior = bool(ci_low is not None and ci_low > NON_INFERIORITY_MARGIN)
    return dominates, non_inferior


def _paired_advantage(paired: Mapping[str, Any], *, is_incumbent: bool) -> float:
    """Estadístico de orden: ventaja pareada media contra el incumbent.

    El incumbent vale exactamente 0 (es su propia referencia). Un candidato sin emparejamiento
    aplicable también vale 0: no ha demostrado ventaja, así que empata con el incumbent y la
    simplicidad decide.
    """
    if is_incumbent or not paired.get("applicable", False):
        return 0.0
    return _number(paired.get("mean_diff"), default=0.0)


def _paired(candidate: Mapping[str, Any], incumbent: Mapping[str, Any]) -> dict[str, Any]:
    """Empareja las cohortes de ambos candidatos por periodo mensual.

    Las fechas exactas no coinciden entre configuraciones: `dataset.py` sitúa cada snapshot en
    `fin_de_mes + execution_lag_days`, así que barrer el lag desplaza toda la rejilla. Emparejar por
    la cadena de fecha dejaría cero fechas comunes y la puerta pareada quedaría vacía. El periodo
    mensual identifica la misma cohorte económica bajo cualquier lag del catálogo.
    """
    left = _cohort_series(candidate)
    right = _cohort_series(incumbent)
    if candidate.get("evaluation_key") == incumbent.get("evaluation_key"):
        return {
            "mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0,
            "n_dates": len(left), "fraction_a_better": 0.0,
            "distinguishable_from_zero": False, "applicable": True,
            "block_size": DEFAULT_BLOCK_SIZE, "same_evaluation": True,
        }
    return paired_difference_ci(
        left, right, block_size=DEFAULT_BLOCK_SIZE,
        n_boot=PAIRED_BOOTSTRAP_DRAWS, confidence=PAIRED_CONFIDENCE,
    )


def _cohort_series(result: Mapping[str, Any]) -> pd.Series:
    rows = [
        (pd.Timestamp(row["date"]).to_period("M"), float(row["rank_ic"]))
        for row in result.get("rank_ic_by_cohort", [])
        if row.get("rank_ic") is not None
    ]
    if not rows:
        return pd.Series(dtype=float)
    series = pd.Series({period: value for period, value in rows})
    return series.sort_index()


def _sort_key(item: Mapping[str, Any]) -> tuple[float, float, float, float, int]:
    """Ordena por la ventaja pareada contra el incumbent, no por la mediana entre eras.

    `median_era_rank_ic` es la mediana de tres números —una por era de selección— y no distingue
    diferencias del orden del ruido. La diferencia pareada usa las 117 cohortes emparejadas y tiene
    varianza mucho menor, porque elimina el factor común de mercado de cada fecha.
    """
    return (
        -_number(item.get("paired_advantage"), default=0.0),
        -_number(item.get("mean_rank_ic")),
        -_number(item.get("positive_fraction")),
        _number(item.get("rank_ic_std"), default=999.0),
        int(item.get("value_complexity_rank", 999)),
    )


def _tied(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return abs(
        _number(left.get("paired_advantage"), default=0.0)
        - _number(right.get("paired_advantage"), default=0.0)
    ) < TIE_TOLERANCE


def _simpler(
    left: Mapping[str, Any], right: Mapping[str, Any], variable_id: str,
) -> Mapping[str, Any]:
    left_rank = _complexity(variable_id, left["value"])
    right_rank = _complexity(variable_id, right["value"])
    return left if left_rank < right_rank else right


def _complexity(variable_id: str, value: Any) -> int:
    ordering = list(BY_ID[variable_id].simplicity)
    try:
        return ordering.index(value)
    except ValueError:
        return len(ordering)


def _reason(
    eligible: bool,
    eras: list[float],
    paired: Mapping[str, Any],
    observations: int,
    *,
    is_incumbent: bool,
) -> str:
    if eligible:
        if is_incumbent:
            return "Incumbent: referencia de la comparación, elegible por definición."
        dominates, _ = _paired_gates(paired, is_incumbent=False)
        if dominates:
            return (
                f"Domina al incumbent: ventaja media {paired['mean_diff']:+.4f} y mejor en el "
                f"{paired['fraction_a_better']:.1%} de las {paired['n_dates']} cohortes emparejadas."
            )
        return "Elegible por suficiencia, estabilidad entre eras y no inferioridad pareada."
    failures = []
    if observations < 3:
        failures.append("observaciones insuficientes")
    if not eras or min(eras) < ERA_FLOOR:
        failures.append("era por debajo del suelo")
    if not is_incumbent:
        if not paired.get("applicable", False):
            failures.append(
                f"emparejamiento no aplicable: solo {paired.get('n_dates', 0)} cohortes comunes "
                f"frente al bloque mínimo de {paired.get('block_size', DEFAULT_BLOCK_SIZE)}"
            )
        else:
            dominates, non_inferior = _paired_gates(paired, is_incumbent=False)
            if not dominates and not non_inferior:
                failures.append("no domina al incumbent ni es no inferior")
    return "; ".join(failures)


def _number(value: Any, default: float = -999.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _finite(value: Any) -> float | None:
    number = _number(value)
    return number if number > -999 else None
