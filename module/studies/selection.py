"""Selección secuencial basada únicamente en Rank-IC robusto."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from module.evaluation.stats import paired_difference_ci
from module.studies.catalog import BY_ID


ERA_FLOOR = -0.02
NON_INFERIORITY_MARGIN = -0.01
TIE_TOLERANCE = 0.002


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
    paired = _paired(candidate["result"], incumbent["result"])
    eligible = bool(
        observations >= 3
        and len(era_values) >= 1
        and min(era_values) >= ERA_FLOOR
        and paired["ci_low"] > NON_INFERIORITY_MARGIN
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "value": candidate["value"],
        "is_incumbent": candidate["candidate_id"] == incumbent["candidate_id"],
        "eligible": eligible,
        "gates": {
            "observations_sufficient": observations >= 3,
            "all_available_eras_above_floor": bool(era_values and min(era_values) >= ERA_FLOOR),
            "paired_bootstrap_non_inferior": paired["ci_low"] > NON_INFERIORITY_MARGIN,
        },
        "paired_bootstrap_90": paired,
        "median_era_rank_ic": _finite(np.median(era_values)) if era_values else None,
        "mean_rank_ic": summary.get("mean_rank_ic"),
        "positive_fraction": summary.get("rank_ic_positive_fraction"),
        "rank_ic_std": summary.get("rank_ic_std"),
        "observations": observations,
        "value_complexity_rank": _complexity(variable_id, candidate["value"]),
        "reason": _reason(eligible, era_values, paired, observations),
    }


def _paired(candidate: Mapping[str, Any], incumbent: Mapping[str, Any]) -> dict[str, Any]:
    left = pd.Series({
        str(row["date"]): row["rank_ic"] for row in candidate.get("rank_ic_by_cohort", [])
        if row.get("rank_ic") is not None
    })
    right = pd.Series({
        str(row["date"]): row["rank_ic"] for row in incumbent.get("rank_ic_by_cohort", [])
        if row.get("rank_ic") is not None
    })
    if candidate.get("evaluation_key") == incumbent.get("evaluation_key"):
        return {
            "mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0,
            "n_dates": len(left), "fraction_a_better": 0.0,
            "distinguishable_from_zero": False,
        }
    return paired_difference_ci(left, right, block_size=12, n_boot=1000, confidence=0.90)


def _sort_key(item: Mapping[str, Any]) -> tuple[float, float, float, float, int]:
    return (
        -_number(item.get("median_era_rank_ic")),
        -_number(item.get("mean_rank_ic")),
        -_number(item.get("positive_fraction")),
        _number(item.get("rank_ic_std"), default=999.0),
        int(item.get("value_complexity_rank", 999)),
    )


def _tied(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return abs(
        _number(left.get("median_era_rank_ic"))
        - _number(right.get("median_era_rank_ic"))
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
    eligible: bool, eras: list[float], paired: Mapping[str, Any], observations: int,
) -> str:
    if eligible:
        return "Elegible por suficiencia, estabilidad entre eras y no inferioridad pareada."
    failures = []
    if observations < 3:
        failures.append("observaciones insuficientes")
    if not eras or min(eras) < ERA_FLOOR:
        failures.append("era por debajo del suelo")
    if paired["ci_low"] <= NON_INFERIORITY_MARGIN:
        failures.append("bootstrap incompatible con no inferioridad")
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
