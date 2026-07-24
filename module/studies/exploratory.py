"""Exploración secuencial: una variable, una decisión, un baseline acumulado."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from module.common.utils import write_json
from module.evaluation.stats import paired_difference_ci
from module.studies.catalog import BY_ID, CATALOG_VERSION, DECISION_REASONS
from module.studies.config import (
    initial_values,
    ordered_optimized_variables,
    settings_payload,
    validate_definition,
)
from module.studies.runner import discard_summary_cache, run_evaluation
from module.storage.datasets import prune_prepared
from module.storage.evidence import (
    STUDIES_ROOT,
    append_ledger,
    create_study,
    freeze_hypothesis,
    read_study,
    safe_path,
    update_study,
)


def exploratory_preflight(payload: Mapping[str, Any]) -> dict[str, Any]:
    _validate_metadata(payload, allowed={"name", "note", "definition"})
    definition, budget = validate_definition(payload.get("definition"))
    return {
        "valid": True,
        "catalog_version": CATALOG_VERSION,
        "definition": definition,
        "budget": budget,
        "optimized_variables": ordered_optimized_variables(definition),
    }


def create_exploratory(payload: Mapping[str, Any]) -> dict[str, Any]:
    preflight = exploratory_preflight(payload)
    values = initial_values(preflight["definition"])
    study_id, _ = create_study(
        "exploratory",
        {
            "name": str(payload.get("name") or "Exploratory Study"),
            "note": str(payload.get("note") or ""),
            "catalog_version": CATALOG_VERSION,
            "definition": preflight["definition"],
            "budget": preflight["budget"],
            "optimized_variables": preflight["optimized_variables"],
            "current_values": values,
            "next_variable_index": 0,
            "pending_decision": None,
        },
    )
    update_study(study_id, status="running")
    baseline = run_evaluation(values)
    append_ledger(study_id, [_ledger_record(
        1, "baseline", "baseline", None, values, baseline, selected=True,
        reason="automatic",
    )])
    update_study(study_id, baseline=baseline)
    return evaluate_next_variable(study_id)


def evaluate_next_variable(study_id: str) -> dict[str, Any]:
    study = read_study(study_id)
    if study["study_type"] != "exploratory":
        raise ValueError("El estudio no es exploratorio.")
    if study["status"] not in {"running", "awaiting_decision"}:
        raise ValueError(f"Estado incompatible: {study['status']}.")
    if study.get("pending_decision"):
        return study

    variables = study["optimized_variables"]
    index = int(study["next_variable_index"])
    if index >= len(variables):
        return update_study(study_id, status="awaiting_freeze", pending_decision=None)

    variable_id = variables[index]
    selection = study["definition"][variable_id]
    current = dict(study["current_values"])
    records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    ledger_path = safe_path(STUDIES_ROOT, study_id) / "evaluation_ledger.parquet"
    start = len(pd.read_parquet(ledger_path)) + 1
    for offset, value in enumerate(selection["values"]):
        candidate_values = {**current, variable_id: value}
        result = run_evaluation(candidate_values)
        candidate_id = f"{variable_id}:{offset}"
        candidates.append({
            "candidate_id": candidate_id,
            "value": value,
            "values": candidate_values,
            "result": result,
        })
        records.append(_ledger_record(
            start + offset, BY_ID[variable_id].stage, variable_id, value,
            candidate_values, result, selected=False, reason=None,
        ))
    baseline = study["baseline"]
    automatic = _select_candidate(variable_id, candidates, baseline)
    append_ledger(study_id, records)
    return update_study(
        study_id,
        status="awaiting_decision",
        pending_decision={
            "variable_id": variable_id,
            "stage": BY_ID[variable_id].stage,
            "candidates": candidates,
            "automatic_candidate_id": automatic["candidate_id"],
            "automatic_reason": automatic["selection_reason"],
        },
    )


def advance_exploratory(
    study_id: str,
    *,
    candidate_id: str | None = None,
    reason: str = "automatic",
) -> dict[str, Any]:
    study = read_study(study_id)
    if study["status"] != "awaiting_decision" or not study.get("pending_decision"):
        raise ValueError("El estudio no espera una decisión.")
    pending = study["pending_decision"]
    chosen_id = candidate_id or pending["automatic_candidate_id"]
    candidates = {item["candidate_id"]: item for item in pending["candidates"]}
    if chosen_id not in candidates:
        raise ValueError("El candidato no pertenece a la comparación pendiente.")
    if reason not in DECISION_REASONS:
        raise ValueError("Motivo de decisión fuera de catálogo.")
    if chosen_id != pending["automatic_candidate_id"] and reason == "automatic":
        raise ValueError("Una intervención humana necesita un motivo explícito.")
    chosen = candidates[chosen_id]
    variable_id = pending["variable_id"]
    _mark_selected(study_id, variable_id, chosen["value"], reason)
    decisions = list(study.get("decisions", []))
    decisions.append({
        "variable_id": variable_id,
        "candidate_id": chosen_id,
        "value": chosen["value"],
        "automatic_candidate_id": pending["automatic_candidate_id"],
        "reason": reason,
        "human_override": chosen_id != pending["automatic_candidate_id"],
    })
    update_study(
        study_id,
        status="running",
        current_values=chosen["values"],
        baseline=chosen["result"],
        decisions=decisions,
        next_variable_index=int(study["next_variable_index"]) + 1,
        pending_decision=None,
    )
    discard_summary_cache([
        candidate["result"]["evaluation_key"]
        for candidate in pending["candidates"]
        if candidate["candidate_id"] != chosen_id
    ])
    prune_prepared(keep={str(chosen["result"]["dataset_hash"])})
    return evaluate_next_variable(study_id)


def freeze_exploratory(study_id: str) -> dict[str, Any]:
    study = read_study(study_id)
    if study["status"] != "awaiting_freeze":
        raise ValueError("Solo puede congelarse una exploración completamente decidida.")
    hypothesis_id, hypothesis_dir = freeze_hypothesis({
        "source_study_id": study_id,
        "catalog_version": study["catalog_version"],
        "definition": study["definition"],
        "configuration": study["current_values"],
        "effective_settings": settings_payload(study["current_values"]),
        "decisions": study.get("decisions", []),
        "selection_metrics": study["baseline"],
        "selection_years": [2015, 2024],
        "known_stress_years": [2025, 2026],
        "statement": _hypothesis_statement(study["current_values"]),
    })
    evidence = hypothesis_dir / "evidence"
    try:
        final_result = run_evaluation(study["current_values"], retain_dir=evidence)
        hypothesis_path = hypothesis_dir / "hypothesis.json"
        payload = json.loads(hypothesis_path.read_text(encoding="utf-8"))
        payload["dataset_hash"] = final_result["dataset_hash"]
        payload["evaluation_key"] = final_result["evaluation_key"]
        payload["evidence_bytes"] = _directory_size(evidence)
        if payload["evidence_bytes"] > 100 * 1024**2:
            raise RuntimeError("La hipótesis supera el límite de 100 MiB.")
        write_json(payload, hypothesis_path)
    except Exception:
        shutil.rmtree(hypothesis_dir, ignore_errors=True)
        raise
    update_study(study_id, status="succeeded", hypothesis_id=hypothesis_id)
    return payload


def _validate_metadata(payload: Mapping[str, Any], *, allowed: set[str]) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Campos desconocidos: {sorted(unknown)}.")
    if "name" in payload and not isinstance(payload["name"], str):
        raise ValueError("name debe ser texto.")
    if "note" in payload and not isinstance(payload["note"], str):
        raise ValueError("note debe ser texto.")


def _select_candidate(
    variable_id: str,
    candidates: list[dict[str, Any]],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    stage = BY_ID[variable_id].stage
    eligible = []
    for candidate in candidates:
        result = candidate["result"]
        if stage == "portfolio":
            ok, reason = _portfolio_eligible(result, baseline)
            score = _portfolio_key(result, variable_id, candidate["value"])
        else:
            ok, reason = _signal_eligible(result, baseline)
            score = _signal_key(result, variable_id, candidate["value"])
        candidate["eligible"] = ok
        candidate["eligibility_reason"] = reason
        candidate["_score"] = score
        if ok:
            eligible.append(candidate)
    winner = max(eligible or candidates, key=lambda item: item["_score"])
    winner["selection_reason"] = (
        "best_eligible_candidate" if eligible else "no_candidate_passed_gates_best_non_inferior"
    )
    for candidate in candidates:
        candidate.pop("_score", None)
    return winner


def _signal_eligible(result: Mapping[str, Any], baseline: Mapping[str, Any]) -> tuple[bool, str]:
    current = result["summary"]
    base = baseline["summary"]
    era_ok = all((era.get("rank_ic") or 0) > -0.02 for era in result["eras"])
    paired = _paired_ci(result, baseline)
    ok = (
        (current.get("mean_rank_ic") or 0) >= (base.get("mean_rank_ic") or 0) - 0.005
        and (current.get("rank_ic_positive_fraction") or 0)
        >= (base.get("rank_ic_positive_fraction") or 0) - 0.03
        and era_ok
        and paired["ci_low"] > -0.01
        and (current.get("tail_spread") or 0) >= (base.get("tail_spread") or 0)
    )
    return ok, "eligible" if ok else "signal_gate_failed"


def _portfolio_eligible(result: Mapping[str, Any], baseline: Mapping[str, Any]) -> tuple[bool, str]:
    current = result["summary"]
    base = baseline["summary"]
    worst = min((era.get("mean_alpha") or 0) for era in result["eras"])
    base_worst = min((era.get("mean_alpha") or 0) for era in baseline["eras"])
    ok = (
        int(current.get("positive_alpha_eras") or 0) >= 2
        and (current.get("information_ratio") or 0) >= (base.get("information_ratio") or 0)
        and (current.get("annualized_turnover") or 0) <= 2.0
        and worst >= base_worst - 0.02
    )
    return ok, "eligible" if ok else "portfolio_gate_failed"


def _paired_ci(result: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    def series(payload: Mapping[str, Any]) -> pd.Series:
        rows = payload.get("rank_ic_by_cohort", [])
        return pd.Series(
            {str(row["date"]): row["rank_ic"] for row in rows if row.get("rank_ic") is not None},
            dtype=float,
        )
    return paired_difference_ci(series(result), series(baseline), block_size=12, n_boot=1000, confidence=0.90)


def _signal_key(result: Mapping[str, Any], variable_id: str, value: Any) -> tuple[float, ...]:
    summary = result["summary"]
    era_spread = [era.get("rank_ic") or -1 for era in result["eras"]]
    return (
        float(pd.Series(era_spread).median()),
        float(summary.get("tail_spread") or -1),
        float(summary.get("mean_rank_ic") or -1),
        -float(summary.get("rank_ic_std") or 99),
        -_complexity_rank(variable_id, value),
    )


def _portfolio_key(result: Mapping[str, Any], variable_id: str, value: Any) -> tuple[float, ...]:
    summary = result["summary"]
    return (
        float(summary.get("information_ratio") or -99),
        -float(summary.get("annualized_turnover") or 99),
        -_complexity_rank(variable_id, value),
    )


def _complexity_rank(variable_id: str, value: Any) -> int:
    order = BY_ID[variable_id].simplicity
    return order.index(value) if value in order else len(order)


def _ledger_record(
    number: int,
    stage: str,
    variable_id: str,
    value: Any,
    values: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    selected: bool,
    reason: str | None,
) -> dict[str, Any]:
    return {
        "evaluation_number": number,
        "phase": "exploratory",
        "stage": stage,
        "variable_id": variable_id,
        "candidate_value": json.dumps(value, ensure_ascii=False),
        "configuration": json.dumps(dict(values), ensure_ascii=False, sort_keys=True),
        "metrics": json.dumps(result["summary"], ensure_ascii=False, sort_keys=True),
        "eras": json.dumps(result["eras"], ensure_ascii=False),
        "evaluation_key": result["evaluation_key"],
        "dataset_hash": result["dataset_hash"],
        "source": result["source"],
        "elapsed_seconds": result.get("elapsed_seconds", 0.0),
        "selected": selected,
        "decision_reason": reason,
    }


def _mark_selected(study_id: str, variable_id: str, value: Any, reason: str) -> None:
    path = safe_path(STUDIES_ROOT, study_id) / "evaluation_ledger.parquet"
    frame = pd.read_parquet(path)
    value_json = json.dumps(value, ensure_ascii=False)
    mask = frame["variable_id"].eq(variable_id) & frame["candidate_value"].eq(value_json)
    frame.loc[mask, "selected"] = True
    frame.loc[mask, "decision_reason"] = reason
    from module.common.utils import write_parquet
    write_parquet(frame, path)


def _hypothesis_statement(values: Mapping[str, Any]) -> str:
    return (
        f"Una señal {values['model_family']} con horizonte {values['target_horizon_months']} meses, "
        f"meta {values['meta_method']} y cartera dinámica por meta-score debería conservar "
        "capacidad de ranking y producir alfa neto estable en las tres eras de selección."
    )


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
