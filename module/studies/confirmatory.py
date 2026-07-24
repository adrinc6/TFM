"""Corroboración cerrada de una hipótesis inmutable."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from module.common.utils import write_json, write_parquet
from module.evaluation.profiles import PROFILE_NAMES
from module.evaluation.robustness import label_permutation_test
from module.evaluation.stats import block_bootstrap_ci
from module.studies.config import CONFIRMATORY_EVALUATIONS
from module.studies.runner import SELECTION_ERAS, run_evaluation, run_profile_evaluation
from module.storage.datasets import validate_dataset_reference
from module.storage.evidence import (
    append_ledger,
    create_model,
    create_study,
    read_hypothesis,
    safe_path,
    update_study,
)


SEEDS = (7, 2026)
COST_CASES = ((0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (15.0, 30.0))
PLACEBO_SEEDS = (101, 102, 103, 104, 105)
CONFIRMATORY_BREAKDOWN = {
    "seeds": len(SEEDS),
    "profiles": len(PROFILE_NAMES),
    "costs": len(COST_CASES),
    "calendar": 1,
    "placebos": len(PLACEBO_SEEDS),
    "bootstrap_and_eras": 1,
    "permutation": 1,
    "random_portfolios": 1,
}
if sum(CONFIRMATORY_BREAKDOWN.values()) != CONFIRMATORY_EVALUATIONS:
    raise RuntimeError("El presupuesto confirmatorio dejó de sumar 23.")


def confirmatory_preflight(payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"hypothesis_id", "name", "note"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Confirmatory no admite overrides: {sorted(unknown)}.")
    hypothesis_id = payload.get("hypothesis_id")
    if not isinstance(hypothesis_id, str):
        raise ValueError("Debe seleccionarse una hypothesis_id.")
    hypothesis = read_hypothesis(hypothesis_id)
    validate_dataset_reference(str(hypothesis["dataset_hash"]))
    return {
        "valid": True,
        "hypothesis_id": hypothesis_id,
        "evaluations": CONFIRMATORY_EVALUATIONS,
        "scientific_overrides_allowed": False,
        "dataset_hash": hypothesis["dataset_hash"],
    }


def run_confirmatory(payload: Mapping[str, Any]) -> dict[str, Any]:
    preflight = confirmatory_preflight(payload)
    hypothesis = read_hypothesis(preflight["hypothesis_id"])
    values = hypothesis["configuration"]
    study_id, study_dir = create_study(
        "confirmatory",
        {
            "name": str(payload.get("name") or "Confirmatory Study"),
            "note": str(payload.get("note") or ""),
            "hypothesis_id": hypothesis["hypothesis_id"],
            "budget": {"evaluations": CONFIRMATORY_EVALUATIONS},
        },
    )
    update_study(study_id, status="running", completed_evaluations=0)
    records: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    number = 0
    ledger_written = False

    def record(group: str, name: str, result: Mapping[str, Any], status: str = "succeeded") -> None:
        nonlocal number
        number += 1
        records.append({
            "evaluation_number": number,
            "phase": "confirmatory",
            "group": group,
            "name": name,
            "status": status,
            "metrics": json.dumps(result, ensure_ascii=False, sort_keys=True, default=str),
        })
        update_study(study_id, completed_evaluations=number, current_group=group, current_name=name)

    try:
        seed_results = []
        for seed in SEEDS:
            result = run_evaluation(values, random_seed=seed)
            seed_results.append(result)
            record("seeds", f"seed_{seed}", result["summary"])
        results["seeds"] = seed_results

        from module.storage.evidence import HYPOTHESES_ROOT
        winner_evidence = safe_path(HYPOTHESES_ROOT, hypothesis["hypothesis_id"]) / "evidence"
        profile_results = []
        for profile in PROFILE_NAMES:
            result = run_profile_evaluation(values, profile, winner_evidence, winner_evidence / "profiles" / profile)
            profile_results.append({"profile": profile, "result": result})
            record("profiles", profile, result["summary"])
        results["profiles"] = profile_results

        cost_results = []
        for commission, slippage in COST_CASES:
            result = run_evaluation(
                values,
                overrides={"commission_bps": commission, "slippage_bps": slippage},
            )
            item = {"commission_bps": commission, "slippage_bps": slippage, "result": result}
            cost_results.append(item)
            record("costs", f"commission_{commission:g}_slippage_{slippage:g}", result["summary"])
        results["costs"] = cost_results

        calendar = run_evaluation(values, overrides={"price_only_strictness_multiplier": 2.0})
        results["conservative_rebalance"] = calendar
        record("calendar", "strictness_two", calendar["summary"])

        placebo_results = []
        prepared = validate_dataset_reference(str(hypothesis["dataset_hash"]))
        real_targets = pd.read_parquet(prepared / "targets_forward.parquet")
        for seed in PLACEBO_SEEDS:
            shuffled = _shuffle_targets(real_targets, seed)
            temp_dir = Path(tempfile.mkdtemp(prefix=f"placebo-{seed}-"))
            try:
                target_path = temp_dir / "targets_forward.parquet"
                write_parquet(shuffled, target_path)
                result = run_evaluation(
                    values,
                    random_seed=seed,
                    target_override=target_path,
                    target_identity=f"placebo-{seed}",
                )
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
            placebo_results.append(result)
            record("placebos", f"labels_{seed}", result["summary"])
        results["placebos"] = placebo_results

        # La ruta persistida no se guarda dentro del JSON: se resuelve de forma segura por ID.
        evidence = safe_path(HYPOTHESES_ROOT, hypothesis["hypothesis_id"]) / "evidence"
        diagnostics = pd.read_parquet(evidence / "rank_ic_diagnostics.parquet")
        scores = pd.read_parquet(evidence / "agent_scores.parquet")
        diagnostics_selection = _meta_selection(diagnostics)

        bootstrap = _bootstrap_and_eras(diagnostics_selection)
        results["bootstrap_and_eras"] = bootstrap
        record("statistics", "bootstrap_and_era_exclusion", bootstrap)

        permutation = _score_permutation(scores, real_targets)
        results["permutation"] = permutation
        record("statistics", "score_return_permutation", permutation)

        random_result = _random_portfolios(prepared, evidence, values)
        results["random_portfolios"] = random_result
        record("statistics", "pit_random_portfolios", random_result)

        if number != CONFIRMATORY_EVALUATIONS:
            raise RuntimeError(
                f"Confirmatory produjo {number} evaluaciones; se esperaban {CONFIRMATORY_EVALUATIONS}."
            )
        append_ledger(study_id, records)
        ledger_written = True

        placebo_ic = [
            float(item["summary"].get("mean_rank_ic") or 0)
            for item in placebo_results
        ]
        placebo_test = label_permutation_test(diagnostics_selection, placebo_ic)
        results["placebo_summary"] = placebo_test
        verdict, reasons = _verdict(hypothesis, results)
        decision = {
            "schema_version": 1,
            "study_id": study_id,
            "hypothesis_id": hypothesis["hypothesis_id"],
            "verdict": verdict,
            "reasons": reasons,
            "configuration": values,
            "selection_years": [2015, 2024],
            "known_stress_years": [2025, 2026],
            "known_stress_role": "known_stress_not_selection",
            "evaluation_budget": CONFIRMATORY_EVALUATIONS,
            "results": results,
        }
        write_json(decision, study_dir / "decision.json")

        model_id = None
        if verdict in {"confirmed", "non_inferior"}:
            model_id, model_dir = create_model({
                "hypothesis_id": hypothesis["hypothesis_id"],
                "confirmatory_study_id": study_id,
                "verdict": verdict,
                "configuration": values,
                "dataset_hash": hypothesis["dataset_hash"],
            })
            final_result = run_evaluation(values, retain_dir=model_dir / "evidence")
            write_json(decision, model_dir / "decision.json")
            model_bytes = _directory_size(model_dir)
            if model_bytes > 250 * 1024**2:
                shutil.rmtree(model_dir, ignore_errors=True)
                raise RuntimeError("La evidencia final supera el límite de 250 MiB.")
            decision["model_id"] = model_id
            decision["final_summary"] = final_result
            write_json(decision, study_dir / "decision.json")
        update_study(
            study_id, status="succeeded", completed_evaluations=number,
            verdict=verdict, model_id=model_id,
        )
        return decision
    except Exception as exc:
        if records and not ledger_written:
            append_ledger(study_id, records)
        update_study(study_id, status="failed", error=str(exc), completed_evaluations=number)
        raise


def _shuffle_targets(targets: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = targets.copy()
    columns = ["forward_return", "forward_benchmark_return", "forward_excess_return"]
    for _, indices in frame.groupby("snapshot_date").groups.items():
        positions = np.asarray(list(indices))
        permutation = rng.permutation(len(positions))
        frame.loc[positions, columns] = frame.loc[positions, columns].to_numpy()[permutation]
    return frame


def _meta_selection(diagnostics: pd.DataFrame) -> pd.DataFrame:
    frame = diagnostics.loc[diagnostics["agent"].eq("meta_final")].copy()
    frame["year"] = pd.to_datetime(frame["prediction_date"]).dt.year
    return frame.loc[frame["year"].le(2024)]


def _bootstrap_and_eras(diagnostics: pd.DataFrame) -> dict[str, Any]:
    values = diagnostics.set_index("prediction_date")["rank_ic"].dropna()
    full = block_bootstrap_ci(values, block_size=12, n_boot=2000, confidence=0.95)
    exclusions = []
    for start, end in SELECTION_ERAS:
        years = pd.to_datetime(values.index).year
        without = values.loc[~pd.Series(years, index=values.index).between(start, end)]
        exclusions.append({
            "excluded_era": f"{start}-{end}",
            "mean_rank_ic": float(without.mean()) if len(without) else None,
            "n_cohorts": int(len(without)),
        })
    return {"bootstrap_95": full, "era_exclusions": exclusions}


def _score_permutation(scores: pd.DataFrame, targets: pd.DataFrame) -> dict[str, Any]:
    merged = scores[["ticker", "snapshot_date", "meta_rank"]].merge(
        targets[["ticker", "snapshot_date", "forward_excess_return", "target_available"]],
        on=["ticker", "snapshot_date"], how="inner",
    )
    merged = merged.loc[merged["target_available"].fillna(False)].copy()
    merged["year"] = pd.to_datetime(merged["snapshot_date"]).dt.year
    merged = merged.loc[merged["year"].le(2024)]
    groups = [
        group[["meta_rank", "forward_excess_return"]].dropna().to_numpy()
        for _, group in merged.groupby("snapshot_date")
    ]
    observed = np.mean([_spearman(group[:, 0], group[:, 1]) for group in groups if len(group) >= 8])
    rng = np.random.default_rng(42)
    exceedances = 0
    n_permutations = 9_999
    for _ in range(n_permutations):
        statistic = np.mean([
            _spearman(group[:, 0], rng.permutation(group[:, 1]))
            for group in groups if len(group) >= 8
        ])
        exceedances += statistic >= observed
    return {
        "observed_mean_rank_ic": float(observed),
        "p_value": float((exceedances + 1) / (n_permutations + 1)),
        "n_permutations": n_permutations,
        "add_one_correction": True,
    }


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(pd.Series(left).rank().corr(pd.Series(right).rank()))


def _random_portfolios(prepared: Path, evidence: Path, values: Mapping[str, Any]) -> dict[str, Any]:
    prices = pd.read_parquet(prepared / "asset_price_point_in_time.parquet")
    scores = pd.read_parquet(evidence / "agent_scores.parquet")
    annual = pd.read_parquet(evidence / "annual_metrics.parquet")
    prices["year"] = pd.to_datetime(prices["snapshot_date"]).dt.year
    prices = prices.loc[prices["year"].le(2024)].sort_values(["ticker", "snapshot_date"])
    returns = prices.groupby(["year", "ticker"])["price"].agg(["first", "last"]).reset_index()
    returns["return"] = returns["last"] / returns["first"] - 1
    pools = {int(year): group["return"].dropna().to_numpy() for year, group in returns.groupby("year")}
    model = annual.loc[annual["year"].le(2024)].set_index("year")["portfolio_return"]
    general = _simulate_random(model, pools, int(values["target_size"]), seed=42)

    scores["year"] = pd.to_datetime(scores["snapshot_date"]).dt.year
    risk = scores.groupby(["year", "ticker"])["risk_rank"].mean().reset_index()
    risk["quintile"] = risk.groupby("year")["risk_rank"].transform(
        lambda values_: pd.qcut(values_.rank(method="first"), 5, labels=False, duplicates="drop")
    )
    matched = returns.merge(risk, on=["year", "ticker"], how="inner")
    central = matched.loc[matched["quintile"].isin([1, 2, 3])]
    matched_pools = {
        int(year): group["return"].dropna().to_numpy()
        for year, group in central.groupby("year")
    }
    risk_matched = _simulate_random(model, matched_pools, int(values["target_size"]), seed=43)
    return {
        "general": general,
        "risk_matched": risk_matched,
        "beats_random_convincingly": (
            general["model_percentile"] >= 0.95 and risk_matched["model_percentile"] >= 0.95
        ),
    }


def _simulate_random(
    model: pd.Series,
    pools: Mapping[int, np.ndarray],
    size: int,
    *,
    seed: int,
) -> dict[str, Any]:
    years = sorted(set(model.index) & set(pools))
    rng = np.random.default_rng(seed)
    simulations = np.zeros(1_000)
    for index in range(len(simulations)):
        yearly = []
        for year in years:
            pool = np.asarray(pools[year])
            pool = pool[np.isfinite(pool)]
            yearly.append(float(rng.choice(pool, min(size, len(pool)), replace=False).mean()) if len(pool) else 0)
        simulations[index] = np.prod(1 + np.asarray(yearly)) ** (1 / max(len(yearly), 1)) - 1
    model_values = model.reindex(years).dropna().to_numpy()
    model_cagr = float(np.prod(1 + model_values) ** (1 / len(model_values)) - 1) if len(model_values) else 0
    return {
        "model_cagr": model_cagr,
        "random_mean": float(simulations.mean()),
        "random_p95": float(np.quantile(simulations, 0.95)),
        "model_percentile": float((simulations < model_cagr).mean()),
        "n_simulations": 1_000,
    }


def _verdict(hypothesis: Mapping[str, Any], results: Mapping[str, Any]) -> tuple[str, list[str]]:
    base = hypothesis["selection_metrics"]["summary"]
    seed_ics = [item["summary"].get("mean_rank_ic") or 0 for item in results["seeds"]]
    high_cost = results["costs"][2]["result"]["summary"]
    bootstrap = results["bootstrap_and_eras"]["bootstrap_95"]
    random_ok = results["random_portfolios"]["beats_random_convincingly"]
    placebo = results["placebo_summary"]
    leakage_ok = (
        (placebo.get("rank_ic_real") or 0) > (placebo.get("placebo_max") or 0)
    )
    signal_ok = (
        min(seed_ics) > -0.02
        and bootstrap.get("ci_low", -1) > -0.01
        and results["permutation"]["p_value"] <= 0.10
        and leakage_ok
    )
    portfolio_ok = (
        int(base.get("positive_alpha_eras") or 0) >= 2
        and (high_cost.get("information_ratio") or 0) >= 0
        and (base.get("annualized_turnover") or 0) <= 2.0
        and random_ok
    )
    non_inferior = (
        min(seed_ics) >= (base.get("mean_rank_ic") or 0) - 0.005
        and bootstrap.get("ci_low", -1) > -0.01
        and (high_cost.get("information_ratio") or 0) >= 0
        and (base.get("annualized_turnover") or 0) <= 2.0
    )
    reasons = [
        f"signal_ok={signal_ok}",
        f"portfolio_ok={portfolio_ok}",
        f"seed_rank_ic_min={min(seed_ics):.6f}",
        f"bootstrap_ci_low={bootstrap.get('ci_low')}",
        f"real_above_placebo_range={leakage_ok}",
        f"random_nulls={random_ok}",
    ]
    if signal_ok and portfolio_ok:
        return "confirmed", reasons
    if signal_ok:
        return "signal_only", reasons
    if non_inferior:
        return "non_inferior", reasons
    return "rejected", reasons


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
