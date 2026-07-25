"""Único ejecutor científico: configuración cerrada → métricas y evidencia."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from environment import DATA_DIR
from module.common.utils import write_json, write_parquet
from module.evaluation.backtest import BacktestResult, run_backtest
from module.modeling.agents import build_agent_scores
from module.studies.catalog import SELECTION_ERAS
from module.studies.config import settings_from_values
from module.storage.cache import canonical_json, enforce_cache_limit
from module.storage.datasets import ensure_prepared


SUMMARY_CACHE = DATA_DIR / "cache" / "evaluations"
log = logging.getLogger(__name__)
KNOWN_STRESS_YEARS = (2025, 2026)


def evaluation_key(
    values: Mapping[str, Any],
    *,
    random_seed: int,
    profile: str,
    overrides: Mapping[str, Any] | None = None,
    target_identity: str = "real",
) -> str:
    payload = {
        "values": dict(values),
        "random_seed": random_seed,
        "profile": profile,
        "overrides": dict(overrides or {}),
        "target_identity": target_identity,
        "runner_schema": 1,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def run_evaluation(
    values: Mapping[str, Any],
    *,
    random_seed: int = 42,
    profile: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    target_override: Path | None = None,
    target_identity: str = "real",
    retain_dir: Path | None = None,
) -> dict[str, Any]:
    effective_profile = profile or "balanced"
    key = evaluation_key(
        values, random_seed=random_seed, profile=effective_profile, overrides=overrides,
        target_identity=target_identity,
    )
    cached = SUMMARY_CACHE / key / "summary.json"
    if cached.exists() and retain_dir is None:
        log.info("Evaluación %s: resumen reutilizado de caché", key[:12])
        payload = json.loads(cached.read_text(encoding="utf-8"))
        payload["source"] = "cached"
        return payload

    started = time.perf_counter()
    log.info("Evaluación %s: inicio (perfil=%s, semilla=%s)", key[:12], effective_profile, random_seed)
    base_settings = settings_from_values(
        values, random_seed=random_seed, profile=effective_profile, overrides=overrides,
    )
    dataset_hash, prepared, dataset_reused = ensure_prepared(base_settings)
    log.info("Evaluación %s: dataset %s (%s)", key[:12], dataset_hash[:12], "reutilizado" if dataset_reused else "creado")
    work = Path(tempfile.mkdtemp(prefix=f"evaluation-{key[:10]}-"))
    try:
        runtime = settings_from_values(
            values, workspace_dir=prepared, random_seed=random_seed, profile=effective_profile,
            overrides=overrides,
        )
        agents_root = work / "agents"
        build_agent_scores(
            runtime,
            target_path_override=target_override,
            run_root=agents_root,
            input_dir_override=prepared,
        )
        agent_dirs = [path for path in agents_root.iterdir() if path.is_dir()]
        if len(agent_dirs) != 1:
            raise RuntimeError("El runner esperaba exactamente un directorio de agente.")
        agent_dir = agent_dirs[0]
        scores = pd.read_parquet(agent_dir / "agent_scores.parquet")
        diagnostics = pd.read_parquet(agent_dir / "rank_ic_diagnostics.parquet")
        prices = pd.read_parquet(prepared / "asset_price_point_in_time.parquet")
        benchmark = pd.read_parquet(prepared / "benchmark_point_in_time.parquet")
        result = run_backtest(scores, prices, benchmark, runtime, diagnostics)
        tail_path = agent_dir / "rank_tail_diagnostics.parquet"
        tail = pd.read_parquet(tail_path) if tail_path.exists() else pd.DataFrame()
        summary = _summary(
            result, diagnostics, tail, dataset_hash=dataset_hash, evaluation_key=key,
            elapsed_seconds=time.perf_counter() - started, dataset_reused=dataset_reused,
        )
        cache_dir = SUMMARY_CACHE / key
        cache_dir.mkdir(parents=True, exist_ok=True)
        write_json(summary, cached)
        enforce_cache_limit(protected={key})
        if retain_dir is not None:
            _retain_evidence(retain_dir, prepared, agent_dir, result, summary)
        log.info("Evaluación %s: terminada en %.1fs", key[:12], summary["elapsed_seconds"])
        return summary
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_profile_evaluation(
    values: Mapping[str, Any], profile: str, evidence_dir: Path, retain_dir: Path | None = None,
) -> dict[str, Any]:
    """Backtest de un perfil sobre los scores ya congelados del ganador, sin reentrenar."""
    log.info("Perfil %s: backtest sobre scores congelados", profile)
    runtime = settings_from_values(values, profile=profile)
    reference = json.loads((evidence_dir / "dataset_reference.json").read_text(encoding="utf-8"))
    prepared = Path(reference["prepared_path"])
    scores = pd.read_parquet(evidence_dir / "agent_scores.parquet")
    diagnostics = pd.read_parquet(evidence_dir / "rank_ic_diagnostics.parquet")
    prices = pd.read_parquet(prepared / "asset_price_point_in_time.parquet")
    benchmark = pd.read_parquet(prepared / "benchmark_point_in_time.parquet")
    result = run_backtest(scores, prices, benchmark, runtime, diagnostics)
    summary = {
        "dataset_hash": reference["dataset_hash"], "profile": profile,
        "summary": result.summary, "eras": _profile_eras(result.annual_metrics),
    }
    if retain_dir is not None:
        retain_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(result.equity, retain_dir / "equity.parquet")
        write_parquet(result.annual_metrics, retain_dir / "annual_metrics.parquet")
        write_parquet(result.positions, retain_dir / "positions.parquet")
        write_parquet(result.orders, retain_dir / "orders.parquet")
        write_json(summary, retain_dir / "summary.json")
    log.info("Perfil %s: terminado (IR=%s, alfa=%s)", profile, result.summary.get("information_ratio"), result.summary.get("mean_annual_alpha"))
    return summary


def _profile_eras(annual: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for start, end in SELECTION_ERAS:
        part = annual.loc[annual["year"].between(start, end)]
        rows.append({"era": f"{start}-{end}", "mean_alpha": _finite(part["alpha"].mean()), "information_ratio": _finite(part["information_ratio_year"].median())})
    return rows


def _selection_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    frame = diagnostics.copy()
    if "agent" in frame and frame["agent"].eq("meta_final").any():
        frame = frame.loc[frame["agent"].eq("meta_final")]
    frame["year"] = pd.to_datetime(frame["prediction_date"]).dt.year
    return frame.loc[frame["year"].le(2024)]


def _summary(
    result: BacktestResult,
    diagnostics: pd.DataFrame,
    tail: pd.DataFrame,
    *,
    dataset_hash: str,
    evaluation_key: str,
    elapsed_seconds: float,
    dataset_reused: bool,
) -> dict[str, Any]:
    selected_ic = _selection_diagnostics(diagnostics)
    annual = result.annual_metrics.loc[result.annual_metrics["year"].le(2024)].copy()
    tail_selection = tail.copy()
    if not tail_selection.empty:
        date_column = "prediction_date" if "prediction_date" in tail_selection else "snapshot_date"
        tail_selection["year"] = pd.to_datetime(tail_selection[date_column]).dt.year
        tail_selection = tail_selection.loc[tail_selection["year"].le(2024)]
    rank_values = pd.to_numeric(selected_ic.get("rank_ic"), errors="coerce").dropna()
    tail_column = next(
        (name for name in ("top_decile_minus_universe", "top_decile_spread") if name in tail_selection),
        None,
    )
    tail_values = (
        pd.to_numeric(tail_selection[tail_column], errors="coerce").dropna()
        if tail_column else pd.Series(dtype=float)
    )
    era_rows = []
    for start, end in SELECTION_ERAS:
        ic = selected_ic.loc[selected_ic["year"].between(start, end), "rank_ic"].dropna()
        years = annual.loc[annual["year"].between(start, end)]
        era_rows.append({
            "era": f"{start}-{end}",
            "rank_ic": _finite(ic.mean()),
            "information_ratio": _finite(years["information_ratio_year"].median()),
            "mean_alpha": _finite(years["alpha"].mean()),
            "positive_alpha_years": int((years["alpha"] > 0).sum()),
        })
    known = result.annual_metrics.loc[result.annual_metrics["year"].isin(KNOWN_STRESS_YEARS)]
    selection_summary = dict(result.summary)
    selection_summary.update({
        "mean_rank_ic": _finite(rank_values.mean()),
        "rank_ic_positive_fraction": _finite((rank_values > 0).mean()),
        "rank_ic_std": _finite(rank_values.std(ddof=1)),
        "tail_spread": _finite(tail_values.mean()),
        "information_ratio": _finite(annual["information_ratio_year"].median()),
        "mean_annual_alpha": _finite(annual["alpha"].mean()),
        "worst_year_alpha": _finite(annual["alpha"].min()),
        "positive_alpha_eras": int(sum((row["mean_alpha"] or 0) > 0 for row in era_rows)),
        "stress_mean_alpha": _finite(known["alpha"].mean()),
    })
    rank_ic_by_cohort = [
        {"date": str(row.prediction_date), "rank_ic": _finite(row.rank_ic)}
        for row in selected_ic[["prediction_date", "rank_ic"]].itertuples(index=False)
    ]
    return {
        "evaluation_key": evaluation_key,
        "dataset_hash": dataset_hash,
        "source": "computed",
        "dataset_reused": dataset_reused,
        "elapsed_seconds": elapsed_seconds,
        "selection_until_year": 2024,
        "summary": selection_summary,
        "eras": era_rows,
        "rank_ic_by_cohort": rank_ic_by_cohort,
        "known_stress_not_selection": known.to_dict("records"),
    }


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _retain_evidence(
    destination: Path,
    prepared: Path,
    agent_dir: Path,
    result: BacktestResult,
    summary: Mapping[str, Any],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    keep = (
        "agent_scores.parquet", "meta_weights.parquet", "rank_ic_diagnostics.parquet",
        "rank_tail_diagnostics.parquet", "model_feature_attribution.parquet",
        "agent_local_attribution.parquet", "feature_diagnostics.parquet",
        "signal_health.parquet", "signal_calibration.parquet", "feature_catalog.json",
        "manifest.json",
    )
    for name in keep:
        source = agent_dir / name
        if source.exists():
            shutil.copy2(source, destination / name)
    write_parquet(result.equity, destination / "equity.parquet")
    write_parquet(result.annual_metrics, destination / "annual_metrics.parquet")
    write_parquet(result.orders, destination / "orders.parquet")
    write_parquet(result.positions, destination / "positions.parquet")
    write_json(dict(summary), destination / "summary.json")
    write_json(
        {"dataset_hash": summary["dataset_hash"], "prepared_path": str(prepared)},
        destination / "dataset_reference.json",
    )


def execute_model_study(study_id: str) -> dict[str, Any]:
    """Ejecuta de principio a fin la única ruta científica del proyecto."""
    from module.evaluation.profiles import PROFILE_NAMES
    from module.studies.catalog import BY_ID
    from module.studies.config import (
        diagnostic_portfolio_variables, initial_values, ordered_predictive_variables,
    )
    from module.studies.selection import choose_candidate
    from module.storage.datasets import validate_dataset_reference
    from module.storage.evidence import write_report, write_winner
    from module.storage.studies import (
        append_event, create_run, directory_size, find_run,
        read_study, safe_study_path, update_run, update_study, write_storage_manifest,
    )

    study = read_study(study_id)
    definition = study["configuration"]
    dev_scope = study.get("run_scope") == "dev"
    directory = safe_study_path(study_id)
    evidence = directory / "evidence"
    evidence.mkdir(exist_ok=True)
    values = initial_values(definition)
    if dev_scope:
        values["target_size"] = 3
    decisions: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    from module.storage.studies import list_runs
    completed = sum(run["status"] == "succeeded" for run in list_runs(study_id))

    def execute(
        logical_key: str,
        phase: str,
        variable_id: str,
        value: Any,
        configuration: Mapping[str, Any],
        *,
        seed: int = 42,
        profile: str = "balanced",
        overrides: Mapping[str, Any] | None = None,
        retain: Path | None = None,
        target_override: Path | None = None,
        target_identity: str = "real",
    ) -> dict[str, Any]:
        nonlocal completed
        previous = find_run(study_id, logical_key)
        if previous and previous["status"] == "succeeded" and previous.get("result"):
            append_event(study_id, "info", "run_reused", f"Run reutilizado: {logical_key}", run_id=previous["run_id"])
            return previous["result"]
        attempt = int(previous.get("attempt", 0)) + 1 if previous else 1
        run = create_run(
            study_id, logical_key=logical_key, phase=phase,
            variable_id=variable_id, value=value, configuration=configuration,
            attempt=attempt,
        )
        started = time.perf_counter()
        update_run(study_id, run["run_id"], status="running", stage="evaluation", progress=0.05)
        update_study(study_id, phase=phase, current_run_id=run["run_id"], heartbeat=_utc_now())
        append_event(
            study_id, "info", "run_started",
            f"Run {run['run_id']} · {phase} · {variable_id}={value}",
            run_id=run["run_id"], phase=phase, variable_id=variable_id,
        )
        try:
            runtime_overrides = dict(overrides or {})
            if dev_scope:
                runtime_overrides.update({
                    "run_scope": "dev",
                    "data_start_date": "2016-01-01",
                    "end_date": "2024-12-31",
                    "min_rank_ic_cross_section": 3,
                    "train_lookback_years": min(
                        int(configuration["train_lookback_years"]), 4,
                    ),
                    "lgbm_n_estimators": min(
                        int(configuration["lgbm_n_estimators"]), 30,
                    ),
                    "lgbm_min_child_samples": min(
                        int(configuration["lgbm_min_child_samples"]), 5,
                    ),
                    "lgbm_n_jobs": 1,
                })
                runtime_overrides.setdefault("target_size", 3)
            result = run_evaluation(
                configuration, random_seed=seed, profile=profile,
                overrides=runtime_overrides, retain_dir=retain,
                target_override=target_override, target_identity=target_identity,
            )
            elapsed = time.perf_counter() - started
            bytes_persisted = directory_size(retain) if retain and retain.exists() else 0
            update_run(
                study_id, run["run_id"], status="succeeded", stage="completed",
                progress=1.0, elapsed_seconds=elapsed,
                bytes_persisted=bytes_persisted, result=result,
            )
            completed += 1
            total = max(int(study.get("budget", {}).get("total_runs", 1)), 1)
            update_study(
                study_id, completed_runs=completed,
                progress=min(0.95, completed / total), heartbeat=_utc_now(),
            )
            append_event(
                study_id, "info", "run_succeeded",
                f"Run {run['run_id']} terminado en {elapsed:.1f}s",
                run_id=run["run_id"], source=result.get("source"),
            )
            return result
        except Exception as exc:
            update_run(
                study_id, run["run_id"], status="failed", stage="failed",
                elapsed_seconds=time.perf_counter() - started,
                error=str(exc), traceback=traceback.format_exc(),
            )
            update_study(study_id, status="failed", error=str(exc), traceback=traceback.format_exc())
            append_event(study_id, "error", "run_failed", f"Run {run['run_id']} falló: {exc}", run_id=run["run_id"])
            raise

    update_study(study_id, status="running", phase="predictive", progress=0.01)
    append_event(study_id, "info", "study_started", "Comienza el Model Study.")
    baseline = execute("predictive:baseline", "temporal", "baseline", "baseline", values)
    incumbent = {
        "candidate_id": "predictive:baseline",
        "value": "baseline",
        "result": baseline,
        "configuration": dict(values),
    }
    ledger.append(_ledger_row(incumbent, "temporal", "baseline", selected=True))

    for variable_id in ordered_predictive_variables(definition):
        spec = BY_ID[variable_id]
        candidates = []
        for value in definition[variable_id]["values"]:
            configuration = dict(values)
            configuration[variable_id] = value
            logical_key = f"predictive:{variable_id}:{_key_value(value)}"
            if configuration == incumbent["configuration"]:
                result = incumbent["result"]
                candidate_id = incumbent["candidate_id"]
            else:
                result = execute(
                    logical_key, spec.stage, variable_id, value, configuration,
                )
                candidate_id = logical_key
            candidates.append({
                "candidate_id": candidate_id,
                "value": value,
                "result": result,
                "configuration": configuration,
            })
        comparison_incumbent = next(
            (candidate for candidate in candidates if candidate["configuration"] == incumbent["configuration"]),
            incumbent,
        )
        decision = choose_candidate(comparison_incumbent, candidates, variable_id)
        winner = next(
            candidate for candidate in candidates
            if candidate["candidate_id"] == decision["winner_candidate_id"]
        )
        values = dict(winner["configuration"])
        incumbent = winner
        decisions.append(decision)
        for candidate in candidates:
            ledger.append(_ledger_row(
                candidate, spec.stage, variable_id,
                selected=candidate["candidate_id"] == winner["candidate_id"],
                decision=decision,
            ))
        append_event(
            study_id, "info", "decision",
            f"{spec.label}: elegido {winner['value']} solo por Rank-IC.",
            variable_id=variable_id, winner=winner["value"],
        )

    write_parquet(pd.DataFrame(ledger), directory / "evaluation_ledger.parquet")
    write_json({"decisions": decisions}, directory / "decisions.json")
    winner_result = execute(
        "winner:evidence", "winner", "winner", "final",
        values, retain=evidence,
    )
    winner_run = find_run(study_id, "winner:evidence")
    winner_payload = {
        "study_id": study_id,
        "winner_run_id": winner_run["run_id"] if winner_run else incumbent["candidate_id"],
        "configuration": values,
        "selection_metric": "rank_ic_only",
        "selection_until_year": 2024,
        "known_stress_role": "known_stress_not_selection",
        "summary": winner_result,
    }
    write_winner(directory / "winner.json", winner_payload)

    append_event(study_id, "info", "post_selection", "Ganador congelado; comienzan diagnósticos informativos.")
    portfolio_rows = []
    for variable_id in diagnostic_portfolio_variables(definition):
        base = values[variable_id]
        for value in definition[variable_id]["values"]:
            if value == base:
                continue
            override = {variable_id: value}
            result = execute(
                f"portfolio:{variable_id}:{_key_value(value)}",
                "portfolio", variable_id, value, values, overrides=override,
            )
            portfolio_rows.append({
                "variable_id": variable_id,
                "base_value": json.dumps(base, ensure_ascii=False),
                "diagnostic_value": json.dumps(value, ensure_ascii=False),
                **result["summary"],
            })
    write_parquet(pd.DataFrame(portfolio_rows), directory / "portfolio_comparison.parquet")

    profile_rows = []
    profiles_root = evidence / "profiles"
    for profile in PROFILE_NAMES:
        logical = f"profile:{profile}"
        run = create_run(
            study_id, logical_key=logical, phase="profiles",
            variable_id="profile", value=profile, configuration=values,
            attempt=1,
        ) if not find_run(study_id, logical) else find_run(study_id, logical)
        if run["status"] == "succeeded" and run.get("result"):
            result = run["result"]
        else:
            update_run(study_id, run["run_id"], status="running", stage="backtest", progress=0.1)
            append_event(study_id, "info", "profile_started", f"Perfil {profile}.", run_id=run["run_id"])
            result = run_profile_evaluation(values, profile, evidence, profiles_root / profile)
            update_run(study_id, run["run_id"], status="succeeded", stage="completed", progress=1.0, result=result)
            completed += 1
        profile_rows.append({"profile": profile, **result["summary"]})
    write_parquet(pd.DataFrame(profile_rows), directory / "profile_comparison.parquet")

    dev = dev_scope
    diagnostics = pd.read_parquet(evidence / "rank_ic_diagnostics.parquet")
    scores = pd.read_parquet(evidence / "agent_scores.parquet")
    reference = json.loads((evidence / "dataset_reference.json").read_text(encoding="utf-8"))
    prepared = validate_dataset_reference(reference["dataset_hash"])
    targets = pd.read_parquet(prepared / "targets_forward.parquet")
    robustness = _robustness_base(
        diagnostics, scores, targets, prepared, evidence, values, winner_result, dev=dev,
    )
    for seed in (7, 2026):
        result = execute(f"robustness:seed:{seed}", "robustness", "seed", seed, values, seed=seed)
        robustness["seeds"].append({"seed": seed, "summary": result["summary"]})
    placebo_count = 2 if dev else 5
    for seed in range(101, 101 + placebo_count):
        shuffled = _shuffle_targets(targets, seed)
        temporary = Path(tempfile.mkdtemp(prefix=f"placebo-{seed}-"))
        try:
            target_path = temporary / "targets_forward.parquet"
            write_parquet(shuffled, target_path)
            result = execute(
                f"robustness:placebo:{seed}", "robustness", "label_placebo", seed,
                values, seed=seed, target_override=target_path,
                target_identity=f"placebo-{seed}",
            )
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        robustness["label_placebos"].append({"seed": seed, "summary": result["summary"]})
    write_json(robustness, directory / "robustness.json")

    update_study(
        study_id, status="succeeded", phase="completed", progress=1.0,
        winner_run_id=winner_payload["winner_run_id"], completed_runs=completed,
        worker_pid=None, heartbeat=_utc_now(), error=None, traceback=None,
    )
    final = {**winner_payload, "status": "succeeded", "robustness": robustness}
    write_report(directory / "report.md", final)
    write_storage_manifest(study_id)
    append_event(study_id, "info", "study_succeeded", "Model Study terminado correctamente.")
    return final


def _ledger_row(
    candidate: Mapping[str, Any],
    phase: str,
    variable_id: str,
    *,
    selected: bool,
    decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = candidate["result"]
    return {
        "logical_key": candidate["candidate_id"],
        "phase": phase,
        "variable_id": variable_id,
        "candidate_value": json.dumps(candidate["value"], ensure_ascii=False),
        "selected": selected,
        "selection_metric": "rank_ic_only",
        "mean_rank_ic": result["summary"].get("mean_rank_ic"),
        "positive_fraction": result["summary"].get("rank_ic_positive_fraction"),
        "rank_ic_std": result["summary"].get("rank_ic_std"),
        "eras": json.dumps(result.get("eras", []), ensure_ascii=False),
        "reason": next(
            (
                row["reason"] for row in (decision or {}).get("candidates", [])
                if row["candidate_id"] == candidate["candidate_id"]
            ),
            "Baseline inicial.",
        ),
        "source": result.get("source"),
        "elapsed_seconds": result.get("elapsed_seconds"),
    }


def _shuffle_targets(targets: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = targets.copy()
    columns = ["forward_return", "forward_benchmark_return", "forward_excess_return"]
    for _, indices in frame.groupby("snapshot_date").groups.items():
        positions = np.asarray(list(indices))
        frame.loc[positions, columns] = frame.loc[positions, columns].to_numpy()[
            rng.permutation(len(positions))
        ]
    return frame


def _robustness_base(
    diagnostics: pd.DataFrame,
    scores: pd.DataFrame,
    targets: pd.DataFrame,
    prepared: Path,
    evidence: Path,
    values: Mapping[str, Any],
    winner_result: Mapping[str, Any],
    *,
    dev: bool,
) -> dict[str, Any]:
    """Construye el bloque de robustez sin seeds ni placebos (esos requieren reentrenar)."""
    from module.research.robustness import bootstrap_and_eras, random_portfolios, score_permutation

    iterations = 199 if dev else 9_999
    bootstrap_iterations = 200 if dev else 2_000
    random_iterations = 100 if dev else 1_000
    return {
        "scientific_selection_effect": "none",
        "dev_only_not_scientific": dev,
        "bootstrap_and_era_exclusion": bootstrap_and_eras(
            diagnostics, iterations=bootstrap_iterations,
        ),
        "permutation": score_permutation(
            scores, targets, iterations=iterations,
            minimum_cross_section=3 if dev else 8,
        ),
        "random_portfolios": random_portfolios(
            prepared, evidence, values, simulations=random_iterations,
        ),
        "seeds": [],
        "label_placebos": [],
        "agent_rank_ic": _agent_summary(diagnostics),
        "meta_weight_stability": _meta_stability(evidence / "meta_weights.parquet"),
        "known_stress_not_selection": winner_result.get("known_stress_not_selection", []),
    }


def _agent_summary(diagnostics: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "agent": str(agent),
            "mean_rank_ic": _finite(group["rank_ic"].mean()),
            "positive_fraction": _finite((group["rank_ic"] > 0).mean()),
            "observations": int(group["rank_ic"].notna().sum()),
        }
        for agent, group in diagnostics.groupby("agent")
    ]


def _meta_stability(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    frame = pd.read_parquet(path)
    if frame.empty or not {"snapshot_date", "agent", "weight"}.issubset(frame.columns):
        return {"available": False}
    weights = (
        frame.pivot(index="snapshot_date", columns="agent", values="weight")
        .fillna(0.0)
        .sort_index()
    )
    turnover = weights.diff().abs().sum(axis=1).iloc[1:] / 2.0
    return {
        "available": True,
        "rows": int(len(frame)),
        "mean_concentration": _finite((weights ** 2).sum(axis=1).mean()),
        "mean_turnover": _finite(turnover.mean()) if not turnover.empty else 0.0,
    }


def _key_value(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]


def _utc_now() -> str:
    from module.storage.studies import utc_now
    return utc_now()
