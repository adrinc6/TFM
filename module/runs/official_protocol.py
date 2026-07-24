"""Protocolo confirmatorio oficial de 48 evaluaciones.

El study manual conserva su exploración amplia. Este módulo ejecuta una lista cerrada, impide
usar 2025-2026 para elegir y materializa solo un final de evidencia.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import socket
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from environment import Settings
from module.common.utils import write_json, write_parquet
from module.evaluation.backtest import run_backtest
from module.evaluation.portfolio import active_fraction
from module.evaluation.profiles import PROFILE_NAMES
from module.evaluation.signal_diagnostics import (
    SELECTION_ERAS, era_summary, holm_adjust, moving_block_bootstrap_delta, summarize_tail,
)
from module.modeling.targets import (
    TARGET_ARTIFACT_NAME, normalize_target_columns, target_artifact_path,
)
from module.runs.recycle import cache_dir
from module.runs.results_store import ResultsStore
from module.scenarios.variables import (
    OFFICIAL_PORTFOLIO_STRUCTURES, OFFICIAL_SIGNAL_CHALLENGERS, OFFICIAL_STUDY_PROTOCOL,
    official_evaluation_budget, validate_official_budget,
)


def official_preflight(settings: Settings, store: ResultsStore | None = None) -> dict[str, Any]:
    """Valida el protocolo sin crear studies, runs ni artefactos."""
    return _preflight(settings, store or ResultsStore())


def run_official_protocol(
    settings: Settings, store: ResultsStore, *, name: str, hypothesis: str = "",
    resume_study_id: str | None = None,
) -> str:
    """Ejecuta el protocolo fijo y deja un estado terminal aunque falle."""
    previous = {
        path.name for path in store.studies_root.iterdir()
    } if store.studies_root.exists() else set()
    try:
        return _run_official_protocol(
            settings, store, name=name, hypothesis=hypothesis,
            resume_study_id=resume_study_id,
        )
    except Exception as exc:
        candidates = [resume_study_id] if resume_study_id else [
            path.name for path in store.studies_root.iterdir()
            if path.name not in previous
        ]
        for study_id in candidates:
            if not study_id:
                continue
            manifest_path = store.studies_root / study_id / "study_manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") not in {"succeeded", "cancelled"}:
                store.update_study_status(
                    study_id, "failed", current_phase="failed", error=str(exc),
                )
        raise


def _run_official_protocol(
    settings: Settings, store: ResultsStore, *, name: str, hypothesis: str = "",
    resume_study_id: str | None = None,
) -> str:
    """Implementación cerrada; no admite catálogo ni expansión dinámica."""
    from module.runs.execution import execute_run

    preflight = official_preflight(settings, store)
    payload = {
        "name": name.strip() or "optimization-official",
        "kind": "optimization",
        "description": hypothesis.strip() or (
            "Protocolo confirmatorio: señal adaptable, traducción a alfa y robustez."
        ),
        "hypothesis": hypothesis.strip(),
        "tags": ["official", "optimization", "confirmatory-v2"],
        "strategy": "confirmatory_rank_to_alpha",
        "protocol_version": 2,
        "owner_pid": os.getpid(),
        "owner_host": socket.gethostname(),
        "selection_metric": "paired_rank_ic_tail_and_net_information_ratio",
        "selection_eras": [list(era) for era in SELECTION_ERAS],
        "known_stress_years": [2025, 2026],
        "evaluation_budget": preflight["evaluation_budget"],
        "fit_budget": preflight["fit_budget"],
        "storage_preflight": preflight["storage"],
    }
    if resume_study_id:
        study_id, study_dir = store.resume_study(resume_study_id)
    else:
        study_id, study_dir = store.create_study(payload)
    store.update_study_status(
        study_id, "running", current_phase="signal_challengers",
        current_scenario=None, completed_evaluations=0,
        owner_pid=os.getpid(), owner_host=socket.gethostname(),
    )
    ledger: list[dict[str, Any]] = []

    signal_records: list[dict[str, Any]] = []
    signal_settings: dict[str, Settings] = {}
    for candidate in OFFICIAL_SIGNAL_CHALLENGERS:
        candidate_name = str(candidate["name"])
        candidate_settings = replace(settings, **dict(candidate["overrides"]))
        signal_settings[candidate_name] = candidate_settings
        run_id = execute_run(
            candidate_settings, mode="full", run_kind="scenario", study_id=study_id,
            label=f"{name} · señal · {candidate_name}",
            description="Challenger de señal pre-registrado.",
            tags=["official", "signal_challenger"],
            grid_definition={"phase": "signal_challengers", "candidate": candidate_name,
                             "overrides": dict(candidate["overrides"])},
            store=store, retention_policy="compact_candidate",
        )
        store.add_to_study(study_id, run_id)
        record = _signal_record(store, run_id, candidate_name)
        signal_records.append(record)
        ledger.append(_ledger("signal_challengers", candidate_name, run_id, record))
        store.update_study_status(
            study_id, "running", current_phase="signal_challengers",
            current_scenario=candidate_name, completed_evaluations=len(ledger),
        )

    signal_records, signal_winner_name = _select_signal(signal_records, store)
    signal_winner = next(row for row in signal_records if row["scenario"] == signal_winner_name)
    model_final_id = str(signal_winner["run_id"])
    final_model_settings = signal_settings[signal_winner_name]
    _promote_to_evidence(store, model_final_id)
    final_agent_dir = store.runs_root / model_final_id / "artifacts"
    write_parquet(pd.DataFrame(signal_records), study_dir / "signal_comparison.parquet")

    # Dos semillas confirmatorias: jamás se elige la más favorable.
    store.update_study_status(study_id, "running", current_phase="seed_confirmation")
    seed_records: list[dict[str, Any]] = []
    for seed in (7, 2026):
        seed_settings = replace(final_model_settings, random_seed=seed)
        run_id = execute_run(
            seed_settings, mode="full", run_kind="stress", study_id=study_id,
            label=f"{name} · semilla {seed}", description="Confirmación, no selección.",
            tags=["official", "seed_confirmation"],
            grid_definition={"phase": "seed_confirmation", "seed": seed},
            store=store, retention_policy="compact_candidate",
        )
        store.add_to_study(study_id, run_id)
        record = _signal_record(store, run_id, f"seed_{seed}")
        seed_records.append(record)
        ledger.append(_ledger("seed_confirmation", f"seed_{seed}", run_id, record))
        store.update_study_status(
            study_id, "running", current_phase="seed_confirmation",
            current_scenario=f"seed_{seed}", completed_evaluations=len(ledger),
        )

    store.update_study_status(study_id, "running", current_phase="portfolio_translation",
                              completed_evaluations=len(ledger))
    portfolio_records, portfolio_settings_by_name, portfolio_winner_name = _run_portfolio_translation(
        final_model_settings, final_agent_dir, store, study_id, name, ledger,
    )
    portfolio_winner = next(
        row for row in portfolio_records if row["scenario"] == portfolio_winner_name
    )
    final_portfolio_settings = portfolio_settings_by_name[str(portfolio_winner["scenario"])]
    portfolio_final_id = str(portfolio_winner["run_id"])
    write_parquet(pd.DataFrame(portfolio_records), study_dir / "portfolio_comparison.parquet")
    _promote_portfolio_to_evidence(store, portfolio_final_id, model_final_id)
    final_agent_dir = store.runs_root / portfolio_final_id / "artifacts"

    # Perfiles: misma estructura y ruta de exposición, siempre equal y sin capacidad de selección.
    store.update_study_status(study_id, "running", current_phase="profiles")
    profile_records: list[dict[str, Any]] = []
    profile_run_ids: dict[str, str] = {}
    for profile in PROFILE_NAMES:
        profile_settings = replace(final_portfolio_settings, profile=profile, sizing_mode="equal")
        run_id = execute_run(
            profile_settings, mode="backtest", run_kind="scenario", study_id=study_id,
            label=f"{name} · perfil {profile}", description="Perfil informativo; no selecciona.",
            tags=["official", "profile"],
            grid_definition={"phase": "profiles", "profile": profile,
                             "parent_run_id": portfolio_final_id},
            store=store, agent_dir=final_agent_dir, retention_policy="compact_backtest",
        )
        store.add_to_study(study_id, run_id)
        record = _portfolio_record(store, run_id, profile)
        record["selection_policy"] = "reported_not_optimized"
        profile_records.append(record)
        profile_run_ids[profile] = run_id
        ledger.append(_ledger("profiles", profile, run_id, record))
        store.update_study_status(
            study_id, "running", current_phase="profiles",
            current_scenario=profile, completed_evaluations=len(ledger),
        )
    write_parquet(pd.DataFrame(profile_records), study_dir / "profile_comparison.parquet")

    # Seis stresses económicos no seleccionables.
    store.update_study_status(study_id, "running", current_phase="stress_and_robustness")
    stress_records = _run_stresses(
        final_portfolio_settings, final_agent_dir, store, study_id, name, ledger,
    )
    write_parquet(pd.DataFrame(stress_records), study_dir / "stress_results.parquet")

    robustness, robustness_ledger = _robustness(
        store, portfolio_final_id, portfolio_final_id, final_portfolio_settings,
    )
    ledger.extend(robustness_ledger)
    store.update_study_status(
        study_id, "running", current_phase="stress_and_robustness",
        current_scenario="statistical_robustness", completed_evaluations=len(ledger),
    )
    if len(ledger) != 48:
        raise RuntimeError(f"Ledger oficial inválido: se esperaban 48 evaluaciones y hay {len(ledger)}.")
    for index, row in enumerate(ledger, start=1):
        row["evaluation_index"] = index
    write_parquet(pd.DataFrame(ledger), study_dir / "evaluation_ledger.parquet")
    write_parquet(_comparison_frame(ledger), study_dir / "comparison_data.parquet")
    write_json({
        "selection_eras": [list(era) for era in SELECTION_ERAS],
        "known_stress_years": [2025, 2026],
        "known_stress_policy": "known_stress_not_selection",
    }, study_dir / "selection_folds.json")
    write_json(robustness, study_dir / "robustness.json")

    known_stress = _known_stress(store, portfolio_final_id, portfolio_final_id)
    verdict = _verdict(signal_records, signal_winner, portfolio_records, portfolio_winner)
    _compact_promoted_model(store, model_final_id, portfolio_final_id)
    storage = _storage_manifest(store, study_dir, ledger, portfolio_final_id)
    fit_consumption = _fit_consumption(store, ledger, robustness)
    decision = {
        "schema_version": 2,
        "strategy": "confirmatory_rank_to_alpha",
        "hypothesis": hypothesis.strip(),
        "verdict": verdict,
        "incumbent": signal_records[0],
        "signal_candidates": signal_records,
        "signal_winner": signal_winner,
        "portfolio_candidates": portfolio_records,
        "portfolio_winner": portfolio_winner,
        "model_final_run_id": model_final_id,
        "final_run_id": portfolio_final_id,
        "final_settings": asdict(final_portfolio_settings),
        "seed_confirmation": seed_records,
        "profile_run_ids": profile_run_ids,
        "profile_reference": "balanced",
        "profile_selection_policy": "reported_not_optimized",
        "known_stress_2025_2026": {
            "status": "known_stress_not_selection", **known_stress,
        },
        "evaluation_budget": preflight["evaluation_budget"],
        "evaluation_consumed": len(ledger),
        "fit_budget": preflight["fit_budget"],
        "fit_consumption": fit_consumption,
        "robustness": robustness,
        "storage": storage,
    }
    write_json(decision, study_dir / "decision.json")
    # Segunda pasada: incluye decision.json y storage_manifest.json en los bytes persistidos.
    for _ in range(2):
        storage = _storage_manifest(store, study_dir, ledger, portfolio_final_id)
        decision["storage"] = storage
        write_json(decision, study_dir / "decision.json")
    if not storage["within_limit"]:
        raise RuntimeError("El study oficial supera 5 GiB persistidos tras la publicación.")
    store.update_study_status(
        study_id, "succeeded", current_phase="complete",
        current_scenario=None, completed_evaluations=len(ledger), verdict=verdict,
        model_final_run_id=model_final_id, final_run_id=portfolio_final_id,
    )
    return study_id


def _preflight(settings: Settings, store: ResultsStore) -> dict[str, Any]:
    budget = official_evaluation_budget()
    expensive = int(OFFICIAL_STUDY_PROTOCOL["estimated_expensive_fits"])
    if expensive > int(budget["max_expensive_fits"]):
        raise RuntimeError("El protocolo supera el límite de walk-forwards caros.")
    # Estimación conservadora. La telemetría histórica se usa para informar, no para relajar
    # ninguno de los límites duros.
    estimated_bytes = 4 * 1024**3
    validate_official_budget(
        dict(budget["breakdown"]), estimated_expensive_fits=expensive,
        estimated_incremental_bytes=estimated_bytes,
    )
    raw = settings.raw_output_dir
    missing = [
        str(raw / name) for name in (
            "finnhub_metrics.parquet", "prices.parquet", "profiles.parquet",
            "report_dates.parquet",
        ) if not (raw / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "El preflight no puede reconstruir el incumbent; faltan inputs: " + ", ".join(missing)
        )
    historical_wall_seconds: list[float] = []
    if store.runs_root.exists():
        for manifest_path in store.runs_root.glob("*/run_manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                execution = manifest.get("execution", {})
                agent_source = execution.get("stage_source", {}).get("agents")
                agent_seconds = execution.get("stage_timings_seconds", {}).get("agents")
                if (
                    agent_source == "computed"
                    and agent_seconds is not None
                    and float(agent_seconds) > 0
                ):
                    historical_wall_seconds.append(float(agent_seconds))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
    median_wall = float(np.median(historical_wall_seconds)) if historical_wall_seconds else None
    estimated_wall = median_wall * expensive if median_wall is not None else None
    return {
        "evaluation_budget": budget,
        "fit_budget": {
            "maximum": int(budget["max_expensive_fits"]),
            "estimated_new": expensive,
            "projected_unique_keys": {
                "dataset": 1,
                "features": 1,
                "agents_fit": 10,
                "agents": 19,
            },
            "recycled_signal_scenarios": 9,
        },
        "time": {
            "historical_runs_sampled": len(historical_wall_seconds),
            "median_historical_expensive_fit_seconds": median_wall,
            "estimated_new_fit_wall_seconds": estimated_wall,
        },
        "storage": {
            "maximum_bytes": int(budget["max_incremental_bytes"]),
            "estimated_incremental_bytes": estimated_bytes,
        },
    }


def _signal_record(store: ResultsStore, run_id: str, scenario: str) -> dict[str, Any]:
    artifacts = store.runs_root / run_id / "artifacts"
    summary = json.loads((artifacts / "backtest_summary.json").read_text(encoding="utf-8"))
    tail = pd.read_parquet(artifacts / "rank_tail_diagnostics.parquet")
    tail = tail.loc[pd.to_datetime(tail["prediction_date"]).dt.year <= 2024].copy()
    eras = era_summary(tail)
    weights_path = artifacts / "meta_weights.parquet"
    weight_turnover = 0.0
    if weights_path.exists():
        weights = pd.read_parquet(weights_path)
        pivot = weights.pivot_table(index="snapshot_date", columns="agent", values="weight")
        weight_turnover = float(pivot.sort_index().diff().abs().sum(axis=1).mean())
    result = {
        "scenario": scenario, "run_id": run_id,
        "mean_rank_ic": float(tail["rank_ic"].mean()),
        "rank_ic_positive_fraction": float((tail["rank_ic"] > 0).mean()),
        "meta_weight_turnover": weight_turnover,
        **summarize_tail(tail),
        "era_metrics": eras.to_dict("records"),
        "eligible": scenario == "incumbent_expanding",
        "rejection_reasons": [],
        "holm_adjusted_p": None,
    }
    result.update({key: summary.get(key) for key in (
        "information_ratio", "mean_annual_alpha", "annualized_turnover",
    )})
    return result


def _select_signal(
    records: list[dict[str, Any]], store: ResultsStore,
) -> tuple[list[dict[str, Any]], str]:
    incumbent = records[0]
    incumbent_tail = _tail_for_record(incumbent, store)
    raw_p: list[float] = []
    comparisons: list[dict[str, Any]] = []
    for record in records[1:]:
        challenger_tail = _tail_for_record(record, store)
        paired = incumbent_tail.merge(
            challenger_tail, on="prediction_date", suffixes=("_inc", "_challenger"),
        )
        boot = moving_block_bootstrap_delta(
            paired["rank_ic_inc"], paired["rank_ic_challenger"], block_size=12, confidence=0.90,
        )
        # Superioridad formal solo si el intervalo queda por encima de cero. La corrección Holm
        # evita convertir el máximo de once alternativas en evidencia espuria.
        raw_p.append(float(boot.get("p_superiority") or 1.0))
        reasons: list[str] = []
        if record["mean_rank_ic"] < incumbent["mean_rank_ic"] - 0.005:
            reasons.append("rank_ic_inferior")
        if record["rank_ic_positive_fraction"] < incumbent["rank_ic_positive_fraction"] - 0.03:
            reasons.append("positive_fraction_inferior")
        if any(
            float(era.get("mean_rank_ic") or -1) <= -0.02 for era in record["era_metrics"]
        ):
            reasons.append("era_rank_ic_bajo")
        if float(boot.get("ci_low") or -1) <= -0.01:
            reasons.append("bootstrap_no_inferioridad_falla")
        if float(record.get("top_decile_minus_universe") or -1) < float(
            incumbent.get("top_decile_minus_universe") or -1
        ):
            reasons.append("cola_no_mejora")
        comparisons.append({"record": record, "bootstrap": boot, "reasons": reasons})
    adjusted = holm_adjust(raw_p)
    for comparison, adjusted_p in zip(comparisons, adjusted, strict=True):
        record = comparison["record"]
        record["paired_bootstrap"] = comparison["bootstrap"]
        record["holm_adjusted_p"] = adjusted_p
        record["rejection_reasons"] = comparison["reasons"]
        record["eligible"] = not comparison["reasons"]
        statistically_superior = adjusted_p < 0.10
        simpler = _signal_complexity(record) < _signal_complexity(incumbent)
        less_variable = (
            float(record["meta_weight_turnover"])
            < float(incumbent["meta_weight_turnover"]) - 1e-12
        )
        record["replacement_basis"] = (
            "holm_superior" if statistically_superior
            else "non_inferior_simpler" if simpler
            else "non_inferior_less_variable" if less_variable
            else None
        )
        if record["eligible"] and record["replacement_basis"] is None:
            record["eligible"] = False
            record["rejection_reasons"].append("sin_superioridad_ni_simplificacion")
    eligible = [record for record in records if record["eligible"]]
    winner = eligible[0]
    for candidate in eligible[1:]:
        candidate_tail = _median_era(candidate, "top_decile_minus_universe")
        winner_tail = _median_era(winner, "top_decile_minus_universe")
        if candidate_tail > winner_tail + 0.005:
            winner = candidate
        elif abs(candidate_tail - winner_tail) <= 0.005:
            if candidate["mean_rank_ic"] > winner["mean_rank_ic"] + 1e-12:
                winner = candidate
            elif abs(candidate["mean_rank_ic"] - winner["mean_rank_ic"]) <= 1e-12:
                if candidate["meta_weight_turnover"] < winner["meta_weight_turnover"]:
                    winner = candidate
    return records, str(winner["scenario"])


def _signal_complexity(record: Mapping[str, Any]) -> int:
    name = str(record.get("scenario") or "")
    if "meta_equal" in name:
        return 0
    if "meta_rank_ic" in name:
        return 1
    if "rolling" in name:
        return 2
    if "exponential" in name:
        return 3
    if "incumbent" in name:
        return 4
    return 5


def _tail_for_record(record: Mapping[str, Any], store: ResultsStore) -> pd.DataFrame:
    run_id = str(record["run_id"])
    root = store.runs_root / run_id / "artifacts"
    return pd.read_parquet(root / "rank_tail_diagnostics.parquet").loc[
        lambda frame: pd.to_datetime(frame["prediction_date"]).dt.year <= 2024
    ]


def _median_era(record: Mapping[str, Any], key: str) -> float:
    values = [float(era[key]) for era in record["era_metrics"] if era.get(key) is not None]
    return float(np.median(values)) if values else float("-inf")


def _promote_to_evidence(store: ResultsStore, run_id: str) -> None:
    """Materializa el candidato ya calculado desde caché; no ejecuta ninguna etapa."""
    run_dir = store.runs_root / run_id
    artifacts = run_dir / "artifacts"
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = manifest.get("execution", {}).get("stage_keys", {})
    wanted_agent = (
        "agent_scores.parquet", "meta_weights.parquet", "rank_ic_diagnostics.parquet",
        "rank_tail_diagnostics.parquet", "signal_health.parquet", "signal_calibration.parquet",
        "model_feature_attribution.parquet", "agent_local_attribution.parquet",
        "feature_diagnostics.parquet", "feature_catalog.json", "manifest.json",
    )
    agents_cache = cache_dir("agents", str(keys.get("agents", "")))
    agent_dirs = [path for path in agents_cache.iterdir() if path.is_dir()] if agents_cache.exists() else []
    if len(agent_dirs) != 1:
        raise FileNotFoundError("No se puede promover el ganador: falta su caché de agentes.")
    for name in wanted_agent:
        source = agent_dirs[0] / name
        if source.exists() and not (artifacts / name).exists():
            shutil.copy2(source, artifacts / name)
    for stage, names in {
        "dataset": (
            "asset_price_point_in_time.parquet", "benchmark_point_in_time.parquet",
            "panel_point_in_time.parquet",
        ),
        "features": ("features_point_in_time.parquet", TARGET_ARTIFACT_NAME),
    }.items():
        source_root = cache_dir(stage, str(keys.get(stage, "")))
        for name in names:
            source = source_root / name
            target_name = "stock_panel.parquet" if name == "panel_point_in_time.parquet" else name
            if source.exists() and not (artifacts / target_name).exists():
                shutil.copy2(source, artifacts / target_name)
    manifest["outputs"]["retention_policy"] = "evidence_final"
    manifest["outputs"]["promoted_without_retraining"] = True
    manifest["outputs"]["artifacts"] = sorted(path.name for path in artifacts.iterdir() if path.is_file())
    persisted_bytes = sum(path.stat().st_size for path in artifacts.rglob("*") if path.is_file())
    size_limit = 250 * 1024**2
    if persisted_bytes > size_limit:
        raise RuntimeError(
            f"El final de evidencia supera 250 MiB: {persisted_bytes} bytes."
        )
    manifest["outputs"]["persisted_bytes"] = persisted_bytes
    manifest["outputs"]["size_limit_bytes"] = size_limit
    write_json(manifest, manifest_path)


def _promote_portfolio_to_evidence(
    store: ResultsStore, portfolio_run_id: str, model_run_id: str,
) -> None:
    """Une señal y cartera ya calculadas en el único run final auditable."""
    portfolio_dir = store.runs_root / portfolio_run_id
    portfolio_artifacts = portfolio_dir / "artifacts"
    model_artifacts = store.runs_root / model_run_id / "artifacts"
    model_evidence = (
        "agent_scores.parquet", "meta_weights.parquet", "rank_ic_diagnostics.parquet",
        "rank_tail_diagnostics.parquet", "signal_health.parquet",
        "signal_calibration.parquet", "model_feature_attribution.parquet",
        "agent_local_attribution.parquet", "feature_diagnostics.parquet",
        "feature_catalog.json", "asset_price_point_in_time.parquet",
        "benchmark_point_in_time.parquet", "stock_panel.parquet",
        TARGET_ARTIFACT_NAME,
    )
    for name in model_evidence:
        source = model_artifacts / name
        if source.exists():
            shutil.copy2(source, portfolio_artifacts / name)
    manifest_path = portfolio_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    persisted_bytes = sum(
        path.stat().st_size for path in portfolio_artifacts.rglob("*") if path.is_file()
    )
    size_limit = 250 * 1024**2
    if persisted_bytes > size_limit:
        raise RuntimeError(
            f"El final de evidencia supera 250 MiB: {persisted_bytes} bytes."
        )
    manifest["outputs"].update({
        "retention_policy": "evidence_final",
        "promoted_without_retraining": True,
        "model_evidence_source_run_id": model_run_id,
        "artifacts": sorted(
            path.name for path in portfolio_artifacts.iterdir() if path.is_file()
        ),
        "persisted_bytes": persisted_bytes,
        "size_limit_bytes": size_limit,
    })
    write_json(manifest, manifest_path)


def _compact_promoted_model(
    store: ResultsStore, model_run_id: str, final_run_id: str,
) -> None:
    """Retira duplicados del modelo solo si el final conserva exactamente los mismos bytes."""
    from module.common.utils import sha256_file

    model_dir = store.runs_root / model_run_id
    model_artifacts = model_dir / "artifacts"
    final_artifacts = store.runs_root / final_run_id / "artifacts"
    compact_names = {
        "backtest_summary.json", "rank_ic_diagnostics.parquet", "meta_weights.parquet",
        "rank_tail_diagnostics.parquet", "annual_metrics.parquet", "manifest.json",
    }
    for source in model_artifacts.iterdir():
        if not source.is_file() or source.name in compact_names:
            continue
        preserved = final_artifacts / source.name
        if preserved.exists() and sha256_file(source) == sha256_file(preserved):
            source.unlink()
    manifest_path = model_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    persisted_bytes = sum(
        path.stat().st_size for path in model_artifacts.rglob("*") if path.is_file()
    )
    manifest["outputs"].update({
        "retention_policy": "compact_candidate",
        "demoted_after_final_promotion": True,
        "evidence_final_run_id": final_run_id,
        "artifacts": sorted(
            path.name for path in model_artifacts.iterdir() if path.is_file()
        ),
        "persisted_bytes": persisted_bytes,
        "size_limit_bytes": 5 * 1024**2,
    })
    if persisted_bytes > 5 * 1024**2:
        raise RuntimeError(
            f"El candidato compacto supera 5 MiB tras promover el final: {persisted_bytes} bytes."
        )
    write_json(manifest, manifest_path)


def _run_portfolio_translation(
    model_settings: Settings, agent_dir: Path, store: ResultsStore, study_id: str,
    name: str, ledger: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Settings], str]:
    from module.runs.execution import execute_run

    records: list[dict[str, Any]] = []
    settings_by_name: dict[str, Settings] = {}

    def evaluate(scenario: str, overrides: Mapping[str, Any], family: str) -> dict[str, Any]:
        candidate_settings = replace(model_settings, **dict(overrides))
        run_id = execute_run(
            candidate_settings, mode="backtest", run_kind="scenario", study_id=study_id,
            label=f"{name} · cartera · {scenario}", description="Traducción a alfa pre-registrada.",
            tags=["official", "portfolio_translation", family],
            grid_definition={"phase": "portfolio_translation", "family": family,
                             "scenario": scenario, "overrides": dict(overrides)},
            store=store, agent_dir=agent_dir, retention_policy="compact_backtest",
        )
        store.add_to_study(study_id, run_id)
        record = _portfolio_record(store, run_id, scenario)
        record["family"] = family
        records.append(record)
        settings_by_name[scenario] = candidate_settings
        ledger.append(_ledger("portfolio_translation", scenario, run_id, record))
        store.update_study_status(
            study_id, "running", current_phase="portfolio_translation",
            current_scenario=scenario, completed_evaluations=len(ledger),
        )
        return record

    structures = [
        evaluate(str(item["name"]), dict(item["overrides"]), "structure")
        for item in OFFICIAL_PORTFOLIO_STRUCTURES
    ]
    structure = _select_portfolio(structures)
    base = asdict(settings_by_name[str(structure["scenario"])])
    # replace solo admite campos de Settings; se acota a las diferencias científicas necesarias.
    base_overrides = {
        key: base[key] for key in (
            "portfolio_policy", "target_size", "vintage_count", "holding_months",
            "active_overlay_mode", "fixed_active_fraction",
        )
    }
    sizing = [
        evaluate("selected_structure_legacy_sizing",
                 {**base_overrides, "sizing_mode": "legacy_linear"}, "sizing"),
        evaluate("selected_structure_calibrated_sizing",
                 {**base_overrides, "sizing_mode": "calibrated_alpha"}, "sizing"),
    ]
    sizing_winner = _select_portfolio([structure, *sizing])
    sizing_settings = settings_by_name.get(
        str(sizing_winner["scenario"]), settings_by_name[str(structure["scenario"])]
    )
    overlay_base = {
        key: getattr(sizing_settings, key) for key in (
            "portfolio_policy", "target_size", "vintage_count", "holding_months", "sizing_mode",
        )
    }
    overlays = [
        evaluate("selected_fixed_50", {**overlay_base, "active_overlay_mode": "fixed",
                                      "fixed_active_fraction": 0.50}, "overlay"),
        evaluate("selected_binary_gate", {**overlay_base, "active_overlay_mode": "binary"}, "overlay"),
        evaluate("selected_continuous_gate", {**overlay_base, "active_overlay_mode": "continuous"},
                 "overlay"),
    ]
    overlay_winner = _select_portfolio([sizing_winner, *overlays])
    overlay_settings = settings_by_name.get(
        str(overlay_winner["scenario"]), sizing_settings
    )
    hurdle_base = {
        key: getattr(overlay_settings, key) for key in (
            "portfolio_policy", "target_size", "vintage_count", "holding_months",
            "active_overlay_mode", "fixed_active_fraction",
        )
    }
    hurdles = [
        evaluate("selected_hurdle_1x", {**hurdle_base, "sizing_mode": "calibrated_alpha",
                                        "cost_hurdle_multiplier": 1.0}, "cost_hurdle"),
        evaluate("selected_hurdle_2x", {**hurdle_base, "sizing_mode": "calibrated_alpha",
                                        "cost_hurdle_multiplier": 2.0}, "cost_hurdle"),
    ]
    final_winner = _select_portfolio([overlay_winner, *hurdles])
    final_winner["final_policy_winner"] = True
    return records, settings_by_name, str(final_winner["scenario"])


def _portfolio_record(store: ResultsStore, run_id: str, scenario: str) -> dict[str, Any]:
    artifacts = store.runs_root / run_id / "artifacts"
    summary = json.loads((artifacts / "backtest_summary.json").read_text(encoding="utf-8"))
    equity = pd.read_parquet(artifacts / "equity.parquet")
    equity["year"] = pd.to_datetime(equity["snapshot_date"]).dt.year
    eras: list[dict[str, Any]] = []
    for start, end in SELECTION_ERAS:
        era = equity.loc[equity["year"].between(start, end)]
        excess = pd.to_numeric(era["excess_return"], errors="coerce").dropna()
        tracking_error = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
        eras.append({
            "era": f"{start}-{end}",
            "information_ratio": float(excess.mean() / tracking_error) if tracking_error > 0 else 0.0,
            "annualized_alpha": float(excess.mean() * 12) if not excess.empty else 0.0,
            "annualized_turnover": float(era["turnover_pct"].mean() * 12) if not era.empty else 0.0,
        })
    selection = equity.loc[equity["year"] <= 2024].copy()
    selection_excess = pd.to_numeric(
        selection["excess_return"], errors="coerce"
    ).dropna()
    selection_te = (
        float(selection_excess.std(ddof=1)) if len(selection_excess) > 1 else 0.0
    )
    selection_ir = (
        float(selection_excess.mean() / selection_te) if selection_te > 0 else 0.0
    )
    high_cost_excess = (
        selection["gross_return"] - selection["turnover_pct"] * 0.003
        - selection["benchmark_return"]
    )
    high_te = float(high_cost_excess.std(ddof=1)) if len(high_cost_excess) > 1 else 0.0
    return {
        "scenario": scenario, "run_id": run_id,
        "information_ratio": selection_ir,
        "mean_annual_alpha": (
            float(selection_excess.mean() * 12) if not selection_excess.empty else 0.0
        ),
        "cagr_portfolio": summary.get("cagr_portfolio"),
        "cagr_difference": summary.get("cagr_difference"),
        "beat_rate": summary.get("beat_rate"),
        "annualized_turnover": (
            float(selection["turnover_pct"].mean() * 12) if not selection.empty else 0.0
        ),
        "mean_active_fraction": (
            float(selection["active_fraction"].mean())
            if not selection.empty and "active_fraction" in selection else 1.0
        ),
        "selection_until_year": 2024,
        "high_cost_information_ratio": float(high_cost_excess.mean() / high_te) if high_te > 0 else 0.0,
        "selection_excess_returns": pd.to_numeric(
            selection["excess_return"], errors="coerce"
        ).fillna(0.0).tolist(),
        "era_metrics": eras,
        "eligible": True,
        "rejection_reasons": [],
    }


def _select_portfolio(records: list[dict[str, Any]]) -> dict[str, Any]:
    incumbent = records[0]
    incumbent_median = _median_portfolio_ir(incumbent)
    incumbent_worst = min(float(era["annualized_alpha"]) for era in incumbent["era_metrics"])
    raw_p: list[float] = []
    for record in records[1:]:
        bootstrap = moving_block_bootstrap_delta(
            pd.Series(incumbent.get("selection_excess_returns", []), dtype=float),
            pd.Series(record.get("selection_excess_returns", []), dtype=float),
            block_size=12, confidence=0.90,
        )
        record["paired_bootstrap"] = bootstrap
        raw_p.append(float(bootstrap.get("p_superiority") or 1.0))
    for record, adjusted in zip(records[1:], holm_adjust(raw_p), strict=True):
        record["holm_adjusted_p"] = adjusted
    incumbent["holm_adjusted_p"] = None
    for record in records:
        reasons: list[str] = []
        positive_eras = sum(float(era["annualized_alpha"]) > 0 for era in record["era_metrics"])
        if positive_eras < 2:
            reasons.append("alpha_no_positivo_en_dos_eras")
        if _median_portfolio_ir(record) <= incumbent_median and record is not incumbent:
            reasons.append("mediana_ir_no_mejora")
        if float(record.get("annualized_turnover") or math.inf) > 2.0:
            reasons.append("turnover_superior_200pct")
        worst = min(float(era["annualized_alpha"]) for era in record["era_metrics"])
        if worst < incumbent_worst - 0.02:
            reasons.append("peor_era_degrada")
        if float(record.get("high_cost_information_ratio") or -1) < 0:
            reasons.append("ir_coste_alto_negativo")
        record["rejection_reasons"] = reasons
        record["eligible"] = not reasons or record is incumbent
    eligible = [record for record in records if record["eligible"]]
    winner = eligible[0]
    for candidate in eligible[1:]:
        delta = _median_portfolio_ir(candidate) - _median_portfolio_ir(winner)
        if delta > 0.01 and float(candidate.get("holm_adjusted_p") or 1.0) < 0.10:
            winner = candidate
        elif abs(delta) <= 0.01 and float(candidate["annualized_turnover"]) < float(
            winner["annualized_turnover"]
        ):
            winner = candidate
    return winner


def _median_portfolio_ir(record: Mapping[str, Any]) -> float:
    return float(np.median([float(era["information_ratio"]) for era in record["era_metrics"]]))


def _run_stresses(
    settings: Settings, agent_dir: Path, store: ResultsStore, study_id: str,
    name: str, ledger: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from module.runs.execution import execute_run

    cases = (
        ("cost_low", {"commission_bps": 0, "slippage_bps": 5}),
        ("cost_base", {"commission_bps": 5, "slippage_bps": 10}),
        ("cost_high", {"commission_bps": 10, "slippage_bps": 20}),
        ("cost_severe", {"commission_bps": 15, "slippage_bps": 30}),
        ("vintage_offset_1m", {"vintage_calendar_offset_months": 1}),
        ("signal_health_8q", {"signal_health_lookback_quarters": 8}),
    )
    rows: list[dict[str, Any]] = []
    for scenario, overrides in cases:
        candidate = replace(settings, **overrides)
        run_id = execute_run(
            candidate, mode="backtest", run_kind="stress", study_id=study_id,
            label=f"{name} · estrés · {scenario}", description="Estrés, no seleccionable.",
            tags=["official", "economic_stress"],
            grid_definition={"phase": "stress_and_robustness", "scenario": scenario,
                             "overrides": overrides},
            store=store, agent_dir=agent_dir, retention_policy="compact_backtest",
        )
        store.add_to_study(study_id, run_id)
        record = _portfolio_record(store, run_id, scenario)
        record["selection_policy"] = "stress_not_selected"
        rows.append(record)
        ledger.append(_ledger("economic_stress", scenario, run_id, record))
        store.update_study_status(
            study_id, "running", current_phase="stress_and_robustness",
            current_scenario=scenario, completed_evaluations=len(ledger),
        )
    return rows


def _robustness(
    store: ResultsStore, model_run_id: str, portfolio_run_id: str, settings: Settings,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from module.runs.execution import _label_permutation

    artifacts = store.runs_root / model_run_id / "artifacts"
    diagnostics = pd.read_parquet(artifacts / "rank_ic_diagnostics.parquet")
    tail = pd.read_parquet(artifacts / "rank_tail_diagnostics.parquet")
    targets_path = target_artifact_path(artifacts)
    targets = normalize_target_columns(pd.read_parquet(targets_path))
    scores = pd.read_parquet(artifacts / "agent_scores.parquet")
    ledger: list[dict[str, Any]] = []

    permutation = _score_label_permutation(scores, targets, n_permutations=9_999)
    ledger.append(_ledger("score_label_permutation", "within_cohort_9999", None, permutation))
    random_result = _pit_random_portfolios(scores, artifacts, settings, simulations=1_000)
    ledger.append(_ledger("pit_random_portfolios", "unconditional_and_risk_matched", None,
                          random_result))
    bootstrap = _bootstrap_and_eras(diagnostics, tail)
    ledger.append(_ledger("bootstrap_and_era_exclusion", "blocks_12_and_eras", None, bootstrap))

    placebo = _label_permutation(
        settings, diagnostics, n_permutations=5,
        targets_path=targets_path, input_dir=artifacts,
    )
    placebo["interpretation"] = "descriptive_leakage_check_not_inferential"
    placebo.pop("signal_above_chance", None)
    placebo.pop("p_value", None)
    for index in range(5):
        ledger.append(_ledger(
            "retraining_placebos", f"lightweight_retrain_{index + 1}", None,
            {"status": "included_in_aggregate", "aggregate": placebo},
        ))
    return {
        "score_label_permutation": permutation,
        "random_portfolios": random_result,
        "bootstrap_and_era_exclusion": bootstrap,
        "retraining_placebo": placebo,
    }, ledger


def _score_label_permutation(
    scores: pd.DataFrame, targets: pd.DataFrame, *, n_permutations: int, seed: int = 42,
) -> dict[str, Any]:
    merged = scores[["ticker", "snapshot_date", "meta_rank"]].merge(
        targets[["ticker", "snapshot_date", "target_available", "forward_excess_return"]],
        on=["ticker", "snapshot_date"], how="inner",
    )
    merged = merged.loc[merged["target_available"].fillna(False)]
    cohorts: list[tuple[np.ndarray, np.ndarray]] = []
    real_values: list[float] = []
    for _, cohort in merged.groupby("snapshot_date"):
        clean = cohort[["meta_rank", "forward_excess_return"]].dropna()
        if len(clean) < 10:
            continue
        x = clean["meta_rank"].rank().to_numpy(dtype=float)
        y = clean["forward_excess_return"].rank().to_numpy(dtype=float)
        x = (x - x.mean()) / (x.std() or 1.0)
        y = (y - y.mean()) / (y.std() or 1.0)
        cohorts.append((x, y))
        real_values.append(float(np.mean(x * y)))
    real = float(np.mean(real_values)) if real_values else 0.0
    rng = np.random.default_rng(seed)
    exceed = 0
    null_sum = 0.0
    for _ in range(n_permutations):
        value = float(np.mean([np.mean(x * rng.permutation(y)) for x, y in cohorts]))
        null_sum += value
        exceed += int(value >= real)
    p_value = (exceed + 1) / (n_permutations + 1)
    return {
        "rank_ic_real": real,
        "null_mean": null_sum / n_permutations,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "add_one_correction": True,
        "signal_above_chance": p_value < 0.05,
    }


def _pit_random_portfolios(
    scores: pd.DataFrame, artifacts: Path, settings: Settings, *, simulations: int,
) -> dict[str, Any]:
    prices = pd.read_parquet(artifacts / "asset_price_point_in_time.parquet")
    benchmark = pd.read_parquet(artifacts / "benchmark_point_in_time.parquet")
    diagnostics = pd.read_parquet(artifacts / "rank_ic_diagnostics.parquet")
    model = run_backtest(scores, prices, benchmark, settings, diagnostics)
    model_cagr = float(model.summary.get("cagr_portfolio", 0.0))
    unconditional, matched = _fast_random_cagrs(
        scores, prices, benchmark, replace(settings, sizing_mode="equal"),
        simulations=simulations, seed=42,
    )
    p_unconditional = float((unconditional < model_cagr).mean())
    p_matched = float((matched < model_cagr).mean())
    return {
        "model_cagr": model_cagr,
        "unconditional_percentile": p_unconditional,
        "risk_matched_percentile": p_matched,
        "beats_random_convincingly": p_unconditional >= 0.95 and p_matched >= 0.95,
        "n_simulations": simulations,
        "engine": "vectorized_point_in_time_policy_simulator",
    }


def _fast_random_cagrs(
    scores: pd.DataFrame, prices: pd.DataFrame, benchmark: pd.DataFrame, settings: Settings,
    *, simulations: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simula los dos nulos sin copiar DataFrames ni relanzar 2.000 backtests completos."""
    dates = sorted(scores["snapshot_date"].astype(str).unique())
    score_by_date = {
        str(date): frame.drop_duplicates("ticker").copy()
        for date, frame in scores.groupby("snapshot_date", sort=False)
    }
    price_by_date = {
        str(date): dict(zip(frame["ticker"].astype(str), pd.to_numeric(frame["price"])))
        for date, frame in prices.groupby("snapshot_date", sort=False)
    }
    benchmark_map = {
        str(row.snapshot_date): float(row.price) for row in benchmark.itertuples(index=False)
    }
    return_by_date: dict[str, dict[str, float]] = {}
    previous_prices: dict[str, float] = {}
    for date in dates:
        current = price_by_date.get(date, {})
        returns: dict[str, float] = {}
        for ticker, price in current.items():
            old = previous_prices.get(ticker)
            value = price / old - 1 if old is not None and old > 0 else 0.0
            returns[ticker] = (
                float(value) if np.isfinite(value)
                and abs(value) <= settings.max_monthly_position_return else 0.0
            )
        return_by_date[date] = returns
        previous_prices.update(current)
    benchmark_returns: dict[str, float] = {}
    previous_benchmark = None
    for date in dates:
        current = benchmark_map.get(date, previous_benchmark)
        benchmark_returns[date] = (
            current / previous_benchmark - 1
            if current is not None and previous_benchmark is not None else 0.0
        )
        previous_benchmark = current

    rng = np.random.default_rng(seed)
    unconditional = np.empty(simulations)
    matched = np.empty(simulations)
    for index in range(simulations):
        unconditional[index] = _one_fast_random_cagr(
            dates, score_by_date, price_by_date, return_by_date, benchmark_returns,
            settings, rng, risk_matched=False,
        )
        matched[index] = _one_fast_random_cagr(
            dates, score_by_date, price_by_date, return_by_date, benchmark_returns,
            settings, rng, risk_matched=True,
        )
    return unconditional, matched


def _one_fast_random_cagr(
    dates: list[str], score_by_date: Mapping[str, pd.DataFrame],
    price_by_date: Mapping[str, dict[str, float]],
    return_by_date: Mapping[str, dict[str, float]],
    benchmark_returns: Mapping[str, float], settings: Settings,
    rng: np.random.Generator, *, risk_matched: bool,
) -> float:
    holdings: dict[str, float] = {}
    vintage_slots: dict[int, dict[str, float]] = {}
    current_budget = 0.0
    previous_active = 0.0
    equity = 100.0
    trading_cost = (settings.commission_bps + settings.slippage_bps) / 10_000

    for date in dates:
        stock_return = sum(
            weight * return_by_date.get(date, {}).get(ticker, 0.0)
            for ticker, weight in holdings.items()
        )
        benchmark_return = float(benchmark_returns.get(date, 0.0))
        equity *= 1.0 + (
            previous_active * stock_return + (1.0 - previous_active) * benchmark_return
        )
        cohort = score_by_date[date]
        requested_active = active_fraction(cohort, settings)
        refresh = _policy_refresh(date, cohort, settings)
        previous_target = dict(holdings)
        if refresh:
            ranked = _random_ranked_tickers(
                cohort, set(price_by_date.get(date, {})), rng, risk_matched=risk_matched,
            )
            if settings.portfolio_policy == "legacy_monthly":
                holdings = _legacy_equal_target(holdings, ranked, settings)
                current_budget = 1.0 if holdings else 0.0
            elif settings.portfolio_policy == "quarterly_top_n":
                selected = ranked[:settings.target_size]
                holdings = _equal_weights(selected)
                current_budget = 1.0 if holdings else 0.0
            else:
                timestamp = pd.Timestamp(date)
                ordinal = timestamp.year * 4 + timestamp.quarter - 1
                slot = int(
                    (ordinal - settings.vintage_calendar_offset_months)
                    % settings.vintage_count
                )
                per_vintage = settings.target_size // settings.vintage_count
                vintage_slots[slot] = _equal_weights(ranked[:per_vintage])
                raw: dict[str, float] = {}
                for weights in vintage_slots.values():
                    for ticker, weight in weights.items():
                        raw[ticker] = raw.get(ticker, 0.0) + weight / settings.vintage_count
                current_budget = sum(raw.values())
                holdings = {
                    ticker: weight / current_budget for ticker, weight in raw.items()
                } if current_budget > 0 else {}
        desired_active = requested_active * current_budget
        before = {
            ticker: previous_active * weight for ticker, weight in previous_target.items()
        }
        after = {ticker: desired_active * weight for ticker, weight in holdings.items()}
        turnover = sum(
            abs(after.get(ticker, 0.0) - before.get(ticker, 0.0))
            for ticker in set(before) | set(after)
        ) + abs((1.0 - desired_active) - (1.0 - previous_active))
        equity *= max(0.0, 1.0 - turnover * trading_cost)
        previous_active = desired_active
    years = max(
        (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25, 1e-9
    )
    return float((equity / 100.0) ** (1.0 / years) - 1.0)


def _policy_refresh(date: str, cohort: pd.DataFrame, settings: Settings) -> bool:
    if settings.portfolio_policy == "legacy_monthly":
        return True
    if settings.vintage_calendar_offset_months:
        return (
            (pd.Timestamp(date).month - 1 - settings.vintage_calendar_offset_months) % 3
        ) == 0
    return bool(cohort.get("is_quarterly", pd.Series([True])).iloc[0])


def _random_ranked_tickers(
    cohort: pd.DataFrame, tradable: set[str], rng: np.random.Generator, *,
    risk_matched: bool,
) -> list[str]:
    frame = cohort.loc[cohort["ticker"].astype(str).isin(tradable)].copy()
    if frame.empty:
        return []
    ranks = pd.to_numeric(frame["meta_rank"], errors="coerce").fillna(0.0).to_numpy()
    if not risk_matched or "risk_rank" not in frame:
        shuffled = rng.permutation(ranks)
    else:
        risk = pd.to_numeric(frame["risk_rank"], errors="coerce")
        buckets = pd.qcut(
            risk.rank(method="first"), 5, labels=False, duplicates="drop",
        ).fillna(-1).astype(int).to_numpy()
        shuffled = ranks.copy()
        for bucket in np.unique(buckets):
            mask = buckets == bucket
            shuffled[mask] = rng.permutation(ranks[mask])
    frame["null_rank"] = shuffled
    return frame.sort_values("null_rank", ascending=False)["ticker"].astype(str).tolist()


def _equal_weights(tickers: list[str]) -> dict[str, float]:
    return {ticker: 1.0 / len(tickers) for ticker in tickers} if tickers else {}


def _legacy_equal_target(
    previous: dict[str, float], ranked: list[str], settings: Settings,
) -> dict[str, float]:
    if not ranked:
        return {}
    percentile = {
        ticker: 100.0 * (len(ranked) - index) / len(ranked)
        for index, ticker in enumerate(ranked)
    }
    survivors = {
        ticker for ticker in previous
        if percentile.get(ticker, -1.0) > settings.min_hold_percentile
    }
    for ticker in ranked:
        if len(survivors) >= settings.target_size:
            break
        survivors.add(ticker)
    while len(survivors) >= settings.target_size:
        outsider = next((ticker for ticker in ranked if ticker not in survivors), None)
        if outsider is None:
            break
        worst = min(survivors, key=lambda ticker: percentile.get(ticker, 0.0))
        if percentile[outsider] - percentile.get(worst, 0.0) < settings.rotation_edge_percentiles:
            break
        survivors.remove(worst)
        survivors.add(outsider)
    ordered = [ticker for ticker in ranked if ticker in survivors][:settings.target_size]
    return _equal_weights(ordered)


def _bootstrap_and_eras(diagnostics: pd.DataFrame, tail: pd.DataFrame) -> dict[str, Any]:
    meta = diagnostics.loc[diagnostics["agent"] == "meta_final"].copy()
    meta = meta.loc[pd.to_datetime(meta["prediction_date"]).dt.year <= 2024]
    zero = pd.Series(0.0, index=range(len(meta)))
    boot90 = moving_block_bootstrap_delta(zero, meta["rank_ic"].reset_index(drop=True),
                                          block_size=12, confidence=0.90)
    boot95 = moving_block_bootstrap_delta(zero, meta["rank_ic"].reset_index(drop=True),
                                          block_size=12, confidence=0.95)
    return {
        "block_size": 12,
        "ci_90": boot90,
        "ci_95": boot95,
        "eras": era_summary(tail.loc[
            pd.to_datetime(tail["prediction_date"]).dt.year <= 2024
        ]).to_dict("records"),
    }


def _known_stress(store: ResultsStore, model_run_id: str, portfolio_run_id: str) -> dict[str, Any]:
    model_artifacts = store.runs_root / model_run_id / "artifacts"
    tail = pd.read_parquet(model_artifacts / "rank_tail_diagnostics.parquet")
    years = pd.to_datetime(tail["prediction_date"]).dt.year
    known = tail.loc[years.isin([2025, 2026])]
    portfolio_artifacts = store.runs_root / portfolio_run_id / "artifacts"
    annual = pd.read_parquet(portfolio_artifacts / "annual_metrics.parquet")
    annual = annual.loc[pd.to_numeric(annual["year"]).isin([2025, 2026])]
    return {
        "rank_ic_mean": float(known["rank_ic"].mean()) if not known.empty else None,
        "cohorts": int(len(known)),
        "annual": annual.to_dict("records"),
    }


def _verdict(
    signal_records: list[dict[str, Any]], signal_winner: Mapping[str, Any],
    portfolio_records: list[dict[str, Any]], portfolio_winner: Mapping[str, Any],
) -> str:
    signal_changed = signal_winner["scenario"] != signal_records[0]["scenario"]
    portfolio_changed = portfolio_winner["scenario"] != portfolio_records[0]["scenario"]
    if signal_changed or portfolio_changed:
        signal_superior = signal_changed and (
            (signal_winner.get("holm_adjusted_p") or 1.0) < 0.10
        )
        portfolio_superior = portfolio_changed and (
            _median_portfolio_ir(portfolio_winner)
            > _median_portfolio_ir(portfolio_records[0]) + 0.01
        )
        if signal_superior or portfolio_superior:
            return "improved"
        return "non_inferior_simpler"
    return "no_improvement"


def _storage_manifest(
    store: ResultsStore, study_dir: Path, ledger: list[dict[str, Any]], final_run_id: str,
) -> dict[str, Any]:
    run_ids = {str(row["run_id"]) for row in ledger if row.get("run_id")}
    run_bytes = sum(
        path.stat().st_size
        for run_id in run_ids
        for path in (store.runs_root / run_id).rglob("*") if path.is_file()
    )
    study_bytes = sum(path.stat().st_size for path in study_dir.rglob("*") if path.is_file())
    final_bytes = sum(
        path.stat().st_size for path in (store.runs_root / final_run_id).rglob("*") if path.is_file()
    )
    heavy_names = {
        "agent_scores.parquet", "model_feature_attribution.parquet",
        "agent_local_attribution.parquet", "feature_diagnostics.parquet",
        "asset_price_point_in_time.parquet", "benchmark_point_in_time.parquet",
        "stock_panel.parquet", TARGET_ARTIFACT_NAME,
    }
    final_artifacts = store.runs_root / final_run_id / "artifacts"
    heavy_reference_bytes = sum(
        (final_artifacts / name).stat().st_size
        for name in heavy_names if (final_artifacts / name).exists()
    )
    compact_runs = 0
    for run_id in run_ids:
        manifest_path = store.runs_root / run_id / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(manifest.get("outputs", {}).get("retention_policy", "")).startswith("compact_"):
            compact_runs += 1
    payload = {
        "run_bytes": run_bytes,
        "study_bytes": study_bytes,
        "total_persisted_bytes": run_bytes + study_bytes,
        "temporary_workspace_bytes_after_publish": 0,
        "estimated_duplicate_bytes_avoided": compact_runs * heavy_reference_bytes,
        "compact_runs": compact_runs,
        "final_evidence_bytes": final_bytes,
        "limit_bytes": int(OFFICIAL_STUDY_PROTOCOL["max_incremental_bytes"]),
        "within_limit": run_bytes + study_bytes <= int(
            OFFICIAL_STUDY_PROTOCOL["max_incremental_bytes"]
        ),
    }
    write_json(payload, study_dir / "storage_manifest.json")
    return payload


def _fit_consumption(
    store: ResultsStore, ledger: list[dict[str, Any]], robustness: Mapping[str, Any],
) -> dict[str, Any]:
    """Resume claves únicas y reutilización observables sin confundir meta con un fit caro."""
    sources_by_key: dict[str, str] = {}
    references = 0
    for item in ledger:
        if item["phase"] not in {"signal_challengers", "seed_confirmation"}:
            continue
        run_id = item.get("run_id")
        if not run_id:
            continue
        manifest_path = store.runs_root / str(run_id) / "artifacts" / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for fit in manifest.get("agents_fit", []):
            key = str(fit.get("key") or "")
            if not key:
                continue
            references += 1
            source = str(fit.get("source") or "unknown")
            if key not in sources_by_key or source == "computed":
                sources_by_key[key] = source
    placebo = robustness.get("retraining_placebo", {})
    placebo_count = int(placebo.get("n_permutations") or 5)
    return {
        "published_unique_agents_fit_keys": len(sources_by_key),
        "published_fit_materializations_computed": sum(
            source == "computed" for source in sources_by_key.values()
        ),
        "published_fit_materializations_recycled": sum(
            source == "recycled" for source in sources_by_key.values()
        ),
        "published_fit_references": references,
        "published_reuse_references": max(0, references - len(sources_by_key)),
        "placebo_retraining_requests": placebo_count,
        "maximum_new_expensive_fits": 10,
    }


def _ledger(
    phase: str, scenario: str, run_id: str | None, metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "evaluation_index": None,
        "phase": phase,
        "scenario": scenario,
        "run_id": run_id,
        "status": "completed",
        "metrics": json.dumps(dict(metrics), ensure_ascii=False, default=str),
    }


def _comparison_frame(ledger: list[dict[str, Any]]) -> pd.DataFrame:
    """Proyección plana compatible con el dashboard histórico."""
    rows: list[dict[str, Any]] = []
    for item in ledger:
        metrics = json.loads(str(item["metrics"]))
        row = {
            "phase": item["phase"],
            "scenario": item["scenario"],
            "run_id": item.get("run_id"),
            "evaluation_index": item.get("evaluation_index"),
            "status": item.get("status"),
        }
        for key in (
            "mean_rank_ic", "top_decile_minus_universe", "information_ratio",
            "mean_annual_alpha", "cagr_portfolio", "cagr_difference", "beat_rate",
            "annualized_turnover", "mean_active_fraction",
            "high_cost_information_ratio", "eligible",
        ):
            if key in metrics:
                row[key] = metrics[key]
        rows.append(row)
    return pd.DataFrame(rows)
