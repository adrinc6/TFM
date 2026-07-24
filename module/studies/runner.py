"""Único ejecutor científico: configuración cerrada → métricas y evidencia."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from environment import DATA_DIR
from module.common.utils import write_json, write_parquet
from module.evaluation.backtest import BacktestResult, run_backtest
from module.modeling.agents import build_agent_scores
from module.studies.config import settings_from_values
from module.storage.cache import canonical_json, enforce_cache_limit
from module.storage.datasets import ensure_prepared


SUMMARY_CACHE = DATA_DIR / "cache" / "evaluations"
SELECTION_ERAS = ((2015, 2018), (2019, 2021), (2022, 2024))
KNOWN_STRESS_YEARS = (2025, 2026)


def discard_summary_cache(keys: list[str]) -> None:
    for key in keys:
        if key and all(char in "0123456789abcdef" for char in key):
            shutil.rmtree(SUMMARY_CACHE / key, ignore_errors=True)


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
        payload = json.loads(cached.read_text(encoding="utf-8"))
        payload["source"] = "cached"
        return payload

    started = time.perf_counter()
    base_settings = settings_from_values(
        values, random_seed=random_seed, profile=effective_profile, overrides=overrides,
    )
    dataset_hash, prepared, dataset_reused = ensure_prepared(base_settings)
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
        return summary
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_profile_evaluation(
    values: Mapping[str, Any], profile: str, evidence_dir: Path, retain_dir: Path | None = None,
) -> dict[str, Any]:
    """Backtest de un perfil sobre los scores ya congelados del ganador, sin reentrenar."""
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
