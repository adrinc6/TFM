"""Consultas del dashboard para el único Model Study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from module.storage.datasets import validate_dataset_reference
from module.storage.studies import (
    list_runs, list_studies, read_events, read_run, read_study, safe_study_path,
)


ANALYSIS_VIEWS = {
    "winner", "learning", "robustness", "profiles", "portfolio-comparisons",
    "portfolio", "stocks", "report",
}


def studies() -> list[dict[str, Any]]:
    rows = []
    for study in list_studies():
        run_rows = list_runs(study["study_id"])
        completed = int(study.get("completed_runs", 0))
        budgeted = int(study.get("budget", {}).get("total_runs", 0))
        rank_values = [
            float(run["result"]["summary"]["mean_rank_ic"])
            for run in run_rows
            if (run.get("result") or {}).get("summary", {}).get("mean_rank_ic") is not None
        ]
        rows.append({
            **study,
            "max_rank_ic": max(rank_values) if rank_values else None,
            "runs_total": budgeted,
            # El presupuesto cuenta también recombinaciones metodológicas sin un run físico. Una
            # vez finalizado el Study no queda trabajo pendiente aunque ambas cifras no coincidan.
            "runs_remaining": 0 if study.get("status") == "succeeded" else max(0, budgeted - completed),
        })
    return rows


def study_detail(study_id: str) -> dict[str, Any]:
    payload = read_study(study_id)
    directory = safe_study_path(study_id)
    payload["runs"] = list_runs(study_id)
    payload["winner"] = _json(directory / "winner.json")
    payload["decisions"] = _json(directory / "decisions.json").get("decisions", [])
    payload["storage"] = _json(directory / "storage_manifest.json")
    return payload


def run_detail(study_id: str, run_id: str) -> dict[str, Any]:
    return read_run(study_id, run_id)


def events(study_id: str, after: int) -> list[dict[str, Any]]:
    return read_events(study_id, after)


def analysis(study_id: str, view: str, query: dict[str, list[str]]) -> dict[str, Any]:
    if view not in ANALYSIS_VIEWS:
        raise ValueError("Vista analítica desconocida.")
    directory = safe_study_path(study_id)
    evidence = directory / "evidence"
    profile = query.get("profile", [None])[0]
    profile_dir = evidence / "profiles" / profile if profile else evidence
    if view == "winner":
        return _json(directory / "winner.json")
    if view == "learning":
        return {
            "rank_ic": _parquet(evidence / "rank_ic_diagnostics.parquet"),
            "tail": _parquet(evidence / "rank_tail_diagnostics.parquet"),
            "weights": _parquet(evidence / "meta_weights.parquet"),
            "features": _parquet(evidence / "feature_diagnostics.parquet", limit=2_000),
            "attribution": _parquet(evidence / "model_feature_attribution.parquet", limit=2_000),
        }
    if view == "robustness":
        return _json(directory / "robustness.json")
    if view == "profiles":
        rows = _parquet(directory / "profile_comparison.parquet")
        annual = []
        for name in ("balanced", "growth", "value", "quality", "momentum", "contrarian", "defensive", "garp"):
            for row in _parquet(evidence / "profiles" / name / "annual_metrics.parquet"):
                annual.append({"profile": name, **row})
        return {"comparison": rows, "annual": annual}
    if view == "portfolio-comparisons":
        return {"rows": _parquet(directory / "portfolio_comparison.parquet")}
    if view == "portfolio":
        positions = _read_frame(profile_dir / "positions.parquet")
        orders = _read_frame(profile_dir / "orders.parquet")
        snapshots = _snapshots(positions, orders)
        selected_snapshot = _selected_snapshot(snapshots, query.get("snapshot", [None])[0])
        return {
            "summary": _json(profile_dir / "summary.json"),
            "equity": _parquet(profile_dir / "equity.parquet"),
            "annual": _parquet(profile_dir / "annual_metrics.parquet"),
            "available_snapshots": snapshots,
            "selected_snapshot": selected_snapshot,
            "positions": _records_at_snapshot(positions, selected_snapshot),
            "orders": _records_at_snapshot(orders, selected_snapshot),
        }
    if view == "stocks":
        ticker = query.get("ticker", [""])[0].strip().upper()
        snapshot = query.get("snapshot", [None])[0]
        parameter = query.get("parameter", [None])[0]
        return _stock(directory, evidence, ticker, snapshot, parameter)
    return {"markdown": (directory / "report.md").read_text(encoding="utf-8") if (directory / "report.md").exists() else ""}


def _stock(
    directory: Path, evidence: Path, ticker: str, snapshot: str | None, parameter: str | None,
) -> dict[str, Any]:
    winner = _json(directory / "winner.json")
    prepared = validate_dataset_reference(str(winner["summary"]["dataset_hash"]))
    panel = pd.read_parquet(prepared / "panel_point_in_time.parquet")
    features = pd.read_parquet(prepared / "features_point_in_time.parquet")
    scores = pd.read_parquet(evidence / "agent_scores.parquet")
    positions = _read_frame(evidence / "positions.parquet")
    orders = _read_frame(evidence / "orders.parquet")
    snapshots = sorted(scores["snapshot_date"].dropna().astype(str).unique().tolist())
    selected_snapshot = _selected_snapshot(snapshots, snapshot)
    tickers = sorted(scores["ticker"].dropna().astype(str).unique().tolist())
    if not ticker:
        return {
            "available_tickers": tickers,
            "available_snapshots": snapshots,
            "selected_snapshot": selected_snapshot,
        }
    if ticker not in tickers:
        raise ValueError("Ticker no disponible en el Study.")
    score_row = scores.loc[
        scores["ticker"].eq(ticker) & scores["snapshot_date"].astype(str).eq(selected_snapshot)
    ]
    panel_row = panel.loc[
        panel["ticker"].eq(ticker) & panel["snapshot_date"].astype(str).eq(selected_snapshot)
    ]
    feature_row = features.loc[
        features["ticker"].eq(ticker) & features["snapshot_date"].astype(str).eq(selected_snapshot)
    ]
    score_columns = [column for column in score_row.columns if column.endswith("_rank") or column in {
        "quality", "value", "growth", "momentum", "risk", "meta_score", "meta_rank",
    }]
    factor_columns = [column for column in feature_row.columns if column.startswith("factor_")]
    value_columns = [
        column for column in panel_row.columns
        if column not in {"ticker", "snapshot_date", "review_type", "in_sp500", "fundamental_period", "fundamental_filed_date"}
        and pd.api.types.is_numeric_dtype(panel_row[column])
    ]
    parameter_options = [
        {"id": column, "label": column.removeprefix("factor_"), "kind": "puntuación"}
        for column in factor_columns
    ] + [
        {"id": column, "label": column, "kind": "valor PIT"}
        for column in value_columns
    ]
    available_parameters = {item["id"] for item in parameter_options}
    selected_parameter = parameter if parameter in available_parameters else (parameter_options[0]["id"] if parameter_options else None)
    history_source = features if selected_parameter in factor_columns else panel
    history = history_source.loc[history_source["ticker"].eq(ticker), ["snapshot_date", selected_parameter]].dropna() if selected_parameter else pd.DataFrame()
    return {
        "ticker": ticker,
        "available_tickers": tickers,
        "available_snapshots": snapshots,
        "selected_snapshot": selected_snapshot,
        "scores": score_row[score_columns].to_dict("records"),
        "parameter_scores": feature_row[factor_columns].to_dict("records"),
        "parameter_values": panel_row[value_columns].to_dict("records"),
        "positions": _records_at_snapshot(positions.loc[positions.get("ticker", pd.Series(dtype=str)).eq(ticker)], selected_snapshot),
        "orders": _records_at_snapshot(orders.loc[orders.get("ticker", pd.Series(dtype=str)).eq(ticker)], selected_snapshot),
        "parameter_options": parameter_options,
        "selected_parameter": selected_parameter,
        "parameter_history": history.to_dict("records"),
    }


def _snapshots(*frames: pd.DataFrame) -> list[str]:
    values = {
        str(value) for frame in frames if not frame.empty and "snapshot_date" in frame.columns
        for value in frame["snapshot_date"].dropna()
    }
    return sorted(values)


def _selected_snapshot(snapshots: list[str], requested: str | None) -> str | None:
    if requested in snapshots:
        return requested
    return snapshots[-1] if snapshots else None


def _records_at_snapshot(frame: pd.DataFrame, snapshot: str | None) -> list[dict[str, Any]]:
    if frame.empty or not snapshot or "snapshot_date" not in frame.columns:
        return []
    selected = frame.loc[frame["snapshot_date"].astype(str).eq(snapshot)]
    return selected.where(pd.notna(selected), None).to_dict("records")


def _read_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _parquet(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    frame = _read_frame(path)
    if limit is not None:
        frame = frame.tail(limit)
    return frame.where(pd.notna(frame), None).to_dict("records") if not frame.empty else []


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
