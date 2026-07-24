"""Consultas acotadas para las vistas analíticas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from module.storage.datasets import validate_dataset_reference
from module.storage.evidence import (
    HYPOTHESES_ROOT,
    MODELS_ROOT,
    STUDIES_ROOT,
    list_entities,
    read_json,
    safe_path,
)


def studies() -> list[dict[str, Any]]:
    return list_entities(STUDIES_ROOT, "study.json")


def hypotheses() -> list[dict[str, Any]]:
    return list_entities(HYPOTHESES_ROOT, "hypothesis.json")


def models() -> list[dict[str, Any]]:
    return list_entities(MODELS_ROOT, "model.json")


def study_detail(study_id: str) -> dict[str, Any]:
    directory = safe_path(STUDIES_ROOT, study_id)
    payload = read_json(directory / "study.json")
    ledger = directory / "evaluation_ledger.parquet"
    payload["ledger"] = pd.read_parquet(ledger).to_dict("records") if ledger.exists() else []
    decision = directory / "decision.json"
    payload["decision"] = read_json(decision) if decision.exists() else None
    return payload


def resolve_evidence(entity_id: str, profile: str | None = None) -> tuple[dict[str, Any], Path]:
    if entity_id.startswith("model-"):
        directory = safe_path(MODELS_ROOT, entity_id)
        metadata, evidence = read_json(directory / "model.json"), directory / "evidence"
        return metadata, evidence / "profiles" / profile if profile else evidence
    if entity_id.startswith("hyp-"):
        directory = safe_path(HYPOTHESES_ROOT, entity_id)
        metadata, evidence = read_json(directory / "hypothesis.json"), directory / "evidence"
        return metadata, evidence / "profiles" / profile if profile else evidence
    raise ValueError("Las vistas analíticas requieren una hipótesis o modelo.")


def performance(entity_id: str, profile: str | None = None) -> dict[str, Any]:
    metadata, evidence = resolve_evidence(entity_id, profile)
    return {
        "entity": metadata,
        "summary": _json(evidence / "summary.json"),
        "equity": _parquet(evidence / "equity.parquet"),
        "annual": _parquet(evidence / "annual_metrics.parquet"),
    }


def learning(entity_id: str) -> dict[str, Any]:
    metadata, evidence = resolve_evidence(entity_id)
    return {
        "entity": metadata,
        "rank_ic": _parquet(evidence / "rank_ic_diagnostics.parquet"),
        "tail": _parquet(evidence / "rank_tail_diagnostics.parquet"),
        "weights": _parquet(evidence / "meta_weights.parquet"),
        "health": _parquet(evidence / "signal_health.parquet"),
        "attribution": _parquet(evidence / "model_feature_attribution.parquet", limit=2_000),
    }


def rankings(entity_id: str, snapshot: str | None = None) -> dict[str, Any]:
    metadata, evidence = resolve_evidence(entity_id)
    path = evidence / "agent_scores.parquet"
    frame = pd.read_parquet(path)
    available = sorted(frame["snapshot_date"].astype(str).unique())
    selected = snapshot if snapshot in available else (available[-1] if available else None)
    if selected:
        frame = frame.loc[frame["snapshot_date"].astype(str).eq(selected)]
    columns = [
        column for column in (
            "ticker", "snapshot_date", "meta_rank", "quality_rank", "value_rank",
            "growth_rank", "momentum_rank", "risk_rank", "expected_excess_return",
        ) if column in frame
    ]
    return {
        "entity": metadata,
        "snapshots": available,
        "selected_snapshot": selected,
        "rows": frame[columns].sort_values("meta_rank", ascending=False).head(200).to_dict("records"),
    }


def portfolio(entity_id: str, profile: str | None = None) -> dict[str, Any]:
    metadata, evidence = resolve_evidence(entity_id, profile)
    return {
        "entity": metadata,
        "positions": _parquet(evidence / "positions.parquet", limit=5_000),
    }


def trades(entity_id: str, profile: str | None = None) -> dict[str, Any]:
    metadata, evidence = resolve_evidence(entity_id, profile)
    return {"entity": metadata, "orders": _parquet(evidence / "orders.parquet", limit=10_000)}


def stock(entity_id: str, ticker: str) -> dict[str, Any]:
    metadata, evidence = resolve_evidence(entity_id)
    ticker = ticker.strip().upper()
    if not ticker or len(ticker) > 12 or not ticker.replace(".", "").replace("-", "").isalnum():
        raise ValueError("Ticker inválido.")
    dataset_hash = str(metadata["dataset_hash"])
    prepared = validate_dataset_reference(dataset_hash)
    panel = pd.read_parquet(prepared / "panel_point_in_time.parquet")
    panel = panel.loc[panel["ticker"].eq(ticker)].sort_values("snapshot_date")
    scores = pd.read_parquet(evidence / "agent_scores.parquet")
    scores = scores.loc[scores["ticker"].eq(ticker)].sort_values("snapshot_date")
    positions_path = evidence / "positions.parquet"
    positions = pd.read_parquet(positions_path) if positions_path.exists() else pd.DataFrame()
    if "ticker" in positions:
        positions = positions.loc[positions["ticker"].eq(ticker)]
    orders_path = evidence / "orders.parquet"
    orders = pd.read_parquet(orders_path) if orders_path.exists() else pd.DataFrame()
    if "ticker" in orders:
        orders = orders.loc[orders["ticker"].eq(ticker)]
    return {
        "entity": metadata,
        "ticker": ticker,
        "panel": panel.tail(240).to_dict("records"),
        "scores": scores.tail(240).to_dict("records"),
        "positions": positions.to_dict("records"),
        "orders": orders.to_dict("records"),
    }


def _parquet(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_parquet(path)
    if limit is not None:
        frame = frame.tail(limit)
    return frame.where(pd.notna(frame), None).to_dict("records")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
