"""Persistencia mínima de studies, hipótesis y modelos confirmados."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

import pandas as pd

from environment import PROJECT_ROOT
from module.common.utils import write_json, write_parquet


RESULTS_ROOT = PROJECT_ROOT / "results"
STUDIES_ROOT = RESULTS_ROOT / "studies"
HYPOTHESES_ROOT = RESULTS_ROOT / "hypotheses"
MODELS_ROOT = RESULTS_ROOT / "models"
SCHEMA_VERSION = 1
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{5,80}$")


def ensure_roots() -> None:
    for path in (STUDIES_ROOT, HYPOTHESES_ROOT, MODELS_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"


def safe_path(root: Path, identifier: str) -> Path:
    if not _ID.fullmatch(identifier):
        raise ValueError("Identificador inválido.")
    path = (root / identifier).resolve()
    path.relative_to(root.resolve())
    return path


def create_study(kind: str, payload: Mapping[str, Any]) -> tuple[str, Path]:
    if kind not in {"exploratory", "confirmatory"}:
        raise ValueError("Tipo de estudio inválido.")
    ensure_roots()
    study_id = new_id("exp" if kind == "exploratory" else "con")
    path = STUDIES_ROOT / study_id
    path.mkdir()
    write_json(
        {
            "schema_version": SCHEMA_VERSION,
            "study_id": study_id,
            "study_type": kind,
            "status": "draft",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            **dict(payload),
        },
        path / "study.json",
    )
    return study_id, path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_study(study_id: str) -> dict[str, Any]:
    path = safe_path(STUDIES_ROOT, study_id) / "study.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe el estudio {study_id}.")
    return read_json(path)


def update_study(study_id: str, **changes: Any) -> dict[str, Any]:
    path = safe_path(STUDIES_ROOT, study_id) / "study.json"
    payload = read_json(path)
    payload.update(changes)
    payload["updated_at"] = utc_now()
    write_json(payload, path)
    return payload


def append_ledger(study_id: str, records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    path = safe_path(STUDIES_ROOT, study_id) / "evaluation_ledger.parquet"
    current = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    incoming = pd.DataFrame(list(records))
    combined = pd.concat([current, incoming], ignore_index=True)
    write_parquet(combined, path)
    return combined


def list_entities(root: Path, filename: str) -> list[dict[str, Any]]:
    ensure_roots()
    rows = []
    for directory in root.iterdir():
        path = directory / filename
        if path.exists():
            rows.append(read_json(path))
    return sorted(rows, key=lambda row: row.get("created_at", ""), reverse=True)


def freeze_hypothesis(payload: Mapping[str, Any]) -> tuple[str, Path]:
    ensure_roots()
    hypothesis_id = new_id("hyp")
    path = HYPOTHESES_ROOT / hypothesis_id
    path.mkdir()
    write_json(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": hypothesis_id,
            "status": "frozen",
            "created_at": utc_now(),
            **dict(payload),
        },
        path / "hypothesis.json",
    )
    return hypothesis_id, path


def read_hypothesis(hypothesis_id: str) -> dict[str, Any]:
    path = safe_path(HYPOTHESES_ROOT, hypothesis_id) / "hypothesis.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe la hipótesis {hypothesis_id}.")
    payload = read_json(path)
    if payload.get("status") != "frozen":
        raise ValueError("La hipótesis no está congelada.")
    return payload


def create_model(payload: Mapping[str, Any]) -> tuple[str, Path]:
    ensure_roots()
    model_id = new_id("model")
    path = MODELS_ROOT / model_id
    path.mkdir()
    write_json(
        {
            "schema_version": SCHEMA_VERSION,
            "model_id": model_id,
            "created_at": utc_now(),
            **dict(payload),
        },
        path / "model.json",
    )
    return model_id, path


def storage_usage() -> dict[str, int]:
    ensure_roots()
    files = [path for path in RESULTS_ROOT.rglob("*") if path.is_file()]
    return {"bytes": sum(path.stat().st_size for path in files)}
