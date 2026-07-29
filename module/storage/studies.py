"""Persistencia auditable de Model Studies y sus runs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from environment import PROJECT_ROOT
from module.common.utils import write_json


RESULTS_ROOT = PROJECT_ROOT / "results"
STUDIES_ROOT = RESULTS_ROOT / "studies"
SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_root() -> None:
    STUDIES_ROOT.mkdir(parents=True, exist_ok=True)


def safe_study_path(study_id: str) -> Path:
    if not study_id.startswith("study-") or not study_id.replace("-", "").isalnum():
        raise ValueError("Identificador de Study inválido.")
    ensure_root()
    path = (STUDIES_ROOT / study_id).resolve()
    path.relative_to(STUDIES_ROOT.resolve())
    return path


def create_study(payload: Mapping[str, Any]) -> tuple[str, Path]:
    ensure_root()
    study_id = f"study-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    directory = safe_study_path(study_id)
    directory.mkdir()
    now = utc_now()
    study = {
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        "study_type": "model_study",
        "status": "queued",
        "phase": "initialization",
        "progress": 0.0,
        "created_at": now,
        "updated_at": now,
        "heartbeat": None,
        "worker_pid": None,
        "completed_runs": 0,
        **dict(payload),
    }
    write_json(study, directory / "study.json")
    write_json(payload["configuration"], directory / "config.json")
    write_json(payload["catalog"], directory / "catalog_snapshot.json")
    (directory / "events.jsonl").write_text("", encoding="utf-8")
    return study_id, directory


def read_study(study_id: str) -> dict[str, Any]:
    path = safe_study_path(study_id) / "study.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe el Study {study_id}.")
    return json.loads(path.read_text(encoding="utf-8"))


def update_study(study_id: str, **changes: Any) -> dict[str, Any]:
    path = safe_study_path(study_id) / "study.json"
    payload = read_study(study_id)
    payload.update(changes)
    payload["updated_at"] = utc_now()
    write_json(payload, path)
    return payload


def list_studies() -> list[dict[str, Any]]:
    ensure_root()
    rows = []
    for directory in STUDIES_ROOT.iterdir():
        metadata = directory / "study.json"
        if metadata.exists():
            rows.append(json.loads(metadata.read_text(encoding="utf-8")))
    return sorted(rows, key=lambda row: row["created_at"], reverse=True)


def create_run(
    study_id: str,
    *,
    logical_key: str,
    phase: str,
    variable_id: str,
    value: Any,
    configuration: Mapping[str, Any],
    attempt: int = 1,
) -> dict[str, Any]:
    directory = safe_study_path(study_id) / "runs"
    directory.mkdir(exist_ok=True)
    existing = find_run(study_id, logical_key)
    run_id = existing["run_id"] if existing else f"run-{uuid4().hex[:12]}"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "study_id": study_id,
        "logical_key": logical_key,
        "attempt": attempt,
        "phase": phase,
        "variable_id": variable_id,
        "value": value,
        "configuration": dict(configuration),
        "status": "queued",
        "stage": "queued",
        "progress": 0.0,
        "heartbeat": utc_now(),
        "created_at": existing.get("created_at") if existing else utc_now(),
        "updated_at": utc_now(),
        "pid": os.getpid(),
        "agent": None,
        "walk_forward_current": 0,
        "walk_forward_total": 0,
        "elapsed_seconds": 0.0,
        "cpu_percent": None,
        "ram_bytes": None,
        "bytes_persisted": 0,
        "error": None,
        "traceback": None,
        "result": None,
    }
    write_json(payload, directory / f"{run_id}.json")
    return payload


def read_run(study_id: str, run_id: str) -> dict[str, Any]:
    if not run_id.startswith("run-") or not run_id.replace("-", "").isalnum():
        raise ValueError("Identificador de run inválido.")
    path = safe_study_path(study_id) / "runs" / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe el run {run_id}.")
    return json.loads(path.read_text(encoding="utf-8"))


def update_run(study_id: str, run_id: str, **changes: Any) -> dict[str, Any]:
    path = safe_study_path(study_id) / "runs" / f"{run_id}.json"
    payload = read_run(study_id, run_id)
    payload.update(changes)
    payload["updated_at"] = utc_now()
    payload["heartbeat"] = utc_now()
    write_json(payload, path)
    return payload


def list_runs(study_id: str) -> list[dict[str, Any]]:
    directory = safe_study_path(study_id) / "runs"
    if not directory.exists():
        return []
    return sorted(
        (json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("run-*.json")),
        key=lambda row: (row["created_at"], row["run_id"]),
    )


def find_run(study_id: str, logical_key: str) -> dict[str, Any] | None:
    return next((run for run in list_runs(study_id) if run["logical_key"] == logical_key), None)


def append_event(
    study_id: str,
    level: str,
    event: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    path = safe_study_path(study_id) / "events.jsonl"
    sequence = 1
    if path.exists():
        with path.open("rb") as stream:
            sequence = sum(1 for _ in stream) + 1
    payload = {
        "sequence": sequence,
        "timestamp": utc_now(),
        "level": level,
        "event": event,
        "message": message,
        **details,
    }
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    time_of_day = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{time_of_day}] [{level.upper()}] {study_id} · {message}", flush=True)
    return payload


def read_events(study_id: str, after: int = 0) -> list[dict[str, Any]]:
    path = safe_study_path(study_id) / "events.jsonl"
    if not path.exists():
        return []
    return [
        payload
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for payload in [json.loads(line)]
        if int(payload["sequence"]) > after
    ]


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_storage_manifest(study_id: str) -> dict[str, Any]:
    directory = safe_study_path(study_id)
    groups: dict[str, int] = {}
    for file in directory.rglob("*"):
        if file.is_file():
            relative = file.relative_to(directory)
            group = relative.parts[0] if len(relative.parts) > 1 else "study"
            groups[group] = groups.get(group, 0) + file.stat().st_size
    payload = {"study_id": study_id, "total_bytes": sum(groups.values()), "groups": groups}
    write_json(payload, directory / "storage_manifest.json")
    return payload
