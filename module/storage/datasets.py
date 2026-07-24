"""Materialización única de datos PIT y features por identidad científica."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from environment import DATA_DIR, Settings
from module.common.utils import sha256_file, write_json
from module.data.dataset import build_point_in_time_dataset
from module.modeling.features import build_features
from module.storage.cache import canonical_json


PREPARED_ROOT = DATA_DIR / "prepared"
CORE_ROOT = PREPARED_ROOT / "_core"
DATASET_SCHEMA_VERSION = 1
CORE_FIELDS = (
    "run_scope", "data_start_date", "end_date", "benchmark_ticker", "execution_year",
    "execution_quarter", "execution_lag_days", "snapshot_step_months",
    "fundamental_step_months", "dataset_code_version",
)
FEATURE_FIELDS = (
    "target_horizon_months", "neutralize_by_sector",
    "fundamental_momentum", "market_regime_feature", "price_momentum_multi",
    "moving_averages", "regime_extended", "quality_growth_derived",
    "enabled_feature_blocks", "metric_winsorization_percentile", "risk_feature_windows",
    "technical_feature_windows", "features_code_version",
)
REQUIRED = (
    "panel_point_in_time.parquet",
    "benchmark_point_in_time.parquet",
    "asset_price_point_in_time.parquet",
    "features_point_in_time.parquet",
    "targets_forward.parquet",
)


def _raw_fingerprint(raw_dir: Path) -> dict[str, dict[str, Any]]:
    names = ("prices.parquet", "finnhub_metrics.parquet", "report_dates.parquet", "profiles.parquet")
    fingerprint: dict[str, dict[str, Any]] = {}
    for name in names:
        path = raw_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Falta el input raw obligatorio: {path}.")
        fingerprint[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return fingerprint


def dataset_identity(settings: Settings) -> tuple[str, dict[str, Any]]:
    core_hash, core = _core_identity(settings)
    payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "core_hash": core_hash,
        "settings": {field: getattr(settings, field) for field in FEATURE_FIELDS},
        "raw": core["raw"],
    }
    key = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return key, payload


def ensure_prepared(settings: Settings) -> tuple[str, Path, bool]:
    key, identity = dataset_identity(settings)
    final = PREPARED_ROOT / key
    manifest = final / "dataset_manifest.json"
    if manifest.exists() and all((final / name).exists() for name in REQUIRED):
        return key, final, True

    core_hash = str(identity["core_hash"])
    core = _ensure_core(settings, core_hash)
    PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{key[:12]}-", dir=PREPARED_ROOT))
    try:
        runtime = replace(settings, workspace_dir=temp)
        for name in (
            "panel_point_in_time.parquet",
            "benchmark_point_in_time.parquet",
            "asset_price_point_in_time.parquet",
        ):
            _link_or_copy(core / name, temp / name)
        build_features(runtime)
        files = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in temp.iterdir() if path.is_file()
        }
        write_json(
            {"schema_version": DATASET_SCHEMA_VERSION, "dataset_hash": key, **identity, "files": files},
            temp / "dataset_manifest.json",
        )
        try:
            temp.replace(final)
        except FileExistsError:
            shutil.rmtree(temp, ignore_errors=True)
    finally:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
    return key, final, False


def _core_identity(settings: Settings) -> tuple[str, dict[str, Any]]:
    payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "settings": {field: getattr(settings, field) for field in CORE_FIELDS},
        "raw": _raw_fingerprint(settings.raw_output_dir),
    }
    key = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return key, payload


def _ensure_core(settings: Settings, key: str) -> Path:
    final = CORE_ROOT / key
    required = (
        "panel_point_in_time.parquet",
        "benchmark_point_in_time.parquet",
        "asset_price_point_in_time.parquet",
    )
    if (final / "core_manifest.json").exists() and all((final / name).exists() for name in required):
        return final
    CORE_ROOT.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{key[:12]}-", dir=CORE_ROOT))
    try:
        build_point_in_time_dataset(replace(settings, workspace_dir=temp))
        write_json(
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "core_hash": key,
                "files": {
                    name: {"bytes": (temp / name).stat().st_size, "sha256": sha256_file(temp / name)}
                    for name in required
                },
            },
            temp / "core_manifest.json",
        )
        try:
            temp.replace(final)
        except FileExistsError:
            shutil.rmtree(temp, ignore_errors=True)
    finally:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
    return final


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def prepared_usage() -> dict[str, int]:
    files = [path for path in PREPARED_ROOT.rglob("*") if path.is_file()] if PREPARED_ROOT.exists() else []
    unique: dict[tuple[int, int] | str, int] = {}
    for path in files:
        stat = path.stat()
        identity: tuple[int, int] | str = (
            (stat.st_dev, stat.st_ino) if stat.st_ino else str(path.resolve())
        )
        unique.setdefault(identity, stat.st_size)
    return {
        "bytes": sum(unique.values()),
        "datasets": sum(1 for _ in PREPARED_ROOT.glob("*/dataset_manifest.json"))
        if PREPARED_ROOT.exists() else 0,
    }


def validate_dataset_reference(dataset_hash: str) -> Path:
    if not dataset_hash or any(char not in "0123456789abcdef" for char in dataset_hash):
        raise ValueError("Hash de dataset inválido.")
    path = PREPARED_ROOT / dataset_hash
    if not (path / "dataset_manifest.json").exists():
        raise FileNotFoundError(f"No existe el dataset {dataset_hash}.")
    return path


def pinned_dataset_hashes() -> set[str]:
    hashes: set[str] = set()
    results = DATA_DIR.parent / "results"
    if not results.exists():
        return hashes
    for metadata in (
        list(results.glob("hypotheses/*/hypothesis.json"))
        + list(results.glob("models/*/model.json"))
    ):
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        value = payload.get("dataset_hash")
        if isinstance(value, str):
            hashes.add(value)
    return hashes


def prune_prepared(*, keep: set[str] | None = None) -> dict[str, int]:
    """Elimina materializaciones no referidas; conserva los cores que aún tienen consumidores."""
    keep_hashes = set(keep or ()) | pinned_dataset_hashes()
    if not PREPARED_ROOT.exists():
        return prepared_usage()
    for directory in PREPARED_ROOT.iterdir():
        if not directory.is_dir() or directory.name == "_core":
            continue
        if directory.name not in keep_hashes:
            shutil.rmtree(directory, ignore_errors=True)
    core_hashes: set[str] = set()
    for manifest in PREPARED_ROOT.glob("*/dataset_manifest.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        core_hash = payload.get("core_hash")
        if isinstance(core_hash, str):
            core_hashes.add(core_hash)
    if CORE_ROOT.exists():
        for directory in CORE_ROOT.iterdir():
            if directory.is_dir() and directory.name not in core_hashes:
                shutil.rmtree(directory, ignore_errors=True)
    return prepared_usage()
