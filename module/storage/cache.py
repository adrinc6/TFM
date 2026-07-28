"""Caché content-addressable compartida por los estudios."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from environment import DATA_DIR, PROJECT_ROOT, Settings
from module.common.utils import link_or_copy, pid_alive, sha256_file, write_json


CACHE_ROOT = DATA_DIR / "cache"
CACHE_SCHEMA_VERSION = 1
CACHE_LIMIT_BYTES = 5 * 1024**3

STAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "agents_fit": (
        "execution_year", "execution_quarter", "execution_lag_days", "train_lookback_years",
        "fundamental_step_months", "min_rank_ic_cross_section", "objective",
        "lgbm_n_estimators", "lgbm_max_depth", "lgbm_learning_rate",
        "lgbm_min_child_samples", "random_seed", "recency_weighting", "enabled_agents",
        "feature_weighting_mode", "feature_selection_max_features_per_agent",
        "enabled_feature_blocks", "metric_winsorization_percentile", "risk_feature_windows",
        "technical_feature_windows", "agents_fit_code_version",
    ),
}


def canonical_json(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def stage_key(
    stage: str,
    settings: Settings,
    inputs: Iterable[Path],
    extra: dict[str, object] | None = None,
) -> str:
    if stage not in STAGE_FIELDS:
        raise ValueError(f"Etapa de caché desconocida: {stage}")
    payload: dict[str, object] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "stage": stage,
        "settings": {name: getattr(settings, name) for name in STAGE_FIELDS[stage]},
        "inputs": {path.name: sha256_file(path) for path in inputs if path.exists()},
        "python": platform.python_version(),
    }
    if extra:
        payload["extra"] = extra
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def cache_dir(stage: str, key: str) -> Path:
    return CACHE_ROOT / stage / key


@contextmanager
def cache_lock(stage: str, key: str, timeout_seconds: float = 86_400):
    lock = CACHE_ROOT / stage / ".locks" / f"{key}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            os.close(descriptor)
            break
        except FileExistsError:
            try:
                owner = int(lock.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                lock.unlink(missing_ok=True)
                continue
            if not pid_alive(owner):
                lock.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timeout esperando caché {stage}:{key[:12]}.")
            time.sleep(0.25)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def restore(stage: str, key: str, destination: Path) -> bool:
    source = cache_dir(stage, key)
    manifest = source / "manifest.json"
    if not manifest.exists():
        return False
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION or metadata.get("key") != key:
        return False
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    validated: list[tuple[Path, Path]] = []
    for relative, expected in artifacts.items():
        item = source / relative
        if (
            not isinstance(expected, dict)
            or not item.is_file()
            or item.stat().st_size != expected.get("size")
            or sha256_file(item) != expected.get("sha256")
        ):
            return False
        validated.append((item, destination / relative))
    for item, target in validated:
        link_or_copy(item, target)
    return True


def publish(
    stage: str,
    key: str,
    source_items: Iterable[Path],
    settings: Settings,
) -> Path:
    final = cache_dir(stage, key)
    if (final / "manifest.json").exists():
        return final
    final.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{key[:12]}-", dir=final.parent))
    try:
        artifacts: dict[str, dict[str, int | str]] = {}
        for item in source_items:
            if not item.is_file():
                continue
            target = temp / item.name
            shutil.copy2(item, target)
            artifacts[item.name] = {
                "size": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        write_json(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "stage": stage,
                "key": key,
                "settings": {name: getattr(settings, name) for name in STAGE_FIELDS[stage]},
                "artifacts": artifacts,
            },
            temp / "manifest.json",
        )
        try:
            os.replace(temp, final)
        except FileExistsError:
            pass
    finally:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
    try:
        enforce_cache_limit(protected={key})
    except RuntimeError:
        if key not in pinned_keys():
            shutil.rmtree(final, ignore_errors=True)
        raise
    return final


def pinned_keys() -> set[str]:
    """Claves referidas por hipótesis o modelos; nunca son candidatas a limpieza."""
    keys: set[str] = set()
    results = PROJECT_ROOT / "results"
    if not results.exists():
        return keys
    for manifest in results.glob("*/**/evidence/manifest.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in payload.get("agents_fit", []):
            key = item.get("key") if isinstance(item, dict) else None
            if isinstance(key, str):
                keys.add(key)
    return keys


def enforce_cache_limit(*, protected: set[str] | None = None) -> None:
    """Elimina solo entradas sin referencias hasta respetar el máximo."""
    protected = set(protected or ()) | pinned_keys()
    if not CACHE_ROOT.exists():
        return
    entries = [
        directory
        for stage in CACHE_ROOT.iterdir() if stage.is_dir()
        for directory in stage.iterdir()
        if directory.is_dir() and directory.name != ".locks"
    ]
    total = sum(
        path.stat().st_size
        for path in CACHE_ROOT.rglob("*")
        if path.is_file()
    )
    candidates = sorted(
        (directory for directory in entries if directory.name not in protected),
        key=lambda directory: directory.stat().st_mtime,
    )
    for directory in candidates:
        if total <= CACHE_LIMIT_BYTES:
            break
        size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        shutil.rmtree(directory, ignore_errors=True)
        total -= size
    if total > CACHE_LIMIT_BYTES:
        raise RuntimeError(
            "La caché referenciada supera 5 GiB; no se eliminó evidencia para ocultarlo."
        )
