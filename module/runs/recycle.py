"""Caché persistente y trazable de etapas costosas.

Cada entrada de ``data/recycle`` es inmutable: su clave depende de los parámetros que afectan a
la etapa y de las huellas de entrada. Una coincidencia reutiliza los mismos artefactos para runs,
studies y optimizaciones posteriores sin relajar el diseño PIT.

La clave NO depende automáticamente del contenido del código: un cambio de logging o de comentario
en un módulo de la etapa no debe invalidar la caché. La invalidación tras un cambio real de lógica
es responsabilidad manual del usuario, subiendo el campo ``*_code_version`` correspondiente en
``environment.Settings`` (ver ``STAGE_FIELDS`` más abajo, ya incluye esos campos).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from environment import DATA_DIR, Settings
from module.runs.results_store import canonical_json, git_revision
from module.common.utils import pid_alive, sha256_file, write_json


RECYCLE_ROOT = DATA_DIR / "recycle"
CACHE_SCHEMA_VERSION = 2

# La etapa cara "agents" se descompone internamente en dos claves de reutilización distintas:
#   - "agents_fit": el fit walk-forward de UNA familia de modelo (lightgbm/catboost/elastic_net).
#     Es lo caro (~90% del tiempo). Su clave incluye la familia concreta (vía ``extra`` en
#     ``stage_key``) y NO depende de ``meta_type``/``meta_ic_lookback_quarters``/
#     ``min_rank_ic_cross_section`` ni de ``intra_agent_ensemble_mode``: cambiar el meta o quitar/
#     añadir OTRA familia reutiliza estos scores intactos.
#   - "agents": la combinación barata (ensemble intra-agente + meta), cuya identidad la aportan los
#     hashes de las salidas ``agents_fit`` (vía ``inputs``) más los campos meta de abajo.
STAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "dataset": ("run_scope", "data_start_date", "end_date", "benchmark_ticker", "execution_lag_days", "snapshot_step_months", "dataset_code_version"),
    "features": ("run_scope", "target_horizon_months", "neutralize_by_sector", "fundamental_momentum", "market_regime_feature", "price_momentum_multi", "moving_averages", "regime_extended", "quality_growth_derived", "enabled_feature_blocks", "metric_winsorization_percentile", "risk_feature_windows", "technical_feature_windows", "features_code_version"),
    "agents_fit": ("execution_year", "execution_quarter", "execution_lag_days", "train_lookback_years", "fundamental_step_months", "min_rank_ic_cross_section", "objective", "lgbm_n_estimators", "lgbm_max_depth", "lgbm_learning_rate", "lgbm_min_child_samples", "random_seed", "recency_weighting", "enabled_agents", "feature_weighting_mode", "feature_selection_min_coverage", "feature_selection_lookback_quarters", "feature_selection_min_permutation_importance", "feature_selection_min_positive_fraction", "feature_selection_max_features_per_agent", "enabled_feature_blocks", "metric_winsorization_percentile", "risk_feature_windows", "technical_feature_windows", "agents_code_version"),
    "agents": ("execution_year", "execution_quarter", "execution_lag_days", "train_lookback_years", "fundamental_step_months", "meta_ic_lookback_quarters", "min_rank_ic_cross_section", "objective", "lgbm_n_estimators", "lgbm_max_depth", "lgbm_learning_rate", "lgbm_min_child_samples", "random_seed", "meta_type", "recency_weighting", "enabled_agents", "enabled_model_families", "intra_agent_ensemble_mode", "feature_weighting_mode", "feature_selection_min_coverage", "feature_selection_lookback_quarters", "feature_selection_min_permutation_importance", "feature_selection_min_positive_fraction", "feature_selection_max_features_per_agent", "enabled_feature_blocks", "metric_winsorization_percentile", "risk_feature_windows", "technical_feature_windows", "agents_code_version"),
    "backtest": ("target_size", "min_hold_percentile", "rotation_edge_percentiles", "commission_bps", "slippage_bps", "rebalance_drift_tolerance", "max_monthly_position_return", "profile", "backtest_code_version"),
}


def stage_key(stage: str, settings: Settings, inputs: Iterable[Path],
              extra: dict[str, object] | None = None) -> str:
    if stage not in STAGE_FIELDS:
        raise ValueError(f"Etapa de reciclaje desconocida: {stage}")
    payload = {
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
    return RECYCLE_ROOT / stage / key


@contextmanager
def cache_lock(stage: str, key: str, timeout_seconds: float = 86_400):
    """Single-flight entre procesos para una transformación content-addressed."""
    lock = RECYCLE_ROOT / stage / ".locks" / f"{key}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            os.close(descriptor)
            break
        except FileExistsError:
            # ``os.kill(pid, 0)`` es inestable en Windows (lanza SystemError para PIDs muertos, no
            # OSError, y puede corromper el estado C del intérprete matando el worker). Se usa
            # ``pid_alive`` (psutil), fiable en toda plataforma — el mismo patrón que
            # ``_full_study_lock``. Un lock ilegible o de un proceso muerto es huérfano y se reclama.
            try:
                owner = int(lock.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                lock.unlink(missing_ok=True)
                continue
            if not pid_alive(owner):
                lock.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timeout esperando caché {stage}:{key[:12]} (PID {owner}).")
            time.sleep(0.25)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _link_or_copy(source: Path, target: Path) -> None:
    """Enlaza (hardlink) el fichero de la caché en destino; copia si el FS no lo soporta.

    Los artefactos de ``data/recycle`` son inmutables y las etapas siguientes solo los leen,
    así que compartir inode por hardlink da bytes idénticos con coste ~0 frente a copiar el
    parquet completo. En NTFS/ext4 funciona; si falla (p.ej. cruce de volumen), se copia.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _restore_tree(source: Path, target: Path) -> None:
    """Replica ``source`` en ``target`` enlazando cada fichero (recursivo para directorios)."""
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            _restore_tree(child, target / child.name)
    else:
        _link_or_copy(source, target)


def restore(stage: str, key: str, destination: Path) -> bool:
    """Restaura una entrada validada; una caché parcial o corrupta nunca se reutiliza."""
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
        if not isinstance(expected, dict) or not item.is_file():
            return False
        if item.stat().st_size != expected.get("size") or sha256_file(item) != expected.get("sha256"):
            return False
        validated.append((item, destination / relative))
    for item, target in validated:
        _link_or_copy(item, target)
    return True


def _replace_with_retry(temp: Path, final: Path, *, attempts: int = 5, delay: float = 0.5) -> None:
    """os.replace con reintentos: en Windows, un antivirus o indexador puede retener brevemente
    un handle sobre el directorio temporal y provocar PermissionError en vez de FileExistsError."""
    for attempt in range(attempts):
        try:
            os.replace(temp, final)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def publish(stage: str, key: str, source_items: Iterable[Path], settings: Settings) -> Path:
    """Publica de forma atómica una entrada si todavía no existe."""
    final = cache_dir(stage, key)
    if (final / "manifest.json").exists():
        return final
    final.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{key[:12]}-", dir=final.parent))
    try:
        artifacts: dict[str, dict[str, int | str]] = {}
        for item in source_items:
            if not item.exists():
                continue
            target = temp / item.name
            if item.is_dir():
                shutil.copytree(item, target)
                for child in target.rglob("*"):
                    if child.is_file():
                        relative = child.relative_to(temp).as_posix()
                        artifacts[relative] = {"size": child.stat().st_size, "sha256": sha256_file(child)}
            else:
                shutil.copy2(item, target)
                artifacts[item.name] = {"size": target.stat().st_size, "sha256": sha256_file(target)}
        write_json({"schema_version": CACHE_SCHEMA_VERSION, "stage": stage, "key": key,
                    "git_revision": git_revision(), "python": platform.python_version(),
                    "settings": {name: getattr(settings, name) for name in STAGE_FIELDS[stage]},
                    "artifacts": artifacts}, temp / "manifest.json")
        try:
            _replace_with_retry(temp, final)
        except FileExistsError:
            shutil.rmtree(temp, ignore_errors=True)
    finally:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
    return final
