"""Caché persistente y trazable de etapas costosas.

Cada entrada de ``data/recycle`` es inmutable: su clave depende de los parámetros que afectan a
la etapa, huellas de entrada y revisión de código. Una coincidencia reutiliza los mismos
artefactos para runs, studies y optimizaciones posteriores sin relajar el diseño PIT.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from environment import DATA_DIR, Settings
from module.runs.results_store import canonical_json, git_revision
from module.common.utils import sha256_file, write_json


RECYCLE_ROOT = DATA_DIR / "recycle"

STAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "dataset": ("run_scope", "data_start_date", "end_date", "benchmark_ticker", "snapshot_day", "snapshot_step_months"),
    "features": ("run_scope", "max_price_age_days", "target_horizon_months", "neutralize_by_sector", "neutralize_min_group", "fundamental_momentum", "market_regime_feature", "price_momentum_multi", "moving_averages", "regime_extended", "quality_growth_derived"),
    "agents": ("execution_year", "execution_quarter", "execution_lag_days", "train_lookback_years", "fundamental_step_months", "meta_ic_lookback_quarters", "min_training_rows", "min_rank_ic_cross_section", "objective", "lgbm_n_estimators", "lgbm_max_depth", "lgbm_learning_rate", "lgbm_min_child_samples", "random_seed", "meta_type"),
    "backtest": ("target_min", "target_max", "entry_min_percentile", "min_hold_percentile", "rotation_edge_percentiles", "max_weight_per_position", "commission_bps", "slippage_bps", "rebalance_drift_tolerance", "max_monthly_position_return", "profile"),
}


def stage_key(stage: str, settings: Settings, inputs: Iterable[Path]) -> str:
    if stage not in STAGE_FIELDS:
        raise ValueError(f"Etapa de reciclaje desconocida: {stage}")
    payload = {
        "stage": stage,
        "settings": {name: getattr(settings, name) for name in STAGE_FIELDS[stage]},
        "inputs": {path.name: sha256_file(path) for path in inputs if path.exists()},
        "git_revision": git_revision(),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def cache_dir(stage: str, key: str) -> Path:
    return RECYCLE_ROOT / stage / key


def restore(stage: str, key: str, destination: Path) -> bool:
    """Restaura una entrada completa con copias; nunca enlaza ni muta la caché."""
    source = cache_dir(stage, key)
    manifest = source / "manifest.json"
    if not manifest.exists():
        return False
    for item in source.iterdir():
        if item.name == "manifest.json":
            continue
        target = destination / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
    return True


def publish(stage: str, key: str, source_items: Iterable[Path], settings: Settings) -> Path:
    """Publica de forma atómica una entrada si todavía no existe."""
    final = cache_dir(stage, key)
    if (final / "manifest.json").exists():
        return final
    final.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{key[:12]}-", dir=final.parent))
    try:
        artifacts: list[str] = []
        for item in source_items:
            if not item.exists():
                continue
            target = temp / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
            artifacts.append(item.name)
        write_json({"stage": stage, "key": key, "git_revision": git_revision(),
                    "settings": {name: getattr(settings, name) for name in STAGE_FIELDS[stage]},
                    "artifacts": artifacts}, temp / "manifest.json")
        try:
            os.replace(temp, final)
        except FileExistsError:
            shutil.rmtree(temp, ignore_errors=True)
    finally:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
    return final
