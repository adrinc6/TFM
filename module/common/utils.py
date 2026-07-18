"""Shared filesystem and logging helpers."""

from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


def setup_logging(log_path: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: Path, produced_by: str | None = None) -> pd.DataFrame:
    """Lee un parquet requerido y explica qué etapa debe producirlo."""
    if not path.exists():
        stage = produced_by or "la etapa anterior"
        raise FileNotFoundError(
            f"No existe el artefacto requerido: {path}. Ejecuta primero {stage} "
            "con el mismo RUN_SCOPE."
        )
    return pd.read_parquet(path)


def sha256_file(path: Path) -> str:
    """Huella estable de un artefacto para trazabilidad entre etapas."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
