"""Shared filesystem and logging helpers."""

from __future__ import annotations

import json
import logging
import hashlib
import os
import tempfile
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
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        df.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_parquet(path: Path, produced_by: str | None = None) -> pd.DataFrame:
    """Lee un parquet requerido y explica qué etapa debe producirlo."""
    if not path.exists():
        stage = produced_by or "la etapa anterior"
        raise FileNotFoundError(
            f"No existe el artefacto requerido: {path}. Ejecuta primero {stage}."
        )
    return pd.read_parquet(path)


def sha256_file(path: Path) -> str:
    """Huella estable de un artefacto para trazabilidad entre etapas."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pid_alive(pid: int) -> bool:
    """True si el proceso ``pid`` sigue vivo en esta máquina.

    Se usa para detectar locks huérfanos (su proceso dueño murió). En Windows ``os.kill(pid, 0)``
    es inconsistente (lanza SystemError/WinError 87 para PIDs muertos), así que se delega en
    ``psutil.pid_exists``, fiable en todas las plataformas.
    """
    import psutil
    return psutil.pid_exists(pid)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
