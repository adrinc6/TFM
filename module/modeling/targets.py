"""Compatibilidad de nombres de targets y contrato neutral respecto al horizonte."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TARGET_ALIASES: dict[str, str] = {
    "forward_return_3m": "forward_return",
    "forward_benchmark_return_3m": "forward_benchmark_return",
    "forward_excess_return_3m": "forward_excess_return",
}
TARGET_ARTIFACT_NAME = "targets_forward.parquet"
HISTORICAL_TARGET_ARTIFACT_NAME = "targets_forward_3m.parquet"


def target_artifact_path(directory: Path) -> Path:
    """Devuelve el artefacto neutral y solo cae al nombre histórico para leer resultados previos."""
    canonical = Path(directory) / TARGET_ARTIFACT_NAME
    if canonical.exists():
        return canonical
    return Path(directory) / HISTORICAL_TARGET_ARTIFACT_NAME


def normalize_target_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Añade nombres neutrales al leer artefactos históricos.

    Los nombres ``*_3m`` eran históricos aunque el horizonte fuese configurable. La función no
    cambia valores y permite leer tanto resultados antiguos como nuevos, sin volver a generar
    columnas históricas en frames modernos.
    """
    result = frame.copy()
    for legacy, neutral in TARGET_ALIASES.items():
        if neutral not in result and legacy in result:
            result[neutral] = result[legacy]
    return result
