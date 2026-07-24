"""Contrato único de targets, neutral respecto al horizonte."""

from pathlib import Path

import pandas as pd


TARGET_ARTIFACT_NAME = "targets_forward.parquet"
TARGET_COLUMNS = (
    "forward_return",
    "forward_benchmark_return",
    "forward_excess_return",
)


def target_artifact_path(directory: Path) -> Path:
    return Path(directory) / TARGET_ARTIFACT_NAME


def normalize_target_columns(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in TARGET_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"El target no cumple el contrato vigente; faltan {missing}.")
    return frame
