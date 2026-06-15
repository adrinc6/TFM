"""Parquet-first IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        raise ValueError(f"Refusing to write empty parquet: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError("Parquet support requires pyarrow or fastparquet.") from exc


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required parquet not found: {path}")
    return pd.read_parquet(path)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
