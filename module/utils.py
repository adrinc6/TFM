"""Shared filesystem and logging helpers."""

from __future__ import annotations

import json
import logging
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


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def price_cache_by_ticker(prices: pd.DataFrame) -> dict[str, tuple[list, list[float]]]:
    """Índice ticker -> (fechas ordenadas, adj_close) para búsqueda point-in-time con bisect. Fuente
    única compartida por ml y backtest (antes duplicada en ambos)."""
    cache: dict[str, tuple[list, list[float]]] = {}
    for ticker, group in prices.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        cache[ticker] = (group["date"].tolist(), group["adj_close"].astype(float).tolist())
    return cache
