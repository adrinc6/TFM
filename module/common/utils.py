from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from environment import (
    ANALYSIS_FREQUENCY,
    FINNHUB_DATA_DIR,
    TP_SL_MAX_HOLDING_DAYS,
)


@dataclass(frozen=True)
class TradingStrategy:
    name: str
    tp_pct: float
    sl_pct: float


DEFAULT_STRATEGIES: tuple[TradingStrategy, ...] = (
    TradingStrategy(name="conservative", tp_pct=0.09, sl_pct=0.06),
    TradingStrategy(name="balanced", tp_pct=0.10, sl_pct=0.10),
    TradingStrategy(name="aggressive", tp_pct=0.15, sl_pct=0.065),
)


def strategies_map() -> Dict[str, TradingStrategy]:
    return {s.name: s for s in DEFAULT_STRATEGIES}


def project_data_dir(data_dir: str | Path = FINNHUB_DATA_DIR) -> Path:
    return Path(data_dir)


def load_master_dataset(data_dir: str | Path = FINNHUB_DATA_DIR) -> pd.DataFrame:
    path = project_data_dir(data_dir) / "master_dataset.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_parquet(path)
    if isinstance(df.index, pd.MultiIndex) and {"ticker", "date"}.issubset(df.index.names):
        out = df.copy()
    elif {"ticker", "date"}.issubset(df.columns):
        out = df.set_index(["ticker", "date"]).copy()
    else:
        raise ValueError("Master dataset must provide ticker/date")

    tickers = out.index.get_level_values("ticker").astype(str)
    dates = pd.to_datetime(out.index.get_level_values("date"), errors="coerce")
    valid = pd.notna(dates)
    if not valid.all():
        out = out[valid]
        tickers = tickers[valid]
        dates = dates[valid]

    out.index = pd.MultiIndex.from_arrays([tickers, dates], names=["ticker", "date"])
    out = out.sort_index()

    if "snapshot_date" in out.columns:
        out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    else:
        out["snapshot_date"] = out.index.get_level_values("date")

    if "year_quarter" not in out.columns:
        out["year_quarter"] = out["snapshot_date"].dt.to_period("Q").astype(str)

    if "sector" not in out.columns:
        out["sector"] = "Unknown"

    return out


def load_prices_for_ticker(ticker: str, data_dir: str | Path = FINNHUB_DATA_DIR) -> pd.DataFrame:
    path = project_data_dir(data_dir) / str(ticker) / "prices.json"
    if not path.exists():
        return pd.DataFrame()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()

    rows = payload.get("data", [])
    if not rows:
        return pd.DataFrame()

    prices = pd.DataFrame(rows)
    if "date" not in prices.columns:
        return pd.DataFrame()

    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.dropna(subset=["date"]).sort_values("date").set_index("date")
    prices = prices.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj_close": "Adj Close",
            "volume": "Volume",
        }
    )

    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in prices.columns:
            prices[col] = pd.to_numeric(prices[col], errors="coerce")

    prices = prices.dropna(subset=["Close"]).sort_index()
    return prices


def load_price_cache(tickers: Iterable[str], data_dir: str | Path = FINNHUB_DATA_DIR) -> Dict[str, pd.DataFrame]:
    unique = sorted({str(t) for t in tickers})
    return {ticker: load_prices_for_ticker(ticker, data_dir=data_dir) for ticker in unique}


def safe_numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.loc[:, list(columns)].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    med = out.median(numeric_only=True)
    out = out.fillna(med)
    out = out.fillna(0.0)
    return out


def analysis_period_keys(snapshot_dates: pd.Series, frequency: str = ANALYSIS_FREQUENCY) -> pd.Series:
    dt = pd.to_datetime(snapshot_dates, errors="coerce")
    freq = str(frequency).strip().lower()
    if freq == "annual":
        return dt.dt.to_period("Y").astype(str)
    return dt.dt.to_period("Q").astype(str)


def split_train_validation_by_time(df: pd.DataFrame, validation_ratio: float = 0.20, frequency: str = ANALYSIS_FREQUENCY) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = analysis_period_keys(df["snapshot_date"], frequency=frequency)
    periods = sorted(keys.dropna().unique().tolist())

    if len(periods) <= 2:
        split = max(int(len(df) * (1.0 - validation_ratio)), 1)
        return df.iloc[:split].copy(), df.iloc[split:].copy()

    n_val = max(int(len(periods) * validation_ratio), 1)
    val_periods = set(periods[-n_val:])

    train = df[keys.isin([p for p in periods if p not in val_periods])].copy()
    val = df[keys.isin(val_periods)].copy()

    if train.empty or val.empty:
        split = max(int(len(df) * (1.0 - validation_ratio)), 1)
        train, val = df.iloc[:split].copy(), df.iloc[split:].copy()

    return train, val


def forward_horizon_end(entry_date: pd.Timestamp, max_holding_days: int = TP_SL_MAX_HOLDING_DAYS) -> pd.Timestamp:
    return pd.Timestamp(entry_date) + pd.Timedelta(days=int(max_holding_days))
