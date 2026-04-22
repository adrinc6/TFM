from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from environment import FINNHUB_DATA_DIR


class DataLayer:
    def __init__(self, base_dir: str | Path = FINNHUB_DATA_DIR):
        self.base_dir = Path(base_dir)
        self.master_dataset_path = self.base_dir / "master_dataset.parquet"

    def load_master_dataset(self) -> pd.DataFrame:
        if not self.master_dataset_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {self.master_dataset_path}")
        df = pd.read_parquet(self.master_dataset_path)
        if isinstance(df.index, pd.MultiIndex) and set(df.index.names) >= {"ticker", "date"}:
            pass
        elif {"ticker", "date"}.issubset(df.columns):
            df = df.set_index(["ticker", "date"])
        else:
            raise ValueError("Master dataset must contain ticker/date index or columns")

        date_values = pd.to_datetime(df.index.get_level_values("date"), errors="coerce")
        valid = pd.notna(date_values)
        if not valid.all():
            df = df[valid]
            date_values = date_values[valid]

        index = pd.MultiIndex.from_arrays(
            [df.index.get_level_values("ticker").astype(str), date_values],
            names=["ticker", "date"],
        )
        df = df.copy()
        df.index = index
        df = df.sort_index()
        if "snapshot_date" in df.columns:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
        else:
            df["snapshot_date"] = date_values
        if "year_quarter" not in df.columns:
            yq = pd.PeriodIndex(df["snapshot_date"].dt.to_period("Q"), freq="Q")
            df["year_quarter"] = yq.astype(str)
        if "sector" not in df.columns:
            df["sector"] = "Unknown"
        return df

    def load_prices_for_ticker(self, ticker: str) -> pd.DataFrame:
        file_path = self.base_dir / ticker / "prices.json"
        if not file_path.exists():
            return pd.DataFrame()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
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
        rename = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj_close": "Adj Close",
            "volume": "Volume",
        }
        prices = prices.rename(columns=rename)
        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if col in prices.columns:
                prices[col] = pd.to_numeric(prices[col], errors="coerce")
        prices = prices.dropna(subset=["Close"]).sort_index()
        return prices

    def load_price_cache(self, tickers: Iterable[str]) -> Dict[str, pd.DataFrame]:
        return {ticker: self.load_prices_for_ticker(str(ticker)) for ticker in sorted(set(map(str, tickers)))}
