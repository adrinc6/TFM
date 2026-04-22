from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from environment import (
    ANALYSIS_FREQUENCY,
    TP_SL_MAX_HOLDING_DAYS,
    TP_SL_PRIMARY_STRATEGY,
)
from module.common.utils import (
    TradingStrategy,
    analysis_period_keys,
    forward_horizon_end,
    strategies_map,
)


@dataclass(frozen=True)
class EventOutcome:
    strategy: str
    tp_pct: float
    sl_pct: float
    entry_price: float
    tp_level: float
    sl_level: float
    outcome: str
    label: int
    days_to_event: int


def ratio_feature_candidates(df: pd.DataFrame) -> list[str]:
    deny_prefixes = ("label_", "outcome_", "days_to_event_", "tp_level_", "sl_level_", "entry_price_")
    deny_cols = {"forward_return", "snapshot_date", "year_quarter", "sector", "industry"}
    candidates: list[str] = []
    for col in df.columns:
        if col in deny_cols or col.startswith(deny_prefixes):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        name = str(col)
        if (
            "_ratio" in name
            or "_margin" in name
            or "_yield" in name
            or name.endswith("_pct")
            or "momentum" in name
            or "volatility" in name
            or "trend" in name
            or "rsi" in name
            or name.startswith("bf_")
            or name.startswith("seq_")
            or "coverage" in name
            or "equity" in name
            or "debt" in name
        ):
            candidates.append(name)
    return candidates


def evaluate_forward_tp_sl(prices: pd.DataFrame, snapshot_date: pd.Timestamp, strategy: TradingStrategy) -> EventOutcome:
    if prices is None or prices.empty:
        return EventOutcome(strategy.name, strategy.tp_pct, strategy.sl_pct, np.nan, np.nan, np.nan, "NO_DATA", 0, TP_SL_MAX_HOLDING_DAYS)

    snapshot_date = pd.Timestamp(snapshot_date)
    future = prices.loc[prices.index >= snapshot_date].copy()
    if future.empty:
        return EventOutcome(strategy.name, strategy.tp_pct, strategy.sl_pct, np.nan, np.nan, np.nan, "NO_DATA", 0, TP_SL_MAX_HOLDING_DAYS)

    entry_price = float(future.iloc[0].get("Close", np.nan))
    if not np.isfinite(entry_price) or entry_price <= 0:
        return EventOutcome(strategy.name, strategy.tp_pct, strategy.sl_pct, np.nan, np.nan, np.nan, "NO_DATA", 0, TP_SL_MAX_HOLDING_DAYS)

    tp_level = entry_price * (1.0 + strategy.tp_pct)
    sl_level = entry_price * (1.0 - strategy.sl_pct)
    horizon = future.loc[future.index <= forward_horizon_end(snapshot_date)]
    if horizon.empty:
        horizon = future

    for dt, row in horizon.iterrows():
        high = float(row.get("High", row.get("Close", np.nan)))
        low = float(row.get("Low", row.get("Close", np.nan)))
        days = max((pd.Timestamp(dt) - snapshot_date).days, 0)
        tp_hit = np.isfinite(high) and high >= tp_level
        sl_hit = np.isfinite(low) and low <= sl_level

        if tp_hit and sl_hit:
            return EventOutcome(strategy.name, strategy.tp_pct, strategy.sl_pct, entry_price, tp_level, sl_level, "SL_FIRST", 0, days)
        if tp_hit:
            return EventOutcome(strategy.name, strategy.tp_pct, strategy.sl_pct, entry_price, tp_level, sl_level, "TP_FIRST", 1, days)
        if sl_hit:
            return EventOutcome(strategy.name, strategy.tp_pct, strategy.sl_pct, entry_price, tp_level, sl_level, "SL_FIRST", 0, days)

    return EventOutcome(strategy.name, strategy.tp_pct, strategy.sl_pct, entry_price, tp_level, sl_level, "NO_HIT", 0, TP_SL_MAX_HOLDING_DAYS)


def generate_strategy_targets(master_df: pd.DataFrame, prices_cache: Dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategies = strategies_map()
    data = master_df.reset_index().copy()
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="coerce")
    data = data.dropna(subset=["snapshot_date"])

    main_records: list[dict] = []
    strategy_records: list[dict] = []

    for row in data.itertuples(index=False):
        ticker = str(getattr(row, "ticker"))
        date = pd.Timestamp(getattr(row, "date"))
        snapshot_date = pd.Timestamp(getattr(row, "snapshot_date"))
        price_df = prices_cache.get(ticker, pd.DataFrame())

        rec = row._asdict()
        for strategy_name, strategy in strategies.items():
            ev = evaluate_forward_tp_sl(price_df, snapshot_date, strategy)
            rec[f"label_{strategy_name}"] = int(ev.label)
            rec[f"outcome_{strategy_name}"] = ev.outcome
            rec[f"days_to_event_{strategy_name}"] = int(ev.days_to_event)
            rec[f"entry_price_{strategy_name}"] = float(ev.entry_price) if np.isfinite(ev.entry_price) else np.nan
            rec[f"tp_level_{strategy_name}"] = float(ev.tp_level) if np.isfinite(ev.tp_level) else np.nan
            rec[f"sl_level_{strategy_name}"] = float(ev.sl_level) if np.isfinite(ev.sl_level) else np.nan

            strategy_records.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "snapshot_date": snapshot_date,
                    "year_quarter": rec.get("year_quarter"),
                    "sector": rec.get("sector", "Unknown"),
                    "strategy": strategy_name,
                    "tp_pct": strategy.tp_pct,
                    "sl_pct": strategy.sl_pct,
                    "entry_price": rec[f"entry_price_{strategy_name}"],
                    "tp_level": rec[f"tp_level_{strategy_name}"],
                    "sl_level": rec[f"sl_level_{strategy_name}"],
                    "actual_outcome": ev.outcome,
                    "label": int(ev.label),
                    "days_to_event": int(ev.days_to_event),
                }
            )

        main_records.append(rec)

    target_df = pd.DataFrame(main_records).set_index(["ticker", "date"]).sort_index()
    strategy_df = pd.DataFrame(strategy_records).set_index(["ticker", "date"]).sort_index()
    return target_df, strategy_df


def primary_label_column() -> str:
    return f"label_{TP_SL_PRIMARY_STRATEGY}"


def analysis_keys_for_dataframe(df: pd.DataFrame) -> pd.Series:
    return analysis_period_keys(df["snapshot_date"], frequency=ANALYSIS_FREQUENCY)
