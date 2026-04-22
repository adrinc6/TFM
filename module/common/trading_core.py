from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from environment import TP_SL_MAX_HOLDING_DAYS
from module.common.metrics import expected_value_from_probability


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


def strategies_map() -> Dict[str, TradingStrategy]:
    return {s.name: s for s in DEFAULT_STRATEGIES}


def _forward_horizon_end(entry_date: pd.Timestamp, max_holding_days: int = TP_SL_MAX_HOLDING_DAYS) -> pd.Timestamp:
    return pd.Timestamp(entry_date) + pd.Timedelta(days=int(max_holding_days))


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
    horizon = future.loc[future.index <= _forward_horizon_end(snapshot_date)]
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


def build_per_stock_diagnostics(strategy_df: pd.DataFrame, score_col: str = "meta_score") -> pd.DataFrame:
    if strategy_df is None or strategy_df.empty:
        return pd.DataFrame()

    df = strategy_df.copy()
    df["expected_value_model"] = expected_value_from_probability(df[score_col], df["tp_pct"], df["sl_pct"])
    sl = pd.to_numeric(df["sl_pct"], errors="coerce").fillna(0.0)
    sl_safe = sl.where(sl.abs() >= 1e-6, np.nan)
    df["risk_reward"] = pd.to_numeric(df["tp_pct"], errors="coerce").fillna(0.0) / sl_safe

    idx = df.groupby(["ticker", "date"])["expected_value_model"].idxmax()
    out = df.loc[idx].copy()
    out = out.sort_values(["snapshot_date", "expected_value_model"], ascending=[True, False])
    return out.reset_index(drop=True)


def add_portfolio_flag(diagnostics: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    out = diagnostics.copy()
    out["selected_in_portfolio"] = False
    if portfolio is None or portfolio.empty:
        return out

    selected = pd.MultiIndex.from_frame(portfolio[["ticker", "date"]])
    diag_index = pd.MultiIndex.from_frame(out[["ticker", "date"]])
    out["selected_in_portfolio"] = diag_index.isin(selected)
    return out


def construct_portfolio(
    diagnostics: pd.DataFrame,
    min_stocks: int = 5,
    max_stocks: int = 8,
    sector_cap: int = 3,
    score_col: str = "meta_score",
) -> pd.DataFrame:
    if diagnostics is None or diagnostics.empty:
        return pd.DataFrame()

    latest_date = pd.to_datetime(diagnostics["snapshot_date"], errors="coerce").max()
    universe = diagnostics[pd.to_datetime(diagnostics["snapshot_date"], errors="coerce") == latest_date].copy()
    if universe.empty:
        return pd.DataFrame()

    universe[score_col] = pd.to_numeric(universe[score_col], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    universe["expected_value_model"] = pd.to_numeric(universe["expected_value_model"], errors="coerce").fillna(0.0)
    universe["risk_reward"] = pd.to_numeric(universe["risk_reward"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
    universe = universe.sort_values([score_col, "expected_value_model", "risk_reward"], ascending=False)

    picks: list[dict] = []
    counts: dict[str, int] = {}
    selected_keys: set[tuple[str, pd.Timestamp]] = set()

    for _, row in universe.iterrows():
        if len(picks) >= max_stocks:
            break
        sector = str(row.get("sector", "Unknown"))
        if sector_cap > 0 and counts.get(sector, 0) >= sector_cap:
            continue
        picks.append(row.to_dict())
        counts[sector] = counts.get(sector, 0) + 1
        selected_keys.add((str(row["ticker"]), pd.Timestamp(row["date"])))

    if len(picks) < min_stocks:
        for _, row in universe.iterrows():
            if len(picks) >= min_stocks:
                break
            key = (str(row["ticker"]), pd.Timestamp(row["date"]))
            if key in selected_keys:
                continue
            picks.append(row.to_dict())
            selected_keys.add(key)

    portfolio = pd.DataFrame(picks)
    if portfolio.empty:
        return portfolio

    raw = portfolio[score_col].clip(lower=1e-8)
    portfolio["weight"] = raw / raw.sum()
    portfolio["selected_in_portfolio"] = True
    return portfolio.reset_index(drop=True)
