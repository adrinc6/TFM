from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyResult:
    strategy: str
    tp_pct: float
    sl_pct: float
    entry_date: pd.Timestamp
    entry_price: float
    tp_level: float
    sl_level: float
    outcome: str
    label: int
    days_to_event: int


class BaseStrategy:
    name: str = "base"
    tp_pct: float = 0.10
    sl_pct: float = 0.10

    def generate_levels(self, entry_price: float) -> tuple[float, float]:
        return entry_price * (1.0 + self.tp_pct), entry_price * (1.0 - self.sl_pct)

    def evaluate_outcome(
        self,
        prices: pd.DataFrame,
        entry_date: pd.Timestamp,
        max_holding_days: int = 90,
    ) -> StrategyResult:
        if prices is None or prices.empty:
            return StrategyResult(
                strategy=self.name,
                tp_pct=self.tp_pct,
                sl_pct=self.sl_pct,
                entry_date=pd.Timestamp(entry_date),
                entry_price=np.nan,
                tp_level=np.nan,
                sl_level=np.nan,
                outcome="NO_DATA",
                label=0,
                days_to_event=max_holding_days,
            )

        entry_date = pd.Timestamp(entry_date)
        future = prices.loc[prices.index >= entry_date].copy()
        if future.empty:
            return StrategyResult(
                strategy=self.name,
                tp_pct=self.tp_pct,
                sl_pct=self.sl_pct,
                entry_date=entry_date,
                entry_price=np.nan,
                tp_level=np.nan,
                sl_level=np.nan,
                outcome="NO_DATA",
                label=0,
                days_to_event=max_holding_days,
            )

        first = future.iloc[0]
        entry_price = float(first.get("Close", np.nan))
        if not np.isfinite(entry_price) or entry_price <= 0:
            return StrategyResult(
                strategy=self.name,
                tp_pct=self.tp_pct,
                sl_pct=self.sl_pct,
                entry_date=entry_date,
                entry_price=np.nan,
                tp_level=np.nan,
                sl_level=np.nan,
                outcome="NO_DATA",
                label=0,
                days_to_event=max_holding_days,
            )

        tp_level, sl_level = self.generate_levels(entry_price)
        window_end = entry_date + pd.Timedelta(days=max_holding_days)
        horizon = future.loc[future.index <= window_end]
        if horizon.empty:
            horizon = future

        for current_date, row in horizon.iterrows():
            high = float(row.get("High", row.get("Close", np.nan)))
            low = float(row.get("Low", row.get("Close", np.nan)))
            days_to_event = max((pd.Timestamp(current_date) - entry_date).days, 0)
            tp_hit = np.isfinite(high) and high >= tp_level
            sl_hit = np.isfinite(low) and low <= sl_level

            if tp_hit and sl_hit:
                return StrategyResult(
                    strategy=self.name,
                    tp_pct=self.tp_pct,
                    sl_pct=self.sl_pct,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    tp_level=tp_level,
                    sl_level=sl_level,
                    outcome="SL_FIRST",
                    label=0,
                    days_to_event=days_to_event,
                )
            if tp_hit:
                return StrategyResult(
                    strategy=self.name,
                    tp_pct=self.tp_pct,
                    sl_pct=self.sl_pct,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    tp_level=tp_level,
                    sl_level=sl_level,
                    outcome="TP_FIRST",
                    label=1,
                    days_to_event=days_to_event,
                )
            if sl_hit:
                return StrategyResult(
                    strategy=self.name,
                    tp_pct=self.tp_pct,
                    sl_pct=self.sl_pct,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    tp_level=tp_level,
                    sl_level=sl_level,
                    outcome="SL_FIRST",
                    label=0,
                    days_to_event=days_to_event,
                )

        return StrategyResult(
            strategy=self.name,
            tp_pct=self.tp_pct,
            sl_pct=self.sl_pct,
            entry_date=entry_date,
            entry_price=entry_price,
            tp_level=tp_level,
            sl_level=sl_level,
            outcome="NO_HIT",
            label=0,
            days_to_event=max_holding_days,
        )


class ConservativeStrategy(BaseStrategy):
    name = "conservative"
    tp_pct = 0.09
    sl_pct = 0.06


class BalancedStrategy(BaseStrategy):
    name = "balanced"
    tp_pct = 0.10
    sl_pct = 0.10


class AggressiveStrategy(BaseStrategy):
    name = "aggressive"
    tp_pct = 0.15
    sl_pct = 0.065


def build_strategies() -> Dict[str, BaseStrategy]:
    strategies = [ConservativeStrategy(), BalancedStrategy(), AggressiveStrategy()]
    return {s.name: s for s in strategies}
