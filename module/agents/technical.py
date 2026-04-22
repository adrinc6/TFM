from __future__ import annotations

from module.agents.base import BaseAgent


TECHNICAL_FEATURE_COLUMNS = [
    "rsi_14",
    "rsi_28",
    "macd",
    "macd_signal",
    "macd_hist",
    "sma_20",
    "sma_50",
    "sma_200",
    "bb_pct",
    "atr_14",
    "volatility_20d",
    "volatility_60d",
    "vol_ratio_20_50",
    "price_vs_52w_high",
    "price_vs_52w_low",
]


class TechnicalAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="technical", feature_pool=TECHNICAL_FEATURE_COLUMNS, min_features=8, max_features=12)
