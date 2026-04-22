from __future__ import annotations

import pandas as pd

from environment import ANALYSIS_FREQUENCY, TP_SL_PRIMARY_STRATEGY
from module.common.trading_core import EventOutcome, generate_strategy_targets
from module.common.utils import analysis_period_keys


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


def primary_label_column() -> str:
    return f"label_{TP_SL_PRIMARY_STRATEGY}"


def analysis_keys_for_dataframe(df: pd.DataFrame) -> pd.Series:
    return analysis_period_keys(df["snapshot_date"], frequency=ANALYSIS_FREQUENCY)


__all__ = [
    "EventOutcome",
    "generate_strategy_targets",
    "ratio_feature_candidates",
    "primary_label_column",
    "analysis_keys_for_dataframe",
]
