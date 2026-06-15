"""Position sizing intelligence."""

from __future__ import annotations

import pandas as pd


def add_position_sizing(holdings: pd.DataFrame) -> pd.DataFrame:
    if holdings.empty:
        return holdings
    rows = []
    for date, group in holdings.groupby("date", sort=True):
        sized = group.copy()
        n = len(sized)
        sized["equal_weight"] = 1 / n if n else 0
        conviction_sum = sized["current_conviction_score"].clip(lower=0).sum()
        if conviction_sum > 0:
            sized["conviction_weight"] = sized["current_conviction_score"].clip(lower=0) / conviction_sum
        else:
            sized["conviction_weight"] = sized["equal_weight"]
        sized["hybrid_weight"] = (0.50 * sized["equal_weight"] + 0.50 * sized["conviction_weight"]).clip(0, 1)
        rows.append(sized)
    return pd.concat(rows, ignore_index=True)
