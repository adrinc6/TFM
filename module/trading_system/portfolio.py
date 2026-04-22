from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from environment import TP_SL_MAX_STOCKS, TP_SL_MIN_STOCKS, TP_SL_SECTOR_CAP

WEIGHT_FLOOR = 1e-8


def construct_portfolio(
    stock_diagnostics: pd.DataFrame,
    as_of: Optional[pd.Timestamp] = None,
    min_positions: int = TP_SL_MIN_STOCKS,
    max_positions: int = TP_SL_MAX_STOCKS,
    sector_cap: int = TP_SL_SECTOR_CAP,
    score_col: str = "meta_score",
) -> pd.DataFrame:
    if stock_diagnostics is None or stock_diagnostics.empty:
        return pd.DataFrame()

    df = stock_diagnostics.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")

    if as_of is None:
        as_of = df["snapshot_date"].max()
    as_of = pd.Timestamp(as_of)

    universe = df[df["snapshot_date"] == as_of].copy()
    if universe.empty:
        return pd.DataFrame()

    universe[score_col] = pd.to_numeric(universe[score_col], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    universe["risk_reward"] = pd.to_numeric(universe.get("risk_reward", np.nan), errors="coerce")
    universe["risk_reward"] = universe["risk_reward"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    universe["expected_value_model"] = pd.to_numeric(universe.get("expected_value_model", np.nan), errors="coerce")
    universe["expected_value_model"] = universe["expected_value_model"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    universe = universe.sort_values([score_col, "expected_value_model", "risk_reward"], ascending=False)

    selected_rows = []
    selected_keys: set[tuple[str, pd.Timestamp]] = set()
    sector_counts: dict[str, int] = {}

    for _, row in universe.iterrows():
        if len(selected_rows) >= max_positions:
            break
        sector = str(row.get("sector", "Unknown"))
        if sector_cap > 0 and sector_counts.get(sector, 0) >= sector_cap:
            continue
        selected_rows.append(row)
        selected_keys.add((str(row["ticker"]), pd.Timestamp(row["date"])))
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    if len(selected_rows) < min_positions:
        for _, row in universe.iterrows():
            if len(selected_rows) >= min_positions:
                break
            row_key = (str(row["ticker"]), pd.Timestamp(row["date"]))
            if row_key in selected_keys:
                continue
            selected_rows.append(row)
            selected_keys.add(row_key)

    portfolio = pd.DataFrame(selected_rows)
    if portfolio.empty:
        return portfolio

    raw_weight = portfolio[score_col].clip(lower=WEIGHT_FLOOR)
    portfolio["weight"] = raw_weight / raw_weight.sum()
    portfolio["selected_in_portfolio"] = True

    return portfolio.reset_index(drop=True)
