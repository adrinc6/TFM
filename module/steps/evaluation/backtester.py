from __future__ import annotations

import numpy as np
import pandas as pd


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
