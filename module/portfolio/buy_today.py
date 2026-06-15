"""Buy Today Engine.

Separates "this position is worth holding" from "this business deserves new
capital today".
"""

from __future__ import annotations

import pandas as pd


def add_buy_today_decision(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rows = []
    for _, group in df.groupby("snapshot_date", sort=True):
        ranked = group.sort_values("thesis_rank_score", ascending=False)
        best_by_ticker = _best_alternatives(ranked)
        group = group.copy()
        group["best_alternative_ticker"] = group["ticker"].map(lambda ticker: best_by_ticker[ticker]["ticker"])
        group["best_alternative_score"] = group["ticker"].map(lambda ticker: best_by_ticker[ticker]["score"])
        group["opportunity_cost_score"] = (group["best_alternative_score"] - group["thesis_rank_score"]).clip(lower=0)
        group["buy_today_score"] = (
            0.30 * group["thesis_score"]
            + 0.25 * group["business_quality_score"]
            + 0.20 * group["positive_expectation_gap"]
            + 0.15 * group["valuation_score"]
            + 0.10 * (1 - group["opportunity_cost_score"].clip(0, 1))
        ).clip(0, 1)
        group["would_buy_today"] = (
            group["buy_today_score"] >= 0.58
        ) & group["thesis_state"].isin({"Improving", "Intact"}) & ~group["opportunity_type"].isin({"Avoid", "Value Trap"})
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def _best_alternatives(group: pd.DataFrame) -> dict[str, dict]:
    alternatives = {}
    records = group[["ticker", "thesis_rank_score"]].to_dict("records")
    for record in records:
        other = [candidate for candidate in records if candidate["ticker"] != record["ticker"]]
        if not other:
            alternatives[record["ticker"]] = {"ticker": "", "score": float(record["thesis_rank_score"])}
            continue
        best = max(other, key=lambda item: item["thesis_rank_score"])
        alternatives[record["ticker"]] = {"ticker": best["ticker"], "score": float(best["thesis_rank_score"])}
    return alternatives
