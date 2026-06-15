"""Position sizing intelligence."""

from __future__ import annotations

import pandas as pd

MIN_POSITION_WEIGHT = 0.04
MAX_POSITION_WEIGHT = 0.18


def add_position_sizing(holdings: pd.DataFrame) -> pd.DataFrame:
    if holdings.empty:
        return holdings
    rows = []
    for date, group in holdings.groupby("date", sort=True):
        sized = group.copy()
        n = len(sized)
        sized["equal_weight"] = 1 / n if n else 0
        sized["sizing_score"] = _sizing_score(sized)
        score_sum = sized["sizing_score"].clip(lower=0).sum()
        if score_sum > 0:
            sized["conviction_weight"] = sized["sizing_score"].clip(lower=0) / score_sum
        else:
            sized["conviction_weight"] = sized["equal_weight"]
        sized["risk_adjusted_weight"] = _bounded_normalize(sized["conviction_weight"], n)
        sized["hybrid_weight"] = (0.35 * sized["equal_weight"] + 0.65 * sized["risk_adjusted_weight"]).clip(0, 1)
        sized["position_action"] = sized.apply(_position_action, axis=1)
        rows.append(sized)
    return pd.concat(rows, ignore_index=True)


def _sizing_score(df: pd.DataFrame) -> pd.Series:
    score = (
        0.30 * df["current_conviction_score"].fillna(0.5)
        + 0.22 * df.get("current_manager_score", df["current_conviction_score"]).fillna(0.5)
        + 0.18 * df.get("current_business_quality_score", df["current_conviction_score"]).fillna(0.5)
        + 0.12 * df.get("current_buy_today_score", df["current_conviction_score"]).fillna(0.5)
        + 0.10 * df.get("current_risk_score", pd.Series(0.5, index=df.index)).fillna(0.5)
        + 0.08 * df.get("current_positive_expectation_gap", pd.Series(0.5, index=df.index)).fillna(0.5)
    )
    penalty = 1 - 0.30 * df.get("current_opportunity_cost_score", pd.Series(0, index=df.index)).fillna(0).clip(0, 1)
    return (score * penalty).clip(0, 1)


def _bounded_normalize(weights: pd.Series, n: int) -> pd.Series:
    if n == 0:
        return weights
    max_weight = min(MAX_POSITION_WEIGHT, 1 / max(n, 1) * 1.65)
    min_weight = min(MIN_POSITION_WEIGHT, 1 / max(n, 1))
    bounded = weights.clip(lower=min_weight, upper=max_weight)
    total = bounded.sum()
    return bounded / total if total > 0 else weights


def _position_action(row: pd.Series) -> str:
    if row.get("current_thesis_state") == "Improving" and row.get("current_would_buy_today") and row.get("sizing_score", 0) >= 0.65:
        return "ADD"
    if row.get("current_thesis_state") == "Maturing" or row.get("not_buy_today_count", 0) >= 2 or row.get("sizing_score", 1) < 0.48:
        return "TRIM"
    return "HOLD"
