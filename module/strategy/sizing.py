"""Position sizing intelligence.

Sizing deliberately leans into the winner/loser asymmetry the backtest shows (excess return is
positive on average even with a ~52% win rate): the book should bet *bigger* on its highest-
conviction ideas, not spread capital evenly. Two levers do this — a hybrid weight tilted mostly to
the conviction/risk-adjusted component (little equal-weight), and a convexity exponent that widens
the gap between top-conviction and merely-good names before weights are formed. The per-position cap
(MAX_POSITION_WEIGHT) acts as a risk brake (applied before renormalization, so it compresses rather
than hard-limits the top names) that keeps the book from collapsing onto a single name.
"""

from __future__ import annotations

import pandas as pd

MIN_POSITION_WEIGHT = 0.04
# Hard per-position cap: the risk limit on concentration. If in practice it binds on the highest-
# conviction names most months (check portfolio_allocation: hybrid_weight clustering at the cap),
# consider widening it to ~0.20-0.22 so conviction can express itself — a deliberate, reviewed
# change, not done here by default.
MAX_POSITION_WEIGHT = 0.18

# Share of the hybrid weight taken from the conviction/risk-adjusted component vs. flat equal weight.
# Tilted hard toward conviction (0.80) so sizing concentrates on the best ideas; equal weight is kept
# only as a small stabilizer against noisy single-snapshot conviction estimates.
EQUAL_WEIGHT_SHARE = 0.20
CONVICTION_WEIGHT_SHARE = 0.80
# Convexity applied to the conviction core (>1): on [0,1] scores this widens the top-vs-middle gap,
# rewarding the highest-conviction ideas disproportionately over the merely good ones.
CONVICTION_CONVEXITY = 1.5


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
        sized["hybrid_weight"] = (
            EQUAL_WEIGHT_SHARE * sized["equal_weight"] + CONVICTION_WEIGHT_SHARE * sized["risk_adjusted_weight"]
        ).clip(0, 1)
        sized["position_action"] = sized.apply(_position_action, axis=1)
        rows.append(sized)
    return pd.concat(rows, ignore_index=True)


def _sizing_score(df: pd.DataFrame) -> pd.Series:
    conviction = df["current_conviction_score"].fillna(0.5)
    manager = df.get("current_manager_score", conviction).fillna(0.5)
    # Conviction core (conviction + manager score) raised to the convexity exponent, so the top ideas
    # pull disproportionately more weight than the middle of the book before the supporting factors
    # (quality, buy-today, momentum, risk, expectation gap) are blended in linearly.
    conviction_core = (0.6 * conviction + 0.4 * manager).clip(0, 1) ** CONVICTION_CONVEXITY
    score = (
        0.52 * conviction_core
        + 0.18 * df.get("current_business_quality_score", conviction).fillna(0.5)
        + 0.12 * df.get("current_buy_today_score", conviction).fillna(0.5)
        + 0.08 * df.get("current_momentum_score", pd.Series(0.5, index=df.index)).fillna(0.5)
        + 0.06 * df.get("current_risk_score", pd.Series(0.5, index=df.index)).fillna(0.5)
        + 0.04 * df.get("current_positive_expectation_gap", pd.Series(0.5, index=df.index)).fillna(0.5)
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
