"""Position thesis, health, conviction and exit scoring."""

from __future__ import annotations

import pandas as pd

from module.portfolio.buy_today import add_buy_today_decision
from module.research.thesis_generator import generate_research


def enrich_with_thesis_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = generate_research(df)
    df["thesis_score"] = (
        0.30 * df["business_quality_score"]
        + 0.25 * df["moat_score"]
        + 0.20 * df["growth_score"]
        + 0.15 * df["valuation_score"]
        + 0.10 * df["catalyst_score"]
    ).clip(0, 1)
    df["position_health_score"] = (
        0.35 * df["business_quality_score"]
        + 0.25 * df["moat_score"]
        + 0.20 * df["risk_score"]
        + 0.20 * df["catalyst_score"]
    ).clip(0, 1)
    df["conviction_score"] = (
        0.45 * df["thesis_score"]
        + 0.30 * df["position_health_score"]
        + 0.15 * df["final_score"]
        + 0.10 * df["valuation_score"]
    ).clip(0, 1)
    df["thesis_rank_score"] = (
        0.45 * df["thesis_score"]
        + 0.30 * df["business_quality_score"]
        + 0.15 * df["conviction_score"]
        + 0.10 * df["valuation_score"]
    ).clip(0, 1)
    df["exit_score"] = (
        1 - (0.45 * df["thesis_score"] + 0.35 * df["position_health_score"] + 0.20 * df["valuation_score"])
    ).clip(0, 1)
    df["thesis_state"] = df.apply(_state, axis=1)
    df = add_buy_today_decision(df)
    df["exit_thesis"] = df.apply(_exit_thesis, axis=1)
    df["buy_reason"] = df.apply(_buy_reason, axis=1)
    return df


def _state(row: pd.Series) -> str:
    if row["thesis_score"] >= 0.78 and row["position_health_score"] >= 0.70:
        return "Improving"
    if row["thesis_score"] >= 0.62:
        return "Intact"
    if row["valuation_score"] < 0.25 and row["quality_score"] >= 0.65:
        return "Maturing"
    if row["thesis_score"] >= 0.45:
        return "Weakening"
    return "Broken"


def _buy_reason(row: pd.Series) -> str:
    return (
        f"{row['opportunity_type']}: {row['investment_thesis']} "
        f"conviction={row['conviction_score']:.2f}"
    )


def _exit_thesis(row: pd.Series) -> str:
    reasons = []
    if row["thesis_state"] == "Broken":
        reasons.append("the investment thesis is broken")
    if row["position_health_score"] < 0.45:
        reasons.append("position health has deteriorated")
    if row["valuation_score"] < 0.25:
        reasons.append("valuation no longer offers a reasonable margin")
    if row["expectation_gap"] < -0.20:
        reasons.append("market expectations appear too optimistic versus expected fundamentals")
    if row["risk_score"] < 0.30:
        reasons.append("financial risk has increased")
    if not reasons:
        reasons.append("exit only if opportunity cost becomes materially superior or thesis persistence weakens")
    return f"Exit thesis for {row['ticker']}: " + "; ".join(reasons) + "."
