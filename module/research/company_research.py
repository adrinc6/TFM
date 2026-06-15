"""Business summaries derived from structured features."""

from __future__ import annotations

import pandas as pd


def describe_company(row: pd.Series) -> str:
    return (
        f"{row['ticker']} is classified as {row.get('sector', 'Unknown')} with "
        f"business quality {row['quality_score']:.2f}, moat {row['moat_score']:.2f}, "
        f"growth {row['growth_score']:.2f} and risk protection {row['risk_score']:.2f}."
    )


def summarize_business(row: pd.Series) -> str:
    if row["business_quality_score"] >= 0.70:
        return "High-quality business profile with attractive durability signals."
    if row["business_quality_score"] >= 0.55:
        return "Acceptable business profile that requires valuation and catalyst support."
    return "Weak business profile; any investment case must be treated cautiously."


def summarize_risks(row: pd.Series) -> str:
    risks = []
    if row["risk_score"] < 0.45:
        risks.append("balance sheet or financial risk")
    if row["valuation_score"] < 0.35:
        risks.append("valuation already discounts too much optimism")
    if row["growth_score"] < 0.40:
        risks.append("weak growth profile")
    return ", ".join(risks) if risks else "No dominant quantitative risk flag."


def summarize_opportunities(row: pd.Series) -> str:
    opportunities = []
    if row["valuation_score"] >= 0.60:
        opportunities.append("reasonable or attractive valuation")
    if row["catalyst_score"] >= 0.60:
        opportunities.append("possible catalyst or acceleration")
    if row["moat_score"] >= 0.65:
        opportunities.append("durable competitive position")
    return ", ".join(opportunities) if opportunities else "Needs clearer upside trigger."
