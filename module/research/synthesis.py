"""Deterministic thesis and narrative synthesis."""

from __future__ import annotations

import pandas as pd


def generate_research(universe: pd.DataFrame) -> pd.DataFrame:
    df = universe.copy()
    df["company_description"] = df.apply(describe_company, axis=1)
    df["business_summary"] = df.apply(summarize_business, axis=1)
    df["moat_analysis"] = df.apply(analyze_moat, axis=1)
    df["moat_durability"] = df.apply(moat_durability, axis=1)
    df["catalyst"] = df.apply(detect_catalyst, axis=1)
    df["news_summary"] = df.apply(summarize_news, axis=1)
    df["risk_summary"] = df.apply(summarize_risks, axis=1)
    df["opportunity_summary"] = df.apply(summarize_opportunities, axis=1)
    df["entry_trigger"] = df.apply(entry_trigger, axis=1)
    df["base_thesis"] = df.apply(base_thesis, axis=1)
    df["bull_thesis"] = df.apply(bull_thesis, axis=1)
    df["bear_thesis"] = df.apply(bear_thesis, axis=1)
    df["investment_thesis"] = df.apply(investment_thesis, axis=1)
    return df


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


def analyze_moat(row: pd.Series) -> str:
    if row["moat_score"] >= 0.75:
        return "Strong moat: persistent profitability and margin quality support durability."
    if row["moat_score"] >= 0.55:
        return "Moderate moat: business quality exists but requires continued monitoring."
    return "Weak moat: limited evidence of durable competitive advantage."


def moat_durability(row: pd.Series) -> str:
    if row["moat_score"] >= 0.70 and row["quality_score"] >= 0.65:
        return "Durable"
    if row["moat_score"] >= 0.50:
        return "Unproven"
    return "Fragile"


def summarize_news(row: pd.Series) -> str:
    return "No structured news signal integrated in this run; decision relies on financial and thesis evidence."


def detect_catalyst(row: pd.Series) -> str:
    if row["catalyst_score"] >= 0.70 and row["valuation_score"] >= 0.50:
        return "Growth acceleration or expectation gap may support rerating."
    if row["catalyst_score"] >= 0.55:
        return "Possible catalyst, but confirmation is required at future reviews."
    if row["valuation_score"] >= 0.75:
        return "Valuation gap exists, but catalyst evidence is limited."
    return "No clear catalyst detected."


def entry_trigger(row: pd.Series) -> str:
    if row["valuation_score"] < 0.40:
        return "Wait for a better valuation or stronger evidence of durable growth."
    if row["catalyst_score"] < 0.50:
        return "Wait for clearer catalyst confirmation."
    if row["business_quality_score"] < 0.55:
        return "Wait for business quality improvement."
    return "Eligible if portfolio capacity and thesis persistence remain favorable."


def investment_thesis(row: pd.Series) -> str:
    return (
        f"{row['opportunity_type']}: {row['business_summary']} "
        f"{row['moat_analysis']} {row['catalyst']}"
    )


def base_thesis(row: pd.Series) -> str:
    valuation = row.get("price_adjusted_valuation_score", row["valuation_score"])
    return (
        f"Base case: business quality {row['business_quality_score']:.2f}, "
        f"price-adjusted valuation {valuation:.2f}, catalyst {row['catalyst_score']:.2f}."
    )


def bull_thesis(row: pd.Series) -> str:
    return f"Bull case: {row['opportunity_summary']}."


def bear_thesis(row: pd.Series) -> str:
    return f"Bear case: {row['risk_summary']}."
