"""Generate analyst-like investment thesis narratives."""

from __future__ import annotations

import pandas as pd

from module.research.catalyst_detector import detect_catalyst, entry_trigger
from module.research.company_research import (
    describe_company,
    summarize_business,
    summarize_opportunities,
    summarize_risks,
)
from module.research.moat_analyzer import analyze_moat, moat_durability
from module.research.news_analyzer import summarize_news


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
    df["base_thesis"] = df.apply(_base_thesis, axis=1)
    df["bull_thesis"] = df.apply(_bull_thesis, axis=1)
    df["bear_thesis"] = df.apply(_bear_thesis, axis=1)
    df["investment_thesis"] = df.apply(_investment_thesis, axis=1)
    return df


def _investment_thesis(row: pd.Series) -> str:
    return (
        f"{row['opportunity_type']}: {row['business_summary']} "
        f"{row['moat_analysis']} {row['catalyst']}"
    )


def _base_thesis(row: pd.Series) -> str:
    return (
        f"Base case: business quality {row['business_quality_score']:.2f}, "
        f"valuation {row['valuation_score']:.2f}, catalyst {row['catalyst_score']:.2f}."
    )


def _bull_thesis(row: pd.Series) -> str:
    return f"Bull case: {row['opportunity_summary']}."


def _bear_thesis(row: pd.Series) -> str:
    return f"Bear case: {row['risk_summary']}."
