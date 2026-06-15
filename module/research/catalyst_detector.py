"""Catalyst and rerating analysis."""

from __future__ import annotations

import pandas as pd


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
