"""Moat and business quality analysis."""

from __future__ import annotations

import pandas as pd


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
