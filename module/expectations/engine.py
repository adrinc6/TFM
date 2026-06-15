"""Estimate expected, implied and realized growth gaps.

The goal is not price prediction. This module makes the GARP thesis explicit:
the system wants businesses where future reality can exceed what valuation
appears to imply.
"""

from __future__ import annotations

import pandas as pd


def add_expectation_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["expected_growth"] = (
        0.45 * df["growth_score"] + 0.30 * df["quality_score"] + 0.25 * df["catalyst_score"]
    ).clip(0, 1)
    df["implied_growth"] = (1 - df["valuation_score"]).clip(0, 1)
    df["realized_growth"] = (
        0.60 * df["growth_score"] + 0.25 * df["quality_score"] + 0.15 * df["moat_score"]
    ).clip(0, 1)
    df["expectation_gap"] = (df["expected_growth"] - df["implied_growth"]).clip(-1, 1)
    df["positive_expectation_gap"] = ((df["expectation_gap"] + 1) / 2).clip(0, 1)
    return df
