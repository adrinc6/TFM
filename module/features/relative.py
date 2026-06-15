"""Relative metrics versus sector, industry and universe."""

from __future__ import annotations

import pandas as pd


BASE_RELATIVE_COLUMNS = [
    "quality_score",
    "growth_score",
    "valuation_score",
    "moat_score",
    "catalyst_score",
    "risk_score",
]


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "industry" not in df.columns:
        df["industry"] = df.get("sector", "Unknown")
    for column in BASE_RELATIVE_COLUMNS:
        df[f"{column}_vs_universe"] = _relative_rank(df, ["snapshot_date"], column)
        df[f"{column}_vs_sector"] = _relative_rank(df, ["snapshot_date", "sector"], column)
        df[f"{column}_vs_industry"] = _relative_rank(df, ["snapshot_date", "industry"], column)
    return df


def _relative_rank(df: pd.DataFrame, groups: list[str], column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.5, index=df.index)
    return df.groupby(groups)[column].rank(pct=True).fillna(0.5)
