"""Feature transforms used by the GARP feature pipeline."""

from __future__ import annotations

import pandas as pd


TEMPORAL_FEATURES = [
    "quality_trend_1y",
    "quality_trend_2y",
    "roic_trend",
    "margin_trend",
    "fcf_trend",
    "growth_acceleration",
    "growth_deceleration",
    "moat_trend",
]


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


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    comparisons = [
        ("quality_score", "quality_score_vs"),
        ("growth_score", "growth_score_vs"),
        ("valuation_score", "valuation_score_vs"),
    ]
    for column, prefix in comparisons:
        df[f"{prefix}_sector"] = _relative_rank(df, ["snapshot_date", "sector"], column)
        df[f"{prefix}_universe"] = _relative_rank(df, ["snapshot_date"], column)
    return df


def add_temporal_business_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["snapshot_date_dt"] = pd.to_datetime(df["snapshot_date"])
    df = df.sort_values(["ticker", "snapshot_date_dt"])
    df["quality_trend_1y"] = _historical_delta(df, "quality_score", months=12)
    df["quality_trend_2y"] = _historical_delta(df, "quality_score", months=24)
    df["roic_trend"] = _historical_delta(df, "roic", months=12)
    df["margin_composite"] = df[["gross_margin", "operating_margin", "net_margin", "fcf_margin"]].mean(axis=1)
    df["margin_trend"] = _historical_delta(df, "margin_composite", months=12)
    df["fcf_trend"] = _historical_delta(df, "fcf_margin", months=12)
    growth_trend = _historical_delta(df, "growth_score", months=12)
    df["growth_acceleration"] = growth_trend.clip(lower=0)
    df["growth_deceleration"] = (-growth_trend).clip(lower=0)
    df["moat_trend"] = _historical_delta(df, "moat_score", months=12)
    df[TEMPORAL_FEATURES] = df[TEMPORAL_FEATURES].fillna(0.0).clip(-1, 1)
    return df.drop(columns=["snapshot_date_dt", "margin_composite"])


def _relative_rank(df: pd.DataFrame, groups: list[str], column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.5, index=df.index)
    return df.groupby(groups)[column].rank(pct=True)


def _historical_delta(df: pd.DataFrame, column: str, months: int) -> pd.Series:
    values = []
    for row in df.itertuples(index=False):
        history = df[
            (df["ticker"] == row.ticker)
            & (df["snapshot_date_dt"] <= row.snapshot_date_dt - pd.DateOffset(months=months))
        ].sort_values("snapshot_date_dt")
        if history.empty or column not in df.columns:
            values.append(0.0)
            continue
        current = getattr(row, column, None)
        previous = history.iloc[-1][column]
        values.append(0.0 if pd.isna(current) or pd.isna(previous) else float(current) - float(previous))
    return pd.Series(values, index=df.index)
