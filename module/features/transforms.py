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
    """Expectation-gap features.

    `expected_growth` (the market's implied expectation) is a proxy derived from valuation:
    a low valuation_score (expensive) implies the market prices in high growth. `realized_growth`
    is the **actually observed** fundamental growth (cross-sectional percentile of the reported
    revenue/eps growth from module/dataset.py's `_historical_growth`), NOT a deterministic
    re-projection of the input scores. The gap between the two is what forward targets exploit.
    """
    df = df.copy()
    # Market-implied growth expectation: cheaper multiples => lower implied growth.
    df["implied_growth"] = (1 - df["valuation_score"]).clip(0, 1)
    df["expected_growth"] = df["implied_growth"]
    # Observed fundamental growth, ranked cross-sectionally so it stays on [0, 1] and comparable.
    reported = [col for col in ("revenue_growth", "eps_growth") if col in df.columns]
    if reported:
        df["realized_growth"] = (
            df.groupby("snapshot_date")[reported].rank(pct=True).mean(axis=1).clip(0, 1)
        )
    else:  # pragma: no cover - defensive; master always carries these
        df["realized_growth"] = df["growth_score"].clip(0, 1)
    # Gap: observed fundamental growth beating the market's implied expectation.
    df["expectation_gap"] = (df["realized_growth"] - df["implied_growth"]).clip(-1, 1)
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
    """current - value `months` ago for the same ticker, via merge_asof (O(n log n)).

    For each row, finds the same ticker's most recent snapshot at or before
    `snapshot_date - months` and subtracts that value from the current one. NaN gaps -> 0.
    """
    if column not in df.columns:
        return pd.Series(0.0, index=df.index)
    left = df[["ticker", "snapshot_date_dt", column]].copy()
    left["_orig_index"] = df.index
    left["_lookup_date"] = left["snapshot_date_dt"] - pd.DateOffset(months=months)
    left = left.sort_values("_lookup_date")
    right = df[["ticker", "snapshot_date_dt", column]].rename(
        columns={"snapshot_date_dt": "_hist_date", column: "_prev"}
    ).sort_values("_hist_date")
    merged = pd.merge_asof(
        left, right, left_on="_lookup_date", right_on="_hist_date",
        by="ticker", direction="backward",
    )
    delta = (merged[column] - merged["_prev"]).fillna(0.0).astype(float)
    return pd.Series(delta.to_numpy(), index=merged["_orig_index"].to_numpy()).reindex(df.index)
