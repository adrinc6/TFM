"""GARP feature pipeline."""

from __future__ import annotations

import logging

import pandas as pd

from environment import PROCESSED_DIR
from module.features.transforms import (
    TEMPORAL_FEATURES,
    add_expectation_features,
    add_relative_features,
    add_temporal_business_features,
)
from module.utils import read_parquet, write_parquet

log = logging.getLogger(__name__)


CORE_FEATURE_COLUMNS = [
    "quality_score",
    "growth_score",
    "valuation_score",
    "price_adjusted_valuation_score",
    "momentum_score",
    "moat_score",
    "catalyst_score",
    "risk_score",
    "garp_score",
    "quality_value_gap",
]

FEATURE_COLUMNS = [
    *CORE_FEATURE_COLUMNS,
    "implied_growth",
    "realized_growth",
    "expectation_gap",
    "positive_expectation_gap",
    "quality_score_vs_sector",
    "quality_score_vs_universe",
    "growth_score_vs_sector",
    "growth_score_vs_universe",
    "valuation_score_vs_sector",
    "valuation_score_vs_universe",
    "price_return_since_fundamental",
    "price_return_1m",
    "price_return_3m",
    "price_return_6m",
    "price_return_12m",
    "stale_fundamental_months",
    *TEMPORAL_FEATURES,
]


def build_features() -> pd.DataFrame:
    df = read_parquet(PROCESSED_DIR.parent / "master" / "master_point_in_time.parquet").copy()
    log.info(
        "Building features from master rows=%s snapshots=%s tickers=%s",
        len(df),
        df["snapshot_date"].nunique(),
        df["ticker"].nunique(),
    )
    df["quality_score"] = _mean_percentile(df, ["roe", "roic", "gross_margin", "operating_margin", "net_margin", "fcf_margin"])
    df["growth_score"] = _mean_percentile(df, ["revenue_growth", "eps_growth"])
    df["valuation_score"] = 1 - _mean_percentile(df, ["pe", "forward_pe", "peg"])
    # Price-adjusted valuation uses the multiples re-adjusted for price moves since the last
    # reported fundamental (pe/forward_pe/peg _price_adjusted from module/dataset.py), so a stock
    # that ran up after its last filing looks more expensive here than in the raw valuation_score.
    adjusted_multiples = [col for col in ("pe_price_adjusted", "forward_pe_price_adjusted", "peg_price_adjusted") if col in df.columns]
    if adjusted_multiples:
        df["price_adjusted_valuation_score"] = 1 - _mean_percentile(df, adjusted_multiples)
    else:
        df["price_adjusted_valuation_score"] = df["valuation_score"]
    df["momentum_score"] = _mean_percentile(df, ["price_return_3m", "price_return_6m", "price_return_12m"])
    # moat = durable capital efficiency (roic + fcf conversion); distinct from quality_score, which
    # is a broad profitability composite. Kept differentiated to avoid a duplicate signal.
    df["moat_score"] = _mean_percentile(df, ["roic", "fcf_margin"])
    # catalyst = earnings momentum relative to sales (eps outgrowing revenue) + recent price
    # momentum — a *change/surprise* signal, NOT a copy of growth_score's level of growth.
    if {"eps_growth", "revenue_growth"}.issubset(df.columns):
        df["_eps_vs_rev"] = pd.to_numeric(df["eps_growth"], errors="coerce") - pd.to_numeric(df["revenue_growth"], errors="coerce")
        df["catalyst_score"] = (
            df.groupby("snapshot_date")["_eps_vs_rev"].rank(pct=True).fillna(0.5)
            + _mean_percentile(df, ["price_return_3m", "price_return_6m"])
        ) / 2
        df = df.drop(columns=["_eps_vs_rev"])
    else:
        df["catalyst_score"] = df["momentum_score"]
    df["risk_score"] = 1 - _mean_percentile(df, ["debt_equity"])
    df["garp_score"] = (
        0.30 * df["quality_score"]
        + 0.20 * df["moat_score"]
        + 0.20 * df["growth_score"]
        + 0.15 * df["valuation_score"]
        + 0.08 * df["catalyst_score"]
        + 0.02 * df["momentum_score"]
        + 0.05 * df["risk_score"]
    )
    # "Undervalued quality" gap: how good the business is (quality+growth) net of how expensive it
    # is (price-adjusted). High = strong fundamentals trading cheap — the concept the project hunts.
    quality_growth = df.groupby("snapshot_date")[["quality_score", "growth_score"]].rank(pct=True).mean(axis=1)
    expensiveness = df.groupby("snapshot_date")["price_adjusted_valuation_score"].rank(pct=True)
    df["quality_value_gap"] = ((quality_growth - (1 - expensiveness)) + 1) / 2
    df["price_return_since_fundamental"] = df.get("price_return_since_fundamental", pd.Series(0, index=df.index)).fillna(0)
    for column in ["price_return_1m", "price_return_3m", "price_return_6m", "price_return_12m"]:
        df[column] = df.get(column, pd.Series(0, index=df.index)).fillna(0)
    df["stale_fundamental_months"] = df.get("stale_fundamental_months", pd.Series(0, index=df.index)).fillna(0)
    df[CORE_FEATURE_COLUMNS] = df[CORE_FEATURE_COLUMNS].fillna(0.5)
    df = add_expectation_features(df)
    df = add_relative_features(df)
    df = add_temporal_business_features(df)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0.5)
    write_parquet(df, PROCESSED_DIR / "features.parquet")
    log.info(
        "Feature dataset rows=%s snapshots=%s valuation_score_range=(%.3f, %.3f)",
        len(df),
        df["snapshot_date"].nunique(),
        float(df["valuation_score"].min()),
        float(df["valuation_score"].max()),
    )
    return df


def _mean_percentile(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    usable = [col for col in columns if col in df.columns]
    if not usable:
        return pd.Series(0.5, index=df.index)
    # Coerce to numeric before ranking: some multiples (notably peg/peg_price_adjusted) arrive as
    # object dtype from the parquet round-trip when they mix None and float, and .rank() on object
    # dtype would rank lexicographically or error. errors="coerce" turns any stray strings into NaN,
    # which rank(pct=True) skips per column, so valuation_score is computed on real numbers.
    numeric = df[usable].apply(pd.to_numeric, errors="coerce")
    numeric["snapshot_date"] = df["snapshot_date"].to_numpy()
    return numeric.groupby("snapshot_date")[usable].rank(pct=True).mean(axis=1)
