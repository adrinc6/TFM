"""Cross-sectional normalized feature engineering for ranking models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_pct_rank(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    if vals.notna().sum() <= 1:
        return pd.Series(0.5, index=s.index, dtype=float)
    return vals.rank(pct=True, method="average").fillna(0.5)


def _numeric_col_or_default(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        source = df[col]
    else:
        source = pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(source, errors="coerce")


def enrich_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sector-relative and volatility-adjusted factors plus interactions."""
    if df is None or df.empty:
        return df

    out = df.copy()
    if not isinstance(out.index, pd.MultiIndex) or "date" not in out.index.names:
        return out

    dates = out.index.get_level_values("date")
    if "sector" in out.columns:
        grp_sector = [dates, out["sector"].astype(str)]

        # Three complementary sector-relative valuation/quality ranks kept for
        # the MetaLearner.  PE and P/B ranks dropped: pe_rank_sector is
        # redundant with ev_ebitda and fcf_yield, while pb_rank_sector is
        # distorted by sector capital-structure differences.
        if "fcf_yield" in out.columns:
            out["fcf_yield_rank_sector"] = out.groupby(grp_sector)["fcf_yield"].transform(_safe_pct_rank)
        if "roic" in out.columns:
            out["roic_rank_sector"] = out.groupby(grp_sector)["roic"].transform(_safe_pct_rank)
        if "ev_to_ebitda" in out.columns:
            # Lower EV/EBITDA = cheaper → invert so higher rank = better value
            out["ev_ebitda_rank_sector"] = 1.0 - out.groupby(grp_sector)["ev_to_ebitda"].transform(_safe_pct_rank)
        value_parts = []
        if "fcf_yield" in out.columns:
            value_parts.append(out.groupby(grp_sector)["fcf_yield"].transform(_safe_pct_rank))
        if "earnings_yield" in out.columns:
            value_parts.append(out.groupby(grp_sector)["earnings_yield"].transform(_safe_pct_rank))
        if "ev_to_ebitda" in out.columns:
            value_parts.append(1.0 - out.groupby(grp_sector)["ev_to_ebitda"].transform(_safe_pct_rank))
        if "pe_ratio" in out.columns:
            value_parts.append(1.0 - out.groupby(grp_sector)["pe_ratio"].transform(_safe_pct_rank))
        if value_parts:
            out["valuation_percentile_sector"] = pd.concat(value_parts, axis=1).mean(axis=1).fillna(0.5)

    # Universe-wide percentile ranks (cross-sectional factor exposures)
    if "roic" in out.columns:
        out["quality_rank_universe"] = out.groupby(dates)["roic"].transform(_safe_pct_rank)
    if "earnings_yield" in out.columns:
        out["value_rank_universe"] = out.groupby(dates)["earnings_yield"].transform(_safe_pct_rank)
    if "piotroski_fscore" in out.columns:
        out["piotroski_rank_universe"] = out.groupby(dates)["piotroski_fscore"].transform(_safe_pct_rank)

    value_universe_parts = []
    for col, invert in [("fcf_yield", False), ("earnings_yield", False), ("ev_to_ebitda", True), ("pe_ratio", True), ("ps_ratio", True)]:
        if col in out.columns:
            r = out.groupby(dates)[col].transform(_safe_pct_rank)
            value_universe_parts.append(1.0 - r if invert else r)
    if value_universe_parts:
        out["valuation_percentile_universe"] = pd.concat(value_universe_parts, axis=1).mean(axis=1).fillna(0.5)


    # GARP mispricing / expectation-gap features. These are point-in-time
    # cross-sectional transforms: no future returns or future fundamentals are
    # used. Higher values mean the current quality/growth profile looks
    # underappreciated by current valuation.
    quality_parts = []
    for col in ["gross_margin", "operating_margin", "fcf_margin", "roic", "piotroski_fscore"]:
        if col in out.columns:
            quality_parts.append(out.groupby(dates)[col].transform(_safe_pct_rank))
    growth_parts = []
    for col in ["revenue_yoy_growth", "fcf_yoy_growth", "eps_growth_trend_3y"]:
        if col in out.columns:
            growth_parts.append(out.groupby(dates)[col].transform(_safe_pct_rank))
    if quality_parts:
        quality_composite = pd.concat(quality_parts, axis=1).mean(axis=1).fillna(0.5)
    else:
        quality_composite = pd.Series(0.5, index=out.index, dtype=float)
    if growth_parts:
        growth_composite = pd.concat(growth_parts, axis=1).mean(axis=1).fillna(0.5)
    else:
        growth_composite = pd.Series(0.5, index=out.index, dtype=float)

    margin_stability = pd.Series(0.5, index=out.index, dtype=float)
    if "gross_margin" in out.columns and isinstance(out.index, pd.MultiIndex) and "ticker" in out.index.names:
        gm = pd.to_numeric(out["gross_margin"], errors="coerce")
        ticker_vals = out.index.get_level_values("ticker")
        rolling_std = gm.groupby(ticker_vals).transform(lambda x: x.rolling(8, min_periods=3).std())
        margin_stability = (1.0 - rolling_std.groupby(dates).transform(_safe_pct_rank)).fillna(0.5)

    out["moat_proxy_score"] = (0.45 * quality_composite + 0.35 * margin_stability + 0.20 * _numeric_col_or_default(out, "roic").groupby(dates).transform(_safe_pct_rank)).fillna(0.5).clip(0.0, 1.0)

    valuation_reasonable = out.get("valuation_percentile_universe", pd.Series(0.5, index=out.index, dtype=float))
    quality_growth = (0.55 * quality_composite + 0.45 * growth_composite).fillna(0.5)
    out["mispricing_quality_growth"] = (quality_growth * valuation_reasonable).fillna(0.5).clip(0.0, 1.0)
    out["expectation_gap_score"] = (0.65 * quality_growth + 0.35 * valuation_reasonable).fillna(0.5).clip(0.0, 1.0)
    out["valuation_to_growth_reasonableness"] = (valuation_reasonable - (growth_composite - 0.5).clip(lower=0.0) * 0.25).fillna(0.5).clip(0.0, 1.0)

    expensive_parts = []
    for col in ["peg_ratio", "ev_to_sales", "pe_vs_5y_median", "ev_ebitda_vs_5y_median", "ps_ratio"]:
        if col in out.columns:
            expensive_parts.append(out.groupby(dates)[col].transform(_safe_pct_rank))
    if expensive_parts:
        expensive = pd.concat(expensive_parts, axis=1).mean(axis=1).fillna(0.5)
    else:
        expensive = 1.0 - valuation_reasonable
    out["overexpectation_penalty"] = (expensive * (0.5 + 0.5 * growth_composite)).fillna(0.5).clip(0.0, 1.0)

    # Analyst earnings revision rank: one of the most consistently predictive
    # forward-looking cross-sectional signals.
    if "eps_revision" in out.columns:
        out["eps_revision_rank_universe"] = out.groupby(dates)["eps_revision"].transform(_safe_pct_rank)

    # Beat rate rank: consistent earnings beaters signal durable competitive
    # advantages.  Cross-sectional rank removes sector-wide easy-beat bias.
    if "beat_rate_4q" in out.columns:
        out["beat_rate_rank_universe"] = out.groupby(dates)["beat_rate_4q"].transform(_safe_pct_rank)

    # Momentum consistency: fraction of short-to-medium momentum windows
    # (1m, 3m, 6m) that are positive.  momentum_12m is deliberately excluded
    # because it requires 252+ prior trading days and is frequently absent in
    # the first analysis year.  When a window value is NaN (insufficient
    # history), fillna(0.0) maps it to a neutral signal that is NOT counted
    # as positive — this is the intended behaviour (missing ≠ positive).
    mom_windows = [c for c in ["momentum_1m", "momentum_3m", "momentum_6m"] if c in out.columns]
    if len(mom_windows) >= 2:
        pos_count = sum(
            (_numeric_col_or_default(out, c).fillna(0.0) > 0).astype(float)
            for c in mom_windows
        )
        out["momentum_consistency"] = pos_count / len(mom_windows)

    value_signal = _numeric_col_or_default(out, "earnings_yield").fillna(0.0)
    mom_signal = _numeric_col_or_default(out, "momentum_6m").fillna(0.0)
    quality_signal = _numeric_col_or_default(out, "roic").fillna(0.0)
    low_vol = 1.0 / (1.0 + _numeric_col_or_default(out, "volatility_60d").abs().fillna(0.0))

    out["value_x_momentum"] = value_signal * mom_signal
    out["quality_x_lowvol"] = quality_signal * low_vol

    # Composite factor: quality × value (high quality + cheap = strong candidate)
    if "quality_rank_universe" in out.columns and "value_rank_universe" in out.columns:
        out["quality_x_value_universe"] = out["quality_rank_universe"] * out["value_rank_universe"]

    return out
