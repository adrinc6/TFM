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

    # Universe-wide percentile ranks (cross-sectional factor exposures)
    if "roic" in out.columns:
        out["quality_rank_universe"] = out.groupby(dates)["roic"].transform(_safe_pct_rank)
    if "earnings_yield" in out.columns:
        out["value_rank_universe"] = out.groupby(dates)["earnings_yield"].transform(_safe_pct_rank)
    if "piotroski_fscore" in out.columns:
        out["piotroski_rank_universe"] = out.groupby(dates)["piotroski_fscore"].transform(_safe_pct_rank)

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
    # because it is systematically unavailable in the current analysis window
    # (requires 252+ prior trading days) and its NaN values would bias the
    # ratio downward via the fillna(0.0) fallback.
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
