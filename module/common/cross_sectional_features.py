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

        if "pe_ratio" in out.columns:
            out["pe_rank_sector"] = out.groupby(grp_sector)["pe_ratio"].transform(_safe_pct_rank)
        if "momentum_12m" in out.columns:
            out["momentum_pct_sector"] = out.groupby(grp_sector)["momentum_12m"].transform(_safe_pct_rank)
        if "roe" in out.columns:
            out["roe_pct_sector"] = out.groupby(grp_sector)["roe"].transform(_safe_pct_rank)

        # Additional sector-relative value/quality ranks
        if "pb_ratio" in out.columns:
            # Lower P/B is cheaper → invert so higher rank = better value
            out["pb_rank_sector"] = 1.0 - out.groupby(grp_sector)["pb_ratio"].transform(_safe_pct_rank)
        if "fcf_yield" in out.columns:
            out["fcf_yield_rank_sector"] = out.groupby(grp_sector)["fcf_yield"].transform(_safe_pct_rank)
        if "roic" in out.columns:
            out["roic_rank_sector"] = out.groupby(grp_sector)["roic"].transform(_safe_pct_rank)
        if "ev_to_ebitda" in out.columns:
            # Lower EV/EBITDA = cheaper → invert
            out["ev_ebitda_rank_sector"] = 1.0 - out.groupby(grp_sector)["ev_to_ebitda"].transform(_safe_pct_rank)
        if "debt_to_ebitda" in out.columns:
            # Lower debt = better → invert
            out["debt_rank_sector"] = 1.0 - out.groupby(grp_sector)["debt_to_ebitda"].transform(_safe_pct_rank)

    # Universe-wide percentile ranks (cross-sectional factor exposures)
    if "momentum_12m" in out.columns:
        out["momentum_12m_rank_universe"] = out.groupby(dates)["momentum_12m"].transform(_safe_pct_rank)
    if "roic" in out.columns:
        out["quality_rank_universe"] = out.groupby(dates)["roic"].transform(_safe_pct_rank)
    if "earnings_yield" in out.columns:
        out["value_rank_universe"] = out.groupby(dates)["earnings_yield"].transform(_safe_pct_rank)
    if "piotroski_fscore" in out.columns:
        out["piotroski_rank_universe"] = out.groupby(dates)["piotroski_fscore"].transform(_safe_pct_rank)

    # Analyst earnings revision rank: one of the most consistently predictive
    # forward-looking cross-sectional signals.  Stocks with top-decile upward
    # revisions exhibit persistent positive price momentum.
    if "eps_revision" in out.columns:
        out["eps_revision_rank_universe"] = out.groupby(dates)["eps_revision"].transform(_safe_pct_rank)

    # Revenue growth acceleration: difference between recent and older growth
    # rate.  Positive = growth is speeding up; negative = decelerating.
    # A company re-accelerating revenue beats one with stable but flat growth.
    if "revenue_yoy_growth" in out.columns:
        rev_growth = _numeric_col_or_default(out, "revenue_yoy_growth").fillna(0.0)
        # Lag one cross-section (one quarter back within each ticker)
        if isinstance(out.index, pd.MultiIndex) and "ticker" in out.index.names:
            rev_lag = (
                rev_growth
                .groupby(out.index.get_level_values("ticker"))
                .shift(1)
                .fillna(0.0)
            )
        else:
            rev_lag = rev_growth.shift(1).fillna(0.0)
        raw_accel = rev_growth - rev_lag
        # Cross-sectionally rank so the meta-learner gets a relative signal
        out["revenue_growth_acceleration"] = raw_accel.groupby(dates).transform(_safe_pct_rank)

    # EPS surprise acceleration: is the magnitude of earnings beats growing?
    # Stocks with improving EPS surprise trend signal that analysts are
    # systematically underestimating earnings power — a leading indicator of
    # persistent positive earnings momentum.
    if "eps_surprise_pct" in out.columns:
        eps_surp = _numeric_col_or_default(out, "eps_surprise_pct").fillna(0.0)
        if isinstance(out.index, pd.MultiIndex) and "ticker" in out.index.names:
            eps_surp_lag = (
                eps_surp
                .groupby(out.index.get_level_values("ticker"))
                .shift(1)
                .fillna(0.0)
            )
        else:
            eps_surp_lag = eps_surp.shift(1).fillna(0.0)
        raw_eps_accel = eps_surp - eps_surp_lag
        out["eps_surprise_acceleration"] = raw_eps_accel.groupby(dates).transform(_safe_pct_rank)

    # Beat rate rank: consistent earnings beaters signal durable competitive
    # advantages and management credibility.  Cross-sectional rank filters out
    # sector-wide easy-beat environments.
    if "beat_rate_4q" in out.columns:
        out["beat_rate_rank_universe"] = out.groupby(dates)["beat_rate_4q"].transform(_safe_pct_rank)

    # Volatility-adjusted 12-month momentum (Sharpe-momentum): dividing realized
    # return by realized risk rewards smooth uptrends and penalises noisy spikes.
    # Cross-sectional rank makes the signal comparable across sectors.
    if "momentum_12m" in out.columns and "volatility_60d" in out.columns:
        vol_safe = _numeric_col_or_default(out, "volatility_60d").replace(0, np.nan)
        vol_median = float(vol_safe.median()) if vol_safe.notna().any() else 1.0
        vol_safe = vol_safe.fillna(vol_median)
        raw_vol_adj = _numeric_col_or_default(out, "momentum_12m").fillna(0.0) / vol_safe
        raw_vol_adj = raw_vol_adj.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["vol_adj_momentum_12m_rank"] = raw_vol_adj.groupby(dates).transform(_safe_pct_rank)

    # Momentum consistency: fraction of momentum windows (1m, 3m, 6m, 12m) that
    # are positive.  A stock with all horizons pointing up has a more reliable
    # underlying trend than one with mixed multi-period signals.
    mom_windows = [c for c in ["momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m"] if c in out.columns]
    if len(mom_windows) >= 2:
        pos_count = sum(
            (_numeric_col_or_default(out, c).fillna(0.0) > 0).astype(float)
            for c in mom_windows
        )
        out["momentum_consistency"] = pos_count / len(mom_windows)

    # ROIC acceleration: cross-sectional rank of quarterly capital-efficiency
    # improvement.  Companies steadily increasing ROIC are compounding their
    # competitive moat and typically command expanding valuation multiples.
    if "roic" in out.columns:
        roic_vals = _numeric_col_or_default(out, "roic").fillna(0.0)
        if isinstance(out.index, pd.MultiIndex) and "ticker" in out.index.names:
            roic_lag = (
                roic_vals
                .groupby(out.index.get_level_values("ticker"))
                .shift(1)
                .fillna(0.0)
            )
        else:
            roic_lag = roic_vals.shift(1).fillna(0.0)
        roic_accel = roic_vals - roic_lag
        out["quality_acceleration_rank"] = roic_accel.groupby(dates).transform(_safe_pct_rank)

    vol = _numeric_col_or_default(out, "volatility_60d").replace(0, np.nan)
    for src, dst in [
        ("momentum_3m", "momentum_vol_adj"),
        ("earnings_yield", "value_vol_adj"),
        ("roic", "quality_vol_adj"),
        ("finbert_sentiment_polarity", "sentiment_vol_adj"),
    ]:
        if src in out.columns:
            out[dst] = (pd.to_numeric(out[src], errors="coerce") / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    value_signal = _numeric_col_or_default(out, "earnings_yield").fillna(0.0)
    mom_signal = _numeric_col_or_default(out, "momentum_6m").fillna(0.0)
    quality_signal = _numeric_col_or_default(out, "roic").fillna(0.0)
    low_vol = 1.0 / (1.0 + _numeric_col_or_default(out, "volatility_60d").abs().fillna(0.0))
    sentiment = _numeric_col_or_default(out, "finbert_sentiment_polarity").fillna(0.0)
    eps_surprise = _numeric_col_or_default(out, "eps_surprise_pct").fillna(0.0)

    out["value_x_momentum"] = value_signal * mom_signal
    out["quality_x_lowvol"] = quality_signal * low_vol
    out["sentiment_x_earnings_surprise"] = sentiment * eps_surprise

    # Composite factor: quality × value (high quality + cheap = strong candidate)
    if "quality_rank_universe" in out.columns and "value_rank_universe" in out.columns:
        out["quality_x_value_universe"] = out["quality_rank_universe"] * out["value_rank_universe"]

    # Momentum divergence: long-run momentum minus short-term reversal.
    # Positive when a stock has strong 12-month trend but a mild 1-month move
    # (trend is intact without recent excessive run-up); negative when the
    # recent 1-month surge has overshot the longer-term trend.
    if "momentum_12m" in out.columns and "momentum_1m" in out.columns:
        mom12 = _numeric_col_or_default(out, "momentum_12m").fillna(0.0)
        mom1 = _numeric_col_or_default(out, "momentum_1m").fillna(0.0)
        out["momentum_quality_signal"] = mom12 - mom1

    return out
