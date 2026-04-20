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

    return out
