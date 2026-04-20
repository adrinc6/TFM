"""Cross-sectional normalized feature engineering for ranking models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_pct_rank(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    if vals.notna().sum() <= 1:
        return pd.Series(0.5, index=s.index, dtype=float)
    return vals.rank(pct=True, method="average").fillna(0.5)


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

    vol = pd.to_numeric(out.get("volatility_60d", 0.0), errors="coerce").replace(0, np.nan)
    for src, dst in [
        ("momentum_3m", "momentum_vol_adj"),
        ("earnings_yield", "value_vol_adj"),
        ("roic", "quality_vol_adj"),
        ("finbert_sentiment_polarity", "sentiment_vol_adj"),
    ]:
        if src in out.columns:
            out[dst] = (pd.to_numeric(out[src], errors="coerce") / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    value_signal = pd.to_numeric(out.get("earnings_yield", 0.0), errors="coerce").fillna(0.0)
    mom_signal = pd.to_numeric(out.get("momentum_6m", 0.0), errors="coerce").fillna(0.0)
    quality_signal = pd.to_numeric(out.get("roic", 0.0), errors="coerce").fillna(0.0)
    low_vol = 1.0 / (1.0 + pd.to_numeric(out.get("volatility_60d", 0.0), errors="coerce").abs().fillna(0.0))
    sentiment = pd.to_numeric(out.get("finbert_sentiment_polarity", 0.0), errors="coerce").fillna(0.0)
    eps_surprise = pd.to_numeric(out.get("eps_surprise_pct", 0.0), errors="coerce").fillna(0.0)

    out["value_x_momentum"] = value_signal * mom_signal
    out["quality_x_lowvol"] = quality_signal * low_vol
    out["sentiment_x_earnings_surprise"] = sentiment * eps_surprise

    return out
