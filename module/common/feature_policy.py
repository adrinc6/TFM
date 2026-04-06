"""Shared feature policy helpers (ratio/normalized-only)."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def is_ratio_or_normalized_feature(col_name: str, series: Optional[pd.Series] = None) -> bool:
    """Strict policy: allow only ratio/normalized features or binary flags."""
    c = str(col_name).lower().strip()
    if not c:
        return False

    # Binary engineered flags are considered normalized features.
    if series is not None:
        vals = pd.Series(series).dropna().unique()
        if len(vals) > 0:
            try:
                as_set = set(pd.Series(vals).astype(float).tolist())
                if as_set.issubset({0.0, 1.0}):
                    return True
            except Exception:
                pass

    allowed_tokens = [
        "ratio", "margin", "yield", "growth", "trend", "momentum", "volatility",
        "rsi", "macd", "beta", "zscore", "zsector", "pct", "coverage",
        "score", "prior", "dispersion", "consensus", "confidence", "quality",
        "fscore", "accrual", "atr", "bb_", "vs_5y", "vs_52w", "debt_to_", "_to_",
        "revision", "surprise", "beater", "overbought", "oversold", "bullish",
        "above_sma", "cross_sma", "expansion", "decline", "losses", "risk",
    ]
    if any(tok in c for tok in allowed_tokens):
        return True

    blocked_prefixes = [
        "revenue", "net_income", "operating_income", "gross_profit", "fcf", "ebitda",
        "total_assets", "total_liabilities", "total_equity", "total_debt", "cash",
        "shares", "eps_est", "eps_reported", "market_cap", "capex", "income_tax",
        "depreciation", "operating_cash_flow",
    ]
    if any(c.startswith(p) for p in blocked_prefixes):
        return False

    return False


def filter_ratio_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only ratio/normalized columns from a DataFrame."""
    if df is None or df.empty:
        return df
    keep_cols = [
        col for col in df.columns
        if is_ratio_or_normalized_feature(col, df[col])
    ]
    return df[keep_cols].copy()
