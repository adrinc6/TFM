"""Point-in-time (as-of) helpers for leakage control and audit."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd


def _resolve_dates(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.Series:
    """Return a normalized datetime series from index or date_col."""
    if df is None or df.empty:
        return pd.Series(dtype="datetime64[ns]")

    if date_col is not None:
        if date_col not in df.columns:
            return pd.Series(dtype="datetime64[ns]")
        return pd.to_datetime(df[date_col], errors="coerce")

    if isinstance(df.index, pd.MultiIndex):
        date_idx = None
        for name in ("date", "datetime", "timestamp"):
            if name in df.index.names:
                date_idx = df.index.get_level_values(name)
                break
        if date_idx is None:
            date_idx = df.index.get_level_values(-1)
        return pd.to_datetime(date_idx, errors="coerce")

    return pd.to_datetime(df.index, errors="coerce")


def filter_asof(df: pd.DataFrame, as_of: pd.Timestamp, date_col: Optional[str] = None) -> pd.DataFrame:
    """Filter rows where date <= as_of. Returns empty df if date cannot be resolved."""
    if df is None or df.empty:
        return df
    dates = _resolve_dates(df, date_col=date_col)
    if dates.empty:
        return df.iloc[0:0].copy()
    as_of_ts = pd.Timestamp(as_of)
    mask = dates <= as_of_ts
    return df.loc[mask].copy()


def detect_future_rows(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    date_col: Optional[str] = None,
) -> Tuple[int, Optional[pd.Timestamp]]:
    """Return number of rows with date > as_of and max future date detected."""
    if df is None or df.empty:
        return 0, None
    dates = _resolve_dates(df, date_col=date_col)
    if dates.empty:
        return 0, None
    as_of_ts = pd.Timestamp(as_of)
    future_mask = dates > as_of_ts
    n_future = int(future_mask.sum())
    if n_future <= 0:
        return 0, None
    max_future = pd.to_datetime(dates[future_mask], errors="coerce").max()
    return n_future, pd.Timestamp(max_future) if pd.notna(max_future) else None


def assert_no_future_data(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    context: str,
    date_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Structured leakage check result for audit exports/logging."""
    n_future, max_future_date = detect_future_rows(df=df, as_of=as_of, date_col=date_col)
    return {
        "context": str(context),
        "as_of": str(pd.Timestamp(as_of).date()),
        "n_rows_future_detected": int(n_future),
        "max_future_date_detected": None if max_future_date is None else str(pd.Timestamp(max_future_date).date()),
        "ok": bool(n_future == 0),
    }
