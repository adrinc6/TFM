"""Point-in-time (as-of) helpers for leakage control and audit."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd


def _resolve_dates(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.Series:
    """Extracts a normalised datetime series from a DataFrame index or column.

    Args:
        df (pd.DataFrame): Source DataFrame.
        date_col (Optional[str]): Column name to use as the date source. If
            None, the DataFrame index is used.

    Returns:
        pd.Series: A datetime64[ns] Series. Empty if df is None or empty.
    """
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
    """Filters rows where date <= as_of to prevent look-ahead bias.

    Args:
        df (pd.DataFrame): DataFrame to filter.
        as_of (pd.Timestamp): The point-in-time cut-off. Rows with dates
            strictly after this value are removed.
        date_col (Optional[str]): Column name holding the date. If None, the
            DataFrame index is used.

    Returns:
        pd.DataFrame: Filtered copy of df. Returns df unchanged if None is
            passed; returns an empty frame if the date cannot be resolved.
    """
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
    """Returns the count of future rows and the latest future date detected.

    Args:
        df (pd.DataFrame): DataFrame to inspect.
        as_of (pd.Timestamp): The point-in-time cut-off.
        date_col (Optional[str]): Column name holding the date. If None, the
            DataFrame index is used.

    Returns:
        Tuple[int, Optional[pd.Timestamp]]: A tuple (n_future, max_future_date)
            where n_future is the count of rows with date > as_of and
            max_future_date is the latest such date (None if n_future == 0).
    """
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
    """Produces a structured leakage-check result suitable for audit exports.

    Args:
        df (pd.DataFrame): DataFrame to audit.
        as_of (pd.Timestamp): The point-in-time cut-off.
        context (str): A label identifying the data source for audit purposes.
        date_col (Optional[str]): Column name holding the date. If None, the
            DataFrame index is used.

    Returns:
        Dict[str, Any]: Audit record with keys:
            - context: the supplied context label
            - as_of: the cut-off date as a string
            - n_rows_future_detected: count of rows with date > as_of
            - max_future_date_detected: the latest future date found (or None)
            - ok: True if no future rows were detected
    """
    n_future, max_future_date = detect_future_rows(df=df, as_of=as_of, date_col=date_col)
    return {
        "context": str(context),
        "as_of": str(pd.Timestamp(as_of).date()),
        "n_rows_future_detected": int(n_future),
        "max_future_date_detected": None if max_future_date is None else str(pd.Timestamp(max_future_date).date()),
        "ok": bool(n_future == 0),
    }
