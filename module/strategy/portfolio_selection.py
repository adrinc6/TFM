"""Portfolio selection: rank stocks and enforce 4–8 stock constraints.

Ranking metric
--------------
We use the **expected value (EV)** of the trade:

    EV = confidence × tp_pct  −  (1 − confidence) × sl_pct

A positive EV means the expected gain exceeds the expected loss; only
stocks with EV > 0 are eligible by default (configurable).

Portfolio constraints
---------------------
* Minimum stocks : 4  (if fewer qualify → no investment)
* Maximum stocks : 8
* Optional per-sector cap (max 3 from the same GICS sector)

Design choices
--------------
* Sector-cap is relaxed when it would prevent meeting ``min_stocks``.
* All stocks are ranked and the top-``max_stocks`` that pass the EV
  threshold and sector cap are selected.
* When fewer than ``min_stocks`` valid candidates exist, the selection
  returns an empty DataFrame to signal "no investment this period".
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


_DEFAULT_MIN_STOCKS = 4
_DEFAULT_MAX_STOCKS = 8
_DEFAULT_SECTOR_CAP = 3


def compute_expected_value(signals: pd.DataFrame) -> pd.Series:
    """Compute expected value for each stock.

    EV = confidence × tp_pct  −  (1 − confidence) × sl_pct

    Parameters
    ----------
    signals:
        DataFrame that must contain ``confidence``, ``tp_pct``, ``sl_pct``.

    Returns
    -------
    pd.Series aligned with *signals* index.
    """
    conf = pd.to_numeric(signals["confidence"], errors="coerce").clip(0.0, 1.0)
    tp = pd.to_numeric(signals["tp_pct"], errors="coerce")
    sl = pd.to_numeric(signals["sl_pct"], errors="coerce")
    ev = conf * tp - (1.0 - conf) * sl
    return ev


def select_portfolio(
    signals: pd.DataFrame,
    *,
    min_stocks: int = _DEFAULT_MIN_STOCKS,
    max_stocks: int = _DEFAULT_MAX_STOCKS,
    sector_cap: int = _DEFAULT_SECTOR_CAP,
    ev_threshold: float = 0.0,
    sector_col: Optional[str] = "sector",
) -> pd.DataFrame:
    """Select a portfolio from the ranked signal universe.

    Parameters
    ----------
    signals:
        DataFrame with at least ``ticker``, ``confidence``, ``tp_pct``,
        ``sl_pct`` columns.  Optionally a ``sector`` column for concentration
        constraints.
    min_stocks:
        Minimum number of stocks required to invest (default 4).
        Returns an **empty** DataFrame when fewer candidates qualify.
    max_stocks:
        Maximum portfolio size (default 8).
    sector_cap:
        Maximum stocks from the same sector (default 3).
    ev_threshold:
        Minimum expected value for a stock to be considered (default 0.0).
    sector_col:
        Column name for sector information.  Set to ``None`` to disable the
        sector cap.

    Returns
    -------
    Sub-DataFrame of *signals* for the selected stocks, sorted by EV
    descending.  Includes the ``ev`` and ``selected`` columns.
    Empty DataFrame if fewer than ``min_stocks`` candidates qualify.
    """
    df = signals.copy()
    df["ev"] = compute_expected_value(df)

    # Filter by EV threshold
    eligible = df[df["ev"] >= ev_threshold].sort_values("ev", ascending=False)

    if eligible.empty or len(eligible) < min_stocks:
        # Not enough candidates — no investment this period
        df["selected"] = False
        df["ev"] = compute_expected_value(df)
        return df.assign(selected=False)

    # Apply sector cap
    selected_rows = []
    sector_counts: dict[str, int] = {}

    for _, row in eligible.iterrows():
        if len(selected_rows) >= max_stocks:
            break
        sector = str(row.get(sector_col, "Unknown")) if sector_col else "Unknown"
        if sector_cap > 0 and sector_counts.get(sector, 0) >= sector_cap:
            continue
        selected_rows.append(row)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    # Relax sector cap if we still don't have enough
    if len(selected_rows) < min_stocks:
        existing_tickers = {str(r["ticker"]) for r in selected_rows}
        for _, row in eligible.iterrows():
            if len(selected_rows) >= min_stocks:
                break
            if str(row["ticker"]) not in existing_tickers:
                selected_rows.append(row)
                existing_tickers.add(str(row["ticker"]))

    if len(selected_rows) < min_stocks:
        # Still not enough after relaxation → no investment
        df["selected"] = False
        return df

    selected_df = pd.DataFrame(selected_rows)
    selected_tickers = set(selected_df["ticker"].astype(str))

    # Mark selected in the full universe
    df["selected"] = df["ticker"].astype(str).isin(selected_tickers)
    return df


def get_portfolio_weights(
    signals: pd.DataFrame,
    selected_only: bool = True,
    weight_by: str = "ev",
) -> pd.Series:
    """Compute normalised portfolio weights.

    Parameters
    ----------
    signals:
        Output of :func:`select_portfolio` (must have ``selected``, ``ev``,
        ``confidence`` columns).
    selected_only:
        When ``True`` (default), only selected stocks receive a weight.
    weight_by:
        Column to use for proportional weighting: ``"ev"`` (default) or
        ``"confidence"``.  Falls back to equal weights on failure.

    Returns
    -------
    pd.Series indexed by ticker.
    """
    df = signals.copy()
    if selected_only and "selected" in df.columns:
        df = df[df["selected"].astype(bool)]

    if df.empty:
        return pd.Series(dtype=float)

    if weight_by in df.columns:
        raw = pd.to_numeric(df[weight_by], errors="coerce").clip(0.0, None)
        total = float(raw.sum())
        if total > 0:
            weights = raw / total
        else:
            weights = pd.Series(1.0 / len(df), index=df.index)
    else:
        weights = pd.Series(1.0 / len(df), index=df.index)

    weights.index = df["ticker"].values
    return weights
