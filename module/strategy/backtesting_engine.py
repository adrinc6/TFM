"""TP/SL backtesting engine.

For each stock, determines whether the take-profit (TP) or stop-loss (SL)
level is hit first within a configurable holding-period window.

Outcome labels
--------------
``"TP"``   – TP price was reached before SL or expiry.
``"SL"``   – SL price was reached before TP or expiry.
``"NONE"`` – Neither level was hit within ``max_holding_days``.

The engine is intentionally look-ahead-free: it only reads prices from
``entry_date`` onward and never peeks at future training data.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


_DEFAULT_MAX_HOLDING_DAYS = 90


def simulate_tp_sl(
    ticker: str,
    prices: pd.Series,
    entry_date: pd.Timestamp,
    tp_pct: float,
    sl_pct: float,
    *,
    max_holding_days: int = _DEFAULT_MAX_HOLDING_DAYS,
) -> Dict[str, object]:
    """Simulate a single TP/SL trade for one stock.

    Parameters
    ----------
    ticker:
        Stock identifier (used only for labelling the result).
    prices:
        Daily close price series for the stock.  Index must be DatetimeIndex.
    entry_date:
        Trade entry date (first bar is the entry price bar).
    tp_pct:
        Take-profit level as a fraction (e.g. 0.08 = 8 %).
    sl_pct:
        Stop-loss level as a fraction (e.g. 0.05 = 5 %).
    max_holding_days:
        Maximum calendar days the position is held (default 90).

    Returns
    -------
    dict with keys:
        ticker, entry_date, entry_price, tp_pct, sl_pct,
        tp_price, sl_price, outcome (TP/SL/NONE), days_to_outcome.
    """
    result: Dict[str, object] = {
        "ticker": ticker,
        "entry_date": entry_date,
        "entry_price": np.nan,
        "tp_pct": float(tp_pct),
        "sl_pct": float(sl_pct),
        "tp_price": np.nan,
        "sl_price": np.nan,
        "outcome": "NONE",
        "days_to_outcome": int(max_holding_days),
    }

    if prices is None or prices.empty:
        return result

    prices = pd.to_numeric(prices, errors="coerce").dropna().sort_index()
    prices.index = pd.to_datetime(prices.index)

    entry_ts = pd.Timestamp(entry_date)
    expiry_ts = entry_ts + pd.Timedelta(days=int(max_holding_days))

    # First bar on or after entry date
    entry_candidates = prices.index[prices.index >= entry_ts]
    if len(entry_candidates) == 0:
        return result
    actual_entry = entry_candidates[0]
    entry_price = float(prices.loc[actual_entry])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return result

    tp_price = entry_price * (1.0 + float(tp_pct))
    sl_price = entry_price * (1.0 - float(sl_pct))

    result["entry_price"] = entry_price
    result["tp_price"] = tp_price
    result["sl_price"] = sl_price

    # Scan forward-only bars within the holding window
    window = prices.loc[
        (prices.index > actual_entry) & (prices.index <= expiry_ts)
    ]

    for dt, px in window.items():
        p = float(px)
        days_elapsed = int((pd.Timestamp(dt) - actual_entry).days)
        if p >= tp_price:
            result["outcome"] = "TP"
            result["days_to_outcome"] = days_elapsed
            return result
        if p <= sl_price:
            result["outcome"] = "SL"
            result["days_to_outcome"] = days_elapsed
            return result

    # No barrier hit → NONE; days_to_outcome = actual days in window
    if not window.empty:
        result["days_to_outcome"] = int(
            (pd.Timestamp(window.index[-1]) - actual_entry).days
        )
    return result


def run_backtest(
    signals: pd.DataFrame,
    prices_dict: Dict[str, object],
    entry_date: pd.Timestamp,
    *,
    max_holding_days: int = _DEFAULT_MAX_HOLDING_DAYS,
) -> pd.DataFrame:
    """Run TP/SL backtest for all tickers in *signals*.

    Parameters
    ----------
    signals:
        DataFrame with columns ``ticker``, ``tp_pct``, ``sl_pct``.
        Typically the output of :func:`~module.strategy.portfolio_selection.select_portfolio`.
    prices_dict:
        Mapping of ticker → price DataFrame or Series.
    entry_date:
        Common trade entry date for all tickers.
    max_holding_days:
        Forwarded to :func:`simulate_tp_sl`.

    Returns
    -------
    *signals* augmented with columns:
    ``entry_price``, ``tp_price``, ``sl_price``, ``outcome``, ``days_to_outcome``.
    """
    rows = []
    for _, row in signals.iterrows():
        ticker = str(row["ticker"])
        tp_pct = float(row.get("tp_pct", 0.08))
        sl_pct = float(row.get("sl_pct", 0.05))

        price_obj = prices_dict.get(ticker)
        prices = _extract_close(price_obj)

        sim = simulate_tp_sl(
            ticker=ticker,
            prices=prices,
            entry_date=entry_date,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            max_holding_days=max_holding_days,
        )
        rows.append(sim)

    backtest_df = pd.DataFrame(rows)
    if backtest_df.empty:
        return signals.copy()

    merge_cols = [c for c in backtest_df.columns if c != "ticker"]
    # Drop duplicated columns that already exist in signals
    merge_cols = [c for c in merge_cols if c not in signals.columns or c in ("entry_date",)]
    out = signals.copy()
    for col in merge_cols:
        out[col] = out["ticker"].map(
            backtest_df.set_index("ticker")[col]
        )
    # Always overwrite outcome and days_to_outcome
    for col in ("outcome", "days_to_outcome", "entry_price", "tp_price", "sl_price", "entry_date"):
        if col in backtest_df.columns:
            out[col] = out["ticker"].map(backtest_df.set_index("ticker")[col])

    return out


def _extract_close(price_obj) -> pd.Series:
    """Extract a close-price Series from various input formats."""
    if price_obj is None:
        return pd.Series(dtype=float)
    if isinstance(price_obj, pd.Series):
        return pd.to_numeric(price_obj, errors="coerce").dropna().sort_index()
    if isinstance(price_obj, pd.DataFrame) and not price_obj.empty:
        col = "Close" if "Close" in price_obj.columns else price_obj.columns[-1]
        return pd.to_numeric(price_obj[col], errors="coerce").dropna().sort_index()
    return pd.Series(dtype=float)
