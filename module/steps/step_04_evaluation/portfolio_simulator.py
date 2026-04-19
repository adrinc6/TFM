"""USD portfolio simulator for fold-level and chained walk-forward backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Position:
    ticker: str
    shares: float


def _get_close_column(prices: pd.DataFrame, fallback_col_idx: int = 3) -> str:
    """Return the close price column name for a price DataFrame.

    Uses 'Close' if present, otherwise falls back to the column at *fallback_col_idx*
    (default 3, corresponding to the C in standard OHLCV layout).
    """
    if "Close" in prices.columns:
        # Canonical path: explicit close column.
        return "Close"
    if len(prices.columns) > fallback_col_idx:
        # OHLCV fallback: index 3 usually corresponds to "Close".
        return str(prices.columns[fallback_col_idx])
    if len(prices.columns) == 1:
        # Single-column inputs are treated as pre-extracted close series.
        return str(prices.columns[0])
    # Defensive fallback for non-standard multi-column inputs.
    return str(prices.columns[-1])


def _extract_close_series(price_obj) -> pd.Series:
    if price_obj is None:
        return pd.Series(dtype=float)
    if isinstance(price_obj, pd.Series):
        s = pd.to_numeric(price_obj, errors="coerce").dropna()
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    if isinstance(price_obj, pd.DataFrame) and not price_obj.empty:
        close_col = _get_close_column(price_obj)
        s = price_obj[close_col]
        s = pd.to_numeric(s, errors="coerce").dropna()
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    return pd.Series(dtype=float)


def _resolve_exec_date(price_series: pd.Series, requested: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Return the first available trading date on or after *requested*."""
    if price_series is None or price_series.empty:
        return None
    ts = pd.Timestamp(requested)
    candidates = price_series.index[price_series.index >= ts]
    if len(candidates) == 0:
        return None
    return pd.Timestamp(candidates[0])


def _resolve_exec_date_on_or_before(price_series: pd.Series, requested: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Return the last available trading date on or before *requested*.

    Used for exit resolution so that a position can be closed at the last
    available price even when the data does not extend all the way to the
    requested exit date (e.g. recent folds whose price download is not yet
    complete, or tickers that were delisted shortly before exit_req).
    """
    if price_series is None or price_series.empty:
        return None
    ts = pd.Timestamp(requested)
    candidates = price_series.index[price_series.index <= ts]
    if len(candidates) == 0:
        return None
    return pd.Timestamp(candidates[-1])


def _price_on_or_before(price_series: pd.Series, date: pd.Timestamp) -> Optional[float]:
    if price_series is None or price_series.empty:
        return None
    subset = price_series.loc[price_series.index <= pd.Timestamp(date)]
    if subset.empty:
        return None
    px = float(subset.iloc[-1])
    if not np.isfinite(px) or px <= 0:
        return None
    return px


def _build_weights(selected_tickers: List[str], weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not selected_tickers:
        return {}
    if not weights:
        eq = 1.0 / len(selected_tickers)
        return {t: eq for t in selected_tickers}

    raw = {t: float(weights.get(t, 0.0)) for t in selected_tickers}
    total = float(sum(max(v, 0.0) for v in raw.values()))
    if total <= 0:
        eq = 1.0 / len(selected_tickers)
        return {t: eq for t in selected_tickers}
    return {t: max(v, 0.0) / total for t, v in raw.items()}


def simulate_fold_usd(
    *,
    fold_id: str,
    prices_dict: Dict[str, object],
    selected_tickers: List[str],
    weights: Optional[Dict[str, float]],
    entry_date_requested,
    exit_date_requested,
    starting_cash_usd: float,
    transaction_fee_usd: float,
    slippage_pct: float,
    allow_fractional_shares: bool = True,
) -> Dict[str, object]:
    """Simulate one long-only fold in USD with deterministic trade rules."""
    entry_req = pd.Timestamp(entry_date_requested)
    exit_req = pd.Timestamp(exit_date_requested)

    tickers = [str(t) for t in selected_tickers]
    tickers = [t for t in tickers if t in prices_dict]

    per_ticker_close: Dict[str, pd.Series] = {t: _extract_close_series(prices_dict.get(t)) for t in tickers}
    # Entry: first trading day ON or AFTER entry_req (do not buy before the entry date).
    entry_candidates = {t: _resolve_exec_date(s, entry_req) for t, s in per_ticker_close.items() if not s.empty}
    # Exit: last trading day ON or BEFORE exit_req.  Using "on or before" prevents entire
    # folds from being skipped when a ticker's price feed does not yet reach exit_req
    # (e.g. recent quarters whose download is incomplete, or recently delisted stocks).
    exit_candidates = {t: _resolve_exec_date_on_or_before(s, exit_req) for t, s in per_ticker_close.items() if not s.empty}

    valid_tickers = [
        t for t in tickers
        if (
            entry_candidates.get(t) is not None
            and exit_candidates.get(t) is not None
            # Exit must be strictly after entry so the holding period is positive.
            and exit_candidates[t] > entry_candidates[t]
        )
    ]

    if not valid_tickers:
        empty_trades = pd.DataFrame(columns=[
            "fold_id", "datetime", "action", "ticker", "raw_price", "exec_price", "shares",
            "notional_usd", "fee_usd", "slippage_pct", "entry_date_requested", "entry_date_used",
            "exit_date_requested", "exit_date_used", "reason",
        ])
        empty_equity = pd.DataFrame(columns=["date", "equity_usd", "cash_usd", "positions_value_usd"])
        return {
            "trades_df": empty_trades,
            "equity_curve_df": empty_equity,
            "fold_summary": {
                "fold_id": fold_id,
                "starting_capital_usd": float(starting_cash_usd),
                "ending_capital_usd": float(starting_cash_usd),
                "pnl_usd": 0.0,
                "pnl_pct": 0.0,
                "total_fees_usd": 0.0,
                "n_buys": 0,
                "n_sells": 0,
                "n_ffill_days": 0,
                "entry_date_used": None,
                "exit_date_used": None,
                "entry_gap_days": None,
                "exit_gap_days": None,
                "n_selected_tickers": 0,
                "leakage_tainted": False,
            },
            "selected_tickers_used": [],
            "weights_used": {},
            "missing_tickers": tickers,
            "missing_reasons": {t: "missing_entry_or_exit_price" for t in tickers},
        }

    # Use the LATEST first-available entry date so that every valid ticker
    # actually has a price on (or before) entry_used.  With min() a ticker whose
    # first available date is later than entry_used would be bought at a stale
    # pre-entry_req price returned by _price_on_or_before.
    entry_used = max(entry_candidates[t] for t in valid_tickers)
    # Use the EARLIEST last-available exit date so that every valid ticker can
    # be sold on the common exit day (each exit_candidate is already <= exit_req).
    exit_used = min(exit_candidates[t] for t in valid_tickers)
    if exit_used < entry_used:
        # Safety fallback: use the latest exit date available across all tickers.
        exit_used = max(exit_candidates[t] for t in valid_tickers)

    entry_gap_days = int((entry_used - entry_req).days)
    exit_gap_days = int((exit_used - exit_req).days)

    weights_used = _build_weights(valid_tickers, weights)
    cash = float(starting_cash_usd)
    # Snapshot initial capital so every ticker's allocation is based on the
    # same starting amount, not on the cash remaining after prior purchases.
    initial_capital = float(starting_cash_usd)
    positions: Dict[str, Position] = {}
    trades: List[Dict[str, object]] = []
    total_fees = 0.0
    n_ffill_days = 0

    for ticker in valid_tickers:
        s = per_ticker_close[ticker]
        raw_price = _price_on_or_before(s, entry_used)
        if raw_price is None:
            continue
        exec_price = raw_price * (1.0 + float(slippage_pct))
        allocated_cash = float(initial_capital * weights_used.get(ticker, 0.0))

        if allocated_cash < float(transaction_fee_usd):
            trades.append({
                "fold_id": fold_id,
                "datetime": entry_used,
                "action": "BUY",
                "ticker": ticker,
                "raw_price": raw_price,
                "exec_price": exec_price,
                "shares": 0.0,
                "notional_usd": 0.0,
                "fee_usd": float(transaction_fee_usd),
                "slippage_pct": float(slippage_pct),
                "entry_date_requested": entry_req,
                "entry_date_used": entry_used,
                "exit_date_requested": exit_req,
                "exit_date_used": exit_used,
                "reason": "insufficient_cash_for_fee",
            })
            continue

        if not allow_fractional_shares:
            shares = max(0.0, float(np.floor((allocated_cash - float(transaction_fee_usd)) / exec_price)))
        else:
            shares = max(0.0, float((allocated_cash - float(transaction_fee_usd)) / exec_price))

        notional = float(shares * exec_price)
        if shares > 0:
            cash -= (notional + float(transaction_fee_usd))
            total_fees += float(transaction_fee_usd)
            positions[ticker] = Position(ticker=ticker, shares=shares)

        trades.append({
            "fold_id": fold_id,
            "datetime": entry_used,
            "action": "BUY",
            "ticker": ticker,
            "raw_price": raw_price,
            "exec_price": exec_price,
            "shares": shares,
            "notional_usd": notional,
            "fee_usd": float(transaction_fee_usd),
            "slippage_pct": float(slippage_pct),
            "entry_date_requested": entry_req,
            "entry_date_used": entry_used,
            "exit_date_requested": exit_req,
            "exit_date_used": exit_used,
            "reason": "rebalance_entry",
        })

    all_dates = sorted(set().union(*[
        [d for d in s.index if (d >= entry_used and d <= exit_used)] for s in per_ticker_close.values() if not s.empty
    ]))
    if not all_dates:
        all_dates = [entry_used, exit_used]

    last_prices: Dict[str, float] = {}
    equity_rows: List[Dict[str, object]] = []

    for dt in all_dates:
        pos_value = 0.0
        for ticker, pos in positions.items():
            s = per_ticker_close.get(ticker, pd.Series(dtype=float))
            px = _price_on_or_before(s, dt)
            if px is None:
                px = last_prices.get(ticker)
                if px is not None:
                    n_ffill_days += 1
            else:
                last_prices[ticker] = px
            if px is not None:
                pos_value += float(pos.shares * px)

        equity_rows.append({
            "date": pd.Timestamp(dt),
            "equity_usd": float(cash + pos_value),
            "cash_usd": float(cash),
            "positions_value_usd": float(pos_value),
        })

    for ticker, pos in list(positions.items()):
        s = per_ticker_close.get(ticker, pd.Series(dtype=float))
        raw_price = _price_on_or_before(s, exit_used)
        if raw_price is None:
            continue
        exec_price = raw_price * (1.0 - float(slippage_pct))
        notional = float(pos.shares * exec_price)
        cash += (notional - float(transaction_fee_usd))
        total_fees += float(transaction_fee_usd)
        trades.append({
            "fold_id": fold_id,
            "datetime": exit_used,
            "action": "SELL",
            "ticker": ticker,
            "raw_price": raw_price,
            "exec_price": exec_price,
            "shares": float(pos.shares),
            "notional_usd": notional,
            "fee_usd": float(transaction_fee_usd),
            "slippage_pct": float(slippage_pct),
            "entry_date_requested": entry_req,
            "entry_date_used": entry_used,
            "exit_date_requested": exit_req,
            "exit_date_used": exit_used,
            "reason": "rebalance_exit",
        })
        positions.pop(ticker, None)

    final_equity = float(cash)
    if equity_rows:
        equity_rows[-1]["cash_usd"] = float(cash)
        equity_rows[-1]["positions_value_usd"] = 0.0
        equity_rows[-1]["equity_usd"] = float(cash)

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["datetime", "action", "ticker"]).reset_index(drop=True)

    equity_curve_df = pd.DataFrame(equity_rows)
    if not equity_curve_df.empty:
        equity_curve_df = equity_curve_df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    fold_summary = {
        "fold_id": fold_id,
        "starting_capital_usd": float(starting_cash_usd),
        "ending_capital_usd": float(final_equity),
        "pnl_usd": float(final_equity - float(starting_cash_usd)),
        "pnl_pct": float((final_equity / float(starting_cash_usd) - 1.0) if float(starting_cash_usd) > 0 else 0.0),
        "total_fees_usd": float(total_fees),
        "n_buys": int((trades_df["action"] == "BUY").sum()) if not trades_df.empty else 0,
        "n_sells": int((trades_df["action"] == "SELL").sum()) if not trades_df.empty else 0,
        "n_ffill_days": int(n_ffill_days),
        "entry_date_used": str(entry_used.date()),
        "exit_date_used": str(exit_used.date()),
        "entry_gap_days": int(entry_gap_days),
        "exit_gap_days": int(exit_gap_days),
        "n_selected_tickers": int(len(valid_tickers)),
        "leakage_tainted": False,
    }

    missing_tickers = [t for t in tickers if t not in valid_tickers]
    missing_reasons = {t: "missing_entry_or_exit_price" for t in missing_tickers}

    return {
        "trades_df": trades_df,
        "equity_curve_df": equity_curve_df,
        "fold_summary": fold_summary,
        "selected_tickers_used": valid_tickers,
        "weights_used": weights_used,
        "missing_tickers": missing_tickers,
        "missing_reasons": missing_reasons,
    }


def compute_max_drawdown_from_equity(equity_curve_df: pd.DataFrame) -> float:
    if equity_curve_df is None or equity_curve_df.empty or "equity_usd" not in equity_curve_df.columns:
        return 0.0
    s = pd.to_numeric(equity_curve_df["equity_usd"], errors="coerce").dropna()
    if s.empty:
        return 0.0
    peak = s.cummax()
    dd = (s - peak) / peak.replace(0, np.nan)
    return float(dd.min()) if not dd.empty else 0.0


def to_daily_returns_from_equity(equity_curve_df: pd.DataFrame) -> pd.Series:
    if equity_curve_df is None or equity_curve_df.empty:
        return pd.Series(dtype=float)
    s = pd.Series(
        pd.to_numeric(equity_curve_df["equity_usd"], errors="coerce").values,
        index=pd.to_datetime(equity_curve_df["date"]),
    ).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return s.pct_change().dropna()
