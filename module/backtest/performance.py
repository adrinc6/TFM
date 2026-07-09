"""Performance, turnover and benchmark comparison helpers."""

from __future__ import annotations

import pandas as pd

from environment import Settings


def position_performance(
    transactions: pd.DataFrame,
    evolution: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_ticker: str,
) -> pd.DataFrame:
    columns = [
        "lot_id", "ticker", "entry_date", "exit_date", "closed", "holding_days",
        "entry_price", "exit_price", "total_return", "annualized_return",
        "benchmark_total_return", "benchmark_annualized_return", "excess_total_return",
        "entry_manager_score", "entry_buy_today_score", "entry_reason", "exit_reason",
        "exit_reason_category",
    ]
    if transactions.empty or evolution.empty:
        return pd.DataFrame(columns=columns)
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    tx = transactions.copy()
    tx["date"] = pd.to_datetime(tx["date"])
    tx["_action_order"] = tx["action"].map({"SELL": 0, "BUY": 1}).fillna(2)
    tx = tx.sort_values(["date", "_action_order", "ticker"])
    end_date = pd.to_datetime(evolution["date"]).max()
    open_lots: dict[str, list[dict]] = {}
    rows = []
    lot_id = 1
    for row in tx.itertuples(index=False):
        if row.action == "BUY":
            open_lots.setdefault(row.ticker, []).append({
                "lot_id": lot_id,
                "ticker": row.ticker,
                "entry_date": row.date,
                "entry_price": last_price(prices, row.ticker, row.date),
                "entry_reason": getattr(row, "reason", ""),
                "entry_manager_score": getattr(row, "manager_score", None),
                "entry_buy_today_score": getattr(row, "buy_today_score", None),
            })
            lot_id += 1
        elif row.action == "SELL":
            lot = open_lots.get(row.ticker, []).pop(0) if open_lots.get(row.ticker) else None
            if lot:
                rows.append(performance_row(lot, row.date, prices, benchmark_ticker, getattr(row, "reason", ""), True))
    for lots in open_lots.values():
        for lot in lots:
            rows.append(performance_row(lot, end_date, prices, benchmark_ticker, "Open", False))
    return pd.DataFrame(rows, columns=columns)


def performance_row(
    lot: dict,
    exit_date: pd.Timestamp,
    prices: pd.DataFrame,
    benchmark_ticker: str,
    exit_reason: str,
    closed: bool,
) -> dict:
    entry_date = lot["entry_date"]
    entry_price = lot["entry_price"]
    exit_price = last_price(prices, lot["ticker"], exit_date)
    benchmark_entry = last_price(prices, benchmark_ticker, entry_date)
    benchmark_exit = last_price(prices, benchmark_ticker, exit_date)
    holding_days = max((exit_date - entry_date).days, 1)
    total_return = safe_return(entry_price, exit_price)
    benchmark_total_return = safe_return(benchmark_entry, benchmark_exit)
    return {
        "lot_id": lot["lot_id"],
        "ticker": lot["ticker"],
        "entry_date": entry_date.date().isoformat(),
        "exit_date": exit_date.date().isoformat(),
        "closed": bool(closed),
        "holding_days": holding_days,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "total_return": total_return,
        "annualized_return": annualize(total_return, holding_days),
        "benchmark_total_return": benchmark_total_return,
        "benchmark_annualized_return": annualize(benchmark_total_return, holding_days),
        "excess_total_return": total_return - benchmark_total_return,
        "entry_manager_score": lot.get("entry_manager_score"),
        "entry_buy_today_score": lot.get("entry_buy_today_score"),
        "entry_reason": lot.get("entry_reason", ""),
        "exit_reason": exit_reason,
        "exit_reason_category": reason_category(exit_reason),
    }


def turnover(transactions: pd.DataFrame, periods: int, performance: pd.DataFrame | None = None) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame([{
            "monthly_turnover": 0,
            "annual_turnover": 0,
            "average_holding_days": None,
            "median_holding_days": None,
            "buys": 0,
            "sells": 0,
            "sell_reason_mix": "",
        }])
    sells = transactions[transactions["action"] == "SELL"]
    buys = transactions[transactions["action"] == "BUY"]
    monthly = len(sells) / max(periods, 1)
    closed = (
        performance[performance["closed"]]
        if performance is not None and not performance.empty and "closed" in performance.columns
        else pd.DataFrame()
    )
    reason_mix = sells["reason"].map(reason_category).value_counts()
    return pd.DataFrame([{
        "monthly_turnover": monthly,
        "annual_turnover": monthly * 12,
        "average_holding_days": float(closed["holding_days"].mean()) if not closed.empty else None,
        "median_holding_days": float(closed["holding_days"].median()) if not closed.empty else None,
        "buys": int(len(buys)),
        "sells": int(len(sells)),
        "sell_reason_mix": "; ".join(f"{reason}={count}" for reason, count in reason_mix.items()),
    }])


def portfolio_vs_benchmark(
    evolution: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_ticker: str,
    transactions: pd.DataFrame,
    settings: Settings,
    weights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    tx = transactions.copy()
    if not tx.empty:
        tx["date"] = pd.to_datetime(tx["date"])
    cost_rate = (settings.transaction_cost_bps + settings.slippage_bps) / 10000
    weight_map = _weight_map(weights)
    rows = []
    previous_gross_value = previous_net_value = previous_benchmark_value = 1.0
    previous_benchmark = previous_date = None
    previous_date_key: str | None = None
    previous_tickers: list[str] = []
    for row in evolution.itertuples(index=False):
        date = pd.to_datetime(row.date)
        tickers = [ticker.strip() for ticker in row.tickers.split(",") if ticker.strip()]
        gross_return = (
            0.0
            if previous_date is None
            else weighted_basket_return(prices, previous_tickers, previous_date, date, weight_map.get(previous_date_key))
        )
        transaction_cost = period_transaction_cost(
            tx, date, len(tickers), cost_rate, weight_map.get(str(row.date)), weight_map.get(previous_date_key)
        )
        net_return = gross_return - transaction_cost
        benchmark_price = last_price(prices, benchmark_ticker, date)
        benchmark_return = 0.0 if previous_benchmark is None or benchmark_price is None else benchmark_price / previous_benchmark - 1
        previous_benchmark = benchmark_price or previous_benchmark
        previous_gross_value *= 1 + gross_return
        previous_net_value *= 1 + net_return
        previous_benchmark_value *= 1 + benchmark_return
        previous_date = date
        previous_date_key = str(row.date)
        previous_tickers = tickers
        rows.append({
            "date": row.date,
            "portfolio_gross_period_return": gross_return,
            "transaction_cost_drag": transaction_cost,
            "portfolio_period_return": net_return,
            "benchmark_period_return": benchmark_return,
            "period_alpha": net_return - benchmark_return,
            "portfolio_gross_value": previous_gross_value,
            "portfolio_value": previous_net_value,
            "benchmark_value": previous_benchmark_value,
        })
    return pd.DataFrame(rows)


def _weight_map(weights: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    if weights is None or weights.empty or "hybrid_weight" not in weights.columns:
        return {}
    result: dict[str, dict[str, float]] = {}
    for date, group in weights.groupby("date"):
        raw = group.set_index("ticker")["hybrid_weight"].fillna(0).clip(lower=0)
        total = raw.sum()
        result[str(date)] = (raw / total).to_dict() if total > 0 else {}
    return result


def safe_return(start_price: float | None, end_price: float | None) -> float:
    if start_price is None or end_price is None or start_price <= 0:
        return 0.0
    return float(end_price / start_price - 1)


def annualize(total_return: float, holding_days: int) -> float:
    if holding_days <= 0 or total_return <= -1:
        return 0.0
    return float((1 + total_return) ** (365.25 / holding_days) - 1)


def reason_category(reason: str | None) -> str:
    if not reason:
        return "Unknown"
    return str(reason).split(":", 1)[0].strip() or "Unknown"


def period_transaction_cost(
    transactions: pd.DataFrame,
    date: pd.Timestamp,
    holding_count: int,
    cost_rate: float,
    current_weights: dict[str, float] | None = None,
    previous_weights: dict[str, float] | None = None,
) -> float:
    if transactions.empty or cost_rate <= 0:
        return 0.0
    period_tx = transactions[transactions["date"] == date]
    if period_tx.empty:
        return 0.0
    if not current_weights and not previous_weights:
        count = int(len(period_tx))
        return float((count / max(holding_count, 1)) * cost_rate)
    notional_traded = 0.0
    for tx_row in period_tx.itertuples(index=False):
        if tx_row.action == "BUY":
            notional_traded += (current_weights or {}).get(tx_row.ticker, 0.0)
        elif tx_row.action == "SELL":
            notional_traded += (previous_weights or {}).get(tx_row.ticker, 0.0)
    return float(notional_traded * cost_rate)


def weighted_basket_return(
    prices: pd.DataFrame,
    tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    weights: dict[str, float] | None = None,
) -> float:
    if not tickers:
        return 0.0
    returns: dict[str, float] = {}
    for ticker in tickers:
        start_price = last_price(prices, ticker, start_date)
        end_price = last_price(prices, ticker, end_date)
        if start_price and end_price and start_price > 0:
            returns[ticker] = end_price / start_price - 1
    if not returns:
        return 0.0
    if weights:
        weighted = {ticker: weights.get(ticker, 0.0) for ticker in returns}
        total_weight = sum(weighted.values())
        if total_weight > 0:
            return float(sum(returns[ticker] * weighted[ticker] for ticker in returns) / total_weight)
    return float(pd.Series(list(returns.values())).mean())


def basket_return(prices: pd.DataFrame, tickers: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    return weighted_basket_return(prices, tickers, start_date, end_date, weights=None)


def last_price(prices: pd.DataFrame, ticker: str, date: pd.Timestamp) -> float | None:
    rows = prices[(prices["ticker"] == ticker) & (prices["date"] <= date)].sort_values("date")
    return None if rows.empty else float(rows.iloc[-1]["adj_close"])
