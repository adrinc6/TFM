"""Backtest contable de la cartera dinámica; SPY es únicamente el benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from environment import Settings
from module.evaluation.portfolio import PortfolioState, decide_orders


@dataclass
class BacktestResult:
    positions: pd.DataFrame
    orders: pd.DataFrame
    equity: pd.DataFrame
    annual_metrics: pd.DataFrame
    summary: dict = field(default_factory=dict)


def run_backtest(scores: pd.DataFrame, prices: pd.DataFrame, benchmark: pd.DataFrame, settings: Settings, diagnostics: pd.DataFrame | None = None) -> BacktestResult:
    from module.evaluation.profiles import apply_profile
    scores = apply_profile(scores, settings.profile).sort_values("snapshot_date")
    prices_by_date = _prices(prices)
    benchmark_by_date = {row.snapshot_date: float(row.price) for row in benchmark.itertuples(index=False)}
    grouped = {date: frame for date, frame in scores.groupby("snapshot_date", sort=False)}
    snapshots = sorted(grouped)
    if not snapshots:
        raise ValueError("No hay snapshots para el backtest.")
    state, value, benchmark_value = PortfolioState.empty(), 100.0, 100.0
    previous_prices: dict[str, float] = {}
    previous_benchmark = benchmark_by_date.get(snapshots[0])
    if not previous_benchmark:
        raise ValueError("Sin benchmark PIT inicial.")
    positions_rows: list[dict] = []
    orders_rows: list[dict] = []
    equity_rows: list[dict] = []
    corrupt: list[dict] = []
    for index, date in enumerate(snapshots):
        stock_return, drifted = _mark_to_market(state.holdings, previous_prices, prices_by_date, date, settings.max_monthly_position_return, corrupt)
        state.holdings = drifted
        price = benchmark_by_date.get(date, previous_benchmark)
        benchmark_return = price / previous_benchmark - 1 if index else 0.0
        benchmark_value *= 1 + benchmark_return
        before_orders = value * (1 + stock_return)
        current_prices = prices_by_date.get(date, {})
        frame = grouped[date]
        tradable = frame.loc[frame["ticker"].map(lambda ticker: bool(current_prices.get(ticker, 0) > 0))]
        orders, target = decide_orders(state, tradable, settings)
        if not target:
            target = dict(state.holdings)
        if not target:
            raise ValueError(f"No hay acciones negociables para invertir el 100 % en {date}.")
        target = _normalise(target)
        priced, drag = _price_orders(orders, current_prices, state.holdings, before_orders, settings, previous_prices)
        value_after = before_orders * (1 - drag)
        prior_entries = dict(state.entry_dates)
        prior_prices = dict(state.entry_prices)
        buy_costs = {row["ticker"]: row["commission_amount"] + row["slippage_amount"] for row in priced if row["side"] == "buy"}
        position_values = {ticker: value_after * weight for ticker, weight in target.items()}
        units = {ticker: position_values[ticker] / current_prices[ticker] for ticker in target}
        state = PortfolioState(
            holdings=target,
            entry_dates={ticker: prior_entries.get(ticker, date) for ticker in target},
            entry_prices={ticker: prior_prices.get(ticker, current_prices[ticker]) for ticker in target},
            units=units,
            entry_costs={ticker: buy_costs.get(ticker, 0.0) for ticker in target},
            cash=0.0,
            costs_paid=state.costs_paid + before_orders * drag,
        )
        previous_prices = {ticker: current_prices[ticker] for ticker in target}
        turnover = sum(abs(row["weight_after"] - row["weight_before"]) for row in priced)
        for ticker, weight in target.items():
            positions_rows.append({"snapshot_date": date, "ticker": ticker, "weight": weight, "entry_date": state.entry_dates[ticker], "entry_price": state.entry_prices[ticker], "valuation_price": current_prices[ticker], "units": units[ticker], "market_value": position_values[ticker], "entry_cost": state.entry_costs[ticker], "months_held": _months(state.entry_dates[ticker], date), "current_percentile": _percentile(ticker, frame)})
        orders_rows.extend(priced)
        equity_rows.append({"snapshot_date": date, "period_start_portfolio_value": value, "period_start_benchmark_value": benchmark_value / (1 + benchmark_return) if 1 + benchmark_return else benchmark_value, "portfolio_value": value_after, "benchmark_value": benchmark_value, "portfolio_return": value_after / value - 1, "benchmark_return": benchmark_return, "excess_return": value_after / value - 1 - benchmark_return, "turnover_pct": turnover, "gross_return": stock_return, "cost_drag": drag, "stock_sleeve_return": stock_return, "positions_value": sum(position_values.values()), "cash": 0.0, "total_weight": sum(target.values()), "accounting_error": value_after - sum(position_values.values()), "cumulative_costs": state.costs_paid})
        value, previous_benchmark = value_after, price
    equity = pd.DataFrame(equity_rows)
    annual = _annual_metrics(equity)
    summary = _summary(equity, annual, settings, diagnostics)
    summary.update({"annualized_turnover": float(equity["turnover_pct"].mean() * 12), "corrupt_returns_neutralized": len(corrupt), "portfolio_policy": "dynamic_meta_rank", "sizing_mode": settings.sizing_mode})
    return BacktestResult(pd.DataFrame(positions_rows), pd.DataFrame(orders_rows), equity, annual, summary)


def _prices(prices: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {str(date): dict(zip(group.ticker, group.price)) for date, group in prices.groupby("snapshot_date")}


def _normalise(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    return {ticker: weight / total for ticker, weight in weights.items()} if total > 0 else {}


def _mark_to_market(holdings: dict[str, float], previous: dict[str, float], prices: dict[str, dict[str, float]], date: str, maximum: float, corrupt: list[dict]) -> tuple[float, dict[str, float]]:
    if not holdings:
        return 0.0, {}
    grown: dict[str, float] = {}
    for ticker, weight in holdings.items():
        old, new = previous.get(ticker), prices.get(date, {}).get(ticker)
        change = new / old - 1 if old and new and old > 0 else 0.0
        if abs(change) > maximum:
            corrupt.append({"snapshot_date": date, "ticker": ticker, "position_return": change})
            change = 0.0
        grown[ticker] = weight * (1 + change)
    total = sum(grown.values())
    return total - 1, _normalise(grown) if total > 0 else {}


def _price_orders(orders: list[dict], prices: dict[str, float], holdings: dict[str, float], value: float, settings: Settings, fallback: dict[str, float]) -> tuple[list[dict], float]:
    rows, total = [], 0.0
    rate = (settings.commission_bps + settings.slippage_bps) / 10_000
    for order in orders:
        before, after = float(order.get("weight_before") or holdings.get(order["ticker"], 0.0)), float(order.get("weight_after") or 0.0)
        notional = abs(after - before) * value
        commission, slippage = notional * settings.commission_bps / 10_000, notional * settings.slippage_bps / 10_000
        rows.append({**order, "weight_before": before, "weight_after": after, "price": prices.get(order["ticker"], fallback.get(order["ticker"])), "notional": notional, "commission": commission, "slippage": slippage, "commission_amount": commission, "slippage_amount": slippage, "price_guard": None})
        total += notional * rate
    return rows, total / value if value else 0.0


def _months(start: str, end: str) -> int:
    left, right = pd.Timestamp(start), pd.Timestamp(end)
    return max(0, (right.year - left.year) * 12 + right.month - left.month)


def _percentile(ticker: str, frame: pd.DataFrame) -> float:
    row = frame.loc[frame.ticker.eq(ticker), "meta_rank"]
    return float(row.iloc[0] * 100) if not row.empty else float("nan")


def _annual_metrics(equity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in equity.assign(year=pd.to_datetime(equity.snapshot_date).dt.year).groupby("year"):
        portfolio_return = group.portfolio_value.iloc[-1] / group.period_start_portfolio_value.iloc[0] - 1
        benchmark_return = group.benchmark_value.iloc[-1] / group.period_start_benchmark_value.iloc[0] - 1
        excess = group.excess_return.to_numpy()
        tracking = np.std(excess, ddof=1) if len(excess) > 1 else 0.0
        drawdown = 1 - group.portfolio_value.to_numpy() / np.maximum.accumulate(group.portfolio_value.to_numpy())
        rows.append({"year": int(year), "portfolio_return": portfolio_return, "benchmark_return": benchmark_return, "alpha": portfolio_return - benchmark_return, "beats_benchmark": portfolio_return > benchmark_return, "max_drawdown_year": float(drawdown.max()), "information_ratio_year": float(np.mean(excess) / tracking) if tracking else 0.0})
    return pd.DataFrame(rows)


def _summary(equity: pd.DataFrame, annual: pd.DataFrame, settings: Settings, diagnostics: pd.DataFrame | None) -> dict:
    dates = pd.to_datetime(equity.snapshot_date)
    years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, 1 / 12)
    cagr = (equity.portfolio_value.iloc[-1] / equity.period_start_portfolio_value.iloc[0]) ** (1 / years) - 1
    benchmark_cagr = (equity.benchmark_value.iloc[-1] / equity.period_start_benchmark_value.iloc[0]) ** (1 / years) - 1
    excess = equity.excess_return.to_numpy()
    tracking = np.std(excess, ddof=1) if len(excess) > 1 else 0.0
    drawdown = 1 - equity.portfolio_value.to_numpy() / np.maximum.accumulate(equity.portfolio_value.to_numpy())
    diagnostic = diagnostics.loc[diagnostics.agent.eq("meta_final")] if diagnostics is not None and "agent" in diagnostics else pd.DataFrame()
    ic = diagnostic.rank_ic.dropna() if not diagnostic.empty else pd.Series(dtype=float)
    return {"mean_rank_ic": float(ic.mean()) if len(ic) else 0.0, "rank_ic_positive_fraction": float((ic > 0).mean()) if len(ic) else 0.0, "rank_ic_std": float(ic.std(ddof=1)) if len(ic) > 1 else 0.0, "beat_rate": float((annual.alpha > 0).mean()) if len(annual) else 0.0, "max_drawdown": float(drawdown.max()), "cagr_portfolio": float(cagr), "cagr_benchmark": float(benchmark_cagr), "cagr_difference": float(cagr - benchmark_cagr), "geometric_excess_return": float(cagr - benchmark_cagr), "mean_annual_alpha": float(annual.alpha.mean()), "median_annual_alpha": float(annual.alpha.median()), "worst_year_alpha": float(annual.alpha.min()), "information_ratio": float(np.mean(excess) / tracking) if tracking else 0.0, "commission_bps": settings.commission_bps, "slippage_bps": settings.slippage_bps, "target_size": settings.target_size, "min_hold_percentile": settings.min_hold_percentile, "rotation_edge_percentiles": settings.rotation_edge_percentiles}
