"""Reglas causales de la cartera dinámica, siempre invertida en acciones."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from environment import Settings


@dataclass
class PortfolioState:
    holdings: dict[str, float] = field(default_factory=dict)
    entry_dates: dict[str, str] = field(default_factory=dict)
    entry_prices: dict[str, float] = field(default_factory=dict)
    units: dict[str, float] = field(default_factory=dict)
    entry_costs: dict[str, float] = field(default_factory=dict)
    cash: float = 0.0
    costs_paid: float = 0.0

    @classmethod
    def empty(cls) -> "PortfolioState":
        return cls()


def decide_orders(
    state: PortfolioState, scores_at_date: pd.DataFrame, settings: Settings,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Aplica expulsión, sustitución por ventaja y rebalanceo con tolerancia.

    El percentil del meta decide quién entra o sale; el ``meta_score`` crudo determina
    el tamaño de cada posición. No existe permanencia mínima temporal: conservar una
    acción depende exclusivamente de que continúe superando los umbrales causales.
    """
    if scores_at_date.empty:
        return [], dict(state.holdings)
    date = str(scores_at_date["snapshot_date"].iloc[0])
    strictness = _strictness(scores_at_date, settings)
    ranked = scores_at_date.dropna(subset=["meta_rank"]).copy()
    ranked["percentile"] = pd.to_numeric(ranked["meta_rank"], errors="coerce") * 100
    ranked = ranked.dropna(subset=["percentile"]).sort_values("percentile", ascending=False)
    percentile = dict(zip(ranked["ticker"].astype(str), ranked["percentile"]))
    rank_by_ticker = {ticker: value / 100.0 for ticker, value in percentile.items()}

    holders = set(state.holdings)
    orders: list[dict[str, Any]] = []
    minimum = settings.min_hold_percentile / strictness
    for ticker in sorted(tuple(holders)):
        if percentile.get(ticker, -1.0) <= minimum:
            holders.remove(ticker)
            orders.append(_order(date, ticker, "sell", "dropped_below_min", state.holdings.get(ticker, 0.0), 0.0))

    for ticker in ranked["ticker"].astype(str):
        if len(holders) >= settings.target_size:
            break
        if ticker not in holders:
            holders.add(ticker)
            orders.append(_order(date, ticker, "buy", "initial_fill", 0.0, None))

    edge = settings.rotation_edge_percentiles * strictness
    while len(holders) >= settings.target_size:
        outsider = next((ticker for ticker in ranked["ticker"].astype(str) if ticker not in holders), None)
        if outsider is None:
            break
        worst = min(holders, key=lambda ticker: percentile.get(ticker, -1.0))
        if percentile[outsider] - percentile.get(worst, -1.0) < edge:
            break
        holders.remove(worst)
        holders.add(outsider)
        orders.extend((_order(date, worst, "sell", "displaced_by_edge", state.holdings.get(worst, 0.0), 0.0),
                       _order(date, outsider, "buy", "edge_over_worst", 0.0, None)))

    target = _weights(
        sorted(holders, key=lambda ticker: -percentile.get(ticker, -1.0)),
        rank_by_ticker, minimum / 100.0, settings,
    )
    target = _apply_rebalance_tolerance(state.holdings, target, settings.rebalance_drift_tolerance * strictness)
    planned = {order["ticker"] for order in orders}
    for ticker, weight in target.items():
        current = state.holdings.get(ticker, 0.0)
        if abs(weight - current) > 1e-12 and ticker not in planned:
            orders.append(_order(date, ticker, "buy" if weight > current else "sell", "rebalance", current, weight))
    for order in orders:
        if order["weight_after"] is None:
            order["weight_after"] = target.get(order["ticker"], 0.0)
    return orders, target


def _strictness(scores: pd.DataFrame, settings: Settings) -> float:
    return 1.0 if "is_quarterly" not in scores or bool(scores["is_quarterly"].iloc[0]) else settings.price_only_strictness_multiplier


def _weights(
    tickers: list[str], meta_rank: dict[str, float], minimum_rank: float, settings: Settings,
) -> dict[str, float]:
    if not tickers:
        return {}
    if settings.sizing_mode == "equal" or len(tickers) == 1:
        return {ticker: 1.0 / len(tickers) for ticker in tickers}
    floor = min(max(float(minimum_rank), 0.0), 0.999999)
    raw = {
        ticker: 1.0 + min(max((float(meta_rank.get(ticker, floor)) - floor) / (1.0 - floor), 0.0), 1.0)
        for ticker in tickers
    }
    total = sum(raw.values())
    return {ticker: weight / total for ticker, weight in raw.items()}


def _apply_rebalance_tolerance(current: dict[str, float], target: dict[str, float], tolerance: float) -> dict[str, float]:
    frozen = {ticker: current[ticker] for ticker in target if current.get(ticker, 0.0) > 0 and abs(target[ticker] - current[ticker]) / current[ticker] < tolerance}
    movable = [ticker for ticker in target if ticker not in frozen]
    available = 1.0 - sum(frozen.values())
    total = sum(target[ticker] for ticker in movable)
    if available < 0 or (movable and total <= 0):
        return target
    return {**frozen, **{ticker: available * target[ticker] / total for ticker in movable}} if movable else frozen


def _order(date: str, ticker: str, side: str, reason: str, before: float, after: float | None) -> dict[str, Any]:
    return {"snapshot_date": date, "ticker": ticker, "side": side, "reason": reason, "weight_before": before, "weight_after": after}
