"""Reglas críticas de la cartera fija: tamaño, umbral de salida, rotación y pesos."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from module.evaluation.portfolio import PortfolioState, decide_orders


def _row(ticker: str, rank: float, date: str = "2000-01-15") -> dict:
    return {"ticker": ticker, "snapshot_date": date, "meta_score": rank,
            "meta_rank": rank, "is_quarterly": True}


def test_fixed_size_weights_sum_to_one_and_respect_two_to_one(portfolio_settings) -> None:
    settings = replace(portfolio_settings, target_size=5, min_hold_percentile=80)
    scores = pd.DataFrame([_row(f"T{i}", 1 - i * 0.01) for i in range(8)])
    orders, weights = decide_orders(PortfolioState.empty(), scores, settings)

    buys = [order for order in orders if order["side"] == "buy"]
    assert len(buys) == 5
    assert len(weights) == 5
    assert sum(weights.values()) == pytest.approx(1.0)
    assert max(weights.values()) <= 2 * min(weights.values()) + 1e-12


def test_holder_at_or_below_maintenance_threshold_is_sold(portfolio_settings) -> None:
    settings = replace(portfolio_settings, target_size=3, min_hold_percentile=80)
    state = PortfolioState.from_holdings({"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3})
    scores = pd.DataFrame([
        _row("AAA", 0.80), _row("BBB", 0.90), _row("CCC", 0.85), _row("DDD", 0.95),
    ])
    orders, weights = decide_orders(state, scores, settings)

    assert any(order["ticker"] == "AAA" and order["side"] == "sell"
               and order["reason"] == "dropped_below_min" for order in orders)
    assert "DDD" in weights and len(weights) == 3


def test_multiple_rotations_require_configured_edge(portfolio_settings) -> None:
    settings = replace(portfolio_settings, target_size=3, min_hold_percentile=60,
                       rotation_edge_percentiles=10)
    state = PortfolioState.from_holdings({"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3})
    scores = pd.DataFrame([
        _row("AAA", 0.92), _row("BBB", 0.80), _row("CCC", 0.75),
        _row("DDD", 0.95), _row("EEE", 0.91),
    ])
    orders, weights = decide_orders(state, scores, settings)

    sells = [order for order in orders if order["side"] == "sell"]
    buys = [order for order in orders if order["side"] == "buy"]
    assert {order["ticker"] for order in sells} == {"BBB", "CCC"}
    assert {order["ticker"] for order in buys} == {"DDD", "EEE"}
    assert len(weights) == 3 and sum(weights.values()) == pytest.approx(1.0)


def test_small_advantage_does_not_rotate(portfolio_settings) -> None:
    settings = replace(portfolio_settings, target_size=3, min_hold_percentile=60,
                       rotation_edge_percentiles=10)
    state = PortfolioState.from_holdings({"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3})
    scores = pd.DataFrame([
        _row("AAA", 0.92), _row("BBB", 0.85), _row("CCC", 0.80), _row("DDD", 0.89),
    ])
    orders, _ = decide_orders(state, scores, settings)
    assert not any(order["side"] == "sell" for order in orders)
    assert not any(order["ticker"] == "DDD" for order in orders)
