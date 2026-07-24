from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from module.evaluation.portfolio import PortfolioState, decide_orders


def _row(ticker: str, rank: float, date: str = "2000-01-15") -> dict:
    return {
        "ticker": ticker,
        "snapshot_date": date,
        "meta_score": rank,
        "meta_rank": rank,
        "is_quarterly": True,
    }


def _state(holdings: dict[str, float]) -> PortfolioState:
    return PortfolioState(
        holdings=dict(holdings),
        entry_dates={ticker: "1900-01-01" for ticker in holdings},
    )


def test_fixed_size_weights_sum_to_one(portfolio_settings) -> None:
    settings = replace(portfolio_settings, target_size=5, min_hold_percentile=80)
    scores = pd.DataFrame([_row(f"T{index}", 1 - index * 0.01) for index in range(8)])
    orders, weights = decide_orders(PortfolioState.empty(), scores, settings)

    assert len([order for order in orders if order["side"] == "buy"]) == 5
    assert len(weights) == 5
    assert sum(weights.values()) == pytest.approx(1.0)


def test_holder_at_or_below_maintenance_threshold_is_sold(portfolio_settings) -> None:
    settings = replace(portfolio_settings, target_size=3, min_hold_percentile=80)
    state = _state({"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3})
    scores = pd.DataFrame([
        _row("AAA", 0.80), _row("BBB", 0.90), _row("CCC", 0.85), _row("DDD", 0.95),
    ])
    orders, weights = decide_orders(state, scores, settings)

    assert any(order["ticker"] == "AAA" and order["reason"] == "dropped_below_min" for order in orders)
    assert "DDD" in weights and len(weights) == 3


def test_multiple_rotations_require_configured_edge(portfolio_settings) -> None:
    settings = replace(portfolio_settings, target_size=3, min_hold_percentile=60, rotation_edge_percentiles=10)
    state = _state({"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3})
    scores = pd.DataFrame([
        _row("AAA", 0.92), _row("BBB", 0.80), _row("CCC", 0.75),
        _row("DDD", 0.95), _row("EEE", 0.91),
    ])
    orders, weights = decide_orders(state, scores, settings)

    assert {order["ticker"] for order in orders if order["side"] == "sell"} == {"BBB", "CCC"}
    assert {order["ticker"] for order in orders if order["side"] == "buy"} == {"DDD", "EEE"}
    assert sum(weights.values()) == pytest.approx(1.0)


def test_small_relative_change_is_frozen_no_order(portfolio_settings) -> None:
    settings = replace(portfolio_settings, target_size=3, min_hold_percentile=80, rebalance_drift_tolerance=0.25)
    state = _state({"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3})
    scores = pd.DataFrame([_row("AAA", 0.98), _row("BBB", 0.98), _row("CCC", 0.98)])
    orders, weights = decide_orders(state, scores, settings)

    assert orders == []
    assert weights == pytest.approx({"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3})


def test_price_only_month_requires_more_edge_to_rotate(portfolio_settings) -> None:
    rows = [_row("AAA", 0.92), _row("BBB", 0.80), _row("CCC", 0.75), _row("DDD", 0.95)]
    settings = replace(
        portfolio_settings, target_size=3, min_hold_percentile=60, rotation_edge_percentiles=10,
        price_only_strictness_multiplier=3.0,
    )
    state = _state({"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3})

    quarterly_orders, _ = decide_orders(state, pd.DataFrame(rows), settings)
    price_only = pd.DataFrame([{**row, "is_quarterly": False} for row in rows])
    price_only_orders, _ = decide_orders(state, price_only, settings)

    assert any(order["ticker"] == "DDD" for order in quarterly_orders)
    assert not any(order["ticker"] == "DDD" for order in price_only_orders)
