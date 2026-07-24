from __future__ import annotations

import pandas as pd

from environment import Settings
from module.evaluation.portfolio import PortfolioState, decide_orders
from module.evaluation.robustness import label_permutation_test


def test_add_one_placebo_p_value_never_zero() -> None:
    result = label_permutation_test(pd.DataFrame({"agent": ["meta_final"], "rank_ic": [1.0]}), [0.0] * 5)
    assert result["p_value"] == 1 / 6


def test_dynamic_portfolio_requires_edge_and_preserves_full_stock_weight() -> None:
    state = PortfolioState(holdings={"A": 0.5, "B": 0.5})
    scores = pd.DataFrame({"ticker": ["A", "B", "C"], "snapshot_date": ["2020-03-31"] * 3, "meta_rank": [0.91, 0.84, 0.92], "meta_score": [0.6, 0.4, 0.7], "is_quarterly": [True] * 3})
    _, weights = decide_orders(state, scores, Settings(target_size=2, min_hold_percentile=80, rotation_edge_percentiles=10))
    assert set(weights) == {"A", "B"}
    assert sum(weights.values()) == 1.0


def test_below_minimum_is_sold_and_best_candidate_fills_slot() -> None:
    state = PortfolioState(holdings={"A": 0.5, "B": 0.5})
    scores = pd.DataFrame({"ticker": ["A", "B", "C"], "snapshot_date": ["2020-03-31"] * 3, "meta_rank": [0.91, 0.70, 0.95], "meta_score": [0.6, 0.4, 0.7], "is_quarterly": [True] * 3})
    orders, weights = decide_orders(state, scores, Settings(target_size=2, min_hold_percentile=80))
    assert {order["reason"] for order in orders} >= {"dropped_below_min", "initial_fill"}
    assert set(weights) == {"A", "C"}


def test_weight_scale_is_anchored_to_minimum_hold_not_to_worst_basket_score() -> None:
    scores = pd.DataFrame({"ticker": ["A", "B"], "snapshot_date": ["2020-03-31"] * 2, "meta_rank": [1.0, 0.80], "meta_score": [0.99, 0.98], "is_quarterly": [True] * 2})
    _, weights = decide_orders(PortfolioState.empty(), scores, Settings(target_size=2, min_hold_percentile=80, sizing_mode="score_linear"))
    assert weights["A"] == 2 / 3
    assert weights["B"] == 1 / 3
