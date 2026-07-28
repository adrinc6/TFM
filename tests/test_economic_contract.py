"""Contrato económico de la cartera: umbrales en puntos básicos, efectivo y contabilidad."""

from __future__ import annotations

import pandas as pd

from environment import Settings
from module.evaluation.portfolio import PortfolioState, decide_orders
from module.research.robustness import score_permutation


def _scores(**columns: object) -> pd.DataFrame:
    frame = pd.DataFrame(columns)
    frame["snapshot_date"] = ["2020-03-31"] * len(frame)
    frame["is_quarterly"] = [True] * len(frame)
    return frame


def test_add_one_placebo_p_value_never_zero() -> None:
    tickers = [f"T{i}" for i in range(8)]
    scores = pd.DataFrame({
        "ticker": tickers, "snapshot_date": ["2020-03-31"] * 8,
        "meta_rank": [i / 7 for i in range(8)],
    })
    targets = pd.DataFrame({
        "ticker": tickers, "snapshot_date": ["2020-03-31"] * 8,
        "forward_excess_return": [i / 7 for i in range(8)], "target_available": [True] * 8,
    })
    result = score_permutation(scores, targets, iterations=5, minimum_cross_section=8)
    assert result["observed_mean_rank_ic"] == 1.0
    assert result["p_value"] == 1 / 6


def test_spy_is_never_a_stock_position_and_fully_invested_holds_no_cash() -> None:
    scores = _scores(ticker=["AAPL", "MSFT"], meta_rank=[0.99, 0.90],
                     expected_excess_return=[0.05, 0.04])
    _, weights = decide_orders(
        PortfolioState.empty(), scores,
        Settings(target_size=2, cash_policy="fully_invested", sizing_mode="equal"),
    )
    assert "SPY" not in weights
    assert sum(weights.values()) == 1.0


def test_rotation_requires_clearing_the_round_trip_cost() -> None:
    """Una rotación solo se autoriza si la ventaja supera el coste de ida y vuelta más el margen."""
    state = PortfolioState(holdings={"A": 0.5, "B": 0.5})
    settings = Settings(
        target_size=2, commission_bps=5, slippage_bps=10, rotation_edge_bps=50,
        exit_expected_alpha_bps=0.0, sizing_mode="equal",
    )
    # Umbral = 2*(5+10) + 50 = 80 pb. Ventaja de C sobre B = 70 pb: insuficiente.
    insufficient = _scores(ticker=["A", "B", "C"], meta_rank=[0.91, 0.84, 0.92],
                           expected_excess_return=[0.0200, 0.0100, 0.0170])
    _, weights = decide_orders(state, insufficient, settings)
    assert set(weights) == {"A", "B"}

    # Ventaja de C sobre B = 90 pb: supera el umbral y la rotación se autoriza.
    sufficient = _scores(ticker=["A", "B", "C"], meta_rank=[0.91, 0.84, 0.92],
                         expected_excess_return=[0.0200, 0.0100, 0.0190])
    orders, weights = decide_orders(state, sufficient, settings)
    assert set(weights) == {"A", "C"}
    assert {order["reason"] for order in orders} >= {"displaced_by_net_edge", "net_edge_over_worst"}


def test_position_below_expected_alpha_threshold_is_sold() -> None:
    state = PortfolioState(holdings={"A": 0.5, "B": 0.5})
    scores = _scores(ticker=["A", "B", "C"], meta_rank=[0.91, 0.70, 0.95],
                     expected_excess_return=[0.0300, 0.0050, 0.0400])
    orders, weights = decide_orders(
        state, scores,
        Settings(target_size=2, exit_expected_alpha_bps=100.0, sizing_mode="equal"),
    )
    assert {order["reason"] for order in orders} >= {"expected_alpha_below_exit"}
    assert set(weights) == {"A", "C"}


def test_uncalibrated_expected_alpha_never_triggers_a_sale() -> None:
    """Sin calibración no hay evidencia económica, y sin evidencia no se actúa."""
    state = PortfolioState(holdings={"A": 0.5, "B": 0.5})
    scores = _scores(ticker=["A", "B"], meta_rank=[0.91, 0.70],
                     expected_excess_return=[float("nan"), float("nan")])
    orders, weights = decide_orders(
        state, scores, Settings(target_size=2, exit_expected_alpha_bps=250.0, sizing_mode="equal"),
    )
    assert not [order for order in orders if order["reason"] == "expected_alpha_below_exit"]
    assert set(weights) == {"A", "B"}


def test_opportunity_cash_leaves_the_slot_empty_and_respects_the_cap() -> None:
    """Si ninguna candidata supera el umbral, la plaza queda en efectivo, nunca por encima del tope."""
    scores = _scores(ticker=["A", "B", "C", "D"], meta_rank=[0.99, 0.95, 0.90, 0.85],
                     expected_excess_return=[0.0400, 0.0300, 0.0005, 0.0002])
    settings = Settings(
        target_size=4, cash_policy="opportunity_cash", max_cash_weight=0.40,
        exit_expected_alpha_bps=100.0, sizing_mode="equal",
    )
    _, weights = decide_orders(PortfolioState.empty(), scores, settings)
    cash = 1.0 - sum(weights.values())
    assert set(weights) == {"A", "B"}
    assert 0 < cash <= settings.max_cash_weight + 1e-12


def test_fully_invested_fills_the_slot_even_below_threshold() -> None:
    scores = _scores(ticker=["A", "B", "C"], meta_rank=[0.99, 0.95, 0.10],
                     expected_excess_return=[0.0400, 0.0300, -0.0500])
    _, weights = decide_orders(
        PortfolioState.empty(), scores,
        Settings(target_size=3, cash_policy="fully_invested", exit_expected_alpha_bps=100.0,
                 sizing_mode="equal"),
    )
    assert set(weights) == {"A", "B", "C"}
    assert sum(weights.values()) == 1.0


def test_alpha_proportional_sizing_caps_at_two_to_one() -> None:
    scores = _scores(ticker=["A", "B"], meta_rank=[1.0, 0.80],
                     expected_excess_return=[0.0600, 0.0200])
    _, weights = decide_orders(
        PortfolioState.empty(), scores,
        Settings(target_size=2, sizing_mode="alpha_proportional", exit_expected_alpha_bps=0.0),
    )
    assert weights["A"] == 2 / 3
    assert weights["B"] == 1 / 3
