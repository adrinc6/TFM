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


def test_position_below_expected_alpha_threshold_is_sold_to_cash() -> None:
    """La venta a efectivo solo existe bajo la política de oportunidad y respeta el suelo."""
    state = PortfolioState(holdings={"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    scores = _scores(ticker=["A", "B", "C", "D"], meta_rank=[0.95, 0.30, 0.90, 0.85],
                     expected_excess_return=[0.0300, 0.0050, 0.0250, 0.0200])
    orders, weights = decide_orders(
        state, scores,
        Settings(target_size=4, cash_policy="opportunity_cash", max_cash_weight=0.25,
                 exit_expected_alpha_bps=100.0, sizing_mode="equal"),
    )
    assert {order["reason"] for order in orders} >= {"expected_alpha_below_exit"}
    assert set(weights) == {"A", "C", "D"}
    assert abs(1.0 - sum(weights.values()) - 0.25) < 1e-12


def test_fully_invested_never_sells_just_to_rebuy_the_same_names() -> None:
    """Con todas las posiciones bajo el umbral y sin retador mejor, no se emite ninguna orden.

    La versión anterior vendía la cartera entera por umbral y la recompraba en el mismo snapshot
    (las vendidas seguían siendo las mejores por ranking): una ida y vuelta completa para quedar
    igual. Una venta sin destino mejor después de costes no es una decisión, es un peaje.
    """
    state = PortfolioState(holdings={"A": 0.5, "B": 0.5})
    scores = _scores(ticker=["A", "B"], meta_rank=[0.91, 0.70],
                     expected_excess_return=[0.0050, 0.0030])
    orders, weights = decide_orders(
        state, scores,
        Settings(target_size=2, cash_policy="fully_invested", exit_expected_alpha_bps=250.0,
                 sizing_mode="equal"),
    )
    assert orders == []
    assert weights == {"A": 0.5, "B": 0.5}


def test_uncalibrated_expected_alpha_never_triggers_a_sale() -> None:
    """Sin calibración no hay evidencia económica, y sin evidencia no se actúa."""
    state = PortfolioState(holdings={"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    scores = _scores(ticker=["A", "B", "C", "D"], meta_rank=[0.91, 0.70, 0.60, 0.50],
                     expected_excess_return=[float("nan")] * 4)
    orders, weights = decide_orders(
        state, scores,
        Settings(target_size=4, cash_policy="opportunity_cash", max_cash_weight=0.25,
                 exit_expected_alpha_bps=250.0, sizing_mode="equal"),
    )
    assert not [order for order in orders if order["reason"] == "expected_alpha_below_exit"]
    assert set(weights) == {"A", "B", "C", "D"}


def test_opportunity_cash_respects_the_cap_and_the_diversification_floor() -> None:
    """Sin candidatas sobre el umbral la plaza queda en efectivo, pero nunca bajo el suelo.

    Con tope del 25 % y 4 plazas, el suelo son 3 posiciones: el 75 % invertido se reparte entre las
    3 mejores por ranking aunque la tercera no supere el umbral. Sin ese suelo, una única admisible
    concentraría el 75 % de la cartera en una acción.
    """
    scores = _scores(ticker=["A", "B", "C", "D"], meta_rank=[0.99, 0.95, 0.90, 0.85],
                     expected_excess_return=[0.0400, 0.0300, 0.0005, 0.0002])
    settings = Settings(
        target_size=4, cash_policy="opportunity_cash", max_cash_weight=0.25,
        exit_expected_alpha_bps=100.0, sizing_mode="equal",
    )
    orders, weights = decide_orders(PortfolioState.empty(), scores, settings)
    cash = 1.0 - sum(weights.values())
    assert set(weights) == {"A", "B", "C"}
    assert {order["reason"] for order in orders} >= {"cash_floor_fill"}
    assert abs(cash - settings.max_cash_weight) < 1e-12


def test_entry_needs_the_exit_threshold_plus_its_own_round_trip() -> None:
    """Histéresis: comprar exige umbral de salida + coste de ida y vuelta; mantener, solo el umbral.

    Sin la banda, una acción oscilando alrededor del umbral se compra y se vende en snapshots
    consecutivos pagando costes con ventaja esperada nula.
    """
    settings = Settings(
        target_size=2, cash_policy="opportunity_cash", max_cash_weight=0.25,
        exit_expected_alpha_bps=100.0, commission_bps=5, slippage_bps=10, sizing_mode="equal",
    )
    # Banda: mantener exige 100 pb; entrar exige 100 + 2*(5+10) = 130 pb. B está en 115 pb.
    in_band = _scores(ticker=["A", "B"], meta_rank=[0.99, 0.95],
                      expected_excess_return=[0.0400, 0.0115])
    orders, weights = decide_orders(PortfolioState.empty(), in_band, settings)
    buys = {order["ticker"] for order in orders if order["reason"] == "initial_fill"}
    assert "B" not in buys

    state = PortfolioState(holdings={"A": 0.5, "B": 0.5})
    orders, weights = decide_orders(state, in_band, settings)
    assert not [order for order in orders if order["reason"] == "expected_alpha_below_exit"]
    assert set(weights) == {"A", "B"}


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
