"""Reglas de decisión de cartera. Se prueban aisladas del simulador (`backtest.py`).

Contrato: `decide_orders(state, scores_at_date, settings)` devuelve la lista de órdenes
que llevan la cartera al estado siguiente, siguiendo las reglas del plan (expulsión,
ventaja, tamaño flexible 5-10, sizing con tope).
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from module.portfolio import PortfolioState, decide_orders


def _scores_row(ticker: str, meta_rank: float, snapshot_date: str = "2000-01-15") -> dict:
    return {"ticker": ticker, "snapshot_date": snapshot_date, "meta_score": meta_rank,
            "meta_rank": meta_rank, "is_quarterly": True}


def test_holder_kept_when_candidate_advantage_below_threshold(portfolio_settings) -> None:
    """Tenente en percentil 60 no rota si el mejor candidato fuera está en 62 (diff < 5)."""
    settings = replace(
        portfolio_settings,
        target_min=5, target_max=5,               # cartera llena, sin huecos
        min_hold_percentile=50,                    # el tenente esta por encima
        rotation_edge_percentiles=5,
        entry_min_percentile=60,
    )
    state = PortfolioState.from_holdings(
        {"AAA": 0.2, "BBB": 0.2, "CCC": 0.2, "DDD": 0.2, "GGG": 0.2},
    )
    scores = pd.DataFrame([
        _scores_row("AAA", 0.90), _scores_row("BBB", 0.80), _scores_row("CCC", 0.75),
        _scores_row("DDD", 0.65), _scores_row("GGG", 0.60),     # peor tenente en 60
        _scores_row("EEE", 0.62),                                # candidato fuera con solo +2
        _scores_row("FFF", 0.50),
    ])
    orders = decide_orders(state, scores, settings)

    # Nadie sale (todos por encima del 50), nadie entra (ventaja < 5).
    assert orders == []


def test_holder_below_min_percentile_is_dropped_even_without_replacement(portfolio_settings) -> None:
    """Un tenente que cae al percentil 40 sale, aunque nadie tenga la ventaja para rellenar."""
    settings = replace(
        portfolio_settings,
        target_min=5, target_max=5,
        min_hold_percentile=50,
        rotation_edge_percentiles=5,
        entry_min_percentile=80,           # nadie fuera lo cumple -> el hueco queda
    )
    state = PortfolioState.from_holdings(
        {"AAA": 0.2, "BBB": 0.2, "CCC": 0.2, "DDD": 0.2, "GGG": 0.2},
    )
    scores = pd.DataFrame([
        _scores_row("AAA", 0.90), _scores_row("BBB", 0.80), _scores_row("CCC", 0.75),
        _scores_row("DDD", 0.65),
        _scores_row("GGG", 0.40),                                # se hunde
        _scores_row("EEE", 0.55), _scores_row("FFF", 0.50),      # nadie llega a 0.80
    ])
    orders = decide_orders(state, scores, settings)

    sells = [order for order in orders if order["side"] == "sell"]
    buys = [order for order in orders if order["side"] == "buy"]
    assert len(sells) == 1 and sells[0]["ticker"] == "GGG"
    assert sells[0]["reason"] == "dropped_below_min"
    assert buys == []


def test_ticker_can_reenter_after_leaving(portfolio_settings) -> None:
    """Ida y vuelta: un ticker con score 90 -> 40 -> 88 entra, sale, y vuelve a entrar."""
    settings = replace(
        portfolio_settings,
        target_min=5, target_max=5,
        min_hold_percentile=50,
        rotation_edge_percentiles=5,
        entry_min_percentile=80,
    )
    # Snapshot 1: entra
    state = PortfolioState.empty()
    scores_1 = pd.DataFrame([
        _scores_row("AAA", 0.90, "2000-01-15"), _scores_row("BBB", 0.85, "2000-01-15"),
        _scores_row("CCC", 0.83, "2000-01-15"), _scores_row("DDD", 0.82, "2000-01-15"),
        _scores_row("EEE", 0.81, "2000-01-15"),
    ])
    orders_1 = decide_orders(state, scores_1, settings)
    assert any(order["ticker"] == "AAA" and order["side"] == "buy" for order in orders_1)

    # Snapshot 2: AAA se hunde a 0.40 -> sale
    state = state.apply(orders_1, prices={"AAA": 100, "BBB": 101, "CCC": 102, "DDD": 103, "EEE": 104})
    scores_2 = pd.DataFrame([
        _scores_row("AAA", 0.40, "2000-02-15"),                     # se cae
        _scores_row("BBB", 0.85, "2000-02-15"), _scores_row("CCC", 0.83, "2000-02-15"),
        _scores_row("DDD", 0.82, "2000-02-15"), _scores_row("EEE", 0.81, "2000-02-15"),
    ])
    orders_2 = decide_orders(state, scores_2, settings)
    assert any(order["ticker"] == "AAA" and order["side"] == "sell"
               and order["reason"] == "dropped_below_min" for order in orders_2)

    # Snapshot 3: AAA vuelve a 0.88 con hueco disponible -> entra
    state = state.apply(orders_2, prices={"AAA": 100, "BBB": 102, "CCC": 103, "DDD": 104, "EEE": 105})
    scores_3 = pd.DataFrame([
        _scores_row("AAA", 0.88, "2000-03-15"),                     # vuelve
        _scores_row("BBB", 0.85, "2000-03-15"), _scores_row("CCC", 0.83, "2000-03-15"),
        _scores_row("DDD", 0.82, "2000-03-15"), _scores_row("EEE", 0.81, "2000-03-15"),
    ])
    orders_3 = decide_orders(state, scores_3, settings)
    assert any(order["ticker"] == "AAA" and order["side"] == "buy" for order in orders_3)


def test_sizing_respects_max_weight_and_sums_to_one(portfolio_settings) -> None:
    """Cuatro posiciones 95/85/80/78 con MAX_WEIGHT=30 %: la mejor recibe 30 %, resto se reparte."""
    settings = replace(
        portfolio_settings,
        target_min=4, target_max=4,
        max_weight_per_position=0.30,
        entry_min_percentile=70,
    )
    state = PortfolioState.empty()
    scores = pd.DataFrame([
        _scores_row("AAA", 0.95), _scores_row("BBB", 0.85),
        _scores_row("CCC", 0.80), _scores_row("DDD", 0.78),
        _scores_row("EEE", 0.40),                                    # no entra
    ])
    orders = decide_orders(state, scores, settings)
    buys = [order for order in orders if order["side"] == "buy"]

    weights = {order["ticker"]: order["weight_after"] for order in buys}
    assert set(weights) == {"AAA", "BBB", "CCC", "DDD"}
    assert weights["AAA"] == pytest.approx(0.30)                     # tope activo
    assert all(w <= 0.30 + 1e-9 for w in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)     # capital invertido = 100 %


def test_portfolio_size_flexes_when_few_candidates_qualify(portfolio_settings) -> None:
    """Si solo 6 candidatos superan ENTRY_MIN_PERCENTILE=80, la cartera es de 6, no de 10."""
    settings = replace(
        portfolio_settings,
        target_min=5, target_max=10,
        max_weight_per_position=0.25,
        entry_min_percentile=80,
    )
    state = PortfolioState.empty()
    scores = pd.DataFrame([
        _scores_row("AAA", 0.95), _scores_row("BBB", 0.92), _scores_row("CCC", 0.90),
        _scores_row("DDD", 0.85), _scores_row("EEE", 0.83), _scores_row("FFF", 0.81),
        _scores_row("GGG", 0.75),                                    # bajo 80: fuera
        _scores_row("HHH", 0.70), _scores_row("III", 0.50), _scores_row("JJJ", 0.30),
    ])
    orders = decide_orders(state, scores, settings)
    buys = [order for order in orders if order["side"] == "buy"]

    assert {order["ticker"] for order in buys} == {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF"}
