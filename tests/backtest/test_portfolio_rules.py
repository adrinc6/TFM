"""Reglas críticas de la cartera fija: tamaño, umbral de salida, rotación y pesos."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from module.evaluation.portfolio import PortfolioState, _compute_weights, decide_orders


def _row(ticker: str, rank: float, date: str = "2000-01-15") -> dict:
    return {"ticker": ticker, "snapshot_date": date, "meta_score": rank,
            "meta_rank": rank, "is_quarterly": True}


def test_fixed_size_weights_sum_to_one(portfolio_settings) -> None:
    settings = replace(portfolio_settings, target_size=5, min_hold_percentile=80)
    scores = pd.DataFrame([_row(f"T{i}", 1 - i * 0.01) for i in range(8)])
    orders, weights = decide_orders(PortfolioState.empty(), scores, settings)

    buys = [order for order in orders if order["side"] == "buy"]
    assert len(buys) == 5
    assert len(weights) == 5
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(weight > 0 for weight in weights.values())


def test_sizing_scale_best_weighs_double_the_worst(portfolio_settings) -> None:
    """Escala min-max in-basket sobre meta_score: el mejor del basket pesa el doble que el peor,
    lineal en medio."""
    # meta_score crudo (no percentiles): el peor del basket es el ancla, el mejor pesa 2x.
    meta = {"TOP": 0.92, "MID": 0.87, "WORST": 0.82}
    weights = _compute_weights(list(meta), meta)
    # r: TOP=1, MID=0.5, WORST=0 => score 2:1.5:1 => peso 2:1.5:1
    assert weights["TOP"] == pytest.approx(2 * weights["WORST"], rel=1e-9)
    assert weights["MID"] == pytest.approx(1.5 * weights["WORST"], rel=1e-9)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_sizing_scale_is_in_basket_not_global(portfolio_settings) -> None:
    """El ancla es el PEOR del basket, no un percentil global: dos meta_scores muy próximos en
    términos absolutos igualmente se separan 2:1 porque uno es el mejor y otro el peor del basket."""
    meta = {"A": 0.951, "B": 0.949}
    weights = _compute_weights(list(meta), meta)
    # Aunque casi iguales en absoluto, A es el mejor y B el peor del basket -> 2:1 exacto.
    assert weights["A"] == pytest.approx(2 * weights["B"], rel=1e-9)
    # Con un tercero en medio, se interpola linealmente.
    meta3 = {"A": 1.0, "MID": 0.75, "B": 0.5}
    w3 = _compute_weights(list(meta3), meta3)
    assert w3["MID"] == pytest.approx(1.5 * w3["B"], rel=1e-9)


def test_sizing_scale_equal_scores_give_equal_weights(portfolio_settings) -> None:
    flat = {"A": 0.9, "B": 0.9, "C": 0.9}
    weights = _compute_weights(list(flat), flat)
    for weight in weights.values():
        assert weight == pytest.approx(1 / 3)


def test_sizing_single_ticker_gets_full_weight(portfolio_settings) -> None:
    assert _compute_weights(["ONLY"], {"ONLY": 0.88}) == {"ONLY": 1.0}


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


# --- Rebalanceo real: umbral relativo (congelar/mover) + reparto por relaciones del target ---


def test_small_relative_change_is_frozen_no_order(portfolio_settings) -> None:
    """Un tenente cuyo peso objetivo cambia menos del umbral relativo se congela: sin orden,
    conserva su peso actual."""
    settings = replace(portfolio_settings, target_size=3, min_hold_percentile=80,
                       rebalance_drift_tolerance=0.25)
    # meta_scores iguales -> target 1/3 cada uno; partiendo de 1/3, el cambio es ~0 -> congelados.
    state = PortfolioState.from_holdings({"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3})
    scores = pd.DataFrame([_row("AAA", 0.98), _row("BBB", 0.98), _row("CCC", 0.98)])
    orders, weights = decide_orders(state, scores, settings)
    assert orders == []                                   # todos congelados
    assert weights == pytest.approx({"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3})


def test_large_relative_change_rebalances(portfolio_settings) -> None:
    """Un tenente muy sobreponderado (cambio >= 25 %) se rebalancea; los estables no."""
    settings = replace(portfolio_settings, target_size=3, min_hold_percentile=80,
                       rebalance_drift_tolerance=0.25)
    # AAA entró con 60 %, su target baja a ~1/3 -> cambio enorme -> se mueve.
    state = PortfolioState.from_holdings({"AAA": 0.60, "BBB": 0.20, "CCC": 0.20})
    scores = pd.DataFrame([_row("AAA", 0.99), _row("BBB", 0.98), _row("CCC", 0.97)])
    orders, weights = decide_orders(state, scores, settings)
    assert any(order["ticker"] == "AAA" for order in orders)  # AAA se rebalancea
    assert weights["AAA"] < 0.60                              # baja hacia su target
    assert sum(weights.values()) == pytest.approx(1.0)


def test_budget_split_preserves_global_relations(portfolio_settings) -> None:
    """El presupuesto liberado se reparte entre móviles conservando las relaciones del target
    global: dos móviles con percentiles parecidos reciben pesos parecidos."""
    settings = replace(portfolio_settings, target_size=3, min_hold_percentile=80,
                       rebalance_drift_tolerance=0.25)
    # CCC congelado (estable); AAA (sobreponderado) y DDD (entra) son móviles con percentiles
    # casi iguales (100 y 98) -> deben repartirse el presupuesto casi a partes iguales.
    state = PortfolioState.from_holdings({"AAA": 0.60, "BBB": 0.20, "CCC": 0.20})
    scores = pd.DataFrame([_row("AAA", 1.00), _row("CCC", 0.85), _row("DDD", 0.98),
                           _row("BBB", 0.50)])  # BBB cae bajo min_hold -> sale
    orders, weights = decide_orders(state, scores, settings)
    assert "DDD" in weights and "AAA" in weights
    # AAA (100) y DDD (98) casi iguales pese a partir de pesos muy distintos (0.60 vs 0).
    assert weights["AAA"] / weights["DDD"] < 1.15
    assert sum(weights.values()) == pytest.approx(1.0)


def test_rebalance_is_independent_of_ticker_order(portfolio_settings) -> None:
    """Clasificar contra el target global fijo hace el resultado independiente del orden."""
    settings = replace(portfolio_settings, target_size=3, min_hold_percentile=80,
                       rebalance_drift_tolerance=0.25)
    state = PortfolioState.from_holdings({"AAA": 0.60, "BBB": 0.20, "CCC": 0.20})
    rows = [_row("AAA", 1.00), _row("BBB", 0.98), _row("CCC", 0.85)]
    _, weights_forward = decide_orders(state, pd.DataFrame(rows), settings)
    _, weights_reversed = decide_orders(state, pd.DataFrame(list(reversed(rows))), settings)
    assert weights_forward == pytest.approx(weights_reversed)


# --- Revisiones sin fundamentales nuevos: price_only_strictness_multiplier ---


def _price_only(rows: list[dict]) -> pd.DataFrame:
    """Mismo corte transversal pero marcado como revisión de solo precio."""
    return pd.DataFrame([{**row, "is_quarterly": False} for row in rows])


def test_price_only_month_keeps_holder_that_would_be_expelled(portfolio_settings) -> None:
    """Con fundamentales nuevos AAA (percentil 80) cae al umbral y sale; sin ellos, el umbral
    efectivo baja (80/2 = 40) y AAA se conserva."""
    rows = [_row("AAA", 0.80), _row("BBB", 0.90), _row("CCC", 0.85), _row("DDD", 0.95)]
    quarterly = replace(portfolio_settings, target_size=3, min_hold_percentile=80,
                        price_only_strictness_multiplier=2.0)
    state = PortfolioState.from_holdings({"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3})

    orders_quarterly, _ = decide_orders(state, pd.DataFrame(rows), quarterly)
    assert any(order["ticker"] == "AAA" and order["reason"] == "dropped_below_min"
               for order in orders_quarterly)

    orders_price_only, weights = decide_orders(state, _price_only(rows), quarterly)
    assert not any(order["ticker"] == "AAA" and order["side"] == "sell"
                   for order in orders_price_only)
    assert "AAA" in weights


def test_price_only_month_requires_more_edge_to_rotate(portfolio_settings) -> None:
    """La ventaja de DDD sobre CCC (20 puntos) basta con fundamentales nuevos, pero no cuando el
    multiplicador eleva la exigencia a 10 * 3 = 30 puntos."""
    rows = [_row("AAA", 0.92), _row("BBB", 0.80), _row("CCC", 0.75), _row("DDD", 0.95)]
    settings = replace(portfolio_settings, target_size=3, min_hold_percentile=60,
                       rotation_edge_percentiles=10, price_only_strictness_multiplier=3.0)
    state = PortfolioState.from_holdings({"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3})

    orders_quarterly, _ = decide_orders(state, pd.DataFrame(rows), settings)
    assert any(order["ticker"] == "DDD" for order in orders_quarterly)

    orders_price_only, _ = decide_orders(state, _price_only(rows), settings)
    assert not any(order["ticker"] == "DDD" for order in orders_price_only)


def test_price_only_month_freezes_weights_that_would_rebalance(portfolio_settings) -> None:
    """Deriva relativa de ~33 %: supera el umbral base (25 %) pero no el endurecido (25 % * 2)."""
    # meta_scores iguales -> target 1/3 para los tres. Partiendo de 0.25, la deriva relativa es
    # (1/3 - 0.25) / 0.25 = 33 % -> se mueve con el umbral base, se congela con el endurecido.
    rows = [_row("AAA", 0.98), _row("BBB", 0.98), _row("CCC", 0.98)]
    settings = replace(portfolio_settings, target_size=3, min_hold_percentile=80,
                       rebalance_drift_tolerance=0.25, price_only_strictness_multiplier=2.0)
    state = PortfolioState.from_holdings({"AAA": 0.25, "BBB": 0.25, "CCC": 0.50})

    orders_quarterly, _ = decide_orders(state, pd.DataFrame(rows), settings)
    assert any(order["ticker"] == "AAA" for order in orders_quarterly)

    orders_price_only, weights = decide_orders(state, _price_only(rows), settings)
    assert orders_price_only == []
    assert weights["AAA"] == pytest.approx(0.25)


def test_multiplier_one_leaves_price_only_months_unchanged(portfolio_settings) -> None:
    """Con el multiplicador neutro, una revisión de solo precio decide igual que una trimestral."""
    rows = [_row("AAA", 0.80), _row("BBB", 0.90), _row("CCC", 0.85), _row("DDD", 0.95)]
    settings = replace(portfolio_settings, target_size=3, min_hold_percentile=80,
                       price_only_strictness_multiplier=1.0)
    state = PortfolioState.from_holdings({"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3})

    _, weights_quarterly = decide_orders(state, pd.DataFrame(rows), settings)
    _, weights_price_only = decide_orders(state, _price_only(rows), settings)
    assert weights_quarterly == pytest.approx(weights_price_only)


def test_missing_is_quarterly_column_keeps_historic_behaviour(portfolio_settings) -> None:
    """Sin la columna (datos antiguos) no se endurece nada, aunque el multiplicador sea alto."""
    rows = [{k: v for k, v in row.items() if k != "is_quarterly"}
            for row in [_row("AAA", 0.80), _row("BBB", 0.90), _row("CCC", 0.85)]]
    settings = replace(portfolio_settings, target_size=3, min_hold_percentile=80,
                       price_only_strictness_multiplier=3.0)
    state = PortfolioState.from_holdings({"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3})

    orders, _ = decide_orders(state, pd.DataFrame(rows), settings)
    assert any(order["ticker"] == "AAA" and order["reason"] == "dropped_below_min"
               for order in orders)
