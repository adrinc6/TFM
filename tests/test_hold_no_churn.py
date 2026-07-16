"""Mantener, no rotar — invariante económico pedido explícitamente.

Cuando al revisar la cartera un nombre que ya se tiene sigue estando seleccionado, se CONSERVA: no se
vende para recomprarlo. Solo se opera el delta (nuevas entradas y salidas reales), y los costes se
cobran únicamente sobre operaciones reales, nunca sobre el solapamiento mantenido.

Pure-logic sobre las funciones reales `review_portfolio` y `period_transaction_cost`.
"""

from __future__ import annotations

import pandas as pd
import pytest

import module.strategy.portfolio as pf
from module.backtest.performance import period_transaction_cost


def _universe_row(ticker: str, snapshot_date: str, **overrides) -> dict:
    """Fila de universo "sana" que califica como entrada y como mantenible. Cubre todas las columnas
    que leen manager_score, _entry_candidates, _refresh_position y _position_from_row."""
    row = {
        "ticker": ticker,
        "snapshot_date": snapshot_date,
        "price": 100.0,
        "thesis_state": "Intact",
        "opportunity_type": "Compounder",
        "exit_score": 0.0,
        "would_buy_today": True,
        "buy_today_score": 0.70,
        "final_score": 0.80,
        "thesis_score": 0.60,
        "thesis_rank_score": 0.60,
        "conviction_score": 0.60,
        "business_quality_score": 0.70,
        "moat_score": 0.60,
        "risk_score": 0.60,
        "valuation_score": 0.60,
        "price_adjusted_valuation_score": 0.60,
        "momentum_score": 0.60,
        "positive_expectation_gap": 0.50,
        "expectation_gap": 0.10,
        "price_return_3m": 0.0,
        "price_return_6m": 0.0,
        "price_return_12m": 0.0,
        "price_return_since_fundamental": 0.0,
        "stale_fundamental_months": 0.0,
        "best_alternative_ticker": "",
        "opportunity_cost_score": 0.0,
        "sector": "Tech",
        "investment_thesis": "Tesis sana",
        "exit_thesis": "Se vende si se rompe",
        "catalyst": "Catalizador",
        "entry_trigger": "Entrada",
        "buy_reason": "Compra",
    }
    row.update(overrides)
    return row


def _universe(tickers: list[str], snapshot_date: str) -> pd.DataFrame:
    return pd.DataFrame([_universe_row(t, snapshot_date) for t in tickers])


def test_overlap_holdings_are_kept_without_sell_or_buy():
    held = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    current = pf.initial_portfolio(_universe(held, "2010-05-15"), "2010-05-15")
    assert set(current) == set(held)

    # Mismo universo sano un mes después: todos siguen seleccionados.
    updated, transactions = pf.review_portfolio(current, _universe(held, "2010-06-15"), "2010-06-15")

    assert set(updated) == set(held), "los nombres que siguen seleccionados deben conservarse"
    assert transactions == [], "conservar un nombre no debe generar ninguna transacción (ni venta ni compra)"


def test_only_the_new_pick_is_bought_overlap_untouched():
    held = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    current = pf.initial_portfolio(_universe(held, "2010-05-15"), "2010-05-15")

    # Aparece un nombre nuevo sano; hay hueco (6 < MAX). Debe comprarse SOLO él.
    universe_next = _universe(held + ["GGG"], "2010-06-15")
    updated, transactions = pf.review_portfolio(current, universe_next, "2010-06-15")

    assert "GGG" in updated
    actions = [(tx["ticker"], tx["action"]) for tx in transactions]
    assert actions == [("GGG", "BUY")], "solo el nombre nuevo se opera; el solapamiento se mantiene"


def test_transaction_cost_charges_only_real_trades():
    # Una sola compra (nombre nuevo) en la fecha; los nombres mantenidos no emiten transacción.
    transactions = pd.DataFrame([
        {"date": pd.Timestamp("2010-06-15"), "ticker": "GGG", "action": "BUY"},
    ])
    current_weights = {"AAA": 0.2, "BBB": 0.2, "CCC": 0.2, "DDD": 0.2, "GGG": 0.2}
    cost = period_transaction_cost(
        transactions, pd.Timestamp("2010-06-15"), holding_count=5, cost_rate=0.0015,
        current_weights=current_weights, previous_weights={},
    )
    # Solo se cobra el peso realmente operado (GGG), no la cartera entera.
    assert cost == pytest.approx(0.2 * 0.0015)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
