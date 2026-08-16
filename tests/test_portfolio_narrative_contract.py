"""Contrato de la narrativa de cartera: es material del TFM, no una vista bonita.

Las tablas que salen de aquí van al manuscrito —los nombres más presentes, quién aportó y quién
restó, las mejores y las peores decisiones—, así que sus modos de fallo son afirmaciones falsas en
un capítulo. Tres importan por encima del resto:

1. Que la contribución por acción no cuadre con el retorno bruto de la cartera. Entonces la suma de
   las partes no da el todo y cualquier «esta acción aportó tanto» es una estimación disfrazada.
2. Que un recorte de rebalanceo se cuente como operación cerrada. Ensuciaría a la vez los aciertos y
   los errores con posiciones que en realidad siguieron abiertas.
3. Que la salvedad del sector se pierda. El sector no es point-in-time y una figura sectorial sin esa
   advertencia se lee como si lo fuera.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from module.evaluation.backtest import run_backtest
from module.research.portfolio_narrative import (
    _episodes,
    _full_exits,
    holding_duration,
    holdings_table,
    round_trips,
    sector_exposure,
)
from tests.test_backtest_contract import _panel, _settings


def _rotating_result(months: int = 60):
    """Cartera que rota de verdad: con rangos fijos nunca habría operaciones cerradas que medir."""
    scores, prices, benchmark = _panel(months=months, tickers=tuple("ABCDEFGH"))
    rng = np.random.default_rng(3)
    scores = scores.copy()
    scores["meta_rank"] = rng.random(len(scores))
    scores["meta_score"] = scores["meta_rank"]
    scores["expected_excess_return"] = scores["meta_rank"] * 0.12 - 0.03
    settings = _settings(commission_bps=5.0, slippage_bps=10.0, target_size=3,
                         exit_expected_alpha_bps=100.0, max_cash_weight=0.2)
    return run_backtest(scores, prices, benchmark, settings), prices, benchmark


def test_contributions_by_ticker_reproduce_the_gross_return_of_the_portfolio() -> None:
    """La suma de las partes da el todo: sin esto la atribución por acción no es contabilidad."""
    result, _, _ = _rotating_result()
    table = holdings_table(result.positions, result.contributions, result.orders, 1)
    assert table["gross_contribution"].sum() == pytest.approx(
        result.equity["gross_return"].sum(), rel=0, abs=1e-9,
    )


def test_imputed_costs_add_up_to_what_the_portfolio_actually_paid() -> None:
    """El coste imputado por acción no puede inventar ni perder fricción por el camino."""
    result, _, _ = _rotating_result()
    table = holdings_table(result.positions, result.contributions, result.orders, 1)
    assert table["cost_contribution"].sum() == pytest.approx(
        result.equity["cost_drag"].sum(), rel=1e-6,
    )


def test_net_contribution_is_gross_minus_the_cost_of_holding_it() -> None:
    result, _, _ = _rotating_result()
    table = holdings_table(result.positions, result.contributions, result.orders, 1)
    assert np.allclose(
        table["net_contribution"], table["gross_contribution"] - table["cost_contribution"],
    )


def test_a_rebalance_trim_is_not_a_closed_position() -> None:
    """Solo cuenta como operación cerrada la venta que deja el peso a cero."""
    orders = pd.DataFrame([
        {"ticker": "A", "snapshot_date": "2020-02-29", "side": "sell", "weight_before": 0.30,
         "weight_after": 0.20, "realized_pnl_pct": 0.05, "reason": "rebalance"},
        {"ticker": "B", "snapshot_date": "2020-02-29", "side": "sell", "weight_before": 0.30,
         "weight_after": 0.0, "realized_pnl_pct": -0.08, "reason": "displaced_by_net_edge"},
        {"ticker": "C", "snapshot_date": "2020-02-29", "side": "buy", "weight_before": 0.0,
         "weight_after": 0.30, "realized_pnl_pct": None, "reason": "initial_fill"},
    ])
    exits = _full_exits(orders)
    assert exits["ticker"].tolist() == ["B"]
    trips = round_trips(orders, pd.DataFrame())
    assert trips["ticker"].tolist() == ["B"]


def test_closed_positions_carry_the_time_they_were_actually_held() -> None:
    """La permanencia se toma del último snapshot en que seguía en cartera, no del día de la venta."""
    positions = pd.DataFrame([
        {"ticker": "A", "snapshot_date": "2020-01-31", "entry_date": "2020-01-31", "months_held": 0},
        {"ticker": "A", "snapshot_date": "2020-02-29", "entry_date": "2020-01-31", "months_held": 1},
    ])
    orders = pd.DataFrame([
        {"ticker": "A", "snapshot_date": "2020-03-31", "side": "sell", "weight_before": 0.3,
         "weight_after": 0.0, "realized_pnl_pct": 0.12, "reason": "expected_alpha_below_exit"},
    ])
    trips = round_trips(orders, positions)
    assert len(trips) == 1
    assert trips.loc[0, "months_held"] == 1
    assert trips.loc[0, "entry_date"] == "2020-01-31"


def test_every_closed_position_of_a_real_backtest_knows_how_long_it_was_held() -> None:
    """Sobre una cartera real, ninguna operación cerrada puede quedarse sin permanencia."""
    result, _, _ = _rotating_result()
    trips = round_trips(result.orders, result.positions)
    assert not trips.empty
    assert trips["months_held"].notna().all()
    assert trips["entry_date"].notna().all()


def test_presence_counts_snapshots_and_episodes_separately() -> None:
    """Estar 30 snapshots en 17 entradas distintas no es lo mismo que estar 30 seguidos."""
    snapshots = [f"2020-{month:02d}-01" for month in range(1, 13)]
    assert _episodes(["2020-01-01", "2020-02-01", "2020-03-01"], snapshots) == 1
    assert _episodes(["2020-01-01", "2020-05-01", "2020-06-01"], snapshots) == 2
    assert _episodes(["2020-01-01", "2020-03-01", "2020-05-01"], snapshots) == 3


def test_holding_duration_counts_one_episode_per_entry_not_per_snapshot() -> None:
    positions = pd.DataFrame([
        {"ticker": "A", "snapshot_date": "2020-01-31", "entry_date": "2020-01-31", "months_held": 0},
        {"ticker": "A", "snapshot_date": "2020-02-29", "entry_date": "2020-01-31", "months_held": 1},
        {"ticker": "A", "snapshot_date": "2020-06-30", "entry_date": "2020-06-30", "months_held": 0},
    ])
    duration = holding_duration(positions, 1)
    assert duration["episodes"] == 2
    assert duration["max_months"] == 1


def test_sector_exposure_always_declares_that_the_sector_is_not_point_in_time() -> None:
    """La salvedad viaja pegada al dato, disponible o no: sin ella la figura se malinterpreta."""
    positions = pd.DataFrame([
        {"ticker": "A", "snapshot_date": "2020-01-31", "weight": 0.5},
        {"ticker": "B", "snapshot_date": "2020-01-31", "weight": 0.5},
    ])
    block = sector_exposure(positions, {"A": "Tech", "B": "Energy"})
    assert block["available"] is True
    assert block["sector_is_point_in_time"] is False
    assert "no PIT" in block["sector_source"]
    assert {row["sector"] for row in block["sectors"]} == {"Tech", "Energy"}

    without_map = sector_exposure(positions, {})
    assert without_map["available"] is False
    assert without_map["sector_is_point_in_time"] is False
