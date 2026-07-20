"""Invariantes de simulación: point-in-time y contabilidad de costes."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from module.evaluation.backtest import run_backtest


def test_mutating_future_scores_does_not_change_past_positions(
    portfolio_settings, synthetic_scores, synthetic_prices, synthetic_benchmark
) -> None:
    """La invariante central del proyecto, aplicada al backtest.

    Si borramos o mutamos scores posteriores a T, las posiciones y equity hasta T deben ser
    idénticos. Cazamos fugas sin saber por dónde entran.
    """
    settings = replace(
        portfolio_settings,
        target_size=5,
        min_hold_percentile=50,
        rotation_edge_percentiles=5,
        commission_bps=5,
        slippage_bps=10,
    )

    first = run_backtest(synthetic_scores, synthetic_prices, synthetic_benchmark, settings)

    cutoff = "2000-02-15"
    mutated_scores = synthetic_scores.copy()
    future_mask = mutated_scores["snapshot_date"] > cutoff
    mutated_scores.loc[future_mask, "meta_rank"] = 0.99
    mutated_scores.loc[future_mask, "meta_score"] = 0.99

    second = run_backtest(mutated_scores, synthetic_prices, synthetic_benchmark, settings)

    first_past = first.positions.loc[first.positions["snapshot_date"] <= cutoff]
    second_past = second.positions.loc[second.positions["snapshot_date"] <= cutoff]
    pd.testing.assert_frame_equal(
        first_past.reset_index(drop=True), second_past.reset_index(drop=True)
    )

    first_equity_past = first.equity.loc[first.equity["snapshot_date"] <= cutoff]
    second_equity_past = second.equity.loc[second.equity["snapshot_date"] <= cutoff]
    pd.testing.assert_frame_equal(
        first_equity_past.reset_index(drop=True), second_equity_past.reset_index(drop=True)
    )


def test_equity_delta_matches_positions_return_minus_costs(
    portfolio_settings, synthetic_scores, synthetic_prices, synthetic_benchmark
) -> None:
    """Contabilidad: sin ganancias ni pérdidas fantasma.

    `equity[t] - equity[t-1]` = suma(peso × retorno) − comisiones − slippage del día. Si no
    cuadra, hay dinero apareciendo o desapareciendo del backtest.
    """
    settings = replace(
        portfolio_settings,
        target_size=5,
        min_hold_percentile=50,
        rotation_edge_percentiles=5,
        commission_bps=5,
        slippage_bps=10,
    )
    result = run_backtest(synthetic_scores, synthetic_prices, synthetic_benchmark, settings)

    equity = result.equity.sort_values("snapshot_date").reset_index(drop=True)
    orders = result.orders

    for index in range(1, len(equity)):
        row = equity.iloc[index]
        previous_value = equity.iloc[index - 1]["portfolio_value"]
        current_value = row["portfolio_value"]
        realised_delta = (current_value - previous_value) / previous_value

        # Costes del día (comisión + slippage sobre el nocional operado)
        day_orders = orders.loc[orders["snapshot_date"] == row["snapshot_date"]]
        cost_drag = -(day_orders["commission"].sum() + day_orders["slippage"].sum()) / previous_value
        gross_return = row["portfolio_return"] - cost_drag

        # portfolio_return YA está neto de costes en el output; verificamos que la delta cuadra:
        assert realised_delta == pytest.approx(row["portfolio_return"], abs=1e-6), (
            f"desajuste en {row['snapshot_date']}: delta={realised_delta} vs "
            f"portfolio_return={row['portfolio_return']}"
        )
        # Y que gross_return - drag reconstruye lo mismo (evita que costes esten mal contados)
        assert row["portfolio_return"] == pytest.approx(gross_return + cost_drag, abs=1e-9)
