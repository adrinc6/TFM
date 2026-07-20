"""Artefactos activables: correctos y sin lookahead (point-in-time)."""

from __future__ import annotations

import pandas as pd

from module.modeling.artifacts import (
    add_moving_averages,
    add_price_momentum_multi,
    add_quality_growth_derived,
    add_regime_extended,
)


def test_price_momentum_multi_from_panel_returns() -> None:
    f = pd.DataFrame({
        "price_return_1m": [0.05, -0.02],
        "price_return_3m": [0.10, 0.01],
        "price_return_6m": [0.15, 0.00],
        "price_return_12m": [0.20, -0.10],
    })
    add_price_momentum_multi(f)
    assert round(f["mom_acceleration"].iloc[0], 3) == -0.10   # 0.10 - 0.20
    assert round(f["mom_reversal_1m"].iloc[0], 3) == -0.05    # -(0.05)
    assert f["mom_volatility"].iloc[0] > 0


def test_moving_averages_are_point_in_time() -> None:
    """Precio vs SMA usa solo precios hasta la fecha; mutar el futuro no cambia el pasado."""
    dates = [f"2000-{m:02d}-15" for m in range(1, 13)]
    prices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    price_series = {"AAA": (dates, prices)}
    frame = pd.DataFrame({"ticker": ["AAA"] * 12, "snapshot_date": dates})
    add_moving_averages(frame, price_series)
    base = frame["ma_price_vs_sma6"].copy()

    # Mutar precios FUTUROS (ultimos 3 meses) no debe cambiar las filas anteriores.
    prices_mut = prices[:9] + [999, 999, 999]
    frame2 = pd.DataFrame({"ticker": ["AAA"] * 12, "snapshot_date": dates})
    add_moving_averages(frame2, {"AAA": (dates, prices_mut)})
    pd.testing.assert_series_equal(base.iloc[:9], frame2["ma_price_vs_sma6"].iloc[:9], check_names=False)
    # En una serie creciente, el precio esta por encima de su SMA (tendencia alcista).
    assert frame["ma_price_vs_sma6"].iloc[-1] > 0


def test_regime_extended_maps_by_date_without_lookahead() -> None:
    bench = pd.DataFrame({
        "snapshot_date": ["2000-01-15", "2000-02-15", "2000-03-15", "2000-04-15"],
        "price": [100.0, 110.0, 99.0, 105.0],
    })
    frame = pd.DataFrame({
        "ticker": ["AAA", "BBB", "AAA", "AAA"],
        "snapshot_date": ["2000-02-15", "2000-02-15", "2000-03-15", "2000-04-15"],
    })
    add_regime_extended(frame, bench)
    # drawdown en 2000-03 (precio 99 desde maximo 110) es negativo.
    dd_march = frame.loc[frame["snapshot_date"] == "2000-03-15", "regime_sp500_drawdown"].iloc[0]
    assert dd_march < 0
    # mismas fechas -> mismo regimen para todos los tickers.
    feb = frame.loc[frame["snapshot_date"] == "2000-02-15", "regime_sp500_drawdown"]
    assert feb.nunique() == 1


def test_quality_growth_uses_only_past_of_same_ticker() -> None:
    """qg_roe_trend compara con la media PASADA del mismo ticker; el futuro no afecta al pasado."""
    frame = pd.DataFrame({
        "ticker": ["AAA"] * 5,
        "snapshot_date": [f"2000-{m:02d}-15" for m in range(1, 6)],
        "roe": [0.10, 0.11, 0.12, 0.13, 0.14],
        "eps_growth_yoy": [0.05] * 5,
        "net_margin": [0.08, 0.09, 0.08, 0.09, 0.08],
    })
    add_quality_growth_derived(frame)
    base_trend = frame["qg_roe_trend"].copy()

    mutated = frame.copy()
    mutated.loc[3:, "roe"] = 0.99          # cambia el futuro
    add_quality_growth_derived(mutated)
    # Las filas <= indice 2 no deben cambiar (usan shift + rolling del pasado).
    pd.testing.assert_series_equal(
        base_trend.iloc[:3], mutated["qg_roe_trend"].iloc[:3], check_names=False
    )
