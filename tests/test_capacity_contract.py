"""Contrato de capacidad: el modo de fallo silencioso es sobreestimar cuánto dinero cabe.

Dos errores producirían una capacidad inflada sin que nada avise. El primero, tratar un ticker sin
volumen medido como si tuviera liquidez infinita: los huecos del panel subirían el límite
precisamente donde menos se sabe. El segundo, romper la linealidad entre patrimonio y participación,
que es lo que permite resolver el máximo en forma cerrada en vez de tantear la escalera.

Un límite de capacidad demasiado alto es peor que no publicarlo: convierte una limitación conocida
en una afirmación falsa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from module.research.capacity import (
    PARTICIPATION_THRESHOLDS,
    _maximum_aum,
    _orders_with_volume,
    _window_block,
    participation,
)
from module.studies.catalog import SELECTION_UNTIL_YEAR


def _frame(volume: float | None = 1e8, snapshots: int = 12) -> pd.DataFrame:
    """Órdenes sintéticas con su volumen, ya en la forma que produce `_orders_with_volume`."""
    dates = pd.date_range("2020-01-31", periods=snapshots, freq=pd.DateOffset(months=1))
    rows = []
    for stamp in dates:
        for ticker, fraction in (("A", 0.10), ("B", 0.05)):
            rows.append({
                "ticker": ticker, "snapshot_date": str(stamp.date()),
                "portfolio_fraction": fraction,
                "median_dollar_volume_21d": np.nan if volume is None else volume,
                "year": stamp.year,
            })
    return pd.DataFrame(rows)


def test_participation_scales_linearly_with_assets_under_management() -> None:
    """Duplicar el patrimonio duplica la participación: es lo que hace exacto el máximo."""
    frame = _frame()
    small = participation(frame, 1e7)
    large = participation(frame, 2e7)
    assert np.allclose(large.to_numpy(), small.to_numpy() * 2.0)


def test_the_maximum_is_higher_for_a_more_permissive_threshold() -> None:
    """Admitir el 10 % del volumen permite exactamente el doble de patrimonio que el 5 %."""
    frame = _frame()
    at_five = _maximum_aum(frame, 0.05)
    at_ten = _maximum_aum(frame, 0.10)
    assert at_ten == pytest.approx(at_five * 2.0)
    assert at_five > 0


def test_the_maximum_respects_the_threshold_it_declares() -> None:
    """Al patrimonio publicado la participación gobernante toca el umbral, no lo cruza."""
    frame = _frame()
    for threshold in PARTICIPATION_THRESHOLDS:
        limit = _maximum_aum(frame, threshold)
        governing = float(np.percentile(participation(frame, limit).dropna(), 95))
        assert governing == pytest.approx(threshold)


def test_missing_volume_is_reported_as_uncovered_never_as_infinite_liquidity() -> None:
    """Un hueco del panel no puede subir la capacidad: se declara y no se cuenta como ejecutable."""
    block = _window_block(_frame(volume=None), (1e6, 1e8))
    assert block["available"] is False
    assert block["volume_coverage"] == 0.0
    assert "Ninguna orden" in block["reason"]


def test_partial_volume_coverage_is_measured_and_published() -> None:
    """Con volumen a medias, el límite sale de lo medido y la cobertura se publica tal cual."""
    frame = _frame()
    frame.loc[frame["ticker"].eq("B"), "median_dollar_volume_21d"] = np.nan
    block = _window_block(frame, (1e6, 1e8))
    assert block["available"] is True
    assert block["volume_coverage"] == pytest.approx(0.5)
    assert block["orders_with_volume"] == len(frame) // 2


def test_orders_are_converted_to_a_fraction_of_the_portfolio_before_scaling() -> None:
    """`notional` viene en unidades de la cartera simulada; sin dividir, el patrimonio no escala."""
    orders = pd.DataFrame([
        {"ticker": "A", "snapshot_date": "2020-01-31", "notional": 25.0},
        {"ticker": "B", "snapshot_date": "2020-01-31", "notional": 10.0},
    ])
    equity = pd.DataFrame([
        {"snapshot_date": "2020-01-31", "period_start_portfolio_value": 250.0},
    ])
    prices = pd.DataFrame([
        {"ticker": "A", "snapshot_date": "2020-01-31", "median_dollar_volume_21d": 1e8},
        {"ticker": "B", "snapshot_date": "2020-01-31", "median_dollar_volume_21d": 1e8},
    ])
    frame = _orders_with_volume(orders, equity, prices)
    assert frame["portfolio_fraction"].tolist() == [0.1, 0.04]


def test_a_panel_without_the_volume_column_degrades_instead_of_crashing() -> None:
    """Un dataset anterior a la columna de volumen no debe tumbar el diagnóstico."""
    orders = pd.DataFrame([{"ticker": "A", "snapshot_date": "2020-01-31", "notional": 25.0}])
    equity = pd.DataFrame([{"snapshot_date": "2020-01-31", "period_start_portfolio_value": 250.0}])
    prices = pd.DataFrame([{"ticker": "A", "snapshot_date": "2020-01-31", "price": 10.0}])
    frame = _orders_with_volume(orders, equity, prices)
    assert frame["median_dollar_volume_21d"].isna().all()
    assert _window_block(frame, (1e6,))["available"] is False


def test_the_two_windows_never_share_orders() -> None:
    """La era reservada se mide aparte, igual que en el resto del proyecto."""
    from module.research.capacity import _window_slice

    frame = pd.concat([_frame(snapshots=6), _frame(snapshots=6).assign(year=2025)])
    selection = _window_slice(frame, "selection")
    confirmation = _window_slice(frame, "confirmation")
    assert (selection["year"] <= SELECTION_UNTIL_YEAR).all()
    assert (confirmation["year"] == 2025).all()
    assert len(selection) + len(confirmation) == len(frame)
