"""Contrato del volumen negociado del panel: hacia atrás y nada más que hacia atrás.

La columna existe para dimensionar capacidad, y su modo de fallo es el mismo que el del resto del
panel: mirar al futuro. Un volumen posterior al snapshot que se colara en la mediana haría que la
capacidad estimada dependiera de información que ese día no existía —justo la clase de fuga que todo
el proyecto está construido para impedir— y encima lo haría en la dirección cómoda, porque las
acciones que después se vuelven líquidas parecerían operables antes de serlo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from module.data.dataset import (
    ASSET_PRICE_COLUMNS,
    DOLLAR_VOLUME_SESSIONS,
    _asset_price_frame,
    _dollar_volume_index,
    _median_dollar_volume,
)


def _prices(sessions: int = 60, volume: float = 1_000.0, price: float = 10.0) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=sessions)
    return pd.DataFrame({
        "ticker": ["A"] * sessions,
        "date": dates,
        "adj_close": [price] * sessions,
        "volume": [volume] * sessions,
    })


def test_the_column_is_part_of_the_artifact_contract() -> None:
    assert "median_dollar_volume_21d" in ASSET_PRICE_COLUMNS


def test_a_later_session_can_never_change_the_value_at_a_snapshot() -> None:
    """La prueba directa de ausencia de fuga: el futuro se altera y el pasado no se mueve."""
    prices = _prices()
    snapshot = pd.Timestamp("2020-02-14")
    index = _dollar_volume_index(prices)
    before = _median_dollar_volume(*index["A"], snapshot)

    future = prices.copy()
    mask = future["date"].gt(snapshot)
    assert mask.any(), "el escenario necesita sesiones posteriores que alterar"
    future.loc[mask, "volume"] = 10_000_000.0
    after = _median_dollar_volume(*_dollar_volume_index(future)["A"], snapshot)

    assert before == after == pytest.approx(10.0 * 1_000.0)


def test_only_the_declared_window_of_sessions_enters_the_median() -> None:
    """Una sesión más antigua que la ventana no puede influir, por extrema que sea."""
    prices = _prices(sessions=60)
    prices.loc[0, "volume"] = 1e12
    snapshot = pd.Timestamp("2020-03-20")
    index = _dollar_volume_index(prices)
    dates, values = index["A"]
    within = sum(1 for date in dates if date <= snapshot)
    assert within > DOLLAR_VOLUME_SESSIONS, "la sesión extrema debe quedar fuera de la ventana"
    assert _median_dollar_volume(dates, values, snapshot) == pytest.approx(10.0 * 1_000.0)


def test_the_median_ignores_a_single_extraordinary_session() -> None:
    """Mediana y no media: un día de volumen anómalo no puede inflar la capacidad estimada."""
    prices = _prices(sessions=30)
    prices.loc[25, "volume"] = 1e9
    dates, values = _dollar_volume_index(prices)["A"]
    snapshot = dates[-1]
    assert _median_dollar_volume(dates, values, snapshot) == pytest.approx(10.0 * 1_000.0)


def test_a_snapshot_before_any_session_has_no_value() -> None:
    """Sin sesiones previas no hay volumen que declarar, y se dice con un nulo, no con un cero."""
    dates, values = _dollar_volume_index(_prices())["A"]
    assert _median_dollar_volume(dates, values, pd.Timestamp("2019-01-01")) is None


def test_ingest_without_volume_leaves_the_column_null_not_zero() -> None:
    """Un cero se leería como «sin liquidez» y un ausente como «no medido». Solo lo segundo es cierto."""
    prices = _prices().drop(columns="volume")
    assert _dollar_volume_index(prices) == {}
    frame = _asset_price_frame(
        {"A": (list(prices["date"]), list(prices["adj_close"]))},
        [pd.Timestamp("2020-02-14")], "SPY", {},
    )
    assert frame["median_dollar_volume_21d"].isna().all()


def test_the_frame_publishes_the_volume_observed_at_each_snapshot() -> None:
    """Camino completo: del raw diario a la columna del artefacto de precios."""
    prices = _prices(sessions=40, volume=2_000.0, price=25.0)
    snapshots = [pd.Timestamp("2020-02-14"), pd.Timestamp("2020-02-21")]
    frame = _asset_price_frame(
        {"A": (list(prices["date"]), list(prices["adj_close"]))},
        snapshots, "SPY", _dollar_volume_index(prices),
    )
    assert len(frame) == 2
    assert np.allclose(frame["median_dollar_volume_21d"], 25.0 * 2_000.0)


def test_the_benchmark_is_excluded_as_it_already_was() -> None:
    prices = _prices()
    frame = _asset_price_frame(
        {"SPY": (list(prices["date"]), list(prices["adj_close"]))},
        [pd.Timestamp("2020-02-14")], "SPY", _dollar_volume_index(prices),
    )
    assert frame.empty
