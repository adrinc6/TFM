from __future__ import annotations

import pandas as pd
import pytest

from module.modeling.agents import _local_contribution_rows
from module.ui.dashboard import normalize_index, stock_series


def _panel() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["AAA", "BBB", "AAA", "BBB"],
        "snapshot_date": ["2020-01-15", "2020-01-15", "2020-02-15", "2020-02-15"],
        "in_sp500": [True, True, True, True],
        "pe": [10.0, 20.0, 12.0, None],
        "price": [100.0, 80.0, 110.0, 90.0],
        "eps": [2.0, 1.0, -2.2, 0.0],
    })


def test_stock_series_uses_point_in_time_universe_mean_excluding_nulls() -> None:
    result = stock_series(_panel(), "AAA", "pe")
    assert [point["sp500_mean"] for point in result["points"]] == [15.0, 12.0]
    assert [point["sp500_observations"] for point in result["points"]] == [2, 1]
    assert [point["price_index"] for point in result["points"]] == pytest.approx([100.0, 110.0])


def test_companion_crossing_zero_is_not_normalized() -> None:
    result = stock_series(_panel(), "AAA", "pe")
    assert result["companion"] == "eps"
    assert not result["companion_indexed"]
    assert [point["companion_display"] for point in result["points"]] == [2.0, -2.2]


def test_normalize_index_requires_positive_series() -> None:
    assert normalize_index([10.0, 15.0, None]) == ([100.0, 150.0, None], True)
    assert normalize_index([10.0, -2.0]) == ([10.0, -2.0], False)


class _Booster:
    def predict(self, frame, pred_contrib=False):
        assert pred_contrib
        return [[0.2, -0.7, 0.1]]


class _Model:
    booster_ = _Booster()


def test_local_attribution_keeps_top_five_and_model_date() -> None:
    scoring = pd.DataFrame({"ticker": ["AAA"], "snapshot_date": ["2020-01-15"], "a": [0.3], "b": [0.8]})
    rows = _local_contribution_rows(_Model(), scoring, ["a", "b"], "quality", pd.Timestamp("2020-01-15"))
    assert [row["feature"] for row in rows] == ["b", "a"]
    assert rows[0]["direction"] == "negative"
    assert rows[0]["model_retrain_date"] == "2020-01-15"
