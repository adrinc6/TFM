from __future__ import annotations

import pandas as pd

from module.ui.dashboard import stock_series


def test_stock_series_uses_point_in_time_universe_mean_excluding_nulls() -> None:
    panel = pd.DataFrame({
        "ticker": ["AAA", "BBB", "CCC"], "snapshot_date": ["2000-01-15"] * 3,
        "in_sp500": [True, True, False], "roe": [0.1, 0.3, 9.0], "price": [100.0, 100.0, 100.0],
    })

    result = stock_series(panel, "AAA", "roe")

    assert result["points"][0]["value"] == 0.1
    assert result["points"][0]["sp500_mean"] == 0.2
