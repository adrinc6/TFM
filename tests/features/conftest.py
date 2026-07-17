from __future__ import annotations

import pandas as pd
import pytest

import environment
from environment import Settings


@pytest.fixture
def feature_settings(monkeypatch, tmp_path) -> Settings:
    processed = tmp_path / "processed" / "dev"
    monkeypatch.setattr(environment, "DEV_PROCESSED_DIR", processed)
    dates = pd.date_range("2000-01-15", "2000-08-15", freq="MS") + pd.Timedelta(days=14)
    tickers = ("AAA", "BBB", "CCC")
    panel_rows = []
    asset_rows = []
    for date_index, date in enumerate(dates):
        date_text = date.date().isoformat()
        for ticker_index, ticker in enumerate(tickers):
            price = 10.0 + ticker_index * 5 + date_index
            panel_rows.append(
                {
                    "ticker": ticker,
                    "snapshot_date": date_text,
                    "review_type": "fundamental_quarterly" if date.month in (2, 5, 8, 11) else "price_monthly",
                    "in_sp500": True,
                    "price": price,
                    "price_as_of_date": date_text,
                    "price_age_days": 8 if ticker == "CCC" and date_index == 1 else 0,
                    "price_return_1m": 0.01 * (ticker_index + 1),
                    "price_return_3m": 0.03 * (ticker_index + 1),
                    "price_return_6m": 0.06 * (ticker_index + 1),
                    "price_return_12m": 0.12 * (ticker_index + 1),
                    "roe": 0.1 * (ticker_index + 1),
                    "roic": 0.08 * (ticker_index + 1),
                    "net_margin": 0.05 * (ticker_index + 1),
                    "operating_margin": 0.06 * (ticker_index + 1),
                    "gross_margin": 0.2 * (ticker_index + 1),
                    "fcf_margin": 0.03 * (ticker_index + 1),
                    "pe": 30 - ticker_index * 5,
                    "pb": 5 - ticker_index,
                    "ps": 4 - ticker_index * 0.5,
                    "ev_ebitda": 20 - ticker_index * 3,
                    "debt_equity": 2 - ticker_index * 0.5,
                    "current_ratio": 1 + ticker_index * 0.2,
                    "eps_growth_yoy": 0.05 * (ticker_index + 1),
                    "sales_per_share_growth_yoy": 0.04 * (ticker_index + 1),
                    "fundamental_period": "1999-12-31",
                    "fundamental_filed_date": "2000-02-01",
                    "fundamental_age_days": 14,
                }
            )
            asset_rows.append(
                {
                    "ticker": ticker,
                    "snapshot_date": date_text,
                    "price": price,
                    "price_as_of_date": date_text,
                    "price_age_days": 0,
                }
            )
    benchmark = pd.DataFrame(
        {
            "snapshot_date": [date.date().isoformat() for date in dates],
            "price": [100 + index for index in range(len(dates))],
            "price_as_of_date": [date.date().isoformat() for date in dates],
            "price_age_days": [0] * len(dates),
            "price_return_1m": [0.01] * len(dates),
            "price_return_3m": [0.02] * len(dates),
            "price_return_6m": [0.03] * len(dates),
            "price_return_12m": [0.04] * len(dates),
        }
    )
    processed.mkdir(parents=True)
    pd.DataFrame(panel_rows).to_parquet(processed / "panel_point_in_time.parquet", index=False)
    pd.DataFrame(asset_rows).to_parquet(processed / "asset_price_point_in_time.parquet", index=False)
    benchmark.to_parquet(processed / "benchmark_point_in_time.parquet", index=False)
    return Settings(run_scope="dev", data_start_date="2000-01-01", end_date="2000-08-15")
