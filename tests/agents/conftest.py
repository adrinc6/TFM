from __future__ import annotations

import pandas as pd
import pytest

import environment
from environment import Settings


@pytest.fixture
def agent_settings(monkeypatch, tmp_path) -> Settings:
    processed = tmp_path / "processed" / "dev"
    monkeypatch.setattr(environment, "DEV_PROCESSED_DIR", processed)
    dates = pd.date_range("1996-02-01", "2001-12-01", freq="MS") + pd.Timedelta(days=14)
    tickers = [f"T{index:02d}" for index in range(12)]
    feature_rows = []
    target_rows = []
    for date_index, date in enumerate(dates):
        date_text = date.date().isoformat()
        quarterly = date.month in (2, 5, 8, 11)
        for ticker_index, ticker in enumerate(tickers):
            base = (ticker_index + 1) / len(tickers)
            feature_rows.append(
                {
                    "ticker": ticker,
                    "snapshot_date": date_text,
                    "review_type": "fundamental_quarterly" if quarterly else "price_monthly",
                    "is_price_fresh": True,
                    "factor_roe": base,
                    "factor_roic": base,
                    "factor_net_margin": base,
                    "factor_operating_margin": base,
                    "factor_gross_margin": base,
                    "factor_fcf_margin": base,
                    "factor_debt_equity": 1 - base,
                    "factor_current_ratio": base,
                    "factor_relative_return_3m": base,
                    "factor_relative_return_6m": base,
                    "factor_relative_return_12m": base,
                    "factor_pe": base,
                    "factor_pb": base,
                    "factor_ps": base,
                    "factor_ev_ebitda": base,
                }
            )
            label_end = date + pd.DateOffset(months=3)
            available = label_end <= dates[-1]
            target_rows.append(
                {
                    "ticker": ticker,
                    "snapshot_date": date_text,
                    "label_end_date": label_end.date().isoformat() if available else None,
                    "target_available": available,
                    "forward_excess_return": base + date_index / 10000 if available else None,
                }
            )
    processed.mkdir(parents=True)
    pd.DataFrame(feature_rows).to_parquet(processed / "features_point_in_time.parquet", index=False)
    pd.DataFrame(target_rows).to_parquet(processed / "targets_forward.parquet", index=False)
    return Settings(
        run_scope="dev",
        data_start_date="1996-01-01",
        end_date="2001-12-15",
        execution_year=2000,
        min_rank_ic_cross_section=5,
    )
