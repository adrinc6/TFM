from __future__ import annotations

import pandas as pd

from module.data.dataset import build_point_in_time_dataset


def _row(panel, ticker: str, date: str):
    return panel.loc[(panel["ticker"] == ticker) & (panel["snapshot_date"] == date)].iloc[0]


def test_panel_uses_each_company_last_published_fundamental(dataset_settings) -> None:
    panel = build_point_in_time_dataset(dataset_settings)

    aaa_february = _row(panel, "AAA", "2000-02-15")
    bbb_february = _row(panel, "BBB", "2000-02-15")
    bbb_march = _row(panel, "BBB", "2000-03-15")

    assert aaa_february["fundamental_period"] == "1999-12-31"
    assert bbb_february["fundamental_period"] == "1999-09-30"
    assert bbb_march["fundamental_period"] == "1999-12-31"
    assert pd.to_datetime(panel["fundamental_filed_date"].dropna()).le(
        pd.to_datetime(panel.loc[panel["fundamental_filed_date"].notna(), "snapshot_date"])
    ).all()
    assert panel["fundamental_age_days"].dropna().ge(0).all()


def test_fundamentals_remain_frozen_until_next_filing_and_ignore_metric_snapshot(dataset_settings) -> None:
    panel = build_point_in_time_dataset(dataset_settings)
    aaa_february = _row(panel, "AAA", "2000-02-15")
    aaa_march = _row(panel, "AAA", "2000-03-15")

    assert aaa_february["roe"] == aaa_march["roe"] == 0.5
    assert aaa_february["roe"] != 999.0
    assert aaa_february["price"] != aaa_march["price"]
    assert aaa_february["price_return_1m"] is not None


def test_removing_future_filing_does_not_change_past_rows(dataset_settings) -> None:
    panel_with_future = build_point_in_time_dataset(dataset_settings)
    reports_path = dataset_settings.raw_output_dir / "report_dates.parquet"
    reports = pd.read_parquet(reports_path)
    reports.loc[reports["filed_date"] <= "2000-03-15"].to_parquet(reports_path, index=False)

    panel_without_future = build_point_in_time_dataset(dataset_settings)
    before_future = panel_with_future.loc[panel_with_future["snapshot_date"] <= "2000-03-15"]
    rebuilt_before_future = panel_without_future.loc[panel_without_future["snapshot_date"] <= "2000-03-15"]

    pd.testing.assert_frame_equal(
        before_future.reset_index(drop=True), rebuilt_before_future.reset_index(drop=True)
    )


def test_panel_contract_excludes_current_profiles_and_sector(dataset_settings) -> None:
    panel = build_point_in_time_dataset(dataset_settings)

    forbidden = {"sector", "marketCapitalization", "finnhubIndustry", "payload", "metric"}
    assert not (forbidden & set(panel.columns))
    assert panel["in_sp500"].all()


def test_dataset_writes_benchmark_and_price_traceability(dataset_settings) -> None:
    panel = build_point_in_time_dataset(dataset_settings)
    benchmark = pd.read_parquet(dataset_settings.processed_output_dir / "benchmark_point_in_time.parquet")
    asset_prices = pd.read_parquet(dataset_settings.processed_output_dir / "asset_price_point_in_time.parquet")

    assert {"price_as_of_date", "price_age_days"} <= set(panel.columns)
    assert not benchmark.empty
    assert benchmark["price_age_days"].ge(0).all()
    assert {"AAA", "BBB"} <= set(asset_prices["ticker"])
    assert "SPY" not in set(asset_prices["ticker"])
