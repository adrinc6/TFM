from __future__ import annotations

import pandas as pd

from module.data.dataset import build_point_in_time_dataset, snapshot_dates


def _row(panel, ticker: str, date: str):
    return panel.loc[(panel["ticker"] == ticker) & (panel["snapshot_date"] == date)].iloc[0]


def _snapshot_after(settings, after: str) -> str:
    """Primer snapshot de la rejilla en/tras `after` (la rejilla cae en fin_de_mes + lag)."""
    grid = snapshot_dates(settings)
    return next(d.date().isoformat() for d in grid if d >= pd.Timestamp(after))


def test_panel_uses_each_company_last_published_fundamental(dataset_settings) -> None:
    panel = build_point_in_time_dataset(dataset_settings)

    # Snapshot tras el 10-K de AAA (filed 2000-02-01) pero antes del 10-K de BBB (filed 2000-03-10),
    # y otro tras el 10-K de BBB.
    feb = _snapshot_after(dataset_settings, "2000-02-05")
    mar = _snapshot_after(dataset_settings, "2000-03-11")
    aaa_february = _row(panel, "AAA", feb)
    bbb_february = _row(panel, "BBB", feb)
    bbb_march = _row(panel, "BBB", mar)

    assert aaa_february["fundamental_period"] == "1999-12-31"
    assert bbb_february["fundamental_period"] == "1999-09-30"
    assert bbb_march["fundamental_period"] == "1999-12-31"
    assert pd.to_datetime(panel["fundamental_filed_date"].dropna()).le(
        pd.to_datetime(panel.loc[panel["fundamental_filed_date"].notna(), "snapshot_date"])
    ).all()
    assert panel["fundamental_age_days"].dropna().ge(0).all()

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
