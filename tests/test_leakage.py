"""Walk-forward leakage guard — the central methodological invariant of the project.

For every snapshot date `t`, a component target may only be used for training if its
`horizon_months`-ahead outcome was already observable at `t`, i.e.
`snapshot_date + horizon <= t`. Any row whose future is not yet observable at `t` must have its
label masked (set to NA) before fitting. This test reproduces that rule and asserts no future
label survives into a training window.

This is a pure-logic test: it does not need downloaded data or a trained model, so it runs in CI
without API keys.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from environment import Settings


HORIZON_MONTHS = 12
COMPONENT_TARGETS = [
    "target_future_alpha",
    "target_quality",
    "target_timing",
    "target_alpha_rank",
]


def _synthetic_universe(n_snapshots: int = 40, tickers: int = 15) -> pd.DataFrame:
    dates = pd.date_range("2015-01-31", periods=n_snapshots, freq="ME")
    rows = []
    for date in dates:
        for i in range(tickers):
            rows.append(
                {
                    "ticker": f"T{i:02d}",
                    "snapshot_date": date.date().isoformat(),
                    "snapshot_date_dt": date,
                    # forward labels are "known" for every row in this synthetic frame; the mask
                    # is what must remove the not-yet-observable ones.
                    **{target: 0.5 for target in COMPONENT_TARGETS},
                }
            )
    return pd.DataFrame(rows)


def _masked_training_window(scored: pd.DataFrame, date: pd.Timestamp, max_history_years: int = 8):
    """Reproduce module/ml.py::_walk_forward_component_scores masking for a single snapshot."""
    label_offset = pd.DateOffset(months=HORIZON_MONTHS)
    max_history = pd.DateOffset(years=max_history_years)
    train_mask = (
        (scored["snapshot_date_dt"] >= (pd.Timestamp(date) - max_history))
        & (scored["snapshot_date_dt"] <= pd.Timestamp(date))
    )
    train = scored[train_mask].copy()
    observable_mask = train["snapshot_date_dt"] + label_offset <= pd.Timestamp(date)
    for target in COMPONENT_TARGETS:
        train.loc[~observable_mask, target] = pd.NA
    return train, observable_mask


def test_no_future_label_survives_masking():
    scored = _synthetic_universe()
    dates = sorted(scored["snapshot_date_dt"].unique())
    for date in dates:
        train, observable_mask = _masked_training_window(scored, pd.Timestamp(date))
        # Every row that KEPT a non-null label must be observable at `date`.
        labeled = train[COMPONENT_TARGETS].notna().any(axis=1)
        leaked = train[labeled & ~observable_mask.reindex(train.index, fill_value=False)]
        assert leaked.empty, (
            f"Leakage at {pd.Timestamp(date).date()}: {len(leaked)} rows kept a forward label "
            f"whose {HORIZON_MONTHS}m outcome is not observable yet."
        )


def test_observable_rows_are_strictly_in_the_past_by_horizon():
    scored = _synthetic_universe()
    dates = sorted(scored["snapshot_date_dt"].unique())
    label_offset = pd.DateOffset(months=HORIZON_MONTHS)
    for date in dates:
        train, observable_mask = _masked_training_window(scored, pd.Timestamp(date))
        observable = train[observable_mask]
        if observable.empty:
            continue
        # The most recent observable snapshot must be at least `horizon` before `date`.
        latest_observable = observable["snapshot_date_dt"].max()
        assert latest_observable + label_offset <= pd.Timestamp(date)


def test_early_snapshots_have_no_observable_labels():
    """The first `horizon` months cannot have any observable forward label."""
    scored = _synthetic_universe()
    first_date = pd.Timestamp(sorted(scored["snapshot_date_dt"].unique())[0])
    _, observable_mask = _masked_training_window(scored, first_date)
    assert not observable_mask.any()


def test_mutating_a_future_row_does_not_change_the_masked_training_window():
    """Direct invariance check: change a value that only affects a not-yet-observable row, and the
    masked training window for an earlier snapshot must be byte-for-byte identical."""
    scored = _synthetic_universe()
    dates = sorted(scored["snapshot_date_dt"].unique())
    probe_date = pd.Timestamp(dates[len(dates) // 2])

    baseline, _ = _masked_training_window(scored, probe_date)

    mutated = scored.copy()
    future_rows = mutated["snapshot_date_dt"] > probe_date
    assert future_rows.any(), "test setup needs at least one future row"
    for target in COMPONENT_TARGETS:
        mutated.loc[future_rows, target] = 0.999

    mutated_window, _ = _masked_training_window(mutated, probe_date)
    pd.testing.assert_frame_equal(
        baseline.reset_index(drop=True), mutated_window.reset_index(drop=True),
        check_dtype=False,
    )


@pytest.fixture
def quarterly_master_dataset(tmp_path, monkeypatch):
    """A tiny synthetic raw dataset with a fundamental fixed for a whole quarter and price moving
    every month, run through the real `module.dataset.build_master_dataset` — the same function the
    pipeline uses — to check the point-in-time contract end to end rather than re-deriving it."""
    import environment
    import module.dataset as dataset_module

    ticker = "T00"
    benchmark = "SPY"
    monkeypatch.setattr(dataset_module, "RAW_DIR", tmp_path)
    monkeypatch.setattr(dataset_module, "MASTER_DIR", tmp_path)
    monkeypatch.setattr(environment, "DEV_TICKERS", [ticker])

    monthly_dates = pd.date_range("2021-01-31", "2021-06-30", freq="ME")
    price_rows = []
    price = 100.0
    for i, d in enumerate(pd.date_range("2020-12-01", "2021-07-31", freq="D")):
        price *= 1.001
        for t in (ticker, benchmark):
            price_rows.append({"ticker": t, "date": d, "adj_close": price})
    prices = pd.DataFrame(price_rows)

    metrics_payload = {
        "metric": {},
        "series": {
            "quarterly": {
                "roic": [
                    {"period": "2020-12-31", "v": 0.10},
                    {"period": "2021-03-31", "v": 0.14},
                ],
                "pe": [{"period": "2020-12-31", "v": 20.0}, {"period": "2021-03-31", "v": 22.0}],
            }
        },
    }
    metrics = pd.DataFrame([{"ticker": ticker, "payload": repr(metrics_payload)}])
    profiles = pd.DataFrame([{"ticker": ticker, "finnhubIndustry": "Tech"}])

    prices.to_parquet(tmp_path / "prices.parquet")
    metrics.to_parquet(tmp_path / "finnhub_metrics.parquet")
    profiles.to_parquet(tmp_path / "profiles.parquet")

    settings = dataclasses.replace(
        Settings(),
        dev_mode=True,
        data_start_date="2021-01-01",
        start_date="2021-01-31",
        end_date="2021-06-30",
        price_update_frequency="M",
        fundamental_review_frequency="Q",
        walk_forward_scoring=False,
    )
    yield settings, monthly_dates, ticker


def test_fundamental_value_fixed_between_quarterly_reviews_while_price_moves(quarterly_master_dataset):
    from module.dataset import build_master_dataset

    settings, monthly_dates, ticker = quarterly_master_dataset
    master = build_master_dataset(settings)
    rows = master[master["ticker"] == ticker].sort_values("snapshot_date")
    assert len(rows) >= 3

    # Within the Q1 2021 window (before the 2021-03-31 print becomes observable), roic/pe must stay
    # fixed at the 2020-12-31 value even though price (and price-derived columns) moves every month.
    pre_q2 = rows[rows["snapshot_date"] < "2021-03-31"]
    assert pre_q2["roic"].nunique() <= 1, "roic must not change between quarterly fundamental reviews"
    assert pre_q2["price"].nunique() > 1, "price must still move month to month in this same window"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
