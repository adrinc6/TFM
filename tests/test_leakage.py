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

import pandas as pd
import pytest


HORIZON_MONTHS = 12
COMPONENT_TARGETS = [
    "target_future_alpha",
    "target_quality",
    "target_improvement",
    "target_mispricing",
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
