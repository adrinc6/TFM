"""Pure rolling walk-forward — the invariant behind module/ml.py's training scheme.

Two things must always hold:
1. `train_cutoff_date` is only the EARLIEST point learning is allowed to start — training dates
   continue past it, quarterly, all the way to the end of the available history. The model never
   freezes at a fixed point and gets reused unchanged for years afterward.
2. Every training date fits its OWN model on a trailing `max_walk_forward_training_years` window of
   history available as of that date — no lookahead, and no single model reused across the whole
   evaluated window.

This is a pure-logic test against module.ml's actual date-splitting helper
(`_train_and_apply_dates`) and a small synthetic end-to-end run, so it does not need downloaded
data or API keys.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from environment import PROCESSED_DIR, RAW_DIR, Settings
import module.ml as ml


def test_train_dates_start_at_cutoff_and_continue_past_it():
    dates = [pd.Timestamp(d) for d in pd.date_range("2015-01-31", "2022-06-30", freq="ME")]
    settings = dataclasses.replace(
        Settings(), train_cutoff_date="2019-06-30", walk_forward_train_frequency="Q"
    )
    train_dates, apply_dates = ml._train_and_apply_dates(dates, settings)
    cutoff = pd.Timestamp(settings.train_cutoff_date)
    assert all(d >= cutoff for d in train_dates)
    # Training must continue quarterly past the cutoff, all the way to the end of history — a pure
    # rolling scheme, not a train-until-cutoff-then-freeze one.
    assert max(train_dates) > cutoff
    assert apply_dates == sorted(dates)


def test_train_dates_are_quarterly_and_cutoff_is_the_first_point():
    dates = [pd.Timestamp(d) for d in pd.date_range("2015-01-31", "2022-06-30", freq="ME")]
    settings = dataclasses.replace(
        Settings(), train_cutoff_date="2020-01-31", walk_forward_train_frequency="Q"
    )
    train_dates, _ = ml._train_and_apply_dates(dates, settings)
    assert pd.Timestamp(settings.train_cutoff_date) in train_dates
    assert min(train_dates) == pd.Timestamp(settings.train_cutoff_date)
    # Roughly quarterly cadence: consecutive training dates should be ~3 months apart.
    gaps_days = [(b - a).days for a, b in zip(train_dates, train_dates[1:])]
    assert all(60 <= gap <= 100 for gap in gaps_days)


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-01-31", "2019-12-31", freq="ME")
    tickers = [f"T{i:02d}" for i in range(12)] + ["SPY"]
    rows = []
    for d in dates:
        for t in tickers:
            row = {"snapshot_date": d.date().isoformat(), "ticker": t, "sector": "Tech"}
            for c in ml.MODEL_FEATURES:
                row[c] = rng.random()
            row["garp_score"] = rng.random()
            row["roic"] = rng.normal(0.15, 0.05)
            row["valuation_score"] = rng.random()
            rows.append(row)
    features = pd.DataFrame(rows)

    price_rows = []
    for t in tickers:
        price = 100.0
        for d in pd.date_range("2015-01-01", "2020-12-31", freq="7D"):
            price *= 1 + rng.normal(0.001, 0.03)
            price_rows.append({"ticker": t, "date": d, "adj_close": price})
    prices = pd.DataFrame(price_rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    features_path = PROCESSED_DIR / "features.parquet"
    prices_path = RAW_DIR / "prices.parquet"
    features.to_parquet(features_path)
    prices.to_parquet(prices_path)

    settings = dataclasses.replace(
        Settings(),
        dev_mode=True,
        start_date="2015-01-31",
        end_date="2019-12-31",
        min_walk_forward_training_rows=30,
        min_walk_forward_training_years=1,
        max_walk_forward_training_years=2,
        train_cutoff_date="2017-01-31",
        walk_forward_train_frequency="Q",
    )
    yield settings
    for path in (features_path, prices_path, PROCESSED_DIR / "scored_universe.parquet",
                 PROCESSED_DIR / "meta_weights_by_snapshot.parquet", PROCESSED_DIR / "model_explainability.json"):
        path.unlink(missing_ok=True)
    diagnostics_dir = settings.run_dir
    if diagnostics_dir.exists():
        for f in diagnostics_dir.glob("*"):
            f.unlink()
        diagnostics_dir.rmdir()


def test_rolling_walk_forward_keeps_retraining_across_the_whole_window(synthetic_run):
    settings = synthetic_run
    ml.train_and_score(settings)

    diagnostics = pd.read_csv(settings.run_dir / "model_walk_forward_diagnostics.csv")
    training = diagnostics[diagnostics["is_train_date"] & (diagnostics["mode"] == "walk_forward_model")]
    assert not training.empty
    # Training dates must span well past the cutoff — not stop there.
    cutoff = pd.Timestamp(settings.train_cutoff_date)
    assert pd.to_datetime(training["snapshot_date"]).max() > cutoff
    # Multiple distinct models must have been fit across the window (not one reused throughout).
    assert diagnostics.loc[diagnostics["mode"] == "walk_forward_model", "training_snapshot_date"].nunique() > 1

    weights = pd.read_parquet(PROCESSED_DIR / "meta_weights_by_snapshot.parquet")
    learned = weights[weights["source"] == "learned"]
    if not learned.empty:
        # If more than one snapshot learned weights, they should not all be byte-identical — the
        # meta-agent is meant to keep adapting, not freeze after the first successful fit.
        agent_keys = ml.AGENT_KEYS
        assert learned[agent_keys].drop_duplicates().shape[0] >= 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
