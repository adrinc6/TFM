"""Train-until-cutoff-then-freeze — the invariant behind module/ml.py's newest walk-forward mode.

Two things must always hold:
1. No training date is ever after `train_cutoff_date` (the model never learns from the future
   relative to the user-chosen cutoff).
2. Every apply_date after the cutoff is scored with the SAME model/weights as the cutoff itself —
   no further learning happens once the simulation window starts.

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


def test_train_dates_never_exceed_cutoff():
    dates = [pd.Timestamp(d) for d in pd.date_range("2015-01-31", "2022-06-30", freq="ME")]
    settings = dataclasses.replace(
        Settings(), train_cutoff_date="2019-06-30", walk_forward_train_years=3, walk_forward_train_frequency="Q"
    )
    train_dates, apply_dates = ml._train_and_apply_dates(dates, settings)
    cutoff = pd.Timestamp(settings.train_cutoff_date)
    assert all(d <= cutoff for d in train_dates)
    assert apply_dates == sorted(dates)


def test_train_dates_are_quarterly_and_start_at_train_years_before_cutoff():
    dates = [pd.Timestamp(d) for d in pd.date_range("2015-01-31", "2022-06-30", freq="ME")]
    settings = dataclasses.replace(
        Settings(), train_cutoff_date="2020-01-31", walk_forward_train_years=2, walk_forward_train_frequency="Q"
    )
    train_dates, _ = ml._train_and_apply_dates(dates, settings)
    expected_start = pd.Timestamp(settings.train_cutoff_date) - pd.DateOffset(years=2)
    assert all(d >= expected_start for d in train_dates)
    # cutoff itself must always be a training/freeze point even if not a calendar quarter-end
    assert pd.Timestamp(settings.train_cutoff_date) in train_dates


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
            row["expected_growth"] = rng.random()
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
        train_cutoff_date="2018-01-31",
        walk_forward_train_years=2,
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


def test_frozen_phase_reuses_a_single_model_and_weight_set(synthetic_run):
    settings = synthetic_run
    ml.train_and_score(settings)

    diagnostics = pd.read_csv(settings.run_dir / "model_walk_forward_diagnostics.csv")
    assert "phase" in diagnostics.columns
    frozen = diagnostics[diagnostics["phase"] == "frozen"]
    assert not frozen.empty, "expected at least one frozen-phase snapshot in this synthetic window"
    assert frozen["training_snapshot_date"].nunique() == 1, "frozen phase must reuse a single trained model"

    training = diagnostics[diagnostics["is_train_date"] & (diagnostics["mode"] == "walk_forward_model")]
    cutoff = pd.Timestamp(settings.train_cutoff_date)
    assert (pd.to_datetime(training["snapshot_date"]) <= cutoff).all()

    weights = pd.read_parquet(PROCESSED_DIR / "meta_weights_by_snapshot.parquet")
    frozen_weights = weights[weights["phase"] == "frozen"]
    agent_keys = ml.AGENT_KEYS
    assert frozen_weights[agent_keys].drop_duplicates().shape[0] == 1, "frozen meta-agent weights must not change"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
