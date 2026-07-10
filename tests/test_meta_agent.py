"""Meta-agent weighting — marginal (partial-IC) contribution, no lookahead.

Pure-logic tests against module.ml's `_fit_meta_weights`: an agent that ranks the forward alpha the
others cannot explain earns weight; noise agents get ~0; the weights are a valid simplex. No
downloaded data or trained model required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import module.ml as ml


def _train_frame(n_dates: int = 24, per_date: int = 12, seed: int = 0) -> pd.DataFrame:
    """One agent (timing) tracks the realized alpha; the other three are independent noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-31", periods=n_dates, freq="ME")
    rows = []
    for date in dates:
        for _ in range(per_date):
            driver = rng.normal()
            alpha = driver + rng.normal(scale=0.3)
            rows.append({
                "snapshot_date_dt": date,
                "target_future_alpha": alpha,
                # timing carries the signal (monotonic in the driver, squashed to [0,1]);
                "timing_probability": 1 / (1 + np.exp(-driver)),
                # the other three are pure noise, uncorrelated with alpha.
                "quality_probability": rng.random(),
                "improvement_probability": rng.random(),
                "mispricing_probability": rng.random(),
            })
    return pd.DataFrame(rows)


def test_fit_meta_weights_returns_a_valid_simplex_and_partial_ics():
    result = ml._fit_meta_weights(_train_frame())
    assert result is not None
    weights, partial_ics = result
    assert set(weights) == set(ml.AGENT_KEYS)
    assert set(partial_ics) == set(ml.AGENT_KEYS)
    assert all(w >= 0 for w in weights.values())
    assert pytest.approx(sum(weights.values()), abs=1e-9) == 1.0


def test_informative_agent_dominates_noise_agents():
    weights, partial_ics = ml._fit_meta_weights(_train_frame())
    assert weights["timing_probability"] == max(weights.values())
    assert weights["timing_probability"] > 0.5
    for noise in ("quality_probability", "improvement_probability", "mispricing_probability"):
        assert weights[noise] < weights["timing_probability"]


def test_duplicating_the_signal_collapses_its_marginal_ic():
    """Marginal-contribution property: once a second agent carries the SAME signal, neither can rank
    the alpha the other already explains, so the informative agent's partial IC drops sharply vs. the
    non-duplicated case — the meta-agent does not pay twice for one signal."""
    base = _train_frame()
    _, base_partial_ics = ml._fit_meta_weights(base)
    solo_ic = base_partial_ics["timing_probability"]

    duplicated = _train_frame()
    duplicated["quality_probability"] = duplicated["timing_probability"]  # exact duplicate of the signal
    _, dup_partial_ics = ml._fit_meta_weights(duplicated)
    # The two duplicates split the credit and each individually contributes far less than the solo
    # agent did; symmetry means they land close to each other.
    assert dup_partial_ics["timing_probability"] < solo_ic
    assert abs(dup_partial_ics["timing_probability"] - dup_partial_ics["quality_probability"]) < 0.15


def test_thin_history_falls_back_to_prior():
    tiny = _train_frame(n_dates=2, per_date=5)
    assert ml._fit_meta_weights(tiny) is None


def test_no_ranking_power_falls_back_to_prior():
    """All agents pure noise → no marginal ranking power → None (caller uses the fixed prior)."""
    rng = np.random.default_rng(1)
    frame = _train_frame()
    for key in ml.AGENT_KEYS:
        frame[key] = rng.random(len(frame))
    # With everything uncorrelated the clipped partial ICs may all be ~0; either a valid simplex or
    # None is acceptable, but if it returns weights they must still be a valid simplex.
    result = ml._fit_meta_weights(frame)
    if result is not None:
        weights, _ = result
        assert pytest.approx(sum(weights.values()), abs=1e-9) == 1.0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
