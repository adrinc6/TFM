"""Feature-layer invariants: score ranges, non-duplication, and non-circular targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from module.features.transforms import add_expectation_features


def _panel(n_tickers: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for i in range(n_tickers):
        rows.append({
            "snapshot_date": "2021-06-30",
            "ticker": f"T{i:02d}",
            "growth_score": rng.random(),
            "quality_score": rng.random(),
            "catalyst_score": rng.random(),
            "moat_score": rng.random(),
            "valuation_score": rng.random(),
            "revenue_growth": rng.normal(0.1, 0.15),
            "eps_growth": rng.normal(0.12, 0.2),
        })
    return pd.DataFrame(rows)


def test_realized_growth_is_observed_not_circular():
    """realized_growth must come from reported revenue/eps growth, not a fixed blend of scores."""
    df = _panel()
    out = add_expectation_features(df)
    # A deterministic 0.6*growth+0.25*quality+0.15*moat blend would correlate ~1 with that formula;
    # the observed-growth percentile should not.
    circular = (0.60 * df["growth_score"] + 0.25 * df["quality_score"] + 0.15 * df["moat_score"]).clip(0, 1)
    corr = out["realized_growth"].corr(circular)
    assert abs(corr) < 0.95, f"realized_growth still tracks the old circular blend (corr={corr:.2f})"


def test_expectation_features_in_range():
    out = add_expectation_features(_panel())
    for col in ["realized_growth", "implied_growth", "expected_growth", "positive_expectation_gap"]:
        assert out[col].between(0, 1).all(), f"{col} out of [0,1]"
    assert out["expectation_gap"].between(-1, 1).all()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
