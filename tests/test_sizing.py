"""Position sizing — conviction concentration and convexity.

Pure tests on module.strategy.sizing: the sizing score is convex in conviction (top ideas pull
disproportionately), and the resulting book concentrates weight on the highest-conviction names
above equal weight while respecting the per-position cap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import module.strategy.sizing as sizing


def _sizing_score_at(conviction_values: list[float]) -> pd.Series:
    df = pd.DataFrame({"current_conviction_score": conviction_values})
    return sizing._sizing_score(df)


def test_sizing_score_is_convex_in_conviction():
    scores = _sizing_score_at([0.3, 0.5, 0.7, 0.9]).to_numpy()
    first_diffs = np.diff(scores)
    assert (first_diffs > 0).all(), "sizing score must increase with conviction"
    # Convexity: equal steps in conviction produce accelerating steps in score.
    assert (np.diff(first_diffs) > 0).all(), "sizing score must be convex in conviction"


def _book(conviction_values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2021-06-30"] * len(conviction_values),
        "ticker": [f"T{i}" for i in range(len(conviction_values))],
        "current_conviction_score": conviction_values,
    })


def test_hybrid_weights_concentrate_on_conviction():
    sized = sizing.add_position_sizing(_book([0.9, 0.6, 0.3])).sort_values("current_conviction_score", ascending=False)
    weights = sized["hybrid_weight"].to_numpy()
    # Non-increasing with conviction (the top names may tie once they hit the per-position cap).
    assert (np.diff(weights) <= 1e-9).all(), "higher conviction must not get less weight"
    assert pytest.approx(weights.sum(), abs=1e-6) == 1.0
    equal = 1 / 3
    assert weights[0] > equal > weights[-1], "the book must concentrate, not equal-weight"


def test_cap_keeps_concentration_bounded():
    """The cap is applied before renormalization, so it compresses (not hard-limits) the top names —
    but it must still keep the book from collapsing onto a single name."""
    sized = sizing.add_position_sizing(_book([0.99, 0.98, 0.10, 0.10, 0.10]))
    assert pytest.approx(sized["hybrid_weight"].sum(), abs=1e-6) == 1.0
    assert sized["hybrid_weight"].max() < 0.45, "no single position should dominate the book"


def test_more_convex_exponent_widens_the_top_to_middle_gap(monkeypatch):
    base = _sizing_score_at([0.5, 0.9])
    base_gap = base.iloc[1] - base.iloc[0]
    monkeypatch.setattr(sizing, "CONVICTION_CONVEXITY", 2.5)
    stronger = _sizing_score_at([0.5, 0.9])
    stronger_gap = stronger.iloc[1] - stronger.iloc[0]
    assert stronger_gap > base_gap


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
