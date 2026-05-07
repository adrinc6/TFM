from __future__ import annotations

import numpy as np
import pandas as pd

from module.steps.step_03_training.agent_diagnostics import compute_agent_redundancy
from module.steps.step_03_training.training import (
    _apply_rule_quality_multipliers,
    _build_rule_quality_multipliers,
)


def test_compute_agent_redundancy_no_high_corr_pairs_does_not_crash() -> None:
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame(
        {
            "fundamental_score": rng.uniform(0.0, 1.0, n),
            "momentum_score": rng.uniform(0.0, 1.0, n),
            "valuation_score": rng.uniform(0.0, 1.0, n),
        }
    )

    out = compute_agent_redundancy(df, ["fundamental_score", "momentum_score", "valuation_score"], corr_cap=0.95)

    assert "high_corr_pairs" in out
    pairs = out["high_corr_pairs"]
    assert isinstance(pairs, pd.DataFrame)
    assert list(pairs.columns) == ["left", "right", "abs_corr"]
    assert pairs.empty


def test_build_rule_quality_multipliers_mutes_inverted_rules() -> None:
    rule_quality = pd.DataFrame(
        [
            {
                "rule_col": "fundamental_rule_signal",
                "n": 180,
                "spearman_ic": -0.04,
                "top_bottom_spread": -0.03,
                "stability": 0.60,
            },
            {
                "rule_col": "momentum_rule_signal",
                "n": 180,
                "spearman_ic": 0.09,
                "top_bottom_spread": 0.11,
                "stability": 0.72,
            },
        ]
    )

    multipliers = _build_rule_quality_multipliers(rule_quality)

    assert multipliers["fundamental"] == 0.0
    assert 0.0 < multipliers["momentum"] <= 1.0


def test_apply_rule_quality_multipliers_recomputes_consensus() -> None:
    df = pd.DataFrame(
        {
            "fundamental_rule_signal": [0.8, -0.6],
            "fundamental_rule_confidence": [0.7, 0.8],
            "momentum_rule_signal": [0.1, 0.2],
            "momentum_rule_confidence": [0.4, 0.3],
        }
    )
    multipliers = {"fundamental": 0.0, "momentum": 1.0}

    out = _apply_rule_quality_multipliers(df, multipliers)

    np.testing.assert_allclose(out["fundamental_rule_signal"].to_numpy(dtype=float), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(out["fundamental_rule_confidence"].to_numpy(dtype=float), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        out["rules_consensus_signal"].to_numpy(dtype=float),
        out["momentum_rule_signal"].to_numpy(dtype=float) / 2.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        out["rules_consensus_confidence"].to_numpy(dtype=float),
        out["momentum_rule_confidence"].to_numpy(dtype=float) / 2.0,
        atol=1e-12,
    )
