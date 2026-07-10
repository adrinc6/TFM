"""Rotation policy — minimum-hold gating on soft exits and the replacement thresholds.

Pure-logic tests on module.strategy.portfolio's `_exit_reason` and `_replacement_target`, using
synthetic rows/positions. They assert (1) soft sell triggers respect MIN_HOLD_MONTHS_BEFORE_ROTATION
while hard triggers fire immediately, and (2) replacement uses the environment thresholds, with no
hardcoded duplicate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from environment import MIN_ROTATION_ADVANTAGE
import module.strategy.portfolio as pf


def _row(**overrides) -> pd.Series:
    base = {
        "thesis_state": "Intact",
        "exit_score": 0.0,
        "valuation_score": 0.6,
        "price_adjusted_valuation_score": 0.6,
        "momentum_score": 0.6,
        "manager_score": 0.30,          # below MIN_HOLD_SCORE -> soft hurdle trigger
        "would_buy_today": False,
        "opportunity_cost_score": 0.0,
    }
    base.update(overrides)
    return pd.Series(base)


def _position(months_since_entry: int, **overrides) -> dict:
    base = {
        "months_since_entry": months_since_entry,
        "not_buy_today_count": 0,
        "thesis_deterioration_count": 0,
    }
    base.update(overrides)
    return base


def test_soft_trigger_suppressed_before_minimum_hold():
    reason = pf._exit_reason(_row(), _position(months_since_entry=pf.MIN_HOLD_MONTHS_BEFORE_ROTATION - 1))
    assert reason is None


def test_soft_trigger_fires_after_minimum_hold():
    reason = pf._exit_reason(_row(), _position(months_since_entry=pf.MIN_HOLD_MONTHS_BEFORE_ROTATION))
    assert reason == "Manager Score Below Hold Hurdle"


def test_hard_trigger_fires_immediately_even_when_freshly_bought():
    reason = pf._exit_reason(_row(thesis_state="Broken"), _position(months_since_entry=0))
    assert reason == "Thesis Broken"


def test_exit_score_hard_trigger_ignores_hold_period():
    reason = pf._exit_reason(_row(exit_score=0.9), _position(months_since_entry=0))
    assert reason == "Exit Score Trigger"


def _candidate(manager_score: float, **overrides) -> SimpleNamespace:
    base = {
        "ticker": "NEW",
        "manager_score": manager_score,
        "conviction_score": 0.6,
        "momentum_score": 0.6,
        "would_buy_today": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _held(manager_score: float, conviction: float = 0.5, momentum: float = 0.5) -> dict:
    return {
        "current_manager_score": manager_score,
        "current_conviction_score": conviction,
        "current_momentum_score": momentum,
        "current_thesis_state": "Intact",
        "months_since_entry": pf.MIN_HOLD_MONTHS_BEFORE_ROTATION,
    }


def test_replacement_requires_the_configured_score_advantage():
    updated = {"OLD": _held(0.50)}
    # A challenger a clear MIN_ROTATION_ADVANTAGE above the weakest holding replaces it.
    strong = _candidate(0.50 + MIN_ROTATION_ADVANTAGE + 0.01)
    assert pf._replacement_target(updated, strong) == ("OLD", updated["OLD"])
    # A negligible edge does not.
    weak = _candidate(0.51, conviction=0.50, momentum=0.50)
    assert pf._replacement_target(updated, weak) is None


def test_replacement_skips_positions_inside_minimum_hold():
    fresh = {"OLD": {**_held(0.40), "months_since_entry": 0, "current_thesis_state": "Intact"}}
    strong = _candidate(0.90)
    assert pf._replacement_target(fresh, strong) is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
