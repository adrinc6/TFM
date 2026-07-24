from __future__ import annotations

from module.evaluation.backtest import _mark_to_market


def test_price_guard_neutralizes_impossible_returns() -> None:
    log = []
    corrupt = _mark_to_market(
        {"AAA": 1.0}, {"AAA": 100.0}, {"d": {"AAA": 1100.0}}, "d", max_return=2.0,
        corrupt_log=log,
    )
    assert corrupt == 0.0
    assert len(log) == 1 and log[0]["ticker"] == "AAA"

    valid = _mark_to_market(
        {"AAA": 1.0}, {"AAA": 100.0}, {"d": {"AAA": 180.0}}, "d", max_return=2.0,
        corrupt_log=[],
    )
    assert valid == 0.80
