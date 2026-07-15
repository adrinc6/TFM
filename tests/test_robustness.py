"""Statistical-robustness helpers — bootstrap CI, cost breakeven, sub-period split.

Pure tests on module.backtest.robustness against synthetic vs-benchmark tables. No pipeline run,
downloaded data, or trained model required.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from environment import Settings
import module.backtest.robustness as rob


def _vs_benchmark(excess: np.ndarray, drag: float = 0.0) -> pd.DataFrame:
    """Monthly table with the real performance semantics: gross is PRE-cost (gross − benchmark =
    excess), and net = gross − drag is what portfolio_period_return carries."""
    benchmark = np.full(excess.size, 0.005)
    gross = benchmark + excess  # gross excess return, before any transaction cost
    return pd.DataFrame({
        "date": pd.date_range("2018-01-31", periods=excess.size, freq="ME").astype(str),
        "portfolio_gross_period_return": gross,
        "transaction_cost_drag": np.full(excess.size, drag),
        "portfolio_period_return": gross - drag,
        "benchmark_period_return": benchmark,
    })


def test_bootstrap_positive_edge_has_positive_interval():
    excess = np.full(48, 0.01)  # a steady +1%/month edge
    summary, distribution = rob.block_bootstrap_excess_returns(_vs_benchmark(excess), n_resamples=500, seed=0)
    assert list(distribution.columns) == ["cumulative_alpha", "information_ratio"]
    assert len(distribution) == 500
    alpha_row = summary[summary["metrica"].str.startswith("Alpha")].iloc[0]
    assert alpha_row["prob_positiva"] == "100%"
    # A constant series has zero spread, so the CI collapses onto the point estimate.
    assert alpha_row["ic_2_5"] == alpha_row["ic_97_5"]


def test_bootstrap_noisy_edge_interval_can_straddle_zero():
    rng = np.random.default_rng(0)
    excess = rng.normal(0.0, 0.05, size=36)  # zero-mean noise
    summary, _ = rob.block_bootstrap_excess_returns(_vs_benchmark(excess), n_resamples=800, seed=1)
    prob = float(summary.iloc[0]["prob_positiva"].rstrip("%"))
    assert 5 < prob < 95, "a zero-mean series should not read as a near-certain edge"


def test_bootstrap_too_short_returns_empty():
    summary, distribution = rob.block_bootstrap_excess_returns(_vs_benchmark(np.array([0.01, 0.02])))
    assert summary.empty and distribution.empty


def test_cost_breakeven_multiplier_matches_closed_form():
    excess = np.full(24, 0.01)
    vs = _vs_benchmark(excess, drag=0.002)
    # A0 = sum(excess) = 24 * 0.01 = 0.24; D = sum(drag) = 24 * 0.002 = 0.048; breakeven = A0/D = 5.
    assert rob.cost_breakeven_multiplier(vs) == pytest.approx(0.24 / 0.048)


def test_cost_breakeven_none_when_no_positive_edge():
    vs = _vs_benchmark(np.full(24, -0.01), drag=0.002)
    assert rob.cost_breakeven_multiplier(vs) is None


def test_cost_sweep_is_monotonic_decreasing():
    vs = _vs_benchmark(np.full(24, 0.01), drag=0.002)
    settings = dataclasses.replace(Settings(), transaction_cost_bps=5.0, slippage_bps=10.0)
    sweep = rob.cost_sensitivity_sweep(vs, settings, multipliers=[0.0, 1.0, 2.0, 5.0, 10.0])
    alphas = sweep["alpha_acumulada_neta"].to_numpy()
    assert (np.diff(alphas) < 0).all()
    # coste_total_bps scales with the base 15 bps.
    assert sweep.loc[sweep["multiplicador"] == 2.0, "coste_total_bps"].iloc[0] == pytest.approx(30.0)
    assert not bool(sweep.loc[sweep["multiplicador"] == 10.0, "sigue_positiva"].iloc[0])


def test_sub_period_alpha_splits_into_n_contiguous_chunks():
    excess = np.full(24, 0.01)
    out = rob.sub_period_alpha(_vs_benchmark(excess), n_periods=3)
    assert len(out) == 3
    assert list(out["sub_periodo"]) == [1, 2, 3]
    assert out["n_periodos"].sum() == 24
    # A steady, constant edge should split evenly across chunks (each chunk sees the same series).
    assert out["alpha_acumulada"].nunique() <= 1 or np.allclose(out["alpha_acumulada"], out["alpha_acumulada"].iloc[0])


def test_sub_period_alpha_flags_a_single_strong_chunk():
    # Only the last third of the window carries any edge; the rest is flat.
    excess = np.concatenate([np.zeros(16), np.full(8, 0.03)])
    out = rob.sub_period_alpha(_vs_benchmark(excess), n_periods=3)
    assert out.iloc[-1]["alpha_acumulada"] > out.iloc[0]["alpha_acumulada"]
    assert out.iloc[0]["alpha_acumulada"] == pytest.approx(0.0)


def test_sub_period_alpha_empty_is_safe():
    assert rob.sub_period_alpha(pd.DataFrame()).empty


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
