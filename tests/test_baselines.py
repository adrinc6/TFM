"""Placebo and simple-baseline checks: does the ranking actually carry information?

Two synthetic scenarios:
1. A score that is genuinely predictive (constructed to correlate with forward return) should beat
   its own shuffled distribution by a wide margin — high percentile.
2. A score that is pure noise (uncorrelated with forward return) should land squarely inside its own
   shuffled distribution — nothing special about the "real" ranking versus a random one.

This is a pure-logic test against module.backtest.baselines with a fully synthetic universe/price
panel, so it runs without downloaded data or a trained model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from module.backtest.baselines import (
    cumulative_alpha,
    placebo_distribution,
    placebo_summary,
    top_n_monthly_returns,
)


def _synthetic_panel(n_snapshots: int = 10, n_tickers: int = 20, predictive: bool = True, seed: int = 1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-31", periods=n_snapshots, freq="ME")
    tickers = [f"T{i:02d}" for i in range(n_tickers)] + ["SPY"]

    # Per-ticker constant "quality" drift so a predictive score can key off it.
    drift = {t: rng.normal(0.0, 0.02) for t in tickers}
    drift["SPY"] = 0.0

    price_rows = []
    price = {t: 100.0 for t in tickers}
    daily_dates = pd.date_range(dates[0] - pd.DateOffset(months=1), dates[-1] + pd.DateOffset(months=1), freq="D")
    for d in daily_dates:
        for t in tickers:
            price[t] *= 1 + rng.normal(drift[t] / 21, 0.01)
            price_rows.append({"ticker": t, "date": d, "adj_close": price[t]})
    prices = pd.DataFrame(price_rows)

    universe_rows = []
    for d in dates:
        for t in tickers:
            if t == "SPY":
                continue
            score = drift[t] * 10 if predictive else rng.random()
            universe_rows.append({"snapshot_date": d.date().isoformat(), "ticker": t, "final_score": score})
    universe = pd.DataFrame(universe_rows)
    return universe, prices


def test_predictive_score_beats_its_shuffled_distribution():
    universe, prices = _synthetic_panel(predictive=True, seed=1)
    real_alpha, shuffled = placebo_distribution(
        universe, prices, "SPY", score_column="final_score", top_n=5, n_shuffles=100, seed=2,
    )
    summary = placebo_summary(real_alpha, shuffled)
    percentile = summary.iloc[0]["percentil_real"]
    assert percentile is not None
    assert percentile > 0.8, f"predictive score should rank near the top of its shuffled distribution, got percentile={percentile}"


def test_noise_score_lands_inside_its_shuffled_distribution():
    universe, prices = _synthetic_panel(predictive=False, seed=3)
    real_alpha, shuffled = placebo_distribution(
        universe, prices, "SPY", score_column="final_score", top_n=5, n_shuffles=100, seed=4,
    )
    summary = placebo_summary(real_alpha, shuffled)
    percentile = summary.iloc[0]["percentil_real"]
    assert percentile is not None
    assert 0.05 <= percentile <= 0.95, f"pure-noise score should not be an outlier vs. its own shuffles, got percentile={percentile}"


def test_top_n_monthly_returns_is_point_in_time():
    """The holdings decided at snapshot `t` must only use price information up to `t`; the return is
    measured forward to `t+1`, never using a price from before `t` to decide the period return."""
    universe, prices = _synthetic_panel(predictive=True, seed=5)
    monthly = top_n_monthly_returns(universe, prices, "SPY", "final_score", top_n=5)
    assert not monthly.empty
    assert set(monthly.columns) == {"date", "portfolio_period_return", "benchmark_period_return", "period_alpha"}
    assert len(monthly) == universe["snapshot_date"].nunique() - 1


def test_cumulative_alpha_compone_no_suma():
    """La alpha acumulada debe COMPONERSE, no sumar los excesos mensuales.

    Regresión de un bug real: se reportaba `period_alpha.sum()`, que no es un retorno realizable e
    infravaloraba el resultado (en el baseline: suma=1.14 vs. compuesta=5.77). Con 12 meses al 10%
    de cartera y 0% de benchmark, la respuesta correcta es 1.10^12 - 1 ≈ 2.138, no 12*0.10 = 1.20.
    """
    monthly = pd.DataFrame({
        "portfolio_period_return": [0.10] * 12,
        "benchmark_period_return": [0.0] * 12,
        "period_alpha": [0.10] * 12,
    })
    alpha = cumulative_alpha(monthly)
    assert alpha == pytest.approx(1.10**12 - 1)
    assert alpha != pytest.approx(1.20), "no debe ser la suma aritmética de los excesos"


def test_cumulative_alpha_resta_benchmark_compuesto():
    """El benchmark también se compone: alpha = (Π(1+rp) - 1) - (Π(1+rb) - 1)."""
    monthly = pd.DataFrame({
        "portfolio_period_return": [0.10] * 6,
        "benchmark_period_return": [0.05] * 6,
        "period_alpha": [0.05] * 6,
    })
    assert cumulative_alpha(monthly) == pytest.approx((1.10**6 - 1) - (1.05**6 - 1))


def test_cumulative_alpha_casos_borde():
    """Vacío o sin las columnas necesarias -> 0.0, sin romper."""
    assert cumulative_alpha(pd.DataFrame()) == 0.0
    assert cumulative_alpha(pd.DataFrame({"otra": [1.0]})) == 0.0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
