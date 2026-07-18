"""Tests de robustez / placebo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from module.robustness import label_permutation_test, leave_one_year_out, random_portfolio_test


def _diag(rank_ics: list[float], years: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "agent": "meta_final",
        "prediction_date": [f"{y}-06-15" for y in years],
        "rank_ic": rank_ics,
    })


def test_permutation_detects_real_signal() -> None:
    """rank-IC real muy por encima del placebo -> señal por encima del azar (p bajo)."""
    real = _diag([0.06, 0.05, 0.07], [2016, 2017, 2018])
    placebo = list(np.random.default_rng(0).normal(0.0, 0.02, 200))   # placebo centrado en 0
    out = label_permutation_test(real, placebo)
    assert out["rank_ic_real"] > 0.05
    assert abs(out["placebo_mean"]) < 0.02
    assert out["p_value"] < 0.05
    assert out["signal_above_chance"]


def test_permutation_flags_no_signal() -> None:
    """rank-IC real indistinguible del placebo -> p alto, no hay señal."""
    real = _diag([0.005, -0.005, 0.0], [2016, 2017, 2018])
    placebo = list(np.random.default_rng(1).normal(0.0, 0.02, 200))
    out = label_permutation_test(real, placebo)
    assert not out["signal_above_chance"]


def test_random_portfolio_percentile() -> None:
    """Una cartera que siempre elige los mejores queda en la cola alta de las aleatorias."""
    years = [2016, 2017, 2018]
    # cada año, activos con retornos de -0.2 a +0.4; el modelo elige los mejores (0.4).
    pools = {y: np.linspace(-0.2, 0.4, 50) for y in years}
    model = pd.Series({y: 0.38 for y in years})
    out = random_portfolio_test(model, pools, portfolio_size=5, n_simulations=300, seed=1)
    assert out["model_percentile"] > 0.95
    assert out["beats_random_convincingly"]


def test_leave_one_year_out_flags_fragile_result() -> None:
    """Un año que sostiene el resultado se detecta: quitarlo cambia mucho el rank-IC."""
    # 2016 con rank-IC altisimo, el resto ~0. Quitar 2016 desploma la media.
    diag = _diag([0.5, 0.5, 0.01, 0.0, -0.01, 0.0], [2016, 2016, 2017, 2018, 2019, 2020])
    table = leave_one_year_out(diag)
    without_2016 = table.loc[table["excluded_year"] == 2016, "rank_ic_without_it"].iloc[0]
    without_2018 = table.loc[table["excluded_year"] == 2018, "rank_ic_without_it"].iloc[0]
    # quitar 2016 baja mucho mas el rank-IC que quitar 2018.
    assert without_2016 < without_2018
