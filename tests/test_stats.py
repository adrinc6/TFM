"""Significancia estadística y diagnóstico temporal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from module.evaluation.stats import (
    block_bootstrap_ci,
    paired_difference_ci,
    rank_ic_by_year,
    rank_ic_window_summary,
)


def test_bootstrap_ci_brackets_the_mean() -> None:
    values = pd.Series(np.random.default_rng(0).normal(0.05, 0.1, 60))
    ci = block_bootstrap_ci(values, n_boot=500, seed=1)
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"]
    assert ci["n_cohorts"] == 60


def test_paired_difference_detects_clear_advantage() -> None:
    """A consistentemente mejor que B por un margen amplio -> distinguible de cero."""
    n = 40
    a = pd.Series(np.full(n, 0.06), index=range(n))
    b = pd.Series(np.full(n, 0.01), index=range(n))
    result = paired_difference_ci(a, b, n_boot=500, seed=1)
    assert result["mean_diff"] > 0
    assert result["fraction_a_better"] == 1.0
    assert result["distinguishable_from_zero"]


def test_paired_difference_noise_is_not_distinguishable() -> None:
    """Diferencia puramente aleatoria centrada en cero -> NO distinguible."""
    rng = np.random.default_rng(3)
    a = pd.Series(rng.normal(0.02, 0.15, 40), index=range(40))
    b = pd.Series(rng.normal(0.02, 0.15, 40), index=range(40))
    result = paired_difference_ci(a, b, n_boot=500, seed=1)
    assert not result["distinguishable_from_zero"]


def _diag() -> pd.DataFrame:
    rows = []
    for year in (2000, 2001, 2016, 2017):
        base = 0.0 if year < 2010 else 0.05   # aprende mejor en anios recientes
        for month in range(1, 13):
            rows.append({
                "agent": "meta_final",
                "prediction_date": f"{year}-{month:02d}-15",
                "rank_ic": base + (month % 3 - 1) * 0.01,
                "observations": 200 if year < 2010 else 400,
            })
    return pd.DataFrame(rows)


def test_rank_ic_by_year_aggregates() -> None:
    table = rank_ic_by_year(_diag())
    assert list(table["year"]) == [2000, 2001, 2016, 2017]
    # los anios recientes tienen mayor rank-IC y mas empresas por cohorte
    assert table.loc[table["year"] == 2016, "rank_ic_mean"].iloc[0] > \
        table.loc[table["year"] == 2000, "rank_ic_mean"].iloc[0]
    assert table.loc[table["year"] == 2016, "avg_names"].iloc[0] == 400


def test_window_summary_prefers_recent_when_better() -> None:
    summary = rank_ic_window_summary(_diag(), start_years=(2000, 2016))
    recent = summary.loc[summary["start_year"] == 2016, "rank_ic_mean"].iloc[0]
    full = summary.loc[summary["start_year"] == 2000, "rank_ic_mean"].iloc[0]
    assert recent > full
