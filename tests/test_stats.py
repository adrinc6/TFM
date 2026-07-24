from __future__ import annotations

import pandas as pd

from module.evaluation.stats import block_bootstrap_ci


def test_bootstrap_ci_brackets_the_mean() -> None:
    values = pd.Series([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    result = block_bootstrap_ci(values, n_boot=200, seed=0)

    assert result["ci_low"] < values.mean() < result["ci_high"]
