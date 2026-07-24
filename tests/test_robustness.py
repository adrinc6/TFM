from __future__ import annotations

import pandas as pd

from module.evaluation.robustness import label_permutation_test


def test_permutation_detects_real_signal() -> None:
    diagnostics = pd.DataFrame({"agent": ["meta_final"] * 4, "rank_ic": [0.2, 0.3, 0.25, 0.35]})
    result = label_permutation_test(diagnostics, [-0.1, 0.0, 0.05, 0.1])

    assert result["signal_above_chance"]
    assert result["rank_ic_real"] > result["placebo_mean"]
