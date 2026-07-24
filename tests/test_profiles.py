from __future__ import annotations

import pandas as pd

from module.evaluation.profiles import apply_profile


def test_bad_stocks_never_selected_by_any_profile() -> None:
    scores = pd.DataFrame({
        "ticker": ["GOOD", "BAD"],
        "snapshot_date": ["2000-01-15", "2000-01-15"],
        "meta_rank": [0.9, 0.2],
        "quality_rank": [0.8, 1.0], "value_rank": [0.8, 1.0],
        "growth_rank": [0.8, 1.0], "momentum_rank": [0.8, 1.0], "risk_rank": [0.8, 1.0],
    })

    for profile in ("growth", "value", "quality", "momentum", "contrarian", "defensive", "garp"):
        result = apply_profile(scores, profile)
        assert result.loc[result["ticker"] == "BAD", "meta_rank"].item() == 0.0
