from __future__ import annotations

import numpy as np
import pandas as pd

from environment import Settings
from module.modeling.meta import AGENT_NAMES, _weights_as_of


def test_meta_type_rank_ic_favours_the_predictive_agent() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for date in pd.date_range("2010-03-15", periods=8, freq="3MS"):
        for index in range(30):
            future = rng.normal()
            for agent, score in {
                "momentum": future + rng.normal(scale=0.3),
                "quality": rng.normal(),
                "value": rng.normal(),
            }.items():
                rows.append({
                    "ticker": f"T{index}", "snapshot_date": date.date().isoformat(),
                    "snapshot_ts": date, "label_end_ts": date + pd.DateOffset(months=3),
                    "is_quarterly": True, "target_available": True, "agent": agent,
                    "score": score, "forward_excess_return_3m": future,
                })

    weights, _ = _weights_as_of(
        pd.DataFrame(rows), pd.Timestamp("2011-06-15"), set(AGENT_NAMES),
        Settings(run_scope="dev", meta_type="rank_ic"),
    )

    assert weights["momentum"] > weights["quality"]
    assert weights["momentum"] > weights["value"]
