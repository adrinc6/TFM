from __future__ import annotations

import pandas as pd

from environment import ANALYSIS_FREQUENCY
from module.common.feature_engineering import analysis_keys_for_dataframe


def build_walk_forward_folds(df: pd.DataFrame, min_train_periods: int = 8, frequency: str = ANALYSIS_FREQUENCY) -> list[tuple[pd.Index, pd.Index]]:
    if df is None or df.empty:
        return []

    if "snapshot_date" not in df.columns:
        raise ValueError("DataFrame must include snapshot_date for temporal walk-forward")

    keys = analysis_keys_for_dataframe(df)
    periods = sorted(keys.dropna().unique().tolist())
    if len(periods) < 2:
        return []

    if len(periods) <= min_train_periods + 1:
        train_periods = periods[:-1]
        test_period = periods[-1]
        return [(df[keys.isin(train_periods)].index, df[keys == test_period].index)]

    folds: list[tuple[pd.Index, pd.Index]] = []
    for i in range(min_train_periods, len(periods)):
        train_periods = periods[:i]
        test_period = periods[i]
        train_idx = df[keys.isin(train_periods)].index
        test_idx = df[keys == test_period].index
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        folds.append((train_idx, test_idx))

    return folds
