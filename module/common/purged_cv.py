"""Purged and embargoed time-series cross-validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import pandas as pd


@dataclass
class PurgedEmbargoKFold:
    """Time-aware splitter that purges overlap and applies embargo around test windows."""

    n_splits: int = 5
    purge_days: int = 30
    embargo_days: int = 15
    allow_future_train: bool = False

    def split(self, dates: pd.Index) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        ts = pd.to_datetime(pd.Index(dates))
        uniq = np.array(sorted(ts.unique()))
        n_unique = len(uniq)
        if n_unique < 2:
            return

        n_splits = min(max(int(self.n_splits), 2), n_unique)
        fold_sizes = np.full(n_splits, n_unique // n_splits, dtype=int)
        fold_sizes[: n_unique % n_splits] += 1

        start = 0
        for fs in fold_sizes:
            stop = start + fs
            test_dates = uniq[start:stop]
            if len(test_dates) == 0:
                start = stop
                continue

            test_start = pd.Timestamp(test_dates[0])
            test_end = pd.Timestamp(test_dates[-1])
            purge_start = test_start - pd.Timedelta(days=max(int(self.purge_days), 0))
            embargo_end = test_end + pd.Timedelta(days=max(int(self.embargo_days), 0))

            test_mask = ts.isin(test_dates)
            if self.allow_future_train:
                train_mask = (~test_mask) & ((ts < purge_start) | (ts > embargo_end))
            else:
                train_mask = (~test_mask) & (ts < purge_start)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx
            start = stop
