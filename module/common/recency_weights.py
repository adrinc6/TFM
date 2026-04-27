"""Shared utility for exponential recency weighting of training observations.

Recent quarters carry more predictive signal than observations from several
years ago because factor premia and market dynamics shift over time
(concept drift).  Weighting observations exponentially by age reduces the
influence of stale data without discarding it entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def compute_recency_weights(
    df: pd.DataFrame,
    enabled: bool,
    halflife_years: float,
) -> Optional[np.ndarray]:
    """Compute per-observation exponential recency weights from the index.

    For each observation at time ``t``, the weight is::

        w(t) = exp(log(2) / halflife_days * (t - t_min))

    where ``t_min`` is the oldest observation in the training window.
    Weights are normalised to mean = 1.0 so that the total gradient
    magnitude is independent of window length.

    Args:
        df: Training DataFrame whose index contains date information.
            Supports both plain DatetimeIndex and MultiIndex with a
            ``"date"`` level.
        enabled: When ``False`` the function returns ``None`` immediately
            (no weighting).
        halflife_years: Half-life of the exponential decay in years.
            An observation ``halflife_years`` years before the most recent
            one receives half the weight.  Must be > 0.

    Returns:
        Float32 array of shape ``(len(df),)`` with mean 1.0, or ``None``
        when weighting is disabled or the date cannot be resolved.
    """
    if not enabled:
        return None
    if halflife_years <= 0:
        return None
    try:
        if isinstance(df.index, pd.MultiIndex) and "date" in df.index.names:
            dates = pd.to_datetime(df.index.get_level_values("date"), errors="coerce")
        else:
            dates = pd.to_datetime(df.index, errors="coerce")

        dates_valid = dates.dropna()
        if len(dates_valid) < 2:
            return None

        t_min = dates_valid.min()
        t_max = dates_valid.max()
        if t_min == t_max:
            return None

        halflife_days = float(halflife_years) * 365.25
        lam = np.log(2.0) / halflife_days

        age_days = np.array(
            [(d - t_min).days if pd.notna(d) else 0.0 for d in dates],
            dtype=float,
        )
        weights = np.exp(lam * age_days)
        # Normalise to mean 1.0 so the effective learning rate is preserved
        weights /= weights.mean()
        return weights.astype(np.float32)
    except Exception:
        return None
