"""Return helpers for the live fold."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def qtd_return(close_series: pd.Series) -> Optional[float]:
    """Return from the first to last available close in the series."""
    series = close_series.dropna()
    if len(series) < 2:
        return None
    return float((series.iloc[-1] - series.iloc[0]) / series.iloc[0])
