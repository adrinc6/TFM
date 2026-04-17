"""Sector normalization for dataset frames."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from module.steps.step_02_dataset.builders.sector import SectorNormalizer


def apply_sector_normalization(
    df: pd.DataFrame,
    sector_map: Dict[str, str],
    normalizer: SectorNormalizer,
    fit: bool = True,
) -> pd.DataFrame:
    """Applies sector-based z-score normalisation to a multi-index DataFrame.

    For each ``*_zsector`` column defined by the SectorNormalizer, computes
    ``(value - sector_mean) / sector_std`` using sector-level statistics.
    When ``fit=True``, the statistics are estimated from the data.
    When ``fit=False``, the previously fitted statistics are reused.

    Args:
        df (pd.DataFrame): Multi-indexed (ticker, date) feature DataFrame.
        sector_map (Dict[str, str]): Mapping ``{ticker: sector}``.
        normalizer (SectorNormalizer): Fitted or unfitted normaliser instance.
        fit (bool): If True, fits the normaliser from df; if False, applies
            previously fitted statistics (for inference / test folds).

    Returns:
        pd.DataFrame: Copy of df with additional ``{col}_zsector`` columns.
    """
    if fit:
        # Build a flat feature dict to fit the normaliser
        features_dict: Dict[str, pd.Series] = {}
        for i, ((ticker, _date), row) in enumerate(df.iterrows()):
            features_dict[f"{ticker}_{i}"] = row
        expanded_sector_map = {
            f"{ticker}_{i}": sector_map.get(ticker, "Unknown")
            for i, (ticker, _date) in enumerate(df.index)
        }
        normalizer.fit(features_dict, expanded_sector_map)

    tickers_col = df.index.get_level_values("ticker")
    sectors = tickers_col.map(lambda t: sector_map.get(t, "Unknown"))

    df_norm = df.copy()
    for col in normalizer.COLS:
        if col not in df_norm.columns:
            df_norm[f"{col}_zsector"] = np.nan
            continue
        zcol = f"{col}_zsector"
        df_norm[zcol] = np.nan
        for sector in sectors.unique():
            stats = normalizer._stats.get(sector, {}).get(col)
            if stats is None:
                continue
            mean, std = stats
            mask = sectors == sector
            if std > 0:
                df_norm.loc[mask, zcol] = (df_norm.loc[mask, col] - mean) / std
            else:
                df_norm.loc[mask, zcol] = 0.0

    return df_norm
