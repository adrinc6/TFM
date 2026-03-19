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
    if fit:
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
