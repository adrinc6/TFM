from __future__ import annotations

from typing import Iterable

import pandas as pd


class PercentileContext:
    def __init__(self, snapshot_col: str = "snapshot_date", sector_col: str = "sector") -> None:
        self.snapshot_col = snapshot_col
        self.sector_col = sector_col

    def add(self, df: pd.DataFrame, features: Iterable[str]) -> tuple[pd.DataFrame, list[str]]:
        out = df.copy()
        base_features = [f for f in features if f in out.columns]
        if self.snapshot_col not in out.columns:
            return out, base_features

        added: list[str] = []
        for feature in base_features:
            global_col = f"{feature}_pct_global"
            out[global_col] = out.groupby(self.snapshot_col)[feature].rank(method="average", pct=True)
            added.append(global_col)

            if self.sector_col in out.columns:
                sector_col = f"{feature}_pct_sector"
                out[sector_col] = out.groupby([self.snapshot_col, self.sector_col])[feature].rank(method="average", pct=True)
                added.append(sector_col)

        if added:
            out[added] = out[added].fillna(0.5)

        return out, base_features + added
