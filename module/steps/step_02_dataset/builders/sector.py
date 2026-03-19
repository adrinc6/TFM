"""Sector normalization utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class SectorNormalizer:
    """
    Z-score relativo al sector para eliminar sesgos sectoriales en los ratios.
    """

    COLS = [
        "roe", "roa", "roi", "roic", "net_margin", "gross_margin", "fcf_margin",
        "ebitda_margin", "operating_margin", "current_ratio",
        "debt_equity", "debt_to_ebitda", "pe_ratio", "pb_ratio", "ev_to_ebitda",
        "fcf_yield", "revenue_yoy_growth", "net_income_yoy_growth", "eps_yoy_growth",
        "momentum_12m", "volatility_20d", "interest_coverage",
        "analyst_buy_ratio", "analyst_consensus", "mspr_3m", "beat_rate_4q",
    ]

    def __init__(self, min_peers: int = 3):
        self.min_peers = min_peers
        self._stats: Dict[str, Dict[str, Tuple[float, float]]] = {}

    def fit(
        self,
        features_dict: Dict[str, pd.Series],
        sector_map: Dict[str, str],
    ) -> "SectorNormalizer":
        from collections import defaultdict

        buckets: Dict[str, List[pd.Series]] = defaultdict(list)
        for ticker, feat in features_dict.items():
            sector = sector_map.get(ticker, "Unknown")
            buckets[sector].append(feat)

        self._stats = {}
        for sector, feats in buckets.items():
            if len(feats) < self.min_peers:
                continue
            combined = pd.DataFrame(feats)
            self._stats[sector] = {}
            for col in self.COLS:
                if col not in combined.columns:
                    continue
                vals = combined[col].dropna()
                if len(vals) >= self.min_peers:
                    self._stats[sector][col] = (float(vals.mean()), float(vals.std()))
        return self

    def transform(self, features: pd.Series, sector: str) -> pd.Series:
        result = features.copy()
        stats = self._stats.get(sector, {})
        for col in self.COLS:
            val = features.get(col, np.nan)
            if col not in stats or pd.isna(val):
                result[f"{col}_zsector"] = np.nan
                continue
            mean, std = stats[col]
            result[f"{col}_zsector"] = (val - mean) / std if std > 0 else 0.0
        return result

    def fit_transform(
        self,
        features_dict: Dict[str, pd.Series],
        sector_map: Dict[str, str],
    ) -> Dict[str, pd.Series]:
        self.fit(features_dict, sector_map)
        return {
            t: self.transform(f, sector_map.get(t, "Unknown"))
            for t, f in features_dict.items()
        }
