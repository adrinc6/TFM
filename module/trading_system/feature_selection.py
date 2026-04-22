from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from environment import (
    FEATURE_CORR_THRESHOLD,
    FEATURE_IMPORTANCE_CUTOFF_FRACTION,
    FEATURE_IMPORTANCE_MAX_KEEP,
    FEATURE_IMPORTANCE_MIN_KEEP,
    FEATURE_SELECTOR_RF_MAX_DEPTH,
    FEATURE_SELECTOR_RF_N_ESTIMATORS,
    FEATURE_SELECTOR_RELEVANCE_WEIGHT,
    FEATURE_TOP_N,
    RANDOM_SEED,
)


@dataclass
class FeatureSelectionResult:
    selected: list[str]
    importances: dict[str, float]


class FeatureSelector:
    def __init__(self) -> None:
        self.corr_threshold = float(FEATURE_CORR_THRESHOLD)
        self.relevance_weight = float(FEATURE_SELECTOR_RELEVANCE_WEIGHT)
        self.top_n = int(FEATURE_TOP_N)
        self.min_keep = int(FEATURE_IMPORTANCE_MIN_KEEP)
        self.max_keep = int(FEATURE_IMPORTANCE_MAX_KEEP)
        self.cutoff_fraction = float(FEATURE_IMPORTANCE_CUTOFF_FRACTION)

    @staticmethod
    def _safe_numeric_frame(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
        x = df.loc[:, list(features)].copy()
        for c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
        x = x.replace([np.inf, -np.inf], np.nan)
        medians = x.median(numeric_only=True)
        x = x.fillna(medians)
        x = x.fillna(0.0)
        return x

    def select(self, df: pd.DataFrame, candidate_features: list[str], y: pd.Series) -> FeatureSelectionResult:
        valid_features = [f for f in candidate_features if f in df.columns]
        if not valid_features:
            return FeatureSelectionResult(selected=[], importances={})

        y_clean = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
        x = self._safe_numeric_frame(df, valid_features)
        if x.empty or y_clean.nunique() < 2:
            selected = valid_features[: min(len(valid_features), self.max_keep)]
            return FeatureSelectionResult(selected=selected, importances={k: 0.0 for k in selected})

        rf = RandomForestClassifier(
            n_estimators=FEATURE_SELECTOR_RF_N_ESTIMATORS,
            max_depth=FEATURE_SELECTOR_RF_MAX_DEPTH,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        rf.fit(x, y_clean)
        rf_imp = pd.Series(rf.feature_importances_, index=x.columns)

        mi_vals = mutual_info_classif(x, y_clean, random_state=RANDOM_SEED)
        mi = pd.Series(mi_vals, index=x.columns)

        rf_norm = rf_imp / (rf_imp.max() if rf_imp.max() > 0 else 1.0)
        mi_norm = mi / (mi.max() if mi.max() > 0 else 1.0)
        combined = self.relevance_weight * mi_norm + (1.0 - self.relevance_weight) * rf_norm
        ranked = combined.sort_values(ascending=False)
        pool = ranked.head(min(self.top_n, len(ranked))).index.tolist()

        filtered = self._correlation_filter(x[pool], ranked)
        if not filtered:
            filtered = pool[: self.min_keep]

        imp = rf_imp.reindex(filtered).fillna(0.0)
        top_imp = float(imp.max()) if len(imp) else 0.0
        cutoff = top_imp * self.cutoff_fraction

        selected = [f for f in filtered if float(imp.get(f, 0.0)) >= cutoff]
        if len(selected) < self.min_keep:
            selected = imp.sort_values(ascending=False).head(self.min_keep).index.tolist()
        if len(selected) > self.max_keep:
            selected = imp.reindex(selected).sort_values(ascending=False).head(self.max_keep).index.tolist()

        return FeatureSelectionResult(
            selected=selected,
            importances={f: float(imp.get(f, 0.0)) for f in selected},
        )

    def _correlation_filter(self, x: pd.DataFrame, rank_series: pd.Series) -> list[str]:
        if x.shape[1] <= 1:
            return list(x.columns)
        corr = x.corr().abs()
        keep: list[str] = []
        for col in rank_series.index:
            if col not in corr.columns:
                continue
            if not keep:
                keep.append(col)
                continue
            max_corr = float(corr.loc[col, keep].max()) if keep else 0.0
            if max_corr < self.corr_threshold:
                keep.append(col)
        return keep
