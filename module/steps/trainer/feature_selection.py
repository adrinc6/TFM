from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from environment import (
    FEATURE_CORR_THRESHOLD,
    FEATURE_IMPORTANCE_CUTOFF_FRACTION,
    FEATURE_IMPORTANCE_MAX_KEEP,
    FEATURE_IMPORTANCE_MIN_KEEP,
    FEATURE_SELECTOR_RELEVANCE_WEIGHT,
    FEATURE_TOP_N,
    RANDOM_SEED,
)
from module.common.utils import safe_numeric_frame


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

    def select(self, df: pd.DataFrame, candidate_features: list[str], y: pd.Series) -> FeatureSelectionResult:
        valid = [c for c in candidate_features if c in df.columns]
        if not valid:
            return FeatureSelectionResult(selected=[], importances={})

        x = safe_numeric_frame(df, valid)
        yt = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

        if x.empty or yt.nunique() < 2:
            picked = valid[: min(len(valid), self.max_keep)]
            return FeatureSelectionResult(selected=picked, importances={f: 0.0 for f in picked})

        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        rf.fit(x, yt)
        imp = pd.Series(rf.feature_importances_, index=x.columns)

        mi = pd.Series(mutual_info_classif(x, yt, random_state=RANDOM_SEED), index=x.columns)
        imp_n = imp / (float(imp.max()) if imp.max() > 0 else 1.0)
        mi_n = mi / (float(mi.max()) if mi.max() > 0 else 1.0)
        score = self.relevance_weight * mi_n + (1.0 - self.relevance_weight) * imp_n

        ranked = score.sort_values(ascending=False).head(min(self.top_n, len(score))).index.tolist()
        filtered = self._correlation_filter(x[ranked], score)

        final_imp = imp.reindex(filtered).fillna(0.0)
        cutoff = float(final_imp.max()) * self.cutoff_fraction if len(final_imp) else 0.0
        selected = [f for f in filtered if float(final_imp.get(f, 0.0)) >= cutoff]

        if len(selected) < self.min_keep:
            selected = final_imp.sort_values(ascending=False).head(self.min_keep).index.tolist()
        if len(selected) > self.max_keep:
            selected = final_imp.reindex(selected).sort_values(ascending=False).head(self.max_keep).index.tolist()

        return FeatureSelectionResult(selected=selected, importances={k: float(final_imp.get(k, 0.0)) for k in selected})

    def _correlation_filter(self, x: pd.DataFrame, ranked_score: pd.Series) -> list[str]:
        if x.shape[1] <= 1:
            return list(x.columns)

        corr = x.corr().abs()
        keep: list[str] = []
        for col in ranked_score.sort_values(ascending=False).index:
            if col not in x.columns or col not in corr.index or col not in corr.columns:
                continue
            if not keep:
                keep.append(col)
                continue
            if float(corr.loc[col, keep].max()) < self.corr_threshold:
                keep.append(col)
        return keep
