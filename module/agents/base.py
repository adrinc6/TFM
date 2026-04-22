from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from environment import MOMENTUM_MAX_DEPTH, MOMENTUM_MIN_SAMPLES_LEAF, MOMENTUM_N_ESTIMATORS, RANDOM_SEED
from module.common.percentile_context import PercentileContext
from module.common.utils import safe_numeric_frame
from module.steps.step_03_trainer.feature_selection import FeatureSelector


@dataclass
class AgentPerformance:
    best_model: str
    model_scores: Dict[str, float] = field(default_factory=dict)


class BaseAgent:
    def __init__(self, name: str, feature_pool: list[str], min_features: int = 8, max_features: int = 12) -> None:
        self.name = name
        self.feature_pool = list(dict.fromkeys(feature_pool))
        self.min_features = int(min_features)
        self.max_features = int(max_features)
        self.selector = FeatureSelector()
        self.percentile_context = PercentileContext(snapshot_col="snapshot_date", sector_col="sector")

        self.model = None
        self.selected_features: list[str] = []
        self.performance = AgentPerformance(best_model="", model_scores={})

    def candidate_features(self, df: pd.DataFrame) -> list[str]:
        available = [f for f in self.feature_pool if f in df.columns]
        numeric = [f for f in available if pd.api.types.is_numeric_dtype(df[f])]
        return numeric

    @staticmethod
    def _base_features(selected_features: Iterable[str]) -> list[str]:
        return [f for f in selected_features if not f.endswith("_pct_global") and not f.endswith("_pct_sector")]

    @staticmethod
    def _candidate_models() -> Dict[str, object]:
        return {
            "logistic": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
            "rf": RandomForestClassifier(
                n_estimators=int(MOMENTUM_N_ESTIMATORS),
                max_depth=int(MOMENTUM_MAX_DEPTH),
                min_samples_leaf=int(MOMENTUM_MIN_SAMPLES_LEAF),
                class_weight="balanced_subsample",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            "gbm": GradientBoostingClassifier(random_state=RANDOM_SEED),
        }

    def _select_features(self, x_train: pd.DataFrame, y_train: pd.Series) -> list[str]:
        candidate = self.candidate_features(x_train)
        selection = self.selector.select(x_train, candidate, y_train)
        selected = selection.selected

        if len(selected) < self.min_features:
            missing = [c for c in candidate if c not in selected]
            selected = (selected + missing)[: max(self.min_features, len(selected))]

        if len(selected) > self.max_features:
            selected = selected[: self.max_features]

        return selected

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series) -> None:
        selected = self._select_features(x_train, y_train)

        train_aug, train_features = self.percentile_context.add(x_train, selected)
        val_aug, val_features = self.percentile_context.add(x_val, selected)
        model_features = [f for f in train_features if f in val_features and f in train_aug.columns and f in val_aug.columns]

        xt = safe_numeric_frame(train_aug, model_features)
        xv = safe_numeric_frame(val_aug, model_features)
        yt = pd.to_numeric(y_train, errors="coerce").fillna(0).astype(int)
        yv = pd.to_numeric(y_val, errors="coerce").fillna(0).astype(int)

        best_model = None
        best_auc = -np.inf
        best_name = ""
        scores: dict[str, float] = {}

        for model_name, model in self._candidate_models().items():
            fitted = clone(model)
            fitted.fit(xt, yt)
            pred = fitted.predict_proba(xv)[:, 1]
            auc = float(roc_auc_score(yv, pred)) if yv.nunique() > 1 else 0.5
            scores[model_name] = auc
            if auc > best_auc:
                best_auc = auc
                best_name = model_name
                best_model = fitted

        if best_model is None:
            best_name = "logistic"
            best_model = clone(self._candidate_models()["logistic"])
            best_model.fit(xt, yt)

        full = pd.concat([x_train, x_val], axis=0)
        full_aug, _ = self.percentile_context.add(full, selected)
        xf = safe_numeric_frame(full_aug, model_features)
        yf = pd.concat([yt, yv], axis=0)
        best_model.fit(xf, yf)

        self.model = best_model
        self.selected_features = model_features
        self.performance = AgentPerformance(best_model=best_name, model_scores=scores)

    def predict_proba(self, x: pd.DataFrame) -> pd.Series:
        if self.model is None or not self.selected_features:
            return pd.Series(0.5, index=x.index, name=f"{self.name}_score")

        base = self._base_features(self.selected_features)
        aug, _ = self.percentile_context.add(x, base)
        use = [c for c in self.selected_features if c in aug.columns]
        xs = safe_numeric_frame(aug, use)
        prob = self.model.predict_proba(xs)[:, 1]
        return pd.Series(np.clip(prob, 0.0, 1.0), index=x.index, name=f"{self.name}_score")
