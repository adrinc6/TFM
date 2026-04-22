from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from environment import META_LR_C, RANDOM_SEED
from module.common.utils import safe_numeric_frame


@dataclass
class MetaModelPerformance:
    best_model: str
    scores: Dict[str, float]
    validation_auc: float


class MetaModel:
    def __init__(self) -> None:
        self.model = None
        self.features: list[str] = []
        self.performance = MetaModelPerformance(best_model="", scores={}, validation_auc=np.nan)

    @staticmethod
    def _candidate_models() -> dict[str, object]:
        return {
            "logistic": LogisticRegression(
                C=META_LR_C,
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
            "rf": RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        }

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series) -> None:
        self.features = list(x_train.columns)
        x_tr = safe_numeric_frame(x_train, self.features)
        x_va = safe_numeric_frame(x_val, self.features)
        y_tr = pd.to_numeric(y_train, errors="coerce").fillna(0).astype(int)
        y_va = pd.to_numeric(y_val, errors="coerce").fillna(0).astype(int)

        best_auc = -np.inf
        best_name = ""
        best_model = None
        scores: dict[str, float] = {}

        for name, model in self._candidate_models().items():
            model.fit(x_tr, y_tr)
            pred = model.predict_proba(x_va)[:, 1]
            auc = float(roc_auc_score(y_va, pred)) if y_va.nunique() > 1 else 0.5
            scores[name] = auc
            if auc > best_auc:
                best_auc = auc
                best_name = name
                best_model = model

        if best_model is None:
            best_name = "logistic"
            best_model = self._candidate_models()["logistic"]
            best_model.fit(x_tr, y_tr)
            best_auc = 0.5

        self.model = best_model
        self.performance = MetaModelPerformance(
            best_model=best_name,
            scores=scores,
            validation_auc=float(best_auc),
        )

    def predict_proba(self, x: pd.DataFrame) -> pd.Series:
        if self.model is None or not self.features:
            return pd.Series(0.5, index=x.index, name="meta_score")

        use = [c for c in self.features if c in x.columns]
        frame = safe_numeric_frame(x, use)
        probs = self.model.predict_proba(frame)[:, 1]
        return pd.Series(np.clip(probs, 0.0, 1.0), index=x.index, name="meta_score")
