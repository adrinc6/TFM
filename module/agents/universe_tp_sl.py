"""Universal TP/SL agent with dynamic feature selection and model search.

This agent is designed for the target:
  P(TP hit before SL within horizon)

Key behavior:
  - Trains on the full universe (no per-sector model split).
  - Selects between 6 and 10 base metrics from agent-specific candidates.
  - Builds percentile transforms for each selected metric:
      * within sector (same snapshot date)
      * within universe (same snapshot date)
  - Tries multiple model families and keeps the best by temporal CV AUC.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from module.agents.base import BaseAgent

log = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    _SK_OK = True
except Exception:
    _SK_OK = False

try:
    import xgboost as xgb

    _XGB_OK = True
except Exception:
    _XGB_OK = False


class UniversalTpSlAgent(BaseAgent):
    """Single-universe TP/SL classifier with percentile-aware feature engineering."""

    def __init__(
        self,
        name: str,
        results_dir: str,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        random_seed: int = 42,
        min_features: int = 6,
        max_features: int = 10,
        neutral_score: float = 0.5,
        save_artifacts: bool = True,
    ):
        super().__init__(name=name, results_dir=results_dir, random_seed=random_seed, save_artifacts=save_artifacts)
        if not _SK_OK:
            raise ImportError("scikit-learn is required for UniversalTpSlAgent")

        self.include_features = list(include_features or [])
        self.exclude_features = set(exclude_features or [])
        self.min_features = int(max(3, min_features))
        self.max_features = int(max(self.min_features, max_features))
        self._neutral_score = float(neutral_score)

        self._selected_base_features: List[str] = []
        self._feature_cols: List[str] = []
        self._model = None
        self._best_model_name: str = ""
        self._cv_summary: Dict[str, Dict[str, float]] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fold: Optional[int] = None,
        sector_col: str = "sector",
    ) -> "UniversalTpSlAgent":
        if X is None or X.empty:
            return self

        X_raw = self._prepare_base_candidate_frame(X)
        X_raw, y_aligned = self.clean_features(X_raw, y)
        if X_raw.empty or y_aligned is None or len(X_raw) < 30:
            log.warning("[%s] Not enough clean samples to train.", self.name)
            return self
        if not self.has_multiple_classes(y_aligned):
            log.warning("[%s] Target has a single class in this fold.", self.name)
            return self

        self._selected_base_features = self._select_base_features(X_raw, y_aligned)
        if not self._selected_base_features:
            log.warning("[%s] No base features selected.", self.name)
            return self

        sector_series = X.loc[X_raw.index, sector_col] if sector_col in X.columns else None
        X_model = self._build_model_frame(X_raw[self._selected_base_features], sector_series)
        X_model, y_aligned = self.clean_features(X_model, y_aligned)

        if X_model.empty or len(X_model) < 30 or not self.has_multiple_classes(y_aligned):
            log.warning("[%s] Feature frame invalid after percentile engineering.", self.name)
            return self

        self._feature_cols = list(X_model.columns)
        best_name, best_model, cv_summary = self._choose_best_model(X_model, y_aligned)
        self._cv_summary = cv_summary
        self._best_model_name = best_name

        if best_model is None:
            log.warning("[%s] Model search failed.", self.name)
            return self

        best_model.fit(X_model, y_aligned)
        self._model = best_model
        self.is_trained = True

        importances = self._model_importance_series(self._model, self._feature_cols)
        self.save_feature_importances(importances, fold)

        self._diagnostics = {
            "selected_base_features": self._selected_base_features,
            "n_base_features": int(len(self._selected_base_features)),
            "n_model_features": int(len(self._feature_cols)),
            "best_model": self._best_model_name,
            "cv_summary": self._cv_summary,
            "top_features": importances.nlargest(15).to_dict(),
        }
        self.record_train_metrics(
            {
                "best_cv_auc": float(self._cv_summary.get(self._best_model_name, {}).get("mean_auc", 0.0)),
                "n_features": float(len(self._feature_cols)),
            },
            fold=fold,
        )
        self.save_diagnostics(fold)
        return self

    def predict_score(self, X: pd.DataFrame, sector_col: str = "sector") -> pd.Series:
        if not self.is_trained or self._model is None:
            raise RuntimeError(f"[{self.name}] Not trained.")

        X_raw = self._prepare_base_candidate_frame(X)
        for col in self._selected_base_features:
            if col not in X_raw.columns:
                X_raw[col] = 0.0
        X_raw = X_raw[self._selected_base_features]
        X_raw = self.clean_features_predict(X_raw)

        sector_series = X.loc[X_raw.index, sector_col] if sector_col in X.columns else None
        X_model = self._build_model_frame(X_raw, sector_series)
        X_model = self.clean_features_predict(X_model)
        X_model = self._align_to_feature_cols(X_model, fill_value=0.0)

        if hasattr(self._model, "predict_proba"):
            score = self._model.predict_proba(X_model)[:, 1]
        else:
            raw = self._model.decision_function(X_model)
            score = 1.0 / (1.0 + np.exp(-raw))

        score = self._apply_calibration(score)
        return pd.Series(np.clip(score, 0.0, 1.0), index=X_model.index, name=f"{self.name}_score")

    def _prepare_base_candidate_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.include_features if c in X.columns and c not in self.exclude_features]
        if not available:
            numeric_cols = [
                c for c in X.columns
                if c not in self.exclude_features and pd.api.types.is_numeric_dtype(X[c])
            ]
            available = numeric_cols
        return X[available].copy()

    def _select_base_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        if X.empty:
            return []

        # Rank by absolute point-biserial proxy (Pearson on binary y).
        scores: Dict[str, float] = {}
        y_num = pd.to_numeric(y, errors="coerce")
        for col in X.columns:
            s = pd.to_numeric(X[col], errors="coerce")
            valid = s.notna() & y_num.notna()
            if valid.sum() < 20:
                continue
            corr = s[valid].corr(y_num[valid])
            if pd.notna(corr):
                scores[col] = float(abs(corr))

        ranked = [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
        if not ranked:
            ranked = list(X.columns)

        keep = int(min(self.max_features, max(self.min_features, min(len(ranked), self.max_features))))
        selected = ranked[:keep]

        if len(selected) < self.min_features:
            for col in X.columns:
                if col not in selected:
                    selected.append(col)
                if len(selected) >= self.min_features:
                    break
        return selected[: self.max_features]

    def _build_model_frame(
        self,
        X_base: pd.DataFrame,
        sector_series: Optional[pd.Series],
    ) -> pd.DataFrame:
        out = X_base.copy()
        date_key = self._resolve_date_key(X_base.index)

        for col in list(X_base.columns):
            out[f"{col}__pct_universe"] = self._percentile_by_group(X_base[col], date_key)

            if sector_series is not None:
                sec = sector_series.reindex(X_base.index).fillna("Unknown").astype(str)
                sec_key = pd.Series(date_key.astype(str), index=X_base.index).astype(str) + "|" + sec
                out[f"{col}__pct_sector"] = self._percentile_by_group(X_base[col], sec_key)

        return out

    @staticmethod
    def _resolve_date_key(index: pd.Index) -> pd.Series:
        if isinstance(index, pd.MultiIndex) and "date" in index.names:
            return pd.Series(index.get_level_values("date"), index=index)
        return pd.Series(index, index=index)

    @staticmethod
    def _percentile_by_group(values: pd.Series, group_key: pd.Series) -> pd.Series:
        df = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"), "g": group_key})
        ranked = df.groupby("g")["v"].rank(pct=True, method="average")
        return ranked.fillna(0.5).astype(float)

    def _choose_best_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[str, Optional[object], Dict[str, Dict[str, float]]]:
        candidates = self._model_candidates()
        summary: Dict[str, Dict[str, float]] = {}

        best_name = ""
        best_model = None
        best_auc = -1.0

        for name, model in candidates.items():
            mean_auc, std_auc = self._temporal_cv_auc(model, X, y)
            summary[name] = {"mean_auc": float(mean_auc), "std_auc": float(std_auc)}
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_name = name
                best_model = model

        return best_name, best_model, summary

    def _model_candidates(self) -> Dict[str, object]:
        models: Dict[str, object] = {
            "logistic": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=400,
                            class_weight="balanced",
                            random_state=self.random_seed,
                        ),
                    ),
                ]
            ),
            # min_samples_leaf=10 consistently applied across all tree-based
            # models to enforce the same regularization floor: at least 10
            # training samples must be in every leaf.  This prevents individual
            # tickers/quarters from dominating splits and is the key guard
            # against overfitting on the ~5Y quarterly dataset.
            "random_forest": RandomForestClassifier(
                n_estimators=320,
                max_depth=6,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=self.random_seed,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=220,
                max_depth=2,
                learning_rate=0.04,
                subsample=0.80,
                min_samples_leaf=10,
                random_state=self.random_seed,
            ),
        }

        if _XGB_OK:
            models["xgboost"] = xgb.XGBClassifier(
                n_estimators=260,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.80,
                colsample_bytree=0.75,
                reg_lambda=2.0,
                reg_alpha=0.5,
                min_child_weight=10,
                eval_metric="auc",
                random_state=self.random_seed,
                n_jobs=-1,
                tree_method="hist",
            )

        return models

    def _temporal_cv_auc(self, model: object, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float]:
        if len(X) < 50:
            return 0.0, 0.0

        X_sorted, y_sorted = self._sort_by_time(X, y)
        n_splits = max(2, min(5, len(X_sorted) // 30))
        if len(X_sorted) <= n_splits:
            return 0.0, 0.0

        tscv = TimeSeriesSplit(n_splits=n_splits)
        aucs: List[float] = []

        for tr, val in tscv.split(X_sorted):
            y_tr = y_sorted.iloc[tr]
            y_val = y_sorted.iloc[val]
            if y_tr.nunique() < 2 or y_val.nunique() < 2:
                continue
            model.fit(X_sorted.iloc[tr], y_tr)
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X_sorted.iloc[val])[:, 1]
            else:
                raw = model.decision_function(X_sorted.iloc[val])
                p = 1.0 / (1.0 + np.exp(-raw))
            aucs.append(float(roc_auc_score(y_val, p)))

        if not aucs:
            return 0.0, 0.0
        return float(np.mean(aucs)), float(np.std(aucs))

    @staticmethod
    def _sort_by_time(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        if isinstance(X.index, pd.MultiIndex) and "date" in X.index.names:
            order = pd.Series(X.index.get_level_values("date")).argsort()
            Xs = X.iloc[order]
            ys = y.reindex(Xs.index)
            return Xs, ys
        return X, y.reindex(X.index)

    @staticmethod
    def _model_importance_series(model: object, feature_cols: List[str]) -> pd.Series:
        if isinstance(model, Pipeline):
            inner = model.named_steps.get("clf")
        else:
            inner = model

        if inner is not None and hasattr(inner, "feature_importances_"):
            arr = np.asarray(getattr(inner, "feature_importances_"), dtype=float)
            return pd.Series(arr, index=feature_cols).fillna(0.0)

        if inner is not None and hasattr(inner, "coef_"):
            coef = np.asarray(getattr(inner, "coef_"), dtype=float)
            if coef.ndim == 2:
                coef = np.abs(coef[0])
            else:
                coef = np.abs(coef)
            return pd.Series(coef, index=feature_cols).fillna(0.0)

        return pd.Series(np.zeros(len(feature_cols), dtype=float), index=feature_cols)
