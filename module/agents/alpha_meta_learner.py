"""Alpha-oriented meta learner with regression + ranking outputs."""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from module.agents.base import BaseAgent
from environment import (
    META_FEATURE_COLUMNS,
    META_FEATURE_EXCLUDE,
    ALPHA_META_REG_N_ESTIMATORS,
    ALPHA_META_REG_MAX_DEPTH,
    ALPHA_META_REG_LEARNING_RATE,
    ALPHA_META_REG_SUBSAMPLE,
    ALPHA_META_REG_COLSAMPLE,
    ALPHA_META_RANK_N_ESTIMATORS,
    ALPHA_META_RANK_MAX_DEPTH,
    ALPHA_META_RANK_LEARNING_RATE,
    ALPHA_META_RANK_SUBSAMPLE,
    ALPHA_META_RANK_COLSAMPLE,
    ALPHA_META_RISK_N_ESTIMATORS,
    ALPHA_META_RISK_MAX_DEPTH,
    ALPHA_META_RISK_LEARNING_RATE,
    ALPHA_META_RISK_SUBSAMPLE,
    ALPHA_META_RISK_COLSAMPLE,
    ALPHA_META_RANK_BLEND,
    ALPHA_META_RISK_BLEND,
)

log = logging.getLogger(__name__)

try:
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error
    _DEPS_OK = True
except Exception:
    _DEPS_OK = False


class AlphaMetaLearner(BaseAgent):
    """Meta model that predicts alpha and ranking-ready portfolio signals."""

    def __init__(self, results_dir: str, random_seed: int = 42):
        super().__init__("alpha_meta_learner", results_dir, random_seed)
        if not _DEPS_OK:
            raise ImportError("xgboost and scikit-learn are required for AlphaMetaLearner")
        self._feature_cols: List[str] = []
        self._feature_medians: Optional[pd.Series] = None
        self._sector_cols: List[str] = []
        self._reg_model = None
        self._rank_model = None
        self._risk_model = None
        self._risk_blend = float(np.clip(ALPHA_META_RISK_BLEND, 0.0, 1.0))

    def _prepare(self, X: pd.DataFrame, sector_col: str = "sector", fit_mode: bool = False) -> pd.DataFrame:
        df = X.copy()

        selected = [c for c in META_FEATURE_COLUMNS if c in df.columns and c not in META_FEATURE_EXCLUDE]

        # Full-feature meta model: keep all numeric non-target columns.
        forbidden = {
            "label",
            "forward_return",
            "target_alpha",
            "target_quintile",
            "target_triple_barrier",
            "predicted_alpha",
            "ranking_score",
            "risk_score",
            "regime_adjusted_score",
            "final_score",
        }
        numeric_cols = []
        for c in df.columns:
            if c in forbidden:
                continue
            try:
                dtype = getattr(df[c], "dtype", np.dtype("float64"))
                if np.issubdtype(dtype, np.number):
                    numeric_cols.append(c)
            except (TypeError, ValueError):
                # Skip columns with incompatible dtypes (e.g., StringDtype)
                pass
        selected += numeric_cols

        if sector_col in df.columns:
            dummies = pd.get_dummies(df[sector_col].astype(str), prefix="sector", dtype=float)
            if fit_mode:
                self._sector_cols = list(dummies.columns)
            else:
                for c in self._sector_cols:
                    if c not in dummies.columns:
                        dummies[c] = 0.0
                dummies = dummies[[c for c in self._sector_cols if c in dummies.columns]]
            df = pd.concat([df, dummies], axis=1)
            selected += list(dummies.columns)

        if fit_mode:
            selected = list(dict.fromkeys([c for c in selected if c in df.columns]))
            out = df[selected].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            medians = out.median(numeric_only=True).reindex(out.columns).fillna(0.0)
            out = out.fillna(medians).fillna(0.0)
            self._feature_cols = list(out.columns)
            self._feature_medians = medians.astype(float)
            return out.astype(np.float32)

        # Inference path: strictly reuse training feature schema for speed and stability.
        aligned_cols = list(self._feature_cols) if self._feature_cols else list(
            dict.fromkeys([c for c in selected if c in df.columns])
        )
        out = df.reindex(columns=aligned_cols).apply(pd.to_numeric, errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan)

        if self._feature_medians is not None and not self._feature_medians.empty:
            out = out.fillna(self._feature_medians.reindex(out.columns))
        else:
            out = out.fillna(out.median(numeric_only=True).reindex(out.columns))

        out = out.fillna(0.0)
        return out.astype(np.float32)

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c in self._feature_cols:
            if c not in out.columns:
                out[c] = 0.0
        out = out[self._feature_cols].replace([np.inf, -np.inf], np.nan)
        if self._feature_medians is not None and not self._feature_medians.empty:
            out = out.fillna(self._feature_medians.reindex(out.columns))
        return out.fillna(0.0).astype(np.float32)

    @staticmethod
    def _group_sizes_from_index(idx: pd.Index) -> List[int]:
        if isinstance(idx, pd.MultiIndex) and "date" in idx.names:
            g = pd.Series(idx.get_level_values("date")).value_counts(sort=False)
            return g.tolist()
        g = pd.Series(idx).value_counts(sort=False)
        return g.tolist()

    @staticmethod
    def _cross_sectional_rank(idx: pd.Index, values: np.ndarray) -> pd.Series:
        s = pd.Series(values, index=idx, dtype=float)
        if isinstance(idx, pd.MultiIndex) and "date" in idx.names:
            return s.groupby(idx.get_level_values("date")).rank(pct=True, method="average")
        return s.rank(pct=True, method="average")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fold: Optional[int] = None,
        sector_col: str = "sector",
        target_alpha: Optional[pd.Series] = None,
    ) -> "AlphaMetaLearner":
        # Handle numpy arrays
        if isinstance(y, np.ndarray):
            y = pd.Series(y, index=X.index)
        if isinstance(target_alpha, np.ndarray):
            target_alpha = pd.Series(target_alpha, index=X.index)
        
        X_prep = self._prepare(X, sector_col=sector_col, fit_mode=True)

        y_cls = pd.to_numeric(y.reindex(X_prep.index), errors="coerce").fillna(0.0)
        alpha = pd.to_numeric((target_alpha if target_alpha is not None else (y_cls - 0.5)), errors="coerce")
        alpha = alpha.reindex(X_prep.index).fillna(alpha.median() if np.isfinite(alpha.median()) else 0.0)

        self._reg_model = xgb.XGBRegressor(
            n_estimators=ALPHA_META_REG_N_ESTIMATORS,
            max_depth=ALPHA_META_REG_MAX_DEPTH,
            learning_rate=ALPHA_META_REG_LEARNING_RATE,
            subsample=ALPHA_META_REG_SUBSAMPLE,
            colsample_bytree=ALPHA_META_REG_COLSAMPLE,
            objective="reg:squarederror",
            random_state=self.random_seed,
            n_jobs=-1,
            tree_method="hist",
        )
        self._reg_model.fit(X_prep, alpha)

        groups = self._group_sizes_from_index(X_prep.index)
        if len(groups) > 1:
            self._rank_model = xgb.XGBRanker(
                n_estimators=ALPHA_META_RANK_N_ESTIMATORS,
                max_depth=ALPHA_META_RANK_MAX_DEPTH,
                learning_rate=ALPHA_META_RANK_LEARNING_RATE,
                subsample=ALPHA_META_RANK_SUBSAMPLE,
                colsample_bytree=ALPHA_META_RANK_COLSAMPLE,
                objective="rank:pairwise",
                random_state=self.random_seed,
                n_jobs=-1,
                tree_method="hist",
            )
            self._rank_model.fit(X_prep, alpha, group=groups)

        self._risk_model = xgb.XGBClassifier(
            n_estimators=ALPHA_META_RISK_N_ESTIMATORS,
            max_depth=ALPHA_META_RISK_MAX_DEPTH,
            learning_rate=ALPHA_META_RISK_LEARNING_RATE,
            subsample=ALPHA_META_RISK_SUBSAMPLE,
            colsample_bytree=ALPHA_META_RISK_COLSAMPLE,
            objective="binary:logistic",
            random_state=self.random_seed,
            n_jobs=-1,
            tree_method="hist",
        )
        self._risk_model.fit(X_prep, (y_cls > 0.5).astype(int))

        self.is_trained = True
        self._diagnostics = {
            "n_features": int(len(self._feature_cols)),
            "alpha_mean_train": float(alpha.mean()),
            "alpha_std_train": float(alpha.std()),
            "rank_blend": float(ALPHA_META_RANK_BLEND),
            "risk_blend": float(self._risk_blend),
        }
        self.save_diagnostics(fold)
        return self

    def predict_components(self, X: pd.DataFrame, sector_col: str = "sector") -> pd.DataFrame:
        if not self.is_trained:
            raise RuntimeError("[AlphaMetaLearner] Not trained")

        X_prep = self._align(self._prepare(X, sector_col=sector_col, fit_mode=False))
        pred_alpha = self._reg_model.predict(X_prep)

        if self._rank_model is not None:
            rank_raw = self._rank_model.predict(X_prep)
        else:
            rank_raw = pred_alpha

        ranking_score = self._cross_sectional_rank(X.index, rank_raw).clip(0.0, 1.0)
        risk_raw = 1.0 - self._risk_model.predict_proba(X_prep)[:, 1]
        risk_score = pd.Series(risk_raw, index=X.index, dtype=float).clip(0.0, 1.0)

        if "regime_adjusted_score" in X.columns:
            regime_adj = pd.to_numeric(X["regime_adjusted_score"], errors="coerce").reindex(X.index).fillna(0.5)
            regime_rank_blend = (
                ALPHA_META_RANK_BLEND * ranking_score + (1.0 - ALPHA_META_RANK_BLEND) * regime_adj
            ).clip(0.0, 1.0)
        else:
            regime_rank_blend = ranking_score

        # Risk-aware calibration of final score: blend ranking signal with
        # risk-model bullish probability (1 - risk_score).
        risk_bullish = (1.0 - risk_score).clip(0.0, 1.0)
        regime_adjusted_score = (
            (1.0 - self._risk_blend) * regime_rank_blend + self._risk_blend * risk_bullish
        ).clip(0.0, 1.0)

        out = pd.DataFrame(
            {
                "predicted_alpha": pd.Series(pred_alpha, index=X.index, dtype=float),
                "ranking_score": ranking_score,
                "risk_score": risk_score,
                "regime_adjusted_score": regime_adjusted_score,
            },
            index=X.index,
        )
        return out

    def predict_score(self, X: pd.DataFrame, sector_col: str = "sector") -> pd.Series:
        comps = self.predict_components(X, sector_col=sector_col)
        return comps["regime_adjusted_score"].rename("final_score")

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sector_col: str = "sector",
        fold: Optional[int] = None,
        target_alpha: Optional[pd.Series] = None,
    ) -> Dict:
        comps = self.predict_components(X, sector_col=sector_col)
        score = comps["regime_adjusted_score"]
        y_al = pd.to_numeric(y.reindex(score.index), errors="coerce").dropna()
        score = score.reindex(y_al.index)

        pred = (score >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_al, pred)),
            "f1": float(f1_score(y_al, pred, zero_division=0)),
        }
        if y_al.nunique() > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_al, score))

        alpha_true = pd.to_numeric((target_alpha if target_alpha is not None else y_al - 0.5), errors="coerce")
        alpha_true = alpha_true.reindex(comps.index).dropna()
        alpha_pred = comps.loc[alpha_true.index, "predicted_alpha"]
        if len(alpha_true) > 1:
            metrics["alpha_rmse"] = float(mean_squared_error(alpha_true, alpha_pred) ** 0.5)
            metrics["alpha_spearman_ic"] = float(alpha_true.corr(alpha_pred, method="spearman"))

        suffix = f"_{fold}" if fold is not None else ""
        out_path = self.results_dir / f"evaluation{suffix}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        return metrics
