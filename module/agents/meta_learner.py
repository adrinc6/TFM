"""Meta-learner (stacking) for the multi-agent stock picker.

Combines the outputs of the configured base agents into a final Outperform prediction.

Inputs:
  Agent scores:  fundamental_score, valuation_score,
                 momentum_score, bear_score, sentiment_score
  Macro context: vix, yield_curve, sp500_momentum_3m, sp500_momentum_12m
  Sector (profiles): sector_* one-hot dummies (so the meta-learner learns
                     different agent weights per sector)

Output:
  Probability [0, 1] of Outperform + binary label
"""
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from module.agents.base import BaseAgent
from module.common.feature_controls import resolve_feature_columns
from module.steps.step_04_evaluation.explainability import AgentExplainer, build_explainer_for_agent
from environment import (
    META_LR_C, META_GBM_N_ESTIMATORS, META_GBM_MAX_DEPTH,
    META_GBM_LEARNING_RATE, META_GBM_SUBSAMPLE,
    BEAR_HARD_THRESHOLD,
    META_ENABLE_CONSENSUS_FEATURES, META_BULLISH_SCORE_THRESHOLD,
    META_ENABLE_SCORE_RECALIBRATION, META_SCORE_RECALIBRATION_TEMPERATURE,
    META_BASE_SCORE_BLEND_WEIGHT,
    META_FEATURE_COLUMNS, META_FEATURE_EXCLUDE,
    META_AGENT_SCORE_COLUMNS,
)

log = logging.getLogger(__name__)


def _score_stats_msg(name: str, s: pd.Series) -> str:
    v = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return f"[{name}] sin datos"
    return (
        f"[{name}] n={len(v)} min={v.min():.4f} q25={v.quantile(0.25):.4f} mean={v.mean():.4f} "
        f"q50={v.quantile(0.50):.4f} q75={v.quantile(0.75):.4f} max={v.max():.4f} "
        f"| >=0.50={(v >= 0.50).mean():.1%} >=0.55={(v >= 0.55).mean():.1%} >=0.60={(v >= 0.60).mean():.1%}"
    )

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (roc_auc_score, accuracy_score,
                                  f1_score, precision_score, recall_score,
                                  confusion_matrix, classification_report)
    from sklearn.model_selection import TimeSeriesSplit
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False




class MetaLearner(BaseAgent):
    """
    Two-level stacking:
      - Level 1: base agent scores (+ macro + sector dummies)
      - Level 2: Logistic Regression (interpretable, avoids overfitting)
                 + GBM (captures non-linearities)
                 → ensemble of both

    The meta-learner also detects when the Bear Agent has raised many
    red flags and applies a direct penalty (hard rule).
    """

    BEAR_HARD_THRESHOLD = BEAR_HARD_THRESHOLD   # Only intervenes in extreme cases


    def __init__(self, results_dir: str, random_seed: int = 42,
                 use_sector_features: bool = True,
                 use_macro_features: bool = True):
        super().__init__("meta_learner", results_dir, random_seed)
        if not _DEPS_OK:
            raise ImportError("scikit-learn requerido.")
        self.use_sector_features = use_sector_features
        self._lr_model:     Optional[Pipeline] = None
        self._gbm_model:    Optional[Pipeline] = None
        self._feature_cols: List[str]          = []
        self._sector_cols:  List[str]          = []
        self._lr_weight:    float              = 0.5
        self._gbm_weight:   float              = 0.5
        self._explainer:    Optional[AgentExplainer] = None
        self._use_consensus_features = bool(META_ENABLE_CONSENSUS_FEATURES)
        self._bullish_score_threshold = float(META_BULLISH_SCORE_THRESHOLD)
        self._enable_score_recalibration = bool(META_ENABLE_SCORE_RECALIBRATION)
        self._score_recal_temperature = max(float(META_SCORE_RECALIBRATION_TEMPERATURE), 1e-3)
        self._base_score_blend_weight = float(np.clip(META_BASE_SCORE_BLEND_WEIGHT, 0.0, 1.0))
        self._score_center: float = 0.5
        self._score_scale: float = 0.1

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series,
            fold: Optional[int] = None, sector_col: str = "sector") -> "MetaLearner":
        """
        X: DataFrame with agent score columns + macro + sector
        y: 1=Outperform, 0=Underperform
        """
        log.info(f"[MetaLearner] Training stacking (LR + GBM) on OOF scores — {len(X)} obs")
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len].copy()
        y = y.iloc[:min_len].copy()
        X_prep       = self._prepare(X, sector_col, fit_mode=True)
        X_prep, y_cl = self.clean_features(X_prep, y.reset_index(drop=True))
        X_prep       = X_prep.reset_index(drop=True)
        y_cl         = y_cl.reset_index(drop=True)

        if not self.has_multiple_classes(y_cl):
            log.warning("[MetaLearner] Label without enough class variance after cleaning - training skipped.")
            return self

        self._feature_cols = list(X_prep.columns)
        bal          = self.class_balance(y_cl)
        log.info(f"[MetaLearner] Balance: {bal['n_positive']} Outperform / {bal['n_negative']} Underperform ({bal['n_positive']/(bal['n_positive']+bal['n_negative']):.1%} positive)")

        # ── Logistic Regression (interpretable)
        self._lr_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                C=META_LR_C, class_weight="balanced", max_iter=1000,
                random_state=self.random_seed, solver="lbfgs",
            )),
        ])

        # ── GBM (captura interacciones)
        self._gbm_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(
                n_estimators=META_GBM_N_ESTIMATORS,
                max_depth=META_GBM_MAX_DEPTH,
                learning_rate=META_GBM_LEARNING_RATE,
                subsample=META_GBM_SUBSAMPLE,
                random_state=self.random_seed,
            )),
        ])

        # CV para calibrar pesos LR vs GBM
        cv_lr, cv_gbm = self._cv_both(X_prep, y_cl)
        auc_lr  = cv_lr["mean_auc"]
        auc_gbm = cv_gbm["mean_auc"]
        total   = auc_lr + auc_gbm
        self._lr_weight  = auc_lr  / total if total > 0 else 0.5
        self._gbm_weight = auc_gbm / total if total > 0 else 0.5
        log.info(f"[MetaLearner] CV — LR AUC={auc_lr:.4f}, GBM AUC={auc_gbm:.4f}  → ensemble weights LR={self._lr_weight:.2f} / GBM={self._gbm_weight:.2f}")

        # Final training
        self._lr_model.fit(X_prep, y_cl)
        self._gbm_model.fit(X_prep, y_cl)
        self.is_trained = True

        # Robust score calibration using in-sample train distribution.
        train_raw = self._raw_ensemble_score(X_prep)
        q1 = float(train_raw.quantile(0.25))
        q3 = float(train_raw.quantile(0.75))
        iqr = max(q3 - q1, 1e-3)
        self._score_center = float(train_raw.median())
        # Robust scale: IQR/1.349 ≈ sigma if the distribution were normal.
        self._score_scale = max(iqr / 1.349, 1e-3)
        log.info(
            "[MetaLearner] Train calibration: center=%.4f scale=%.6f (q25=%.4f q75=%.4f iqr=%.6f)",
            self._score_center,
            self._score_scale,
            q1,
            q3,
            iqr,
        )
        log.info(_score_stats_msg("MetaLearner/train_raw_ensemble", train_raw))

        # Coeficientes LR → interpretabilidad directa
        lr_coef = pd.Series(
            self._lr_model.named_steps["clf"].coef_[0],
            index=self._feature_cols,
        )
        gbm_fitted = self._gbm_model.named_steps["clf"]
        gbm_imp = pd.Series(gbm_fitted.feature_importances_, index=self._feature_cols)
        self.save_feature_importances(gbm_imp, fold)

        self._diagnostics = {
            "class_balance":    bal,
            "cv_lr":            cv_lr,
            "cv_gbm":           cv_gbm,
            "lr_weight":        self._lr_weight,
            "gbm_weight":       self._gbm_weight,
            "consensus_features_enabled": self._use_consensus_features,
            "bullish_score_threshold": self._bullish_score_threshold,
            "score_recalibration_enabled": self._enable_score_recalibration,
            "score_recalibration_temperature": self._score_recal_temperature,
            "score_center_train": self._score_center,
            "score_scale_train": self._score_scale,
            "lr_coefficients":  lr_coef.to_dict(),
            "gbm_importances":  gbm_imp.nlargest(10).to_dict(),
            "bear_hard_threshold": self.BEAR_HARD_THRESHOLD,
        }
        self.record_train_metrics({**cv_lr, **{f"gbm_{k}": v for k, v in cv_gbm.items()}}, fold)
        self.save_diagnostics(fold)
        self._save_lr_report(lr_coef, fold)

        # SHAP explainability (over GBM, more informative than LR)
        gbm_model = self._gbm_model.named_steps["clf"]
        self._explainer = build_explainer_for_agent(
            self.name, gbm_model, self._feature_cols,
            X_prep, self.results_dir.parent.as_posix(), fold, model_type="tree"
        )
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_score(self, X: pd.DataFrame, sector_col: str = "sector") -> pd.Series:
        if not self.is_trained:
            raise RuntimeError("[MetaLearner] Not trained.")
        X_prep = self.clean_features_predict(self._prepare(X, sector_col, fit_mode=False))
        X_al   = self._align(X_prep)

        score = self._raw_ensemble_score(X_al)
        score.index = X.index
        log.info(_score_stats_msg("MetaPredict/raw_ensemble", score))
        score = self._blend_with_base_consensus(score, X)
        if self._enable_score_recalibration:
            score = self._recalibrate_scores(score)
            log.info(_score_stats_msg("MetaPredict/recalibrated", score))
        score.name = "final_score"

        # Hard risk rule: high risk -> force Underperform.
        # Prefer explicit bear_risk_score; fallback to (1 - bear_score safety).
        if "bear_risk_score" in X.columns:
            bear_risk = X["bear_risk_score"].reindex(X_prep.index).fillna(0.5)
        elif "bear_score" in X.columns:
            bear_risk = (1.0 - X["bear_score"].reindex(X_prep.index).fillna(0.5)).clip(0.0, 1.0)
        else:
            bear_risk = pd.Series(0.5, index=score.index)

        if len(bear_risk) > 0:
            n_hard = int((bear_risk >= self.BEAR_HARD_THRESHOLD).sum())
            log.info(
                "[MetaPredict] Hard-risk gate threshold=%.2f -> affected=%d/%d (%.1f%%)",
                self.BEAR_HARD_THRESHOLD,
                n_hard,
                len(bear_risk),
                (n_hard / len(bear_risk) * 100.0) if len(bear_risk) else 0.0,
            )
            score = score.where(bear_risk < self.BEAR_HARD_THRESHOLD, 0.05)
            log.info(_score_stats_msg("MetaPredict/final_after_risk", score))

        return score

    def _raw_ensemble_score(self, X_al: pd.DataFrame) -> pd.Series:
        lr_p = self._lr_model.predict_proba(X_al)[:, 1]
        gbm_p = self._gbm_model.predict_proba(X_al)[:, 1]
        return pd.Series(self._lr_weight * lr_p + self._gbm_weight * gbm_p, index=X_al.index)

    def _blend_with_base_consensus(self, meta_score: pd.Series, X: pd.DataFrame) -> pd.Series:
        base_cols = list(META_AGENT_SCORE_COLUMNS)
        available = [c for c in base_cols if c in X.columns]
        if not available or self._base_score_blend_weight <= 0.0:
            return meta_score

        base_mean = (
            X[available]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .mean(axis=1, skipna=True)
            .reindex(meta_score.index)
            .fillna(0.5)
            .clip(0.0, 1.0)
        )
        w = self._base_score_blend_weight
        blended = ((1.0 - w) * meta_score + w * base_mean).clip(0.0, 1.0)
        log.info(
            "[MetaPredict] base-consensus blend weight=%.2f (meta=%.2f, base=%.2f)",
            w,
            1.0 - w,
            w,
        )
        log.info(_score_stats_msg("MetaPredict/base_consensus", base_mean))
        log.info(_score_stats_msg("MetaPredict/blended_pre_recal", blended))
        return blended

    def _recalibrate_scores(self, score: pd.Series) -> pd.Series:
        # Re-centres and re-scales relative to train to keep the 0.5 threshold interpretable.
        # A minimum scale and blend with the raw score are applied to avoid extreme collapse.
        eff_scale = max(self._score_scale, self.RECAL_MIN_SCALE)
        z = (score - self._score_center) / (eff_scale * self._score_recal_temperature)
        recal_sigmoid = 1.0 / (1.0 + np.exp(-z.clip(-10, 10)))
        recal = (1.0 - self.RECAL_BLEND_WITH_RAW) * score + self.RECAL_BLEND_WITH_RAW * recal_sigmoid
        out = pd.Series(recal.clip(0.01, 0.99), index=score.index)
        log.info(
            "[MetaLearner] Recal params: enabled=%s temp=%.3f center=%.4f scale=%.6f eff_scale=%.6f blend=%.2f",
            str(self._enable_score_recalibration),
            self._score_recal_temperature,
            self._score_center,
            self._score_scale,
            eff_scale,
            self.RECAL_BLEND_WITH_RAW,
        )
        return out

    def evaluate(self, X: pd.DataFrame, y: pd.Series,
                 sector_col: str = "sector", fold: Optional[int] = None) -> Dict:
        """Evaluate the meta-learner and save a complete report to disk."""
        scores = self.predict_score(X, sector_col)
        preds  = (scores >= 0.5).astype(int)
        y_al   = y.reindex(scores.index).dropna()
        preds  = preds.reindex(y_al.index)
        scores = scores.reindex(y_al.index)

        metrics = {
            "accuracy":  float(accuracy_score(y_al, preds)),
            "precision": float(precision_score(y_al, preds, zero_division=0)),
            "recall":    float(recall_score(y_al, preds, zero_division=0)),
            "f1":        float(f1_score(y_al, preds, zero_division=0)),
        }
        if y_al.nunique() > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_al, scores))

        cm = confusion_matrix(y_al, preds)
        report = classification_report(y_al, preds, target_names=["Underperform","Outperform"])
        log.info(
            f"[MetaLearner] Evaluation fold {fold} — "
            f"AUC={metrics.get('roc_auc',0):.4f}  Acc={metrics['accuracy']:.4f}  "
            f"Prec={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}"
        )
        log.info(f"\n{report}")

        # Save report
        suffix = f"_{fold}" if fold is not None else ""
        path   = self.results_dir / f"evaluation{suffix}.json"
        with open(path, "w") as f:
            json.dump({**metrics, "confusion_matrix": cm.tolist(),
                       "classification_report": report}, f, indent=2)

        preds_df = pd.DataFrame({"score": scores, "pred": preds, "label": y_al})
        self.save_predictions(preds_df, fold)
        return metrics

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _prepare(self, X: pd.DataFrame, sector_col: str, fit_mode: bool) -> pd.DataFrame:
        df = X.copy()
        selected = resolve_feature_columns(
            default_cols=[],
            available_cols=list(df.columns),
            include_cols=META_FEATURE_COLUMNS,
            exclude_cols=META_FEATURE_EXCLUDE,
            logger=log,
            owner="MetaLearner",
        )

        # Sector dummies (pesos distintos por sector)
        if self.use_sector_features and sector_col in df.columns:
            dummies = pd.get_dummies(df[sector_col], prefix="sector", dtype=float)
            if fit_mode:
                self._sector_cols = list(dummies.columns)
            else:
                for c in self._sector_cols:
                    if c not in dummies.columns:
                        dummies[c] = 0.0
                dummies = dummies[[c for c in self._sector_cols if c in dummies.columns]]
            df = pd.concat([df, dummies], axis=1)
            selected += self._sector_cols

        # Percentile rank of the score within each sector (relative position)
        # Captures whether the ticker is in the top quartile of its sector, not of the full universe
        if sector_col in df.columns:
            for score_col in ["fundamental_score", "valuation_score", "momentum_score"]:
                if score_col in df.columns:
                    rank_col = f"{score_col}_sector_rank"
                    df[rank_col] = df.groupby(df[sector_col])[score_col].rank(pct=True)
                    selected.append(rank_col)

        # Cross-agent interaction features (only financially meaningful pairs)
        if "fundamental_score" in df.columns and "valuation_score" in df.columns:
            df["fund_x_val"] = df["fundamental_score"] * df["valuation_score"]
            selected.append("fund_x_val")
        # mom_x_safety: high when momentum is strong and risk is low.
        if "momentum_score" in df.columns and "bear_score" in df.columns:
            df["mom_x_safety"] = df["momentum_score"] * df["bear_score"]
            selected.append("mom_x_safety")

        # Consensus/confidence signals across agents (investment-oriented).
        # Keep only the most financially meaningful consensus features to avoid
        # feature explosion that dilutes signal with noise.
        if self._use_consensus_features:
            available_agent_scores = [c for c in META_AGENT_SCORE_COLUMNS if c in df.columns]
            if available_agent_scores:
                score_mat = df[available_agent_scores].astype(float)
                score_mat = score_mat.replace([np.inf, -np.inf], np.nan)

                df["agent_score_mean"] = score_mat.mean(axis=1, skipna=True).fillna(0.5)
                df["agent_score_std"] = score_mat.std(axis=1, skipna=True).fillna(0.0)

                bullish_mask = score_mat.ge(self._bullish_score_threshold)
                df["bullish_agent_count"] = bullish_mask.sum(axis=1).astype(float)

                selected += [
                    "agent_score_mean",
                    "agent_score_std",
                    "bullish_agent_count",
                ]

        selected = list(dict.fromkeys(selected))
        result   = df[[c for c in selected if c in df.columns]].copy()
        if fit_mode:
            self._feature_cols = list(result.columns)
        return result

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        for c in self._feature_cols:
            if c not in X.columns:
                X[c] = 0.0
        return X[self._feature_cols]

    def _cv_both(self, X: pd.DataFrame, y: pd.Series):
        date_order = X.index.get_level_values("date") if "date" in X.index.names else X.index
        sort_idx = date_order.argsort()
        X = X.iloc[sort_idx]
        y = y.reindex(X.index)
        tss = TimeSeriesSplit(n_splits=5)
        lr_aucs, lr_f1s = [], []
        gbm_aucs, gbm_f1s = [], []
        for tr, val in tss.split(X):
            y_tr = y.iloc[tr]
            y_val = y.iloc[val]
            if not self.has_multiple_classes(y_tr):
                continue
            lr  = Pipeline([("s", StandardScaler()),
                            ("c", LogisticRegression(C=META_LR_C, class_weight="balanced",
                                                     max_iter=1000, random_state=self.random_seed))])
            gbm = Pipeline([("s", StandardScaler()),
                            ("c", GradientBoostingClassifier(
                                n_estimators=META_GBM_N_ESTIMATORS,
                                max_depth=META_GBM_MAX_DEPTH,
                                learning_rate=META_GBM_LEARNING_RATE,
                                subsample=META_GBM_SUBSAMPLE,
                                random_state=self.random_seed))])
            lr.fit(X.iloc[tr], y_tr)
            gbm.fit(X.iloc[tr], y_tr)
            for model, aucs, f1s in [(lr, lr_aucs, lr_f1s), (gbm, gbm_aucs, gbm_f1s)]:
                p = model.predict_proba(X.iloc[val])[:, 1]
                if self.has_multiple_classes(y_val):
                    aucs.append(roc_auc_score(y_val, p))
                f1s.append(f1_score(y_val, (p>=0.5).astype(int), zero_division=0))
        def _agg(aucs, f1s):
            return {"mean_auc": float(np.mean(aucs)) if aucs else 0.0,
                    "std_auc":  float(np.std(aucs))  if aucs else 0.0,
                    "mean_f1":  float(np.mean(f1s)) if f1s else 0.0}
        return _agg(lr_aucs, lr_f1s), _agg(gbm_aucs, gbm_f1s)

    def _save_lr_report(self, coef: pd.Series, fold: Optional[int | str]):
        suffix = f"_{fold}" if fold is not None else ""
        path   = self.results_dir / f"lr_coefficients{suffix}.json"
        sorted_coef = coef.sort_values(ascending=False)
        report = {
            "positive_drivers":  sorted_coef[sorted_coef > 0].to_dict(),
            "negative_drivers":  sorted_coef[sorted_coef < 0].to_dict(),
            "interpretation":    "Positive coefficients → contribute to Outperform",
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"[MetaLearner] LR coefficients → {path.name}")
