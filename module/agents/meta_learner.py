# =============================================================================
# module/agents/meta_learner.py — Meta-Learner (Stacking)
# =============================================================================
# Combina las salidas de los 5 agentes en una predicción final.
#
# DATOS QUE CONSUME:
#   Scores de agentes:  fundamental_score, valuation_score,
#                       momentum_score, bear_score, sentiment_score
#   Macro context:      vix, yield_curve, sp500_momentum_3m, sp500_momentum_12m
#   Sector (profiles):  sector_* (one-hot para que el meta-learner
#                       aprenda pesos distintos por sector)
#
# SALIDA:
#   Probabilidad [0,1] de Outperform + etiqueta binaria
# =============================================================================
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from module.agents.base import BaseAgent
from module.steps.step_04_evaluation.explainability import AgentExplainer, build_explainer_for_agent
from environment import (
    META_LR_C, META_GBM_N_ESTIMATORS, META_GBM_MAX_DEPTH,
    META_GBM_LEARNING_RATE, META_GBM_SUBSAMPLE,
    BEAR_HARD_THRESHOLD,
)

log = logging.getLogger(__name__)

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

AGENT_SCORE_COLS = [
    "fundamental_score",
    "valuation_score",
    "momentum_score",
    "bear_score",        # Se usará como (1 - bear_score) en _prepare
    "sentiment_score",   # Consenso analistas + insiders + beat rate
    "sector_score",      # Top-down: probabilidad de que el sector supere al S&P
]

MACRO_COLS: list = []  # Macro eliminada: es igual para todos los tickers y enmascara señales de stock picking


class MetaLearner(BaseAgent):
    """
    Stacking de dos niveles:
      - Nivel 1: scores de los 4 agentes (+ macro + sector dummies)
      - Nivel 2: Logistic Regression (interpretable, evita overfitting)
                 + GBM (captura no-linealidades)
                 → ensemble de ambos

    El meta-learner también detecta si el Bear Agent ha activado muchas
    red flags y aplica un penalizador directo (hard rule).
    """

    BEAR_HARD_THRESHOLD = BEAR_HARD_THRESHOLD   # Solo interviene en casos extremos
    BEAR_SOFT_PENALTY   = 0.0                   # Desactivado: evita doble penalización

    def __init__(self, results_dir: str, random_seed: int = 42,
                 use_sector_features: bool = True,
                 use_macro_features: bool = True):
        super().__init__("meta_learner", results_dir, random_seed)
        if not _DEPS_OK:
            raise ImportError("scikit-learn requerido.")
        self.use_sector_features = use_sector_features
        self.use_macro_features  = use_macro_features
        self._lr_model:     Optional[Pipeline] = None
        self._gbm_model:    Optional[Pipeline] = None
        self._feature_cols: List[str]          = []
        self._sector_cols:  List[str]          = []
        self._lr_weight:    float              = 0.5
        self._gbm_weight:   float              = 0.5
        self._explainer:    Optional[AgentExplainer] = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series,
            fold: Optional[int] = None, sector_col: str = "sector") -> "MetaLearner":
        """
        X: DataFrame con columnas de scores de agentes + macro + sector
        y: 1=Outperform, 0=Underperform
        """
        log.info(f"[MetaLearner] Entrenando stacking (LR + GBM) sobre scores OOF — {len(X)} obs")
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len].copy()
        y = y.iloc[:min_len].copy()
        X_prep       = self._prepare(X, sector_col, fit_mode=True)
        X_prep, y_cl = self.clean_features(X_prep, y.reset_index(drop=True))
        X_prep       = X_prep.reset_index(drop=True)
        y_cl         = y_cl.reset_index(drop=True)
        self._feature_cols = list(X_prep.columns)
        bal          = self.class_balance(y_cl)
        log.info(f"[MetaLearner] Balance: {bal['n_positive']} Outperform / {bal['n_negative']} Underperform ({bal['n_positive']/(bal['n_positive']+bal['n_negative']):.1%} positivos)")

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
        log.info(f"[MetaLearner] CV — LR AUC={auc_lr:.4f}, GBM AUC={auc_gbm:.4f}  → ensemble pesos LR={self._lr_weight:.2f} / GBM={self._gbm_weight:.2f}")

        # Entrenamiento final
        self._lr_model.fit(X_prep, y_cl)
        self._gbm_model.fit(X_prep, y_cl)
        self.is_trained = True

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
            "lr_coefficients":  lr_coef.to_dict(),
            "gbm_importances":  gbm_imp.nlargest(10).to_dict(),
            "bear_hard_threshold": self.BEAR_HARD_THRESHOLD,
        }
        self.record_train_metrics({**cv_lr, **{f"gbm_{k}": v for k, v in cv_gbm.items()}}, fold)
        self.save_diagnostics(fold)
        self._save_lr_report(lr_coef, fold)

        # Explicabilidad SHAP (sobre el GBM, más informativo que LR)
        gbm_model = self._gbm_model.named_steps["clf"]
        self._explainer = build_explainer_for_agent(
            self.name, gbm_model, self._feature_cols,
            X_prep, self.results_dir.parent.as_posix(), fold, model_type="tree"
        )
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_score(self, X: pd.DataFrame, sector_col: str = "sector") -> pd.Series:
        if not self.is_trained:
            raise RuntimeError("[MetaLearner] No entrenado.")
        X_prep = self.clean_features_predict(self._prepare(X, sector_col, fit_mode=False))
        X_al   = self._align(X_prep)

        lr_p  = self._lr_model.predict_proba(X_al)[:, 1]
        gbm_p = self._gbm_model.predict_proba(X_al)[:, 1]
        score = pd.Series(
            self._lr_weight * lr_p + self._gbm_weight * gbm_p,
            index=X.index, name="final_score",
        )

        # Hard rule: Bear muy alto → fuerza Underperform
        if "bear_score" in X.columns:
            bear = X["bear_score"].reindex(X_prep.index).fillna(0.5)
            score = score.where(bear < self.BEAR_HARD_THRESHOLD, 0.05)
            # Soft penalty proporcional al bear_score
            penalty = (bear * self.BEAR_SOFT_PENALTY).clip(0, 0.4)
            score   = (score - penalty).clip(0.0, 1.0)

        return score

    def evaluate(self, X: pd.DataFrame, y: pd.Series,
                 sector_col: str = "sector", fold: Optional[int] = None) -> Dict:
        """Evalúa el meta-learner y guarda reporte completo."""
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
            f"[MetaLearner] Evaluación fold {fold} — "
            f"AUC={metrics.get('roc_auc',0):.4f}  Acc={metrics['accuracy']:.4f}  "
            f"Prec={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}"
        )
        log.info(f"\n{report}")

        # Guardar reporte
        suffix = f"_fold{fold}" if fold is not None else ""
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
        # bear_score invertido: alto bear = alto riesgo = queremos bajo input para el MetaLearner
        if "bear_score" in df.columns:
            df["bear_score"] = 1.0 - df["bear_score"]
        selected = [c for c in AGENT_SCORE_COLS if c in df.columns]

        # Macro features
        if self.use_macro_features:
            selected += [c for c in MACRO_COLS if c in df.columns]

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

        # Ranking percentil del final_score dentro del sector (posición relativa)
        # Captura si el ticker está en el cuartil alto de su sector, no del universo completo
        if sector_col in df.columns:
            for score_col in ["fundamental_score", "valuation_score", "momentum_score", "sentiment_score"]:
                if score_col in df.columns:
                    rank_col = f"{score_col}_sector_rank"
                    df[rank_col] = df.groupby(df[sector_col])[score_col].rank(pct=True)
                    selected.append(rank_col)

        # Features de interacción entre agentes
        if "fundamental_score" in df.columns and "valuation_score" in df.columns:
            df["fund_x_val"] = df["fundamental_score"] * df["valuation_score"]
            selected.append("fund_x_val")
        # mom_x_safety: alto cuando momentum es alto Y riesgo es bajo.
        # bear_score ya está invertido (= safety score) así que el producto es correcto.
        if "momentum_score" in df.columns and "bear_score" in df.columns:
            df["mom_x_safety"] = df["momentum_score"] * df["bear_score"]
            selected.append("mom_x_safety")
        if "sentiment_score" in df.columns and "fundamental_score" in df.columns:
            df["fund_x_sentiment"] = df["fundamental_score"] * df["sentiment_score"]
            selected.append("fund_x_sentiment")
        if "sentiment_score" in df.columns and "momentum_score" in df.columns:
            df["mom_x_sentiment"] = df["momentum_score"] * df["sentiment_score"]
            selected.append("mom_x_sentiment")

        # Interacción sector × stock: ticker fuerte en sector fuerte = señal potente
        if "sector_score" in df.columns and "fundamental_score" in df.columns:
            df["sector_x_fundamental"] = df["sector_score"] * df["fundamental_score"]
            selected.append("sector_x_fundamental")
        if "sector_score" in df.columns and "momentum_score" in df.columns:
            df["sector_x_momentum"] = df["sector_score"] * df["momentum_score"]
            selected.append("sector_x_momentum")

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
            lr.fit(X.iloc[tr], y.iloc[tr])
            gbm.fit(X.iloc[tr], y.iloc[tr])
            for model, aucs, f1s in [(lr, lr_aucs, lr_f1s), (gbm, gbm_aucs, gbm_f1s)]:
                p = model.predict_proba(X.iloc[val])[:, 1]
                if y.iloc[val].nunique() > 1:
                    aucs.append(roc_auc_score(y.iloc[val], p))
                f1s.append(f1_score(y.iloc[val], (p>=0.5).astype(int), zero_division=0))
        def _agg(aucs, f1s):
            return {"mean_auc": float(np.mean(aucs)) if aucs else 0.0,
                    "std_auc":  float(np.std(aucs))  if aucs else 0.0,
                    "mean_f1":  float(np.mean(f1s))}
        return _agg(lr_aucs, lr_f1s), _agg(gbm_aucs, gbm_f1s)

    def _save_lr_report(self, coef: pd.Series, fold: Optional[int]):
        suffix = f"_fold{fold}" if fold is not None else ""
        path   = self.results_dir / f"lr_coefficients{suffix}.json"
        sorted_coef = coef.sort_values(ascending=False)
        report = {
            "positive_drivers":  sorted_coef[sorted_coef > 0].to_dict(),
            "negative_drivers":  sorted_coef[sorted_coef < 0].to_dict(),
            "interpretation":    "Coeficientes positivos → contribuyen a Outperform",
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"[MetaLearner] LR coeficientes → {path.name}")
