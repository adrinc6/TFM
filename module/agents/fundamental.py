# =============================================================================
# module/agents/fundamental_agent.py — Agente Fundamental (XGBoost)
# =============================================================================
# Evalúa la salud financiera de una empresa.
#
# DATOS QUE CONSUME (del consolidated + feature_engineering):
#   Rentabilidad:  roe, roa, roi, roic, net_margin, gross_margin,
#                  fcf_margin, ebitda_margin, operating_margin
#   Liquidez:      current_ratio, quick_ratio
#   Solvencia:     debt_equity, debt_to_ebitda, interest_coverage
#   Crecimiento:   revenue_yoy_growth, net_income_yoy_growth, eps_yoy_growth,
#                  fcf_yoy_growth, operating_income_yoy_growth
#   Calidad:       accruals_ratio, capex_to_revenue, consecutive_losses
#   Sector Z:      *_zsector  (columnas añadidas por SectorNormalizer)
#   Sector dummy:  sector_*   (one-hot del sector de companies.csv)
# =============================================================================
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from module.agents.base import BaseAgent, FeatureSelector
from module.steps.step_04_evaluation.explainability import build_explainer_for_agent, AgentExplainer
from environment import (
    FUNDAMENTAL_N_ESTIMATORS, FUNDAMENTAL_MAX_DEPTH, FUNDAMENTAL_LEARNING_RATE,
    FUNDAMENTAL_SUBSAMPLE, FUNDAMENTAL_COLSAMPLE, FUNDAMENTAL_MIN_CHILD_WEIGHT,
    FEATURE_CORR_THRESHOLD, FEATURE_TOP_N,
)

log = logging.getLogger(__name__)

try:
    import xgboost as xgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    from sklearn.model_selection import TimeSeriesSplit
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False
    log.error("[FundamentalAgent] Instala: pip install xgboost scikit-learn")

# Features base que consume este agente.
# Fuente: FinancialDataConsolidator → FundamentalFeatureBuilder.
# Columnas que falten se rellenan con 0 en _align (sin romper la predicción).
FEATURE_COLS = [
    # Rentabilidad
    "roe", "roa", "roi", "roic",
    "net_margin", "gross_margin", "fcf_margin", "ebitda_margin", "operating_margin",
    # Liquidez / Solvencia
    "current_ratio", "quick_ratio",
    "debt_equity", "debt_to_ebitda", "interest_coverage",
    # Crecimiento YoY (calculado en FundamentalFeatureBuilder._yoy_growth)
    "revenue_yoy_growth", "net_income_yoy_growth", "eps_yoy_growth",
    "fcf_yoy_growth", "operating_income_yoy_growth", "total_debt_yoy_growth",
    # Calidad contable / eficiencia
    "accruals_ratio", "capex_to_revenue", "consecutive_losses",
    # EPS (base para cálculo de P/E en ValuationAgent)
    "eps",
    # Nota: las columnas _zsector y sector_* las añade _prepare dinámicamente.
]


class FundamentalAgent(BaseAgent):
    """
    XGBoost calibrado (Isotonic) que aprende qué combinación de ratios
    financieros predice mejor el Outperform a 1 año.

    El sector de companies.csv entra de dos formas:
      1. Como dummies one-hot (sector_Technology, sector_Healthcare, ...)
      2. Como ratios normalizados sectorialmente (_zsector)
    """

    def __init__(self, results_dir: str, random_seed: int = 42,
                 n_estimators: int = FUNDAMENTAL_N_ESTIMATORS,
                 max_depth: int = FUNDAMENTAL_MAX_DEPTH,
                 learning_rate: float = FUNDAMENTAL_LEARNING_RATE,
                 subsample: float = FUNDAMENTAL_SUBSAMPLE,
                 colsample_bytree: float = FUNDAMENTAL_COLSAMPLE,
                 min_child_weight: int = FUNDAMENTAL_MIN_CHILD_WEIGHT):
        super().__init__("fundamental", results_dir, random_seed)
        if not _DEPS_OK:
            raise ImportError("xgboost y scikit-learn requeridos.")
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.learning_rate    = learning_rate
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self._model:          Optional[CalibratedClassifierCV] = None
        self._feature_cols:   List[str] = []
        self._sector_dummies: List[str] = []
        self._selector:       Optional[FeatureSelector] = None
        self._explainer:      Optional[AgentExplainer] = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series,
            fold: Optional[int] = None, sector_col: str = "sector") -> "FundamentalAgent":
        log.info(f"[FundamentalAgent] Entrenando — {len(X)} muestras")
        # Alinear X e y por posición antes de cualquier procesado
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len].copy()
        y = y.iloc[:min_len].copy()
        X_prep       = self._prepare(X, sector_col, fit_mode=True)
        X_prep, y_cl = self.clean_features(X_prep, y.reset_index(drop=True))
        X_prep       = X_prep.reset_index(drop=True)
        y_cl         = y_cl.reset_index(drop=True)

        # Selección de features: solo con datos de train (sin leakage)
        self._selector = FeatureSelector(corr_threshold=FEATURE_CORR_THRESHOLD, top_n=FEATURE_TOP_N,
                                         min_features=8, random_seed=self.random_seed)
        X_prep = self._selector.fit_transform(X_prep, y_cl, agent_name="fundamental")

        self._feature_cols = list(X_prep.columns)
        bal          = self.class_balance(y_cl)
        spw          = bal["n_negative"] / bal["n_positive"] if bal["n_positive"] > 0 else 1.0

        base = xgb.XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree, min_child_weight=self.min_child_weight,
            scale_pos_weight=spw, eval_metric="auc",
            random_state=self.random_seed, n_jobs=-1, tree_method="hist",
        )
        self._model = CalibratedClassifierCV(base, method="isotonic", cv=5)

        cv = self._cv(X_prep, y_cl, spw)
        log.info(f"[FundamentalAgent] CV AUC={cv['mean_auc']:.4f} ± {cv['std_auc']:.4f}")

        self._model.fit(X_prep, y_cl)
        self.is_trained = True

        # Feature importances del estimador XGB subyacente
        imp = pd.Series(
            self._model.calibrated_classifiers_[0].estimator.feature_importances_,
            index=self._feature_cols,
        )
        self.save_feature_importances(imp, fold)

        self._diagnostics = {
            "class_balance": bal, "cv_metrics": cv,
            "n_features": len(self._feature_cols),
            "top_features": imp.nlargest(15).to_dict(),
            "feature_selection": self._selector.report(),
        }
        self.record_train_metrics(cv, fold)
        self.save_diagnostics(fold)
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_score(self, X: pd.DataFrame, sector_col: str = "sector") -> pd.Series:
        if not self.is_trained:
            raise RuntimeError("[FundamentalAgent] No entrenado.")
        X_prep = self.clean_features_predict(self._prepare(X, sector_col, fit_mode=False))
        if self._selector is not None:
            X_prep = self._selector.transform(X_prep)
        X_al = self._align(X_prep)
        proba = self._model.predict_proba(X_al)[:, 1]
        return pd.Series(proba, index=X.index, name="fundamental_score")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _prepare(self, X: pd.DataFrame, sector_col: str, fit_mode: bool) -> pd.DataFrame:
        result = self._prepare_with_sector_dummies(
            X=X,
            feature_cols=FEATURE_COLS,
            sector_col=sector_col,
            fit_mode=fit_mode,
            include_zsector=True,
            dummies_attr="_sector_dummies",
        )
        if fit_mode:
            self._feature_cols = list(result.columns)
        return result

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._align_to_feature_cols(X, fill_value=0.0)

    def _cv(self, X: pd.DataFrame, y: pd.Series, spw: float) -> Dict:
        date_order = X.index.get_level_values("date") if "date" in X.index.names else X.index
        sort_idx = date_order.argsort()
        X = X.iloc[sort_idx]
        y = y.reindex(X.index)
        tss = TimeSeriesSplit(n_splits=5)
        aucs, accs, f1s = [], [], []
        for tr, val in tss.split(X):
            clf = xgb.XGBClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate, subsample=self.subsample,
                colsample_bytree=self.colsample_bytree, min_child_weight=self.min_child_weight,
                scale_pos_weight=spw, eval_metric="auc",
                random_state=self.random_seed, n_jobs=-1, tree_method="hist",
            )
            clf.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[val], y.iloc[val])], verbose=False)
            p = clf.predict_proba(X.iloc[val])[:, 1]
            if y.iloc[val].nunique() > 1:
                aucs.append(roc_auc_score(y.iloc[val], p))
            accs.append(accuracy_score(y.iloc[val], (p >= 0.5).astype(int)))
            f1s.append(f1_score(y.iloc[val], (p >= 0.5).astype(int), zero_division=0))
        return {
            "mean_auc": float(np.mean(aucs)) if aucs else 0.0,
            "std_auc":  float(np.std(aucs))  if aucs else 0.0,
            "mean_acc": float(np.mean(accs)),
            "mean_f1":  float(np.mean(f1s)),
        }
