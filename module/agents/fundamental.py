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
from module.common.feature_controls import resolve_feature_columns
from module.steps.step_04_evaluation.explainability import build_explainer_for_agent, AgentExplainer
from environment import (
    FUNDAMENTAL_N_ESTIMATORS, FUNDAMENTAL_MAX_DEPTH, FUNDAMENTAL_LEARNING_RATE,
    FUNDAMENTAL_SUBSAMPLE, FUNDAMENTAL_COLSAMPLE, FUNDAMENTAL_MIN_CHILD_WEIGHT,
    FEATURE_CORR_THRESHOLD, FUNDAMENTAL_FEATURE_TOP_N,
    FUNDAMENTAL_FEATURE_COLUMNS, FUNDAMENTAL_FEATURE_EXCLUDE,
)

log = logging.getLogger(__name__)

try:
    import xgboost as xgb
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
    # Cambios YoY de ratios (tendencia de calidad)
    "roa_change_yoy", "gross_margin_change_yoy", "current_ratio_change_yoy",
    # Calidad contable / eficiencia
    "accruals_ratio", "capex_to_revenue", "consecutive_losses",
    # earnings_quality = FCF / Net Income: alto → beneficios respaldados por caja real.
    # Bajo o negativo → beneficios contables inflados (accruals). Señal de calidad.
    "earnings_quality",
    # Piotroski F-score: composite de 8 señales binarias de calidad financiera.
    # Validado académicamente como predictor de rendimiento a largo plazo.
    # Rango [0,1]: 1 = empresa financieramente sana en todas las dimensiones.
    "piotroski_fscore",
    # EPS (base para cálculo de P/E en ValuationAgent)
    "eps",
    # Tendencias de mejora/deterioro de calidad a 2-3 años (slope normalizado).
    # Capturan si la empresa está mejorando estructuralmente, no solo el nivel actual.
    "roe_trend_2y", "roe_trend_3y",
    "net_margin_trend_2y", "net_margin_trend_3y",
    "gross_margin_trend_3y",
    # Nota: las columnas _zsector las añade _prepare dinámicamente.
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
                 min_child_weight: int = FUNDAMENTAL_MIN_CHILD_WEIGHT,
                 save_artifacts: bool = True):
        super().__init__("fundamental", results_dir, random_seed, save_artifacts)
        if not _DEPS_OK:
            raise ImportError("xgboost y scikit-learn requeridos.")
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.learning_rate    = learning_rate
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self._model = None
        self._feature_cols: List[str] = []
        self._selector:     Optional[FeatureSelector] = None
        self._explainer:    Optional[AgentExplainer] = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series,
            fold: Optional[int] = None, sector_col: str = "sector") -> "FundamentalAgent":
        log.info(f"[FundamentalAgent] Entrenando XGBoost — {len(X)} obs, {len(X.columns)} features")
        # Alinear X e y por posición antes de cualquier procesado
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len].copy()
        y = y.iloc[:min_len].copy()
        X_prep       = self._prepare(X, sector_col, fit_mode=True)
        X_prep, y_cl = self.clean_features(X_prep, y.reset_index(drop=True))
        X_prep       = X_prep.reset_index(drop=True)
        y_cl         = y_cl.reset_index(drop=True)

        # Selección de features: solo con datos de train (sin leakage)
        self._selector = FeatureSelector(corr_threshold=FEATURE_CORR_THRESHOLD, top_n=FUNDAMENTAL_FEATURE_TOP_N,
                                         min_features=3, random_seed=self.random_seed,
                                         zsector_pair_policy="force_zsector")
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
        self._model = base

        cv = self._cv(X_prep, y_cl, spw)
        log.info(f"[FundamentalAgent] CV AUC={cv['mean_auc']:.4f} ± {cv['std_auc']:.4f}  ({len(self._feature_cols)} features seleccionadas)")

        self._model.fit(X_prep, y_cl)
        self.is_trained = True

        # Feature importances del estimador XGB
        imp = pd.Series(self._model.feature_importances_, index=self._feature_cols)
        self.save_feature_importances(imp, fold)

        self._diagnostics = {
            "class_balance": bal, "cv_metrics": cv,
            "n_features": len(self._feature_cols),
            "top_features": imp.nlargest(15).to_dict(),
            "feature_selection": self._selector.report(),
        }
        self.record_train_metrics(cv, fold)
        self.save_diagnostics(fold)
        self._explainer = build_explainer_for_agent(
            self.name, self._model, self._feature_cols,
            X_prep, self.results_dir.parent.as_posix(), fold, model_type="tree"
        )
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
        # Solo columnas base + _zsector (calculadas por SectorNormalizer).
        # Los dummies one-hot de sector se eliminaron: aprenden el nivel medio del
        # sector, no la posición relativa del ticker — eso ya lo cubren los _zsector.
        selected = resolve_feature_columns(
            default_cols=FEATURE_COLS,
            available_cols=list(X.columns),
            include_cols=FUNDAMENTAL_FEATURE_COLUMNS,
            exclude_cols=FUNDAMENTAL_FEATURE_EXCLUDE,
            logger=log,
            owner="FundamentalAgent",
        )
        zsector_cols = [c for c in X.columns if c.endswith("_zsector")]
        result = X[list(dict.fromkeys(selected + zsector_cols))].copy()
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
