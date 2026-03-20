# =============================================================================
# module/agents/valuation_agent.py — Agente de Valoración (GBM)
# =============================================================================
# Detecta infravaloración relativa.
#
# DATOS QUE CONSUME:
#   Múltiplos propios:    pe_ratio, pb_ratio, ps_ratio, ev_to_ebitda,
#                         fcf_yield, earnings_yield
#   Vs. historial:        pe_vs_5y_median, pb_vs_5y_median, ev_ebitda_vs_5y_median
#   Vs. sector (calculado aquí en fit usando companies.csv):
#                         pe_sector_pct, pb_sector_pct, evebitda_sector_pct
#   Analistas:            eps_surprise_pct, eps_revision, eps_est, eps_reported
# =============================================================================
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from module.agents.base import BaseAgent, FeatureSelector
from module.steps.step_04_evaluation.explainability import build_explainer_for_agent, AgentExplainer
from environment import (
    VALUATION_N_ESTIMATORS, VALUATION_MAX_DEPTH, VALUATION_LEARNING_RATE,
    VALUATION_SUBSAMPLE, FEATURE_CORR_THRESHOLD, FEATURE_TOP_N,
)

log = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    from sklearn.model_selection import TimeSeriesSplit
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

# Features base que consume este agente.
# Fuente: ValuationFeatureBuilder (cruce precio × fundamentales).
# Las columnas _sector_pct se calculan internamente en _prepare.
FEATURE_COLS = [
    # Múltiplos actuales (calculados en ValuationFeatureBuilder)
    "pe_ratio", "pb_ratio", "ps_ratio", "ev_to_ebitda", "fcf_yield", "earnings_yield",
    # Comparativa vs historial propio (mean-reversion)
    "pe_vs_5y_median", "pb_vs_5y_median", "ev_ebitda_vs_5y_median",
    # Señales de analistas (ValuationFeatureBuilder._analyst_features)
    "eps_surprise_pct", "eps_revision", "eps_est", "eps_reported",
    # Nota: pe_sector_pct, pb_sector_pct, evebitda_sector_pct, fcfyield_sector_pct
    # se añaden dinámicamente en _prepare usando estadísticas sectoriales de train.
]

# Múltiplos para los que se calcula el percentil sectorial (normalización interna)
MULTIPLES_FOR_SECTOR = ["pe_ratio", "pb_ratio", "ev_to_ebitda", "fcf_yield"]


class ValuationAgent(BaseAgent):
    """
    Gradient Boosting calibrado que detecta infravaloración combinando:
      1. Múltiplos vs propio historial (mean-reversion)
      2. Percentil del múltiplo dentro del sector (companies.csv)
      3. Señales de analistas (EPS surprise / revisiones)
    """

    def __init__(self, results_dir: str, random_seed: int = 42,
                 n_estimators: int = VALUATION_N_ESTIMATORS,
                 max_depth: int = VALUATION_MAX_DEPTH,
                 learning_rate: float = VALUATION_LEARNING_RATE,
                 subsample: float = VALUATION_SUBSAMPLE):
        super().__init__("valuation", results_dir, random_seed)
        if not _DEPS_OK:
            raise ImportError("scikit-learn requerido.")
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.subsample     = subsample
        self._model:              Optional[Pipeline] = None
        self._feature_cols:       List[str]          = []
        self._sector_stats:       Dict               = {}
        self._selector:           Optional[FeatureSelector] = None
        self._explainer:          Optional[AgentExplainer] = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series,
            fold: Optional[int] = None, sector_col: str = "sector") -> "ValuationAgent":
        log.info(f"[ValuationAgent] Entrenando — {len(X)} muestras")
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len].copy()
        y = y.iloc[:min_len].copy()
        X_prep       = self._prepare(X, sector_col, fit_mode=True)
        X_prep, y_cl = self.clean_features(X_prep, y.reset_index(drop=True))
        X_prep       = X_prep.reset_index(drop=True)
        y_cl         = y_cl.reset_index(drop=True)

        # Selección de features: solo con datos de train (sin leakage)
        self._selector = FeatureSelector(corr_threshold=FEATURE_CORR_THRESHOLD, top_n=FEATURE_TOP_N,
                                         min_features=6, random_seed=self.random_seed)
        X_prep = self._selector.fit_transform(X_prep, y_cl, agent_name="valuation")

        self._feature_cols = list(X_prep.columns)
        bal          = self.class_balance(y_cl)

        gbc = GradientBoostingClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            random_state=self.random_seed,
        )
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    CalibratedClassifierCV(gbc, method="sigmoid", cv=5)),
        ])

        cv = self._cv(X_prep, y_cl)
        log.info(f"[ValuationAgent] CV AUC={cv['mean_auc']:.4f} ± {cv['std_auc']:.4f}")
        self._model.fit(X_prep, y_cl)
        self.is_trained = True

        gbc_fitted = self._model.named_steps["clf"].calibrated_classifiers_[0].estimator
        imp = pd.Series(gbc_fitted.feature_importances_, index=self._feature_cols)
        self.save_feature_importances(imp, fold)

        # Guardar distribución de múltiplos por sector para análisis
        sector_summary = self._sector_summary(X, sector_col)
        self._diagnostics = {
            "class_balance": bal, "cv_metrics": cv,
            "sector_multiples_distribution": sector_summary,
            "top_features": imp.nlargest(10).to_dict(),
            "feature_selection": self._selector.report(),
        }
        self.record_train_metrics(cv, fold)
        self.save_diagnostics(fold)
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_score(self, X: pd.DataFrame, sector_col: str = "sector") -> pd.Series:
        if not self.is_trained:
            raise RuntimeError("[ValuationAgent] No entrenado.")
        X_prep = self.clean_features_predict(self._prepare(X, sector_col, fit_mode=False))
        if self._selector is not None:
            X_prep = self._selector.transform(X_prep)
        X_al = self._align(X_prep)
        return pd.Series(self._model.predict_proba(X_al)[:, 1],
                         index=X.index, name="valuation_score")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _prepare(self, X: pd.DataFrame, sector_col: str, fit_mode: bool) -> pd.DataFrame:
        df       = X.copy()
        selected = [c for c in FEATURE_COLS if c in df.columns]

        if sector_col in df.columns:
            if fit_mode:
                # Calcular estadísticas sectoriales SOLO en train
                self._sector_stats = {}
                for mult in MULTIPLES_FOR_SECTOR:
                    if mult not in df.columns:
                        continue
                    self._sector_stats[mult] = {}
                    for sec, grp in df.groupby(sector_col):
                        vals = grp[mult].dropna()
                        if len(vals) >= 3:
                            self._sector_stats[mult][sec] = {
                                "mean": float(vals.mean()),
                                "std":  float(vals.std()) or 1e-6,
                            }
            # Añadir percentil sectorial (disponible en fit y transform)
            for mult in MULTIPLES_FOR_SECTOR:
                if mult not in df.columns:
                    continue
                pct_col = f"{mult.split('_')[0]}_sector_pct"
                stats   = self._sector_stats.get(mult, {})

                def _pct(row, _mult=mult, _stats=stats, _sc=sector_col):
                    s = _stats.get(row[_sc])
                    v = row.get(_mult, np.nan)
                    if s is None or pd.isna(v):
                        return np.nan
                    z = (v - s["mean"]) / s["std"]
                    return float(1 / (1 + np.exp(-z)))  # sigmoide → [0,1]

                df[pct_col] = df.apply(_pct, axis=1)
                selected.append(pct_col)

        selected = list(dict.fromkeys(selected))
        result   = df[[c for c in selected if c in df.columns]].copy()
        if fit_mode:
            self._feature_cols = list(result.columns)
        return result

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._align_to_feature_cols(X, fill_value=np.nan)

    def _cv(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        date_order = X.index.get_level_values("date") if "date" in X.index.names else X.index
        sort_idx = date_order.argsort()
        X = X.iloc[sort_idx]
        y = y.reindex(X.index)
        tss = TimeSeriesSplit(n_splits=5)
        aucs, accs, f1s = [], [], []
        for tr, val in tss.split(X):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(
                    n_estimators=self.n_estimators, max_depth=self.max_depth,
                    learning_rate=self.learning_rate, subsample=self.subsample,
                    random_state=self.random_seed)),
            ])
            pipe.fit(X.iloc[tr], y.iloc[tr])
            p = pipe.predict_proba(X.iloc[val])[:, 1]
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

    def _sector_summary(self, X: pd.DataFrame, sector_col: str) -> Dict:
        if sector_col not in X.columns:
            return {}
        out = {}
        for mult in MULTIPLES_FOR_SECTOR:
            if mult not in X.columns:
                continue
            out[mult] = (X.groupby(sector_col)[mult]
                          .agg(["mean","median","std","count"])
                          .to_dict(orient="index"))
        return out
