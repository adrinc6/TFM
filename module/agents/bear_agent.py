# =============================================================================
# module/agents/bear_agent.py — Agente Bear (Filtro de Riesgo)
# =============================================================================
# Busca "red flags" que indiquen riesgo elevado. Su score es INVERSO:
#   score alto → muchas red flags → penaliza en el meta-learner.
#
# DATOS QUE CONSUME:
#   Fundamentales:  total_debt_yoy_growth, debt_equity, debt_to_ebitda,
#                   fcf, current_ratio, consecutive_losses,
#                   revenue_decline, interest_coverage
#   Insiders:       insider_net_shares_90d, insider_sell_ratio
#   Analistas:      eps_surprise_pct, eps_revision
#
# FLAGS QUE EVALÚA (lógica + modelo):
#   F1  Deuda creciendo >20% YoY
#   F2  Debt/Equity > 3.0
#   F3  Debt/EBITDA > 6.0
#   F4  FCF negativo
#   F5  Current Ratio < 1.0 (riesgo de liquidez)
#   F6  ≥2 trimestres consecutivos con pérdidas
#   F7  Caída de revenue YoY
#   F8  Interest Coverage < 1.5
#   F9  Insiders vendiendo neto (sell_ratio > 0.7)
#   F10 EPS surprise negativo en último trimestre
# =============================================================================
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from module.agents.base_agent import BaseAgent
from module.explainer import build_explainer_for_agent, AgentExplainer

log = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

# Definición de flags: (nombre, columna, operador, umbral, descripción)
FLAG_DEFINITIONS = [
    ("debt_growth_high",   "total_debt_yoy_growth", ">",  0.20, "Deuda creciendo >20% YoY"),
    ("debt_equity_high",   "debt_equity",           ">",  3.00, "Debt/Equity > 3"),
    ("debt_ebitda_high",   "debt_to_ebitda",        ">",  6.00, "Debt/EBITDA > 6"),
    ("fcf_negative",       "fcf",                   "<",  0.00, "FCF negativo"),
    ("liquidity_risk",     "current_ratio",         "<",  1.00, "Current Ratio < 1"),
    ("consecutive_losses", "consecutive_losses",    ">=", 2.00, "≥2 trimestres con pérdidas"),
    ("revenue_decline",    "revenue_decline",       "==", 1.00, "Caída de revenue YoY"),
    ("low_coverage",       "interest_coverage",     "<",  1.50, "Interest Coverage < 1.5"),
    ("insider_selling",    "insider_sell_ratio",    ">",  0.70, "Insiders vendiendo >70%"),
    ("eps_miss",           "eps_surprise_pct",      "<", -5.00, "EPS miss >5%"),
]

FEATURE_COLS = [
    "total_debt_yoy_growth","debt_equity","debt_to_ebitda","fcf",
    "current_ratio","consecutive_losses","revenue_decline","interest_coverage",
    "insider_net_shares_90d","insider_sell_ratio",
    "eps_surprise_pct","eps_revision",
]


class BearAgent(BaseAgent):
    """
    Agente de riesgo que opera en dos capas:
      1. Capa de reglas: evalúa flags binarios (rápido, interpretable)
      2. Capa ML: Random Forest que aprende patrones de riesgo más sutiles

    El score final es la media ponderada de ambas capas.
    score [0,1] donde 1 = máximo riesgo (el meta-learner lo usa invertido).
    """

    RULE_WEIGHT = 0.5   # Peso de la capa de reglas en el score final
    ML_WEIGHT   = 0.5   # Peso de la capa ML

    def __init__(self, results_dir: str, random_seed: int = 42,
                 n_estimators: int = 200, max_depth: int = 6,
                 n_cv_folds: int = 5):
        super().__init__("bear", results_dir, random_seed)
        if not _DEPS_OK:
            raise ImportError("scikit-learn requerido.")
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.n_cv_folds   = n_cv_folds
        self._model:        Optional[Pipeline] = None
        self._feature_cols: List[str]          = []
        self._explainer:    Optional[AgentExplainer] = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series,
            fold: Optional[int] = None) -> "BearAgent":
        """
        y aquí es el label INVERTIDO: 1 = Underperform (evento de riesgo).
        El pipeline principal se encarga de invertir y antes de llamar a fit.
        """
        log.info(f"[BearAgent] Entrenando — {len(X)} muestras")
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len].copy()
        y = y.iloc[:min_len].copy()
        X_flags      = self._add_flag_cols(X)
        X_prep       = self._prepare(X_flags, fit_mode=True)
        X_prep, y_cl = self.clean_features(X_prep, y.reset_index(drop=True))
        X_prep       = X_prep.reset_index(drop=True)
        y_cl         = y_cl.reset_index(drop=True)
        self._feature_cols = list(X_prep.columns)
        bal          = self.class_balance(y_cl)

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            class_weight="balanced", random_state=self.random_seed, n_jobs=-1,
        )
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    CalibratedClassifierCV(rf, method="sigmoid", cv=3)),
        ])

        cv = self._cv(X_prep, y_cl)
        log.info(f"[BearAgent] CV AUC={cv['mean_auc']:.4f} ± {cv['std_auc']:.4f}")
        self._model.fit(X_prep, y_cl)
        self.is_trained = True

        rf_fitted = self._model.named_steps["clf"].calibrated_classifiers_[0].estimator
        imp = pd.Series(rf_fitted.feature_importances_, index=self._feature_cols)
        self.save_feature_importances(imp, fold)

        # Estadísticas de flags en train
        flag_stats = self._flag_statistics(X)
        self._diagnostics = {
            "class_balance": bal, "cv_metrics": cv,
            "flag_statistics": flag_stats,
            "flag_definitions": [f[0] for f in FLAG_DEFINITIONS],
            "top_features": imp.nlargest(10).to_dict(),
        }
        self.record_train_metrics(cv, fold)
        self.save_diagnostics(fold)
        self._save_flag_report(flag_stats, fold)

        # Explicabilidad SHAP
        rf_model = self._model.named_steps["clf"].calibrated_classifiers_[0].estimator
        self._explainer = build_explainer_for_agent(
            self.name, rf_model, self._feature_cols,
            X_prep, self.results_dir.parent.as_posix(), fold, model_type="tree"
        )
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        """
        Score de RIESGO [0,1]. En el meta-learner se usa como (1 - bear_score)
        para que contribuya como señal bajista.
        """
        if not self.is_trained:
            raise RuntimeError("[BearAgent] No entrenado.")

        X_flags = self._add_flag_cols(X)
        rule_score = self._rule_score(X)
        X_prep = self.clean_features_predict(self._prepare(X_flags, fit_mode=False))
        X_al   = self._align(X_prep)
        ml_score = pd.Series(self._model.predict_proba(X_al)[:, 1], index=X.index)

        # Combinar reglas + ML
        combined = (self.RULE_WEIGHT * rule_score.reindex(X.index).fillna(0.5) +
                    self.ML_WEIGHT   * ml_score)
        return combined.rename("bear_score")

    def flag_report(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Devuelve DataFrame con qué flags dispara cada observación.
        Útil para análisis individual de un ticker.
        """
        return self._add_flag_cols(X)[[f[0] for f in FLAG_DEFINITIONS
                                       if f[0] in self._add_flag_cols(X).columns]]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _add_flag_cols(X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        ops = {">": lambda a, b: a > b, "<": lambda a, b: a < b,
               ">=": lambda a, b: a >= b, "==": lambda a, b: a == b}
        for name, col, op, thresh, _ in FLAG_DEFINITIONS:
            if col in df.columns:
                df[name] = ops[op](df[col], thresh).astype(float)
            else:
                df[name] = 0.0
        return df

    @staticmethod
    def _rule_score(X: pd.DataFrame) -> pd.Series:
        """Score de reglas: proporción de flags activos, normalizado a [0,1]."""
        df    = BearAgent._add_flag_cols(X)
        flags = [f[0] for f in FLAG_DEFINITIONS]
        available = [f for f in flags if f in df.columns]
        if not available:
            return pd.Series(0.5, index=X.index)
        return df[available].mean(axis=1)

    def _prepare(self, X: pd.DataFrame, fit_mode: bool) -> pd.DataFrame:
        df       = X.copy()
        base     = [c for c in FEATURE_COLS if c in df.columns]
        flag_col = [f[0] for f in FLAG_DEFINITIONS]
        flag_col = [c for c in flag_col if c in df.columns]
        selected = list(dict.fromkeys(base + flag_col))
        result   = df[selected].copy()
        if fit_mode:
            self._feature_cols = list(result.columns)
        return result

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        for c in self._feature_cols:
            if c not in X.columns:
                X[c] = 0.0
        return X[self._feature_cols]

    def _cv(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        skf  = StratifiedKFold(n_splits=self.n_cv_folds, shuffle=True, random_state=self.random_seed)
        aucs, accs, f1s = [], [], []
        for tr, val in skf.split(X, y):
            rf   = RandomForestClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                class_weight="balanced", random_state=self.random_seed, n_jobs=-1)
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", rf)])
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

    @staticmethod
    def _flag_statistics(X: pd.DataFrame) -> Dict:
        df    = BearAgent._add_flag_cols(X)
        stats = {}
        for name, col, op, thresh, desc in FLAG_DEFINITIONS:
            if name in df.columns:
                rate = float(df[name].mean())
                stats[name] = {"activation_rate": rate, "description": desc,
                               "threshold": thresh, "column": col}
        return stats

    def _save_flag_report(self, flag_stats: Dict, fold: Optional[int]):
        suffix = f"_fold{fold}" if fold is not None else ""
        path   = self.results_dir / f"flag_report{suffix}.json"
        with open(path, "w") as f:
            json.dump(flag_stats, f, indent=2)
        log.info(f"[BearAgent] Flag report → {path.name}")
