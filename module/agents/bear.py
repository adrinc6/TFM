# =============================================================================
# module/agents/bear_agent.py — Agente Bear (Filtro de Riesgo)
# =============================================================================
# Detecta riesgo elevado mediante un score estructurado en dos sub-scores:
#
#   Sub-score financiero (deterioro empresarial):
#     F1  Deuda creciendo >20% YoY
#     F2  Debt/Equity > 3.0
#     F3  Debt/EBITDA > 6.0
#     F4  FCF negativo
#     F5  ≥2 trimestres consecutivos con pérdidas
#     F6  Caída de revenue YoY
#     F7  Interest Coverage < 1.5
#
#   Sub-score de mercado (señales externas de presión):
#     F8  Current Ratio < 1.0 (riesgo de liquidez)
#     F9  Insiders vendiendo neto (sell_ratio > 0.7)
#     F10 EPS miss >5%
#
# Score final = 0.6 × sub_financiero + 0.4 × sub_mercado (capa de reglas)
# El score de reglas se combina con la capa ML ponderada por BEAR_RULE_WEIGHT.
#
# DATOS QUE CONSUME:
#   Fundamentales:  total_debt_yoy_growth, debt_equity, debt_to_ebitda,
#                   fcf, current_ratio, consecutive_losses,
#                   revenue_decline, interest_coverage
#   Insiders:       insider_net_ratio_90d, insider_sell_ratio
#   Analistas:      eps_surprise_pct, eps_revision
# =============================================================================
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from module.agents.base import BaseAgent, FeatureSelector
from module.common.feature_controls import resolve_feature_columns
from module.steps.step_04_evaluation.explainability import build_explainer_for_agent, AgentExplainer
from environment import (
    BEAR_N_ESTIMATORS, BEAR_MAX_DEPTH,
    BEAR_RULE_WEIGHT, BEAR_ML_WEIGHT, FEATURE_CORR_THRESHOLD, BEAR_FEATURE_TOP_N,
    BEAR_FEATURE_COLUMNS, BEAR_FEATURE_EXCLUDE,
)

log = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    from sklearn.model_selection import TimeSeriesSplit
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

# Definición de flags: (nombre, columna, operador, umbral, descripción, peso)
# Peso relativo dentro de su sub-score (se normalizan internamente).
FLAG_DEFINITIONS = [
    # ── Sub-score financiero ─────────────────────────────────────────────────
    ("debt_growth_high",   "total_debt_yoy_growth", ">",  0.20, "Deuda creciendo >20% YoY",    1.0),
    ("debt_equity_high",   "debt_equity",           ">",  3.00, "Debt/Equity > 3",              1.5),
    ("debt_ebitda_high",   "debt_to_ebitda",        ">",  6.00, "Debt/EBITDA > 6",              1.5),
    ("fcf_negative",       "fcf_margin",            "<",  0.00, "FCF margin negativo",          2.0),
    ("consecutive_losses", "consecutive_losses",    ">=", 2.00, "≥2 trimestres con pérdidas",   2.0),
    ("revenue_decline",    "revenue_decline",       "==", 1.00, "Caída de revenue YoY",         1.0),
    ("low_coverage",       "interest_coverage",     "<",  1.50, "Interest Coverage < 1.5",      1.5),
    # ── Sub-score de mercado ─────────────────────────────────────────────────
    ("liquidity_risk",     "current_ratio",         "<",  1.00, "Current Ratio < 1",            1.5),
    ("insider_selling",    "insider_sell_ratio",    ">",  0.70, "Insiders vendiendo >70%",      1.0),
    ("eps_miss",           "eps_surprise_pct",      "<", -5.00, "EPS miss >5%",                 1.0),
]

# Flags que pertenecen al sub-score financiero (el resto es sub-score de mercado)
_FINANCIAL_FLAGS = {
    "debt_growth_high", "debt_equity_high", "debt_ebitda_high",
    "fcf_negative", "consecutive_losses", "revenue_decline", "low_coverage",
}

# Peso de sub-score financiero vs. mercado en la capa de reglas
_FINANCIAL_WEIGHT = 0.60
_MARKET_WEIGHT    = 0.40

# Features base que consume este agente.
FEATURE_COLS = [
    # Fundamentales de riesgo (FundamentalFeatureBuilder)
    "total_debt_yoy_growth",   # Crecimiento de deuda YoY
    "debt_equity",             # Deuda / Equity
    "debt_to_ebitda",          # Deuda / EBITDA
    "fcf_margin",              # Margen de FCF (negativo = riesgo)
    "current_ratio",           # Liquidez a corto plazo
    "consecutive_losses",      # Trimestres consecutivos con pérdidas
    "revenue_decline",         # 1 si revenue cayó YoY
    "interest_coverage",       # EBIT / gastos de interés
    # Insiders (InsiderFeatureBuilder, ventana 90 días)
    "insider_net_ratio_90d",   # Balance neto insider normalizado [-1,1]
    "insider_sell_ratio",      # Proporción de ventas sobre total [0,1]
    # Analistas (ValuationFeatureBuilder._analyst_features)
    "eps_surprise_pct",        # Sorpresa EPS último trimestre
    "eps_revision",            # Revisión de estimación EPS
]


class BearAgent(BaseAgent):
    """
    Agente de riesgo estructurado en dos capas:

      1. Capa de reglas ponderadas: evalúa flags con pesos distintos según
         su severidad, agrupados en sub-score financiero y de mercado.
      2. Capa ML: Random Forest que aprende patrones de riesgo más sutiles.

    El score final [0,1] es la media ponderada de ambas capas.
    Score 1 = máximo riesgo (el meta-learner lo usa invertido).
    """

    RULE_WEIGHT = BEAR_RULE_WEIGHT
    ML_WEIGHT   = BEAR_ML_WEIGHT

    def __init__(self, results_dir: str, random_seed: int = 42,
                 n_estimators: int = BEAR_N_ESTIMATORS,
                 max_depth: int = BEAR_MAX_DEPTH,
                 save_artifacts: bool = True):
        super().__init__("bear", results_dir, random_seed, save_artifacts)
        if not _DEPS_OK:
            raise ImportError("scikit-learn requerido.")
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self._model:        Optional[Pipeline] = None
        self._feature_cols: List[str]          = []
        self._selector:     Optional[FeatureSelector] = None
        self._explainer:    Optional[AgentExplainer] = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series,
            fold: Optional[int] = None) -> "BearAgent":
        """
        y aquí es el label INVERTIDO: 1 = Underperform (evento de riesgo).
        El pipeline principal se encarga de invertir y antes de llamar a fit.
        """
        log.info(f"[BearAgent] Entrenando RandomForest (detección de riesgo) — {len(X)} obs")
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len].copy()
        y = y.iloc[:min_len].copy()
        X_flags      = self._add_flag_cols(X)
        X_prep       = self._prepare(X_flags, fit_mode=True)
        X_prep, y_cl = self.clean_features(X_prep, y.reset_index(drop=True))
        X_prep       = X_prep.reset_index(drop=True)
        y_cl         = y_cl.reset_index(drop=True)

        # Selección de features: solo con datos de train (sin leakage)
        # BearAgent usa todas sus features (pocas), top_n >= nº flags+base
        self._selector = FeatureSelector(corr_threshold=FEATURE_CORR_THRESHOLD, top_n=BEAR_FEATURE_TOP_N,
                                         min_features=5, random_seed=self.random_seed)
        X_prep = self._selector.fit_transform(X_prep, y_cl, agent_name="bear")

        self._feature_cols = list(X_prep.columns)
        bal          = self.class_balance(y_cl)

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            class_weight="balanced", random_state=self.random_seed, n_jobs=-1,
        )
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    rf),
        ])

        cv = self._cv(X_prep, y_cl)
        log.info(f"[BearAgent] CV AUC={cv['mean_auc']:.4f} ± {cv['std_auc']:.4f}  ({len(self._feature_cols)} features seleccionadas)")
        self._model.fit(X_prep, y_cl)
        self.is_trained = True

        imp = pd.Series(self._model.named_steps["clf"].feature_importances_, index=self._feature_cols)
        self.save_feature_importances(imp, fold)

        flag_stats = self._flag_statistics(X)
        self._diagnostics = {
            "class_balance": bal, "cv_metrics": cv,
            "flag_statistics": flag_stats,
            "flag_definitions": [f[0] for f in FLAG_DEFINITIONS],
            "top_features": imp.nlargest(10).to_dict(),
            "feature_selection": self._selector.report(),
            "rule_structure": {
                "financial_flags": list(_FINANCIAL_FLAGS),
                "financial_weight": _FINANCIAL_WEIGHT,
                "market_weight": _MARKET_WEIGHT,
            },
        }
        self.record_train_metrics(cv, fold)
        self.save_diagnostics(fold)
        self._save_flag_report(flag_stats, fold)

        rf_model = self._model.named_steps["clf"]
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

        X_flags    = self._add_flag_cols(X)
        rule_score = self._rule_score_from_flags(X_flags)
        X_prep     = self.clean_features_predict(self._prepare(X_flags, fit_mode=False))
        if self._selector is not None:
            X_prep = self._selector.transform(X_prep)
        X_al       = self._align(X_prep)
        ml_score   = pd.Series(self._model.predict_proba(X_al)[:, 1], index=X.index)

        combined = (self.RULE_WEIGHT * rule_score.reindex(X.index).fillna(0.5) +
                    self.ML_WEIGHT   * ml_score)
        return combined.rename("bear_score")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _add_flag_cols(X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        ops = {">": lambda a, b: a > b, "<": lambda a, b: a < b,
               ">=": lambda a, b: a >= b, "==": lambda a, b: a == b}
        for name, col, op, thresh, _, _w in FLAG_DEFINITIONS:
            if col in df.columns:
                df[name] = ops[op](df[col], thresh).astype(float)
            else:
                df[name] = 0.0
        return df

    @staticmethod
    def _rule_score_from_flags(df_flags: pd.DataFrame) -> pd.Series:
        """
        Score de reglas estructurado en dos sub-scores ponderados.

        Sub-score financiero: media ponderada de flags de deterioro empresarial.
        Sub-score de mercado: media ponderada de flags de presión externa.
        Score final = _FINANCIAL_WEIGHT × fin + _MARKET_WEIGHT × mkt
        """
        fin_flags = [(f[0], f[5]) for f in FLAG_DEFINITIONS if f[0] in _FINANCIAL_FLAGS]
        mkt_flags = [(f[0], f[5]) for f in FLAG_DEFINITIONS if f[0] not in _FINANCIAL_FLAGS]

        def _weighted_mean(flags: List, df: pd.DataFrame) -> pd.Series:
            available = [(name, w) for name, w in flags if name in df.columns]
            if not available:
                return pd.Series(0.0, index=df.index)
            total_w = sum(w for _, w in available)
            score   = sum(df[name] * w for name, w in available) / total_w
            return score

        fin_score = _weighted_mean(fin_flags, df_flags)
        mkt_score = _weighted_mean(mkt_flags, df_flags)
        return (_FINANCIAL_WEIGHT * fin_score + _MARKET_WEIGHT * mkt_score).rename("rule_score")

    def _prepare(self, X: pd.DataFrame, fit_mode: bool = False) -> pd.DataFrame:
        """Selecciona columnas base + flags presentes en X."""
        base = resolve_feature_columns(
            default_cols=FEATURE_COLS,
            available_cols=list(X.columns),
            include_cols=BEAR_FEATURE_COLUMNS,
            exclude_cols=BEAR_FEATURE_EXCLUDE,
            logger=log,
            owner="BearAgent",
        )
        flag_col = [f[0] for f in FLAG_DEFINITIONS if f[0] in X.columns]
        return self._prepare_base_features(X, base + flag_col)

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._align_to_feature_cols(X, fill_value=0.0)

    def _cv(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        date_order = X.index.get_level_values("date") if "date" in X.index.names else X.index
        sort_idx = date_order.argsort()
        X = X.iloc[sort_idx]
        y = y.reindex(X.index)
        tss = TimeSeriesSplit(n_splits=5)
        aucs, accs, f1s = [], [], []
        for tr, val in tss.split(X):
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
        for name, col, op, thresh, desc, weight in FLAG_DEFINITIONS:
            if name in df.columns:
                rate = float(df[name].mean())
                sub  = "financial" if name in _FINANCIAL_FLAGS else "market"
                stats[name] = {
                    "activation_rate": rate,
                    "description":     desc,
                    "threshold":       thresh,
                    "column":          col,
                    "weight":          weight,
                    "sub_score":       sub,
                }
        return stats

    def _save_flag_report(self, flag_stats: Dict, fold: Optional[int | str]):
        suffix = f"_{fold}" if fold is not None else ""
        path   = self.results_dir / f"flag_report{suffix}.json"
        with open(path, "w") as f:
            json.dump(flag_stats, f, indent=2)
        log.info(f"[BearAgent] Flag report → {path.name}")
