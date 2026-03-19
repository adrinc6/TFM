# =============================================================================
# module/agents/sentiment_agent.py — Agente de Sentiment (Random Forest)
# =============================================================================
# DATOS QUE CONSUME (calculados por SentimentFeatureBuilder):
#   Analistas:   analyst_buy_ratio, analyst_bearish_score, analyst_consensus,
#                analyst_dispersion, analyst_strong_buy_pct,
#                analyst_consensus_change
#   Insiders:    mspr_3m, mspr_trend, insider_net_shares_90d, insider_sell_ratio
#   EPS:         beat_rate_4q, eps_surprise_avg_4q, eps_surprise_pct
#
# Lógica: captura el "wisdom of the crowd" institucional + señales de insiders
# como factor predictivo de comportamiento relativo a 1 trimestre vista.
# Complementa al ValuationAgent (que usa EPS surprises como feature secundaria).
# =============================================================================
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from module.agents.base_agent import BaseAgent, FeatureSelector
from environment import (
    SENTIMENT_N_ESTIMATORS, SENTIMENT_MAX_DEPTH, SENTIMENT_MIN_SAMPLES_LEAF,
    SENTIMENT_CV_FOLDS, FEATURE_CORR_THRESHOLD, FEATURE_TOP_N,
)

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


FEATURE_COLS = [
    # Señales de analistas (recommendation_trends.json)
    "analyst_buy_ratio",
    "analyst_bearish_score",
    "analyst_consensus",
    "analyst_dispersion",
    "analyst_strong_buy_pct",
    "analyst_consensus_change",
    # Insider sentiment MSPR (insider_sentiment.json)
    "mspr_3m",
    "mspr_trend",
    # Transacciones insider (insider_transactions.json)
    "insider_net_shares_90d",
    "insider_sell_ratio",
    # EPS surprises (eps_surprises.json)
    "beat_rate_4q",
    "eps_surprise_avg_4q",
    "eps_surprise_pct",
]


class SentimentAgent(BaseAgent):
    """
    Random Forest calibrado sobre señales de sentimiento institucional.

    Captura:
      - Consenso y tendencia de analistas (recommendation_trends)
      - MSPR mensual de insiders (insider_sentiment) — señal predictiva 30-90d
      - Net buying de insiders (insider_transactions)
      - Beat rate y sorpresas de EPS (eps_surprises)

    Score [0,1]: 1 = sentimiento muy positivo → señal alcista
    """

    def __init__(
        self,
        results_dir: str,
        random_seed: int = 42,
        n_estimators: int = SENTIMENT_N_ESTIMATORS,
        max_depth: int = SENTIMENT_MAX_DEPTH,
        min_samples_leaf: int = SENTIMENT_MIN_SAMPLES_LEAF,
        n_cv_folds: int = SENTIMENT_CV_FOLDS,
    ):
        super().__init__("sentiment", results_dir, random_seed)
        if not _DEPS_OK:
            raise ImportError("scikit-learn requerido.")
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_cv_folds       = n_cv_folds
        self._model:        Optional[Pipeline]         = None
        self._feature_cols: List[str]                  = []
        self._selector:     Optional[FeatureSelector]  = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series,
            fold: Optional[int] = None) -> "SentimentAgent":
        log.info(f"[SentimentAgent] Entrenando — {len(X)} muestras")

        min_len = min(len(X), len(y))
        X = X.iloc[:min_len].copy()
        y = y.iloc[:min_len].copy()

        X_prep       = self._prepare(X)
        X_prep, y_cl = self.clean_features(X_prep, y.reset_index(drop=True))
        X_prep       = X_prep.reset_index(drop=True)
        y_cl         = y_cl.reset_index(drop=True)

        if len(X_prep) < 20:
            log.warning("[SentimentAgent] Insuficientes muestras para entrenar.")
            return self

        # Selección de features
        self._selector = FeatureSelector(
            corr_threshold=FEATURE_CORR_THRESHOLD,
            top_n=FEATURE_TOP_N,
            min_features=5,
            random_seed=self.random_seed,
        )
        X_prep = self._selector.fit_transform(X_prep, y_cl, agent_name="sentiment")
        self._feature_cols = list(X_prep.columns)

        bal = self.class_balance(y_cl)

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced",
            random_state=self.random_seed,
            n_jobs=-1,
        )
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    CalibratedClassifierCV(rf, method="sigmoid", cv=self.n_cv_folds)),
        ])

        cv = self._cv(X_prep, y_cl)
        log.info(f"[SentimentAgent] CV AUC={cv['mean_auc']:.4f} ± {cv['std_auc']:.4f}")
        self._model.fit(X_prep, y_cl)
        self.is_trained = True

        # Importancias
        try:
            rf_fitted = self._model.named_steps["clf"].calibrated_classifiers_[0].estimator
            imp = pd.Series(rf_fitted.feature_importances_, index=self._feature_cols)
        except Exception:
            imp = pd.Series(1.0 / len(self._feature_cols),
                            index=self._feature_cols)
        self.save_feature_importances(imp, fold)

        self._diagnostics = {
            "class_balance":     bal,
            "cv_metrics":        cv,
            "top_features":      imp.nlargest(10).to_dict(),
            "feature_selection": self._selector.report(),
            "coverage": {
                "pct_has_analyst": float((X_prep.get("analyst_buy_ratio", pd.Series(np.nan)).notna()).mean()),
                "pct_has_mspr":    float((X_prep.get("mspr_3m", pd.Series(np.nan)).notna()).mean()),
                "pct_has_beat":    float((X_prep.get("beat_rate_4q", pd.Series(np.nan)).notna()).mean()),
            },
        }
        self.record_train_metrics(cv, fold)
        self.save_diagnostics(fold)
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        if not self.is_trained:
            raise RuntimeError("[SentimentAgent] No entrenado.")
        X_prep = self.clean_features_predict(self._prepare(X))
        if self._selector is not None:
            X_prep = self._selector.transform(X_prep)
        X_al = self._align(X_prep)
        return pd.Series(
            self._model.predict_proba(X_al)[:, 1],
            index=X.index,
            name="sentiment_score",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Selecciona las columnas del agente y añade features derivados."""
        df = X.copy()
        selected = [c for c in FEATURE_COLS if c in df.columns]

        # Feature derivado: señal alcista neta (buy_ratio - bearish_score)
        if "analyst_buy_ratio" in df.columns and "analyst_bearish_score" in df.columns:
            df["analyst_net_bullish"] = df["analyst_buy_ratio"] - df["analyst_bearish_score"]
            selected.append("analyst_net_bullish")

        # Feature derivado: insider neto normalizado
        if "insider_net_shares_90d" in df.columns:
            net = df["insider_net_shares_90d"]
            std = net.std()
            if std > 0:
                df["insider_net_zscore"] = (net - net.mean()) / std
            else:
                df["insider_net_zscore"] = 0.0
            selected.append("insider_net_zscore")

        # Feature derivado: bandera de MSPR positivo (insiders comprando activamente)
        if "mspr_3m" in df.columns:
            df["mspr_positive"] = (df["mspr_3m"] > 20).astype(float)
            df["mspr_negative"] = (df["mspr_3m"] < -20).astype(float)
            selected += ["mspr_positive", "mspr_negative"]

        # Feature derivado: beat streak (beat_rate > 75% = empresa que supera consistentemente)
        if "beat_rate_4q" in df.columns:
            df["consistent_beater"] = (df["beat_rate_4q"] >= 0.75).astype(float)
            selected.append("consistent_beater")

        selected = list(dict.fromkeys(selected))
        return df[[c for c in selected if c in df.columns]].copy()

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        for c in self._feature_cols:
            if c not in X.columns:
                X[c] = 0.0
        return X[self._feature_cols]

    def _cv(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        skf = StratifiedKFold(
            n_splits=self.n_cv_folds, shuffle=True, random_state=self.random_seed
        )
        aucs, accs, f1s = [], [], []
        for tr, val in skf.split(X, y):
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight="balanced",
                random_state=self.random_seed,
                n_jobs=-1,
            )
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
