# =============================================================================
# module/agents/momentum_agent.py — Agente de Momentum (Random Forest)
# =============================================================================
# DATOS QUE CONSUME (calculados por TechnicalFeatureBuilder):
#   Osciladores:   rsi_14, rsi_28
#   Tendencia:     macd, macd_signal, macd_hist, sma_20/50/200 (distancia %)
#   Bandas:        bb_pct
#   52 semanas:    price_vs_52w_high, price_vs_52w_low
#   Momentum:      momentum_1m/3m/6m/12m
#   Volatilidad:   volatility_20d, volatility_60d, atr_14
#   Volumen:       vol_ratio_20_50
#   Macro:         vix, yield_curve, sp500_momentum_3m, sp500_momentum_12m
#
# Features derivados que construye internamente:
#   rsi_overbought, rsi_oversold, above_sma200, macd_bullish,
#   momentum_quality, vol_expansion, high_vix_regime,
#   inverted_yield_curve, cross_sma_20_50
# =============================================================================
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from module.agents.base import BaseAgent, FeatureSelector
from module.steps.step_04_evaluation.explainability import build_explainer_for_agent, AgentExplainer
from environment import (
    MOMENTUM_N_ESTIMATORS, MOMENTUM_MAX_DEPTH, MOMENTUM_MIN_SAMPLES_LEAF,
    FEATURE_CORR_THRESHOLD, MOMENTUM_FEATURE_TOP_N,
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

# Features base que consume este agente.
# Fuente: TechnicalFeatureBuilder (OHLCV diario) + SentimentFeatureBuilder (EPS).
# Columnas que falten (ej. atr_14 con pocos días) se rellenan con 0 en _align.
FEATURE_COLS = [
    # Osciladores
    "rsi_14", "rsi_28",
    # Tendencia MACD
    "macd", "macd_signal", "macd_hist",
    # SMAs como distancia % al precio actual (positivo = precio > SMA)
    "sma_20", "sma_50", "sma_200",
    # Posición dentro de Bollinger Bands 20d [0=lower, 1=upper]
    "bb_pct",
    # Posición relativa 52 semanas
    "price_vs_52w_high", "price_vs_52w_low",
    # Momentum puro (retorno sin ajuste)
    "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m",
    # Volatilidad realizada anualizada
    "volatility_20d", "volatility_60d", "atr_14",
    # Volumen relativo
    "vol_ratio_20_50",
    # Earnings momentum: la señal de momentum más potente en la literatura académica.
    # beat_rate_4q: % de trimestres en que el EPS superó la estimación (últimos 4Q).
    # eps_surprise_avg_4q: sorpresa media de EPS en los últimos 4 trimestres.
    # eps_revision: cambio reciente en las estimaciones de EPS de analistas.
    # Estas features son momentum de beneficios, no sentimiento — aquí es su lugar.
    "beat_rate_4q", "eps_surprise_avg_4q", "eps_revision",
    # Nota: rsi_overbought, rsi_oversold, above_sma200, macd_bullish,
    # cross_sma_20_50, momentum_quality, vol_expansion se derivan en _prepare.
]


class MomentumAgent(BaseAgent):
    """
    Random Forest calibrado sobre indicadores técnicos y earnings momentum.

    Usa TimeSeriesSplit para la CV interna (respeta el orden temporal
    de los datos de precio, evitando leakage en la validación).
    """

    def __init__(self, results_dir: str, random_seed: int = 42,
                 n_estimators: int = MOMENTUM_N_ESTIMATORS,
                 max_depth: int = MOMENTUM_MAX_DEPTH,
                 min_samples_leaf: int = MOMENTUM_MIN_SAMPLES_LEAF,
                 save_artifacts: bool = True):
        super().__init__("momentum", results_dir, random_seed, save_artifacts)
        if not _DEPS_OK:
            raise ImportError("scikit-learn requerido.")
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._model:        Optional[Pipeline] = None
        self._feature_cols: List[str]          = []
        self._selector:     Optional[FeatureSelector] = None
        self._explainer:    Optional[AgentExplainer] = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series,
            fold: Optional[int] = None) -> "MomentumAgent":
        log.info(f"[MomentumAgent] Entrenando RandomForest — {len(X)} obs, {len(X.columns)} features")
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len].copy()
        y = y.iloc[:min_len].copy()
        X_prep       = self._prepare(X, fit_mode=True)
        X_prep, y_cl = self.clean_features(X_prep, y.reset_index(drop=True))
        X_prep       = X_prep.reset_index(drop=True)
        y_cl         = y_cl.reset_index(drop=True)

        # Selección de features: solo con datos de train (sin leakage)
        self._selector = FeatureSelector(corr_threshold=FEATURE_CORR_THRESHOLD, top_n=MOMENTUM_FEATURE_TOP_N,
                                         min_features=3, random_seed=self.random_seed)
        X_prep = self._selector.fit_transform(X_prep, y_cl, agent_name="momentum")

        self._feature_cols = list(X_prep.columns)
        bal          = self.class_balance(y_cl)

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf, class_weight="balanced",
            random_state=self.random_seed, n_jobs=-1,
        )
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    rf),
        ])

        cv = self._cv(X_prep, y_cl)
        log.info(f"[MomentumAgent] CV AUC={cv['mean_auc']:.4f} ± {cv['std_auc']:.4f}  ({len(self._feature_cols)} features seleccionadas)")
        self._model.fit(X_prep, y_cl)
        self.is_trained = True

        imp = pd.Series(self._model.named_steps["clf"].feature_importances_, index=self._feature_cols)
        self.save_feature_importances(imp, fold)

        # Guardar distribución de régimen macro
        regime_summary = self._regime_summary(X_prep)
        self._diagnostics = {
            "class_balance": bal, "cv_metrics": cv,
            "regime_summary": regime_summary,
            "top_features": imp.nlargest(10).to_dict(),
            "feature_selection": self._selector.report(),
        }
        self.record_train_metrics(cv, fold)
        self.save_diagnostics(fold)
        rf_model = self._model.named_steps["clf"]
        self._explainer = build_explainer_for_agent(
            self.name, rf_model, self._feature_cols,
            X_prep, self.results_dir.parent.as_posix(), fold, model_type="tree"
        )
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        if not self.is_trained:
            raise RuntimeError("[MomentumAgent] No entrenado.")
        X_prep = self.clean_features_predict(self._prepare(X, fit_mode=False))
        if self._selector is not None:
            X_prep = self._selector.transform(X_prep)
        X_al = self._align(X_prep)
        return pd.Series(self._model.predict_proba(X_al)[:, 1],
                         index=X.index, name="momentum_score")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _prepare(self, X: pd.DataFrame, fit_mode: bool) -> pd.DataFrame:
        df = X.copy()
        selected = [c for c in FEATURE_COLS if c in df.columns]

        # Features derivados (señales binarias y compuestas)
        if "rsi_14" in df.columns:
            df["rsi_overbought"] = (df["rsi_14"] > 70).astype(float)
            df["rsi_oversold"]   = (df["rsi_14"] < 30).astype(float)
            selected += ["rsi_overbought", "rsi_oversold"]
        if "sma_200" in df.columns:
            df["above_sma200"] = (df["sma_200"] > 0).astype(float)
            selected.append("above_sma200")
        if "macd" in df.columns and "macd_signal" in df.columns:
            df["macd_bullish"] = (df["macd"] > df["macd_signal"]).astype(float)
            selected.append("macd_bullish")
        if "sma_20" in df.columns and "sma_50" in df.columns:
            df["cross_sma_20_50"] = (df["sma_20"] > df["sma_50"]).astype(float)
            selected.append("cross_sma_20_50")
        if "momentum_12m" in df.columns and "momentum_1m" in df.columns:
            df["momentum_quality"] = df["momentum_12m"] - df["momentum_1m"]
            selected.append("momentum_quality")
        if "vol_ratio_20_50" in df.columns:
            df["vol_expansion"] = (df["vol_ratio_20_50"] > 1.5).astype(float)
            selected.append("vol_expansion")

        # Earnings momentum derivados
        if "beat_rate_4q" in df.columns:
            # Empresa que supera consistentemente las estimaciones → señal alcista
            df["consistent_beater"] = (df["beat_rate_4q"] >= 0.75).astype(float)
            selected.append("consistent_beater")
        if "eps_surprise_avg_4q" in df.columns and "eps_revision" in df.columns:
            # Combinación: sorpresa positiva + revisión al alza = señal doble
            df["earnings_momentum"] = (
                (df["eps_surprise_avg_4q"].fillna(0) > 0).astype(float) +
                (df["eps_revision"].fillna(0) > 0).astype(float)
            )
            selected.append("earnings_momentum")

        selected = self._unique_existing_columns(df, selected)
        result = df[selected].copy()
        if fit_mode:
            self._feature_cols = list(result.columns)
        return result

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
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf, class_weight="balanced",
                random_state=self.random_seed, n_jobs=-1,
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

    @staticmethod
    def _regime_summary(X: pd.DataFrame) -> Dict:
        out = {}
        if "above_sma200" in X.columns:
            out["pct_above_sma200"]   = float(X["above_sma200"].mean())
        if "rsi_overbought" in X.columns:
            out["pct_rsi_overbought"] = float(X["rsi_overbought"].mean())
        if "rsi_oversold" in X.columns:
            out["pct_rsi_oversold"]   = float(X["rsi_oversold"].mean())
        if "macd_bullish" in X.columns:
            out["pct_macd_bullish"]   = float(X["macd_bullish"].mean())
        if "consistent_beater" in X.columns:
            out["pct_consistent_beater"] = float(X["consistent_beater"].mean())
        return out
