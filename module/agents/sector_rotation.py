"""Sector rotation agent (top-down predictor) for the multi-agent stock picker.

Predicts whether a sector will outperform the S&P 500 next quarter.

Logic:
  Operates at SECTOR level, not ticker level. For each (sector, quarter):
    - Aggregates features from all tickers in the sector (median).
    - Adds sectoral price momentum and analyst flow features.
    - Predicts whether the mean sector return will beat SPY.

  Output (sector_score) enters the MetaLearner as top-down context:
    - A strong ticker in a strong sector receives a boosting signal.
    - A strong ticker in a weak sector receives a penalty.

Features consumed (sector-level medians of):
  Fundamentals: roe, roa, net_margin, revenue_yoy_growth, fcf_margin,
                debt_to_ebitda, interest_coverage
  Valuation:    pe_ratio, pb_ratio, ev_to_ebitda, fcf_yield
  Momentum:     momentum_1m, momentum_3m, momentum_6m, momentum_12m,
                volatility_20d, rsi_14
  Sentiment:    analyst_buy_ratio, analyst_consensus, beat_rate_4q,
                eps_surprise_pct, mspr_3m
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from module.agents.base import BaseAgent, FeatureSelector
from environment import FEATURE_CORR_THRESHOLD, FEATURE_TOP_N
from module.common.feature_controls import resolve_feature_columns
from environment import SECTOR_ROTATION_FEATURE_COLUMNS, SECTOR_ROTATION_FEATURE_EXCLUDE

log = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, f1_score
    from sklearn.model_selection import TimeSeriesSplit
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

# Hiperparámetros propios (no dependen de los de los otros agentes)
_N_ESTIMATORS = 200
_MAX_DEPTH = 3
_LEARNING_RATE = 0.05
_SUBSAMPLE = 0.8

class SectorRotationAgent(BaseAgent):
    """
    Agente top-down que predice si un sector va a superar al S&P 500.

    Proceso:
      1. fit(): recibe el DataFrame de tickers con multi-index (ticker, date).
                Agrega a nivel (sector, quarter) usando medianas robustas.
                Construye el label sectorial: 1 si el retorno medio del sector
                superó al SPY en ese quarter.
                Entrena GBM sobre esas observaciones sectoriales.

      2. predict_sector_scores(): dado un DataFrame de tickers, agrega a nivel
                sector y devuelve un dict {sector: score [0,1]}.

      3. map_to_tickers(): expande el dict de sectores a una Series indexada
                por los mismos tickers del DataFrame de entrada.
    """

    def __init__(
        self,
        results_dir: str,
        random_seed: int = 42,
        save_artifacts: bool = True,
    ):
        super().__init__("sector_rotation", results_dir, random_seed, save_artifacts)
        if not _DEPS_OK:
            raise ImportError("scikit-learn requerido.")
        self._model: Optional[Pipeline] = None
        self._feature_cols: List[str] = []
        self._selector: Optional[FeatureSelector] = None
        self._sector_return_history: Dict[str, float] = {}  # para logging
        self._sector_map: Dict[str, str] = {}  # guardado en fit() para predict_score()

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        sector_map: Dict[str, str],
        spy_prices: Optional[pd.Series] = None,
        fold: Optional[int] = None,
    ) -> "SectorRotationAgent":
        """
        df: DataFrame con multi-index (ticker, date) y columna forward_return.
        sector_map: {ticker: sector}.
        spy_prices: precios diarios del SPY para calcular retorno trimestral.
        """
        log.info(f"[SectorRotationAgent] Construyendo dataset sectorial — {len(df)} obs de tickers")

        sector_df = self._build_sector_dataset(df, sector_map, spy_prices)
        if sector_df.empty or len(sector_df) < 10:
            log.warning("[SectorRotationAgent] Dataset sectorial insuficiente — agente no entrenado.")
            return self

        # Separar features y label
        label_col = "sector_outperform"
        feature_cols = [c for c in sector_df.columns if c not in [label_col, "sector", "quarter"]]
        X = sector_df[feature_cols].copy()
        y = sector_df[label_col].copy()

        X, y = self.clean_features(X, y.reset_index(drop=True))
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        if len(X) < 10 or y.nunique() < 2:
            log.warning(f"[SectorRotationAgent] Datos insuficientes o sin varianza en label ({len(X)} obs) — agente no entrenado.")
            return self

        self._selector = FeatureSelector(
            corr_threshold=FEATURE_CORR_THRESHOLD,
            top_n=FEATURE_TOP_N,
            min_features=3,
            random_seed=self.random_seed,
        )
        X = self._selector.fit_transform(X, y, agent_name="sector_rotation")
        self._feature_cols = list(X.columns)

        bal = self.class_balance(y)
        log.info(
            f"[SectorRotationAgent] {len(sector_df)} obs sectoriales | "
            f"{bal['n_positive']} sectores outperform / {bal['n_negative']} underperform | "
            f"{len(self._feature_cols)} features"
        )

        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=_N_ESTIMATORS,
                max_depth=_MAX_DEPTH,
                learning_rate=_LEARNING_RATE,
                subsample=_SUBSAMPLE,
                random_state=self.random_seed,
            )),
        ])

        cv = self._cv(X, y)
        log.info(f"[SectorRotationAgent] CV AUC={cv['mean_auc']:.4f} ± {cv['std_auc']:.4f}")

        self._model.fit(X, y)
        self.is_trained = True

        imp = pd.Series(
            self._model.named_steps["clf"].feature_importances_,
            index=self._feature_cols,
        )
        self.save_feature_importances(imp, fold)

        self._diagnostics = {
            "class_balance": bal,
            "cv_metrics": cv,
            "n_sector_obs": len(sector_df),
            "top_features": imp.nlargest(10).to_dict(),
            "feature_selection": self._selector.report(),
        }
        self.record_train_metrics(cv, fold)
        self.save_diagnostics(fold)
        self._sector_map = dict(sector_map)  # guardar para predict_score()
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        """
        Implementa la interfaz abstracta de BaseAgent.
        Usa el sector_map guardado en fit() para devolver scores por ticker.
        """
        sector_scores = self.predict_sector_scores(X, self._sector_map)
        return self.map_to_tickers(X, self._sector_map, sector_scores).rename("sector_rotation_score")

    def predict_sector_scores(
        self,
        df: pd.DataFrame,
        sector_map: Dict[str, str],
    ) -> Dict[str, float]:
        """
        Devuelve un dict {sector: score [0,1]} donde score es la probabilidad
        de que ese sector supere al S&P 500 el próximo quarter.
        """
        if not self.is_trained:
            log.warning("[SectorRotationAgent] No entrenado — devolviendo score neutro 0.5 para todos los sectores.")
            sectors = set(sector_map.values())
            return {s: 0.5 for s in sectors}

        sector_features = self._aggregate_ticker_features(df, sector_map)
        if sector_features.empty:
            sectors = set(sector_map.values())
            return {s: 0.5 for s in sectors}

        X = sector_features.copy()
        X = self.clean_features_predict(X)
        if self._selector is not None:
            X = self._selector.transform(X)
        X = self._align(X)

        proba = self._model.predict_proba(X)[:, 1]
        scores = dict(zip(sector_features.index, proba))

        log.info("[SectorRotationAgent] Scores sectoriales:")
        for sector, score in sorted(scores.items(), key=lambda x: -x[1]):
            log.info(f"    {sector:<30}  score={score:.3f}")

        return scores

    def map_to_tickers(
        self,
        df: pd.DataFrame,
        sector_map: Dict[str, str],
        sector_scores: Dict[str, float],
    ) -> pd.Series:
        """
        Expande sector_scores a una Serie por ticker, alineada con df.index.
        Tickers sin sector conocido reciben score neutro 0.5.
        """
        tickers = df.index.get_level_values("ticker")
        mapped = pd.Series(
            [sector_scores.get(sector_map.get(t, ""), 0.5) for t in tickers],
            index=df.index,
            name="sector_score",
        )
        return mapped

    # ── Helpers internos ──────────────────────────────────────────────────────

    def _build_sector_dataset(
        self,
        df: pd.DataFrame,
        sector_map: Dict[str, str],
        spy_prices: Optional[pd.Series],
    ) -> pd.DataFrame:
        """
        Construye un DataFrame (sector, quarter) con:
          - Features agregadas (media de los tickers del sector en ese quarter)
          - Label: 1 si el retorno medio del sector superó al SPY
        """
        tickers = df.index.get_level_values("ticker")
        dates = df.index.get_level_values("date")

        sector_series = pd.Series(tickers, index=df.index).map(sector_map).fillna("Unknown")
        quarter_series = pd.Series(dates.to_period("Q").astype(str), index=df.index)

        temp = df.copy()
        temp["_sector"] = sector_series.values
        temp["_quarter"] = quarter_series.values

        # Features a agregar
        available_feat_cols = resolve_feature_columns(
            default_cols=[],
            available_cols=list(temp.columns),
            include_cols=SECTOR_ROTATION_FEATURE_COLUMNS,
            exclude_cols=SECTOR_ROTATION_FEATURE_EXCLUDE,
            logger=log,
            owner="SectorRotationAgent",
        )

        records = []
        for (sector, quarter), grp in temp.groupby(["_sector", "_quarter"]):
            if sector == "Unknown" or len(grp) < 3:
                continue

            feat = {col: float(grp[col].mean()) for col in available_feat_cols if grp[col].notna().any()}

            # Label sectorial: ¿el retorno medio del sector supera al SPY?
            if "forward_return" not in grp.columns:
                continue
            sector_return = float(grp["forward_return"].mean())

            if spy_prices is not None and not spy_prices.empty:
                spy_ret_by_q = _spy_quarterly_returns_dict(spy_prices)
                spy_ret = spy_ret_by_q.get(quarter, np.nan)
                label = int(sector_return > spy_ret) if pd.notna(spy_ret) else int(sector_return > 0)
            else:
                label = int(sector_return > 0)

            feat["sector"] = sector
            feat["quarter"] = quarter
            feat["sector_outperform"] = label
            feat["_sector_return"] = sector_return
            records.append(feat)

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records)
        log.debug(f"[SectorRotationAgent] Dataset: {len(result)} observaciones (sector × quarter)")
        return result

    def _aggregate_ticker_features(
        self,
        df: pd.DataFrame,
        sector_map: Dict[str, str],
    ) -> pd.DataFrame:
        """Agrega las features de ticker a nivel sector (media) para predicción."""
        tickers = df.index.get_level_values("ticker")
        sector_series = pd.Series(tickers, index=df.index).map(sector_map).fillna("Unknown")

        temp = df.copy()
        temp["_sector"] = sector_series.values
        available_feat_cols = resolve_feature_columns(
            default_cols=[],
            available_cols=list(temp.columns),
            include_cols=SECTOR_ROTATION_FEATURE_COLUMNS,
            exclude_cols=SECTOR_ROTATION_FEATURE_EXCLUDE,
            logger=log,
            owner="SectorRotationAgent",
        )

        agg = temp.groupby("_sector")[available_feat_cols].mean()
        agg = agg[agg.index != "Unknown"]
        return agg

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._align_to_feature_cols(X, fill_value=np.nan)

    def _cv(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        tss = TimeSeriesSplit(n_splits=min(3, len(X) // 4))
        aucs, f1s = [], []
        for tr, val in tss.split(X):
            if y.iloc[val].nunique() < 2:
                continue
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(
                    n_estimators=_N_ESTIMATORS,
                    max_depth=_MAX_DEPTH,
                    learning_rate=_LEARNING_RATE,
                    subsample=_SUBSAMPLE,
                    random_state=self.random_seed,
                )),
            ])
            pipe.fit(X.iloc[tr], y.iloc[tr])
            p = pipe.predict_proba(X.iloc[val])[:, 1]
            aucs.append(roc_auc_score(y.iloc[val], p))
            f1s.append(f1_score(y.iloc[val], (p >= 0.5).astype(int), zero_division=0))
        return {
            "mean_auc": float(np.mean(aucs)) if aucs else 0.0,
            "std_auc":  float(np.std(aucs))  if aucs else 0.0,
            "mean_f1":  float(np.mean(f1s)) if f1s else 0.0,
        }


def _spy_quarterly_returns_dict(spy_prices: pd.Series) -> Dict[str, float]:
    """Retornos trimestrales del SPY como dict {quarter_str: retorno}."""
    spy = spy_prices.sort_index().dropna()
    quarterly = spy.resample("QE").last()
    result: Dict[str, float] = {}
    for i in range(1, len(quarterly)):
        p0 = quarterly.iloc[i - 1]
        p1 = quarterly.iloc[i]
        period = quarterly.index[i].to_period("Q")
        if p0 > 0:
            result[str(period)] = float(p1 / p0 - 1)
    return result
