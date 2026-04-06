# =============================================================================
# module/agents/base_agent.py — Clase base abstracta para todos los agentes
# =============================================================================
import json
import logging
import numpy as np
import pandas as pd

from module.common.feature_policy import is_ratio_or_normalized_feature
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from environment import (
    FEATURE_SELECTOR_RELEVANCE_WEIGHT,
    FEATURE_SELECTOR_RF_N_ESTIMATORS,
    FEATURE_SELECTOR_RF_MAX_DEPTH,
)

log = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Contrato común de todos los agentes del sistema multi-agente.

    Cada agente:
      - Recibe features específicos de su dominio (fundamentales, precio, etc.)
      - Entrena sin ver datos futuros (el pipeline garantiza el orden temporal)
      - Devuelve un score [0.0, 1.0] donde 1 = señal alcista / Outperform
      - Guarda diagnósticos completos en results/agents/<nombre>/
    """

    def __init__(self, name: str, results_dir: str, random_seed: int = 42,
                 save_artifacts: bool = True):
        self.name          = name
        self.results_dir   = Path(results_dir) / name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed   = random_seed
        self.is_trained    = False
        self.save_artifacts = save_artifacts
        self._diagnostics:   Dict[str, Any]  = {}
        self._train_history: List[Dict]       = []

    # ── Interfaz pública ──────────────────────────────────────────────────────

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "BaseAgent":
        """Entrena el agente con datos de un fold de walk-forward."""
        ...

    @abstractmethod
    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        """Devuelve scores [0,1] por observación. 1 = Outperform."""
        ...

    # ── Guardado de diagnósticos ──────────────────────────────────────────────

    def save_diagnostics(self, fold: Optional[int | str] = None, extra: Optional[Dict] = None):
        if not self.save_artifacts:
            return
        data = {
            "agent":     self.name,
            "timestamp": datetime.now().isoformat(),
            "fold":      fold,
            **self._diagnostics,
            **(extra or {}),
        }
        suffix = f"_{fold}" if fold is not None else ""
        path   = self.results_dir / f"diagnostics{suffix}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info(f"[{self.name}] Diagnósticos → {path.name}")

    def save_feature_importances(self, importances: pd.Series, fold: Optional[int | str] = None):
        if not self.save_artifacts:
            return
        suffix = f"_{fold}" if fold is not None else ""
        path   = self.results_dir / f"feature_importances{suffix}.csv"
        importances.sort_values(ascending=False).to_csv(path, header=["importance"])
        log.info(f"[{self.name}] Top-5 features: "
                 + " | ".join(f"{k}={v:.3f}" for k, v in importances.nlargest(5).items()))

    def save_predictions(self, preds_df: pd.DataFrame, fold: Optional[int | str] = None):
        if not self.save_artifacts:
            return
        suffix = f"_{fold}" if fold is not None else ""
        path   = self.results_dir / f"predictions{suffix}.csv"
        preds_df.to_csv(path)
        log.info(f"[{self.name}] Predicciones ({len(preds_df)} obs) → {path.name}")

    def record_train_metrics(self, metrics: Dict[str, float], fold: Optional[int] = None):
        self._diagnostics["last_train_metrics"] = metrics
        if not self.save_artifacts:
            return
        entry = {"fold": fold, "ts": datetime.now().isoformat(), **metrics}
        self._train_history.append(entry)
        path = self.results_dir / "train_history.json"
        with open(path, "w") as f:
            json.dump(self._train_history, f, indent=2, default=str)

    # ── Helpers de preprocesamiento ───────────────────────────────────────────

    @staticmethod
    def _is_ratio_or_normalized_feature(col_name: str, series: Optional[pd.Series] = None) -> bool:
        """Strict policy: allow only ratio/normalized features or binary flags."""
        return is_ratio_or_normalized_feature(col_name, series)

    @staticmethod
    def _filter_ratio_normalized_columns(X: pd.DataFrame) -> pd.DataFrame:
        """Drop non ratio/normalized magnitude features under strict policy."""
        if X is None or X.empty:
            return X
        keep_cols = [
            col for col in X.columns
            if BaseAgent._is_ratio_or_normalized_feature(col, X[col])
        ]
        # Keep strict policy; if no columns survive, return empty frame and let caller fail fast.
        return X[keep_cols].copy()

    @staticmethod
    def _clean_numeric(X: pd.DataFrame) -> pd.DataFrame:
        """
        Limpieza numérica compartida por clean_features y clean_features_predict:
          1. Reemplaza inf/-inf por NaN.
          2. Elimina columnas 100% vacías.
          3. Imputa NaN por mediana de columna; NaN residuales → 0.
          4. Clip absoluto ±1e15 (evita overflow al calcular std).
          5. Clip ±10σ por columna (elimina outliers extremos).
          6. Garantía final: sin inf/NaN para XGBoost y sklearn.
        """
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna(axis=1, how="all")
        medians = X.median(numeric_only=True)
        X = X.fillna(medians).fillna(0)
        X = X.clip(-1e15, 1e15)
        means = X.mean(numeric_only=True)
        stds  = X.std(numeric_only=True).replace(0, 1)
        lower = (means - 10 * stds).reindex(X.columns)
        upper = (means + 10 * stds).reindex(X.columns)
        X = X.clip(lower=lower, upper=upper, axis=1)
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        return X

    @staticmethod
    def clean_features(
        X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> tuple:
        """
        Limpieza para entrenamiento:
          - Elimina filas con >50% NaN antes de imputar (preserva calidad del train).
          - Alinea X e y por índice; fallback posicional si no comparten índice.
          - Aplica _clean_numeric sobre el resultado.

        Args:
            X: DataFrame de features.
            y: Labels opcionales para alinear tras el filtrado de filas.

        Returns:
            Tupla (X_limpio, y_alineado) si y no es None, sino (X_limpio, None).
        """
        X = X.replace([np.inf, -np.inf], np.nan)
        # Eliminar filas con >50% NaN (solo en train; no en predict)
        X = X.dropna(thresh=max(1, int(len(X.columns) * 0.5)), axis=0)

        if y is not None:
            common = X.index.intersection(y.index)
            if len(common) == 0:
                # Fallback posicional si los índices no comparten nada
                min_len = min(len(X), len(y))
                X = X.iloc[:min_len].reset_index(drop=True)
                y = y.iloc[:min_len].reset_index(drop=True)
            else:
                y = y.loc[common].dropna()
                X = X.loc[y.index]

        X = BaseAgent._filter_ratio_normalized_columns(X)
        if X.shape[1] == 0:
            raise ValueError(
                "No quedaron features tras aplicar filtro estricto ratio/normalizado. "
                "Ajusta las features del agente para usar ratios, zscores o flags binarias."
            )
        X = BaseAgent._clean_numeric(X)
        return (X, y) if y is not None else (X, None)

    @staticmethod
    def clean_features_predict(X: pd.DataFrame) -> pd.DataFrame:
        """
        Limpieza para inferencia: imputa y clipea pero NO elimina filas.
        Garantiza que predict_score devuelve exactamente len(X) filas.

        Args:
            X: DataFrame de features de test o live.

        Returns:
            DataFrame limpio con el mismo número de filas que la entrada.
        """
        X = BaseAgent._filter_ratio_normalized_columns(X)
        return BaseAgent._clean_numeric(X)

    @staticmethod
    def class_balance(y: pd.Series) -> Dict[str, float]:
        counts = y.value_counts()
        total  = len(y)
        return {
            "n_samples":      total,
            "n_positive":     int(counts.get(1, 0)),
            "n_negative":     int(counts.get(0, 0)),
            "positive_ratio": float(counts.get(1, 0) / total) if total > 0 else 0.0,
        }

    # ── Helpers estructurales de features ───────────────────────────────────

    @staticmethod
    def _unique_existing_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
        """Deduplica preservando orden y filtra solo columnas presentes en df."""
        return [c for c in dict.fromkeys(columns) if c in df.columns]

    def _prepare_base_features(self, X: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Selecciona columnas base del agente, preservando orden y sin duplicados."""
        df = X.copy()
        selected = self._unique_existing_columns(df, feature_cols)
        return df[selected].copy()

    def _prepare_with_sector_dummies(
        self,
        X: pd.DataFrame,
        feature_cols: List[str],
        sector_col: str,
        fit_mode: bool,
        include_zsector: bool = True,
        dummies_attr: str = "_sector_dummies",
    ) -> pd.DataFrame:
        """Construye matriz de features base + zsector opcional + dummies sectoriales."""
        df = X.copy()
        selected = self._unique_existing_columns(df, feature_cols)

        if include_zsector:
            zsec_cols = [c for c in df.columns if c.endswith("_zsector")]
            selected = self._unique_existing_columns(df, selected + zsec_cols)

        if sector_col in df.columns:
            dummies = pd.get_dummies(df[sector_col], prefix="sector", dtype=float)
            dummies.index = df.index
            if fit_mode:
                setattr(self, dummies_attr, list(dummies.columns))
            else:
                stored = getattr(self, dummies_attr, [])
                for col in stored:
                    if col not in dummies.columns:
                        dummies[col] = 0.0
                dummies = dummies[[c for c in stored if c in dummies.columns]]
            df = pd.concat([df, dummies], axis=1)
            selected = self._unique_existing_columns(df, selected + list(dummies.columns))

        return df[selected].copy()

    def _align_to_feature_cols(self, X: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
        """Alinea columnas al esquema entrenado en self._feature_cols."""
        if not hasattr(self, "_feature_cols"):
            return X
        for col in self._feature_cols:
            if col not in X.columns:
                X[col] = fill_value
        return X[self._feature_cols]


class FeatureSelector:
    """
    Selección de features en dos pasos — debe fitarse solo con datos de train.

    Paso 1 — Eliminación de redundancia por correlación (>= corr_threshold):
        En cada par de features altamente correlacionadas se elimina la que
        tenga MENOR correlación punto-biserial con el target y (la que menos
        se alinea con subidas). Si no se puede calcular, se elimina la segunda
        del par (la habitual).

    Paso 2 — Ranking por señal a y + importancia de modelo:
        Calcula un score combinado por feature:
            combined = w_relevance * norm(|pb_y|) + (1-w_relevance) * norm(importancia_RF)
        donde pb_y es la correlación punto-biserial con y (relevancia directa
        con outperformance) e importancia_RF es la importancia Gini del
        RandomForest auxiliar. Así se evita depender solo de importancias
        relativas del modelo cuando hay colinealidad.

    Nota: NO se aplican pesos a las features seleccionadas. Los modelos de árbol
    (XGBoost, RF, GBM) son invariantes al escalado monotónico, y para la LR
    el StandardScaler normaliza antes de ajustar. La selección es suficiente.

    Uso:
        selector = FeatureSelector()
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel  = selector.transform(X_test)
    """

    def __init__(self, corr_threshold: float = 0.95, top_n: int = 10,
                 min_features: int = 5, random_seed: int = 42,
                 relevance_weight: float = FEATURE_SELECTOR_RELEVANCE_WEIGHT,
                 rf_n_estimators: int = FEATURE_SELECTOR_RF_N_ESTIMATORS,
                 rf_max_depth: int = FEATURE_SELECTOR_RF_MAX_DEPTH,
                 zsector_pair_policy: str = "auto"):
        self.corr_threshold  = corr_threshold
        self.top_n           = top_n
        self.min_features    = min_features
        self.random_seed     = random_seed
        self.relevance_weight = float(max(0.0, min(1.0, relevance_weight)))
        self.rf_n_estimators = max(int(rf_n_estimators), 20)
        self.rf_max_depth = max(int(rf_max_depth), 2)
        policy = str(zsector_pair_policy).strip().lower()
        self.zsector_pair_policy = policy if policy in {"auto", "force_zsector"} else "auto"
        self._selected_cols: List[str] = []
        self._dropped_pair:  List[str] = []
        self._dropped_corr:  List[str] = []
        self._dropped_imp:   List[str] = []

    # ── fit_transform ──────────────────────────────────────────────────────────

    def fit_transform(self, X: pd.DataFrame, y: pd.Series,
                      agent_name: str = "") -> pd.DataFrame:
        """
        Ajusta el selector y devuelve X con las top-N features seleccionadas.

        Args:
            X: DataFrame de features (ya limpio).
            y: Target binario alineado con X.
            agent_name: Nombre del agente para los logs (opcional).

        Returns:
            X filtrado: solo las top-N features seleccionadas (sin escalar).
        """
        cols    = list(X.columns)
        prefix  = f"[FeatureSelector/{agent_name}]" if agent_name else "[FeatureSelector]"

        # ── Paso 1: correlación punto-biserial con y ───────────────────────────
        # Se usa para decidir, ante un par muy correlacionado, cuál descartar.
        try:
            from scipy.stats import pointbiserialr
            pb_corr: Dict[str, float] = {}
            for c in cols:
                try:
                    r, _ = pointbiserialr(y, X[c])
                    pb_corr[c] = abs(float(r))
                except Exception:
                    pb_corr[c] = 0.0
        except ImportError:
            # Fallback: correlación de Pearson con y
            pb_corr = {c: abs(float(X[c].corr(y.astype(float)))) for c in cols}

        # ── Paso 0: exclusión obligatoria base vs _zsector ───────────────────
        # Regla dura: nunca permitir ambas versiones del mismo indicador.
        # Para cada par (base, base_zsector), conservar la que tenga mayor |pb_y|.
        # En empate, preferir _zsector para mantener comparabilidad sectorial.
        to_drop_pair: List[str] = []
        pair_drop_detail: List[str] = []
        for col in cols:
            if not col.endswith("_zsector"):
                continue
            base_col = col[:-8]
            if base_col not in cols:
                continue

            pb_base = float(pb_corr.get(base_col, 0.0))
            pb_zsec = float(pb_corr.get(col, 0.0))
            if self.zsector_pair_policy == "force_zsector":
                loser = base_col
                winner = col
                decision_txt = "policy=force_zsector"
            elif pb_zsec >= pb_base:
                loser = base_col
                winner = col
                decision_txt = "policy=auto"
            else:
                loser = col
                winner = base_col
                decision_txt = "policy=auto"

            if loser not in to_drop_pair:
                to_drop_pair.append(loser)
                pair_drop_detail.append(
                    f"{loser} (par excluyente con {winner}; "
                    f"pb_y({loser})={pb_corr.get(loser, 0.0):.3f} <= "
                    f"pb_y({winner})={pb_corr.get(winner, 0.0):.3f}; {decision_txt})"
                )

        cols_after_pair = [c for c in cols if c not in to_drop_pair]
        self._dropped_pair = to_drop_pair

        if to_drop_pair:
            log.info(f"{prefix} Paso 0 — exclusión base vs _zsector: "
                     f"eliminadas {len(to_drop_pair)} / {len(cols)} features "
                     f"(policy={self.zsector_pair_policy})")
            for detail in pair_drop_detail:
                log.info(f"{prefix}   ELIMINADA por par excluyente: {detail}")

        # ── Paso 1: eliminación greedy por correlación entre features ─────────
        # Algoritmo:
        #   1. Calcular la correlación media de cada feature con las demás
        #      (las features más "redundantes" tienen mayor correlación media).
        #   2. Ordenar features de MAYOR a MENOR correlación media → las más
        #      redundantes se evalúan primero.
        #   3. Para cada feature aún no descartada, revisar si tiene |corr| > umbral
        #      con ALGUNA de las features ya marcadas como "kept".
        #      Si sí: comparar pb_y de ambas → descartar la de menor pb_y.
        #   Este enfoque garantiza que todos los pares se evalúan correctamente,
        #   no solo los de la triangular superior en orden de columna.
        corr_matrix = X[cols_after_pair].corr().abs()

        # Orden: de más a menos correlación media con el resto (más redundantes primero)
        mean_corr = corr_matrix.mean()
        cols_sorted = list(mean_corr.sort_values(ascending=False).index)

        to_drop_corr: List[str]  = []
        kept_so_far:  List[str]  = []
        corr_drop_detail: List[str] = []

        for feat in cols_sorted:
            if feat in to_drop_corr:
                continue
            # Buscar si esta feature está muy correlacionada con ALGUNA ya conservada
            rival = None
            rival_corr = 0.0
            for k in kept_so_far:
                c_val = float(corr_matrix.at[feat, k])
                if c_val > self.corr_threshold and c_val > rival_corr:
                    rival      = k
                    rival_corr = c_val
            if rival is not None:
                # Descartar la que menos se alinea con y
                if pb_corr.get(feat, 0) >= pb_corr.get(rival, 0):
                    # 'feat' es mejor → descartar rival y conservar feat
                    to_drop_corr.append(rival)
                    kept_so_far.remove(rival)
                    kept_so_far.append(feat)
                    corr_drop_detail.append(
                        f"{rival} (|corr|={rival_corr:.3f} con {feat}; "
                        f"pb_y({rival})={pb_corr.get(rival, 0):.3f} < "
                        f"pb_y({feat})={pb_corr.get(feat, 0):.3f})"
                    )
                else:
                    # rival es mejor → descartar feat
                    to_drop_corr.append(feat)
                    corr_drop_detail.append(
                        f"{feat} (|corr|={rival_corr:.3f} con {rival}; "
                        f"pb_y({feat})={pb_corr.get(feat, 0):.3f} < "
                        f"pb_y({rival})={pb_corr.get(rival, 0):.3f})"
                    )
            else:
                kept_so_far.append(feat)

        remaining = [c for c in cols_after_pair if c not in to_drop_corr]
        if len(remaining) < self.min_features:
            # Revertir: no hay suficientes features tras el filtro
            remaining        = cols_after_pair
            to_drop_corr     = []
            corr_drop_detail = []
        self._dropped_corr = to_drop_corr

        log.info(f"{prefix} Paso 1 — correlación >{self.corr_threshold:.0%}: "
                 f"eliminadas {len(to_drop_corr)} / {len(cols)} features")
        for detail in corr_drop_detail:
            log.info(f"{prefix}   ELIMINADA por corr: {detail}")

        # ── Paso 2: score combinado (relevancia con y + importancia RF) ───────
        n_remaining = len(remaining)
        imp_all = pd.Series(1.0 / n_remaining if n_remaining else 0.0, index=remaining)
        pb_abs_all = pd.Series({c: float(pb_corr.get(c, 0.0)) for c in remaining})

        def _minmax_norm(s: pd.Series) -> pd.Series:
            if s is None or s.empty:
                return pd.Series(dtype=float)
            s = s.astype(float)
            s_min = float(s.min())
            s_max = float(s.max())
            if s_max - s_min <= 1e-12:
                return pd.Series(1.0, index=s.index)
            return (s - s_min) / (s_max - s_min)

        try:
            from sklearn.ensemble import RandomForestClassifier as _RFC
            rf = _RFC(n_estimators=self.rf_n_estimators, max_depth=self.rf_max_depth, n_jobs=-1,
                      random_state=self.random_seed, class_weight="balanced")
            rf.fit(X[remaining], y)
            imp_all = pd.Series(rf.feature_importances_, index=remaining).sort_values(ascending=False)
        except Exception:
            pass

        pb_norm = _minmax_norm(pb_abs_all)
        imp_norm = _minmax_norm(imp_all.reindex(remaining).fillna(0.0))
        combined = (
            self.relevance_weight * pb_norm.reindex(remaining).fillna(0.0)
            + (1.0 - self.relevance_weight) * imp_norm.reindex(remaining).fillna(0.0)
        ).sort_values(ascending=False)

        keep_n = max(self.min_features, min(self.top_n, len(combined)))
        self._selected_cols = list(combined.head(keep_n).index)
        self._dropped_imp = [c for c in remaining if c not in self._selected_cols]

        # ── Logs detallados ────────────────────────────────────────────────────
        log.info(
            f"{prefix} Paso 2 — hasta top-{self.top_n} por score combinado "
            f"(w_relevance={self.relevance_weight:.2f}, w_importance={1.0 - self.relevance_weight:.2f}):"
        )
        log.info(f"{prefix}   {'Feature':<35} {'Combined':>10} {'Imp_RF':>10} {'pb_y':>8}")
        log.info(f"{prefix}   {'-'*35} {'-'*10} {'-'*10} {'-'*8}")
        for rank, feat in enumerate(self._selected_cols, 1):
            imp_abs = float(imp_all[feat]) if feat in imp_all.index else 0.0
            pb_val  = pb_corr.get(feat, 0.0)
            cmb_val = float(combined.get(feat, 0.0))
            log.info(f"{prefix}   {rank:2}. {feat:<33} {cmb_val:>10.4f} {imp_abs:>10.4f} {pb_val:>8.3f}")

        if self._dropped_imp:
            log.info(f"{prefix}   Descartadas por score combinado bajo ({len(self._dropped_imp)}): "
                     + ", ".join(self._dropped_imp))

        log.info(
            f"{prefix} Resumen: {len(cols)} features → "
            f"pair_drop={len(self._dropped_pair)} → "
            f"corr_drop={len(self._dropped_corr)} → "
            f"imp_drop={len(self._dropped_imp)} → "
            f"{len(self._selected_cols)} seleccionadas (máximo top-{self.top_n})"
        )

        return X[self._selected_cols].copy()

    # ── transform ─────────────────────────────────────────────────────────────

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.reindex(columns=self._selected_cols, fill_value=0.0)
        return result.copy()

    def report(self) -> Dict:
        return {
            "corr_threshold": self.corr_threshold,
            "top_n": self.top_n,
            "min_features": self.min_features,
            "relevance_weight": self.relevance_weight,
            "importance_weight": 1.0 - self.relevance_weight,
            "rf_n_estimators": self.rf_n_estimators,
            "rf_max_depth": self.rf_max_depth,
            "zsector_pair_policy": self.zsector_pair_policy,
            "n_selected":   len(self._selected_cols),
            "selected":     self._selected_cols,
            "dropped_pair": self._dropped_pair,
            "dropped_corr": self._dropped_corr,
            "dropped_imp":  self._dropped_imp,
        }
