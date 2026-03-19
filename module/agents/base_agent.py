# =============================================================================
# module/agents/base_agent.py — Clase base abstracta para todos los agentes
# =============================================================================
import json
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    def __init__(self, name: str, results_dir: str, random_seed: int = 42):
        self.name        = name
        self.results_dir = Path(results_dir) / name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        self.is_trained  = False
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

    def predict_label(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """Etiquetas binarias desde scores."""
        return (self.predict_score(X) >= threshold).astype(int)

    # ── Guardado de diagnósticos ──────────────────────────────────────────────

    def save_diagnostics(self, fold: Optional[int] = None, extra: Optional[Dict] = None):
        data = {
            "agent":     self.name,
            "timestamp": datetime.now().isoformat(),
            "fold":      fold,
            **self._diagnostics,
            **(extra or {}),
        }
        suffix = f"_fold{fold}" if fold is not None else ""
        path   = self.results_dir / f"diagnostics{suffix}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info(f"[{self.name}] Diagnósticos → {path.name}")

    def save_feature_importances(self, importances: pd.Series, fold: Optional[int] = None):
        suffix = f"_fold{fold}" if fold is not None else ""
        path   = self.results_dir / f"feature_importances{suffix}.csv"
        importances.sort_values(ascending=False).to_csv(path, header=["importance"])
        log.info(f"[{self.name}] Top-5 features: "
                 + " | ".join(f"{k}={v:.3f}" for k, v in importances.nlargest(5).items()))

    def save_predictions(self, preds_df: pd.DataFrame, fold: Optional[int] = None):
        suffix = f"_fold{fold}" if fold is not None else ""
        path   = self.results_dir / f"predictions{suffix}.csv"
        preds_df.to_csv(path)
        log.info(f"[{self.name}] Predicciones ({len(preds_df)} obs) → {path.name}")

    def record_train_metrics(self, metrics: Dict[str, float], fold: Optional[int] = None):
        entry = {"fold": fold, "ts": datetime.now().isoformat(), **metrics}
        self._train_history.append(entry)
        self._diagnostics["last_train_metrics"] = metrics
        path = self.results_dir / "train_history.json"
        with open(path, "w") as f:
            json.dump(self._train_history, f, indent=2, default=str)

    # ── Helpers de preprocesamiento ───────────────────────────────────────────

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


class FeatureSelector:
    """
    Selección de features en dos pasos — debe fitarse solo con datos de train.

    Paso 1 — Eliminación de redundancia por correlación (>= corr_threshold):
        En cada par de features altamente correlacionadas se elimina la que
        tenga MENOR correlación punto-biserial con el target y (la que menos
        se alinea con subidas). Si no se puede calcular, se elimina la segunda
        del par (la habitual).

    Paso 2 — Selección top-N por importancia del modelo:
        Entrena un RandomForest rápido y conserva las `top_n` features con
        mayor importancia Gini. A partir de sus importancias se calculan pesos
        normalizados [0,1] que se usan para ponderar las columnas antes del
        entrenamiento final (multiplicando cada feature por su peso).

    Uso:
        selector = FeatureSelector()
        X_train_sel, weights = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)      # aplica pesos automáticamente
    """

    def __init__(self, corr_threshold: float = 0.95, top_n: int = 10,
                 min_features: int = 5, random_seed: int = 42):
        self.corr_threshold  = corr_threshold
        self.top_n           = top_n
        self.min_features    = min_features
        self.random_seed     = random_seed
        self._selected_cols: List[str]         = []
        self._dropped_corr:  List[str]         = []
        self._dropped_imp:   List[str]         = []
        self._feature_weights: Dict[str, float] = {}   # feature → peso normalizado

    # ── fit_transform ──────────────────────────────────────────────────────────

    def fit_transform(self, X: pd.DataFrame, y: pd.Series,
                      agent_name: str = "") -> pd.DataFrame:
        """
        Ajusta el selector y devuelve X con las features seleccionadas y
        ponderadas. Los pesos se almacenan en self._feature_weights.

        Args:
            X: DataFrame de features (ya limpio).
            y: Target binario alineado con X.
            agent_name: Nombre del agente para los logs (opcional).

        Returns:
            X transformado: top-N features escaladas por su peso normalizado.
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

        # ── Paso 2: eliminación greedy por correlación entre features ─────────
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
        corr_matrix = X[cols].corr().abs()

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

        remaining = [c for c in cols if c not in to_drop_corr]
        if len(remaining) < self.min_features:
            # Revertir: no hay suficientes features tras el filtro
            remaining        = cols
            to_drop_corr     = []
            corr_drop_detail = []
        self._dropped_corr = to_drop_corr

        log.info(f"{prefix} Paso 1 — correlación >{self.corr_threshold:.0%}: "
                 f"eliminadas {len(to_drop_corr)} / {len(cols)} features")
        for detail in corr_drop_detail:
            log.info(f"{prefix}   ELIMINADA por corr: {detail}")

        # Diagnóstico: matriz de correlaciones entre features supervivientes agrupadas
        # por raíz común (ej. roi / roi_zsector / roic / roic_zsector)
        roots = {}
        for c in remaining:
            root = c.replace("_zsector", "").replace("_yoy_growth", "")
            roots.setdefault(root, []).append(c)
        groups = {r: fs for r, fs in roots.items() if len(fs) > 1}
        if groups:
            log.info(f"{prefix}   [diag] Correlaciones entre features con raíz común (supervivientes):")
            for root, feats in groups.items():
                for i, fa in enumerate(feats):
                    for fb in feats[i+1:]:
                        cv = float(corr_matrix.at[fa, fb])
                        log.info(f"{prefix}   [diag]   {fa} ↔ {fb}: |corr|={cv:.3f}")

        # ── Paso 3: importancia mediante RF rápido ─────────────────────────────
        n_remaining = len(remaining)
        imp_all = pd.Series(1.0 / n_remaining if n_remaining else 0.0, index=remaining)
        try:
            from sklearn.ensemble import RandomForestClassifier as _RFC
            rf = _RFC(n_estimators=100, max_depth=5, n_jobs=-1,
                      random_state=self.random_seed, class_weight="balanced")
            rf.fit(X[remaining], y)
            imp_all = pd.Series(rf.feature_importances_, index=remaining).sort_values(ascending=False)

            keep_n = max(self.min_features, min(self.top_n, len(imp_all)))
            kept   = list(imp_all.head(keep_n).index)
            self._dropped_imp   = [c for c in remaining if c not in kept]
            self._selected_cols = kept

            # Pesos normalizados (suma = 1) para las top-N seleccionadas
            imp_top = imp_all[kept]
            imp_sum = imp_top.sum()
            weights = imp_top / imp_sum if imp_sum > 0 else pd.Series(1.0 / len(imp_top), index=kept)
            self._feature_weights = weights.to_dict()

        except Exception:
            self._selected_cols   = remaining[:max(self.min_features, self.top_n)]
            self._dropped_imp     = [c for c in remaining if c not in self._selected_cols]
            n = len(self._selected_cols)
            self._feature_weights = {c: 1.0 / n for c in self._selected_cols}

        # ── Logs detallados ────────────────────────────────────────────────────
        log.info(f"{prefix} Paso 2 — top-{self.top_n} features por importancia RF:")
        log.info(f"{prefix}   {'Feature':<35} {'Importancia':>12} {'Peso norm.':>10} {'pb_y':>8}")
        log.info(f"{prefix}   {'-'*35} {'-'*12} {'-'*10} {'-'*8}")
        for rank, feat in enumerate(self._selected_cols, 1):
            imp_val = self._feature_weights[feat]
            imp_abs = float(imp_all[feat]) if feat in imp_all.index else imp_val
            pb_val  = pb_corr.get(feat, 0.0)
            log.info(f"{prefix}   {rank:2}. {feat:<33} {imp_abs:>12.4f} {imp_val:>10.4f} {pb_val:>8.3f}")

        if self._dropped_imp:
            log.info(f"{prefix}   Descartadas por importancia baja ({len(self._dropped_imp)}): "
                     + ", ".join(self._dropped_imp))

        log.info(
            f"{prefix} Resumen: {len(cols)} features → "
            f"corr_drop={len(self._dropped_corr)} → "
            f"imp_drop={len(self._dropped_imp)} → "
            f"{len(self._selected_cols)} seleccionadas (top-{self.top_n})"
        )

        # Aplicar pesos: escalar cada feature por su peso normalizado
        return self._apply_weights(X[self._selected_cols])

    # ── transform ─────────────────────────────────────────────────────────────

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in self._selected_cols if c in X.columns]
        missing = [c for c in self._selected_cols if c not in X.columns]
        result  = X[present].copy()
        for c in missing:
            result[c] = 0.0
        return self._apply_weights(result[self._selected_cols])

    # ── helpers ────────────────────────────────────────────────────────────────

    def _apply_weights(self, X: pd.DataFrame) -> pd.DataFrame:
        """Multiplica cada columna por su peso normalizado."""
        X = X.copy()
        for col, w in self._feature_weights.items():
            if col in X.columns:
                X[col] = X[col] * w
        return X

    def report(self) -> Dict:
        return {
            "n_selected":      len(self._selected_cols),
            "selected":        self._selected_cols,
            "feature_weights": self._feature_weights,
            "dropped_corr":    self._dropped_corr,
            "dropped_imp":     self._dropped_imp,
        }
