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
    def clean_features(
        X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> tuple:
        """
        Limpieza estándar:
          1. Elimina columnas 100% vacías
          2. Elimina filas con >50% NaN
          3. Imputación por mediana
          4. Clip outliers ±10σ
        """
        # 1. Eliminar inf/-inf ANTES de cualquier otra operación
        #    XGBoost no tolera inf aunque missing no sea inf
        X = X.replace([np.inf, -np.inf], np.nan)

        # 2. Eliminar columnas 100% vacías y filas con >50% NaN
        X = X.dropna(axis=1, how="all")
        X = X.dropna(thresh=max(1, int(len(X.columns) * 0.5)), axis=0)

        if y is not None:
            # Alineación robusta: ambos ya deben tener índice 0..N-1 (reset hecho en fit)
            # Usar reindex con fill para no perder filas de X válidas
            common = X.index.intersection(y.index)
            if len(common) == 0:
                # Fallback posicional si los índices no comparten nada
                min_len = min(len(X), len(y))
                X = X.iloc[:min_len].reset_index(drop=True)
                y = y.iloc[:min_len].reset_index(drop=True)
            else:
                y = y.loc[common].dropna()
                X = X.loc[y.index]

        # 3. Imputación por mediana (ya sin inf, la mediana es fiable)
        medians = X.median(numeric_only=True)
        X = X.fillna(medians)
        # Si aún hay NaN (columna 100% NaN tras el dropna parcial) → 0
        X = X.fillna(0)

        # 4. Pre-clip absoluto para evitar overflow al calcular std
        #    (p.ej. 1e308 hace que std -> inf y el clip posterior no funciona)
        X = X.clip(-1e15, 1e15)

        # 5. Clip ±10σ por columna (axis=1 para alinear la Series de bounds)
        means = X.mean(numeric_only=True)
        stds  = X.std(numeric_only=True).replace(0, 1)
        lower = (means - 10 * stds).reindex(X.columns)
        upper = (means + 10 * stds).reindex(X.columns)
        X = X.clip(lower=lower, upper=upper, axis=1)

        # 6. Verificación final: garantía absoluta de no inf/NaN para XGBoost
        X = X.replace([np.inf, -np.inf], 0).fillna(0)

        return (X, y) if y is not None else (X, None)

    @staticmethod
    def clean_features_predict(X: pd.DataFrame) -> pd.DataFrame:
        """
        Limpieza para inferencia: imputa y clipea pero NO elimina filas.
        Garantiza que predict_score devuelve exactamente len(X) filas.
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
    def class_balance(y: pd.Series) -> Dict[str, float]:
        counts = y.value_counts()
        total  = len(y)
        return {
            "n_samples":      total,
            "n_positive":     int(counts.get(1, 0)),
            "n_negative":     int(counts.get(0, 0)),
            "positive_ratio": float(counts.get(1, 0) / total) if total > 0 else 0.0,
        }
