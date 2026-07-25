"""Catálogo cerrado del único flujo científico: Model Study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


CATALOG_VERSION = 2
AGENT_NAMES = ("quality", "value", "growth", "momentum", "risk")
PROFILE_NAMES = (
    "balanced", "growth", "value", "quality",
    "momentum", "contrarian", "defensive", "garp",
)
PREDICTIVE_STAGES = ("temporal", "representation", "model", "meta")
STAGE_ORDER = (*PREDICTIVE_STAGES, "portfolio")
STAGE_DETAILS = {
    "temporal": {
        "label": "Temporal",
        "description": "Define cuándo se observa la información, cuánto futuro se predice y cuánta historia aprende el modelo.",
        "question": "¿Qué definición temporal conserva más capacidad predictiva fuera de muestra?",
    },
    "representation": {
        "label": "Representación",
        "description": "Define los bloques de información point-in-time y su transformación antes del aprendizaje.",
        "question": "¿Qué representación mejora el Rank-IC de forma estable entre eras?",
    },
    "model": {
        "label": "Modelo",
        "description": "Compara capacidad y regularización del modelo de cada uno de los cinco agentes.",
        "question": "¿Qué modelo aprende mejor sin depender de una sola era?",
    },
    "meta": {
        "label": "Meta-agente",
        "description": "Combina causalmente los cinco rankings usando únicamente etiquetas ya cerradas.",
        "question": "¿Equiponderar o aprender pesos rolling mejora el Rank-IC robusto?",
    },
    "portfolio": {
        "label": "Cartera informativa",
        "description": "Traduce el ganador a una cartera 100 % acciones sin intervenir en la selección predictiva.",
        "question": "¿Cómo se comporta económicamente la señal ya elegida?",
    },
}

FEATURE_PRESETS = {
    "core": ("quality_core", "value_core", "growth_acceleration", "momentum_core", "price_risk"),
    "fundamental": (
        "quality_core", "quality_efficiency", "financial_strength", "value_core",
        "value_cashflow", "growth_acceleration", "fundamental_stability",
    ),
    "technical": ("momentum_core", "momentum_trend", "price_risk", "market_liquidity"),
    "all": (
        "quality_core", "quality_efficiency", "financial_strength", "value_core",
        "value_cashflow", "growth_acceleration", "fundamental_stability",
        "momentum_core", "momentum_trend", "price_risk", "market_liquidity",
    ),
}


@dataclass(frozen=True)
class VariableSpec:
    id: str
    label: str
    description: str
    stage: str
    values: tuple[Any, ...]
    recommended: Any
    invalidates: str
    cost: str
    order: int
    predictive: bool = True
    depends_on: tuple[tuple[str, tuple[Any, ...]], ...] = ()
    simplicity: tuple[Any, ...] = ()

    @property
    def modes(self) -> tuple[str, ...]:
        return ("fixed", "optimize") if self.predictive else ("fixed", "diagnostic")

    def public(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["values"] = list(self.values)
        payload["modes"] = list(self.modes)
        payload["depends_on"] = [
            {"variable": variable, "values": list(values)}
            for variable, values in self.depends_on
        ]
        payload["value_options"] = [
            {
                "value": value,
                "label": _label(self.id, value),
                "description": _description(self.id, value),
            }
            for value in self.values
        ]
        return payload


def _v(
    identifier: str,
    label: str,
    description: str,
    stage: str,
    values: tuple[Any, ...],
    recommended: Any,
    invalidates: str,
    cost: str,
    order: int,
    *,
    predictive: bool = True,
    depends_on: tuple[tuple[str, tuple[Any, ...]], ...] = (),
    simplicity: tuple[Any, ...] = (),
) -> VariableSpec:
    return VariableSpec(
        identifier, label, description, stage, values, recommended,
        invalidates, cost, order, predictive, depends_on,
        simplicity or values,
    )


VARIABLES: tuple[VariableSpec, ...] = (
    _v("snapshot_step_months", "Cadencia de snapshots", "Frecuencia con la que se vuelve a observar y puntuar el universo.", "temporal", (1, 3, 6, 12), 3, "dataset", "fit", 10),
    _v("target_horizon_months", "Horizonte objetivo", "Meses del retorno futuro cuya ordenación aprende el modelo.", "temporal", (3, 6, 12), 12, "features", "fit", 20),
    _v("train_lookback_years", "Historia de entrenamiento", "Años anteriores disponibles en cada ajuste walk-forward.", "temporal", (4, 8, 12), 8, "fit", "fit", 30),
    _v("execution_lag_days", "Lag PIT", "Días exigidos entre el cierre fiscal y la disponibilidad operativa de fundamentales.", "temporal", (30, 45, 60), 60, "dataset", "fit", 40),
    _v("recency_weighting", "Peso de recencia", "Importancia adicional de las observaciones recientes dentro del entrenamiento.", "temporal", ("off", "linear", "exponential"), "off", "fit", "fit", 50),
    _v("objective", "Objetivo de aprendizaje", "Regresión del rango continuo o ranking directo.", "temporal", ("rank_regression", "ranking"), "rank_regression", "fit", "fit", 60),
    _v("feature_preset", "Preset de información", "Conjunto cerrado de bloques de factores que reciben los agentes.", "representation", tuple(FEATURE_PRESETS), "core", "features", "fit", 10),
    _v("fundamental_momentum", "Momentum fundamental", "Añade cambios point-in-time de los fundamentales.", "representation", (False, True), True, "features", "fit", 20),
    _v("market_regime_feature", "Régimen de mercado", "Añade contexto causal del mercado en cada snapshot.", "representation", (False, True), True, "features", "fit", 30),
    _v("neutralize_by_sector", "Neutralización sectorial", "Compara las empresas dentro de su sector antes de aprender.", "representation", (False, True), False, "features", "fit", 40),
    _v("winsorization", "Winsorización", "Recorta extremos transversales al percentil indicado.", "representation", (0.0, 0.01, 0.025), 0.0, "features", "fit", 50),
    _v("max_features_per_agent", "Máximo de features", "Límite de variables conservadas por cada agente.", "representation", (8, 12, 20), 8, "fit", "fit", 60),
    _v("feature_weighting_mode", "Selección de features", "Usa selección nativa o poda basada en estabilidad fuera de muestra.", "representation", ("model_native", "oos_stability_prune"), "oos_stability_prune", "fit", "fit", 70),
    _v("model_family", "Familia del modelo", "Modelo usado de forma independiente por los cinco agentes.", "model", ("lightgbm", "elastic_net"), "lightgbm", "fit", "fit", 10),
    _v("lgbm_max_depth", "Profundidad", "Profundidad máxima de cada árbol LightGBM.", "model", (3, 4, 6), 3, "fit", "fit", 20, depends_on=(("model_family", ("lightgbm",)),)),
    _v("lgbm_n_estimators", "Estimadores", "Número de árboles del boosting.", "model", (100, 200, 400), 100, "fit", "fit", 30, depends_on=(("model_family", ("lightgbm",)),)),
    _v("lgbm_learning_rate", "Learning rate", "Paso de actualización de cada árbol.", "model", (0.03, 0.05, 0.10), 0.05, "fit", "fit", 40, depends_on=(("model_family", ("lightgbm",)),)),
    _v("lgbm_min_child_samples", "Mínimo por hoja", "Observaciones mínimas necesarias en una hoja.", "model", (20, 50, 100), 50, "fit", "fit", 50, depends_on=(("model_family", ("lightgbm",)),)),
    _v("meta_method", "Combinación de agentes", "Equiponderación o Ridge rolling causal con límites de peso.", "meta", ("equal", "stacked_rolling_free", "stacked_rolling_bounded"), "stacked_rolling_bounded", "meta", "meta", 10),
    _v("meta_history_quarters", "Ventana del meta", "Número de cohortes trimestrales cerradas usadas por el stacker.", "meta", (8, 16), 16, "meta", "meta", 20, depends_on=(("meta_method", ("stacked_rolling_free", "stacked_rolling_bounded")),)),
    _v("target_size", "Número de posiciones", "Número objetivo de acciones simultáneas.", "portfolio", (8, 12, 16), 12, "backtest", "backtest", 10, predictive=False),
    _v("min_hold_percentile", "Mínimo para conservar", "Percentil mínimo del meta-rank antes de vender una posición.", "portfolio", (70.0, 80.0, 90.0), 80.0, "backtest", "backtest", 20, predictive=False),
    _v("rotation_edge_percentiles", "Ventaja para sustituir", "Ventaja mínima del candidato sobre la peor posición.", "portfolio", (5.0, 10.0, 15.0), 10.0, "backtest", "backtest", 30, predictive=False),
    _v("rebalance_drift_tolerance", "Tolerancia de rebalanceo", "Desviación relativa mínima necesaria para emitir una orden.", "portfolio", (0.0, 0.10, 0.25), 0.25, "backtest", "backtest", 40, predictive=False),
    _v("price_only_strictness_multiplier", "Prudencia sin fundamentales", "Multiplica los umbrales en snapshots sin fundamentales nuevos.", "portfolio", (1.0, 1.5, 2.0), 1.5, "backtest", "backtest", 50, predictive=False),
    _v("sizing_mode", "Reparto de pesos", "Equiponderación o escala lineal 1:2 anclada al umbral efectivo.", "portfolio", ("equal", "score_linear"), "score_linear", "backtest", "backtest", 60, predictive=False),
    _v("commission_bps", "Comisión", "Comisión aplicada sobre el nocional operado.", "portfolio", (0.0, 5.0, 10.0), 5.0, "backtest", "backtest", 70, predictive=False),
    _v("slippage_bps", "Slippage", "Impacto estimado aplicado sobre el nocional operado.", "portfolio", (5.0, 10.0, 20.0), 10.0, "backtest", "backtest", 80, predictive=False),
)
BY_ID = {variable.id: variable for variable in VARIABLES}


def _label(identifier: str, value: Any) -> str:
    labels = {
        "snapshot_step_months": {1: "Mensual", 3: "Trimestral", 6: "Semestral", 12: "Anual"},
        "recency_weighting": {"off": "Sin ponderación", "linear": "Lineal", "exponential": "Exponencial"},
        "objective": {"rank_regression": "Regresión de ranking", "ranking": "Ranking directo"},
        "model_family": {"lightgbm": "LightGBM", "elastic_net": "Elastic Net"},
        "meta_method": {
            "equal": "Equiponderado 20 %",
            "stacked_rolling_free": "Ridge rolling libre",
            "stacked_rolling_bounded": "Ridge rolling 10–50 %",
        },
        "sizing_mode": {"equal": "Equiponderado", "score_linear": "Lineal por meta-rank"},
    }
    if isinstance(value, bool):
        return "Activado" if value else "Desactivado"
    return labels.get(identifier, {}).get(value, str(value))


def _description(identifier: str, value: Any) -> str:
    specific = {
        "execution_lag_days": {
            30: "Hipótesis exigente de disponibilidad; mayor riesgo de retrasos reales.",
            45: "Compromiso prudente entre actualidad y disponibilidad.",
            60: "Supuesto conservador de publicación point-in-time.",
        },
        "meta_method": {
            "equal": "Cada agente pesa exactamente 20 %.",
            "stacked_rolling_free": "Ridge no negativo; cualquier agente puede pesar entre 0 % y 100 %.",
            "stacked_rolling_bounded": "Ridge no negativo; cada agente pesa entre 10 % y 50 %.",
        },
        "sizing_mode": {
            "equal": "Todas las posiciones objetivo reciben el mismo peso.",
            "score_linear": "La escala parte de 1 en el umbral de conservación y llega a 2 en meta-rank 1.",
        },
    }
    if identifier in specific and value in specific[identifier]:
        return specific[identifier][value]
    if isinstance(value, bool):
        return "Incluye esta capacidad." if value else "No incluye esta capacidad."
    return f"Utiliza {value} como valor de esta variable."


def default_definition() -> dict[str, dict[str, Any]]:
    return {
        variable.id: {"mode": "fixed", "values": [variable.recommended]}
        for variable in VARIABLES
    }


def recommended_definition() -> dict[str, dict[str, Any]]:
    definition = default_definition()
    choices = {
        "target_horizon_months": [6, 12],
        "train_lookback_years": [8, 12],
        "execution_lag_days": [45, 60],
        "recency_weighting": ["off", "linear"],
        "feature_preset": ["core", "fundamental", "all"],
        "fundamental_momentum": [False, True],
        "meta_method": ["equal", "stacked_rolling_free", "stacked_rolling_bounded"],
    }
    for identifier, values in choices.items():
        definition[identifier] = {"mode": "optimize", "values": values}
    for variable in VARIABLES:
        if not variable.predictive:
            alternatives = list(variable.values)
            definition[variable.id] = {
                "mode": "diagnostic",
                "values": [variable.recommended, *[value for value in alternatives if value != variable.recommended][:1]],
            }
    return definition


def public_catalog() -> dict[str, Any]:
    raw = {
        "version": CATALOG_VERSION,
        "stage_order": list(STAGE_ORDER),
        "predictive_stages": list(PREDICTIVE_STAGES),
        "stages": [{"id": stage, **STAGE_DETAILS[stage]} for stage in STAGE_ORDER],
        "variables": [variable.public() for variable in VARIABLES],
        "recommended_definition": recommended_definition(),
        "profiles": list(PROFILE_NAMES),
    }
    raw["hash"] = hashlib.sha256(
        json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return raw
