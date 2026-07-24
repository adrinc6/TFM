"""Catálogo científico cerrado y versionado."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from module.evaluation.profiles import PROFILE_NAMES


CATALOG_VERSION = 1
STAGE_ORDER = ("temporal", "representation", "model", "meta", "portfolio")
STAGE_DETAILS = {
    "temporal": {
        "label": "Temporal",
        "description": "Define cuándo se observa el universo, cuánto futuro se predice y cuánta historia aprende el modelo.",
        "question": "¿Qué ritmo de decisión y horizonte conservan información predictiva sin introducir lookahead?",
    },
    "representation": {
        "label": "Representación",
        "description": "Define qué bloques de información point-in-time llegan a los agentes y cómo se transforman.",
        "question": "¿Qué representación separa mejor la cola operable de acciones?",
    },
    "model": {
        "label": "Modelo y agentes",
        "description": "Define familia predictiva, agentes especializados y regularización del ajuste walk-forward.",
        "question": "¿Qué capacidad de aprendizaje mejora la señal sin sacrificar estabilidad?",
    },
    "meta": {
        "label": "Meta-agente",
        "description": "Define cómo se combinan causalmente los rankings de los agentes ya entrenados.",
        "question": "¿Cómo se adapta la combinación de señales sin usar etiquetas aún abiertas?",
    },
    "portfolio": {
        "label": "Cartera",
        "description": "Define la traducción de la señal en posiciones, exposición activa y costes netos.",
        "question": "¿Qué construcción convierte Rank-IC en alfa neto con rotación sostenible?",
    },
}
DECISION_REASONS = (
    "automatic",
    "greater_simplicity",
    "lower_compute_cost",
    "lower_turnover",
    "greater_stability",
    "methodological_constraint",
)

FEATURE_PRESETS = {
    "core": (
        "quality_core", "value_core", "growth_acceleration", "momentum_core", "price_risk",
    ),
    "fundamental": (
        "quality_core", "quality_efficiency", "financial_strength", "value_core",
        "value_cashflow", "growth_acceleration", "fundamental_stability",
    ),
    "technical": ("momentum_core", "momentum_trend", "price_risk", "market_liquidity"),
    "all": (
        "quality_core", "quality_efficiency", "financial_strength", "value_core",
        "value_cashflow", "growth_acceleration", "fundamental_stability", "momentum_core",
        "momentum_trend", "price_risk", "market_liquidity",
    ),
}

AGENT_NAMES = ("quality", "value", "growth", "momentum", "risk")


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
    modes: tuple[str, ...] = ("fixed", "optimize")
    max_values: int = 3
    depends_on: tuple[tuple[str, tuple[Any, ...]], ...] = ()
    simplicity: tuple[Any, ...] = ()

    def public(self) -> dict[str, Any]:
        value = asdict(self)
        value["values"] = list(self.values)
        value["modes"] = list(self.modes)
        value["depends_on"] = [
            {"variable": variable, "values": list(values)}
            for variable, values in self.depends_on
        ]
        value["value_options"] = [
            {
                "value": option,
                "label": _option_label(self.id, option),
                "description": _option_description(self, option),
            }
            for option in self.values
        ]
        return value


_OPTION_LABELS: dict[str, dict[Any, str]] = {
    "snapshot_step_months": {1: "Mensual", 3: "Trimestral", 6: "Semestral", 12: "Anual"},
    "recency_weighting": {"off": "Sin recencia", "linear": "Recencia lineal", "exponential": "Recencia exponencial"},
    "objective": {"rank_regression": "Regresión de ranking", "ranking": "Ranking directo"},
    "feature_preset": {"core": "Core", "fundamental": "Fundamental", "technical": "Técnico", "all": "Todos los bloques"},
    "model_family": {"lightgbm": "LightGBM", "elastic_net": "Elastic Net"},
    "feature_weighting_mode": {"model_native": "Nativa del modelo", "oos_stability_prune": "Poda por estabilidad OOS"},
    "meta_method": {"equal": "Equiponderado", "rank_ic": "Ponderado por Rank-IC", "stacked_rolling": "Stacking móvil", "stacked_exponential": "Stacking exponencial"},
    "sizing_mode": {"equal": "Equiponderado", "score_linear": "Lineal por meta-score"},
    "investor_profile": {"balanced": "Balanced", "growth": "Growth", "value": "Value", "quality": "Quality", "momentum": "Momentum", "contrarian": "Contrarian", "defensive": "Defensive", "garp": "GARP"},
}


def _option_label(variable_id: str, value: Any) -> str:
    if isinstance(value, bool):
        return "Activar" if value else "No activar"
    return _OPTION_LABELS.get(variable_id, {}).get(value, str(value))


def _option_description(spec: "VariableSpec", value: Any) -> str:
    details: dict[str, dict[Any, str]] = {
        "snapshot_step_months": {1: "Observa cada mes; más muestras y más cambios potenciales.", 3: "Decide cada trimestre; se alinea con la información fundamental nueva.", 6: "Decide cada semestre; prioriza señales persistentes.", 12: "Decide una vez al año; es la versión de menor rotación."},
        "target_horizon_months": {3: "Predice a tres meses: más reactivo y potencialmente más ruidoso.", 6: "Predice a seis meses: equilibrio entre reacción y persistencia.", 12: "Predice a doce meses: prioriza señales lentas y mantenimiento prolongado."},
        "train_lookback_years": {4: "Usa cuatro años: se adapta rápido, con menos historia.", 8: "Usa ocho años: equilibrio entre historia y adaptación.", 12: "Usa doce años: más historia, con menor sensibilidad a cambios recientes."},
        "investor_profile": {"balanced": "Usa el meta-rank sin sesgo adicional.", "growth": "Prioriza crecimiento, calidad y momentum.", "value": "Prioriza valoración atractiva filtrada por calidad y riesgo.", "quality": "Prioriza negocios de alta calidad y crecimiento sostenible.", "momentum": "Prioriza fuerza de precio y acepta su riesgo asociado.", "contrarian": "Busca acciones castigadas con valoración y riesgo controlados.", "defensive": "Prioriza bajo riesgo, calidad y preservación de capital.", "garp": "Combina crecimiento, valoración razonable y calidad."},
        "sizing_mode": {"equal": "Da el mismo peso a cada acción seleccionada.", "score_linear": "Da más peso al mayor meta-score, con una relación máxima 2:1."},
    }
    if spec.id in details and value in details[spec.id]:
        return details[spec.id][value]
    if isinstance(value, bool):
        return ("Incluye" if value else "Excluye") + " este componente en la evaluación."
    return f"Usa el valor {value} para {spec.label.lower()}."


def _v(
    id: str,
    label: str,
    description: str,
    stage: str,
    values: tuple[Any, ...],
    recommended: Any,
    invalidates: str,
    cost: str,
    order: int,
    *,
    max_values: int = 3,
    depends_on: tuple[tuple[str, tuple[Any, ...]], ...] = (),
    simplicity: tuple[Any, ...] = (),
) -> VariableSpec:
    return VariableSpec(
        id, label, description, stage, values, recommended, invalidates, cost, order,
        max_values=max_values, depends_on=depends_on, simplicity=simplicity or values,
    )


VARIABLES: tuple[VariableSpec, ...] = (
    _v("snapshot_step_months", "Cadencia de snapshots", "Frecuencia de observación, ranking y mark-to-market.", "temporal",
       (1, 3, 6, 12), 1, "dataset", "fit", 5, max_values=4),
    _v("target_horizon_months", "Horizonte objetivo", "Meses del retorno futuro.", "temporal",
       (3, 6, 12), 12, "features", "fit", 10),
    _v("train_lookback_years", "Historia de entrenamiento", "Años usados en cada fit.", "temporal",
       (4, 8, 12), 8, "fit", "fit", 20),
    _v("execution_lag_days", "Lag de ejecución", "Retraso PIT tras cierre de periodo.", "temporal",
       (30, 45, 60), 60, "dataset", "fit", 30),
    _v("recency_weighting", "Peso de recencia", "Ponderación temporal del entrenamiento.", "temporal",
       ("off", "linear", "exponential"), "off", "fit", "fit", 40),
    _v("objective", "Objetivo", "Función de aprendizaje transversal.", "temporal",
       ("rank_regression", "ranking"), "rank_regression", "fit", "fit", 50),

    _v("feature_preset", "Preset de features", "Bloques de factores predefinidos.", "representation",
       tuple(FEATURE_PRESETS), "core", "features", "fit", 10, max_values=4),
    _v("fundamental_momentum", "Momentum fundamental", "Tendencia de fundamentales PIT.",
       "representation", (False, True), False, "features", "fit", 20, max_values=2),
    _v("market_regime_feature", "Régimen de mercado", "Contexto de mercado PIT.",
       "representation", (False, True), False, "features", "fit", 30, max_values=2),
    _v("neutralize_by_sector", "Neutralización sectorial", "Ranking dentro de sector.",
       "representation", (False, True), False, "features", "fit", 40, max_values=2),
    _v("winsorization", "Winsorización", "Recorte transversal de extremos.", "representation",
       (0.0, 0.01, 0.025), 0.0, "features", "fit", 50),
    _v("max_features_per_agent", "Features por agente", "Límite tras selección OOS.",
       "representation", (8, 12, 20), 8, "fit", "fit", 60),

    _v("model_family", "Familia de modelo", "Modelo predictivo por agente.", "model",
       ("lightgbm", "elastic_net"), "lightgbm", "fit", "fit", 10),
    _v("lgbm_max_depth", "Profundidad", "Profundidad máxima de LightGBM.", "model",
       (3, 4, 6), 3, "fit", "fit", 30,
       depends_on=(("model_family", ("lightgbm",)),)),
    _v("lgbm_n_estimators", "Estimadores", "Número de árboles.", "model",
       (100, 200, 400), 100, "fit", "fit", 40,
       depends_on=(("model_family", ("lightgbm",)),)),
    _v("lgbm_learning_rate", "Learning rate", "Paso de boosting.", "model",
       (0.03, 0.05, 0.1), 0.05, "fit", "fit", 50,
       depends_on=(("model_family", ("lightgbm",)),)),
    _v("lgbm_min_child_samples", "Mínimo por hoja", "Regularización por tamaño de hoja.", "model",
       (20, 50, 100), 50, "fit", "fit", 60,
       depends_on=(("model_family", ("lightgbm",)),)),
    _v("feature_weighting_mode", "Selección de features", "Uso de diagnóstico OOS.", "model",
       ("model_native", "oos_stability_prune"), "oos_stability_prune", "fit", "fit", 70),

    _v("meta_method", "Meta-agente", "Combinación causal de agentes.", "meta",
       ("equal", "rank_ic", "stacked_rolling", "stacked_exponential"), "equal",
       "meta", "meta", 10, max_values=4),
    _v("meta_history_quarters", "Ventana del meta", "Trimestres cerrados usados.", "meta",
       (8, 16), 16, "meta", "meta", 20,
       depends_on=(("meta_method", ("rank_ic", "stacked_rolling", "stacked_exponential")),)),
    _v("meta_weight_min", "Peso mínimo por agente", "Suelo causal para que ningún agente desaparezca del meta.", "meta",
       (0.0, 0.05, 0.10), 0.10, "meta", "meta", 30,
       depends_on=(("meta_method", ("rank_ic", "stacked_rolling", "stacked_exponential")),)),
    _v("meta_weight_cap", "Peso máximo por agente", "Techo causal para evitar la concentración del meta.", "meta",
       (0.30, 0.50, 1.0), 0.50, "meta", "meta", 40,
       depends_on=(("meta_method", ("stacked_rolling", "stacked_exponential")),)),
    _v("meta_equal_shrinkage", "Contracción a equal", "Contracción de pesos aprendidos.", "meta",
       (0.0, 0.25, 0.5), 0.0, "meta", "meta", 50,
       depends_on=(("meta_method", ("stacked_rolling", "stacked_exponential")),)),
    _v("meta_half_life_quarters", "Semivida del meta", "Decaimiento de cohortes cerradas.", "meta",
       (4.0, 8.0), 8.0, "meta", "meta", 50,
       depends_on=(("meta_method", ("stacked_exponential",)),), max_values=2),

    _v("target_size", "Posiciones", "Número objetivo de acciones.", "portfolio",
       (8, 12, 16), 12, "backtest", "backtest", 10),
    _v("min_hold_percentile", "Mínimo para conservar", "Por debajo de este percentil una posición se vende.", "portfolio",
       (70.0, 80.0, 90.0), 80.0, "backtest", "backtest", 20),
    _v("rotation_edge_percentiles", "Ventaja para sustituir", "Diferencia mínima de percentil para desplazar a la peor posición.", "portfolio",
       (5.0, 10.0, 15.0), 10.0, "backtest", "backtest", 30),
    _v("rebalance_drift_tolerance", "Tolerancia de rebalanceo", "Cambio relativo mínimo de peso que justifica una orden.", "portfolio",
       (0.0, 0.10, 0.25), 0.25, "backtest", "backtest", 40),
    _v("price_only_strictness_multiplier", "Prudencia sin fundamentales", "Endurece las reglas en snapshots de solo precio.", "portfolio",
       (1.0, 1.5, 2.0), 1.5, "backtest", "backtest", 50),
    _v("sizing_mode", "Sizing", "Asignación del satélite activo.", "portfolio",
       ("equal", "score_linear"), "score_linear", "backtest", "backtest", 60),
    _v("commission_bps", "Comisión", "Comisión por operación.", "portfolio",
       (0.0, 5.0, 10.0), 5.0, "backtest", "backtest", 60),
    _v("slippage_bps", "Slippage", "Impacto estimado por operación.", "portfolio",
       (5.0, 10.0, 20.0), 10.0, "backtest", "backtest", 70),
)

BY_ID = {variable.id: variable for variable in VARIABLES}


def public_catalog() -> dict[str, Any]:
    return {
        "version": CATALOG_VERSION,
        "stage_order": list(STAGE_ORDER),
        "stages": [{"id": stage, **STAGE_DETAILS[stage]} for stage in STAGE_ORDER],
        "decision_reasons": list(DECISION_REASONS),
        "recommended_exploratory_definition": recommended_exploratory_definition(),
        "recommended_exploratory_rationale": (
            "Prioriza horizonte, representación, combinación causal de los cinco agentes y reglas "
            "de cartera que reducen rotación improductiva; mantiene fijos controles PIT y costes base."
        ),
        "limits": {
            "exploratory_evaluations": None,
            "expensive_fits": None,
            "incremental_bytes": None,
            "confirmatory_evaluations": 23,
        },
        "variables": [variable.public() for variable in VARIABLES],
    }


def default_definition() -> dict[str, dict[str, Any]]:
    return {
        variable.id: {"mode": "fixed", "values": [variable.recommended]}
        for variable in VARIABLES
    }


def recommended_exploratory_definition() -> dict[str, dict[str, Any]]:
    """Diseño inicial amplio, secuencial y defendible para señal → alfa."""
    definition = default_definition()
    optimized = {
        "target_horizon_months": [6, 12],
        "train_lookback_years": [8, 12],
        "recency_weighting": ["off", "linear"],
        "feature_preset": ["core", "fundamental", "all"],
        "fundamental_momentum": [False, True],
        "neutralize_by_sector": [False, True],
        "winsorization": [0.0, 0.01],
        "max_features_per_agent": [8, 12],
        "meta_method": ["equal", "rank_ic", "stacked_rolling", "stacked_exponential"],
        "target_size": [8, 12, 16],
        "min_hold_percentile": [70.0, 80.0, 90.0],
        "rotation_edge_percentiles": [5.0, 10.0, 15.0],
        "rebalance_drift_tolerance": [0.0, 0.10, 0.25],
        "price_only_strictness_multiplier": [1.0, 1.5, 2.0],
        "sizing_mode": ["equal", "score_linear"],
    }
    for variable_id, values in optimized.items():
        definition[variable_id] = {"mode": "optimize", "values": values}
    return definition
