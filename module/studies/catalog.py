"""Catálogo cerrado del único flujo científico: Model Study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from module.evaluation.profiles import PROFILE_NAMES
# Fuente única de los nombres de agente: el catálogo de features es quien decide qué agentes
# existen, porque es donde se declara qué factores recibe cada uno.
from module.modeling.catalog import AGENT_NAMES


CATALOG_VERSION = 3
SELECTION_ERAS = ((2015, 2018), (2019, 2021), (2022, 2024))
# Frontera única entre la ventana que decide y la que solo confirma. Toda métrica de selección se
# recorta aquí; 2025-26 no participó en ninguna decisión y por eso es la confirmación fuera de
# muestra. Vive en el catálogo porque es un parámetro científico, no un literal de implementación.
SELECTION_UNTIL_YEAR = 2024
KNOWN_STRESS_YEARS = (2025, 2026)
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
    _v("target_size", "Número de posiciones", "Número objetivo de acciones simultáneas.", "portfolio", (8, 12, 16, 25, 50), 12, "backtest", "backtest", 10, predictive=False),
    _v("exit_expected_alpha_bps", "Alfa mínimo para conservar", "Alfa esperado mínimo, en puntos básicos, antes de vender una posición.", "portfolio", (0.0, 100.0, 250.0), 100.0, "backtest", "backtest", 20, predictive=False),
    _v("rotation_edge_bps", "Ventaja para sustituir", "Ventaja de alfa esperado, en puntos básicos, exigida por encima del coste de ida y vuelta.", "portfolio", (25.0, 50.0, 100.0), 50.0, "backtest", "backtest", 30, predictive=False),
    _v("cash_policy", "Política de efectivo", "Invertir siempre el 100 % o dejar efectivo cuando no hay oportunidades por encima del umbral.", "portfolio", ("fully_invested", "opportunity_cash"), "fully_invested", "backtest", "backtest", 35, predictive=False),
    _v("max_cash_weight", "Efectivo máximo", "Peso máximo que puede quedar sin invertir bajo la política de oportunidad.", "portfolio", (0.0, 0.20, 0.40), 0.20, "backtest", "backtest", 36, predictive=False, depends_on=(("cash_policy", ("opportunity_cash",)),)),
    _v("rebalance_drift_tolerance", "Tolerancia de rebalanceo", "Desviación relativa mínima necesaria para emitir una orden.", "portfolio", (0.0, 0.10, 0.25), 0.25, "backtest", "backtest", 40, predictive=False),
    _v("price_only_strictness_multiplier", "Prudencia sin fundamentales", "Endurece los umbrales en snapshots sin fundamentales nuevos.", "portfolio", (1.0, 1.5, 2.0), 1.5, "backtest", "backtest", 50, predictive=False),
    _v("sizing_mode", "Reparto de pesos", "Equiponderación o peso proporcional al alfa esperado, con tope 2:1.", "portfolio", ("equal", "alpha_proportional"), "alpha_proportional", "backtest", "backtest", 60, predictive=False),
    _v("commission_bps", "Comisión", "Comisión aplicada sobre el nocional operado.", "portfolio", (0.0, 5.0, 10.0), 5.0, "backtest", "backtest", 70, predictive=False),
    _v("slippage_bps", "Slippage", "Impacto estimado aplicado sobre el nocional operado.", "portfolio", (5.0, 10.0, 20.0), 10.0, "backtest", "backtest", 80, predictive=False),
)
BY_ID = {variable.id: variable for variable in VARIABLES}


def _label(identifier: str, value: Any) -> str:
    labels = {
        "snapshot_step_months": {1: "Mensual", 3: "Trimestral", 6: "Semestral", 12: "Anual"},
        "target_horizon_months": {3: "3 meses", 6: "6 meses", 12: "12 meses"},
        "train_lookback_years": {4: "4 años", 8: "8 años", 12: "12 años"},
        "execution_lag_days": {30: "30 días", 45: "45 días", 60: "60 días"},
        "recency_weighting": {"off": "Sin ponderación", "linear": "Lineal", "exponential": "Exponencial"},
        "objective": {"rank_regression": "Regresión de ranking", "ranking": "Ranking directo"},
        "feature_preset": {
            "core": "Core (lo esencial de cada agente)",
            "fundamental": "Fundamental (todo lo contable)",
            "technical": "Técnico (todo lo de precio)",
            "all": "Todo (fundamental + técnico)",
        },
        "winsorization": {0.0: "Sin recorte", 0.01: "1 %", 0.025: "2,5 %"},
        "max_features_per_agent": {8: "8 variables", 12: "12 variables", 20: "20 variables"},
        "feature_weighting_mode": {
            "model_native": "Selección nativa del modelo",
            "oos_stability_prune": "Poda por estabilidad fuera de muestra",
        },
        "model_family": {"lightgbm": "LightGBM (árboles)", "elastic_net": "Elastic Net (lineal)"},
        "lgbm_max_depth": {3: "Poco profundo (3)", 4: "Medio (4)", 6: "Profundo (6)"},
        "lgbm_n_estimators": {100: "100 árboles", 200: "200 árboles", 400: "400 árboles"},
        "lgbm_learning_rate": {0.03: "Lento (0,03)", 0.05: "Medio (0,05)", 0.10: "Rápido (0,10)"},
        "lgbm_min_child_samples": {20: "20 observaciones", 50: "50 observaciones", 100: "100 observaciones"},
        "meta_method": {
            "equal": "Equiponderado 20 %",
            "stacked_rolling_free": "Ridge rolling libre",
            "stacked_rolling_bounded": "Ridge rolling 10–50 %",
        },
        "meta_history_quarters": {8: "8 trimestres (2 años)", 16: "16 trimestres (4 años)"},
        "target_size": {8: "8 posiciones", 12: "12 posiciones", 16: "16 posiciones", 25: "25 posiciones", 50: "50 posiciones"},
        "exit_expected_alpha_bps": {0.0: "0 pb (solo alfa negativo)", 100.0: "100 pb (1 %)", 250.0: "250 pb (2,5 %)"},
        "rotation_edge_bps": {25.0: "25 pb sobre coste", 50.0: "50 pb sobre coste", 100.0: "100 pb sobre coste"},
        "cash_policy": {"fully_invested": "Siempre 100 % invertido", "opportunity_cash": "Efectivo por oportunidad"},
        "max_cash_weight": {0.0: "Sin efectivo", 0.20: "Hasta 20 %", 0.40: "Hasta 40 %"},
        "rebalance_drift_tolerance": {0.0: "Sin tolerancia", 0.10: "10 %", 0.25: "25 %"},
        "price_only_strictness_multiplier": {1.0: "Sin extra (x1,0)", 1.5: "Prudente (x1,5)", 2.0: "Muy prudente (x2,0)"},
        "sizing_mode": {"equal": "Equiponderado", "alpha_proportional": "Proporcional al alfa esperado"},
        "commission_bps": {0.0: "Sin comisión", 5.0: "5 pb (0,05 %)", 10.0: "10 pb (0,10 %)"},
        "slippage_bps": {5.0: "5 pb (0,05 %)", 10.0: "10 pb (0,10 %)", 20.0: "20 pb (0,20 %)"},
    }
    if isinstance(value, bool):
        return "Activado" if value else "Desactivado"
    return labels.get(identifier, {}).get(value, str(value))


def _description(identifier: str, value: Any) -> str:
    specific: dict[str, dict[Any, str]] = {
        "snapshot_step_months": {
            1: "El sistema vuelve a mirar y puntuar todas las acciones cada mes. Es la cadencia más rápida: reacciona antes a cambios, pero genera muchas más decisiones que revisar y más coste de cálculo.",
            3: "El sistema vuelve a mirar y puntuar todas las acciones cada 3 meses (trimestral), en línea con la publicación habitual de resultados de las empresas. Es el punto intermedio entre reaccionar rápido y no generar ruido innecesario.",
            6: "El sistema vuelve a mirar y puntuar todas las acciones cada 6 meses. Cadencia lenta: menos decisiones, pero puede tardar más en detectar cambios de tendencia.",
            12: "El sistema vuelve a mirar y puntuar todas las acciones una vez al año. Es la cadencia más lenta y la más conservadora en número de decisiones.",
        },
        "target_horizon_months": {
            3: "El modelo aprende a ordenar acciones según su retorno esperado en los próximos 3 meses. Horizonte corto: más sensible al ruido de corto plazo.",
            6: "El modelo aprende a ordenar acciones según su retorno esperado en los próximos 6 meses. Horizonte intermedio.",
            12: "El modelo aprende a ordenar acciones según su retorno esperado en los próximos 12 meses (un año). Horizonte largo: suaviza el ruido de corto plazo pero tarda más en confirmar si acertó.",
        },
        "train_lookback_years": {
            4: "Cada vez que se reentrena, el modelo solo mira los últimos 4 años de historia. Menos historia significa que se adapta más rápido a regímenes de mercado recientes, pero con menos ejemplos para aprender.",
            8: "Cada vez que se reentrena, el modelo mira los últimos 8 años de historia. Compromiso entre tener suficientes ejemplos y no arrastrar demasiado pasado ya poco relevante.",
            12: "Cada vez que se reentrena, el modelo mira los últimos 12 años de historia. Máxima cantidad de ejemplos disponibles, a costa de incluir regímenes de mercado más antiguos y posiblemente menos parecidos al presente.",
        },
        "execution_lag_days": {
            30: "Se asume que unos resultados financieros están disponibles para operar solo 30 días después del cierre fiscal de la empresa. Es la hipótesis más optimista/exigente: mayor riesgo de estar usando datos que en la realidad aún no se habían publicado (lookahead), porque muchas empresas tardan más en publicar.",
            45: "Se asume que unos resultados financieros están disponibles para operar 45 días después del cierre fiscal. Compromiso prudente entre actualidad de la información y margen de seguridad frente a publicaciones tardías.",
            60: "Se asume que unos resultados financieros están disponibles para operar solo 60 días después del cierre fiscal. Es el supuesto más conservador: se tarda más en \"creer\" el dato, pero es el que menos riesgo tiene de usar información que aún no existía en ese momento (evita lookahead).",
        },
        "recency_weighting": {
            "off": "Todas las observaciones históricas pesan igual al entrenar, da igual si son de hace 1 mes o de hace 8 años.",
            "linear": "Las observaciones más recientes pesan progresivamente más que las antiguas, de forma proporcional a su antigüedad (una recta). El modelo se adapta algo más rápido a la situación actual del mercado.",
            "exponential": "Las observaciones más recientes pesan mucho más que las antiguas, y ese peso cae rápidamente cuanto más atrás se mira (una curva, no una recta). El modelo prioriza fuertemente lo último visto, al precio de \"olvidar\" antes lo lejano.",
        },
        "objective": {
            "rank_regression": "El modelo aprende a predecir un número (el retorno futuro estimado) y luego ese número se usa para ordenar las acciones. Es un enfoque indirecto: predice una magnitud, no un orden.",
            "ranking": "El modelo aprende directamente a poner unas acciones por delante de otras (aprende el orden, no intenta acertar la magnitud del retorno). Suele ajustar mejor a lo que realmente se necesita: un ranking, no una predicción exacta.",
        },
        "feature_preset": {
            "core": "Cada agente recibe solo su bloque de información más esencial y directo: Calidad usa solo rentabilidad (ROE, márgenes...); Valor usa solo múltiplos de precio (PER, P/VC...); Crecimiento usa solo aceleración de resultados; Momentum usa solo retornos relativos recientes; Riesgo usa solo volatilidad y drawdown de precio. Es el conjunto más pequeño y simple de factores: menos ruido, pero también menos matices.",
            "fundamental": "Los agentes reciben todos los bloques que vienen de los estados financieros de la empresa: rentabilidad, eficiencia operativa, solidez de balance (deuda), múltiplos de valoración, flujo de caja, aceleración de crecimiento y estabilidad de los fundamentales a lo largo del tiempo. No incluye nada calculado a partir del precio de mercado (momentum, volatilidad). Útil para aislar si lo que aporta valor es \"lo que dicen las cuentas de la empresa\".",
            "technical": "Los agentes reciben todos los bloques calculados a partir del precio y el volumen de cotización: momentum a distintos plazos, tendencia respecto a medias móviles, volatilidad/drawdown y liquidez de mercado. No incluye nada de los estados financieros. Útil para aislar si lo que aporta valor es \"cómo se ha movido el precio\", sin mirar las cuentas.",
            "all": "Cada agente recibe absolutamente todos los bloques de información disponibles en el catálogo: fundamentales completos más todos los factores técnicos de precio. Es el conjunto más rico y también el más caro de calcular; puede aportar más señal, pero también más ruido y riesgo de sobreajuste.",
        },
        "winsorization": {
            0.0: "No se recorta ningún valor extremo (outlier) de los factores antes de entrenar. Un dato anómalo (por ejemplo un PER disparatado por un beneficio casi nulo) puede distorsionar el aprendizaje.",
            0.01: "Antes de entrenar, se recortan los valores más extremos de cada factor: el 1 % más alto y el 1 % más bajo de cada variable se \"aplanan\" al límite de ese percentil. Reduce el efecto de outliers sin eliminar observaciones.",
            0.025: "Antes de entrenar, se recortan los valores más extremos de cada factor: el 2,5 % más alto y el 2,5 % más bajo de cada variable se \"aplanan\" al límite de ese percentil. Recorte más agresivo que el 1 %, más protección frente a outliers pero también más pérdida de información real en los extremos.",
        },
        "max_features_per_agent": {
            8: "Cada agente se queda como máximo con las 8 variables más relevantes de su bloque de información, descartando el resto. Modelo más simple y con menos riesgo de sobreajuste, pero puede perder matices útiles.",
            12: "Cada agente se queda como máximo con las 12 variables más relevantes de su bloque de información. Punto intermedio entre simplicidad y riqueza de información.",
            20: "Cada agente puede usar hasta 20 variables de su bloque de información, prácticamente sin recortar. Modelo más rico en información pero con más riesgo de sobreajuste y más lento de entrenar.",
        },
        "feature_weighting_mode": {
            "model_native": "Se deja que el propio modelo (LightGBM o Elastic Net) decida internamente qué variables usar más o menos, sin ningún filtro previo externo.",
            "oos_stability_prune": "Antes de entrenar el modelo final, se descartan las variables cuya utilidad para predecir no se mantiene estable cuando se prueba en datos fuera de la muestra de entrenamiento (out-of-sample). Solo sobreviven las variables que demuestran ser consistentemente útiles, no solo por azar en el pasado.",
        },
        "model_family": {
            "lightgbm": "Se usa LightGBM, un modelo basado en muchos árboles de decisión combinados (gradient boosting). Puede capturar relaciones no lineales y combinaciones complejas entre variables, a cambio de ser menos interpretable y con más parámetros que ajustar.",
            "elastic_net": "Se usa Elastic Net, un modelo lineal regularizado (combina penalización Ridge y Lasso). Es más simple e interpretable que LightGBM, asume relaciones aproximadamente lineales entre los factores y el retorno futuro, y es más resistente al sobreajuste con pocos datos.",
        },
        "lgbm_max_depth": {
            3: "Cada árbol de LightGBM puede hacer como máximo 3 preguntas encadenadas antes de dar un resultado. Árboles poco profundos: más simples, menos riesgo de memorizar ruido (overfitting), pero pueden no capturar interacciones complejas entre factores.",
            4: "Cada árbol de LightGBM puede hacer como máximo 4 preguntas encadenadas. Profundidad intermedia.",
            6: "Cada árbol de LightGBM puede hacer hasta 6 preguntas encadenadas antes de dar un resultado. Árboles más profundos: capturan interacciones más complejas entre factores, pero con más riesgo de memorizar ruido específico de los datos de entrenamiento (overfitting).",
        },
        "lgbm_n_estimators": {
            100: "El modelo combina 100 árboles de decisión sucesivos. Menos árboles: entrena más rápido, pero puede quedarse \"corto\" de capacidad predictiva.",
            200: "El modelo combina 200 árboles de decisión sucesivos. Cantidad intermedia, buen compromiso entre capacidad y velocidad de entrenamiento.",
            400: "El modelo combina 400 árboles de decisión sucesivos. Más árboles: más capacidad de ajuste, pero más lento de entrenar y con más riesgo de sobreajuste si no se controla bien la profundidad y el learning rate.",
        },
        "lgbm_learning_rate": {
            0.03: "Cada árbol nuevo corrige el error de los anteriores con pasos muy pequeños (0,03). Aprendizaje lento pero más estable; suele necesitar más árboles (n_estimators alto) para compensar.",
            0.05: "Cada árbol nuevo corrige el error de los anteriores con pasos moderados (0,05). Punto intermedio entre velocidad de aprendizaje y estabilidad.",
            0.10: "Cada árbol nuevo corrige el error de los anteriores con pasos grandes (0,10). Aprendizaje rápido: converge con menos árboles, pero con más riesgo de \"pasarse\" y sobreajustar.",
        },
        "lgbm_min_child_samples": {
            20: "Un árbol de LightGBM puede crear una hoja final (una regla muy específica) con tan solo 20 observaciones detrás. Permite reglas muy específicas, con más riesgo de que sean simple ruido de pocas acciones concretas.",
            50: "Un árbol de LightGBM necesita al menos 50 observaciones detrás para crear una hoja final. Exige más evidencia antes de crear una regla, reduciendo el riesgo de memorizar casos aislados.",
            100: "Un árbol de LightGBM necesita al menos 100 observaciones detrás para crear una hoja final. Exigencia alta: solo se aceptan patrones respaldados por muchas observaciones, lo que da reglas más robustas pero también más generales (menos matizadas).",
        },
        "meta_method": {
            "equal": "El meta-agente combina a los cinco agentes (calidad, valor, crecimiento, momentum, riesgo) dándoles siempre el mismo peso: 20 % cada uno, sin importar cuál lo esté haciendo mejor o peor. Es la opción más simple y la más difícil de sobreajustar, pero no aprovecha que unos agentes puedan ser más fiables que otros en cada momento.",
            "stacked_rolling_free": "El meta-agente aprende automáticamente (con una regresión Ridge, usando solo información ya cerrada y pasada, nunca del futuro) qué peso dar a cada uno de los cinco agentes, sin ningún límite: en teoría un solo agente podría llegar a pesar el 100 % y el resto el 0 %. Máxima flexibilidad para premiar al mejor agente, pero también máximo riesgo de depender casi en exclusiva de uno solo.",
            "stacked_rolling_bounded": "El meta-agente aprende automáticamente (igual que la versión libre, con Ridge y solo datos ya cerrados) qué peso dar a cada agente, pero obligado a que cada uno de los cinco pese entre un 10 % y un 50 %. Evita los dos extremos: ni ignora completamente a un agente, ni depende de uno solo. Es la opción recomendada por defecto por este equilibrio.",
        },
        "meta_history_quarters": {
            8: "El meta-agente, para decidir los pesos de cada agente, solo mira los últimos 8 trimestres cerrados (2 años) de resultados pasados. Ventana corta: se adapta más rápido a cambios recientes de qué agente funciona mejor, con menos datos para decidirlo con solidez.",
            16: "El meta-agente, para decidir los pesos de cada agente, mira los últimos 16 trimestres cerrados (4 años) de resultados pasados. Ventana más larga: decisión más estable y con más respaldo histórico, pero tarda más en reaccionar si un agente empieza a fallar recientemente.",
        },
        "target_size": {
            8: "La cartera final mantiene 8 acciones distintas a la vez. Cartera concentrada: más impacto de acertar o fallar en cada acción individual.",
            12: "La cartera final mantiene 12 acciones distintas a la vez. Diversificación intermedia.",
            16: "La cartera final mantiene 16 acciones distintas a la vez. Cartera más diversificada: cada acierto o fallo individual pesa menos sobre el resultado total.",
            25: "La cartera final mantiene 25 acciones distintas a la vez. Más amplitud (breadth): por la ley fundamental de la gestión activa, el ratio de información crece con la raíz del número de apuestas independientes, así que ampliar posiciones recupera señal que una cartera concentrada desperdicia.",
            50: "La cartera final mantiene 50 acciones distintas a la vez. Máxima amplitud del catálogo: cada acción pesa poco y el resultado depende de la calidad media de la ordenación, no de unos pocos aciertos. Es la configuración que más se acerca a cosechar el Rank-IC completo, a costa de parecerse más al índice.",
        },
        "exit_expected_alpha_bps": {
            0.0: "Una acción en cartera solo se vende cuando su alfa esperado se vuelve negativo. Umbral mínimo: la posición se conserva mientras el modelo espere de ella algo mejor que el mercado, por poco que sea. Es el criterio que menos opera.",
            100.0: "Una acción en cartera se vende cuando su alfa esperado cae por debajo de 100 puntos básicos (1 %) sobre el índice, en el horizonte del modelo. Exige una expectativa positiva clara para seguir ocupando una plaza.",
            250.0: "Una acción en cartera se vende cuando su alfa esperado cae por debajo de 250 puntos básicos (2,5 %) sobre el índice. Umbral exigente: solo conserva convicciones fuertes, lo que aumenta la rotación y, con ella, los costes.",
        },
        "rotation_edge_bps": {
            25.0: "Para sustituir una posición por una candidata externa, la ventaja de alfa esperado debe superar el coste completo de ida y vuelta de la operación más 25 puntos básicos. Umbral bajo: rota con relativa facilidad, pero nunca por debajo de lo que cuesta operar.",
            50.0: "Para sustituir una posición por una candidata externa, la ventaja de alfa esperado debe superar el coste completo de ida y vuelta más 50 puntos básicos. Exigencia intermedia.",
            100.0: "Para sustituir una posición por una candidata externa, la ventaja de alfa esperado debe superar el coste de ida y vuelta más 100 puntos básicos. Umbral alto: solo rota ante una mejora económica clara, lo que reduce mucho la rotación y el coste asociado.",
        },
        "cash_policy": {
            "fully_invested": "La cartera está siempre invertida al 100 % en acciones: si una plaza queda libre se rellena con la mejor candidata disponible, aunque su alfa esperado sea bajo. Es la política de referencia y la que se compara contra el índice sin ninguna ventaja de posicionamiento.",
            "opportunity_cash": "Cuando ninguna candidata supera el umbral de alfa esperado, la plaza se deja en efectivo en lugar de comprar la menos mala. El efectivo se remunera al 0 %, así que nunca aporta rentabilidad: solo puede ayudar evitando malas compras y ahorrando costes de operación. La decisión sale exclusivamente de la sección transversal de candidatas, nunca de una previsión sobre el mercado.",
        },
        "max_cash_weight": {
            0.0: "No se permite efectivo aunque la política lo autorice: equivale a estar siempre invertido.",
            0.20: "Como máximo un 20 % de la cartera puede quedar en efectivo a la espera de oportunidades. El 80 % restante permanece invertido incluso si las candidatas son mediocres.",
            0.40: "Como máximo un 40 % de la cartera puede quedar en efectivo. Tolerancia amplia: en periodos sin oportunidades claras la cartera se repliega bastante, lo que reduce el riesgo pero también la exposición a subidas del mercado.",
        },
        "rebalance_drift_tolerance": {
            0.0: "Cualquier desviación, por mínima que sea, entre el peso actual de una posición y su peso objetivo genera una orden de ajuste. Máxima precisión de pesos, al coste de generar muchas más operaciones (y comisiones).",
            0.10: "Solo se ajusta una posición cuando su peso real se desvía más de un 10 % (en términos relativos) de su peso objetivo. Reduce operaciones innecesarias por desviaciones pequeñas del peso, a cambio de tolerar carteras algo menos ajustadas al objetivo teórico.",
            0.25: "Solo se ajusta una posición cuando su peso real se desvía más de un 25 % (en términos relativos) de su peso objetivo. Tolerancia amplia: pocas operaciones de puro rebalanceo, pero los pesos reales pueden alejarse bastante del objetivo teórico entre ajustes.",
        },
        "price_only_strictness_multiplier": {
            1.0: "Cuando en un snapshot no hay fundamentales nuevos publicados (solo hay precio actualizado), se exigen exactamente los mismos umbrales de venta/compra que cuando sí hay fundamentales frescos. Sin prudencia extra.",
            1.5: "Cuando en un snapshot no hay fundamentales nuevos publicados, los umbrales de venta/compra se multiplican por 1,5 (se vuelven más exigentes). Precaución moderada: evita mover la cartera solo por ruido de precio sin confirmación fundamental reciente.",
            2.0: "Cuando en un snapshot no hay fundamentales nuevos publicados, los umbrales de venta/compra se multiplican por 2 (mucho más exigentes). Máxima prudencia: casi no se opera una posición basándose solo en el movimiento de precio, sin datos fundamentales frescos que lo respalden.",
        },
        "sizing_mode": {
            "equal": "Todas las posiciones objetivo de la cartera reciben exactamente el mismo peso (equiponderado), sin importar si una acción tiene una puntuación mucho mejor que otra.",
            "alpha_proportional": "El peso de cada posición escala con su alfa esperado calibrado: la posición con menor alfa esperado de la cartera recibe peso base (x1) y la de mayor alfa esperado recibe el doble (x2). A diferencia de escalar por percentil, aquí el peso responde a una magnitud económica estimada y no a la posición relativa en un ranking.",
        },
        "commission_bps": {
            0.0: "No se descuenta ninguna comisión por operar. Escenario idealizado, sobreestima el resultado neto real.",
            5.0: "Se descuenta una comisión de 5 puntos básicos (0,05 %) sobre el importe de cada operación. Coste de bróker moderado.",
            10.0: "Se descuenta una comisión de 10 puntos básicos (0,10 %) sobre el importe de cada operación. Coste de bróker más alto; penaliza más las estrategias con muchas operaciones (turnover alto).",
        },
        "slippage_bps": {
            5.0: "Se asume que, además de la comisión, cada operación pierde 5 puntos básicos (0,05 %) extra por la diferencia entre el precio esperado y el precio real de ejecución (slippage). Supuesto optimista de liquidez.",
            10.0: "Se asume un slippage de 10 puntos básicos (0,10 %) por operación. Supuesto intermedio.",
            20.0: "Se asume un slippage de 20 puntos básicos (0,20 %) por operación. Supuesto conservador: penaliza más las estrategias que rotan mucho la cartera, reflejando peor liquidez o impacto de mercado.",
        },
    }
    booleans: dict[str, dict[bool, str]] = {
        "fundamental_momentum": {
            True: "Se añaden variables que miden cómo han cambiado los fundamentales de la empresa en el tiempo (por ejemplo si el ROE está mejorando o empeorando trimestre a trimestre), no solo su valor actual. Da al modelo información de tendencia, no solo de nivel.",
            False: "El modelo solo ve el valor actual de cada fundamental (por ejemplo el ROE de hoy), sin información de si viene mejorando o empeorando en el tiempo.",
        },
        "market_regime_feature": {
            True: "Se añade una variable con el contexto general del mercado en cada fecha (por ejemplo si el mercado en conjunto está en un momento alcista o bajista), calculada solo con información ya disponible en ese momento. Ayuda al modelo a adaptar su criterio según el entorno de mercado.",
            False: "El modelo no recibe ninguna información sobre el estado general del mercado; solo ve datos propios de cada empresa, sin contexto macro.",
        },
        "neutralize_by_sector": {
            True: "Antes de aprender, cada empresa se compara solo contra otras empresas de su mismo sector (por ejemplo, tecnológicas contra tecnológicas), no contra todo el universo. Evita que el modelo confunda \"esta empresa es buena\" con \"este sector entero lo está haciendo bien\".",
            False: "Todas las empresas se comparan entre sí directamente, sin ajustar por sector. Más simple, pero un sector en auge puede parecer sistemáticamente \"mejor\" sin que sea mérito de las empresas concretas.",
        },
    }
    if identifier in specific and value in specific[identifier]:
        return specific[identifier][value]
    if identifier in booleans and isinstance(value, bool):
        return booleans[identifier][value]
    if isinstance(value, bool):
        return "Incluye esta capacidad." if value else "No incluye esta capacidad."
    return f"Utiliza {value} como valor de esta variable."


def default_definition() -> dict[str, dict[str, Any]]:
    return {
        variable.id: {"mode": "fixed", "values": [variable.recommended], "baseline": variable.recommended}
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
        definition[identifier] = {
            "mode": "optimize", "values": values, "baseline": BY_ID[identifier].recommended,
        }
    # Las variables de cartera se barren como diagnóstico con el ganador predictivo ya congelado:
    # nunca modifican la selección. Una variable dependiente se deja fija cuando su controlador es
    # el que se está barriendo, porque comparar sus valores bajo dos regímenes distintos del
    # controlador mezclaría dos preguntas en un único resultado.
    swept = {
        variable.id for variable in VARIABLES
        if not variable.predictive and len(variable.values) > 1
    }
    for variable in VARIABLES:
        if variable.predictive:
            continue
        controllers_swept = any(controller in swept for controller, _ in variable.depends_on)
        if controllers_swept:
            definition[variable.id] = {
                "mode": "fixed", "values": [variable.recommended], "baseline": variable.recommended,
            }
            continue
        alternatives = [value for value in variable.values if value != variable.recommended]
        definition[variable.id] = {
            "mode": "diagnostic",
            "values": [variable.recommended, *alternatives[:1]],
            "baseline": variable.recommended,
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
