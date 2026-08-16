"""Configuración científica inmutable usada por el runner único."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_JSON_DIR = RAW_DIR / "json"
DEV_RAW_DIR = RAW_DIR / "dev"
PREPARED_DIR = DATA_DIR / "prepared"
RUN_SCOPES = ("dev", "full")


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")

# Ventana temporal descargada: permite entrenar hacia atrás desde el año 2000.
DATA_START_DATE = "2003-01-01"
DATA_END_DATE = "2026-07-15"
DEV_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
BENCHMARK_TICKER = "SPY"
# Fase 1: el panel se construye para toda la historia disponible. La fecha ancla
# se usará después para iniciar la evaluación OOS.
# Fecha ancla FIJA: el sistema evalua la simulación OOS desde 2015-Q1 (~10 años de OOS hasta hoy
# con 2025-26 reservados), entrenando 8 años hacia atras. El año/trimestre ancla NO se barren en
# el study (son fijos); sí se barre el retardo de publicación (execution_lag_days).
EXECUTION_YEAR = 2015
EXECUTION_QUARTER = 1
EXECUTION_LAG_DAYS = 45
TRAIN_LOOKBACK_YEARS = 8
SNAPSHOT_STEP_MONTHS = 1
FUNDAMENTAL_STEP_MONTHS = 3
TARGET_HORIZON_MONTHS = 6
MAX_PRICE_AGE_DAYS = 7
META_HISTORY_QUARTERS = 16
META_WEIGHT_CAP = 1.0
MIN_TRAINING_ROWS = 30
MIN_RANK_IC_CROSS_SECTION = 10

# Pesos de recencia en el entrenamiento: da más peso a los años recientes de la ventana.
#   "off"         -> todas las filas pesan igual (comportamiento por defecto).
#   "linear"      -> el año más reciente pesa más, decreciendo linealmente hacia el más antiguo.
#   "exponential" -> decaimiento exponencial con vida media RECENCY_HALFLIFE_YEARS.
# El catálogo permite contrastar si priorizar lo reciente mejora el Rank-IC.
RECENCY_WEIGHTING = "off"
RECENCY_HALFLIFE_YEARS = 3.0       # vida media (años) del decaimiento exponencial; fija

# Mismo mecanismo que RECENCY_WEIGHTING, pero sobre el Ridge del meta-agente: pondera las cohortes
# ya cerradas por antigüedad al aprender el peso de cada agente. La ventana del stacker
# (META_HISTORY_QUARTERS) es rectangular, así que sin esto una cohorte de hace cuatro años pesa lo
# mismo que la última al decidir en qué agente confiar. Reutiliza RECENCY_HALFLIFE_YEARS.
META_RECENCY_WEIGHTING = "off"

# Mismo mecanismo que RECENCY_WEIGHTING, pero aplicado al Ridge del meta-agente (module/modeling/meta.py):
# da más peso a las cohortes trimestrales recientes al aprender los pesos entre los cinco agentes.
# Reutiliza RECENCY_HALFLIFE_YEARS para el modo "exponential".
META_RECENCY_WEIGHTING = "off"

# --- Modelo LightGBM ---
# Objetivo de aprendizaje:
#   "rank_regression" (principal) -> regresion sobre el PERCENTIL transversal del retorno (0..1)
#                                    dentro de cada snapshot; alinea el entrenamiento con el rank-IC.
#   "ranking"  -> LGBMRanker (lambdarank) agrupado por snapshot; optimiza el orden directamente.
OBJECTIVE = "rank_regression"
# Hiperparametros LightGBM. Defaults conservadores (arboles poco profundos, muchas muestras
# mínimas por hoja) por el número limitado de eras independientes.
LGBM_N_ESTIMATORS = 200
LGBM_MAX_DEPTH = 4
LGBM_LEARNING_RATE = 0.05
LGBM_MIN_CHILD_SAMPLES = 50
# Semilla base del modelo. Cada agente entrena SEED_ENSEMBLE réplicas que solo difieren en la
# semilla y promedia sus scores. No es una variable científica a optimizar sino reducción de varianza
# del estimador: el Rank-IC apenas se movía entre semillas (±0,001) pero el resultado económico de
# una cartera concentrada llegaba a cambiar de signo, porque 12 posiciones amplifican el ruido de
# inicialización de LightGBM. El barrido de semillas se conserva como diagnóstico de robustez, ahora
# sobre el ensemble completo.
RANDOM_SEED = 42
SEED_ENSEMBLE = 5
# Meta-agente: como se combinan los agentes.
#   "equal"       -> promedio equiponderado de los rangos de los agentes
#   "stacked_oos" -> Ridge no negativo causal sobre cohortes ya cerradas
META_TYPE = "stacked_oos"

# --- Laboratorio de factores/ML -------------------------------------------------
# Tuplas (no listas) para que Settings siga siendo inmutable y serializable en huellas.
ENABLED_FEATURE_BLOCKS = (
    "quality_core", "quality_efficiency", "financial_strength", "value_core", "value_cashflow",
    "growth_acceleration", "fundamental_stability", "momentum_core", "momentum_trend",
    "price_risk", "market_liquidity",
)
ENABLED_AGENTS = ("quality", "value", "growth", "momentum", "risk")
ENABLED_MODEL_FAMILIES = ("lightgbm",)
FEATURE_WEIGHTING_MODE = "model_native"
FEATURE_SELECTION_MAX_FEATURES_PER_AGENT = 0  # 0 = sin límite
# Constantes de la poda causal de features. No son variables científicas: el catálogo no las expone,
# así que no pueden variar entre evaluaciones y no tienen sitio en las huellas de caché.
FEATURE_SELECTION_MIN_COVERAGE = 0.55
FEATURE_SELECTION_LOOKBACK_QUARTERS = 12
FEATURE_SELECTION_MIN_POSITIVE_FRACTION = 0.50
METRIC_WINSORIZATION_PERCENTILE = 0.01
RISK_FEATURE_WINDOWS = (63, 126, 252)
TECHNICAL_FEATURE_WINDOWS = (21, 63, 252)

# --- Bloques opcionales del catálogo, todos point-in-time ---
NEUTRALIZE_BY_SECTOR = False       # rankear factores dentro de sector en vez de global
FUNDAMENTAL_MOMENTUM = False       # tendencia de fundamentales + descomposicion P/E precio vs fundamental
MARKET_REGIME_FEATURE = False      # regimen bull/bear del SP500 + interacciones factor x regimen
NEUTRALIZE_MIN_GROUP = 5           # tamano minimo de grupo para neutralizar por sector

# --- Cartera ---
# Los umbrales son económicos, en puntos básicos de alfa esperado ANUALES, no percentiles del
# ranking: se convierten geométricamente al horizonte real del modelo antes de compararlos (ver
# module/evaluation/portfolio.py::_annual_to_horizon_bps). Definirlos en anual los hace comparables
# entre configuraciones con distinto target_horizon_months o snapshot_step_months.
TARGET_SIZE = 8
EXIT_EXPECTED_ALPHA_BPS = 100.0   # pb/año; alfa esperado por debajo del equivalente -> venta
ROTATION_EDGE_BPS = 50.0          # pb/año; ventaja exigida POR ENCIMA del coste de ida y vuelta
# Tope de efectivo: peso máximo que puede quedar sin invertir cuando ninguna candidata supera el
# umbral de alfa esperado. Es la ÚNICA variable que gobierna el efectivo; `MAX_CASH_WEIGHT = 0`
# significa "siempre 100 % invertido" y hace que el suelo de diversificación coincida con
# `TARGET_SIZE`, así que no hace falta una política aparte que diga lo mismo. El efectivo se
# remunera al 0 %: es una cota inferior conservadora, nunca aporta rentabilidad, solo evita malas
# compras.
MAX_CASH_WEIGHT = 0.25
COMMISSION_BPS = 5                # comision por operacion, en puntos basicos
SLIPPAGE_BPS = 10                 # slippage por operacion, en puntos basicos
REBALANCE_DRIFT_TOLERANCE = 0.25  # fracción mínima de cambio RELATIVO a la posición para rebalancear
                                  # (0.25 = 25 %); por debajo, el tenente se "congela" y no opera
# Mínimo de meses en cartera antes de poder vender una posición POR NINGÚN MOTIVO (ni caída de alfa
# ni rotación), como fracción del horizonte del modelo. No busca más alfa: busca estabilidad y menos
# rotación de alta frecuencia sobre el mismo modelo ya congelado.
#   "none"           -> sin mínimo (comportamiento histórico)
#   "quarter_horizon" -> ceil(horizonte / 4)
#   "half_horizon"    -> ceil(horizonte / 2)
#   "full_horizon"    -> horizonte completo
MINIMUM_HOLDING_PERIOD = "none"
MINIMUM_HOLDING_PERIODS = ("none", "quarter_horizon", "half_horizon", "full_horizon")
# Suelo de cobertura del ranking, en percentil 0-100. Una posición que cae por debajo deja de
# pertenecer a una cartera construida sobre este ranking, con independencia de su alfa esperado: es
# la generalización de `missing_current_score` (percentil ausente) a "percentil demasiado bajo". No
# se compara contra ningún coste, porque no decide si una operación es rentable sino si la acción
# sigue perteneciendo al universo invertible: es un mandato, no una decisión económica. A diferencia
# de la pérdida de cobertura, sí respeta MINIMUM_HOLDING_PERIOD. 0 desactiva la regla.
COVERAGE_PERCENTILE_FLOOR = 0.0
# En un snapshot que solo trae precio nuevo (sin fundamentales frescos), permite vender una posición
# que ya no cumple pero prohíbe comprar cualquier reemplazo (ni compra nueva, ni relleno obligatorio,
# ni rotación): no hay información nueva que justifique elegir una acción distinta a la ya elegida
# con datos reales. False = comportamiento histórico (compras y rotación normales todo el tiempo).
PRICE_ONLY_SELL_ONLY = False
# Con snapshot_step_months=1 y fundamental_step_months=3 hay revisiones que solo traen precio nuevo,
# sin fundamentales nuevos. Este factor endurece expulsión, rotación y rebalanceo en esas revisiones
# para no rotar la cartera por ruido de precio. 1.0 = sin efecto (comportamiento histórico).
PRICE_ONLY_STRICTNESS_MULTIPLIER = 1.0
# Guarda anti-artefactos de datos: un retorno mensual de una posicion mayor que esto se trata
# como dato corrupto (split mal ajustado, ticker reciclado) y se neutraliza, registrandolo.
MAX_MONTHLY_POSITION_RETURN = 2.0  # +200 % en un mes es imposible para una accion normal

# Traducción señal → cartera. Son defaults internos; el catálogo fija la ejecución real.
SIZING_MODE = "alpha_proportional"
SIZING_MODES = ("equal", "alpha_proportional")
META_WEIGHT_MIN = 0.10

RUN_SCOPE = os.getenv("RUN_SCOPE", "full").strip().lower()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
EDGAR_USER_AGENT = os.getenv(
    "EDGAR_USER_AGENT", "TFM Academic Research adri4nc2001@gmail.com"
)

# Un snapshot por día de mercado (1996-2026): fuente del universo dinámico.
SP500_COMPONENTS_CSV = DATA_DIR / "S&P 500 Historical Components & Changes.csv"


@dataclass(frozen=True)
class Settings:
    data_start_date: str = DATA_START_DATE
    end_date: str = DATA_END_DATE
    run_scope: str = RUN_SCOPE
    benchmark_ticker: str = BENCHMARK_TICKER
    execution_year: int = EXECUTION_YEAR
    execution_quarter: int = EXECUTION_QUARTER
    execution_lag_days: int = EXECUTION_LAG_DAYS
    train_lookback_years: int = TRAIN_LOOKBACK_YEARS
    snapshot_step_months: int = SNAPSHOT_STEP_MONTHS
    fundamental_step_months: int = FUNDAMENTAL_STEP_MONTHS
    target_horizon_months: int = TARGET_HORIZON_MONTHS
    min_rank_ic_cross_section: int = MIN_RANK_IC_CROSS_SECTION
    objective: str = OBJECTIVE
    lgbm_n_estimators: int = LGBM_N_ESTIMATORS
    lgbm_max_depth: int = LGBM_MAX_DEPTH
    lgbm_learning_rate: float = LGBM_LEARNING_RATE
    lgbm_min_child_samples: int = LGBM_MIN_CHILD_SAMPLES
    # Hilos de LightGBM por fit. No altera la definición científica.
    lgbm_n_jobs: int = -1
    random_seed: int = RANDOM_SEED
    meta_type: str = META_TYPE
    meta_history_quarters: int = META_HISTORY_QUARTERS
    meta_weight_cap: float = META_WEIGHT_CAP
    recency_weighting: str = RECENCY_WEIGHTING
    meta_recency_weighting: str = META_RECENCY_WEIGHTING
    enabled_feature_blocks: tuple[str, ...] = ENABLED_FEATURE_BLOCKS
    enabled_agents: tuple[str, ...] = ENABLED_AGENTS
    enabled_model_families: tuple[str, ...] = ENABLED_MODEL_FAMILIES
    feature_weighting_mode: str = FEATURE_WEIGHTING_MODE
    feature_selection_max_features_per_agent: int = FEATURE_SELECTION_MAX_FEATURES_PER_AGENT
    metric_winsorization_percentile: float = METRIC_WINSORIZATION_PERCENTILE
    risk_feature_windows: tuple[int, ...] = RISK_FEATURE_WINDOWS
    technical_feature_windows: tuple[int, ...] = TECHNICAL_FEATURE_WINDOWS
    # artefactos activables
    neutralize_by_sector: bool = NEUTRALIZE_BY_SECTOR
    fundamental_momentum: bool = FUNDAMENTAL_MOMENTUM
    market_regime_feature: bool = MARKET_REGIME_FEATURE
    # cartera
    target_size: int = TARGET_SIZE
    exit_expected_alpha_bps: float = EXIT_EXPECTED_ALPHA_BPS
    rotation_edge_bps: float = ROTATION_EDGE_BPS
    max_cash_weight: float = MAX_CASH_WEIGHT
    commission_bps: float = COMMISSION_BPS
    slippage_bps: float = SLIPPAGE_BPS
    rebalance_drift_tolerance: float = REBALANCE_DRIFT_TOLERANCE
    minimum_holding_period: str = MINIMUM_HOLDING_PERIOD
    coverage_percentile_floor: float = COVERAGE_PERCENTILE_FLOOR
    price_only_sell_only: bool = PRICE_ONLY_SELL_ONLY
    price_only_strictness_multiplier: float = PRICE_ONLY_STRICTNESS_MULTIPLIER
    max_monthly_position_return: float = MAX_MONTHLY_POSITION_RETURN
    profile: str = "balanced"   # perfil de inversor para la seleccion de cartera (ver module/profiles.py)
    sizing_mode: str = SIZING_MODE
    meta_weight_min: float = META_WEIGHT_MIN
    # Versiones explícitas para invalidar materializaciones y caché tras cambios científicos.
    dataset_code_version: int = 2
    features_code_version: int = 2
    # El fit de las familias y la combinación meta tienen versiones separadas: cambiar la
    # combinación no debe invalidar los fits LightGBM ya calculados.
    agents_fit_code_version: int = 3
    # Ruta privada de ejecución. No es una variable científica ni se incluye en fingerprints;
    # el orquestador la asigna después de crear el run.
    workspace_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.run_scope not in RUN_SCOPES:
            raise ValueError(
                f"RUN_SCOPE inválido: {self.run_scope!r}. Valores admitidos: {', '.join(RUN_SCOPES)}."
            )

        if self.execution_quarter not in (1, 2, 3, 4):
            raise ValueError("EXECUTION_QUARTER debe estar entre 1 y 4.")
        if self.objective not in ("rank_regression", "ranking"):
            raise ValueError(
                f"OBJECTIVE inválido: {self.objective!r}. Usa 'rank_regression' o 'ranking'."
            )
        if self.meta_type not in ("equal", "stacked_oos"):
            raise ValueError(
                f"META_TYPE invalido: {self.meta_type!r}. Usa 'equal' o 'stacked_oos'."
            )
        if self.meta_history_quarters < 1:
            raise ValueError("META_HISTORY_QUARTERS debe ser positivo.")
        if not 0 < self.meta_weight_cap <= 1:
            raise ValueError("META_WEIGHT_CAP debe estar en (0, 1].")
        if self.recency_weighting not in ("off", "linear", "exponential"):
            raise ValueError(
                f"RECENCY_WEIGHTING invalido: {self.recency_weighting!r}. "
                "Usa 'off', 'linear' o 'exponential'."
            )
        if self.meta_recency_weighting not in ("off", "linear", "exponential"):
            raise ValueError(
                f"META_RECENCY_WEIGHTING invalido: {self.meta_recency_weighting!r}. "
                "Usa 'off', 'linear' o 'exponential'."
            )
        if self.feature_weighting_mode not in ("model_native", "oos_stability_prune"):
            raise ValueError("FEATURE_WEIGHTING_MODE no reconocido.")
        if not self.enabled_agents:
            raise ValueError("ENABLED_AGENTS no puede estar vacío.")
        if not 0 <= self.metric_winsorization_percentile < 0.5:
            raise ValueError("METRIC_WINSORIZATION_PERCENTILE debe estar en [0, 0.5).")
        if self.train_lookback_years <= 0 or self.target_horizon_months <= 0:
            raise ValueError("Las ventanas temporales deben ser positivas.")
        if self.target_size < 1:
            raise ValueError("TARGET_SIZE debe ser positivo.")
        if self.sizing_mode not in SIZING_MODES:
            raise ValueError(f"SIZING_MODE no reconocido: {self.sizing_mode!r}.")
        if not 0 <= self.meta_weight_min <= self.meta_weight_cap:
            raise ValueError("META_WEIGHT_MIN debe estar entre 0 y META_WEIGHT_CAP.")
        if not 0 <= self.max_cash_weight < 1:
            raise ValueError("MAX_CASH_WEIGHT es una fracción de la cartera y debe estar en [0, 1).")
        if not 0 <= self.coverage_percentile_floor < 100:
            raise ValueError(
                "COVERAGE_PERCENTILE_FLOOR es un percentil del ranking en escala 0-100 y debe estar "
                "en [0, 100). 0 desactiva la regla."
            )
        if self.rotation_edge_bps < 0:
            raise ValueError(
                "ROTATION_EDGE_BPS es un margen POR ENCIMA del coste de ida y vuelta y no puede ser "
                "negativo: una rotación nunca debe autorizarse por debajo de lo que cuesta operarla."
            )
        if self.rebalance_drift_tolerance < 0:
            raise ValueError(
                "REBALANCE_DRIFT_TOLERANCE es una fracción relativa (0.25 = 25 %) y debe ser >= 0."
            )
        if self.price_only_strictness_multiplier < 1:
            raise ValueError(
                "PRICE_ONLY_STRICTNESS_MULTIPLIER debe ser >= 1: endurece las revisiones sin "
                "fundamentales nuevos, nunca las relaja."
            )
        if self.minimum_holding_period not in MINIMUM_HOLDING_PERIODS:
            raise ValueError(
                f"MINIMUM_HOLDING_PERIOD inválido: {self.minimum_holding_period!r}. "
                f"Usa uno de {MINIMUM_HOLDING_PERIODS}."
            )

    @property
    def dev_mode(self) -> bool:
        """Compatibilidad semántica: el alcance ``dev`` usa una muestra aislada."""
        return self.run_scope == "dev"

    @property
    def raw_output_dir(self) -> Path:
        """Directorio de agregados; nunca mezcla resultados dev y full."""
        return DEV_RAW_DIR if self.dev_mode else RAW_DIR

    @property
    def processed_output_dir(self) -> Path:
        """Directorio de artefactos procesados, aislado por alcance."""
        if self.workspace_dir is not None:
            return Path(self.workspace_dir)
        return PREPARED_DIR / ("standalone-dev" if self.dev_mode else "standalone")

    @property
    def tickers(self) -> list[str]:
        """Universo histórico completo o muestra de desarrollo más el benchmark."""
        from module.data.universe import historical_universe

        base = DEV_TICKERS if self.dev_mode else sorted(historical_universe())
        return list(dict.fromkeys([*base, self.benchmark_ticker]))
