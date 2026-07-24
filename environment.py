"""Configuración central y modos de ejecución del proyecto.

Ejemplos en PowerShell:
    $env:RUN_MODE = "download"; $env:RUN_SCOPE = "dev"; python main.py
    $env:RUN_MODE = "download"; $env:RUN_SCOPE = "full"; python main.py

``RUN_MODE`` selecciona la etapa que se ejecuta y ``RUN_SCOPE`` el alcance de
sus datos. Así se puede repetir una etapa sin volver a descargar las anteriores.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_JSON_DIR = RAW_DIR / "json"
DEV_RAW_DIR = RAW_DIR / "dev"
PROCESSED_DIR = DATA_DIR / "processed"
DEV_PROCESSED_DIR = PROCESSED_DIR / "dev"

RUN_MODES = (
    "download",
    "dataset",
    "features",
    "agents",
    "backtest",
    "report",
    "experiments",
    "full_study",
    "full",
)
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
# se usará después para iniciar una simulación o un escenario concreto.
# Fecha ancla FIJA: el sistema evalua la simulación OOS desde 2015-Q1 (~10 años de OOS hasta hoy
# con 2025-26 reservados), entrenando 8 años hacia atras. El año/trimestre ancla NO se barren en
# el study (son fijos); sí se barre el retardo de publicación (execution_lag_days).
EXECUTION_YEAR = 2015
EXECUTION_QUARTER = 1
EXECUTION_LAG_DAYS = 45
TRAIN_LOOKBACK_YEARS = 8
SNAPSHOT_STEP_MONTHS = 1           # cadencia de revision de cartera; barrida como escenario
FUNDAMENTAL_STEP_MONTHS = 3        # cadencia de reentreno; barrida como escenario (3/12/1)
TARGET_HORIZON_MONTHS = 6
MAX_PRICE_AGE_DAYS = 7
META_IC_LOOKBACK_QUARTERS = 12
MIN_TRAINING_ROWS = 30
MIN_RANK_IC_CROSS_SECTION = 10

# Pesos de recencia en el entrenamiento: da más peso a los años recientes de la ventana.
#   "off"         -> todas las filas pesan igual (comportamiento por defecto).
#   "linear"      -> el año más reciente pesa más, decreciendo linealmente hacia el más antiguo.
#   "exponential" -> decaimiento exponencial con vida media RECENCY_HALFLIFE_YEARS.
# Barrida como escenario para medir si priorizar lo reciente mejora el rank-IC.
RECENCY_WEIGHTING = "off"
RECENCY_HALFLIFE_YEARS = 3.0       # vida media (años) del decaimiento exponencial; fija

# --- Modelo LightGBM ---
# Objetivo de aprendizaje:
#   "rank_regression" (principal) -> regresion sobre el PERCENTIL transversal del retorno (0..1)
#                                    dentro de cada snapshot; alinea el entrenamiento con el rank-IC.
#   "ranking"  -> LGBMRanker (lambdarank) agrupado por snapshot; optimiza el orden directamente.
#   "quartile" -> clasifica cuartil superior vs inferior (ablacion).
OBJECTIVE = "rank_regression"
# Hiperparametros LightGBM. Defaults conservadores (arboles poco profundos, muchas muestras
# minimas por hoja) por el numero limitado de eras independientes. Barridos como escenario.
LGBM_N_ESTIMATORS = 200
LGBM_MAX_DEPTH = 4
LGBM_LEARNING_RATE = 0.05
LGBM_MIN_CHILD_SAMPLES = 50
# Semilla del modelo. Fija en la comparacion principal; se barre para medir robustez del ganador.
RANDOM_SEED = 42
# Meta-agente: como se combinan los 3 agentes (equal | rank_ic | regime).
#   "equal"   -> promedio equiponderado de los rangos de los agentes
#   "rank_ic" -> ponderacion por rank-IC reciente de cada agente (mejor medido: bate al equal)
#   "regime"  -> pesos distintos segun regimen bull/bear
META_TYPE = "rank_ic"

# --- Laboratorio de factores/ML -------------------------------------------------
# Tuplas (no listas) para que Settings siga siendo inmutable y serializable en huellas.
ENABLED_FEATURE_BLOCKS = (
    "quality_core", "quality_efficiency", "financial_strength", "value_core", "value_cashflow",
    "growth_acceleration", "fundamental_stability", "momentum_core", "momentum_trend",
    "price_risk", "market_liquidity",
)
ENABLED_AGENTS = ("quality", "value", "growth", "momentum", "risk")
ENABLED_MODEL_FAMILIES = ("lightgbm", "elastic_net", "catboost")
INTRA_AGENT_ENSEMBLE_MODE = "rank_ic_weighted"
FEATURE_WEIGHTING_MODE = "diagnostic_only"
FEATURE_SELECTION_MIN_COVERAGE = 0.55
FEATURE_SELECTION_LOOKBACK_QUARTERS = 12
FEATURE_SELECTION_MIN_PERMUTATION_IMPORTANCE = 0.0
FEATURE_SELECTION_MIN_POSITIVE_FRACTION = 0.50
FEATURE_SELECTION_MAX_FEATURES_PER_AGENT = 0  # 0 = sin límite
METRIC_WINSORIZATION_PERCENTILE = 0.01
RISK_FEATURE_WINDOWS = (63, 126, 252)
TECHNICAL_FEATURE_WINDOWS = (21, 63, 252)

# --- Artefactos activables (bloques de features/contexto que el barrido activa como ablations) ---
# Cada uno es point-in-time. El barrido mide si suben el rank-IC del meta_final. Ver module/artifacts.py.
NEUTRALIZE_BY_SECTOR = False       # rankear factores dentro de sector en vez de global
FUNDAMENTAL_MOMENTUM = False       # tendencia de fundamentales + descomposicion P/E precio vs fundamental
MARKET_REGIME_FEATURE = False      # regimen bull/bear del SP500 + interacciones factor x regimen
PRICE_MOMENTUM_MULTI = False       # aceleracion (r3m-r12m), reversion (-r1m), volatilidad reciente
MOVING_AVERAGES = False            # precio vs SMA200/SMA50, distancia a maximo de 12m
REGIME_EXTENDED = False            # vol del SP500, drawdown del indice, amplitud
QUALITY_GROWTH_DERIVED = False     # tendencia de ROE/margenes, estabilidad, sorpresa de crecimiento
NEUTRALIZE_MIN_GROUP = 5           # tamano minimo de grupo para neutralizar por sector

# --- Cartera ---
# Cartera fija: el top-N del meta-rank entra y permanece mientras conserve calidad relativa.
TARGET_SIZE = 8
MIN_HOLD_PERCENTILE = 80          # percentil <= este umbral -> venta obligatoria
ROTATION_EDGE_PERCENTILES = 10    # ventaja mínima para que un candidato desplace a un tenente
COMMISSION_BPS = 5                # comision por operacion, en puntos basicos
SLIPPAGE_BPS = 10                 # slippage por operacion, en puntos basicos
REBALANCE_DRIFT_TOLERANCE = 0.25  # fracción mínima de cambio RELATIVO a la posición para rebalancear
                                  # (0.25 = 25 %); por debajo, el tenente se "congela" y no opera
# Con snapshot_step_months=1 y fundamental_step_months=3 hay revisiones que solo traen precio nuevo,
# sin fundamentales nuevos. Este factor endurece expulsión, rotación y rebalanceo en esas revisiones
# para no rotar la cartera por ruido de precio. 1.0 = sin efecto (comportamiento histórico).
PRICE_ONLY_STRICTNESS_MULTIPLIER = 1.0
# Guarda anti-artefactos de datos: un retorno mensual de una posicion mayor que esto se trata
# como dato corrupto (split mal ajustado, ticker reciclado) y se neutraliza, registrandolo.
MAX_MONTHLY_POSITION_RETURN = 2.0  # +200 % en un mes es imposible para una accion normal

# Se pueden establecer temporalmente desde la consola sin editar este archivo.
RUN_MODE = os.getenv("RUN_MODE", "download").strip().lower()
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
    run_mode: str = RUN_MODE
    run_scope: str = RUN_SCOPE
    benchmark_ticker: str = BENCHMARK_TICKER
    execution_year: int = EXECUTION_YEAR
    execution_quarter: int = EXECUTION_QUARTER
    execution_lag_days: int = EXECUTION_LAG_DAYS
    train_lookback_years: int = TRAIN_LOOKBACK_YEARS
    snapshot_step_months: int = SNAPSHOT_STEP_MONTHS
    fundamental_step_months: int = FUNDAMENTAL_STEP_MONTHS
    target_horizon_months: int = TARGET_HORIZON_MONTHS
    meta_ic_lookback_quarters: int = META_IC_LOOKBACK_QUARTERS
    min_rank_ic_cross_section: int = MIN_RANK_IC_CROSS_SECTION
    objective: str = OBJECTIVE
    lgbm_n_estimators: int = LGBM_N_ESTIMATORS
    lgbm_max_depth: int = LGBM_MAX_DEPTH
    lgbm_learning_rate: float = LGBM_LEARNING_RATE
    lgbm_min_child_samples: int = LGBM_MIN_CHILD_SAMPLES
    # Hilos de LightGBM por fit. -1 = todos los núcleos. No afecta a los resultados (LightGBM es
    # determinista con hilos), solo a la velocidad. El full study reduce esto a
    # MODEL_THREADS_PER_WORKER dentro de cada worker paralelo para no sobre-suscribir la máquina.
    lgbm_n_jobs: int = -1
    random_seed: int = RANDOM_SEED
    meta_type: str = META_TYPE
    recency_weighting: str = RECENCY_WEIGHTING
    enabled_feature_blocks: tuple[str, ...] = ENABLED_FEATURE_BLOCKS
    enabled_agents: tuple[str, ...] = ENABLED_AGENTS
    enabled_model_families: tuple[str, ...] = ENABLED_MODEL_FAMILIES
    intra_agent_ensemble_mode: str = INTRA_AGENT_ENSEMBLE_MODE
    feature_weighting_mode: str = FEATURE_WEIGHTING_MODE
    feature_selection_min_coverage: float = FEATURE_SELECTION_MIN_COVERAGE
    feature_selection_lookback_quarters: int = FEATURE_SELECTION_LOOKBACK_QUARTERS
    feature_selection_min_permutation_importance: float = FEATURE_SELECTION_MIN_PERMUTATION_IMPORTANCE
    feature_selection_min_positive_fraction: float = FEATURE_SELECTION_MIN_POSITIVE_FRACTION
    feature_selection_max_features_per_agent: int = FEATURE_SELECTION_MAX_FEATURES_PER_AGENT
    metric_winsorization_percentile: float = METRIC_WINSORIZATION_PERCENTILE
    risk_feature_windows: tuple[int, ...] = RISK_FEATURE_WINDOWS
    technical_feature_windows: tuple[int, ...] = TECHNICAL_FEATURE_WINDOWS
    # artefactos activables
    neutralize_by_sector: bool = NEUTRALIZE_BY_SECTOR
    fundamental_momentum: bool = FUNDAMENTAL_MOMENTUM
    market_regime_feature: bool = MARKET_REGIME_FEATURE
    price_momentum_multi: bool = PRICE_MOMENTUM_MULTI
    moving_averages: bool = MOVING_AVERAGES
    regime_extended: bool = REGIME_EXTENDED
    quality_growth_derived: bool = QUALITY_GROWTH_DERIVED
    # cartera
    target_size: int = TARGET_SIZE
    min_hold_percentile: float = MIN_HOLD_PERCENTILE
    rotation_edge_percentiles: float = ROTATION_EDGE_PERCENTILES
    commission_bps: float = COMMISSION_BPS
    slippage_bps: float = SLIPPAGE_BPS
    rebalance_drift_tolerance: float = REBALANCE_DRIFT_TOLERANCE
    price_only_strictness_multiplier: float = PRICE_ONLY_STRICTNESS_MULTIPLIER
    max_monthly_position_return: float = MAX_MONTHLY_POSITION_RETURN
    profile: str = "balanced"   # perfil de inversor para la seleccion de cartera (ver module/profiles.py)
    # Versión manual del código de cada etapa cacheada: entra en la clave de reciclado
    # (module/runs/recycle.py STAGE_FIELDS) en vez de un hash automático del código. Súbela a mano
    # tras un cambio real de lógica en esa etapa para invalidar su caché; un cambio de logging o
    # de comentario no requiere tocarla.
    dataset_code_version: int = 1
    features_code_version: int = 1
    agents_code_version: int = 1
    backtest_code_version: int = 1
    # Ruta privada de ejecución. No es una variable científica ni se incluye en fingerprints;
    # el orquestador la asigna después de crear el run.
    workspace_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.run_mode not in RUN_MODES:
            raise ValueError(
                f"RUN_MODE inválido: {self.run_mode!r}. Valores admitidos: {', '.join(RUN_MODES)}."
            )
        if self.run_scope not in RUN_SCOPES:
            raise ValueError(
                f"RUN_SCOPE inválido: {self.run_scope!r}. Valores admitidos: {', '.join(RUN_SCOPES)}."
            )

        if self.execution_quarter not in (1, 2, 3, 4):
            raise ValueError("EXECUTION_QUARTER debe estar entre 1 y 4.")
        if self.objective not in ("rank_regression", "ranking", "quartile"):
            raise ValueError(
                f"OBJECTIVE invalido: {self.objective!r}. Usa 'rank_regression', 'ranking' o 'quartile'."
            )
        if self.meta_type not in ("equal", "rank_ic", "regime", "stacked_oos"):
            raise ValueError(
                f"META_TYPE invalido: {self.meta_type!r}. Usa 'equal', 'rank_ic' o 'regime'."
            )
        if self.recency_weighting not in ("off", "linear", "exponential"):
            raise ValueError(
                f"RECENCY_WEIGHTING invalido: {self.recency_weighting!r}. "
                "Usa 'off', 'linear' o 'exponential'."
            )
        if self.feature_weighting_mode not in ("model_native", "diagnostic_only", "oos_stability_prune",
                                               "regularized_linear_ensemble", "block_gated"):
            raise ValueError("FEATURE_WEIGHTING_MODE no reconocido.")
        if self.intra_agent_ensemble_mode not in ("single", "equal_rank", "rank_ic_weighted"):
            raise ValueError("INTRA_AGENT_ENSEMBLE_MODE no reconocido.")
        if not self.enabled_agents:
            raise ValueError("ENABLED_AGENTS no puede estar vacío.")
        if not 0 <= self.feature_selection_min_coverage <= 1:
            raise ValueError("FEATURE_SELECTION_MIN_COVERAGE debe estar en [0, 1].")
        if not 0 <= self.metric_winsorization_percentile < 0.5:
            raise ValueError("METRIC_WINSORIZATION_PERCENTILE debe estar en [0, 0.5).")
        if self.train_lookback_years <= 0 or self.target_horizon_months <= 0:
            raise ValueError("Las ventanas temporales deben ser positivas.")
        if self.target_size < 1:
            raise ValueError("TARGET_SIZE debe ser positivo.")
        for name, value in (("min_hold_percentile", self.min_hold_percentile),):
            if not 0 <= value <= 100:
                raise ValueError(f"{name} debe estar en [0, 100], recibido {value!r}.")
        if self.rebalance_drift_tolerance < 0:
            raise ValueError(
                "REBALANCE_DRIFT_TOLERANCE es una fracción relativa (0.25 = 25 %) y debe ser >= 0."
            )
        if self.price_only_strictness_multiplier < 1:
            raise ValueError(
                "PRICE_ONLY_STRICTNESS_MULTIPLIER debe ser >= 1: endurece las revisiones sin "
                "fundamentales nuevos, nunca las relaja."
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
        return DEV_PROCESSED_DIR if self.dev_mode else PROCESSED_DIR

    @property
    def tickers(self) -> list[str]:
        """Universo histórico completo o muestra de desarrollo más el benchmark."""
        from module.data.universe import historical_universe

        base = DEV_TICKERS if self.dev_mode else sorted(historical_universe())
        return list(dict.fromkeys([*base, self.benchmark_ticker]))


def ensure_directories(settings: Settings) -> None:
    for path in (RAW_DIR, RAW_JSON_DIR, PROCESSED_DIR, settings.raw_output_dir, settings.processed_output_dir):
        path.mkdir(parents=True, exist_ok=True)
