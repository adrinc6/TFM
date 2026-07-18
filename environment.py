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
DATA_START_DATE = "1990-01-01"
DATA_END_DATE = "2026-07-15"
DEV_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
BENCHMARK_TICKER = "SPY"
FORCE_RAW_DOWNLOAD = False

# Fase 1: el panel se construye para toda la historia disponible. La fecha ancla
# se usará después para iniciar una simulación o un escenario concreto.
# Fecha ancla: el sistema evalua desde 2016 (mayor cobertura, menos sesgo de supervivencia),
# entrenando 8-12 anios hacia atras (datos de sobra desde ~2004). Se olvida la ventana desde 2000.
EXECUTION_YEAR = 2016
EXECUTION_QUARTER = 1
EXECUTION_LAG_DAYS = 45
TRAIN_LOOKBACK_YEARS = 10
SNAPSHOT_STEP_MONTHS = 1           # cadencia de revision de cartera; barrida como escenario
FUNDAMENTAL_STEP_MONTHS = 3        # cadencia de reentreno; barrida como escenario (3/12/1)
SNAPSHOT_DAY = 15
TARGET_HORIZON_MONTHS = 3
MAX_PRICE_AGE_DAYS = 7
META_IC_LOOKBACK_QUARTERS = 12
MIN_TRAINING_ROWS = 30
MIN_RANK_IC_CROSS_SECTION = 10

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
# Por defecto 8-12 posiciones, peso maximo 15 % (minimo implicito ~5 % por el reparto). Barrida.
TARGET_MIN = 8
TARGET_MAX = 12
ENTRY_MIN_PERCENTILE = 80         # solo entran candidatos por encima de este percentil (0..100)
MIN_HOLD_PERCENTILE = 50          # tenente cae por debajo -> sale, aunque nadie le supere
ROTATION_EDGE_PERCENTILES = 5     # umbral de ventaja para que un candidato desplace a un tenente
MAX_WEIGHT_PER_POSITION = 0.15    # tope de peso por posicion; el excedente se reparte
COMMISSION_BPS = 5                # comision por operacion, en puntos basicos
SLIPPAGE_BPS = 10                 # slippage por operacion, en puntos basicos
REBALANCE_DRIFT_TOLERANCE = 1.5   # solo re-sizing si un peso excede MAX_WEIGHT * este factor
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
    snapshot_day: int = SNAPSHOT_DAY
    target_horizon_months: int = TARGET_HORIZON_MONTHS
    max_price_age_days: int = MAX_PRICE_AGE_DAYS
    meta_ic_lookback_quarters: int = META_IC_LOOKBACK_QUARTERS
    min_training_rows: int = MIN_TRAINING_ROWS
    min_rank_ic_cross_section: int = MIN_RANK_IC_CROSS_SECTION
    objective: str = OBJECTIVE
    lgbm_n_estimators: int = LGBM_N_ESTIMATORS
    lgbm_max_depth: int = LGBM_MAX_DEPTH
    lgbm_learning_rate: float = LGBM_LEARNING_RATE
    lgbm_min_child_samples: int = LGBM_MIN_CHILD_SAMPLES
    random_seed: int = RANDOM_SEED
    meta_type: str = META_TYPE
    # artefactos activables
    neutralize_by_sector: bool = NEUTRALIZE_BY_SECTOR
    neutralize_min_group: int = NEUTRALIZE_MIN_GROUP
    fundamental_momentum: bool = FUNDAMENTAL_MOMENTUM
    market_regime_feature: bool = MARKET_REGIME_FEATURE
    price_momentum_multi: bool = PRICE_MOMENTUM_MULTI
    moving_averages: bool = MOVING_AVERAGES
    regime_extended: bool = REGIME_EXTENDED
    quality_growth_derived: bool = QUALITY_GROWTH_DERIVED
    # cartera
    target_min: int = TARGET_MIN
    target_max: int = TARGET_MAX
    entry_min_percentile: float = ENTRY_MIN_PERCENTILE
    min_hold_percentile: float = MIN_HOLD_PERCENTILE
    rotation_edge_percentiles: float = ROTATION_EDGE_PERCENTILES
    max_weight_per_position: float = MAX_WEIGHT_PER_POSITION
    commission_bps: float = COMMISSION_BPS
    slippage_bps: float = SLIPPAGE_BPS
    rebalance_drift_tolerance: float = REBALANCE_DRIFT_TOLERANCE
    max_monthly_position_return: float = MAX_MONTHLY_POSITION_RETURN
    profile: str = "balanced"   # perfil de inversor para la seleccion de cartera (ver module/profiles.py)

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
        if self.meta_type not in ("equal", "rank_ic", "regime"):
            raise ValueError(
                f"META_TYPE invalido: {self.meta_type!r}. Usa 'equal', 'rank_ic' o 'regime'."
            )
        if self.train_lookback_years <= 0 or self.target_horizon_months <= 0:
            raise ValueError("Las ventanas temporales deben ser positivas.")
        if not 1 <= self.target_min <= self.target_max:
            raise ValueError("TARGET_MIN y TARGET_MAX deben cumplir 1 <= min <= max.")
        for name, value in (
            ("entry_min_percentile", self.entry_min_percentile),
            ("min_hold_percentile", self.min_hold_percentile),
        ):
            if not 0 <= value <= 100:
                raise ValueError(f"{name} debe estar en [0, 100], recibido {value!r}.")
        if not 0 < self.max_weight_per_position <= 1:
            raise ValueError("MAX_WEIGHT_PER_POSITION debe estar en (0, 1].")
        if self.max_weight_per_position * self.target_min < 0.999:
            raise ValueError(
                "MAX_WEIGHT_PER_POSITION * TARGET_MIN < 1: la cartera minima no puede llegar al 100 %."
            )
        if self.max_price_age_days < 0:
            raise ValueError("MAX_PRICE_AGE_DAYS no puede ser negativo.")

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
