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
EXECUTION_YEAR = 2000
EXECUTION_QUARTER = 1
EXECUTION_LAG_DAYS = 45
TRAIN_LOOKBACK_YEARS = 8
SNAPSHOT_STEP_MONTHS = 1
FUNDAMENTAL_STEP_MONTHS = 3
SNAPSHOT_DAY = 15
TARGET_HORIZON_MONTHS = 3
MAX_PRICE_AGE_DAYS = 7
# B1: neutralizar factores rankeando dentro de sector en vez de sobre todo el universo.
# El sector viene de profiles.parquet (snapshot actual): se usa SOLO para agrupar, nunca
# como senal. Con menos de NEUTRALIZE_MIN_GROUP miembros, el grupo cae a ranking global.
NEUTRALIZE_BY_SECTOR = False
NEUTRALIZE_MIN_GROUP = 5
META_IC_LOOKBACK_QUARTERS = 12
RIDGE_ALPHA = 1.0
MIN_TRAINING_ROWS = 30
MIN_RANK_IC_CROSS_SECTION = 10

# Motor de aprendizaje de los agentes: "ridge" (lineal) o "lightgbm" (arboles, no lineal).
# LightGBM captura interacciones que el lineal promedia a cero. Ver docs/informe_situacion_y_critica.md.
MODEL_TYPE = "lightgbm"
# Objetivo de aprendizaje:
#   "rank_regression" -> regresion sobre el PERCENTIL transversal del retorno (0..1) dentro de
#                        cada snapshot. Objetivo principal: alinea el entrenamiento con el rank-IC.
#   "regression"      -> exceso de retorno crudo (tratado por LABEL_TRANSFORM).
#   "ranking"         -> LGBMRanker agrupado por snapshot (lambdarank). Solo LightGBM.
#   "quartile"        -> clasifica cuartil superior vs inferior; ablacion secundaria.
OBJECTIVE = "rank_regression"
# Hiperparametros LightGBM. Defaults conservadores anti-overfitting (arboles poco profundos,
# muchas muestras minimas por hoja) porque hay pocas eras independientes.
LGBM_N_ESTIMATORS = 200
LGBM_MAX_DEPTH = 4
LGBM_LEARNING_RATE = 0.05
LGBM_MIN_CHILD_SAMPLES = 50
# Semilla del modelo. Fija en la comparacion principal; se barre para medir sensibilidad del
# candidato ganador (una senal que solo aparece con una semilla es sospechosa).
RANDOM_SEED = 42
# Meta-agente: como se combinan los 3 agentes.
#   "equal"   -> promedio equiponderado de los rangos de los agentes
#   "rank_ic" -> ponderacion por rank-IC reciente de cada agente
#   "regime"  -> pesos distintos segun regimen bull/bear
#   "stacker" -> stacker lineal (Ridge) walk-forward sobre scores + contexto de mercado
# Se empieza por rank_ic; el stacker solo se prueba tras superar la Puerta 1 (ver plan).
META_TYPE = "rank_ic"
# B2: tratamiento de la etiqueta de entrenamiento para reducir ruido.
#   "none"   -> exceso de retorno crudo
#   "winsor" -> recorta las colas al percentil LABEL_WINSOR_PCT (reduce outliers)
#   "rank"   -> percentil transversal del retorno futuro (entrena contra el orden, Spearman)
# El tratamiento se aplica SOLO a la etiqueta de entrenamiento, nunca al scoring.
# "rank" es el mejor medido (rank-IC -0.0058 -> +0.0011, frac 0.486 -> 0.534): entrenar contra
# el orden alinea la perdida con el rank-IC. Ver docs/bitacora.md (B2).
LABEL_TRANSFORM = "rank"
LABEL_WINSOR_PCT = 0.02   # recorta el 2 % de cada cola (solo si LABEL_TRANSFORM="winsor")
# B3: anadir features de tendencia de fundamentales (cambio del ratio respecto a la publicacion
# anterior de la misma empresa) y descomposicion del cambio de valoracion en su parte de precio
# y su parte fundamental. Todo point-in-time. Ver docs/bitacora.md (B3).
FUNDAMENTAL_MOMENTUM = False
# B5: anadir el regimen de mercado (bull/bear, detectado SOLO con datos pasados del SP500) como
# feature, junto con interacciones factor x regimen, para que el modelo aprenda a ponderar
# distinto en cada regimen. Mejora marginal (rank-IC +0.0011 -> +0.0015) y ayuda a los agentes
# fundamentales (quality cruza a positivo); se conserva activo. Ver docs/bitacora.md (B5).
MARKET_REGIME_FEATURE = True

# Parametros de cartera (Fase 4). Todos con valores por defecto conservadores.
# La logica que sigue: expulsa a los que se hunden, protege del ruido con umbral de ventaja,
# no fija tenencia minima. Ver `docs/plan_fases.md` (Fase 4) para el razonamiento completo.
TARGET_MIN = 5                    # cartera nunca baja de 5 mientras haya candidatos que cumplan
TARGET_MAX = 10                   # cartera nunca supera 10
ENTRY_MIN_PERCENTILE = 80         # solo entran candidatos por encima de este percentil (0..100)
MIN_HOLD_PERCENTILE = 50          # tenente cae por debajo -> sale, aunque nadie le supere
ROTATION_EDGE_PERCENTILES = 5     # umbral de ventaja para que un candidato desplace a un tenente
MAX_WEIGHT_PER_POSITION = 0.20    # tope de peso por posicion; el excedente se reparte
COMMISSION_BPS = 5                # comision por operacion, en puntos basicos
SLIPPAGE_BPS = 10                 # slippage por operacion, en puntos basicos
REBALANCE_DRIFT_TOLERANCE = 1.5   # solo re-sizing si un peso excede MAX_WEIGHT * este factor

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
    neutralize_by_sector: bool = NEUTRALIZE_BY_SECTOR
    neutralize_min_group: int = NEUTRALIZE_MIN_GROUP
    meta_ic_lookback_quarters: int = META_IC_LOOKBACK_QUARTERS
    ridge_alpha: float = RIDGE_ALPHA
    min_training_rows: int = MIN_TRAINING_ROWS
    min_rank_ic_cross_section: int = MIN_RANK_IC_CROSS_SECTION
    model_type: str = MODEL_TYPE
    objective: str = OBJECTIVE
    lgbm_n_estimators: int = LGBM_N_ESTIMATORS
    lgbm_max_depth: int = LGBM_MAX_DEPTH
    lgbm_learning_rate: float = LGBM_LEARNING_RATE
    lgbm_min_child_samples: int = LGBM_MIN_CHILD_SAMPLES
    random_seed: int = RANDOM_SEED
    meta_type: str = META_TYPE
    label_transform: str = LABEL_TRANSFORM
    label_winsor_pct: float = LABEL_WINSOR_PCT
    fundamental_momentum: bool = FUNDAMENTAL_MOMENTUM
    market_regime_feature: bool = MARKET_REGIME_FEATURE
    target_min: int = TARGET_MIN
    target_max: int = TARGET_MAX
    entry_min_percentile: float = ENTRY_MIN_PERCENTILE
    min_hold_percentile: float = MIN_HOLD_PERCENTILE
    rotation_edge_percentiles: float = ROTATION_EDGE_PERCENTILES
    max_weight_per_position: float = MAX_WEIGHT_PER_POSITION
    commission_bps: float = COMMISSION_BPS
    slippage_bps: float = SLIPPAGE_BPS
    rebalance_drift_tolerance: float = REBALANCE_DRIFT_TOLERANCE

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
        if self.model_type not in ("ridge", "lightgbm"):
            raise ValueError(f"MODEL_TYPE invalido: {self.model_type!r}. Usa 'ridge' o 'lightgbm'.")
        if self.objective not in ("rank_regression", "regression", "ranking", "quartile"):
            raise ValueError(
                f"OBJECTIVE invalido: {self.objective!r}. Usa 'rank_regression', 'regression', "
                "'ranking' o 'quartile'."
            )
        if self.objective == "ranking" and self.model_type != "lightgbm":
            raise ValueError("OBJECTIVE='ranking' requiere MODEL_TYPE='lightgbm' (LGBMRanker).")
        if self.meta_type not in ("equal", "rank_ic", "regime", "stacker"):
            raise ValueError(
                f"META_TYPE invalido: {self.meta_type!r}. Usa 'equal', 'rank_ic', 'regime' o 'stacker'."
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
        from module.universe import historical_universe

        base = DEV_TICKERS if self.dev_mode else sorted(historical_universe())
        return list(dict.fromkeys([*base, self.benchmark_ticker]))


def ensure_directories(settings: Settings) -> None:
    for path in (RAW_DIR, RAW_JSON_DIR, PROCESSED_DIR, settings.raw_output_dir, settings.processed_output_dir):
        path.mkdir(parents=True, exist_ok=True)
