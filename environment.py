# =============================================================================
# Parámetros globales del proyecto
# =============================================================================

# Modo de ejecucion: si True, salta el backtest y solo ejecuta el fold live Q1 2026
SKIP_BACKTEST = False

## Proceso
FORCE_DOWNLOAD = False
FORCE_TRAINING = False
NO_PLOT = True

## Rutas
DATA_DIR     = "data"
HISTORY_FILE = f"{DATA_DIR}/ticker_history.json"
RESULTS_DIR  = "results"

## Período de datos
START_DATE  = "2021-01-01"
END_DATE    = "2026-01-01"
DAYS_UPDATE = 90

# =============================================================================
# Parámetros ML Pipeline (añadidos para el framework multi-agente)
# =============================================================================
import os

# Sub-rutas de datos
PRICES_DIR       = os.path.join(DATA_DIR, "prices")
CONSOLIDATED_DIR = os.path.join(DATA_DIR, "consolidated")
INSIDER_DIR      = os.path.join(DATA_DIR, "insider")
ANALYST_DIR      = os.path.join(DATA_DIR, "analyst")
MACRO_DIR        = os.path.join(DATA_DIR, "macro")
COMPANIES_FILE   = os.path.join(DATA_DIR, "companies.csv")  # Company Name,Ticker,Market Cap,Sector,Industry

# Sub-rutas de resultados
AGENTS_RESULTS_DIR   = os.path.join(RESULTS_DIR, "agents")
BACKTEST_RESULTS_DIR = os.path.join(RESULTS_DIR, "backtest")
PLOTS_DIR            = os.path.join(RESULTS_DIR, "plots")

# Anti-leakage: días desde fin de trimestre hasta que el informe es público
FILING_LAG_DAYS = 45

# Horizonte de prediccion del label en dias de trading (~1 quarter = 63 dias)
# Alineado con WALKFORWARD_TEST_QUARTERS para que el label mida exactamente
# el retorno en el periodo de test
FORWARD_RETURN_DAYS = 63

# Minimo de trimestres historicos por ticker para incluirlo en train
MIN_HISTORY_QUARTERS = 4

# Normalizacion sectorial: minimo de peers para calcular Z-score
SECTOR_ZSCORE_MIN_PEERS = 3

# Walk-forward backtesting
WALKFORWARD_TRAIN_YEARS   = 1   # Minimo de anos de train (siempre anos enteros)
WALKFORWARD_TEST_QUARTERS = 1   # Periodo de test: 1 quarter (~63 dias de trading)

# Tasa libre de riesgo anualizada (Sharpe ratio)
RISK_FREE_RATE = 0.04

# Reproducibilidad
RANDOM_SEED = 42
