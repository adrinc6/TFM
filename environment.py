# =============================================================================
# environment.py — Parámetros globales del proyecto
# =============================================================================
"""
Fuente única de verdad para toda la configuración del pipeline.

Organización:
  1. Flags de ejecución
  2. Rutas de datos y resultados
  3. Fechas del período de análisis
  4. Parámetros del pipeline ML
  5. Walk-forward backtesting
  6. Hiperparámetros de los agentes
  7. Reproducibilidad

Nota: el backend matplotlib Agg está configurado en los módulos de visualización,
por lo que todos los gráficos se guardan siempre en disco (modo headless).
"""
import os

# =============================================================================
# 1. Flags de ejecución
# =============================================================================

# Si True, salta el backtest walk-forward y solo ejecuta el fold live
SKIP_BACKTEST = False

# Si True, re-descarga todos los datos aunque ya existan en disco
FORCE_DOWNLOAD = False

# Si True, reintenta descargar datos para los tickers eliminados por falta de datos
RETRY_MISSING_TICKERS = False

# =============================================================================
# 2. API Keys
# =============================================================================

FINNHUB_API_KEY = "d6ttu99r01qhkb45jm5gd6ttu99r01qhkb45jm60"

# =============================================================================
# 3. Rutas de datos y resultados
# =============================================================================

# Directorio raíz de datos Finnhub (nueva fuente de datos)
FINNHUB_DATA_DIR = "data_finnhub"

# Directorio raíz de datos legacy (yfinance) — mantenido por compatibilidad
DATA_DIR     = "data"
RESULTS_DIR  = "results"
HISTORY_FILE = os.path.join(DATA_DIR, "ticker_history.json")

# Sub-rutas de datos Finnhub
FINNHUB_MACRO_DIR       = os.path.join(FINNHUB_DATA_DIR, "_macro")
FINNHUB_CONSOLIDATED_DIR = os.path.join(FINNHUB_DATA_DIR, "consolidated")

# Sub-rutas legacy (compatibilidad con módulos no migrados aún)
PRICES_DIR       = os.path.join(DATA_DIR, "prices")
CONSOLIDATED_DIR = os.path.join(DATA_DIR, "consolidated")
INSIDER_DIR      = os.path.join(DATA_DIR, "insider")
ANALYST_DIR      = os.path.join(DATA_DIR, "analyst")
MACRO_DIR        = os.path.join(DATA_DIR, "macro")
COMPANIES_FILE   = os.path.join(DATA_DIR, "companies.csv")

# Sub-rutas de resultados
AGENTS_RESULTS_DIR   = os.path.join(RESULTS_DIR, "agents")
BACKTEST_RESULTS_DIR = os.path.join(RESULTS_DIR, "backtest")
PLOTS_DIR            = os.path.join(RESULTS_DIR, "plots")

# =============================================================================
# 4. Período de análisis
# =============================================================================

START_DATE  = "2015-01-01"   # Inicio del histórico — mínimo 5 años de entrenamiento
END_DATE    = "2026-01-01"   # Fin del histórico / inicio del fold live

# Días tras los que se considera que un ticker necesita actualización de datos
DAYS_UPDATE = 90

# =============================================================================
# 5. Parámetros del pipeline ML
# =============================================================================

# Horizonte del label en días de trading (~1 quarter ≈ 63 días bursátiles).
FORWARD_RETURN_DAYS = 63

# Mínimo de trimestres históricos por ticker para incluirlo en train
MIN_HISTORY_QUARTERS = 4

# Mínimo de empresas del mismo sector para calcular Z-score sectorial
SECTOR_ZSCORE_MIN_PEERS = 3

# Número de folds KFold internos para generar OOF scores del meta-learner
OOF_N_SPLITS = 3

# =============================================================================
# 6. Walk-forward backtesting
# =============================================================================

# Mínimo de años de datos de entrenamiento (con historia desde 2015, usamos 5Y)
WALKFORWARD_TRAIN_YEARS   = 5

# Trimestres de test por fold (siempre 1)
WALKFORWARD_TEST_QUARTERS = 1

# Tasa libre de riesgo anualizada para Sharpe / Sortino
RISK_FREE_RATE = 0.04

# Número de stocks seleccionados en la cartera long por fold
TOP_N_STOCKS = 10

# =============================================================================
# 7. Hiperparámetros de los agentes
# =============================================================================

# ── FundamentalAgent (XGBoost) ────────────────────────────────────────────────
FUNDAMENTAL_N_ESTIMATORS    = 400
FUNDAMENTAL_MAX_DEPTH       = 5
FUNDAMENTAL_LEARNING_RATE   = 0.05
FUNDAMENTAL_SUBSAMPLE       = 0.8
FUNDAMENTAL_COLSAMPLE       = 0.7
FUNDAMENTAL_MIN_CHILD_WEIGHT = 5
FUNDAMENTAL_CV_FOLDS        = 5

# ── ValuationAgent (GBM) ─────────────────────────────────────────────────────
VALUATION_N_ESTIMATORS  = 200
VALUATION_MAX_DEPTH     = 4
VALUATION_LEARNING_RATE = 0.05
VALUATION_SUBSAMPLE     = 0.8
VALUATION_CV_FOLDS      = 5

# ── MomentumAgent (Random Forest) ────────────────────────────────────────────
MOMENTUM_N_ESTIMATORS    = 300
MOMENTUM_MAX_DEPTH       = 8
MOMENTUM_MIN_SAMPLES_LEAF = 10
MOMENTUM_CV_FOLDS        = 5

# ── BearAgent (Random Forest híbrido) ────────────────────────────────────────
BEAR_N_ESTIMATORS = 200
BEAR_MAX_DEPTH    = 6
BEAR_CV_FOLDS     = 5
# Peso de la capa de reglas vs ML en el score final
BEAR_RULE_WEIGHT  = 0.5
BEAR_ML_WEIGHT    = 0.5
# Score de riesgo por encima del cual el meta-learner fuerza Underperform
BEAR_HARD_THRESHOLD = 0.90

# ── SentimentAgent (Random Forest) ───────────────────────────────────────────
SENTIMENT_N_ESTIMATORS    = 200
SENTIMENT_MAX_DEPTH       = 6
SENTIMENT_MIN_SAMPLES_LEAF = 8
SENTIMENT_CV_FOLDS        = 5

# ── FeatureSelector (compartido por todos los agentes) ───────────────────────
FEATURE_CORR_THRESHOLD = 0.85
FEATURE_TOP_N = 10

# ── MetaLearner (LR + GBM stacking) ──────────────────────────────────────────
META_LR_C             = 0.5
META_GBM_N_ESTIMATORS = 150
META_GBM_MAX_DEPTH    = 3
META_GBM_LEARNING_RATE = 0.05
META_GBM_SUBSAMPLE    = 0.8
META_CV_FOLDS         = 5

# =============================================================================
# 8. Reproducibilidad
# =============================================================================

RANDOM_SEED = 42
