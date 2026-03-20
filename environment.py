# =============================================================================
# environment.py — Parámetros globales del proyecto
# =============================================================================
"""
Fuente única de verdad para toda la configuración del pipeline.

Organización:
  1. Flags de ejecución
  2. API keys
  3. Rutas de datos y resultados
  4. Universo de tickers
  5. Período de análisis
  6. Parámetros del pipeline ML
  7. Walk-forward backtesting
  8. Hiperparámetros de los agentes
  9. Reproducibilidad

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

# Si True, ejecuta el estudio de ablation para medir la contribución de agentes
RUN_ABLATION_STUDY = False

# Descarga paralela
DOWNLOAD_MAX_WORKERS = 8

# Finnhub rate limit (segundos entre requests globales)
FINNHUB_MIN_INTERVAL = 1

# =============================================================================
# 2. API Keys
# =============================================================================

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "d6ttu99r01qhkb45jm5gd6ttu99r01qhkb45jm60")

# =============================================================================
# 3. Rutas de datos y resultados
# =============================================================================

FINNHUB_DATA_DIR         = "data_finnhub"

RESULTS_DIR          = "results"
AGENTS_RESULTS_DIR   = os.path.join(RESULTS_DIR, "agents")
BACKTEST_RESULTS_DIR = os.path.join(RESULTS_DIR, "backtest")
PLOTS_DIR            = os.path.join(RESULTS_DIR, "plots")

# =============================================================================
# 4. Universo de tickers
# =============================================================================
# Lista de tickers a analizar. Edita aquí para cambiar el universo.
TICKERS = [
    "NVDA","AAPL","MSFT","AMZN","GOOGL","GOOG","META","AVGO","TSLA","BRK-B","WMT","LLY","JPM","XOM","V","JNJ","MU","COST","MA","ORCL","NFLX","CVX","ABBV","PLTR","PG","BAC","HD","KO","AMD","CAT","GE","CSCO","MRK","LRCX","AMAT","RTX","PM","UNH","MS","GS","IBM","WFC","GEV","TMUS","LIN","MCD","INTC","PEP","VZ","AXP","KLAC","T","NEE","C","AMGN","ABT","CRM","DIS","GILD","TXN","TMO","ANET","TJX","ISRG","SCHW","BA","UBER","APH","DE","PFE","COP","BLK","ADI","LMT","APP","HON","WELL","UNP","QCOM","BKNG","ETN","PANW","DHR","SYK","LOW","CB","SPGI","INTU","PLD","ACN","BMY","NOW","PGR","PH","VRTX","CEG","MCK","MDT","COF","HCA","CME","CRWD","GLW","MO","NEM","SO","SBUX","BSX","SNDK","CMCSA","NOC","DUK","WDC","ADBE","DELL","HWM","EQIX","GD","WM","TT","CVS","STX","WMB","ICE","BX","MAR","PWR","ADP","AMT","MRSH","JCI","UPS","FDX","SNPS","PNC","USB","KKR","CDNS","REGN","BK","NKE","ABNB","MCO","SHW","MSI","FCX","MMM","ITW","CTAS","CMI","ECL","EOG","ORLY","CSX","MNST","RCL","EMR","KMI","MDLZ","VLO","DASH","AEP","CL","CI","MPC","PSX","TDG","RSG","LHX","SLB","HLT","AON","WBD","ROST","HOOD","CRH","GM","ELV","TRV","APO","NSC","COR","APD","FTNT","SRE","SPG","DLR","PCAR","O","OXY","TEL","BKR","VST","AFL","AZO","TFC","D","OKE","CIEN","FANG","AJG","CTVA","COIN","ALL","MPWR","ADSK","TGT","FAST","EXC","TRGP","EA","CAH","XEL","FIX","ZTS","GWW","PSA","AME","KEYS","NXPI","NDAQ","CARR","EW","ETR","F","DDOG","TER","URI","IDXX","BDX","KR","MET","GRMN","YUM","HSY","PEG","CMG","CVNA","DAL","EBAY","ED","AXON","PYPL","MSCI","VTR","WAB","EQT","PCG","AMP","DHI","ROK","AIG","CBRE","FITB","SYY","ODFL","TTWO","WEC","LYV","CCI","TPL","NUE","KDP","HIG","ROP","LVS","MCHP","WDAY","XYZ","MLM","ADM","VMC","NRG","STT","CCL","KVUE","RMD","KMB","EME","ACGL","PAYX","PRU","IR","GEHC","CPRT","A","IRM","EL","ATO","OTIS","AEE","HAL","HBAN","FISV","IBKR","CBOE","DTE","DVN","UAL","VICI","TDY","WAT","FE","MTB","XYL","EXPE","CTSH","EXR","PPL","DOV","HPE","FICO","CNP","TPR","RJF","EIX","VRSK","DG","ES","IQV","WTW","JBL","DOW","AWK","BIIB","CHTR","STZ","KHC","DXCM","ROL","CTRA","EXE","FIS","HUBB","WRB","NTRS","CINF","LYB","STLD","TSCO","CFG","ARES","MTD","BG","Q","LEN","CMS","ON","OMC","AVB","DRI","ULTA","PPG","BRO","CHD","SYF","EQR","PHM","NI","VLTO","EFX","WSM","VRSN","LH","RF","L","DGX","TSN","DLTR","STE","FSLR","LDOS","RL","KEY","MRNA","BR","HUM","CHRW","CF","GIS","SW","NTAP","GPN","LUV","CPAY","LULU","EXPD","TROW","ALB","EVRG","IP","SBAC","PFG","SNA","PKG","INCY","LNT","JBHT","AMCR","SMCI","CSGP","DD","NVR","IFF","PTC","CNC","ZBH","WST","WY","FTV","HOLX","HPQ","LII","HII","PODD","BALL","FFIV","ESS","TXT","VTRS","AKAM","TKO","TRMB","KIM","J","INVH","CDW","MAA","APTV","NDSN","MKC","TYL","DECK","PNR","IEX","GPC","REG","COO","BBY","CLX","HST","APA","ALGN","HAS","EG","DPZ","AVY","ERIE","HRL","GEN","BEN","ALLE","MAS","DOC","PNW","JKHY","GNRC","SOLV","FOX","UHS","UDR","FOXA","IT","TTD","GDDY","SWK","SJM","GL","WYNN","AIZ","BF-B","IVZ","CPT","ZBRA","PSKY","AES","DVA","BLDR","RVTY","MGM","FRT","MOS","NCLH","AOS","NWSA","BAX","HSIC","ARE","BXP","SWKS","TECH","TAP","CRL","FDS","MOH","POOL","CAG","EPAM","MTCH","PAYC","CPB","LW","NWS"
    ][:20]

# =============================================================================
# 5. Período de análisis
# =============================================================================

# Fecha desde la que se descargan los datos brutos.
DOWNLOAD_START_DATE = "2015-01-01"

# Inicio del período de testeo/backtest walk-forward, expresado como año y trimestre.
TEST_START_YEAR = 2025
TEST_START_QUARTER = 3

# Fin del histórico / referencia para el fold live, expresado como año y trimestre.
END_YEAR = 2026
END_QUARTER = 1

# Si True, ejecuta el fold live al final del pipeline.
RUN_LIVE_FOLD = True

# =============================================================================
# 6. Parámetros del pipeline ML
# =============================================================================

# Mínimo de trimestres históricos por ticker para incluirlo en train
MIN_HISTORY_QUARTERS = 4

# Mínimo de empresas del mismo sector para calcular Z-score sectorial
SECTOR_ZSCORE_MIN_PEERS = 3

# Número de folds KFold internos para generar OOF scores del meta-learner
OOF_N_SPLITS = 3

# Umbral mínimo de score para incluir un stock en la cartera long / shortlist
PORTFOLIO_MIN_SCORE = 0.5

# =============================================================================
# 7. Walk-forward backtesting
# =============================================================================

# Ventana de entrenamiento del walk-forward, en años.
WALKFORWARD_TRAIN_LOOKBACK_YEARS = 8

# Trimestres de test por fold (siempre 1)
WALKFORWARD_TEST_QUARTERS = 1

# Tasa libre de riesgo anualizada para Sharpe / Sortino
RISK_FREE_RATE = 0.04

# Máximo de stocks seleccionados en la cartera long por fold
TOP_N_STOCKS = 10

# Si True, pondera la cartera: el ticker #1 pesa (1 + N/10) veces más que el #N.
# Distribución lineal entre ambos extremos, normalizada a suma 1.
# Ejemplo: N=10 → el primero pesa el doble que el último.
#          N=5  → el primero pesa un 50% más que el último.
# Si False, todos los tickers tienen el mismo peso.
SCORE_WEIGHTED_PORTFOLIO = True

# =============================================================================
# 8. Hiperparámetros de los agentes
# =============================================================================

# ── FundamentalAgent (XGBoost) ────────────────────────────────────────────────
FUNDAMENTAL_N_ESTIMATORS    = 400
FUNDAMENTAL_MAX_DEPTH       = 5
FUNDAMENTAL_LEARNING_RATE   = 0.05
FUNDAMENTAL_SUBSAMPLE       = 0.8
FUNDAMENTAL_COLSAMPLE       = 0.7
FUNDAMENTAL_MIN_CHILD_WEIGHT = 5
# ── ValuationAgent (GBM) ─────────────────────────────────────────────────────
VALUATION_N_ESTIMATORS  = 200
VALUATION_MAX_DEPTH     = 4
VALUATION_LEARNING_RATE = 0.05
VALUATION_SUBSAMPLE     = 0.8

# ── MomentumAgent (Random Forest) ────────────────────────────────────────────
MOMENTUM_N_ESTIMATORS    = 300
MOMENTUM_MAX_DEPTH       = 8
MOMENTUM_MIN_SAMPLES_LEAF = 10

# ── BearAgent (Random Forest híbrido) ────────────────────────────────────────
BEAR_N_ESTIMATORS = 200
BEAR_MAX_DEPTH    = 6
# Peso de la capa de reglas vs ML en el score final
BEAR_RULE_WEIGHT  = 0.5
BEAR_ML_WEIGHT    = 0.5
# Score de riesgo por encima del cual el meta-learner fuerza Underperform
BEAR_HARD_THRESHOLD = 0.90

# ── SentimentAgent (Random Forest) ───────────────────────────────────────────
SENTIMENT_N_ESTIMATORS    = 200
SENTIMENT_MAX_DEPTH       = 6
SENTIMENT_MIN_SAMPLES_LEAF = 8
# ── FeatureSelector (compartido por todos los agentes) ───────────────────────
FEATURE_CORR_THRESHOLD = 0.85
# Máximo de features conservadas por agente; si quedan menos, no pasa nada
FEATURE_TOP_N = 10

# ── MetaLearner (LR + GBM stacking) ──────────────────────────────────────────
META_LR_C             = 0.5
META_GBM_N_ESTIMATORS = 150
META_GBM_MAX_DEPTH    = 3
META_GBM_LEARNING_RATE = 0.05
META_GBM_SUBSAMPLE    = 0.8

# =============================================================================
# 9. Reproducibilidad
# =============================================================================

RANDOM_SEED = 42
