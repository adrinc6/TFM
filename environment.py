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

# Si True, solo actualiza precios y macro (sin consolidacion ni entrenamiento)
UPDATE_PRICES_ONLY = False

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
    ]

# =============================================================================
# 5. Período de análisis
# =============================================================================

# Fecha desde la que se descargan los datos brutos.
DOWNLOAD_START_DATE = "2015-01-01"

# Inicio del período de análisis/backtest walk-forward (quarters de snapshot a analizar).
ANALYSIS_START_YEAR = 2023
ANALYSIS_START_QUARTER = 3

# Fin del período de análisis/backtest walk-forward.
ANALYSIS_END_YEAR = 2026
ANALYSIS_END_QUARTER = 2

# Frecuencia del análisis walk-forward:
# - "quarterly": ejecuta un fold por quarter (comportamiento histórico).
# - "annual": ejecuta un fold por año (misma lógica de features, menor frecuencia).
ANALYSIS_FREQUENCY = "annual"

# Fecha ancla opcional para modo anual (formato "YYYY-MM-DD").
# Si es None en modo anual, se usa automáticamente:
#   1 de enero de ANALYSIS_START_YEAR + SNAPSHOT_LAG_DAYS.
ANALYSIS_ANNUAL_START_DATE = None

# Retraso (en días) desde el cierre del quarter hasta el momento real de análisis/entrada.
# Ejemplo: snapshot Q1 (Mar 31) + 45 días => entrada aproximada a mitad de Q2.
SNAPSHOT_LAG_DAYS = 60

# Si True, cuando un ticker no tenga un reporte del quarter analizado,
# se extrapolan los features usando el promedio de los últimos N quarters disponibles.
# Esto permite mantener el ticker en el universo de test con un snapshot estimado.
ENABLE_FALLBACK_EXTRAPOLATION = True

# Número de quarters previos a usar para extrapolación de features cuando falte el reporte exacto.
# Requiere que existan al menos este número de reports históricos.
FALLBACK_LOOK_BACK_QUARTERS = 4

# Duración del holding de la cartera desde la fecha de entrada.
# 3 meses = aproximación natural a "un trimestre" desplazado.
HOLDING_PERIOD_MONTHS = 3

# =============================================================================
# 6. Parámetros del pipeline ML
# =============================================================================

# Mínimo de trimestres históricos por ticker para incluirlo en train
MIN_HISTORY_QUARTERS = 4

# Mínimo de empresas del mismo sector para calcular Z-score sectorial
SECTOR_ZSCORE_MIN_PEERS = 3

# Número de folds KFold internos para generar OOF scores del meta-learner
OOF_N_SPLITS = 3

# Umbral mínimo de score para incluir un stock en la cartera long / shortlist.
# 0.55 filtra solo los tickers con señal positiva clara; con score-weighted
# portfolio + min_stocks floor, garantiza cartera aunque haya pocos cualificados.
PORTFOLIO_MIN_SCORE = 0.55

# -----------------------------------------------------------------------------
# Ajustes de robustez de scoring (sector + dispersión)
# -----------------------------------------------------------------------------

# Penaliza sectores con pocos peers: sector_confidence = min(1, sqrt(n_peers / k)).
SECTOR_CONFIDENCE_PEERS = 10

# Prior suave del sector sobre el score final:
# final_score *= (SECTOR_SCORE_PRIOR_BASE + SECTOR_SCORE_PRIOR_WEIGHT * sector_score)
SECTOR_SCORE_PRIOR_BASE = 0.5
SECTOR_SCORE_PRIOR_WEIGHT = 0.5

# Si un score de agente tiene baja dispersión, se contrae hacia 0.5.
# scale = min(1, std / SCORE_DISPERSION_MIN_STD)
SCORE_DISPERSION_MIN_STD = 0.03
# Suelo de escala para evitar colapsos a 0.5 cuando std en train es casi 0.
# Solo aplica cuando hay shrink (scale<1) y preserva algo de señal en test.
SCORE_DISPERSION_MIN_SCALE = 0.35

# Ventana de precios usada para features técnicas (RSI, momentum, volatilidad, etc.).
# Se reduce frente al valor histórico de 400 para mantener suficiente contexto
# sin exigir tanto histórico innecesario.
TECHNICAL_LOOKBACK_DAYS = 300

# =============================================================================
# 7. Walk-forward backtesting
# =============================================================================

# Ventana de entrenamiento del walk-forward, en años.
WALKFORWARD_TRAIN_LOOKBACK_YEARS = 8

# Trimestres de test por fold (siempre 1)
WALKFORWARD_TEST_QUARTERS = 1

# Mínimo de empresas en el universo de test de un fold.
# Se calcula dinámicamenbte como un porcentaje del universo total de tickers.
# Ejemplo: si hay 500 tickers totales, será 250 (50%).
MIN_TEST_TICKERS_PERCENT = 50  # porcentaje del universo total

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
# min_samples_leaf=5: hojas más pequeñas → probabilidades más extremas (mejor dispersión)
# Con 300 árboles el riesgo de overfitting es bajo incluso con hojas pequeñas.
MOMENTUM_MIN_SAMPLES_LEAF = 5

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
SENTIMENT_MIN_SAMPLES_LEAF = 5
# ── FeatureSelector ──────────────────────────────────────────────────────────
FEATURE_CORR_THRESHOLD = 0.85
# Peso del score combinado de selección de features:
# combined = w * relevancia_con_y + (1-w) * importancia_modelo
FEATURE_SELECTOR_RELEVANCE_WEIGHT = 0.65
# Modelo auxiliar interno del selector (RandomForest rápido)
FEATURE_SELECTOR_RF_N_ESTIMATORS = 120
FEATURE_SELECTOR_RF_MAX_DEPTH = 5
# Top-N features por agente. Los agentes con más señales legítimas usan más.
# FundamentalAgent: ~30 ratios candidatos → 12 para no perder señal.
# MomentumAgent: técnicos + earnings momentum → 12.
# ValuationAgent, BearAgent, SentimentAgent: universos más acotados → 8.
# MetaLearner: solo scores de agentes + interacciones → sin límite estricto (20).
FEATURE_TOP_N             = 8   # default si el agente no especifica uno propio
FUNDAMENTAL_FEATURE_TOP_N = 12
MOMENTUM_FEATURE_TOP_N    = 12
VALUATION_FEATURE_TOP_N   = 8
BEAR_FEATURE_TOP_N        = 8
SENTIMENT_FEATURE_TOP_N   = 8

# ── MetaLearner (LR + GBM stacking) ──────────────────────────────────────────
META_LR_C             = 0.5
META_GBM_N_ESTIMATORS = 150
META_GBM_MAX_DEPTH    = 3
META_GBM_LEARNING_RATE = 0.05
META_GBM_SUBSAMPLE    = 0.8
# Si True, añade señales de consenso/confianza entre agentes como features extra.
META_ENABLE_CONSENSUS_FEATURES = True
# Umbral para contar agentes claramente alcistas en el snapshot.
META_BULLISH_SCORE_THRESHOLD = 0.55
# Recalibración robusta de score del meta-learner para evitar colapso en <0.5
# cuando la probabilidad cruda sale comprimida o sesgada por drift temporal.
META_ENABLE_SCORE_RECALIBRATION = False
# Temperatura >1 suaviza; <1 hace más agresiva la separación.
META_SCORE_RECALIBRATION_TEMPERATURE = 1.0
# Mezcla del score meta con el consenso medio de agentes base para evitar
# que el meta colapse por drift y pierda toda la señal cross-sectional.
META_BASE_SCORE_BLEND_WEIGHT = 0.55

# =============================================================================
# 9. Reproducibilidad
# =============================================================================

RANDOM_SEED = 42
