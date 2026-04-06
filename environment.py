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

# Si True, habilita caché de artefactos intermedios para reutilizar cálculo
ENABLE_CACHE = True

# Carpeta raíz de caché
CACHE_DIR = "cache"

# Versión de esquema de caché. Súbela cuando cambie la estructura de artefactos
# o la política de columnas del dataset.
CACHE_SCHEMA_VERSION = 2

# Reutiliza dataset maestro si coincide contexto (tickers + parámetros)
CACHE_USE_MASTER_DATASET = True

# Reutiliza artefactos derivados de DataRouter (tickers disponibles y datos de mercado preparados)
CACHE_USE_ROUTER_DERIVED = True

# Reutiliza resumen final walk-forward y salta recomputar backtest completo
CACHE_USE_WALKFORWARD_SUMMARY = False

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
# Lista manual de fallback (legacy).
# Nota: por defecto el pipeline usa universo dinámico desde sp500_historic.csv.
# Esta lista solo se usa si desactivas USE_DYNAMIC_SP500_UNIVERSE.
TICKERS = [
    "NVDA","AAPL","MSFT","AMZN","GOOGL","GOOG","META","AVGO","TSLA","BRK-B","WMT","LLY","JPM","XOM","V","JNJ","MU","COST","MA","ORCL","NFLX","CVX","ABBV","PLTR","PG","BAC","HD","KO","AMD","CAT","GE","CSCO","MRK","LRCX","AMAT","RTX","PM","UNH","MS","GS","IBM","WFC","GEV","TMUS","LIN","MCD","INTC","PEP","VZ","AXP","KLAC","T","NEE","C","AMGN","ABT","CRM","DIS","GILD","TXN","TMO","ANET","TJX","ISRG","SCHW","BA","UBER","APH","DE","PFE","COP","BLK","ADI","LMT","APP","HON","WELL","UNP","QCOM","BKNG","ETN","PANW","DHR","SYK","LOW","CB","SPGI","INTU","PLD","ACN","BMY","NOW","PGR","PH","VRTX","CEG","MCK","MDT","COF","HCA","CME","CRWD","GLW","MO","NEM","SO","SBUX","BSX","SNDK","CMCSA","NOC","DUK","WDC","ADBE","DELL","HWM","EQIX","GD","WM","TT","CVS","STX","WMB","ICE","BX","MAR","PWR","ADP","AMT","MRSH","JCI","UPS","FDX","SNPS","PNC","USB","KKR","CDNS","REGN","BK","NKE","ABNB","MCO","SHW","MSI","FCX","MMM","ITW","CTAS","CMI","ECL","EOG","ORLY","CSX","MNST","RCL","EMR","KMI","MDLZ","VLO","DASH","AEP","CL","CI","MPC","PSX","TDG","RSG","LHX","SLB","HLT","AON","WBD","ROST","HOOD","CRH","GM","ELV","TRV","APO","NSC","COR","APD","FTNT","SRE","SPG","DLR","PCAR","O","OXY","TEL","BKR","VST","AFL","AZO","TFC","D","OKE","CIEN","FANG","AJG","CTVA","COIN","ALL","MPWR","ADSK","TGT","FAST","EXC","TRGP","EA","CAH","XEL","FIX","ZTS","GWW","PSA","AME","KEYS","NXPI","NDAQ","CARR","EW","ETR","F","DDOG","TER","URI","IDXX","BDX","KR","MET","GRMN","YUM","HSY","PEG","CMG","CVNA","DAL","EBAY","ED","AXON","PYPL","MSCI","VTR","WAB","EQT","PCG","AMP","DHI","ROK","AIG","CBRE","FITB","SYY","ODFL","TTWO","WEC","LYV","CCI","TPL","NUE","KDP","HIG","ROP","LVS","MCHP","WDAY","XYZ","MLM","ADM","VMC","NRG","STT","CCL","KVUE","RMD","KMB","EME","ACGL","PAYX","PRU","IR","GEHC","CPRT","A","IRM","EL","ATO","OTIS","AEE","HAL","HBAN","FISV","IBKR","CBOE","DTE","DVN","UAL","VICI","TDY","WAT","FE","MTB","XYL","EXPE","CTSH","EXR","PPL","DOV","HPE","FICO","CNP","TPR","RJF","EIX","VRSK","DG","ES","IQV","WTW","JBL","DOW","AWK","BIIB","CHTR","STZ","KHC","DXCM","ROL","CTRA","EXE","FIS","HUBB","WRB","NTRS","CINF","LYB","STLD","TSCO","CFG","ARES","MTD","BG","Q","LEN","CMS","ON","OMC","AVB","DRI","ULTA","PPG","BRO","CHD","SYF","EQR","PHM","NI","VLTO","EFX","WSM","VRSN","LH","RF","L","DGX","TSN","DLTR","STE","FSLR","LDOS","RL","KEY","MRNA","BR","HUM","CHRW","CF","GIS","SW","NTAP","GPN","LUV","CPAY","LULU","EXPD","TROW","ALB","EVRG","IP","SBAC","PFG","SNA","PKG","INCY","LNT","JBHT","AMCR","SMCI","CSGP","DD","NVR","IFF","PTC","CNC","ZBH","WST","WY","FTV","HOLX","HPQ","LII","HII","PODD","BALL","FFIV","ESS","TXT","VTRS","AKAM","TKO","TRMB","KIM","J","INVH","CDW","MAA","APTV","NDSN","MKC","TYL","DECK","PNR","IEX","GPC","REG","COO","BBY","CLX","HST","APA","ALGN","HAS","EG","DPZ","AVY","ERIE","HRL","GEN","BEN","ALLE","MAS","DOC","PNW","JKHY","GNRC","SOLV","FOX","UHS","UDR","FOXA","IT","TTD","GDDY","SWK","SJM","GL","WYNN","AIZ","BF-B","IVZ","CPT","ZBRA","PSKY","AES","DVA","BLDR","RVTY","MGM","FRT","MOS","NCLH","AOS","NWSA","BAX","HSIC","ARE","BXP","SWKS","TECH","TAP","CRL","FDS","MOH","POOL","CAG","EPAM","MTCH","PAYC","CPB","LW","NWS"
    ]

# Universo dinámico S&P 500 (recomendado)
# Si True, el pipeline ignora la lista manual TICKERS y construye el universo
# desde data_finnhub/sp500_historic.csv en función del periodo analizado.
USE_DYNAMIC_SP500_UNIVERSE = True

# CSV de miembros históricos del S&P 500 (columnas: ticker,start_date,end_date)
SP500_HISTORIC_CSV_PATH = os.path.join(FINNHUB_DATA_DIR, "sp500_historic.csv")

# Número objetivo de tickers por año tras ranking por market cap histórico.
# - Si pones 200/300/400: aplica Top-N anual por market cap y usa la unión.
# - Si pones False o 0: usa todo el universo activo del rango sin recorte.
SP500_DYNAMIC_TOP_N = False

# =============================================================================
# 5. Período de análisis
# =============================================================================

# Fecha desde la que se descargan los datos brutos.
DOWNLOAD_START_DATE = "2000-01-01"

# Inicio del período de análisis/backtest walk-forward (quarters de snapshot a analizar).
ANALYSIS_START_YEAR = 2015
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

# Objetivo de entrenamiento para agentes base (todos menos SectorRotation):
# - "vs_sector": y=1 si la compañía supera la mediana de su sector en el snapshot.
# - "vs_universe": y=1 si supera la mediana del universo en el snapshot.
BASE_AGENTS_LABEL_MODE = "vs_sector"

# Mínimo de peers por sector x snapshot para usar benchmark sectorial en labels.
# Si no se alcanza, se usa fallback a mediana del universo en ese snapshot.
BASE_LABEL_SECTOR_MIN_PEERS = 3

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

# Ventana máxima de entrenamiento del walk-forward, en años.
# El pipeline intentará este máximo y, si no cumple cobertura mínima de test,
# reducirá progresivamente hasta WALKFORWARD_TRAIN_MIN_YEARS.
WALKFORWARD_TRAIN_LOOKBACK_YEARS = 10

# Límite inferior de ventana de entrenamiento dinámica del walk-forward.
WALKFORWARD_TRAIN_MIN_YEARS = 5

# Trimestres de test por fold (siempre 1)
WALKFORWARD_TEST_QUARTERS = 1

# Mínimo de empresas en el universo de test de un fold.
# Se calcula dinámicamenbte como un porcentaje del universo total de tickers.
# Ejemplo: si hay 500 tickers totales, será 250 (50%).
MIN_TEST_TICKERS_PERCENT = 80  # porcentaje del universo total

# Tasa libre de riesgo anualizada para Sharpe / Sortino
RISK_FREE_RATE = 0.04

# Máximo de stocks seleccionados en la cartera long por fold
TOP_N_STOCKS = 10

# Capital inicial para simulación en USD (modo backtest monetario)
INITIAL_CAPITAL_USD = 1000.0

# Coste fijo por transacción (cada BUY y cada SELL por ticker)
TRANSACTION_FEE_USD = 1.0

# Slippage porcentual aplicado al precio de ejecución (0.01 = 1%)
SLIPPAGE_PCT = 0.0

# Si True, además de métricas de retorno, ejecuta backtest monetario en USD.
USE_DOLLAR_BACKTEST = True

# Siempre permitir acciones fraccionarias (sin redondeo a enteros)
ALLOW_FRACTIONAL_SHARES = True

# Ejecutar benchmark y baselines adicionales para comparativa robusta
RUN_BASELINES = True

# Número de simulaciones para baseline random-topN
N_RANDOM_BASELINE_SIMS = 100

# Ventana de momentum para baseline de 12 meses
BASELINE_MOMENTUM_LOOKBACK_DAYS = 252

# Exporta artefactos adicionales del run (config, calidad, resúmenes)
EXPORT_RUN_ARTIFACTS = True

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

# Control de columnas por agente:
# - *_FEATURE_COLUMNS es la lista AUTORITATIVA que usan los agentes.
# - *_FEATURE_EXCLUDE es informativa/documental: esas columnas tambien se
#   garantizan en dataset maestro, pero el entrenamiento usa solo *_FEATURE_COLUMNS.
FUNDAMENTAL_FEATURE_COLUMNS = [
  "roe", "roa", "roi", "roic",
  "net_margin", "gross_margin", "fcf_margin", "ebitda_margin", "operating_margin",
  "current_ratio", "quick_ratio",
  "debt_equity", "debt_to_ebitda", "interest_coverage",
  "revenue_yoy_growth", "net_income_yoy_growth", "eps_yoy_growth",
  "fcf_yoy_growth", "operating_income_yoy_growth", "total_debt_yoy_growth",
  "roa_change_yoy", "gross_margin_change_yoy", "current_ratio_change_yoy",
  "accruals_ratio", "capex_to_revenue", "consecutive_losses",
  "earnings_quality", "piotroski_fscore",
  "eps",
  "roe_trend_2y", "roe_trend_3y",
  "net_margin_trend_2y", "net_margin_trend_3y",
  "gross_margin_trend_3y",
]
FUNDAMENTAL_FEATURE_EXCLUDE = []
# ── ValuationAgent (GBM) ─────────────────────────────────────────────────────
VALUATION_N_ESTIMATORS  = 200
VALUATION_MAX_DEPTH     = 4
VALUATION_LEARNING_RATE = 0.05
VALUATION_SUBSAMPLE     = 0.8
VALUATION_FEATURE_COLUMNS = [
  "pe_ratio", "pb_ratio", "ps_ratio", "ev_to_ebitda", "fcf_yield", "earnings_yield",
  "pe_vs_5y_median", "pb_vs_5y_median", "ev_ebitda_vs_5y_median",
  "eps_surprise_pct", "eps_revision", "eps_est", "eps_reported",
]
VALUATION_FEATURE_EXCLUDE = []

# ── MomentumAgent (Random Forest) ────────────────────────────────────────────
MOMENTUM_N_ESTIMATORS    = 300
MOMENTUM_MAX_DEPTH       = 8
# min_samples_leaf=5: hojas más pequeñas → probabilidades más extremas (mejor dispersión)
# Con 300 árboles el riesgo de overfitting es bajo incluso con hojas pequeñas.
MOMENTUM_MIN_SAMPLES_LEAF = 5
MOMENTUM_FEATURE_COLUMNS = [
  "rsi_14", "rsi_28",
  "macd", "macd_signal", "macd_hist",
  "sma_20", "sma_50", "sma_200",
  "bb_pct",
  "price_vs_52w_high", "price_vs_52w_low",
  "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m",
  "volatility_20d", "volatility_60d", "atr_14",
  "vol_ratio_20_50",
  "beat_rate_4q", "eps_surprise_avg_4q", "eps_revision",
]
MOMENTUM_FEATURE_EXCLUDE = []

# ── BearAgent (Random Forest híbrido) ────────────────────────────────────────
BEAR_N_ESTIMATORS = 200
BEAR_MAX_DEPTH    = 6
# Peso de la capa de reglas vs ML en el score final
BEAR_RULE_WEIGHT  = 0.5
BEAR_ML_WEIGHT    = 0.5
# Score de riesgo por encima del cual el meta-learner fuerza Underperform
BEAR_HARD_THRESHOLD = 0.90
BEAR_FEATURE_COLUMNS = [
  "total_debt_yoy_growth",
  "debt_equity",
  "debt_to_ebitda",
  "fcf_margin",
  "current_ratio",
  "consecutive_losses",
  "revenue_decline",
  "interest_coverage",
  "insider_net_ratio_90d",
  "insider_sell_ratio",
  "eps_surprise_pct",
  "eps_revision",
]
BEAR_FEATURE_EXCLUDE = []

# ── SentimentAgent (Random Forest) ───────────────────────────────────────────
SENTIMENT_N_ESTIMATORS    = 200
SENTIMENT_MAX_DEPTH       = 6
SENTIMENT_MIN_SAMPLES_LEAF = 5
SENTIMENT_FEATURE_COLUMNS = [
  "analyst_buy_ratio",
  "analyst_bearish_score",
  "analyst_consensus",
  "analyst_dispersion",
  "analyst_strong_buy_pct",
  "analyst_consensus_change",
  "mspr_3m",
  "mspr_trend",
  "insider_net_ratio_90d",
  "insider_sell_ratio",
  "beat_rate_4q",
  "eps_surprise_avg_4q",
  "eps_surprise_pct",
]
SENTIMENT_FEATURE_EXCLUDE = []

# SectorRotationAgent
SECTOR_ROTATION_FEATURE_COLUMNS = [
  "roe", "roa", "net_margin", "gross_margin", "fcf_margin", "ebitda_margin",
  "operating_margin",
  "revenue_yoy_growth", "net_income_yoy_growth", "eps_yoy_growth",
  "debt_to_ebitda", "debt_equity", "interest_coverage", "current_ratio",
  "quick_ratio",
  "pe_ratio", "pb_ratio", "ev_to_ebitda", "fcf_yield",
  "earnings_yield",
  "pe_vs_5y_median", "pb_vs_5y_median",
  "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m",
  "volatility_20d", "volatility_60d", "rsi_14",
  "analyst_buy_ratio", "analyst_consensus", "analyst_dispersion", "analyst_bearish_score",
  "insider_net_ratio_90d", "insider_sell_ratio",
  "beat_rate_4q", "eps_surprise_pct", "eps_surprise_avg_4q", "eps_revision", "mspr_3m",
]
SECTOR_ROTATION_FEATURE_EXCLUDE = []
# ── FeatureSelector ──────────────────────────────────────────────────────────
FEATURE_CORR_THRESHOLD = 0.85
# Peso del score combinado de selección de features:
# combined = w * relevancia_con_y + (1-w) * importancia_modelo
FEATURE_SELECTOR_RELEVANCE_WEIGHT = 0.65

# Si True, exporta por fold un reporte con columnas pedidas vs realmente usadas.
EXPORT_FEATURE_USAGE_REPORT = True
# Modelo auxiliar interno del selector (RandomForest rápido)
FEATURE_SELECTOR_RF_N_ESTIMATORS = 120
FEATURE_SELECTOR_RF_MAX_DEPTH = 5
# Regla de selección final por importancia del selector:
# - conservar features con importancia >= (top_importance * FEATURE_IMPORTANCE_CUTOFF_FRACTION)
# - y luego acotar entre [FEATURE_IMPORTANCE_MIN_KEEP, FEATURE_IMPORTANCE_MAX_KEEP].
FEATURE_IMPORTANCE_CUTOFF_FRACTION = 0.50
FEATURE_IMPORTANCE_MIN_KEEP = 4
FEATURE_IMPORTANCE_MAX_KEEP = 10
# Top-N global para el pre-filtrado del FeatureSelector (todos los agentes).
# La selección FINAL en todos los agentes se controla uniformemente por:
#   - FEATURE_IMPORTANCE_CUTOFF_FRACTION = 0.50
#   - FEATURE_IMPORTANCE_MIN_KEEP = 4
#   - FEATURE_IMPORTANCE_MAX_KEEP = 10
# Estos límites garantizan que todos los agentes usen entre 4 y 10 features finales.
FEATURE_TOP_N = 14

# ── MetaLearner (LR + GBM stacking) ──────────────────────────────────────────
META_LR_C             = 0.5
META_GBM_N_ESTIMATORS = 150
META_GBM_MAX_DEPTH    = 3
META_GBM_LEARNING_RATE = 0.05
META_GBM_SUBSAMPLE    = 0.8
META_FEATURE_COLUMNS = [
  "fundamental_score",
  "valuation_score",
  "momentum_score",
  "bear_score",
  "sentiment_score",
  "sector_score",
]
META_FEATURE_EXCLUDE = []
# Columnas de scores base sobre las que el meta calcula consenso/interacciones.
META_AGENT_SCORE_COLUMNS = [
  "fundamental_score",
  "valuation_score",
  "momentum_score",
  "bear_score",
  "sentiment_score",
  "sector_score",
]
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
