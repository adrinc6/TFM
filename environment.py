# =============================================================================
# environment.py — Global project parameters
# =============================================================================
"""
Single source of truth for all pipeline configuration.

Organization:
  1. Execution flags
  2. API keys
  3. Data and results paths
  4. Ticker universe
  5. Analysis period
  6. ML pipeline parameters
  7. Walk-forward backtesting
  8. Agent hyperparameters
  9. Reproducibility

Note: the matplotlib Agg backend is configured in visualization modules,
so all plots are always saved to disk (headless mode).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# =============================================================================
# 1. Execution flags
# =============================================================================

# If True, skips walk-forward backtest and runs only the live fold
SKIP_BACKTEST = False

# If True, re-downloads all data even if it already exists on disk
FORCE_DOWNLOAD = False

# If True, updates only prices and macro data (without consolidation or training)
UPDATE_PRICES_ONLY = False

# If True, retries downloading data for tickers removed due to missing data
RETRY_MISSING_TICKERS = False

# If True, runs the ablation study to measure agent contribution
RUN_ABLATION_STUDY = False

# If True, exports per-ticker debug CSVs (e.g., AAPL) for agent auditing
DEBUG_EXPORT_AGENT_INPUTS = False

# If True, forces recomputation of the master dataset even if it already
# exists on disk.  If False and data_finnhub/master_dataset.parquet exists,
# the dataset is loaded directly without rebuilding.
REBUILD_MASTER_DATASET = False

# Parallel download
DOWNLOAD_MAX_WORKERS = 8

# Finnhub rate limit (seconds between global requests)
FINNHUB_MIN_INTERVAL = 1

# If True, downloads optional endpoints that are not currently used in the
# feature pipeline (company_news, peers, quote, earnings_calendar).
# Set to False to save API quota and storage.
DOWNLOAD_OPTIONAL_ENDPOINTS = True

# =============================================================================
# 2. API Keys
# =============================================================================

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# =============================================================================
# 3. Data and results paths
# =============================================================================

FINNHUB_DATA_DIR         = "data_finnhub"

RESULTS_DIR              = os.path.join("results", "general")
AGENTS_RESULTS_DIR       = os.path.join("results", "agents")
AGENT_MODELS_RESULTS_DIR = os.path.join("results", "agent_models")
BACKTEST_RESULTS_DIR     = os.path.join("results", "backtest")
PLOTS_DIR                = os.path.join("results", "plots")

# =============================================================================
# 4. Ticker universe
# =============================================================================
# Manual fallback list (legacy).
# Note: by default the pipeline uses a dynamic universe from sp500_historic.csv.
# This list is used only if USE_DYNAMIC_SP500_UNIVERSE is disabled.
TICKERS = [
    "NVDA","AAPL","MSFT","AMZN","GOOGL","GOOG","META","AVGO","TSLA","BRK-B","WMT","LLY","JPM","XOM","V","JNJ","MU","COST","MA","ORCL","NFLX","CVX","ABBV","PLTR","PG","BAC","HD","KO","AMD","CAT","GE","CSCO","MRK","LRCX","AMAT","RTX","PM","UNH","MS","GS","IBM","WFC","GEV","TMUS","LIN","MCD","INTC","PEP","VZ","AXP","KLAC","T","NEE","C","AMGN","ABT","CRM","DIS","GILD","TXN","TMO","ANET","TJX","ISRG","SCHW","BA","UBER","APH","DE","PFE","COP","BLK","ADI","LMT","APP","HON","WELL","UNP","QCOM","BKNG","ETN","PANW","DHR","SYK","LOW","CB","SPGI","INTU","PLD","ACN","BMY","NOW","PGR","PH","VRTX","CEG","MCK","MDT","COF","HCA","CME","CRWD","GLW","MO","NEM","SO","SBUX","BSX","SNDK","CMCSA","NOC","DUK","WDC","ADBE","DELL","HWM","EQIX","GD","WM","TT","CVS","STX","WMB","ICE","BX","MAR","PWR","ADP","AMT","MRSH","JCI","UPS","FDX","SNPS","PNC","USB","KKR","CDNS","REGN","BK","NKE","ABNB","MCO","SHW","MSI","FCX","MMM","ITW","CTAS","CMI","ECL","EOG","ORLY","CSX","MNST","RCL","EMR","KMI","MDLZ","VLO","DASH","AEP","CL","CI","MPC","PSX","TDG","RSG","LHX","SLB","HLT","AON","WBD","ROST","HOOD","CRH","GM","ELV","TRV","APO","NSC","COR","APD","FTNT","SRE","SPG","DLR","PCAR","O","OXY","TEL","BKR","VST","AFL","AZO","TFC","D","OKE","CIEN","FANG","AJG","CTVA","COIN","ALL","MPWR","ADSK","TGT","FAST","EXC","TRGP","EA","CAH","XEL","FIX","ZTS","GWW","PSA","AME","KEYS","NXPI","NDAQ","CARR","EW","ETR","F","DDOG","TER","URI","IDXX","BDX","KR","MET","GRMN","YUM","HSY","PEG","CMG","CVNA","DAL","EBAY","ED","AXON","PYPL","MSCI","VTR","WAB","EQT","PCG","AMP","DHI","ROK","AIG","CBRE","FITB","SYY","ODFL","TTWO","WEC","LYV","CCI","TPL","NUE","KDP","HIG","ROP","LVS","MCHP","WDAY","XYZ","MLM","ADM","VMC","NRG","STT","CCL","KVUE","RMD","KMB","EME","ACGL","PAYX","PRU","IR","GEHC","CPRT","A","IRM","EL","ATO","OTIS","AEE","HAL","HBAN","FISV","IBKR","CBOE","DTE","DVN","UAL","VICI","TDY","WAT","FE","MTB","XYL","EXPE","CTSH","EXR","PPL","DOV","HPE","FICO","CNP","TPR","RJF","EIX","VRSK","DG","ES","IQV","WTW","JBL","DOW","AWK","BIIB","CHTR","STZ","KHC","DXCM","ROL","CTRA","EXE","FIS","HUBB","WRB","NTRS","CINF","LYB","STLD","TSCO","CFG","ARES","MTD","BG","Q","LEN","CMS","ON","OMC","AVB","DRI","ULTA","PPG","BRO","CHD","SYF","EQR","PHM","NI","VLTO","EFX","WSM","VRSN","LH","RF","L","DGX","TSN","DLTR","STE","FSLR","LDOS","RL","KEY","MRNA","BR","HUM","CHRW","CF","GIS","SW","NTAP","GPN","LUV","CPAY","LULU","EXPD","TROW","ALB","EVRG","IP","SBAC","PFG","SNA","PKG","INCY","LNT","JBHT","AMCR","SMCI","CSGP","DD","NVR","IFF","PTC","CNC","ZBH","WST","WY","FTV","HOLX","HPQ","LII","HII","PODD","BALL","FFIV","ESS","TXT","VTRS","AKAM","TKO","TRMB","KIM","J","INVH","CDW","MAA","APTV","NDSN","MKC","TYL","DECK","PNR","IEX","GPC","REG","COO","BBY","CLX","HST","APA","ALGN","HAS","EG","DPZ","AVY","ERIE","HRL","GEN","BEN","ALLE","MAS","DOC","PNW","JKHY","GNRC","SOLV","FOX","UHS","UDR","FOXA","IT","TTD","GDDY","SWK","SJM","GL","WYNN","AIZ","BF-B","IVZ","CPT","ZBRA","PSKY","AES","DVA","BLDR","RVTY","MGM","FRT","MOS","NCLH","AOS","NWSA","BAX","HSIC","ARE","BXP","SWKS","TECH","TAP","CRL","FDS","MOH","POOL","CAG","EPAM","MTCH","PAYC","CPB","LW","NWS"
    ]

# Dynamic S&P 500 universe (recommended)
# If True, the pipeline ignores manual TICKERS and builds the universe
# from data_finnhub/sp500_historic.csv based on the analyzed period.
USE_DYNAMIC_SP500_UNIVERSE = True

# CSV of historical S&P 500 members (columns: ticker,start_date,end_date)
SP500_HISTORIC_CSV_PATH = os.path.join(FINNHUB_DATA_DIR, "sp500_historic.csv")

# Target number of tickers per year after historical market cap ranking.
# - If set to 200/300/400: applies annual Top-N by market cap and uses the union.
# - If set to False or 0: uses the full active universe in range without trimming.
SP500_DYNAMIC_TOP_N = False

# =============================================================================
# 5. Analysis period
# =============================================================================

# Date from which raw data is downloaded.
DOWNLOAD_START_DATE = "2000-01-01"

# Quarter range to analyze (both quarterly and annual modes use these).
# - "quarterly": one fold per quarter from START to END.
# - "annual":    one fold per year starting at the START quarter.
#
# Example for 2025Q2 snapshot:
#   ANALYSIS_START_YEAR = 2025, ANALYSIS_START_QUARTER = 2
#   => snapshot closes Jun 30 · entry ~Aug 14 (+ SNAPSHOT_LAG_DAYS)
ANALYSIS_START_YEAR = 2024
ANALYSIS_START_QUARTER = 4

ANALYSIS_END_YEAR = 2025
ANALYSIS_END_QUARTER = 4

# Walk-forward analysis frequency:
# - "quarterly": runs one fold per quarter.
# - "annual":    runs one fold per year.
ANALYSIS_FREQUENCY = "annual"

# Lag (in days) from quarter close to real analysis/entry time.
# Example: Q1 snapshot (Mar 31) + 45 days => approximate entry mid-Q2.
SNAPSHOT_LAG_DAYS = 45

# If True, when a ticker has no report for the analyzed quarter,
# features are extrapolated using the average of the last N available quarters.
# This keeps the ticker in the test universe with an estimated snapshot.
ENABLE_FALLBACK_EXTRAPOLATION = True

# Number of prior quarters used for feature extrapolation when the exact report is missing.
# Requires at least this number of historical reports.
FALLBACK_LOOK_BACK_QUARTERS = 4

# Portfolio holding duration from entry date.
# 3 months = natural approximation to a shifted quarter holding period.
HOLDING_PERIOD_MONTHS = 3

# =============================================================================
# 6. ML pipeline parameters
# =============================================================================

# Minimum historical quarters per ticker to include it in training
MIN_HISTORY_QUARTERS = 4

# Minimum observations required to train an independent model in each sector.
# Under-sampled sectors fall back to the agent-specific score policy defined
# in training config instead of blindly returning 0.5.
SECTOR_SPECIALIST_MIN_SAMPLES = 40

# Number of internal KFold splits to generate OOF scores for the meta-learner
OOF_N_SPLITS = 3

# Purged/embargoed validation gaps (in days) for temporal integrity.
PURGED_CV_GAP_DAYS = 90
EMBARGO_DAYS = 30

# When a sector-specific model is degenerate (e.g. all feature importances are 0),
# use a conservative fallback score instead of a neutral 0.5 to avoid accidental
# promotion of low-confidence candidates.
DEGENERATE_MODEL_FALLBACK_SCORE = 0.25
# Conservative fallback score for long-oriented sector-specialized agents when a
# sector model cannot be trained or a fold fails. Kept aligned with the
# degenerate-model policy so missing/no-signal sectors are penalized consistently.
SECTOR_SPECIALIST_LONG_FALLBACK_SCORE = DEGENERATE_MODEL_FALLBACK_SCORE
# Numerical threshold to consider feature importance mass as zero.
DEGENERATE_MODEL_IMPORTANCE_EPS = 1e-12

# Minimum score threshold to include a stock in the long portfolio / shortlist.
# 0.55 keeps only tickers with clear positive signal; with a score-weighted
# portfolio + min_stocks floor, it still guarantees a portfolio with few qualifiers.
PORTFOLIO_MIN_SCORE = 0.55
# Max number of selected stocks per sector (0 disables sector cap).
# Prevents concentration in a single winning theme/regime.
PORTFOLIO_MAX_STOCKS_PER_SECTOR = 3
# Maximum portfolio weight per ticker (0 disables weight cap).
# Final weights are re-normalized after capping.
PORTFOLIO_MAX_STOCK_WEIGHT = 0.15

# -----------------------------------------------------------------------------
# Scoring robustness settings (sector + dispersion)
# -----------------------------------------------------------------------------

# Penalizes sectors with few peers: sector_confidence = min(1, sqrt(n_peers / k)).
SECTOR_CONFIDENCE_PEERS = 10

# Soft sector prior over final score (additive tilt model):
# final_score += (sector_score - 0.5) * SECTOR_SCORE_PRIOR_WEIGHT * sector_confidence
# A sector_score of 0.7 with full confidence adds +0.06 to each ticker in that sector.
SECTOR_SCORE_PRIOR_BASE = 0.5
SECTOR_SCORE_PRIOR_WEIGHT = 0.4

# If an agent score has low dispersion, it is shrunk toward 0.5.
# scale = min(1, std / SCORE_DISPERSION_MIN_STD)
SCORE_DISPERSION_MIN_STD = 0.03
# Scale floor to avoid collapse to 0.5 when train std is close to 0.
# Applies only when shrink is active (scale<1), preserving some test signal.
SCORE_DISPERSION_MIN_SCALE = 0.35

# Price window used for technical features (RSI, momentum, volatility, etc.).
# Reduced from the historical value of 400 to preserve enough context
# without requiring unnecessary extra history.
TECHNICAL_LOOKBACK_DAYS = 300

# =============================================================================
# 7. Walk-forward backtesting
# =============================================================================

# Maximum walk-forward training window, in years.
# The pipeline will try this maximum and, if it does not meet minimum test coverage,
# will progressively reduce it down to WALKFORWARD_TRAIN_MIN_YEARS.
WALKFORWARD_TRAIN_LOOKBACK_YEARS = 5
# Lower bound for dynamic walk-forward training window.
WALKFORWARD_TRAIN_MIN_YEARS = 4

# Test quarters per fold (always 1)
WALKFORWARD_TEST_QUARTERS = 1

# Minimum companies in a fold test universe.
# Computed dynamically as a percentage of the total ticker universe.
# Example: with 500 total tickers, this is 250 (50%).
MIN_TEST_TICKERS_PERCENT = 75  # percentage of total universe
# Annualized risk-free rate for Sharpe / Sortino
RISK_FREE_RATE = 0.04

# Maximum selected stocks in the long portfolio per fold
TOP_N_STOCKS = 10

# Initial capital for USD simulation (monetary backtest mode)
INITIAL_CAPITAL_USD = 1000.0
# Fixed transaction cost (each BUY and each SELL per ticker)
TRANSACTION_FEE_USD = 1.0
# Percentage slippage applied to execution price (0.01 = 1%)
SLIPPAGE_PCT = 0.001
# If True, runs USD monetary backtest in addition to return metrics.
USE_DOLLAR_BACKTEST = True
# Always allow fractional shares (no integer rounding)
ALLOW_FRACTIONAL_SHARES = True

# Run benchmark and additional baselines for robust comparison
RUN_BASELINES = True

# Number of simulations for random-topN baseline
N_RANDOM_BASELINE_SIMS = 100

# Momentum window for 12-month baseline
BASELINE_MOMENTUM_LOOKBACK_DAYS = 252

# Export additional run artifacts (config, quality, summaries)
EXPORT_RUN_ARTIFACTS = True

# If True, weights the portfolio: ticker #1 weighs (1 + N/10) times more than #N.
# Linear distribution between both ends, normalized to sum to 1.
# Example: N=10 -> first weighs double the last.
#          N=5  -> first weighs 50% more than the last.
# If False, all tickers have equal weight.
SCORE_WEIGHTED_PORTFOLIO = True

# Portfolio optimizer used after model ranking: hrp | risk_parity | markowitz
PORTFOLIO_OPTIMIZER = "hrp"

# =============================================================================
# 8. Agent hyperparameters
# =============================================================================

# ── FundamentalAgent (XGBoost) ────────────────────────────────────────────────
FUNDAMENTAL_N_ESTIMATORS    = 400
FUNDAMENTAL_MAX_DEPTH       = 5
FUNDAMENTAL_LEARNING_RATE   = 0.05
FUNDAMENTAL_SUBSAMPLE       = 0.8
FUNDAMENTAL_COLSAMPLE       = 0.7
FUNDAMENTAL_MIN_CHILD_WEIGHT = 5

# Per-agent column controls:
# - *_FEATURE_COLUMNS is the AUTHORITATIVE list used by agents.
# - *_FEATURE_EXCLUDE is informational/documentary: those columns may also
#   be present in the master dataset, but training uses only *_FEATURE_COLUMNS.
FUNDAMENTAL_FEATURE_COLUMNS = [
  # Profitability (core)
  "roa",
  "roe",
  "roi",
  "roic",
  "capex_to_revenue",

  # Margins
  "net_margin",
  "gross_margin",
  "ebitda_margin",
  "operating_margin",
  "fcf_margin",

  # Liquidity
  "current_ratio",
  "quick_ratio",

  # Leverage / solvency
  "debt_equity",
  "debt_to_ebitda",
  "interest_coverage",

  # Growth (cleaned set)
  "revenue_yoy_growth",
  "fcf_yoy_growth",

  # Efficiency / quality
  "piotroski_fscore",

  # Long-term trends
  "roe_trend_3y",
  "net_margin_trend_3y",
  "gross_margin_trend_3y",
]

FUNDAMENTAL_FEATURE_EXCLUDE = [
  # Highly correlated growth metrics (keep simpler set)
  "net_income_yoy_growth",
  "eps_yoy_growth",
  "operating_income_yoy_growth",

  # Noisy YoY changes of already included metrics
  "roa_change_yoy",
  "gross_margin_change_yoy",
  "current_ratio_change_yoy",

  # Low cross-sector comparability
  "accruals_ratio",
  "total_debt_yoy_growth",

  # Weak or redundant signals
  "consecutive_losses",
  "earnings_quality",

  # Redundant short-term trends (keep longer horizon)
  "roe_trend_2y",
  "net_margin_trend_2y",
]
# ── ValuationAgent (GBM) ─────────────────────────────────────────────────────
VALUATION_N_ESTIMATORS  = 200
VALUATION_MAX_DEPTH     = 4
VALUATION_LEARNING_RATE = 0.05
VALUATION_SUBSAMPLE     = 0.8
VALUATION_FEATURE_COLUMNS = [
  # Core valuation multiples
  "pe_ratio",
  "ps_ratio",
  "ev_to_ebitda",
  "pb_ratio",

  # Yield-based valuation (very informative cross-sector)
  "fcf_yield",
  "earnings_yield",

  # Relative valuation vs own history
  "pe_vs_5y_median",
  "ev_ebitda_vs_5y_median",
]

VALUATION_FEATURE_EXCLUDE = [
  # Redundant historical comparisons
  "pb_vs_5y_median",

  # Weak / noisy signals for valuation vs sector
  "eps_surprise_pct",         # short-term, more trading signal
  "eps_revision",             # analyst-driven, not pure valuation
]

# ── MomentumAgent (Random Forest) ────────────────────────────────────────────
MOMENTUM_N_ESTIMATORS    = 300
MOMENTUM_MAX_DEPTH       = 8
# min_samples_leaf=5: smaller leaves -> more extreme probabilities (better dispersion)
# With 300 trees, overfitting risk remains low even with small leaves.
MOMENTUM_MIN_SAMPLES_LEAF = 5
MOMENTUM_FEATURE_COLUMNS = [
  # Price position
  "price_vs_52w_high",

  # Momentum (multi-horizon but non-redundant)
  "momentum_3m",
  "momentum_6m",
  "momentum_12m",

  # Trend (long-term signal)
  "sma_50",
  "sma_200",

  # Volatility (regime detection)
  "volatility_60d",

  # Volume confirmation
  "vol_ratio_20_50",

  # RSI signals
  "rsi_14",
  "rsi_28",
]

MOMENTUM_FEATURE_EXCLUDE = [
  # MACD components (highly correlated, noisy for cross-sectional use)
  "macd",
  "macd_signal",
  "macd_hist",

  # Moving average redundancy
  "sma_20",

  # Overlapping technical indicators
  "bb_pct",

  # Redundant price positioning
  "price_vs_52w_low",

  # Overlapping momentum horizons
  "momentum_1m",

  # Volatility redundancy
  "volatility_20d",
  "atr_14",
]

# ── BearAgent (Hybrid Random Forest) ─────────────────────────────────────────
BEAR_N_ESTIMATORS = 200
BEAR_MAX_DEPTH    = 6
# Rule-layer vs ML-layer weight in final score.
# ML layer gets higher weight because it captures non-linear risk interactions
# that simple threshold rules miss (e.g., high debt is fine for utilities).
BEAR_RULE_WEIGHT  = 0.35
BEAR_ML_WEIGHT    = 0.65
# Risk score above which the meta-learner forces Underperform
BEAR_HARD_THRESHOLD = 0.90
BEAR_FEATURE_COLUMNS = [
  "debt_equity",
  "debt_to_ebitda",
  "interest_coverage",
  "fcf_margin",
  "current_ratio",
  "revenue_decline",
  "insider_sell_ratio",
  "consecutive_losses",
  "total_debt_yoy_growth",
  "insider_net_ratio_90d",
  "eps_surprise_pct",
  "eps_revision",
]

BEAR_FEATURE_EXCLUDE = [
]

# ── SentimentAgent (Random Forest) ───────────────────────────────────────────
# Enables/disables the standalone sentiment agent in the base stack.
# Disabled by default because this signal is often sparse/neutral in current data.
ENABLE_SENTIMENT_AGENT = True
SENTIMENT_N_ESTIMATORS    = 200
SENTIMENT_MAX_DEPTH       = 6
SENTIMENT_MIN_SAMPLES_LEAF = 5
SENTIMENT_FEATURE_COLUMNS = [
  # Analyst sentiment (core)
  "analyst_consensus",
  "analyst_dispersion",          # disagreement = uncertainty
  "analyst_consensus_change",    # revisions trend

  # Market-implied sentiment
  "mspr_3m",
  "mspr_trend",

  # Insider signal (keep only one clean proxy)
  "insider_net_ratio_90d",

  # Earnings expectation vs reality
  "beat_rate_4q",
  "eps_surprise_avg_4q",

  # NLP sentiment (FinBERT)
  "finbert_sentiment_polarity",
  "finbert_uncertainty_score",
  "finbert_risk_intensity",
  "finbert_bullish_tone",
]

SENTIMENT_FEATURE_EXCLUDE = [
  # Redundant analyst metrics (overlapping with consensus)
  "analyst_buy_ratio",
  "analyst_bearish_score",
  "analyst_strong_buy_pct",

  # Redundant insider metric
  "insider_sell_ratio",

  # Noisy / too short-term
  "eps_surprise_pct",
]

# SectorRotationAgent
SECTOR_ROTATION_FEATURE_COLUMNS = [
  # =========================
  # 1. MOMENTUM (CORE DRIVER)
  # =========================
  "momentum_3m",
  "momentum_6m",
  "momentum_12m",
  "price_vs_52w_high",

  # =========================
  # 2. GROWTH (CONFIRMATION)
  # =========================
  "revenue_yoy_growth",

  # =========================
  # 3. VALUATION (ENTRY TIMING)
  # =========================
  "fcf_yield",
  "earnings_yield",
  "ev_to_ebitda",
  "pe_vs_5y_median",

  # =========================
  # 4. QUALITY (FILTER)
  # =========================
  "roic",
  "fcf_margin",
  "net_margin",

  # =========================
  # 5. RISK / BALANCE SHEET
  # =========================
  "debt_to_ebitda",
  "interest_coverage",

  # =========================
  # 6. SENTIMENT (ACCELERATOR)
  # =========================
  "eps_revision",
  "analyst_consensus_change",
  "analyst_dispersion",

  # =========================
  # 7. REGIME / RISK CONTEXT
  # =========================
  "volatility_60d",
]

SECTOR_ROTATION_FEATURE_EXCLUDE = []

# ── FeatureSelector ──────────────────────────────────────────────────────────
FEATURE_CORR_THRESHOLD = 0.85
# Weight of the combined feature-selection score:
# combined = w * relevance_to_y + (1-w) * model_importance
FEATURE_SELECTOR_RELEVANCE_WEIGHT = 0.65

# If True, exports a per-fold report of requested vs actually used columns.
EXPORT_FEATURE_USAGE_REPORT = True
# Selector internal helper model (fast RandomForest)
FEATURE_SELECTOR_RF_N_ESTIMATORS = 120
FEATURE_SELECTOR_RF_MAX_DEPTH = 5
# Final selector importance rule:
# - keep features with importance >= (top_importance * FEATURE_IMPORTANCE_CUTOFF_FRACTION)
# - then cap final count between [FEATURE_IMPORTANCE_MIN_KEEP, FEATURE_IMPORTANCE_MAX_KEEP].
FEATURE_IMPORTANCE_CUTOFF_FRACTION = 0.40
FEATURE_IMPORTANCE_MIN_KEEP = 6
FEATURE_IMPORTANCE_MAX_KEEP = 10
# Global Top-N for FeatureSelector pre-filtering (all agents).
# FINAL selection across all agents is uniformly controlled by:
#   - FEATURE_IMPORTANCE_CUTOFF_FRACTION = 0.40
#   - FEATURE_IMPORTANCE_MIN_KEEP = 6
#   - FEATURE_IMPORTANCE_MAX_KEEP = 10
# These limits ensure all agents use between 6 and 10 final features.
FEATURE_TOP_N = 14

# ── MetaLearner (LR + GBM stacking) ──────────────────────────────────────────
META_LR_C             = 0.5
META_GBM_N_ESTIMATORS = 150
META_GBM_MAX_DEPTH    = 3
META_GBM_LEARNING_RATE = 0.05
META_GBM_SUBSAMPLE    = 0.8
META_FEATURE_COLUMNS = [
  # Base agent scores (stacking layer)
  "fundamental_score",
  "valuation_score",
  "momentum_score",
  "bear_score",
  "sentiment_score",
  "sector_score",
  "regime_adjusted_score",

  # Sector-relative percentile ranks (original)
  "pe_rank_sector",
  "momentum_pct_sector",
  "roe_pct_sector",

  # Sector-relative value/quality ranks (new)
  "pb_rank_sector",
  "fcf_yield_rank_sector",
  "roic_rank_sector",
  "ev_ebitda_rank_sector",
  "debt_rank_sector",

  # Universe-wide percentile ranks (new)
  "momentum_12m_rank_universe",
  "quality_rank_universe",
  "value_rank_universe",
  "piotroski_rank_universe",

  # Volatility-adjusted signals
  "momentum_vol_adj",
  "value_vol_adj",
  "quality_vol_adj",

  # Interaction features
  "value_x_momentum",
  "quality_x_lowvol",
  "sentiment_x_earnings_surprise",
  "quality_x_value_universe",
  "momentum_quality_signal",

  # Macro regime context
  "vix",
  "yield_curve",
  "sp500_momentum_3m",
  "sp500_momentum_12m",
]
META_FEATURE_EXCLUDE = []
# Base score columns on which meta computes consensus/interactions.
META_AGENT_SCORE_COLUMNS = [
  "fundamental_score",
  "valuation_score",
  "momentum_score",
  "bear_score",
  "sentiment_score",
  "sector_score",
]
# If True, adds agent consensus/confidence signals as extra features.
META_ENABLE_CONSENSUS_FEATURES = True
# Threshold to count clearly bullish agents in the snapshot.
META_BULLISH_SCORE_THRESHOLD = 0.55
# Robust meta-learner score recalibration to avoid collapse below 0.5
# when raw probabilities are compressed or time-drift biased.
META_ENABLE_SCORE_RECALIBRATION = False
# Temperature >1 smooths; <1 makes separation more aggressive.
META_SCORE_RECALIBRATION_TEMPERATURE = 1.0
# Blend meta score with average base-agent consensus to avoid
# meta collapse from drift and preserve cross-sectional signal.
# Reduced from 0.58 to give the learned meta-model more influence
# over the final score while still keeping a regularising anchor.
META_BASE_SCORE_BLEND_WEIGHT = 0.40

# ── AlphaMetaLearner (XGBoost Regressor + Ranker + Classifier) ───────────────
ALPHA_META_REG_N_ESTIMATORS  = 450
ALPHA_META_REG_MAX_DEPTH     = 5
ALPHA_META_REG_LEARNING_RATE = 0.03
ALPHA_META_REG_SUBSAMPLE     = 0.85
ALPHA_META_REG_COLSAMPLE     = 0.8

ALPHA_META_RANK_N_ESTIMATORS  = 300
ALPHA_META_RANK_MAX_DEPTH     = 4
ALPHA_META_RANK_LEARNING_RATE = 0.05
ALPHA_META_RANK_SUBSAMPLE     = 0.9
ALPHA_META_RANK_COLSAMPLE     = 0.8

ALPHA_META_RISK_N_ESTIMATORS  = 250
ALPHA_META_RISK_MAX_DEPTH     = 4
ALPHA_META_RISK_LEARNING_RATE = 0.05
ALPHA_META_RISK_SUBSAMPLE     = 0.9
ALPHA_META_RISK_COLSAMPLE     = 0.8

# Blend weight: regime_adjusted_score = ALPHA_META_RANK_BLEND * ranking_score
#               + (1 - ALPHA_META_RANK_BLEND) * regime_adj
ALPHA_META_RANK_BLEND = 0.65

# ── MomentumAgent TFT-lite blend weight ──────────────────────────────────────
# Final score = (1 - MOMENTUM_DEEP_BLEND_WEIGHT) * RF_score
#               + MOMENTUM_DEEP_BLEND_WEIGHT * TFT_score
MOMENTUM_DEEP_BLEND_WEIGHT = 0.35



# =============================================================================
# 9. Reproducibility
# =============================================================================

RANDOM_SEED = 42

# =============================================================================
# 10. TP/SL + Confidence Strategy
# =============================================================================

# --- Signal generation (take-profit / stop-loss) ----------------------------

# Baseline take-profit percentage applied when agent score == 0.5
TP_SL_BASE_TP = 0.08       # 8 %

# Baseline stop-loss percentage applied when agent score == 0.5
TP_SL_BASE_SL = 0.05       # 5 %

# Maximum shift in TP as score moves from 0.5 → 1.0 (or 0.5 → 0.0)
TP_SL_TP_SENSITIVITY = 0.10

# Maximum shift in SL as score moves from 0.5 → 1.0 (or 0.5 → 0.0)
TP_SL_SL_SENSITIVITY = 0.04

# Hard bounds on TP and SL percentages
TP_SL_MIN_TP = 0.02
TP_SL_MAX_TP = 0.25
TP_SL_MIN_SL = 0.01
TP_SL_MAX_SL = 0.15

# --- Confidence model -------------------------------------------------------

# Relative weight of the raw model score vs. historical calibration
TP_SL_CONFIDENCE_SCORE_WEIGHT = 0.50
TP_SL_CONFIDENCE_CALIBRATION_WEIGHT = 0.50

# --- Portfolio construction -------------------------------------------------

# Minimum stocks required to invest (0 candidates → no investment)
TP_SL_MIN_STOCKS = 4

# Maximum portfolio size
TP_SL_MAX_STOCKS = 8

# Maximum stocks from the same GICS sector in the portfolio
TP_SL_SECTOR_CAP = 3

# Minimum expected value (EV) for a stock to be eligible
# EV = confidence × tp_pct − (1 − confidence) × sl_pct
TP_SL_EV_THRESHOLD = 0.0

# --- Backtesting engine -----------------------------------------------------

# Maximum calendar days to hold a position before forcing closure ("NONE")
TP_SL_MAX_HOLDING_DAYS = 90

# --- Agent weight tracker ---------------------------------------------------

# EWMA decay factor for historical hit-rate tracking (higher = slower to forget)
TP_SL_WEIGHT_DECAY = 0.85

# Prior hit-rate before any fold data are available (maximum uncertainty = 0.5)
TP_SL_WEIGHT_PRIOR = 0.50

# Floor weight so every agent stays active (prevents weight collapse)
TP_SL_WEIGHT_MIN = 0.05
