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
import json

# Load environment variables from .env file
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


def _coerce_override_value(raw_value, current_value):
  """Coerce override value to the current constant type when possible."""
  if isinstance(current_value, bool):
    if isinstance(raw_value, bool):
      return raw_value
    if isinstance(raw_value, str):
      norm = raw_value.strip().lower()
      if norm in {"1", "true", "yes", "y", "on"}:
        return True
      if norm in {"0", "false", "no", "n", "off"}:
        return False
    return bool(raw_value)

  if isinstance(current_value, int) and not isinstance(current_value, bool):
    return int(raw_value)

  if isinstance(current_value, float):
    return float(raw_value)

  if isinstance(current_value, str):
    return str(raw_value)

  # For lists/dicts/tuples/other objects we trust JSON types as-is.
  return raw_value


def _apply_environment_overrides(namespace: dict) -> None:
  """Apply runtime overrides from ENV_OVERRIDES_JSON onto module constants.

  Example:
    ENV_OVERRIDES_JSON='{"WALKFORWARD_TRAIN_LOOKBACK_YEARS": 10}'
  """
  raw = os.getenv("ENV_OVERRIDES_JSON", "").strip()
  if not raw:
    return

  try:
    payload = json.loads(raw)
  except Exception:
    return

  if not isinstance(payload, dict):
    return

  for key, value in payload.items():
    if not isinstance(key, str):
      continue
    if key not in namespace:
      continue
    if not key.isupper():
      continue

    current = namespace[key]
    try:
      namespace[key] = _coerce_override_value(value, current)
    except Exception:
      # Skip invalid overrides without breaking module import.
      continue

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
# Manual fallback list.
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
# 5. Analysis schedule
# =============================================================================

# Date from which raw data is downloaded.
DOWNLOAD_START_DATE = "2000-01-01"

# Anchor date for the first test entry.
# The pipeline then creates additional test windows by moving backwards in
# HOLDING_PERIOD_MONTHS steps.
# Example:
#   ANALYSIS_REFERENCE_DATE = "2026-02-15"
#   HOLDING_PERIOD_MONTHS = 12
#   WALKFORWARD_NUM_TESTS = 4
#   -> test entries at 2026-02-15, 2025-02-15, 2024-02-15, 2023-02-15.
ANALYSIS_REFERENCE_DATE = "2026-02-15"

# Number of historical test windows generated from ANALYSIS_REFERENCE_DATE.
WALKFORWARD_NUM_TESTS = 8

# If True, when a ticker has no report for the analyzed quarter,
# features are extrapolated using the average of the last N available quarters.
# This keeps the ticker in the test universe with an estimated snapshot.
ENABLE_FALLBACK_EXTRAPOLATION = True

# Number of prior quarters used for feature extrapolation when the exact report is missing.
# Requires at least this number of historical reports.
FALLBACK_LOOK_BACK_QUARTERS = 4

# Portfolio holding duration from entry date.
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

# ── Recency weighting ──────────────────────────────────────────────────────────
# If True, training observations are weighted exponentially so that recent
# quarters receive more weight than older ones.  This combats concept drift:
# factor premia and market dynamics from 4-5 years ago are less predictive
# of current quarters than the most recent 1-2 years of data.
ENABLE_RECENCY_WEIGHTING = True
# Half-life of the exponential decay, in years.
# A value of 2.0 means an observation from 2 years ago gets half the weight
# of a current observation. Tuned empirically for quarterly S&P 500 data:
# - Too short (< 1Y): ignores 80% of training data, high variance
# - Too long (> 4Y): essentially uniform weights, no recency benefit
TRAINING_RECENCY_HALFLIFE_YEARS = 2.0

# Temporal validation of base agents (in-fold, train-only) to detect
# persistence and trend in each agent's quality before scoring test data.
ENABLE_TEMPORAL_AGENT_VALIDATION = True
# Fraction of top-scored tickers per quarter used to evaluate each agent.
TEMPORAL_VALIDATION_TOP_PCT = 0.15
# Minimum top-k selected per quarter for temporal agent validation.
TEMPORAL_VALIDATION_MIN_TOP_K = 12
# Exponential half-life in quarters for temporal validation weights.
TEMPORAL_VALIDATION_HALFLIFE_QUARTERS = 6
# Relative contribution of performance trend to reliability multiplier.
TEMPORAL_VALIDATION_TREND_WEIGHT = 0.30
# Bounds for reliability multiplier applied to agent score dispersion.
TEMPORAL_VALIDATION_WEIGHT_CLIP_MIN = 0.75
TEMPORAL_VALIDATION_WEIGHT_CLIP_MAX = 1.25

# When a sector-specific model is degenerate (e.g. all feature importances are 0),
# use a conservative fallback score instead of a neutral 0.5 to avoid accidental
# promotion of low-confidence candidates.
DEGENERATE_MODEL_FALLBACK_SCORE = 0.25
# Conservative fallback score for long-oriented sector-specialized agents when a
# sector model cannot be trained or a fold fails. Kept aligned with the
# degenerate-model policy so missing/no-signal sectors are penalized consistently.
SECTOR_SPECIALIST_LONG_FALLBACK_SCORE = DEGENERATE_MODEL_FALLBACK_SCORE
# Global neutral score policy used in training-time score sanitation:
# exact neutral values (typically 0.5) can be pushed down so missing/unknown
# evidence does not accidentally rank above truly strong names.
NEUTRAL_SCORE_PENALTY_ENABLED = True
NEUTRAL_SCORE_VALUE = 0.5
NEUTRAL_SCORE_PENALIZED_VALUE = 0.25
NEUTRAL_SCORE_EPS = 1e-9
# Numerical threshold to consider feature importance mass as zero.
DEGENERATE_MODEL_IMPORTANCE_EPS = 1e-12

# Minimum score threshold to include a stock in the long portfolio / shortlist.
# 0.55 keeps only tickers with clear positive signal; with a score-weighted
# portfolio + min_stocks floor, it still guarantees a portfolio with few qualifiers.
# Updated after 2025Q1-2025Q4 parallel batch (20260429_212415):
# 7-3 portfolio with slightly higher score threshold delivered the best
# alpha, Sharpe and drawdown profile among tested configurations.
PORTFOLIO_MIN_SCORE = 0.57
# Max number of selected stocks per sector (0 disables sector cap).
# Prevents concentration in a single winning theme/regime.
PORTFOLIO_MAX_STOCKS_PER_SECTOR = 3
# Maximum portfolio weight per ticker (0 disables weight cap).
# Final weights are re-normalized after capping.
PORTFOLIO_MAX_STOCK_WEIGHT = 0.20

# -----------------------------------------------------------------------------
# Scoring robustness settings (sector + dispersion)
# -----------------------------------------------------------------------------

# Penalizes sectors with few peers: sector_confidence = min(1, sqrt(n_peers / k)).
SECTOR_CONFIDENCE_PEERS = 10

# Soft sector prior over final score (additive tilt model):
# final_score += (sector_score - 0.5) * SECTOR_SCORE_PRIOR_WEIGHT * sector_confidence
# A sector_score of 0.7 with full confidence adds +0.06 to each ticker in that sector.
# Reduced from 0.25 to 0.15 to limit sector-rotation noise in the final score.
SECTOR_SCORE_PRIOR_BASE = 0.5
SECTOR_SCORE_PRIOR_WEIGHT = 0.15

# If an agent score has low dispersion, it is shrunk toward 0.5.
# scale = min(1, std / SCORE_DISPERSION_MIN_STD)
# Increased from 0.03 to 0.05 to trigger shrinkage more aggressively for
# poorly-calibrated agents (e.g., sentiment when sparse data is available).
SCORE_DISPERSION_MIN_STD = 0.05
# Scale floor to avoid collapse to 0.5 when train std is close to 0.
# Applies only when shrink is active (scale<1), preserving some test signal.
SCORE_DISPERSION_MIN_SCALE = 0.35

# Price window used for technical features (RSI, momentum, volatility, etc.).
# Reduced from the historical value of 400 to preserve enough context
# without requiring unnecessary extra history.
TECHNICAL_LOOKBACK_DAYS = 300


# =============================================================================
# 6B. GARP / Value-Growth strategy profile
# =============================================================================

# New default research objective: learn 12M alpha/ranking for GARP/value-growth
# stock selection. TP/SL labels are still available as secondary research
# artefacts, but they are no longer the primary training label.
PRIMARY_STRATEGY_PROFILE = "garp_value_growth"
ENABLE_TP_SL_AS_SECONDARY_EVALUATION = True

# Composite score weights used for transparent reporting / fallback scoring.
GARP_SCORE_WEIGHTS = {
  "quality": 0.20,
  "growth": 0.20,
  "valuation": 0.25,
  "fundamental_trend": 0.15,
  "catalyst": 0.10,
  "technical_guardrail": 0.05,
  "risk_penalty": 0.20,
}

# Composite training label weights. Positive terms reward forward 12M alpha
# and robust value-growth fundamentals; negative terms penalize severe downside,
# value traps and expensive low-quality growth.
GARP_TARGET_WEIGHTS = {
  "spy_alpha": 0.30,
  "sector_alpha": 0.15,
  "future_fundamental_improvement": 0.20,
  "expectation_gap": 0.15,
  "initial_valuation_reasonableness": 0.10,
  "overexpectation_penalty": 0.05,
  "downside_penalty": 0.05,
}
GARP_OUTPERFORM_QUANTILE = 0.60
GARP_VALUE_TRAP_MAX_SCORE = 0.35
GARP_TECHNICAL_GUARDRAIL_WEIGHT = 0.05
REQUIRED_GARP_AGENTS = [
  "quality", "growth", "valuation", "fundamental_trend",
  "catalyst", "risk_bear", "technical_guardrail",
]
GARP_FORBIDDEN_FEATURE_PATTERNS = [
  "forward_", "future_", "target", "alpha_test", "return_12m_fwd",
  "tp_level", "sl_level", "tp_sl", "outcome",
]
GARP_CRITICAL_FEATURES = [
  "roic", "fcf_margin", "gross_margin", "revenue_yoy_growth", "fcf_yoy_growth",
  "fcf_yield", "earnings_yield", "ev_to_ebitda", "pe_ratio",
  "roic_trend_2y", "eps_growth_trend_3y", "debt_to_ebitda", "volatility_60d",
]
GARP_MIN_STOCKS = 5
GARP_MAX_STOCKS = 10
GARP_SECTOR_CAP = 3
RISK_BEAR_HARD_THRESHOLD = 0.90

# =============================================================================
# 7. Walk-forward backtesting
# =============================================================================

# Maximum walk-forward training window, in years.
# The pipeline will try this maximum and, if it does not meet minimum test coverage,
# will progressively reduce it down to WALKFORWARD_TRAIN_MIN_YEARS.
# Current default favors long-memory training (12Y -> 8Y fallback) because the
# latest comparative runs showed better alpha stability and Sharpe versus the
# shorter 6Y configuration.
WALKFORWARD_TRAIN_LOOKBACK_YEARS = 12
# Lower bound for dynamic walk-forward training window.
# 8Y preserves regime diversity while still keeping enough eligible tickers in
# recent folds under the minimum test coverage constraint.
WALKFORWARD_TRAIN_MIN_YEARS = 8

# Debug output profile:
# - "focused": keep only high-value artifacts for training/test and TP/SL debugging.
# - "full": keep every available artifact.
DEBUG_OUTPUT_PROFILE = "focused"

# Optional detailed artifacts (typically only useful with DEBUG_OUTPUT_PROFILE="full").
EXPORT_TP_SL_UNIVERSE_MATRIX = False
EXPORT_GLOBAL_TP_SL_UNIVERSE_MATRIX = False
EXPORT_SNAPSHOT_AGENT_AUDITS = False
EXPORT_ALL_FOLDS_SCORES = False
EXPORT_DETAILED_TRADES_REPORT = False

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

# Frequency for live thesis-managed portfolio reviews: M | 2M | Q.
PORTFOLIO_REVIEW_FREQUENCY = "M"

# Entrypoint selected when running analyzer.py directly:
#   "pipeline"             -> full data/dataset/backtest pipeline
#   "portfolio_review"     -> review current positions from an existing dataset
#   "portfolio_evolution"  -> simulate thesis-managed portfolio evolution
MAIN_ACTION = "pipeline"

# Lightweight portfolio-review inputs. These replace command-line flags.
# Use either PORTFOLIO_REVIEW_POSITIONS_CSV or PORTFOLIO_REVIEW_TICKERS.
PORTFOLIO_MASTER_DATASET_PATH = os.path.join(FINNHUB_DATA_DIR, "master_dataset.parquet")
PORTFOLIO_REVIEW_TICKERS = []
PORTFOLIO_REVIEW_POSITIONS_CSV = ""
PORTFOLIO_REVIEW_DATE = ANALYSIS_REFERENCE_DATE
PORTFOLIO_REVIEW_OUTPUT_DIR = os.path.join(RESULTS_DIR, "portfolio_review")
PORTFOLIO_EVOLUTION_OUTPUT_DIR = os.path.join(RESULTS_DIR, "portfolio_evolution")

# Rotation discipline for live portfolio evolution.
MIN_ROTATION_ADVANTAGE = 0.15
MIN_SCORE_ADVANTAGE_TO_REPLACE = 0.12
MIN_CONVICTION_ADVANTAGE = 10
MIN_OPPORTUNITY_COST_THRESHOLD = 0.15
HOLD_WINNER_BONUS = 0.05
THESIS_INTACT_HOLD_PREFERENCE = 0.08
PORTFOLIO_WEIGHTING_MODE = "equal"  # equal | conviction
# Static viewer can use local assets when available; CDN remains the default.
VIEWER_OFFLINE_MODE = False

# =============================================================================
# 8. Agent hyperparameters
# =============================================================================

# Old fundamental/momentum/bear agent feature sets removed: GARP domain feature sets below are authoritative.

# ── SentimentAgent (Random Forest) ───────────────────────────────────────────
# Enables/disables the standalone sentiment agent in the base stack.
# Empirically shown to hurt alpha: 10.44% without vs 9.94% with (8-scenario analysis Q3-Q4 2025).
# Disabled by default to improve signal quality.
ENABLE_SENTIMENT_AGENT = False
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
  "eps_surprise_avg_4q",

  # NLP sentiment (FinBERT when available, lexical fallback otherwise)
  "finbert_sentiment_polarity",
  "finbert_risk_intensity",
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

  # Removed: consistently absent in early folds; marginal importance when present
  "beat_rate_4q",

  # Removed: redundant with finbert_sentiment_polarity (positive-only vs positive−negative)
  "finbert_bullish_tone",

  # Removed: near-zero importance; partially overlaps finbert_risk_intensity
  "finbert_uncertainty_score",
]

# SectorRotationAgent
SECTOR_ROTATION_FEATURE_COLUMNS = [
  # =========================
  # 1. MOMENTUM (CORE DRIVER)
  # =========================
  "momentum_3m",
  "momentum_6m",
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
  # 5. SENTIMENT (ACCELERATOR)
  # =========================
  "eps_revision",

  # =========================
  # 6. REGIME / RISK CONTEXT
  # =========================
  "volatility_60d",
]

SECTOR_ROTATION_FEATURE_EXCLUDE = [
  # Not available in the current analysis window (requires 252+ trading days)
  "momentum_12m",

  # Debt/coverage metrics have no sector-rotation importance after aggregation;
  # captured by bear_score in the meta-learner
  "debt_to_ebitda",
  "interest_coverage",

  # Analyst dispersion and consensus-change lose signal when aggregated to sector level
  "analyst_consensus_change",
  "analyst_dispersion",
]


# ── GARP / Value-Growth agent feature universes ──────────────────────────────
QUALITY_FEATURE_COLUMNS = [
  "gross_margin", "operating_margin", "net_margin", "ebitda_margin", "fcf_margin",
  "roic", "roe", "roa", "asset_turnover", "current_ratio", "debt_equity",
  "debt_to_ebitda", "interest_coverage", "piotroski_fscore", "accruals_ratio",
  "earnings_quality", "cash_conversion", "dilution_yoy", "moat_proxy_score",
]
QUALITY_FEATURE_EXCLUDE = []

GROWTH_FEATURE_COLUMNS = [
  "revenue_yoy_growth", "eps_yoy_growth", "fcf_yoy_growth", "gross_profit_yoy_growth",
  "operating_income_yoy_growth", "revenue_growth_3y", "eps_growth_trend_3y",
  "revenue_growth_acceleration", "fcf_growth_acceleration", "market_growth_proxy",
  "analyst_consensus_change", "eps_revision",
]
GROWTH_FEATURE_EXCLUDE = []

GARP_VALUATION_FEATURE_COLUMNS = [
  "pe_ratio", "forward_pe", "peg_ratio", "ps_ratio", "ev_to_ebitda", "ev_to_sales",
  "ev_sales_to_gross_margin", "ev_sales_to_operating_margin", "pb_ratio", "p_fcf",
  "fcf_yield", "earnings_yield", "pe_vs_5y_median", "ev_ebitda_vs_5y_median",
  "valuation_percentile_sector", "valuation_percentile_universe", "valuation_percentile_history",
  "fcf_yield_rank_sector", "ev_ebitda_rank_sector", "quality_x_value_universe",
  "expectation_gap_score", "overexpectation_penalty", "valuation_to_growth_reasonableness",
]
GARP_VALUATION_FEATURE_EXCLUDE = []

FUNDAMENTAL_TREND_FEATURE_COLUMNS = [
  "roic_trend_2y", "roe_trend_2y", "net_margin_trend_2y", "gross_margin_trend_2y",
  "operating_margin_trend_2y", "fcf_margin_trend_2y", "leverage_trend",
  "revenue_growth_acceleration", "eps_growth_trend_3y", "fcf_yoy_growth",
  "total_debt_yoy_growth", "current_ratio_change_yoy", "gross_margin_change_yoy",
  "roa_change_yoy", "eps_revision",
]
FUNDAMENTAL_TREND_FEATURE_EXCLUDE = []

CATALYST_FEATURE_COLUMNS = [
  "eps_revision", "analyst_consensus_change", "eps_surprise_pct", "eps_surprise_avg_4q",
  "beat_rate_4q", "insider_net_ratio_90d", "mspr_3m", "mspr_trend",
  "buyback_yield", "sector_score", "sp500_momentum_3m", "sector_momentum",
  "finbert_sentiment_polarity", "expectation_gap_score", "mispricing_quality_growth",
  "revenue_growth_acceleration", "fcf_growth_acceleration", "fcf_margin_trend_2y",
]
CATALYST_FEATURE_EXCLUDE = []

RISK_BEAR_FEATURE_COLUMNS = [
  "debt_equity", "debt_to_ebitda", "interest_coverage", "current_ratio",
  "fcf_margin", "fcf_yoy_growth", "consecutive_losses", "revenue_decline",
  "total_debt_yoy_growth", "insider_sell_ratio", "insider_net_ratio_90d",
  "volatility_60d", "price_vs_52w_high", "momentum_6m", "max_drawdown_252d",
  "dilution_yoy", "accruals_ratio",
]
RISK_BEAR_FEATURE_EXCLUDE = []

TECHNICAL_GUARDRAIL_FEATURE_COLUMNS = [
  "momentum_3m", "momentum_6m", "momentum_12m", "price_vs_52w_high",
  "price_vs_52w_low", "volatility_60d", "rsi_14", "rsi_28", "sma_200",
  "distance_to_200dma", "max_drawdown_252d", "vol_ratio_20_50",
]
TECHNICAL_GUARDRAIL_FEATURE_EXCLUDE = []

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
  # Base agent scores (stacking layer): GARP/value-growth agents.
  "quality_score",
  "growth_score",
  "valuation_score",
  "fundamental_trend_score",
  "catalyst_score",
  "risk_bear_score",
  "technical_guardrail_score",
  "sentiment_score",
  "sector_score",
  "regime_adjusted_score",
  "rules_consensus_signal",
  "rules_consensus_confidence",
  "agent_score_mean",
  "agent_score_std",
  "agent_disagreement",
  "bullish_agents",
  "bearish_agents",
  "agent_contradiction_flag",

  # Sector-relative percentile ranks (3 non-redundant angles)
  "fcf_yield_rank_sector",    # cheapness vs sector (robust to earnings distortion)
  "roic_rank_sector",         # capital efficiency vs sector
  "ev_ebitda_rank_sector",    # enterprise value vs sector

  # Universe-wide percentile ranks
  "quality_rank_universe",       # ROIC vs all tickers
  "value_rank_universe",         # earnings_yield vs all tickers
  "piotroski_rank_universe",     # financial health score vs all tickers
  "eps_revision_rank_universe",  # analyst revision momentum (proven alpha)
  "beat_rate_rank_universe",     # consistent earnings beaters

  # Momentum consistency: fraction of 1m/3m/6m windows that are positive.
  # Directional agreement across horizons (momentum_12m excluded — unavailable).
  "momentum_consistency",

  # Interaction features (bounded, robust combinations)
  "value_x_momentum",         # cheap + momentum = value-momentum blend
  "quality_x_lowvol",         # high ROIC + low volatility = quality/defensive
  "quality_x_value_universe", # rank × rank composite (double-confirmation)

  # Macro regime context
  "vix",
  "yield_curve",
  "sp500_momentum_3m",
  "sp500_momentum_12m",
]
META_FEATURE_EXCLUDE = [
  # Removed sector ranks: redundant or noisy after aggregation
  "pe_rank_sector",           # redundant with ev_ebitda_rank_sector + fcf_yield_rank_sector
  "pb_rank_sector",           # too sector-specific (e.g., banks); distorts cross-sector ranking
  "roe_pct_sector",           # ROE is leverage-influenced; replaced by roic_rank_sector
  "momentum_pct_sector",      # redundant with universe-level momentum ranks
  "debt_rank_sector",         # captured by bear_score; redundant

  # Removed: always NaN in current analysis window (momentum_12m unavailable)
  "momentum_12m_rank_universe",
  "vol_adj_momentum_12m_rank",
  "momentum_quality_signal",

  # Removed: second-derivative signals — noisy with sparse quarterly data
  "revenue_growth_acceleration",
  "eps_surprise_acceleration",
  "quality_acceleration_rank",

  # Removed: raw volatility-adjusted values (not cross-sectionally ranked;
  # near-zero for sentiment-based version; redundant with agent scores)
  "momentum_vol_adj",
  "value_vol_adj",
  "quality_vol_adj",
  "sentiment_vol_adj",

  # Removed: finbert_sentiment_polarity ≈ 0 makes this interaction near-zero
  "sentiment_x_earnings_surprise",
]
# Base score columns on which meta computes consensus/interactions.
META_AGENT_SCORE_COLUMNS = [
  "quality_score",
  "growth_score",
  "valuation_score",
  "fundamental_trend_score",
  "catalyst_score",
  "risk_bear_score",
  "technical_guardrail_score",
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
ALPHA_META_REG_MAX_DEPTH     = 4
ALPHA_META_REG_LEARNING_RATE = 0.03
ALPHA_META_REG_SUBSAMPLE     = 0.80
ALPHA_META_REG_COLSAMPLE     = 0.7
ALPHA_META_REG_L2_REG        = 2.0
ALPHA_META_REG_L1_REG        = 0.5
ALPHA_META_REG_MIN_CHILD     = 10

ALPHA_META_RANK_N_ESTIMATORS  = 300
ALPHA_META_RANK_MAX_DEPTH     = 3
ALPHA_META_RANK_LEARNING_RATE = 0.05
ALPHA_META_RANK_SUBSAMPLE     = 0.85
ALPHA_META_RANK_COLSAMPLE     = 0.7
ALPHA_META_RANK_L2_REG        = 2.0
ALPHA_META_RANK_L1_REG        = 0.5
ALPHA_META_RANK_MIN_CHILD     = 10

ALPHA_META_RISK_N_ESTIMATORS  = 250
ALPHA_META_RISK_MAX_DEPTH     = 3
ALPHA_META_RISK_LEARNING_RATE = 0.05
ALPHA_META_RISK_SUBSAMPLE     = 0.85
ALPHA_META_RISK_COLSAMPLE     = 0.7
ALPHA_META_RISK_L2_REG        = 2.0
ALPHA_META_RISK_L1_REG        = 0.5
ALPHA_META_RISK_MIN_CHILD     = 10

# Blend weight: regime_adjusted_score = ALPHA_META_RANK_BLEND * ranking_score
#               + (1 - ALPHA_META_RANK_BLEND) * regime_adj
# Increased from 0.65 to 0.70 to give the pairwise ranking model more influence,
# since cross-sectional ordering is what drives portfolio construction quality.
ALPHA_META_RANK_BLEND = 0.70
# Additional risk-aware blend applied after ranking/regime blend:
# final_score = (1 - ALPHA_META_RISK_BLEND) * regime_rank_blend
#               + ALPHA_META_RISK_BLEND * (1 - risk_score)
# Reduced from 0.20 to 0.15 to dampen the anti-momentum bias introduced by the
# risk classifier when markets trend strongly upward.
# Trade-off: this slightly reduces the portfolio's downside protection from the
# risk model in bear markets, but that drawdown is considered acceptable given
# that the risk classifier itself has limited predictive AUC (~0.5) on this
# dataset.  If the risk model's out-of-sample AUC improves significantly in
# future runs, this value should be revisited upward.
ALPHA_META_RISK_BLEND = 0.15

# ── MomentumAgent TFT-lite blend weight ──────────────────────────────────────
# Final score = (1 - MOMENTUM_DEEP_BLEND_WEIGHT) * RF_score
#               + MOMENTUM_DEEP_BLEND_WEIGHT * TFT_score
MOMENTUM_DEEP_BLEND_WEIGHT = 0.35

# ── Agent rule learning (explicit pattern mining) ───────────────────────────
# When enabled, each base agent mines historical TP/SL rules from its own
# feature universe (single-feature ranges and feature-pair interactions).
# The resulting rule signal is blended with model probabilities at inference.
ENABLE_AGENT_RULE_ENGINE = True

# Number of quantile bins used to discretize each metric for rule discovery.
AGENT_RULE_N_BINS = 5

# Minimum samples required for a rule candidate to be considered valid.
AGENT_RULE_MIN_SAMPLES = 45

# Minimum absolute uplift vs global TP rate required to keep a rule.
AGENT_RULE_MIN_EDGE = 0.035

# Minimum temporal stability required to keep a mined rule.
# Stability is estimated from quarter-to-quarter variance of rule hit-rate.
AGENT_RULE_MIN_STABILITY = 0.52

# Maximum number of strongest rules kept per agent and fold.
AGENT_RULE_MAX_RULES = 40

# Maximum number of top features considered for pairwise interaction rules.
AGENT_RULE_MAX_PAIR_FEATURES = 7

# Blend intensity for rule signal over model probability.
# final = clip(model_proba + AGENT_RULE_BLEND * rule_signal, 0, 1)
AGENT_RULE_BLEND = 0.22

# Minimum number of training samples a sector must have to receive its own
# independent rule set.  Sectors below this threshold are pooled together
# under an "_other" group so their data is not wasted.
AGENT_RULE_SECTOR_MIN_SAMPLES = 80

# OOF rule-quality gate: downweights per-agent rule telemetry when the
# out-of-fold signal quality is weak or inverted.
RULE_QUALITY_GATE_ENABLED = True

# Minimum OOF quality required to keep a non-zero rule contribution.
RULE_QUALITY_MIN_IC = 0.015
RULE_QUALITY_MIN_SPREAD = 0.0
RULE_QUALITY_MIN_STABILITY = 0.45

# Reference scales used to map good rule quality toward multiplier ~= 1.0.
RULE_QUALITY_REF_IC = 0.08
RULE_QUALITY_REF_SPREAD = 0.10



# =============================================================================
# 9. Reproducibility
# =============================================================================

RANDOM_SEED = 42

# =============================================================================
# 10. Secondary exit diagnostics (not a training objective)
# =============================================================================

# --- Signal generation (take-profit / stop-loss) ----------------------------

# Baseline take-profit percentage applied when agent score == 0.5
# Raised to 15% for 12-month holding: 8% was calibrated for quarterly windows.
TP_SL_BASE_TP = 0.15       # 15 %

# Baseline stop-loss percentage applied when agent score == 0.5
# Raised to 10% for 12-month holding.
TP_SL_BASE_SL = 0.10       # 10 %

# Maximum shift in TP as score moves from 0.5 → 1.0 (or 0.5 → 0.0)
TP_SL_TP_SENSITIVITY = 0.10

# Maximum shift in SL as score moves from 0.5 → 1.0 (or 0.5 → 0.0)
TP_SL_SL_SENSITIVITY = 0.04

# Hard bounds on TP and SL percentages — calibrated for 12-month holding.
# S&P500 stocks can gain 40-80% in a strong year and drop 25-40% in severe corrections.
# Previous MAX_TP=30% was cutting off the natural upside that drives most alpha.
TP_SL_MIN_TP = 0.08
TP_SL_MAX_TP = 0.55
TP_SL_MIN_SL = 0.08
TP_SL_MAX_SL = 0.35

# --- Confidence model -------------------------------------------------------

# Relative weight of the raw model score vs. historical calibration
TP_SL_CONFIDENCE_SCORE_WEIGHT = 0.50
TP_SL_CONFIDENCE_CALIBRATION_WEIGHT = 0.50

# --- Portfolio construction -------------------------------------------------

# Minimum stocks required to invest (0 candidates → no investment)

# Maximum portfolio size

# Maximum stocks from the same GICS sector in the portfolio

# Minimum expected value (EV) for a stock to be eligible
# EV = confidence × tp_pct − (1 − confidence) × sl_pct
TP_SL_EV_THRESHOLD = 0.0

# --- Backtesting engine -----------------------------------------------------

# Maximum calendar days to hold a position before forcing closure ("NONE")
# 365 days matches the full 12-month holding period window.
TP_SL_MAX_HOLDING_DAYS = 365


# --- TP/SL vs Buy & Hold counterfactual evaluation --------------------------

# If True, evaluate a pure 12M Buy & Hold counterfactual on the exact same
# tickers, entry date and initial weights selected for the TP/SL strategy.
ENABLE_BUY_HOLD_COUNTERFACTUAL = True

# If True, close Buy & Hold positions at the last available close when a ticker
# does not have data through the target 12M exit date. This is evaluation-only
# and does not affect portfolio selection.
BUY_HOLD_EXIT_ON_LAST_AVAILABLE_PRICE = True

# If True, export fold/ticker CSVs, JSON summary and comparison charts under
# results/strategy (or the run-specific strategy directory).
EXPORT_TP_SL_VS_BUY_HOLD = True

# Research switch for future controlled TP/SL variants. The default mode is
# "base" so the production strategy remains unchanged unless explicitly
# overridden.
ENABLE_TP_SL_RESEARCH_VARIANTS = False
TP_SL_VARIANT_MODE = "base"  # secondary diagnostic only; not a training profile
TP_SL_HYBRID_MIN_TRAIN_PATHS = 6
TP_SL_HYBRID_TRAILING_MIN_PCT = 0.06
TP_SL_HYBRID_TRAILING_MAX_PCT = 0.28
TP_SL_HYBRID_PROFIT_REVIEW_DAYS = 10

# --- Agent weight tracker ---------------------------------------------------

# EWMA decay factor for historical hit-rate tracking (higher = slower to forget)
TP_SL_WEIGHT_DECAY = 0.85

# Prior hit-rate before any fold data are available (maximum uncertainty = 0.5)
TP_SL_WEIGHT_PRIOR = 0.50

# Floor weight so every agent stays active (prevents weight collapse)
TP_SL_WEIGHT_MIN = 0.05

# --- TP/SL edge overlay (fold-level, no leakage) ----------------------------

# If True, adjusts test-fold confidence using each ticker's historical
# TP-before-SL behavior observed only in the train window of that fold.
TP_EDGE_ENABLE = True

# Bayesian prior strength used to estimate ticker TP probability.
# Higher values mean stronger pull toward neutral 0.5 when few observations.
TP_EDGE_PRIOR_STRENGTH = 10.0

# Reliability shrink factor by number of historical observations.
# reliability = sqrt(n / (n + TP_EDGE_RELIABILITY_K))
TP_EDGE_RELIABILITY_K = 12.0

# Outcome score for NONE when estimating TP edge.
# TP=1.0, SL=0.0, NONE=TP_EDGE_NONE_SCORE
TP_EDGE_NONE_SCORE = 0.40

# Confidence overlay intensity applied to the model score.
# adjusted_conf = clip(conf + TP_EDGE_CONFIDENCE_BLEND * historical_tp_edge)
# [Batch 20260430] Lowered from 0.25 → 0.10 (winner in round-3 sensitivity sweep, +5.7 bps).
TP_EDGE_CONFIDENCE_BLEND = 0.10

# Penalization for overly ambitious TP levels vs historical TP reach.
# feasibility = clip(1 - penalty * max(tp/stretch_ref - 1, 0), min, 1)
TP_EDGE_TP_STRETCH_PENALTY = 0.60

# Minimum feasibility floor for TP stretch penalization.
TP_EDGE_MIN_FEASIBILITY = 0.35

# Rule signal contribution in ticker-level strategy ranking.
# risk_benefit_score is multiplied by (1 + weight * rules_consensus_signal).
TP_SL_RULE_SIGNAL_RBS_WEIGHT = 0.25

# Minimum holding fraction before TP/SL logic activates.
# For a 12-month holding period, 0.25 → first 3 months are a grace period
# where neither TP nor SL can trigger.  Prevents noise-driven early exits
# caused by normal short-term volatility right after entry.
TP_SL_GRACE_PERIOD_FRACTION = 0.5

# --- Trailing stop (TP-converted-SL) ----------------------------------------

# Once the price crosses the TP level, the position is NOT closed.
# Instead, the TP becomes a hard floor for a trailing stop that ratchets up
# each TP_SL_TRAILING_REVIEW_DAYS calendar days based on the running peak.
# The trailing stop never moves down; it exits with outcome "TP" when hit.
#
# Review interval in calendar days (default 30 = monthly).
TP_SL_TRAILING_REVIEW_DAYS = 30

# Quantile of the rolling-peak drawdown distribution used to set the trailing
# distance per ticker.  Higher quantile = wider trailing stop = more tolerance.
# 0.65: the trailing stop is set at the 65th percentile of historical pullbacks
# from rolling peak, so the stock would need an unusually large pullback
# (bigger than 65% of historical ones) to trigger the stop prematurely.
TP_SL_TRAILING_DRAWDOWN_QUANTILE = 0.65

# Fold-level TP/SL strategy fine-tuning loop. If the best base strategy has
# weak utility/hit-rate, evaluator tries relaxed variants automatically.
TP_SL_ENABLE_STRATEGY_FINE_TUNING = True
TP_SL_FINE_TUNE_MAX_RELAX_STEPS = 2
TP_SL_FINE_TUNE_MIN_HIT_RATE = 0.48
TP_SL_FINE_TUNE_MIN_UTILITY = 0.0

# Final ticker selection preferences: prioritize realizable TP with high
# confidence over extreme but unlikely targets.
TP_SL_MIN_ACCEPTABLE_TP = 0.12
TP_SL_SELECTION_CERTAINTY_WEIGHT = 0.35
TP_SL_SELECTION_TP_QUALITY_WEIGHT = 0.25


# Apply optional runtime overrides after all defaults are declared.
_apply_environment_overrides(globals())

