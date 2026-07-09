"""Central project configuration.

Run with:
    python main.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_JSON_DIR = RAW_DIR / "json"
PROCESSED_DIR = DATA_DIR / "processed"
MASTER_DIR = DATA_DIR / "master"
RESULTS_DIR = PROJECT_ROOT / "results"


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

# Editable project configuration. Keep .env only for API keys/secrets.
RUN_MODE = "full"
DATA_START_DATE = "2000-01-01"
PORTFOLIO_START_DATE = "2018-02-15"
PORTFOLIO_END_DATE = "2026-06-15"
PORTFOLIO_REVIEW_FREQUENCY = "M"
FUNDAMENTAL_REVIEW_FREQUENCY = "Q"
PRICE_UPDATE_FREQUENCY = "M"

DEV_MODE = False
DEV_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
BENCHMARK_TICKER = "SPY"
FORCE_RAW_DOWNLOAD = False

MIN_PORTFOLIO_SIZE = 5
MAX_PORTFOLIO_SIZE = 10
MIN_ROTATION_ADVANTAGE = 0.08
MIN_SCORE_ADVANTAGE_TO_REPLACE = 0.06
MIN_CONVICTION_ADVANTAGE = 0.05
MIN_OPPORTUNITY_COST_THRESHOLD = 0.07
WALK_FORWARD_SCORING = True
WALK_FORWARD_LABEL_HORIZON_MONTHS = 12
MIN_WALK_FORWARD_TRAINING_ROWS = 120
MIN_WALK_FORWARD_TRAINING_YEARS = 4
MAX_WALK_FORWARD_TRAINING_YEARS = 8
# Train-until-cutoff-then-freeze: the walk-forward loop only reruns/relearns (agents + meta-agent
# weights) on TRAIN_CUTOFF_DATE and earlier, at WALK_FORWARD_TRAIN_FREQUENCY cadence, starting
# WALK_FORWARD_TRAIN_YEARS before the cutoff. From the cutoff onward, the last trained model/weights
# are FROZEN and only used to predict (no further learning) — the portfolio is still reviewed
# monthly (PORTFOLIO_REVIEW_FREQUENCY) using that frozen model. This replaces the previous "pure"
# walk-forward that kept relearning every month all the way through the live-portfolio window.
TRAIN_CUTOFF_DATE = PORTFOLIO_START_DATE
WALK_FORWARD_TRAIN_YEARS = 8
WALK_FORWARD_TRAIN_FREQUENCY = "Q"
TRANSACTION_COST_BPS = 5.0
SLIPPAGE_BPS = 10.0
# Fundamentals carry a PERIOD-END date but are reported to the market weeks later. A fundamental is
# only treated as observable this many weeks after its period end, removing a subtle lookahead.
FUNDAMENTAL_PUBLICATION_LAG_WEEKS = 7

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-5.5"
ENABLE_OPENAI_RESEARCH = False

# Methodological limitation (survivorship bias): this is a static, present-day large-cap roster
# applied retroactively back to DATA_START_DATE. It excludes names delisted/acquired/dropped from
# the index over that period and includes recent IPOs/spin-offs that did not exist historically.
# The live portfolio only trades within PORTFOLIO_START_DATE/PORTFOLIO_END_DATE, which bounds but
# does not eliminate the bias; the ML training window (MAX_WALK_FORWARD_TRAINING_YEARS of lookback)
# inherits it too. See CLAUDE.md "Methodological limitations" and report.py's conclusions section.
TICKERS = [
    "NVDA","AAPL","MSFT","AMZN","GOOGL","GOOG","META","AVGO","TSLA","BRK-B","WMT","LLY","JPM","XOM","V","JNJ","MU","COST","MA","ORCL","NFLX","CVX","ABBV","PLTR","PG","BAC","HD","KO","AMD","CAT","GE","CSCO","MRK","LRCX","AMAT","RTX","PM","UNH","MS","GS","IBM","WFC","GEV","TMUS","LIN","MCD","INTC","PEP","VZ","AXP","KLAC","T","NEE","C","AMGN","ABT","CRM","DIS","GILD","TXN","TMO","ANET","TJX","ISRG","SCHW","BA","UBER","APH","DE","PFE","COP","BLK","ADI","LMT","APP","HON","WELL","UNP","QCOM","BKNG","ETN","PANW","DHR","SYK","LOW","CB","SPGI","INTU","PLD","ACN","BMY","NOW","PGR","PH","VRTX","CEG","MCK","MDT","COF","HCA","CME","CRWD","GLW","MO","NEM","SO","SBUX","BSX","SNDK","CMCSA","NOC","DUK","WDC","ADBE","DELL","HWM","EQIX","GD","WM","TT","CVS","STX","WMB","ICE","BX","MAR","PWR","ADP","AMT","MRSH","JCI","UPS","FDX","SNPS","PNC","USB","KKR","CDNS","REGN","BK","NKE","ABNB","MCO","SHW","MSI","FCX","MMM","ITW","CTAS","CMI","ECL","EOG","ORLY","CSX","MNST","RCL","EMR","KMI","MDLZ","VLO","DASH","AEP","CL","CI","MPC","PSX","TDG","RSG","LHX","SLB","HLT","AON","WBD","ROST","HOOD","CRH","GM","ELV","TRV","APO","NSC","COR","APD","FTNT","SRE","SPG","DLR","PCAR","O","OXY","TEL","BKR","VST","AFL","AZO","TFC","D","OKE","CIEN","FANG","AJG","CTVA","COIN","ALL","MPWR","ADSK","TGT","FAST","EXC","TRGP","EA","CAH","XEL","FIX","ZTS","GWW","PSA","AME","KEYS","NXPI","NDAQ","CARR","EW","ETR","F","DDOG","TER","URI","IDXX","BDX","KR","MET","GRMN","YUM","HSY","PEG","CMG","CVNA","DAL","EBAY","ED","AXON","PYPL","MSCI","VTR","WAB","EQT","PCG","AMP","DHI","ROK","AIG","CBRE","FITB","SYY","ODFL","TTWO","WEC","LYV","CCI","TPL","NUE","KDP","HIG","ROP","LVS","MCHP","WDAY","XYZ","MLM","ADM","VMC","NRG","STT","CCL","KVUE","RMD","KMB","EME","ACGL","PAYX","PRU","IR","GEHC","CPRT","A","IRM","EL","ATO","OTIS","AEE","HAL","HBAN","FISV","IBKR","CBOE","DTE","DVN","UAL","VICI","TDY","WAT","FE","MTB","XYL","EXPE","CTSH","EXR","PPL","DOV","HPE","FICO","CNP","TPR","RJF","EIX","VRSK","DG","ES","IQV","WTW","JBL","DOW","AWK","BIIB","CHTR","STZ","KHC","DXCM","ROL","CTRA","EXE","FIS","HUBB","WRB","NTRS","CINF","LYB","STLD","TSCO","CFG","ARES","MTD","BG","Q","LEN","CMS","ON","OMC","AVB","DRI","ULTA","PPG","BRO","CHD","SYF","EQR","PHM","NI","VLTO","EFX","WSM","VRSN","LH","RF","L","DGX","TSN","DLTR","STE","FSLR","LDOS","RL","KEY","MRNA","BR","HUM","CHRW","CF","GIS","SW","NTAP","GPN","LUV","CPAY","LULU","EXPD","TROW","ALB","EVRG","IP","SBAC","PFG","SNA","PKG","INCY","LNT","JBHT","AMCR","SMCI","CSGP","DD","NVR","IFF","PTC","CNC","ZBH","WST","WY","FTV","HOLX","HPQ","LII","HII","PODD","BALL","FFIV","ESS","TXT","VTRS","AKAM","TKO","TRMB","KIM","J","INVH","CDW","MAA","APTV","NDSN","MKC","TYL","DECK","PNR","IEX","GPC","REG","COO","BBY","CLX","HST","APA","ALGN","HAS","EG","DPZ","AVY","ERIE","HRL","GEN","BEN","ALLE","MAS","DOC","PNW","JKHY","GNRC","SOLV","FOX","UHS","UDR","FOXA","IT","TTD","GDDY","SWK","SJM","GL","WYNN","AIZ","BF-B","IVZ","CPT","ZBRA","PSKY","AES","DVA","BLDR","RVTY","MGM","FRT","MOS","NCLH","AOS","NWSA","BAX","HSIC","ARE","BXP","SWKS","TECH","TAP","CRL","FDS","MOH","POOL","CAG","EPAM","MTCH","PAYC","CPB","LW","NWS"
]


@dataclass(frozen=True)
class Settings:
    run_mode: str = RUN_MODE
    data_start_date: str = DATA_START_DATE
    start_date: str = PORTFOLIO_START_DATE
    end_date: str = PORTFOLIO_END_DATE
    review_frequency: str = PORTFOLIO_REVIEW_FREQUENCY
    fundamental_review_frequency: str = FUNDAMENTAL_REVIEW_FREQUENCY
    price_update_frequency: str = PRICE_UPDATE_FREQUENCY
    walk_forward_scoring: bool = WALK_FORWARD_SCORING
    walk_forward_label_horizon_months: int = WALK_FORWARD_LABEL_HORIZON_MONTHS
    min_walk_forward_training_rows: int = MIN_WALK_FORWARD_TRAINING_ROWS
    min_walk_forward_training_years: int = MIN_WALK_FORWARD_TRAINING_YEARS
    max_walk_forward_training_years: int = MAX_WALK_FORWARD_TRAINING_YEARS
    train_cutoff_date: str = TRAIN_CUTOFF_DATE
    walk_forward_train_years: int = WALK_FORWARD_TRAIN_YEARS
    walk_forward_train_frequency: str = WALK_FORWARD_TRAIN_FREQUENCY
    transaction_cost_bps: float = TRANSACTION_COST_BPS
    slippage_bps: float = SLIPPAGE_BPS
    fundamental_publication_lag_weeks: int = FUNDAMENTAL_PUBLICATION_LAG_WEEKS
    dev_mode: bool = DEV_MODE
    benchmark_ticker: str = BENCHMARK_TICKER

    @property
    def tickers(self) -> list[str]:
        base = DEV_TICKERS if self.dev_mode else TICKERS
        return list(dict.fromkeys([*base, self.benchmark_ticker]))

    @property
    def investable_tickers(self) -> list[str]:
        return [ticker for ticker in self.tickers if ticker != self.benchmark_ticker]

    @property
    def run_name(self) -> str:
        scope = "dev" if self.dev_mode else "full"
        return f"{scope}_{self.start_date}_{self.end_date}_{self.review_frequency}_cutoff{self.train_cutoff_date}"

    @property
    def run_dir(self) -> Path:
        return RESULTS_DIR / self.run_name


def ensure_directories() -> None:
    for path in (RAW_DIR, RAW_JSON_DIR, PROCESSED_DIR, MASTER_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
