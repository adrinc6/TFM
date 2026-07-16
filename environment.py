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
# RUN_MODE controla qué etapa(s) ejecuta main.py: una etapa suelta, "full" (el pipeline entero) o
# "experiments" (barre escenarios con module/experiments; no ejecuta el pipeline normal).
RUN_MODE = "experiments"
# Con RUN_MODE="experiments", qué escenarios corre main.py: la ruta de un fichero de escenarios
# (p.ej. "experiments/escenarios_aprendizaje.py") o el literal "todos" para juntar los cuatro bloques
# (aprendizaje + estabilidad + utilidad + pesos_meta) en un solo barrido.
EXPERIMENTS_FILE = "todos"
# Con RUN_MODE="experiments": si True, reanuda el último barrido en su carpeta y SALTA los escenarios
# ya completados (los que dejaron su fila persistida), corriendo solo los que faltan y reescribiendo
# la comparación con todos. Ponlo a False para forzar un barrido limpio en carpeta nueva.
EXPERIMENTS_RESUME = False
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
# Rotation / replacement thresholds, all consumed in module/strategy/portfolio.py. These are the
# A/B knob for how aggressively the book rotates: editing them here is the only place a value lives,
# there is no hardcoded duplicate in the strategy code. Raising them (or the minimum holding period)
# lowers turnover.
MIN_ROTATION_ADVANTAGE = 0.10          # primary: manager_score edge a challenger needs to replace a holding
MIN_SCORE_ADVANTAGE_TO_REPLACE = 0.06  # secondary path: a smaller score edge is enough only if conviction is also better
MIN_CONVICTION_ADVANTAGE = 0.05        # secondary path: conviction edge required alongside the smaller score edge
MIN_OPPORTUNITY_COST_THRESHOLD = 0.08  # soft sell: opportunity_cost_score above this = capital is better used elsewhere
WALK_FORWARD_SCORING = True
WALK_FORWARD_LABEL_HORIZON_MONTHS = 12
MIN_WALK_FORWARD_TRAINING_ROWS = 120
MIN_WALK_FORWARD_TRAINING_YEARS = 4
# MAX_WALK_FORWARD_TRAINING_YEARS is the actual trailing-history window used at every relearning
# point (module/ml.py's `max_history`) — a shorter window adapts faster to regime change (e.g. the
# 2023-2025 AI/megacap rally looking nothing like 2010-2018) at the cost of less data per fit.
MAX_WALK_FORWARD_TRAINING_YEARS = 4
# Pure rolling walk-forward, no freeze: the walk-forward loop reruns/relearns (agents + meta-agent
# weights) every WALK_FORWARD_TRAIN_FREQUENCY across the ENTIRE evaluated window, from
# TRAIN_CUTOFF_DATE through PORTFOLIO_END_DATE — never freezing the model at a fixed point. Each
# relearning step uses only the trailing MAX_WALK_FORWARD_TRAINING_YEARS of history available as of
# that date (no lookahead), so the live portfolio is always scored by a model trained on recent
# data, not one trained once in 2010-2018 and deployed unchanged for 8 years afterward — the
# earlier train-until-cutoff-then-freeze scheme was replaced because it produced a near-zero/
# negative out-of-sample rank-IC in the later years of the evaluated window (the frozen model never
# saw the market regime it was being scored against).
TRAIN_CUTOFF_DATE = PORTFOLIO_START_DATE
WALK_FORWARD_TRAIN_FREQUENCY = "Q"
TRANSACTION_COST_BPS = 5.0
SLIPPAGE_BPS = 10.0

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

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
    walk_forward_train_frequency: str = WALK_FORWARD_TRAIN_FREQUENCY
    transaction_cost_bps: float = TRANSACTION_COST_BPS
    slippage_bps: float = SLIPPAGE_BPS
    dev_mode: bool = DEV_MODE
    benchmark_ticker: str = BENCHMARK_TICKER
    experiments_file: str = EXPERIMENTS_FILE
    experiments_resume: bool = EXPERIMENTS_RESUME
    # Cuando el runner de experimentos ejecuta un escenario aislado, fija aquí la carpeta destino
    # (results/experiments/<exp_id>/<escenario>/) para que sus artefactos no colisionen con los de
    # otro escenario que comparta fechas/frecuencia. En una ejecución normal queda None y run_dir
    # deriva del run_name clásico, sin cambio de comportamiento.
    run_dir_override: str | None = None

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
        if self.run_dir_override:
            return Path(self.run_dir_override)
        return RESULTS_DIR / self.run_name


def ensure_directories() -> None:
    for path in (RAW_DIR, RAW_JSON_DIR, PROCESSED_DIR, MASTER_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
