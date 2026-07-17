"""Central project configuration.

Run with:
    python main.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


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
# (p.ej. "experiments/rejilla.py") o el literal "todos", que resuelve al generador de rejilla
# (barrido sistemático: ventana × cadencia + horizonte + ablations de pesos/meta).
EXPERIMENTS_FILE = "todos"
# Con RUN_MODE="experiments": si True, reanuda el último barrido en su carpeta y SALTA los escenarios
# ya completados (los que dejaron su fila persistida), corriendo solo los que faltan y reescribiendo
# la comparación con todos. Ponlo a False para forzar un barrido limpio en carpeta nueva.
EXPERIMENTS_RESUME = False
DATA_START_DATE = "2000-01-01"

# Punto de arranque de la simulación, expresado como TRIMESTRE (auto-documentado): "2010Q1" = primer
# día de ese trimestre (Q1->1-ene, Q2->1-abr, Q3->1-jul, Q4->1-oct). Sobre ese día se suma el retardo
# de publicación de fundamentales para obtener la FECHA ANCLA desde la que se entrena y evalúa. La
# idea: al arrancar en 2010Q1 + 45 días = 15-feb-2010, ya están publicados los resultados del
# trimestre anterior (Q4-2009) de todas las empresas, que es cuando cambian de verdad los ratios.
EVAL_START_QUARTER = "2010Q1"
# Días tras el cierre de un periodo en que su fundamental pasa a ser observable (un 10-K/10-Q tarda en
# publicarse). Antes de este retardo, ese periodo NO existe para el modelo: corrige el lookahead sutil
# de tratar el fundamental como conocido el mismo día del cierre de periodo.
FUNDAMENTAL_PUBLICATION_LAG_DAYS = 45


def _quarter_to_anchor(spec: str, lag_days: int) -> str:
    """'2010Q1' + lag -> fecha ancla ISO. Q1->1-ene, Q2->1-abr, Q3->1-jul, Q4->1-oct, + lag_days."""
    year_str, quarter_str = spec.upper().split("Q")
    year, quarter = int(year_str), int(quarter_str)
    if quarter not in (1, 2, 3, 4):
        raise ValueError(f"Trimestre inválido en '{spec}': usa Q1..Q4.")
    quarter_start = pd.Timestamp(year=year, month=(quarter - 1) * 3 + 1, day=1)
    return (quarter_start + pd.Timedelta(days=lag_days)).date().isoformat()


# Fecha ancla derivada del trimestre + retardo. Es el punto desde el que arranca el walk-forward (se
# entrena por primera vez) y desde el que opera la cartera: simula "haber ejecutado el pipeline aquí".
EVAL_START_DATE = _quarter_to_anchor(EVAL_START_QUARTER, FUNDAMENTAL_PUBLICATION_LAG_DAYS)
PORTFOLIO_START_DATE = EVAL_START_DATE
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
# weights) every WALK_FORWARD_TRAIN_FREQUENCY across the ENTIRE evaluated window, from the eval anchor
# (TRAIN_CUTOFF_DATE = EVAL_START_DATE) through PORTFOLIO_END_DATE — never freezing the model. Each
# relearning step uses only the trailing MAX_WALK_FORWARD_TRAINING_YEARS of history available as of
# that date (no lookahead), so the model is always trained on recent data.
TRAIN_CUTOFF_DATE = EVAL_START_DATE
# Cadencia de ENTRENAMIENTO (reajuste del modelo): "Q" trimestral o "A"/"Y" anual. Solo tiene sentido
# reentrenar cuando hay fundamentales nuevos; la revisión de cartera es mensual (PRICE_UPDATE_FREQUENCY)
# y re-precia sin reentrenar. El entrenamiento mensual se descarta a propósito (3 meses con los mismos
# fundamentales). Es uno de los ejes del barrido de escenarios.
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
    eval_start_quarter: str = EVAL_START_QUARTER
    fundamental_publication_lag_days: int = FUNDAMENTAL_PUBLICATION_LAG_DAYS
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
    def eval_start_date(self) -> str:
        """Fecha ancla ISO desde la que arranca el walk-forward y opera la cartera. Coincide con
        train_cutoff_date; se expone con nombre propio para leerla semánticamente en ml/dataset."""
        return self.train_cutoff_date

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
