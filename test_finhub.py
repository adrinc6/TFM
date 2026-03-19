"""
finnhub_downloader.py
=====================
Descarga todo lo relevante para entrenar modelos ML en bolsa usando
exclusivamente Finnhub (plan gratuito) y la API pública de Yahoo Finance
(HTTP directo, sin librerías externas) para los precios OHLCV.

Estructura de salida:
    data_finnhub/
    ├── _registry.json               ← control de descargas (qué/cuándo)
    ├── _macro/
    │   ├── vix.json                 ← serie histórica VIX
    │   ├── sp500.json               ← serie histórica S&P 500
    │   ├── us10y.json               ← yield 10Y Treasury
    │   └── us2y.json                ← yield 2Y Treasury
    └── AAPL/
        ├── prices.json                       ← OHLCV diario (Yahoo Finance HTTP)
        ├── profile.json                      ← nombre, sector, market cap, IPO
        ├── basic_financials.json             ← ratios: P/E, ROE, márgenes, beta…
        ├── financials_reported_annual.json   ← BS + IS + CF anuales (10-K, SEC)
        ├── financials_reported_quarterly.json← BS + IS + CF trimestrales (10-Q)
        ├── eps_surprises.json                ← EPS real vs estimado (20 trimestres)
        ├── earnings_calendar.json            ← fechas resultados + revenue
        ├── recommendation_trends.json        ← buy/hold/sell por período
        ├── insider_transactions.json         ← compras/ventas insiders
        ├── insider_sentiment.json            ← MSPR mensual
        ├── company_news.json                 ← noticias último año
        ├── peers.json                        ← competidores mismo sector
        └── quote.json                        ← precio y cambio del día

Control de descargas (_registry.json):
    - Cada endpoint de cada ticker queda registrado con su timestamp.
    - Si FORCE_DOWNLOAD = False, un endpoint ya registrado NO se vuelve
      a descargar aunque re-ejecutes el script.
    - Pon FORCE_DOWNLOAD = True para forzar la re-descarga completa.

Ejecutar:
    pip install requests tqdm
    python finnhub_downloader.py
"""

import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import requests

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("finnhub_dl")


# ═══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS — edita aquí
# ═══════════════════════════════════════════════════════════════════════════════

FINNHUB_API_KEY = "d6ttu99r01qhkb45jm5gd6ttu99r01qhkb45jm60"

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "UNH",
]

START          = "2015-01-01"   # Fecha de inicio YYYY-MM-DD
END            = None           # Fecha de fin YYYY-MM-DD — None = hoy
BASE_DIR       = "data_finnhub" # Directorio raíz de salida
FORCE_DOWNLOAD = False          # True = re-descarga todo ignorando el registro
VERBOSE        = False          # True = logs de debug

# Pon SP500 = True para descargar los ~500 tickers del S&P 500 (ignora TICKERS)
SP500 = False

# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRO DE DESCARGAS
# ═══════════════════════════════════════════════════════════════════════════════

class Registry:
    """
    Controla qué endpoints se han descargado ya.
    Se persiste en BASE_DIR/_registry.json.

    Estructura interna:
        {
            "AAPL": {
                "prices":                     "2024-03-01T10:23:11",
                "profile":                    "2024-03-01T10:23:12",
                "financials_reported_annual": "2024-03-01T10:23:14",
                ...
            },
            "_macro": {
                "vix":   "2024-03-01T10:23:00",
                ...
            }
        }
    """

    def __init__(self, base_dir: Path):
        self.path = base_dir / "_registry.json"
        self._data: dict = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def is_done(self, group: str, endpoint: str) -> bool:
        """True si este endpoint ya fue descargado (y FORCE_DOWNLOAD=False)."""
        return endpoint in self._data.get(group, {})

    def mark_done(self, group: str, endpoint: str):
        """Registra que el endpoint se descargó ahora."""
        if group not in self._data:
            self._data[group] = {}
        self._data[group][endpoint] = datetime.utcnow().isoformat()
        self.save()   # guardar en cada marca para no perder progreso

    def clear(self, group: str = None):
        """Borra el registro completo o solo un grupo."""
        if group:
            self._data.pop(group, None)
        else:
            self._data = {}
        self.save()


# ═══════════════════════════════════════════════════════════════════════════════
# CLIENTE FINNHUB
# ═══════════════════════════════════════════════════════════════════════════════

class FinnhubClient:
    """Cliente HTTP para Finnhub con rate limiting conservador (plan free)."""

    BASE         = "https://finnhub.io/api/v1"
    MIN_INTERVAL = 1.05   # ~57 req/min  (límite free: 60/min)

    def __init__(self, api_key: str):
        self.api_key    = api_key
        self._last_call = 0.0
        self.session    = requests.Session()
        self.session.headers.update({"X-Finnhub-Token": api_key})

    def _get(self, endpoint: str, params: dict = None):
        params = params or {}
        url    = f"{self.BASE}{endpoint}"

        wait = self.MIN_INTERVAL - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)

        try:
            resp = self.session.get(url, params=params, timeout=20)
            self._last_call = time.time()

            if resp.status_code == 429:
                log.warning("Rate limit — esperando 65 s…")
                time.sleep(65)
                return self._get(endpoint, params)

            if resp.status_code in (401, 403):
                log.debug(f"  {resp.status_code} → {endpoint} (Premium / no autorizado)")
                return None

            resp.raise_for_status()
            data = resp.json()
            return data if data else None

        except requests.exceptions.RequestException as e:
            log.warning(f"  Error en {endpoint}: {e}")
            return None

    # ── Endpoints gratuitos ───────────────────────────────────────────────────

    def company_profile2(self, symbol):
        """Nombre, sector, market cap, exchange, IPO date, logo, web."""
        return self._get("/stock/profile2", {"symbol": symbol})

    def basic_financials(self, symbol):
        """
        Ratios calculados point-in-time + series anuales:
        P/E, P/B, P/S, EV/FCF, ROE, ROA, ROIC, márgenes (neto/bruto/EBITDA/FCF),
        debt/equity, current ratio, interest coverage, beta, 52w high/low,
        10/52-day avg volume, y series anuales de revenue, netIncome, EPS, FCF,
        currentRatio, salesPerShare, netMargin, etc.
        """
        return self._get("/stock/metric", {"symbol": symbol, "metric": "all"})

    def financials_as_reported(self, symbol, freq="annual"):
        """
        Estados financieros RAW tal como fueron reportados al SEC.
        Contiene secciones 'bs' (Balance Sheet), 'ic' (Income Statement)
        y 'cf' (Cash Flow) para cada filing 10-K (annual) o 10-Q (quarterly).
        Requiere postprocesado para extraer líneas concretas.
        freq: 'annual' | 'quarterly'
        """
        return self._get("/stock/financials-reported", {
            "symbol": symbol,
            "freq":   freq,
        })

    def eps_surprises(self, symbol, limit=20):
        """EPS real vs estimado + surprise% por trimestre (hasta 20Q)."""
        return self._get("/stock/earnings", {"symbol": symbol, "limit": limit})

    def earnings_calendar(self, from_date, to_date, symbol=""):
        """Fechas de resultados con EPS y revenue real vs estimado."""
        params = {"from": from_date, "to": to_date}
        if symbol:
            params["symbol"] = symbol
        return self._get("/calendar/earnings", params)

    def recommendation_trends(self, symbol):
        """Tendencias de analistas: strongBuy/buy/hold/sell/strongSell por mes."""
        return self._get("/stock/recommendation", {"symbol": symbol})

    def insider_transactions(self, symbol, from_date, to_date):
        """Compras y ventas de insiders (Forms 3, 4, 5). + = compra, - = venta."""
        return self._get("/stock/insider-transactions", {
            "symbol": symbol, "from": from_date, "to": to_date,
        })

    def insider_sentiment(self, symbol, from_date, to_date):
        """
        MSPR mensual (Monthly Share Purchase Ratio), rango -100 a 100.
        Señal predictiva documentada para los siguientes 30-90 días.
        """
        return self._get("/stock/insider-sentiment", {
            "symbol": symbol, "from": from_date, "to": to_date,
        })

    def company_news(self, symbol, from_date, to_date):
        """Noticias recientes de la compañía (free tier: último año)."""
        return self._get("/company-news", {
            "symbol": symbol, "from": from_date, "to": to_date,
        })

    def peers(self, symbol):
        """Peers en el mismo país y sub-industria."""
        return self._get("/stock/peers", {"symbol": symbol})

    def quote(self, symbol):
        """Precio actual, cambio día, máximo/mínimo diario, cierre anterior."""
        return self._get("/quote", {"symbol": symbol})

    def sp500_symbols(self):
        """Componentes del S&P 500 scrapeados de Wikipedia."""
        try:
            import pandas as pd
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )
            return (
                tables[0]["Symbol"]
                .str.replace(".", "-", regex=False)
                .tolist()
            )
        except Exception as e:
            log.error(f"No se pudo obtener el S&P 500: {e}")
            return []


# ═══════════════════════════════════════════════════════════════════════════════
# CLIENTE YAHOO FINANCE (HTTP directo, sin librerías)
# ═══════════════════════════════════════════════════════════════════════════════

class YahooClient:
    """
    Descarga OHLCV diario de Yahoo Finance usando la API v8 directamente
    por HTTP. No requiere yfinance ni ninguna librería externa.
    """

    BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._last_call = 0.0

    def _dt(self, date_str: str) -> int:
        """Convierte YYYY-MM-DD a UNIX timestamp."""
        return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())

    def ohlcv(self, ticker: str, start: str, end: str) -> dict | None:
        """
        Descarga OHLCV diario ajustado para un ticker.
        Devuelve dict con 'data': lista de {date, open, high, low, close, volume}
        o None si falla.
        """
        # Pequeño throttle para no saturar Yahoo
        wait = 0.5 - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)

        params = {
            "period1":  self._dt(start),
            "period2":  self._dt(end),
            "interval": "1d",
            "events":   "div,splits",
            "includeAdjustedClose": "true",
        }

        url = f"{self.BASE}/{ticker}"

        try:
            resp = self.session.get(url, params=params, timeout=20)
            self._last_call = time.time()

            if resp.status_code == 429:
                log.warning(f"[{ticker}] Yahoo rate limit — esperando 30 s…")
                time.sleep(30)
                return self.ohlcv(ticker, start, end)

            if resp.status_code != 200:
                log.warning(f"[{ticker}] Yahoo status {resp.status_code}")
                return None

            raw = resp.json()
            result = raw.get("chart", {}).get("result")
            if not result:
                log.warning(f"[{ticker}] Yahoo sin resultado")
                return None

            result  = result[0]
            meta    = result.get("meta", {})
            timestamps = result.get("timestamp", [])
            indicators = result.get("indicators", {})
            quote_data = indicators.get("quote", [{}])[0]
            adj_close  = indicators.get("adjclose", [{}])[0].get("adjclose", [])

            records = []
            for i, ts in enumerate(timestamps):
                date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                records.append({
                    "date":      date_str,
                    "open":      _safe(quote_data.get("open",   []), i),
                    "high":      _safe(quote_data.get("high",   []), i),
                    "low":       _safe(quote_data.get("low",    []), i),
                    "close":     _safe(quote_data.get("close",  []), i),
                    "adj_close": _safe(adj_close, i),
                    "volume":    _safe(quote_data.get("volume", []), i),
                })

            # Eventos: dividendos y splits
            events = result.get("events", {})
            dividends = [
                {"date": datetime.utcfromtimestamp(int(k)).strftime("%Y-%m-%d"),
                 "amount": v.get("amount")}
                for k, v in events.get("dividends", {}).items()
            ]
            splits = [
                {"date": datetime.utcfromtimestamp(int(k)).strftime("%Y-%m-%d"),
                 "ratio": f"{v.get('numerator')}/{v.get('denominator')}"}
                for k, v in events.get("splits", {}).items()
            ]

            return {
                "currency":  meta.get("currency"),
                "exchange":  meta.get("exchangeName"),
                "data":      records,
                "dividends": dividends,
                "splits":    splits,
            }

        except Exception as e:
            log.warning(f"[{ticker}] Yahoo error: {e}")
            return None


def _safe(lst: list, i: int):
    """Acceso seguro a lista, devuelve None si fuera de rango o es NaN."""
    try:
        v = lst[i]
        return None if v != v else v   # NaN check
    except (IndexError, TypeError):
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES JSON
# ═══════════════════════════════════════════════════════════════════════════════

def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _meta(ticker: str) -> dict:
    return {
        "_ticker":        ticker,
        "_downloaded_at": datetime.utcnow().isoformat(),
    }


def _count(payload: dict) -> str:
    """Devuelve '(N items)' buscando la primera lista en el payload."""
    for key in ("data", "earningsCalendar", "series", "historicalConstituents"):
        val = payload.get(key)
        if isinstance(val, list):
            return f" ({len(val)} items)"
        if isinstance(val, dict):
            # p.ej. basic_financials tiene series.annual.*
            return " (dict)"
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# DESCARGA GENÉRICA CON REGISTRO
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_and_save(
    group:     str,
    endpoint:  str,
    fn,
    out_path:  Path,
    registry:  Registry,
    force:     bool,
    wrap:      str = None,
) -> str:
    """
    Ejecuta fn() y guarda el resultado en out_path si:
      - FORCE_DOWNLOAD = True, o
      - el endpoint no está en el registro.

    group:    clave de agrupación en el registro (ticker o '_macro')
    endpoint: nombre del archivo / clave en el registro
    fn:       callable sin argumentos que devuelve los datos
    wrap:     si la respuesta es una lista, la envuelve en {wrap: [...]}
    """
    if not force and registry.is_done(group, endpoint):
        return "⏭  ya descargado"

    data = fn()
    if not data:
        return "❌ sin datos"

    if isinstance(data, list):
        payload = {wrap or "data": data}
    else:
        payload = dict(data)

    payload.update(_meta(group))
    save_json(payload, out_path)
    registry.mark_done(group, endpoint)
    return f"✅{_count(payload)}"


# ═══════════════════════════════════════════════════════════════════════════════
# DESCARGA DE PRECIOS
# ═══════════════════════════════════════════════════════════════════════════════

def download_prices(
    ticker:    str,
    ticker_dir: Path,
    start:     str,
    end:       str,
    registry:  Registry,
    force:     bool,
    yahoo:     YahooClient,
) -> str:
    endpoint = "prices"
    if not force and registry.is_done(ticker, endpoint):
        return "⏭  ya descargado"

    data = yahoo.ohlcv(ticker, start, end)
    if not data or not data.get("data"):
        return "❌ sin datos"

    payload = {**_meta(ticker), "start": start, "end": end, "source": "yahoo_v8", **data}
    save_json(payload, ticker_dir / "prices.json")
    registry.mark_done(ticker, endpoint)
    return f"✅ ({len(data['data'])} días)"


# ═══════════════════════════════════════════════════════════════════════════════
# DESCARGA POR TICKER
# ═══════════════════════════════════════════════════════════════════════════════

def download_ticker(
    ticker:    str,
    client:    FinnhubClient,
    yahoo:     YahooClient,
    base_dir:  Path,
    registry:  Registry,
    start:     str,
    end:       str,
    force:     bool,
) -> dict:
    """
    Descarga todos los endpoints para un ticker.
    Devuelve {endpoint: estado} para el log final.
    """
    ticker_dir = base_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    r = {}   # resultados

    # ── 1. Precios OHLCV diarios (Yahoo Finance HTTP) ─────────────────────────
    r["prices"] = download_prices(
        ticker, ticker_dir, start, end, registry, force, yahoo
    )

    # ── 2. Perfil de empresa ──────────────────────────────────────────────────
    # nombre, sector, market cap, exchange, IPO date, logo, web
    r["profile"] = fetch_and_save(
        ticker, "profile",
        lambda: client.company_profile2(ticker),
        ticker_dir / "profile.json",
        registry, force,
    )

    # ── 3. Ratios y métricas calculadas (Basic Financials) ────────────────────
    # P/E, P/B, P/S, EV/FCF, ROE, ROA, ROIC, márgenes, debt/equity,
    # current ratio, interest coverage, beta, 52w high/low, avg volume,
    # + series anuales de revenue, netIncome, EPS, FCF, currentRatio…
    r["basic_financials"] = fetch_and_save(
        ticker, "basic_financials",
        lambda: client.basic_financials(ticker),
        ticker_dir / "basic_financials.json",
        registry, force,
    )

    # ── 4. Estados financieros anuales crudos (10-K → SEC EDGAR) ─────────────
    # Secciones: bs (Balance Sheet), ic (Income Statement), cf (Cash Flow)
    # Exactamente como se reportaron, requieren postprocesado para extraer
    # líneas como Assets, Liabilities, Revenue, NetIncome, CapEx, etc.
    r["financials_reported_annual"] = fetch_and_save(
        ticker, "financials_reported_annual",
        lambda: client.financials_as_reported(ticker, freq="annual"),
        ticker_dir / "financials_reported_annual.json",
        registry, force,
    )

    # ── 5. Estados financieros trimestrales crudos (10-Q → SEC EDGAR) ────────
    r["financials_reported_quarterly"] = fetch_and_save(
        ticker, "financials_reported_quarterly",
        lambda: client.financials_as_reported(ticker, freq="quarterly"),
        ticker_dir / "financials_reported_quarterly.json",
        registry, force,
    )

    # ── 6. EPS surprises (hasta 20 trimestres) ────────────────────────────────
    # EPS real vs estimado, surprise absoluto y en %, quarter y year
    r["eps_surprises"] = fetch_and_save(
        ticker, "eps_surprises",
        lambda: client.eps_surprises(ticker, limit=20),
        ticker_dir / "eps_surprises.json",
        registry, force, wrap="data",
    )

    # ── 7. Earnings calendar ──────────────────────────────────────────────────
    # Fecha de publicación de resultados + EPS y revenue real vs estimado
    r["earnings_calendar"] = fetch_and_save(
        ticker, "earnings_calendar",
        lambda: client.earnings_calendar(start, end, symbol=ticker),
        ticker_dir / "earnings_calendar.json",
        registry, force,
    )

    # ── 8. Recommendation trends ──────────────────────────────────────────────
    # strongBuy / buy / hold / sell / strongSell por período mensual
    r["recommendation_trends"] = fetch_and_save(
        ticker, "recommendation_trends",
        lambda: client.recommendation_trends(ticker),
        ticker_dir / "recommendation_trends.json",
        registry, force, wrap="data",
    )

    # ── 9. Insider transactions ───────────────────────────────────────────────
    # change > 0 = compra, change < 0 = venta
    # transactionCode: S=sale, P=purchase, M=option exercise, etc.
    r["insider_transactions"] = fetch_and_save(
        ticker, "insider_transactions",
        lambda: client.insider_transactions(ticker, start, end),
        ticker_dir / "insider_transactions.json",
        registry, force,
    )

    # ── 10. Insider sentiment (MSPR mensual) ──────────────────────────────────
    # Monthly Share Purchase Ratio: -100 (muy bajista) a 100 (muy alcista)
    r["insider_sentiment"] = fetch_and_save(
        ticker, "insider_sentiment",
        lambda: client.insider_sentiment(ticker, start, end),
        ticker_dir / "insider_sentiment.json",
        registry, force,
    )

    # ── 11. Company news (último año — límite free tier) ─────────────────────
    news_start = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    r["company_news"] = fetch_and_save(
        ticker, "company_news",
        lambda: client.company_news(ticker, news_start, end),
        ticker_dir / "company_news.json",
        registry, force, wrap="data",
    )

    # ── 12. Peers ─────────────────────────────────────────────────────────────
    # Lista de tickers del mismo país y sub-industria
    r["peers"] = fetch_and_save(
        ticker, "peers",
        lambda: client.peers(ticker),
        ticker_dir / "peers.json",
        registry, force, wrap="peers",
    )

    # ── 13. Quote (precio actual) ─────────────────────────────────────────────
    # current price, change, % change, high/low del día, previous close
    r["quote"] = fetch_and_save(
        ticker, "quote",
        lambda: client.quote(ticker),
        ticker_dir / "quote.json",
        registry, force,
    )

    return r


# ═══════════════════════════════════════════════════════════════════════════════
# DESCARGA MACRO
# ═══════════════════════════════════════════════════════════════════════════════

MACRO_SERIES = {
    "vix":   "^VIX",
    "sp500": "^GSPC",
    "us10y": "^TNX",
    "us2y":  "^IRX",
}

def download_macro(
    base_dir:  Path,
    registry:  Registry,
    start:     str,
    end:       str,
    force:     bool,
    yahoo:     YahooClient,
):
    macro_dir = base_dir / "_macro"
    macro_dir.mkdir(parents=True, exist_ok=True)

    for name, yticker in MACRO_SERIES.items():
        endpoint = name
        if not force and registry.is_done("_macro", endpoint):
            log.info(f"  _macro/{name}.json  ⏭  ya descargado")
            continue

        data = yahoo.ohlcv(yticker, start, end)
        if not data or not data.get("data"):
            log.warning(f"  _macro/{name}.json  ❌ sin datos")
            continue

        payload = {
            "_name":         name,
            "_source_ticker": yticker,
            "_downloaded_at": datetime.utcnow().isoformat(),
            "start": start,
            "end":   end,
            "data":  [{"date": r["date"], "close": r["close"]}
                      for r in data["data"]],
        }
        save_json(payload, macro_dir / f"{name}.json")
        registry.mark_done("_macro", endpoint)
        log.info(f"  _macro/{name}.json  ✅ ({len(payload['data'])} filas)")


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def run_download(
    api_key:  str,
    tickers:  list,
    start:    str  = "2015-01-01",
    end:      str  = None,
    base_dir: str  = "data_finnhub",
    force:    bool = False,
):
    end      = end or datetime.today().strftime("%Y-%m-%d")
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    client   = FinnhubClient(api_key)
    yahoo    = YahooClient()
    registry = Registry(base_dir)

    if force:
        log.warning("FORCE_DOWNLOAD = True — se ignorará el registro y se re-descargará todo")
        registry.clear()

    log.info("════════════════════════════════════════════")
    log.info("  Finnhub + Yahoo Finance Downloader")
    log.info(f"  Tickers  : {len(tickers)}")
    log.info(f"  Período  : {start} → {end}")
    log.info(f"  Destino  : {base_dir.resolve()}")
    log.info(f"  Registro : {registry.path}")
    log.info("════════════════════════════════════════════")

    # ── Macro ─────────────────────────────────────────────────────────────────
    log.info("\n── Macro (_macro/) ──────────────────────────")
    download_macro(base_dir, registry, start, end, force, yahoo)

    # ── Por ticker ────────────────────────────────────────────────────────────
    log.info(f"\n── Tickers ({len(tickers)}) ──────────────────────────")

    summary = {"ok": 0, "partial": 0, "fail": 0}
    iterable = (
        tqdm(tickers, desc="Descargando", unit="ticker")
        if TQDM_AVAILABLE else tickers
    )

    for ticker in iterable:
        log.info(f"\n[{ticker}]")
        results = download_ticker(
            ticker=ticker,
            client=client,
            yahoo=yahoo,
            base_dir=base_dir,
            registry=registry,
            start=start,
            end=end,
            force=force,
        )

        ok    = sum(1 for v in results.values() if v.startswith("✅"))
        skip  = sum(1 for v in results.values() if v.startswith("⏭"))
        total = len(results)

        for ep, status in results.items():
            log.info(f"  {ep:<35} {status}")

        if ok + skip == total:
            summary["ok"] += 1
        elif ok + skip > 0:
            summary["partial"] += 1
        else:
            summary["fail"] += 1

    # ── Resumen final ─────────────────────────────────────────────────────────
    log.info("\n════════════════════════════════════════════")
    log.info("  DESCARGA COMPLETADA")
    log.info(f"  ✅ Completos  : {summary['ok']}")
    log.info(f"  ⚠️  Parciales  : {summary['partial']}")
    log.info(f"  ❌ Sin datos  : {summary['fail']}")
    log.info(f"  Directorio   : {base_dir.resolve()}")
    log.info(f"  Registro     : {registry.path}")
    log.info("════════════════════════════════════════════\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if VERBOSE:
        logging.getLogger().setLevel(logging.DEBUG)

    if SP500:
        _tmp = FinnhubClient(FINNHUB_API_KEY)
        tickers = _tmp.sp500_symbols()
        if not tickers:
            log.error("No se pudo obtener el S&P 500.")
        else:
            log.info(f"S&P 500: {len(tickers)} tickers cargados")
            run_download(FINNHUB_API_KEY, tickers, START, END, BASE_DIR, FORCE_DOWNLOAD)
    else:
        run_download(FINNHUB_API_KEY, TICKERS, START, END, BASE_DIR, FORCE_DOWNLOAD)