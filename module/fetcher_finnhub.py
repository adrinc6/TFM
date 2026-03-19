# =============================================================================
# module/fetcher_finnhub.py — Descarga de datos via Finnhub + Yahoo Finance HTTP
# =============================================================================
"""
Reemplaza al fetcher original (yfinance) con Finnhub como fuente principal
y Yahoo Finance HTTP directo para precios OHLCV y macro.

Estructura de salida en data_finnhub/:
    data_finnhub/
    ├── _registry.json               ← control de descargas por endpoint
    ├── _macro/
    │   ├── vix.json
    │   ├── sp500.json
    │   ├── us10y.json
    │   └── us2y.json
    └── AAPL/
        ├── prices.json
        ├── profile.json
        ├── basic_financials.json
        ├── financials_reported_annual.json
        ├── financials_reported_quarterly.json
        ├── eps_surprises.json
        ├── earnings_calendar.json
        ├── recommendation_trends.json
        ├── insider_transactions.json
        ├── insider_sentiment.json
        ├── company_news.json        ← descargado pero no procesado
        ├── peers.json
        └── quote.json
"""
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

log = logging.getLogger(__name__)


# =============================================================================
# REGISTRO DE DESCARGAS
# =============================================================================

class Registry:
    """
    Controla qué endpoints se han descargado ya.
    Se persiste en BASE_DIR/_registry.json.
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
        return endpoint in self._data.get(group, {})

    def mark_done(self, group: str, endpoint: str):
        if group not in self._data:
            self._data[group] = {}
        self._data[group][endpoint] = datetime.utcnow().isoformat()
        self.save()

    def clear(self, group: str = None):
        if group:
            self._data.pop(group, None)
        else:
            self._data = {}
        self.save()


# =============================================================================
# CLIENTE FINNHUB
# =============================================================================

class FinnhubClient:
    """Cliente HTTP para Finnhub con rate limiting conservador (plan free)."""

    BASE         = "https://finnhub.io/api/v1"
    MIN_INTERVAL = 1.05   # ~57 req/min (límite free: 60/min)

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
                log.warning("Rate limit — esperando 65 s...")
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

    def company_profile2(self, symbol):
        """Nombre, sector, market cap, exchange, IPO date, logo, web."""
        return self._get("/stock/profile2", {"symbol": symbol})

    def basic_financials(self, symbol):
        """
        Ratios calculados point-in-time + series anuales:
        P/E, P/B, P/S, EV/FCF, ROE, ROA, ROIC, márgenes, debt/equity,
        current ratio, interest coverage, beta, 52w high/low,
        avg volume, y series anuales de revenue, netIncome, EPS, FCF, etc.
        """
        return self._get("/stock/metric", {"symbol": symbol, "metric": "all"})

    def financials_as_reported(self, symbol, freq="annual"):
        """
        Estados financieros RAW tal como fueron reportados al SEC.
        Secciones 'bs' (Balance Sheet), 'ic' (Income Statement), 'cf' (Cash Flow).
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
        """Compras y ventas de insiders. change > 0 = compra, < 0 = venta."""
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
        """Noticias recientes (descargadas pero no procesadas por el pipeline)."""
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


# =============================================================================
# CLIENTE YAHOO FINANCE (HTTP directo, sin librerías externas)
# =============================================================================

class YahooClient:
    """
    Descarga OHLCV diario de Yahoo Finance usando la API v8 directamente.
    No requiere yfinance.
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
        return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())

    def ohlcv(self, ticker: str, start: str, end: str) -> Optional[dict]:
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
                log.warning(f"[{ticker}] Yahoo rate limit — esperando 30 s...")
                time.sleep(30)
                return self.ohlcv(ticker, start, end)

            if resp.status_code != 200:
                log.warning(f"[{ticker}] Yahoo status {resp.status_code}")
                return None

            raw = resp.json()
            result = raw.get("chart", {}).get("result")
            if not result:
                return None

            result     = result[0]
            meta       = result.get("meta", {})
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

            events    = result.get("events", {})
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
    try:
        v = lst[i]
        return None if v != v else v
    except (IndexError, TypeError):
        return None


# =============================================================================
# UTILIDADES JSON
# =============================================================================

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
    for key in ("data", "earningsCalendar", "series", "historicalConstituents"):
        val = payload.get(key)
        if isinstance(val, list):
            return f" ({len(val)} items)"
        if isinstance(val, dict):
            return " (dict)"
    return ""


# =============================================================================
# DESCARGA GENÉRICA CON REGISTRO
# =============================================================================

def fetch_and_save(
    group:     str,
    endpoint:  str,
    fn,
    out_path:  Path,
    registry:  Registry,
    force:     bool,
    wrap:      str = None,
) -> str:
    if not force and registry.is_done(group, endpoint):
        return "skip"

    data = fn()
    if not data:
        return "nodata"

    if isinstance(data, list):
        payload = {wrap or "data": data}
    else:
        payload = dict(data)

    payload.update(_meta(group))
    save_json(payload, out_path)
    registry.mark_done(group, endpoint)
    return f"ok{_count(payload)}"


# =============================================================================
# DESCARGA DE PRECIOS
# =============================================================================

def download_prices(
    ticker:     str,
    ticker_dir: Path,
    start:      str,
    end:        str,
    registry:   Registry,
    force:      bool,
    yahoo:      YahooClient,
) -> str:
    endpoint = "prices"
    if not force and registry.is_done(ticker, endpoint):
        return "skip"

    data = yahoo.ohlcv(ticker, start, end)
    if not data or not data.get("data"):
        return "nodata"

    payload = {**_meta(ticker), "start": start, "end": end, "source": "yahoo_v8", **data}
    save_json(payload, ticker_dir / "prices.json")
    registry.mark_done(ticker, endpoint)
    return f"ok ({len(data['data'])} días)"


# =============================================================================
# DESCARGA MACRO
# =============================================================================

MACRO_SERIES = {
    "vix":   "^VIX",
    "sp500": "^GSPC",
    "us10y": "^TNX",
    "us2y":  "^IRX",
}


def download_macro(
    base_dir: Path,
    registry: Registry,
    start:    str,
    end:      str,
    force:    bool,
    yahoo:    YahooClient,
):
    macro_dir = base_dir / "_macro"
    macro_dir.mkdir(parents=True, exist_ok=True)

    for name, yticker in MACRO_SERIES.items():
        if not force and registry.is_done("_macro", name):
            log.info(f"  _macro/{name}.json  skip")
            continue

        data = yahoo.ohlcv(yticker, start, end)
        if not data or not data.get("data"):
            log.warning(f"  _macro/{name}.json  sin datos")
            continue

        payload = {
            "_name":          name,
            "_source_ticker": yticker,
            "_downloaded_at": datetime.utcnow().isoformat(),
            "start": start,
            "end":   end,
            "data":  [{"date": r["date"], "close": r["close"]}
                      for r in data["data"]],
        }
        save_json(payload, macro_dir / f"{name}.json")
        registry.mark_done("_macro", name)
        log.info(f"  _macro/{name}.json  ok ({len(payload['data'])} filas)")


# =============================================================================
# DESCARGA POR TICKER
# =============================================================================

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
    ticker_dir = base_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    r = {}

    # 1. Precios OHLCV diarios (Yahoo Finance HTTP)
    r["prices"] = download_prices(ticker, ticker_dir, start, end, registry, force, yahoo)

    # 2. Perfil de empresa: sector, market cap, IPO date
    r["profile"] = fetch_and_save(
        ticker, "profile",
        lambda: client.company_profile2(ticker),
        ticker_dir / "profile.json",
        registry, force,
    )

    # 3. Ratios calculados + series anuales (P/E, ROE, márgenes, beta, etc.)
    r["basic_financials"] = fetch_and_save(
        ticker, "basic_financials",
        lambda: client.basic_financials(ticker),
        ticker_dir / "basic_financials.json",
        registry, force,
    )

    # 4. Estados financieros anuales crudos (10-K → SEC EDGAR)
    r["financials_reported_annual"] = fetch_and_save(
        ticker, "financials_reported_annual",
        lambda: client.financials_as_reported(ticker, freq="annual"),
        ticker_dir / "financials_reported_annual.json",
        registry, force,
    )

    # 5. Estados financieros trimestrales crudos (10-Q → SEC EDGAR)
    r["financials_reported_quarterly"] = fetch_and_save(
        ticker, "financials_reported_quarterly",
        lambda: client.financials_as_reported(ticker, freq="quarterly"),
        ticker_dir / "financials_reported_quarterly.json",
        registry, force,
    )

    # 6. EPS surprises (hasta 20 trimestres)
    r["eps_surprises"] = fetch_and_save(
        ticker, "eps_surprises",
        lambda: client.eps_surprises(ticker, limit=20),
        ticker_dir / "eps_surprises.json",
        registry, force, wrap="data",
    )

    # 7. Earnings calendar (fechas + EPS y revenue real vs estimado)
    r["earnings_calendar"] = fetch_and_save(
        ticker, "earnings_calendar",
        lambda: client.earnings_calendar(start, end, symbol=ticker),
        ticker_dir / "earnings_calendar.json",
        registry, force,
    )

    # 8. Recommendation trends (strongBuy/buy/hold/sell/strongSell por mes)
    r["recommendation_trends"] = fetch_and_save(
        ticker, "recommendation_trends",
        lambda: client.recommendation_trends(ticker),
        ticker_dir / "recommendation_trends.json",
        registry, force, wrap="data",
    )

    # 9. Insider transactions (change > 0 = compra, < 0 = venta)
    r["insider_transactions"] = fetch_and_save(
        ticker, "insider_transactions",
        lambda: client.insider_transactions(ticker, start, end),
        ticker_dir / "insider_transactions.json",
        registry, force,
    )

    # 10. Insider sentiment (MSPR mensual: -100 bajista a 100 alcista)
    r["insider_sentiment"] = fetch_and_save(
        ticker, "insider_sentiment",
        lambda: client.insider_sentiment(ticker, start, end),
        ticker_dir / "insider_sentiment.json",
        registry, force,
    )

    # 11. Company news — descargado pero NO procesado por el pipeline ML
    news_start = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    r["company_news"] = fetch_and_save(
        ticker, "company_news",
        lambda: client.company_news(ticker, news_start, end),
        ticker_dir / "company_news.json",
        registry, force, wrap="data",
    )

    # 12. Peers (mismo país y sub-industria)
    r["peers"] = fetch_and_save(
        ticker, "peers",
        lambda: client.peers(ticker),
        ticker_dir / "peers.json",
        registry, force, wrap="peers",
    )

    # 13. Quote (precio actual del día)
    r["quote"] = fetch_and_save(
        ticker, "quote",
        lambda: client.quote(ticker),
        ticker_dir / "quote.json",
        registry, force,
    )

    return r


# =============================================================================
# PIPELINE PRINCIPAL DE DESCARGA
# =============================================================================

def run_download(
    api_key:  str,
    tickers:  list,
    start:    str  = "2015-01-01",
    end:      str  = None,
    base_dir: str  = "data_finnhub",
    force:    bool = False,
):
    """
    Descarga todos los datos para la lista de tickers.
    Función principal a llamar desde data_ops.py.
    """
    end      = end or datetime.today().strftime("%Y-%m-%d")
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    client   = FinnhubClient(api_key)
    yahoo    = YahooClient()
    registry = Registry(base_dir)

    if force:
        log.warning("FORCE_DOWNLOAD=True — re-descargando todo")
        registry.clear()

    log.info("=" * 50)
    log.info("  Finnhub + Yahoo Finance Downloader")
    log.info(f"  Tickers  : {len(tickers)}")
    log.info(f"  Período  : {start} → {end}")
    log.info(f"  Destino  : {base_dir.resolve()}")
    log.info("=" * 50)

    # Macro primero
    log.info("Descargando macro...")
    download_macro(base_dir, registry, start, end, force, yahoo)

    # Tickers
    log.info(f"Descargando {len(tickers)} tickers...")
    summary = {"ok": 0, "partial": 0, "fail": 0}

    iterable = (
        tqdm(tickers, desc="Tickers", unit="ticker")
        if TQDM_AVAILABLE else tickers
    )

    for ticker in iterable:
        try:
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
            ok   = sum(1 for v in results.values() if v.startswith("ok"))
            skip = sum(1 for v in results.values() if v == "skip")
            total = len(results)

            if ok + skip == total:
                summary["ok"] += 1
            elif ok + skip > 0:
                summary["partial"] += 1
            else:
                summary["fail"] += 1

            log.debug(f"[{ticker}] ok={ok} skip={skip} nodata={total-ok-skip}")
        except Exception as e:
            log.warning(f"[{ticker}] Error inesperado: {e}")
            summary["fail"] += 1

    log.info("=" * 50)
    log.info("  DESCARGA COMPLETADA")
    log.info(f"  Completos  : {summary['ok']}")
    log.info(f"  Parciales  : {summary['partial']}")
    log.info(f"  Sin datos  : {summary['fail']}")
    log.info("=" * 50)


def fetch_all_finnhub(
    tickers:  list,
    start:    str,
    end:      str,
    base_dir: str,
    api_key:  str,
    force:    bool = False,
):
    """
    Punto de entrada desde data_ops.py.
    Descarga todos los datos Finnhub para los tickers dados.
    """
    run_download(
        api_key=api_key,
        tickers=tickers,
        start=start,
        end=end,
        base_dir=base_dir,
        force=force,
    )
