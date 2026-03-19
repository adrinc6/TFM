"""Download helpers for Finnhub and Yahoo data."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from module.steps.step_01_data.clients import FinnhubClient, YahooClient
from module.steps.step_01_data.registry import Registry

log = logging.getLogger(__name__)

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


MACRO_SERIES = {
    "vix": "^VIX",
    "sp500": "^GSPC",
    "us10y": "^TNX",
    "us2y": "^IRX",
}


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _meta(ticker: str) -> dict:
    return {
        "_ticker": ticker,
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
# Generic download with registry
# =============================================================================

def fetch_and_save(
    group: str,
    endpoint: str,
    fn,
    out_path: Path,
    registry: Registry,
    force: bool,
    wrap: str | None = None,
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
# Prices and macro
# =============================================================================

def download_prices(
    ticker: str,
    ticker_dir: Path,
    start: str,
    end: str,
    registry: Registry,
    force: bool,
    yahoo: YahooClient,
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
    return f"ok ({len(data['data'])} dias)"


def download_macro(
    base_dir: Path,
    registry: Registry,
    start: str,
    end: str,
    force: bool,
    yahoo: YahooClient,
) -> None:
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
            "_name": name,
            "_source_ticker": yticker,
            "_downloaded_at": datetime.utcnow().isoformat(),
            "start": start,
            "end": end,
            "data": [{"date": r["date"], "close": r["close"]} for r in data["data"]],
        }
        save_json(payload, macro_dir / f"{name}.json")
        registry.mark_done("_macro", name)
        log.info(f"  _macro/{name}.json  ok ({len(payload['data'])} filas)")


# =============================================================================
# Per-ticker download
# =============================================================================

def download_ticker(
    ticker: str,
    client: FinnhubClient,
    yahoo: YahooClient,
    base_dir: Path,
    registry: Registry,
    start: str,
    end: str,
    force: bool,
) -> dict:
    ticker_dir = base_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    r: dict = {}

    r["prices"] = download_prices(ticker, ticker_dir, start, end, registry, force, yahoo)

    r["profile"] = fetch_and_save(
        ticker,
        "profile",
        lambda: client.company_profile2(ticker),
        ticker_dir / "profile.json",
        registry,
        force,
    )

    r["basic_financials"] = fetch_and_save(
        ticker,
        "basic_financials",
        lambda: client.basic_financials(ticker),
        ticker_dir / "basic_financials.json",
        registry,
        force,
    )

    r["financials_reported_annual"] = fetch_and_save(
        ticker,
        "financials_reported_annual",
        lambda: client.financials_as_reported(ticker, freq="annual"),
        ticker_dir / "financials_reported_annual.json",
        registry,
        force,
    )

    r["financials_reported_quarterly"] = fetch_and_save(
        ticker,
        "financials_reported_quarterly",
        lambda: client.financials_as_reported(ticker, freq="quarterly"),
        ticker_dir / "financials_reported_quarterly.json",
        registry,
        force,
    )

    r["eps_surprises"] = fetch_and_save(
        ticker,
        "eps_surprises",
        lambda: client.eps_surprises(ticker, limit=20),
        ticker_dir / "eps_surprises.json",
        registry,
        force,
        wrap="data",
    )

    r["earnings_calendar"] = fetch_and_save(
        ticker,
        "earnings_calendar",
        lambda: client.earnings_calendar(start, end, symbol=ticker),
        ticker_dir / "earnings_calendar.json",
        registry,
        force,
    )

    r["recommendation_trends"] = fetch_and_save(
        ticker,
        "recommendation_trends",
        lambda: client.recommendation_trends(ticker),
        ticker_dir / "recommendation_trends.json",
        registry,
        force,
        wrap="data",
    )

    r["insider_transactions"] = fetch_and_save(
        ticker,
        "insider_transactions",
        lambda: client.insider_transactions(ticker, start, end),
        ticker_dir / "insider_transactions.json",
        registry,
        force,
    )

    r["insider_sentiment"] = fetch_and_save(
        ticker,
        "insider_sentiment",
        lambda: client.insider_sentiment(ticker, start, end),
        ticker_dir / "insider_sentiment.json",
        registry,
        force,
    )

    news_start = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    r["company_news"] = fetch_and_save(
        ticker,
        "company_news",
        lambda: client.company_news(ticker, news_start, end),
        ticker_dir / "company_news.json",
        registry,
        force,
        wrap="data",
    )

    r["peers"] = fetch_and_save(
        ticker,
        "peers",
        lambda: client.peers(ticker),
        ticker_dir / "peers.json",
        registry,
        force,
        wrap="peers",
    )

    r["quote"] = fetch_and_save(
        ticker,
        "quote",
        lambda: client.quote(ticker),
        ticker_dir / "quote.json",
        registry,
        force,
    )

    return r


# =============================================================================
# Main download pipeline
# =============================================================================

def run_download(
    api_key: str,
    tickers: list,
    start: str = "2015-01-01",
    end: str | None = None,
    base_dir: str = "data_finnhub",
    force: bool = False,
) -> None:
    end = end or datetime.today().strftime("%Y-%m-%d")
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    client = FinnhubClient(api_key)
    yahoo = YahooClient()
    registry = Registry(base_dir)

    if force:
        log.warning("FORCE_DOWNLOAD=True — re-descargando todo")
        registry.clear()

    log.info("=" * 50)
    log.info("  Finnhub + Yahoo Finance Downloader")
    log.info(f"  Tickers  : {len(tickers)}")
    log.info(f"  Periodo  : {start} -> {end}")
    log.info(f"  Destino  : {base_dir.resolve()}")
    log.info("=" * 50)

    log.info("Descargando macro...")
    download_macro(base_dir, registry, start, end, force, yahoo)

    log.info(f"Descargando {len(tickers)} tickers...")
    summary = {"ok": 0, "partial": 0, "fail": 0}

    iterable = tqdm(tickers, desc="Tickers", unit="ticker") if TQDM_AVAILABLE else tickers

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
            ok = sum(1 for v in results.values() if v.startswith("ok"))
            skip = sum(1 for v in results.values() if v == "skip")
            total = len(results)

            if ok + skip == total:
                summary["ok"] += 1
            elif ok + skip > 0:
                summary["partial"] += 1
            else:
                summary["fail"] += 1

            log.debug(f"[{ticker}] ok={ok} skip={skip} nodata={total - ok - skip}")
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
    tickers: list,
    start: str,
    end: str,
    base_dir: str,
    api_key: str,
    force: bool = False,
) -> None:
    run_download(
        api_key=api_key,
        tickers=tickers,
        start=start,
        end=end,
        base_dir=base_dir,
        force=force,
    )
