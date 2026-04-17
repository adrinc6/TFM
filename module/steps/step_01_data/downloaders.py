"""Download helpers for Finnhub and Yahoo data."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "sp500": "^GSPC",  # Only macro data used: benchmark for forward_return and backtester
}


_TICKER_ENDPOINT_FILES = {
    "prices": "prices.json",
    "profile": "profile.json",
    "basic_financials": "basic_financials.json",
    "financials_reported_annual": "financials_reported_annual.json",
    "financials_reported_quarterly": "financials_reported_quarterly.json",
    "eps_surprises": "eps_surprises.json",
    "earnings_calendar": "earnings_calendar.json",
    "recommendation_trends": "recommendation_trends.json",
    "insider_transactions": "insider_transactions.json",
    "insider_sentiment": "insider_sentiment.json",
    "company_news": "company_news.json",
    "peers": "peers.json",
    "quote": "quote.json",
}


def _required_endpoints(prices_only: bool) -> list[str]:
    if prices_only:
        return ["prices"]
    return list(_TICKER_ENDPOINT_FILES.keys())


def _ticker_is_fully_done(base_dir: Path, ticker: str, registry: Registry, prices_only: bool) -> bool:
    ticker_dir = base_dir / ticker
    for endpoint in _required_endpoints(prices_only):
        file_name = _TICKER_ENDPOINT_FILES[endpoint]
        if not registry.is_done(ticker, endpoint):
            return False
        if not (ticker_dir / file_name).exists():
            return False
    return True


def _ticker_is_partial(base_dir: Path, ticker: str, registry: Registry, prices_only: bool) -> bool:
    ticker_dir = base_dir / ticker
    required = _required_endpoints(prices_only)

    done_count = 0
    missing_count = 0
    failed_count = 0

    for endpoint in required:
        file_name = _TICKER_ENDPOINT_FILES[endpoint]
        file_exists = (ticker_dir / file_name).exists()
        done = registry.is_done(ticker, endpoint)
        entry = registry.get_endpoint_entry(ticker, endpoint)
        failed = isinstance(entry, dict) and entry.get("status") == "failed"

        if done and file_exists:
            done_count += 1
        else:
            missing_count += 1

        if failed:
            failed_count += 1

    # Partial = has something already downloaded or a persisted failure, but is not complete.
    return (done_count > 0 or failed_count > 0) and missing_count > 0


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
    registry_lock: threading.Lock | None = None,
    rate_limiter: "RateLimiter" | None = None,
    allow_retry_failed: bool = False,
) -> str:
    if registry_lock:
        with registry_lock:
            cooldown = None if not allow_retry_failed else 0
            if not force and registry.should_skip_retry(group, endpoint, cooldown_hours=cooldown):
                return "skip_retry_cooldown"
            if not force and registry.is_done(group, endpoint) and out_path.exists():
                return "skip"
    else:
        cooldown = None if not allow_retry_failed else 0
        if not force and registry.should_skip_retry(group, endpoint, cooldown_hours=cooldown):
            return "skip_retry_cooldown"
        if not force and registry.is_done(group, endpoint) and out_path.exists():
            return "skip"

    if rate_limiter is not None:
        rate_limiter.wait()

    data = fn()
    if not data:
        if registry_lock:
            with registry_lock:
                registry.mark_failed(
                    group,
                    endpoint,
                    terminal=False,
                    reason="nodata_or_empty",
                )
        else:
            registry.mark_failed(
                group,
                endpoint,
                terminal=False,
                reason="nodata_or_empty",
            )
        return "nodata"

    if isinstance(data, list):
        payload = {wrap or "data": data}
    else:
        payload = dict(data)

    payload.update(_meta(group))
    save_json(payload, out_path)
    if registry_lock:
        with registry_lock:
            registry.mark_done(group, endpoint)
    else:
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
    registry_lock: threading.Lock | None = None,
    allow_retry_failed: bool = False,
) -> str:
    endpoint = "prices"
    prices_path = ticker_dir / "prices.json"
    if registry_lock:
        with registry_lock:
            if not force and registry.is_terminal_failure(ticker, endpoint):
                return "skip_terminal"
            cooldown = None if not allow_retry_failed else 0
            if not force and registry.should_skip_retry(ticker, endpoint, cooldown_hours=cooldown):
                return "skip_retry_cooldown"
            if not force and registry.is_done(ticker, endpoint) and prices_path.exists():
                return "skip"
    else:
        if not force and registry.is_terminal_failure(ticker, endpoint):
            return "skip_terminal"
        cooldown = None if not allow_retry_failed else 0
        if not force and registry.should_skip_retry(ticker, endpoint, cooldown_hours=cooldown):
            return "skip_retry_cooldown"
        if not force and registry.is_done(ticker, endpoint) and prices_path.exists():
            return "skip"

    data = yahoo.ohlcv(ticker, start, end)
    if not data or not data.get("data"):
        status_code = getattr(yahoo, "last_status_code", None)
        # 404 typically indicates a non-existent or delisted ticker in Yahoo for that symbol.
        # Mark as a terminal failure to avoid retrying on every run.
        if status_code == 404:
            if registry_lock:
                with registry_lock:
                    registry.mark_failed(
                        ticker,
                        endpoint,
                        terminal=True,
                        reason="yahoo_not_found",
                        status_code=404,
                    )
            else:
                registry.mark_failed(
                    ticker,
                    endpoint,
                    terminal=True,
                    reason="yahoo_not_found",
                    status_code=404,
                )
        else:
            if registry_lock:
                with registry_lock:
                    registry.mark_failed(
                        ticker,
                        endpoint,
                        terminal=False,
                        reason="yahoo_nodata_or_empty",
                        status_code=status_code,
                    )
            else:
                registry.mark_failed(
                    ticker,
                    endpoint,
                    terminal=False,
                    reason="yahoo_nodata_or_empty",
                    status_code=status_code,
                )
        return "nodata"

    payload = {**_meta(ticker), "start": start, "end": end, "source": "yahoo_v8", **data}
    save_json(payload, ticker_dir / "prices.json")
    if registry_lock:
        with registry_lock:
            registry.mark_done(ticker, endpoint)
    else:
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
    prices_only: bool = False,
    registry_lock: threading.Lock | None = None,
    rate_limiter: "RateLimiter" | None = None,
    allow_retry_failed: bool = False,
) -> dict:
    ticker_dir = base_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    r: dict = {}

    r["prices"] = download_prices(
        ticker,
        ticker_dir,
        start,
        end,
        registry,
        force,
        yahoo,
        registry_lock=registry_lock,
        allow_retry_failed=allow_retry_failed,
    )

    # If Yahoo marked prices as a terminal failure (e.g. 404 for a non-existent/delisted ticker),
    # there is no point continuing to call the remaining endpoints for this ticker.
    if r["prices"] == "skip_terminal":
        return r
    if r["prices"] == "nodata" and registry.is_terminal_failure(ticker, "prices"):
        return r

    if prices_only:
        return r

    r["profile"] = fetch_and_save(
        ticker,
        "profile",
        lambda: client.company_profile2(ticker),
        ticker_dir / "profile.json",
        registry,
        force,
        registry_lock=registry_lock,
        rate_limiter=rate_limiter,
        allow_retry_failed=allow_retry_failed,
    )

    r["basic_financials"] = fetch_and_save(
        ticker,
        "basic_financials",
        lambda: client.basic_financials(ticker),
        ticker_dir / "basic_financials.json",
        registry,
        force,
        registry_lock=registry_lock,
        rate_limiter=rate_limiter,
        allow_retry_failed=allow_retry_failed,
    )

    r["financials_reported_annual"] = fetch_and_save(
        ticker,
        "financials_reported_annual",
        lambda: client.financials_as_reported(ticker, freq="annual"),
        ticker_dir / "financials_reported_annual.json",
        registry,
        force,
        registry_lock=registry_lock,
        rate_limiter=rate_limiter,
        allow_retry_failed=allow_retry_failed,
    )

    r["financials_reported_quarterly"] = fetch_and_save(
        ticker,
        "financials_reported_quarterly",
        lambda: client.financials_as_reported(ticker, freq="quarterly"),
        ticker_dir / "financials_reported_quarterly.json",
        registry,
        force,
        registry_lock=registry_lock,
        rate_limiter=rate_limiter,
        allow_retry_failed=allow_retry_failed,
    )

    r["eps_surprises"] = fetch_and_save(
        ticker,
        "eps_surprises",
        lambda: client.eps_surprises(ticker, limit=20),
        ticker_dir / "eps_surprises.json",
        registry,
        force,
        wrap="data",
        registry_lock=registry_lock,
        rate_limiter=rate_limiter,
        allow_retry_failed=allow_retry_failed,
    )

    r["earnings_calendar"] = fetch_and_save(
        ticker,
        "earnings_calendar",
        lambda: client.earnings_calendar(start, end, symbol=ticker),
        ticker_dir / "earnings_calendar.json",
        registry,
        force,
        registry_lock=registry_lock,
        rate_limiter=rate_limiter,
        allow_retry_failed=allow_retry_failed,
    )

    r["recommendation_trends"] = fetch_and_save(
        ticker,
        "recommendation_trends",
        lambda: client.recommendation_trends(ticker),
        ticker_dir / "recommendation_trends.json",
        registry,
        force,
        wrap="data",
        registry_lock=registry_lock,
        rate_limiter=rate_limiter,
        allow_retry_failed=allow_retry_failed,
    )

    r["insider_transactions"] = fetch_and_save(
        ticker,
        "insider_transactions",
        lambda: client.insider_transactions(ticker, start, end),
        ticker_dir / "insider_transactions.json",
        registry,
        force,
        registry_lock=registry_lock,
        rate_limiter=rate_limiter,
        allow_retry_failed=allow_retry_failed,
    )

    r["insider_sentiment"] = fetch_and_save(
        ticker,
        "insider_sentiment",
        lambda: client.insider_sentiment(ticker, start, end),
        ticker_dir / "insider_sentiment.json",
        registry,
        force,
        registry_lock=registry_lock,
        rate_limiter=rate_limiter,
        allow_retry_failed=allow_retry_failed,
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
        registry_lock=registry_lock,
        rate_limiter=rate_limiter,
        allow_retry_failed=allow_retry_failed,
    )

    r["peers"] = fetch_and_save(
        ticker,
        "peers",
        lambda: client.peers(ticker),
        ticker_dir / "peers.json",
        registry,
        force,
        wrap="peers",
        registry_lock=registry_lock,
        rate_limiter=rate_limiter,
        allow_retry_failed=allow_retry_failed,
    )

    r["quote"] = fetch_and_save(
        ticker,
        "quote",
        lambda: client.quote(ticker),
        ticker_dir / "quote.json",
        registry,
        force,
        registry_lock=registry_lock,
        rate_limiter=rate_limiter,
        allow_retry_failed=allow_retry_failed,
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
    prices_only: bool = False,
    max_workers: int | None = None,
    min_interval: float | None = None,
    allow_retry_failed: bool = False,
) -> None:
    end = end or datetime.today().strftime("%Y-%m-%d")
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    yahoo = YahooClient()
    registry = Registry(base_dir)

    if force:
        log.warning("FORCE_DOWNLOAD=True — re-descargando todo")
        registry.clear(delete_file=True)

    log.info("=" * 50)
    log.info("  Finnhub + Yahoo Finance Downloader")
    log.info(f"  Tickers  : {len(tickers)}")
    log.info(f"  Periodo  : {start} -> {end}")
    log.info(f"  Destino  : {base_dir.resolve()}")
    log.info("=" * 50)

    log.info("Descargando macro...")
    download_macro(base_dir, registry, start, end, force, yahoo)

    tickers_requested = list(tickers)
    skipped_terminal = 0
    skipped_complete = 0
    skipped_partial = 0
    tickers_to_process: list[str] = []

    for ticker in tickers_requested:
        if not force and registry.is_terminal_failure(ticker, "prices"):
            skipped_terminal += 1
            continue
        if not force and not allow_retry_failed and _ticker_is_partial(base_dir, ticker, registry, prices_only):
            skipped_partial += 1
            continue
        if not force and _ticker_is_fully_done(base_dir, ticker, registry, prices_only):
            skipped_complete += 1
            continue
        tickers_to_process.append(ticker)

    if skipped_terminal:
        log.info("  Skip terminal prices (404/delisted): %s", skipped_terminal)
    if skipped_complete:
        log.info("  Skip tickers completos en registry: %s", skipped_complete)
    if skipped_partial:
        log.info("  Skip tickers incompletos/parciales: %s", skipped_partial)

    tickers = tickers_to_process

    log.info(f"Descargando {len(tickers)} tickers...")
    summary = {"ok": 0, "partial": 0, "fail": 0}

    registry_lock = threading.Lock()
    max_workers = max_workers or int(os.getenv("DOWNLOAD_MAX_WORKERS", "4"))
    max_workers = max(1, min(max_workers, 16))

    min_interval = min_interval or float(os.getenv("FINNHUB_MIN_INTERVAL", "1.05"))
    rate_limiter = RateLimiter(min_interval=min_interval)

    def _worker(ticker: str):
        local_client = FinnhubClient(api_key)
        local_client.MIN_INTERVAL = 0.0
        local_yahoo = YahooClient()
        return ticker, download_ticker(
            ticker=ticker,
            client=local_client,
            yahoo=local_yahoo,
            base_dir=base_dir,
            registry=registry,
            start=start,
            end=end,
            force=force,
            prices_only=prices_only,
            registry_lock=registry_lock,
            rate_limiter=rate_limiter,
            allow_retry_failed=allow_retry_failed,
        )

    if max_workers == 1:
        iterable = tqdm(tickers, desc="Tickers", unit="ticker") if TQDM_AVAILABLE else tickers
        for ticker in iterable:
            try:
                _ticker, results = _worker(ticker)
                ok = sum(1 for v in results.values() if v.startswith("ok"))
                skip = sum(1 for v in results.values() if isinstance(v, str) and v.startswith("skip"))
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
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker, ticker): ticker for ticker in tickers}
            iterable = as_completed(futures)
            if TQDM_AVAILABLE:
                iterable = tqdm(iterable, total=len(futures), desc="Tickers", unit="ticker")
            for fut in iterable:
                ticker = futures[fut]
                try:
                    _ticker, results = fut.result()
                    ok = sum(1 for v in results.values() if v.startswith("ok"))
                    skip = sum(1 for v in results.values() if isinstance(v, str) and v.startswith("skip"))
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
    log.info(f"  No data  : {summary['fail']}")
    log.info("=" * 50)




class RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            wait = self.min_interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
