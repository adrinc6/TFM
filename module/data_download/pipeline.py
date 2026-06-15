"""Raw data acquisition only: download, validate, store."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Any

import pandas as pd

from clients_download import FinnhubClient, YahooClient
from environment import FINNHUB_API_KEY, FORCE_RAW_DOWNLOAD, RAW_DIR, RAW_JSON_DIR, Settings
from module.common.io import write_parquet

log = logging.getLogger(__name__)


def download_raw_data(settings: Settings) -> None:
    finnhub = FinnhubClient(FINNHUB_API_KEY) if FINNHUB_API_KEY else None
    yahoo = YahooClient()
    tickers = settings.tickers
    RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Preparing raw data for %s tickers (force_download=%s)", len(tickers), FORCE_RAW_DOWNLOAD)

    profiles: list[dict] = []
    metrics: list[dict] = []
    prices: list[dict] = []
    news: list[dict] = []
    downloaded_at = datetime.utcnow().isoformat(timespec="seconds")

    for ticker in tickers:
        log.info("[%s] raw data", ticker)
        profile = _cached_json(
            _json_path("finnhub", ticker, "profile"),
            lambda: _require_finnhub(finnhub).company_profile2(ticker),
        )
        if profile:
            profiles.append({"ticker": ticker, "downloaded_at": profile.get("_downloaded_at", downloaded_at), **_strip_meta(profile)})

        metric = _cached_json(
            _json_path("finnhub", ticker, "basic_financials"),
            lambda: _require_finnhub(finnhub).basic_financials(ticker),
        )
        if metric:
            metrics.append({"ticker": ticker, "downloaded_at": metric.get("_downloaded_at", downloaded_at), "payload": _strip_meta(metric)})

        ohlcv = _cached_json(
            _json_path("yahoo", ticker, f"ohlcv_{settings.data_start_date}_{settings.end_date}"),
            lambda: yahoo.ohlcv(ticker, settings.data_start_date, settings.end_date),
        )
        if ohlcv and ohlcv.get("data"):
            for row in ohlcv["data"]:
                prices.append({"ticker": ticker, **row})

        company_news = _cached_json(
            _json_path("finnhub", ticker, f"company_news_{settings.data_start_date}_{settings.end_date}"),
            lambda: _require_finnhub(finnhub).company_news(ticker, settings.data_start_date, settings.end_date),
        )
        if company_news:
            for item in company_news[:25]:
                news.append({"ticker": ticker, "downloaded_at": item.get("_downloaded_at", downloaded_at), **_strip_meta(item)})

    _require_rows(profiles, "profiles")
    _require_rows(metrics, "metrics")
    _require_rows(prices, "prices")

    write_parquet(pd.DataFrame(profiles), RAW_DIR / "profiles.parquet")
    write_parquet(pd.DataFrame(metrics), RAW_DIR / "finnhub_metrics.parquet")
    write_parquet(pd.DataFrame(prices), RAW_DIR / "prices.parquet")
    if news:
        write_parquet(pd.DataFrame(news), RAW_DIR / "news.parquet")


def _require_rows(rows: list[dict], name: str) -> None:
    if not rows:
        raise RuntimeError(f"No raw {name} downloaded. Failing fast.")


def _cached_json(path: Path, fetcher: Callable[[], Any]) -> Any:
    if path.exists() and not FORCE_RAW_DOWNLOAD:
        log.info("Using cached raw JSON: %s", path)
        return json.loads(path.read_text(encoding="utf-8"))

    data = fetcher()
    if data is None:
        return None
    payload = _with_meta(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("Saved raw JSON: %s", path)
    return payload


def _json_path(source: str, ticker: str, dataset: str) -> Path:
    safe_ticker = ticker.replace("/", "-")
    safe_dataset = dataset.replace("/", "-")
    return RAW_JSON_DIR / source / safe_ticker / f"{safe_dataset}.json"


def _with_meta(data: Any) -> Any:
    downloaded_at = datetime.utcnow().isoformat(timespec="seconds")
    if isinstance(data, dict):
        return {"_downloaded_at": downloaded_at, **data}
    if isinstance(data, list):
        return [{"_downloaded_at": downloaded_at, **item} if isinstance(item, dict) else item for item in data]
    return data


def _strip_meta(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: value for key, value in data.items() if not key.startswith("_")}
    if isinstance(data, list):
        return [_strip_meta(item) for item in data]
    return data


def _require_finnhub(client: FinnhubClient | None) -> FinnhubClient:
    if client is None:
        raise RuntimeError(
            "FINNHUB_API_KEY is required because a Finnhub raw JSON file is missing. "
            "Add the key to .env or set FORCE_RAW_DOWNLOAD=False with existing cache."
        )
    return client
