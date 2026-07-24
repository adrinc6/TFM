"""Adquisición, validación y consolidación de datos crudos."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from environment import (
    EDGAR_USER_AGENT,
    FINNHUB_API_KEY,
    RAW_JSON_DIR,
    Settings,
)
from module.data.ingest.clients import FinnhubClient, YahooClient
from module.data.ingest.edgar import EdgarClient
from module.data.universe import annual_membership_dates, is_recycled_ticker, members_at
from module.common.utils import write_json, write_parquet

log = logging.getLogger(__name__)

REPORT_DATE_COLUMNS = ["ticker", "cik", "form", "period", "filed_date"]
FAILURE_COLUMNS = ["ticker", "dataset", "reason"]


def download_raw_data(settings: Settings) -> None:
    """Descarga el universo solicitado y escribe agregados en su alcance aislado."""
    finnhub = FinnhubClient(FINNHUB_API_KEY) if FINNHUB_API_KEY else None
    yahoo = YahooClient()
    edgar = EdgarClient(EDGAR_USER_AGENT, RAW_JSON_DIR / "edgar", force_download=False)
    cik_by_ticker = edgar.ticker_to_cik()
    tickers = settings.tickers
    output_dir = settings.raw_output_dir
    RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)

    log.info(
        "Preparando datos crudos: tickers=%s scope=%s force_download=%s",
        len(tickers),
        settings.run_scope,
        False,
    )
    profiles: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    prices: list[dict[str, Any]] = []
    news: list[dict[str, Any]] = []
    report_dates: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    coverage = {
        "profiles": 0,
        "metrics": 0,
        "prices": 0,
        "news": 0,
        "report_dates": 0,
        "benchmark_price_rows": 0,
    }
    observations: dict[str, dict[str, set[str]]] = {}
    recycled_tickers: set[str] = set()
    downloaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started_at = time.perf_counter()

    for ticker_index, ticker in enumerate(tickers, start=1):
        log.info("[%s] datos crudos (%s/%s)", ticker, ticker_index, len(tickers))
        try:
            ohlcv = _cached_json(
                _json_path("yahoo", ticker, f"ohlcv_{settings.data_start_date}_{settings.end_date}"),
                lambda: yahoo.ohlcv(ticker, settings.data_start_date, settings.end_date),
            )
            price_rows = (ohlcv or {}).get("data") or []
            if not price_rows:
                dataset = "benchmark_ohlcv" if ticker == settings.benchmark_ticker else "ohlcv"
                failures.append(_failure(ticker, dataset, "missing"))
                continue

            first_price_date = min(row["date"] for row in price_rows if row.get("date"))
            if is_recycled_ticker(ticker, first_price_date):
                recycled_tickers.add(ticker)
                failures.append(
                    _failure(ticker, "ohlcv", f"recycled_ticker:first_price_date={first_price_date}")
                )
                continue

            coverage["prices"] += 1
            prices.extend({"ticker": ticker, **row} for row in price_rows)
            if ticker == settings.benchmark_ticker:
                coverage["benchmark_price_rows"] = len(price_rows)
                # SPY es una serie de referencia, no una empresa del universo: no
                # necesita perfil, fundamentales, CIK ni noticias para el pipeline.
                continue
            observations[ticker] = {
                "price_dates": {_date_only(row["date"]) for row in price_rows if row.get("date")},
                "matched_filed_dates": set(),
            }

            profile = _cached_json(
                _json_path("finnhub", ticker, "profile"),
                lambda: _require_finnhub(finnhub).company_profile2(ticker),
            )
            if profile:
                coverage["profiles"] += 1
                profiles.append(
                    {
                        "ticker": ticker,
                        "downloaded_at": profile.get("_downloaded_at", downloaded_at),
                        **_strip_meta(profile),
                    }
                )
            else:
                failures.append(_failure(ticker, "profile", "missing"))

            metric = _cached_json(
                _json_path("finnhub", ticker, "basic_financials"),
                lambda: _require_finnhub(finnhub).basic_financials(ticker),
            )
            metric_periods: set[str] = set()
            if metric:
                payload = _strip_meta(metric)
                metric_periods = _metric_periods(payload)
                coverage["metrics"] += 1
                metrics.append(
                    {
                        "ticker": ticker,
                        "downloaded_at": metric.get("_downloaded_at", downloaded_at),
                        "payload": payload,
                    }
                )
            else:
                failures.append(_failure(ticker, "basic_financials", "missing"))

            cik = cik_by_ticker.get(ticker)
            ticker_reports = edgar.report_dates(ticker, cik) if cik else []
            if not ticker_reports:
                fallback_cik = edgar.lookup_cik(ticker)
                if fallback_cik and fallback_cik != cik:
                    fallback_reports = edgar.report_dates(ticker, fallback_cik)
                    if fallback_reports:
                        log.warning(
                            "[%s] CIK %s sin informes periódicos; se usa CIK %s de búsqueda SEC",
                            ticker,
                            cik,
                            fallback_cik,
                        )
                        cik = fallback_cik
                        ticker_reports = fallback_reports

            if not cik:
                failures.append(_failure(ticker, "edgar", "missing_cik"))
            elif not ticker_reports:
                failures.append(_failure(ticker, "edgar", "missing_reports"))
            else:
                matched_count = 0
                for report in ticker_reports:
                    period = _date_only(report["period"])
                    filed_date = _date_only(report["filed_date"])
                    report_dates.append(
                        {
                            "ticker": ticker,
                            "cik": cik,
                            "form": report["form"],
                            "period": period,
                            "filed_date": filed_date,
                        }
                    )
                    if period in metric_periods:
                        observations[ticker]["matched_filed_dates"].add(filed_date)
                        matched_count += 1
                coverage["report_dates"] += matched_count
                if not matched_count:
                    failures.append(_failure(ticker, "edgar", "no_metric_period_match"))

            company_news = _cached_json(
                _json_path("finnhub", ticker, f"company_news_{settings.data_start_date}_{settings.end_date}"),
                lambda: _require_finnhub(finnhub).company_news(
                    ticker, settings.data_start_date, settings.end_date
                ),
            )
            if company_news:
                coverage["news"] += 1
                news.extend(
                    {
                        "ticker": ticker,
                        "downloaded_at": item.get("_downloaded_at", downloaded_at),
                        **_strip_meta(item),
                    }
                    for item in company_news[:25]
                )
            else:
                failures.append(_failure(ticker, "company_news", "missing"))
        except Exception as exc:
            log.exception("[%s] fallo al descargar datos crudos", ticker)
            failures.append(_failure(ticker, "all", str(exc)))

    benchmark_rows = [row for row in prices if row["ticker"] == settings.benchmark_ticker]
    if not benchmark_rows:
        raise RuntimeError(
            f"No se descargaron precios del benchmark {settings.benchmark_ticker}. "
            "La preparación de features requiere su serie OHLCV."
        )

    _require_rows(profiles, "profiles")
    _require_rows(metrics, "finnhub_metrics")
    _require_rows(prices, "prices")

    deduped_reports = (
        pd.DataFrame(report_dates, columns=REPORT_DATE_COLUMNS)
        .sort_values(["ticker", "period", "filed_date"])
        .drop_duplicates(subset=["ticker", "period"], keep="first")
    )
    write_parquet(pd.DataFrame(profiles), output_dir / "profiles.parquet")
    write_parquet(pd.DataFrame(metrics), output_dir / "finnhub_metrics.parquet")
    write_parquet(pd.DataFrame(prices), output_dir / "prices.parquet")
    write_parquet(deduped_reports, output_dir / "report_dates.parquet")
    if news:
        write_parquet(pd.DataFrame(news), output_dir / "news.parquet")

    elapsed_seconds = time.perf_counter() - started_at
    write_json(
        {
            "run_scope": settings.run_scope,
            "tickers": len(tickers),
            "coverage": coverage,
            "benchmark": {
                "ticker": settings.benchmark_ticker,
                "available": True,
                "price_rows": len(benchmark_rows),
                "first_date": min(_date_only(row["date"]) for row in benchmark_rows),
                "last_date": max(_date_only(row["date"]) for row in benchmark_rows),
            },
            "failure_count": len(failures),
            "elapsed_seconds": round(elapsed_seconds, 1),
        },
        output_dir / "download_coverage.json",
    )
    pd.DataFrame(failures, columns=FAILURE_COLUMNS).to_csv(
        output_dir / "download_failures.csv", index=False
    )
    write_json(
        _universe_coverage(settings, observations, recycled_tickers),
        output_dir / "universe_coverage.json",
    )
    log.info(
        "Datos crudos finalizados: tickers=%s informes=%s fallos=%s elapsed_seconds=%.1f",
        len(tickers),
        len(deduped_reports),
        len(failures),
        elapsed_seconds,
    )


def _universe_coverage(
    settings: Settings,
    observations: dict[str, dict[str, set[str]]],
    recycled_tickers: set[str],
) -> dict[str, Any]:
    """Mide la cobertura histórica observable; la muestra dev no es representativa."""
    if settings.dev_mode:
        return {
            "run_scope": "dev",
            "representative": False,
            "sampled_tickers": settings.tickers,
            "message": "La cobertura de desarrollo no representa el universo histórico completo.",
            "years": [],
        }

    years: list[dict[str, Any]] = []
    for as_of in annual_membership_dates():
        members = members_at(as_of)
        reasons = {"recycled_ticker": 0, "missing_price": 0, "missing_fundamental_or_report": 0}
        eligible = 0
        as_of_date = as_of.date().isoformat()
        for ticker in members:
            if ticker in recycled_tickers:
                reasons["recycled_ticker"] += 1
                continue
            ticker_observations = observations.get(ticker, {})
            if not any(date <= as_of_date for date in ticker_observations.get("price_dates", set())):
                reasons["missing_price"] += 1
                continue
            if not any(
                date <= as_of_date
                for date in ticker_observations.get("matched_filed_dates", set())
            ):
                reasons["missing_fundamental_or_report"] += 1
                continue
            eligible += 1
        member_count = len(members)
        years.append(
            {
                "year": as_of.year,
                "as_of_date": as_of_date,
                "sp500_members": member_count,
                "panel_eligible_tickers": eligible,
                "coverage_pct": round(100 * eligible / member_count, 2) if member_count else 0.0,
                "exclusions": reasons,
            }
        )
    return {
        "run_scope": "full",
        "representative": True,
        "eligibility": "precio observable y periodo fundamental con informe SEC publicado",
        "years": years,
    }


def _metric_periods(payload: dict[str, Any]) -> set[str]:
    periods: set[str] = set()
    for frequency in (payload.get("series") or {}).values():
        if not isinstance(frequency, dict):
            continue
        for values in frequency.values():
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, dict) and value.get("period"):
                    periods.add(_date_only(value["period"]))
    return periods


def _date_only(value: Any) -> str:
    return str(value).split(" ", maxsplit=1)[0]


def _failure(ticker: str, dataset: str, reason: str) -> dict[str, str]:
    return {"ticker": ticker, "dataset": dataset, "reason": reason}


def _require_rows(rows: list[dict[str, Any]], name: str) -> None:
    if not rows:
        raise RuntimeError(f"No se descargaron filas para {name}.")


def _cached_json(path: Path, fetcher: Callable[[], Any]) -> Any:
    if path.exists():
        log.info("Usando caché JSON: %s", path)
        return json.loads(path.read_text(encoding="utf-8"))

    data = fetcher()
    if data is None:
        return None
    payload = _with_meta(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("Guardada caché JSON: %s", path)
    return payload


def _json_path(source: str, ticker: str, dataset: str) -> Path:
    return RAW_JSON_DIR / source / ticker.replace("/", "-") / f"{dataset.replace('/', '-')}.json"


def _with_meta(data: Any) -> Any:
    downloaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if isinstance(data, dict):
        return {"_downloaded_at": downloaded_at, **data}
    if isinstance(data, list):
        return [
            {"_downloaded_at": downloaded_at, **item} if isinstance(item, dict) else item
            for item in data
        ]
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
            "FINNHUB_API_KEY es obligatoria si falta una caché JSON de Finnhub."
        )
    return client
