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
    TICKER_ALIASES_CSV,
    Settings,
)
from module.data.ingest.clients import FinnhubClient, YahooClient
from module.data.ingest.edgar import EdgarClient
from module.data.universe import (
    annual_membership_dates,
    is_recycled_ticker,
    members_at,
    membership_span,
)
from module.common.utils import write_json, write_parquet

log = logging.getLogger(__name__)

REPORT_DATE_COLUMNS = ["ticker", "cik", "form", "period", "filed_date"]
FAILURE_COLUMNS = ["ticker", "dataset", "reason", "detail"]

# Clasificación de por qué un ticker del universo histórico no llega al panel. El orden es de
# precedencia y refleja el flujo de `download_raw_data`: un fallo temprano impide llegar a los
# siguientes, de modo que cada ticker se cuenta exactamente una vez y los recuentos suman el
# universo. Sin ese orden, un ticker con varios fallos aparecería en varias categorías.
#
# Importa distinguirlas porque "no resuelve" NO es sinónimo de "la empresa murió": un cambio de
# símbolo, un emisor extranjero que presenta 20-F en vez de 10-K, o una clase de acción con
# puntuación distinta producen el mismo síntoma que una quiebra. Contarlas juntas sobreestima la
# mortalidad y convierte un defecto de resolución en un sesgo de supervivencia aparente.
RESOLUTION_REASONS: tuple[tuple[str, str], ...] = (
    ("recycled_ticker", "Símbolo reutilizado por otra empresa: los precios no son los de la histórica"),
    ("symbol_withdrawn", "El proveedor de precios no reconoce el símbolo (retirado de su API)"),
    ("download_failed", "La descarga de precios falló por red o límite de peticiones: reintentable"),
    ("missing_price", "Sin serie de precios observable"),
    ("missing_cik", "El símbolo no resuelve a ningún CIK de la SEC"),
    ("missing_reports", "CIK resuelto, pero sin informes periódicos"),
    ("no_metric_period_match", "Informes publicados que no casan con ningún periodo de fundamentales"),
    ("missing_fundamentals", "Sin serie de fundamentales"),
    ("download_error", "El ticker abortó con una excepción durante la descarga"),
)

# Qué fila de `download_failures.csv` corresponde a cada categoría. `profile` y `company_news` no
# aparecen: no excluyen del panel, que exige precio, fundamentales e informe publicado.
#
# `symbol_withdrawn` y `download_failed` se separan de `missing_price` a propósito: el primero es
# una propiedad del proveedor (el símbolo existió pero ya no se sirve) y el segundo es una avería
# reintentable. Sumarlos a "sin precios" convierte ambos en mortalidad empresarial aparente.
_FAILURE_TO_REASON = {
    ("ohlcv", "not_found"): "symbol_withdrawn",
    ("ohlcv", "bad_request"): "symbol_withdrawn",
    ("ohlcv", "empty_series"): "missing_price",
    ("ohlcv", "rate_limited"): "download_failed",
    ("ohlcv", "http_error"): "download_failed",
    ("ohlcv", "transport_error"): "download_failed",
    ("ohlcv", "missing"): "missing_price",  # vocabulario anterior, por compatibilidad
    ("edgar", "missing_cik"): "missing_cik",
    ("edgar", "missing_reports"): "missing_reports",
    ("edgar", "no_metric_period_match"): "no_metric_period_match",
    ("basic_financials", "missing"): "missing_fundamentals",
    ("all", "download_error"): "download_error",
}


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
        "Preparando datos crudos: tickers=%s scope=%s",
        len(tickers),
        settings.run_scope,
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
    diagnostics: dict[str, dict[str, Any]] = {}
    alias_cik = _load_alias_cik()
    downloaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started_at = time.perf_counter()

    for ticker_index, ticker in enumerate(tickers, start=1):
        log.info("[%s] datos crudos (%s/%s)", ticker, ticker_index, len(tickers))
        try:
            # Se limpia antes de cada ticker: `_cached_json` puede servir de disco sin llamar al
            # cliente, y entonces el motivo que quedara ahí sería el del ticker anterior.
            yahoo.last_failure_reason = None
            ohlcv = _cached_json(
                _json_path("yahoo", ticker, f"ohlcv_{settings.data_start_date}_{settings.end_date}"),
                lambda: yahoo.ohlcv(ticker, settings.data_start_date, settings.end_date),
            )
            price_rows = (ohlcv or {}).get("data") or []
            has_prices = bool(price_rows)
            if not has_prices:
                dataset = "benchmark_ohlcv" if ticker == settings.benchmark_ticker else "ohlcv"
                # El motivo real, no un `missing` que mete en el mismo saco un símbolo retirado por
                # el proveedor y un corte de red. Sin esta distinción, una avería de infraestructura
                # se contabiliza como mortalidad empresarial.
                reason = yahoo.last_failure_reason or "empty_series"
                failures.append(_failure(ticker, dataset, reason))
                if ticker == settings.benchmark_ticker:
                    continue

            if has_prices:
                first_price_date = min(row["date"] for row in price_rows if row.get("date"))
                if is_recycled_ticker(ticker, first_price_date, settings.data_start_date):
                    recycled_tickers.add(ticker)
                    failures.append(
                        _failure(ticker, "ohlcv", f"recycled_ticker:first_price_date={first_price_date}")
                    )
                    # Un símbolo reciclado conserva historia legítima mientras estuvo en el índice:
                    # se trunca a ese tramo en vez de descartar el ticker entero, que tiraba también
                    # perfil, fundamentales e informes.
                    price_rows = _rows_within_membership(ticker, price_rows)
                    has_prices = bool(price_rows)

            diagnostic = diagnostics.setdefault(ticker, _blank_diagnostic(ticker))
            diagnostic["price_status"] = "ok" if has_prices else (
                yahoo.last_failure_reason or "empty_series"
            )
            diagnostic["price_rows"] = len(price_rows)
            if price_rows:
                observed = sorted(_date_only(row["date"]) for row in price_rows if row.get("date"))
                diagnostic["price_first"], diagnostic["price_last"] = observed[0], observed[-1]

            if has_prices:
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
            diagnostic["has_profile"] = bool(profile)
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
            diagnostic["has_fundamentals"] = bool(metric)
            if metric:
                payload = _strip_meta(metric)
                metric_periods = _metric_periods(payload)
                if metric_periods:
                    ordered = sorted(metric_periods)
                    diagnostic["fundamentals_first"] = ordered[0]
                    diagnostic["fundamentals_last"] = ordered[-1]
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

            cik = cik_by_ticker.get(ticker) or alias_cik.get(ticker)
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

            diagnostic["cik"] = cik or ""
            diagnostic["has_reports"] = bool(ticker_reports)
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
            # Motivo fijo, no `str(exc)`: el mensaje de la excepción es irrepetible y no casaría con
            # `_FAILURE_TO_REASON`, de modo que el ticker se contaba como `in_panel` pese a no haber
            # aportado ningún dato. El detalle va a una columna aparte.
            failures.append(_failure(ticker, "all", "download_error", detail=str(exc)))

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
        _universe_coverage(settings, observations, recycled_tickers, failures),
        output_dir / "universe_coverage.json",
    )
    _write_ticker_diagnostics(settings, diagnostics, failures, recycled_tickers, output_dir)
    log.info(
        "Datos crudos finalizados: tickers=%s informes=%s fallos=%s elapsed_seconds=%.1f",
        len(tickers),
        len(deduped_reports),
        len(failures),
        elapsed_seconds,
    )


DIAGNOSTIC_COLUMNS = [
    "ticker",
    "first_membership",
    "last_membership",
    "price_status",
    "price_rows",
    "price_first",
    "price_last",
    "has_profile",
    "has_fundamentals",
    "fundamentals_first",
    "fundamentals_last",
    "cik",
    "has_reports",
    "in_panel",
    "exclusion_reason",
]


def _blank_diagnostic(ticker: str) -> dict[str, Any]:
    """Fila de diagnóstico con los campos de pertenencia ya resueltos."""
    span = membership_span(ticker)
    return {
        "ticker": ticker,
        "first_membership": span[0].date().isoformat() if span else "",
        "last_membership": span[1].date().isoformat() if span else "",
        "price_status": "not_attempted",
        "price_rows": 0,
        "price_first": "",
        "price_last": "",
        "has_profile": False,
        "has_fundamentals": False,
        "fundamentals_first": "",
        "fundamentals_last": "",
        "cik": "",
        "has_reports": False,
        "in_panel": False,
        "exclusion_reason": "",
    }


def _load_alias_cik() -> dict[str, str]:
    """Mapa `ticker histórico -> CIK` para símbolos que la SEC ya no indexa.

    `company_tickers.json` solo lista emisores vivos bajo su símbolo actual, así que las empresas
    renombradas o absorbidas (AET->CVS, ESRX->Cigna, TWX...) no resuelven. Sin la tabla, un cambio
    de nombre es indistinguible de una desaparición.
    """
    if not TICKER_ALIASES_CSV.exists():
        return {}
    frame = pd.read_csv(TICKER_ALIASES_CSV, dtype=str).fillna("")
    return {
        row.historical_ticker.strip().upper(): row.cik.strip()
        for row in frame.itertuples(index=False)
        if getattr(row, "historical_ticker", "").strip() and getattr(row, "cik", "").strip()
    }


def _exclusion_reason_by_ticker(
    failures: list[dict[str, str]],
    recycled_tickers: set[str],
) -> dict[str, str]:
    """Motivo único de exclusión por ticker, resuelto por precedencia.

    Un ticker puede acumular varios fallos (sin precio y además sin CIK); se queda con el primero
    del flujo, que es el que realmente lo excluyó. Así cada ticker se cuenta una sola vez y los
    recuentos suman el universo.
    """
    reason_by_ticker: dict[str, str] = {ticker: "recycled_ticker" for ticker in recycled_tickers}
    precedence = {reason: index for index, (reason, _) in enumerate(RESOLUTION_REASONS)}
    for failure in failures:
        reason = _FAILURE_TO_REASON.get((failure["dataset"], failure["reason"]))
        if reason is None:
            continue
        current = reason_by_ticker.get(failure["ticker"])
        if current is None or precedence[reason] < precedence[current]:
            reason_by_ticker[failure["ticker"]] = reason
    return reason_by_ticker


def _write_ticker_diagnostics(
    settings: Settings,
    diagnostics: dict[str, dict[str, Any]],
    failures: list[dict[str, str]],
    recycled_tickers: set[str],
    output_dir: Path,
) -> None:
    """Una fila por ticker del universo con qué se pudo descargar y por qué falta lo demás.

    Responde «¿por qué no está X?» sin reconstruirlo a mano cruzando artefactos. Importa que
    incluya a los tickers que no llegan al panel: son justo los que sostienen la discusión sobre el
    sesgo de cobertura, y sin esta tabla su ausencia es un agujero sin explicación.
    """
    reason_by_ticker = _exclusion_reason_by_ticker(failures, recycled_tickers)
    universe = [ticker for ticker in settings.tickers if ticker != settings.benchmark_ticker]
    rows = []
    for ticker in universe:
        row = diagnostics.get(ticker) or _blank_diagnostic(ticker)
        reason = reason_by_ticker.get(ticker, "")
        row["exclusion_reason"] = reason
        row["in_panel"] = not reason
        rows.append(row)
    frame = pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS).sort_values("ticker")
    frame.to_csv(output_dir / "ticker_diagnostics.csv", index=False)
    log.info(
        "Diagnóstico por ticker: %s filas, %s en panel, %s excluidos",
        len(frame),
        int(frame["in_panel"].sum()),
        int((~frame["in_panel"]).sum()),
    )


def _rows_within_membership(ticker: str, price_rows: list[dict]) -> list[dict]:
    """Filas de precio dentro del periodo en que `ticker` estuvo en el índice.

    Para un símbolo reciclado, las filas posteriores a su salida pertenecen a la empresa que
    reutilizó el símbolo y contaminarían el backtest; las anteriores son de la histórica y son
    justo las que el panel necesita.
    """
    span = membership_span(ticker)
    if span is None:
        return []
    first, last = span[0].date().isoformat(), span[1].date().isoformat()
    return [row for row in price_rows if row.get("date") and first <= _date_only(row["date"]) <= last]


def _ticker_resolution(
    settings: Settings,
    failures: list[dict[str, str]],
    recycled_tickers: set[str],
) -> dict[str, Any]:
    """Reparte el universo histórico entre el panel y cada motivo de exclusión.

    Existe para que el tamaño del agujero de cobertura sea una **medida** y no una interpretación:
    saber cuántos tickers no resuelven no dice cuántas empresas murieron, y el reparto por motivo es
    lo único que permite separar mortalidad real de fallo de resolución (ver `RESOLUTION_REASONS`).
    """
    universe = [ticker for ticker in settings.tickers if ticker != settings.benchmark_ticker]
    reason_by_ticker = _exclusion_reason_by_ticker(failures, recycled_tickers)

    counts = {reason: 0 for reason, _ in RESOLUTION_REASONS}
    for ticker in universe:
        reason = reason_by_ticker.get(ticker)
        if reason is not None:
            counts[reason] += 1
    excluded = sum(counts.values())
    return {
        "universe_tickers": len(universe),
        "in_panel": len(universe) - excluded,
        "excluded": excluded,
        "by_reason": [
            {"reason": reason, "description": description, "tickers": counts[reason]}
            for reason, description in RESOLUTION_REASONS
        ],
        "unresolved_sample": sorted(
            ticker for ticker, reason in reason_by_ticker.items()
            if reason == "missing_cik" and ticker in set(universe)
        )[:50],
        "note": (
            "Un símbolo que no resuelve no prueba que la empresa desapareciera: puede haber "
            "cambiado de ticker, presentar formularios de emisor extranjero o llevar una clase de "
            "acción con otra puntuación. `missing_cik` es por tanto una cota superior de la "
            "mortalidad, no una medida de ella."
        ),
    }


def _universe_coverage(
    settings: Settings,
    observations: dict[str, dict[str, set[str]]],
    recycled_tickers: set[str],
    failures: list[dict[str, str]],
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
        "ticker_resolution": _ticker_resolution(settings, failures, recycled_tickers),
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


def _failure(ticker: str, dataset: str, reason: str, detail: str = "") -> dict[str, str]:
    """Una fila de `download_failures.csv`.

    `reason` es vocabulario cerrado (lo consume `_FAILURE_TO_REASON`); `detail` es texto libre
    para el diagnóstico humano y no se interpreta.
    """
    return {"ticker": ticker, "dataset": dataset, "reason": reason, "detail": detail}


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
