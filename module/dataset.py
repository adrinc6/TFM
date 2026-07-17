"""Construcción de artefactos point-in-time a partir de los agregados raw."""

from __future__ import annotations

import json
import logging
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from environment import Settings
from module.universe import first_membership_date, members_at
from module.utils import read_parquet, write_parquet

log = logging.getLogger(__name__)

PANEL_COLUMNS = [
    "ticker",
    "snapshot_date",
    "review_type",
    "in_sp500",
    "price",
    "price_as_of_date",
    "price_age_days",
    "price_return_1m",
    "price_return_3m",
    "price_return_6m",
    "price_return_12m",
    "roe",
    "roic",
    "net_margin",
    "operating_margin",
    "gross_margin",
    "fcf_margin",
    "pe",
    "pb",
    "ps",
    "ev_ebitda",
    "debt_equity",
    "current_ratio",
    "eps_growth_yoy",
    "sales_per_share_growth_yoy",
    "fundamental_period",
    "fundamental_filed_date",
    "fundamental_age_days",
]

BENCHMARK_COLUMNS = [
    "snapshot_date",
    "price",
    "price_as_of_date",
    "price_age_days",
    "price_return_1m",
    "price_return_3m",
    "price_return_6m",
    "price_return_12m",
]

ASSET_PRICE_COLUMNS = [
    "ticker",
    "snapshot_date",
    "price",
    "price_as_of_date",
    "price_age_days",
]

# Tolerancia al emparejar un trimestre con el del año anterior: absorbe los cierres
# fiscales que se mueven unos días entre ejercicios, sin llegar a saltarse un trimestre.
YOY_TOLERANCE_DAYS = 45

# Cada métrica declara de qué frecuencias puede leerse, y ese contrato se respeta.
#
# Las TTM y las de balance (`pb`, `debt_equity`, `current_ratio`) admiten el bloque anual: las
# primeras ya son de doce meses y las segundas son un saldo a una fecha, así que el valor no
# cambia de significado según de dónde salga.
#
# Las de flujo NO lo admiten (solo `quarterly`): un cierre anual y su Q4 comparten fecha
# (1999-12-31 es a la vez el Q4 y el cierre del ejercicio), de modo que el fallback mezclaría
# un margen de doce meses con uno de tres dentro del mismo corte transversal. Sin frecuencia
# trimestral, NA. Ver `docs/plan_fases.md` (Hallazgo 3).
METRIC_CANDIDATES = {
    "roe": (("roeTTM", "roe"), ("quarterly", "annual")),
    "roic": (("roicTTM", "roic"), ("quarterly", "annual")),
    "net_margin": (("netMargin",), ("quarterly",)),
    "operating_margin": (("operatingMargin",), ("quarterly",)),
    "gross_margin": (("grossMargin",), ("quarterly",)),
    "fcf_margin": (("fcfMargin",), ("quarterly",)),
    "pe": (("peTTM", "pe"), ("quarterly", "annual")),
    "pb": (("pb",), ("quarterly", "annual")),
    "ps": (("psTTM", "ps"), ("quarterly", "annual")),
    "ev_ebitda": (("evEbitdaTTM", "evEbitda"), ("quarterly", "annual")),
    "debt_equity": (("totalDebtToEquity",), ("quarterly", "annual")),
    "current_ratio": (("currentRatio",), ("quarterly", "annual")),
}


def build_point_in_time_dataset(settings: Settings) -> pd.DataFrame:
    """Genera panel, benchmark y precios PIT sin usar información futura."""
    raw_dir = settings.raw_output_dir
    prices = read_parquet(raw_dir / "prices.parquet", "RUN_MODE='download'")
    metrics = read_parquet(raw_dir / "finnhub_metrics.parquet", "RUN_MODE='download'")
    reports = read_parquet(raw_dir / "report_dates.parquet", "RUN_MODE='download'")

    price_by_ticker = _price_index(prices)
    series_by_ticker = _series_index(metrics)
    reports_by_ticker = _reports_index(reports)
    snapshots = snapshot_dates(settings)
    rows: list[dict[str, Any]] = []

    for snapshot in snapshots:
        review_type = _review_type(snapshot, settings)
        for ticker in members_at(snapshot):
            price_dates, price_values = price_by_ticker.get(ticker, ([], []))
            price, as_of, age_days = _observed_price(price_dates, price_values, snapshot)
            if price is None:
                continue
            row: dict[str, Any] = {
                "ticker": ticker,
                "snapshot_date": snapshot.date().isoformat(),
                "review_type": review_type,
                "in_sp500": True,
                "price": price,
                "price_as_of_date": as_of.date().isoformat() if as_of is not None else None,
                "price_age_days": age_days,
                "price_return_1m": _trailing_return(price_dates, price_values, snapshot, 1),
                "price_return_3m": _trailing_return(price_dates, price_values, snapshot, 3),
                "price_return_6m": _trailing_return(price_dates, price_values, snapshot, 6),
                "price_return_12m": _trailing_return(price_dates, price_values, snapshot, 12),
            }
            row.update(
                _fundamentals_at(
                    reports_by_ticker.get(ticker, []), series_by_ticker.get(ticker, {}), snapshot
                )
            )
            rows.append(row)

    panel = pd.DataFrame(rows, columns=PANEL_COLUMNS)
    if panel.empty:
        raise RuntimeError("El panel point-in-time quedó vacío. Revisa los agregados raw y su alcance.")
    panel.sort_values(["snapshot_date", "ticker"], inplace=True, ignore_index=True)

    benchmark = _benchmark_frame(price_by_ticker.get(settings.benchmark_ticker), snapshots)
    if benchmark.empty:
        log.warning("No hay precios PIT para el benchmark %s.", settings.benchmark_ticker)
    asset_prices = _asset_price_frame(price_by_ticker, snapshots, settings.benchmark_ticker)

    output_dir = settings.processed_output_dir
    write_parquet(panel, output_dir / "panel_point_in_time.parquet")
    write_parquet(benchmark, output_dir / "benchmark_point_in_time.parquet")
    write_parquet(asset_prices, output_dir / "asset_price_point_in_time.parquet")
    log.info(
        "Panel point-in-time: rows=%s snapshots=%s tickers=%s output=%s",
        len(panel),
        panel["snapshot_date"].nunique(),
        panel["ticker"].nunique(),
        output_dir / "panel_point_in_time.parquet",
    )
    return panel


def snapshot_dates(settings: Settings) -> list[pd.Timestamp]:
    start = max(pd.Timestamp(settings.data_start_date), first_membership_date())
    end = pd.Timestamp(settings.end_date)
    first = start.replace(day=min(settings.snapshot_day, start.days_in_month))
    if first < start:
        first += pd.DateOffset(months=1)
    dates: list[pd.Timestamp] = []
    current = first
    while current <= end:
        dates.append(current)
        next_month = current + pd.DateOffset(months=settings.snapshot_step_months)
        current = next_month.replace(day=min(settings.snapshot_day, next_month.days_in_month))
    return dates


def _review_type(snapshot: pd.Timestamp, settings: Settings) -> str:
    quarter_start_month = (settings.execution_quarter - 1) * 3 + 1
    anchor = pd.Timestamp(year=settings.execution_year, month=quarter_start_month, day=1)
    anchor += pd.Timedelta(days=settings.execution_lag_days)
    month_distance = (snapshot.year - anchor.year) * 12 + snapshot.month - anchor.month
    return (
        "fundamental_quarterly"
        if month_distance % settings.fundamental_step_months == 0
        else "price_monthly"
    )


def _price_index(prices: pd.DataFrame) -> dict[str, tuple[list[pd.Timestamp], list[float]]]:
    frame = prices.copy()
    required = {"ticker", "date", "adj_close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prices.parquet no contiene las columnas requeridas: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
    frame.dropna(subset=["ticker", "date", "adj_close"], inplace=True)
    result: dict[str, tuple[list[pd.Timestamp], list[float]]] = {}
    for ticker, group in frame.sort_values("date").groupby("ticker"):
        deduped = group.drop_duplicates("date", keep="last")
        result[ticker] = (list(deduped["date"]), list(deduped["adj_close"]))
    return result


def _benchmark_frame(
    series: tuple[list[pd.Timestamp], list[float]] | None, snapshots: list[pd.Timestamp]
) -> pd.DataFrame:
    if series is None:
        return pd.DataFrame(columns=BENCHMARK_COLUMNS)
    dates, values = series
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        price, as_of, age_days = _observed_price(dates, values, snapshot)
        if price is None:
            continue
        rows.append(
            {
                "snapshot_date": snapshot.date().isoformat(),
                "price": price,
                "price_as_of_date": as_of.date().isoformat() if as_of is not None else None,
                "price_age_days": age_days,
                "price_return_1m": _trailing_return(dates, values, snapshot, 1),
                "price_return_3m": _trailing_return(dates, values, snapshot, 3),
                "price_return_6m": _trailing_return(dates, values, snapshot, 6),
                "price_return_12m": _trailing_return(dates, values, snapshot, 12),
            }
        )
    return pd.DataFrame(rows, columns=BENCHMARK_COLUMNS)


def _asset_price_frame(
    price_by_ticker: dict[str, tuple[list[pd.Timestamp], list[float]]],
    snapshots: list[pd.Timestamp],
    benchmark_ticker: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker, (dates, values) in price_by_ticker.items():
        if ticker == benchmark_ticker:
            continue
        for snapshot in snapshots:
            price, as_of, age_days = _observed_price(dates, values, snapshot)
            if price is None:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "snapshot_date": snapshot.date().isoformat(),
                    "price": price,
                    "price_as_of_date": as_of.date().isoformat() if as_of is not None else None,
                    "price_age_days": age_days,
                }
            )
    return pd.DataFrame(rows, columns=ASSET_PRICE_COLUMNS)


def _reports_index(reports: pd.DataFrame) -> dict[str, list[dict[str, pd.Timestamp]]]:
    required = {"ticker", "period", "filed_date"}
    missing = required - set(reports.columns)
    if missing:
        raise ValueError(f"report_dates.parquet no contiene las columnas requeridas: {sorted(missing)}")
    frame = reports.copy()
    frame["period"] = pd.to_datetime(frame["period"])
    frame["filed_date"] = pd.to_datetime(frame["filed_date"])
    frame.dropna(subset=["ticker", "period", "filed_date"], inplace=True)
    result: dict[str, list[dict[str, pd.Timestamp]]] = {}
    for ticker, group in frame.sort_values(["filed_date", "period"]).groupby("ticker"):
        result[ticker] = group[["period", "filed_date"]].to_dict("records")
    return result


def _series_index(metrics: pd.DataFrame) -> dict[str, dict[str, dict[str, dict[pd.Timestamp, float]]]]:
    required = {"ticker", "payload"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"finnhub_metrics.parquet no contiene las columnas requeridas: {sorted(missing)}")
    result: dict[str, dict[str, dict[str, dict[pd.Timestamp, float]]]] = {}
    for row in metrics[["ticker", "payload"]].itertuples(index=False):
        payload = _as_mapping(row.payload)
        ticker_series: dict[str, dict[str, dict[pd.Timestamp, float]]] = {}
        for frequency in ("quarterly", "annual"):
            frequency_series = (_as_mapping(payload.get("series")).get(frequency) or {})
            parsed_frequency: dict[str, dict[pd.Timestamp, float]] = {}
            for metric, values in _as_mapping(frequency_series).items():
                parsed_values: dict[pd.Timestamp, float] = {}
                if values is None:
                    continue
                for value in values:
                    item = _as_mapping(value)
                    if item.get("period") is None or item.get("v") is None:
                        continue
                    numeric = pd.to_numeric(item["v"], errors="coerce")
                    if pd.notna(numeric):
                        parsed_values[pd.Timestamp(item["period"])] = float(numeric)
                if parsed_values:
                    parsed_frequency[metric] = parsed_values
            ticker_series[frequency] = parsed_frequency
        result[row.ticker] = ticker_series
    return result


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, Mapping) else {}


def _fundamentals_at(
    reports: list[dict[str, pd.Timestamp]],
    series: dict[str, dict[str, dict[pd.Timestamp, float]]],
    snapshot: pd.Timestamp,
) -> dict[str, Any]:
    empty = {column: None for column in PANEL_COLUMNS[PANEL_COLUMNS.index("roe"):]} 
    published = [report for report in reports if report["filed_date"] <= snapshot]
    if not published:
        return empty
    report = max(published, key=lambda item: item["period"])
    period = report["period"]
    values = {
        column: _series_value(series, period, candidates, frequencies)
        for column, (candidates, frequencies) in METRIC_CANDIDATES.items()
    }
    values["eps_growth_yoy"] = _yoy_growth(series, "eps", period)
    values["sales_per_share_growth_yoy"] = _yoy_growth(series, "salesPerShare", period)
    values.update(
        {
            "fundamental_period": period.date().isoformat(),
            "fundamental_filed_date": report["filed_date"].date().isoformat(),
            "fundamental_age_days": int((snapshot - report["filed_date"]).days),
        }
    )
    return values


def _series_value(
    series: dict[str, dict[str, dict[pd.Timestamp, float]]],
    period: pd.Timestamp,
    candidates: Iterable[str],
    frequencies: Iterable[str],
) -> float | None:
    """Primer valor disponible para `period`, buscando solo en las frecuencias permitidas."""
    for frequency in frequencies:
        for metric in candidates:
            value = series.get(frequency, {}).get(metric, {}).get(period)
            if value is not None:
                return value
    return None


def _yoy_growth(
    series: dict[str, dict[str, dict[pd.Timestamp, float]]], metric: str, period: pd.Timestamp
) -> float | None:
    """Crecimiento frente al mismo trimestre del año anterior, emparejado POR FECHA.

    La pareja se busca alrededor de `period - 12 meses`, no cuatro posiciones atrás: las series
    de Finnhub tienen huecos, y contar posiciones compara contra un trimestre arbitrario (por
    ejemplo 15 meses atrás) etiquetándolo como interanual. Si no hay un trimestre dentro de la
    tolerancia, se devuelve None: perder la fila es preferible a inventar la magnitud.
    """
    values = series.get("quarterly", {}).get(metric, {})
    if period not in values:
        return None
    target = period - pd.DateOffset(months=12)
    candidates = [
        candidate
        for candidate in values
        if abs((candidate - target).days) <= YOY_TOLERANCE_DAYS
    ]
    if not candidates:
        return None
    match = min(candidates, key=lambda candidate: abs((candidate - target).days))
    previous = values[match]
    if previous == 0:
        return None
    return values[period] / previous - 1


def _observed_price(
    dates: list[pd.Timestamp], values: list[float], target: pd.Timestamp
) -> tuple[float | None, pd.Timestamp | None, int | None]:
    index = bisect_right(dates, target) - 1
    if index < 0:
        return None, None, None
    as_of = pd.Timestamp(dates[index])
    return float(values[index]), as_of, int((target.normalize() - as_of.normalize()).days)


def _trailing_return(
    dates: list[pd.Timestamp], values: list[float], snapshot: pd.Timestamp, months: int
) -> float | None:
    current, _, _ = _observed_price(dates, values, snapshot)
    previous, _, _ = _observed_price(dates, values, snapshot - pd.DateOffset(months=months))
    if current is None or previous in (None, 0):
        return None
    return current / previous - 1
