"""Construcción de artefactos point-in-time a partir de los agregados raw."""

from __future__ import annotations

import json
import logging
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from environment import Settings
from module.data.universe import first_membership_date, members_at
from module.common.utils import read_parquet, write_parquet

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
    "roa", "rotc", "pretax_margin", "asset_turnover", "inventory_turnover",
    "receivables_turnover", "sga_to_sales", "quick_ratio", "cash_ratio",
    "pe",
    "pb",
    "ps",
    "ev_ebitda",
    "ev_revenue", "pfcf", "ptbv",
    "debt_equity",
    "debt_assets", "debt_capital", "long_debt_equity", "long_debt_assets", "long_debt_capital",
    "net_debt_equity", "net_debt_capital",
    "current_ratio",
    # Series auxiliares para interpretar los múltiplos en el explorador de acciones.
    # No se usan como features por sí solas: conservan el valor conocido a cada fecha PIT.
    "eps",
    "book_value",
    "sales_per_share",
    "ebitda",
    "fcf_per_share",
    "eps_growth_yoy",
    "sales_per_share_growth_yoy",
    "ebitda_growth_yoy",
    "fcf_per_share_growth_yoy",
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
    "median_dollar_volume_21d",
]

# Sesiones que entran en el volumen negociado de referencia. Un mes bursátil: suficiente para que
# un día anómalo no domine y corto para que siga describiendo la liquidez del momento.
DOLLAR_VOLUME_SESSIONS = 21

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
# trimestral, NA.
METRIC_CANDIDATES = {
    "roe": (("roeTTM", "roe"), ("quarterly", "annual")),
    "roic": (("roicTTM", "roic"), ("quarterly", "annual")),
    "net_margin": (("netMargin",), ("quarterly",)),
    "operating_margin": (("operatingMargin",), ("quarterly",)),
    "gross_margin": (("grossMargin",), ("quarterly",)),
    "fcf_margin": (("fcfMargin",), ("quarterly",)),
    "roa": (("roaTTM", "roa"), ("quarterly", "annual")),
    "rotc": (("rotcTTM", "rotc"), ("quarterly", "annual")),
    "pretax_margin": (("pretaxMargin",), ("quarterly", "annual")),
    "asset_turnover": (("assetTurnoverTTM", "assetTurnover"), ("quarterly", "annual")),
    "inventory_turnover": (("inventoryTurnoverTTM", "inventoryTurnover"), ("quarterly", "annual")),
    "receivables_turnover": (("receivablesTurnoverTTM", "receivablesTurnover"), ("quarterly", "annual")),
    "sga_to_sales": (("sgaToSale",), ("quarterly", "annual")),
    "quick_ratio": (("quickRatio",), ("quarterly", "annual")),
    "cash_ratio": (("cashRatio",), ("quarterly", "annual")),
    "pe": (("peTTM", "pe"), ("quarterly", "annual")),
    "pb": (("pb",), ("quarterly", "annual")),
    "ps": (("psTTM", "ps"), ("quarterly", "annual")),
    "ev_ebitda": (("evEbitdaTTM", "evEbitda"), ("quarterly", "annual")),
    "ev_revenue": (("evRevenueTTM", "evRevenue"), ("quarterly", "annual")),
    "pfcf": (("pfcfTTM", "pfcf"), ("quarterly", "annual")),
    "ptbv": (("ptbv",), ("quarterly", "annual")),
    "debt_equity": (("totalDebtToEquity",), ("quarterly", "annual")),
    "debt_assets": (("totalDebtToTotalAsset",), ("quarterly", "annual")),
    "debt_capital": (("totalDebtToTotalCapital",), ("quarterly", "annual")),
    "long_debt_equity": (("longtermDebtTotalEquity",), ("quarterly", "annual")),
    "long_debt_assets": (("longtermDebtTotalAsset",), ("quarterly", "annual")),
    "long_debt_capital": (("longtermDebtTotalCapital",), ("quarterly", "annual")),
    "net_debt_equity": (("netDebtToTotalEquity",), ("quarterly", "annual")),
    "net_debt_capital": (("netDebtToTotalCapital",), ("quarterly", "annual")),
    "current_ratio": (("currentRatio",), ("quarterly", "annual")),
    "eps": (("eps",), ("quarterly", "annual")),
    "book_value": (("bookValue",), ("quarterly", "annual")),
    "sales_per_share": (("salesPerShare",), ("quarterly", "annual")),
    "ebitda": (("ebitda",), ("quarterly", "annual")),
    "fcf_per_share": (("fcfPerShareTTM",), ("quarterly",)),
}


def build_point_in_time_dataset(settings: Settings) -> pd.DataFrame:
    """Genera panel, benchmark y precios PIT sin usar información futura."""
    raw_dir = settings.raw_output_dir
    prices = read_parquet(raw_dir / "prices.parquet", "la ingesta raw")
    metrics = read_parquet(raw_dir / "finnhub_metrics.parquet", "la ingesta raw")
    reports = read_parquet(raw_dir / "report_dates.parquet", "la ingesta raw")

    price_by_ticker = _price_index(prices)
    series_by_ticker = _series_index(metrics)
    reports_by_ticker = _reports_index(reports)
    snapshots = snapshot_dates(settings)
    review_types = _review_types(snapshots, settings)
    rows: list[dict[str, Any]] = []

    for snapshot, review_type in zip(snapshots, review_types, strict=True):
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
    asset_prices = _asset_price_frame(
        price_by_ticker, snapshots, settings.benchmark_ticker, _dollar_volume_index(prices),
    )

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
    """Rejilla de observación: cada snapshot cae en `fin_de_periodo + execution_lag_days`.

    El día de observación NO es fijo: lo define el retardo de publicación. Cerrado un periodo
    (mes si `snapshot_step_months=1`, trimestre si `=3`), los fundamentales tardan `execution_lag_days`
    en estar disponibles; la rejilla observa justo entonces. Así el lag gobierna cuándo se miran los
    datos y barrerlo produce rejillas distintas (información real), en vez de un día de mes arbitrario.
    El PIT sigue garantizado: `_fundamentals_at`/`_observed_price` solo ven lo publicado antes del snapshot.
    """
    # `panel_start_date`, no `data_start_date`: la descarga baja más historia de la que el panel
    # usa (para resolver el universo y alimentar medias móviles), pero la rejilla de snapshots
    # arranca donde siempre, de modo que ampliar la descarga no cambia el periodo evaluado.
    start = max(pd.Timestamp(settings.panel_start_date), first_membership_date())
    end = pd.Timestamp(settings.end_date)
    lag = pd.Timedelta(days=settings.execution_lag_days)
    step = settings.snapshot_step_months
    # Fin del periodo que contiene `start`, avanzando de `step` en `step` meses.
    period_end = start + pd.offsets.MonthEnd(0)
    dates: list[pd.Timestamp] = []
    current = period_end + lag
    while current < start:
        period_end += pd.offsets.MonthEnd(step)
        current = period_end + lag
    while current <= end:
        dates.append(current)
        period_end += pd.offsets.MonthEnd(step)
        current = period_end + lag
    return dates


def _review_types(snapshots: list[pd.Timestamp], settings: Settings) -> list[str]:
    """Marca cada snapshot como revisión fundamental o de precio.

    Un snapshot es "fundamental_quarterly" (reentreno) cuando dista un número entero de pasos
    fundamentales del snapshot ancla dentro de la propia rejilla. La clave es contar en pasos de
    la rejilla (`fundamental_step_months // snapshot_step_months`), no en meses absolutos: así la
    fase de reentreno se alinea SIEMPRE con las fechas reales de la rejilla, sea mensual o
    trimestral. Anclar a un mes absoluto derivado del retardo de publicación fallaba cuando la
    rejilla no pisaba ese mes (p. ej. rejilla trimestral en Ene/Abr/Jul/Oct frente a un ancla en
    febrero: cero reentrenos).
    """
    quarter_start_month = (settings.execution_quarter - 1) * 3 + 1
    # La rejilla ya incorpora execution_lag_days en cada fecha (ver snapshot_dates), así que el
    # ancla NO vuelve a sumarlo: basta el inicio del trimestre de ejecución para fijar la fase.
    anchor = pd.Timestamp(year=settings.execution_year, month=quarter_start_month, day=1)
    step = max(1, settings.fundamental_step_months // settings.snapshot_step_months)
    # Índice del primer snapshot en/tras el ancla; fija la fase de reentreno de la rejilla.
    anchor_index = next((i for i, date in enumerate(snapshots) if date >= anchor), 0)
    return [
        "fundamental_quarterly" if (index - anchor_index) % step == 0 else "price_monthly"
        for index in range(len(snapshots))
    ]


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


def _dollar_volume_index(prices: pd.DataFrame) -> dict[str, tuple[list[pd.Timestamp], list[float]]]:
    """Nocional negociado por sesión y ticker, para dimensionar capacidad.

    El precio esta ajustado por splits y dividendos y el volumen solo por splits, asi que el
    producto es una **aproximación** del nocional: sirve para saber si una orden cabe en el mercado,
    no como dato de mercado citable. La salvedad viaja en `docs/architecture.md`.

    Si la ingesta raw no trae volumen, se devuelve un índice vacío y la columna queda a nulo. Es
    deliberado: un cero se leería como «sin liquidez» y un ausente como «no medido», y solo lo
    segundo es cierto.
    """
    if "volume" not in prices.columns:
        return {}
    frame = prices[["ticker", "date", "adj_close", "volume"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("adj_close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.dropna(subset=["ticker", "date", "adj_close", "volume"], inplace=True)
    frame = frame.loc[frame["adj_close"].gt(0) & frame["volume"].ge(0)]
    frame["dollar_volume"] = frame["adj_close"] * frame["volume"]
    result: dict[str, tuple[list[pd.Timestamp], list[float]]] = {}
    for ticker, group in frame.sort_values("date").groupby("ticker"):
        deduped = group.drop_duplicates("date", keep="last")
        result[ticker] = (list(deduped["date"]), list(deduped["dollar_volume"]))
    return result


def _median_dollar_volume(
    dates: list[pd.Timestamp], values: list[float], target: pd.Timestamp
) -> float | None:
    """Mediana del nocional de las últimas `DOLLAR_VOLUME_SESSIONS` sesiones hasta el snapshot.

    Estrictamente hacia atrás, con el mismo `bisect_right` que `_observed_price`: una sesión
    posterior al snapshot no puede alterar el valor. Mediana y no media porque un único día de
    volumen extraordinario —una entrada en el índice, una fusión— inflaría la capacidad estimada.
    """
    end = bisect_right(dates, target)
    if end <= 0:
        return None
    window = values[max(0, end - DOLLAR_VOLUME_SESSIONS):end]
    return float(np.median(window)) if window else None


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
    dollar_volume_by_ticker: dict[str, tuple[list[pd.Timestamp], list[float]]] | None = None,
) -> pd.DataFrame:
    volume_by_ticker = dollar_volume_by_ticker or {}
    rows: list[dict[str, Any]] = []
    for ticker, (dates, values) in price_by_ticker.items():
        if ticker == benchmark_ticker:
            continue
        volume_dates, volume_values = volume_by_ticker.get(ticker, ([], []))
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
                    "median_dollar_volume_21d": _median_dollar_volume(
                        volume_dates, volume_values, snapshot
                    ),
                }
            )
    return pd.DataFrame(rows, columns=ASSET_PRICE_COLUMNS)


ReportTimeline = tuple[list[pd.Timestamp], list[dict[str, pd.Timestamp]]]


def _reports_index(reports: pd.DataFrame) -> dict[str, ReportTimeline]:
    required = {"ticker", "period", "filed_date"}
    missing = required - set(reports.columns)
    if missing:
        raise ValueError(f"report_dates.parquet no contiene las columnas requeridas: {sorted(missing)}")
    frame = reports.copy()
    frame["period"] = pd.to_datetime(frame["period"])
    frame["filed_date"] = pd.to_datetime(frame["filed_date"])
    frame.dropna(subset=["ticker", "period", "filed_date"], inplace=True)
    result: dict[str, ReportTimeline] = {}
    for ticker, group in frame.sort_values(["filed_date", "period"]).groupby("ticker"):
        filed_dates: list[pd.Timestamp] = []
        best_reports: list[dict[str, pd.Timestamp]] = []
        best: dict[str, pd.Timestamp] | None = None
        for report in group[["period", "filed_date"]].to_dict("records"):
            if best is None or report["period"] > best["period"]:
                best = report
            filed_dates.append(report["filed_date"])
            best_reports.append(best)
        result[ticker] = (filed_dates, best_reports)
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
    reports: ReportTimeline | tuple,
    series: dict[str, dict[str, dict[pd.Timestamp, float]]],
    snapshot: pd.Timestamp,
) -> dict[str, Any]:
    empty = {column: None for column in PANEL_COLUMNS[PANEL_COLUMNS.index("roe"):]} 
    if not reports:
        return empty
    filed_dates, best_reports = reports
    index = bisect_right(filed_dates, snapshot) - 1
    if index < 0:
        return empty
    report = best_reports[index]
    period = report["period"]
    values = {
        column: _series_value(series, period, candidates, frequencies)
        for column, (candidates, frequencies) in METRIC_CANDIDATES.items()
    }
    values["eps_growth_yoy"] = _yoy_growth(series, "eps", period)
    values["sales_per_share_growth_yoy"] = _yoy_growth(series, "salesPerShare", period)
    values["ebitda_growth_yoy"] = _yoy_growth(series, "ebitda", period)
    values["fcf_per_share_growth_yoy"] = _yoy_growth(series, "fcfPerShareTTM", period)
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
