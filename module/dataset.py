"""Build auditable ticker x snapshot_date rows."""

from __future__ import annotations

import logging
from ast import literal_eval
from bisect import bisect_right
from collections.abc import Iterable

import pandas as pd

from environment import MASTER_DIR, RAW_DIR, Settings
from module.utils import read_parquet, write_parquet

log = logging.getLogger(__name__)


METRIC_MAP = {
    "pe": "peBasicExclExtraTTM",
    "forward_pe": "forwardPE",
    "peg": "pegRatio",
    "roe": "roeTTM",
    "roic": "roicTTM",
    "gross_margin": "grossMarginTTM",
    "operating_margin": "operatingMarginTTM",
    "net_margin": "netProfitMarginTTM",
    "debt_equity": "totalDebt/totalEquityQuarterly",
    "revenue_growth": "revenueGrowthTTMYoy",
    "eps_growth": "epsGrowthTTMYoy",
    "fcf_margin": "fcfMarginTTM",
}

HISTORICAL_SERIES_MAP = {
    "pe": "pe",
    "roe": "roe",
    "roic": "roic",
    "gross_margin": "grossMargin",
    "operating_margin": "operatingMargin",
    "net_margin": "netMargin",
    "debt_equity": "totalDebtToEquity",
    "fcf_margin": "fcfMargin",
}


def build_master_dataset(settings: Settings) -> pd.DataFrame:
    prices = read_parquet(RAW_DIR / "prices.parquet")
    metrics = read_parquet(RAW_DIR / "finnhub_metrics.parquet")
    profiles = read_parquet(RAW_DIR / "profiles.parquet")

    prices["date"] = pd.to_datetime(prices["date"])
    snapshots = _snapshot_grid(settings)
    fundamental_dates = _fundamental_date_set(settings)
    rows: list[dict] = []
    price_by_ticker = {
        ticker: _ticker_price_series(group)
        for ticker, group in prices.groupby("ticker")
    }
    benchmark_dates, _ = price_by_ticker.get(settings.benchmark_ticker, ([], []))

    lag_days = settings.fundamental_publication_lag_days
    publication_dates = _load_publication_dates()
    payload_by_ticker = {
        row.ticker: _metric_cache(_parse_payload(row.payload), lag_days, publication_dates.get(row.ticker, {}))
        for row in metrics.itertuples(index=False)
    }
    profile_by_ticker = profiles.drop_duplicates("ticker").set_index("ticker").to_dict("index")

    for snapshot_date in snapshots:
        log.info("Building monthly universe snapshot %s", snapshot_date.date().isoformat())
        if _latest_price(benchmark_dates, [], snapshot_date, require_value=False) is None:
            continue
        for ticker in settings.investable_tickers:
            price_dates, price_values = price_by_ticker.get(ticker, ([], []))
            current_price = _latest_price(price_dates, price_values, snapshot_date)
            if current_price is None:
                continue
            row = {
                "ticker": ticker,
                "snapshot_date": snapshot_date.date().isoformat(),
                "review_type": _review_type(snapshot_date, fundamental_dates),
                "price": current_price,
                "price_return_1m": _trailing_return(price_dates, price_values, snapshot_date, months=1),
                "price_return_3m": _trailing_return(price_dates, price_values, snapshot_date, months=3),
                "price_return_6m": _trailing_return(price_dates, price_values, snapshot_date, months=6),
                "price_return_12m": _trailing_return(price_dates, price_values, snapshot_date, months=12),
                "sector": profile_by_ticker.get(ticker, {}).get("finnhubIndustry", "Unknown"),
            }
            row.update(
                _payload_to_metrics(
                    payload_by_ticker.get(ticker, {}),
                    snapshot_date,
                    current_price,
                    price_dates,
                    price_values,
                )
            )
            rows.append(row)

    master = pd.DataFrame(rows)
    if master.empty:
        raise RuntimeError("Master dataset is empty. Check raw data and date range.")
    write_parquet(master, MASTER_DIR / "master_point_in_time.parquet")
    log.info(
        "Master dataset rows=%s snapshots=%s tickers=%s",
        len(master),
        master["snapshot_date"].nunique(),
        master["ticker"].nunique(),
    )
    return master


def _phased_dates(anchor: pd.Timestamp, panel_start: pd.Timestamp, end: pd.Timestamp, step_months: int) -> list[pd.Timestamp]:
    """Fechas ancladas al DÍA del `anchor` (p.ej. día 15), separadas `step_months`, cubriendo
    [panel_start, end] tanto hacia adelante como hacia atrás desde el ancla. Anclar a la fecha de
    publicación hace que el ancla y los refrescos trimestrales caigan en snapshots reales."""
    dates: list[pd.Timestamp] = []
    current = anchor
    while current <= end:
        dates.append(current)
        current = current + pd.DateOffset(months=step_months)
    current = anchor - pd.DateOffset(months=step_months)
    while current >= panel_start:
        dates.append(current)
        current = current - pd.DateOffset(months=step_months)
    return sorted(dates)


def _snapshot_grid(settings: Settings) -> list[pd.Timestamp]:
    """Rejilla MENSUAL de revisión, fasada a la fecha ancla. Cada mes re-precia los fundamentales
    vigentes; cada trimestre (ver _fundamental_date_set) hay fundamentales nuevos."""
    anchor = pd.Timestamp(settings.eval_start_date)
    panel_start = _snapshot_start_date(settings)
    end = pd.Timestamp(settings.end_date)
    grid = _phased_dates(anchor, panel_start, end, step_months=1)
    log.info(
        "Master snapshot window anchor=%s start=%s end=%s raw_data_start=%s max_walk_forward_years=%s snapshots=%s",
        anchor.date().isoformat(),
        panel_start.date().isoformat(),
        end.date().isoformat(),
        pd.Timestamp(settings.data_start_date).date().isoformat(),
        settings.max_walk_forward_training_years,
        len(grid),
    )
    return grid


def _fundamental_date_set(settings: Settings) -> set:
    """Fechas TRIMESTRALES (refresco de fundamentales), fasadas al ancla: son las únicas con datos
    nuevos y los puntos donde tiene sentido reentrenar."""
    anchor = pd.Timestamp(settings.eval_start_date)
    panel_start = _snapshot_start_date(settings)
    end = pd.Timestamp(settings.end_date)
    return {date.date() for date in _phased_dates(anchor, panel_start, end, step_months=3)}


def _snapshot_start_date(settings: Settings) -> pd.Timestamp:
    raw_start = pd.Timestamp(settings.data_start_date)
    if not settings.walk_forward_scoring:
        return raw_start
    needed_start = pd.Timestamp(settings.eval_start_date) - pd.DateOffset(years=settings.max_walk_forward_training_years)
    return max(raw_start, needed_start)


def _review_type(snapshot_date: pd.Timestamp, fundamental_dates: set) -> str:
    if snapshot_date.date() in fundamental_dates:
        return "quarterly_fundamental_review"
    return "monthly_price_update"


def _parse_payload(payload: object) -> dict:
    if isinstance(payload, str):
        payload = literal_eval(payload)
    return payload if isinstance(payload, dict) else {}


def _load_publication_dates() -> dict[str, dict[pd.Timestamp, pd.Timestamp]]:
    """Mapa {ticker: {fin_de_periodo -> fecha_de_publicación_real}} desde report_dates.parquet (si
    existe). Se usa para decidir la observabilidad de cada fundamental por su fecha de publicación
    REAL; si el fichero no existe o falta un periodo, el dataset cae al retardo fijo."""
    path = RAW_DIR / "report_dates.parquet"
    if not path.exists():
        return {}
    table = read_parquet(path)
    if table.empty or not {"ticker", "period", "filed_date"}.issubset(table.columns):
        return {}
    table = table.assign(
        period=pd.to_datetime(table["period"], errors="coerce").dt.normalize(),
        filed_date=pd.to_datetime(table["filed_date"], errors="coerce").dt.normalize(),
    ).dropna(subset=["period", "filed_date"])
    result: dict[str, dict[pd.Timestamp, pd.Timestamp]] = {}
    for ticker, group in table.groupby("ticker"):
        result[ticker] = dict(zip(group["period"], group["filed_date"]))
    return result


def _ticker_price_series(group: pd.DataFrame) -> tuple[list[pd.Timestamp], list[float]]:
    sorted_group = group.sort_values("date")
    dates = [pd.Timestamp(value) for value in sorted_group["date"].tolist()]
    values = [float(value) if pd.notna(value) else float("nan") for value in sorted_group["adj_close"].tolist()]
    return dates, values


def _metric_cache(payload: dict, lag_days: int, publication_map: dict[pd.Timestamp, pd.Timestamp]) -> dict:
    metric = payload.get("metric", {}) if isinstance(payload, dict) else {}
    metric = metric if isinstance(metric, dict) else {}
    raw_series = payload.get("series", {}) if isinstance(payload, dict) else {}
    raw_series = raw_series if isinstance(raw_series, dict) else {}
    annual = raw_series.get("annual") or {}
    quarterly = raw_series.get("quarterly") or {}
    annual = annual if isinstance(annual, dict) else {}
    quarterly = quarterly if isinstance(quarterly, dict) else {}
    series = {}
    for key in set(HISTORICAL_SERIES_MAP.values()) | {"salesPerShare", "eps"}:
        series[key] = _prepared_rows(quarterly, key, lag_days, publication_map) or _prepared_rows(annual, key, lag_days, publication_map)
    return {"metric": metric, "series": series}


def _payload_to_metrics(
    cache: dict,
    snapshot_date: pd.Timestamp,
    current_price: float,
    price_dates: list[pd.Timestamp],
    price_values: list[float],
) -> dict[str, float | str | None]:
    metric = cache.get("metric", {}) if isinstance(cache, dict) else {}
    series = cache.get("series", {}) if isinstance(cache, dict) else {}
    values: dict[str, float | str | None] = {}
    asof_dates: list[pd.Timestamp] = []

    for name, source in HISTORICAL_SERIES_MAP.items():
        value, asof = _historical_value(series, source, snapshot_date)
        values[name] = value
        if asof is not None:
            asof_dates.append(asof)

    revenue_growth, revenue_asof = _historical_growth(series, "salesPerShare", snapshot_date)
    eps_growth, eps_asof = _historical_growth(series, "eps", snapshot_date)
    values["revenue_growth"] = revenue_growth
    values["eps_growth"] = eps_growth
    asof_dates.extend([date for date in (revenue_asof, eps_asof) if date is not None])

    for name, source in METRIC_MAP.items():
        if values.get(name) is None:
            values[name] = _num(metric.get(source))

    fundamental_asof = max(asof_dates) if asof_dates else snapshot_date
    price_at_fundamental = _latest_price(price_dates, price_values, fundamental_asof)
    price_change = current_price / price_at_fundamental - 1 if price_at_fundamental and price_at_fundamental > 0 else 0.0
    values["fundamental_asof_date"] = fundamental_asof.date().isoformat()
    values["price_at_fundamental_asof"] = price_at_fundamental
    values["price_return_since_fundamental"] = price_change
    values["stale_fundamental_months"] = max((snapshot_date.year - fundamental_asof.year) * 12 + snapshot_date.month - fundamental_asof.month, 0)

    for multiple in ("pe", "forward_pe", "peg"):
        base_value = _num(values.get(multiple))
        adjusted = base_value * (1 + price_change) if base_value is not None else None
        values[f"{multiple}_base"] = base_value
        values[f"{multiple}_price_adjusted"] = adjusted
        values[multiple] = adjusted
    return values


def _historical_value(
    series: dict,
    key: str,
    snapshot_date: pd.Timestamp,
) -> tuple[float | None, pd.Timestamp | None]:
    row = _latest_row(series.get(key, []), snapshot_date)
    if row is None:
        return None, None
    return row[1], row[0]


def _historical_growth(
    series: dict,
    key: str,
    snapshot_date: pd.Timestamp,
) -> tuple[float | None, pd.Timestamp | None]:
    rows = series.get(key, [])
    index = bisect_right([row[0] for row in rows], snapshot_date) - 1
    if index < 1:
        return None, None
    current = rows[index][1]
    previous = rows[index - 1][1]
    if current is None or previous in (None, 0):
        return None, rows[index][0]
    return current / previous - 1, rows[index][0]


def _prepared_rows(
    series: dict,
    key: str,
    lag_days: int,
    publication_map: dict[pd.Timestamp, pd.Timestamp],
) -> list[tuple[pd.Timestamp, float | None]]:
    rows = series.get(key, [])
    if not isinstance(rows, Iterable) or isinstance(rows, (str, bytes, dict)):
        return []
    # Un fundamental de un periodo NO es observable el mismo día del cierre de periodo: un 10-K/10-Q
    # tarda en publicarse. Su fecha observable es la fecha de publicación REAL (filedDate, de
    # report_dates.parquet, cruzada por fin de periodo); si no se conoce, cae al fallback
    # cierre_de_periodo + lag_days. Esa fecha observable es la que usa el resto del pipeline (bisect
    # point-in-time). Corrige el lookahead de tratar el fundamental como conocido en la fecha de cierre.
    lag = pd.Timedelta(days=lag_days)
    valid = []
    for row in rows:
        period = pd.to_datetime(row.get("period"), errors="coerce")
        if pd.isna(period):
            continue
        period = pd.Timestamp(period).normalize()
        observable = publication_map.get(period)
        if observable is None:
            observable = period + lag
        valid.append((observable, _num(row.get("v"))))
    return sorted(valid, key=lambda item: item[0])


def _latest_row(rows: list[tuple[pd.Timestamp, float | None]], date: pd.Timestamp) -> tuple[pd.Timestamp, float | None] | None:
    index = bisect_right([row[0] for row in rows], date) - 1
    if index < 0:
        return None
    return rows[index]


def _latest_price(
    dates: list[pd.Timestamp],
    values: list[float],
    date: pd.Timestamp,
    require_value: bool = True,
) -> float | None:
    index = bisect_right(dates, date) - 1
    if index < 0:
        return None
    if not require_value:
        return 1.0
    value = values[index]
    if pd.isna(value):
        return None
    return float(value)


def _trailing_return(
    dates: list[pd.Timestamp],
    values: list[float],
    date: pd.Timestamp,
    months: int,
) -> float | None:
    current = _latest_price(dates, values, date)
    past = _latest_price(dates, values, date - pd.DateOffset(months=months))
    if current is None or past in (None, 0):
        return None
    return current / past - 1


def _num(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
