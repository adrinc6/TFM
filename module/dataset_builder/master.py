"""Build auditable ticker x snapshot_date rows."""

from __future__ import annotations

import logging
from ast import literal_eval

import pandas as pd

from environment import MASTER_DIR, RAW_DIR, Settings
from module.common.io import read_parquet, write_parquet

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
    snapshots = _review_dates(settings)
    rows: list[dict] = []

    payload_by_ticker = {
        row.ticker: _parse_payload(row.payload)
        for row in metrics.itertuples(index=False)
    }
    profile_by_ticker = profiles.drop_duplicates("ticker").set_index("ticker").to_dict("index")

    for snapshot_date in snapshots:
        for ticker in settings.investable_tickers:
            history = prices[(prices["ticker"] == ticker) & (prices["date"] <= snapshot_date)]
            benchmark = prices[(prices["ticker"] == settings.benchmark_ticker) & (prices["date"] <= snapshot_date)]
            if history.empty or benchmark.empty:
                continue
            row = {
                "ticker": ticker,
                "snapshot_date": snapshot_date.date().isoformat(),
                "price": float(history.sort_values("date").iloc[-1]["adj_close"]),
                "sector": profile_by_ticker.get(ticker, {}).get("finnhubIndustry", "Unknown"),
            }
            row.update(_payload_to_metrics(payload_by_ticker.get(ticker, {}), snapshot_date))
            rows.append(row)

    master = pd.DataFrame(rows)
    if master.empty:
        raise RuntimeError("Master dataset is empty. Check raw data and date range.")
    write_parquet(master, MASTER_DIR / "master_point_in_time.parquet")
    log.info("Master dataset rows: %s", len(master))
    return master


def _review_dates(settings: Settings) -> pd.DatetimeIndex:
    aliases = {"M": "ME", "2M": "2ME", "Q": "QE"}
    frequency = aliases.get(settings.review_frequency, settings.review_frequency)
    return pd.date_range(settings.data_start_date, settings.end_date, freq=frequency)


def _parse_payload(payload: object) -> dict:
    if isinstance(payload, str):
        payload = literal_eval(payload)
    return payload if isinstance(payload, dict) else {}


def _payload_to_metrics(payload: dict, snapshot_date: pd.Timestamp) -> dict[str, float | None]:
    metric = payload.get("metric", {}) if isinstance(payload, dict) else {}
    annual = payload.get("series", {}).get("annual", {}) if isinstance(payload, dict) else {}
    values = {name: _historical_value(annual, source, snapshot_date) for name, source in HISTORICAL_SERIES_MAP.items()}
    values["forward_pe"] = _num(metric.get(METRIC_MAP["forward_pe"]))
    values["peg"] = _num(metric.get(METRIC_MAP["peg"]))
    values["revenue_growth"] = _historical_growth(annual, "salesPerShare", snapshot_date)
    values["eps_growth"] = _historical_growth(annual, "eps", snapshot_date)
    for name, source in METRIC_MAP.items():
        if values.get(name) is None:
            values[name] = _num(metric.get(source))
    return values


def _historical_value(annual: dict, key: str, snapshot_date: pd.Timestamp) -> float | None:
    rows = _historical_rows(annual, key, snapshot_date)
    if not rows:
        return None
    return _num(rows[-1].get("v"))


def _historical_growth(annual: dict, key: str, snapshot_date: pd.Timestamp) -> float | None:
    rows = _historical_rows(annual, key, snapshot_date)
    if len(rows) < 2:
        return None
    current = _num(rows[-1].get("v"))
    previous = _num(rows[-2].get("v"))
    if current is None or previous in (None, 0):
        return None
    return current / previous - 1


def _historical_rows(annual: dict, key: str, snapshot_date: pd.Timestamp) -> list[dict]:
    rows = annual.get(key, [])
    if not isinstance(rows, list):
        return []
    valid = []
    for row in rows:
        period = pd.to_datetime(row.get("period"), errors="coerce")
        if pd.notna(period) and period <= snapshot_date:
            valid.append(row)
    return sorted(valid, key=lambda item: item.get("period", ""))


def _num(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
