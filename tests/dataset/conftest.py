from __future__ import annotations

import pandas as pd
import pytest

import environment
import module.data.universe as universe
from environment import Settings


def _periods() -> list[str]:
    return ["1998-12-31", "1999-03-31", "1999-06-30", "1999-09-30", "1999-12-31", "2000-03-31"]


def _payload(multiplier: float) -> dict:
    periods = _periods()
    def values(start: float) -> list[dict]:
        return [
            {"period": period, "v": multiplier * start * (index + 1)}
            for index, period in enumerate(periods)
        ]
    return {
        "metric": {"roeTTM": 999.0},
        "series": {
            "quarterly": {
                "roeTTM": values(0.1),
                "roicTTM": values(0.2),
                "netMargin": values(0.3),
                "operatingMargin": values(0.4),
                "grossMargin": values(0.5),
                "fcfMargin": values(0.6),
                "peTTM": values(10),
                "pb": values(2),
                "psTTM": values(3),
                "evEbitdaTTM": values(4),
                "totalDebtToEquity": values(1),
                "currentRatio": values(1.5),
                "eps": values(1),
                "salesPerShare": values(5),
            }
        },
    }


@pytest.fixture
def dataset_settings(monkeypatch, tmp_path) -> Settings:
    raw_dir = tmp_path / "raw" / "dev"
    processed_dir = tmp_path / "processed" / "dev"
    components = tmp_path / "sp500_components.csv"
    components.write_text(
        "date,tickers\n"
        '1999-01-02,"AAA,BBB"\n'
        '2000-01-03,"AAA,BBB"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(environment, "DEV_RAW_DIR", raw_dir)
    monkeypatch.setattr(environment, "DEV_PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(universe, "SP500_COMPONENTS_CSV", components)
    universe._snapshots.cache_clear()

    # Precios diarios: así hay precio fresco en cualquier fecha de la rejilla (que ahora cae en
    # fin_de_mes + execution_lag_days, no en un día de mes fijo). El valor crece de forma
    # monótona con el tiempo para conservar el orden que comprueban los tests.
    price_rows = []
    for ticker, base in (("AAA", 10.0), ("BBB", 20.0), ("SPY", 100.0)):
        for index, date in enumerate(pd.date_range("1999-01-01", "2000-04-15", freq="D")):
            price_rows.append(
                {
                    "ticker": ticker,
                    "date": date.date().isoformat(),
                    "open": base + index / 30.0,
                    "high": base + index / 30.0,
                    "low": base + index / 30.0,
                    "close": base + index / 30.0,
                    "adj_close": base + index / 30.0,
                    "volume": 1000,
                }
            )
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(price_rows).to_parquet(raw_dir / "prices.parquet", index=False)
    pd.DataFrame(
        [
            {"ticker": "AAA", "payload": _payload(1.0)},
            {"ticker": "BBB", "payload": _payload(2.0)},
        ]
    ).to_parquet(raw_dir / "finnhub_metrics.parquet", index=False)
    pd.DataFrame(
        [
            {"ticker": "AAA", "cik": "1", "form": "10-Q", "period": "1999-09-30", "filed_date": "1999-11-01"},
            {"ticker": "AAA", "cik": "1", "form": "10-K", "period": "1999-12-31", "filed_date": "2000-02-01"},
            {"ticker": "AAA", "cik": "1", "form": "10-Q", "period": "2000-03-31", "filed_date": "2000-04-10"},
            {"ticker": "BBB", "cik": "2", "form": "10-Q", "period": "1999-09-30", "filed_date": "1999-11-15"},
            {"ticker": "BBB", "cik": "2", "form": "10-K", "period": "1999-12-31", "filed_date": "2000-03-10"},
        ]
    ).to_parquet(raw_dir / "report_dates.parquet", index=False)

    yield Settings(run_scope="dev", data_start_date="1999-01-01", end_date="2000-04-15")
    universe._snapshots.cache_clear()
