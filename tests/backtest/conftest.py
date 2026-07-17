"""Fixture minima para tests de cartera: scores controlados sobre un universo pequeno."""

from __future__ import annotations

import pandas as pd
import pytest

from environment import Settings


@pytest.fixture
def portfolio_settings() -> Settings:
    """Config con parametros de cartera por defecto; los tests los sobreescriben con `replace`."""
    return Settings(run_scope="dev", data_start_date="2000-01-01", end_date="2001-12-31")


@pytest.fixture
def synthetic_scores() -> pd.DataFrame:
    """Panel de scores controlado. `meta_rank` es el percentil transversal (0..1), lo que ya
    produce Fase 3 en `agent_scores.parquet`.

    Snapshots: 2000-01-15, 2000-02-15, 2000-03-15, 2000-04-15 (dos trimestrales, dos mensuales).
    Universo: AAA, BBB, CCC, DDD, EEE, FFF, GGG, HHH, III, JJJ (10 tickers).
    """
    rows: list[dict] = []
    snapshots = [
        ("2000-01-15", True),   # fundamental_quarterly
        ("2000-02-15", False),  # price_monthly
        ("2000-03-15", False),
        ("2000-04-15", True),
    ]
    # Por snapshot, doy percentiles distintos para cada ticker para poder guionar entradas/salidas.
    # Este es el caso base: ranking estable, sin osificacion.
    ranks_by_snapshot = {
        "2000-01-15": {"AAA": 0.95, "BBB": 0.90, "CCC": 0.88, "DDD": 0.85, "EEE": 0.82,
                        "FFF": 0.75, "GGG": 0.60, "HHH": 0.40, "III": 0.20, "JJJ": 0.10},
        "2000-02-15": {"AAA": 0.92, "BBB": 0.91, "CCC": 0.87, "DDD": 0.83, "EEE": 0.80,
                        "FFF": 0.70, "GGG": 0.55, "HHH": 0.38, "III": 0.22, "JJJ": 0.12},
        "2000-03-15": {"AAA": 0.90, "BBB": 0.93, "CCC": 0.86, "DDD": 0.84, "EEE": 0.79,
                        "FFF": 0.72, "GGG": 0.58, "HHH": 0.35, "III": 0.25, "JJJ": 0.15},
        "2000-04-15": {"AAA": 0.88, "BBB": 0.94, "CCC": 0.85, "DDD": 0.86, "EEE": 0.81,
                        "FFF": 0.68, "GGG": 0.62, "HHH": 0.40, "III": 0.28, "JJJ": 0.18},
    }
    for snapshot_date, is_quarterly in snapshots:
        for ticker, meta_rank in ranks_by_snapshot[snapshot_date].items():
            rows.append(
                {
                    "ticker": ticker,
                    "snapshot_date": snapshot_date,
                    "is_quarterly": is_quarterly,
                    "meta_score": meta_rank,   # simplificacion: meta_score = meta_rank en fixtures
                    "meta_rank": meta_rank,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Precios PIT alineados con los snapshots. Todos los tickers suben 1 % por snapshot
    (deterministas) salvo cuando un test cambia el precio para forzar cash/costes concretos.
    """
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ"]
    snapshots = ["2000-01-15", "2000-02-15", "2000-03-15", "2000-04-15"]
    rows: list[dict] = []
    for step, snapshot in enumerate(snapshots):
        for ticker in tickers:
            base = 100.0 + tickers.index(ticker)  # AAA=100, BBB=101, ...
            rows.append(
                {
                    "ticker": ticker,
                    "snapshot_date": snapshot,
                    "price": base * (1.01 ** step),
                    "price_as_of_date": snapshot,
                    "price_age_days": 0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_benchmark() -> pd.DataFrame:
    """SPY sube 0.5 % por snapshot: da alfa positiva a la cartera base."""
    rows: list[dict] = []
    for step, snapshot in enumerate(["2000-01-15", "2000-02-15", "2000-03-15", "2000-04-15"]):
        rows.append(
            {
                "snapshot_date": snapshot,
                "price": 400.0 * (1.005 ** step),
                "price_as_of_date": snapshot,
                "price_age_days": 0,
                "price_return_1m": 0.005,
                "price_return_3m": 0.015,
                "price_return_6m": 0.03,
                "price_return_12m": 0.06,
            }
        )
    return pd.DataFrame(rows)
