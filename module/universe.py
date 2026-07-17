"""Universo dinámico: qué tickers formaban parte del S&P 500 en cada fecha.

El universo NO es una lista estática de los líderes actuales: se lee de un CSV con la
composición histórica real del índice (un snapshot por día de mercado, 1996-2026). Así, una
señal fechada en 2000 solo puede mirar a las empresas que estaban en el índice en 2000, y no a
las que entraron después. Eso elimina el sesgo de inclusión anticipada.

Ver `docs/plan_fases.md` (Fase 0) para el contexto y las limitaciones.
"""

from __future__ import annotations

import functools
import logging

import pandas as pd

from environment import SP500_COMPONENTS_CSV

log = logging.getLogger(__name__)


def normalize_ticker(ticker: str) -> str:
    """Convierte el formato del CSV al de Yahoo/Finnhub: `BRK.B` -> `BRK-B`.

    El CSV usa punto para las clases de acción (11 casos: BRK.B, BF.B, RDS.A...); las APIs
    que consultamos usan guion.
    """
    return ticker.strip().upper().replace(".", "-")


@functools.lru_cache(maxsize=1)
def _snapshots() -> list[tuple[pd.Timestamp, frozenset[str]]]:
    """Composición del índice por fecha, ordenada. Cacheada: el CSV no cambia en un run."""
    if not SP500_COMPONENTS_CSV.exists():
        raise FileNotFoundError(
            f"No se encuentra el CSV de composición histórica del S&P 500: {SP500_COMPONENTS_CSV}. "
            "Es la fuente del universo dinámico (ver docs/plan_fases.md, Fase 0)."
        )
    frame = pd.read_csv(SP500_COMPONENTS_CSV)
    rows = [
        (pd.Timestamp(row.date), frozenset(normalize_ticker(t) for t in row.tickers.split(",") if t.strip()))
        for row in frame.itertuples(index=False)
    ]
    rows.sort(key=lambda item: item[0])
    log.info(
        "Composición histórica del S&P 500: %s snapshots (%s..%s)",
        len(rows),
        rows[0][0].date(),
        rows[-1][0].date(),
    )
    return rows


def members_at(date) -> frozenset[str]:
    """Miembros del índice en `date`, según el último snapshot disponible <= `date`.

    Antes del primer snapshot (1996-01-02) devuelve vacío: no inventamos composición.
    """
    target = pd.Timestamp(date)
    snapshots = _snapshots()
    result: frozenset[str] = frozenset()
    for snapshot_date, tickers in snapshots:
        if snapshot_date > target:
            break
        result = tickers
    return result


def historical_universe() -> frozenset[str]:
    """Todos los tickers que pertenecieron al índice en algún momento (1206 únicos).

    Incluye deslistados y quebrados. Es el universo a intentar descargar: los que no tengan
    datos quedan registrados como evidencia del sesgo de supervivencia.
    """
    result: set[str] = set()
    for _, tickers in _snapshots():
        result |= tickers
    return frozenset(result)


def annual_membership_dates() -> list[pd.Timestamp]:
    """Último snapshot disponible de la composición para cada año natural."""
    latest_by_year: dict[int, pd.Timestamp] = {}
    for snapshot_date, _ in _snapshots():
        latest_by_year[snapshot_date.year] = snapshot_date
    return [latest_by_year[year] for year in sorted(latest_by_year)]


def first_membership_date() -> pd.Timestamp:
    """Primera fecha para la que existe composición histórica del índice."""
    return _snapshots()[0][0]


def membership_span(ticker: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Primera y última fecha en que `ticker` estuvo en el índice, o None si nunca estuvo.

    La última fecha es la que usa la guarda de reciclaje de tickers (ver `is_recycled_ticker`).
    """
    normalized = normalize_ticker(ticker)
    dates = [date for date, tickers in _snapshots() if normalized in tickers]
    if not dates:
        return None
    return dates[0], dates[-1]


def is_recycled_ticker(ticker: str, first_price_date) -> bool:
    """True si los precios de `ticker` son de OTRA empresa que reutilizó el símbolo.

    Un símbolo liberado por una empresa (quiebra, fusión) puede reasignarse años después. Sus
    precios no son los de la empresa que estuvo en el índice y contaminarían el backtest.
    Ejemplos reales: CPQ (Compaq salió del índice en 2002-05-01, pero Yahoo devuelve precios
    desde 2004) y MOB (Mobil salió en 1999-11-30, precios desde 2022).

    La regla: si el primer precio disponible es POSTERIOR a la última fecha en el índice, esos
    datos no pueden ser de la empresa histórica.
    """
    span = membership_span(ticker)
    if span is None or first_price_date is None:
        return False
    return pd.Timestamp(first_price_date) > span[1]
