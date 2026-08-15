"""Contrato de la medición de cobertura del universo.

El sesgo de supervivencia de este proyecto no está en la composición del índice —que es
point-in-time— sino en la cobertura de datos: las empresas que murieron, cambiaron de símbolo o
presentan formularios de emisor extranjero no llegan al panel. Ambas mediciones existen para que
ese agujero sea una cifra y no una interpretación, y las dos tienen un modo de fallo silencioso:

1. `_ticker_resolution` clasifica cada ticker por un solo motivo. Si la precedencia falla, un
   ticker con varios fallos se cuenta más de una vez, los recuentos dejan de sumar el universo y
   la exclusión aparenta ser mayor de lo que es.
2. `_coverage_by_year` publicaba la calidad **dentro** del panel sin denominador, de modo que
   marcaba 100 % mientras faltaba media lista del S&P 500.
"""

from __future__ import annotations

import pandas as pd

from module.data.ingest.pipeline import RESOLUTION_REASONS, _ticker_resolution
from module.research.attribution import _coverage_by_year


class _Universe:
    """Doble mínimo: `_ticker_resolution` solo necesita `tickers` y `benchmark_ticker`."""

    def __init__(self, tickers: list[str], benchmark: str = "SPY"):
        self.tickers = [*tickers, benchmark]
        self.benchmark_ticker = benchmark


def _failure(ticker: str, dataset: str, reason: str) -> dict[str, str]:
    return {"ticker": ticker, "dataset": dataset, "reason": reason}


def test_resolution_counts_add_up_to_the_universe():
    """Cada ticker cuenta una vez: `in_panel` + `excluded` debe ser el universo, sin el benchmark.

    Es la comprobación que detecta una doble contabilización, que inflaría la exclusión.
    """
    universe = _Universe(["AAA", "BBB", "CCC", "DDD"])
    failures = [
        _failure("BBB", "ohlcv", "missing"),
        _failure("CCC", "edgar", "missing_cik"),
    ]

    resolution = _ticker_resolution(universe, failures, recycled_tickers=set())

    assert resolution["universe_tickers"] == 4
    assert resolution["in_panel"] + resolution["excluded"] == 4
    assert resolution["in_panel"] == 2
    assert sum(row["tickers"] for row in resolution["by_reason"]) == resolution["excluded"]


def test_benchmark_never_counts_as_universe():
    """SPY es referencia, no empresa: el pipeline le salta perfil y fundamentales a propósito.

    Si contase, aparecería siempre como excluido por falta de fundamentales y ensuciaría la cifra.
    """
    universe = _Universe(["AAA"])

    resolution = _ticker_resolution(universe, failures=[], recycled_tickers=set())

    assert resolution["universe_tickers"] == 1
    assert resolution["in_panel"] == 1


def test_earliest_failure_wins_when_a_ticker_fails_several_times():
    """Un ticker sin precios ni CIK se cuenta como `missing_price`, no en ambas categorías.

    La precedencia sigue el flujo real de la descarga: sin precios nunca se llega a consultar EDGAR,
    así que atribuirlo al CIK describiría mal la causa.
    """
    universe = _Universe(["AAA"])
    failures = [
        _failure("AAA", "edgar", "missing_cik"),
        _failure("AAA", "ohlcv", "missing"),
    ]

    resolution = _ticker_resolution(universe, failures, recycled_tickers=set())

    counts = {row["reason"]: row["tickers"] for row in resolution["by_reason"]}
    assert counts["missing_price"] == 1
    assert counts["missing_cik"] == 0
    assert resolution["excluded"] == 1


def test_recycled_ticker_takes_precedence_over_every_other_failure():
    """Un símbolo reutilizado es un problema de identidad, no de disponibilidad de datos."""
    universe = _Universe(["AAA"])
    failures = [_failure("AAA", "edgar", "missing_cik")]

    resolution = _ticker_resolution(universe, failures, recycled_tickers={"AAA"})

    counts = {row["reason"]: row["tickers"] for row in resolution["by_reason"]}
    assert counts["recycled_ticker"] == 1
    assert counts["missing_cik"] == 0


def test_unresolved_symbols_are_not_reported_as_deaths():
    """`missing_cik` es una cota superior de la mortalidad, y el artefacto debe decirlo.

    Es la afirmación que el trabajo no puede hacer a la ligera: un símbolo que no resuelve puede ser
    un cambio de ticker o un emisor extranjero, no una quiebra.
    """
    universe = _Universe(["AAA", "BBB"])
    failures = [_failure("BBB", "edgar", "missing_cik")]

    resolution = _ticker_resolution(universe, failures, recycled_tickers=set())

    assert resolution["unresolved_sample"] == ["BBB"]
    assert "cota superior" in resolution["note"]
    assert {reason for reason, _ in RESOLUTION_REASONS} == {
        row["reason"] for row in resolution["by_reason"]
    }


def test_coverage_by_year_reports_the_index_as_denominator(monkeypatch):
    """La cobertura del panel se mide contra los miembros reales del índice, no contra sí misma.

    Con un panel de 2 tickers sobre un índice de 4, `usable_fraction` puede valer 100 % —todas las
    filas del panel son utilizables— mientras la cobertura real es del 50 %. Sin el denominador, la
    tabla afirmaba lo primero y callaba lo segundo.
    """
    monkeypatch.setattr(
        "module.research.attribution._index_members_by_year", lambda: {2015: 4},
    )
    features = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "snapshot_date": ["2015-03-31", "2015-03-31"],
            "is_price_fresh": [True, True],
        }
    )

    rows = _coverage_by_year(features)

    assert len(rows) == 1
    assert rows[0]["distinct_tickers"] == 2
    assert rows[0]["sp500_members"] == 4
    assert rows[0]["panel_coverage_fraction"] == 0.5
    assert rows[0]["usable_fraction"] == 1.0


def test_coverage_degrades_without_the_index_composition(monkeypatch):
    """Sin composición histórica la cobertura queda `None`, nunca 0.

    Cero significaría «ningún miembro del índice llegó al panel», que es una afirmación falsa y
    mucho peor que declarar que no se pudo medir.
    """
    monkeypatch.setattr("module.research.attribution._index_members_by_year", dict)
    features = pd.DataFrame(
        {"ticker": ["AAA"], "snapshot_date": ["2015-03-31"], "is_price_fresh": [True]}
    )

    rows = _coverage_by_year(features)

    assert rows[0]["sp500_members"] == 0
    assert rows[0]["panel_coverage_fraction"] is None
