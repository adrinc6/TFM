from __future__ import annotations

import pytest

import module.universe as universe


@pytest.fixture(autouse=True)
def historical_components_csv(monkeypatch, tmp_path):
    """Aísla las pruebas de descarga de los datos crudos del usuario."""
    components = tmp_path / "sp500_components.csv"
    components.write_text(
        "date,tickers\n"
        '1996-01-02,"AAPL,CPQ,MOB,BF.B"\n'
        '1999-11-30,"AAPL,CPQ"\n'
        '2000-01-03,"AAPL,ENRNQ,CPQ"\n'
        '2002-05-01,"AAPL,ENRNQ"\n'
        '2004-10-14,"AAPL"\n'
        '2026-06-30,"AAPL"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(universe, "SP500_COMPONENTS_CSV", components)
    universe._snapshots.cache_clear()
    yield
    universe._snapshots.cache_clear()
