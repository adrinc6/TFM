"""Calendario de la simulación: fecha ancla (trimestre + retardo), cadencia de entrenamiento
(trimestral/anual) y retardo de publicación de fundamentales.

Comprueba tres invariantes nuevos del esquema:
1. `eval_start_quarter` + `fundamental_publication_lag_days` derivan la fecha ancla correcta.
2. La cadencia anual reentrena una vez al año desde el ancla, sin cruzarla hacia el futuro.
3. Un fundamental NO es observable hasta `cierre_de_periodo + lag` (corrige el lookahead sutil).

Pure-logic + un master sintético con la función real `build_master_dataset`; sin datos descargados.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from environment import Settings, _quarter_to_anchor
import module.dataset as dataset
import module.ml as ml


def test_quarter_plus_lag_derives_the_anchor():
    assert _quarter_to_anchor("2010Q1", 45) == "2010-02-15"
    assert _quarter_to_anchor("2007Q3", 45) == "2007-08-15"
    assert _quarter_to_anchor("2010Q1", 0) == "2010-01-01"


def test_annual_cadence_trains_once_a_year_from_the_anchor():
    dates = [pd.Timestamp(d) for d in pd.date_range("2005-02-15", "2015-02-15", freq=pd.DateOffset(months=1))]
    settings = dataclasses.replace(Settings(), train_cutoff_date="2010-02-15", walk_forward_train_frequency="A")
    train_dates, apply_dates = ml._train_and_apply_dates(dates, settings)
    assert apply_dates == sorted(dates)
    assert min(train_dates) == pd.Timestamp("2010-02-15")
    assert all(d >= pd.Timestamp("2010-02-15") for d in train_dates)
    gaps = [(b - a).days for a, b in zip(train_dates, train_dates[1:])]
    assert gaps and all(360 <= gap <= 372 for gap in gaps), "los reentrenos deben distar ~1 año"


def test_quarterly_cadence_trains_every_three_months_from_the_anchor():
    dates = [pd.Timestamp(d) for d in pd.date_range("2005-02-15", "2015-02-15", freq=pd.DateOffset(months=1))]
    settings = dataclasses.replace(Settings(), train_cutoff_date="2010-02-15", walk_forward_train_frequency="Q")
    train_dates, _ = ml._train_and_apply_dates(dates, settings)
    assert min(train_dates) == pd.Timestamp("2010-02-15")
    gaps = [(b - a).days for a, b in zip(train_dates, train_dates[1:])]
    assert all(80 <= gap <= 100 for gap in gaps)


def test_prepared_rows_shift_observability_by_the_publication_lag():
    series = {"roic": [{"period": "2020-12-31", "v": 0.10}, {"period": "2021-03-31", "v": 0.14}]}
    rows = dataset._prepared_rows(series, "roic", lag_days=45)
    assert rows[0][0] == pd.Timestamp("2020-12-31") + pd.Timedelta(days=45)
    assert rows[1][0] == pd.Timestamp("2021-03-31") + pd.Timedelta(days=45)
    assert [value for _, value in rows] == [0.10, 0.14]


@pytest.fixture
def lag_master(tmp_path, monkeypatch):
    import environment

    ticker, benchmark = "T00", "SPY"
    monkeypatch.setattr(dataset, "RAW_DIR", tmp_path)
    monkeypatch.setattr(dataset, "MASTER_DIR", tmp_path)
    monkeypatch.setattr(environment, "DEV_TICKERS", [ticker])

    price_rows = []
    price = 100.0
    for day in pd.date_range("2020-11-01", "2021-07-31", freq="D"):
        price *= 1.001
        for symbol in (ticker, benchmark):
            price_rows.append({"ticker": symbol, "date": day, "adj_close": price})
    pd.DataFrame(price_rows).to_parquet(tmp_path / "prices.parquet")

    payload = {"metric": {}, "series": {"quarterly": {"roic": [
        {"period": "2020-12-31", "v": 0.10},
        {"period": "2021-03-31", "v": 0.14},
    ]}}}
    pd.DataFrame([{"ticker": ticker, "payload": repr(payload)}]).to_parquet(tmp_path / "finnhub_metrics.parquet")
    pd.DataFrame([{"ticker": ticker, "finnhubIndustry": "Tech"}]).to_parquet(tmp_path / "profiles.parquet")

    settings = dataclasses.replace(
        Settings(),
        dev_mode=True,
        data_start_date="2021-01-01",
        train_cutoff_date="2021-02-15",  # ancla real: rejilla fasada al día 15
        end_date="2021-06-30",
        walk_forward_scoring=False,
        fundamental_publication_lag_days=45,
    )
    return settings, ticker


def test_publication_lag_delays_fundamental_observability(lag_master):
    from module.dataset import build_master_dataset

    settings, ticker = lag_master
    master = build_master_dataset(settings)
    by_date = master[master["ticker"] == ticker].set_index("snapshot_date")["roic"]
    # El fundamental de Q4-2020 (cierre 2020-12-31) se publica el 2021-02-14: NO es observable el
    # 2021-01-15, y sí el 2021-02-15. Con lag=0 sí estaría el 2021-01-15 -> el test aísla el retardo.
    assert pd.isna(by_date.get("2021-01-15"))
    assert by_date.get("2021-02-15") == pytest.approx(0.10)
    # El de Q1-2021 (cierre 2021-03-31) se publica el 2021-05-15: aún 0.10 en abril, 0.14 desde mayo.
    assert by_date.get("2021-04-15") == pytest.approx(0.10)
    assert by_date.get("2021-05-15") == pytest.approx(0.14)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
