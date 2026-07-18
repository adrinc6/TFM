"""Emparejado de cada snapshot con su fecha de etiqueta.

La fecha de etiqueta se obtiene tomando, en la propia rejilla de snapshots
(`module.dataset.snapshot_dates`), la que está N posiciones más adelante — no sumando meses por
calendario. La rejilla clampa los fines de mes con `min(snapshot_day, days_in_month)`, y una
suma de calendario (`snapshot + DateOffset(months=horizon)`) clampa con otra regla distinta:
con `SNAPSHOT_DAY = 31` no coincidían, el `merge` de la etiqueta no encontraba pareja y el
target se degradaba a NaN sin ningún error, perdiendo ~40 % del entrenamiento en silencio.
Emparejar por posición en la rejilla hace que la fecha de etiqueta exista por construcción,
sea cual sea `SNAPSHOT_DAY` — uno de los parámetros que la Fase 6 va a barrer.
"""

from __future__ import annotations

import pandas as pd
import pytest

import environment
import module.data.universe as universe
from environment import Settings
from module.data.dataset import build_point_in_time_dataset
from module.modeling.features import build_features


def test_every_fresh_row_before_the_horizon_gets_a_target(feature_settings) -> None:
    """Las filas con precio fresco y horizonte completo deben tener etiqueta, no NaN.

    Protege contra que un cambio en el emparejado vuelva a perder etiquetas en silencio.
    """
    build_features(feature_settings)
    targets = pd.read_parquet(feature_settings.processed_output_dir / "targets_forward_3m.parquet")
    features = pd.read_parquet(feature_settings.processed_output_dir / "features_point_in_time.parquet")

    last_snapshot = pd.to_datetime(features["snapshot_date"]).max()
    horizon_cutoff = last_snapshot - pd.DateOffset(months=feature_settings.target_horizon_months)
    merged = features.merge(targets, on=["ticker", "snapshot_date"], validate="one_to_one")
    within_horizon = merged.loc[
        merged["is_price_fresh"].fillna(False)
        & pd.to_datetime(merged["snapshot_date"]).le(horizon_cutoff)
    ]

    assert not within_horizon.empty
    assert within_horizon["target_available"].all()


@pytest.fixture
def non_default_snapshot_day_settings(monkeypatch, tmp_path) -> Settings:
    """Dos años de precios e informes, para que el horizonte de 3 meses tenga margen real."""
    raw_dir = tmp_path / "raw" / "dev"
    processed_dir = tmp_path / "processed" / "dev"
    components = tmp_path / "sp500_components.csv"
    components.write_text(
        "date,tickers\n1999-01-02,\"AAA,BBB\"\n2000-01-03,\"AAA,BBB\"\n", encoding="utf-8"
    )
    monkeypatch.setattr(environment, "DEV_RAW_DIR", raw_dir)
    monkeypatch.setattr(environment, "DEV_PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(universe, "SP500_COMPONENTS_CSV", components)
    universe._snapshots.cache_clear()

    price_rows = []
    for ticker, base in (("AAA", 10.0), ("BBB", 20.0), ("SPY", 100.0)):
        for index, date in enumerate(pd.date_range("1999-01-31", "2001-06-30", freq="ME")):
            price_rows.append(
                {
                    "ticker": ticker,
                    "date": date.date().isoformat(),
                    "open": base + index,
                    "high": base + index,
                    "low": base + index,
                    "close": base + index,
                    "adj_close": base + index,
                    "volume": 1000,
                }
            )
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(price_rows).to_parquet(raw_dir / "prices.parquet", index=False)

    periods = ["1998-12-31", "1999-03-31", "1999-06-30", "1999-09-30", "1999-12-31", "2000-03-31"]
    values = lambda start: [
        {"period": period, "v": start * (index + 1)} for index, period in enumerate(periods)
    ]
    payload = {
        "metric": {"roeTTM": 999.0},
        "series": {"quarterly": {"roeTTM": values(0.1), "peTTM": values(10)}},
    }
    pd.DataFrame(
        [{"ticker": "AAA", "payload": payload}, {"ticker": "BBB", "payload": payload}]
    ).to_parquet(raw_dir / "finnhub_metrics.parquet", index=False)
    pd.DataFrame(
        [
            {"ticker": "AAA", "cik": "1", "form": "10-K", "period": "1999-12-31", "filed_date": "2000-02-01"},
            {"ticker": "BBB", "cik": "2", "form": "10-K", "period": "1999-12-31", "filed_date": "2000-03-10"},
        ]
    ).to_parquet(raw_dir / "report_dates.parquet", index=False)

    yield Settings(
        run_scope="dev", data_start_date="1999-01-01", end_date="2001-06-30", snapshot_day=31
    )
    universe._snapshots.cache_clear()


def test_non_default_snapshot_day_does_not_lose_targets(non_default_snapshot_day_settings) -> None:
    """Extremo a extremo (`dataset` -> `features`) con `SNAPSHOT_DAY = 31`.

    Antes del arreglo del Hallazgo 2, este caso perdía ~40 % de los targets sin ningún error:
    la fecha de etiqueta se calculaba sumando meses por calendario y no coincidía con la
    rejilla de snapshots, que clampa los fines de mes de otra forma.
    """
    settings = non_default_snapshot_day_settings
    build_point_in_time_dataset(settings)
    build_features(settings)

    features = pd.read_parquet(settings.processed_output_dir / "features_point_in_time.parquet")
    targets = pd.read_parquet(settings.processed_output_dir / "targets_forward_3m.parquet")

    last_snapshot = pd.to_datetime(features["snapshot_date"]).max()
    horizon_cutoff = last_snapshot - pd.DateOffset(months=settings.target_horizon_months)
    merged = features.merge(targets, on=["ticker", "snapshot_date"], validate="one_to_one")
    within_horizon = merged.loc[
        merged["is_price_fresh"].fillna(False)
        & pd.to_datetime(merged["snapshot_date"]).le(horizon_cutoff)
    ]

    assert not within_horizon.empty
    assert within_horizon["target_available"].all()
