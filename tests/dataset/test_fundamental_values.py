"""Cómo se leen los valores fundamentales de las series de Finnhub.

No son pruebas de fuga temporal (esas están en `test_point_in_time.py`): comprueban que un
número que sale del panel es de verdad lo que su nombre de columna dice que es.
"""

from __future__ import annotations

import pandas as pd
import pytest

from module.data.dataset import build_point_in_time_dataset, snapshot_dates


def _snapshot_after(settings, after: str) -> str:
    """Primer snapshot de la rejilla en/tras `after` (cae en fin_de_mes + execution_lag_days)."""
    grid = snapshot_dates(settings)
    return next(d.date().isoformat() for d in grid if d >= pd.Timestamp(after))


FULL_HISTORY = [
    "1997-03-31", "1997-06-30", "1997-09-30", "1997-12-31",
    "1998-03-31", "1998-06-30", "1998-09-30", "1998-12-31",
    "1999-03-31", "1999-06-30", "1999-09-30", "1999-12-31",
]


def _rewrite_aaa_quarterly(settings, *, eps_periods: list[str]) -> None:
    """Reescribe la serie de AAA con historia larga: `eps` según `eps_periods`, ventas completas.

    Hace falta historia suficiente para que 1999-12-31 quede en la posición 4 o posterior: con
    menos, la guarda de "no hay 4 trimestres previos" corta antes y el test pasaría sin llegar
    a ejercitar el emparejado, que es justo lo que se quiere probar.
    """
    metrics_path = settings.raw_output_dir / "finnhub_metrics.parquet"
    metrics = pd.read_parquet(metrics_path)
    for row in metrics.itertuples():
        if row.ticker != "AAA":
            continue
        quarterly = row.payload["series"]["quarterly"]
        quarterly["eps"] = [
            {"period": period, "v": 1.0 + index} for index, period in enumerate(eps_periods)
        ]
        quarterly["salesPerShare"] = [
            {"period": period, "v": 5.0 + index} for index, period in enumerate(FULL_HISTORY)
        ]
    metrics.to_parquet(metrics_path, index=False)


def test_yoy_growth_is_na_when_the_prior_year_quarter_is_missing(dataset_settings) -> None:
    """Sin el trimestre de hace un año, `eps_growth_yoy` debe ser NA, no un número inventado.

    Falta 1998-12-31, la pareja real de 1999-12-31. Contando cuatro posiciones atrás se cae en
    1998-09-30 —quince meses— y el resultado se etiqueta como interanual igualmente.
    """
    _rewrite_aaa_quarterly(
        dataset_settings, eps_periods=[p for p in FULL_HISTORY if p != "1998-12-31"]
    )
    panel = build_point_in_time_dataset(dataset_settings)
    snapshot = _snapshot_after(dataset_settings, "2000-02-05")
    row = panel.loc[
        (panel["ticker"] == "AAA") & (panel["snapshot_date"] == snapshot)
    ].iloc[0]

    assert row["fundamental_period"] == "1999-12-31"
    assert pd.isna(row["eps_growth_yoy"])
    # La serie de ventas conserva su historia completa: no debe verse afectada.
    assert pd.notna(row["sales_per_share_growth_yoy"])


def test_yoy_growth_survives_a_gap_that_shifts_positions(dataset_settings) -> None:
    """Un hueco que NO es la pareja no debe impedir el cálculo.

    Falta 1999-03-31: desplaza los índices, pero 1998-12-31 sigue ahí. Emparejando por fecha el
    interanual se calcula; contando posiciones se compararía contra el trimestre equivocado.
    Sin este caso, el arreglo podría limitarse a devolver NA siempre y parecer correcto.
    """
    _rewrite_aaa_quarterly(
        dataset_settings, eps_periods=[p for p in FULL_HISTORY if p != "1999-03-31"]
    )
    panel = build_point_in_time_dataset(dataset_settings)
    snapshot = _snapshot_after(dataset_settings, "2000-02-05")
    row = panel.loc[
        (panel["ticker"] == "AAA") & (panel["snapshot_date"] == snapshot)
    ].iloc[0]

    # eps: 1998-12-31 -> 8.0 (indice 7), 1999-12-31 -> 11.0 (indice 10 tras quitar uno).
    assert row["eps_growth_yoy"] == pytest.approx(11.0 / 8.0 - 1)


@pytest.fixture
def annual_only_margin_settings(dataset_settings):
    """`netMargin` de AAA solo existe en el bloque anual, con el mismo cierre que el Q4."""
    metrics_path = dataset_settings.raw_output_dir / "finnhub_metrics.parquet"
    metrics = pd.read_parquet(metrics_path)
    for row in metrics.itertuples():
        if row.ticker != "AAA":
            continue
        series = row.payload["series"]
        series["annual"] = {"netMargin": [{"period": "1999-12-31", "v": 0.99}]}
        del series["quarterly"]["netMargin"]
    metrics.to_parquet(metrics_path, index=False)
    return dataset_settings


def test_non_ttm_margins_do_not_fall_back_to_annual(annual_only_margin_settings) -> None:
    """Un margen anual no es un margen trimestral: mezclarlos contamina el corte transversal.

    El cierre anual y el Q4 comparten fecha (1999-12-31), así que el fallback los confunde:
    AAA aportaría un margen de doce meses donde BBB aporta uno de tres.
    """
    panel = build_point_in_time_dataset(annual_only_margin_settings)
    snapshot = _snapshot_after(annual_only_margin_settings, "2000-02-05")
    row = panel.loc[
        (panel["ticker"] == "AAA") & (panel["snapshot_date"] == snapshot)
    ].iloc[0]

    assert pd.isna(row["net_margin"])
    assert row["net_margin"] != 0.99
