"""Bloques nuevos del informe: curva de tamaño de cartera y tabla de desempeño anual.

Tests puros sobre las funciones de datos/HTML (sin generar PNG ni correr el pipeline).
"""

from __future__ import annotations

import pandas as pd
import pytest

from module.viewer.charts import breadth_curve
from module.viewer.pages import _annual_performance_table, _breadth_size_block


def _diag() -> pd.DataFrame:
    return pd.DataFrame({
        "snapshot_date": ["2020-01-15", "2020-02-15"],
        "mode": ["walk_forward_model", "walk_forward_model"],
        "top5_alpha_lift": [0.03, 0.05],
        "top10_alpha_lift": [0.02, 0.02],
        "top20_alpha_lift": [0.01, 0.01],
        "top50_alpha_lift": [0.0, 0.0],
    })


def test_breadth_curve_promedia_por_tamano():
    curve = breadth_curve(_diag())
    assert list(curve["n"]) == [5, 10, 20, 50]
    assert curve.loc[curve["n"] == 5, "lift"].iloc[0] == pytest.approx(0.04)
    assert curve.loc[curve["n"] == 50, "lift"].iloc[0] == pytest.approx(0.0)


def test_breadth_curve_ignora_snapshots_en_fallback():
    diag = _diag()
    diag.loc[len(diag)] = {"snapshot_date": "2019-01-15", "mode": "fallback_garp",
                           "top5_alpha_lift": 9.0, "top10_alpha_lift": 9.0,
                           "top20_alpha_lift": 9.0, "top50_alpha_lift": 9.0}
    curve = breadth_curve(diag)
    # El fallback (9.0) no debe contaminar la media.
    assert curve.loc[curve["n"] == 5, "lift"].iloc[0] == pytest.approx(0.04)


def test_breadth_block_html():
    out = _breadth_size_block({}, _diag())
    assert "Tamaño de cartera óptimo" in out


def test_tabla_desempeno_anual():
    vs = pd.DataFrame({
        "date": ["2020-06-15", "2020-12-15", "2021-06-15"],
        "portfolio_period_return": [0.10, 0.10, 0.05],
        "benchmark_period_return": [0.05, 0.05, 0.05],
    })
    out = _annual_performance_table(vs)
    assert "Desempeño anual" in out
    assert "2020" in out and "2021" in out


def test_tabla_anual_vacia_no_rompe():
    assert _annual_performance_table(pd.DataFrame()) == ""


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
