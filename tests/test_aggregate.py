"""Agregación global y selección automática del sistema final (module/experiments/aggregate.py).

Fija dos cosas críticas para la honestidad del TFM:
1. La selección se hace por ESTABILIDAD (media alta + poca dispersión + años positivos), no por la
   media cruda: un config con mayor rank-IC medio pero errático NO debe ganar.
2. La elección se hace sobre la era de DESARROLLO; la era de confirmación solo COMPRUEBA que
   generaliza (no participa en la elección).
"""

from __future__ import annotations

import pandas as pd

from module.experiments.aggregate import aggregate_scenarios


def _rows() -> list[dict]:
    return [
        # Estable, positivo y útil: media moderada, poca dispersión, muchos años positivos, buen lift.
        {"name": "estable", "block": "ventana_cadencia",
         "rank_ic_dev_mean": 0.03, "rank_ic_dev_std": 0.01, "rank_ic_dev_positive_years": 8,
         "rank_ic_dev_n_years": 10, "rank_ic_conf_mean": 0.02, "top_n_alpha_lift": 0.01},
        # Mayor media pero ERRÁTICO y con confirmación negativa: no debe ganar pese al mejor dev_mean.
        {"name": "erratico", "block": "ventana_cadencia",
         "rank_ic_dev_mean": 0.05, "rank_ic_dev_std": 0.08, "rank_ic_dev_positive_years": 6,
         "rank_ic_dev_n_years": 10, "rank_ic_conf_mean": -0.01, "top_n_alpha_lift": 0.005},
        # Muy estable pero media baja y sin lift.
        {"name": "plano", "block": "ventana_cadencia",
         "rank_ic_dev_mean": 0.01, "rank_ic_dev_std": 0.005, "rank_ic_dev_positive_years": 9,
         "rank_ic_dev_n_years": 10, "rank_ic_conf_mean": 0.015, "top_n_alpha_lift": 0.0},
    ]


def test_selecciona_el_estable_no_el_de_mayor_media(tmp_path):
    verdict = aggregate_scenarios(_rows(), tmp_path)
    assert verdict["winner"] == "estable"
    assert verdict["winner"] != "erratico", "el mayor rank-IC medio no debe ganar si es inestable"
    assert verdict["generalizes"] is True  # confirmación 0.02 > 0
    assert (tmp_path / "system_selection.csv").exists()
    assert (tmp_path / "system_selection.json").exists()


def test_confirmacion_no_participa_en_la_eleccion(tmp_path):
    # 'erratico' tiene la mejor confirmación imaginable, pero pésimo desarrollo -> no debe ganar.
    rows = _rows()
    rows[1]["rank_ic_conf_mean"] = 0.99
    verdict = aggregate_scenarios(rows, tmp_path)
    assert verdict["winner"] == "estable"


def test_sin_metricas_de_desarrollo_no_elige(tmp_path):
    rows = [{"name": "x", "block": "b", "rank_ic_dev_mean": float("nan")}]
    assert aggregate_scenarios(rows, tmp_path) == {}


def test_banner_se_renderiza_en_el_html(tmp_path):
    from module.experiments.report import write_comparison

    verdict = aggregate_scenarios(_rows(), tmp_path)
    # rows mínimos para el informe (necesita 'name'/'block'); baseline para los KPIs.
    rows = [{"name": "baseline", "block": "baseline"}, *_rows()]
    path = write_comparison(rows, tmp_path, selection=verdict)
    html_text = path.read_text(encoding="utf-8")
    assert "Sistema final" in html_text and "estable" in html_text


if __name__ == "__main__":  # pragma: no cover
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
