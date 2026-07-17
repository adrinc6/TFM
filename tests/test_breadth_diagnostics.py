"""Métricas de breadth (top-N) del diagnóstico de la señal maestra.

El rank-IC mide el orden de TODO el universo, pero la cartera solo compra el top-N. Estas pruebas
fijan que `top_n_alpha` / `top_n_alpha_lift` midan de verdad el tramo que se ejecuta, incluido el
caso que motivó la métrica: un ranking global bueno puede tener un top-N malo (y al revés).

Tests puros sobre tablas sintéticas: no requieren datos descargados ni modelo entrenado.
"""

from __future__ import annotations

import pandas as pd
import pytest

import module.ml as ml


def _labeled(scores: list[float], alphas: list[float], snapshot: str = "2020-01-31") -> pd.DataFrame:
    return pd.DataFrame({
        "snapshot_date": [snapshot] * len(scores),
        "final_score": scores,
        "target_future_alpha": alphas,
    })


def _diagnostics(snapshot: str = "2020-01-31") -> pd.DataFrame:
    return pd.DataFrame({"snapshot_date": [snapshot]})


def test_top_n_alpha_mide_solo_los_comprados():
    """top_n_alpha es el alpha medio de los N mejores por final_score, no el del universo."""
    # 10 nombres: el score ordena perfectamente; los 2 mejores tienen alpha 1.0 y 0.8.
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    alphas = [1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    out = ml._master_signal_diagnostics(_diagnostics(), _labeled(scores, alphas), top_n=2)
    assert out.iloc[0]["top_n_alpha"] == pytest.approx(0.9)  # media de 1.0 y 0.8
    # lift = alpha del top-2 menos la media del universo (1.8/10 = 0.18)
    assert out.iloc[0]["top_n_alpha_lift"] == pytest.approx(0.9 - 0.18)


def test_rank_ic_alto_puede_tener_top_n_malo():
    """El caso que motiva la métrica: ranking global muy bueno pero el top-N es el peor tramo.

    Universo de 100 nombres cuyo alpha crece con el score EXCEPTO en los 2 primeros, invertidos y
    negativos. Como son 2 de 100, el rank-IC global apenas se resiente (sigue >0.9) pero justo lo
    que la cartera compra pierde dinero. Es el patrón `solo_alpha` del barrido real: el mejor
    rank-IC (0.332) con la peor alpha (0.796). Ilustra por qué el rank-IC del universo no basta.
    """
    n = 100
    scores = [i / n for i in range(n, 0, -1)]          # 1.00 descendente
    alphas = [-0.5, -0.4] + [i / n for i in range(n - 2, 0, -1)]  # top-2 invertido y negativo
    out = ml._master_signal_diagnostics(_diagnostics(), _labeled(scores, alphas), top_n=2)
    assert out.iloc[0]["rank_ic_final"] > 0.85, "el orden global sigue siendo muy bueno"
    assert out.iloc[0]["top_n_alpha"] < 0, "pero lo que se compra pierde dinero"
    assert out.iloc[0]["top_n_alpha_lift"] < 0, "y va por debajo de la media del universo"


def test_top_n_lift_positivo_cuando_el_top_es_bueno():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    alphas = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    out = ml._master_signal_diagnostics(_diagnostics(), _labeled(scores, alphas), top_n=3)
    assert out.iloc[0]["top_n_alpha_lift"] > 0


def test_curva_breadth_por_tamano_de_cartera():
    """La curva top-N (BREADTH_TOP_NS) permite elegir el tamaño óptimo de cartera desde el ranking:
    concentrar en menos nombres (los mejores) da más alpha medio cuando el orden es bueno."""
    n = 50
    scores = [i / n for i in range(n, 0, -1)]
    alphas = list(scores)  # alpha crece con el score: orden perfecto
    out = ml._master_signal_diagnostics(_diagnostics(), _labeled(scores, alphas))
    row = out.iloc[0]
    for size in ml.BREADTH_TOP_NS:
        assert f"top{size}_alpha" in out.columns and f"top{size}_alpha_lift" in out.columns
    assert row["top5_alpha"] > row["top10_alpha"] > row["top20_alpha"] > row["top50_alpha"]
    # top_n_alpha es alias del tamaño real de cartera (MAX_PORTFOLIO_SIZE por defecto).
    assert row["top_n_alpha"] == pytest.approx(row[f"top{ml.MAX_PORTFOLIO_SIZE}_alpha"])


def test_sin_final_score_no_rompe():
    """Sin la columna de señal, las métricas quedan a NA en vez de reventar."""
    labeled = pd.DataFrame({"snapshot_date": ["2020-01-31"], "target_future_alpha": [0.1]})
    out = ml._master_signal_diagnostics(_diagnostics(), labeled)
    assert pd.isna(out.iloc[0]["top_n_alpha"])
    assert pd.isna(out.iloc[0]["top_n_alpha_lift"])


def test_media_anual_ignora_los_snapshots_en_fallback():
    """Un año mixto no puede heredar el IC del `garp_score` determinista.

    Antes del cutoff no hay modelo y `final_score` cae al baseline, cuyo IC (~0.6) mide su
    correlación consigo mismo. Un solo snapshot así dentro de un año duplicaba la media anual real
    (2018: 0.098 vs. 0.048), y esa media es la que compara los escenarios del TFM.
    """
    # Enero en fallback con IC perfecto; febrero con modelo e IC ~0. La media del año debe ser la de
    # febrero, no el promedio de ambos.
    labeled = pd.concat([
        _labeled([0.9, 0.8, 0.7, 0.6, 0.5], [0.9, 0.8, 0.7, 0.6, 0.5], snapshot="2018-01-31"),
        _labeled([0.9, 0.8, 0.7, 0.6, 0.5], [0.1, 0.5, 0.2, 0.4, 0.3], snapshot="2018-02-28"),
    ])
    diagnostics = pd.DataFrame({
        "snapshot_date": ["2018-01-31", "2018-02-28"],
        "mode": ["fallback_garp", "walk_forward_model"],
    })
    out = ml._master_signal_diagnostics(diagnostics, labeled)

    fallback_ic = out.loc[out["mode"] == "fallback_garp", "rank_ic_final"].iloc[0]
    model_ic = out.loc[out["mode"] == "walk_forward_model", "rank_ic_final"].iloc[0]
    assert fallback_ic == pytest.approx(1.0)  # el baseline correlado consigo mismo
    # El IC por snapshot se conserva sin enmascarar (es auditable); lo que se limpia es la agregación.
    assert out["rank_ic_final_year_mean"].nunique() == 1
    assert out["rank_ic_final_year_mean"].iloc[0] == pytest.approx(model_ic)


def test_sin_columna_mode_agrega_todo():
    """Diagnósticos antiguos sin `mode` siguen agregando sin romper."""
    out = ml._master_signal_diagnostics(
        _diagnostics(), _labeled([0.9, 0.8, 0.7, 0.6, 0.5], [0.9, 0.8, 0.7, 0.6, 0.5])
    )
    assert out.iloc[0]["rank_ic_final_year_mean"] == pytest.approx(1.0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
