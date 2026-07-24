"""Perfiles de inversor: reordenan las buenas del meta segun estilo, sin tocar el resto."""

from __future__ import annotations

import pandas as pd

from module.evaluation.profiles import PROFILE_NAMES, apply_profile


def _scores() -> pd.DataFrame:
    # 4 acciones: dos buenas (meta_rank alto), dos malas. Entre las buenas, A es mas value y menos
    # arriesgada (risk_rank alto = poca volatilidad); B es mas momentum, mas growth y mas arriesgada.
    return pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "snapshot_date": ["2016-02-15"] * 4,
        "meta_rank": [0.95, 0.90, 0.40, 0.20],
        "quality_rank": [0.7, 0.7, 0.5, 0.3],
        "momentum_rank": [0.3, 0.95, 0.4, 0.2],
        "value_rank": [0.95, 0.3, 0.5, 0.4],
        "growth_rank": [0.3, 0.9, 0.5, 0.4],
        "risk_rank": [0.9, 0.2, 0.5, 0.5],
    })


def test_balanced_returns_meta_unchanged() -> None:
    scores = _scores()
    out = apply_profile(scores, "balanced")
    pd.testing.assert_series_equal(out["meta_rank"], scores["meta_rank"])


def test_profile_fails_loudly_when_agent_rank_missing() -> None:
    """Si falta el rango de un agente que el perfil pondera, apply_profile debe fallar, no callar.

    Antes se saltaba en silencio (deformando el estilo). El full_study fija los 5 agentes, pero un
    run manual sin `quality` no debe producir un perfil quality falso, sino un error visible.
    """
    import pytest
    scores = _scores().drop(columns=["quality_rank"])  # finalista sin agente quality
    with pytest.raises(ValueError, match="quality_rank"):
        apply_profile(scores, "quality")


def test_value_profile_prefers_value_stock_among_the_good() -> None:
    """Entre las buenas (A, B), el perfil value pone a A (value alto) por encima de B."""
    out = apply_profile(_scores(), "value")
    a = out.loc[out["ticker"] == "A", "meta_rank"].iloc[0]
    b = out.loc[out["ticker"] == "B", "meta_rank"].iloc[0]
    assert a > b


def test_momentum_profile_prefers_momentum_stock() -> None:
    """El perfil momentum usa risk_rank negativo; verifica que aun asi rankea por momentum."""
    out = apply_profile(_scores(), "momentum")
    a = out.loc[out["ticker"] == "A", "meta_rank"].iloc[0]
    b = out.loc[out["ticker"] == "B", "meta_rank"].iloc[0]
    assert b > a          # B tiene momentum alto (aunque su risk_rank sea bajo)


def test_defensive_profile_prefers_low_risk_stock() -> None:
    """Entre las buenas, el defensivo pone arriba la de risk_rank alto (menos volatilidad)."""
    out = apply_profile(_scores(), "defensive")
    a = out.loc[out["ticker"] == "A", "meta_rank"].iloc[0]  # risk_rank 0.9 (poca volatilidad)
    b = out.loc[out["ticker"] == "B", "meta_rank"].iloc[0]  # risk_rank 0.2 (mucha volatilidad)
    assert a > b


def test_contrarian_profile_penalizes_high_momentum() -> None:
    """El contrarian usa momentum_rank negativo: penaliza a la de momentum alto (B) frente a A."""
    out = apply_profile(_scores(), "contrarian")
    a = out.loc[out["ticker"] == "A", "meta_rank"].iloc[0]  # momentum bajo, value alto
    b = out.loc[out["ticker"] == "B", "meta_rank"].iloc[0]  # momentum alto -> penalizado
    assert a > b


def test_bad_stocks_never_selected_by_any_profile() -> None:
    """Las acciones que no son buenas (meta_rank < umbral) quedan a 0 en todos los perfiles."""
    for profile in PROFILE_NAMES:
        if profile == "balanced":
            continue
        out = apply_profile(_scores(), profile)
        bad = out.loc[out["ticker"].isin(["C", "D"]), "meta_rank"]
        assert (bad == 0.0).all(), f"{profile} no deberia seleccionar las malas"
