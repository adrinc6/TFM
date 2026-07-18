"""Perfiles de inversor: reordenan las buenas del meta segun estilo, sin tocar el resto."""

from __future__ import annotations

import pandas as pd

from module.profiles import PROFILE_NAMES, apply_profile


def _scores() -> pd.DataFrame:
    # 4 acciones: dos buenas (meta_rank alto), dos malas. Entre las buenas, A es mas value,
    # B es mas momentum.
    return pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "snapshot_date": ["2016-02-15"] * 4,
        "meta_rank": [0.95, 0.90, 0.40, 0.20],
        "quality_rank": [0.7, 0.7, 0.5, 0.3],
        "momentum_rank": [0.3, 0.95, 0.4, 0.2],
        "value_rank": [0.95, 0.3, 0.5, 0.4],
    })


def test_balanced_returns_meta_unchanged() -> None:
    scores = _scores()
    out = apply_profile(scores, "balanced")
    pd.testing.assert_series_equal(out["meta_rank"], scores["meta_rank"])


def test_value_profile_prefers_value_stock_among_the_good() -> None:
    """Entre las buenas (A, B), el perfil value pone a A (value alto) por encima de B."""
    out = apply_profile(_scores(), "value")
    a = out.loc[out["ticker"] == "A", "meta_rank"].iloc[0]
    b = out.loc[out["ticker"] == "B", "meta_rank"].iloc[0]
    assert a > b


def test_momentum_profile_prefers_momentum_stock() -> None:
    out = apply_profile(_scores(), "momentum")
    a = out.loc[out["ticker"] == "A", "meta_rank"].iloc[0]
    b = out.loc[out["ticker"] == "B", "meta_rank"].iloc[0]
    assert b > a          # B tiene momentum alto


def test_bad_stocks_never_selected_by_any_profile() -> None:
    """Las acciones que no son buenas (meta_rank < umbral) quedan a 0 en todos los perfiles."""
    for profile in PROFILE_NAMES:
        if profile == "balanced":
            continue
        out = apply_profile(_scores(), profile)
        bad = out.loc[out["ticker"].isin(["C", "D"]), "meta_rank"]
        assert (bad == 0.0).all(), f"{profile} no deberia seleccionar las malas"
