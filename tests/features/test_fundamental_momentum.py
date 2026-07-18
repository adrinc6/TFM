"""B3: tendencia de fundamentales y descomposicion precio/fundamental, point-in-time."""

from __future__ import annotations

import numpy as np
import pandas as pd

from module.modeling.features import _add_fundamental_momentum


def _panel() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["AAA"] * 5,
        "snapshot_date": ["2000-01-15", "2000-02-15", "2000-03-15", "2000-04-15", "2000-05-15"],
        "fundamental_period": ["1999-09-30", "1999-12-31", "1999-12-31", "2000-03-31", "2000-03-31"],
        "roe": [0.10, 0.15, 0.15, 0.12, 0.12],
        "roic": [0.0] * 5, "net_margin": [0.0] * 5,
        "operating_margin": [0.0] * 5, "eps_growth_yoy": [0.0] * 5,
        "pe": [20.0, 18.0, 19.0, 25.0, 22.0],
        "price": [100.0, 108.0, 114.0, 120.0, 110.0],
    })


def test_momentum_is_constant_between_publications() -> None:
    """El delta se mantiene entre publicaciones: no hay informacion nueva, no cae a cero."""
    out = _add_fundamental_momentum(_panel())
    mom = out["mom_roe"].tolist()
    assert np.isnan(mom[0])                         # sin publicacion anterior
    assert round(mom[1], 3) == 0.05                 # 0.15 - 0.10
    assert round(mom[2], 3) == 0.05                 # mismo periodo -> se mantiene
    assert round(mom[3], 3) == -0.03                # 0.12 - 0.15
    assert round(mom[4], 3) == -0.03                # se mantiene


def test_pe_decomposition_separates_price_and_fundamental() -> None:
    """El cambio de valoracion se separa en su parte de precio y su parte de EPS."""
    out = _add_fundamental_momentum(_panel())
    # En 2000-04: EPS baja (precio/pe: 120/25=4.8 vs 114/19=6.0) -> componente fundamental < 0.
    assert out["pe_fundamental_component"].iloc[3] < 0
    # y el precio subio (114 -> 120) -> componente precio > 0.
    assert out["pe_price_component"].iloc[3] > 0


def test_no_lookahead_only_past_publications_used() -> None:
    """Mutar una publicacion FUTURA no cambia los deltas de filas anteriores."""
    panel = _panel()
    baseline = _add_fundamental_momentum(panel)

    mutated = panel.copy()
    mutated.loc[3:, "roe"] = 0.99          # cambia el futuro (2000-04 en adelante)
    after = _add_fundamental_momentum(mutated)

    # Las filas <= 2000-03 no deben cambiar.
    pd.testing.assert_series_equal(
        baseline["mom_roe"].iloc[:3], after["mom_roe"].iloc[:3], check_names=False
    )
