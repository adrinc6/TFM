"""B1: neutralizacion por sector en el ranking transversal, con fallback a global."""

from __future__ import annotations

import pandas as pd

from module.modeling.features import _cross_section_rank


def _frame(groups: list[str]) -> pd.DataFrame:
    n = len(groups)
    return pd.DataFrame({
        "snapshot_date": ["d"] * n,
        "_group": groups,
        "is_price_fresh": [True] * n,
    })


def test_large_group_ranks_within_sector() -> None:
    """Un grupo con >= min_group miembros se rankea dentro del sector, no globalmente."""
    frame = _frame(["A", "A", "A", "A"])
    values = pd.Series([10.0, 20, 30, 40], index=frame.index)
    ranked = _cross_section_rank(frame, values, ascending=True, min_group=3)
    # 4 miembros -> percentiles 0.25/0.50/0.75/1.0 dentro del propio grupo
    assert list(ranked.round(3)) == [0.25, 0.5, 0.75, 1.0]


def test_small_group_falls_back_to_global() -> None:
    """Un grupo con menos de min_group miembros se rankea sobre todo el corte, no aislado."""
    frame = _frame(["A", "A", "A", "A", "B", "B"])
    values = pd.Series([10.0, 20, 30, 40, 5, 50], index=frame.index)
    ranked = _cross_section_rank(frame, values, ascending=True, min_group=3)
    # B tiene 2 miembros (< 3): cae a global. val=5 es el minimo global (1/6=0.167),
    # val=50 es el maximo global (1.0). Si se hubiera rankeado dentro de B, val=5 seria 0.5.
    b_low = ranked.iloc[4]
    b_high = ranked.iloc[5]
    assert round(b_low, 3) == 0.167
    assert round(b_high, 3) == 1.0


def test_min_group_zero_is_pure_global() -> None:
    """min_group=0 reproduce exactamente el ranking global (comportamiento base sin B1)."""
    frame = _frame(["A", "A", "B", "B"])
    values = pd.Series([10.0, 40, 20, 30], index=frame.index)
    grouped = _cross_section_rank(frame, values, ascending=True, min_group=0)
    # ranking global independiente del grupo: 10<20<30<40 -> 0.25/1.0/0.5/0.75
    assert list(grouped.round(3)) == [0.25, 1.0, 0.5, 0.75]
