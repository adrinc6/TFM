"""Contrato de seleccion: rango medio de las 4 dimensiones anuales elige al mas estable,
no al de mayor alfa."""

from __future__ import annotations

import pandas as pd

from module.experiments import select_winner


def test_stable_scenario_beats_spike_year_by_composite_rank() -> None:
    """Un escenario que gana 20 anios por 2 % gana a otro que suma 30 % pero solo un anio."""
    summary = pd.DataFrame([
        # spike: alfa total alta, pero derivada de un solo anio brutal.
        {"scenario": "spike", "beat_rate": 0.4, "median_alpha": 0.005,
         "worst_year_alpha": -0.20, "max_drawdown": 0.35, "total_alpha": 1.20},
        # steady: nada espectacular pero todos positivos y drawdown pequeno.
        {"scenario": "steady", "beat_rate": 1.0, "median_alpha": 0.02,
         "worst_year_alpha": 0.01, "max_drawdown": 0.08, "total_alpha": 0.15},
        # middle: mezcla.
        {"scenario": "middle", "beat_rate": 0.6, "median_alpha": 0.03,
         "worst_year_alpha": -0.05, "max_drawdown": 0.15, "total_alpha": 0.30},
    ])

    winner, ranked = select_winner(summary)

    assert winner == "steady", f"esperaba 'steady', salio {winner!r}"
    # El "spike" no debe estar arriba.
    assert ranked.iloc[0]["scenario"] == "steady"
    assert ranked.iloc[-1]["scenario"] == "spike"
