"""Contrato de seleccion: gana el que APRENDE (rank-IC + estabilidad), no el de mas alfa.

La seleccion del sistema final se basa en aprendizaje y consistencia, nunca en rentabilidad.
Elegir por alfa cuando el rank-IC es debil seria seleccionar ruido. Ver docs/doc.md (§8).
"""

from __future__ import annotations

import pandas as pd

from module.experiments import select_winner


def test_learner_beats_high_alpha_scenario() -> None:
    """Un escenario con rank-IC alto y estable gana a otro con mucho alfa pero rank-IC ~0."""
    summary = pd.DataFrame([
        # high_alpha: alfa espectacular pero NO aprende (rank-IC nulo). Es ruido afortunado.
        {"scenario": "high_alpha", "mean_rank_ic": 0.001, "rank_ic_positive_fraction": 0.49,
         "beat_rate": 0.9, "max_drawdown": 0.30, "annualized_alpha": 0.25},
        # learner: aprende de verdad (rank-IC positivo, consistente) aunque su alfa sea modesto.
        {"scenario": "learner", "mean_rank_ic": 0.06, "rank_ic_positive_fraction": 0.68,
         "beat_rate": 0.6, "max_drawdown": 0.20, "annualized_alpha": 0.03},
        # middle.
        {"scenario": "middle", "mean_rank_ic": 0.02, "rank_ic_positive_fraction": 0.55,
         "beat_rate": 0.7, "max_drawdown": 0.25, "annualized_alpha": 0.10},
    ])

    winner, ranked = select_winner(summary)

    assert winner == "learner", f"esperaba 'learner', salio {winner!r}"
    # El de mayor alfa NO debe ganar por su alfa.
    assert ranked.iloc[-1]["scenario"] == "high_alpha"


def test_alpha_columns_do_not_affect_the_ranking() -> None:
    """Cambiar el alfa (a igualdad de senales de aprendizaje) no cambia el ganador."""
    base = [
        {"scenario": "a", "mean_rank_ic": 0.05, "rank_ic_positive_fraction": 0.65,
         "beat_rate": 0.6, "max_drawdown": 0.20},
        {"scenario": "b", "mean_rank_ic": 0.02, "rank_ic_positive_fraction": 0.52,
         "beat_rate": 0.55, "max_drawdown": 0.28},
    ]
    winner_low, _ = select_winner(pd.DataFrame([{**r, "annualized_alpha": 0.01} for r in base]))
    winner_high, _ = select_winner(pd.DataFrame([
        {**base[0], "annualized_alpha": 0.01},   # el que aprende, poco alfa
        {**base[1], "annualized_alpha": 5.00},   # el que no aprende, alfa absurdo
    ]))
    assert winner_low == winner_high == "a", "el alfa no debe alterar la seleccion"


def test_composite_is_mean_of_learning_ranks() -> None:
    """El rango compuesto es la media de los rangos de las dimensiones de aprendizaje."""
    summary = pd.DataFrame([
        {"scenario": "best", "mean_rank_ic": 0.10, "rank_ic_positive_fraction": 0.70,
         "beat_rate": 0.8, "max_drawdown": 0.10, "annualized_alpha": 0.0},
        {"scenario": "worst", "mean_rank_ic": -0.05, "rank_ic_positive_fraction": 0.40,
         "beat_rate": 0.3, "max_drawdown": 0.50, "annualized_alpha": 0.0},
    ])
    _, ranked = select_winner(summary)
    best = ranked.loc[ranked["scenario"] == "best"].iloc[0]
    # best es el mejor en las 4 dimensiones -> rango 1 en todas -> media 1.0
    assert best["composite_rank_mean"] == 1.0
