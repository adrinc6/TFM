"""Rejilla base de escenarios (placeholder de Fase 1; se completa en Fase 4).

El sistema es LightGBM + rank_regression. Los escenarios activan artefactos y varian
hiperparametros/ventanas/cadencia como ablations dirigidas. La rejilla completa se define en
Fase 4 del plan.
"""

from module.experiments import ScenarioSpec


SCENARIOS = [
    ScenarioSpec(name="baseline", overrides={}),
]
