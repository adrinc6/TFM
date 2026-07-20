"""Barrido de robustez del umbral de rebalanceo (`rebalance_drift_tolerance`).

Este eje es MECÁNICO (regla de fricción, no señal predictiva; ver docs/doc.md). Por eso NO se
trata como eje de optimización: no se elige el valor que maximiza rentabilidad. El objetivo del
barrido es DEMOSTRAR ROBUSTEZ — que las métricas (rank-IC y, como consecuencia, rentabilidad neta)
apenas se mueven al variar la tolerancia entre 1.0 y 2.5 pp alrededor del default (1.5).

`rebalance_drift_tolerance` es un campo de cartera (PORTFOLIO_FIELDS): solo interviene en el
backtest, no reentrena agentes. Los tres escenarios comparten dataset/features/agents por
fingerprint y solo difieren en el re-backtest, así que el barrido es barato.

Uso:
    python -c "from environment import Settings; from module.runs.experiments import run_scenarios; \
        from environment import PROJECT_ROOT; \
        run_scenarios(PROJECT_ROOT/'module'/'scenarios'/'estres.py', Settings())"

El resultado (results/escenarios/comparison.html) debe mostrar rank-IC idéntico entre los tres
(mismo aprendizaje) y rentabilidad/rotación estables. Si lo son -> el parámetro no es sensible y
se documenta como fortaleza. Si NO lo son -> hay sobre-rotación que habría que investigar.
"""

from __future__ import annotations

from module.runs.experiments import ScenarioSpec

# El baseline (1.5) va explícito para que aparezca en el comparativo junto a las variantes.
SCENARIOS = [
    ScenarioSpec(name="drift_1_0", overrides={"rebalance_drift_tolerance": 1.0}),
    ScenarioSpec(name="drift_1_5", overrides={"rebalance_drift_tolerance": 1.5}),
    ScenarioSpec(name="drift_2_5", overrides={"rebalance_drift_tolerance": 2.5}),
]
