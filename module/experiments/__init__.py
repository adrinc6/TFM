"""Sistema de experimentos: barre escenarios (ML + estrategia) reusando el scoring caro, y produce
una tabla e informe HTML comparativos. Ver módulo raíz CLAUDE.md y docs/ para la metodología."""

from .runner import load_scenarios, run_experiment, run_scenario, scoring_cache_key
from .scenario import Scenario

__all__ = ["Scenario", "load_scenarios", "run_experiment", "run_scenario", "scoring_cache_key"]
