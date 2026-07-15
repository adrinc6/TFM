"""Punto de entrada del runner de experimentos.

Uso:
    python -m module.experiments run <ruta/al/fichero_de_escenarios.py>
    python -m module.experiments run todos

El fichero de escenarios debe definir una lista ``SCENARIOS: list[Scenario]``. El literal ``todos``
junta los cuatro bloques. El runner añade un escenario ``baseline`` si no hay ninguno, ejecuta todos,
y escribe ``comparison.csv`` + ``index.html`` en ``results/experiments/<fecha>/``.
"""

from __future__ import annotations

import sys

from module.utils import setup_logging

from .runner import load_scenarios, run_experiment


def main(argv: list[str]) -> int:
    setup_logging()
    if len(argv) != 2 or argv[0] != "run":
        print("Uso: python -m module.experiments run <fichero_de_escenarios.py | todos>", file=sys.stderr)
        return 2
    try:
        scenarios = load_scenarios(argv[1])
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    exp_dir = run_experiment(scenarios)
    print(f"Experimento completo. Informe: {exp_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
