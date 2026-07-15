"""Todos los bloques juntos: aprendizaje + estabilidad + utilidad + pesos_meta en un solo barrido.

Reúne los SCENARIOS de los cuatro ficheros de bloque. El runner deduplica los 'baseline' repetidos
(cada bloque define el suyo) y agrupa por clave de scoring para no re-entrenar de más. Es el barrido
más largo (muchos re-entrenamientos): úsalo para la tanda completa, no para iterar rápido.

Lanzar con:
    python -m module.experiments run experiments/escenarios_todos.py

o, desde main.py, con RUN_MODE="experiments" y EXPERIMENTS_FILE="todos".
"""

from experiments.escenarios_aprendizaje import SCENARIOS as _aprendizaje
from experiments.escenarios_estabilidad import SCENARIOS as _estabilidad
from experiments.escenarios_pesos_meta import SCENARIOS as _pesos_meta
from experiments.escenarios_utilidad import SCENARIOS as _utilidad

SCENARIOS = [*_aprendizaje, *_estabilidad, *_utilidad, *_pesos_meta]
