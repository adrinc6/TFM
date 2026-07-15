"""Bloque A — ¿La IA aprende de verdad? (ablaciones y placebo)

Cada escenario DESACTIVA una pieza del aprendizaje y se lee contra el baseline. La lógica: si apagar
el aprendizaje NO empeora el rank-IC out-of-sample ni el percentil placebo, es que el sistema no
estaba aprendiendo nada útil (resultado negativo válido y honesto). Si empeora, es evidencia de que
el aprendizaje aportaba señal real, no overfitting ni suerte.

Estos escenarios cambian el scoring del ML -> RE-ENTRENAN (más lentos que los de estrategia).

Lanzar con:
    python -m module.experiments run experiments/escenarios_aprendizaje.py
"""

from module.experiments import Scenario

_Q, _T, _A = "quality_probability", "timing_probability", "alpha_probability"

SCENARIOS = [
    Scenario(
        name="baseline",
        why="Aprendizaje completo activo: meta-agente aprendido + prior tilteado a calidad. Referencia.",
        block="baseline",
    ),
    Scenario(
        name="sin_meta_aprendido",
        why="Fija los pesos al prior sin aprender por trimestre (los agentes siguen puntuando walk-forward). Si el rank-IC OOS baja poco, el meta-agente aprendido no aporta.",
        overrides={"ml.LEARN_META_WEIGHTS": False},
        block="aprendizaje",
    ),
    Scenario(
        name="pesos_equal",
        why="Quita el prior tilteado a calidad (1/3 cada agente). ¿La calibración de pesos aporta, o vale igual repartir?",
        overrides={"ml.AGENT_PRIOR_WEIGHTS": {_Q: 1 / 3, _T: 1 / 3, _A: 1 / 3}},
        block="aprendizaje",
    ),
    Scenario(
        name="solo_calidad",
        why="Un único agente (el más estable). Aísla cuánto del rank-IC viene de la combinación vs. de calidad sola.",
        overrides={"ml.AGENT_PRIOR_WEIGHTS": {_Q: 1.0, _T: 0.0, _A: 0.0}},
        block="aprendizaje",
    ),
    Scenario(
        name="solo_alpha",
        why="El agente que rankea alpha directo pero es inestable. Debe mostrar rank-IC alto y errático (pocos años positivos).",
        overrides={"ml.AGENT_PRIOR_WEIGHTS": {_Q: 0.0, _T: 0.0, _A: 1.0}},
        block="aprendizaje",
    ),
    Scenario(
        name="sin_consistencia",
        why="Apaga la penalización de varianza entre sub-folds del meta-agente. ¿Sube la dispersión año a año del rank-IC?",
        overrides={"ml.CONSISTENCY_LAMBDA": 0.0},
        block="aprendizaje",
    ),
    Scenario(
        name="sin_shrinkage",
        why="Permite soluciones esquina 100/0/0. Debe reaparecer el colapso a un agente que el shrinkage evita.",
        overrides={"ml.META_WEIGHT_FLOOR": 0.0},
        block="aprendizaje",
    ),
]
