"""Bloque C — ¿El sistema es útil? (mejora económica real)

Cuantifica cuánto añade el sistema completo sobre reglas simples y auditables, variando la
agresividad de la cartera SIN tocar el ML. Todos estos escenarios comparten el mismo scoring que el
baseline -> reutilizan la caché (re_scored=False), son rápidos.

Lectura clave: alpha_vs_best_baseline (¿el sistema bate a equal_weight/momentum/valuation?) y la
alpha NETA de costes (cost_breakeven_multiplier). Si no supera claramente a las reglas triviales, es
un hallazgo honesto que el TFM debe reportar.

Lanzar con:
    python -m module.experiments run experiments/escenarios_utilidad.py
"""

from module.experiments import Scenario

SCENARIOS = [
    Scenario(name="baseline", why="Sistema completo con la configuración de cartera por defecto. Referencia.", block="baseline"),
    Scenario(
        name="concentrada",
        why="Concentra en las de mayor convicción (cap 22%, más convexidad). ¿Sube la alpha o solo la varianza?",
        overrides={"strategy.MAX_POSITION_WEIGHT": 0.22, "strategy.CONVICTION_CONVEXITY": 1.6},
        block="utilidad",
    ),
    Scenario(
        name="rotacion_baja",
        why="Menos turnover (umbral de rotación 0.15, mínimo 6 meses de tenencia). ¿Cae la alpha bruta pero sube la neta de costes?",
        overrides={"strategy.MIN_ROTATION_ADVANTAGE": 0.15, "strategy.MIN_HOLD_MONTHS_BEFORE_ROTATION": 6},
        block="utilidad",
    ),
    Scenario(
        name="cartera_amplia",
        why="Más nombres (hasta 15), menos concentración. Trade-off diversificación vs. dilución de la señal.",
        overrides={"strategy.MAX_PORTFOLIO_SIZE": 15},
        block="utilidad",
    ),
]
