"""Bloque B — ¿El sistema es estable y consistente?

Comprueba que el resultado no depende de una semilla afortunada ni de un único tramo temporal, y que
sobrevive a supuestos más duros. La estabilidad se mide con la DISPERSIÓN de rank_ic_final_mean y
cumulative_alpha entre las 5 semillas, y con los sub-períodos (¿el edge está repartido o concentrado
en un solo tramo?).

Los escenarios de semilla y de ventana/horizonte RE-ENTRENAN; los de costes reutilizan scoring.

Lanzar con:
    python -m module.experiments run experiments/escenarios_estabilidad.py
"""

from module.experiments import Scenario

SCENARIOS = [
    Scenario(name="baseline", why="Semilla 42, costes 5/10 bps, ventana 4 años, horizonte 12m. Referencia."),
    *[
        Scenario(
            name=f"semilla_{s}",
            why=f"Re-siembra LightGBM con random_state={s}. Un sistema estable no cambia de conclusión al re-sembrar.",
            overrides={"ml.RANDOM_STATE": s},
        )
        for s in (1, 7, 13, 29)
    ],
    Scenario(
        name="costes_realistas",
        why="Costes al doble (10/20 bps). ¿La utilidad sobrevive? Se lee contra el breakeven de costes.",
        overrides={"settings.transaction_cost_bps": 10.0, "settings.slippage_bps": 20.0},
    ),
    Scenario(
        name="ventana_corta",
        why="Ventana de entrenamiento de 3 años en vez de 4. ¿Adaptarse más rápido al régimen mejora o empeora el rank-IC?",
        # Baja min Y max a 3: si solo se bajara max, min (4) > max (3) invalidaría la ventana y todo
        # caería a fallback (rank-IC degenerado ~1.0). Ambos a 3 = ventana real de 3 años.
        overrides={
            "settings.max_walk_forward_training_years": 3,
            "settings.min_walk_forward_training_years": 3,
        },
    ),
    Scenario(
        name="horizonte_6m",
        why="Horizonte de etiqueta a 6m en vez de 12m. ¿Rankea mejor el forward a medio plazo?",
        overrides={"settings.walk_forward_label_horizon_months": 6},
    ),
]
