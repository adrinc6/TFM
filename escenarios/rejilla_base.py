"""Rejilla del barrido — Etapa A: motor y objetivo (plan revisado, Camino C).

Esta etapa AISLA el efecto del motor de aprendizaje y del objetivo. Todos los escenarios usan
el mismo meta simple (rank_ic), las mismas features, ventana, fechas y cartera de diagnostico.
Asi la comparacion es limpia: lo unico que cambia es el modelo.

La seleccion se hace por rank-IC del meta_final (no por rentabilidad). La Puerta 1 decide si
LightGBM aporta senal sobre Ridge de forma estable; solo entonces se pasa a las Etapas B/C/D
(regularizacion, meta, cartera), que se activan editando esta rejilla.

Ver docs/plan_camino_c_revisado.md y la bitacora para el detalle metodologico.
"""

from module.experiments import ScenarioSpec


SCENARIOS = [
    # --- Etapa A: motor x objetivo, mismo meta simple ---------------------------------
    # Control conocido: Ridge lineal sobre el percentil del retorno (la mejor config de la Parte B).
    ScenarioSpec(
        name="A_ridge_rankreg",
        overrides={"model_type": "ridge", "objective": "regression",
                   "label_transform": "rank", "meta_type": "rank_ic"},
    ),
    # Comparacion limpia: cambia SOLO el motor (LightGBM), mismo objetivo de ranking.
    ScenarioSpec(
        name="A_lgbm_rankreg",
        overrides={"model_type": "lightgbm", "objective": "rank_regression",
                   "meta_type": "rank_ic"},
    ),
    # LGBMRanker (lambdarank) agrupado por snapshot: optimiza el orden directamente.
    ScenarioSpec(
        name="A_lgbm_ranker",
        overrides={"model_type": "lightgbm", "objective": "ranking",
                   "meta_type": "rank_ic"},
    ),
    # Ablacion: clasificacion de cuartiles (descarta el centro del universo en entrenamiento).
    ScenarioSpec(
        name="A_lgbm_quartile",
        overrides={"model_type": "lightgbm", "objective": "quartile",
                   "meta_type": "rank_ic"},
    ),
]
