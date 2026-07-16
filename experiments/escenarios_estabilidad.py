"""Bloque B — ¿El sistema es estable y consistente?

Comprueba que el resultado no depende de un único tramo temporal ni de un supuesto afortunado, y que
sobrevive a supuestos más duros. La estabilidad se lee con los sub-períodos (¿el edge está repartido
o concentrado en un solo tramo?) y con la sensibilidad a ventana, horizonte y costes.

NOTA METODOLÓGICA — por qué NO hay escenarios de semilla:
El modelo es DETERMINISTA por construcción. `LGBM_PARAM_GRID` (module/ml.py) no fija `subsample` ni
`colsample_bytree`, así que LightGBM corre con submuestreo de filas y columnas al 100%: un GBDT sin
componente estocástico. Variar `ml.RANDOM_STATE` no cambia NADA — comprobado empíricamente en el
barrido 20260716_105304, donde semilla_1/7/13/29 devolvieron rank-IC y alpha bit-idénticos al
baseline (0.215461 / 1.137596) pese a re-entrenar de verdad. Mantener esos escenarios gastaba ~85
min por barrido para producir una columna constante y, peor, presentaba como "robustez ante la
semilla" algo que solo demostraba determinismo. Se documenta como limitación en vez de simularlo.

Los escenarios de ventana/horizonte RE-ENTRENAN; los de costes reutilizan scoring.

Lanzar con:
    python -m module.experiments run experiments/escenarios_estabilidad.py
"""

from module.experiments import Scenario

SCENARIOS = [
    Scenario(name="baseline", why="Costes 5/10 bps, ventana 4 años, horizonte 12m. Referencia.", block="baseline"),
    Scenario(
        name="costes_realistas",
        why="Costes al doble (10/20 bps). ¿La utilidad sobrevive? Se lee contra el breakeven de costes.",
        overrides={"settings.transaction_cost_bps": 10.0, "settings.slippage_bps": 20.0},
        block="estabilidad",
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
        block="estabilidad",
    ),
    Scenario(
        name="horizonte_6m",
        why="Horizonte de etiqueta a 6m en vez de 12m. ¿Rankea mejor el forward a medio plazo?",
        overrides={"settings.walk_forward_label_horizon_months": 6},
        block="estabilidad",
    ),
    Scenario(
        name="ventana_larga",
        why="Ventana de 6 años en vez de 4. Contrapeso de ventana_corta: si el resultado cambia mucho "
            "entre 3, 4 y 6 años, el edge depende de la ventana elegida y no es estable.",
        overrides={
            "settings.max_walk_forward_training_years": 6,
            "settings.min_walk_forward_training_years": 4,
        },
        block="estabilidad",
    ),
    Scenario(
        name="horizonte_24m",
        why="Horizonte de etiqueta a 24m. Junto a 6m y 12m traza la sensibilidad al horizonte: un edge "
            "real no debería vivir solo en un horizonte concreto.",
        overrides={"settings.walk_forward_label_horizon_months": 24},
        block="estabilidad",
    ),
    Scenario(
        name="costes_extremos",
        why="Costes 4x (20/40 bps). Junto a costes_realistas acota cuánto coste aguanta la utilidad "
            "antes de desaparecer; se lee contra el breakeven de costes.",
        overrides={"settings.transaction_cost_bps": 20.0, "settings.slippage_bps": 40.0},
        block="estabilidad",
    ),
]
