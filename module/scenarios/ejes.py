"""Ejes del sistema aislados sobre un baseline comun (consumido por la Fase 1 del orquestador).

Filosofia (igual que antes, pero ampliada): NO producto cartesiano. Se AISLA el
efecto de cada eje moviendo solo ese eje y dejando el resto fijo en el baseline. Asi el rank-IC
del meta_final de cada escenario mide el efecto de UNA cosa. La Fase 2 (generada despues) combina
los mejores niveles de cada eje.

Ejes barridos (baseline = ancla 2016, ventana 10a, horizonte 3m, profundidad 4, cadencia
trimestral, sin artefactos):

  - Ventana de entrenamiento: 5 / 6 / 7 / 8 / 10 / 12 anios.
  - Horizonte de etiqueta:    1 / 3 / 6 / 12 meses  (cambia QUE se predice).
  - Ancla de evaluacion:      2012 / 2014 / 2016 / 2018 / 2020.
  - Profundidad LightGBM:     3 / 4 / 5 / 6.
  - Cadencia de reentreno:    trimestral / semestral / anual.
  - Artefactos:               los 7, uno a uno.

El orquestador (`module.experiments.decide_best_config`) lee `ARTIFACTS` y `AXES` y compone la
configuracion final: mejor nivel de cada eje + artefactos aceptados. Ver docs/doc.md.

Los valores baseline de cada eje (train_10y, horizon_3m, ancla_2016, depth_4, cadence_trimestral)
apuntan al MISMO escenario `baseline` para que el orquestador pueda elegir "quedarse como esta".
"""

from module.runs.experiments import ScenarioSpec


# --- Artefactos (uno a uno, sobre el baseline) ---------------------------------------------
ARTIFACTS = {
    "sector": {"neutralize_by_sector": True},
    "fund_momentum": {"fundamental_momentum": True},
    "regime_bull_bear": {"market_regime_feature": True},
    "price_momentum": {"price_momentum_multi": True},
    "moving_averages": {"moving_averages": True},
    "regime_extended": {"regime_extended": True},
    "quality_growth": {"quality_growth_derived": True},
}


# --- Ejes con niveles: {eje -> {escenario -> overrides}} -----------------------------------
# El nivel que coincide con el baseline apunta a "baseline" (mismo run, sin duplicar).
AXES: dict[str, dict[str, dict]] = {
    "train_lookback_years": {
        "train_5y": {"train_lookback_years": 5},
        "train_6y": {"train_lookback_years": 6},
        "train_7y": {"train_lookback_years": 7},
        "train_8y": {"train_lookback_years": 8},
        "baseline": {},                                  # 10 anios (default)
        "train_12y": {"train_lookback_years": 12},
    },
    "target_horizon_months": {
        "horizon_1m": {"target_horizon_months": 1},
        "baseline": {},                                  # 3 meses (default)
        "horizon_6m": {"target_horizon_months": 6},
        "horizon_12m": {"target_horizon_months": 12},
    },
    "execution_year": {
        "anchor_2012": {"execution_year": 2012},
        "anchor_2014": {"execution_year": 2014},
        "baseline": {},                                  # ancla 2016 (default)
        "anchor_2018": {"execution_year": 2018},
        "anchor_2020": {"execution_year": 2020},
    },
    "lgbm_max_depth": {
        "depth_3": {"lgbm_max_depth": 3},
        "baseline": {},                                  # profundidad 4 (default)
        "depth_5": {"lgbm_max_depth": 5},
        "depth_6": {"lgbm_max_depth": 6},
    },
    "fundamental_step_months": {
        "baseline": {},                                  # trimestral (default)
        "cadence_semestral": {"fundamental_step_months": 6},
        "cadence_anual": {"fundamental_step_months": 12},
    },
}


def _axis_scenarios() -> list[ScenarioSpec]:
    """Un ScenarioSpec por nivel no-baseline de cada eje (el baseline se define una sola vez)."""
    seen: set[str] = set()
    specs: list[ScenarioSpec] = []
    for levels in AXES.values():
        for name, overrides in levels.items():
            if name == "baseline" or name in seen:
                continue
            seen.add(name)
            specs.append(ScenarioSpec(name=name, overrides=overrides))
    return specs


SCENARIOS = [
    ScenarioSpec(name="baseline", overrides={}),
    *[ScenarioSpec(name=f"artifact_{name}", overrides=overrides)
      for name, overrides in ARTIFACTS.items()],
    *_axis_scenarios(),
]
