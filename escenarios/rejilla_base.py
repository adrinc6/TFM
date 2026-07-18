"""Rejilla de ablations dirigidas — el estudio automatico del sistema.

Filosofia: NO producto cartesiano (seria inmanejable y sobreajustaria por seleccion). En su
lugar, ablations que AISLAN el efecto de cada cosa sobre el rank-IC del meta_final:

  - Bloque 1: baseline vs baseline + cada artefacto por separado (que aporta cada uno).
  - Bloque 2: hiperparametros del modelo sobre el baseline (profundidad).
  - Bloque 3: cadencia de reentreno (trimestral / semestral / anual).
  - Bloque 4: ventana de entrenamiento (8 / 10 / 12 anios).

El sistema es LightGBM + rank_regression + meta rank_ic, ancla 2016. Tras el barrido, el
orquestador (`module.experiments.decide_accepted_artifacts`) marca automaticamente que artefactos
mejoran el rank-IC de forma estable y compone la configuracion final. Ver docs/doc.md.
"""

from module.experiments import ScenarioSpec


# Los artefactos que se prueban de uno en uno (flag de Settings + nombre legible).
ARTIFACTS = {
    "sector": {"neutralize_by_sector": True},
    "fund_momentum": {"fundamental_momentum": True},
    "regime_bull_bear": {"market_regime_feature": True},
    "price_momentum": {"price_momentum_multi": True},
    "moving_averages": {"moving_averages": True},
    "regime_extended": {"regime_extended": True},
    "quality_growth": {"quality_growth_derived": True},
}


SCENARIOS = [
    # --- Bloque 1: baseline + cada artefacto aislado -------------------------------------
    ScenarioSpec(name="baseline", overrides={}),
    *[
        ScenarioSpec(name=f"artifact_{name}", overrides=overrides)
        for name, overrides in ARTIFACTS.items()
    ],
    # --- Bloque 2: hiperparametros del modelo (sobre baseline) ---------------------------
    ScenarioSpec(name="depth_3", overrides={"lgbm_max_depth": 3}),
    ScenarioSpec(name="depth_6", overrides={"lgbm_max_depth": 6}),
    # --- Bloque 3: cadencia de reentreno -------------------------------------------------
    ScenarioSpec(name="cadence_semestral", overrides={"fundamental_step_months": 6}),
    ScenarioSpec(name="cadence_anual", overrides={"fundamental_step_months": 12}),
    # --- Bloque 4: ventana de entrenamiento ----------------------------------------------
    ScenarioSpec(name="train_8y", overrides={"train_lookback_years": 8}),
    ScenarioSpec(name="train_12y", overrides={"train_lookback_years": 12}),
]
