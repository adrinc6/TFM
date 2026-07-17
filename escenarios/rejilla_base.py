"""Rejilla base de escenarios para el barrido de la Fase 6.

Cada `ScenarioSpec` es una config nombrada con overrides sobre `environment.Settings`.
Selección dirigida (no producto cartesiano) organizada por categoría. La reutilización de
etapas se decide automáticamente por huella: los escenarios que solo cambian el backtest
comparten dataset/features/agents con `baseline`.

Los categorías:

- **Ancla**: distintas fechas de arranque, para ver si el sistema no solo funciona en 2000.
- **Entrenamiento**: ventana de historia usada para reentrenar.
- **Cadencia**: cada cuánto se reentrena el fundamental.
- **Etiqueta**: horizonte de la etiqueta (3/6/12 meses).
- **Cartera** (solo cambian backtest): tamaño y política de rotación.
- **Ablations** (solo cambia agentes): quitar un agente para ver qué aporta.
- **Observabilidad**: margen de frescura del precio.
"""

from module.experiments import ScenarioSpec


SCENARIOS = [
    # --- baseline ---------------------------------------------------------------------
    ScenarioSpec(name="baseline_2000q1", overrides={}),

    # --- ancla temporal ---------------------------------------------------------------
    ScenarioSpec(name="anchor_2003q1", overrides={"execution_year": 2003}),
    ScenarioSpec(name="anchor_2008q1", overrides={"execution_year": 2008}),
    ScenarioSpec(name="anchor_2013q1", overrides={"execution_year": 2013}),
    ScenarioSpec(name="anchor_2018q1", overrides={"execution_year": 2018}),

    # --- entrenamiento ----------------------------------------------------------------
    ScenarioSpec(name="train_5y", overrides={"train_lookback_years": 5}),
    ScenarioSpec(name="train_12y", overrides={"train_lookback_years": 12}),

    # --- cadencia fundamental ---------------------------------------------------------
    ScenarioSpec(name="cadence_semestral", overrides={"fundamental_step_months": 6}),

    # --- horizonte de etiqueta --------------------------------------------------------
    ScenarioSpec(name="target_6m", overrides={"target_horizon_months": 6}),
    ScenarioSpec(name="target_12m", overrides={"target_horizon_months": 12}),

    # --- cartera (solo cambian backtest) ----------------------------------------------
    ScenarioSpec(name="portfolio_3_7", overrides={"target_min": 3, "target_max": 7,
                                                    "max_weight_per_position": 0.35}),
    ScenarioSpec(name="portfolio_8_15", overrides={"target_min": 8, "target_max": 15,
                                                     "max_weight_per_position": 0.15}),
    ScenarioSpec(name="rotation_strict", overrides={"rotation_edge_percentiles": 10}),
    ScenarioSpec(name="rotation_loose", overrides={"rotation_edge_percentiles": 3}),
    ScenarioSpec(name="entry_high_bar", overrides={"entry_min_percentile": 90}),

    # --- observabilidad --------------------------------------------------------------
    ScenarioSpec(name="stale_prices_3d", overrides={"max_price_age_days": 3}),
    ScenarioSpec(name="stale_prices_14d", overrides={"max_price_age_days": 14}),
]
