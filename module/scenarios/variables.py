"""Catálogo de valores admitidos por variable, fuente única para la consola y el orquestador.

`STUDY_OPTIONS` define, para cada campo de `Settings` barrible, la lista de valores permitidos.
Lo usan tanto la UI (consola: selects guiados de Experimental y Study) como el orquestador de
estudios (`module/runs/execution.py`), que deriva de aquí el barrido completo del `full_study`.
Vive fuera de `module/ui/` para que el orquestador no dependa de la capa de presentación.

La separación entre variables de modelo y de cartera NO se decide aquí: es autoritativa en
`module/runs/experiments.py` (`MODEL_FIELDS` / `PORTFOLIO_FIELDS`, derivadas de `FINGERPRINT_FIELDS`).
"""

from __future__ import annotations


# The full catalogue is the baseline.  These complementary configurations turn the
# study into a genuine ablation: every block is removed once, while the remaining
# blocks stay enabled.  This isolates the incremental contribution of a block
# without making the study try every one of the 2^11 possible subsets.
FULL_FEATURE_BLOCKS = (
    "quality_core", "quality_efficiency", "financial_strength", "value_core", "value_cashflow",
    "growth_acceleration", "fundamental_stability", "momentum_core", "momentum_trend",
    "price_risk", "market_liquidity",
)
BLOCK_ABLATIONS = [tuple(block for block in FULL_FEATURE_BLOCKS if block != removed)
                   for removed in FULL_FEATURE_BLOCKS]
FULL_AGENTS = ("quality", "value", "growth", "momentum", "risk")
# Igual que con los bloques, cada escenario conserva el resto de agentes.  De ese
# modo la diferencia frente al baseline es el aporte incremental del agente
# retirado, no una comparación confusa entre arquitecturas distintas.
AGENT_ABLATIONS = [tuple(agent for agent in FULL_AGENTS if agent != removed)
                   for removed in FULL_AGENTS]
COST_STRESS_CASES = tuple(
    {"name": f"commission_{commission}_slippage_{slippage}",
     "overrides": {"commission_bps": commission, "slippage_bps": slippage}}
    for commission in (0, 5, 10) for slippage in (5, 10, 20)
)


STUDY_OPTIONS: dict[str, list] = {
    # El ancla temporal (execution_year=2015, execution_quarter=1) es FIJA y no se barre: así todos
    # los escenarios comparten el mismo periodo OOS (2015→hoy) y 2025-26 reservados. Sí se barre el
    # retardo de publicación de fundamentales (execution_lag_days).
    "execution_lag_days": [15, 30, 45, 60], "train_lookback_years": [2, 4, 6, 8, 10, 12],
    "snapshot_step_months": [1, 3], "fundamental_step_months": [3, 6, 12],
    "target_horizon_months": [1, 3, 6, 12], "objective": ["rank_regression", "ranking", "quartile"],
    "lgbm_n_estimators": [100, 200, 400], "lgbm_max_depth": [3, 4, 5, 6, 8],
    "lgbm_learning_rate": [0.02, 0.03, 0.05, 0.10], "lgbm_min_child_samples": [20, 50, 100],
    # meta_type combina los agentes de forma distinta (ver module/modeling/meta.py).
    # Sus parámetros temporales son ejes de modelo porque pueden mover el Rank-IC OOS.
    "meta_type": ["equal", "rank_ic", "regime", "stacked_oos"],
    "meta_ic_lookback_quarters": [8, 12, 16], "min_rank_ic_cross_section": [8, 10, 12],
    "recency_weighting": ["off", "linear", "exponential"],
    "neutralize_by_sector": [False, True],
    "fundamental_momentum": [False, True],
    "market_regime_feature": [False, True], "price_momentum_multi": [False, True],
    "moving_averages": [False, True], "regime_extended": [False, True],
    "quality_growth_derived": [False, True], "target_size": [5, 8, 10, 12, 15],
    # Ablaciones del laboratorio: el catálogo sigue completo, pero cada escenario prueba la
    # contribución de bloques, agentes y familias de modelos de forma aislada.
    "enabled_feature_blocks": [
        ("quality_core", "value_core", "momentum_core"),
        *BLOCK_ABLATIONS,
        FULL_FEATURE_BLOCKS,
    ],
    "enabled_agents": [
        ("quality", "value", "momentum"),
        *AGENT_ABLATIONS,
        FULL_AGENTS,
    ],
    "enabled_model_families": [("lightgbm",), ("lightgbm", "elastic_net"),
                                ("lightgbm", "elastic_net", "catboost")],
    "intra_agent_ensemble_mode": ["single", "equal_rank", "rank_ic_weighted"],
    "feature_weighting_mode": ["model_native", "diagnostic_only", "oos_stability_prune",
                                 "regularized_linear_ensemble", "block_gated"],
    "feature_selection_min_coverage": [0.40, 0.55, 0.70],
    "feature_selection_lookback_quarters": [8, 12, 16],
    "feature_selection_min_permutation_importance": [0.0, 0.001, 0.005],
    "feature_selection_min_positive_fraction": [0.40, 0.50, 0.60],
    "feature_selection_max_features_per_agent": [0, 8, 12, 20],
    "metric_winsorization_percentile": [0.0, 0.01, 0.025],
    "risk_feature_windows": [(63, 126, 252), (21, 63, 126), (63, 252)],
    "technical_feature_windows": [(21, 63, 252), (10, 21, 63), (21, 126, 252)],
    "min_hold_percentile": [60, 70, 80, 85], "rotation_edge_percentiles": [5, 10, 15],
    "profile": ["balanced", "conservative", "aggressive",
                                             "value", "quality", "momentum", "garp", "contrarian"],
}

# Experimental conserva el catálogo amplio de ajustes individuales. Study y full_study comparten
# en cambio exactamente el mismo contrato científico y la misma asignación de fases.
EXPERIMENT_OPTIONS: dict[str, list] = {axis: list(values) for axis, values in STUDY_OPTIONS.items()}
EXPERIMENT_OPTIONS["commission_bps"] = [0, 5, 10]
EXPERIMENT_OPTIONS["slippage_bps"] = [5, 10, 20]
STUDY_OPTIONS["target_horizon_months"] = [3, 6, 12]
for _axis in ("commission_bps", "slippage_bps", "profile"):
    STUDY_OPTIONS.pop(_axis, None)


# --- Contrato común study/full_study -----------------------------------------------------------
# Ambos comparten ejes, valores y fases. El study manual permite elegir subconjuntos; el full_study
# ejecuta el catálogo completo. Los hiperparámetros de Fase 3 se retiran de Fases 1–2, no del
# catálogo científico. `snapshot_day` no existe: la rejilla la define execution_lag_days.
FULL_STUDY_LEVEL_OVERRIDES: dict[str, list] = {}

FULL_STUDY_OPTIONS: dict[str, list] = {
    axis: FULL_STUDY_LEVEL_OVERRIDES.get(axis, list(values))
    for axis, values in STUDY_OPTIONS.items()
}
# Estos ejes se afinan exclusivamente en Fase 3 para no duplicar reentrenos en Fase 1.
FULL_STUDY_PHASE3_OPTIONS: dict[str, list] = {
    axis: list(STUDY_OPTIONS[axis])
    for axis in ("lgbm_max_depth", "lgbm_n_estimators", "lgbm_learning_rate",
                 "lgbm_min_child_samples")
}
for _axis in ("lgbm_max_depth", "lgbm_n_estimators", "lgbm_learning_rate", "lgbm_min_child_samples"):
    FULL_STUDY_OPTIONS.pop(_axis, None)
# El eje de cartera es idéntico en study y full_study.
