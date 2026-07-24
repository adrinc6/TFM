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


def build_ablations(selected: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Conjunto completo + una variante quitando cada item uno a uno, sobre `selected`.

    Es el mismo patrón "full-minus-one" que usan las ablaciones fijas del full study, pero
    parametrizado sobre un conjunto base arbitrario (el que el usuario elija en el study), en vez
    de sobre `FULL_FEATURE_BLOCKS`/`FULL_AGENTS`. Con un solo item devuelve solo ese conjunto (no
    tiene sentido quitarlo y quedarse sin nada). Preserva el orden y elimina duplicados.
    """
    base = tuple(dict.fromkeys(selected))  # dedup conservando orden
    if len(base) <= 1:
        return [base] if base else []
    ablated = [tuple(item for item in base if item != removed) for removed in base]
    return [base, *ablated]


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
    # los escenarios comparten el mismo periodo OOS (2015→hoy) y 2025-26 queda como estrés conocido. Sí se barre el
    # retardo de publicación de fundamentales (execution_lag_days).
    "execution_lag_days": [15, 30, 45, 60], "train_lookback_years": [2, 4, 6, 8, 10, 12],
    "snapshot_step_months": [1, 3], "fundamental_step_months": [3, 6, 12],
    "target_horizon_months": [1, 3, 6, 12], "objective": ["rank_regression", "ranking", "quartile"],
    "lgbm_n_estimators": [100, 200, 400], "lgbm_max_depth": [3, 4, 5, 6, 8],
    "lgbm_learning_rate": [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20],
    "lgbm_min_child_samples": [20, 50, 100],
    # meta_type combina los agentes de forma distinta (ver module/modeling/meta.py).
    # Sus parámetros temporales son ejes de modelo porque pueden mover el Rank-IC OOS.
    "meta_type": ["equal", "rank_ic", "regime", "stacked_oos"],
    "meta_ic_lookback_quarters": [8, 12, 16], "min_rank_ic_cross_section": [8, 10, 12],
    "recency_weighting": ["off", "linear", "exponential"],
    "neutralize_by_sector": [False, True],
    "fundamental_momentum": [False, True],
    "market_regime_feature": [False, True], "price_momentum_multi": [False, True],
    "moving_averages": [False, True], "regime_extended": [False, True],
    "quality_growth_derived": [False, True], 
    # "target_size": [5, 8, 10, 12, 15],
    "target_size": [8, 10, 12, 15, 20],
    # Ablaciones del laboratorio: el catálogo sigue completo, pero cada escenario prueba la
    # contribución de bloques, agentes y familias de modelos de forma aislada.
    # Ablación homogénea: SOLO el catálogo completo y el completo-menos-cada-uno. No hay un
    # subconjunto "histórico" privilegiado: todos los bloques/agentes se tratan por igual, y la
    # diferencia frente al full mide la contribución incremental de lo retirado.
    "enabled_feature_blocks": [
        *BLOCK_ABLATIONS,
        FULL_FEATURE_BLOCKS,
    ],
    "enabled_agents": [
        *AGENT_ABLATIONS,
        FULL_AGENTS,
    ],
    "enabled_model_families": [("lightgbm",), ("lightgbm", "elastic_net"),
                                ("lightgbm", "elastic_net", "catboost")],
    "intra_agent_ensemble_mode": ["single", "equal_rank", "rank_ic_weighted"],
    "feature_weighting_mode": ["model_native", "diagnostic_only", "oos_stability_prune",
                                 "regularized_linear_ensemble", "block_gated"],
    "feature_selection_min_coverage": [0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80],
    "feature_selection_lookback_quarters": [8, 12, 16],
    "feature_selection_min_permutation_importance": [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01],
    "feature_selection_min_positive_fraction": [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70],
    "feature_selection_max_features_per_agent": [0, 8, 12, 20],
    "metric_winsorization_percentile": [0.0, 0.005, 0.01, 0.02, 0.025, 0.05],
    "risk_feature_windows": [(63, 126, 252), (21, 63, 126), (63, 252)],
    "technical_feature_windows": [(21, 63, 252), (10, 21, 63), (21, 126, 252)],
    "min_hold_percentile": [60, 70, 80, 85], "rotation_edge_percentiles": [5, 10, 15],
    "rebalance_drift_tolerance": [0.15, 0.20, 0.25, 0.30, 0.35],
    # Endurece expulsión/rotación/rebalanceo en las revisiones sin fundamentales nuevos (solo
    # precio). Eje MECÁNICO: se estresa para demostrar estabilidad, nunca se elige el más rentable.
    "price_only_strictness_multiplier": [1.0, 1.5, 2.0],
    "profile": ["balanced", "growth", "value", "quality",
                                             "momentum", "contrarian", "defensive", "garp"],
}

# Experimental y study manual conservan el catálogo amplio de ajustes individuales. El full study
# oficial usa más abajo un protocolo confirmatorio separado.
EXPERIMENT_OPTIONS: dict[str, list] = {axis: list(values) for axis, values in STUDY_OPTIONS.items()}
EXPERIMENT_OPTIONS["commission_bps"] = [0, 5, 10]
EXPERIMENT_OPTIONS["slippage_bps"] = [5, 10, 20]
STUDY_OPTIONS["target_horizon_months"] = [3, 6, 12]
for _axis in ("commission_bps", "slippage_bps", "profile"):
    STUDY_OPTIONS.pop(_axis, None)


# --- Contrato histórico/manual ----------------------------------------------------------------
# Se conserva para studies exploratorios y compatibilidad con resultados antiguos. `snapshot_day`
# no existe: la rejilla la define execution_lag_days.
MANUAL_STUDY_LEVEL_OVERRIDES: dict[str, list] = {}

MANUAL_STUDY_MODEL_AND_PORTFOLIO_OPTIONS: dict[str, list] = {
    axis: MANUAL_STUDY_LEVEL_OVERRIDES.get(axis, list(values))
    for axis, values in STUDY_OPTIONS.items()
}
# Estos ejes se afinan exclusivamente en Fase 3 para no duplicar reentrenos en Fase 1.
MANUAL_STUDY_PHASE3_OPTIONS: dict[str, list] = {
    axis: list(STUDY_OPTIONS[axis])
    for axis in ("lgbm_max_depth", "lgbm_n_estimators", "lgbm_learning_rate",
                 "lgbm_min_child_samples")
}
for _axis in ("lgbm_max_depth", "lgbm_n_estimators", "lgbm_learning_rate", "lgbm_min_child_samples"):
    MANUAL_STUDY_MODEL_AND_PORTFOLIO_OPTIONS.pop(_axis, None)
# `enabled_agents` NO se optimiza en el full_study: el finalista conserva siempre los 5 agentes.
# Motivo metodológico: los perfiles de inversor (fase final) ponderan los rangos de los 5 agentes
# (`apply_profile`); si el finalista prescinde de uno (p.ej. quality), esa columna desaparece y los
# perfiles que dependen de ella quedan deformes sin aviso, produciendo resultados que no representan
# el arquetipo. Manteniendo los 5 agentes fijos, los perfiles son comparables y honestos. El aporte
# incremental de cada agente sigue siendo estudiable como ablación informativa desde la consola
# Experimental (donde `enabled_agents` sí sigue disponible en STUDY_OPTIONS), no como eje que altera
# el modelo recomendado.
MANUAL_STUDY_MODEL_AND_PORTFOLIO_OPTIONS.pop("enabled_agents", None)
# El eje de cartera es idéntico en study y full_study.

# El catálogo amplio sigue siendo la fuente del study manual. El full study oficial dejó de ser
# "todo el catálogo": desde study-2 es un protocolo confirmatorio pre-registrado y acotado.
MANUAL_STUDY_OPTIONS: dict[str, list] = {
    axis: list(values) for axis, values in STUDY_OPTIONS.items()
}

INCUMBENT_FEATURE_BLOCKS = (
    "quality_core", "quality_efficiency", "financial_strength", "value_core", "value_cashflow",
    "growth_acceleration", "fundamental_stability", "momentum_trend", "price_risk",
    "market_liquidity",
)

INCUMBENT_MODEL_OVERRIDES: dict[str, object] = {
    "execution_lag_days": 60,
    "train_lookback_years": 12,
    "target_horizon_months": 12,
    "meta_type": "stacked_oos",
    "meta_ic_lookback_quarters": 16,
    "fundamental_momentum": True,
    "market_regime_feature": True,
    "price_momentum_multi": True,
    "moving_averages": True,
    "enabled_feature_blocks": INCUMBENT_FEATURE_BLOCKS,
    "enabled_agents": FULL_AGENTS,
    "enabled_model_families": ("lightgbm",),
    "intra_agent_ensemble_mode": "single",
    "feature_weighting_mode": "oos_stability_prune",
    "feature_selection_min_coverage": 0.30,
    "feature_selection_min_positive_fraction": 0.45,
    "feature_selection_max_features_per_agent": 8,
    "metric_winsorization_percentile": 0.0,
    "lgbm_max_depth": 3,
    "lgbm_n_estimators": 100,
    "lgbm_learning_rate": 0.05,
    "lgbm_min_child_samples": 20,
    "random_seed": 42,
}


def _signal_candidate(name: str, **overrides: object) -> dict[str, object]:
    return {"name": name, "overrides": {**INCUMBENT_MODEL_OVERRIDES, **overrides}}


OFFICIAL_SIGNAL_CHALLENGERS: tuple[dict[str, object], ...] = (
    _signal_candidate(
        "incumbent_expanding",
        meta_history_mode="expanding", meta_history_quarters=16,
        meta_weight_cap=1.0, meta_equal_shrinkage=0.0,
    ),
    _signal_candidate("meta_equal", meta_type="equal"),
    _signal_candidate("meta_rank_ic_8q", meta_type="rank_ic", meta_ic_lookback_quarters=8),
    _signal_candidate("meta_rank_ic_16q", meta_type="rank_ic", meta_ic_lookback_quarters=16),
    _signal_candidate("stacked_rolling_8q", meta_history_mode="rolling", meta_history_quarters=8),
    _signal_candidate("stacked_rolling_16q", meta_history_mode="rolling", meta_history_quarters=16),
    _signal_candidate(
        "stacked_exponential_hl8",
        meta_history_mode="exponential", meta_history_quarters=16,
        meta_decay_half_life_quarters=8.0,
    ),
    _signal_candidate(
        "stacked_rolling_16q_cap50", meta_history_mode="rolling",
        meta_history_quarters=16, meta_weight_cap=0.50,
    ),
    _signal_candidate(
        "stacked_rolling_16q_shrink50", meta_history_mode="rolling",
        meta_history_quarters=16, meta_equal_shrinkage=0.50,
    ),
    _signal_candidate(
        "stacked_rolling_16q_cap50_shrink50", meta_history_mode="rolling",
        meta_history_quarters=16, meta_weight_cap=0.50, meta_equal_shrinkage=0.50,
    ),
    _signal_candidate(
        "adaptive_lookback_8y", meta_history_mode="rolling", meta_history_quarters=16,
        meta_weight_cap=0.50, meta_equal_shrinkage=0.50, train_lookback_years=8,
    ),
    _signal_candidate(
        "adaptive_linear_recency", meta_history_mode="rolling", meta_history_quarters=16,
        meta_weight_cap=0.50, meta_equal_shrinkage=0.50, recency_weighting="linear",
    ),
)

OFFICIAL_PORTFOLIO_STRUCTURES: tuple[dict[str, object], ...] = (
    {"name": "legacy_monthly_top10", "overrides": {
        "portfolio_policy": "legacy_monthly", "target_size": 10,
        "sizing_mode": "legacy_linear", "active_overlay_mode": "full",
    }},
    {"name": "quarterly_top10_equal", "overrides": {
        "portfolio_policy": "quarterly_top_n", "target_size": 10,
        "sizing_mode": "equal", "active_overlay_mode": "full",
    }},
    {"name": "vintages_4x2_equal", "overrides": {
        "portfolio_policy": "staggered_vintages", "vintage_count": 4,
        "holding_months": 12, "target_size": 8, "sizing_mode": "equal",
        "active_overlay_mode": "full",
    }},
    {"name": "vintages_4x3_equal", "overrides": {
        "portfolio_policy": "staggered_vintages", "vintage_count": 4,
        "holding_months": 12, "target_size": 12, "sizing_mode": "equal",
        "active_overlay_mode": "full",
    }},
    {"name": "vintages_4x4_equal", "overrides": {
        "portfolio_policy": "staggered_vintages", "vintage_count": 4,
        "holding_months": 12, "target_size": 16, "sizing_mode": "equal",
        "active_overlay_mode": "full",
    }},
)

OFFICIAL_EVALUATION_BUDGET: dict[str, int] = {
    "signal_challengers": 12,
    "seed_confirmation": 2,
    "portfolio_translation": 12,
    "profiles": 8,
    "economic_stress": 6,
    "retraining_placebos": 5,
    "score_label_permutation": 1,
    "pit_random_portfolios": 1,
    "bootstrap_and_era_exclusion": 1,
}

OFFICIAL_STUDY_PROTOCOL: dict[str, object] = {
    "version": 2,
    "selection_eras": ((2015, 2018), (2019, 2021), (2022, 2024)),
    "known_stress_years": (2025, 2026),
    "signal_challengers": OFFICIAL_SIGNAL_CHALLENGERS,
    "portfolio_structures": OFFICIAL_PORTFOLIO_STRUCTURES,
    "profiles": ("balanced", "growth", "value", "quality", "momentum", "contrarian",
                 "defensive", "garp"),
    "max_evaluations": 50,
    "max_expensive_fits": 10,
    "estimated_expensive_fits": 10,
    "max_incremental_bytes": 5 * 1024**3,
}


def official_evaluation_budget() -> dict[str, object]:
    """Presupuesto determinista del protocolo oficial; falla si alguien lo expande sin revisarlo."""
    breakdown = dict(OFFICIAL_EVALUATION_BUDGET)
    return validate_official_budget(breakdown)


def validate_official_budget(
    breakdown: dict[str, int], *, estimated_expensive_fits: int | None = None,
    estimated_incremental_bytes: int = 0,
) -> dict[str, object]:
    """Valida una proyección antes de crear runs; se expone para tests de integración."""
    total = sum(breakdown.values())
    maximum = int(OFFICIAL_STUDY_PROTOCOL["max_evaluations"])
    if total > maximum:
        raise ValueError(f"El protocolo oficial supera el máximo de {maximum}: {total}.")
    expensive = int(
        OFFICIAL_STUDY_PROTOCOL["estimated_expensive_fits"]
        if estimated_expensive_fits is None else estimated_expensive_fits
    )
    if expensive > int(OFFICIAL_STUDY_PROTOCOL["max_expensive_fits"]):
        raise ValueError("El protocolo oficial supera el máximo de walk-forwards caros.")
    if estimated_incremental_bytes > int(OFFICIAL_STUDY_PROTOCOL["max_incremental_bytes"]):
        raise ValueError("El protocolo oficial supera el máximo de almacenamiento incremental.")
    if breakdown == OFFICIAL_EVALUATION_BUDGET and total != 48:
        raise ValueError(f"El protocolo oficial debe contener exactamente 48 evaluaciones, no {total}.")
    return {"total": total, "maximum": maximum, "max_evaluations": maximum,
            "breakdown": breakdown, "groups": breakdown,
            "max_expensive_fits": int(OFFICIAL_STUDY_PROTOCOL["max_expensive_fits"]),
            "estimated_expensive_fits": expensive,
            "max_incremental_bytes": int(OFFICIAL_STUDY_PROTOCOL["max_incremental_bytes"])}
