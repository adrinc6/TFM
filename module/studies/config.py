"""Validación del catálogo y cálculo determinista del presupuesto."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from environment import Settings
from module.studies.catalog import AGENT_NAMES, BY_ID, FEATURE_PRESETS, STAGE_ORDER, VARIABLES, default_definition


CONFIRMATORY_EVALUATIONS = 23


class ConfigurationError(ValueError):
    """La definición no pertenece al catálogo o excede el protocolo."""


def normalized_definition(raw: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    unknown = set(raw or {}) - set(BY_ID)
    if unknown:
        raise ConfigurationError(f"Variables desconocidas: {sorted(unknown)}.")
    result = default_definition()
    for variable_id, selection in (raw or {}).items():
        if not isinstance(selection, Mapping):
            raise ConfigurationError(f"{variable_id} debe declarar mode y values.")
        spec = BY_ID[variable_id]
        extra = set(selection) - {"mode", "values"}
        if extra:
            raise ConfigurationError(f"Campos desconocidos en {variable_id}: {sorted(extra)}.")
        mode = selection.get("mode")
        values = selection.get("values")
        if mode not in spec.modes:
            raise ConfigurationError(f"Modo {mode!r} no permitido para {variable_id}.")
        if not isinstance(values, list):
            raise ConfigurationError(f"values de {variable_id} debe ser una lista.")
        if mode == "fixed" and len(values) != 1:
            raise ConfigurationError(f"{variable_id} fijo necesita exactamente un valor.")
        if mode == "optimize" and not 2 <= len(values) <= spec.max_values:
            raise ConfigurationError(
                f"{variable_id} optimizable necesita entre 2 y {spec.max_values} valores."
            )
        canonical_values = [_canonical_catalog_value(value, spec.values) for value in values]
        if any(value is None for value in canonical_values):
            invalid = [
                value for value, canonical in zip(values, canonical_values, strict=True)
                if canonical is None
            ]
            raise ConfigurationError(f"Valores fuera de catálogo en {variable_id}: {invalid}.")
        if len({_stable_value(value) for value in canonical_values}) != len(canonical_values):
            raise ConfigurationError(f"{variable_id} contiene valores repetidos.")
        result[variable_id] = {"mode": mode, "values": canonical_values}
    return result


def _stable_value(value: Any) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _canonical_catalog_value(value: Any, allowed_values: tuple[Any, ...]) -> Any | None:
    """Devuelve el objeto del catálogo equivalente a un valor transportado por JSON.

    JSON no diferencia ``0`` de ``0.0``. La conversión solo se permite si el catálogo espera un
    flotante y el número recibido tiene exactamente ese valor; de ese modo se mantiene la
    distinción int/bool del resto del contrato cerrado.
    """
    for allowed in allowed_values:
        if _stable_value(value) == _stable_value(allowed):
            return allowed
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        for allowed in allowed_values:
            if isinstance(allowed, float) and value == allowed:
                return allowed
    return None


def _possible_values(definition: Mapping[str, dict[str, Any]], variable_id: str) -> set[Any]:
    return set(definition[variable_id]["values"])


def is_active(variable_id: str, definition: Mapping[str, dict[str, Any]]) -> bool:
    spec = BY_ID[variable_id]
    for dependency, allowed in spec.depends_on:
        if not (_possible_values(definition, dependency) & set(allowed)):
            return False
    return True


def validate_definition(raw: Mapping[str, Any] | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    definition = normalized_definition(raw)
    for variable in VARIABLES:
        if definition[variable.id]["mode"] != "optimize":
            continue
        for dependency, _ in variable.depends_on:
            if definition[dependency]["mode"] != "fixed":
                raise ConfigurationError(
                    f"{variable.id} solo puede optimizarse cuando {dependency} está fijo; "
                    "así el presupuesto es determinista."
                )
    budget = evaluation_budget(definition)
    _validate_temporal_compatibility(definition)
    return definition, budget


def _validate_temporal_compatibility(definition: Mapping[str, dict[str, Any]]) -> None:
    """Impide una etiqueta más corta que el intervalo entre snapshots.

    Los valores siguen siendo escogibles desde el catálogo. Solo se rechazan combinaciones que no
    pueden materializar una cantidad entera y positiva de snapshots para el horizonte elegido.
    """
    snapshots = definition["snapshot_step_months"]["values"]
    horizons = definition["target_horizon_months"]["values"]
    invalid = [
        (snapshot, horizon)
        for snapshot in snapshots
        for horizon in horizons
        if horizon < snapshot or horizon % snapshot
    ]
    if invalid:
        pairs = ", ".join(f"{snapshot}m→{horizon}m" for snapshot, horizon in invalid)
        raise ConfigurationError(
            "Cadencia y horizonte incompatibles: " + pairs + ". "
            "El horizonte debe ser un múltiplo entero de la cadencia."
        )


def evaluation_budget(definition: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    optimized = [
        variable for variable in VARIABLES
        if is_active(variable.id, definition) and definition[variable.id]["mode"] == "optimize"
    ]
    breakdown = {
        variable.id: len(definition[variable.id]["values"])
        for variable in optimized
    }
    exploratory = 1 + sum(breakdown.values())
    expensive = 1 + sum(
        len(definition[variable.id]["values"])
        for variable in optimized
        if variable.invalidates in {"dataset", "features", "fit"}
    )
    meta = sum(
        len(definition[variable.id]["values"])
        for variable in optimized
        if variable.invalidates == "meta"
    )
    backtests = sum(
        len(definition[variable.id]["values"])
        for variable in optimized
        if variable.invalidates == "backtest"
    )
    estimated_bytes = 200 * 1024**2 + expensive * 400 * 1024**2 + meta * 20 * 1024**2 + backtests * 10 * 1024**2
    estimated_minutes = expensive * 35 + meta * 3 + backtests * 2
    return {
        "exploratory_evaluations": exploratory,
        "expensive_fits": expensive,
        "meta_recombinations": meta,
        "backtests": backtests,
        "reusable_evaluations": max(0, exploratory - expensive),
        "confirmatory_evaluations": CONFIRMATORY_EVALUATIONS,
        "total_cycle_evaluations": exploratory + CONFIRMATORY_EVALUATIONS,
        "estimated_incremental_bytes": estimated_bytes,
        "estimated_minutes": estimated_minutes,
        "breakdown": breakdown,
    }


def initial_values(definition: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for variable in VARIABLES:
        selected = definition[variable.id]["values"]
        values[variable.id] = variable.recommended if variable.recommended in selected else selected[0]
    return values


def settings_from_values(
    values: Mapping[str, Any],
    *,
    workspace_dir: Path | None = None,
    random_seed: int = 42,
    profile: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    meta_method = str(values["meta_method"])
    fields: dict[str, Any] = {
        "snapshot_step_months": int(values["snapshot_step_months"]),
        "execution_lag_days": int(values["execution_lag_days"]),
        "train_lookback_years": int(values["train_lookback_years"]),
        "target_horizon_months": int(values["target_horizon_months"]),
        "recency_weighting": str(values["recency_weighting"]),
        "objective": str(values["objective"]),
        "enabled_feature_blocks": FEATURE_PRESETS[str(values["feature_preset"])],
        "fundamental_momentum": bool(values["fundamental_momentum"]),
        "market_regime_feature": bool(values["market_regime_feature"]),
        "neutralize_by_sector": bool(values["neutralize_by_sector"]),
        "metric_winsorization_percentile": float(values["winsorization"]),
        "feature_selection_max_features_per_agent": int(values["max_features_per_agent"]),
        "enabled_model_families": (str(values["model_family"]),),
        "enabled_agents": AGENT_NAMES,
        "lgbm_max_depth": int(values["lgbm_max_depth"]),
        "lgbm_n_estimators": int(values["lgbm_n_estimators"]),
        "lgbm_learning_rate": float(values["lgbm_learning_rate"]),
        "lgbm_min_child_samples": int(values["lgbm_min_child_samples"]),
        "feature_weighting_mode": str(values["feature_weighting_mode"]),
        "meta_type": (
            meta_method if meta_method in {"equal", "rank_ic"} else "stacked_oos"
        ),
        "meta_history_mode": (
            "rolling" if meta_method == "stacked_rolling"
            else "exponential" if meta_method == "stacked_exponential"
            else "expanding"
        ),
        "meta_ic_lookback_quarters": int(values["meta_history_quarters"]),
        "meta_history_quarters": int(values["meta_history_quarters"]),
        "meta_weight_cap": float(values["meta_weight_cap"]),
        "meta_weight_min": float(values["meta_weight_min"]),
        "meta_equal_shrinkage": float(values["meta_equal_shrinkage"]),
        "meta_decay_half_life_quarters": float(values["meta_half_life_quarters"]),
        "target_size": int(values["target_size"]),
        "min_hold_percentile": float(values["min_hold_percentile"]),
        "rotation_edge_percentiles": float(values["rotation_edge_percentiles"]),
        "rebalance_drift_tolerance": float(values["rebalance_drift_tolerance"]),
        "price_only_strictness_multiplier": float(values["price_only_strictness_multiplier"]),
        "sizing_mode": str(values["sizing_mode"]),
        "commission_bps": float(values["commission_bps"]),
        "slippage_bps": float(values["slippage_bps"]),
        "random_seed": random_seed,
        "profile": profile or "balanced",
        "workspace_dir": workspace_dir,
    }
    fields.update(dict(overrides or {}))
    return Settings(**fields)


def settings_payload(values: Mapping[str, Any]) -> dict[str, Any]:
    payload = asdict(settings_from_values(values))
    payload["workspace_dir"] = None
    return payload


def ordered_optimized_variables(
    definition: Mapping[str, dict[str, Any]],
) -> list[str]:
    return [
        variable.id
        for stage in STAGE_ORDER
        for variable in sorted(
            (item for item in VARIABLES if item.stage == stage),
            key=lambda item: item.order,
        )
        if is_active(variable.id, definition)
        and definition[variable.id]["mode"] == "optimize"
    ]
