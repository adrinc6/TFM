"""Validación y presupuesto del Model Study."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from environment import Settings
from module.studies.catalog import (
    AGENT_NAMES, BY_ID, FEATURE_PRESETS, PREDICTIVE_STAGES,
    VARIABLES, default_definition,
)


class ConfigurationError(ValueError):
    pass


def _stable(value: Any) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _canonical(value: Any, allowed: tuple[Any, ...]) -> Any | None:
    for option in allowed:
        if _stable(value) == _stable(option):
            return option
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        matches = [option for option in allowed if isinstance(option, float) and value == option]
        if len(matches) == 1:
            return matches[0]
    return None


def normalized_definition(raw: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    unknown = set(raw or {}) - set(BY_ID)
    if unknown:
        raise ConfigurationError(f"Variables desconocidas: {sorted(unknown)}.")
    result = default_definition()
    for identifier, selection in (raw or {}).items():
        if not isinstance(selection, Mapping):
            raise ConfigurationError(f"{identifier} debe declarar mode y values.")
        extra = set(selection) - {"mode", "values", "baseline"}
        if extra:
            raise ConfigurationError(f"Campos desconocidos en {identifier}: {sorted(extra)}.")
        spec = BY_ID[identifier]
        mode, values = selection.get("mode"), selection.get("values")
        if mode not in spec.modes:
            raise ConfigurationError(f"Modo {mode!r} no permitido para {identifier}.")
        if not isinstance(values, list):
            raise ConfigurationError(f"values de {identifier} debe ser una lista.")
        minimum = 1 if mode == "fixed" else 2
        maximum = 1 if mode == "fixed" else len(spec.values)
        if not minimum <= len(values) <= maximum:
            raise ConfigurationError(
                f"{identifier} en modo {mode} necesita entre {minimum} y {maximum} valores."
            )
        canonical = [_canonical(value, spec.values) for value in values]
        invalid = [value for value, converted in zip(values, canonical, strict=True) if converted is None]
        if invalid:
            raise ConfigurationError(f"Valores fuera de catálogo en {identifier}: {invalid}.")
        if len({_stable(value) for value in canonical}) != len(canonical):
            raise ConfigurationError(f"{identifier} contiene valores repetidos.")
        baseline = _canonical(selection.get("baseline", canonical[0]), spec.values)
        if baseline is None or not any(_stable(baseline) == _stable(value) for value in canonical):
            raise ConfigurationError(f"baseline de {identifier} debe ser uno de los values seleccionados.")
        result[identifier] = {"mode": mode, "values": canonical, "baseline": baseline}
    return result


def is_active(identifier: str, definition: Mapping[str, dict[str, Any]]) -> bool:
    for controller, allowed in BY_ID[identifier].depends_on:
        if not set(definition[controller]["values"]) & set(allowed):
            return False
    return True


def validate_definition(raw: Mapping[str, Any] | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    definition = normalized_definition(raw)
    for spec in VARIABLES:
        if not is_active(spec.id, definition) or definition[spec.id]["mode"] == "fixed":
            continue
        for controller, _ in spec.depends_on:
            if definition[controller]["mode"] != "fixed":
                raise ConfigurationError(
                    f"{spec.id} solo puede comparar valores si {controller} permanece fijo."
                )
    _validate_temporal(definition)
    return definition, evaluation_budget(definition)


def _validate_temporal(definition: Mapping[str, dict[str, Any]]) -> None:
    invalid = [
        (step, horizon)
        for step in definition["snapshot_step_months"]["values"]
        for horizon in definition["target_horizon_months"]["values"]
        if horizon < step or horizon % step
    ]
    if invalid:
        text = ", ".join(f"{step}m→{horizon}m" for step, horizon in invalid)
        raise ConfigurationError(f"Cadencia y horizonte incompatibles: {text}.")


def evaluation_budget(definition: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    predictive = [
        spec for spec in VARIABLES
        if spec.predictive and is_active(spec.id, definition)
        and definition[spec.id]["mode"] == "optimize"
    ]
    diagnostics = [
        spec for spec in VARIABLES
        if not spec.predictive and definition[spec.id]["mode"] == "diagnostic"
    ]
    breakdown = {spec.id: len(definition[spec.id]["values"]) for spec in predictive}
    predictive_evaluations = 1 + sum(breakdown.values())
    expensive = 1 + sum(
        len(definition[spec.id]["values"])
        for spec in predictive if spec.invalidates in {"dataset", "features", "fit"}
    )
    meta = sum(
        len(definition[spec.id]["values"])
        for spec in predictive if spec.invalidates == "meta"
    )
    portfolio = sum(max(0, len(definition[spec.id]["values"]) - 1) for spec in diagnostics)
    robustness = 2 + 5 + 3
    estimated_minutes = expensive * 35 + meta * 3 + portfolio * 2 + 30
    estimated_bytes = 250 * 1024**2 + expensive * 400 * 1024**2 + meta * 20 * 1024**2
    return {
        "predictive_evaluations": predictive_evaluations,
        "expensive_fits": expensive,
        "meta_recombinations": meta,
        "portfolio_diagnostics": portfolio,
        "profiles": 8,
        "robustness_groups": robustness,
        "total_runs": predictive_evaluations + portfolio + 8 + robustness,
        "estimated_minutes": estimated_minutes,
        "estimated_incremental_bytes": estimated_bytes,
        "breakdown": breakdown,
        "blocking_limits": False,
    }


def initial_values(definition: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    return {spec.id: definition[spec.id]["baseline"] for spec in VARIABLES}


def ordered_predictive_variables(definition: Mapping[str, dict[str, Any]]) -> list[str]:
    return [
        spec.id
        for stage in PREDICTIVE_STAGES
        for spec in sorted(
            (candidate for candidate in VARIABLES if candidate.stage == stage),
            key=lambda candidate: candidate.order,
        )
        if is_active(spec.id, definition) and definition[spec.id]["mode"] == "optimize"
    ]


def diagnostic_portfolio_variables(definition: Mapping[str, dict[str, Any]]) -> list[str]:
    return [
        spec.id for spec in VARIABLES
        if not spec.predictive and definition[spec.id]["mode"] == "diagnostic"
    ]


def settings_from_values(
    values: Mapping[str, Any],
    *,
    workspace_dir: Path | None = None,
    random_seed: int = 42,
    profile: str = "balanced",
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    method = str(values["meta_method"])
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
        "feature_weighting_mode": str(values["feature_weighting_mode"]),
        "enabled_model_families": (str(values["model_family"]),),
        "enabled_agents": AGENT_NAMES,
        "lgbm_max_depth": int(values["lgbm_max_depth"]),
        "lgbm_n_estimators": int(values["lgbm_n_estimators"]),
        "lgbm_learning_rate": float(values["lgbm_learning_rate"]),
        "lgbm_min_child_samples": int(values["lgbm_min_child_samples"]),
        "meta_type": "equal" if method == "equal" else "stacked_oos",
        "meta_ic_lookback_quarters": int(values["meta_history_quarters"]),
        "meta_history_quarters": int(values["meta_history_quarters"]),
        "meta_weight_min": 0.10 if method == "stacked_rolling_bounded" else 0.0,
        "meta_weight_cap": 0.50 if method == "stacked_rolling_bounded" else 1.0,
        "meta_equal_shrinkage": 0.0,
        "target_size": int(values["target_size"]),
        "min_hold_percentile": float(values["min_hold_percentile"]),
        "rotation_edge_percentiles": float(values["rotation_edge_percentiles"]),
        "rebalance_drift_tolerance": float(values["rebalance_drift_tolerance"]),
        "price_only_strictness_multiplier": float(values["price_only_strictness_multiplier"]),
        "sizing_mode": str(values["sizing_mode"]),
        "commission_bps": float(values["commission_bps"]),
        "slippage_bps": float(values["slippage_bps"]),
        "random_seed": int(random_seed),
        "profile": profile,
        "workspace_dir": workspace_dir,
    }
    fields.update(dict(overrides or {}))
    return Settings(**fields)
