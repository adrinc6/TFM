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


# Coste aproximado en disco de la evidencia completa de un run: scores, diagnósticos de Rank-IC,
# atribución, pesos del meta-agente, equity, órdenes y posiciones (ver runner._retain_evidence).
RETAINED_BYTES_PER_RUN = 60 * 1024**2

# Coste medido de una combinación de cartera: reutiliza los scores congelados del ganador y solo
# rehace el backtest, así que cuesta segundos y no los minutos de un run predictivo con ajuste.
PORTFOLIO_SECONDS_PER_COMBINATION = 6.0


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
    # Cada variable predictiva reutiliza sin ejecutar el candidato igual al baseline
    # vigente (ver runner.py: `configuration == incumbent["configuration"]`), así que
    # solo len(values) - 1 candidatos por variable generan un run real.
    breakdown = {spec.id: len(definition[spec.id]["values"]) - 1 for spec in predictive}
    predictive_evaluations = 1 + sum(breakdown.values())
    expensive = 1 + sum(
        len(definition[spec.id]["values"]) - 1
        for spec in predictive if spec.invalidates in {"dataset", "features", "fit"}
    )
    meta = sum(
        len(definition[spec.id]["values"]) - 1
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


def validate_portfolio_definition(
    raw: Mapping[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Valida la rejilla de un Portfolio Study y calcula su presupuesto.

    Dos reglas, y cada una responde a un motivo distinto:

    - De las **seis variables optimizables** se puede marcar cualquier subconjunto de valores (uno o
      todos): son las que el cartesiano explora, y marcar un solo valor equivale a fijarla.
    - De las **demás variables de cartera** hay que marcar **exactamente uno**. No se optimizan, así
      que no admiten alternativas: `commission_bps` y `slippage_bps` son supuestos de coste
      —optimizarlos sería elegir el mundo en el que la estrategia luce mejor— y el resto gobierna
      cuándo se opera bajo información incompleta, que se estresa aparte y no se elige por IR.
    """
    from module.studies.portfolio_study import PORTFOLIO_STUDY_VARIABLES

    raw = raw or {}
    unknown = set(raw) - {spec.id for spec in VARIABLES if not spec.predictive}
    if unknown:
        raise ConfigurationError(
            f"El Portfolio Study solo admite variables de cartera; sobran: {sorted(unknown)}."
        )
    definition: dict[str, dict[str, Any]] = {}
    for spec in VARIABLES:
        if spec.predictive:
            continue
        selection = raw.get(spec.id)
        values = list(selection["values"]) if isinstance(selection, Mapping) and "values" in selection else [spec.recommended]
        canonical = [_canonical(value, spec.values) for value in values]
        if any(value is None for value in canonical):
            invalid = [value for value, ok in zip(values, canonical) if ok is None]
            raise ConfigurationError(f"Valores fuera de catálogo en {spec.id}: {invalid}.")
        if len({_stable(value) for value in canonical}) != len(canonical):
            raise ConfigurationError(f"{spec.id} contiene valores repetidos.")
        if not canonical:
            raise ConfigurationError(f"{spec.id} necesita al menos un valor.")
        if spec.id not in PORTFOLIO_STUDY_VARIABLES and len(canonical) != 1:
            raise ConfigurationError(
                f"{spec.id} no se optimiza: debe fijarse en exactamente un valor, no {len(canonical)}."
            )
        definition[spec.id] = {
            "mode": "grid" if spec.id in PORTFOLIO_STUDY_VARIABLES else "fixed",
            "values": canonical,
            "baseline": canonical[0],
        }
    return definition, portfolio_budget(definition)


def portfolio_budget(definition: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """Presupuesto del cartesiano de cartera.

    Cada combinación reutiliza los scores ya congelados del ganador y solo rehace el backtest, del
    orden de segundos, frente a los minutos que cuesta un run predictivo con su ajuste.
    """
    from module.studies.portfolio_study import PORTFOLIO_STUDY_VARIABLES

    breakdown = {
        variable_id: len(definition[variable_id]["values"])
        for variable_id in PORTFOLIO_STUDY_VARIABLES
        if variable_id in definition
    }
    combinations = 1
    for count in breakdown.values():
        combinations *= count
    return {
        "combinations": combinations,
        "grid_variables": len([count for count in breakdown.values() if count > 1]),
        "fixed_variables": len([count for count in breakdown.values() if count == 1]),
        "breakdown": breakdown,
        "seconds_per_combination": PORTFOLIO_SECONDS_PER_COMBINATION,
        "estimated_minutes": round(combinations * PORTFOLIO_SECONDS_PER_COMBINATION / 60),
        # Solo sobrevive la evidencia del mejor vigente, así que el disco no crece con la rejilla.
        "estimated_incremental_bytes": RETAINED_BYTES_PER_RUN,
        "blocking_limits": False,
    }


def post_winner_budget(budget: Mapping[str, Any], post_winner_diagnostics: bool) -> dict[str, Any]:
    """Descuenta del presupuesto los diagnósticos posteriores al ganador cuando se desactivan.

    Un Study cuyo único fin es elegir configuración termina al congelar el ganador: carteras
    diagnósticas, perfiles, robustez y atribución no llegan a ejecutarse. El presupuesto debe
    declararlo antes de lanzar, porque `total_runs` es también el denominador del progreso
    (ver runner.py) y anunciar runs que nunca ocurren dejaría la barra corta para siempre.
    """
    payload = dict(budget)
    payload["post_winner_diagnostics"] = bool(post_winner_diagnostics)
    if post_winner_diagnostics:
        return payload
    skipped = (
        int(payload.get("portfolio_diagnostics", 0))
        + int(payload.get("profiles", 0))
        + int(payload.get("robustness_groups", 0))
    )
    payload["total_runs"] = max(1, int(payload.get("total_runs", 0)) - skipped)
    payload["estimated_minutes"] = max(
        0, int(payload.get("estimated_minutes", 0)) - int(payload.get("portfolio_diagnostics", 0)) * 2 - 30,
    )
    payload["portfolio_diagnostics"] = 0
    payload["profiles"] = 0
    payload["robustness_groups"] = 0
    return payload


def retention_budget(budget: Mapping[str, Any], retain_all_runs: bool) -> dict[str, Any]:
    """Ajusta el presupuesto cuando se conserva la evidencia completa de todos los runs.

    La regla 5 del proyecto (los descartados guardan solo resúmenes) existe por coste de disco, no
    por ciencia: la selección sigue dependiendo únicamente del Rank-IC. Al desactivarla, cada
    evaluación materializa cartera, órdenes, posiciones y pesos del meta-agente, así que el
    presupuesto debe declarar ese coste antes de lanzar y no descubrirlo a mitad de ejecución.
    """
    payload = dict(budget)
    payload["retain_all_runs"] = bool(retain_all_runs)
    if not retain_all_runs:
        return payload
    # Los runs que ya retenían evidencia (baseline y ganador) no se cuentan dos veces.
    additional = max(0, int(payload.get("total_runs", 0)) - 2)
    payload["retained_run_evidence"] = additional
    payload["estimated_incremental_bytes"] = (
        int(payload.get("estimated_incremental_bytes", 0)) + additional * RETAINED_BYTES_PER_RUN
    )
    return payload


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
    """Variables de cartera a barrer como diagnóstico tras congelar el ganador.

    Devuelve la lista vacía mientras el catálogo fije las variables de cartera (`modes` == `fixed`),
    que es el caso desde que su optimización pasó al Portfolio Study. Se conserva porque la fase
    sigue existiendo en el runner y porque el modo `diagnostic` puede reactivarse sin tocar el
    recorrido.
    """
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
        "meta_history_quarters": int(values["meta_history_quarters"]),
        "meta_recency_weighting": str(values["meta_recency_weighting"]),
        "meta_weight_min": 0.10 if method == "stacked_rolling_bounded" else 0.0,
        "meta_weight_cap": 0.50 if method == "stacked_rolling_bounded" else 1.0,
        "target_size": int(values["target_size"]),
        "exit_expected_alpha_bps": float(values["exit_expected_alpha_bps"]),
        "rotation_edge_bps": float(values["rotation_edge_bps"]),
        # El tope de efectivo gobierna por sí solo la exposición: 0 significa siempre invertido, y
        # el suelo de diversificación se deriva de él. No hay una política aparte que diga lo mismo.
        "max_cash_weight": float(values["max_cash_weight"]),
        "rebalance_drift_tolerance": float(values["rebalance_drift_tolerance"]),
        "minimum_holding_period": str(values["minimum_holding_period"]),
        "coverage_percentile_floor": float(values["coverage_percentile_floor"]),
        "price_only_sell_only": bool(values["price_only_sell_only"]),
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
