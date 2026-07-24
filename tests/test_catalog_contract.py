from __future__ import annotations

import pytest

from module.studies.catalog import (
    BY_ID,
    CATALOG_VERSION,
    default_definition,
    public_catalog,
    recommended_exploratory_definition,
)
from module.studies.config import (
    ConfigurationError,
    evaluation_budget,
    settings_from_values,
    validate_definition,
    initial_values,
)


def test_catalog_is_versioned_and_closed() -> None:
    catalog = public_catalog()
    assert catalog["version"] == CATALOG_VERSION == 1
    assert len(catalog["variables"]) == len(BY_ID)
    assert all(item["values"] for item in catalog["variables"])
    assert [item["id"] for item in catalog["stages"]] == catalog["stage_order"]
    assert all(
        len(item["value_options"]) == len(item["values"])
        and all(option["label"] and option["description"] for option in item["value_options"])
        for item in catalog["variables"]
    )


def test_snapshot_cadence_maps_to_runtime_settings() -> None:
    values = initial_values(default_definition())
    values["snapshot_step_months"] = 3
    settings = settings_from_values(values)
    assert settings.snapshot_step_months == 3


def test_snapshot_and_horizon_must_form_a_whole_number_of_snapshots() -> None:
    with pytest.raises(ConfigurationError, match="Cadencia y horizonte incompatibles"):
        validate_definition({
            "snapshot_step_months": {"mode": "fixed", "values": [6]},
            "target_horizon_months": {"mode": "fixed", "values": [3]},
        })


def test_recommended_exploratory_definition_is_valid_and_within_exact_budget() -> None:
    catalog = public_catalog()
    definition, budget = validate_definition(catalog["recommended_exploratory_definition"])
    assert definition == recommended_exploratory_definition()
    assert budget["exploratory_evaluations"] > 20
    assert budget["expensive_fits"] >= 10
    assert budget["total_cycle_evaluations"] == budget["exploratory_evaluations"] + 23


def test_unknown_variable_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="desconocidas"):
        validate_definition({"free_parameter": {"mode": "fixed", "values": [1]}})


def test_value_outside_catalog_is_rejected_even_if_python_compares_equal() -> None:
    with pytest.raises(ConfigurationError, match="fuera de catálogo"):
        validate_definition({"fundamental_momentum": {"mode": "fixed", "values": [0]}})


def test_json_number_for_float_catalog_value_is_canonicalized() -> None:
    definition, _ = validate_definition({
        "winsorization": {"mode": "fixed", "values": [0]},
        "commission_bps": {"mode": "fixed", "values": [5]},
        "slippage_bps": {"mode": "fixed", "values": [10]},
    })
    assert definition["winsorization"]["values"] == [0.0]
    assert definition["commission_bps"]["values"] == [5.0]
    assert definition["slippage_bps"]["values"] == [10.0]


def test_optimized_variable_requires_two_values() -> None:
    with pytest.raises(ConfigurationError, match="entre 2"):
        validate_definition({
            "target_horizon_months": {"mode": "optimize", "values": [12]},
        })


def test_budget_formula_is_baseline_plus_selected_values() -> None:
    definition = default_definition()
    definition["target_horizon_months"] = {"mode": "optimize", "values": [3, 6, 12]}
    definition["meta_method"] = {"mode": "optimize", "values": ["equal", "rank_ic"]}
    normalized, budget = validate_definition(definition)
    assert budget["exploratory_evaluations"] == 1 + 3 + 2
    assert budget["total_cycle_evaluations"] == 29
    assert evaluation_budget(normalized) == budget


def test_hidden_dependency_does_not_consume_budget() -> None:
    definition = default_definition()
    definition["model_family"] = {"mode": "fixed", "values": ["elastic_net"]}
    definition["lgbm_max_depth"] = {"mode": "optimize", "values": [3, 4]}
    _, budget = validate_definition(definition)
    assert "lgbm_max_depth" not in budget["breakdown"]


def test_dependent_axis_requires_fixed_controller() -> None:
    definition = default_definition()
    definition["model_family"] = {
        "mode": "optimize", "values": ["lightgbm", "elastic_net"],
    }
    definition["lgbm_max_depth"] = {"mode": "optimize", "values": [3, 4]}
    with pytest.raises(ConfigurationError, match="presupuesto es determinista"):
        validate_definition(definition)


def test_large_catalog_definition_reports_budget_without_a_global_fit_limit() -> None:
    definition = default_definition()
    for variable_id in (
        "target_horizon_months", "train_lookback_years", "execution_lag_days",
        "recency_weighting",
    ):
        values = list(BY_ID[variable_id].values)
        definition[variable_id] = {"mode": "optimize", "values": values}
    _, budget = validate_definition(definition)
    assert budget["expensive_fits"] > 10
    assert budget["exploratory_evaluations"] > 1


def test_catalog_values_map_to_runtime_settings() -> None:
    values = initial_values(default_definition())
    values.update({
        "feature_preset": "core",
        "meta_method": "stacked_exponential",
        "meta_weight_min": 0.10,
        "meta_weight_cap": 0.50,
        "sizing_mode": "score_linear",
    })
    settings = settings_from_values(values)
    assert settings.meta_type == "stacked_oos"
    assert settings.meta_history_mode == "exponential"
    assert settings.meta_weight_min == 0.10
    assert settings.meta_weight_cap == 0.50
    assert settings.sizing_mode == "score_linear"
    assert settings.profile == "balanced"
    assert settings.enabled_agents == ("quality", "value", "growth", "momentum", "risk")
