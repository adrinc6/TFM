from __future__ import annotations

import pytest

from module.studies.catalog import AGENT_NAMES, public_catalog, recommended_definition
from module.studies.config import ConfigurationError, settings_from_values, validate_definition


def test_catalog_is_closed_and_has_five_agents() -> None:
    catalog = public_catalog()
    assert catalog["version"] == 2
    assert len(catalog["hash"]) == 64
    assert AGENT_NAMES == ("quality", "value", "growth", "momentum", "risk")


def test_recommendation_is_valid_without_blocking_limits() -> None:
    definition, budget = validate_definition(recommended_definition())
    assert definition == recommended_definition()
    assert 12 <= budget["predictive_evaluations"] <= 20
    assert budget["blocking_limits"] is False


def test_unknown_and_out_of_catalog_values_are_rejected() -> None:
    with pytest.raises(ConfigurationError):
        validate_definition({"free_parameter": {"mode": "fixed", "values": [1]}})
    with pytest.raises(ConfigurationError):
        validate_definition({"execution_lag_days": {"mode": "fixed", "values": [15]}})


def test_json_zero_is_canonicalized_for_winsorization() -> None:
    definition, _ = validate_definition({"winsorization": {"mode": "fixed", "values": [0]}})
    assert definition["winsorization"]["values"] == [0.0]


def test_budget_is_sequential_not_cartesian() -> None:
    definition = {
        key: {"mode": "fixed", "values": [selection["values"][0]]}
        for key, selection in recommended_definition().items()
    }
    definition["execution_lag_days"] = {"mode": "optimize", "values": [45, 60]}
    definition["meta_method"] = {
        "mode": "optimize",
        "values": ["equal", "stacked_rolling_free", "stacked_rolling_bounded"],
    }
    _, budget = validate_definition(definition)
    assert budget["predictive_evaluations"] == 1 + 2 + 3


def test_meta_candidates_map_to_exact_weight_contracts() -> None:
    values = {key: selection["values"][0] for key, selection in recommended_definition().items()}
    values["meta_method"] = "equal"
    assert settings_from_values(values).meta_type == "equal"
    values["meta_method"] = "stacked_rolling_free"
    free = settings_from_values(values)
    assert (free.meta_weight_min, free.meta_weight_cap) == (0.0, 1.0)
    values["meta_method"] = "stacked_rolling_bounded"
    bounded = settings_from_values(values)
    assert (bounded.meta_weight_min, bounded.meta_weight_cap) == (0.10, 0.50)
