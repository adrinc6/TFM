from __future__ import annotations

import pytest

from environment import Settings
from module.studies.catalog import (
    AGENT_NAMES,
    CATALOG_VERSION,
    FEATURE_PRESETS,
    public_catalog,
    recommended_definition,
)
from module.studies.config import (
    ConfigurationError,
    settings_from_values,
    validate_definition,
)


def test_catalog_is_closed_and_has_five_agents() -> None:
    catalog = public_catalog()
    assert catalog["version"] == CATALOG_VERSION
    assert len(catalog["hash"]) == 64
    assert AGENT_NAMES == ("quality", "value", "growth", "momentum", "risk")


def test_every_preset_feeds_all_five_agents() -> None:
    """Ningún preset puede dejar a un agente sin features.

    Un preset que vacía a un agente lo elimina de hecho del sistema, y entonces la comparación deja
    de medir «qué información necesita cada agente» para medir «qué pasa si amputo parte de la
    arquitectura». Eran los casos de `fundamental` (sin momentum ni risk) y `technical` (sin quality,
    value ni growth), ya retirados del catálogo.
    """
    from module.modeling.catalog import catalog_by_agent

    for preset, blocks in FEATURE_PRESETS.items():
        by_agent = catalog_by_agent(blocks, AGENT_NAMES)
        starved = sorted(agent for agent, columns in by_agent.items() if not columns)
        assert not starved, f"El preset {preset!r} deja sin features a {starved}"


def test_portfolio_thresholds_are_economic_not_percentiles() -> None:
    """Los umbrales de cartera se expresan en puntos básicos de alfa esperado.

    Un percentil no dice cuánto se espera ganar, así que no puede compararse contra el coste de
    operar. Con umbrales en pb, una rotación solo se autoriza si cubre su propio coste.

    `coverage_percentile_floor` no es una excepción a esa doctrina: no se compara contra ningún
    coste. Es una regla de mandato —la generalización de `missing_current_score`, con el percentil
    ausente sustituido por un percentil demasiado bajo— que decide pertenencia al universo
    invertible, no si una operación concreta es rentable.
    """
    identifiers = {variable["id"] for variable in public_catalog()["variables"]}
    assert {"exit_expected_alpha_bps", "rotation_edge_bps"} <= identifiers
    assert not {"min_hold_percentile", "rotation_edge_percentiles"} & identifiers
    catalog = {variable["id"]: variable for variable in public_catalog()["variables"]}
    assert catalog["coverage_percentile_floor"]["predictive"] is False
    assert catalog["coverage_percentile_floor"]["stage"] == "portfolio"


def test_cash_is_a_portfolio_diagnostic_and_never_touches_selection() -> None:
    """El tope de efectivo no puede alterar la elección del modelo."""
    catalog = {variable["id"]: variable for variable in public_catalog()["variables"]}
    assert catalog["max_cash_weight"]["predictive"] is False
    assert catalog["max_cash_weight"]["stage"] == "portfolio"
    # El tope del 25 % implica un suelo de diversificación de al menos el 75 % de las plazas.
    assert set(catalog["max_cash_weight"]["values"]) == {0.0, 0.10, 0.25}


def test_cash_exposure_is_governed_by_a_single_variable() -> None:
    """`max_cash_weight` es el único grado de libertad del efectivo; 0 significa siempre invertido.

    Existió además una `cash_policy` con los valores `fully_invested`/`opportunity_cash`, pero era
    redundante: el catálogo ya forzaba el tope a 0 bajo `fully_invested`, y el suelo de
    diversificación se deriva del tope. Dos variables para una decisión invitan a combinaciones
    inconsistentes, así que se dejó una.
    """
    identifiers = {variable["id"] for variable in public_catalog()["variables"]}
    assert "cash_policy" not in identifiers
    assert not hasattr(Settings(), "cash_policy")
    values = {key: selection["values"][0] for key, selection in recommended_definition().items()}
    values["max_cash_weight"] = 0.0
    assert settings_from_values(values).max_cash_weight == 0.0
    values["max_cash_weight"] = 0.25
    assert settings_from_values(values).max_cash_weight == 0.25


def test_recommendation_is_valid_without_blocking_limits() -> None:
    definition, budget = validate_definition(recommended_definition())
    assert definition == recommended_definition()
    assert 6 <= budget["predictive_evaluations"] <= 14
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
    # 1 baseline + (2-1) + (3-1): por variable, el candidato igual al baseline
    # vigente se reutiliza sin ejecutar (ver runner.py), así que no cuenta como run.
    assert budget["predictive_evaluations"] == 1 + 1 + 2


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


def test_portfolio_variables_are_fixed_in_the_model_study() -> None:
    """El Model Study no barre reglas de cartera: solo admite elegir un valor.

    Optimiza Rank-IC, que mide la ordenación transversal de scores y es anterior a cualquier
    decisión de cartera: `target_size` o `commission_bps` no pueden cambiarlo. Barrerlas gastaba un
    run por variable para reproducir el mismo Rank-IC, y quien las optimiza es el Portfolio Study.
    """
    from module.studies.catalog import VARIABLES

    definition = recommended_definition()
    portfolio = [spec for spec in VARIABLES if not spec.predictive]
    assert portfolio, "el catálogo debe declarar variables de cartera"
    for spec in portfolio:
        assert spec.modes == ("fixed",), spec.id
        assert definition[spec.id]["mode"] == "fixed", spec.id
        assert len(definition[spec.id]["values"]) == 1, spec.id


def test_the_model_study_rejects_sweeping_a_portfolio_variable() -> None:
    """Pedir un barrido de cartera falla en validación, no en mitad del estudio."""
    with pytest.raises(ConfigurationError):
        validate_definition({"target_size": {"mode": "diagnostic", "values": [12, 5]}})
    with pytest.raises(ConfigurationError):
        validate_definition({"target_size": {"mode": "fixed", "values": [12, 5]}})


def test_a_model_study_budget_has_no_portfolio_runs() -> None:
    """El presupuesto ya no reserva runs de cartera: eran 12 que nunca movían al ganador."""
    from module.studies.config import evaluation_budget

    budget = evaluation_budget(recommended_definition())
    assert budget["portfolio_diagnostics"] == 0
