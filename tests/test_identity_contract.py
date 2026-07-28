"""Contrato de identidad: dos evaluaciones distintas nunca comparten clave.

Una colisión de identidad no rompe nada visible. Devuelve un resultado plausible calculado sobre
otros datos, y esa es exactamente la clase de fallo que sobrevive a una revisión y llega al tribunal.
"""

from __future__ import annotations

from dataclasses import fields

from environment import Settings
from module.storage.cache import STAGE_FIELDS
from module.storage.datasets import CORE_FIELDS, FEATURE_FIELDS
from module.studies.catalog import VARIABLES, recommended_definition
from module.studies.config import settings_from_values
from module.studies.runner import evaluation_key


def _values() -> dict:
    return {key: selection["values"][0] for key, selection in recommended_definition().items()}


def test_evaluation_key_separates_identical_configurations_on_different_datasets() -> None:
    values = _values()
    first = evaluation_key(values, dataset_hash="a" * 64, random_seed=42, profile="balanced")
    second = evaluation_key(values, dataset_hash="b" * 64, random_seed=42, profile="balanced")
    assert first != second


def test_evaluation_key_separates_seed_profile_and_target_identity() -> None:
    values = _values()
    base = dict(values=values, dataset_hash="a" * 64, random_seed=42, profile="balanced")
    assert evaluation_key(**base) != evaluation_key(**{**base, "random_seed": 7})
    assert evaluation_key(**base) != evaluation_key(**{**base, "profile": "value"})
    assert evaluation_key(**base) != evaluation_key(**base, target_identity="placebo-101")


def test_every_scientific_setting_belongs_to_exactly_one_fingerprint() -> None:
    """Un campo científico fuera de toda huella es un cambio que no invalida ninguna caché.

    Las listas se mantienen a mano en dos módulos distintos; sin este contrato, añadir una variable
    al catálogo y olvidarla en la huella hace que la caché devuelva el resultado de la variable
    anterior sin ningún aviso.
    """
    covered = set(CORE_FIELDS) | set(FEATURE_FIELDS)
    for names in STAGE_FIELDS.values():
        covered |= set(names)
    # Campos que no describen ciencia: rutas, paralelismo y contexto de ejecución.
    operational = {
        "workspace_dir", "lgbm_n_jobs", "profile", "tickers", "dev_mode",
        "raw_output_dir", "processed_output_dir",
    }
    # El backtest se recalcula siempre a partir de los scores, así que sus parámetros no participan
    # en ninguna caché de fit; su identidad la fija `evaluation_key` a través de `values`.
    portfolio = {variable.id for variable in VARIABLES if not variable.predictive}
    portfolio |= {
        "target_size", "exit_expected_alpha_bps", "rotation_edge_bps", "cash_policy",
        "max_cash_weight", "rebalance_drift_tolerance", "price_only_strictness_multiplier",
        "sizing_mode", "commission_bps", "slippage_bps", "max_monthly_position_return",
        "meta_weight_min", "meta_type", "meta_history_quarters", "meta_weight_cap",
        "min_rank_ic_cross_section", "enabled_model_families",
    }
    missing = {
        field.name for field in fields(Settings)
    } - covered - operational - portfolio
    assert not missing, f"Campos científicos fuera de toda huella: {sorted(missing)}"


def test_portfolio_settings_reach_the_backtest_from_the_catalog() -> None:
    """Toda variable de cartera del catálogo tiene un campo homónimo en Settings."""
    values = _values()
    settings = settings_from_values(values)
    for variable in VARIABLES:
        if variable.predictive:
            continue
        assert hasattr(settings, variable.id), f"{variable.id} no llega a Settings"
