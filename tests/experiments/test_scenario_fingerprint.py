"""Contrato de huellas: mismo Settings -> misma huella; cambiar un parametro
irrelevante para una etapa no cambia su huella; cambiar uno relevante si."""

from __future__ import annotations

from dataclasses import replace

from environment import Settings
from module.experiments import stage_fingerprint


def test_same_settings_yield_same_fingerprint() -> None:
    a = Settings(run_scope="dev", data_start_date="2000-01-01", end_date="2001-01-01")
    b = Settings(run_scope="dev", data_start_date="2000-01-01", end_date="2001-01-01")
    for stage in ("dataset", "features", "agents", "backtest"):
        assert stage_fingerprint(stage, a) == stage_fingerprint(stage, b)


def test_snapshot_day_change_affects_dataset_and_downstream() -> None:
    """SNAPSHOT_DAY afecta al dataset (rejilla de snapshots), luego a todo."""
    base = Settings(run_scope="dev", data_start_date="2000-01-01", end_date="2001-01-01")
    changed = replace(base, snapshot_day=31)
    for stage in ("dataset", "features", "agents", "backtest"):
        assert stage_fingerprint(stage, base) != stage_fingerprint(stage, changed), (
            f"SNAPSHOT_DAY deberia afectar a la huella de {stage}"
        )


def test_only_portfolio_change_affects_only_backtest() -> None:
    """Cambiar top-N solo cambia la huella del backtest, no las anteriores.

    Es el caso que menciono el usuario: 'escoger algo y volver a cambiar la cartera para
    ver con los resultados que hay sin reentrenar'.
    """
    base = Settings(run_scope="dev", data_start_date="2000-01-01", end_date="2001-01-01")
    changed = replace(base, target_max=15, target_min=8)

    for stage in ("dataset", "features", "agents"):
        assert stage_fingerprint(stage, base) == stage_fingerprint(stage, changed), (
            f"TARGET_MAX/TARGET_MIN no deberian afectar a la huella de {stage}"
        )
    assert stage_fingerprint("backtest", base) != stage_fingerprint("backtest", changed)


def test_ridge_alpha_change_affects_agents_and_backtest_but_not_features() -> None:
    """RIDGE_ALPHA es del modelo: solo agents (y por dependencia backtest) cambian."""
    base = Settings(run_scope="dev", data_start_date="2000-01-01", end_date="2001-01-01")
    changed = replace(base, ridge_alpha=2.0)

    assert stage_fingerprint("dataset", base) == stage_fingerprint("dataset", changed)
    assert stage_fingerprint("features", base) == stage_fingerprint("features", changed)
    assert stage_fingerprint("agents", base) != stage_fingerprint("agents", changed)
    assert stage_fingerprint("backtest", base) != stage_fingerprint("backtest", changed)
