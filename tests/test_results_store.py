from __future__ import annotations

from dataclasses import replace

from environment import Settings
from module.runs.results_store import config_hash


def test_hash_changes_when_effective_setting_changes() -> None:
    base = Settings(run_scope="dev")
    changed = replace(base, target_size=12)

    assert config_hash(base, run_kind="test", mode="full", stages=("backtest",)) != config_hash(
        changed, run_kind="test", mode="full", stages=("backtest",)
    )


def test_execution_hash_ignores_run_role_but_not_mode() -> None:
    from module.runs.results_store import execution_hash

    settings = Settings(run_scope="dev")
    common = {"stages": ("backtest",), "inputs": {"scores": "abc"}}
    assert execution_hash(settings, run_kind="scenario", mode="full", **common) == execution_hash(
        settings, run_kind="final", mode="full", **common
    )
    assert execution_hash(settings, run_kind="scenario", mode="full", **common) != execution_hash(
        settings, run_kind="scenario", mode="backtest", **common
    )
