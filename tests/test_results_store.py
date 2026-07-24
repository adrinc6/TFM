from __future__ import annotations

from dataclasses import replace

import json
import pandas as pd

from environment import Settings
from module.runs.results_store import ResultsStore, config_hash


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


def test_compact_candidate_does_not_publish_heavy_inputs(tmp_path) -> None:
    store = ResultsStore(tmp_path / "results")
    settings = Settings(run_scope="dev")
    _, run_dir, _ = store.create_run(
        settings, run_kind="scenario", mode="full", stages=("agents",),
        label="compact",
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "backtest_summary.json").write_text("{}", encoding="utf-8")
    pd.DataFrame({"rank_ic": [0.1]}).to_parquet(
        source / "rank_ic_diagnostics.parquet", index=False
    )
    pd.DataFrame({"ticker": ["AAA"]}).to_parquet(
        source / "agent_scores.parquet", index=False
    )
    pd.DataFrame({"ticker": ["AAA"]}).to_parquet(
        source / "features_point_in_time.parquet", index=False
    )

    store.publish_artifacts(run_dir, source, retention_policy="compact_candidate")

    artifacts = run_dir / "artifacts"
    assert (artifacts / "rank_ic_diagnostics.parquet").exists()
    assert not (artifacts / "agent_scores.parquet").exists()
    assert not (artifacts / "features_point_in_time.parquet").exists()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"]["retention_policy"] == "compact_candidate"
