from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd

from environment import Settings
from module.runs.results_store import ResultsStore, canonical_json, config_hash, execution_hash
from module.runs.recycle import publish, restore, stage_key
from module.runs.execution import _cache_contract


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})


def test_hash_changes_when_effective_setting_changes():
    settings = Settings(run_mode="agents", run_scope="dev")
    common = {"run_kind": "experimental", "mode": "agents", "stages": ["agents"]}
    # Se usa un valor distinto del default (8) para asegurar que el cambio altera el hash.
    assert config_hash(settings, **common) != config_hash(replace(settings, train_lookback_years=6), **common)


def test_execution_hash_ignores_presentation_intent():
    settings = Settings(run_mode="agents", run_scope="dev")
    common = {"run_kind": "experimental", "mode": "agents", "stages": ["agents"], "inputs": {}}
    assert execution_hash(settings, **common) == execution_hash(settings, **common)


def test_execution_hash_ignores_run_role_but_not_mode():
    settings = Settings(run_mode="agents", run_scope="dev")
    base = execution_hash(settings, run_kind="scenario", mode="agents", stages=["agents"], inputs={})
    promoted = execution_hash(settings, run_kind="optimization_final", mode="agents",
                              stages=["agents"], inputs={})
    other_mode = execution_hash(settings, run_kind="scenario", mode="full", stages=["agents"], inputs={})
    assert base == promoted
    assert base != other_mode


def test_run_directory_uses_day_and_repetition_suffix(tmp_path):
    store = ResultsStore(tmp_path / "results")
    settings = Settings(run_mode="agents", run_scope="dev")
    first_id, first_dir, _ = store.create_run(settings, run_kind="experimental", mode="agents",
                                               stages=["agents"], label="Prueba")
    second_id, second_dir, _ = store.create_run(settings, run_kind="experimental", mode="agents",
                                                 stages=["agents"], label="Prueba")
    assert len(first_id.split("--")[0]) == 8
    assert second_id.endswith("--r02")
    assert first_dir.exists() and second_dir.exists()


def test_registry_is_append_only(tmp_path):
    store = ResultsStore(tmp_path / "results")
    settings = Settings(run_mode="agents", run_scope="dev")
    _, run_dir, _ = store.create_run(settings, run_kind="experimental", mode="agents",
                                     stages=["agents"], label="Prueba")
    store.complete(run_dir, {"mean_rank_ic": 0.03})
    rows = [json.loads(line) for line in store.registry_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["summary"]["mean_rank_ic"] == 0.03


def test_recycle_key_and_restore_are_deterministic(tmp_path, monkeypatch):
    import module.runs.recycle as recycle

    monkeypatch.setattr(recycle, "RECYCLE_ROOT", tmp_path / "recycle")
    settings = Settings(run_mode="dataset", run_scope="dev")
    source = tmp_path / "source.parquet"
    source.write_bytes(b"artefacto")
    key = stage_key("dataset", settings, [source])
    publish("dataset", key, [source], settings)
    destination = tmp_path / "destination"
    assert restore("dataset", key, destination)
    assert (destination / "source.parquet").read_bytes() == b"artefacto"


def test_agent_recycle_key_includes_recency_weighting(tmp_path):
    source = tmp_path / "features.parquet"
    source.write_bytes(b"features")
    off = Settings(run_mode="agents", run_scope="dev", recency_weighting="off")
    weighted = replace(off, recency_weighting="exponential")
    assert stage_key("agents", off, [source]) != stage_key("agents", weighted, [source])


def test_recycle_rejects_corrupted_artifact(tmp_path, monkeypatch):
    import module.runs.recycle as recycle

    monkeypatch.setattr(recycle, "RECYCLE_ROOT", tmp_path / "recycle")
    settings = Settings(run_mode="dataset", run_scope="dev")
    source = tmp_path / "source.parquet"
    source.write_bytes(b"original")
    key = stage_key("dataset", settings, [source])
    cached = publish("dataset", key, [source], settings)
    (cached / source.name).write_bytes(b"corrupto")
    assert not restore("dataset", key, tmp_path / "destination")


def test_publish_artifacts_copies_immutable_stock_panel_and_local_attribution(tmp_path):
    store = ResultsStore(tmp_path / "results")
    settings = Settings(run_mode="agents", run_scope="dev")
    _, run_dir, _ = store.create_run(settings, run_kind="experimental", mode="agents",
                                     stages=["agents"], label="Stocks")
    processed = tmp_path / "processed"
    source = processed / "agents" / "lgbm-test"
    source.mkdir(parents=True)
    pd.DataFrame({"ticker": ["AAA"], "snapshot_date": ["2020-01-15"], "pe": [10.0]}).to_parquet(
        processed / "panel_point_in_time.parquet"
    )
    pd.DataFrame({"ticker": ["AAA"], "agent": ["value"], "local_contribution": [0.2]}).to_parquet(
        source / "agent_local_attribution.parquet"
    )
    store.publish_artifacts(run_dir, source)
    assert (run_dir / "artifacts" / "stock_panel.parquet").exists()
    assert (run_dir / "artifacts" / "agent_local_attribution.parquet").exists()


def test_backtest_cache_contract_uses_explicit_agent_not_latest_directory(tmp_path):
    processed = tmp_path / "processed"
    selected = processed / "agents" / "lgbm-a"
    other = processed / "agents" / "lgbm-z"
    selected.mkdir(parents=True)
    other.mkdir(parents=True)
    for path in (selected / "agent_scores.parquet", processed / "asset_price_point_in_time.parquet",
                 processed / "benchmark_point_in_time.parquet"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"input")

    inputs, outputs, destination = _cache_contract("backtest", processed, agent_dir=selected)

    assert inputs[0] == selected / "agent_scores.parquet"
    assert destination == selected
    assert all(path.parent == selected for path in outputs())
