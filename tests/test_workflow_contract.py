from __future__ import annotations

import json
from pathlib import Path

from module.studies.selection import choose_candidate
from module.storage import studies
from module.web import queries


def _result(rank_ic: float) -> dict:
    return {
        "evaluation_key": str(rank_ic),
        "summary": {"mean_rank_ic": rank_ic, "rank_ic_positive_fraction": 0.75, "rank_ic_std": 0.02},
        "eras": [
            {"era": "2015-2018", "rank_ic": rank_ic},
            {"era": "2019-2021", "rank_ic": rank_ic},
            {"era": "2022-2024", "rank_ic": rank_ic},
        ],
        "rank_ic_by_cohort": [
            {"date": f"20{year:02d}-12-31", "rank_ic": rank_ic}
            for year in range(15, 25)
        ],
        "known_stress_not_selection": [{"year": 2025, "alpha": 99}],
    }


def test_selection_uses_rank_ic_and_excludes_known_stress() -> None:
    incumbent = {"candidate_id": "a", "value": 60, "result": _result(0.03)}
    challenger = {"candidate_id": "b", "value": 45, "result": _result(0.05)}
    decision = choose_candidate(incumbent, [incumbent, challenger], "execution_lag_days")
    assert decision["winner_candidate_id"] == "b"
    assert decision["selection_metric"] == "rank_ic_only"
    assert decision["known_stress_excluded"] is True


def test_run_exists_before_result_and_resume_keeps_identity(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(studies, "STUDIES_ROOT", tmp_path / "studies")
    study_id, _ = studies.create_study({"name": "test", "configuration": {}, "catalog": {}})
    first = studies.create_run(
        study_id, logical_key="baseline", phase="temporal",
        variable_id="baseline", value="baseline", configuration={},
    )
    second = studies.create_run(
        study_id, logical_key="baseline", phase="temporal",
        variable_id="baseline", value="baseline", configuration={}, attempt=2,
    )
    assert first["status"] == "queued"
    assert second["run_id"] == first["run_id"]
    assert second["attempt"] == 2


def test_events_are_persistent_and_incremental(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(studies, "STUDIES_ROOT", tmp_path / "studies")
    study_id, _ = studies.create_study({"name": "test", "configuration": {}, "catalog": {}})
    studies.append_event(study_id, "info", "one", "Primero.")
    studies.append_event(study_id, "info", "two", "Segundo.")
    assert [row["sequence"] for row in studies.read_events(study_id, after=1)] == [2]


def test_study_contains_only_model_study_identity(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(studies, "STUDIES_ROOT", tmp_path / "studies")
    _, directory = studies.create_study({"name": "test", "configuration": {}, "catalog": {}})
    study_id = json.loads((directory / "study.json").read_text(encoding="utf-8"))["study_id"]
    studies.create_run(
        study_id, logical_key="baseline", phase="temporal",
        variable_id="baseline", value="baseline", configuration={},
    )
    payload = json.loads((directory / "study.json").read_text(encoding="utf-8"))
    assert payload["study_type"] == "model_study"
    assert set(payload) >= {"study_id", "study_type", "status", "configuration"}
    assert queries.studies()[0]["max_rank_ic"] is None
    studies.update_study(study_id, status="succeeded", completed_runs=35)
    assert queries.studies()[0]["runs_remaining"] == 0
