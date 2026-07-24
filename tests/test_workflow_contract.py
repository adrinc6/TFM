from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from module.studies.confirmatory import (
    CONFIRMATORY_BREAKDOWN,
    confirmatory_preflight,
)
from module.studies.config import CONFIRMATORY_EVALUATIONS
from module.studies.exploratory import _hypothesis_statement
from module.studies import exploratory
from module.studies.catalog import default_definition
from module.studies.runner import _selection_diagnostics
from module.storage import evidence


def test_confirmatory_budget_is_exactly_23() -> None:
    assert CONFIRMATORY_EVALUATIONS == 23
    assert sum(CONFIRMATORY_BREAKDOWN.values()) == 23


def test_confirmatory_rejects_scientific_overrides_before_execution() -> None:
    with pytest.raises(ValueError, match="no admite overrides"):
        confirmatory_preflight({"hypothesis_id": "hyp-valid", "target_size": 99})


def test_selection_diagnostics_exclude_known_stress() -> None:
    diagnostics = pd.DataFrame({
        "agent": ["meta_final"] * 4,
        "prediction_date": ["2023-03-31", "2024-03-31", "2025-03-31", "2026-03-31"],
        "rank_ic": [0.1, 0.2, 0.9, 0.9],
    })
    selected = _selection_diagnostics(diagnostics)
    assert selected["year"].tolist() == [2023, 2024]


def test_frozen_hypothesis_is_immutable_by_contract(tmp_path: Path, monkeypatch) -> None:
    hypotheses = tmp_path / "hypotheses"
    monkeypatch.setattr(evidence, "HYPOTHESES_ROOT", hypotheses)
    hypothesis_id, path = evidence.freeze_hypothesis({"configuration": {"x": 1}})
    first = evidence.read_hypothesis(hypothesis_id)
    assert first["status"] == "frozen"
    payload = json.loads((path / "hypothesis.json").read_text(encoding="utf-8"))
    payload["status"] = "draft"
    (path / "hypothesis.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="no está congelada"):
        evidence.read_hypothesis(hypothesis_id)


def test_hypothesis_statement_is_structured() -> None:
    values = {
        "model_family": "lightgbm",
        "target_horizon_months": 12,
        "meta_method": "equal",
    }
    statement = _hypothesis_statement(values)
    assert "12 meses" in statement
    assert "cartera dinámica" in statement


def test_exploratory_advances_one_variable_and_keeps_compact_ledger(
    tmp_path: Path, monkeypatch,
) -> None:
    studies = tmp_path / "studies"
    monkeypatch.setattr(evidence, "STUDIES_ROOT", studies)
    monkeypatch.setattr(evidence, "HYPOTHESES_ROOT", tmp_path / "hypotheses")
    monkeypatch.setattr(evidence, "MODELS_ROOT", tmp_path / "models")
    monkeypatch.setattr(exploratory, "STUDIES_ROOT", studies)
    monkeypatch.setattr(exploratory, "prune_prepared", lambda **_: {})
    monkeypatch.setattr(exploratory, "discard_summary_cache", lambda _: None)

    def fake_run(values, **_):
        horizon = values["target_horizon_months"]
        rank_ic = horizon / 100
        return {
            "evaluation_key": f"{horizon:064x}",
            "dataset_hash": f"{horizon + 100:064x}",
            "source": "computed",
            "summary": {
                "mean_rank_ic": rank_ic,
                "rank_ic_positive_fraction": 1.0,
                "rank_ic_std": 0.01,
                "tail_spread": rank_ic,
                "information_ratio": 0.2,
                "annualized_turnover": 1.0,
                "positive_alpha_eras": 3,
            },
            "eras": [
                {"era": "2015-2018", "rank_ic": rank_ic, "mean_alpha": 0.01},
                {"era": "2019-2021", "rank_ic": rank_ic, "mean_alpha": 0.01},
                {"era": "2022-2024", "rank_ic": rank_ic, "mean_alpha": 0.01},
            ],
            "rank_ic_by_cohort": [
                {"date": "2023-03-31", "rank_ic": rank_ic},
                {"date": "2024-03-31", "rank_ic": rank_ic},
            ],
        }

    monkeypatch.setattr(exploratory, "run_evaluation", fake_run)
    definition = default_definition()
    definition["target_horizon_months"] = {
        "mode": "optimize", "values": [3, 6, 12],
    }
    pending = exploratory.create_exploratory({"definition": definition})
    assert pending["status"] == "awaiting_decision"
    assert pending["pending_decision"]["variable_id"] == "target_horizon_months"
    finished = exploratory.advance_exploratory(pending["study_id"])
    assert finished["status"] == "awaiting_freeze"
    ledger = pd.read_parquet(studies / pending["study_id"] / "evaluation_ledger.parquet")
    assert len(ledger) == 4
    assert ledger["selected"].sum() == 2  # baseline + ganador de la variable


def test_no_mojibake_in_runtime_and_app() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for base in ("module", "app"):
        for path in (root / base).rglob("*"):
            if path.suffix not in {".py", ".js", ".html", ".css"}:
                continue
            text = path.read_text(encoding="utf-8")
            if any(token in text for token in (chr(0xC3), chr(0xC2), chr(0xFFFD))):
                offenders.append(str(path.relative_to(root)))
    assert offenders == []
