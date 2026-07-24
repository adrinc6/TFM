from __future__ import annotations

from dataclasses import replace

import pandas as pd

import main
from environment import Settings
from module.runs.experiments import select_winner, split_variables, stage_fingerprint, stress_variables
from module.scenarios.variables import FULL_STUDY_OPTIONS, FULL_STUDY_PHASE3_OPTIONS


def test_only_portfolio_change_affects_only_backtest() -> None:
    base = Settings(run_scope="dev")
    changed = replace(base, target_size=12)

    assert stage_fingerprint("dataset", base) == stage_fingerprint("dataset", changed)
    assert stage_fingerprint("features", base) == stage_fingerprint("features", changed)
    assert stage_fingerprint("agents", base) == stage_fingerprint("agents", changed)
    assert stage_fingerprint("backtest", base) != stage_fingerprint("backtest", changed)


def test_stage_selector_exposes_the_supported_pipeline() -> None:
    assert main.stages_for_run("full") == (
        "download", "dataset", "features", "agents", "backtest", "report",
    )
    assert main.stages_for_run("full_study") == ("full_study",)


def test_selection_ignores_alpha_columns() -> None:
    summary = pd.DataFrame([
        {"scenario": "learner", "mean_rank_ic": 0.10, "rank_ic_positive_fraction": 0.8,
         "beat_rate": 0.6, "max_drawdown": 0.2, "mean_annual_alpha": -1.0},
        {"scenario": "high_alpha", "mean_rank_ic": 0.02, "rank_ic_positive_fraction": 0.4,
         "beat_rate": 0.4, "max_drawdown": 0.4, "mean_annual_alpha": 99.0},
    ])

    winner, ranked = select_winner(summary)

    assert winner == "learner"
    assert "mean_annual_alpha" not in {column for column in ranked if column.startswith("rank_")}


def test_mechanical_portfolio_rules_are_stressed_not_optimized() -> None:
    variables = {"target_size": [8, 12], "min_hold_percentile": [70, 80], "objective": ["rank_regression"]}
    model, portfolio = split_variables(variables)

    assert model == {"objective": ["rank_regression"]}
    assert portfolio == {"target_size": [8, 12]}
    assert stress_variables(variables) == {"min_hold_percentile": [70, 80]}


def test_study_and_full_study_share_the_same_catalogue_by_phase() -> None:
    assert "lgbm_max_depth" not in FULL_STUDY_OPTIONS
    assert "lgbm_max_depth" in FULL_STUDY_PHASE3_OPTIONS
    assert "target_size" in FULL_STUDY_OPTIONS
