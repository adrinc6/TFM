from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from module.steps.step_03_trainer.pipeline import TrainingArtifacts, run_training_pipeline
from module.steps.step_04_evaluation.backtester import construct_portfolio
from module.steps.step_04_evaluation.diagnostics import add_portfolio_flag, build_per_stock_diagnostics
from module.steps.step_04_evaluation.metrics import evaluate_per_model_per_strategy


@dataclass
class EvaluationOutput:
    diagnostics_all_strategies: pd.DataFrame
    diagnostics_per_stock: pd.DataFrame
    model_strategy_metrics: pd.DataFrame
    model_performance: pd.DataFrame
    strategy_performance: pd.DataFrame
    portfolio: pd.DataFrame


def run_evaluation_engine(training_artifacts: TrainingArtifacts | None = None) -> EvaluationOutput:
    artifacts = training_artifacts or run_training_pipeline()

    all_strategies = artifacts.strategy_diagnostics.copy()
    per_stock = build_per_stock_diagnostics(all_strategies, score_col="meta_score")
    portfolio = construct_portfolio(per_stock, min_stocks=5, max_stocks=8, sector_cap=3, score_col="meta_score")
    per_stock = add_portfolio_flag(per_stock, portfolio)

    model_strategy_metrics = evaluate_per_model_per_strategy(all_strategies)

    return EvaluationOutput(
        diagnostics_all_strategies=all_strategies,
        diagnostics_per_stock=per_stock,
        model_strategy_metrics=model_strategy_metrics,
        model_performance=artifacts.model_performance.copy(),
        strategy_performance=artifacts.strategy_performance.copy(),
        portfolio=portfolio,
    )
