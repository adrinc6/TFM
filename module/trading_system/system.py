from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from module.trading_system.data import DataLayer
from module.trading_system.evaluation import choose_best_strategy_per_stock, evaluate_model_predictions
from module.trading_system.portfolio import construct_portfolio
from module.trading_system.training import TrainingArtifacts, train_multi_agent_system


@dataclass
class TradingSystemOutput:
    diagnostics_all_strategies: pd.DataFrame
    diagnostics_per_stock: pd.DataFrame
    portfolio: pd.DataFrame
    model_performance: pd.DataFrame
    strategy_performance: pd.DataFrame
    evaluation_summary: pd.DataFrame


class MultiAgentTradingSystem:
    def __init__(self, data_layer: DataLayer | None = None) -> None:
        self.data_layer = data_layer or DataLayer()

    def run(self) -> TradingSystemOutput:
        artifacts: TrainingArtifacts = train_multi_agent_system(data_layer=self.data_layer)

        diagnostics_all_strategies = artifacts.strategy_diagnostics.copy()
        diagnostics_per_stock = choose_best_strategy_per_stock(diagnostics_all_strategies, score_col="meta_score")

        portfolio = construct_portfolio(
            stock_diagnostics=diagnostics_per_stock,
            score_col="meta_score",
        )

        diagnostics_per_stock = diagnostics_per_stock.copy()
        diagnostics_per_stock["selected_in_portfolio"] = False
        if not portfolio.empty:
            selected_index = pd.MultiIndex.from_frame(portfolio[["ticker", "date"]])
            diagnostics_index = pd.MultiIndex.from_frame(diagnostics_per_stock[["ticker", "date"]])
            diagnostics_per_stock["selected_in_portfolio"] = diagnostics_index.isin(selected_index)

        evaluation_summary = evaluate_model_predictions(diagnostics_all_strategies)

        return TradingSystemOutput(
            diagnostics_all_strategies=diagnostics_all_strategies,
            diagnostics_per_stock=diagnostics_per_stock,
            portfolio=portfolio,
            model_performance=artifacts.model_performance,
            strategy_performance=artifacts.strategy_performance,
            evaluation_summary=evaluation_summary,
        )


def run_trading_system() -> TradingSystemOutput:
    system = MultiAgentTradingSystem()
    return system.run()
