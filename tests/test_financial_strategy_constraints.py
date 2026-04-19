"""Tests for financial strategy simplifications and portfolio constraints."""

from __future__ import annotations

import pandas as pd

from module.steps.step_03_training.agent_config import build_agents_config
from module.steps.step_04_evaluation.backtester import WalkForwardBacktester


def _make_prices(start: str = "2024-01-02", periods: int = 15, base: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=periods)
    vals = [base + i for i in range(periods)]
    return pd.DataFrame({"Close": vals}, index=idx)


def test_sentiment_agent_disabled_by_default():
    cfg = build_agents_config(agent_models_results_dir=".", random_seed=42)
    assert "sentiment" not in cfg


def test_backtester_limits_sector_concentration(tmp_path):
    backtester = WalkForwardBacktester(
        train_years=5,
        test_quarters=1,
        risk_free=0.0,
        results_dir=str(tmp_path / "backtest"),
        top_n_stocks=5,
        score_weighted=True,
    )

    predictions = pd.DataFrame(
        {
            "ticker": ["A1", "A2", "A3", "A4", "B1", "B2"],
            "score": [0.90, 0.89, 0.88, 0.87, 0.86, 0.85],
            "sector": ["Tech", "Tech", "Tech", "Tech", "Health", "Energy"],
        }
    )
    prices_dict = {tk: _make_prices(base=100.0 + i * 10.0) for i, tk in enumerate(predictions["ticker"])}
    bench_idx = pd.bdate_range("2024-01-02", periods=15)
    benchmark = pd.Series(0.001, index=bench_idx)

    result = backtester.simulate_portfolio(
        predictions_df=predictions,
        prices_dict=prices_dict,
        benchmark=benchmark,
        fold_id="F1",
        test_start=pd.Timestamp("2024-01-02"),
        test_end=pd.Timestamp("2024-01-22"),
    )

    selected = result["selected_tickers"]
    selected_sectors = predictions.set_index("ticker").loc[selected, "sector"]
    assert int((selected_sectors == "Tech").sum()) <= 3


def test_backtester_applies_position_weight_cap(tmp_path):
    backtester = WalkForwardBacktester(
        train_years=5,
        test_quarters=1,
        risk_free=0.0,
        results_dir=str(tmp_path / "backtest"),
        top_n_stocks=8,
        score_weighted=True,
    )

    tickers = [f"T{i}" for i in range(1, 9)]
    predictions = pd.DataFrame(
        {
            "ticker": tickers,
            "score": [0.99, 0.93, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80],
            "sector": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"],
        }
    )
    prices_dict = {tk: _make_prices(base=50.0 + i) for i, tk in enumerate(tickers)}
    bench_idx = pd.bdate_range("2024-01-02", periods=15)
    benchmark = pd.Series(0.001, index=bench_idx)

    result = backtester.simulate_portfolio(
        predictions_df=predictions,
        prices_dict=prices_dict,
        benchmark=benchmark,
        fold_id="F2",
        test_start=pd.Timestamp("2024-01-02"),
        test_end=pd.Timestamp("2024-01-22"),
    )

    weights = result["ticker_weights"]
    assert weights
    assert max(weights.values()) <= 0.150001
