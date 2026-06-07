"""Tests for TP/SL vs Buy & Hold counterfactual evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from module.steps.step_04_evaluation.backtesting import WalkForwardBacktester


def _prices(values, start="2024-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.DataFrame({"Close": values}, index=idx)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "score": [0.90, 0.80],
            "sector": ["Tech", "Health"],
            "tp_pct": [0.05, 0.05],
            "sl_pct": [0.05, 0.05],
            "max_holding_days": [2, 2],
        }
    )


def test_buy_hold_uses_same_tickers_and_weights_and_ignores_tp_sl(tmp_path):
    backtester = WalkForwardBacktester(
        train_years=3,
        risk_free=0.0,
        results_dir=str(tmp_path / "backtest"),
        strategy_dir=str(tmp_path / "strategy"),
        top_n_stocks=2,
        score_weighted=False,
        portfolio_optimizer="none",
    )
    # AAA crosses TP early, then keeps rising. Buy & Hold must ignore TP and
    # therefore beat the TP/SL ticker return.
    prices_dict = {
        "AAA": _prices([100, 106, 120, 130, 140, 150]),
        "BBB": _prices([100, 100, 101, 101, 102, 102]),
    }
    benchmark = pd.Series(0.0, index=pd.bdate_range("2024-01-02", periods=6))

    result = backtester.simulate_portfolio(
        predictions_df=_predictions(),
        prices_dict=prices_dict,
        benchmark=benchmark,
        fold_id="F_BH",
        test_start=pd.Timestamp("2024-01-02"),
        test_end=pd.Timestamp("2024-01-09"),
    )

    assert result["buy_hold_selected_tickers"] == list(result["ticker_weights"].keys())
    assert set(result["buy_hold_selected_tickers"]) == set(result["ticker_returns"].keys())
    assert abs(sum(result["ticker_weights"].values()) - 1.0) < 1e-6
    assert result["buy_hold_ticker_returns"]["AAA"] > result["ticker_returns"]["AAA"]
    assert result["ticker_exit_reasons"]["AAA"] == "tp_hit"


def test_buy_hold_exit_is_target_or_last_available_and_exports(tmp_path):
    backtester = WalkForwardBacktester(
        train_years=3,
        risk_free=0.0,
        results_dir=str(tmp_path / "backtest"),
        strategy_dir=str(tmp_path / "strategy"),
        top_n_stocks=2,
        score_weighted=False,
        portfolio_optimizer="none",
    )
    prices_dict = {
        "AAA": _prices([100, 101, 102, 103]),
        "BBB": _prices([100, 99, 98, 97]),
    }
    benchmark = pd.Series(0.0, index=pd.bdate_range("2024-01-02", periods=8))

    result = backtester.simulate_portfolio(
        predictions_df=_predictions(),
        prices_dict=prices_dict,
        benchmark=benchmark,
        fold_id="F_EXPORT",
        test_start=pd.Timestamp("2024-01-02"),
        test_end=pd.Timestamp("2024-01-15"),
    )
    backtester.fold_results.append(result)
    summary = backtester.summarize()

    assert result["buy_hold_exit_dates"]["AAA"] == "2024-01-05"
    assert result["buy_hold_days_held"]["AAA"] == 3
    assert summary["tp_sl_vs_buy_hold"]["enabled"] is True
    assert (tmp_path / "strategy" / "tp_sl_vs_buy_hold_by_fold.csv").exists()
    assert (tmp_path / "strategy" / "tp_sl_vs_buy_hold_by_ticker.csv").exists()
    summary_path = tmp_path / "strategy" / "tp_sl_vs_buy_hold_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text())
    assert "mean_tp_sl_minus_buy_hold" in payload


def test_counterfactual_does_not_mutate_predictions_or_selection(tmp_path):
    backtester = WalkForwardBacktester(
        train_years=3,
        risk_free=0.0,
        results_dir=str(tmp_path / "backtest"),
        strategy_dir=str(tmp_path / "strategy"),
        top_n_stocks=2,
        score_weighted=False,
        portfolio_optimizer="none",
    )
    preds = _predictions()
    before = preds.copy(deep=True)
    prices_dict = {"AAA": _prices([100, 101, 102]), "BBB": _prices([100, 101, 102])}
    benchmark = pd.Series(0.0, index=pd.bdate_range("2024-01-02", periods=3))

    result = backtester.simulate_portfolio(
        predictions_df=preds,
        prices_dict=prices_dict,
        benchmark=benchmark,
        fold_id="F_NO_LEAK",
        test_start=pd.Timestamp("2024-01-02"),
        test_end=pd.Timestamp("2024-01-04"),
    )

    pd.testing.assert_frame_equal(preds, before)
    assert result["selected_tickers"] == ["AAA", "BBB"]
    assert result["buy_hold_selected_tickers"] == ["AAA", "BBB"]



def test_base_variant_preserves_original_tp_sl_outputs(tmp_path, monkeypatch):
    import module.steps.step_04_evaluation.backtesting as backtesting_mod

    monkeypatch.setattr(backtesting_mod, "TP_SL_VARIANT_MODE", "base")
    backtester = WalkForwardBacktester(
        train_years=3,
        risk_free=0.0,
        results_dir=str(tmp_path / "backtest"),
        strategy_dir=str(tmp_path / "strategy"),
        top_n_stocks=2,
        score_weighted=False,
        portfolio_optimizer="none",
    )
    preds = _predictions().assign(
        hybrid_tp_pct=[0.50, 0.50],
        hybrid_sl_pct=[0.30, 0.30],
        hybrid_trailing_stop_pct=[0.20, 0.20],
        hybrid_momentum_used=[0.25, 0.25],
        hybrid_volatility_used=[0.15, 0.15],
        hybrid_regime=["Risk-On", "Risk-On"],
    )
    prices_dict = {
        "AAA": _prices([100, 106, 120, 130, 140, 150]),
        "BBB": _prices([100, 96, 95, 94, 93, 92]),
    }
    benchmark = pd.Series(0.0, index=pd.bdate_range("2024-01-02", periods=6))

    result = backtester.simulate_portfolio(
        predictions_df=preds,
        prices_dict=prices_dict,
        benchmark=benchmark,
        fold_id="F_BASE_COMPAT",
        test_start=pd.Timestamp("2024-01-02"),
        test_end=pd.Timestamp("2024-01-09"),
    )

    assert result["selected_tickers"] == result["buy_hold_selected_tickers"]
    assert result["ticker_returns"] == result["base_ticker_returns"]
    assert result["ticker_exit_reasons"] == result["base_ticker_exit_reasons"]
    assert result["ticker_exit_dates"] == result["base_ticker_exit_dates"]
    assert result["ticker_weights"].keys() == result["base_ticker_returns"].keys()


def test_base_hybrid_and_buy_hold_use_same_portfolio_and_weights(tmp_path):
    backtester = WalkForwardBacktester(
        train_years=3,
        risk_free=0.0,
        results_dir=str(tmp_path / "backtest"),
        strategy_dir=str(tmp_path / "strategy"),
        top_n_stocks=2,
        score_weighted=False,
        portfolio_optimizer="none",
    )
    prices_dict = {"AAA": _prices([100, 110, 120]), "BBB": _prices([100, 90, 95])}
    benchmark = pd.Series(0.0, index=pd.bdate_range("2024-01-02", periods=3))

    result = backtester.simulate_portfolio(
        predictions_df=_predictions(),
        prices_dict=prices_dict,
        benchmark=benchmark,
        fold_id="F_SAME_PORT",
        test_start=pd.Timestamp("2024-01-02"),
        test_end=pd.Timestamp("2024-01-04"),
    )

    weight_tickers = set(result["ticker_weights"].keys())
    assert weight_tickers == set(result["base_ticker_returns"].keys())
    assert weight_tickers == set(result["hybrid_ticker_returns"].keys())
    assert weight_tickers == set(result["buy_hold_ticker_returns"].keys())
    assert result["buy_hold_selected_tickers"] == list(result["ticker_weights"].keys())


def test_hybrid_learned_levels_are_train_only_and_exportable_shape():
    from module.steps.step_04_evaluation.evaluator import _build_tp_sl_strategy_universe_matrix

    hist_dates = pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31", "2021-03-31", "2021-06-30", "2021-12-31"])
    hist_index = pd.MultiIndex.from_product([hist_dates, ["AAA"]], names=["date", "ticker"])
    history = pd.DataFrame(
        {
            "momentum_6m": [0.10, 0.12, 0.15, 0.08, 0.18, 0.11, 0.22],
            "volatility_60d": [0.12, 0.13, 0.11, 0.14, 0.12, 0.13, 0.16],
        },
        index=hist_index,
    )
    price_idx = pd.bdate_range("2020-01-01", "2022-12-31")
    prices = pd.DataFrame({"Close": [100.0 * (1.0008 ** i) for i in range(len(price_idx))]}, index=price_idx)
    preds = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "date": [pd.Timestamp("2022-03-31")],
            "score": [0.85],
            "confidence": [0.75],
            "momentum_6m": [0.20],
            "volatility_60d": [0.12],
            "regime_state": ["Risk-On"],
        }
    )

    matrix = _build_tp_sl_strategy_universe_matrix(
        preds_df=preds,
        history_source_df=history,
        prices_dict={"AAA": prices},
        entry_date=pd.Timestamp("2022-04-01"),
        lag_days=1,
        holding_period_months=12,
    )

    assert not matrix.empty
    assert {"hybrid_tp_pct", "hybrid_sl_pct", "hybrid_trailing_stop_pct"}.issubset(matrix.columns)
    assert {"max_runup_train_p50", "recovery_prob_after_10pct_drawdown", "probability_reach_30pct"}.issubset(matrix.columns)
    assert matrix["hybrid_train_paths"].max() >= 1
    latest_path_end = pd.to_datetime(matrix["latest_train_path_end"].dropna())
    assert (latest_path_end < pd.Timestamp("2022-03-31")).all()
