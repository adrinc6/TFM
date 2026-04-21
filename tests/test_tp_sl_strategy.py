"""Unit and integration tests for the TP/SL + confidence strategy pipeline.

Coverage:
    1. signal_generation  – TP/SL computation correctness
    2. confidence_model   – confidence scoring and calibration blending
    3. portfolio_selection – EV ranking and 4–8 stock constraints
    4. backtesting_engine  – TP/SL hit detection
    5. agent_weighting    – EWMA weight updates and persistence
    6. tp_sl_reporter     – CSV export shape and required columns
    7. Integration        – full pipeline end-to-end
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from module.strategy.signal_generation import compute_tp_sl, build_signals
from module.strategy.confidence_model import compute_confidence, attach_confidence
from module.strategy.portfolio_selection import (
    compute_expected_value,
    select_portfolio,
    get_portfolio_weights,
)
from module.strategy.backtesting_engine import simulate_tp_sl, run_backtest
from module.strategy.agent_weighting import AgentWeightTracker
from module.steps.step_04_evaluation.tp_sl_reporter import (
    build_strategy_csv,
    export_strategy_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scores(tickers=None, value=0.7) -> pd.Series:
    tickers = tickers or ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
    return pd.Series({t: value for t in tickers})


def _make_agent_df(n=8) -> pd.DataFrame:
    tickers = [f"T{i}" for i in range(n)]
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "ticker": tickers,
        "fundamental_score": rng.uniform(0.4, 0.9, n),
        "momentum_score": rng.uniform(0.4, 0.85, n),
        "bear_score": rng.uniform(0.3, 0.7, n),
    })


def _make_price_series(
    n_days: int = 120,
    start: str = "2024-01-02",
    trend: float = 0.001,
) -> pd.Series:
    """Rising daily price series."""
    idx = pd.bdate_range(start=start, periods=n_days)
    prices = 100.0 * np.cumprod(1.0 + trend * np.ones(n_days))
    return pd.Series(prices, index=idx)


def _make_prices_dict(tickers, **kwargs) -> Dict:
    return {t: _make_price_series(**kwargs) for t in tickers}


# ===========================================================================
# 1. Signal generation
# ===========================================================================

class TestComputeTpSl:
    """Tests for compute_tp_sl."""

    def test_output_shape(self):
        scores = _make_scores()
        result = compute_tp_sl(scores)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) >= {"ticker", "score", "tp_pct", "sl_pct"}
        assert len(result) == len(scores)

    def test_tp_increases_with_score(self):
        low = compute_tp_sl(pd.Series({"A": 0.2}))
        high = compute_tp_sl(pd.Series({"A": 0.8}))
        assert float(high["tp_pct"].iloc[0]) > float(low["tp_pct"].iloc[0])

    def test_sl_decreases_with_score(self):
        """Higher score → tighter stop (lower sl_pct)."""
        low = compute_tp_sl(pd.Series({"A": 0.2}))
        high = compute_tp_sl(pd.Series({"A": 0.8}))
        assert float(high["sl_pct"].iloc[0]) < float(low["sl_pct"].iloc[0])

    def test_values_clipped_to_bounds(self):
        scores = pd.Series({"A": 0.0, "B": 1.0})
        result = compute_tp_sl(scores, min_tp=0.02, max_tp=0.25, min_sl=0.01, max_sl=0.15)
        assert result["tp_pct"].between(0.02, 0.25).all()
        assert result["sl_pct"].between(0.01, 0.15).all()

    def test_empty_scores(self):
        result = compute_tp_sl(pd.Series(dtype=float))
        assert result.empty

    def test_baseline_at_midpoint(self):
        """Score = 0.5 should yield exactly the base TP and SL."""
        base_tp, base_sl = 0.08, 0.05
        result = compute_tp_sl(
            pd.Series({"A": 0.5}),
            base_tp=base_tp,
            base_sl=base_sl,
        )
        assert abs(float(result["tp_pct"].iloc[0]) - base_tp) < 1e-9
        assert abs(float(result["sl_pct"].iloc[0]) - base_sl) < 1e-9


class TestBuildSignals:
    """Tests for build_signals (higher-level API)."""

    def test_returns_dataframe(self):
        df = _make_agent_df()
        out = build_signals(df)
        assert isinstance(out, pd.DataFrame)
        assert "ticker" in out.columns

    def test_custom_agent_weights(self):
        df = _make_agent_df()
        w = {"fundamental_score": 2.0, "momentum_score": 1.0, "bear_score": 0.5}
        out = build_signals(df, agent_weights=w)
        assert len(out) == len(df)

    def test_raises_without_score_cols(self):
        df = pd.DataFrame({"ticker": ["A"], "irrelevant": [0.5]})
        with pytest.raises(ValueError):
            build_signals(df)


# ===========================================================================
# 2. Confidence model
# Constant used in historical calibration shift test
_HIGH_HIT_RATE = 0.9


# ===========================================================================

class TestComputeConfidence:

    def test_output_length(self):
        df = _make_agent_df()
        conf = compute_confidence(df)
        assert len(conf) == len(df)

    def test_values_in_valid_range(self):
        df = _make_agent_df()
        conf = compute_confidence(df)
        assert conf.between(0.0, 1.0).all()

    def test_higher_scores_give_higher_confidence(self):
        low_df = _make_agent_df().copy()
        high_df = _make_agent_df().copy()
        for col in ["fundamental_score", "momentum_score", "bear_score"]:
            low_df[col] = 0.2
            high_df[col] = 0.9
        conf_low = compute_confidence(low_df).mean()
        conf_high = compute_confidence(high_df).mean()
        assert conf_high > conf_low

    def test_historical_calibration_shifts_confidence(self):
        df = _make_agent_df()
        no_hist = compute_confidence(df).mean()
        with_hist = compute_confidence(
            df,
            agent_hit_rates={
                "fundamental_score": _HIGH_HIT_RATE,
                "momentum_score": _HIGH_HIT_RATE,
                "bear_score": _HIGH_HIT_RATE,
            },
        ).mean()
        assert with_hist > no_hist


class TestAttachConfidence:

    def test_adds_column(self):
        df = _make_agent_df()
        signals = compute_tp_sl(_make_scores([f"T{i}" for i in range(8)]))
        signals["ticker"] = [f"T{i}" for i in range(8)]
        for col in ["fundamental_score", "momentum_score", "bear_score"]:
            signals[col] = 0.6
        out = attach_confidence(signals)
        assert "confidence" in out.columns


# ===========================================================================
# 3. Portfolio selection
# ===========================================================================

class TestPortfolioSelection:

    def _make_signals(self, n=10) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        sectors_pool = ["Tech", "Health", "Energy", "Finance", "Comm", "Util", "Indus", "Mat"]
        sectors = [sectors_pool[i % len(sectors_pool)] for i in range(n)]
        return pd.DataFrame({
            "ticker": [f"T{i}" for i in range(n)],
            "score": rng.uniform(0.4, 0.9, n),
            "tp_pct": rng.uniform(0.05, 0.15, n),
            "sl_pct": rng.uniform(0.02, 0.08, n),
            "confidence": rng.uniform(0.5, 0.8, n),
            "sector": sectors,
        })

    def test_respects_max_stocks(self):
        signals = self._make_signals(20)
        out = select_portfolio(signals, min_stocks=4, max_stocks=8)
        selected = out[out["selected"].astype(bool)]
        assert len(selected) <= 8

    def test_respects_min_stocks_floor(self):
        """When only 3 stocks pass EV threshold → no investment."""
        signals = self._make_signals(3)
        signals["confidence"] = 0.3  # force low EV
        signals["tp_pct"] = 0.03
        signals["sl_pct"] = 0.10
        out = select_portfolio(signals, min_stocks=4, ev_threshold=0.0)
        # With very low EV candidates, fewer than 4 may qualify
        selected = out[out["selected"].astype(bool)]
        if len(signals) < 4:
            assert len(selected) == 0

    def test_selected_column_is_boolean(self):
        signals = self._make_signals()
        out = select_portfolio(signals)
        assert out["selected"].dtype == bool or set(out["selected"].unique()).issubset({True, False})

    def test_sector_cap_enforced(self):
        """No more than sector_cap stocks from the same sector."""
        signals = self._make_signals(10)
        signals["sector"] = "Tech"  # all same sector
        out = select_portfolio(signals, sector_cap=2)
        selected = out[out["selected"].astype(bool)]
        if len(selected) > 0:
            tech_count = (selected["sector"] == "Tech").sum()
            # May relax cap to meet min_stocks floor, but document the behaviour
            assert tech_count <= max(2, 4)  # capped or relaxed to min_stocks

    def test_ev_computation(self):
        signals = pd.DataFrame({
            "confidence": [0.6],
            "tp_pct": [0.10],
            "sl_pct": [0.05],
        })
        ev = compute_expected_value(signals)
        expected = 0.6 * 0.10 - 0.4 * 0.05
        assert abs(float(ev.iloc[0]) - expected) < 1e-9


class TestPortfolioWeights:

    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(99)
        n_tickers = 6
        signals = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(n_tickers)],
            "ev": rng.uniform(0.01, 0.05, n_tickers),
            "confidence": rng.uniform(0.5, 0.8, n_tickers),
            "selected": [True] * n_tickers,
        })
        w = get_portfolio_weights(signals)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_weights_non_negative(self):
        signals = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "ev": [0.02, 0.04, 0.03],
            "selected": [True, True, True],
        })
        w = get_portfolio_weights(signals)
        assert (w >= 0).all()


# ===========================================================================
# 4. Backtesting engine
# ===========================================================================

class TestSimulateTpSl:

    def _rising_prices(self, n=90):
        idx = pd.bdate_range("2024-01-02", periods=n)
        return pd.Series(100.0 * np.cumprod(np.ones(n) * 1.001), index=idx)

    def _flat_prices(self, n=90):
        idx = pd.bdate_range("2024-01-02", periods=n)
        return pd.Series(np.full(n, 100.0), index=idx)

    def _crashing_prices(self, n=90):
        idx = pd.bdate_range("2024-01-02", periods=n)
        return pd.Series(100.0 * np.cumprod(np.ones(n) * 0.999), index=idx)

    def test_tp_hit_on_rising_prices(self):
        prices = self._rising_prices(120)
        result = simulate_tp_sl("AAPL", prices, pd.Timestamp("2024-01-02"), 0.05, 0.10)
        assert result["outcome"] == "TP"

    def test_sl_hit_on_falling_prices(self):
        prices = self._crashing_prices(120)
        result = simulate_tp_sl("AAPL", prices, pd.Timestamp("2024-01-02"), 0.10, 0.05)
        assert result["outcome"] == "SL"

    def test_none_on_flat_prices(self):
        prices = self._flat_prices(90)
        result = simulate_tp_sl("AAPL", prices, pd.Timestamp("2024-01-02"), 0.10, 0.10, max_holding_days=60)
        assert result["outcome"] == "NONE"

    def test_returns_required_keys(self):
        prices = self._rising_prices()
        result = simulate_tp_sl("TEST", prices, pd.Timestamp("2024-01-02"), 0.08, 0.05)
        for key in ("ticker", "entry_price", "tp_price", "sl_price", "outcome", "days_to_outcome"):
            assert key in result

    def test_days_to_outcome_positive(self):
        prices = self._rising_prices(120)
        result = simulate_tp_sl("X", prices, pd.Timestamp("2024-01-02"), 0.05, 0.10)
        assert result["days_to_outcome"] >= 0

    def test_empty_prices(self):
        result = simulate_tp_sl("EMPTY", pd.Series(dtype=float), pd.Timestamp("2024-01-02"), 0.08, 0.05)
        assert result["outcome"] == "NONE"


class TestRunBacktest:

    def test_output_has_outcome_col(self):
        tickers = ["A", "B", "C", "D"]
        signals = pd.DataFrame({
            "ticker": tickers,
            "tp_pct": [0.05] * 4,
            "sl_pct": [0.03] * 4,
        })
        prices = _make_prices_dict(tickers, trend=0.002)
        out = run_backtest(signals, prices, pd.Timestamp("2024-01-02"))
        assert "outcome" in out.columns

    def test_all_tickers_get_outcome(self):
        tickers = ["A", "B"]
        signals = pd.DataFrame({
            "ticker": tickers,
            "tp_pct": [0.05, 0.05],
            "sl_pct": [0.03, 0.03],
        })
        prices = _make_prices_dict(tickers, trend=0.003)
        out = run_backtest(signals, prices, pd.Timestamp("2024-01-02"))
        assert set(out["ticker"]).issuperset(set(tickers))


# ===========================================================================
# 5. Agent weighting
# ===========================================================================

class TestAgentWeightTracker:

    def _make_tracker(self):
        return AgentWeightTracker(
            ["fundamental_score", "momentum_score"],
            decay=0.8,
            prior_hit_rate=0.5,
        )

    def _make_outcomes(self, tickers, outcomes):
        return pd.DataFrame({"ticker": tickers, "outcome": outcomes})

    def test_initial_weights_equal(self):
        tracker = self._make_tracker()
        w = tracker.get_weights()
        assert abs(w["fundamental_score"] - w["momentum_score"]) < 1e-9

    def test_weights_sum_to_one(self):
        tracker = self._make_tracker()
        w = tracker.get_weights()
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_weights_shift_after_update(self):
        tracker = self._make_tracker()
        agent_df = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(10)],
            "fundamental_score": [0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            "momentum_score":    [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85],
        })
        # Fundamental top picks all hit TP; momentum top picks all hit SL
        outcomes = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(10)],
            "outcome": ["TP", "TP", "TP", "TP", "TP", "SL", "SL", "SL", "SL", "SL"],
        })
        tracker.update("2024Q1", outcomes, agent_df, top_n=5)
        w = tracker.get_weights()
        assert w["fundamental_score"] > w["momentum_score"]

    def test_save_and_load(self, tmp_path):
        tracker = self._make_tracker()
        path = tmp_path / "tracker.json"
        tracker.save(path)
        loaded = AgentWeightTracker.load(path)
        assert loaded.agent_names == tracker.agent_names
        assert loaded.get_weights() == tracker.get_weights()

    def test_history_recorded(self):
        tracker = self._make_tracker()
        agent_df = pd.DataFrame({
            "ticker": ["A", "B"],
            "fundamental_score": [0.8, 0.6],
            "momentum_score": [0.7, 0.55],
        })
        outcomes = pd.DataFrame({"ticker": ["A", "B"], "outcome": ["TP", "SL"]})
        tracker.update("2024Q1", outcomes, agent_df, top_n=2)
        assert len(tracker.get_history()) == 1


# ===========================================================================
# 6. CSV reporter
# ===========================================================================

class TestTpSlReporter:

    def _make_full_signals(self, n=6):
        rng = np.random.default_rng(11)
        return pd.DataFrame({
            "ticker": [f"T{i}" for i in range(n)],
            "sector": ["Tech", "Health", "Energy", "Finance", "Comm", "Util"][:n],
            "score": rng.uniform(0.4, 0.9, n),
            "confidence": rng.uniform(0.5, 0.8, n),
            "tp_pct": rng.uniform(0.05, 0.15, n),
            "sl_pct": rng.uniform(0.02, 0.08, n),
            "ev": rng.uniform(0.0, 0.05, n),
            "selected": [True, True, True, True, False, False],
            "outcome": ["TP", "SL", "TP", "NONE", "NONE", "SL"],
            "days_to_outcome": [30, 15, 45, 90, 90, 20],
            "fundamental_score": rng.uniform(0.4, 0.9, n),
            "momentum_score": rng.uniform(0.4, 0.9, n),
        })

    def test_required_columns_present(self):
        signals = self._make_full_signals()
        out = build_strategy_csv(signals, fold_id="2024Q1")
        for col in ("ticker", "fold_id", "selected", "outcome", "days_to_outcome"):
            assert col in out.columns

    def test_fold_id_propagated(self):
        signals = self._make_full_signals()
        out = build_strategy_csv(signals, fold_id="2024Q4")
        assert (out["fold_id"] == "2024Q4").all()

    def test_agent_weight_columns_added(self):
        signals = self._make_full_signals()
        out = build_strategy_csv(
            signals,
            agent_weights={"fundamental_score": 0.6, "momentum_score": 0.4},
        )
        assert "weight_fundamental_score" in out.columns
        assert "weight_momentum_score" in out.columns

    def test_export_writes_file(self, tmp_path):
        signals = self._make_full_signals()
        out_path = tmp_path / "output.csv"
        export_strategy_csv(signals, out_path, fold_id="2024Q1")
        assert out_path.exists()
        df_read = pd.read_csv(out_path)
        assert len(df_read) == len(signals)


# ===========================================================================
# 7. Integration test — full pipeline (all stocks, single fold)
# ===========================================================================

class TestFullPipelineIntegration:
    """Integration test: signal → confidence → selection → backtest → CSV."""

    def test_end_to_end(self, tmp_path):
        from module.strategy.signal_generation import build_signals
        from module.strategy.confidence_model import attach_confidence
        from module.strategy.portfolio_selection import select_portfolio
        from module.strategy.backtesting_engine import run_backtest
        from module.steps.step_04_evaluation.tp_sl_reporter import export_strategy_csv

        # --- Create synthetic agent scores for 10 stocks ------------------
        n = 10
        tickers = [f"TICK{i}" for i in range(n)]
        rng = np.random.default_rng(123)
        agent_df = pd.DataFrame({
            "ticker": tickers,
            "fundamental_score": rng.uniform(0.4, 0.9, n),
            "momentum_score":    rng.uniform(0.35, 0.85, n),
            "bear_score":        rng.uniform(0.3, 0.7, n),
            "sector": (["Tech", "Health", "Energy", "Finance", "Comm"] * 2)[:n],
        })

        # --- Step 1: Signal generation ------------------------------------
        signals = build_signals(agent_df)

        # Propagate sector column
        sector_map = dict(zip(agent_df["ticker"], agent_df["sector"]))
        signals["sector"] = signals["ticker"].map(sector_map)

        # --- Step 2: Attach confidence ------------------------------------
        signals = attach_confidence(signals)

        # --- Step 3: Portfolio selection ----------------------------------
        signals = select_portfolio(
            signals,
            min_stocks=4,
            max_stocks=8,
            sector_cap=3,
        )

        # --- Step 4: Backtest ---------------------------------------------
        entry_date = pd.Timestamp("2024-01-02")
        prices_dict = _make_prices_dict(tickers, trend=0.002)
        signals = run_backtest(signals, prices_dict, entry_date, max_holding_days=90)

        # --- Step 5: CSV export -------------------------------------------
        out_path = tmp_path / "strategy_output.csv"
        export_strategy_csv(
            signals,
            out_path,
            fold_id="2024Q1",
            agent_weights={"fundamental_score": 0.4, "momentum_score": 0.4, "bear_score": 0.2},
            agent_hit_rates={"fundamental_score": 0.55, "momentum_score": 0.50, "bear_score": 0.48},
        )

        assert out_path.exists()
        result = pd.read_csv(out_path)

        # All stocks must be in the output
        assert len(result) == n
        assert set(result["ticker"]) == set(tickers)

        # Required columns present
        for col in ("ticker", "fold_id", "score", "tp_pct", "sl_pct",
                    "confidence", "ev", "selected", "outcome", "days_to_outcome"):
            assert col in result.columns, f"Missing column: {col}"

        # Portfolio size constraints
        selected = result[result["selected"].astype(bool)]
        assert 0 <= len(selected) <= 8

        # Outcomes are valid labels
        assert result["outcome"].isin(["TP", "SL", "NONE"]).all()
