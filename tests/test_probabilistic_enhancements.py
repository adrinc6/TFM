"""Tests for the probabilistic multi-agent system enhancements.

Coverage:
    1. brier_score                    – perfect, random, and constant predictors
    2. tp_sl_classification_metrics   – accuracy, precision, recall, F1, edge cases
    3. compute_all_metrics            – with and without classification metrics
    4. VolatilityRegimeTpSlLearner    – fit/predict with 3 clusters
    5. BaseAgent.fit_calibrator       – isotonic + platt, edge cases
    6. BaseAgent._apply_calibration   – applies calibrator / no-op without one
    7. get_portfolio_weights           – confidence-proportional + max_weight cap
    8. apply_regime_exposure           – BULL/NEUTRAL/BEAR multipliers
    9. get_regime_exposure_multiplier  – all three regime labels + unknown fallback
   10. build_tp_sl_targets             – with VolatilityRegimeTpSlLearner learner
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_multiindex_df(tickers=None, date="2024-03-31"):
    tickers = tickers or ["AAA", "BBB", "CCC"]
    idx = pd.MultiIndex.from_tuples(
        [(t, pd.Timestamp(date)) for t in tickers],
        names=["ticker", "date"],
    )
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "volatility_60d": rng.uniform(0.01, 0.05, len(tickers)),
            "forward_return": rng.uniform(-0.15, 0.20, len(tickers)),
            "snapshot_date": pd.Timestamp(date),
        },
        index=idx,
    )


def _make_price_series(ticker, start, days, tp_hit=True):
    """Create a price series that hits TP (if tp_hit=True) or SL."""
    dates = pd.bdate_range(start, periods=days)
    prices = [100.0] * days
    if tp_hit and days > 2:
        prices[2] = 109.0   # hits 8% TP on day 2
    elif not tp_hit and days > 2:
        prices[2] = 94.0    # hits 5% SL on day 2
    return pd.Series(prices, index=dates, name="Close")


# ===========================================================================
# 1–3. Performance metrics
# ===========================================================================

class TestBrierScore:
    def test_perfect_predictor_is_zero(self):
        from module.common.performance_metrics import brier_score
        y_true = pd.Series([1, 0, 1, 0])
        y_prob = pd.Series([1.0, 0.0, 1.0, 0.0])
        assert brier_score(y_true, y_prob) == pytest.approx(0.0)

    def test_random_predictor_near_quarter(self):
        from module.common.performance_metrics import brier_score
        rng = np.random.default_rng(42)
        y_true = pd.Series(rng.integers(0, 2, 1000).astype(float))
        y_prob = pd.Series(np.full(1000, 0.5))
        bs = brier_score(y_true, y_prob)
        assert abs(bs - 0.25) < 0.02

    def test_empty_series_returns_nan(self):
        from module.common.performance_metrics import brier_score
        assert np.isnan(brier_score(pd.Series(dtype=float), pd.Series(dtype=float)))

    def test_worst_predictor(self):
        from module.common.performance_metrics import brier_score
        y_true = pd.Series([1.0, 1.0])
        y_prob = pd.Series([0.0, 0.0])
        assert brier_score(y_true, y_prob) == pytest.approx(1.0)


class TestTpSlClassificationMetrics:
    def test_perfect_classifier(self):
        from module.common.performance_metrics import tp_sl_classification_metrics
        y_true = pd.Series([1, 0, 1, 0, 1])
        y_prob = pd.Series([0.9, 0.1, 0.8, 0.2, 0.7])
        m = tp_sl_classification_metrics(y_true, y_prob)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["n_samples"] == 5
        assert m["n_tp"] == 3
        assert m["n_sl"] == 2

    def test_empty_series(self):
        from module.common.performance_metrics import tp_sl_classification_metrics
        m = tp_sl_classification_metrics(pd.Series(dtype=float), pd.Series(dtype=float))
        assert m["n_samples"] == 0
        assert np.isnan(m["accuracy"])

    def test_brier_score_embedded(self):
        from module.common.performance_metrics import tp_sl_classification_metrics
        y_true = pd.Series([1.0, 0.0])
        y_prob = pd.Series([1.0, 0.0])
        m = tp_sl_classification_metrics(y_true, y_prob)
        assert m["brier_score"] == pytest.approx(0.0)


class TestComputeAllMetricsWithClassification:
    def test_includes_classification_when_provided(self):
        from module.common.performance_metrics import compute_all_metrics
        returns = pd.Series([0.01, -0.005, 0.02])
        y_true = pd.Series([1, 0, 1])
        y_prob = pd.Series([0.8, 0.2, 0.7])
        m = compute_all_metrics(returns, y_true=y_true, y_prob=y_prob)
        assert "strategy_brier_score" in m
        assert "strategy_accuracy" in m
        assert "strategy_recall" in m

    def test_no_classification_without_args(self):
        from module.common.performance_metrics import compute_all_metrics
        returns = pd.Series([0.01, -0.005, 0.02])
        m = compute_all_metrics(returns)
        assert "strategy_brier_score" not in m
        assert "strategy_sharpe" in m


# ===========================================================================
# 4. VolatilityRegimeTpSlLearner
# ===========================================================================

class TestVolatilityRegimeTpSlLearner:
    def test_fit_and_predict_shape(self):
        from module.common.target_engineering import VolatilityRegimeTpSlLearner
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "volatility_60d": rng.uniform(0.005, 0.06, 300),
            "forward_return": rng.uniform(-0.20, 0.25, 300),
        })
        learner = VolatilityRegimeTpSlLearner(n_clusters=3, min_samples_per_cluster=10)
        learner.fit(df)
        assert learner.is_fitted

        test_df = pd.DataFrame({"volatility_60d": [0.01, 0.03, 0.055]})
        tp, sl = learner.predict(test_df)
        assert len(tp) == 3
        assert len(sl) == 3
        assert (tp >= 0.02).all()
        assert (tp <= 0.25).all()
        assert (sl >= 0.01).all()
        assert (sl <= 0.15).all()

    def test_higher_vol_cluster_gets_wider_levels(self):
        from module.common.target_engineering import VolatilityRegimeTpSlLearner
        rng = np.random.default_rng(2)
        # Low-vol stocks have small swings; high-vol stocks have large swings
        n = 200
        vol_low = rng.uniform(0.005, 0.015, n)
        ret_low = rng.uniform(-0.03, 0.05, n)
        vol_high = rng.uniform(0.05, 0.10, n)
        ret_high = rng.uniform(-0.15, 0.25, n)
        df = pd.DataFrame({
            "volatility_60d": np.concatenate([vol_low, vol_high]),
            "forward_return": np.concatenate([ret_low, ret_high]),
        })
        learner = VolatilityRegimeTpSlLearner(n_clusters=2, min_samples_per_cluster=10)
        learner.fit(df)
        tp_low, _ = learner.predict(pd.DataFrame({"volatility_60d": [0.01]}))
        tp_high, _ = learner.predict(pd.DataFrame({"volatility_60d": [0.08]}))
        # Higher vol regime should have >= TP level
        assert float(tp_high.iloc[0]) >= float(tp_low.iloc[0])

    def test_predict_without_fit_returns_defaults(self):
        from module.common.target_engineering import VolatilityRegimeTpSlLearner
        learner = VolatilityRegimeTpSlLearner()
        df = pd.DataFrame({"volatility_60d": [0.02]})
        tp, sl = learner.predict(df)
        assert len(tp) == 1

    def test_summary_contains_global(self):
        from module.common.target_engineering import VolatilityRegimeTpSlLearner
        rng = np.random.default_rng(3)
        df = pd.DataFrame({
            "volatility_60d": rng.uniform(0.01, 0.05, 100),
            "forward_return": rng.uniform(-0.10, 0.15, 100),
        })
        learner = VolatilityRegimeTpSlLearner(n_clusters=2)
        learner.fit(df)
        summary = learner.summary()
        assert "_global" in summary

    def test_empty_df_does_not_raise(self):
        from module.common.target_engineering import VolatilityRegimeTpSlLearner
        learner = VolatilityRegimeTpSlLearner()
        learner.fit(pd.DataFrame())
        assert not learner.is_fitted


# ===========================================================================
# 5–6. BaseAgent calibration
# ===========================================================================

class TestBaseAgentCalibration:
    """Use a minimal concrete subclass to test BaseAgent calibration API."""

    def _make_agent(self, tmp_path):
        from module.agents.base import BaseAgent

        class _DummyAgent(BaseAgent):
            def fit(self, X, y, **kwargs):
                self.is_trained = True
                return self

            def predict_score(self, X):
                return pd.Series(0.5, index=X.index)

        return _DummyAgent(name="test", results_dir=str(tmp_path), save_artifacts=False)

    def test_no_calibrator_returns_input_unchanged(self, tmp_path):
        agent = self._make_agent(tmp_path)
        arr = np.array([0.3, 0.7, 0.5])
        out = agent._apply_calibration(arr)
        np.testing.assert_array_equal(out, arr)

    def test_isotonic_calibration_fits_and_applies(self, tmp_path):
        pytest.importorskip("sklearn")
        agent = self._make_agent(tmp_path)
        rng = np.random.default_rng(42)
        oof_proba = rng.uniform(0, 1, 200)
        # Ground truth: high probability → more likely TP
        oof_labels = (oof_proba + rng.normal(0, 0.2, 200) > 0.5).astype(float)
        agent.fit_calibrator(oof_proba, oof_labels, method="isotonic")
        assert agent._calibration_method == "isotonic"
        assert agent._calibrator is not None
        out = agent._apply_calibration(np.array([0.2, 0.5, 0.8]))
        assert out.shape == (3,)
        assert ((out >= 0.0) & (out <= 1.0)).all()

    def test_platt_calibration_fits_and_applies(self, tmp_path):
        pytest.importorskip("sklearn")
        agent = self._make_agent(tmp_path)
        rng = np.random.default_rng(7)
        oof_proba = rng.uniform(0, 1, 200)
        oof_labels = (oof_proba > 0.5).astype(float)
        agent.fit_calibrator(oof_proba, oof_labels, method="platt")
        assert agent._calibration_method == "platt"
        out = agent._apply_calibration(np.array([0.1, 0.9]))
        assert len(out) == 2

    def test_insufficient_samples_skips_calibration(self, tmp_path):
        pytest.importorskip("sklearn")
        agent = self._make_agent(tmp_path)
        agent.fit_calibrator(np.array([0.5, 0.6]), np.array([1.0, 1.0]))
        assert agent._calibrator is None

    def test_single_class_labels_skips_calibration(self, tmp_path):
        pytest.importorskip("sklearn")
        agent = self._make_agent(tmp_path)
        agent.fit_calibrator(np.ones(50) * 0.5, np.ones(50))
        assert agent._calibrator is None

    def test_unknown_method_skips_calibration(self, tmp_path):
        pytest.importorskip("sklearn")
        agent = self._make_agent(tmp_path)
        rng = np.random.default_rng(0)
        proba = rng.uniform(0, 1, 100)
        labels = (proba > 0.5).astype(float)
        agent.fit_calibrator(proba, labels, method="unknown_method")
        assert agent._calibrator is None


# ===========================================================================
# 7. get_portfolio_weights — confidence-proportional + max_weight cap
# ===========================================================================

class TestGetPortfolioWeights:
    def _make_signals(self, confidences):
        tickers = [f"T{i}" for i in range(len(confidences))]
        return pd.DataFrame({
            "ticker": tickers,
            "confidence": confidences,
            "selected": [True] * len(confidences),
        })

    def test_proportional_to_confidence(self):
        from module.steps.step_04_evaluation.strategy import get_portfolio_weights
        signals = self._make_signals([0.6, 0.3, 0.1])
        w = get_portfolio_weights(signals, weight_by="confidence")
        assert w.sum() == pytest.approx(1.0)
        # T0 should have double weight of T1
        assert w["T0"] == pytest.approx(0.6)
        assert w["T1"] == pytest.approx(0.3)
        assert w["T2"] == pytest.approx(0.1)

    def test_max_weight_cap_applied(self):
        from module.steps.step_04_evaluation.strategy import get_portfolio_weights
        signals = self._make_signals([0.9, 0.05, 0.05])
        w = get_portfolio_weights(signals, weight_by="confidence", max_weight=0.5)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert w.max() <= 0.5 + 1e-9

    def test_equal_weights_when_column_missing(self):
        from module.steps.step_04_evaluation.strategy import get_portfolio_weights
        signals = self._make_signals([0.5, 0.5])
        signals = signals.drop(columns=["confidence"])
        w = get_portfolio_weights(signals, weight_by="confidence")
        assert w.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(w.values, [0.5, 0.5])

    def test_empty_signals(self):
        from module.steps.step_04_evaluation.strategy import get_portfolio_weights
        w = get_portfolio_weights(pd.DataFrame())
        assert len(w) == 0

    def test_selected_only_filters_rows(self):
        from module.steps.step_04_evaluation.strategy import get_portfolio_weights
        signals = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "confidence": [0.6, 0.3, 0.1],
            "selected": [True, True, False],
        })
        w = get_portfolio_weights(signals, weight_by="confidence", selected_only=True)
        assert "C" not in w.index
        assert w.sum() == pytest.approx(1.0)


# ===========================================================================
# 8. apply_regime_exposure
# ===========================================================================

class TestApplyRegimeExposure:
    def _make_weights(self):
        return pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})

    def test_bull_full_exposure(self):
        from module.steps.step_04_evaluation.strategy import apply_regime_exposure
        from module.common.regime import REGIME_RISK_ON
        w = self._make_weights()
        result = apply_regime_exposure(w, REGIME_RISK_ON)
        # No cash in bull mode
        assert "_CASH" not in result.index
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_bear_minimal_exposure_adds_cash(self):
        from module.steps.step_04_evaluation.strategy import apply_regime_exposure
        from module.common.regime import REGIME_RISK_OFF
        w = self._make_weights()
        result = apply_regime_exposure(w, REGIME_RISK_OFF)
        assert "_CASH" in result.index
        assert result["_CASH"] > 0.5   # mostly cash in bear
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_neutral_partial_exposure(self):
        from module.steps.step_04_evaluation.strategy import apply_regime_exposure
        from module.common.regime import REGIME_NEUTRAL
        w = self._make_weights()
        result = apply_regime_exposure(w, REGIME_NEUTRAL)
        invested = result[result.index != "_CASH"].sum()
        assert 0.50 < float(invested) < 0.85

    def test_sum_always_one(self):
        from module.steps.step_04_evaluation.strategy import apply_regime_exposure
        from module.common.regime import REGIME_RISK_ON, REGIME_NEUTRAL, REGIME_RISK_OFF
        w = self._make_weights()
        for regime in [REGIME_RISK_ON, REGIME_NEUTRAL, REGIME_RISK_OFF]:
            assert apply_regime_exposure(w, regime).sum() == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# 9. get_regime_exposure_multiplier
# ===========================================================================

class TestRegimeExposureMultiplier:
    def test_bull_is_one(self):
        from module.common.regime import get_regime_exposure_multiplier, REGIME_RISK_ON
        assert get_regime_exposure_multiplier(REGIME_RISK_ON) == pytest.approx(1.0)

    def test_bear_is_low(self):
        from module.common.regime import get_regime_exposure_multiplier, REGIME_RISK_OFF
        mult = get_regime_exposure_multiplier(REGIME_RISK_OFF)
        assert mult < 0.5

    def test_neutral_is_between(self):
        from module.common.regime import get_regime_exposure_multiplier, REGIME_NEUTRAL, REGIME_RISK_ON, REGIME_RISK_OFF
        mult_bull = get_regime_exposure_multiplier(REGIME_RISK_ON)
        mult_bear = get_regime_exposure_multiplier(REGIME_RISK_OFF)
        mult_neutral = get_regime_exposure_multiplier(REGIME_NEUTRAL)
        assert mult_bear < mult_neutral < mult_bull

    def test_unknown_regime_falls_back_to_neutral(self):
        from module.common.regime import get_regime_exposure_multiplier, REGIME_NEUTRAL
        unknown = get_regime_exposure_multiplier("UNKNOWN_REGIME")
        neutral = get_regime_exposure_multiplier(REGIME_NEUTRAL)
        assert unknown == pytest.approx(neutral)

    def test_bull_alias_works(self):
        from module.common.regime import get_regime_exposure_multiplier, REGIME_BULL, REGIME_RISK_ON
        assert REGIME_BULL == REGIME_RISK_ON
        assert get_regime_exposure_multiplier(REGIME_BULL) == pytest.approx(1.0)


# ===========================================================================
# 10. build_tp_sl_targets with VolatilityRegimeTpSlLearner
# ===========================================================================

class TestBuildTpSlTargetsWithLearner:
    def test_learner_overrides_default_levels(self):
        from module.common.target_engineering import (
            build_tp_sl_targets,
            VolatilityRegimeTpSlLearner,
        )
        rng = np.random.default_rng(5)
        df = pd.DataFrame({
            "volatility_60d": rng.uniform(0.01, 0.04, 100),
            "forward_return": rng.uniform(-0.10, 0.15, 100),
        })
        learner = VolatilityRegimeTpSlLearner(n_clusters=2, min_samples_per_cluster=10)
        learner.fit(df)

        idx = pd.MultiIndex.from_tuples(
            [("AAA", pd.Timestamp("2024-03-31"))],
            names=["ticker", "date"],
        )
        test_df = pd.DataFrame(
            {"snapshot_date": [pd.Timestamp("2024-04-01")], "volatility_60d": [0.02]},
            index=idx,
        )
        prices = pd.Series(
            [100.0, 110.0, 115.0, 120.0],
            index=pd.to_datetime(["2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04"]),
        )
        bundle = build_tp_sl_targets(
            test_df,
            prices_dict={"AAA": prices},
            lag_days=0,
            max_holding_days=10,
            tp_sl_learner=learner,
        )
        assert len(bundle.tp_level) == 1
        assert len(bundle.sl_level) == 1
        assert 0.02 <= float(bundle.tp_level.iloc[0]) <= 0.25
