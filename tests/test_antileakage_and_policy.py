"""Tests for anti-leakage guarantees and feature policy filter.

Priority coverage areas:
  1. filter_asof prevents future data leakage
  2. assert_no_future_data audit reports
  3. Feature policy: only ratio/normalized features pass
  4. Feature policy: absolute magnitude features are blocked
  5. FeatureSelector basic smoke test
"""

import numpy as np
import pandas as pd
import pytest

from module.common.asof import filter_asof, detect_future_rows, assert_no_future_data
from module.common.feature_policy import is_ratio_or_normalized_feature, filter_ratio_normalized_columns


# ---------------------------------------------------------------------------
# 1. filter_asof
# ---------------------------------------------------------------------------

class TestFilterAsof:
    """filter_asof must exclude all rows with date > as_of."""

    def _make_df(self, dates):
        return pd.DataFrame(
            {"value": range(len(dates))},
            index=pd.to_datetime(dates),
        )

    def test_filters_future_rows(self):
        df = self._make_df(["2020-01-01", "2020-06-01", "2021-01-01", "2021-06-01"])
        result = filter_asof(df, pd.Timestamp("2020-06-01"))
        assert len(result) == 2
        assert result.index.max() <= pd.Timestamp("2020-06-01")

    def test_returns_all_when_all_past(self):
        df = self._make_df(["2019-01-01", "2019-06-01"])
        result = filter_asof(df, pd.Timestamp("2020-01-01"))
        assert len(result) == 2

    def test_returns_empty_when_all_future(self):
        df = self._make_df(["2025-01-01", "2025-06-01"])
        result = filter_asof(df, pd.Timestamp("2020-01-01"))
        assert len(result) == 0

    def test_handles_none(self):
        result = filter_asof(None, pd.Timestamp("2020-01-01"))
        assert result is None

    def test_handles_empty_df(self):
        df = pd.DataFrame()
        result = filter_asof(df, pd.Timestamp("2020-01-01"))
        assert result.empty

    def test_with_date_column(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]),
            "value": [1, 2, 3],
        })
        result = filter_asof(df, pd.Timestamp("2020-06-01"), date_col="date")
        assert len(result) == 2

    def test_boundary_exact_match_included(self):
        """Row with date == as_of must be included (<=, not <)."""
        df = self._make_df(["2020-06-01"])
        result = filter_asof(df, pd.Timestamp("2020-06-01"))
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 2. detect_future_rows / assert_no_future_data
# ---------------------------------------------------------------------------

class TestDetectFutureRows:
    def test_detects_future(self):
        df = pd.DataFrame(
            {"v": [1, 2, 3]},
            index=pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
        )
        n, max_date = detect_future_rows(df, pd.Timestamp("2020-06-01"))
        assert n == 2
        assert max_date == pd.Timestamp("2022-01-01")

    def test_no_future(self):
        df = pd.DataFrame(
            {"v": [1, 2]},
            index=pd.to_datetime(["2019-01-01", "2020-01-01"]),
        )
        n, max_date = detect_future_rows(df, pd.Timestamp("2020-06-01"))
        assert n == 0
        assert max_date is None

    def test_assert_returns_ok(self):
        df = pd.DataFrame(
            {"v": [1]},
            index=pd.to_datetime(["2019-01-01"]),
        )
        result = assert_no_future_data(df, pd.Timestamp("2020-01-01"), context="test")
        assert result["ok"] is True
        assert result["n_rows_future_detected"] == 0

    def test_assert_detects_leak(self):
        df = pd.DataFrame(
            {"v": [1, 2]},
            index=pd.to_datetime(["2019-01-01", "2025-01-01"]),
        )
        result = assert_no_future_data(df, pd.Timestamp("2020-01-01"), context="test")
        assert result["ok"] is False
        assert result["n_rows_future_detected"] == 1


# ---------------------------------------------------------------------------
# 3. Feature policy — ratio/normalized features pass
# ---------------------------------------------------------------------------

class TestFeaturePolicyAllowed:
    """Features that are ratios or normalized must pass the filter."""

    @pytest.mark.parametrize("col", [
        # These features contain allowed tokens in the feature policy
        "net_margin", "gross_margin", "fcf_margin",
        "current_ratio", "pe_ratio",
        "revenue_yoy_growth", "eps_yoy_growth",
        "rsi_14", "macd", "momentum_3m", "volatility_20d",
        "analyst_buy_ratio", "piotroski_fscore",
        "accruals_ratio", "interest_coverage",
        "bb_pct", "price_vs_52w_high",
        "eps_surprise_pct", "eps_revision",
        "insider_net_ratio_90d", "insider_sell_ratio",
        "debt_to_ebitda",
    ])
    def test_ratio_features_pass(self, col):
        assert is_ratio_or_normalized_feature(col) is True

    @pytest.mark.parametrize("col", [
        # roe, roa, debt_equity don't contain allowed tokens by name alone,
        # but pass through agent-level FEATURE_COLUMNS configuration.
        # The policy function returns False for bare names without tokens.
        "roe", "roa", "debt_equity",
    ])
    def test_features_needing_agent_config(self, col):
        """Features that pass through agent config, not the generic policy filter."""
        assert is_ratio_or_normalized_feature(col) is False


# ---------------------------------------------------------------------------
# 4. Feature policy — absolute magnitude features blocked
# ---------------------------------------------------------------------------

class TestFeaturePolicyBlocked:
    """Absolute magnitude features must be blocked."""

    @pytest.mark.parametrize("col", [
        "revenue", "net_income", "total_assets",
        "total_debt", "shares", "market_cap",
        "cash", "operating_income",
    ])
    def test_absolute_features_blocked(self, col):
        assert is_ratio_or_normalized_feature(col) is False


# ---------------------------------------------------------------------------
# 5. filter_ratio_normalized_columns
# ---------------------------------------------------------------------------

class TestFilterRatioNormalizedColumns:
    def test_keeps_ratio_columns(self):
        df = pd.DataFrame({
            "current_ratio": [1.5, 2.0],
            "revenue": [1e9, 2e9],
            "net_margin": [0.05, 0.10],
            "total_assets": [5e9, 6e9],
        })
        result = filter_ratio_normalized_columns(df)
        assert "current_ratio" in result.columns
        assert "net_margin" in result.columns
        assert "revenue" not in result.columns
        assert "total_assets" not in result.columns

    def test_handles_empty(self):
        df = pd.DataFrame()
        result = filter_ratio_normalized_columns(df)
        assert result.empty

    def test_binary_columns_pass(self):
        """Binary 0/1 columns should be treated as normalized."""
        df = pd.DataFrame({
            "some_flag": [0.0, 1.0, 0.0, 1.0],
        })
        result = filter_ratio_normalized_columns(df)
        assert "some_flag" in result.columns


# ---------------------------------------------------------------------------
# 6. Feature controls — resolve_feature_columns
# ---------------------------------------------------------------------------

class TestResolveFeatureColumns:
    def test_resolves_include(self):
        import logging
        from module.common.feature_controls import resolve_feature_columns
        result = resolve_feature_columns(
            default_cols=[],
            available_cols=["roe", "roa", "revenue", "net_margin"],
            include_cols=["roe", "roa"],
            exclude_cols=[],
            logger=logging.getLogger("test"),
            owner="test",
        )
        assert result == ["roe", "roa"]

    def test_missing_returns_subset(self):
        import logging
        from module.common.feature_controls import resolve_feature_columns
        result = resolve_feature_columns(
            default_cols=[],
            available_cols=["roe"],
            include_cols=["roe", "roa"],
            exclude_cols=[],
            logger=logging.getLogger("test"),
            owner="test",
        )
        assert result == ["roe"]


# ---------------------------------------------------------------------------
# 7. Temporal ordering in OOF (smoke test)
# ---------------------------------------------------------------------------

class TestOOFTemporalOrder:
    """OOF splits must respect temporal ordering."""

    def test_timeseries_split_ordering(self):
        from sklearn.model_selection import TimeSeriesSplit

        n = 100
        dates = pd.date_range("2015-01-01", periods=n, freq="QE")
        X = pd.DataFrame({"f1": np.random.randn(n)}, index=dates)

        tss = TimeSeriesSplit(n_splits=3)
        for train_idx, val_idx in tss.split(X):
            train_max = X.index[train_idx].max()
            val_min = X.index[val_idx].min()
            assert train_max < val_min, "Train data must precede validation data"


# ---------------------------------------------------------------------------
# 6. DataRouter ticker validation (path traversal prevention)
# ---------------------------------------------------------------------------

class TestDataRouterTickerValidation:
    """DataRouter must reject tickers that could cause path traversal."""

    def _make_router(self, tmp_path):
        from module.common.data_router import DataRouter
        return DataRouter(str(tmp_path))

    def test_valid_tickers_accepted(self, tmp_path):
        router = self._make_router(tmp_path)
        for ticker in ["AAPL", "BRK-B", "BF-B", "GOOG", "META"]:
            assert router._validate_ticker(ticker) == ticker

    def test_path_traversal_rejected(self, tmp_path):
        router = self._make_router(tmp_path)
        with pytest.raises(ValueError, match="[Ii]nvalid"):
            router._validate_ticker("../../etc/passwd")

    def test_empty_ticker_rejected(self, tmp_path):
        router = self._make_router(tmp_path)
        with pytest.raises(ValueError, match="[Ii]nvalid"):
            router._validate_ticker("")

    def test_long_ticker_rejected(self, tmp_path):
        router = self._make_router(tmp_path)
        with pytest.raises(ValueError, match="[Ii]nvalid"):
            router._validate_ticker("A" * 20)

    def test_special_chars_rejected(self, tmp_path):
        router = self._make_router(tmp_path)
        for bad in ["AAPL;rm -rf", "AAPL/../..", "ticker/../../"]:
            with pytest.raises(ValueError, match="[Ii]nvalid"):
                router._validate_ticker(bad)
