"""Fundamental feature builder."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class FundamentalFeatureBuilder:
    """
    Enriquece el DataFrame consolidado con ratios y senales adicionales.
    """

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._yoy_growth(df)
        df = self._quality_metrics(df)
        df = self._coverage_ratios(df)
        df = self._risk_flags(df)
        df = self._trend_features(df)
        return df

    def _yoy_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        pairs = {
            "revenue_yoy_growth": "revenue",
            "net_income_yoy_growth": "net_income",
            "operating_income_yoy_growth": "operating_income",
            "fcf_yoy_growth": "fcf",
            "eps_yoy_growth": "eps",
            "total_debt_yoy_growth": "total_debt",
        }
        for feat, col in pairs.items():
            if col in df.columns:
                df[feat] = df[col].pct_change(periods=4)
        return df

    def _quality_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        if "accruals_ratio" not in df.columns:
            if {"net_income", "operating_cash_flow", "total_assets"}.issubset(df.columns):
                df["accruals_ratio"] = (
                    (df["net_income"] - df["operating_cash_flow"]) / df["total_assets"].replace(0, np.nan)
                )

        if "capex" in df.columns and "revenue" in df.columns:
            df["capex_to_revenue"] = df["capex"].abs() / df["revenue"].replace(0, np.nan)

        if "fcf" in df.columns and "net_income" in df.columns:
            df["earnings_quality"] = df["fcf"] / df["net_income"].replace(0, np.nan)

        if "ebitda_margin" not in df.columns:
            if "ebitda" in df.columns and "revenue" in df.columns:
                df["ebitda_margin"] = df["ebitda"] / df["revenue"].replace(0, np.nan)

        if "operating_margin" not in df.columns:
            if "operating_income" in df.columns and "revenue" in df.columns:
                df["operating_margin"] = df["operating_income"] / df["revenue"].replace(0, np.nan)

        return df

    def _coverage_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        if "interest_coverage" not in df.columns:
            if "operating_income" in df.columns and "interest_expense" in df.columns:
                ie = df["interest_expense"].replace(0, np.nan).abs()
                df["interest_coverage"] = df["operating_income"] / ie
        return df

    def _risk_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        if "net_income" in df.columns:
            losses = (df["net_income"] < 0).astype(int)
            groups = (losses != losses.shift()).cumsum()
            df["consecutive_losses"] = losses * (losses.groupby(groups).cumcount() + 1)
        if "revenue_yoy_growth" in df.columns:
            df["revenue_decline"] = (df["revenue_yoy_growth"] < 0).astype(int)
        return df

    def _trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        def _slope(series: pd.Series, n: int = 8) -> float:
            vals = series.dropna().tail(n)
            if len(vals) < 3:
                return np.nan
            x = np.arange(len(vals), dtype=float)
            y = vals.values.astype(float)
            if y.std() == 0:
                return 0.0
            coeffs = np.polyfit(x, y / (abs(y.mean()) + 1e-10), 1)
            return float(coeffs[0])

        for col, feat in [
            ("bf_roe", "roe_trend_3y"),
            ("bf_gross_margin", "gross_margin_trend_3y"),
            ("bf_net_margin", "net_margin_trend_3y"),
            ("roe", "roe_trend_2y"),
            ("net_margin", "net_margin_trend_2y"),
        ]:
            if col in df.columns:
                df[feat] = df[col].expanding(3).apply(lambda s: _slope(s, n=8), raw=False)

        return df
