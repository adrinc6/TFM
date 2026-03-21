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
        """
        Enriquece el DataFrame de fundamentales trimestrales con ratios y señales.
        NO incluye trend_features aquí porque expanding() sobre todo el df
        causaría look-ahead: una fila de 2021 vería tendencias calculadas con
        datos de 2023. Las trend features se calculan en snapshot_trends(),
        llamado desde dataset.py con solo los datos hasta as_of.
        """
        df = df.copy()
        df = self._yoy_growth(df)
        df = self._quality_metrics(df)
        df = self._coverage_ratios(df)
        df = self._risk_flags(df)
        return df

    def snapshot_trends(self, fund_hist_asof: pd.DataFrame) -> Dict:
        """
        Calcula features de tendencia usando SOLO los datos hasta as_of.
        Llamar con fund_hist_asof = fund_enriched[fund_enriched.index <= as_of].
        Devuelve un dict de features listos para añadir al record.
        """
        out: Dict = {}
        for col, feat in [
            ("bf_roe",         "roe_trend_3y"),
            ("bf_gross_margin","gross_margin_trend_3y"),
            ("bf_net_margin",  "net_margin_trend_3y"),
            ("roe",            "roe_trend_2y"),
            ("net_margin",     "net_margin_trend_2y"),
        ]:
            if col in fund_hist_asof.columns:
                vals = fund_hist_asof[col].dropna().tail(8)
                out[feat] = self._slope(vals)
        return out

    @staticmethod
    def _slope(series: pd.Series, n: int = 8) -> float:
        vals = series.tail(n)
        if len(vals) < 3:
            return np.nan
        x = np.arange(len(vals), dtype=float)
        y = vals.values.astype(float)
        if y.std() == 0:
            return 0.0
        coeffs = np.polyfit(x, y / (abs(y.mean()) + 1e-10), 1)
        return float(coeffs[0])

    def _yoy_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        pairs = {
            "revenue_yoy_growth": "revenue",
            "net_income_yoy_growth": "net_income",
            "operating_income_yoy_growth": "operating_income",
            "fcf_yoy_growth": "fcf",
            "eps_yoy_growth": "eps",
            "total_debt_yoy_growth": "total_debt",
            # Cambios YoY de ratios (usados en Piotroski y como features directos)
            "roa_change_yoy": "roa",
            "gross_margin_change_yoy": "gross_margin",
            "current_ratio_change_yoy": "current_ratio",
        }
        for feat, col in pairs.items():
            if col in df.columns:
                df[feat] = df[col].diff(periods=4)
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

        # ── Piotroski F-score (Piotroski 2000) ───────────────────────────────
        # 8 señales binarias normalizadas a [0,1]. Score alto = empresa sana.
        # Fuente: "Value Investing: The Use of Historical Financial Statement
        # Information to Separate Winners from Losers" (Piotroski, JAR 2000).
        piotroski = pd.Series(0.0, index=df.index)
        n_signals = 0

        # Rentabilidad
        if "roa" in df.columns:
            piotroski += (df["roa"] > 0).astype(float)           # F1: ROA positivo
            n_signals += 1
            piotroski += (df["roa"].diff(4) > 0).astype(float)   # F3: ROA mejorando YoY
            n_signals += 1
        if "operating_cash_flow" in df.columns:
            piotroski += (df["operating_cash_flow"] > 0).astype(float)  # F2: CFO positivo
            n_signals += 1
        # F4: calidad de beneficios (CFO > Net Income → accruals bajos)
        if "accruals_ratio" in df.columns:
            piotroski += (df["accruals_ratio"] < 0).astype(float)
            n_signals += 1

        # Apalancamiento / liquidez
        if "total_debt_yoy_growth" in df.columns:
            piotroski += (df["total_debt_yoy_growth"] < 0.05).astype(float)  # F5: deuda no crece
            n_signals += 1
        if "current_ratio" in df.columns:
            piotroski += (df["current_ratio"].diff(4) > 0).astype(float)     # F6: liquidez mejora
            n_signals += 1

        # Eficiencia operativa
        if "gross_margin" in df.columns:
            piotroski += (df["gross_margin"].diff(4) > 0).astype(float)  # F7: margen bruto mejora
            n_signals += 1
        if "revenue_yoy_growth" in df.columns:
            piotroski += (df["revenue_yoy_growth"] > 0).astype(float)    # F8: ingresos crecen
            n_signals += 1

        if n_signals > 0:
            df["piotroski_fscore"] = piotroski / n_signals

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

