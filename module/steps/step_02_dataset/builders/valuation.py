"""Valuation feature builder."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class ValuationFeatureBuilder:
    """Calcula multiplos de valoracion cruzando precio con fundamentales."""

    def build(
        self,
        prices_df: pd.DataFrame,
        fund_snapshot: pd.Series,
        hist_fund: pd.DataFrame,
        as_of: pd.Timestamp,
        analyst_df: Optional[pd.DataFrame] = None,
        recommendation_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        f: Dict = {}

        recent = prices_df[prices_df.index <= as_of]
        if recent.empty:
            return pd.Series(dtype=float)
        cc = "Close" if "Close" in recent.columns else recent.columns[0]
        price = float(recent[cc].iloc[-1])

        shares = fund_snapshot.get("shares_diluted", np.nan)

        eps_ttm = fund_snapshot.get("eps_ttm", np.nan)
        eps_q = fund_snapshot.get("eps", fund_snapshot.get("eps_diluted", np.nan))
        eps = eps_ttm if pd.notna(eps_ttm) else eps_q
        if pd.notna(eps) and eps > 0:
            f["pe_ratio"] = price / eps
            f["earnings_yield"] = eps / price

        equity = fund_snapshot.get("total_equity", np.nan)
        if pd.notna(equity) and pd.notna(shares) and shares > 0 and equity > 0:
            f["pb_ratio"] = price / (equity / shares)

        rev_ttm = fund_snapshot.get("revenue_ttm", np.nan)
        rev_q = fund_snapshot.get("revenue", np.nan)
        rev = rev_ttm if pd.notna(rev_ttm) else (rev_q * 4 if pd.notna(rev_q) else np.nan)
        if pd.notna(rev) and pd.notna(shares) and shares > 0 and rev > 0:
            f["ps_ratio"] = price / (rev / shares)

        mktcap = price * shares if pd.notna(shares) and shares > 0 else np.nan

        ebitda_ttm = fund_snapshot.get("ebitda_ttm", np.nan)
        ebitda_q = fund_snapshot.get("ebitda", np.nan)
        ebitda = ebitda_ttm if pd.notna(ebitda_ttm) else ebitda_q
        debt = fund_snapshot.get("total_debt", np.nan)
        cash = fund_snapshot.get("cash", 0.0) or 0.0
        if all(pd.notna(x) for x in [mktcap, ebitda, debt]) and ebitda > 0:
            ev = mktcap + debt - cash
            f["ev_to_ebitda"] = ev / ebitda

        fcf_ttm = fund_snapshot.get("fcf_ttm", np.nan)
        fcf_q = fund_snapshot.get("fcf", np.nan)
        fcf = fcf_ttm if pd.notna(fcf_ttm) else (fcf_q * 4 if pd.notna(fcf_q) else np.nan)
        if pd.notna(fcf) and pd.notna(mktcap) and mktcap > 0:
            f["fcf_yield"] = fcf / mktcap

        for bf_col, feat_col in [
            ("bf_ev_ebitda", "bf_ev_ebitda"),
            ("bf_fcf_yield", "bf_fcf_yield"),
            ("bf_pe", "bf_pe"),
            ("bf_pb", "bf_pb_annual"),
            ("bf_ps", "bf_ps_ttm"),
        ]:
            val = fund_snapshot.get(bf_col, np.nan)
            if pd.notna(val):
                f[feat_col] = float(val)

        f.update(self._vs_history(f, hist_fund, prices_df, shares))

        if analyst_df is not None and not analyst_df.empty:
            f.update(self._eps_surprise_features(analyst_df, as_of))

        if recommendation_df is not None and not recommendation_df.empty:
            f.update(self._recommendation_features(recommendation_df, as_of))

        return pd.Series(f)

    def _vs_history(self, current: Dict, hist: pd.DataFrame,
                    prices: pd.DataFrame, shares: float) -> Dict:
        out = {}
        cc = "Close" if "Close" in prices.columns else prices.columns[0]
        close_series = prices[cc].sort_index()
        try:
            eps_col = "eps_ttm" if "eps_ttm" in hist.columns else (
                "eps" if "eps" in hist.columns else "eps_diluted"
            )
            if eps_col in hist.columns and "pe_ratio" in current:
                hist_pe = []
                for dt, e in hist[eps_col].dropna().items():
                    if e > 0:
                        p = close_series.asof(dt)
                        if pd.notna(p) and p > 0:
                            hist_pe.append(p / e)
                if len(hist_pe) >= 4:
                    med = float(np.nanmedian(hist_pe))
                    out["pe_vs_5y_median"] = current["pe_ratio"] / med - 1 if med else np.nan

            if "total_equity" in hist.columns and "pb_ratio" in current:
                shr_hist = hist.get("shares_diluted", pd.Series(dtype=float))
                hist_pb = []
                for dt, eq in hist["total_equity"].dropna().items():
                    sh = shr_hist.get(dt, np.nan) if isinstance(shr_hist, pd.Series) else np.nan
                    if pd.notna(sh) and sh > 0 and eq > 0:
                        p = close_series.asof(dt)
                        if pd.notna(p) and p > 0:
                            hist_pb.append(p / (eq / sh))
                if len(hist_pb) >= 4:
                    med = float(np.nanmedian(hist_pb))
                    out["pb_vs_5y_median"] = current.get("pb_ratio", np.nan) / med - 1 if med else np.nan

            ebitda_col = "ebitda_ttm" if "ebitda_ttm" in hist.columns else "ebitda"
            if ebitda_col in hist.columns and "ev_to_ebitda" in current:
                hist_ev = []
                for dt, eb in hist[ebitda_col].dropna().items():
                    if eb > 0:
                        p = close_series.asof(dt)
                        sh = hist["shares_diluted"].get(dt, np.nan) if "shares_diluted" in hist.columns else np.nan
                        d = hist["total_debt"].get(dt, 0) if "total_debt" in hist.columns else 0
                        c = hist["cash"].get(dt, 0) if "cash" in hist.columns else 0
                        if pd.notna(p) and p > 0 and pd.notna(sh) and sh > 0:
                            ev = p * sh + d - c
                            hist_ev.append(ev / eb)
                if len(hist_ev) >= 4:
                    med = float(np.nanmedian(hist_ev))
                    out["ev_ebitda_vs_5y_median"] = current.get("ev_to_ebitda", np.nan) / med - 1 if med else np.nan
        except Exception as e:
            log.debug(f"[ValuationFeat] comparativa historica: {e}")
        return out

    @staticmethod
    def _eps_surprise_features(analyst_df: pd.DataFrame, as_of: pd.Timestamp) -> Dict:
        available = analyst_df[analyst_df.index <= as_of]
        if available.empty:
            return {}
        f: Dict = {}

        last = available.iloc[-1]
        for col in ["eps_surprise_pct", "eps_actual", "eps_estimate"]:
            if col in last.index and pd.notna(last[col]):
                f[col] = float(last[col])

        last4 = available.tail(4)
        if "eps_beat" in last4.columns:
            f["beat_rate_4q"] = float(last4["eps_beat"].mean())
        if "eps_surprise_pct" in last4.columns:
            f["eps_surprise_avg_4q"] = float(last4["eps_surprise_pct"].mean())

        if "eps_estimate" in available.columns and len(available) >= 2:
            e_last = available["eps_estimate"].dropna()
            if len(e_last) >= 2:
                f["eps_revision"] = float(e_last.iloc[-1] / e_last.iloc[-2] - 1) if e_last.iloc[-2] != 0 else np.nan

        return f

    @staticmethod
    def _recommendation_features(rec_df: pd.DataFrame, as_of: pd.Timestamp) -> Dict:
        available = rec_df[rec_df.index <= as_of]
        if available.empty:
            return {}
        f: Dict = {}

        last = available.iloc[-1]
        for col in [
            "analyst_buy_ratio",
            "analyst_bearish_score",
            "analyst_consensus",
            "analyst_dispersion",
            "analyst_strong_buy_pct",
        ]:
            if col in last.index and pd.notna(last[col]):
                f[col] = float(last[col])

        if "analyst_consensus" in available.columns and len(available) >= 2:
            c_now = available["analyst_consensus"].dropna()
            if len(c_now) >= 2:
                f["analyst_consensus_change"] = float(c_now.iloc[-1] - c_now.iloc[-2])

        return f
