"""Sentiment feature builder."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from module.common.asof import filter_asof


class SentimentFeatureBuilder:
    """Construye el vector de features para el SentimentAgent."""

    def build(
        self,
        recommendation_df: Optional[pd.DataFrame],
        mspr_df: Optional[pd.DataFrame],
        insider_df: Optional[pd.DataFrame],
        eps_df: Optional[pd.DataFrame],
        as_of: pd.Timestamp,
    ) -> pd.Series:
        f: Dict = {}

        if recommendation_df is not None and not recommendation_df.empty:
            available = filter_asof(recommendation_df, as_of=as_of)
            if not available.empty:
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
                    c = available["analyst_consensus"].dropna()
                    if len(c) >= 2:
                        f["analyst_consensus_change"] = float(c.iloc[-1] - c.iloc[-2])

        if mspr_df is not None and not mspr_df.empty and "mspr" in mspr_df.columns:
            available_mspr = filter_asof(mspr_df, as_of=as_of)
            if not available_mspr.empty:
                mspr_vals = available_mspr["mspr"].dropna()
                if len(mspr_vals) >= 1:
                    f["mspr_3m"] = float(mspr_vals.tail(3).mean())
                if len(mspr_vals) >= 3:
                    f["mspr_trend"] = float(mspr_vals.iloc[-1] - mspr_vals.iloc[-3])

        if insider_df is not None and not insider_df.empty:
            available_insider = filter_asof(insider_df, as_of=as_of, date_col="date")
            # Ventana rolling de 90 días sin mirar el futuro.
            if not available_insider.empty and "date" in available_insider.columns:
                start = pd.Timestamp(as_of) - pd.Timedelta(days=90)
                available_insider = available_insider.loc[
                    pd.to_datetime(available_insider["date"], errors="coerce") >= start
                ]

            buys = float(available_insider.loc[available_insider["is_buy"] == 1, "shares"].sum()) if not available_insider.empty else 0.0
            sells = float(available_insider.loc[available_insider["is_sell"] == 1, "shares"].sum()) if not available_insider.empty else 0.0
            total = buys + sells
            f["insider_net_ratio_90d"] = (buys - sells) / total if total > 0 else 0.0
            f["insider_sell_ratio"] = sells / total if total > 0 else 0.5

        if eps_df is not None and not eps_df.empty:
            available_eps = filter_asof(eps_df, as_of=as_of)
            if not available_eps.empty:
                last = available_eps.iloc[-1]
                if "eps_estimate" in last.index and pd.notna(last["eps_estimate"]):
                    f["eps_est"] = float(last["eps_estimate"])
                if "eps_actual" in last.index and pd.notna(last["eps_actual"]):
                    f["eps_reported"] = float(last["eps_actual"])
                if "eps_surprise_pct" in last.index and pd.notna(last["eps_surprise_pct"]):
                    f["eps_surprise_pct"] = float(last["eps_surprise_pct"])

                est_series = available_eps.get("eps_estimate")
                if est_series is not None:
                    est_series = est_series.dropna()
                    if len(est_series) >= 2 and est_series.iloc[-2] != 0:
                        f["eps_revision"] = float((est_series.iloc[-1] - est_series.iloc[-2]) / abs(est_series.iloc[-2]))
                    elif len(est_series) >= 2:
                        f["eps_revision"] = float(est_series.iloc[-1] - est_series.iloc[-2])

                last4 = available_eps.tail(4)
                if "eps_beat" in last4.columns:
                    f["beat_rate_4q"] = float(last4["eps_beat"].mean())
                if "eps_surprise_pct" in last4.columns:
                    f["eps_surprise_avg_4q"] = float(last4["eps_surprise_pct"].mean())

        return pd.Series(f)
