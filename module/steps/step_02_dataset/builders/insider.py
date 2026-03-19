"""Insider feature builder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Dict, Optional


class InsiderFeatureBuilder:
    """Procesa transacciones insider y MSPR mensual (Finnhub)."""

    def build(
        self,
        insider_df: Optional[pd.DataFrame] = None,
        mspr_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        f: Dict = {
            "insider_net_shares_90d": 0.0,
            "insider_sell_ratio": 0.5,
            "mspr_3m": np.nan,
            "mspr_trend": np.nan,
        }

        if insider_df is not None and not insider_df.empty:
            buys = float(insider_df.loc[insider_df["is_buy"] == 1, "shares"].sum())
            sells = float(insider_df.loc[insider_df["is_sell"] == 1, "shares"].sum())
            total = buys + sells
            f["insider_net_shares_90d"] = buys - sells
            f["insider_sell_ratio"] = sells / total if total > 0 else 0.5

        if mspr_df is not None and not mspr_df.empty and "mspr" in mspr_df.columns:
            mspr_vals = mspr_df["mspr"].dropna()
            if len(mspr_vals) >= 1:
                f["mspr_3m"] = float(mspr_vals.tail(3).mean())
            if len(mspr_vals) >= 3:
                f["mspr_trend"] = float(mspr_vals.iloc[-1] - mspr_vals.iloc[-3])

        return pd.Series(f)
