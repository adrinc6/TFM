"""Insider feature builder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Dict, Optional

from module.common.asof import filter_asof


class InsiderFeatureBuilder:
    """Procesa transacciones insider y MSPR mensual (Finnhub)."""

    def build(
        self,
        insider_df: Optional[pd.DataFrame] = None,
        mspr_df: Optional[pd.DataFrame] = None,
        as_of: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        f: Dict = {
            "insider_net_ratio_90d": 0.0,
            "insider_sell_ratio": 0.5,
            "mspr_3m": np.nan,
            "mspr_trend": np.nan,
        }

        if insider_df is not None and not insider_df.empty:
            use_insider = insider_df
            if as_of is not None:
                use_insider = filter_asof(use_insider, as_of=as_of, date_col="date")
                start = pd.Timestamp(as_of) - pd.Timedelta(days=90)
                if not use_insider.empty and "date" in use_insider.columns:
                    use_insider = use_insider.loc[
                        pd.to_datetime(use_insider["date"], errors="coerce") >= start
                    ]
            buys = float(use_insider.loc[use_insider["is_buy"] == 1, "shares"].sum()) if not use_insider.empty else 0.0
            sells = float(use_insider.loc[use_insider["is_sell"] == 1, "shares"].sum()) if not use_insider.empty else 0.0
            total = buys + sells
            f["insider_net_ratio_90d"] = (buys - sells) / total if total > 0 else 0.0
            f["insider_sell_ratio"] = sells / total if total > 0 else 0.5

        if mspr_df is not None and not mspr_df.empty and "mspr" in mspr_df.columns:
            use_mspr = mspr_df
            if as_of is not None:
                use_mspr = filter_asof(use_mspr, as_of=as_of)
            mspr_vals = use_mspr["mspr"].dropna()
            if len(mspr_vals) >= 1:
                f["mspr_3m"] = float(mspr_vals.tail(3).mean())
            if len(mspr_vals) >= 3:
                f["mspr_trend"] = float(mspr_vals.iloc[-1] - mspr_vals.iloc[-3])

        return pd.Series(f)
