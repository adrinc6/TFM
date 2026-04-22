from __future__ import annotations

import numpy as np
import pandas as pd

from module.common.metrics import expected_value_from_probability


def build_per_stock_diagnostics(strategy_df: pd.DataFrame, score_col: str = "meta_score") -> pd.DataFrame:
    if strategy_df is None or strategy_df.empty:
        return pd.DataFrame()

    df = strategy_df.copy()
    df["expected_value_model"] = expected_value_from_probability(df[score_col], df["tp_pct"], df["sl_pct"])
    sl = pd.to_numeric(df["sl_pct"], errors="coerce").fillna(0.0)
    sl_safe = sl.where(sl.abs() >= 1e-6, np.nan)
    df["risk_reward"] = pd.to_numeric(df["tp_pct"], errors="coerce").fillna(0.0) / sl_safe

    idx = df.groupby(["ticker", "date"])["expected_value_model"].idxmax()
    out = df.loc[idx].copy()
    out = out.sort_values(["snapshot_date", "expected_value_model"], ascending=[True, False])
    return out.reset_index(drop=True)


def add_portfolio_flag(diagnostics: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    out = diagnostics.copy()
    out["selected_in_portfolio"] = False
    if portfolio is None or portfolio.empty:
        return out

    selected = pd.MultiIndex.from_frame(portfolio[["ticker", "date"]])
    diag_index = pd.MultiIndex.from_frame(out[["ticker", "date"]])
    out["selected_in_portfolio"] = diag_index.isin(selected)
    return out
