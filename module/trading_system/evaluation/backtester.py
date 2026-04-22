from __future__ import annotations

import pandas as pd

from module.trading_system.evaluation import choose_best_strategy_per_stock


def backtest_strategy_diagnostics(strategy_diagnostics: pd.DataFrame, score_col: str = "meta_score") -> pd.DataFrame:
    if strategy_diagnostics is None or strategy_diagnostics.empty:
        return pd.DataFrame()

    best = choose_best_strategy_per_stock(strategy_diagnostics, score_col=score_col)
    if best.empty:
        return best

    out = best.copy()
    out["is_tp_first"] = out["label"].astype(int)
    out["is_sl_first"] = 1 - out["is_tp_first"]
    return out
