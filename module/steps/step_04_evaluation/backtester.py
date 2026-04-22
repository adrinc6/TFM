from __future__ import annotations

import pandas as pd

from module.common.trading_core import construct_portfolio


def backtest_strategy_diagnostics(diagnostics: pd.DataFrame, score_col: str = "meta_score") -> pd.DataFrame:
    return construct_portfolio(diagnostics, min_stocks=5, max_stocks=8, sector_cap=3, score_col=score_col)


__all__ = ["construct_portfolio", "backtest_strategy_diagnostics"]
