"""Financial metrics used in evaluation and backtesting."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def cumulative_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    n = len(returns)
    if n == 0:
        return 0.0
    cr = cumulative_return(returns)
    base = 1 + cr
    if base <= 0:
        return -1.0
    return float(base ** (periods_per_year / n) - 1)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.04,
                 periods_per_year: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    excess = returns - risk_free / periods_per_year
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    return float(dd.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return float(annualized_return(returns, periods_per_year) / mdd)


def sortino_ratio(returns: pd.Series, risk_free: float = 0.04,
                  periods_per_year: int = 252) -> float:
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0].std()
    if downside == 0:
        return 0.0
    return float(excess.mean() / downside * np.sqrt(periods_per_year))


def hit_rate(predictions: pd.Series, labels: pd.Series) -> float:
    mask = predictions == 1
    if mask.sum() == 0:
        return 0.0
    return float(labels[mask].mean())


def compute_all_metrics(
    returns: pd.Series, risk_free: float = 0.04, label: str = "strategy"
) -> Dict:
    return {
        f"{label}_cumulative_return": cumulative_return(returns),
        f"{label}_sharpe": sharpe_ratio(returns, risk_free),
        f"{label}_sortino": sortino_ratio(returns, risk_free),
        f"{label}_max_drawdown": max_drawdown(returns),
        f"{label}_calmar": calmar_ratio(returns),
        f"{label}_volatility": float(returns.std() * np.sqrt(252)),
        f"{label}_n_periods": len(returns),
    }
