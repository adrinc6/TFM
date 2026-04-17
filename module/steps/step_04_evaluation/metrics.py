"""Financial metrics used in evaluation and backtesting."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def cumulative_return(returns: pd.Series) -> float:
    """Computes the total cumulative return of a return series.

    Args:
        returns (pd.Series): Periodic return series.

    Returns:
        float: Cumulative return as a decimal (e.g., 0.25 = 25%).
    """
    return float((1 + returns).prod() - 1)


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Computes the compound annual growth rate (CAGR) of a return series.

    Args:
        returns (pd.Series): Periodic return series.
        periods_per_year (int): Number of periods in a year (252 for daily).

    Returns:
        float: Annualized return as a decimal.
    """
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
    """Computes the annualised Sharpe ratio.

    Args:
        returns (pd.Series): Periodic return series.
        risk_free (float): Annual risk-free rate.
        periods_per_year (int): Number of periods in a year.

    Returns:
        float: Sharpe ratio; 0.0 if standard deviation is zero.
    """
    if returns.std() == 0:
        return 0.0
    excess = returns - risk_free / periods_per_year
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Computes the maximum drawdown (worst peak-to-trough loss).

    Args:
        returns (pd.Series): Periodic return series.

    Returns:
        float: Maximum drawdown as a negative decimal (e.g., -0.30 = -30%).
    """
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    return float(dd.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Computes the Calmar ratio (annualised return / |max drawdown|).

    Args:
        returns (pd.Series): Periodic return series.
        periods_per_year (int): Number of periods in a year.

    Returns:
        float: Calmar ratio; 0.0 if max drawdown is zero.
    """
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return float(annualized_return(returns, periods_per_year) / mdd)


def sortino_ratio(returns: pd.Series, risk_free: float = 0.04,
                  periods_per_year: int = 252) -> float:
    """Computes the annualised Sortino ratio (only penalises downside volatility).

    Args:
        returns (pd.Series): Periodic return series.
        risk_free (float): Annual risk-free rate.
        periods_per_year (int): Number of periods in a year.

    Returns:
        float: Sortino ratio; 0.0 if downside standard deviation is zero.
    """
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0].std()
    if downside == 0:
        return 0.0
    return float(excess.mean() / downside * np.sqrt(periods_per_year))


def compute_all_metrics(
    returns: pd.Series, risk_free: float = 0.04, label: str = "strategy"
) -> Dict:
    """Computes all standard performance metrics for a return series.

    Args:
        returns (pd.Series): Periodic return series.
        risk_free (float): Annual risk-free rate for Sharpe/Sortino.
        label (str): Prefix applied to all metric keys in the output dictionary.

    Returns:
        Dict: Dictionary containing cumulative_return, sharpe, sortino,
            max_drawdown, calmar, volatility, and n_periods metrics.
    """
    return {
        f"{label}_cumulative_return": cumulative_return(returns),
        f"{label}_sharpe": sharpe_ratio(returns, risk_free),
        f"{label}_sortino": sortino_ratio(returns, risk_free),
        f"{label}_max_drawdown": max_drawdown(returns),
        f"{label}_calmar": calmar_ratio(returns),
        f"{label}_volatility": float(returns.std() * np.sqrt(252)),
        f"{label}_n_periods": len(returns),
    }
