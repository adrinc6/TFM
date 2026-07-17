"""Simple top-N monthly-rebalance simulator used for two robustness checks:

1. Baselines: rank the universe by a single, well-understood signal (equal-weight, momentum,
   valuation) instead of the full system's `manager_score`, and see how much the full pipeline
   (thesis logic, rotation rules, hybrid sizing) actually adds over rules a reader can verify by eye.
2. Placebo: shuffle the ranking score within each snapshot_date (same universe, same dates, same
   portfolio size) and compare the resulting alpha to the real ranking's alpha. If a random ranking
   produces a similar alpha, the ranking itself carries no information; if the real ranking is a
   clear outlier versus the shuffled distribution, that is evidence the ranking matters.

Deliberately independent of module/strategy/portfolio.py's thesis-state/rotation state machine: that
logic answers "should we trade this month given everything we know," which is the right model for the
live strategy but too complex to serve as a simple, auditable benchmark. This module only asks "if we
mechanically held the top N names by score, equal-weighted, rebalanced every snapshot, what would the
alpha have been" — a much simpler question with an answer anyone can hand-check.
"""

from __future__ import annotations

import logging
import time
from bisect import bisect_right

import numpy as np
import pandas as pd

from environment import MAX_PORTFOLIO_SIZE
from module.utils import price_cache_by_ticker

from .performance import compound_alpha

log = logging.getLogger(__name__)


def top_n_monthly_returns(
    universe: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_ticker: str,
    score_column: str,
    top_n: int = MAX_PORTFOLIO_SIZE,
    price_cache: dict[str, tuple[list[pd.Timestamp], list[float]]] | None = None,
) -> pd.DataFrame:
    """Equal-weight top-N-by-score, rebalanced at every snapshot_date in `universe`.

    Holds whatever the top N by `score_column` are AT the snapshot date, then measures the
    equal-weighted return to the NEXT snapshot date (point-in-time: only the score available as of
    each snapshot decides that period's holdings, and the price move is always forward from the
    decision date). Returns one row per period with the portfolio's and benchmark's period return.

    `price_cache` (ticker -> sorted (dates, closes)) can be precomputed once by the caller and reused
    across many calls (e.g. the placebo shuffle loop) to avoid rebuilding it every time.
    """
    if universe.empty or score_column not in universe.columns:
        return pd.DataFrame(columns=["date", "portfolio_period_return", "benchmark_period_return", "period_alpha"])
    if price_cache is None:
        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"])
        price_cache = price_cache_by_ticker(prices)
    dates = sorted(pd.to_datetime(universe["snapshot_date"]).unique())
    rows = []
    for i in range(len(dates) - 1):
        start, end = dates[i], dates[i + 1]
        snapshot = universe[pd.to_datetime(universe["snapshot_date"]) == start]
        top = snapshot.sort_values(score_column, ascending=False).head(top_n)
        tickers = top["ticker"].tolist()
        portfolio_return = _equal_weight_return(price_cache, tickers, start, end)
        benchmark_return = _equal_weight_return(price_cache, [benchmark_ticker], start, end)
        rows.append({
            "date": end.date().isoformat(),
            "portfolio_period_return": portfolio_return,
            "benchmark_period_return": benchmark_return,
            "period_alpha": portfolio_return - benchmark_return,
        })
    return pd.DataFrame(rows)


def cumulative_alpha(monthly_returns: pd.DataFrame) -> float:
    """Alpha acumulada compuesta de una serie de retornos por periodo.

    Delega en `compound_alpha` para que baselines, placebo y sistema usen la MISMA definición.
    """
    return compound_alpha(monthly_returns)


def baseline_comparison(
    universe: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_ticker: str,
    system_cumulative_alpha: float,
    top_n: int = MAX_PORTFOLIO_SIZE,
) -> pd.DataFrame:
    """Compare the full system's realized cumulative alpha against simple, single-signal baselines
    computed on the same universe and dates: equal-weight (no selection at all — every investable
    ticker gets weight, a pure "own the universe" line), momentum-only, and valuation-only.
    """
    baselines = {
        "equal_weight_universe": None,  # special-cased below: every ticker, not top-N
        "momentum_only": "momentum_score",
        "valuation_only": "price_adjusted_valuation_score" if "price_adjusted_valuation_score" in universe.columns else "valuation_score",
    }
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    price_cache = price_cache_by_ticker(prices)
    rows = [{"estrategia": "Sistema completo (manager_score + tesis + rotación)", "alpha_acumulada": system_cumulative_alpha}]
    for name, score_column in baselines.items():
        if name == "equal_weight_universe":
            # Equal-weight-universe means every ticker is held, so ranking by a constant column
            # (everyone tied) and setting top_n to the full universe size keeps them all in.
            universe_flagged = universe.copy()
            universe_flagged["_constant"] = 1.0
            monthly = top_n_monthly_returns(
                universe_flagged, prices, benchmark_ticker, "_constant",
                top_n=len(universe["ticker"].unique()), price_cache=price_cache,
            )
        else:
            monthly = top_n_monthly_returns(universe, prices, benchmark_ticker, score_column, top_n=top_n, price_cache=price_cache)
        rows.append({"estrategia": name, "alpha_acumulada": cumulative_alpha(monthly)})
    return pd.DataFrame(rows)


def placebo_distribution(
    universe: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_ticker: str,
    score_column: str = "final_score",
    top_n: int = MAX_PORTFOLIO_SIZE,
    n_shuffles: int = 200,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Shuffle `score_column` within each snapshot_date and recompute cumulative alpha `n_shuffles`
    times. Returns (real_alpha, shuffled_alphas) so the caller can report where the real ranking
    falls in the shuffled distribution — the plain-English version of "does the ranking matter."
    """
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    price_cache = price_cache_by_ticker(prices)
    real_monthly = top_n_monthly_returns(universe, prices, benchmark_ticker, score_column, top_n=top_n, price_cache=price_cache)
    real_alpha = cumulative_alpha(real_monthly)
    if universe.empty or score_column not in universe.columns:
        return real_alpha, np.array([])

    rng = np.random.default_rng(seed)
    shuffled_alphas = np.empty(n_shuffles)
    loop_start = time.perf_counter()
    for i in range(n_shuffles):
        shuffled = universe.copy()
        shuffled[score_column] = (
            shuffled.groupby("snapshot_date")[score_column]
            .transform(lambda s: rng.permutation(s.to_numpy()))
        )
        monthly = top_n_monthly_returns(shuffled, prices, benchmark_ticker, score_column, top_n=top_n, price_cache=price_cache)
        shuffled_alphas[i] = cumulative_alpha(monthly)
        if (i + 1) % 50 == 0 or i + 1 == n_shuffles:
            log.info("Placebo shuffle %s/%s elapsed=%.1fs", i + 1, n_shuffles, time.perf_counter() - loop_start)
    return real_alpha, shuffled_alphas


def placebo_summary(real_alpha: float, shuffled_alphas: np.ndarray) -> pd.DataFrame:
    if shuffled_alphas.size == 0:
        return pd.DataFrame([{
            "alpha_real": real_alpha, "alpha_medio_aleatorio": None, "percentil_real": None,
            "n_barajados": 0,
        }])
    percentile = float((shuffled_alphas < real_alpha).mean())
    return pd.DataFrame([{
        "alpha_real": real_alpha,
        "alpha_medio_aleatorio": float(shuffled_alphas.mean()),
        "alpha_std_aleatorio": float(shuffled_alphas.std(ddof=1)) if shuffled_alphas.size > 1 else 0.0,
        "percentil_real": percentile,
        "n_barajados": int(shuffled_alphas.size),
    }])


def _equal_weight_return(
    price_cache: dict[str, tuple[list[pd.Timestamp], list[float]]], tickers: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> float:
    if not tickers:
        return 0.0
    returns = []
    for ticker in tickers:
        start_price = _last_price(price_cache, ticker, start)
        end_price = _last_price(price_cache, ticker, end)
        if start_price and end_price and start_price > 0:
            returns.append(end_price / start_price - 1)
    return float(np.mean(returns)) if returns else 0.0


def _last_price(price_cache: dict[str, tuple[list[pd.Timestamp], list[float]]], ticker: str, date: pd.Timestamp) -> float | None:
    series = price_cache.get(ticker)
    if series is None:
        return None
    dates, closes = series
    index = bisect_right(dates, date) - 1
    return None if index < 0 else float(closes[index])
