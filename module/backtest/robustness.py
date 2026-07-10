"""Statistical-robustness checks on an already-finished backtest.

The backtest window is only a few dozen monthly observations, so a single t-stat overstates how much
we know. This module re-reads the tables the backtest already produced (no re-scoring, no re-running
selection) and adds three cheap, honest robustness views:

D1  Block bootstrap of the monthly excess return — a confidence interval on the cumulative alpha and
    the information ratio that preserves month-to-month autocorrelation via multi-month blocks.
D2  Cost/slippage sensitivity — because the per-period cost drag is already stored and scales
    linearly with the cost rate, we can recompute net alpha at any cost multiplier without touching
    the pipeline. The headline is one number: the multiplier at which cumulative alpha crosses zero.
D3  Multi-cutoff comparison — a pure aggregator that stacks the executive summaries of several runs
    with different TRAIN_CUTOFF_DATE into one table. Launching those runs is out of scope here; the
    aggregation is ready to consume them.

Following the project's executive/audit tiering, `build_robustness` returns compact summary tables
for the viewer plus one heavy per-resample distribution table meant to live under results/<run>/audit
and be linked, not embedded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from environment import Settings

BOOTSTRAP_BLOCK_MONTHS = 3
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
COST_MULTIPLIERS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
MONTHS_PER_YEAR = 12


def _monthly_excess(vs_benchmark: pd.DataFrame) -> np.ndarray:
    if vs_benchmark.empty or "portfolio_period_return" not in vs_benchmark.columns:
        return np.array([], dtype=float)
    excess = (
        vs_benchmark["portfolio_period_return"].fillna(0)
        - vs_benchmark["benchmark_period_return"].fillna(0)
    )
    return excess.to_numpy(dtype=float)


def _information_ratio(excess: np.ndarray) -> float:
    if excess.size < 2:
        return 0.0
    std = excess.std(ddof=1)
    return float(excess.mean() * np.sqrt(MONTHS_PER_YEAR) / std) if std > 0 else 0.0


def block_bootstrap_excess_returns(
    vs_benchmark: pd.DataFrame,
    block_months: int = BOOTSTRAP_BLOCK_MONTHS,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Circular block bootstrap on the monthly excess return.

    Returns (compact_summary, distribution). The summary carries the point estimate, the 2.5–97.5%
    confidence interval and the share of resamples above zero, for both the cumulative alpha
    (Σ of monthly excess returns, matching tracking_dashboard's cumulative_alpha) and the information
    ratio. The distribution is the raw per-resample values (heavy — meant for the audit folder).
    """
    excess = _monthly_excess(vs_benchmark)
    n = excess.size
    if n < 2 * block_months:
        empty_summary = pd.DataFrame(columns=[
            "metrica", "estimacion", "ic_2_5", "ic_97_5", "prob_positiva", "n_resamples", "bloque_meses",
        ])
        return empty_summary, pd.DataFrame(columns=["cumulative_alpha", "information_ratio"])

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_months))
    offsets = np.arange(block_months)
    cumulative = np.empty(n_resamples)
    ratios = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.integers(0, n, size=n_blocks)
        index = (starts[:, None] + offsets).ravel() % n
        sample = excess[index][:n]
        cumulative[i] = sample.sum()
        ratios[i] = _information_ratio(sample)

    distribution = pd.DataFrame({"cumulative_alpha": cumulative, "information_ratio": ratios})
    summary = pd.DataFrame([
        _bootstrap_row("Alpha acumulada (Σ exceso mensual)", excess.sum(), cumulative, n_resamples, block_months, as_pct=True),
        _bootstrap_row("Information Ratio", _information_ratio(excess), ratios, n_resamples, block_months, as_pct=False),
    ])
    return summary, distribution


def _bootstrap_row(name: str, point: float, samples: np.ndarray, n_resamples: int, block_months: int, as_pct: bool) -> dict:
    low, high = np.percentile(samples, [2.5, 97.5])
    fmt = (lambda v: f"{v:.1%}") if as_pct else (lambda v: f"{v:.2f}")
    return {
        "metrica": name,
        "estimacion": fmt(point),
        "ic_2_5": fmt(float(low)),
        "ic_97_5": fmt(float(high)),
        "prob_positiva": f"{float((samples > 0).mean()):.0%}",
        "n_resamples": n_resamples,
        "bloque_meses": block_months,
    }


def cost_sensitivity_sweep(
    vs_benchmark: pd.DataFrame,
    settings: Settings,
    multipliers: list[float] | None = None,
) -> pd.DataFrame:
    """Recompute cumulative net alpha at several cost/slippage multipliers.

    The stored `transaction_cost_drag` is exactly (cost_rate × notional traded) per period, so a
    cost multiplier m gives net_return = gross_return − m × drag with no need to redo selection or
    sizing. Cumulative alpha is therefore linear in m: A(m) = Σ(gross − benchmark) − m × Σ(drag).
    """
    multipliers = multipliers if multipliers is not None else COST_MULTIPLIERS
    base_cost_bps = settings.transaction_cost_bps + settings.slippage_bps
    if vs_benchmark.empty or "portfolio_gross_period_return" not in vs_benchmark.columns:
        return pd.DataFrame(columns=["multiplicador", "coste_total_bps", "alpha_acumulada_neta", "sigue_positiva"])
    gross = vs_benchmark["portfolio_gross_period_return"].fillna(0)
    drag = vs_benchmark["transaction_cost_drag"].fillna(0)
    benchmark = vs_benchmark["benchmark_period_return"].fillna(0)
    rows = []
    for multiplier in multipliers:
        cumulative_alpha = float((gross - multiplier * drag - benchmark).sum())
        rows.append({
            "multiplicador": multiplier,
            "coste_total_bps": round(base_cost_bps * multiplier, 2),
            "alpha_acumulada_neta": cumulative_alpha,
            "sigue_positiva": bool(cumulative_alpha > 0),
        })
    return pd.DataFrame(rows)


def cost_breakeven_multiplier(vs_benchmark: pd.DataFrame) -> float | None:
    """Cost multiplier at which cumulative alpha crosses zero (None if it never does or no drag).

    A(m) = A0 − m·D with A0 = Σ(gross − benchmark) and D = Σ(drag); breakeven is A0/D when D > 0 and
    A0 > 0 (an edge that positive costs can erode). If A0 ≤ 0 the alpha is already non-positive at
    zero extra cost; if D ≤ 0 there is nothing to erode.
    """
    if vs_benchmark.empty or "portfolio_gross_period_return" not in vs_benchmark.columns:
        return None
    gross = vs_benchmark["portfolio_gross_period_return"].fillna(0)
    drag = vs_benchmark["transaction_cost_drag"].fillna(0)
    benchmark = vs_benchmark["benchmark_period_return"].fillna(0)
    a0 = float((gross - benchmark).sum())
    total_drag = float(drag.sum())
    if total_drag <= 0 or a0 <= 0:
        return None
    return a0 / total_drag


def robustness_summary(vs_benchmark: pd.DataFrame, bootstrap: pd.DataFrame, breakeven: float | None) -> pd.DataFrame:
    """One-row machine-readable summary for the viewer prose (numeric, not pre-formatted)."""
    excess = _monthly_excess(vs_benchmark)
    return pd.DataFrame([{
        "cumulative_alpha_point": float(excess.sum()) if excess.size else 0.0,
        "information_ratio_point": _information_ratio(excess),
        "breakeven_cost_multiplier": breakeven if breakeven is not None else float("nan"),
        "n_periods": int(excess.size),
        "n_bootstrap_resamples": BOOTSTRAP_RESAMPLES if not bootstrap.empty else 0,
        "bootstrap_block_months": BOOTSTRAP_BLOCK_MONTHS,
    }])


def compare_train_cutoffs(summaries_by_cutoff: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack several runs' executive summaries (one per TRAIN_CUTOFF_DATE) into a comparison table.

    Each value is the run's `executive_summary` DataFrame (or its first row as a dict/Series). The
    orchestration that produces those runs lives outside this module; this only aggregates whatever
    is handed to it, so a future A/B over cutoffs has a ready place to land.
    """
    key_metrics = [
        "cumulative_alpha", "information_ratio", "tracking_error_annualized", "excess_return_t_stat",
        "annual_turnover", "closed_position_win_rate_vs_benchmark", "average_closed_excess_return",
    ]
    rows = []
    for cutoff, summary in summaries_by_cutoff.items():
        if summary is None or (isinstance(summary, pd.DataFrame) and summary.empty):
            continue
        record = summary.iloc[0] if isinstance(summary, pd.DataFrame) else pd.Series(summary)
        rows.append({"train_cutoff_date": cutoff, **{metric: record.get(metric) for metric in key_metrics}})
    return pd.DataFrame(rows).sort_values("train_cutoff_date") if rows else pd.DataFrame(columns=["train_cutoff_date", *key_metrics])


def build_robustness(outputs: dict[str, pd.DataFrame], settings: Settings) -> dict[str, pd.DataFrame]:
    """Assemble every robustness table from the finished backtest outputs.

    Returns compact tables for the viewer plus one heavy distribution table for the audit folder.
    """
    vs_benchmark = outputs.get("portfolio_vs_benchmark", pd.DataFrame())
    bootstrap, distribution = block_bootstrap_excess_returns(vs_benchmark)
    breakeven = cost_breakeven_multiplier(vs_benchmark)
    return {
        "robustness_bootstrap": bootstrap,
        "robustness_bootstrap_distribution": distribution,
        "robustness_cost_sensitivity": cost_sensitivity_sweep(vs_benchmark, settings),
        "robustness_summary": robustness_summary(vs_benchmark, bootstrap, breakeven),
    }
