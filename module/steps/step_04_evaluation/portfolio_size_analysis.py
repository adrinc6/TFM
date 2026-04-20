"""Portfolio size analysis using alpha vs benchmark.

For each fold, this module reuses the already-scored ticker table,
ranks tickers by model score, builds equally weighted portfolios of
sizes 1..MAX_PORTFOLIO_SIZE, and computes alpha versus benchmark.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MAX_PORTFOLIO_SIZE = 20


def _compute_ticker_return(
    prices_dict: Dict[str, pd.DataFrame],
    ticker: str,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
) -> Optional[float]:
    """Compute simple return for a ticker over [entry_date, exit_date]."""
    prices = prices_dict.get(ticker)
    if prices is None or prices.empty:
        return None

    if isinstance(prices, pd.Series):
        close = prices
    elif isinstance(prices, pd.DataFrame):
        if "Close" in prices.columns:
            close = prices["Close"]
        elif len(prices.columns) > 3:
            close = prices.iloc[:, 3]
        else:
            close = prices.iloc[:, 0]
    else:
        return None

    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    entry_window = close.loc[close.index >= entry_date]
    if entry_window.empty:
        entry_window = close.loc[close.index <= entry_date]
        if entry_window.empty:
            return None
        p0 = float(entry_window.iloc[-1])
    else:
        p0 = float(entry_window.iloc[0])

    exit_window = close.loc[close.index <= exit_date]
    if exit_window.empty:
        return None
    p1 = float(exit_window.iloc[-1])

    if p0 <= 0 or pd.isna(p0) or pd.isna(p1):
        return None
    return (p1 - p0) / p0


def compute_portfolio_size_alpha_matrix(
    fold_contexts: List[Dict],
    prices_dict: Dict[str, pd.DataFrame],
    benchmark_prices,
    max_size: int = MAX_PORTFOLIO_SIZE,
) -> pd.DataFrame:
    """Build a matrix of alpha vs benchmark by fold and portfolio size.

    Args:
        fold_contexts: List of dicts with keys:
            - fold_id (str): e.g. "2022YQ1"
            - entry_date (Timestamp)
            - exit_date (Timestamp)
            - preds_df (DataFrame): must have 'ticker' and 'score' columns
        prices_dict: ticker -> price DataFrame
        benchmark_prices: SPY (or benchmark proxy) price series/dataframe
        max_size: Maximum portfolio size (default 20)

    Returns:
        DataFrame with fold_id as index and columns 1..max_size representing
        equally weighted portfolio alpha (portfolio return - benchmark return).
    """
    rows = []

    for ctx in fold_contexts:
        fold_id = str(ctx["fold_id"])
        entry_date = pd.Timestamp(ctx["entry_date"])
        exit_date = pd.Timestamp(ctx["exit_date"])
        preds_df = ctx.get("preds_df", pd.DataFrame())

        if preds_df.empty or "score" not in preds_df.columns or "ticker" not in preds_df.columns:
            log.warning("[PortfolioSize] Fold %s: no predictions available — skipped", fold_id)
            continue

        # Reuse scored predictions already produced by the fold (no model rerun).
        # Rank tickers by score (descending)
        ranked = (
            preds_df[["ticker", "score"]]
            .drop_duplicates(subset=["ticker"], keep="last")
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

        benchmark_return = _compute_ticker_return(
            prices_dict={"__BENCH__": benchmark_prices},
            ticker="__BENCH__",
            entry_date=entry_date,
            exit_date=exit_date,
        )
        if benchmark_return is None:
            log.warning("[PortfolioSize] Fold %s: benchmark return unavailable — skipped", fold_id)
            continue

        # Compute return for each ranked ticker
        ticker_returns = {}
        for _, row in ranked.iterrows():
            tk = str(row["ticker"])
            ret = _compute_ticker_return(prices_dict, tk, entry_date, exit_date)
            if ret is not None:
                ticker_returns[tk] = ret

        # Build equally weighted portfolios of size 1..max_size and convert to alpha.
        ordered_tickers = [
            str(row["ticker"]) for _, row in ranked.iterrows()
            if str(row["ticker"]) in ticker_returns
        ]

        row_data = {"fold": fold_id}
        for n in range(1, max_size + 1):
            if n <= len(ordered_tickers):
                selected = ordered_tickers[:n]
                portfolio_return = np.mean([ticker_returns[tk] for tk in selected])
                row_data[n] = float(portfolio_return - benchmark_return)
            else:
                row_data[n] = np.nan

        rows.append(row_data)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("fold")
    df.columns = [int(c) if isinstance(c, (int, np.integer)) else c for c in df.columns]
    return df


def save_portfolio_size_csv(matrix: pd.DataFrame, output_path: Path) -> None:
    """Save the portfolio size matrix as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path, float_format="%.6f")
    log.info("[PortfolioSize] CSV matrix saved to %s", output_path)


def plot_portfolio_size_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str = "Portfolio Size vs Alpha (Benchmark)",
) -> None:
    """Generate a heatmap showing alpha by fold and size.

    Args:
        matrix: DataFrame with folds as rows, portfolio sizes as columns.
        output_path: Path to save the PNG image.
        title: Plot title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if matrix.empty:
        log.warning("[PortfolioSize] Empty matrix — skipping heatmap")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = matrix.values.astype(float) * 100.0  # convert to percentages
    fold_labels = [str(f) for f in matrix.index]
    size_labels = [str(c) for c in matrix.columns]

    fig, ax = plt.subplots(figsize=(max(12, len(size_labels) * 0.8), max(4, len(fold_labels) * 0.6)))

    # Diverging colormap centered at zero
    vmin = np.nanmin(data) if np.any(np.isfinite(data)) else -10
    vmax = np.nanmax(data) if np.any(np.isfinite(data)) else 10
    abs_max = max(abs(vmin), abs(vmax), 1e-6)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", norm=norm, interpolation="nearest")

    ax.set_xticks(range(len(size_labels)))
    ax.set_xticklabels(size_labels, fontsize=8)
    ax.set_yticks(range(len(fold_labels)))
    ax.set_yticklabels(fold_labels, fontsize=8)
    ax.set_xlabel("Number of Stocks in Portfolio")
    ax.set_ylabel("Fold")
    ax.set_title(title, fontweight="bold", fontsize=12)

    # Add text annotations
    for i in range(len(fold_labels)):
        for j in range(len(size_labels)):
            val = data[i, j]
            if np.isfinite(val):
                color = "white" if abs(val) > abs_max * 0.6 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Alpha (%)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("[PortfolioSize] Heatmap saved to %s", output_path)


def run_portfolio_size_analysis(
    fold_contexts: List[Dict],
    prices_dict: Dict[str, pd.DataFrame],
    benchmark_prices,
    output_dir: Path,
    max_size: int = MAX_PORTFOLIO_SIZE,
) -> pd.DataFrame:
    """Run the full portfolio size alpha analysis: matrix, CSV, heatmap and summary.

    Args:
        fold_contexts: List of fold context dicts from the walk-forward pipeline.
        prices_dict: ticker -> price DataFrame.
        benchmark_prices: SPY (or benchmark proxy) price series/dataframe.
        output_dir: Directory where CSV and heatmap are saved.
        max_size: Maximum portfolio size to evaluate (default 20).

    Returns:
        The portfolio size alpha matrix.
    """
    log.info("[PortfolioSize] Running portfolio size alpha analysis (1..%d stocks)", max_size)

    matrix = compute_portfolio_size_alpha_matrix(
        fold_contexts=fold_contexts,
        prices_dict=prices_dict,
        benchmark_prices=benchmark_prices,
        max_size=max_size,
    )
    if matrix.empty:
        log.warning("[PortfolioSize] No data to analyze")
        return matrix

    output_dir = Path(output_dir)
    save_portfolio_size_csv(matrix, output_dir / "portfolio_size_alpha_vs_benchmark.csv")
    plot_portfolio_size_heatmap(
        matrix,
        output_dir / "portfolio_size_alpha_heatmap.png",
        title="Portfolio Size vs Alpha (EW Portfolio - Benchmark)",
    )

    # Summary statistics
    mean_by_size = matrix.mean(axis=0)
    summary = pd.DataFrame({
        "portfolio_size": mean_by_size.index,
        "mean_alpha": mean_by_size.values,
        "std_alpha": matrix.std(axis=0).values,
        "min_alpha": matrix.min(axis=0).values,
        "max_alpha": matrix.max(axis=0).values,
        "median_alpha": matrix.median(axis=0).values,
    })
    summary.to_csv(output_dir / "portfolio_size_alpha_summary.csv", index=False, float_format="%.6f")
    log.info("[PortfolioSize] Summary saved to %s", output_dir / "portfolio_size_alpha_summary.csv")

    return matrix
