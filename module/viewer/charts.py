"""Charts for the report viewer — only the plots that answer a TFM/debug question.

Rendered with matplotlib (Agg, no extra deps) using the validated dataviz palette and a single
consistent style. Each chart carries direct legend labels, so the low-contrast palette slots are
covered by the relief rule.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from .shared import PALETTE

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.edgecolor": "#c8d0d2",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e7ecec",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": "#16232b",
    "axes.labelcolor": "#56656c",
    "xtick.color": "#56656c",
    "ytick.color": "#56656c",
})


def build_charts(charts_dir: Path, tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    for old in charts_dir.glob("*.png"):
        old.unlink()
    charts: dict[str, str] = {}
    vs = tables.get("portfolio_vs_benchmark", pd.DataFrame())
    _chart_value(charts_dir, charts, vs)
    _chart_cumulative_alpha(charts_dir, charts, vs)
    _chart_drawdown(charts_dir, charts, vs)
    _chart_learned_weights(charts_dir, charts, tables.get("meta_weights_by_snapshot", pd.DataFrame()))
    _chart_rank_ic(charts_dir, charts, tables.get("model_walk_forward_diagnostics", pd.DataFrame()))
    _chart_position_performance(charts_dir, charts, tables.get("position_performance", pd.DataFrame()))
    _chart_horizon_comparison(charts_dir, charts, tables.get("label_horizon_comparison", pd.DataFrame()))
    return charts


def _new(height: float = 3.4):
    fig, ax = plt.subplots(figsize=(8.4, height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def _save(fig, charts_dir: Path, charts: dict[str, str], name: str) -> None:
    fig.tight_layout()
    path = charts_dir / f"{name}.png"
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    charts[name] = f"charts/{name}.png"


def _has(df: pd.DataFrame, cols: list[str]) -> bool:
    return not df.empty and all(c in df.columns for c in cols)


def _chart_value(charts_dir, charts, df) -> None:
    if not _has(df, ["date", "portfolio_value", "benchmark_value"]):
        return
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    fig, ax = _new()
    ax.plot(d["date"], d["portfolio_value"], color=PALETTE["portfolio"], linewidth=2.2, label="Cartera (neta)")
    ax.plot(d["date"], d["benchmark_value"], color=PALETTE["benchmark"], linewidth=2.0, label="Benchmark (SPY)")
    ax.set_ylabel("Crecimiento de 1 €")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, charts_dir, charts, "value_vs_benchmark")


def _chart_cumulative_alpha(charts_dir, charts, df) -> None:
    if not _has(df, ["date", "cumulative_alpha"]):
        return
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    fig, ax = _new(2.9)
    ax.fill_between(d["date"], d["cumulative_alpha"], 0, color=PALETTE["portfolio"], alpha=0.16)
    ax.plot(d["date"], d["cumulative_alpha"], color=PALETTE["portfolio"], linewidth=2.0)
    ax.axhline(0, color=PALETTE["muted"], linewidth=1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylabel("Alpha acumulado vs SPY")
    _save(fig, charts_dir, charts, "cumulative_alpha")


def _chart_drawdown(charts_dir, charts, df) -> None:
    if not _has(df, ["date", "portfolio_value"]):
        return
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    dd = d["portfolio_value"] / d["portfolio_value"].cummax() - 1
    fig, ax = _new(2.6)
    ax.fill_between(d["date"], dd, 0, color=PALETTE["negative"], alpha=0.18)
    ax.plot(d["date"], dd, color=PALETTE["negative"], linewidth=1.6)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylabel("Drawdown")
    _save(fig, charts_dir, charts, "drawdown")


def _chart_learned_weights(charts_dir, charts, df) -> None:
    """Evolution of the meta-agent's learned agent weights — the visual proof of learning.

    Pure rolling walk-forward: the meta-agent relearns every quarter across the entire simulated
    window, so this whole series reflects active learning, not a fixed combination applied late.
    """
    agents = {
        "quality_probability": ("Calidad", PALETTE["portfolio"]),
        "timing_probability": ("Temporización", PALETTE["benchmark"]),
        "alpha_probability": ("Alpha (ranking directo)", PALETTE["series_3"]),
    }
    if df.empty or "snapshot_date" not in df.columns or not all(a in df.columns for a in agents):
        return
    d = df.copy()
    d["snapshot_date"] = pd.to_datetime(d["snapshot_date"])
    d = d.sort_values("snapshot_date")
    fig, ax = _new(3.2)
    for col, (label, color) in agents.items():
        ax.plot(d["snapshot_date"], d[col], linewidth=2.0, color=color, label=label)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylabel("Peso aprendido")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    _save(fig, charts_dir, charts, "learned_weights")


def _chart_rank_ic(charts_dir, charts, df) -> None:
    """Out-of-sample rank-IC of the master signal (`final_score`, the meta-agent output) per
    snapshot — model quality over time.

    Pure rolling walk-forward: every snapshot is scored with a model retrained as of that quarter,
    so the whole series measures genuine OOS predictive power, not a fixed model aging.
    """
    if df.empty or "snapshot_date" not in df.columns or "rank_ic_final" not in df.columns:
        return
    d = df.copy()
    d["rank_ic_final"] = pd.to_numeric(d["rank_ic_final"], errors="coerce")
    d = d.dropna(subset=["rank_ic_final"])
    if d.empty:
        return
    d["snapshot_date"] = pd.to_datetime(d["snapshot_date"])
    d = d.sort_values("snapshot_date")
    colors = [PALETTE["positive"] if v >= 0 else PALETTE["negative"] for v in d["rank_ic_final"]]
    fig, ax = _new(2.8)
    ax.bar(d["snapshot_date"], d["rank_ic_final"], width=20, color=colors, alpha=0.55, label="Rank-IC por snapshot")
    ax.axhline(0, color=PALETTE["muted"], linewidth=1.0)
    if "rank_ic_final_rolling" in d.columns and d["rank_ic_final_rolling"].notna().any():
        # Rolling mean shows whether OOS predictive power trends up, stays flat, or oscillates —
        # with ~90 snapshots this is descriptive, not a significance test; shown as-is either way.
        ax.plot(d["snapshot_date"], d["rank_ic_final_rolling"], color=PALETTE["portfolio"],
                 linewidth=2.2, label="Media móvil (12 snapshots)")
    else:
        ax.axhline(float(d["rank_ic_final"].mean()), color=PALETTE["portfolio"], linewidth=1.4,
                   linestyle="--", label=f"Media {d['rank_ic_final'].mean():.2f}")
    ax.set_ylabel("Rank-IC OOS (señal maestra)")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, charts_dir, charts, "rank_ic")


def _chart_horizon_comparison(charts_dir, charts, df) -> None:
    """Rank-IC of the combined master signal at 3/6/12-month horizons — answers whether the
    strategy's edge lives at a shorter or longer horizon, with data instead of assumption.
    """
    if df.empty or "horizon_months" not in df.columns or "rank_ic_mean" not in df.columns:
        return
    d = df.copy()
    d["rank_ic_mean"] = pd.to_numeric(d["rank_ic_mean"], errors="coerce")
    d = d.dropna(subset=["rank_ic_mean"])
    if d.empty:
        return
    d = d.sort_values("horizon_months")
    colors = [
        PALETTE["portfolio"] if bool(row.get("is_current_default")) else PALETTE["muted"]
        for _, row in d.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.bar([f"{m}m" for m in d["horizon_months"]], d["rank_ic_mean"], color=colors, width=0.55)
    ax.axhline(0, color=PALETTE["muted"], linewidth=1.0)
    ax.set_ylabel("Rank-IC medio (señal maestra)")
    ax.set_xlabel("Horizonte de predicción")
    _save(fig, charts_dir, charts, "horizon_comparison")


def _chart_position_performance(charts_dir, charts, df) -> None:
    required = ["ticker", "excess_total_return"]
    if not _has(df, required):
        return
    d = df.copy()
    d["excess_total_return"] = pd.to_numeric(d["excess_total_return"], errors="coerce")
    d = d.dropna(subset=["excess_total_return"])
    if d.empty:
        return
    d = d.groupby("ticker")["excess_total_return"].sum().sort_values()
    d = pd.concat([d.head(8), d.tail(8)]).drop_duplicates()
    colors = [PALETTE["positive"] if v >= 0 else PALETTE["negative"] for v in d.values]
    fig, ax = _new(max(2.6, 0.32 * len(d)))
    ax.barh(d.index, d.values, color=colors)
    ax.axvline(0, color=PALETTE["muted"], linewidth=1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlabel("Retorno en exceso acumulado vs benchmark")
    _save(fig, charts_dir, charts, "position_performance")
