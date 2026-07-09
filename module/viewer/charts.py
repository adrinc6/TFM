"""Chart generation for the static viewer."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
from environment import PROCESSED_DIR
from matplotlib.ticker import PercentFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_charts(charts_dir: Path, tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    for old_chart in charts_dir.glob("*.png"):
        old_chart.unlink()
    charts: dict[str, str] = {}
    chart_portfolio_vs_benchmark(charts_dir, charts, tables.get("portfolio_vs_benchmark", pd.DataFrame()))
    chart_period_alpha(charts_dir, charts, tables.get("portfolio_vs_benchmark", pd.DataFrame()))
    chart_drawdown(charts_dir, charts, tables.get("portfolio_vs_benchmark", pd.DataFrame()))
    chart_turnover(charts_dir, charts, tables.get("portfolio_turnover", pd.DataFrame()))
    chart_latest_allocation(charts_dir, charts, tables.get("portfolio_allocation", pd.DataFrame()))
    chart_allocation_drift(charts_dir, charts, tables.get("portfolio_allocation", pd.DataFrame()))
    chart_sector_exposure(charts_dir, charts, tables.get("sector_exposure", pd.DataFrame()))
    chart_watchlist_map(charts_dir, charts, tables.get("watchlist", pd.DataFrame()))
    chart_thesis_persistence(charts_dir, charts, tables.get("portfolio_monthly_holdings", pd.DataFrame()))
    chart_position_performance(charts_dir, charts, tables.get("position_performance", pd.DataFrame()))
    chart_feature_importance(charts_dir, charts)
    chart_position_pages(charts_dir, charts, tables.get("portfolio_monthly_holdings", pd.DataFrame()))
    return charts


def chart_portfolio_vs_benchmark(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    if not has_columns(df, ["date", "portfolio_value", "benchmark_value"]):
        return
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    fig, ax = new_figure()
    ax.plot(plot_df["date"], plot_df["portfolio_value"], color="#1f7a8c", linewidth=2.6, label="Cartera neta")
    if "portfolio_gross_value" in plot_df.columns:
        ax.plot(plot_df["date"], plot_df["portfolio_gross_value"], color="#665191", linewidth=1.8, linestyle="--", label="Cartera bruta")
    ax.plot(plot_df["date"], plot_df["benchmark_value"], color="#b23a48", linewidth=2.2, label="Benchmark SPY")
    ax.set_title("Valor de la cartera frente al benchmark")
    ax.set_ylabel("Crecimiento de 1 dolar")
    ax.legend()
    save_chart(fig, charts_dir, charts, "portfolio_vs_benchmark")


def chart_period_alpha(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    if not has_columns(df, ["date", "period_alpha"]):
        return
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    colors = ["#1f7a8c" if value >= 0 else "#b23a48" for value in plot_df["period_alpha"]]
    fig, ax = new_figure()
    ax.bar(plot_df["date"], plot_df["period_alpha"], color=colors, width=18)
    ax.axhline(0, color="#172026", linewidth=0.8)
    ax.set_title("Alpha mensual frente al benchmark")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    save_chart(fig, charts_dir, charts, "period_alpha")


def chart_drawdown(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    if not has_columns(df, ["date", "portfolio_value"]):
        return
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    drawdown = plot_df["portfolio_value"] / plot_df["portfolio_value"].cummax() - 1
    fig, ax = new_figure()
    ax.fill_between(plot_df["date"], drawdown, 0, color="#b23a48", alpha=0.28)
    ax.plot(plot_df["date"], drawdown, color="#b23a48", linewidth=1.8)
    ax.set_title("Drawdown de la cartera")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    save_chart(fig, charts_dir, charts, "drawdown")


def chart_turnover(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    if df.empty:
        return
    if "date" not in df.columns:
        row = df.iloc[0]
        metrics = {"Rotacion mensual": row.get("monthly_turnover", 0), "Rotacion anual": row.get("annual_turnover", 0), "Compras": row.get("buys", 0), "Ventas": row.get("sells", 0)}
        fig, ax = new_figure(height=4.6)
        ax.bar(metrics.keys(), metrics.values(), color=["#1f7a8c", "#665191", "#7a5195", "#b23a48"])
        ax.set_title("Resumen de rotacion de cartera")
        save_chart(fig, charts_dir, charts, "turnover")
        return
    value_col = first_existing(df, ["turnover", "portfolio_turnover", "turnover_rate", "sells", "buys"])
    if not value_col:
        return
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    fig, ax = new_figure()
    ax.bar(plot_df["date"], plot_df[value_col], color="#7a5195", width=18)
    ax.set_title(f"Rotacion por periodo de revision ({value_col})")
    save_chart(fig, charts_dir, charts, "turnover")


def chart_latest_allocation(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    if not has_columns(df, ["date", "ticker", "hybrid_weight"]):
        return
    latest = df[df["date"] == df["date"].max()].sort_values("hybrid_weight", ascending=True)
    fig, ax = new_figure(height=5.8)
    ax.barh(latest["ticker"], latest["hybrid_weight"], color="#1f7a8c")
    ax.set_title(f"Asignacion mas reciente ({latest['date'].max()})")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    save_chart(fig, charts_dir, charts, "latest_allocation")


def chart_allocation_drift(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    if not has_columns(df, ["date", "ticker", "hybrid_weight"]):
        return
    pivot = df.assign(date=pd.to_datetime(df["date"])).pivot_table(index="date", columns="ticker", values="hybrid_weight", aggfunc="last").fillna(0)
    top_tickers = pivot.mean().sort_values(ascending=False).head(12).index.tolist()
    other = pivot.drop(columns=top_tickers, errors="ignore").sum(axis=1)
    plot_df = pivot[top_tickers].copy()
    if (other > 0).any():
        plot_df["Otros"] = other
    fig, ax = new_figure(height=5.8)
    ax.stackplot(plot_df.index, [plot_df[col] for col in plot_df.columns], labels=plot_df.columns)
    ax.set_title("Evolucion de la asignacion en el tiempo")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(ncol=5, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.14))
    save_chart(fig, charts_dir, charts, "allocation_drift")


def chart_sector_exposure(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    if not has_columns(df, ["date", "sector", "weight"]):
        return
    pivot = df.assign(date=pd.to_datetime(df["date"])).pivot_table(index="date", columns="sector", values="weight", aggfunc="sum").fillna(0)
    if pivot.empty:
        return
    top_sectors = pivot.mean().sort_values(ascending=False).head(8).index.tolist()
    other = pivot.drop(columns=top_sectors, errors="ignore").sum(axis=1)
    plot_df = pivot[top_sectors].copy()
    if (other > 0).any():
        plot_df["Other"] = other
    fig, ax = new_figure(height=5.8)
    ax.stackplot(plot_df.index, [plot_df[col] for col in plot_df.columns], labels=plot_df.columns)
    ax.set_title("Exposicion sectorial en el tiempo")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.14))
    save_chart(fig, charts_dir, charts, "sector_exposure")


def chart_watchlist_map(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    if not has_columns(df, ["ticker", "valuation_score", "conviction_score"]):
        return
    plot_df = df.dropna(subset=["valuation_score", "conviction_score"]).copy()
    if plot_df.empty:
        return
    fig, ax = new_figure()
    ax.scatter(plot_df["valuation_score"], plot_df["conviction_score"], s=70, color="#1f7a8c", alpha=0.78)
    for _, row in plot_df.sort_values("conviction_score", ascending=False).head(12).iterrows():
        ax.annotate(str(row["ticker"]), (row["valuation_score"], row["conviction_score"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Watchlist: valoracion frente a conviccion")
    save_chart(fig, charts_dir, charts, "watchlist_map")


def chart_thesis_persistence(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    if not has_columns(df, ["ticker", "thesis_persistence_score"]):
        return
    summary = df.groupby("ticker", as_index=False)["thesis_persistence_score"].mean().sort_values("thesis_persistence_score")
    fig, ax = new_figure(height=5.8)
    ax.barh(summary["ticker"], summary["thesis_persistence_score"], color="#2f4b7c")
    ax.set_title("Persistencia media de tesis por posicion")
    save_chart(fig, charts_dir, charts, "thesis_persistence")


def chart_position_performance(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    required = ["ticker", "holding_days", "total_return", "annualized_return", "benchmark_annualized_return", "excess_total_return"]
    if not has_columns(df, required):
        return
    plot_df = df.copy()
    for column in required[1:]:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce").fillna(0)
    rows = []
    for ticker, group in plot_df.groupby("ticker"):
        weights = group["holding_days"].fillna(1).clip(lower=1)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            continue
        rows.append({
            "ticker": ticker,
            "total_return": float((1 + group["total_return"]).prod() - 1),
            "annualized_return": float((group["annualized_return"] * weights).sum() / weight_sum),
            "benchmark_annualized_return": float((group["benchmark_annualized_return"] * weights).sum() / weight_sum),
            "excess_total_return": float(group["excess_total_return"].sum()),
        })
    summary = pd.DataFrame(rows).sort_values("excess_total_return", ascending=False).head(20).sort_values("excess_total_return")
    fig, ax = new_figure(height=max(5.8, len(summary) * 0.36))
    y = range(len(summary))
    bar_height = 0.24
    ax.barh([i - bar_height for i in y], summary["total_return"], height=bar_height, color="#1f7a8c", label="Retorno total accion")
    ax.barh(y, summary["annualized_return"], height=bar_height, color="#665191", label="Retorno anualizado accion")
    ax.barh([i + bar_height for i in y], summary["benchmark_annualized_return"], height=bar_height, color="#b23a48", label="Retorno anualizado benchmark")
    ax.axvline(0, color="#172026", linewidth=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(summary["ticker"])
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower right", fontsize=8)
    save_chart(fig, charts_dir, charts, "position_performance_bars")


def chart_feature_importance(charts_dir: Path, charts: dict[str, str]) -> None:
    path = PROCESSED_DIR / "model_explainability.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for component, features in payload.get("feature_importance", {}).items():
        if isinstance(features, dict):
            rows.extend({"label": f"{component}: {feature}", "importance": importance} for feature, importance in features.items())
        elif isinstance(features, (int, float)):
            rows.append({"label": component, "importance": features})
    if not rows:
        return
    plot_df = pd.DataFrame(rows).sort_values("importance", ascending=False).head(18).sort_values("importance")
    fig, ax = new_figure(height=6.4)
    ax.barh(plot_df["label"], plot_df["importance"], color="#665191")
    ax.set_title("Variables mas relevantes del modelo")
    save_chart(fig, charts_dir, charts, "feature_importance")


def chart_position_pages(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    if not has_columns(df, ["date", "ticker", "current_conviction_score", "hybrid_weight"]):
        return
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    for ticker, ticker_df in plot_df.groupby("ticker"):
        fig, ax1 = new_figure(height=4.8)
        ax2 = ax1.twinx()
        ax1.plot(ticker_df["date"], ticker_df["current_conviction_score"], color="#1f7a8c", marker="o", label="Conviccion")
        ax2.plot(ticker_df["date"], ticker_df["hybrid_weight"], color="#b23a48", marker="s", label="Peso")
        ax2.yaxis.set_major_formatter(PercentFormatter(1))
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")
        ax1.set_title(f"{ticker}: conviccion y peso")
        save_chart(fig, charts_dir, charts, f"position_{ticker}")


def new_figure(width: float = 10.8, height: float = 5.2):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color="#e6ecec", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def save_chart(fig, charts_dir: Path, charts: dict[str, str], name: str) -> None:
    fig.autofmt_xdate()
    fig.tight_layout()
    output = charts_dir / f"{name}.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    charts[name] = f"charts/{name}.png"


def has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    return df is not None and not df.empty and all(column in df.columns for column in columns)


def first_existing(df: pd.DataFrame, columns: list[str]) -> str | None:
    for column in columns:
        if column in df.columns:
            return column
    numeric_cols = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    return numeric_cols[0] if numeric_cols else None
