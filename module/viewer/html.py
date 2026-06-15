"""Generate the required static viewer pages."""

from __future__ import annotations

import html
import json
import logging
import math
from pathlib import Path

import matplotlib
import pandas as pd

from environment import PROCESSED_DIR, Settings

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

log = logging.getLogger(__name__)


PAGES = [
    "index.html",
    "current_portfolio.html",
    "tracking_dashboard.html",
    "portfolio_vs_benchmark.html",
    "action_journal.html",
    "position_performance.html",
    "buy_rationale.html",
    "sell_reasons.html",
    "sector_exposure.html",
    "allocation_dashboard.html",
    "watchlist.html",
    "top_opportunities.html",
    "strategy_learning.html",
    "model_explainability.html",
    "audit.html",
]


def build_viewer(settings: Settings) -> Path:
    run_dir = settings.run_dir
    viewer_dir = run_dir / "viewer"
    charts_dir = viewer_dir / "charts"
    viewer_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    tables = {path.stem: pd.read_csv(path) for path in (run_dir / "audit").glob("*.csv")} if (run_dir / "audit").exists() else {}
    tables.update({path.stem: pd.read_csv(path) for path in run_dir.glob("*.csv")})
    log.info("Building viewer for %s tables=%s", run_dir.name, sorted(tables))
    if "watchlist" not in tables and (PROCESSED_DIR / "watchlist.parquet").exists():
        tables["watchlist"] = pd.read_parquet(PROCESSED_DIR / "watchlist.parquet")

    charts = _build_charts(charts_dir, tables)
    nav = " ".join(f'<a href="{page}">{page.replace(".html", "")}</a>' for page in PAGES)
    _remove_stale_pages(viewer_dir)
    for page in PAGES:
        body = _page_body(page, tables, charts)
        (viewer_dir / page).write_text(_layout(page, nav, body), encoding="utf-8")

    holdings = tables.get("portfolio_monthly_holdings", pd.DataFrame())
    for ticker in sorted(holdings.get("ticker", pd.Series(dtype=str)).dropna().unique()):
        ticker_rows = holdings[holdings["ticker"] == ticker]
        body = (
            f"<h1>{html.escape(ticker)}</h1>"
            + _figure(charts.get(f"position_{ticker}"), f"{ticker} thesis and allocation over time")
            + _table(ticker_rows)
        )
        (viewer_dir / f"position_{ticker}.html").write_text(_layout(ticker, nav, body), encoding="utf-8")
    _write_results_explainer(run_dir, viewer_dir, charts, tables)
    log.info("Viewer written to %s pages=%s charts=%s", viewer_dir, len(list(viewer_dir.glob("*.html"))), len(charts))
    return viewer_dir


def _page_body(page: str, tables: dict[str, pd.DataFrame], charts: dict[str, str]) -> str:
    if page == "index.html":
        return (
            "<h1>GARP AI Portfolio</h1>"
            + "<h2>Executive Summary</h2>"
            + _table(tables.get("executive_summary", pd.DataFrame()))
            + _figure(charts.get("portfolio_vs_benchmark"), "Portfolio value versus SPY benchmark")
            + _figure(charts.get("period_alpha"), "Monthly alpha contribution")
            + _figure(charts.get("position_performance_bars"), "Return earned by stock versus benchmark during each holding period")
            + _figure(charts.get("sector_exposure"), "Sector exposure through time")
            + "<h2>Current Portfolio</h2>"
            + _table(tables.get("current_portfolio", pd.DataFrame()))
            + "<h2>Recent Actions</h2>"
            + _table(tables.get("action_journal", pd.DataFrame()).tail(20))
        )
    if page == "current_portfolio.html":
        return "<h1>Current Portfolio</h1>" + _table(tables.get("current_portfolio", pd.DataFrame()))
    if page == "action_journal.html":
        cols = [
            "date", "ticker", "action", "reason_category", "rank", "manager_score",
            "buy_today_score", "holding_days", "total_return", "benchmark_total_return",
            "excess_total_return", "reason",
        ]
        return "<h1>Action Journal</h1>" + _table(_select(tables.get("action_journal", pd.DataFrame()), cols))
    if page == "audit.html":
        return "<h1>Audit Files</h1>" + _table(_audit_file_table(tables))
    if page == "top_opportunities.html":
        return "<h1>Top Opportunities</h1>" + _table(tables.get("top_opportunities_latest", pd.DataFrame()))
    if page == "strategy_learning.html":
        return (
            "<h1>Strategy Learning</h1>"
            + "<h2>Improvement Backlog</h2>"
            + _table(tables.get("improvement_backlog", pd.DataFrame()))
            + "<h2>Evidence Log</h2>"
            + _table(tables.get("strategy_learning_log", pd.DataFrame()))
        )
    if page == "portfolio_review.html":
        cols = [
            "date", "rank", "ticker", "in_portfolio", "thesis_rank_score",
            "weakest_holding", "score_advantage_vs_weakest", "replacement_candidate", "reason",
        ]
        return "<h1>Portfolio Review</h1>" + _table(_select(tables.get("portfolio_review_diagnostics", pd.DataFrame()), cols))
    if page == "tracking_dashboard.html":
        cols = [
            "date", "portfolio_value", "portfolio_gross_value", "benchmark_value",
            "portfolio_period_return", "portfolio_gross_period_return", "transaction_cost_drag",
            "benchmark_period_return", "period_alpha", "cumulative_alpha",
            "holdings", "buys", "sells", "tickers",
        ]
        return "<h1>Tracking Dashboard</h1>" + _table(_select(tables.get("tracking_dashboard", pd.DataFrame()), cols))
    if page == "portfolio_evolution.html":
        return (
            "<h1>Portfolio Evolution</h1>"
            + _figure(charts.get("portfolio_vs_benchmark"), "Portfolio value versus benchmark")
            + _figure(charts.get("drawdown"), "Portfolio drawdown")
            + _table(tables.get("portfolio_evolution", pd.DataFrame()))
        )
    if page == "portfolio_turnover.html":
        cols = ["monthly_turnover", "annual_turnover", "average_holding_days", "median_holding_days", "buys", "sells", "sell_reason_mix"]
        return "<h1>Turnover</h1>" + _figure(charts.get("turnover"), "Monthly portfolio turnover") + _table(_select(tables.get("portfolio_turnover", pd.DataFrame()), cols))
    if page == "sector_exposure.html":
        return "<h1>Sector Exposure</h1>" + _figure(charts.get("sector_exposure"), "Sector weights through time") + _table(tables.get("sector_exposure", pd.DataFrame()))
    if page == "portfolio_vs_benchmark.html":
        vs = tables.get("portfolio_vs_benchmark", pd.DataFrame())
        return (
            "<h1>Portfolio Vs Benchmark</h1>"
            + _figure(charts.get("portfolio_vs_benchmark"), "Portfolio value versus SPY benchmark")
            + _figure(charts.get("period_alpha"), "Monthly excess return versus benchmark")
            + _figure(charts.get("drawdown"), "Drawdown profile")
            + _table(vs)
        )
    if page == "watchlist.html":
        return "<h1>Watchlist</h1>" + _figure(charts.get("watchlist_map"), "Watchlist valuation versus conviction") + _table(tables.get("watchlist", pd.DataFrame()))
    if page == "rebalance_report.html":
        return "<h1>Rebalance Report</h1>" + _table(tables.get("rebalance_report", pd.DataFrame()))
    if page == "buy_rationale.html":
        cols = [
            "date", "ticker", "rank", "manager_score", "buy_today_score", "thesis_rank_score",
            "business_quality_score", "price_adjusted_valuation_score", "momentum_score",
            "alpha_probability", "opportunity_type", "best_alternative_ticker",
            "opportunity_cost_score", "reason",
        ]
        return "<h1>Buy Rationale</h1>" + _table(_select(tables.get("buy_rationale", pd.DataFrame()), cols))
    if page == "sell_reasons.html":
        return "<h1>Sell Reasons</h1>" + _table(tables.get("sell_reasons_summary", pd.DataFrame()))
    if page == "research.html":
        cols = ["date", "ticker", "investment_thesis", "bull_thesis", "bear_thesis", "catalyst", "moat_analysis"]
        return "<h1>Research</h1>" + _table(_select(tables.get("portfolio_monthly_holdings", pd.DataFrame()), cols))
    if page == "position_performance.html":
        cols = [
            "ticker", "entry_date", "exit_date", "closed", "holding_days", "total_return",
            "annualized_return", "benchmark_annualized_return", "excess_total_return",
            "exit_reason_category",
        ]
        data = tables.get("position_performance", pd.DataFrame())
        data = data.sort_values("excess_total_return", ascending=False) if "excess_total_return" in data.columns else data
        return (
            "<h1>Position Performance</h1>"
            + _figure(charts.get("position_performance_bars"), "Stock return, annualized stock return and annualized benchmark return by ticker")
            + _table(_select(data, cols))
        )
    if page == "exit_thesis.html":
        cols = ["date", "ticker", "current_thesis_state", "exit_thesis", "reason"]
        decisions = tables.get("portfolio_decision_log", pd.DataFrame())
        tx = tables.get("portfolio_transactions", pd.DataFrame())
        return "<h1>Exit Thesis</h1>" + _table(_select(tx, ["date", "ticker", "action", "reason", "exit_thesis"])) + _table(_select(decisions, cols))
    if page == "allocation_dashboard.html":
        return (
            "<h1>Position Sizing</h1>"
            + _figure(charts.get("latest_allocation"), "Latest hybrid allocation")
            + _figure(charts.get("allocation_drift"), "Weight by ticker through time")
            + _table(tables.get("portfolio_allocation", pd.DataFrame()))
        )
    if page == "model_explainability.html":
        return "<h1>Model Explainability</h1>" + _figure(charts.get("feature_importance"), "Top model feature importances") + _explainability()
    if page == "decision_log.html":
        return "<h1>Decision Log</h1>" + _table(tables.get("portfolio_decision_log", pd.DataFrame()))
    if page == "thesis_persistence.html":
        cols = ["date", "ticker", "current_thesis_state", "thesis_persistence_score", "months_thesis_intact"]
        return "<h1>Thesis Persistence</h1>" + _figure(charts.get("thesis_persistence"), "Average thesis persistence by ticker") + _table(_select(tables.get("portfolio_monthly_holdings", pd.DataFrame()), cols))
    return "<h1>Unavailable</h1><p>No dedicated data for this page.</p>"


def _layout(title: str, nav: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; color: #172026; background: #f7f8f8; }}
nav {{ position: sticky; top: 0; display: flex; gap: 12px; overflow-x: auto; padding: 12px 18px; background: #172026; }}
nav a {{ color: white; text-decoration: none; white-space: nowrap; font-size: 13px; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 28px; margin: 8px 0 18px; }}
table {{ width: 100%; border-collapse: collapse; margin: 18px 0; background: white; border: 1px solid #d7dddd; }}
th, td {{ padding: 9px 10px; border-bottom: 1px solid #e7ebeb; text-align: left; font-size: 13px; vertical-align: top; }}
th {{ background: #eef2f2; }}
figure {{ margin: 18px 0 24px; background: white; border: 1px solid #d7dddd; padding: 12px; }}
figure img {{ display: block; width: 100%; height: auto; }}
figcaption {{ color: #516066; font-size: 13px; margin-top: 8px; }}
</style>
</head>
<body><nav>{nav}</nav><main>{body}</main></body>
</html>"""


def _table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p>No data available.</p>"
    return df.to_html(index=False, escape=True)


def _select(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df[[col for col in columns if col in df.columns]]


def _remove_stale_pages(viewer_dir: Path) -> None:
    keep = set(PAGES)
    for path in viewer_dir.glob("*.html"):
        if path.name.startswith("position_") and path.name != "position_sizing.html":
            continue
        if path.name not in keep:
            path.unlink()


def _explainability() -> str:
    path = PROCESSED_DIR / "model_explainability.json"
    if not path.exists():
        return "<p>No explainability artifact available.</p>"
    payload = json.loads(path.read_text(encoding="utf-8"))
    importance = pd.DataFrame(
        [{"feature": key, "importance": value} for key, value in payload.get("feature_importance", {}).items()]
    )
    shap = payload.get("shap", {})
    shap_values = shap.get("mean_abs_contribution", {}) if shap.get("available") else {}
    shap_table = pd.DataFrame([{"feature": key, "mean_abs_shap": value} for key, value in shap_values.items()])
    reason = "" if shap.get("available") else f"<p>{html.escape(shap.get('reason', 'SHAP unavailable.'))}</p>"
    return "<h2>Feature Importance</h2>" + _table(importance) + "<h2>SHAP</h2>" + reason + _table(shap_table)


def _research_ai(settings, tables: dict[str, pd.DataFrame]) -> str:
    if "research_ai" in tables:
        cols = ["ticker", "business_summary", "moat", "catalysts", "risks", "recent_news", "thesis", "exit_thesis", "classification", "source"]
        return _table(_select(tables["research_ai"], cols))
    return "<p>No Research AI artifact available.</p>"


def _audit_file_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    audit_names = [
        "portfolio_monthly_holdings",
        "rebalance_report",
        "universe_monthly_scores",
        "universe_monthly_price_update",
        "universe_quarterly_fundamental_review",
        "universe_top_candidates",
        "research_ai_history",
        "watchlist_history",
    ]
    rows = []
    descriptions = {
        "portfolio_monthly_holdings": "Historico completo posicion x mes usado para paginas por ticker y sizing.",
        "rebalance_report": "Union detallada de transacciones y decisiones HOLD/WATCH/REDUCE.",
        "universe_monthly_scores": "Scoring mensual completo de todo el universo.",
        "universe_monthly_price_update": "Subconjunto de meses intermedios con actualizacion por precio.",
        "universe_quarterly_fundamental_review": "Subconjunto de revisiones trimestrales de fundamentales.",
        "universe_top_candidates": "Top candidatos historicos por fecha.",
        "research_ai_history": "Research historico por compania y snapshot.",
        "watchlist_history": "Watchlist historica por snapshot.",
    }
    for name in audit_names:
        df = tables.get(name, pd.DataFrame())
        rows.append({
            "file": f"audit/{name}.csv",
            "purpose": descriptions[name],
            "rows": len(df),
            "columns": len(df.columns) if not df.empty else 0,
            "read_first": "No",
        })
    return pd.DataFrame(rows)


def _build_charts(charts_dir: Path, tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    for old_chart in charts_dir.glob("*.png"):
        old_chart.unlink()
    charts: dict[str, str] = {}
    _chart_portfolio_vs_benchmark(charts_dir, charts, tables.get("portfolio_vs_benchmark", pd.DataFrame()))
    _chart_period_alpha(charts_dir, charts, tables.get("portfolio_vs_benchmark", pd.DataFrame()))
    _chart_drawdown(charts_dir, charts, tables.get("portfolio_vs_benchmark", pd.DataFrame()))
    _chart_turnover(charts_dir, charts, tables.get("portfolio_turnover", pd.DataFrame()))
    _chart_latest_allocation(charts_dir, charts, tables.get("portfolio_allocation", pd.DataFrame()))
    _chart_allocation_drift(charts_dir, charts, tables.get("portfolio_allocation", pd.DataFrame()))
    _chart_sector_exposure(charts_dir, charts, tables.get("sector_exposure", pd.DataFrame()))
    _chart_watchlist_map(charts_dir, charts, tables.get("watchlist", pd.DataFrame()))
    _chart_thesis_persistence(charts_dir, charts, tables.get("portfolio_monthly_holdings", pd.DataFrame()))
    _chart_position_performance(charts_dir, charts, tables.get("position_performance", pd.DataFrame()))
    _chart_feature_importance(charts_dir, charts)
    _chart_position_pages(charts_dir, charts, tables.get("portfolio_monthly_holdings", pd.DataFrame()))
    return charts


def _chart_portfolio_vs_benchmark(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    if not _has_columns(df, ["date", "portfolio_value", "benchmark_value"]):
        return
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    fig, ax = _new_figure()
    ax.plot(plot_df["date"], plot_df["portfolio_value"], color="#1f7a8c", linewidth=2.6, label="Portfolio net")
    if "portfolio_gross_value" in plot_df.columns:
        ax.plot(plot_df["date"], plot_df["portfolio_gross_value"], color="#665191", linewidth=1.8, linestyle="--", label="Portfolio gross")
    ax.plot(plot_df["date"], plot_df["benchmark_value"], color="#b23a48", linewidth=2.2, label="Benchmark SPY")
    ax.set_title("Portfolio value versus benchmark")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    _save_chart(fig, charts_dir, charts, "portfolio_vs_benchmark")


def _chart_period_alpha(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    if not _has_columns(df, ["date", "period_alpha"]):
        return
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    colors = ["#1f7a8c" if value >= 0 else "#b23a48" for value in plot_df["period_alpha"]]
    fig, ax = _new_figure()
    ax.bar(plot_df["date"], plot_df["period_alpha"], color=colors, width=18)
    ax.axhline(0, color="#172026", linewidth=0.8)
    ax.set_title("Monthly alpha versus benchmark")
    ax.set_ylabel("Alpha")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    _save_chart(fig, charts_dir, charts, "period_alpha")


def _chart_drawdown(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    if not _has_columns(df, ["date", "portfolio_value"]):
        return
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    drawdown = plot_df["portfolio_value"] / plot_df["portfolio_value"].cummax() - 1
    fig, ax = _new_figure()
    ax.fill_between(plot_df["date"], drawdown, 0, color="#b23a48", alpha=0.28)
    ax.plot(plot_df["date"], drawdown, color="#b23a48", linewidth=1.8)
    ax.set_title("Portfolio drawdown")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    _save_chart(fig, charts_dir, charts, "drawdown")


def _chart_turnover(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    if df.empty:
        return
    if "date" not in df.columns:
        row = df.iloc[0]
        metrics = {
            "Monthly turnover": row.get("monthly_turnover", 0),
            "Annual turnover": row.get("annual_turnover", 0),
            "Buys": row.get("buys", 0),
            "Sells": row.get("sells", 0),
        }
        fig, ax = _new_figure(height=4.6)
        ax.bar(metrics.keys(), metrics.values(), color=["#1f7a8c", "#665191", "#7a5195", "#b23a48"])
        ax.set_title("Portfolio turnover summary")
        ax.set_ylabel("Rate or count")
        _save_chart(fig, charts_dir, charts, "turnover")
        return
    date_col = "date" if "date" in df.columns else df.columns[0]
    value_col = _first_existing(df, ["turnover", "portfolio_turnover", "turnover_rate", "sells", "buys"])
    if not value_col:
        return
    plot_df = df.copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    fig, ax = _new_figure()
    ax.bar(plot_df[date_col], plot_df[value_col], color="#7a5195", width=18)
    ax.set_title(f"Turnover by review period ({value_col})")
    ax.set_ylabel(value_col.replace("_", " ").title())
    _save_chart(fig, charts_dir, charts, "turnover")


def _chart_latest_allocation(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    if not _has_columns(df, ["date", "ticker", "hybrid_weight"]):
        return
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].sort_values("hybrid_weight", ascending=True)
    fig, ax = _new_figure(height=5.8)
    ax.barh(latest["ticker"], latest["hybrid_weight"], color="#1f7a8c")
    ax.set_title(f"Latest allocation ({latest_date})")
    ax.set_xlabel("Hybrid weight")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    _save_chart(fig, charts_dir, charts, "latest_allocation")


def _chart_allocation_drift(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    if not _has_columns(df, ["date", "ticker", "hybrid_weight"]):
        return
    pivot = df.assign(date=pd.to_datetime(df["date"])).pivot_table(
        index="date", columns="ticker", values="hybrid_weight", aggfunc="last"
    ).fillna(0)
    fig, ax = _new_figure(height=5.8)
    ax.stackplot(pivot.index, [pivot[col] for col in pivot.columns], labels=pivot.columns)
    ax.set_title("Allocation drift through time")
    ax.set_ylabel("Portfolio weight")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(ncol=5, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.14))
    _save_chart(fig, charts_dir, charts, "allocation_drift")


def _chart_sector_exposure(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    if not _has_columns(df, ["date", "sector", "weight"]):
        return
    pivot = df.assign(date=pd.to_datetime(df["date"])).pivot_table(
        index="date", columns="sector", values="weight", aggfunc="sum"
    ).fillna(0)
    if pivot.empty:
        return
    top_sectors = pivot.mean().sort_values(ascending=False).head(8).index.tolist()
    other = pivot.drop(columns=top_sectors, errors="ignore").sum(axis=1)
    plot_df = pivot[top_sectors].copy()
    if (other > 0).any():
        plot_df["Other"] = other
    fig, ax = _new_figure(height=5.8)
    ax.stackplot(plot_df.index, [plot_df[col] for col in plot_df.columns], labels=plot_df.columns)
    ax.set_title("Sector exposure through time")
    ax.set_ylabel("Portfolio weight")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.14))
    _save_chart(fig, charts_dir, charts, "sector_exposure")


def _chart_watchlist_map(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    if not _has_columns(df, ["ticker", "valuation_score", "conviction_score"]):
        return
    plot_df = df.dropna(subset=["valuation_score", "conviction_score"]).copy()
    if plot_df.empty:
        return
    fig, ax = _new_figure()
    ax.scatter(plot_df["valuation_score"], plot_df["conviction_score"], s=70, color="#1f7a8c", alpha=0.78)
    for _, row in plot_df.sort_values("conviction_score", ascending=False).head(12).iterrows():
        ax.annotate(str(row["ticker"]), (row["valuation_score"], row["conviction_score"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Watchlist: valuation versus conviction")
    ax.set_xlabel("Valuation score")
    ax.set_ylabel("Conviction score")
    _save_chart(fig, charts_dir, charts, "watchlist_map")


def _chart_thesis_persistence(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    if not _has_columns(df, ["ticker", "thesis_persistence_score"]):
        return
    summary = df.groupby("ticker", as_index=False)["thesis_persistence_score"].mean().sort_values("thesis_persistence_score")
    fig, ax = _new_figure(height=5.8)
    ax.barh(summary["ticker"], summary["thesis_persistence_score"], color="#2f4b7c")
    ax.set_title("Average thesis persistence by holding")
    ax.set_xlabel("Persistence score")
    ax.set_xlim(0, max(1, float(summary["thesis_persistence_score"].max())))
    _save_chart(fig, charts_dir, charts, "thesis_persistence")


def _chart_position_performance(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    required = ["ticker", "holding_days", "total_return", "annualized_return", "benchmark_annualized_return", "excess_total_return"]
    if not _has_columns(df, required):
        return
    plot_df = df.copy()
    for column in required[1:]:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce").fillna(0)
    rows = []
    for ticker, group in plot_df.groupby("ticker"):
        weights = group["holding_days"].clip(lower=1)
        rows.append({
            "ticker": ticker,
            "total_return": float((1 + group["total_return"]).prod() - 1),
            "annualized_return": float((group["annualized_return"] * weights).sum() / weights.sum()),
            "benchmark_annualized_return": float((group["benchmark_annualized_return"] * weights).sum() / weights.sum()),
            "excess_total_return": float(group["excess_total_return"].sum()),
        })
    if not rows:
        return
    summary = (
        pd.DataFrame(rows)
        .sort_values("excess_total_return", ascending=False)
        .head(20)
        .sort_values("excess_total_return")
    )
    y = range(len(summary))
    fig, ax = _new_figure(height=max(5.8, len(summary) * 0.36))
    bar_height = 0.24
    ax.barh([i - bar_height for i in y], summary["total_return"], height=bar_height, color="#1f7a8c", label="Stock total return")
    ax.barh(y, summary["annualized_return"], height=bar_height, color="#665191", label="Stock annualized return")
    ax.barh([i + bar_height for i in y], summary["benchmark_annualized_return"], height=bar_height, color="#b23a48", label="Benchmark annualized return")
    ax.axvline(0, color="#172026", linewidth=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(summary["ticker"])
    ax.set_title("Return earned by stock versus benchmark during holding period")
    ax.set_xlabel("Return")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower right", fontsize=8)
    _save_chart(fig, charts_dir, charts, "position_performance_bars")


def _chart_feature_importance(charts_dir: Path, charts: dict[str, str]) -> None:
    path = PROCESSED_DIR / "model_explainability.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for component, features in payload.get("feature_importance", {}).items():
        if isinstance(features, dict):
            for feature, importance in features.items():
                rows.append({"label": f"{component}: {feature}", "importance": importance})
        elif isinstance(features, (int, float)):
            rows.append({"label": component, "importance": features})
    if not rows:
        return
    plot_df = pd.DataFrame(rows).sort_values("importance", ascending=False).head(18).sort_values("importance")
    fig, ax = _new_figure(height=6.4)
    ax.barh(plot_df["label"], plot_df["importance"], color="#665191")
    ax.set_title("Top model feature importances")
    ax.set_xlabel("Importance")
    _save_chart(fig, charts_dir, charts, "feature_importance")


def _chart_position_pages(charts_dir: Path, charts: dict[str, str], df: pd.DataFrame) -> None:
    required = ["date", "ticker", "current_conviction_score", "hybrid_weight"]
    if not _has_columns(df, required):
        return
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    for ticker, ticker_df in plot_df.groupby("ticker"):
        fig, ax1 = _new_figure(height=4.8)
        ax2 = ax1.twinx()
        ax1.plot(ticker_df["date"], ticker_df["current_conviction_score"], color="#1f7a8c", marker="o", label="Conviction")
        ax2.plot(ticker_df["date"], ticker_df["hybrid_weight"], color="#b23a48", marker="s", label="Weight")
        ax1.set_title(f"{ticker}: conviction and weight")
        ax1.set_ylabel("Conviction")
        ax2.set_ylabel("Hybrid weight")
        ax2.yaxis.set_major_formatter(PercentFormatter(1))
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")
        _save_chart(fig, charts_dir, charts, f"position_{ticker}")


def _new_figure(width: float = 10.8, height: float = 5.2):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color="#e6ecec", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def _save_chart(fig, charts_dir: Path, charts: dict[str, str], name: str) -> None:
    fig.autofmt_xdate()
    fig.tight_layout()
    output = charts_dir / f"{name}.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    charts[name] = f"charts/{name}.png"


def _figure(src: str | None, caption: str) -> str:
    if not src:
        return ""
    escaped_src = html.escape(src)
    escaped_caption = html.escape(caption)
    return f'<figure><img src="{escaped_src}" alt="{escaped_caption}"><figcaption>{escaped_caption}</figcaption></figure>'


def _has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    return df is not None and not df.empty and all(column in df.columns for column in columns)


def _first_existing(df: pd.DataFrame, columns: list[str]) -> str | None:
    for column in columns:
        if column in df.columns:
            return column
    numeric_cols = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    return numeric_cols[0] if numeric_cols else None


def _write_results_explainer(run_dir: Path, viewer_dir: Path, charts: dict[str, str], tables: dict[str, pd.DataFrame]) -> None:
    csv_descriptions = {
        "executive_summary.csv": "Resumen de una fila con las metricas clave de la ejecucion.",
        "current_portfolio.csv": "Cartera actual limpia: pesos, scores, tesis, estado y motivo de seguimiento.",
        "action_journal.csv": "Diario unificado de compras y ventas con razon, scores y resultado de operaciones cerradas.",
        "top_opportunities_latest.csv": "Mejores oportunidades del universo en la ultima fecha, en formato resumido.",
        "strategy_learning_log.csv": "Resumen de patrones de entradas/salidas que funcionaron o fallaron frente al benchmark.",
        "improvement_backlog.csv": "Lista automatica de hipotesis de mejora para futuras iteraciones.",
        "watchlist.csv": "Ranking final de oportunidades candidatas, con tesis, catalizador, moat, valoracion y conviccion.",
        "research_ai.csv": "Resumen enriquecido de investigacion por compania, generado por la capa de research automatizado si esta disponible.",
        "portfolio_monthly_summary.json": "Resumen agregado de la simulacion: holdings iniciales/finales, compras, ventas y alpha acumulado.",
        "portfolio_monthly_holdings.csv": "Foto mensual de cada posicion en cartera, incluyendo tesis, persistencia, conviccion y pesos.",
        "portfolio_evolution.csv": "Evolucion mensual de la cartera y estado agregado de las posiciones.",
        "portfolio_decision_log.csv": "Registro de decisiones de compra, mantenimiento o venta con motivos y scores.",
        "portfolio_allocation.csv": "Pesos equal-weight, por conviccion e hibridos usados para dimensionar posiciones.",
        "portfolio_turnover.csv": "Actividad de rotacion por periodo de revision.",
        "position_performance.csv": "Atribucion por lote/accion: retorno total, retorno anualizado, benchmark comparable durante el holding y motivo de salida.",
        "tracking_dashboard.csv": "Tabla ejecutiva mensual con valor de cartera, benchmark, alpha, compras, ventas y tickers mantenidos.",
        "buy_rationale.csv": "Resumen de cada compra con ranking, scores principales, alternativa y tesis que justifico la entrada.",
        "sell_reasons_summary.csv": "Resumen agregado de ventas por categoria de salida.",
        "sector_exposure.csv": "Exposicion sectorial mensual de la cartera con pesos y tickers.",
        "model_walk_forward_diagnostics.csv": "Diagnostico del entrenamiento walk-forward: filas historicas usadas y fechas con fallback.",
        "portfolio_transactions.csv": "Operaciones simuladas de compra y venta con fecha, ticker y motivo.",
        "portfolio_review_diagnostics.csv": "Diagnostico de revision: compara candidatos, holdings existentes y posible reemplazo.",
        "portfolio_vs_benchmark.csv": "Serie principal de rendimiento: retornos mensuales, alpha, valor de cartera y benchmark.",
        "rebalance_report.csv": "Detalle de rebalanceos propuestos o ejecutados en cada revision.",
        "final_report.html": "Informe ejecutivo del experimento con metricas, mejores/peores tesis y conclusiones automaticas.",
    }
    page_descriptions = {
        "index.html": "Vista de entrada al visor con rendimiento, alpha reciente, cartera actual y ultimas transacciones.",
        "current_portfolio.html": "Cartera actual, pesos, scores y tesis de salida.",
        "tracking_dashboard.html": "Seguimiento mensual compacto de performance, alpha, compras, ventas y composicion.",
        "portfolio_vs_benchmark.html": "Comparativa completa entre cartera y benchmark, con curva de valor, alpha mensual y drawdown.",
        "portfolio_evolution.html": "Evolucion de la cartera y soporte tabular para auditar el backtest.",
        "allocation_dashboard.html": "Panel de pesos por ticker, con asignacion actual y deriva historica de la cartera.",
        "watchlist.html": "Mapa de oportunidades: cruza valoracion y conviccion para identificar candidatos atractivos.",
        "top_opportunities.html": "Mejores oportunidades actuales segun el ranking independiente del universo.",
        "strategy_learning.html": "Pistas automaticas para futuras mejoras de reglas, pesos y umbrales.",
        "thesis_persistence.html": "Persistencia de tesis por posicion y por mes.",
        "model_explainability.html": "Importancia de componentes y variables del modelo de scoring.",
        "action_journal.html": "Diario unificado de compras y ventas con resultado economico cuando la posicion esta cerrada.",
        "portfolio_turnover.html": "Rotacion de cartera por periodo.",
        "sector_exposure.html": "Exposicion sectorial de la cartera a traves del tiempo.",
        "position_performance.html": "Rendimiento ganado por accion frente al benchmark durante el periodo real de holding.",
        "buy_rationale.html": "Explicacion tabular de por que se compro cada accion.",
        "sell_reasons.html": "Resumen de categorias de salida y tickers afectados.",
        "decision_log.html": "Bitacora completa de decisiones.",
        "portfolio_review.html": "Diagnostico de revision de cartera, candidatos de reemplazo y razones.",
        "rebalance_report.html": "Detalle tabular del rebalanceo.",
        "research.html": "Tesis de inversion, bull case, bear case, catalizadores y moat.",
        "exit_thesis.html": "Criterios de salida y trazabilidad de ventas o razones de mantenimiento.",
        "audit.html": "Mapa de ficheros pesados de auditoria que no conviene leer de entrada.",
    }
    chart_descriptions = {
        "portfolio_vs_benchmark": "Curva de crecimiento de 1 dolar invertido en la cartera frente a SPY.",
        "period_alpha": "Barras de alpha mensual; positivo significa que la cartera supera al benchmark ese mes.",
        "drawdown": "Caida desde maximos historicos de la cartera, util para medir riesgo vivido.",
        "turnover": "Rotacion o actividad de cartera por periodo, segun las columnas disponibles.",
        "latest_allocation": "Peso hibrido mas reciente de cada posicion.",
        "allocation_drift": "Evolucion historica de los pesos para ver concentracion y cambios.",
        "sector_exposure": "Peso de cartera por sector a lo largo del tiempo.",
        "watchlist_map": "Dispersion valoracion-conviccion de la watchlist; los tickers anotados son los de mayor conviccion.",
        "thesis_persistence": "Persistencia media de tesis por ticker mantenido.",
        "position_performance_bars": "Retorno por accion: total logrado, anualizado logrado y retorno anualizado del benchmark en el mismo periodo de holding.",
        "feature_importance": "Variables mas relevantes del modelo de scoring.",
    }

    lines = [
        "# Explicacion de resultados",
        "",
        f"Run analizado: `{run_dir.name}`.",
        "",
        "## Graficos principales",
        "",
    ]
    for key, src in sorted(charts.items()):
        if key.startswith("position_"):
            continue
        lines.append(f"- `viewer/{src}`: {chart_descriptions.get(key, 'Grafico generado para apoyar el analisis visual del visor.')}")
    lines.extend(["", "## HTML del visor", ""])
    for page in PAGES:
        lines.append(f"- `viewer/{page}`: {page_descriptions.get(page, 'Pagina HTML del visor de resultados.')}")
    position_pages = sorted(path.name for path in viewer_dir.glob("position_*.html"))
    if position_pages:
        lines.append(f"- `viewer/position_*.html`: paginas individuales por posicion ({len(position_pages)} tickers) con tabla historica y grafico de conviccion/peso.")
    lines.extend(["", "## Archivos de resultados", ""])
    for path in sorted(run_dir.glob("*")):
        if path.is_file() and path.name != "expl_results.md":
            description = csv_descriptions.get(path.name, "Artefacto auxiliar generado por el pipeline.")
            extra = ""
            if path.suffix == ".csv" and path.stem in tables:
                df = tables[path.stem]
                extra = f" Contiene {len(df)} filas y {len(df.columns)} columnas."
            lines.append(f"- `{path.name}`: {description}{extra}")
    audit_dir = run_dir / "audit"
    if audit_dir.exists():
        lines.extend(["", "## Archivos de auditoria", ""])
        for path in sorted(audit_dir.glob("*.csv")):
            description = csv_descriptions.get(path.name, "Tabla pesada para trazabilidad y depuracion.")
            df = tables.get(path.stem, pd.DataFrame())
            extra = f" Contiene {len(df)} filas y {len(df.columns)} columnas." if not df.empty else ""
            lines.append(f"- `audit/{path.name}`: {description}{extra}")
    lines.extend([
        "",
        "## Lectura recomendada",
        "",
        "1. Empieza por `viewer/index.html` para ver resultado global y operaciones recientes.",
        "2. Usa `viewer/current_portfolio.html` para ver que hay ahora y por que se mantiene.",
        "3. Usa `viewer/action_journal.html` y `viewer/position_performance.html` para auditar compras, ventas y ganancias.",
        "4. Revisa `viewer/buy_rationale.html` y `viewer/sell_reasons.html` para entender por que entra o sale capital.",
        "5. Consulta `viewer/watchlist.html` y `viewer/top_opportunities.html` para candidatos futuros.",
        "6. Revisa `viewer/strategy_learning.html` para ver pistas de mejora acumuladas.",
        "7. Entra en `viewer/audit.html` solo si necesitas reconstruir o depurar la simulacion completa.",
    ])
    (run_dir / "expl_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "result_manifest.json").write_text(
        json.dumps(_artifact_manifest(run_dir, charts, tables, csv_descriptions, page_descriptions, chart_descriptions), indent=2, default=str),
        encoding="utf-8",
    )


def _artifact_manifest(
    run_dir: Path,
    charts: dict[str, str],
    tables: dict[str, pd.DataFrame],
    csv_descriptions: dict[str, str],
    page_descriptions: dict[str, str],
    chart_descriptions: dict[str, str],
) -> list[dict]:
    rows: list[dict] = []
    first_read = {
        "executive_summary.csv", "current_portfolio.csv", "tracking_dashboard.csv",
        "action_journal.csv", "position_performance.csv", "buy_rationale.csv",
        "sell_reasons_summary.csv", "portfolio_vs_benchmark.csv", "strategy_learning_log.csv",
        "improvement_backlog.csv",
        "index.html", "current_portfolio.html", "action_journal.html",
        "position_performance.html", "portfolio_vs_benchmark.html", "strategy_learning.html",
    }
    for path in sorted(run_dir.glob("*.csv")):
        df = tables.get(path.stem, pd.DataFrame())
        rows.append({
            "path": path.name,
            "kind": "csv",
            "tier": "executive",
            "read_first": path.name in first_read,
            "rows": len(df),
            "columns": len(df.columns) if not df.empty else 0,
            "purpose": csv_descriptions.get(path.name, "Artefacto de resultados."),
        })
    audit_dir = run_dir / "audit"
    if audit_dir.exists():
        for path in sorted(audit_dir.glob("*.csv")):
            df = tables.get(path.stem, pd.DataFrame())
            rows.append({
                "path": f"audit/{path.name}",
                "kind": "csv",
                "tier": "audit",
                "read_first": False,
                "rows": len(df),
                "columns": len(df.columns) if not df.empty else 0,
                "purpose": csv_descriptions.get(path.name, "Tabla pesada para trazabilidad y depuracion."),
            })
    for page in PAGES:
        rows.append({
            "path": f"viewer/{page}",
            "kind": "html",
            "tier": "executive" if page != "audit.html" else "audit",
            "read_first": page in first_read,
            "purpose": page_descriptions.get(page, "Pagina del visor."),
        })
    for key, src in sorted(charts.items()):
        if key.startswith("position_"):
            tier = "detail"
        else:
            tier = "executive"
        rows.append({
            "path": f"viewer/{src}",
            "kind": "png",
            "tier": tier,
            "read_first": key in {"portfolio_vs_benchmark", "period_alpha", "position_performance_bars", "sector_exposure"},
            "purpose": chart_descriptions.get(key, "Grafico de apoyo."),
        })
    return rows
