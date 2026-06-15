"""Generate the required static viewer pages."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from environment import PROCESSED_DIR, Settings


PAGES = [
    "index.html",
    "portfolio_review.html",
    "portfolio_evolution.html",
    "portfolio_vs_benchmark.html",
    "portfolio_turnover.html",
    "decision_log.html",
    "thesis_persistence.html",
    "allocation_dashboard.html",
    "watchlist.html",
    "rebalance_report.html",
    "research.html",
    "research_ai.html",
    "exit_thesis.html",
    "model_explainability.html",
]


def build_viewer(settings: Settings) -> Path:
    run_dir = settings.run_dir
    viewer_dir = run_dir / "viewer"
    viewer_dir.mkdir(parents=True, exist_ok=True)
    tables = {path.stem: pd.read_csv(path) for path in run_dir.glob("*.csv")}
    if "watchlist" not in tables and (PROCESSED_DIR / "watchlist.parquet").exists():
        tables["watchlist"] = pd.read_parquet(PROCESSED_DIR / "watchlist.parquet")

    nav = " ".join(f'<a href="{page}">{page.replace(".html", "")}</a>' for page in PAGES)
    _remove_stale_pages(viewer_dir)
    for page in PAGES:
        body = _page_body(page, tables)
        (viewer_dir / page).write_text(_layout(page, nav, body), encoding="utf-8")

    holdings = tables.get("portfolio_monthly_holdings", pd.DataFrame())
    for ticker in sorted(holdings.get("ticker", pd.Series(dtype=str)).dropna().unique()):
        ticker_rows = holdings[holdings["ticker"] == ticker]
        body = f"<h1>{html.escape(ticker)}</h1>{_table(ticker_rows)}"
        (viewer_dir / f"position_{ticker}.html").write_text(_layout(ticker, nav, body), encoding="utf-8")
    return viewer_dir


def _page_body(page: str, tables: dict[str, pd.DataFrame]) -> str:
    if page == "index.html":
        vs = tables.get("portfolio_vs_benchmark", pd.DataFrame())
        return (
            "<h1>GARP AI Portfolio</h1>"
            + _line_chart(vs, "portfolio_value", "benchmark_value")
            + "<h2>Current Portfolio</h2>"
            + _table(tables.get("portfolio_evolution", pd.DataFrame()).tail(1))
            + "<h2>Transactions</h2>"
            + _table(tables.get("portfolio_transactions", pd.DataFrame()).tail(20))
        )
    if page == "portfolio_review.html":
        cols = [
            "date", "rank", "ticker", "in_portfolio", "thesis_rank_score",
            "weakest_holding", "score_advantage_vs_weakest", "replacement_candidate", "reason",
        ]
        return "<h1>Portfolio Review</h1>" + _table(_select(tables.get("portfolio_review_diagnostics", pd.DataFrame()), cols))
    if page == "portfolio_evolution.html":
        return "<h1>Portfolio Evolution</h1>" + _table(tables.get("portfolio_evolution", pd.DataFrame()))
    if page == "portfolio_turnover.html":
        return "<h1>Turnover</h1>" + _table(tables.get("portfolio_turnover", pd.DataFrame()))
    if page == "portfolio_vs_benchmark.html":
        vs = tables.get("portfolio_vs_benchmark", pd.DataFrame())
        return "<h1>Portfolio Vs Benchmark</h1>" + _line_chart(vs, "portfolio_value", "benchmark_value") + _table(vs)
    if page == "watchlist.html":
        return "<h1>Watchlist</h1>" + _table(tables.get("watchlist", pd.DataFrame()))
    if page == "rebalance_report.html":
        return "<h1>Rebalance Report</h1>" + _table(tables.get("rebalance_report", pd.DataFrame()))
    if page == "research.html":
        cols = ["date", "ticker", "investment_thesis", "bull_thesis", "bear_thesis", "catalyst", "moat_analysis"]
        return "<h1>Research</h1>" + _table(_select(tables.get("portfolio_monthly_holdings", pd.DataFrame()), cols))
    if page == "research_ai.html":
        return "<h1>Research AI</h1>" + _research_ai(settings=None, tables=tables)
    if page == "exit_thesis.html":
        cols = ["date", "ticker", "current_thesis_state", "exit_thesis", "reason"]
        decisions = tables.get("portfolio_decision_log", pd.DataFrame())
        tx = tables.get("portfolio_transactions", pd.DataFrame())
        return "<h1>Exit Thesis</h1>" + _table(_select(tx, ["date", "ticker", "action", "reason", "exit_thesis"])) + _table(_select(decisions, cols))
    if page == "allocation_dashboard.html":
        return "<h1>Position Sizing</h1>" + _table(tables.get("portfolio_allocation", pd.DataFrame()))
    if page == "model_explainability.html":
        return "<h1>Model Explainability</h1>" + _explainability()
    if page == "decision_log.html":
        return "<h1>Decision Log</h1>" + _table(tables.get("portfolio_decision_log", pd.DataFrame()))
    if page == "thesis_persistence.html":
        cols = ["date", "ticker", "current_thesis_state", "thesis_persistence_score", "months_thesis_intact"]
        return "<h1>Thesis Persistence</h1>" + _table(_select(tables.get("portfolio_monthly_holdings", pd.DataFrame()), cols))
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
.chart {{ width: 100%; max-width: 100%; height: auto; background: white; border: 1px solid #d7dddd; margin: 18px 0; }}
.portfolio-line {{ fill: none; stroke: #1f7a8c; stroke-width: 3; }}
.benchmark-line {{ fill: none; stroke: #b23a48; stroke-width: 3; }}
.axis {{ stroke: #c8d0d0; stroke-width: 1; }}
.legend {{ font-size: 13px; fill: #172026; }}
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


def _line_chart(df: pd.DataFrame, portfolio_col: str, benchmark_col: str) -> str:
    if df.empty or portfolio_col not in df.columns or benchmark_col not in df.columns:
        return "<p>No chart data available.</p>"
    width, height = 980, 360
    pad = 42
    values = pd.concat([df[portfolio_col], df[benchmark_col]]).dropna()
    if values.empty:
        return "<p>No chart data available.</p>"
    y_min = float(values.min())
    y_max = float(values.max())
    if y_min == y_max:
        y_min -= 0.01
        y_max += 0.01

    def points(series: pd.Series) -> str:
        coords = []
        n = max(len(series) - 1, 1)
        for i, value in enumerate(series):
            x = pad + (width - 2 * pad) * i / n
            y = height - pad - (height - 2 * pad) * (float(value) - y_min) / (y_max - y_min)
            coords.append(f"{x:.1f},{y:.1f}")
        return " ".join(coords)

    latest_portfolio = float(df[portfolio_col].iloc[-1])
    latest_benchmark = float(df[benchmark_col].iloc[-1])
    return f"""
<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Portfolio versus benchmark chart">
  <line class="axis" x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" />
  <line class="axis" x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" />
  <polyline class="portfolio-line" points="{points(df[portfolio_col])}" />
  <polyline class="benchmark-line" points="{points(df[benchmark_col])}" />
  <text class="legend" x="{pad}" y="24">Portfolio: {latest_portfolio:.2f}</text>
  <text class="legend" x="{pad + 180}" y="24">Benchmark: {latest_benchmark:.2f}</text>
</svg>"""


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
