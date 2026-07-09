"""Shared HTML helpers for the static viewer."""

from __future__ import annotations

import html

import pandas as pd


PAGES = [
    "index.html",
    "dashboard.html",
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

PAGE_GROUPS = {
    "Principal": [
        "dashboard.html",
        "current_portfolio.html",
        "portfolio_vs_benchmark.html",
        "position_performance.html",
        "top_opportunities.html",
    ],
    "Secundario / Auditoria": [
        "index.html",
        "tracking_dashboard.html",
        "action_journal.html",
        "buy_rationale.html",
        "sell_reasons.html",
        "sector_exposure.html",
        "allocation_dashboard.html",
        "watchlist.html",
        "strategy_learning.html",
        "model_explainability.html",
        "audit.html",
    ],
}


def layout(title: str, nav: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; color: #172026; background: #f7f8f8; }}
nav {{ position: sticky; top: 0; display: flex; gap: 16px; align-items: center; overflow-x: auto; padding: 12px 18px; background: #172026; }}
nav .nav-group {{ display: flex; gap: 10px; align-items: center; }}
nav .nav-sep {{ width: 1px; align-self: stretch; background: #3a464a; }}
nav a {{ color: white; text-decoration: none; white-space: nowrap; font-size: 13px; }}
nav .nav-group.secondary a {{ color: #a9b6b9; }}
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


PCT_COLUMNS = {
    "hybrid_weight", "equal_weight", "conviction_weight", "risk_adjusted_weight", "weight",
    "total_return", "annualized_return", "benchmark_total_return", "benchmark_annualized_return",
    "excess_total_return", "period_alpha", "cumulative_alpha", "portfolio_period_return",
    "portfolio_gross_period_return", "benchmark_period_return", "transaction_cost_drag",
    "avg_excess_return", "avg_total_return", "win_rate_vs_benchmark", "closed_position_win_rate_vs_benchmark",
    "information_ratio", "tracking_error_annualized", "monthly_turnover", "annual_turnover",
}
X_COLUMNS = {"portfolio_value", "portfolio_gross_value", "benchmark_value"}


def format_for_html(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    formatted = df.copy()
    for column in formatted.columns:
        if column in PCT_COLUMNS:
            formatted[column] = formatted[column].map(lambda v: "" if pd.isna(v) else f"{v:.2%}")
        elif column in X_COLUMNS:
            formatted[column] = formatted[column].map(lambda v: "" if pd.isna(v) else f"{v:.2f}x")
        elif pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    return formatted


def table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p>No hay datos disponibles.</p>"
    return format_for_html(df).to_html(index=False, escape=True)


def select(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df[[column for column in columns if column in df.columns]]


def figure(src: str | None, caption: str) -> str:
    if not src:
        return ""
    escaped_src = html.escape(src)
    escaped_caption = html.escape(caption)
    return f'<figure><img src="{escaped_src}" alt="{escaped_caption}"><figcaption>{escaped_caption}</figcaption></figure>'
