"""Generate the final experimental report for the TFM."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from environment import PROCESSED_DIR, Settings


def build_final_report(settings: Settings) -> Path:
    run_dir = settings.run_dir
    vs = _read_csv(run_dir / "portfolio_vs_benchmark.csv")
    transactions = _read_csv(run_dir / "portfolio_transactions.csv")
    holdings = _read_csv(run_dir / "portfolio_monthly_holdings.csv")
    decisions = _read_csv(run_dir / "portfolio_decision_log.csv")
    explainability = _read_json(PROCESSED_DIR / "model_explainability.json")

    metrics = _metrics(vs)
    body = (
        "<h1>Final GARP AI Report</h1>"
        + _executive_summary(metrics)
        + "<h2>Performance Metrics</h2>"
        + _table(pd.DataFrame([metrics]))
        + "<h2>Portfolio Vs Benchmark</h2>"
        + _table(vs.tail(24))
        + "<h2>Best Decisions</h2>"
        + _table(_best_decisions(decisions))
        + "<h2>Worst Decisions</h2>"
        + _table(_worst_decisions(decisions))
        + "<h2>Best Theses</h2>"
        + _table(_best_theses(holdings))
        + "<h2>Weakest Theses</h2>"
        + _table(_worst_theses(holdings))
        + "<h2>Opportunity Classification</h2>"
        + _table(_classification_summary(holdings))
        + "<h2>Component Importance</h2>"
        + _table(_component_importance(explainability))
        + "<h2>Feature Importance</h2>"
        + _table(_feature_importance(explainability))
        + "<h2>Automatic Conclusions</h2>"
        + _conclusions(metrics, holdings, transactions)
    )
    path = run_dir / "final_report.html"
    path.write_text(_layout(body), encoding="utf-8")
    return path


def _metrics(vs: pd.DataFrame) -> dict:
    if vs.empty:
        return {"CAGR": 0, "Sharpe": 0, "Sortino": 0, "Max Drawdown": 0, "Alpha": 0}
    returns = vs["portfolio_period_return"].fillna(0)
    benchmark_returns = vs["benchmark_period_return"].fillna(0)
    years = max((pd.to_datetime(vs["date"]).max() - pd.to_datetime(vs["date"]).min()).days / 365.25, 1 / 12)
    ending = float(vs["portfolio_value"].iloc[-1])
    benchmark_ending = float(vs.get("benchmark_value", pd.Series([1])).iloc[-1])
    downside = returns[returns < 0]
    return {
        "CAGR": ending ** (1 / years) - 1,
        "Benchmark CAGR": benchmark_ending ** (1 / years) - 1,
        "Sharpe": _annualized_ratio(returns),
        "Sortino": _annualized_ratio(returns, downside_only=True),
        "Max Drawdown": _max_drawdown(vs["portfolio_value"]),
        "Alpha": (ending - 1) - (benchmark_ending - 1),
        "Average Period Alpha": float((returns - benchmark_returns).mean()),
    }


def _annualized_ratio(returns: pd.Series, downside_only: bool = False) -> float:
    series = returns[returns < 0] if downside_only else returns
    denom = float(series.std())
    if denom == 0 or pd.isna(denom):
        return 0.0
    return float(returns.mean() / denom * (12 ** 0.5))


def _max_drawdown(values: pd.Series) -> float:
    running_max = values.cummax()
    drawdown = values / running_max - 1
    return float(drawdown.min())


def _best_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    return decisions.sort_values(["conviction_score", "thesis_persistence_score"], ascending=False).head(10)


def _worst_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    return decisions.sort_values(["conviction_score", "thesis_persistence_score"], ascending=True).head(10)


def _best_theses(holdings: pd.DataFrame) -> pd.DataFrame:
    cols = ["date", "ticker", "current_thesis_state", "current_conviction_score", "thesis_persistence_score", "investment_thesis"]
    return _select(holdings.sort_values(["thesis_persistence_score", "current_conviction_score"], ascending=False), cols).head(10)


def _worst_theses(holdings: pd.DataFrame) -> pd.DataFrame:
    cols = ["date", "ticker", "current_thesis_state", "current_conviction_score", "thesis_persistence_score", "current_exit_thesis"]
    return _select(holdings.sort_values(["thesis_persistence_score", "current_conviction_score"], ascending=True), cols).head(10)


def _classification_summary(holdings: pd.DataFrame) -> pd.DataFrame:
    if holdings.empty or "opportunity_type_original" not in holdings.columns:
        return pd.DataFrame()
    return holdings.groupby("opportunity_type_original").agg(
        observations=("ticker", "count"),
        avg_conviction=("current_conviction_score", "mean"),
        avg_persistence=("thesis_persistence_score", "mean"),
    ).reset_index().sort_values("avg_persistence", ascending=False)


def _component_importance(payload: dict) -> pd.DataFrame:
    formula = payload.get("garp_score_formula", {})
    return pd.DataFrame([{"component": key, "weight": value} for key, value in formula.items()])


def _feature_importance(payload: dict) -> pd.DataFrame:
    rows = []
    for component, features in payload.get("feature_importance", {}).items():
        if isinstance(features, dict):
            for feature, importance in features.items():
                rows.append({"component": component, "feature": feature, "importance": importance})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).head(30) if rows else pd.DataFrame()


def _executive_summary(metrics: dict) -> str:
    alpha = metrics.get("Alpha", 0)
    result = "outperformed" if alpha > 0 else "underperformed"
    return f"<p>The systematic GARP business manager {result} the benchmark by alpha {alpha:.2%} in this run.</p>"


def _conclusions(metrics: dict, holdings: pd.DataFrame, transactions: pd.DataFrame) -> str:
    turnover = int((transactions.get("action", pd.Series(dtype=str)) == "SELL").sum()) if not transactions.empty else 0
    avg_persistence = holdings.get("thesis_persistence_score", pd.Series([0])).mean() if not holdings.empty else 0
    return (
        "<ul>"
        f"<li>Alpha versus benchmark: {metrics.get('Alpha', 0):.2%}.</li>"
        f"<li>Average thesis persistence: {avg_persistence:.2f}.</li>"
        f"<li>Sell decisions: {turnover}, indicating the portfolio is managed as a live thesis book rather than a short-term ranking list.</li>"
        "<li>The architecture separates buy-today decisions from hold decisions, which is essential for long-term GARP portfolio management.</li>"
        "</ul>"
    )


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _select(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[column for column in columns if column in df.columns]] if not df.empty else df


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No data available.</p>"
    return df.to_html(index=False, escape=True)


def _layout(body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Final GARP AI Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; color: #172026; background: #f7f8f8; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
h1 {{ font-size: 30px; }}
h2 {{ margin-top: 30px; font-size: 20px; }}
table {{ width: 100%; border-collapse: collapse; margin: 14px 0; background: white; border: 1px solid #d7dddd; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #e7ebeb; text-align: left; font-size: 13px; vertical-align: top; }}
th {{ background: #eef2f2; }}
</style>
</head>
<body><main>{body}</main></body>
</html>"""
