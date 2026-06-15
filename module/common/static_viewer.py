"""Static HTML viewer generation for GARP run and portfolio-review artifacts.

The viewer is presentation-only: it reads existing CSV/JSON/Markdown artifacts and
never recalculates training, ranking, backtest or thesis metrics.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from environment import VIEWER_OFFLINE_MODE
except Exception:  # pragma: no cover - import fallback for isolated tests
    VIEWER_OFFLINE_MODE = False


PAGES = [
    ("run_summary.html", "Run Summary"),
    ("portfolio_review.html", "Portfolio Review"),
    ("portfolio_health.html", "Portfolio Health"),
    ("portfolio_evolution.html", "Portfolio Evolution"),
    ("portfolio_vs_benchmark.html", "Portfolio vs Benchmark"),
    ("portfolio_timeline.html", "Portfolio Timeline"),
    ("portfolio_lifecycle.html", "Portfolio Lifecycle"),
    ("portfolio_turnover.html", "Portfolio Turnover"),
    ("decision_log.html", "Decision Log"),
    ("thesis_persistence.html", "Thesis Persistence"),
    ("hold_winners.html", "Hold Winners"),
    ("allocation_dashboard.html", "Allocation"),
    ("thesis_history.html", "Thesis History"),
    ("thesis_events.html", "Thesis Events"),
    ("thesis_change_report.html", "Thesis Changes"),
    ("alerts.html", "Alerts"),
    ("opportunity_cost.html", "Opportunity Cost"),
    ("watchlist.html", "Watchlist"),
]


def _find_first(root: Path, names: Iterable[str]) -> Path | None:
    wanted = set(names)
    for p in root.rglob("*"):
        if p.is_file() and p.name in wanted:
            return p
    return None


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_id(value: object) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value))[:80]


def _ensure_offline_assets(viewer_dir: Path) -> str:
    assets = viewer_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "viewer.css").write_text("table{border-collapse:collapse}td,th{border:1px solid #e5e7eb;padding:4px}.dataTables_wrapper{overflow-x:auto}", encoding="utf-8")
    (assets / "viewer.js").write_text(
        "window.Plotly=window.Plotly||{newPlot:function(id){var e=document.getElementById(id);if(e){e.innerHTML='<p>Plotly asset unavailable in offline placeholder mode.</p>';}}};"
        "if(!window.jQuery){window.jQuery=function(){return {DataTable:function(){}}};window.$=window.jQuery;}",
        encoding="utf-8",
    )
    return '<link rel="stylesheet" href="assets/viewer.css"><script src="assets/viewer.js"></script>'


def _asset_tags(viewer_dir: Path) -> str:
    if bool(VIEWER_OFFLINE_MODE):
        return _ensure_offline_assets(viewer_dir)
    return """<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css">
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>"""


def _html_shell(title: str, body: str, run_root: Path, viewer_dir: Path, pages: list[tuple[str, str]] | None = None) -> str:
    nav_items = pages or PAGES
    nav = "".join(f'<a href="{href}">{label}</a>' for href, label in nav_items)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  {_asset_tags(viewer_dir)}
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #f6f7fb; color: #172033; }}
    header {{ background: #111827; color: white; padding: 18px 28px; }}
    header h1 {{ margin: 0; font-size: 24px; }}
    nav {{ display: flex; gap: 10px; flex-wrap: wrap; padding: 12px 28px; background: #1f2937; }}
    nav a {{ color: #e5e7eb; text-decoration: none; padding: 7px 10px; border-radius: 6px; background: #374151; }}
    main {{ padding: 24px 28px 60px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 18px; }}
    .card {{ background: white; border-radius: 10px; padding: 16px; box-shadow: 0 1px 4px rgba(15,23,42,.08); }}
    .metric {{ font-size: 28px; font-weight: 700; }}
    .muted {{ color: #6b7280; }}
    .pill {{ display: inline-block; padding: 3px 8px; border-radius: 999px; background: #e5e7eb; margin: 2px; }}
    .Critical {{ background: #fee2e2; color: #991b1b; }} .High {{ background: #ffedd5; color: #9a3412; }} .Medium {{ background: #fef9c3; color: #854d0e; }} .Low {{ background: #dcfce7; color: #166534; }}
    table.dataTable {{ background: white; }}
    pre {{ white-space: pre-wrap; background: #111827; color: #e5e7eb; padding: 12px; border-radius: 8px; }}
  </style>
</head>
<body>
<header><h1>{html.escape(title)}</h1><div class="muted">Source: {html.escape(str(run_root))}</div></header>
<nav>{nav}</nav>
<main>{body}</main>
<script>$(document).ready(function(){{ $('table.viewer-table').DataTable({{pageLength: 25, scrollX: true}}); }});</script>
</body>
</html>"""


def _write(path: Path, title: str, body: str, run_root: Path) -> None:
    path.write_text(_html_shell(title, body, run_root, path.parent), encoding="utf-8")


def _table(df: pd.DataFrame, *, columns: list[str] | None = None, max_rows: int = 500) -> str:
    if df.empty:
        return '<div class="card muted">No artifact found for this section.</div>'
    view = df.copy()
    if columns:
        view = view[[c for c in columns if c in view.columns]]
    if len(view) > max_rows:
        view = view.head(max_rows)
    return view.to_html(index=False, classes="viewer-table", escape=True)


def _cards(metrics: dict[str, object]) -> str:
    return '<div class="grid">' + ''.join(
        f'<div class="card"><div class="muted">{html.escape(str(k))}</div><div class="metric">{html.escape(str(v))}</div></div>'
        for k, v in metrics.items()
    ) + '</div>'


def _plot_script(div_id: str, traces: list[dict], layout: dict | None = None) -> str:
    return f"<div class='card'><div id='{div_id}' style='height:420px'></div></div><script>Plotly.newPlot('{div_id}', {json.dumps(traces)}, {json.dumps(layout or {})}, {{responsive:true}});</script>"


def _series_plot(df: pd.DataFrame, ticker_col: str, x_col: str, y_cols: list[str], title: str) -> str:
    if df.empty or ticker_col not in df.columns or x_col not in df.columns:
        return '<div class="card muted">No time-series artifact found.</div>'
    traces = []
    for ticker, group in df.groupby(ticker_col):
        for y in y_cols:
            if y in group.columns:
                traces.append({"x": group[x_col].astype(str).tolist(), "y": pd.to_numeric(group[y], errors="coerce").tolist(), "mode": "lines+markers", "name": f"{ticker} {y}"})
    return _plot_script("plot_" + _safe_id(title), traces, {"title": title, "xaxis": {"title": x_col}, "yaxis": {"title": "score"}})


def _bar_plot(df: pd.DataFrame, col: str, title: str) -> str:
    if df.empty or col not in df.columns:
        return '<div class="card muted">No distribution artifact found.</div>'
    counts = df[col].fillna("Unknown").astype(str).value_counts().sort_values(ascending=False)
    return _plot_script("bar_" + _safe_id(col), [{"x": counts.index.tolist(), "y": counts.values.tolist(), "type": "bar"}], {"title": title})


def _hist_plot(df: pd.DataFrame, col: str, title: str) -> str:
    if df.empty or col not in df.columns:
        return '<div class="card muted">No score distribution artifact found.</div>'
    values = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
    return _plot_script("hist_" + _safe_id(col), [{"x": values, "type": "histogram", "nbinsx": 20}], {"title": title})


def _build_index(viewer_dir: Path, run_root: Path, available_pages: list[tuple[str, str]]) -> None:
    cards = ''.join(f'<div class="card"><h3><a href="{href}">{label}</a></h3><p class="muted">Open {label.lower()}.</p></div>' for href, label in available_pages)
    _write(viewer_dir / "index.html", "GARP Results Viewer", f'<div class="grid">{cards}</div>', run_root)


def _build_run_summary(viewer_dir: Path, run_root: Path, positions: pd.DataFrame, summary: dict) -> None:
    metrics = {
        "positions": len(positions) if not positions.empty else summary.get("positions_reviewed", "-"),
        "avg health": round(float(summary.get("average_position_health_score", 0) or 0), 1) if summary else "-",
        "critical": len(positions[positions.get("review_priority", pd.Series(dtype=str)).eq("Critical")]) if not positions.empty and "review_priority" in positions else 0,
        "best opportunity": summary.get("best_new_opportunity", "-") if summary else "-",
    }
    body = _cards(metrics)
    body += _bar_plot(positions, "opportunity_type", "Opportunity Types")
    body += _hist_plot(positions, "thesis_score", "Thesis Score Distribution")
    body += '<h2>Top Picks / Positions</h2>' + _table(positions.sort_values("thesis_score", ascending=False) if "thesis_score" in positions else positions, columns=["ticker", "thesis_score", "conviction_score", "position_health_score", "opportunity_type", "review_priority"], max_rows=50)
    _write(viewer_dir / "run_summary.html", "Run Summary", body, run_root)


def _build_portfolio_review(viewer_dir: Path, run_root: Path, positions: pd.DataFrame) -> None:
    body = _cards({
        "positions": len(positions),
        "strong/buy": int(positions.get("buy_hold_sell_rating", pd.Series(dtype=str)).isin(["Strong Buy", "Buy"]).sum()) if not positions.empty else 0,
        "review/reduce/sell": int(positions.get("buy_hold_sell_rating", pd.Series(dtype=str)).isin(["Review", "Reduce", "Sell"]).sum()) if not positions.empty else 0,
    })
    body += _table(positions, columns=["ticker", "conviction_score", "thesis_score", "position_health_score", "valuation_status", "buy_hold_sell_rating", "review_priority", "exit_score", "exit_reason"])
    _write(viewer_dir / "portfolio_review.html", "Portfolio Review", body, run_root)


def _build_portfolio_health(viewer_dir: Path, run_root: Path, positions: pd.DataFrame) -> None:
    body = _bar_plot(positions, "review_priority", "Review Priority")
    body += _bar_plot(positions, "thesis_status", "Thesis Status")
    body += _hist_plot(positions, "position_health_score", "Position Health Distribution")
    body += '<h2>Weak / Critical Positions</h2>' + _table(positions.sort_values("position_health_score") if "position_health_score" in positions else positions, columns=["ticker", "position_health_score", "conviction_score", "thesis_status", "review_priority", "action_recommended", "exit_reason"])
    _write(viewer_dir / "portfolio_health.html", "Portfolio Health", body, run_root)


def _build_thesis_history(viewer_dir: Path, run_root: Path, history: pd.DataFrame) -> None:
    body = _series_plot(history, "ticker", "date", ["thesis_score", "conviction_score", "position_health_score", "moat_proxy_score", "catalyst_score", "expectation_gap_score"], "Thesis History")
    body += _table(history)
    _write(viewer_dir / "thesis_history.html", "Thesis History", body, run_root)


def _build_thesis_events(viewer_dir: Path, run_root: Path, events: pd.DataFrame) -> None:
    body = _bar_plot(events.assign(event=events.get("thesis_events", pd.Series(dtype=str)).astype(str).str.split("; ")).explode("event") if not events.empty else events, "event", "Thesis Event Counts")
    body += _table(events.sort_values("date", ascending=False) if "date" in events else events)
    _write(viewer_dir / "thesis_events.html", "Thesis Events Timeline", body, run_root)


def _build_opportunity_cost(viewer_dir: Path, run_root: Path, positions: pd.DataFrame, opportunities: pd.DataFrame) -> None:
    body = '<h2>Best Alternatives</h2>' + _table(opportunities, max_rows=100)
    replaceable = positions[positions.get("opportunity_cost_flag", pd.Series(False, index=positions.index)).astype(bool)] if not positions.empty else positions
    body += '<h2>Positions with Opportunity Cost Flag</h2>' + _table(replaceable, columns=["ticker", "thesis_score", "best_alternative_ticker", "best_alternative_score", "opportunity_cost_flag", "review_priority"])
    _write(viewer_dir / "opportunity_cost.html", "Opportunity Cost", body, run_root)


def _build_watchlist(viewer_dir: Path, run_root: Path, opportunities: pd.DataFrame, positions: pd.DataFrame) -> None:
    source = opportunities if not opportunities.empty else positions
    body = _table(source.sort_values("thesis_score", ascending=False) if "thesis_score" in source else source, max_rows=150)
    body += _bar_plot(source, "opportunity_type", "Watchlist Opportunity Types")
    _write(viewer_dir / "watchlist.html", "Watchlist", body, run_root)


def _build_position_pages(viewer_dir: Path, run_root: Path, positions: pd.DataFrame, history: pd.DataFrame) -> list[tuple[str, str]]:
    pages: list[tuple[str, str]] = []
    if positions.empty or "ticker" not in positions.columns:
        return pages
    for _, row in positions.iterrows():
        ticker = str(row["ticker"])
        filename = f"position_{_safe_id(ticker)}.html"
        details = pd.DataFrame([row.to_dict()])
        hist = history[history.get("ticker", pd.Series(dtype=str)).astype(str).eq(ticker)] if not history.empty and "ticker" in history else pd.DataFrame()
        body = _cards({
            "conviction": row.get("conviction_score", "-"),
            "health": row.get("position_health_score", "-"),
            "thesis": row.get("thesis_score", "-"),
            "rating": row.get("buy_hold_sell_rating", "-"),
        })
        body += _series_plot(hist, "ticker", "date", ["thesis_score", "conviction_score", "position_health_score", "valuation_score", "moat_proxy_score", "catalyst_score", "expectation_gap_score"], f"{ticker} Thesis History")
        body += f'<p><a href="snapshot_compare_{_safe_id(ticker)}.html">Snapshot comparator</a> · <a href="thesis_radar_{_safe_id(ticker)}.html">Thesis radar</a></p>'
        body += '<h2>Current vs Original Thesis</h2>' + _table(details)
        _write(viewer_dir / filename, f"Position {ticker}", body, run_root)
        _build_snapshot_compare_page(viewer_dir, run_root, row)
        _build_thesis_radar_page(viewer_dir, run_root, row)
        pages.append((filename, f"Position {ticker}"))
    return pages


def _original_scores(row: pd.Series) -> dict[str, float]:
    try:
        payload = json.loads(str(row.get("original_scores_json", "{}") or "{}"))
        return {str(k): float(v) for k, v in payload.items() if pd.notna(v)}
    except Exception:
        return {}


def _comparison_frame(row: pd.Series) -> pd.DataFrame:
    original = _original_scores(row)
    metrics = [
        ("Quality", "quality_score"),
        ("Growth", "growth_score"),
        ("Valuation", "valuation_score"),
        ("Moat", "moat_proxy_score"),
        ("Catalyst", "catalyst_score"),
        ("Risk", "risk_score"),
        ("Expectation Gap", "expectation_gap_score"),
        ("Thesis Score", "thesis_score"),
        ("Position Health", "position_health_score"),
        ("Conviction", "conviction_score"),
    ]
    rows = []
    for label, col in metrics:
        cur = pd.to_numeric(pd.Series([row.get(col, None)]), errors="coerce").iloc[0]
        orig = original.get(col, None)
        if orig is None and col in {"position_health_score", "conviction_score"}:
            orig = np.nan
        diff = float(cur) - float(orig) if pd.notna(cur) and pd.notna(orig) else np.nan
        pct = diff / abs(float(orig)) if pd.notna(diff) and float(orig) != 0 else np.nan
        rows.append({"metric": label, "original": orig, "current": cur, "absolute_diff": diff, "pct_diff": pct})
    return pd.DataFrame(rows)


def _build_snapshot_compare_page(viewer_dir: Path, run_root: Path, row: pd.Series) -> None:
    ticker = str(row.get("ticker", "UNKNOWN"))
    cmp_df = _comparison_frame(row)
    improved = cmp_df[cmp_df["absolute_diff"] > 0.03]["metric"].tolist()
    worsened = cmp_df[cmp_df["absolute_diff"] < -0.03]["metric"].tolist()
    body = _cards({"ticker": ticker, "recommendation": row.get("buy_hold_sell_rating", "-"), "priority": row.get("review_priority", "-"), "exit score": row.get("exit_score", "-")})
    body += "<h2>Snapshot compra vs snapshot actual</h2>" + _table(cmp_df)
    body += f"<div class='card'><b>Improved:</b> {html.escape(', '.join(improved) or '-')}<br><b>Worsened:</b> {html.escape(', '.join(worsened) or '-')}<br><b>Recommendation explanation:</b> {html.escape(str(row.get('current_vs_original_summary', row.get('exit_reason', ''))))}</div>"
    _write(viewer_dir / f"snapshot_compare_{_safe_id(ticker)}.html", f"Snapshot Compare {ticker}", body, run_root)


def _build_thesis_radar_page(viewer_dir: Path, run_root: Path, row: pd.Series) -> None:
    ticker = str(row.get("ticker", "UNKNOWN"))
    cmp_df = _comparison_frame(row)
    radar = cmp_df[cmp_df["metric"].isin(["Quality", "Growth", "Moat", "Catalyst", "Valuation", "Expectation Gap", "Risk"])]
    theta = radar["metric"].tolist()
    original = pd.to_numeric(radar["original"], errors="coerce").fillna(0).tolist()
    current = pd.to_numeric(radar["current"], errors="coerce").fillna(0).tolist()
    traces = [
        {"type": "scatterpolar", "r": original + original[:1], "theta": theta + theta[:1], "fill": "toself", "name": "Original"},
        {"type": "scatterpolar", "r": current + current[:1], "theta": theta + theta[:1], "fill": "toself", "name": "Current"},
    ]
    body = _plot_script("radar_" + _safe_id(ticker), traces, {"title": f"{ticker} Thesis Radar", "polar": {"radialaxis": {"visible": True, "range": [0, 1]}}})
    _write(viewer_dir / f"thesis_radar_{_safe_id(ticker)}.html", f"Thesis Radar {ticker}", body, run_root)


def _build_portfolio_evolution_page(viewer_dir: Path, run_root: Path, evolution: pd.DataFrame) -> None:
    body = _series_plot(evolution, "portfolio", "date", [], "Portfolio Evolution") if False else ""
    if not evolution.empty:
        traces = []
        for col in ["avg_thesis_score", "avg_conviction_score", "avg_position_health_score", "n_improving", "n_intact", "n_maturing", "n_weakening", "n_broken"]:
            if col in evolution.columns:
                traces.append({"x": evolution["date"].astype(str).tolist(), "y": pd.to_numeric(evolution[col], errors="coerce").tolist(), "mode": "lines+markers", "name": col})
        body += _plot_script("portfolio_evolution", traces, {"title": "Portfolio evolution through reviews"})
    body += _table(evolution)
    _write(viewer_dir / "portfolio_evolution.html", "Portfolio Evolution", body, run_root)


def _build_vs_benchmark_page(viewer_dir: Path, run_root: Path, evolution: pd.DataFrame) -> None:
    traces = []
    if not evolution.empty:
        for col in ["portfolio_equity", "benchmark_equity", "alpha_equity"]:
            if col in evolution.columns:
                traces.append({"x": evolution["date"].astype(str).tolist(), "y": pd.to_numeric(evolution[col], errors="coerce").tolist(), "mode": "lines+markers", "name": col})
    body = _plot_script("portfolio_vs_benchmark", traces, {"title": "Portfolio vs Benchmark"}) if traces else '<div class="card muted">No continuous benchmark artifact found.</div>'
    body += _table(evolution, columns=["date", "portfolio_equity", "benchmark_equity", "alpha_equity"])
    _write(viewer_dir / "portfolio_vs_benchmark.html", "Portfolio vs Benchmark", body, run_root)


def _build_allocation_dashboard(viewer_dir: Path, run_root: Path, holdings: pd.DataFrame, positions: pd.DataFrame) -> None:
    source = holdings if not holdings.empty else positions
    body = _bar_plot(source, "sector", "Weight by Sector") if "sector" in source else '<div class="card muted">No sector allocation artifact found.</div>'
    for col in ["industry", "opportunity_type", "thesis_status"]:
        body += _bar_plot(source, col, f"Allocation by {col}") if col in source else ""
    if "conviction_score" in source:
        buckets = source.copy()
        buckets["conviction_bucket"] = pd.cut(pd.to_numeric(buckets["conviction_score"], errors="coerce"), bins=[-1, 40, 60, 75, 100], labels=["Low", "Medium", "High", "Very High"])
        body += _bar_plot(buckets, "conviction_bucket", "Conviction Buckets")
    body += _table(source)
    _write(viewer_dir / "allocation_dashboard.html", "Allocation Dashboard", body, run_root)


def _alerts_from_artifacts(positions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not positions.empty:
        for _, row in positions.iterrows():
            ticker = row.get("ticker", "")
            if row.get("thesis_status") == "Broken":
                rows.append({"severity": "Critical", "ticker": ticker, "alert": "Thesis Broken", "detail": row.get("exit_reason", "")})
            if row.get("thesis_status") == "Weakening":
                rows.append({"severity": "High", "ticker": ticker, "alert": "Thesis Weakening", "detail": row.get("thesis_changes", "")})
            if bool(row.get("opportunity_cost_flag", False)):
                rows.append({"severity": "High", "ticker": ticker, "alert": "Opportunity Cost High", "detail": row.get("best_alternative_ticker", "")})
    event_severity = {
        "Overvaluation Risk": "Medium", "Catalyst Exhausted": "Medium",
        "Quality Upgrade": "Low", "Growth Acceleration": "Low", "Re-rating Opportunity": "Low",
    }
    if not events.empty:
        for _, row in events.iterrows():
            for event in str(row.get("thesis_events", "")).split("; "):
                if event in event_severity:
                    rows.append({"severity": event_severity[event], "ticker": row.get("ticker", ""), "alert": event, "detail": row.get("date", "")})
    return pd.DataFrame(rows)


def _build_alerts(viewer_dir: Path, run_root: Path, positions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    alerts = _alerts_from_artifacts(positions, events)
    (viewer_dir / "alerts.json").write_text(json.dumps(alerts.to_dict(orient="records"), indent=2, default=str), encoding="utf-8")
    body = _bar_plot(alerts, "severity", "Alerts by Severity") + _table(alerts)
    _write(viewer_dir / "alerts.html", "Alerts", body, run_root)
    return alerts


def _build_thesis_change_report(viewer_dir: Path, run_root: Path, events: pd.DataFrame) -> None:
    latest = events.copy()
    if not latest.empty and "date" in latest:
        max_date = latest["date"].max()
        latest = latest[latest["date"].eq(max_date)]
    body = _table(latest)
    _write(viewer_dir / "thesis_change_report.html", "Thesis Change Report", body, run_root)


def _build_lifecycle_pages(viewer_dir: Path, run_root: Path, holdings: pd.DataFrame, transactions: pd.DataFrame, decision_log: pd.DataFrame, turnover: pd.DataFrame) -> None:
    lifecycle = holdings.copy()
    if not lifecycle.empty and {"ticker", "months_since_entry"}.issubset(lifecycle.columns):
        lifecycle = lifecycle.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    _write(viewer_dir / "portfolio_lifecycle.html", "Portfolio Lifecycle", _table(lifecycle), run_root)
    _write(viewer_dir / "portfolio_turnover.html", "Portfolio Turnover", _hist_plot(turnover, "monthly_turnover", "Monthly Turnover") + _table(turnover), run_root)
    _write(viewer_dir / "decision_log.html", "Decision Log", _table(decision_log, max_rows=1000), run_root)
    _write(viewer_dir / "thesis_persistence.html", "Thesis Persistence", _hist_plot(lifecycle, "thesis_persistence_score", "Thesis Persistence") + _table(lifecycle, columns=["ticker", "months_since_entry", "months_thesis_intact", "thesis_persistence_score", "original_opportunity_type", "original_buy_reason"]), run_root)
    winners = lifecycle[pd.to_numeric(lifecycle.get("thesis_persistence_score", pd.Series(dtype=float)), errors="coerce").fillna(0) >= 65] if not lifecycle.empty else lifecycle
    _write(viewer_dir / "hold_winners.html", "Hold Winners", _table(winners), run_root)
    timeline = transactions.sort_values("date") if not transactions.empty and "date" in transactions else transactions
    _write(viewer_dir / "portfolio_timeline.html", "Portfolio Timeline", _bar_plot(timeline, "action", "Actions over lifecycle") + _table(timeline, max_rows=1000), run_root)


def generate_static_viewer(run_root: str | Path, *, viewer_dir: str | Path | None = None) -> Path:
    """Generate a static HTML viewer from existing run/portfolio-review artifacts."""
    root = Path(run_root)
    out = Path(viewer_dir) if viewer_dir is not None else root / "viewer"
    out.mkdir(parents=True, exist_ok=True)

    positions = _read_csv(_find_first(root, ["portfolio_review_positions.csv"]))
    history = _read_csv(_find_first(root, ["portfolio_thesis_history.csv"]))
    events = _read_csv(_find_first(root, ["portfolio_thesis_events.csv"]))
    opportunities = _read_csv(_find_first(root, ["portfolio_review_opportunity_cost.csv"]))
    evolution = _read_csv(_find_first(root, ["portfolio_evolution.csv"]))
    holdings = _read_csv(_find_first(root, ["portfolio_monthly_holdings.csv"]))
    transactions = _read_csv(_find_first(root, ["portfolio_transactions.csv"]))
    decision_log = _read_csv(_find_first(root, ["portfolio_decision_log.csv"]))
    turnover = _read_csv(_find_first(root, ["portfolio_turnover.csv"]))
    summary = _read_json(_find_first(root, ["portfolio_review_summary.json", "summary.json", "backtest_summary.json"]))

    _build_run_summary(out, root, positions, summary)
    _build_portfolio_review(out, root, positions)
    _build_portfolio_health(out, root, positions)
    _build_portfolio_evolution_page(out, root, evolution)
    _build_vs_benchmark_page(out, root, evolution)
    _build_lifecycle_pages(out, root, holdings, transactions, decision_log, turnover)
    _build_allocation_dashboard(out, root, holdings, positions)
    _build_thesis_history(out, root, history)
    _build_thesis_events(out, root, events)
    _build_thesis_change_report(out, root, events)
    _build_alerts(out, root, positions, events)
    _build_opportunity_cost(out, root, positions, opportunities)
    _build_watchlist(out, root, opportunities, positions)
    position_pages = _build_position_pages(out, root, positions, history)
    _build_index(out, root, PAGES + position_pages[:50])
    return out
