"""Static HTML viewer generation for GARP run and portfolio-review artifacts.

The viewer is presentation-only: it reads existing CSV/JSON/Markdown artifacts and
never recalculates training, ranking, backtest or thesis metrics.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


PAGES = [
    ("run_summary.html", "Run Summary"),
    ("portfolio_review.html", "Portfolio Review"),
    ("portfolio_health.html", "Portfolio Health"),
    ("thesis_history.html", "Thesis History"),
    ("thesis_events.html", "Thesis Events"),
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


def _html_shell(title: str, body: str, run_root: Path, pages: list[tuple[str, str]] | None = None) -> str:
    nav_items = pages or PAGES
    nav = "".join(f'<a href="{href}">{label}</a>' for href, label in nav_items)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css">
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
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
    path.write_text(_html_shell(title, body, run_root), encoding="utf-8")


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
        body += '<h2>Current vs Original Thesis</h2>' + _table(details)
        _write(viewer_dir / filename, f"Position {ticker}", body, run_root)
        pages.append((filename, f"Position {ticker}"))
    return pages


def generate_static_viewer(run_root: str | Path, *, viewer_dir: str | Path | None = None) -> Path:
    """Generate a static HTML viewer from existing run/portfolio-review artifacts."""
    root = Path(run_root)
    out = Path(viewer_dir) if viewer_dir is not None else root / "viewer"
    out.mkdir(parents=True, exist_ok=True)

    positions = _read_csv(_find_first(root, ["portfolio_review_positions.csv"]))
    history = _read_csv(_find_first(root, ["portfolio_thesis_history.csv"]))
    events = _read_csv(_find_first(root, ["portfolio_thesis_events.csv"]))
    opportunities = _read_csv(_find_first(root, ["portfolio_review_opportunity_cost.csv"]))
    summary = _read_json(_find_first(root, ["portfolio_review_summary.json", "summary.json", "backtest_summary.json"]))

    _build_run_summary(out, root, positions, summary)
    _build_portfolio_review(out, root, positions)
    _build_portfolio_health(out, root, positions)
    _build_thesis_history(out, root, history)
    _build_thesis_events(out, root, events)
    _build_opportunity_cost(out, root, positions, opportunities)
    _build_watchlist(out, root, opportunities, positions)
    position_pages = _build_position_pages(out, root, positions, history)
    _build_index(out, root, PAGES + position_pages[:50])
    return out
