"""Shared helpers and design system for the single-page report viewer.

The viewer is one professional Spanish-language HTML report (see pages.py). Every element must
justify its place: it answers a TFM question (is there alpha? does the system learn? what was
bought/sold and why?) or it is a debugging aid. Bulk per-ticker/per-snapshot audit tables stay on
disk as CSV under results/<run>/audit/ and are only linked, never embedded.
"""

from __future__ import annotations

import html

import pandas as pd


# Chart palette (validated with the dataviz skill, light+dark). Series always carry direct legend
# labels, so the low-contrast slots are covered by the relief rule.
PALETTE = {
    "portfolio": "#2a78d6",   # blue
    "benchmark": "#e34948",   # red
    "series_3": "#1baf7a",    # aqua
    "series_4": "#eda100",    # yellow
    "positive": "#1baf7a",
    "negative": "#e34948",
    "muted": "#8a9296",
}


def report_layout(title: str, body: str) -> str:
    """Self-contained, theme-aware report shell. All styling inline (no external assets)."""
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  --bg: #f4f6f7; --surface: #ffffff; --border: #e2e7e8; --ink: #16232b;
  --ink-soft: #56656c; --accent: #2a78d6; --pos: #12855b; --neg: #c8342f;
  --shadow: 0 1px 2px rgba(20,35,43,.06), 0 1px 8px rgba(20,35,43,.04);
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #10161a; --surface: #182229; --border: #26343d; --ink: #eaf0f2;
    --ink-soft: #9fb0b8; --accent: #3987e5; --pos: #2bbd82; --neg: #e66767;
    --shadow: none;
  }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink);
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.5; -webkit-font-smoothing: antialiased; }}
header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 22px 0; }}
header .wrap {{ display: flex; align-items: baseline; justify-content: space-between; gap: 16px; }}
header h1 {{ font-size: 21px; margin: 0; letter-spacing: -.01em; }}
header .meta {{ color: var(--ink-soft); font-size: 13px; }}
nav {{ position: sticky; top: 0; z-index: 5; background: var(--surface);
  border-bottom: 1px solid var(--border); overflow-x: auto; }}
nav .wrap {{ display: flex; gap: 4px; padding: 6px 0; }}
nav a {{ color: var(--ink-soft); text-decoration: none; font-size: 13px; font-weight: 500;
  padding: 7px 12px; border-radius: 7px; white-space: nowrap; }}
nav a:hover {{ background: var(--bg); color: var(--ink); }}
.wrap {{ max-width: 1080px; margin: 0 auto; padding: 0 24px; }}
main {{ padding: 8px 0 64px; }}
section {{ margin-top: 40px; scroll-margin-top: 56px; }}
section > h2 {{ font-size: 17px; margin: 0 0 4px; letter-spacing: -.01em; }}
section > p.lead {{ color: var(--ink-soft); font-size: 13.5px; margin: 0 0 18px; max-width: 68ch; }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; }}
.kpi {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 16px 18px; box-shadow: var(--shadow); }}
.kpi .label {{ color: var(--ink-soft); font-size: 12px; font-weight: 600;
  text-transform: uppercase; letter-spacing: .04em; }}
.kpi .value {{ font-size: 26px; font-weight: 650; margin-top: 6px; letter-spacing: -.02em;
  font-variant-numeric: tabular-nums; }}
.kpi .sub {{ color: var(--ink-soft); font-size: 12px; margin-top: 3px; }}
.value.pos {{ color: var(--pos); }} .value.neg {{ color: var(--neg); }}
figure {{ margin: 0; background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 14px; box-shadow: var(--shadow); }}
figure img {{ display: block; width: 100%; height: auto; border-radius: 4px; }}
figcaption {{ color: var(--ink-soft); font-size: 12.5px; margin-top: 10px; }}
.table-scroll {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; box-shadow: var(--shadow); }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px;
  font-variant-numeric: tabular-nums; }}
th, td {{ padding: 9px 14px; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
thead th {{ color: var(--ink-soft); font-weight: 600; font-size: 11.5px;
  text-transform: uppercase; letter-spacing: .03em; border-bottom: 1px solid var(--border); }}
tbody tr + tr td {{ border-top: 1px solid var(--border); }}
.note {{ color: var(--ink-soft); font-size: 12.5px; }}
ul.findings {{ padding-left: 18px; }} ul.findings li {{ margin: 6px 0; }}
a.audit {{ color: var(--accent); text-decoration: none; }} a.audit:hover {{ text-decoration: underline; }}
.empty {{ color: var(--ink-soft); font-size: 13px; padding: 14px; }}
</style>
</head>
<body>{body}</body>
</html>"""


PCT_COLUMNS = {
    "hybrid_weight", "equal_weight", "conviction_weight", "risk_adjusted_weight", "weight",
    "total_return", "annualized_return", "benchmark_total_return", "benchmark_annualized_return",
    "excess_total_return", "period_alpha", "cumulative_alpha", "portfolio_period_return",
    "benchmark_period_return", "transaction_cost_drag", "depth", "avg_excess_return",
    "avg_total_return", "win_rate_vs_benchmark", "information_ratio", "tracking_error_annualized",
}


def fmt_pct(value) -> str:
    return "" if pd.isna(value) else f"{value:.1%}"


def fmt_num(value, decimals: int = 2) -> str:
    return "" if pd.isna(value) else f"{value:,.{decimals}f}"


def format_for_html(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for column in out.columns:
        if column in PCT_COLUMNS:
            out[column] = out[column].map(lambda v: "" if pd.isna(v) else f"{v:.2%}")
        elif pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    return out


def table(df: pd.DataFrame, rename: dict[str, str] | None = None) -> str:
    if df is None or df.empty:
        return '<div class="empty">No hay datos disponibles.</div>'
    shown = format_for_html(df)
    if rename:
        shown = shown.rename(columns=rename)
    return '<div class="table-scroll">' + shown.to_html(index=False, escape=True, border=0) + "</div>"


def select(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df[[c for c in columns if c in df.columns]]


def figure(src: str | None, caption: str) -> str:
    if not src:
        return ""
    return (f'<figure><img src="{html.escape(src)}" alt="{html.escape(caption)}">'
            f'<figcaption>{html.escape(caption)}</figcaption></figure>')


def kpi(label: str, value: str, sub: str = "", tone: str = "") -> str:
    tone_class = f" {tone}" if tone in ("pos", "neg") else ""
    sub_html = f'<div class="sub">{html.escape(sub)}</div>' if sub else ""
    return (f'<div class="kpi"><div class="label">{html.escape(label)}</div>'
            f'<div class="value{tone_class}">{html.escape(value)}</div>{sub_html}</div>')
