"""Interactive dashboard page for the static viewer."""

from __future__ import annotations

import json
import math

import pandas as pd

from .shared import select


def dashboard_body(tables: dict[str, pd.DataFrame], charts: dict[str, str]) -> str:
    payload_json = json.dumps(_sanitize(dashboard_payload(tables, charts)), ensure_ascii=True, allow_nan=False)
    template = r"""
<style>
.dash { display: grid; gap: 18px; }
.dash-hero { display: grid; gap: 16px; align-items: end; grid-template-columns: minmax(0, 1fr) auto; }
.dash-hero h1 { margin-bottom: 4px; }
.dash-date { color: #647174; font-size: 13px; text-align: right; }
.kpi-grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; }
.kpi { background: white; border: 1px solid #d8e0df; border-radius: 8px; padding: 12px; min-height: 76px; }
.kpi-label { color: #647174; font-size: 12px; text-transform: uppercase; }
.kpi-value { color: #142124; font-size: 23px; font-weight: 700; margin-top: 8px; }
.kpi-value.good { color: #116a48; }
.kpi-value.bad { color: #a43838; }
.control-bar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; background: #ffffff; border: 1px solid #d8e0df; border-radius: 8px; padding: 10px; }
.control-bar input, .control-bar select { border: 1px solid #c8d2d0; border-radius: 6px; padding: 8px 10px; color: #172026; background: #fff; font-size: 13px; min-height: 36px; }
.control-bar input { flex: 1 1 220px; }
.control-bar select { flex: 0 0 180px; }
.panel-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
.panel { background: white; border: 1px solid #d8e0df; border-radius: 8px; padding: 14px; min-width: 0; }
.panel.wide { grid-column: 1 / -1; }
.panel h2 { margin: 0 0 12px; font-size: 17px; }
.chart-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.chart-tile { border: 1px solid #e2e7e6; border-radius: 8px; padding: 10px; background: #fbfcfc; }
.chart-tile img { display: block; width: 100%; height: auto; }
.chart-tile span { display: block; margin-top: 8px; color: #526063; font-size: 12px; }
.table-wrap { overflow: auto; max-height: 520px; border: 1px solid #e2e7e6; border-radius: 8px; }
.table-wrap table { margin: 0; border: 0; }
.table-wrap th { position: sticky; top: 0; z-index: 1; }
.badge { display: inline-flex; align-items: center; min-height: 22px; border-radius: 999px; padding: 0 8px; font-size: 12px; background: #eef2f2; color: #344044; }
.badge.buy, .badge.add, .badge.hold { background: #e5f3ed; color: #116a48; }
.badge.sell, .badge.reduce { background: #f8e9e9; color: #a43838; }
.badge.watch { background: #fff3dc; color: #7a4f00; }
.empty-state { color: #647174; margin: 10px 0; }
.spark { height: 46px; width: 100%; display: block; }
.section-links { display: flex; gap: 8px; flex-wrap: wrap; }
.section-links a { text-decoration: none; border: 1px solid #ccd5d3; border-radius: 6px; color: #172026; padding: 7px 9px; font-size: 13px; background: white; }
@media (max-width: 980px) { .dash-hero, .panel-grid, .chart-grid { grid-template-columns: 1fr; } .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } .dash-date { text-align: left; } }
@media (max-width: 560px) { .kpi-grid { grid-template-columns: 1fr; } .control-bar select { flex: 1 1 160px; } }
</style>
<div class="dash">
  <section class="dash-hero">
    <div>
      <h1>Dashboard de resultados</h1>
      <div class="section-links">
        <a href="current_portfolio.html">Cartera</a>
        <a href="portfolio_vs_benchmark.html">Benchmark</a>
        <a href="position_performance.html">Posiciones</a>
        <a href="top_opportunities.html">Oportunidades</a>
        <a href="strategy_learning.html">Aprendizaje</a>
      </div>
    </div>
    <div class="dash-date" id="runDate"></div>
  </section>
  <section class="kpi-grid" id="kpis"></section>
  <section class="control-bar">
    <input id="searchBox" type="search" placeholder="Ticker, sector, tesis o razon">
    <select id="viewSelect"><option value="portfolio">Cartera actual</option><option value="opportunities">Oportunidades</option><option value="actions">Operaciones</option><option value="performance">Performance</option></select>
    <select id="sectorSelect"><option value="">Todos los sectores</option></select>
    <select id="actionSelect"><option value="">Todas las acciones</option></select>
  </section>
  <section class="panel-grid">
    <article class="panel wide"><h2>Vision global</h2><div class="chart-grid" id="charts"></div></article>
    <article class="panel"><h2>Comparador</h2><canvas class="spark" id="spark"></canvas><div class="table-wrap" id="trackingTable"></div></article>
    <article class="panel"><h2>Exposicion sectorial</h2><div class="table-wrap" id="sectorTable"></div></article>
    <article class="panel wide"><h2 id="mainTitle">Cartera actual</h2><div class="table-wrap" id="mainTable"></div></article>
    <article class="panel"><h2>Backlog de mejora</h2><div class="table-wrap" id="backlogTable"></div></article>
    <article class="panel"><h2>Evidencia de estrategia</h2><div class="table-wrap" id="learningTable"></div></article>
  </section>
</div>
<script>
const DASHBOARD_DATA = __PAYLOAD__;
const state = { view: "portfolio", query: "", sector: "", action: "" };
const columns = {
  portfolio: ["ticker", "sector", "hybrid_weight", "position_action", "current_manager_score", "current_buy_today_score", "current_thesis_state", "investment_thesis"],
  opportunities: ["rank", "ticker", "universe_action", "classification", "manager_score", "buy_today_score", "valuation_score", "conviction_score", "price_adjusted_valuation_score", "momentum_score", "opportunity_type", "investment_thesis", "entry_trigger"],
  actions: ["date", "ticker", "action", "reason_category", "manager_score", "buy_today_score", "total_return", "excess_total_return", "reason"],
  performance: ["ticker", "closed", "holding_days", "total_return", "annualized_return", "benchmark_annualized_return", "excess_total_return", "exit_reason_category"]
};
const titles = { portfolio: "Cartera actual", opportunities: "Oportunidades actuales", actions: "Operaciones", performance: "Performance por posicion" };
const numericFormats = { hybrid_weight: "pct", current_manager_score: "score", current_buy_today_score: "score", manager_score: "score", buy_today_score: "score", price_adjusted_valuation_score: "score", momentum_score: "score", total_return: "pct", annualized_return: "pct", valuation_score: "score", conviction_score: "score", benchmark_annualized_return: "pct", excess_total_return: "pct", portfolio_value: "x", benchmark_value: "x", period_alpha: "pct", cumulative_alpha: "pct", weight: "pct", avg_manager_score: "score", transaction_cost_drag: "pct" };
function fmt(value, key) { if (value === null || value === undefined || value === "") return ""; const num = Number(value); if (!Number.isFinite(num)) return String(value); if (numericFormats[key] === "pct") return `${(num * 100).toFixed(1)}%`; if (numericFormats[key] === "score") return num.toFixed(2); if (numericFormats[key] === "x") return `${num.toFixed(2)}x`; if (Math.abs(num) < 1 && !Number.isInteger(num)) return num.toFixed(3); return String(value); }
function badge(value) { const text = fmt(value, ""); const cls = text.toLowerCase(); if (["buy","sell","hold","watch","add","reduce"].includes(cls)) return `<span class="badge ${cls}">${text}</span>`; return text; }
function rowMatches(row) { const query = state.query.trim().toLowerCase(); const sectorOk = !state.sector || row.sector === state.sector; const actionValue = row.action || row.universe_action || row.position_action || row.exit_reason_category || ""; const actionOk = !state.action || actionValue === state.action; if (!query) return sectorOk && actionOk; return sectorOk && actionOk && Object.values(row).some(value => String(value ?? "").toLowerCase().includes(query)); }
function table(rows, cols, limit = 80) { const shown = rows.slice(0, limit); if (!shown.length) return `<p class="empty-state">Sin datos para esta seleccion.</p>`; const head = cols.map(col => `<th>${col.replaceAll("_", " ")}</th>`).join(""); const body = shown.map(row => `<tr>${cols.map(col => `<td>${badge(row[col])}</td>`).join("")}</tr>`).join(""); return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`; }
function renderKpis() { document.getElementById("kpis").innerHTML = (DASHBOARD_DATA.kpis || []).map(item => `<div class="kpi"><div class="kpi-label">${item.label}</div><div class="kpi-value ${item.tone || ""}">${item.value}</div></div>`).join(""); }
function renderCharts() { document.getElementById("charts").innerHTML = (DASHBOARD_DATA.charts || []).map(chart => `<div class="chart-tile"><img src="${chart.src}" alt="${chart.label}"><span>${chart.label}</span></div>`).join(""); }
function renderMain() { const rows = (DASHBOARD_DATA[state.view] || []).filter(rowMatches); document.getElementById("mainTitle").textContent = titles[state.view]; document.getElementById("mainTable").innerHTML = table(rows, columns[state.view]); }
function renderSupportTables() { document.getElementById("trackingTable").innerHTML = table(DASHBOARD_DATA.tracking || [], ["date", "portfolio_value", "benchmark_value", "period_alpha", "cumulative_alpha", "buys", "sells"], 18); document.getElementById("sectorTable").innerHTML = table(DASHBOARD_DATA.sectors || [], ["sector", "weight", "positions", "avg_manager_score", "tickers"], 18); document.getElementById("backlogTable").innerHTML = table(DASHBOARD_DATA.backlog || [], ["priority", "theme", "evidence", "suggested_next_step"], 12); document.getElementById("learningTable").innerHTML = table(DASHBOARD_DATA.learning || [], ["area", "segment", "observations", "win_rate_vs_benchmark", "avg_excess_return", "hint"], 12); }
function fillFilters() { const sectors = [...new Set((DASHBOARD_DATA.portfolio || []).map(row => row.sector).filter(Boolean))].sort(); const actions = [...new Set([...(DASHBOARD_DATA.actions || []), ...(DASHBOARD_DATA.opportunities || []), ...(DASHBOARD_DATA.portfolio || [])].map(row => row.action || row.universe_action || row.position_action || row.exit_reason_category).filter(Boolean))].sort(); document.getElementById("sectorSelect").innerHTML += sectors.map(v => `<option value="${v}">${v}</option>`).join(""); document.getElementById("actionSelect").innerHTML += actions.map(v => `<option value="${v}">${v}</option>`).join(""); }
function drawSpark() { const canvas = document.getElementById("spark"); const ctx = canvas.getContext("2d"); const width = canvas.clientWidth * window.devicePixelRatio; const height = canvas.clientHeight * window.devicePixelRatio; canvas.width = width; canvas.height = height; ctx.clearRect(0, 0, width, height); const rows = DASHBOARD_DATA.tracking || []; if (rows.length < 2) return; const series = [{ key: "portfolio_value", color: "#1f7a8c" }, { key: "benchmark_value", color: "#b23a48" }]; const values = rows.flatMap(row => series.map(item => Number(row[item.key]))).filter(Number.isFinite); const min = Math.min(...values); const max = Math.max(...values); const pad = 5 * window.devicePixelRatio; series.forEach(item => { ctx.beginPath(); ctx.strokeStyle = item.color; ctx.lineWidth = 2 * window.devicePixelRatio; rows.forEach((row, index) => { const value = Number(row[item.key]); const x = pad + index * ((width - pad * 2) / Math.max(rows.length - 1, 1)); const y = height - pad - ((value - min) / Math.max(max - min, 0.0001)) * (height - pad * 2); if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.stroke(); }); }
function bindControls() { document.getElementById("searchBox").addEventListener("input", event => { state.query = event.target.value; renderMain(); }); document.getElementById("viewSelect").addEventListener("change", event => { state.view = event.target.value; renderMain(); }); document.getElementById("sectorSelect").addEventListener("change", event => { state.sector = event.target.value; renderMain(); }); document.getElementById("actionSelect").addEventListener("change", event => { state.action = event.target.value; renderMain(); }); window.addEventListener("resize", drawSpark); }
document.getElementById("runDate").textContent = DASHBOARD_DATA.run_label || "";
renderKpis(); renderCharts(); fillFilters(); renderMain(); renderSupportTables(); drawSpark(); bindControls();
</script>
"""
    return template.replace("__PAYLOAD__", payload_json)


def _sanitize(obj):
    if isinstance(obj, dict):
        return {key: _sanitize(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(value) for value in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def dashboard_payload(tables: dict[str, pd.DataFrame], charts: dict[str, str]) -> dict:
    summary = tables.get("executive_summary", pd.DataFrame())
    vs = tables.get("portfolio_vs_benchmark", pd.DataFrame())
    latest_vs = vs.iloc[-1].to_dict() if not vs.empty else {}
    summary_row = summary.iloc[0].to_dict() if not summary.empty else {}
    latest_date = latest_vs.get("date") or summary_row.get("date") or ""
    portfolio_value = as_float(latest_vs.get("portfolio_value", summary_row.get("portfolio_value")))
    benchmark_value = as_float(latest_vs.get("benchmark_value", summary_row.get("benchmark_value")))
    alpha = as_float(summary_row.get("cumulative_alpha", latest_vs.get("cumulative_alpha")))
    win_rate = as_float(summary_row.get("closed_position_win_rate_vs_benchmark"))
    kpis = [
        kpi("Cartera", format_dashboard_number(portfolio_value, "x"), offset_signal(portfolio_value, 1)),
        kpi("Benchmark", format_dashboard_number(benchmark_value, "x"), offset_signal(benchmark_value, 1)),
        kpi("Alpha acum.", format_dashboard_number(alpha, "pct"), alpha),
        kpi("Win rate", format_dashboard_number(win_rate, "pct"), offset_signal(win_rate, 0.5)),
        kpi("Compras", format_dashboard_number(summary_row.get("buys"), "int"), None),
        kpi("Ventas", format_dashboard_number(summary_row.get("sells"), "int"), None),
    ]
    if not summary_row and vs.empty:
        watchlist = tables.get("watchlist", pd.DataFrame())
        research = tables.get("research_ai", pd.DataFrame())
        latest_watchlist = latest_by_date(watchlist, "snapshot_date")
        latest_snapshot = latest_watchlist["snapshot_date"].max() if "snapshot_date" in latest_watchlist.columns and not latest_watchlist.empty else ""
        classification_count = int(research["classification"].nunique()) if "classification" in research.columns else None
        kpis = [
            kpi("Watchlist", format_dashboard_number(len(latest_watchlist), "int"), None),
            kpi("Research AI", format_dashboard_number(len(research), "int"), None),
            kpi("Fecha", str(latest_snapshot) if latest_snapshot else "-", None),
            kpi("Tipos", format_dashboard_number(classification_count, "int"), None),
            kpi("Conviccion med.", format_dashboard_number(mean_or_none(latest_watchlist, "conviction_score"), "score"), None),
            kpi("Valoracion med.", format_dashboard_number(mean_or_none(latest_watchlist, "valuation_score"), "score"), None),
        ]
    opportunities = tables.get("top_opportunities_latest", pd.DataFrame())
    if opportunities.empty:
        opportunities = latest_by_date(tables.get("watchlist", pd.DataFrame()), "snapshot_date").copy()
        if not opportunities.empty:
            sort_cols = [col for col in ["conviction_score", "valuation_score"] if col in opportunities.columns]
            opportunities = opportunities.sort_values(sort_cols, ascending=False)
            opportunities.insert(0, "rank", range(1, len(opportunities) + 1))
    chart_items = [
        ("portfolio_vs_benchmark", "Cartera frente a benchmark"),
        ("period_alpha", "Alpha mensual"),
        ("drawdown", "Drawdown"),
        ("position_performance_bars", "Rendimiento por posicion"),
        ("sector_exposure", "Sectores"),
        ("latest_allocation", "Asignacion actual"),
        ("watchlist_map", "Watchlist valoracion-conviccion"),
    ]
    latest_sectors = latest_by_date(tables.get("sector_exposure", pd.DataFrame()), "date")
    performance = tables.get("position_performance", pd.DataFrame())
    performance = performance.sort_values("excess_total_return", ascending=False) if "excess_total_return" in performance.columns else performance
    return {
        "run_label": f"Ultima fecha: {latest_date}" if latest_date else "",
        "kpis": kpis,
        "charts": [{"src": charts[key], "label": label} for key, label in chart_items if key in charts],
        "portfolio": records(select(tables.get("current_portfolio", pd.DataFrame()), ["ticker", "sector", "hybrid_weight", "position_action", "current_manager_score", "current_buy_today_score", "current_thesis_state", "investment_thesis", "current_exit_thesis"])),
        "opportunities": records(select(opportunities, ["rank", "ticker", "universe_action", "manager_score", "buy_today_score", "valuation_score", "conviction_score", "price_adjusted_valuation_score", "momentum_score", "opportunity_type", "investment_thesis", "entry_trigger", "classification"])),
        "actions": records(select(tables.get("action_journal", pd.DataFrame()).tail(250), ["date", "ticker", "action", "reason_category", "manager_score", "buy_today_score", "total_return", "excess_total_return", "reason"])),
        "performance": records(select(performance, ["ticker", "closed", "holding_days", "total_return", "annualized_return", "benchmark_annualized_return", "excess_total_return", "exit_reason_category"])),
        "tracking": records(select(tables.get("tracking_dashboard", pd.DataFrame()).tail(36), ["date", "portfolio_value", "benchmark_value", "period_alpha", "cumulative_alpha", "transaction_cost_drag", "buys", "sells"])),
        "sectors": records(select(latest_sectors.sort_values("weight", ascending=False) if "weight" in latest_sectors.columns else latest_sectors, ["sector", "weight", "positions", "avg_manager_score", "tickers"])),
        "backlog": records(tables.get("improvement_backlog", pd.DataFrame())),
        "learning": records(tables.get("strategy_learning_log", pd.DataFrame())),
    }


def latest_by_date(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    if df.empty or date_column not in df.columns:
        return df
    return df[df[date_column] == df[date_column].max()].copy()


def records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    clean = df.copy().where(pd.notna(df), None)
    return [{str(key): json_ready(value) for key, value in row.items()} for row in clean.to_dict(orient="records")]


def json_ready(value):
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def as_float(value) -> float | None:
    try:
        return None if value is None or pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return None


def format_dashboard_number(value, kind: str) -> str:
    number = as_float(value)
    if number is None:
        return "-"
    if kind == "pct":
        return f"{number:.1%}"
    if kind == "x":
        return f"{number:.2f}x"
    if kind == "int":
        return f"{int(round(number))}"
    if kind == "score":
        return f"{number:.2f}"
    return f"{number:.2f}"


def mean_or_none(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df.columns:
        return None
    value = pd.to_numeric(df[column], errors="coerce").mean()
    return None if pd.isna(value) else float(value)


def offset_signal(value, baseline: float) -> float | None:
    number = as_float(value)
    return None if number is None else number - baseline


def kpi(label: str, value: str, signal: float | None) -> dict:
    tone = "" if signal is None else ("good" if signal >= 0 else "bad")
    return {"label": label, "value": value, "tone": tone}
