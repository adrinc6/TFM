"""Static page rendering and build orchestration for the result viewer."""

from __future__ import annotations

import html
import json
import logging

import pandas as pd

from pathlib import Path

from environment import PROCESSED_DIR, Settings

from .charts import build_charts
from .manifest import write_results_explainer
from .shared import PAGE_GROUPS, PAGES, figure, layout, select, table
from .dashboard import dashboard_body

log = logging.getLogger(__name__)


def page_body(page: str, tables: dict[str, pd.DataFrame], charts: dict[str, str]) -> str:
    if page == "dashboard.html":
        return dashboard_body(tables, charts)
    if page == "index.html":
        return ("<h1>Cartera GARP AI</h1><h2>Resumen Ejecutivo</h2>" + table(tables.get("executive_summary", pd.DataFrame())) + figure(charts.get("portfolio_vs_benchmark"), "Valor de la cartera frente al benchmark SPY") + figure(charts.get("period_alpha"), "Contribucion mensual de alpha") + figure(charts.get("position_performance_bars"), "Retorno por accion frente al benchmark durante cada periodo de holding") + figure(charts.get("sector_exposure"), "Exposicion sectorial en el tiempo") + "<h2>Cartera Actual</h2>" + table(tables.get("current_portfolio", pd.DataFrame())) + "<h2>Operaciones Recientes</h2>" + table(tables.get("action_journal", pd.DataFrame()).tail(20)))
    if page == "current_portfolio.html":
        return "<h1>Cartera Actual</h1>" + table(tables.get("current_portfolio", pd.DataFrame()))
    if page == "tracking_dashboard.html":
        return "<h1>Seguimiento de la Cartera</h1>" + table(select(tables.get("tracking_dashboard", pd.DataFrame()), ["date", "portfolio_value", "portfolio_gross_value", "benchmark_value", "portfolio_period_return", "portfolio_gross_period_return", "transaction_cost_drag", "benchmark_period_return", "period_alpha", "cumulative_alpha", "holdings", "buys", "sells", "tickers"]))
    if page == "portfolio_vs_benchmark.html":
        vs = tables.get("portfolio_vs_benchmark", pd.DataFrame())
        return "<h1>Cartera Frente a Benchmark</h1>" + figure(charts.get("portfolio_vs_benchmark"), "Valor de la cartera frente al benchmark SPY") + figure(charts.get("period_alpha"), "Retorno mensual en exceso frente al benchmark") + figure(charts.get("drawdown"), "Perfil de drawdown") + drawdown_episode_table(vs) + table(vs)
    if page == "action_journal.html":
        return "<h1>Diario de Operaciones</h1>" + table(select(tables.get("action_journal", pd.DataFrame()), ["date", "ticker", "action", "reason_category", "rank", "manager_score", "buy_today_score", "holding_days", "total_return", "benchmark_total_return", "excess_total_return", "reason"]))
    if page == "position_performance.html":
        data = tables.get("position_performance", pd.DataFrame())
        data = data.sort_values("excess_total_return", ascending=False) if "excess_total_return" in data.columns else data
        return "<h1>Rendimiento de Posiciones</h1>" + figure(charts.get("position_performance_bars"), "Retorno de la accion, retorno anualizado de la accion y retorno anualizado del benchmark por ticker") + table(select(data, ["ticker", "entry_date", "exit_date", "closed", "holding_days", "total_return", "annualized_return", "benchmark_annualized_return", "excess_total_return", "exit_reason_category"]))
    if page == "buy_rationale.html":
        return "<h1>Justificacion de Compra</h1>" + table(select(tables.get("buy_rationale", pd.DataFrame()), ["date", "ticker", "rank", "manager_score", "buy_today_score", "thesis_rank_score", "business_quality_score", "price_adjusted_valuation_score", "momentum_score", "alpha_probability", "opportunity_type", "best_alternative_ticker", "opportunity_cost_score", "reason"]))
    if page == "sell_reasons.html":
        return "<h1>Motivos de Venta</h1>" + table(tables.get("sell_reasons_summary", pd.DataFrame()))
    if page == "sector_exposure.html":
        return "<h1>Exposicion Sectorial</h1>" + figure(charts.get("sector_exposure"), "Pesos sectoriales en el tiempo") + table(tables.get("sector_exposure", pd.DataFrame()))
    if page == "allocation_dashboard.html":
        return "<h1>Dimensionamiento de Posiciones</h1>" + figure(charts.get("latest_allocation"), "Ultima asignacion hibrida") + figure(charts.get("allocation_drift"), "Peso por ticker en el tiempo") + table(tables.get("portfolio_allocation", pd.DataFrame()))
    if page == "watchlist.html":
        return "<h1>Watchlist</h1>" + figure(charts.get("watchlist_map"), "Watchlist: valoracion frente a conviccion") + table(tables.get("watchlist", pd.DataFrame()))
    if page == "top_opportunities.html":
        return "<h1>Mejores Oportunidades</h1>" + table(tables.get("top_opportunities_latest", pd.DataFrame()))
    if page == "strategy_learning.html":
        return "<h1>Aprendizaje de la Estrategia</h1><h2>Backlog de Mejora</h2>" + table(tables.get("improvement_backlog", pd.DataFrame())) + "<h2>Registro de Evidencia</h2>" + table(tables.get("strategy_learning_log", pd.DataFrame()))
    if page == "model_explainability.html":
        return "<h1>Explicabilidad del Modelo</h1>" + figure(charts.get("feature_importance"), "Variables mas relevantes del modelo") + explainability()
    if page == "audit.html":
        return "<h1>Archivos de Auditoria</h1>" + table(audit_file_table(tables))
    return "<h1>No disponible</h1><p>No hay datos para esta pagina.</p>"


def explainability() -> str:
    path = PROCESSED_DIR / "model_explainability.json"
    if not path.exists():
        return "<p>No hay artefacto de explicabilidad disponible.</p>"
    payload = json.loads(path.read_text(encoding="utf-8"))
    importance = pd.DataFrame([{"feature": key, "importance": value} for key, value in payload.get("feature_importance", {}).items()])
    shap = payload.get("shap", {})
    shap_values = shap.get("mean_abs_contribution", {}) if shap.get("available") else {}
    shap_table = pd.DataFrame([{"feature": key, "mean_abs_shap": value} for key, value in shap_values.items()])
    reason = "" if shap.get("available") else f"<p>{html.escape(shap.get('reason', 'SHAP no disponible.'))}</p>"
    return (
        "<h2>Importancia de Variables</h2>" + table(importance)
        + "<h2>SHAP</h2>" + reason + table(shap_table)
        + "<h2>Diagnostico Walk-Forward</h2>"
        + "<p>Muestra, para cada snapshot, si el modelo se entreno con datos suficientes o si se uso el "
        "fallback determinista GARP, y cuantas etiquetas futuras eran observables en ese momento.</p>"
        + walk_forward_diagnostics_table()
    )


def walk_forward_diagnostics_table() -> str:
    path = Settings().run_dir / "model_walk_forward_diagnostics.csv"
    if not path.exists():
        return "<p>No hay diagnostico walk-forward disponible para esta ejecucion.</p>"
    diagnostics = pd.read_csv(path)
    cols = [
        "snapshot_date", "mode", "fallback_reason", "training_rows", "training_years",
        "training_tickers", "alpha_label_observable_rows", "alpha_label_fallback_rows",
    ]
    return table(select(diagnostics, cols))


def drawdown_episode_table(vs: pd.DataFrame) -> str:
    if vs.empty or "portfolio_value" not in vs.columns or "date" not in vs.columns:
        return ""
    from module.report import drawdown_episodes

    episodes = drawdown_episodes(vs)
    if episodes.empty:
        return "<h2>Episodios de Drawdown</h2><p>No se detectaron caidas relevantes.</p>"
    return "<h2>Episodios de Drawdown</h2>" + table(episodes)


def audit_file_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
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
    rows = []
    for name in descriptions:
        df = tables.get(name, pd.DataFrame())
        rows.append({"file": f"audit/{name}.csv", "purpose": descriptions[name], "rows": len(df), "columns": len(df.columns) if not df.empty else 0, "read_first": "No"})
    return pd.DataFrame(rows)


def build_nav() -> str:
    groups = []
    for group_name, pages in PAGE_GROUPS.items():
        css_class = "nav-group" if group_name == "Principal" else "nav-group secondary"
        links = " ".join(f'<a href="{page}">{page.replace(".html", "")}</a>' for page in pages)
        groups.append(f'<span class="{css_class}">{links}</span>')
    return '<span class="nav-sep"></span>'.join(groups)


def build_viewer(settings: Settings) -> Path:
    run_dir = settings.run_dir
    viewer_dir = run_dir / "viewer"
    charts_dir = viewer_dir / "charts"
    viewer_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    tables = {path.stem: pd.read_csv(path) for path in (run_dir / "audit").glob("*.csv")} if (run_dir / "audit").exists() else {}
    tables.update({path.stem: pd.read_csv(path) for path in run_dir.glob("*.csv")})
    if "watchlist" not in tables and (PROCESSED_DIR / "watchlist.parquet").exists():
        tables["watchlist"] = pd.read_parquet(PROCESSED_DIR / "watchlist.parquet")
    charts = build_charts(charts_dir, tables)
    nav = build_nav()
    remove_stale_pages(viewer_dir)
    for page in PAGES:
        try:
            body = page_body(page, tables, charts)
        except Exception:
            log.exception("Failed to render viewer page %s, writing placeholder instead", page)
            body = f"<h1>{page}</h1><p>Esta pagina no pudo generarse en esta ejecucion. Revisa los logs del pipeline.</p>"
        (viewer_dir / page).write_text(layout(page, nav, body), encoding="utf-8")
    holdings = tables.get("portfolio_monthly_holdings", pd.DataFrame())
    for ticker in sorted(holdings.get("ticker", pd.Series(dtype=str)).dropna().unique()):
        try:
            ticker_rows = holdings[holdings["ticker"] == ticker]
            body = f"<h1>{ticker}</h1>" + figure(charts.get(f"position_{ticker}"), f"{ticker} thesis and allocation over time") + table(ticker_rows)
        except Exception:
            log.exception("Failed to render position page for %s, writing placeholder instead", ticker)
            body = f"<h1>{ticker}</h1><p>Esta pagina no pudo generarse en esta ejecucion.</p>"
        (viewer_dir / f"position_{ticker}.html").write_text(layout(ticker, nav, body), encoding="utf-8")
    write_results_explainer(run_dir, viewer_dir, charts, tables)
    return viewer_dir


def remove_stale_pages(viewer_dir: Path) -> None:
    keep = set(PAGES)
    for path in viewer_dir.glob("*.html"):
        if path.name.startswith("position_"):
            continue
        if path.name not in keep:
            path.unlink()
