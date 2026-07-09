"""Artifact catalog and results explainer for the static viewer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .shared import PAGES


def write_results_explainer(run_dir: Path, viewer_dir: Path, charts: dict[str, str], tables: dict[str, pd.DataFrame]) -> None:
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
        "dashboard.html": "Dashboard principal interactivo: KPIs, graficos, filtros, cartera, oportunidades, operaciones y aprendizaje.",
        "current_portfolio.html": "Cartera actual, pesos, scores y tesis de salida.",
        "tracking_dashboard.html": "Seguimiento mensual compacto de performance, alpha, compras, ventas y composicion.",
        "portfolio_vs_benchmark.html": "Comparativa completa entre cartera y benchmark, con curva de valor, alpha mensual y drawdown.",
        "allocation_dashboard.html": "Panel de pesos por ticker, con asignacion actual y deriva historica de la cartera.",
        "watchlist.html": "Mapa de oportunidades: cruza valoracion y conviccion para identificar candidatos atractivos.",
        "top_opportunities.html": "Mejores oportunidades actuales segun el ranking independiente del universo.",
        "strategy_learning.html": "Pistas automaticas para futuras mejoras de reglas, pesos y umbrales.",
        "model_explainability.html": "Importancia de componentes y variables del modelo de scoring.",
        "action_journal.html": "Diario unificado de compras y ventas con resultado economico cuando la posicion esta cerrada.",
        "sector_exposure.html": "Exposicion sectorial de la cartera a traves del tiempo.",
        "position_performance.html": "Rendimiento ganado por accion frente al benchmark durante el periodo real de holding.",
        "buy_rationale.html": "Explicacion tabular de por que se compro cada accion.",
        "sell_reasons.html": "Resumen de categorias de salida y tickers afectados.",
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
    lines = ["# Explicacion de resultados", "", f"Run analizado: `{run_dir.name}`.", "", "## Graficos principales", ""]
    for key, src in sorted(charts.items()):
        if not key.startswith("position_"):
            lines.append(f"- `viewer/{src}`: {chart_descriptions.get(key, 'Grafico generado para apoyar el analisis visual del visor.')}")
    lines.extend(["", "## HTML del visor", ""])
    lines.extend(f"- `viewer/{page}`: {page_descriptions.get(page, 'Pagina HTML del visor de resultados.')}" for page in PAGES)
    position_pages = sorted(path.name for path in viewer_dir.glob("position_*.html"))
    if position_pages:
        lines.append(f"- `viewer/position_*.html`: paginas individuales por posicion ({len(position_pages)} tickers) con tabla historica y grafico de conviccion/peso.")
    lines.extend(["", "## Archivos de resultados", ""])
    for path in sorted(run_dir.glob("*")):
        if path.is_file() and path.name != "expl_results.md":
            description = csv_descriptions.get(path.name, "Artefacto auxiliar generado por el pipeline.")
            extra = f" Contiene {len(tables[path.stem])} filas y {len(tables[path.stem].columns)} columnas." if path.suffix == ".csv" and path.stem in tables else ""
            lines.append(f"- `{path.name}`: {description}{extra}")
    if (run_dir / "audit").exists():
        lines.extend(["", "## Archivos de auditoria", ""])
        for path in sorted((run_dir / "audit").glob("*.csv")):
            df = tables.get(path.stem, pd.DataFrame())
            extra = f" Contiene {len(df)} filas y {len(df.columns)} columnas." if not df.empty else ""
            lines.append(f"- `audit/{path.name}`: {csv_descriptions.get(path.name, 'Tabla pesada para trazabilidad y depuracion.')}{extra}")
    lines.extend(["", "## Lectura recomendada", "", "1. Empieza por `viewer/dashboard.html` para ver KPIs, graficos, filtros, cartera, oportunidades y aprendizaje en una sola vista.", "2. Usa `viewer/index.html` si prefieres la vista estatica clasica.", "3. Usa `viewer/current_portfolio.html` para ver que hay ahora y por que se mantiene.", "4. Usa `viewer/action_journal.html` y `viewer/position_performance.html` para auditar compras, ventas y ganancias.", "5. Revisa `viewer/buy_rationale.html` y `viewer/sell_reasons.html` para entender por que entra o sale capital.", "6. Consulta `viewer/watchlist.html` y `viewer/top_opportunities.html` para candidatos futuros.", "7. Revisa `viewer/strategy_learning.html` para ver pistas de mejora acumuladas.", "8. Entra en `viewer/audit.html` solo si necesitas reconstruir o depurar la simulacion completa."])
    (run_dir / "expl_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "result_manifest.json").write_text(json.dumps(artifact_manifest(run_dir, charts, tables, csv_descriptions, page_descriptions, chart_descriptions), indent=2, default=str), encoding="utf-8")


def artifact_manifest(run_dir: Path, charts: dict[str, str], tables: dict[str, pd.DataFrame], csv_descriptions: dict[str, str], page_descriptions: dict[str, str], chart_descriptions: dict[str, str]) -> list[dict]:
    rows: list[dict] = []
    first_read = {"executive_summary.csv", "current_portfolio.csv", "tracking_dashboard.csv", "action_journal.csv", "position_performance.csv", "buy_rationale.csv", "sell_reasons_summary.csv", "portfolio_vs_benchmark.csv", "strategy_learning_log.csv", "improvement_backlog.csv", "index.html", "dashboard.html", "current_portfolio.html", "action_journal.html", "position_performance.html", "portfolio_vs_benchmark.html", "strategy_learning.html"}
    for path in sorted(run_dir.glob("*.csv")):
        df = tables.get(path.stem, pd.DataFrame())
        rows.append({"path": path.name, "kind": "csv", "tier": "executive", "read_first": path.name in first_read, "rows": len(df), "columns": len(df.columns) if not df.empty else 0, "purpose": csv_descriptions.get(path.name, "Artefacto de resultados.")})
    if (run_dir / "audit").exists():
        for path in sorted((run_dir / "audit").glob("*.csv")):
            df = tables.get(path.stem, pd.DataFrame())
            rows.append({"path": f"audit/{path.name}", "kind": "csv", "tier": "audit", "read_first": False, "rows": len(df), "columns": len(df.columns) if not df.empty else 0, "purpose": csv_descriptions.get(path.name, "Tabla pesada para trazabilidad y depuracion.")})
    for page in PAGES:
        rows.append({"path": f"viewer/{page}", "kind": "html", "tier": "executive" if page != "audit.html" else "audit", "read_first": page in first_read, "purpose": page_descriptions.get(page, "Pagina del visor.")})
    for key, src in sorted(charts.items()):
        rows.append({"path": f"viewer/{src}", "kind": "png", "tier": "detail" if key.startswith("position_") else "executive", "read_first": key in {"portfolio_vs_benchmark", "period_alpha", "position_performance_bars", "sector_exposure"}, "purpose": chart_descriptions.get(key, "Grafico de apoyo.")})
    return rows
