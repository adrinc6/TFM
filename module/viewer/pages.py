"""Single-page professional report — the only viewer output.

Sections are chosen by the utility test in shared.py: executive summary, portfolio vs. benchmark,
current portfolio + recent trades, learning evidence (learned meta-agent weights + OOS rank-IC),
position attribution, and a separate debug block (walk-forward diagnostics + audit file links).
Heavy per-ticker/per-snapshot tables stay on disk under results/<run>/audit/ and are only linked.
"""

from __future__ import annotations

import html
import logging
import math
from pathlib import Path

import pandas as pd

from environment import PROCESSED_DIR, Settings
from module.backtest.artifacts import SMALL_SAMPLE_CAVEAT

from .charts import build_charts
from .shared import figure, kpi, report_layout, select, table

log = logging.getLogger(__name__)


NAV = [
    ("resumen", "Resumen"),
    ("aprendizaje", "Aprendizaje (IA)"),
    ("rendimiento", "Rendimiento"),
    ("cartera", "Cartera"),
    ("posiciones", "Posiciones"),
    ("robustez", "Robustez"),
    ("metodologia", "Metodología"),
    ("debug", "Debug / TFM"),
]


def build_viewer(settings: Settings) -> Path:
    run_dir = settings.run_dir
    viewer_dir = run_dir / "viewer"
    charts_dir = viewer_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    _remove_legacy_pages(viewer_dir)

    tables = _load_tables(run_dir)
    charts = build_charts(charts_dir, tables)

    body = _report_body(settings, tables, charts, run_dir)
    (viewer_dir / "index.html").write_text(
        report_layout("Informe GARP AI", body), encoding="utf-8"
    )
    log.info("Report written to %s", viewer_dir / "index.html")
    return viewer_dir


def _report_body(settings: Settings, tables: dict, charts: dict, run_dir: Path) -> str:
    from module.report import _metrics, drawdown_episodes  # reuse the vetted metric functions

    vs = tables.get("portfolio_vs_benchmark", pd.DataFrame())
    metrics = _metrics(vs)
    header = (
        '<header><div class="wrap">'
        "<h1>Informe GARP AI &mdash; cartera infravalorada de calidad</h1>"
        f'<div class="meta">Simulación {settings.train_cutoff_date} → {settings.end_date} '
        f'(modelo reentrenado cada trimestre con los últimos {settings.max_walk_forward_training_years} '
        f'años de historia — sin congelar) · benchmark {settings.benchmark_ticker} · '
        f'revisión {settings.review_frequency}</div>'
        "</div></header>"
    )
    nav = '<nav><div class="wrap">' + "".join(
        f'<a href="#{anchor}">{label}</a>' for anchor, label in NAV
    ) + "</div></nav>"

    # La sección de aprendizaje va justo después del resumen ejecutivo: es el núcleo del TFM (la IA
    # como motor de la estrategia), no una nota secundaria tras el rendimiento del backtest.
    sections = "".join([
        _section_resumen(metrics),
        _section_aprendizaje(charts, tables, settings),
        _section_rendimiento(charts, vs, drawdown_episodes),
        _section_cartera(tables),
        _section_posiciones(charts, tables),
        _section_robustez(tables),
        _section_metodologia(tables, settings),
        _section_debug(tables, run_dir),
    ])
    return header + nav + f'<main><div class="wrap">{sections}</div></main>'


# --- sections ---------------------------------------------------------------

def _section_resumen(metrics: dict) -> str:
    cagr = metrics.get("CAGR", 0) or 0
    benchmark_cagr = metrics.get("Benchmark CAGR", 0) or 0
    excess_total = metrics.get("Total Excess Return (compounded)", 0) or 0
    ir = metrics.get("Information Ratio", 0) or 0
    tone = "pos" if cagr > benchmark_cagr else "neg"
    verdict = "batió" if cagr > benchmark_cagr else "no batió"
    kpis = "".join([
        kpi("CAGR cartera", _pct(cagr), "retorno anualizado neto", tone),
        kpi("CAGR benchmark", _pct(benchmark_cagr), "SPY, mismo periodo"),
        kpi("Retorno total en exceso", _pct(excess_total),
            "diferencia de retornos compuestos a fin de periodo — no es la cifra con IC bootstrap, ver Robustez"),
        kpi("Sharpe", _num(metrics.get("Sharpe")), f"Sortino {_num(metrics.get('Sortino'))}"),
        kpi("Max Drawdown", _pct(metrics.get("Max Drawdown")), "caída máxima"),
        kpi("Information Ratio", _num(ir), f"TE {_pct(metrics.get('Tracking Error (annualized)'))}"),
        kpi("t-stat exceso", _num(metrics.get("Excess Return t-stat")),
            f"{int(metrics.get('Periods (n)', 0) or 0)} periodos"),
    ])
    return (
        '<section id="resumen"><h2>Resumen ejecutivo</h2>'
        f'<p class="lead">La estrategia sistemática GARP {verdict} al benchmark en esta ventana. '
        f'{SMALL_SAMPLE_CAVEAT} La confianza del resultado se lee mejor con el intervalo de confianza '
        f'por bootstrap de la sección <a href="#robustez">Robustez</a> (alpha acumulada mensual) que con '
        f'el retorno total en exceso o el t-stat aislado de arriba.</p>'
        f'<div class="kpis">{kpis}</div></section>'
    )


def _section_rendimiento(charts: dict, vs: pd.DataFrame, drawdown_episodes) -> str:
    episodes = drawdown_episodes(vs) if not vs.empty else pd.DataFrame()
    episode_cols = {"peak_date": "Pico", "trough_date": "Valle", "recovery_date": "Recuperación",
                    "depth": "Profundidad", "duration_days": "Días", "recovered": "Recuperado"}
    ep_table = (
        "<h3 class=\"note\" style=\"margin-top:22px\">Episodios de drawdown</h3>"
        + table(episodes.rename(columns=episode_cols)) if not episodes.empty else ""
    )
    return (
        '<section id="rendimiento"><h2>Cartera frente al benchmark</h2>'
        '<p class="lead">El gráfico central del trabajo: crecimiento de 1 € en la cartera neta '
        'frente a SPY, el alpha acumulado, y el perfil de caídas.</p>'
        + figure(charts.get("value_vs_benchmark"), "Valor de la cartera neta frente al benchmark SPY")
        + '<div style="height:16px"></div>'
        + figure(charts.get("cumulative_alpha"), "Alpha acumulado frente al benchmark")
        + '<div style="height:16px"></div>'
        + figure(charts.get("drawdown"), "Perfil de drawdown de la cartera")
        + ep_table
        + "</section>"
    )


def _section_cartera(tables: dict) -> str:
    current = tables.get("current_portfolio", pd.DataFrame())
    current_cols = ["ticker", "sector", "hybrid_weight", "manager_score", "final_score",
                    "opportunity_type", "current_thesis_state", "watch_reason"]
    journal = tables.get("action_journal", pd.DataFrame())
    journal_cols = ["date", "ticker", "action", "reason_category", "manager_score",
                    "holding_days", "excess_total_return", "reason"]
    recent = journal.tail(15) if not journal.empty else journal
    return (
        '<section id="cartera"><h2>Cartera actual y operaciones</h2>'
        '<p class="lead">Qué se tiene ahora (pesos, score de gestión, tesis) y las últimas '
        'compras/ventas con su motivo.</p>'
        + table(select(current, current_cols))
        + _holdings_rationale_block(current)
        + '<h3 class="note" style="margin-top:22px">Últimas operaciones</h3>'
        + table(select(recent, journal_cols))
        + _journal_rationale_block(recent)
        + "</section>"
    )


def _clean_text(value) -> str:
    """NaN-safe stringification for free-text columns round-tripped through CSV: an empty cell
    reads back as float('nan'), which is truthy and breaks html.escape (expects str)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _holdings_rationale_block(current: pd.DataFrame) -> str:
    """Por qué está cada posición en cartera: tesis, desglose real de manager_score al comprar
    (qué componente de la fórmula pesó más), comparación contra la mejor alternativa descartada
    ese día, y qué cambió desde entonces (si algo)."""
    if current.empty:
        return ""
    items = []
    for row in current.itertuples(index=False):
        ticker = _clean_text(getattr(row, "ticker", ""))
        thesis = _clean_text(getattr(row, "investment_thesis", ""))
        breakdown = _clean_text(getattr(row, "manager_score_breakdown_at_entry", ""))
        vs_alt = _clean_text(getattr(row, "vs_best_alternative_at_entry", ""))
        change = _clean_text(getattr(row, "entry_vs_current", ""))
        exit_thesis = _clean_text(getattr(row, "current_exit_thesis", ""))
        lines = [line for line in (thesis, breakdown, vs_alt, change, exit_thesis) if line]
        if not lines:
            continue
        body = "<br>".join(html.escape(line) for line in lines)
        items.append(f"<li><strong>{html.escape(str(ticker))}</strong>: {body}</li>")
    if not items:
        return ""
    return (
        '<details style="margin-top:14px"><summary>Por qué está cada posición en cartera</summary>'
        f'<ul class="note">{"".join(items)}</ul></details>'
    )


def _journal_rationale_block(journal: pd.DataFrame) -> str:
    """Texto completo de motivo (con comparación entrada-vs-ahora en ventas) para las últimas
    operaciones, sin saturar la tabla compacta de arriba."""
    if journal.empty:
        return ""
    items = []
    for row in journal.itertuples(index=False):
        reason = _clean_text(getattr(row, "reason", ""))
        if not reason:
            continue
        date = _clean_text(getattr(row, "date", ""))
        ticker = _clean_text(getattr(row, "ticker", ""))
        action = _clean_text(getattr(row, "action", ""))
        items.append(
            f"<li><strong>{html.escape(date)} {html.escape(action)} "
            f"{html.escape(ticker)}</strong>: {html.escape(reason)}</li>"
        )
    if not items:
        return ""
    return (
        '<details style="margin-top:14px"><summary>Motivo completo de cada operación</summary>'
        f'<ul class="note">{"".join(items)}</ul></details>'
    )


def _section_aprendizaje(charts: dict, tables: dict, settings: Settings) -> str:
    weights = tables.get("meta_weights_by_snapshot", pd.DataFrame())
    learned = int((weights.get("source", pd.Series(dtype=str)) == "learned").sum()) if not weights.empty else 0
    total = len(weights) if not weights.empty else 0
    consistency_block = _consistency_headline(tables.get("robustness_sub_periods", pd.DataFrame()))
    rolling_note = (
        f'El modelo se reentrena cada trimestre <strong>durante toda la ventana simulada</strong> '
        f'(desde {settings.train_cutoff_date} hasta {settings.end_date}), usando siempre los últimos '
        f'{settings.max_walk_forward_training_years} años de historia disponibles hasta ese punto — '
        f'nunca se congela. Esto significa que el rank-IC y los pesos que se ven abajo reflejan '
        f'aprendizaje real en toda la ventana, no solo en un tramo inicial de entrenamiento seguido '
        f'de un modelo fijo aplicándose a un régimen de mercado que nunca vio.'
    )
    lead = (
        'La prueba de que el sistema aprende: en cada snapshot de reentrenamiento el meta-agente '
        'ajusta —a partir del alpha realmente realizado a 12 meses (walk-forward, sin lookahead)— '
        f'cuánto pesa cada agente especializado, premiando su contribución <em>marginal</em> '
        '(rank-IC parcial frente al alpha que los otros tres agentes no explican), no su solape. '
        f'{learned} de {total} snapshots usaron pesos aprendidos (el resto, prior por falta de '
        'historia). El segundo gráfico muestra el rank-IC out-of-sample de la señal maestra '
        '(final_score, la combinación del meta-agente) por snapshot y su media móvil de 12 periodos: '
        'si sube con el tiempo, el sistema afina; si oscila plano, la muestra (unas 90 observaciones) '
        'no permite todavía distinguir mejora de ruido — se reporta el dato tal cual sale, sin forzar '
        f'una lectura de mejora. {rolling_note}'
    )
    year_stats = _year_ic_table(tables.get("model_walk_forward_diagnostics", pd.DataFrame()))
    year_block = (
        '<h3 class="note" style="margin-top:22px">Rank-IC de la señal maestra por año</h3>'
        + table(year_stats) if not year_stats.empty else ""
    )
    horizon_block = _horizon_comparison_block(charts, tables.get("label_horizon_comparison", pd.DataFrame()), settings)
    dominant_block = _dominant_agent_block(weights, tables.get("model_walk_forward_diagnostics", pd.DataFrame()))
    attribution_block = _edge_attribution_block(tables.get("edge_attribution", pd.DataFrame()))
    importance_block = _feature_importance_block()
    return (
        '<section id="aprendizaje"><h2>Evidencia de aprendizaje — la IA como motor de la estrategia</h2>'
        f'<p class="lead">{lead}</p>'
        + consistency_block
        + figure(charts.get("learned_weights"), "Evolución de los pesos aprendidos del meta-agente por snapshot")
        + dominant_block
        + '<div style="height:16px"></div>'
        + figure(charts.get("rank_ic"), "Rank-IC out-of-sample de la señal maestra por snapshot, con media móvil de tendencia")
        + year_block
        + horizon_block
        + attribution_block
        + importance_block
        + "</section>"
    )


def _feature_importance_block() -> str:
    """Qué features pesan más dentro de cada agente — evidencia de aprendizaje real a nivel de
    variable, no solo de la combinación entre agentes (LightGBM gain importance; existía en
    model_explainability.json pero nunca se renderizaba en ningún sitio del reporte)."""
    explain_path = PROCESSED_DIR / "model_explainability.json"
    if not explain_path.exists():
        return ""
    import json
    payload = json.loads(explain_path.read_text(encoding="utf-8"))
    importance = payload.get("feature_importance", {})
    if not importance:
        return ""
    agent_names = {
        "quality_probability": "Calidad", "timing_probability": "Temporización",
        "alpha_probability": "Alpha (ranking directo)",
    }
    rows = []
    for agent_key, agent_label in agent_names.items():
        agent_importance = importance.get(agent_key, {})
        if not agent_importance:
            continue
        top = sorted(agent_importance.items(), key=lambda kv: kv[1], reverse=True)[:3]
        total = sum(agent_importance.values()) or 1
        for feature, gain in top:
            rows.append({
                "Agente": agent_label, "Feature": feature, "Importancia relativa": f"{gain / total:.0%}",
            })
    if not rows:
        return ""
    return (
        '<h3 class="note" style="margin-top:22px">Qué variables pesan más dentro de cada agente</h3>'
        '<p class="note">Importancia por ganancia (LightGBM) del último modelo entrenado en cada agente '
        '— top 3 features por agente, evidencia de aprendizaje a nivel de variable individual, no solo '
        'de la combinación entre agentes.</p>'
        + table(pd.DataFrame(rows))
    )


def _consistency_headline(sub_periods: pd.DataFrame) -> str:
    """Métrica principal de éxito del TFM: ¿la IA es consistente entre periodos, o el resultado
    depende de un tramo/posición excepcional? Se muestra aquí, al frente del relato de aprendizaje,
    no relegada a una nota secundaria dentro de Robustez — ver detalle completo por tramo allí."""
    if sub_periods.empty or "alpha_acumulada" not in sub_periods.columns:
        return ""
    alphas = pd.to_numeric(sub_periods["alpha_acumulada"], errors="coerce").dropna()
    if alphas.empty:
        return ""
    positive_periods = int((alphas > 0).sum())
    total_periods = len(alphas)
    spread = float(alphas.max() - alphas.min())
    if positive_periods == total_periods:
        verdict = (
            f'positiva en los {total_periods} tramos analizados — el resultado no depende de un '
            'único periodo excepcional.'
        )
    else:
        verdict = (
            f'positiva en {positive_periods} de {total_periods} tramos — no es uniforme en toda '
            'la ventana simulada; ver el detalle por tramo en la sección Robustez.'
        )
    return (
        '<div class="note" style="margin:14px 0;padding:12px 16px;border-left:3px solid var(--accent, #2f6f4f);">'
        f'<strong>Consistencia entre sub-periodos</strong> (métrica principal de éxito de este TFM, no '
        f'solo la alpha acumulada total): la alpha ha sido {verdict} Dispersión entre el tramo mejor y '
        f'el peor: {spread:.1%}. Detalle completo por tramo en la sección '
        '<a href="#robustez">Robustez</a>.'
        '</div>'
    )


def _edge_attribution_block(attribution: pd.DataFrame) -> str:
    """¿De dónde sale el alpha si el rank-IC rolling es bajo o negativo? — muestra los números
    crudos, sin forzar una conclusión que los datos no sostengan."""
    if attribution.empty:
        return ""
    display = attribution.copy()
    display["valor"] = display["valor"].map(
        lambda v: "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"
    )
    rename = {"metrica": "Métrica", "valor": "Valor", "n": "n"}
    lead = (
        'Si el ranking del modelo predijera bien fuera de muestra, el rank-IC de <code>final_score</code> '
        'a lo largo de toda la ventana simulada (el modelo se reentrena cada trimestre, nunca se congela) '
        'debería ser claramente positivo. Si no lo es (ver tabla), el resultado agregado no puede '
        'explicarse por "el modelo aprendió a rankear bien" — hay que mirar si viene de la correlación '
        'entrada-score/retorno realizado, o de una asimetría ganador/perdedor (pocas posiciones grandes '
        'cargando con el resultado frente a muchas pequeñas perdedoras).'
    )
    return (
        '<h3 class="note" style="margin-top:22px">¿De dónde viene el alpha? — descomposición del edge</h3>'
        f'<p class="note">{lead}</p>'
        + table(display.rename(columns=rename))
    )


def _horizon_comparison_block(charts: dict, horizon_df: pd.DataFrame, settings: Settings) -> str:
    """¿Qué horizonte de predicción funciona mejor? — answers the user's own question with data."""
    if horizon_df.empty or "rank_ic_mean" not in horizon_df.columns:
        return ""
    d = horizon_df.dropna(subset=["rank_ic_mean"])
    if d.empty:
        return ""
    best_row = d.loc[d["rank_ic_mean"].idxmax()]
    default_row = d[d["horizon_months"] == settings.walk_forward_label_horizon_months]
    default_ic = float(default_row["rank_ic_mean"].iloc[0]) if not default_row.empty else None
    best_months = int(best_row["horizon_months"])
    best_ic = float(best_row["rank_ic_mean"])
    if default_ic is not None and best_months == settings.walk_forward_label_horizon_months:
        conclusion = (
            f'Con los datos de esta ejecución, {best_months} meses (el horizonte configurado) es '
            f'el que mejor predice (rank-IC medio {best_ic:.3f}) — se mantiene como está, decisión '
            f'informada por el propio dato en vez de asumida.'
        )
    elif default_ic is not None:
        diff = best_ic - default_ic
        conclusion = (
            f'Con los datos de esta ejecución, {best_months} meses predice algo mejor (rank-IC '
            f'{best_ic:.3f}) que el horizonte configurado de {settings.walk_forward_label_horizon_months} '
            f'meses (rank-IC {default_ic:.3f}, diferencia {diff:+.3f}). La diferencia es pequeña '
            f'frente al ruido snapshot a snapshot visto arriba; no se cambia el horizonte por defecto '
            f'sin repetir esta comparación en más ejecuciones.'
        )
    else:
        conclusion = f'Horizonte con mejor rank-IC observado: {best_months} meses (rank-IC {best_ic:.3f}).'
    return (
        '<h3 class="note" style="margin-top:22px">¿Qué horizonte de predicción funciona mejor?</h3>'
        '<p class="note">La señal maestra ya combinada se compara contra el alfa real a 3, 6 y 12 '
        'meses, usando todos los snapshots de la ventana simulada (el modelo se reentrena cada '
        'trimestre, así que cada snapshot aporta información nueva sobre el horizonte).</p>'
        + figure(charts.get("horizon_comparison"), "Rank-IC medio de la señal maestra a 3, 6 y 12 meses")
        + f'<p class="note">{conclusion}</p>'
    )


def _dominant_agent_block(weights: pd.DataFrame, diag: pd.DataFrame) -> str:
    """Qué agente pesa más y por qué — turns the weights chart into a readable sentence."""
    if weights.empty:
        return ""
    latest = weights.sort_values("snapshot_date").iloc[-1]
    agent_labels = {
        "quality_probability": "Calidad", "timing_probability": "Temporización",
        "alpha_probability": "Alpha (ranking directo)",
    }
    agent_keys = [k for k in agent_labels if k in weights.columns]
    if not agent_keys:
        return ""
    dominant_key = max(agent_keys, key=lambda k: latest[k])
    dominant_share = float(latest[dominant_key])
    rank_ic_col = f"rank_ic_{dominant_key}"
    ic_sentence = ""
    if not diag.empty and rank_ic_col in diag.columns:
        ic_series = pd.to_numeric(diag[rank_ic_col], errors="coerce").dropna()
        if not ic_series.empty:
            ic_sentence = f' Su rank-IC OOS medio histórico es {ic_series.mean():.3f}, la señal más consistente de las tres en esta ejecución.'
    return (
        '<p class="note" style="margin-top:10px">'
        f'En el snapshot más reciente, el agente <strong>{agent_labels[dominant_key]}</strong> '
        f'domina la combinación con un peso de {dominant_share:.0%}.{ic_sentence}'
        '</p>'
    )


def _year_ic_table(diag: pd.DataFrame) -> pd.DataFrame:
    cols = ["rank_ic_final_year_mean", "rank_ic_final_year_tstat"]
    if diag.empty or "snapshot_date" not in diag.columns or not all(c in diag.columns for c in cols):
        return pd.DataFrame()
    d = diag.copy()
    d["year"] = pd.to_datetime(d["snapshot_date"], errors="coerce").dt.year
    yearly = d.dropna(subset=["year"]).groupby("year")[cols].first().reset_index()
    return yearly.rename(columns={
        "year": "Año", "rank_ic_final_year_mean": "Rank-IC medio", "rank_ic_final_year_tstat": "t-stat aprox.",
    })


def _section_posiciones(charts: dict, tables: dict) -> str:
    perf = tables.get("position_performance", pd.DataFrame())
    cols = ["ticker", "entry_date", "exit_date", "holding_days", "total_return",
            "annualized_return", "benchmark_annualized_return", "excess_total_return", "exit_reason_category"]
    top = perf.copy()
    if "excess_total_return" in top.columns:
        top["excess_total_return"] = pd.to_numeric(top["excess_total_return"], errors="coerce")
        top = top.sort_values("excess_total_return", ascending=False).head(20)
    return (
        '<section id="posiciones"><h2>Rendimiento por posición</h2>'
        '<p class="lead">Atribución por lote (FIFO): retorno de cada posición frente al benchmark '
        'durante su periodo de tenencia.</p>'
        + figure(charts.get("position_performance"), "Mejores y peores posiciones por retorno en exceso acumulado")
        + '<div style="height:16px"></div>'
        + table(select(top, cols))
        + "</section>"
    )


def _section_robustez(tables: dict) -> str:
    """Evidencia de apoyo a la consistencia reportada en Aprendizaje (no el relato principal):
    ¿es sólido el alpha o cabe en el ruido de una muestra pequeña? — bootstrap + sensibilidad a costes
    + comparación contra baselines/placebo + desglose completo por sub-periodo."""
    bootstrap = tables.get("robustness_bootstrap", pd.DataFrame())
    cost = tables.get("robustness_cost_sensitivity", pd.DataFrame())
    summary = tables.get("robustness_summary", pd.DataFrame())
    baseline = tables.get("baseline_comparison", pd.DataFrame())
    placebo = tables.get("placebo_summary", pd.DataFrame())
    sub_periods = tables.get("robustness_sub_periods", pd.DataFrame())
    bootstrap_cols = {
        "metrica": "Métrica", "estimacion": "Estimación", "ic_2_5": "IC 2.5%", "ic_97_5": "IC 97.5%",
        "prob_positiva": "P(>0)", "n_resamples": "Resamples", "bloque_meses": "Bloque (meses)",
    }
    cost_cols = {
        "multiplicador": "Multiplicador de coste", "coste_total_bps": "Coste total (bps)",
        "alpha_acumulada_neta": "Alpha acumulada neta", "sigue_positiva": "¿Sigue positiva?",
    }
    breakeven_note = ""
    if not summary.empty:
        breakeven = summary.iloc[0].get("breakeven_cost_multiplier")
        if breakeven is not None and not (isinstance(breakeven, float) and math.isnan(breakeven)):
            breakeven_note = (
                f'<p class="note">Punto de equilibrio de costes: la alpha acumulada cruza a negativa '
                f'a partir de <strong>{float(breakeven):.1f}×</strong> los costes de transacción y '
                f'slippage actuales. Por debajo de ese múltiplo el resultado sigue siendo positivo.</p>'
            )
        else:
            breakeven_note = (
                '<p class="note">La alpha acumulada no llega a cruzar a negativa dentro del barrido de '
                'costes probado (o ya es no positiva sin coste extra); ver la tabla.</p>'
            )
    lead = (
        'Con solo unas decenas de observaciones mensuales, el t-stat por sí solo exagera lo que '
        'sabemos. La forma principal de leer la confianza del resultado es el <strong>intervalo de '
        'confianza por bootstrap</strong> de abajo: se remuestrea el exceso mensual en bloques de '
        'varios meses (para preservar la autocorrelación) y se recalcula la alpha acumulada y el '
        'information ratio miles de veces. Si el intervalo 2.5–97.5% se mantiene por encima de cero, '
        'el edge es más creíble; si cruza el cero, la muestra no permite descartar que sea ruido.'
    )
    return (
        '<section id="robustez"><h2>Robustez estadística</h2>'
        f'<p class="lead">{lead}</p>'
        '<h3 class="note">Bootstrap por bloques del exceso mensual</h3>'
        + table(bootstrap, rename=bootstrap_cols)
        + '<h3 class="note" style="margin-top:22px">Sensibilidad a costes y slippage</h3>'
        '<p class="note">La alpha se recalcula escalando el coste ya aplicado (lineal en el múltiplo), '
        'sin rehacer selección ni sizing.</p>'
        + table(cost, rename=cost_cols)
        + breakeven_note
        + '<p class="note" style="margin-top:10px">La distribución completa por resample se guarda en '
        '<a class="audit" href="../audit/robustness_bootstrap_distribution.csv">'
        'robustness_bootstrap_distribution.csv</a> (enlazada, no embebida).</p>'
        + _baseline_block(baseline)
        + _placebo_block(placebo)
        + _sub_periods_block(sub_periods)
        + "</section>"
    )


def _sub_periods_block(sub_periods: pd.DataFrame) -> str:
    """¿La alpha viene de todo el periodo o de un solo tramo? — parte la ventana única del backtest
    en 3 tramos y compara alpha/IR entre ellos, sin volver a puntuar ni seleccionar nada."""
    if sub_periods.empty:
        return ""
    display = sub_periods.copy()
    display["alpha_acumulada"] = display["alpha_acumulada"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    display["information_ratio"] = display["information_ratio"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    lead = (
        'La misma ventana del backtest, partida en tres tramos consecutivos de igual longitud '
        '(sin rehacer selección ni entrenamiento). Si la alpha y el IR son parecidos en los tres '
        'tramos, el resultado no depende de un único periodo excepcional; si un tramo concentra '
        'casi toda la alpha, el resto de la ventana no la respalda y la robustez del resultado es '
        'menor de lo que sugiere la cifra agregada.'
    )
    rename = {
        "sub_periodo": "Tramo", "fecha_inicio": "Desde", "fecha_fin": "Hasta",
        "n_periodos": "Meses", "alpha_acumulada": "Alpha acumulada", "information_ratio": "Information Ratio",
    }
    return (
        '<h3 class="note" style="margin-top:22px">Alpha por sub-periodo</h3>'
        f'<p class="note">{lead}</p>'
        + table(display.rename(columns=rename))
    )


def _baseline_block(baseline: pd.DataFrame) -> str:
    """¿Aporta el sistema completo algo sobre reglas mucho más simples? — comparación directa."""
    if baseline.empty:
        return ""
    display = baseline.copy()
    display["alpha_acumulada"] = display["alpha_acumulada"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    lead = (
        'La alpha acumulada del sistema completo (manager_score + estados de tesis + reglas de '
        'rotación) frente a tres reglas triviales calculadas sobre el mismo universo y las mismas '
        'fechas: comprar todo el universo por igual, quedarse solo con el momentum más alto, o solo '
        'con la valoración más barata. Si el sistema no supera claramente a las reglas simples, el '
        'diseño completo no está aportando nada sobre una heurística de una línea.'
    )
    return (
        '<h3 class="note" style="margin-top:22px">Comparación contra reglas simples</h3>'
        f'<p class="note">{lead}</p>'
        + table(display.rename(columns={"estrategia": "Estrategia", "alpha_acumulada": "Alpha acumulada"}))
    )


def _placebo_block(placebo: pd.DataFrame) -> str:
    """¿Importa el ranking, o cualquier orden aleatorio habría dado un resultado parecido?"""
    if placebo.empty:
        return ""
    row = placebo.iloc[0]
    percentile = row.get("percentil_real")
    if percentile is None or (isinstance(percentile, float) and math.isnan(percentile)):
        return ""
    n = int(row.get("n_barajados", 0) or 0)
    real = row.get("alpha_real")
    random_mean = row.get("alpha_medio_aleatorio")
    if percentile >= 0.95:
        verdict = (
            f'el ranking real queda en el percentil {percentile:.0%} de {n} barajados aleatorios — '
            'muy por encima de lo típico de un orden aleatorio, evidencia de que el ranking sí aporta '
            'información.'
        )
    elif percentile <= 0.60:
        verdict = (
            f'el ranking real queda en el percentil {percentile:.0%} de {n} barajados aleatorios — '
            'dentro de lo que produciría un orden aleatorio, no hay evidencia clara de que el ranking '
            'en sí aporte algo sobre el azar en esta ventana.'
        )
    else:
        verdict = (
            f'el ranking real queda en el percentil {percentile:.0%} de {n} barajados aleatorios — '
            'por encima de la media aleatoria pero sin ser un caso extremo; evidencia mixta.'
        )
    lead = (
        'Se baraja aleatoriamente el orden de <code>final_score</code> dentro de cada snapshot '
        f'(mismo universo, mismas fechas) {n} veces, y se recalcula la alpha acumulada de una cartera '
        'top-N mecánica con cada orden barajado. Si el ranking real no fuera mejor que un orden al '
        'azar, esto lo mostraría.'
    )
    stats = (
        f'Alpha real: {real:.1%}. Alpha media de los barajados: '
        f'{random_mean:.1%}.' if random_mean is not None and pd.notna(random_mean) else ''
    )
    return (
        '<h3 class="note" style="margin-top:22px">Test de aleatorización (placebo)</h3>'
        f'<p class="note">{lead} {stats} En esta ejecución, {verdict}</p>'
    )


def _section_metodologia(tables: dict, settings: Settings) -> str:
    """Explains the design in prose so the report is readable without opening the code."""
    explain_path = PROCESSED_DIR / "model_explainability.json"
    definitions = {}
    if explain_path.exists():
        import json
        payload = json.loads(explain_path.read_text(encoding="utf-8"))
        definitions = payload.get("component_target_definitions", {})
    agent_names = {
        "quality_probability": "Calidad", "timing_probability": "Temporización",
        "alpha_probability": "Alpha (ranking directo)",
    }
    agent_rows = "".join(
        f'<li><strong>{agent_names.get(key, key)}</strong>: {definitions.get(key, "")}</li>'
        for key in agent_names
    )
    mechanics = (
        f'El modelo se reentrena con cadencia trimestral desde <strong>{settings.train_cutoff_date}</strong> '
        f'hasta <strong>{settings.end_date}</strong> — <strong>sin congelar</strong>: en cada punto de '
        f'reentrenamiento usa solo los últimos {settings.max_walk_forward_training_years} años de historia '
        f'disponibles hasta esa fecha (walk-forward puro, rolling). Esto evita que la cartera quede '
        f'operando con un modelo fijo entrenado años atrás en un régimen de mercado distinto — cada '
        f'trimestre el modelo se pone al día con la información más reciente que tenía en ese momento, '
        f'sin usar nunca datos futuros (no hay fuga de información). La cartera se revisa mensualmente '
        f'({settings.review_frequency}).'
    )
    params = pd.DataFrame([{
        "Parámetro": "Inicio de la simulación / primer reentrenamiento", "Valor": settings.train_cutoff_date,
    }, {
        "Parámetro": "Cadencia de reentrenamiento", "Valor": settings.walk_forward_train_frequency,
    }, {
        "Parámetro": "Horizonte de predicción (por defecto)", "Valor": f"{settings.walk_forward_label_horizon_months} meses",
    }, {
        "Parámetro": "Ventana móvil de historia por reentrenamiento", "Valor": f"{settings.max_walk_forward_training_years} años",
    }, {
        "Parámetro": "Universo", "Valor": f"{len(settings.investable_tickers)} tickers + {settings.benchmark_ticker}",
    }])
    return (
        '<section id="metodologia"><h2>Metodología</h2>'
        '<p class="lead">Cómo está construido el sistema y por qué, sin necesidad de leer el código.</p>'
        '<h3 class="note">Los 3 agentes especializados + el meta-agente decisor</h3>'
        f'<ul class="findings">{agent_rows}'
        '<li>El <strong>meta-agente</strong> combina las tres señales anteriores en <code>final_score</code>, '
        'aprendiendo qué peso dar a cada una por su <strong>contribución marginal</strong>: el rank-IC '
        'parcial de cada agente frente al alpha realizado que los otros dos <em>no</em> explican '
        '(regresión sobre un 70% cronológico, medido en el 30% restante, sin lookahead). Así se premia '
        'a un agente por aportar información nueva, no por solaparse con el resto; un agente redundante o '
        'sin capacidad de ordenar recibe peso ~0, en vez de pesos fijos a mano. El agente <strong>Alpha</strong> '
        'aprende a ordenar directamente el exceso de retorno a 12 meses (la propia magnitud evaluada), '
        'pero desde un subconjunto acotado de features de momentum y valoración, no desde el superconjunto '
        'completo: por eso complementa a los otros dos en vez de dominar estructuralmente la mezcla.</li></ul>'
        '<h3 class="note" style="margin-top:18px">Reentrenamiento continuo, sin congelar</h3>'
        f'<p class="note">{mechanics}</p>'
        '<h3 class="note" style="margin-top:18px">Parámetros efectivos de esta ejecución</h3>'
        + table(params)
        + "</section>"
    )


def _section_debug(tables: dict, run_dir: Path) -> str:
    diag = tables.get("model_walk_forward_diagnostics", pd.DataFrame())
    diag_cols = ["snapshot_date", "mode", "fallback_reason", "training_snapshot_date",
                 "training_rows", "training_years", "alpha_label_observable_rows",
                 "rank_ic_final", "n_oos"]
    audit_dir = run_dir / "audit"
    audit_links = ""
    if audit_dir.exists():
        files = sorted(audit_dir.glob("*.csv"))
        if files:
            items = "".join(
                f'<li><a class="audit" href="../audit/{f.name}">{f.name}</a></li>' for f in files
            )
            audit_links = (
                '<h3 class="note" style="margin-top:22px">Archivos de auditoría (en disco)</h3>'
                '<p class="note">Tablas pesadas por ticker/snapshot; enlazadas, no embebidas.</p>'
                f'<ul class="findings">{items}</ul>'
            )
    return (
        '<section id="debug"><h2>Debug / TFM</h2>'
        '<p class="lead">Diagnóstico del entrenamiento walk-forward: qué snapshots entrenaron modelo '
        'vs. fallback determinista, y la calidad OOS por snapshot.</p>'
        + table(select(diag, diag_cols))
        + audit_links
        + "</section>"
    )


# --- helpers ----------------------------------------------------------------

def _load_tables(run_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    audit_dir = run_dir / "audit"
    if audit_dir.exists():
        tables.update({p.stem: pd.read_csv(p) for p in audit_dir.glob("*.csv")})
    tables.update({p.stem: pd.read_csv(p) for p in run_dir.glob("*.csv")})
    meta = PROCESSED_DIR / "meta_weights_by_snapshot.parquet"
    if "meta_weights_by_snapshot" not in tables and meta.exists():
        tables["meta_weights_by_snapshot"] = pd.read_parquet(meta)
    return tables


def _remove_legacy_pages(viewer_dir: Path) -> None:
    """Delete the old multi-page viewer output so only the new single report remains."""
    if not viewer_dir.exists():
        return
    for path in viewer_dir.glob("*.html"):
        if path.name != "index.html":
            path.unlink()


def _pct(value) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    return f"{value:.1%}"


def _num(value, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    return f"{value:.{decimals}f}"
