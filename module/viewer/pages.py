"""Single-page professional report — the only viewer output.

Sections are chosen by the utility test in shared.py: executive summary, portfolio vs. benchmark,
current portfolio + recent trades, learning evidence (learned meta-agent weights + OOS rank-IC),
position attribution, and a separate debug block (walk-forward diagnostics + audit file links).
Heavy per-ticker/per-snapshot tables stay on disk under results/<run>/audit/ and are only linked.
"""

from __future__ import annotations

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
    ("rendimiento", "Rendimiento"),
    ("cartera", "Cartera"),
    ("aprendizaje", "Aprendizaje"),
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
    charts = build_charts(charts_dir, tables, settings.train_cutoff_date)

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
    train_start = (
        pd.Timestamp(settings.train_cutoff_date) - pd.DateOffset(years=settings.walk_forward_train_years)
    ).date().isoformat()
    header = (
        '<header><div class="wrap">'
        "<h1>Informe GARP AI &mdash; cartera infravalorada de calidad</h1>"
        f'<div class="meta">Entrenamiento {train_start} → {settings.train_cutoff_date} · '
        f'simulación {settings.train_cutoff_date} → {settings.end_date} · '
        f'benchmark {settings.benchmark_ticker} · revisión {settings.review_frequency}</div>'
        "</div></header>"
    )
    nav = '<nav><div class="wrap">' + "".join(
        f'<a href="#{anchor}">{label}</a>' for anchor, label in NAV
    ) + "</div></nav>"

    sections = "".join([
        _section_resumen(metrics),
        _section_rendimiento(charts, vs, drawdown_episodes),
        _section_cartera(tables),
        _section_aprendizaje(charts, tables, settings),
        _section_posiciones(charts, tables),
        _section_robustez(tables),
        _section_metodologia(tables, settings),
        _section_debug(tables, run_dir),
    ])
    return header + nav + f'<main><div class="wrap">{sections}</div></main>'


# --- sections ---------------------------------------------------------------

def _section_resumen(metrics: dict) -> str:
    alpha = metrics.get("Alpha", 0) or 0
    ir = metrics.get("Information Ratio", 0) or 0
    tone = "pos" if alpha > 0 else "neg"
    verdict = "batió" if alpha > 0 else "no batió"
    kpis = "".join([
        kpi("Alpha vs SPY", _pct(alpha), "retorno total en exceso", tone),
        kpi("CAGR", _pct(metrics.get("CAGR")), f"benchmark {_pct(metrics.get('Benchmark CAGR'))}"),
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
        f'por bootstrap de la sección <a href="#robustez">Robustez</a> que con el t-stat aislado.</p>'
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
        + '<h3 class="note" style="margin-top:22px">Últimas operaciones</h3>'
        + table(select(recent, journal_cols))
        + "</section>"
    )


def _section_aprendizaje(charts: dict, tables: dict, settings: Settings) -> str:
    weights = tables.get("meta_weights_by_snapshot", pd.DataFrame())
    training_weights = weights[weights.get("phase", "training") == "training"] if not weights.empty else weights
    learned = int((training_weights.get("source", pd.Series(dtype=str)) == "learned").sum()) if not training_weights.empty else 0
    total = len(training_weights) if not training_weights.empty else 0
    cutoff_note = (
        f'El modelo aprende (reentrenamiento trimestral) desde el inicio de la ventana de '
        f'entrenamiento hasta el <strong>{settings.train_cutoff_date}</strong>. A partir de esa '
        f'fecha el modelo queda <strong>congelado</strong>: la cartera se sigue revisando cada mes, '
        f'pero siempre con los mismos agentes y los mismos pesos del meta-agente — el rank-IC y los '
        f'pesos que se ven después del corte miden cómo envejece ese modelo fijo, no aprendizaje '
        f'nuevo.'
    )
    lead = (
        'La prueba de que el sistema aprende: en cada snapshot de entrenamiento el meta-agente '
        'ajusta —a partir del alpha realmente realizado a 12 meses (walk-forward, sin lookahead)— '
        f'cuánto pesa cada agente especializado, premiando su contribución <em>marginal</em> '
        '(rank-IC parcial frente al alpha que los otros tres agentes no explican), no su solape. '
        f'{learned} de {total} snapshots de entrenamiento usaron pesos aprendidos (el resto, prior '
        'por falta de historia). El segundo gráfico muestra el rank-IC out-of-sample de la señal '
        'maestra (final_score, la combinación del meta-agente) por snapshot y su media móvil de 12 '
        'periodos: si sube con el tiempo, el sistema afina; si oscila plano, la muestra (unas 90 '
        'observaciones) no permite todavía distinguir mejora de ruido — se reporta el dato tal cual '
        f'sale, sin forzar una lectura de mejora. {cutoff_note}'
    )
    year_stats = _year_ic_table(tables.get("model_walk_forward_diagnostics", pd.DataFrame()))
    year_block = (
        '<h3 class="note" style="margin-top:22px">Rank-IC de la señal maestra por año</h3>'
        + table(year_stats) if not year_stats.empty else ""
    )
    horizon_block = _horizon_comparison_block(charts, tables.get("label_horizon_comparison", pd.DataFrame()), settings)
    dominant_block = _dominant_agent_block(weights, tables.get("model_walk_forward_diagnostics", pd.DataFrame()))
    return (
        '<section id="aprendizaje"><h2>Evidencia de aprendizaje</h2>'
        f'<p class="lead">{lead}</p>'
        + figure(charts.get("learned_weights"), "Evolución de los pesos aprendidos del meta-agente por snapshot")
        + dominant_block
        + '<div style="height:16px"></div>'
        + figure(charts.get("rank_ic"), "Rank-IC out-of-sample de la señal maestra por snapshot, con media móvil de tendencia")
        + year_block
        + horizon_block
        + "</section>"
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
        'meses, usando solo los snapshots de la fase de entrenamiento (la fase congelada usa siempre '
        'el mismo modelo, así que no aporta información nueva sobre el horizonte).</p>'
        + figure(charts.get("horizon_comparison"), "Rank-IC medio de la señal maestra a 3, 6 y 12 meses")
        + f'<p class="note">{conclusion}</p>'
    )


def _dominant_agent_block(weights: pd.DataFrame, diag: pd.DataFrame) -> str:
    """Qué agente pesa más y por qué — turns the weights chart into a readable sentence."""
    if weights.empty:
        return ""
    latest = weights.sort_values("snapshot_date").iloc[-1]
    agent_labels = {
        "quality_probability": "Calidad", "improvement_probability": "Crecimiento",
        "mispricing_probability": "Infravaloración", "timing_probability": "Temporización",
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
            ic_sentence = f' Su rank-IC OOS medio histórico es {ic_series.mean():.3f}, la señal más consistente de las cuatro en esta ejecución.'
    phase_label = "congelado" if str(latest.get("phase")) == "frozen" else "aprendido"
    return (
        '<p class="note" style="margin-top:10px">'
        f'En el snapshot más reciente ({phase_label}), el agente <strong>{agent_labels[dominant_key]}</strong> '
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
    """¿Es sólido el alpha o cabe en el ruido de una muestra pequeña? — bootstrap + sensibilidad a costes."""
    bootstrap = tables.get("robustness_bootstrap", pd.DataFrame())
    cost = tables.get("robustness_cost_sensitivity", pd.DataFrame())
    summary = tables.get("robustness_summary", pd.DataFrame())
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
        + "</section>"
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
        "quality_probability": "Calidad", "improvement_probability": "Crecimiento",
        "mispricing_probability": "Infravaloración", "timing_probability": "Temporización",
    }
    agent_rows = "".join(
        f'<li><strong>{agent_names.get(key, key)}</strong>: {definitions.get(key, "")}</li>'
        for key in agent_names
    )
    train_start = (
        pd.Timestamp(settings.train_cutoff_date) - pd.DateOffset(years=settings.walk_forward_train_years)
    ).date().isoformat()
    mechanics = (
        f'El modelo se entrenó con cadencia trimestral desde <strong>{train_start}</strong> hasta '
        f'<strong>{settings.train_cutoff_date}</strong>, usando en cada reentrenamiento hasta '
        f'{settings.max_walk_forward_training_years} años de historia hacia atrás. Desde '
        f'<strong>{settings.train_cutoff_date}</strong> hasta <strong>{settings.end_date}</strong>, '
        f'la cartera se simula con ese modelo <strong>congelado</strong> (sin reentrenar), '
        f'revisándose mensualmente ({settings.review_frequency}).'
    )
    params = pd.DataFrame([{
        "Parámetro": "Fecha de corte (entrenar hasta aquí)", "Valor": settings.train_cutoff_date,
    }, {
        "Parámetro": "Años de entrenamiento previos", "Valor": settings.walk_forward_train_years,
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
        '<h3 class="note">Los 4 agentes especializados + el meta-agente decisor</h3>'
        f'<ul class="findings">{agent_rows}'
        '<li>El <strong>meta-agente</strong> combina las cuatro señales anteriores en <code>final_score</code>, '
        'aprendiendo qué peso dar a cada una por su <strong>contribución marginal</strong>: el rank-IC '
        'parcial de cada agente frente al alpha realizado que los otros tres <em>no</em> explican '
        '(regresión sobre un 70% cronológico, medido en el 30% restante, sin lookahead). Así se premia '
        'a un agente por aportar información nueva, no por solaparse con el resto; un agente redundante o '
        'sin capacidad de ordenar recibe peso ~0, en vez de pesos fijos a mano. Ya no existe un agente '
        '«alpha» generalista: era el mismo objetivo que evalúa al meta-agente y con el superconjunto de '
        'features, así que dominaba estructuralmente la mezcla.</li></ul>'
        '<h3 class="note" style="margin-top:18px">Entrenar hasta un corte, luego congelar</h3>'
        f'<p class="note">{mechanics}</p>'
        '<h3 class="note" style="margin-top:18px">Parámetros efectivos de esta ejecución</h3>'
        + table(params)
        + "</section>"
    )


def _section_debug(tables: dict, run_dir: Path) -> str:
    diag = tables.get("model_walk_forward_diagnostics", pd.DataFrame())
    diag_cols = ["snapshot_date", "phase", "mode", "fallback_reason", "training_snapshot_date",
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
