"""Informes HTML autocontenidos. Dos productos:

  1. `build_run_report(run_dir)` — HTML por run con 7 hojas (Resumen, Rendimiento,
     Aprendizaje, Ranking, Cartera, Cobertura, Posiciones). Los graficos van embebidos como
     PNG base64. Las tablas grandes (ranking completo por agentes, historico de rank-IC,
     posiciones, ordenes) se sirven como CSVs al lado, cargados por `fetch` en JS minimo.
  2. `build_comparison_report(scenarios_root)` — HTML del barrido con 5 hojas y ranking
     por metrica compuesta de estabilidad (rango medio de beat_rate, median_alpha,
     worst_year_alpha, max_drawdown).

Ambos son autocontenidos: se abren en navegador local sin necesidad de servidor. Los CSVs
sueltos requieren un servidor HTTP local o un navegador que permita `fetch` de ficheros
locales (Chromium con `--allow-file-access-from-files`).
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path

import matplotlib
import pandas as pd

from environment import PROJECT_ROOT

matplotlib.use("Agg")   # sin display, para que corra en cabeceras sin GUI (CI, servidores)
import matplotlib.pyplot as plt   # noqa: E402

# Tema oscuro de las figuras, a juego con la paleta de la app (negros y grises, sin azul).
_DARK_BG = "#141416"
_DARK_INK = "#ececee"
_DARK_MUTED = "#9a9aa2"
_DARK_LINE = "#2a2a2e"
_SERIES_1 = "#d4d4d8"
_SERIES_2 = "#8a8a92"
_POS = "#4ade80"
_NEG = "#f87171"
plt.rcParams.update({
    "figure.facecolor": _DARK_BG, "axes.facecolor": _DARK_BG, "savefig.facecolor": _DARK_BG,
    "axes.edgecolor": _DARK_LINE, "axes.labelcolor": _DARK_INK, "text.color": _DARK_INK,
    "xtick.color": _DARK_MUTED, "ytick.color": _DARK_MUTED, "grid.color": _DARK_LINE,
    "axes.grid": True, "legend.facecolor": _DARK_BG, "legend.edgecolor": _DARK_LINE,
})

log = logging.getLogger(__name__)


# -------- HTML por run ------------------------------------------------------------------


def build_run_report(run_dir: Path) -> Path:
    """Genera `run_dir/report.html` con las 6 hojas + CSVs sueltos al lado."""
    run_dir = Path(run_dir)
    summary = _read_json(run_dir / "backtest_summary.json") or {}
    equity = _read_parquet(run_dir / "equity.parquet")
    annual = _read_parquet(run_dir / "annual_metrics.parquet")
    positions = _read_parquet(run_dir / "positions.parquet")
    orders = _read_parquet(run_dir / "orders.parquet")
    scores = _read_parquet(run_dir / "agent_scores.parquet")
    weights = _read_parquet(run_dir / "meta_weights.parquet")
    diagnostics = _read_parquet(run_dir / "rank_ic_diagnostics.parquet")

    _dump_csv(positions, run_dir / "positions_history.csv")
    _dump_csv(orders, run_dir / "orders_history.csv")
    _dump_csv(scores, run_dir / "ranking_by_snapshot.csv")
    _dump_csv(_ranking_by_agents(scores), run_dir / "ranking_by_agents.csv")
    _dump_csv(diagnostics, run_dir / "rank_ic_diagnostics.csv")
    _dump_csv(annual, run_dir / "annual_metrics.csv")

    resumen_body = _section_resumen(summary, annual, equity)
    rendimiento_body = _section_rendimiento(equity, annual)
    aprendizaje_body = _section_aprendizaje(diagnostics, weights)
    ranking_body = _section_ranking_agentes(scores)
    cartera_body = _section_cartera(positions, orders)
    cobertura_body = _section_cobertura(run_dir)
    posiciones_body = _section_posiciones(positions, orders, equity)

    html = _render_run_html({
        "resumen": resumen_body,
        "rendimiento": rendimiento_body,
        "aprendizaje": aprendizaje_body,
        "ranking": ranking_body,
        "cartera": cartera_body,
        "cobertura": cobertura_body,
        "posiciones": posiciones_body,
    })
    output_path = run_dir / "report.html"
    output_path.write_text(html, encoding="utf-8")
    log.info("Informe generado: %s", output_path)
    return output_path


def _section_resumen(summary: dict, annual: pd.DataFrame, equity: pd.DataFrame) -> str:
    if not summary:
        return "<p>Sin datos de backtest.</p>"

    mean_rank_ic = summary.get("mean_rank_ic", 0)
    ic_positive_pct = int(summary.get("rank_ic_positive_fraction", 0) * 100)
    beat_pct = int(summary.get("beat_rate", 0) * 100)
    dd_pct = summary.get("max_drawdown", 0) * 100
    cagr_pf = summary.get("cagr_portfolio", 0) * 100
    cagr_bench = summary.get("cagr_benchmark", 0) * 100
    cagr_diff = summary.get("cagr_difference", 0) * 100

    equity_chart = _plot_equity_curve(equity) if not equity.empty else ""
    alpha_bars = _plot_annual_alpha_bars(annual) if not annual.empty else ""
    annual_table = _annual_table_compact(annual)

    return f"""
        <div class="cards">
            <div class="card"><div class="metric">{mean_rank_ic:.4f}</div><div>rank-IC medio (meta_final)</div></div>
            <div class="card"><div class="metric">{ic_positive_pct}%</div><div>cohortes con rank-IC&gt;0</div></div>
            <div class="card"><div class="metric">{beat_pct}%</div><div>anios batiendo SPY</div></div>
            <div class="card"><div class="metric">-{dd_pct:.2f}%</div><div>drawdown maximo</div></div>
        </div>
        <p class="muted">Rendimiento economico (informativo, no es el criterio de seleccion):
        CAGR cartera <strong>{cagr_pf:.2f}%</strong> vs SPY <strong>{cagr_bench:.2f}%</strong>
        (diferencia {cagr_diff:+.2f}% al anio). El rank-IC mide si el sistema <em>aprende a
        ordenar</em>; el rendimiento es una consecuencia, no la evidencia.</p>
        <h3>Equity vs SPY</h3>
        {equity_chart}
        <h3>Alfa anual</h3>
        {alpha_bars}
        <h3>Anios (resumen)</h3>
        {annual_table}
    """


def _section_rendimiento(equity: pd.DataFrame, annual: pd.DataFrame) -> str:
    if equity.empty:
        return "<p>Sin datos de equity.</p>"
    drawdown = _plot_drawdown_series(equity)
    annual_full = _annual_table_full(annual)
    return f"""
        <h3>Drawdown continuo</h3>
        {drawdown}
        <h3>Tabla anual completa</h3>
        {annual_full}
    """


def _section_aprendizaje(diagnostics: pd.DataFrame, weights: pd.DataFrame) -> str:
    if diagnostics.empty:
        return "<p>Sin diagnosticos de rank-IC.</p>"

    # Resumen por agente: rank-IC medio, fraccion positiva, ICIR (media/desv) y nº de cohortes.
    per_agent_rows = []
    for agent, group in diagnostics.groupby("agent"):
        ic = group["rank_ic"].astype(float)
        icir = (ic.mean() / ic.std(ddof=1)) if len(ic) > 1 and ic.std(ddof=1) > 0 else 0.0
        cls = "pos" if ic.mean() > 0 else "neg"
        per_agent_rows.append(
            f"<tr><td>{agent}</td><td class='{cls}'>{ic.mean():.4f}</td>"
            f"<td>{(ic > 0).mean()*100:.0f}%</td><td>{icir:.2f}</td><td>{len(ic)}</td></tr>"
        )
    per_agent = (
        "<table><thead><tr><th>agente</th><th>rank-IC medio</th><th>% cohortes IC&gt;0</th>"
        "<th>ICIR</th><th># cohortes</th></tr></thead>"
        f"<tbody>{''.join(per_agent_rows)}</tbody></table>"
    )

    ic_chart = _plot_rank_ic_over_time(diagnostics)
    weight_chart = _plot_meta_weights_over_time(weights) if not weights.empty else ""

    return f"""
        <p>Evidencia de si los agentes ordenan activos <strong>fuera de muestra</strong>. El
        rank-IC (correlacion de Spearman entre el ranking predicho y el retorno realizado) se
        reporta por agente y por revision, separado del alfa. El <code>meta_final</code> es el
        score que opera la cartera: es sobre el que se mide el aprendizaje del sistema.</p>
        <h3>Resumen por agente (todas las cohortes)</h3>
        {per_agent}
        <h3>Rank-IC del meta_final a lo largo del tiempo</h3>
        {ic_chart}
        <h3>Evolucion de pesos del meta-agente</h3>
        {weight_chart}
        <h3>Historico completo de rank-IC (por agente y fecha)</h3>
        <p><em>Servido desde <code>rank_ic_diagnostics.csv</code> (carga bajo demanda).</em></p>
        <div id="rankic-detalle" data-src="rank_ic_diagnostics.csv"></div>
    """


def _ranking_by_agents(scores: pd.DataFrame) -> pd.DataFrame:
    """Ranking completo ticker x fecha con el score y rango de cada agente + el meta, ordenado."""
    if scores.empty:
        return pd.DataFrame()
    cols = [c for c in ["snapshot_date", "ticker", "quality_rank", "momentum_rank",
                        "value_rank", "meta_rank", "meta_score", "quality", "momentum", "value"]
            if c in scores.columns]
    frame = scores[cols].copy()
    return frame.sort_values(["snapshot_date", "meta_rank"], ascending=[True, False])


def _section_ranking_agentes(scores: pd.DataFrame) -> str:
    """Tabla del ranking de la ultima revision ordenado por el meta, con el rango de cada agente,
    para ver por que cada accion esta arriba. El historico completo va servido desde CSV."""
    if scores.empty or "meta_rank" not in scores.columns:
        return "<p>Sin scores de agentes.</p>"
    latest_date = scores["snapshot_date"].max()
    latest = (scores.loc[scores["snapshot_date"] == latest_date]
              .sort_values("meta_rank", ascending=False).head(40))

    def _pct(v):
        return f"{float(v)*100:.0f}" if pd.notna(v) else "-"

    rows = "".join(
        f"<tr><td>{i+1}</td><td>{r['ticker']}</td>"
        f"<td><strong>{_pct(r.get('meta_rank'))}</strong></td>"
        f"<td>{_pct(r.get('quality_rank'))}</td>"
        f"<td>{_pct(r.get('momentum_rank'))}</td>"
        f"<td>{_pct(r.get('value_rank'))}</td></tr>"
        for i, r in enumerate(latest.to_dict("records"))
    )
    return f"""
        <p>El ranking de cada revision, ordenado por el <code>meta_rank</code>, con el
        <strong>percentil de cada agente</strong> (calidad, momentum, valor) al lado: asi se ve
        <em>por que</em> cada accion esta arriba (que agente la empuja). Los percentiles van de
        0 (peor) a 100 (mejor) dentro de la seccion transversal de esa fecha.</p>
        <h3>Ranking de la ultima revision ({latest_date}) — top 40</h3>
        <table><thead><tr><th>#</th><th>ticker</th><th>meta</th>
        <th>calidad</th><th>momentum</th><th>valor</th></tr></thead>
        <tbody>{rows}</tbody></table>
        <h3>Ranking completo (todas las fechas, ordenado por meta)</h3>
        <p><em>Servido desde <code>ranking_by_agents.csv</code> — el ranking total ticker x fecha
        con score y rango de cada agente (carga bajo demanda con <code>fetch</code>).</em></p>
        <div id="ranking-agentes-detalle" data-src="ranking_by_agents.csv"></div>
    """


def _section_cartera(positions: pd.DataFrame, orders: pd.DataFrame) -> str:
    if positions.empty:
        return "<p>Sin posiciones.</p>"
    latest_date = positions["snapshot_date"].max()
    latest = positions.loc[positions["snapshot_date"] == latest_date].sort_values("weight", ascending=False)
    latest_rows = "".join(
        f"<tr><td>{r['ticker']}</td><td>{r['weight']*100:.1f}%</td>"
        f"<td>{r.get('current_percentile', 0):.0f}</td><td>{r.get('months_held', 0)}</td></tr>"
        for r in latest.to_dict("records")
    )

    turnover = "<p>Sin ordenes registradas.</p>"
    if not orders.empty:
        turnover_frame = orders.groupby("snapshot_date").size().reset_index(name="ordenes")
        rows = "".join(
            f"<tr><td>{r['snapshot_date']}</td><td>{r['ordenes']}</td></tr>"
            for r in turnover_frame.tail(20).to_dict("records")
        )
        turnover = f"<table><thead><tr><th>fecha</th><th># ordenes</th></tr></thead><tbody>{rows}</tbody></table>"

    return f"""
        <p>La cartera se decide en cada snapshot. Un tenente que cae del percentil 50 sale,
        aunque nadie le supere. Un candidato solo desplaza si su ventaja es &ge; 5 percentiles.</p>
        <h3>Composicion actual ({latest_date})</h3>
        <table><thead><tr><th>ticker</th><th>peso</th><th>percentil</th><th>meses en cartera</th></tr></thead>
        <tbody>{latest_rows}</tbody></table>
        <h3>Ordenes por revision (ultimas 20)</h3>
        {turnover}
        <h3>Historico completo</h3>
        <p><em>Datos servidos desde <code>positions_history.csv</code> y
        <code>orders_history.csv</code> (carga bajo demanda con
        <code>fetch</code>).</em></p>
        <div id="posiciones-detalle" data-src="positions_history.csv"></div>
        <div id="ordenes-detalle" data-src="orders_history.csv"></div>
    """


def _section_cobertura(run_dir: Path) -> str:
    coverage_json = None
    processed_dir = run_dir.parent.parent
    for candidate in (
        processed_dir / "universe_coverage.json",
        run_dir.parent.parent.parent / "raw" / "universe_coverage.json",
        run_dir.parent.parent.parent / "raw" / "dev" / "universe_coverage.json",
    ):
        if candidate.exists():
            coverage_json = json.loads(candidate.read_text(encoding="utf-8"))
            break

    body = "<p>No hay <code>universe_coverage.json</code> accesible desde el run_dir.</p>"
    if coverage_json and coverage_json.get("years"):
        rows = "".join(
            f"<tr><td>{y['year']}</td><td>{y['sp500_members']}</td>"
            f"<td>{y['panel_eligible_tickers']}</td><td>{y['coverage_pct']:.1f}%</td></tr>"
            for y in coverage_json["years"]
        )
        body = (
            "<table><thead><tr><th>anio</th><th>miembros</th>"
            "<th>observables</th><th>cobertura</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    return f"""
        <p>Sesgo de supervivencia por anio: quebrados y absorbidos no tienen datos en
        fuentes gratuitas. La cobertura se mide, no se estima.</p>
        <h3>Universo observable vs indice historico</h3>
        {body}
        <h3>Prohibiciones metodologicas activas</h3>
        <ul>
            <li><code>sector</code> excluido del panel: viene de <code>profiles.parquet</code>,
                que es un snapshot de hoy. Meterlo seria lookahead.</li>
            <li><code>payload.metric</code> prohibido: son ratios calculados hoy, no
                observables en fecha pasada.</li>
            <li>Metricas no-TTM (<code>net_margin</code>, <code>gross_margin</code>...) solo
                se leen de <code>quarterly</code>, nunca de <code>annual</code>, para no mezclar
                magnitudes en el corte transversal.</li>
        </ul>
    """


def _section_posiciones(positions: pd.DataFrame, orders: pd.DataFrame, equity: pd.DataFrame) -> str:
    if positions.empty:
        return "<p>Sin posiciones para analizar.</p>"

    ticker_periods = _compute_holding_returns(positions, orders)
    rows = "".join(
        f"<tr><td>{p['ticker']}</td><td>{p['entry_date']}</td>"
        f"<td>{p.get('exit_date', 'aun dentro')}</td><td>{p['months_held']}</td>"
        f"<td>{p['return_pct']:.2f}%</td></tr>"
        for p in ticker_periods[:50]
    )
    return f"""
        <p>Cuanto rento cada ticker mientras estuvo en cartera. Ayuda a ver si el sistema
        mantiene ganadores o corta demasiado pronto.</p>
        <table><thead><tr><th>ticker</th><th>entrada</th><th>salida</th>
        <th>meses</th><th>retorno</th></tr></thead>
        <tbody>{rows}</tbody></table>
    """


def _compute_holding_returns(positions: pd.DataFrame, orders: pd.DataFrame) -> list[dict]:
    """Une entradas y salidas por ticker para calcular retorno del holding."""
    if orders.empty:
        return []
    orders_by_ticker = orders.sort_values("snapshot_date").groupby("ticker")
    result: list[dict] = []
    for ticker, group in orders_by_ticker:
        buys = group.loc[group["side"] == "buy"]
        sells = group.loc[group["side"] == "sell"]
        if buys.empty:
            continue
        entry = buys.iloc[0]
        exit_row = sells.iloc[-1] if not sells.empty else None
        entry_price = float(entry.get("price") or 0)
        exit_price = float(exit_row.get("price") or 0) if exit_row is not None else 0.0
        if entry_price > 0 and exit_price > 0:
            ret = exit_price / entry_price - 1
        else:
            ret = 0.0
        months = 0
        if exit_row is not None:
            months = (pd.Timestamp(exit_row["snapshot_date"]) -
                      pd.Timestamp(entry["snapshot_date"])).days // 30
        result.append({
            "ticker": ticker,
            "entry_date": entry["snapshot_date"],
            "exit_date": exit_row["snapshot_date"] if exit_row is not None else "aun dentro",
            "months_held": months,
            "return_pct": ret * 100,
        })
    result.sort(key=lambda item: item["return_pct"], reverse=True)
    return result


# -------- HTML del barrido --------------------------------------------------------------


def build_comparison_report(scenarios_root: Path) -> Path:
    """Rankea todos los escenarios por metrica de aprendizaje y produce el HTML global.

    Sin eras: un unico ranking global sobre todos los anios disponibles. La seleccion usa
    senales de aprendizaje (rank-IC) y consistencia, NUNCA alfa. Ver `select_winner`.
    """
    from module.runs.experiments import SELECTION_DIMENSIONS, select_winner

    scenarios_root = Path(scenarios_root)
    scenario_dirs = sorted(
        path for path in scenarios_root.iterdir()
        if path.is_dir() and (path / "scenario_config.json").exists()
    )
    if not scenario_dirs:
        raise RuntimeError(f"No hay escenarios en {scenarios_root}.")

    summary_rows = _collect_scenario_summaries(scenario_dirs)
    if not summary_rows:
        raise RuntimeError("No hay summaries de backtest en los escenarios.")

    winner_name, summary = select_winner(pd.DataFrame(summary_rows))
    summary.to_parquet(scenarios_root / "scenarios_summary.parquet", index=False)
    summary.to_csv(scenarios_root / "scenarios_summary.csv", index=False)

    winner_row = summary.iloc[0]
    selection = {
        "winner": winner_name,
        "composite_rank_mean": float(winner_row["composite_rank_mean"]),
        "selection_dimensions": [name for name, _ in SELECTION_DIMENSIONS],
        "ranks": {
            name: int(winner_row[f"rank_{name}"]) for name, _ in SELECTION_DIMENSIONS
        },
        "winner_learning": {
            "mean_rank_ic": float(winner_row.get("mean_rank_ic", 0)),
            "rank_ic_positive_fraction": float(winner_row.get("rank_ic_positive_fraction", 0)),
            "beat_rate": float(winner_row.get("beat_rate", 0)),
            "max_drawdown": float(winner_row.get("max_drawdown", 0)),
        },
        "winner_economics_reported_only": {
            "cagr_portfolio": float(winner_row.get("cagr_portfolio", 0)),
            "cagr_benchmark": float(winner_row.get("cagr_benchmark", 0)),
            "geometric_excess_return": float(winner_row.get("geometric_excess_return", 0)),
        },
        "top_3": summary.head(3)["scenario"].tolist(),
    }
    (scenarios_root / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )

    resumen = _comparison_summary_section(summary, winner_name)
    anual = _comparison_annual_heatmap_section(scenario_dirs)
    sensibilidad = _comparison_sensitivity_section(summary, scenario_dirs)
    seleccion = _comparison_selection_section(selection, summary)
    todos = _comparison_all_runs_section(summary, scenario_dirs)

    html = _render_comparison_html({
        "resumen": resumen,
        "anual": anual,
        "sensibilidad": sensibilidad,
        "seleccion": seleccion,
        "todos": todos,
    })
    output_path = scenarios_root / "comparison.html"
    output_path.write_text(html, encoding="utf-8")
    log.info("Informe de barrido generado: %s", output_path)
    return output_path


def _collect_scenario_summaries(scenario_dirs: list[Path]) -> list[dict]:
    """Una fila por escenario con sus senales de aprendizaje, consistencia y alfa (reportada).

    Todo se calcula sobre TODOS los anios disponibles: sin separacion en eras.
    """
    rows: list[dict] = []
    for scenario_dir in scenario_dirs:
        run_dir = _pick_run_dir(scenario_dir)
        if run_dir is None:
            continue
        summary = _read_json(run_dir / "backtest_summary.json")
        annual = _read_parquet(run_dir / "annual_metrics.parquet")
        if summary is None or annual.empty:
            continue
        rows.append({
            "scenario": scenario_dir.name,
            # senales de seleccion (aprendizaje + consistencia + riesgo)
            "mean_rank_ic": float(summary.get("mean_rank_ic", 0)),
            "rank_ic_positive_fraction": float(summary.get("rank_ic_positive_fraction", 0)),
            "rank_ic_std": float(summary.get("rank_ic_std", 0)),
            "beat_rate": float((annual["alpha"] > 0).mean()),
            "max_drawdown": float(summary.get("max_drawdown", 0)),
            # rendimiento economico: SOLO reportado, no selecciona
            "cagr_portfolio": float(summary.get("cagr_portfolio", 0)),
            "cagr_benchmark": float(summary.get("cagr_benchmark", 0)),
            "geometric_excess_return": float(summary.get("geometric_excess_return", 0)),
            "mean_annual_alpha": float(annual["alpha"].mean()),
            "information_ratio": float(summary.get("information_ratio", 0)),
        })
    return rows


def _pick_run_dir(scenario_dir: Path) -> Path | None:
    """Busca `agents/<run_id>` primero directamente y despues bajo `processed/`.

    Los escenarios que corren dataset/features propios acaban con la estructura
    `<scenario>/processed/agents/<run_id>`. Los que reutilizan solo tienen el enlace
    en `<scenario>/agents/<run_id>`.
    """
    for candidate_root in (scenario_dir / "agents", scenario_dir / "processed" / "agents"):
        if not candidate_root.exists():
            continue
        candidates = sorted(path for path in candidate_root.iterdir() if path.is_dir())
        if candidates:
            return candidates[-1]
    return None


def _comparison_summary_section(summary: pd.DataFrame, winner: str) -> str:
    ordered = summary.sort_values("composite_rank_mean")
    rows = "".join(
        f"<tr class='{'winner' if row['scenario'] == winner else ''}'>"
        f"<td>{index + 1}</td><td>{row['scenario']}</td>"
        f"<td>{row['composite_rank_mean']:.2f}</td>"
        f"<td>{row['mean_rank_ic']:.4f}</td>"
        f"<td>{row['rank_ic_positive_fraction']*100:.0f}%</td>"
        f"<td>{row['beat_rate']*100:.0f}%</td>"
        f"<td>-{row['max_drawdown']*100:.2f}%</td>"
        f"<td class='muted'>{row.get('geometric_excess_return',0)*100:.2f}%</td></tr>"
        for index, row in enumerate(ordered.to_dict("records"))
    )
    return f"""
        <p>Ranking por metrica de <strong>aprendizaje y estabilidad</strong> (rango medio de
        rank-IC medio, fraccion de cohortes con rank-IC positivo, beat rate y drawdown).
        Menor rango medio es mejor. <strong>El alfa NO participa en la seleccion</strong>:
        se muestra en gris a la derecha solo como consecuencia, no como criterio. El ganador
        se destaca en verde.</p>
        <table><thead><tr><th>#</th><th>escenario</th><th>rango medio</th>
        <th>rank-IC medio</th><th>% cohortes IC&gt;0</th><th>beat rate</th>
        <th>drawdown max</th><th class='muted'>alfa anualiz. (info)</th></tr></thead>
        <tbody>{rows}</tbody></table>
    """


def _comparison_annual_heatmap_section(scenario_dirs: list[Path]) -> str:
    rows = []
    for scenario_dir in scenario_dirs:
        run_dir = _pick_run_dir(scenario_dir)
        if run_dir is None:
            continue
        annual = _read_parquet(run_dir / "annual_metrics.parquet")
        for row in annual.to_dict("records"):
            rows.append({"scenario": scenario_dir.name, "year": row["year"], "alpha": row["alpha"]})
    if not rows:
        return "<p>Sin datos anuales.</p>"
    frame = pd.DataFrame(rows).pivot(index="scenario", columns="year", values="alpha").fillna(0)
    chart = _plot_heatmap(frame, title="Alfa anual por escenario (todos los anios)", cmap="RdYlGn")
    return f"""
        <p>Cada fila es un escenario, cada columna un anio (todos los disponibles, sin
        separacion en eras). Verde &rarr; alfa positivo, rojo &rarr; alfa negativo. Es la
        vista honesta de la consistencia: un escenario con muchas columnas rojas y una muy
        verde gano por un anio excepcional, no por aprender.</p>
        {chart}
    """


def _comparison_sensitivity_section(summary: pd.DataFrame, scenario_dirs: list[Path]) -> str:
    """Muestra la senal de aprendizaje de cada escenario junto a sus overrides."""
    rows = []
    for scenario_dir in scenario_dirs:
        config = _read_json(scenario_dir / "scenario_config.json") or {}
        overrides = config.get("overrides", {})
        match = summary.loc[summary["scenario"] == scenario_dir.name]
        if match.empty:
            continue
        mean_ic = float(match.iloc[0]["mean_rank_ic"])
        overrides_text = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "(baseline)"
        rows.append((scenario_dir.name, overrides_text, mean_ic))
    rows.sort(key=lambda item: item[2], reverse=True)
    body = "".join(
        f"<tr><td>{name}</td><td><code>{overrides}</code></td><td>{ic:.4f}</td></tr>"
        for name, overrides, ic in rows
    )
    return f"""
        <p>Que overrides mueven el aprendizaje (rank-IC medio). Se elige el valor mas estable
        en aprendizaje, no el de mayor alfa puntual.</p>
        <table><thead><tr><th>escenario</th><th>overrides</th><th>rank-IC medio</th></tr></thead>
        <tbody>{body}</tbody></table>
    """


def _comparison_selection_section(selection: dict, summary: pd.DataFrame) -> str:
    top3 = summary.head(3).to_dict("records")
    top_rows = "".join(
        f"<tr><td>{row['scenario']}</td>"
        f"<td>{row['rank_mean_rank_ic']}</td>"
        f"<td>{row['rank_rank_ic_positive_fraction']}</td>"
        f"<td>{row['rank_beat_rate']}</td>"
        f"<td>{row['rank_max_drawdown']}</td>"
        f"<td><strong>{row['composite_rank_mean']:.2f}</strong></td></tr>"
        for row in top3
    )
    learning = selection.get("winner_learning", {})
    alpha_info = selection.get("winner_alpha_reported_only", {})
    learning_body = "".join(f"<li>{key}: {value:.4f}</li>" for key, value in learning.items())
    alpha_body = "".join(f"<li>{key}: {value*100:.2f}%</li>" for key, value in alpha_info.items())
    return f"""
        <h3>Ganador: <code>{selection['winner']}</code></h3>
        <p>Rango medio: <strong>{selection['composite_rank_mean']:.2f}</strong>. Se rankea por
        aprendizaje (rank-IC), estabilidad del aprendizaje, beat rate y drawdown. <strong>El
        alfa no interviene.</strong></p>
        <table><thead><tr><th>escenario</th><th>rk rank-IC</th><th>rk estab. IC</th>
        <th>rk beat</th><th>rk drawdown</th><th>rango medio</th></tr></thead>
        <tbody>{top_rows}</tbody></table>
        <h3>Senales de aprendizaje del ganador</h3>
        <ul>{learning_body}</ul>
        <h3>Alfa del ganador (solo informativo, no selecciono)</h3>
        <ul>{alpha_body}</ul>
        <p><em>Si el rank-IC medio del ganador es cercano a cero, la conclusion honesta es que
        el sistema NO aprende a ordenar activos de forma estable, y cualquier alfa observado no
        es atribuible al aprendizaje. Ese es un resultado valido del TFM.</em></p>
    """


def _comparison_all_runs_section(summary: pd.DataFrame, scenario_dirs: list[Path]) -> str:
    rows = "".join(
        f"<tr><td><a href='{row['scenario']}/agents/'>{row['scenario']}</a></td>"
        f"<td>{row['mean_rank_ic']:.4f}</td>"
        f"<td>{row['rank_ic_positive_fraction']*100:.0f}%</td>"
        f"<td>{row['beat_rate']*100:.0f}%</td>"
        f"<td class='muted'>{row.get('geometric_excess_return',0)*100:.2f}%</td></tr>"
        for row in summary.sort_values("composite_rank_mean").to_dict("records")
    )
    return f"""
        <p>Tabla completa con enlace al HTML de cada escenario, para poder auditar
        cualquiera de los runs. El alfa (en gris) es informativo.</p>
        <table><thead><tr><th>escenario</th><th>rank-IC medio</th><th>% cohortes IC&gt;0</th>
        <th>beat rate</th><th class='muted'>alfa anualiz. (info)</th></tr></thead>
        <tbody>{rows}</tbody></table>
    """


# -------- Helpers --------------------------------------------------------------


def _plot_equity_curve(equity: pd.DataFrame) -> str:
    figure, axes = plt.subplots(figsize=(8, 3))
    axes.plot(pd.to_datetime(equity["snapshot_date"]), equity["portfolio_value"], label="cartera", color=_SERIES_1)
    axes.plot(pd.to_datetime(equity["snapshot_date"]), equity["benchmark_value"], label="SPY", color=_SERIES_2, linestyle="--")
    axes.set_ylabel("valor (base 100)")
    axes.legend()
    axes.grid(alpha=0.3)
    return _figure_to_img(figure)


def _plot_annual_alpha_bars(annual: pd.DataFrame) -> str:
    figure, axes = plt.subplots(figsize=(8, 3))
    colors = [_POS if alpha > 0 else _NEG for alpha in annual["alpha"]]
    axes.bar(annual["year"].astype(str), annual["alpha"] * 100, color=colors)
    axes.axhline(0, color=_DARK_MUTED, linewidth=0.5)
    axes.set_ylabel("alfa (%)")
    axes.grid(alpha=0.3, axis="y")
    return _figure_to_img(figure)


def _plot_drawdown_series(equity: pd.DataFrame) -> str:
    import numpy as np
    equity_series = equity["portfolio_value"].to_numpy()
    drawdown = 1 - equity_series / np.maximum.accumulate(equity_series)
    figure, axes = plt.subplots(figsize=(8, 3))
    axes.fill_between(pd.to_datetime(equity["snapshot_date"]), -drawdown * 100, 0,
                       color=_NEG, alpha=0.4)
    axes.set_ylabel("drawdown (%)")
    axes.grid(alpha=0.3)
    return _figure_to_img(figure)


def _plot_rank_ic_over_time(diagnostics: pd.DataFrame) -> str:
    meta = diagnostics.loc[diagnostics["agent"] == "meta_final"].copy()
    if meta.empty:
        return "<p>Sin cohortes de meta_final.</p>"
    meta["prediction_date"] = pd.to_datetime(meta["prediction_date"])
    meta = meta.sort_values("prediction_date")
    figure, axes = plt.subplots(figsize=(8, 3))
    colors = [_POS if v > 0 else _NEG for v in meta["rank_ic"]]
    axes.bar(meta["prediction_date"], meta["rank_ic"], width=60, color=colors, alpha=0.5)
    rolling = meta["rank_ic"].rolling(4, min_periods=1).mean()
    axes.plot(meta["prediction_date"], rolling, color=_SERIES_1, linewidth=1.5, label="media movil (4)")
    axes.axhline(0, color=_DARK_MUTED, linewidth=0.6)
    axes.axhline(float(meta["rank_ic"].mean()), color=_SERIES_2, linestyle="--", linewidth=1,
                 label=f"media global {meta['rank_ic'].mean():.4f}")
    axes.set_ylabel("rank-IC")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3, axis="y")
    return _figure_to_img(figure)


def _plot_meta_weights_over_time(weights: pd.DataFrame) -> str:
    figure, axes = plt.subplots(figsize=(8, 3))
    for agent, group in weights.groupby("agent"):
        axes.plot(pd.to_datetime(group["snapshot_date"]), group["weight"], label=agent)
    axes.set_ylabel("peso")
    axes.legend()
    axes.grid(alpha=0.3)
    return _figure_to_img(figure)


def _plot_heatmap(frame: pd.DataFrame, title: str, cmap: str) -> str:
    figure, axes = plt.subplots(figsize=(10, max(3, 0.3 * len(frame))))
    image = axes.imshow(frame.values, aspect="auto", cmap=cmap, vmin=-0.3, vmax=0.3)
    axes.set_xticks(range(len(frame.columns)))
    axes.set_xticklabels(frame.columns, rotation=45)
    axes.set_yticks(range(len(frame.index)))
    axes.set_yticklabels(frame.index)
    axes.set_title(title)
    figure.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    return _figure_to_img(figure)


def _figure_to_img(figure) -> str:
    buffer = io.BytesIO()
    figure.tight_layout()
    figure.savefig(buffer, format="png", dpi=100)
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" style="max-width:100%">'


def _annual_table_compact(annual: pd.DataFrame) -> str:
    if annual.empty:
        return "<p>Sin datos anuales.</p>"
    rows = "".join(
        f"<tr><td>{row['year']}</td><td>{row['alpha']*100:.2f}%</td>"
        f"<td>{'si' if row['beats_benchmark'] else 'no'}</td></tr>"
        for row in annual.head(10).to_dict("records")
    )
    return f"<table><thead><tr><th>anio</th><th>alfa</th><th>bate SPY</th></tr></thead><tbody>{rows}</tbody></table>"


def _annual_table_full(annual: pd.DataFrame) -> str:
    if annual.empty:
        return "<p>Sin datos anuales.</p>"
    rows = "".join(
        f"<tr><td>{row['year']}</td><td>{row['portfolio_return']*100:.2f}%</td>"
        f"<td>{row['benchmark_return']*100:.2f}%</td>"
        f"<td>{row['alpha']*100:.2f}%</td><td>-{row['max_drawdown_year']*100:.2f}%</td>"
        f"<td>{row['information_ratio_year']:.2f}</td></tr>"
        for row in annual.to_dict("records")
    )
    return (
        "<table><thead><tr><th>anio</th><th>ret cartera</th><th>ret SPY</th>"
        "<th>alfa</th><th>drawdown</th><th>IR</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        path.write_text("", encoding="utf-8")
        return
    frame.to_csv(path, index=False)


# -------- Template HTML --------------------------------------------------------------


_TABS_JS = """
<script>
document.querySelectorAll('.tab-link').forEach(link => {
  link.addEventListener('click', event => {
    event.preventDefault();
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-link').forEach(other => other.classList.remove('active'));
    document.querySelector(link.dataset.target).classList.add('active');
    link.classList.add('active');
  });
});
const MAX_ROWS = 500;   // tope de filas renderizadas por tabla (el CSV completo esta al lado)
document.querySelectorAll('[data-src]').forEach(container => {
  fetch(container.dataset.src)
    .then(response => { if (!response.ok) throw new Error('http ' + response.status); return response.text(); })
    .then(csv => {
      const rows = csv.split('\\n').filter(line => line);
      if (!rows.length) { container.innerHTML = '<p><em>Sin datos.</em></p>'; return; }
      const total = rows.length - 1;
      const shown = Math.min(rows.length, MAX_ROWS + 1);
      const table = document.createElement('table');
      table.innerHTML = rows.slice(0, shown).map((row, index) => {
        const cells = row.split(',');
        const tag = index === 0 ? 'th' : 'td';
        return '<tr>' + cells.map(cell => `<${tag}>${cell}</${tag}>`).join('') + '</tr>';
      }).join('');
      if (total > MAX_ROWS) {
        const note = document.createElement('p');
        note.className = 'muted';
        note.textContent = 'Mostrando ' + MAX_ROWS + ' de ' + total + ' filas. El fichero completo es ' +
          container.dataset.src + '.';
        container.appendChild(note);
      }
      container.appendChild(table);
    })
    .catch(error => {
      container.innerHTML = '<p><em>No se pudo cargar ' + container.dataset.src +
        ' (requiere servidor local: <code>python servir_html.py</code>).</em></p>';
    });
});
</script>
"""

# El estilo oscuro vive en app/report/report.css (misma paleta que la app) y se
# inyecta inline para que el informe siga siendo autocontenido (se abre sin servidor).
_REPORT_CSS_PATH = PROJECT_ROOT / "app" / "report" / "report.css"


def _load_style() -> str:
    css = _REPORT_CSS_PATH.read_text(encoding="utf-8")
    return f"<style>\n{css}\n</style>"


_STYLE = _load_style()


def _render_run_html(sections: dict[str, str]) -> str:
    tab_links = "".join(
        f'<a href="#" class="tab-link{" active" if index == 0 else ""}" '
        f'data-target="#tab-{key}">{key.capitalize()}</a>'
        for index, key in enumerate(sections)
    )
    tab_bodies = "".join(
        f'<section id="tab-{key}" class="tab{" active" if index == 0 else ""}">{body}</section>'
        for index, (key, body) in enumerate(sections.items())
    )
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><title>Informe del run</title>{_STYLE}</head>
<body>
<h1>Informe del run</h1>
<nav class="tabs">{tab_links}</nav>
{tab_bodies}
{_TABS_JS}
</body></html>"""


def _render_comparison_html(sections: dict[str, str]) -> str:
    tab_links = "".join(
        f'<a href="#" class="tab-link{" active" if index == 0 else ""}" '
        f'data-target="#tab-{key}">{key.capitalize()}</a>'
        for index, key in enumerate(sections)
    )
    tab_bodies = "".join(
        f'<section id="tab-{key}" class="tab{" active" if index == 0 else ""}">{body}</section>'
        for index, (key, body) in enumerate(sections.items())
    )
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><title>Comparacion de escenarios</title>{_STYLE}</head>
<body>
<h1>Comparacion de escenarios</h1>
<nav class="tabs">{tab_links}</nav>
{tab_bodies}
{_TABS_JS}
</body></html>"""


# -------- Entry point ------------------------------------------------------------


def build_report_from_settings(settings) -> None:
    """Handler para `RUN_MODE=report`: localiza el ultimo run_dir de agentes y lo procesa."""
    processed = settings.processed_output_dir
    agents_root = processed / "agents"
    if not agents_root.exists():
        raise RuntimeError(f"No hay run_dir de agentes en {agents_root}.")
    run_dirs = sorted(path for path in agents_root.iterdir() if path.is_dir())
    if not run_dirs:
        raise RuntimeError(f"Sin ningun run en {agents_root}.")
    build_run_report(run_dirs[-1])
