"""Informes HTML autocontenidos. Dos productos:

  1. `build_run_report(run_dir)` — HTML por run con 6 hojas (Resumen, Rendimiento,
     Aprendizaje, Cartera, Cobertura, Posiciones). Los graficos van embebidos como PNG
     base64. Las tablas grandes se sirven como CSVs al lado, cargados por `fetch` en JS
     minimo.
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

matplotlib.use("Agg")   # sin display, para que corra en cabeceras sin GUI (CI, servidores)
import matplotlib.pyplot as plt   # noqa: E402

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

    resumen_body = _section_resumen(summary, annual, equity)
    rendimiento_body = _section_rendimiento(equity, annual)
    aprendizaje_body = _section_aprendizaje(diagnostics, weights)
    cartera_body = _section_cartera(positions, orders)
    cobertura_body = _section_cobertura(run_dir)
    posiciones_body = _section_posiciones(positions, orders, equity)

    html = _render_run_html({
        "resumen": resumen_body,
        "rendimiento": rendimiento_body,
        "aprendizaje": aprendizaje_body,
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

    beat_pct = int(summary.get("beat_rate", 0) * 100)
    alfa_pct = summary.get("total_alpha", 0) * 100
    dd_pct = summary.get("max_drawdown", 0) * 100
    ir = summary.get("information_ratio", 0)

    equity_chart = _plot_equity_curve(equity) if not equity.empty else ""
    alpha_bars = _plot_annual_alpha_bars(annual) if not annual.empty else ""
    annual_table = _annual_table_compact(annual)

    return f"""
        <div class="cards">
            <div class="card"><div class="metric">{alfa_pct:.2f}%</div><div>alfa total neta</div></div>
            <div class="card"><div class="metric">{beat_pct}%</div><div>anios batiendo SPY</div></div>
            <div class="card"><div class="metric">{ir:.2f}</div><div>information ratio</div></div>
            <div class="card"><div class="metric">-{dd_pct:.2f}%</div><div>drawdown maximo</div></div>
        </div>
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
    diag = "<p>Sin diagnosticos de rank-IC.</p>"
    if not diagnostics.empty:
        rows = [f"<tr><td>{r['agent']}</td><td>{r['prediction_date']}</td>"
                f"<td>{r['rank_ic']:.3f}</td><td>{r['observations']}</td></tr>"
                for r in diagnostics.head(30).to_dict("records")]
        diag = (
            "<table><thead><tr><th>agente</th><th>fecha</th>"
            "<th>rank-IC</th><th>observaciones</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    weight_chart = _plot_meta_weights_over_time(weights) if not weights.empty else ""

    return f"""
        <p>Evidencia de que los agentes ordenan activos fuera de muestra.
        El rank-IC se reporta por agente y por era, separado del alfa.</p>
        <h3>Rank-IC por agente y por revision</h3>
        {diag}
        <h3>Evolucion de pesos del meta-agente</h3>
        {weight_chart}
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


def build_comparison_report(
    scenarios_root: Path,
    dev_era: tuple[int, int] = (1990, 2015),
    confirmation_era: tuple[int, int] = (2016, 2100),
) -> Path:
    """Rankea todos los escenarios por metrica compuesta y produce el HTML global."""
    scenarios_root = Path(scenarios_root)
    scenario_dirs = sorted(
        path for path in scenarios_root.iterdir()
        if path.is_dir() and (path / "scenario_config.json").exists()
    )
    if not scenario_dirs:
        raise RuntimeError(f"No hay escenarios en {scenarios_root}.")

    summary_rows = _collect_scenario_summaries(scenario_dirs, dev_era)
    if not summary_rows:
        raise RuntimeError("No hay summaries de backtest en los escenarios.")

    summary = pd.DataFrame(summary_rows)
    summary = _rank_scenarios(summary)
    summary.to_parquet(scenarios_root / "scenarios_summary.parquet", index=False)
    summary.to_csv(scenarios_root / "scenarios_summary.csv", index=False)

    winner_row = summary.sort_values("composite_rank_mean").iloc[0]
    winner_name = str(winner_row["scenario"])

    confirmation_metrics = _confirmation_era_metrics(scenarios_root, winner_name, confirmation_era)
    selection = {
        "winner": winner_name,
        "composite_rank_mean": float(winner_row["composite_rank_mean"]),
        "rank_beat_rate": int(winner_row["rank_beat_rate"]),
        "rank_median_alpha": int(winner_row["rank_median_alpha"]),
        "rank_worst_year_alpha": int(winner_row["rank_worst_year_alpha"]),
        "rank_max_drawdown": int(winner_row["rank_max_drawdown"]),
        "dev_era": list(dev_era),
        "confirmation_era": list(confirmation_era),
        "confirmation_metrics": confirmation_metrics,
        "top_3": summary.sort_values("composite_rank_mean").head(3)["scenario"].tolist(),
    }
    (scenarios_root / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )

    resumen = _comparison_summary_section(summary, winner_name)
    anual = _comparison_annual_heatmap_section(scenario_dirs, dev_era)
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


def _collect_scenario_summaries(scenario_dirs: list[Path], dev_era: tuple[int, int]) -> list[dict]:
    rows: list[dict] = []
    for scenario_dir in scenario_dirs:
        run_dir = _pick_run_dir(scenario_dir)
        if run_dir is None:
            continue
        summary = _read_json(run_dir / "backtest_summary.json")
        annual = _read_parquet(run_dir / "annual_metrics.parquet")
        if summary is None or annual.empty:
            continue
        dev_annual = annual.loc[
            annual["year"].between(dev_era[0], dev_era[1])
        ]
        if dev_annual.empty:
            dev_annual = annual
        rows.append({
            "scenario": scenario_dir.name,
            "beat_rate": float((dev_annual["alpha"] > 0).mean()),
            "median_alpha": float(dev_annual["alpha"].median()),
            "worst_year_alpha": float(dev_annual["alpha"].min()),
            "max_drawdown": float(summary.get("max_drawdown", 0)),
            "total_alpha": float(summary.get("total_alpha", 0)),
            "information_ratio": float(summary.get("information_ratio", 0)),
        })
    return rows


def _rank_scenarios(summary: pd.DataFrame) -> pd.DataFrame:
    """Rango medio de las cuatro dimensiones. Menor es mejor.

    Para las tres dimensiones donde "mas grande es mejor" (beat_rate, median_alpha,
    worst_year_alpha), rankeo descendente. Para max_drawdown, ascendente.
    """
    ranked = summary.copy()
    ranked["rank_beat_rate"] = ranked["beat_rate"].rank(ascending=False, method="min").astype(int)
    ranked["rank_median_alpha"] = ranked["median_alpha"].rank(ascending=False, method="min").astype(int)
    ranked["rank_worst_year_alpha"] = ranked["worst_year_alpha"].rank(ascending=False, method="min").astype(int)
    ranked["rank_max_drawdown"] = ranked["max_drawdown"].rank(ascending=True, method="min").astype(int)
    ranked["composite_rank_mean"] = ranked[
        ["rank_beat_rate", "rank_median_alpha", "rank_worst_year_alpha", "rank_max_drawdown"]
    ].mean(axis=1)
    return ranked


def _confirmation_era_metrics(
    scenarios_root: Path, winner: str, confirmation_era: tuple[int, int]
) -> dict:
    scenario_dir = scenarios_root / winner
    run_dir = _pick_run_dir(scenario_dir)
    if run_dir is None:
        return {"note": "sin run_dir accesible"}
    annual = _read_parquet(run_dir / "annual_metrics.parquet")
    reserved = annual.loc[annual["year"].between(confirmation_era[0], confirmation_era[1])]
    if reserved.empty:
        return {"note": "sin anios en la era reservada"}
    return {
        "years_covered": int(len(reserved)),
        "beat_rate": float((reserved["alpha"] > 0).mean()),
        "median_alpha": float(reserved["alpha"].median()),
        "worst_year_alpha": float(reserved["alpha"].min()),
    }


def _pick_run_dir(scenario_dir: Path) -> Path | None:
    agents_root = scenario_dir / "agents"
    if not agents_root.exists():
        return None
    candidates = sorted(path for path in agents_root.iterdir() if path.is_dir())
    return candidates[-1] if candidates else None


def _comparison_summary_section(summary: pd.DataFrame, winner: str) -> str:
    ordered = summary.sort_values("composite_rank_mean")
    rows = "".join(
        f"<tr class='{'winner' if row['scenario'] == winner else ''}'>"
        f"<td>{index + 1}</td><td>{row['scenario']}</td>"
        f"<td>{row['composite_rank_mean']:.2f}</td>"
        f"<td>{row['beat_rate']*100:.0f}%</td>"
        f"<td>{row['median_alpha']*100:.2f}%</td>"
        f"<td>{row['worst_year_alpha']*100:.2f}%</td>"
        f"<td>-{row['max_drawdown']*100:.2f}%</td></tr>"
        for index, row in enumerate(ordered.to_dict("records"))
    )
    return f"""
        <p>Ranking por metrica compuesta de estabilidad (rango medio de las cuatro
        dimensiones). Menor rango medio es mejor. El ganador se destaca en verde.</p>
        <table><thead><tr><th>#</th><th>escenario</th><th>rango medio</th>
        <th>beat rate</th><th>alfa mediana</th><th>peor anio</th><th>drawdown max</th></tr></thead>
        <tbody>{rows}</tbody></table>
    """


def _comparison_annual_heatmap_section(scenario_dirs: list[Path], dev_era: tuple[int, int]) -> str:
    rows = []
    for scenario_dir in scenario_dirs:
        run_dir = _pick_run_dir(scenario_dir)
        if run_dir is None:
            continue
        annual = _read_parquet(run_dir / "annual_metrics.parquet")
        for row in annual.to_dict("records"):
            if not (dev_era[0] <= row["year"] <= dev_era[1]):
                continue
            rows.append({"scenario": scenario_dir.name, "year": row["year"], "alpha": row["alpha"]})
    if not rows:
        return "<p>Sin datos anuales.</p>"
    frame = pd.DataFrame(rows).pivot(index="scenario", columns="year", values="alpha").fillna(0)
    chart = _plot_heatmap(frame, title="Alfa anual por escenario", cmap="RdYlGn")
    return f"""
        <p>Cada fila es un escenario, cada columna un anio. Verde &rarr; alfa positivo,
        rojo &rarr; alfa negativo. Un escenario con muchas columnas rojas y una muy verde
        se identifica de inmediato como &laquo;gana por un anio excepcional&raquo; y no
        deberia ganar el ranking.</p>
        {chart}
    """


def _comparison_sensitivity_section(summary: pd.DataFrame, scenario_dirs: list[Path]) -> str:
    """Extrae parametros de `scenario_config.json` y muestra sensibilidad basica."""
    configs = {
        scenario_dir.name: _read_json(scenario_dir / "scenario_config.json")
        for scenario_dir in scenario_dirs
    }
    return """
        <p>Como cambia la metrica compuesta al mover cada parametro del barrido. Se elige
        el valor mas estable, no el maximo puntual (evita <em>overfitting por seleccion</em>).</p>
        <p><em>Panel completo pendiente de definir cuando la rejilla real este cargada.</em></p>
    """


def _comparison_selection_section(selection: dict, summary: pd.DataFrame) -> str:
    top3 = summary.sort_values("composite_rank_mean").head(3).to_dict("records")
    top_rows = "".join(
        f"<tr><td>{row['scenario']}</td>"
        f"<td>{row['rank_beat_rate']}</td>"
        f"<td>{row['rank_median_alpha']}</td>"
        f"<td>{row['rank_worst_year_alpha']}</td>"
        f"<td>{row['rank_max_drawdown']}</td>"
        f"<td><strong>{row['composite_rank_mean']:.2f}</strong></td></tr>"
        for row in top3
    )
    confirmation = selection.get("confirmation_metrics", {})
    conf_body = "".join(f"<li>{key}: {value}</li>" for key, value in confirmation.items())
    return f"""
        <h3>Ganador: <code>{selection['winner']}</code></h3>
        <p>Rango medio: <strong>{selection['composite_rank_mean']:.2f}</strong>. Se rankea
        en las cuatro dimensiones y el rango medio decide.</p>
        <table><thead><tr><th>escenario</th><th>rk beat</th><th>rk mediana</th>
        <th>rk peor anio</th><th>rk drawdown</th><th>rango medio</th></tr></thead>
        <tbody>{top_rows}</tbody></table>
        <h3>Validacion en la era reservada ({selection['confirmation_era'][0]}-{selection['confirmation_era'][1]})</h3>
        <p>La era reservada NO participa en el ranking. Solo valida al ganador. Si se
        hunde aqui, se reporta como resultado negativo, no se elige otro.</p>
        <ul>{conf_body}</ul>
    """


def _comparison_all_runs_section(summary: pd.DataFrame, scenario_dirs: list[Path]) -> str:
    rows = "".join(
        f"<tr><td><a href='{row['scenario']}/agents/'>{row['scenario']}</a></td>"
        f"<td>{row['beat_rate']*100:.0f}%</td>"
        f"<td>{row['median_alpha']*100:.2f}%</td>"
        f"<td>{row['worst_year_alpha']*100:.2f}%</td>"
        f"<td>-{row['max_drawdown']*100:.2f}%</td></tr>"
        for row in summary.sort_values("composite_rank_mean").to_dict("records")
    )
    return f"""
        <p>Tabla completa con enlace al HTML de cada escenario, para poder auditar
        cualquiera de los runs.</p>
        <table><thead><tr><th>escenario</th><th>beat rate</th>
        <th>alfa mediana</th><th>peor anio</th><th>drawdown max</th></tr></thead>
        <tbody>{rows}</tbody></table>
    """


# -------- Helpers --------------------------------------------------------------


def _plot_equity_curve(equity: pd.DataFrame) -> str:
    figure, axes = plt.subplots(figsize=(8, 3))
    axes.plot(pd.to_datetime(equity["snapshot_date"]), equity["portfolio_value"], label="cartera")
    axes.plot(pd.to_datetime(equity["snapshot_date"]), equity["benchmark_value"], label="SPY")
    axes.set_ylabel("valor (base 100)"); axes.legend(); axes.grid(alpha=0.3)
    return _figure_to_img(figure)


def _plot_annual_alpha_bars(annual: pd.DataFrame) -> str:
    figure, axes = plt.subplots(figsize=(8, 3))
    colors = ["#2ca02c" if alpha > 0 else "#d62728" for alpha in annual["alpha"]]
    axes.bar(annual["year"].astype(str), annual["alpha"] * 100, color=colors)
    axes.axhline(0, color="black", linewidth=0.5)
    axes.set_ylabel("alfa (%)"); axes.grid(alpha=0.3, axis="y")
    return _figure_to_img(figure)


def _plot_drawdown_series(equity: pd.DataFrame) -> str:
    import numpy as np
    equity_series = equity["portfolio_value"].to_numpy()
    drawdown = 1 - equity_series / np.maximum.accumulate(equity_series)
    figure, axes = plt.subplots(figsize=(8, 3))
    axes.fill_between(pd.to_datetime(equity["snapshot_date"]), -drawdown * 100, 0,
                       color="#d62728", alpha=0.4)
    axes.set_ylabel("drawdown (%)"); axes.grid(alpha=0.3)
    return _figure_to_img(figure)


def _plot_meta_weights_over_time(weights: pd.DataFrame) -> str:
    figure, axes = plt.subplots(figsize=(8, 3))
    for agent, group in weights.groupby("agent"):
        axes.plot(pd.to_datetime(group["snapshot_date"]), group["weight"], label=agent)
    axes.set_ylabel("peso"); axes.legend(); axes.grid(alpha=0.3)
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
document.querySelectorAll('[data-src]').forEach(container => {
  fetch(container.dataset.src)
    .then(response => response.text())
    .then(csv => {
      const rows = csv.split('\\n').filter(line => line);
      if (!rows.length) return;
      const table = document.createElement('table');
      table.innerHTML = rows.map((row, index) => {
        const cells = row.split(',');
        const tag = index === 0 ? 'th' : 'td';
        return '<tr>' + cells.map(cell => `<${tag}>${cell}</${tag}>`).join('') + '</tr>';
      }).join('');
      container.appendChild(table);
    })
    .catch(error => {
      container.innerHTML = '<p><em>No se pudo cargar ' + container.dataset.src +
        ' (requiere servidor local o navegador con acceso a ficheros locales).</em></p>';
    });
});
</script>
"""

_STYLE = """
<style>
body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 1em; color: #222; }
h1 { border-bottom: 2px solid #333; }
nav.tabs { display: flex; gap: 1em; border-bottom: 1px solid #ccc; margin: 1em 0; }
.tab-link { text-decoration: none; color: #555; padding: 0.5em 1em; }
.tab-link.active { color: #000; border-bottom: 3px solid #2ca02c; font-weight: bold; }
.tab { display: none; }
.tab.active { display: block; }
table { border-collapse: collapse; margin: 1em 0; width: 100%; }
th, td { border: 1px solid #ddd; padding: 0.4em 0.8em; text-align: left; }
th { background: #f5f5f5; }
.cards { display: flex; gap: 1em; margin: 1em 0; }
.card { flex: 1; border: 1px solid #ddd; padding: 1em; text-align: center; border-radius: 6px; }
.metric { font-size: 2em; font-weight: bold; }
tr.winner td { background: #d4edda; font-weight: bold; }
</style>
"""


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
