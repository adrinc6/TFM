"""Generate the final experimental report for the TFM."""

from __future__ import annotations

import html
import json
import logging
import math
from pathlib import Path

import pandas as pd

from environment import PROCESSED_DIR, Settings
from module.backtest.artifacts import SMALL_SAMPLE_CAVEAT, excess_return_statistics
from module.viewer.shared import format_for_html

log = logging.getLogger(__name__)


def build_final_report(settings: Settings) -> Path:
    run_dir = settings.run_dir
    log.info("Building final report for %s", run_dir.name)
    vs = _read_result_csv(run_dir, "portfolio_vs_benchmark")
    transactions = _read_result_csv(run_dir, "portfolio_transactions")
    holdings = _read_result_csv(run_dir, "portfolio_monthly_holdings")
    decisions = _read_result_csv(run_dir, "portfolio_decision_log")
    position_performance = _read_result_csv(run_dir, "position_performance")
    sell_reasons = _read_result_csv(run_dir, "sell_reasons_summary")
    sector_exposure = _read_result_csv(run_dir, "sector_exposure")
    explainability = _read_json(PROCESSED_DIR / "model_explainability.json")

    metrics = _metrics(vs)
    body = (
        "<h1>Informe Final GARP AI</h1>"
        + _executive_summary(metrics)
        + "<h2>Metricas de Rendimiento</h2>"
        + _table(pd.DataFrame([metrics]))
        + "<h2>Cartera Frente a Benchmark</h2>"
        + _figure("viewer/charts/portfolio_vs_benchmark.png", "Valor de la cartera frente al benchmark SPY")
        + _figure("viewer/charts/period_alpha.png", "Alpha mensual frente al benchmark")
        + _figure("viewer/charts/drawdown.png", "Drawdown de la cartera")
        + "<h3>Episodios de Drawdown</h3>"
        + _table(drawdown_episodes(vs))
        + _table(vs.tail(24))
        + "<h2>Rendimiento de Posiciones</h2>"
        + _figure("viewer/charts/position_performance_bars.png", "Retorno de la accion, retorno anualizado de la accion y retorno anualizado del benchmark durante cada periodo de holding")
        + _table(_position_performance_summary(position_performance))
        + "<h2>Exposicion Sectorial</h2>"
        + _figure("viewer/charts/sector_exposure.png", "Exposicion sectorial de la cartera en el tiempo")
        + _table(sector_exposure.tail(20))
        + "<h2>Resumen de Motivos de Venta</h2>"
        + _table(sell_reasons)
        + "<h2>Mejores Decisiones</h2>"
        + _table(_best_decisions(decisions))
        + "<h2>Peores Decisiones</h2>"
        + _table(_worst_decisions(decisions))
        + "<h2>Mejores Tesis</h2>"
        + _table(_best_theses(holdings))
        + "<h2>Tesis mas Debiles</h2>"
        + _table(_worst_theses(holdings))
        + "<h2>Clasificacion de Oportunidades</h2>"
        + _table(_classification_summary(holdings))
        + "<h2>Importancia de Componentes</h2>"
        + _figure("viewer/charts/feature_importance.png", "Variables mas relevantes del modelo")
        + _table(_component_importance(explainability))
        + "<h2>Importancia de Variables</h2>"
        + _table(_feature_importance(explainability))
        + "<h2>Conclusiones Automaticas</h2>"
        + _conclusions(metrics, holdings, transactions)
    )
    path = run_dir / "final_report.html"
    path.write_text(_layout(body), encoding="utf-8")
    log.info("Final report written to %s", path)
    return path


def drawdown_episodes(vs: pd.DataFrame) -> pd.DataFrame:
    if vs.empty or "portfolio_value" not in vs.columns or "date" not in vs.columns:
        return pd.DataFrame()
    values = vs["portfolio_value"].reset_index(drop=True)
    dates = pd.to_datetime(vs["date"]).reset_index(drop=True)
    running_max = values.cummax()
    drawdown = values / running_max - 1
    episodes = []
    in_drawdown = False
    peak_index = 0
    for i in range(len(values)):
        if drawdown[i] < 0 and not in_drawdown:
            in_drawdown = True
            peak_index = i - 1 if i > 0 else 0
        is_last = i == len(values) - 1
        if in_drawdown and (drawdown[i] >= 0 or is_last):
            window = drawdown[peak_index:i + 1]
            trough_offset = int(window.values.argmin())
            trough_index = peak_index + trough_offset
            recovered = drawdown[i] >= 0
            episodes.append({
                "peak_date": dates[peak_index].date().isoformat(),
                "trough_date": dates[trough_index].date().isoformat(),
                "recovery_date": dates[i].date().isoformat() if recovered else "",
                "depth": float(drawdown[trough_index]),
                "duration_days": int((dates[i] - dates[peak_index]).days),
                "recovered": bool(recovered),
            })
            in_drawdown = False
    return pd.DataFrame(episodes).sort_values("depth").head(10) if episodes else pd.DataFrame()


def _metrics(vs: pd.DataFrame) -> dict:
    if vs.empty:
        return {"CAGR": 0, "Sharpe": 0, "Sortino": 0, "Max Drawdown": 0, "Alpha": 0}
    returns = vs.get("portfolio_period_return", pd.Series(dtype=float)).fillna(0)
    benchmark_returns = vs.get("benchmark_period_return", pd.Series(dtype=float)).fillna(0)
    years = max((pd.to_datetime(vs["date"]).max() - pd.to_datetime(vs["date"]).min()).days / 365.25, 1 / 12)
    ending = float(vs["portfolio_value"].iloc[-1])
    gross_ending = float(vs["portfolio_gross_value"].iloc[-1]) if "portfolio_gross_value" in vs.columns else ending
    benchmark_ending = float(vs.get("benchmark_value", pd.Series([1])).iloc[-1])
    downside = returns[returns < 0]
    stats = excess_return_statistics(vs)
    return {
        "CAGR": ending ** (1 / years) - 1,
        "Gross CAGR": gross_ending ** (1 / years) - 1,
        "Benchmark CAGR": benchmark_ending ** (1 / years) - 1,
        "Sharpe": _annualized_ratio(returns),
        "Sortino": _annualized_ratio(returns, downside_only=True),
        "Max Drawdown": _max_drawdown(vs["portfolio_value"]),
        "Alpha": (ending - 1) - (benchmark_ending - 1),
        "Gross Alpha": (gross_ending - 1) - (benchmark_ending - 1),
        "Total Cost Drag": float(vs.get("transaction_cost_drag", pd.Series([0])).sum()),
        "Average Period Alpha": float((returns - benchmark_returns).mean()),
        "Information Ratio": stats["information_ratio"],
        "Tracking Error (annualized)": stats["tracking_error_annualized"],
        "Excess Return t-stat": stats["excess_return_t_stat"],
        "Periods (n)": stats["periods_n"],
    }


def _annualized_ratio(returns: pd.Series, downside_only: bool = False) -> float:
    series = returns[returns < 0] if downside_only else returns
    denom = float(series.std())
    if denom == 0 or pd.isna(denom):
        return 0.0
    return float(returns.mean() / denom * (12 ** 0.5))


def _max_drawdown(values: pd.Series) -> float:
    running_max = values.cummax()
    drawdown = values / running_max - 1
    return float(drawdown.min())


def _best_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    return decisions.sort_values(["conviction_score", "thesis_persistence_score"], ascending=False).head(10)


def _worst_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    return decisions.sort_values(["conviction_score", "thesis_persistence_score"], ascending=True).head(10)


def _best_theses(holdings: pd.DataFrame) -> pd.DataFrame:
    cols = ["date", "ticker", "current_thesis_state", "current_conviction_score", "thesis_persistence_score", "investment_thesis"]
    return _select(holdings.sort_values(["thesis_persistence_score", "current_conviction_score"], ascending=False), cols).head(10)


def _worst_theses(holdings: pd.DataFrame) -> pd.DataFrame:
    cols = ["date", "ticker", "current_thesis_state", "current_conviction_score", "thesis_persistence_score", "current_exit_thesis"]
    return _select(holdings.sort_values(["thesis_persistence_score", "current_conviction_score"], ascending=True), cols).head(10)


def _classification_summary(holdings: pd.DataFrame) -> pd.DataFrame:
    if holdings.empty or "opportunity_type_original" not in holdings.columns:
        return pd.DataFrame()
    return holdings.groupby("opportunity_type_original").agg(
        observations=("ticker", "count"),
        avg_conviction=("current_conviction_score", "mean"),
        avg_persistence=("thesis_persistence_score", "mean"),
    ).reset_index().sort_values("avg_persistence", ascending=False)


def _position_performance_summary(position_performance: pd.DataFrame) -> pd.DataFrame:
    required = ["ticker", "holding_days", "total_return", "annualized_return", "benchmark_annualized_return", "excess_total_return"]
    if position_performance.empty or any(column not in position_performance.columns for column in required):
        return pd.DataFrame()
    df = position_performance.copy()
    for column in required[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    rows = []
    for ticker, group in df.groupby("ticker"):
        weights = group["holding_days"].fillna(1).clip(lower=1)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            continue
        rows.append({
            "ticker": ticker,
            "lots": int(len(group)),
            "total_return": float((1 + group["total_return"]).prod() - 1),
            "annualized_return": float((group["annualized_return"] * weights).sum() / weight_sum),
            "benchmark_annualized_return": float((group["benchmark_annualized_return"] * weights).sum() / weight_sum),
            "excess_total_return": float(group["excess_total_return"].sum()),
        })
    return pd.DataFrame(rows).sort_values("excess_total_return", ascending=False).head(20)


def _component_importance(payload: dict) -> pd.DataFrame:
    formula = payload.get("garp_score_formula", {})
    return pd.DataFrame([{"component": key, "weight": value} for key, value in formula.items()])


def _feature_importance(payload: dict) -> pd.DataFrame:
    rows = []
    for component, features in payload.get("feature_importance", {}).items():
        if isinstance(features, dict):
            for feature, importance in features.items():
                rows.append({"component": component, "feature": feature, "importance": importance})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).head(30) if rows else pd.DataFrame()


def _pct(value) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "no disponible"
    return f"{value:.2%}"


def _executive_summary(metrics: dict) -> str:
    alpha = metrics.get("Alpha", 0)
    if alpha is None or (isinstance(alpha, float) and math.isnan(alpha)):
        return "<p>El alpha frente al benchmark no esta disponible para esta ejecucion.</p>"
    result = "supero" if alpha > 0 else "quedo por debajo de"
    return f"<p>La gestion sistematica GARP {result} al benchmark con un alpha de {_pct(alpha)} en esta ejecucion.</p>"


def _conclusions(metrics: dict, holdings: pd.DataFrame, transactions: pd.DataFrame) -> str:
    turnover = int((transactions.get("action", pd.Series(dtype=str)) == "SELL").sum()) if not transactions.empty else 0
    avg_persistence = holdings.get("thesis_persistence_score", pd.Series([0])).mean() if not holdings.empty else 0
    avg_persistence = 0.0 if pd.isna(avg_persistence) else avg_persistence
    return (
        "<ul>"
        f"<li>Alpha frente al benchmark: {_pct(metrics.get('Alpha'))}.</li>"
        f"<li>Information ratio: {metrics.get('Information Ratio', 0):.2f}; "
        f"tracking error (anualizado): {_pct(metrics.get('Tracking Error (annualized)'))}; "
        f"t-stat del retorno en exceso: {metrics.get('Excess Return t-stat', 0):.2f} "
        f"sobre {metrics.get('Periods (n)', 0)} periodos mensuales. {SMALL_SAMPLE_CAVEAT}</li>"
        f"<li>Persistencia media de tesis: {avg_persistence:.2f}.</li>"
        f"<li>Decisiones de venta: {turnover}, lo que indica que la cartera se gestiona como un libro de tesis vivo y no como un ranking de corto plazo.</li>"
        "<li>La arquitectura separa las decisiones de compra-hoy de las decisiones de mantener, algo esencial para la gestion de una cartera GARP a largo plazo.</li>"
        "<li><strong>Limitacion metodologica:</strong> el universo de tickers es un listado estatico y actual "
        "aplicado retroactivamente; esto constituye un sesgo de supervivencia. La ventana de entrenamiento del "
        "modelo ML hereda este sesgo, y aunque la cartera viva solo opera dentro de las fechas de inicio/fin "
        "configuradas (lo cual acota, pero no elimina, el problema), las afirmaciones de rendimiento deben leerse "
        "con esta salvedad en mente.</li>"
        "</ul>"
    )


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_result_csv(run_dir: Path, name: str) -> pd.DataFrame:
    root_path = run_dir / f"{name}.csv"
    audit_path = run_dir / "audit" / f"{name}.csv"
    if root_path.exists():
        return pd.read_csv(root_path)
    if audit_path.exists():
        return pd.read_csv(audit_path)
    return pd.DataFrame()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _select(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[column for column in columns if column in df.columns]] if not df.empty else df


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No hay datos disponibles.</p>"
    return format_for_html(df).to_html(index=False, escape=True)


def _figure(src: str, caption: str) -> str:
    return (
        f'<figure><img src="{html.escape(src)}" alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def _layout(body: str) -> str:
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Informe Final GARP AI</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; color: #172026; background: #f7f8f8; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
h1 {{ font-size: 30px; }}
h2 {{ margin-top: 30px; font-size: 20px; }}
table {{ width: 100%; border-collapse: collapse; margin: 14px 0; background: white; border: 1px solid #d7dddd; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #e7ebeb; text-align: left; font-size: 13px; vertical-align: top; }}
th {{ background: #eef2f2; }}
figure {{ margin: 18px 0 24px; background: white; border: 1px solid #d7dddd; padding: 12px; }}
figure img {{ display: block; width: 100%; height: auto; }}
figcaption {{ color: #516066; font-size: 13px; margin-top: 8px; }}
</style>
</head>
<body><main>{body}</main></body>
</html>"""
