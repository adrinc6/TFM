"""Resume un backtest terminado en una fila de métricas comparables entre escenarios.

Prioriza las métricas de APRENDIZAJE (rank-IC out-of-sample, placebo, mejora sobre baselines) por
encima de las económicas (alpha, IR, breakeven de costes, turnover): la pregunta del TFM es si la IA
aprende, y solo después si eso es útil.

No recalcula nada: lee las tablas que ``run_backtest`` ya devuelve en su dict ``outputs`` más el
``model_walk_forward_diagnostics.csv`` que la etapa ml escribe en ``run_dir`` (el diagnóstico de
rank-IC no viaja dentro de ``outputs``). Cada extracción tolera que falte una columna o tabla
—devuelve ``nan``— para que un escenario degradado no rompa el barrido entero.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


def _f(value) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _learning_metrics(run_dir: Path) -> dict[str, float]:
    """Rank-IC OOS de final_score: media, t-stat medio por año y nº de años con IC positivo."""
    path = run_dir / "model_walk_forward_diagnostics.csv"
    if not path.exists():
        return {"rank_ic_final_mean": float("nan"), "rank_ic_final_tstat": float("nan"), "rank_ic_positive_years": float("nan")}
    diag = pd.read_csv(path)
    ic = pd.to_numeric(diag.get("rank_ic_final"), errors="coerce").dropna()
    # Las columnas *_year_* traen el mismo valor repetido en cada snapshot del año. Deduplicamos a
    # un valor por año antes de agregar, para no ponderar de más los años con más snapshots.
    per_year = diag.assign(
        year=pd.to_datetime(diag.get("snapshot_date"), errors="coerce").dt.year,
        year_mean=pd.to_numeric(diag.get("rank_ic_final_year_mean"), errors="coerce"),
        year_tstat=pd.to_numeric(diag.get("rank_ic_final_year_tstat"), errors="coerce"),
    ).dropna(subset=["year"]).groupby("year").agg(year_mean=("year_mean", "first"), year_tstat=("year_tstat", "first"))
    year_mean = per_year["year_mean"].dropna()
    year_tstat = per_year["year_tstat"].dropna()
    return {
        "rank_ic_final_mean": _f(ic.mean()) if len(ic) else float("nan"),
        # t-stat medio de la media anual de rank-IC: un valor por año, luego promedio.
        "rank_ic_final_tstat": _f(year_tstat.mean()) if len(year_tstat) else float("nan"),
        "rank_ic_positive_years": float((year_mean > 0).sum()) if len(year_mean) else float("nan"),
    }


def _placebo_percentile(outputs: dict[str, pd.DataFrame]) -> float:
    placebo = outputs.get("placebo_summary", pd.DataFrame())
    if placebo.empty or "percentil_real" not in placebo.columns:
        return float("nan")
    return _f(placebo.iloc[0]["percentil_real"])


def _alpha_vs_best_baseline(outputs: dict[str, pd.DataFrame]) -> float:
    """Alpha del sistema completo menos la mejor de las baselines simples (equal/momentum/valuation).
    Positivo = el sistema bate a la mejor regla trivial; negativo = no añade sobre lo simple."""
    comp = outputs.get("baseline_comparison", pd.DataFrame())
    if comp.empty or "alpha_acumulada" not in comp.columns or "estrategia" not in comp.columns:
        return float("nan")
    system = comp[comp["estrategia"].str.startswith("Sistema completo")]
    baselines = comp[~comp["estrategia"].str.startswith("Sistema completo")]
    if system.empty or baselines.empty:
        return float("nan")
    return _f(system.iloc[0]["alpha_acumulada"] - baselines["alpha_acumulada"].max())


def _entry_score_excess_corr(outputs: dict[str, pd.DataFrame]) -> float:
    """Correlación (Spearman) entre el manager_score de entrada y el exceso realizado, de
    edge_attribution: ¿un score de entrada más alto predice de verdad un mejor resultado?"""
    edge = outputs.get("edge_attribution", pd.DataFrame())
    if edge.empty or "metrica" not in edge.columns or "valor" not in edge.columns:
        return float("nan")
    row = edge[edge["metrica"].str.contains("manager_score", case=False, na=False)]
    return _f(row.iloc[0]["valor"]) if not row.empty else float("nan")


def _economic_metrics(outputs: dict[str, pd.DataFrame]) -> dict[str, float]:
    summary = outputs.get("portfolio_monthly_summary", {})
    if isinstance(summary, pd.DataFrame):
        summary = summary.iloc[0].to_dict() if not summary.empty else {}
    robustness = outputs.get("robustness_summary", pd.DataFrame())
    breakeven = float("nan")
    if not robustness.empty and "breakeven_cost_multiplier" in robustness.columns:
        breakeven = _f(robustness.iloc[0]["breakeven_cost_multiplier"])
    turnover = outputs.get("portfolio_turnover", pd.DataFrame())
    annual_turnover = float("nan")
    if not turnover.empty and "annual_turnover" in turnover.columns:
        annual_turnover = _f(turnover.iloc[0]["annual_turnover"])
    return {
        "cumulative_alpha": _f(summary.get("cumulative_alpha")),
        "information_ratio": _f(summary.get("information_ratio")),
        "excess_return_t_stat": _f(summary.get("excess_return_t_stat")),
        "cost_breakeven_multiplier": breakeven,
        "annual_turnover": annual_turnover,
    }


def collect_metrics(outputs: dict[str, pd.DataFrame], run_dir: Path) -> dict[str, float]:
    """Una fila de métricas para el escenario: aprendizaje primero, economía después."""
    metrics = _learning_metrics(run_dir)
    metrics["placebo_percentile"] = _placebo_percentile(outputs)
    metrics["alpha_vs_best_baseline"] = _alpha_vs_best_baseline(outputs)
    metrics["entry_score_excess_corr"] = _entry_score_excess_corr(outputs)
    metrics.update(_economic_metrics(outputs))
    return metrics
