"""Tabla comparativa (CSV/Parquet) e informe HTML de un experimento.

Reutiliza el sistema de diseño del viewer (``report_layout``, ``PALETTE``, ``table``, ``kpi``) para
que el informe comparativo tenga el mismo aspecto que el informe de un run. Los gráficos se dibujan
como SVG inline (sin matplotlib) para no añadir dependencias y que el HTML sea autocontenido.

El orden de lectura es deliberado: primero el aprendizaje (rank-IC OOS, placebo, mejora sobre
baselines), después la economía. Un escenario que apaga el aprendizaje y empeora esas métricas es la
evidencia de que el aprendizaje aportaba señal real.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import pandas as pd

from module.viewer.shared import PALETTE, kpi, report_layout, table

# Columnas de la tabla comparativa, en orden (aprendizaje primero) y con su etiqueta en español.
_COLUMN_LABELS = {
    "name": "Escenario",
    "rank_ic_final_mean": "Rank-IC medio (OOS)",
    "rank_ic_final_tstat": "Rank-IC t-stat (anual)",
    "rank_ic_positive_years": "Años rank-IC > 0",
    "placebo_percentile": "Percentil placebo",
    "alpha_vs_best_baseline": "Alpha vs. mejor baseline",
    "entry_score_excess_corr": "Corr. score-entrada / exceso",
    "cumulative_alpha": "Alpha acumulada",
    "information_ratio": "Information Ratio",
    "excess_return_t_stat": "t-stat exceso",
    "cost_breakeven_multiplier": "Breakeven de costes (×)",
    "annual_turnover": "Turnover anual",
    "re_scored": "Re-entrenado",
    "seconds": "Segundos",
}
_TABLE_COLUMNS = list(_COLUMN_LABELS.keys())


def _svg_bars(rows: pd.DataFrame, value_col: str, title: str) -> str:
    """Barras horizontales de una métrica por escenario (positivas verde, negativas rojo)."""
    data = [(str(r["name"]), r[value_col]) for _, r in rows.iterrows() if pd.notna(r[value_col])]
    if not data:
        return ""
    width, row_h, pad_left, pad_top = 640, 26, 200, 34
    height = pad_top + row_h * len(data) + 12
    vmax = max((abs(v) for _, v in data), default=1.0) or 1.0
    zero_x = pad_left
    span = width - pad_left - 20
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="expchart">']
    parts.append(f'<text x="12" y="20" class="expchart-title">{html.escape(title)}</text>')
    for i, (name, value) in enumerate(data):
        y = pad_top + i * row_h
        bar_w = (value / vmax) * span if math.isfinite(value) else 0
        color = PALETTE["positive"] if value >= 0 else PALETTE["negative"]
        x = zero_x if value >= 0 else zero_x + bar_w
        parts.append(f'<text x="{pad_left - 8}" y="{y + 16}" text-anchor="end" class="expchart-label">{html.escape(name)}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y + 4}" width="{abs(bar_w):.1f}" height="{row_h - 10}" fill="{color}" rx="2"/>')
        parts.append(f'<text x="{zero_x + (bar_w if value >= 0 else 0) + (6 if value >= 0 else -6):.1f}" y="{y + 16}" '
                     f'text-anchor="{"start" if value >= 0 else "end"}" class="expchart-value">{value:.3f}</text>')
    parts.append(f'<line x1="{zero_x}" y1="{pad_top}" x2="{zero_x}" y2="{height - 12}" class="expchart-axis"/>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_scatter(rows: pd.DataFrame) -> str:
    """Dispersión rank-IC (x) vs. alpha acumulada (y): visualiza el trade-off estabilidad/rentabilidad."""
    pts = [(str(r["name"]), r["rank_ic_final_mean"], r["cumulative_alpha"]) for _, r in rows.iterrows()
           if pd.notna(r["rank_ic_final_mean"]) and pd.notna(r["cumulative_alpha"])]
    if len(pts) < 2:
        return ""
    width, height, pad = 640, 380, 52
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xr = (xmax - xmin) or 1.0
    yr = (ymax - ymin) or 1.0

    def px(v):
        return pad + (v - xmin) / xr * (width - 2 * pad)

    def py(v):
        return height - pad - (v - ymin) / yr * (height - 2 * pad)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="expchart">']
    parts.append('<text x="12" y="20" class="expchart-title">Rank-IC medio (x) vs. alpha acumulada (y)</text>')
    parts.append(f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" class="expchart-axis"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" class="expchart-axis"/>')
    if xmin < 0 < xmax:
        parts.append(f'<line x1="{px(0):.1f}" y1="{pad}" x2="{px(0):.1f}" y2="{height - pad}" class="expchart-grid"/>')
    for name, x, y in pts:
        cx, cy = px(x), py(y)
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{PALETTE["portfolio"]}"/>')
        parts.append(f'<text x="{cx + 8:.1f}" y="{cy + 4:.1f}" class="expchart-label">{html.escape(name)}</text>')
    parts.append("</svg>")
    return "".join(parts)


_EXTRA_CSS = """
<style>
.expchart { width: 100%; height: auto; margin: 8px 0 18px; }
.expchart-title { font-size: 14px; font-weight: 600; fill: var(--ink); }
.expchart-label { font-size: 12px; fill: var(--ink-soft); }
.expchart-value { font-size: 11px; fill: var(--ink-soft); }
.expchart-axis { stroke: var(--border); stroke-width: 1.5; }
.expchart-grid { stroke: var(--border); stroke-width: 1; stroke-dasharray: 3 3; }
.scenario-why { color: var(--ink-soft); font-size: 13px; margin: 2px 0 10px; }
</style>
"""


def build_html(rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    ranked = (
        df.sort_values("rank_ic_final_mean", ascending=False, na_position="last").reset_index(drop=True)
        if "rank_ic_final_mean" in df.columns
        else df.reset_index(drop=True)
    )

    baseline = df[df["name"] == "baseline"]
    kpis = ""
    if not baseline.empty:
        b = baseline.iloc[0]
        def _fmt(v, pct=False):
            if pd.isna(v):
                return "—"
            return f"{v:.1%}" if pct else f"{v:.3f}"
        kpis = (
            kpi("Rank-IC medio (baseline)", _fmt(b.get("rank_ic_final_mean")),
                "señal OOS de final_score", "pos" if (b.get("rank_ic_final_mean") or 0) > 0 else "neg")
            + kpi("Percentil placebo", _fmt(b.get("placebo_percentile"), pct=True), "vs. ranking barajado")
            + kpi("Alpha vs. baseline", _fmt(b.get("alpha_vs_best_baseline"), pct=True), "sobre la mejor regla simple",
                  "pos" if (b.get("alpha_vs_best_baseline") or 0) > 0 else "neg")
            + kpi("Alpha acumulada", _fmt(b.get("cumulative_alpha"), pct=True), "sistema completo")
        )

    table_df = ranked[[c for c in _TABLE_COLUMNS if c in ranked.columns]]
    scenario_notes = "".join(
        f'<p class="scenario-why"><strong>{html.escape(str(r["name"]))}:</strong> '
        f'{html.escape(str(r["why"]))} <code>{html.escape(str(r["overrides"]))}</code></p>'
        for _, r in df.iterrows()
    )

    body = f"""
{_EXTRA_CSS}
<h1>Comparación de escenarios</h1>
<p class="scenario-why">Ordenado por rank-IC medio out-of-sample (¿la IA rankea bien el alpha
futuro?). El bloque de KPIs resume el escenario <strong>baseline</strong> (configuración por defecto).</p>
<div class="kpis">{kpis}</div>
<h2>Tabla comparativa</h2>
{table(table_df, rename=_COLUMN_LABELS)}
<h2>Aprendizaje por escenario</h2>
{_svg_bars(ranked, "rank_ic_final_mean", "Rank-IC medio out-of-sample por escenario")}
{_svg_scatter(ranked)}
<h2>Utilidad económica</h2>
{_svg_bars(ranked, "alpha_vs_best_baseline", "Alpha del sistema menos la mejor baseline simple")}
<h2>Qué prueba cada escenario</h2>
{scenario_notes}
"""
    return report_layout("Comparación de escenarios — TFM", body)


def write_comparison(rows: list[dict], exp_dir: Path) -> Path:
    df = pd.DataFrame(rows)
    df.to_csv(exp_dir / "comparison.csv", index=False)
    try:
        df.to_parquet(exp_dir / "comparison.parquet", index=False)
    except Exception:  # parquet es opcional; el CSV es la fuente de verdad
        pass
    index_path = exp_dir / "index.html"
    index_path.write_text(build_html(rows), encoding="utf-8")
    return index_path
