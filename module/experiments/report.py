"""Tabla comparativa (CSV/Parquet) e informe HTML de un experimento.

Reutiliza el sistema de diseño del viewer (``report_layout``, ``PALETTE``, ``table``, ``kpi``) para
que el informe comparativo tenga el mismo aspecto que el informe de un run. Los gráficos se dibujan
como SVG inline (sin matplotlib) para no añadir dependencias y que el HTML sea autocontenido.

El informe es profundo: además de la vista global (todos los escenarios juntos + deltas frente al
baseline), abre una subsección por bloque temático (aprendizaje / estabilidad / utilidad /
pesos_meta), donde cada escenario se compara por pares contra el baseline. Cada escenario enlaza a su
propio ``viewer/index.html``, porque ahora cada uno corre el pipeline completo y deja su informe
navegable en ``<nombre>/viewer/index.html``.

El orden de lectura es deliberado: primero el aprendizaje (rank-IC OOS, placebo, mejora sobre
baselines), después la economía. Un escenario que apaga el aprendizaje y empeora esas métricas es la
evidencia de que el aprendizaje aportaba señal real.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import pandas as pd

from module.viewer.shared import PALETTE, format_for_html, kpi, report_layout

# Columnas de métricas absolutas de la tabla comparativa, en orden (aprendizaje primero) y con su
# etiqueta en español.
_COLUMN_LABELS = {
    "name": "Escenario",
    "block": "Bloque",
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
# Tabla global: todas las métricas + bloque. Las tablas por bloque omiten la columna 'block' (es
# constante dentro del bloque).
_GLOBAL_COLUMNS = list(_COLUMN_LABELS.keys())
_BLOCK_COLUMNS = [c for c in _GLOBAL_COLUMNS if c != "block"]

# Métricas para las que calculamos delta = escenario - baseline (las de lectura clave del TFM). El
# baseline queda a 0 por construcción. Se escriben en comparison.csv como delta_<metrica>.
_DELTA_METRICS = {
    "rank_ic_final_mean": "Δ Rank-IC medio",
    "placebo_percentile": "Δ Percentil placebo",
    "alpha_vs_best_baseline": "Δ Alpha vs. baseline",
    "cumulative_alpha": "Δ Alpha acumulada",
    "information_ratio": "Δ Information Ratio",
}
_DELTA_COLUMNS = ["name", *[f"delta_{m}" for m in _DELTA_METRICS]]
_DELTA_LABELS = {"name": "Escenario", **{f"delta_{m}": lbl for m, lbl in _DELTA_METRICS.items()}}

# Eje característico de cada bloque: la métrica cuya lectura resume la pregunta del bloque. Se dibuja
# como barras dentro de su subsección.
_BLOCK_AXIS = {
    "aprendizaje": ("rank_ic_final_mean", "Rank-IC medio out-of-sample por escenario"),
    "estabilidad": ("cumulative_alpha", "Alpha acumulada por escenario (dispersión entre semillas y supuestos)"),
    "utilidad": ("alpha_vs_best_baseline", "Alpha del sistema menos la mejor baseline simple"),
    "pesos_meta": ("rank_ic_final_mean", "Rank-IC medio out-of-sample por prior de pesos"),
}
# Orden de lectura de los bloques (aprende → estable → útil → calibración de pesos).
_BLOCK_ORDER = ["aprendizaje", "estabilidad", "utilidad", "pesos_meta"]
_BLOCK_TITLES = {
    "aprendizaje": "Bloque — Aprendizaje (¿la IA aprende?)",
    "estabilidad": "Bloque — Estabilidad (¿es consistente?)",
    "utilidad": "Bloque — Utilidad (¿es económicamente útil?)",
    "pesos_meta": "Bloque — Pesos del meta-agente (calibración del prior)",
}


def _with_block(df: pd.DataFrame) -> pd.DataFrame:
    """Garantiza la columna 'block' (vacía si el escenario no la trae)."""
    out = df.copy()
    if "block" not in out.columns:
        out["block"] = ""
    out["block"] = out["block"].fillna("")
    return out


def _with_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas delta_<metrica> = escenario - baseline para las métricas clave.

    El baseline se identifica por name=='baseline'. Sin baseline, no hay referencia y las deltas
    quedan NaN.
    """
    out = _with_block(df)
    baseline = out[out["name"] == "baseline"]
    b = baseline.iloc[0] if not baseline.empty else None
    for metric in _DELTA_METRICS:
        col = f"delta_{metric}"
        if metric in out.columns and b is not None:
            out[col] = pd.to_numeric(out[metric], errors="coerce") - _num(b.get(metric))
        else:
            out[col] = float("nan")
    return out


def _num(value) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _ranked(df: pd.DataFrame) -> pd.DataFrame:
    if "rank_ic_final_mean" in df.columns:
        return df.sort_values("rank_ic_final_mean", ascending=False, na_position="last").reset_index(drop=True)
    return df.reset_index(drop=True)


def _svg_bars(rows: pd.DataFrame, value_col: str, title: str) -> str:
    """Barras horizontales de una métrica por escenario (positivas verde, negativas rojo)."""
    if value_col not in rows.columns:
        return ""
    data = [(str(r["name"]), _num(r[value_col])) for _, r in rows.iterrows() if pd.notna(r[value_col])]
    data = [(n, v) for n, v in data if math.isfinite(v)]
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
    if "rank_ic_final_mean" not in rows.columns or "cumulative_alpha" not in rows.columns:
        return ""
    pts = [(str(r["name"]), _num(r["rank_ic_final_mean"]), _num(r["cumulative_alpha"])) for _, r in rows.iterrows()
           if pd.notna(r["rank_ic_final_mean"]) and pd.notna(r["cumulative_alpha"])]
    pts = [(n, x, y) for n, x, y in pts if math.isfinite(x) and math.isfinite(y)]
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


def _linked_table(df: pd.DataFrame, columns: list[str], labels: dict[str, str]) -> str:
    """Como ``table``, pero enlaza el nombre de cada escenario a su ``<nombre>/viewer/index.html``.

    Cada escenario corre ahora el pipeline completo y deja su informe navegable en esa ruta relativa
    al índice del experimento, así que la comparativa enlaza directamente a él.
    """
    cols = [c for c in columns if c in df.columns]
    if df.empty or not cols:
        return '<div class="empty">No hay datos disponibles.</div>'
    shown = format_for_html(df[cols].reset_index(drop=True))
    names = df["name"].reset_index(drop=True)
    header = "".join(f"<th>{html.escape(labels.get(c, c))}</th>" for c in cols)
    body_rows = []
    for i in range(len(shown)):
        cells = []
        for c in cols:
            value = shown.iloc[i][c]
            text = "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)
            if c == "name":
                name = str(names.iloc[i])
                href = html.escape(f"{name}/viewer/index.html")
                cells.append(f'<td><a class="audit" href="{href}">{html.escape(name)}</a></td>')
            else:
                cells.append(f"<td>{html.escape(text)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return ('<div class="table-scroll"><table border="0"><thead><tr>'
            + header + "</tr></thead><tbody>" + "".join(body_rows) + "</tbody></table></div>")


_EXTRA_CSS = """
<style>
.expchart { width: 100%; height: auto; margin: 8px 0 18px; }
.expchart-title { font-size: 14px; font-weight: 600; fill: var(--ink); }
.expchart-label { font-size: 12px; fill: var(--ink-soft); }
.expchart-value { font-size: 11px; fill: var(--ink-soft); }
.expchart-axis { stroke: var(--border); stroke-width: 1.5; }
.expchart-grid { stroke: var(--border); stroke-width: 1; stroke-dasharray: 3 3; }
.scenario-why { color: var(--ink-soft); font-size: 13px; margin: 2px 0 10px; }
.block-section { margin-top: 44px; padding-top: 8px; border-top: 2px solid var(--border); }
.subtle { color: var(--ink-soft); font-size: 13px; margin: 14px 0 6px; font-weight: 600; }
</style>
"""


def _baseline_kpis(df: pd.DataFrame) -> str:
    baseline = df[df["name"] == "baseline"]
    if baseline.empty:
        return ""
    b = baseline.iloc[0]

    def _fmt(v, pct=False):
        if pd.isna(v):
            return "—"
        return f"{v:.1%}" if pct else f"{v:.3f}"

    return (
        kpi("Rank-IC medio (baseline)", _fmt(b.get("rank_ic_final_mean")),
            "señal OOS de final_score", "pos" if _num(b.get("rank_ic_final_mean")) > 0 else "neg")
        + kpi("Percentil placebo", _fmt(b.get("placebo_percentile"), pct=True), "vs. ranking barajado")
        + kpi("Alpha vs. baseline", _fmt(b.get("alpha_vs_best_baseline"), pct=True), "sobre la mejor regla simple",
              "pos" if _num(b.get("alpha_vs_best_baseline")) > 0 else "neg")
        + kpi("Alpha acumulada", _fmt(b.get("cumulative_alpha"), pct=True), "sistema completo")
    )


def _why_notes(rows: pd.DataFrame) -> str:
    return "".join(
        f'<p class="scenario-why"><strong>{html.escape(str(r["name"]))}:</strong> '
        f'{html.escape(str(r.get("why", "")))} <code>{html.escape(str(r.get("overrides", "")))}</code></p>'
        for _, r in rows.iterrows()
    )


def _block_section(block: str, df: pd.DataFrame) -> str:
    """Subsección de un bloque: baseline + sus escenarios, con tabla absoluta, deltas por pares y
    barras del eje del bloque."""
    baseline = df[df["name"] == "baseline"]
    block_rows = df[df["block"] == block]
    if block_rows.empty:
        return ""
    scoped = _ranked(pd.concat([baseline, block_rows]).drop_duplicates(subset="name"))
    axis_col, axis_title = _BLOCK_AXIS.get(block, ("rank_ic_final_mean", "Rank-IC medio por escenario"))
    title = _BLOCK_TITLES.get(block, f"Bloque — {block}")
    return f"""
<section class="block-section">
<h2>{html.escape(title)}</h2>
<p class="scenario-why">Solo los escenarios de este bloque, comparados contra el <strong>baseline</strong>.</p>
{_linked_table(scoped, _BLOCK_COLUMNS, _COLUMN_LABELS)}
<p class="subtle">Comparación por pares: delta de cada escenario frente al baseline</p>
{_linked_table(_ranked(block_rows), _DELTA_COLUMNS, _DELTA_LABELS)}
{_svg_bars(scoped, axis_col, axis_title)}
<p class="subtle">Qué prueba cada escenario</p>
{_why_notes(block_rows)}
"""


def build_html(rows: list[dict]) -> str:
    df = _with_deltas(pd.DataFrame(rows))
    ranked = _ranked(df)

    blocks_present = [b for b in _BLOCK_ORDER if b in set(df["block"])]
    # Bloques no previstos (por si aparece uno nuevo): se añaden al final, en orden estable.
    extra_blocks = [b for b in sorted(set(df["block"])) if b and b not in _BLOCK_ORDER and b != "baseline"]
    block_sections = "".join(_block_section(b, df) for b in [*blocks_present, *extra_blocks])

    body = f"""
{_EXTRA_CSS}
<h1>Comparación de escenarios</h1>
<p class="scenario-why">Ordenado por rank-IC medio out-of-sample (¿la IA rankea bien el alpha
futuro?). El bloque de KPIs resume el escenario <strong>baseline</strong> (configuración por defecto).
Cada escenario corre el pipeline completo: pincha su nombre para abrir su informe propio.</p>
<div class="kpis">{_baseline_kpis(df)}</div>

<h2>Global — todos los escenarios</h2>
{_linked_table(ranked, _GLOBAL_COLUMNS, _COLUMN_LABELS)}
{_svg_bars(ranked, "rank_ic_final_mean", "Rank-IC medio out-of-sample por escenario")}
{_svg_scatter(ranked)}
{_svg_bars(ranked, "alpha_vs_best_baseline", "Alpha del sistema menos la mejor baseline simple")}

<h2>Global — deltas frente al baseline</h2>
<p class="scenario-why">Diferencia de cada escenario respecto al baseline en las métricas clave
(positivo = mejora sobre la referencia). El baseline queda a cero por construcción.</p>
{_linked_table(ranked, _DELTA_COLUMNS, _DELTA_LABELS)}

{block_sections}
"""
    return report_layout("Comparación de escenarios — TFM", body)


def write_comparison(rows: list[dict], exp_dir: Path) -> Path:
    df = _with_deltas(pd.DataFrame(rows))
    df.to_csv(exp_dir / "comparison.csv", index=False)
    try:
        df.to_parquet(exp_dir / "comparison.parquet", index=False)
    except Exception:  # parquet es opcional; el CSV es la fuente de verdad
        pass
    index_path = exp_dir / "index.html"
    index_path.write_text(build_html(rows), encoding="utf-8")
    return index_path
