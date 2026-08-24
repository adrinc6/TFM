"""Exporta activos de los estudios de referencia para el manuscrito XeLaTeX.

Uso desde la raíz del repositorio:
    python latex/scripts/export_study_assets.py \
        --study-id study-20260817-094411-568bd37e \
        --chain-study-id study-20260816-182345-3cc1a5fb \
        --chain-study-id study-20260817-021135-b5926b62 \
        --chain-study-id study-20260817-094411-568bd37e \
        --portfolio-study-id study-20260817-212856-f86ca822

``--study-id`` aporta la evidencia predictiva; ``--portfolio-study-id``, la económica, leída del
ganador de la rejilla de cartera; y los ``--chain-study-id``, en orden de ejecución, las figuras de
la cadena de studies encadenados.

Los PNG se generan a 300 dpi, con fondo blanco y dimensiones de impresión, y están preparados
para incluirse directamente en Overleaf. Cada salida queda registrada en
``latex/asset_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LATEX = ROOT / "latex"

# Esta migración editorial queda deliberadamente anclada a la cadena adoptada. El exportador no
# acepta estudios "parecidos" ni el estudio posterior: mezclar una sola cifra invalidaría la
# comparación que sostienen memoria y defensa.
ADOPTED_MODEL_STUDIES = (
    "study-20260816-182345-3cc1a5fb",
    "study-20260817-021135-b5926b62",
    "study-20260817-094411-568bd37e",
)
ADOPTED_PORTFOLIO_STUDY = "study-20260817-212856-f86ca822"

SPIVA_SOURCE = {
    "title": "SPIVA U.S. Scorecard Year-End 2025",
    "url": "https://www.spglobal.com/spdji/en/documents/spiva/spiva-us-year-end-2025.pdf",
    "table": "Report 1a",
    "data_as_of": "2025-12-31",
    "series": "All Large-Cap Funds frente al S&P 500, rentabilidad absoluta",
}
SPIVA_UNDERPERFORMANCE = {
    1: 78.78,
    3: 66.84,
    5: 88.96,
    10: 85.59,
    15: 89.93,
    20: 92.89,
}

# Paleta sobria, apta para impresión y distinguible con daltonismo.
NAVY = "#16324F"
TEAL = "#007C83"
GOLD = "#C58B00"
RED = "#B33A3A"
SLATE = "#5D6D7E"
LIGHT = "#E8EEF2"
GREEN = "#2C7A4B"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#D9E1E7",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
    }
)


@dataclass(frozen=True)
class Paths:
    study: Path
    evidence: Path
    figures: Path
    tables: Path


def pct(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{100 * float(value):.{digits}f}".replace(".", ",") + r"\%"


def num(value: float | None, digits: int = 4) -> str:
    """Número con coma decimal; ``—`` si la cifra no existe.

    Una celda vacía no es un fallo: una pasada que corrió sin diagnósticos posteriores al ganador
    nunca produjo ese número, y escribirlo como ausente es más honesto que omitir la fila.
    """
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}".replace(".", ",")


def integer(value: int | float) -> str:
    """Entero con separador de millares español, apto para texto LaTeX.

    Redondea en vez de truncar: los recuentos ya son enteros y no cambian, pero una magnitud
    continua como el equilibrio de costes (294,79 pb) se imprimía como 294 mientras el texto la
    citaba como 295, y esa discrepancia de una unidad entre macro y prosa es exactamente la clase de
    desajuste que la auditoría no puede detectar.
    """
    return f"{round(float(value)):,}".replace(",", ".")


def macro_content(values: dict[str, str]) -> str:
    """Serializa cifras auditables sin exponer su procedencia al lector."""
    lines = [
        "% GENERADO por latex/scripts/export_study_assets.py. No editar a mano.",
        "% Las rutas y los identificadores se verifican internamente; el PDF solo consume macros.",
    ]
    lines.extend(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in values.items())
    return "\n".join(lines) + "\n"


def tex(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def tex_free(value: object) -> str:
    """Valor legible para etiquetas de figura (matplotlib, sin escapes LaTeX).

    Los valores del barrido de cartera llegan serializados como JSON, de modo
    que las cadenas conservan sus comillas (``"half_horizon"``).
    """
    text = str(value).strip('"')
    return text.replace("_", " ")


def load_features(paths: "Paths") -> pd.DataFrame:
    """Une el catálogo declarado con los diagnósticos calculados.

    Las dos fuentes no coinciden: ``feature_catalog.json`` describe 68 features
    con su bloque y sus agentes, mientras que ``feature_diagnostics.parquet``
    contiene 73 filas porque incluye columnas derivadas que el catálogo no
    declara por separado. Se hace una unión externa y se marca el origen, de
    forma que la asimetría quede visible en la tabla y no se oculte.
    """
    catalog = read_json(paths.evidence / "feature_catalog.json")
    declared = pd.DataFrame(
        [
            {
                "feature": item["name"],
                "block": item["block"],
                "agents": ", ".join(item["agents"]),
                "direction": item.get("direction"),
                "source": item.get("source"),
            }
            for item in catalog["features"]
        ]
    )
    diagnostics = pd.read_parquet(paths.evidence / "feature_diagnostics.parquet")
    merged = declared.merge(diagnostics, on="feature", how="outer", indicator=True)
    merged["origin"] = merged["_merge"].map(
        {"both": "catálogo", "left_only": "solo catálogo", "right_only": "solo diagnóstico"}
    )
    merged["block"] = merged["block"].fillna("(no declarado)")
    merged["agents"] = merged["agents"].fillna("—")
    return merged.drop(columns="_merge")


def load_paths(study_id: str) -> Paths:
    study = ROOT / "results" / "studies" / study_id
    evidence = study / "evidence"
    if not (study / "winner.json").is_file() or not evidence.is_dir():
        raise FileNotFoundError(f"No se encontraron artefactos completos para {study_id}.")
    # El proyecto separa los activos por tipo: los PNG en figures/ y los cuerpos de tabla —junto
    # con study_macros.tex, que sale del mismo paso— en tables/. Los capítulos viven en chapters/
    # y no los genera este script.
    figures = LATEX / "figures"
    tables = LATEX / "tables"
    for carpeta in (figures, tables):
        carpeta.mkdir(parents=True, exist_ok=True)
    return Paths(study, evidence, figures, tables)


def load_chain(study_ids: list[str]) -> pd.DataFrame:
    """Métricas comparables de la cadena de Model Studies, en orden de ejecución.

    Cada pasada tomó como baseline al ganador de la anterior, así que la fila *i* no es un
    experimento independiente sino el punto de partida de la *i+1*. Todo se lee de los artefactos:
    ninguna cifra de la cadena se escribe a mano.

    ``newey_west_t`` no está en ``summary.json`` —vive en ``attribution.json``, bajo
    ``ic_significance/selection``— y las cifras de la era reservada salen de ``confirmation``, que
    solo existe porque el ganador se reevaluó sobre la serie completa.

    ``attribution.json`` solo existe si la pasada corrió con diagnósticos posteriores al ganador.
    Las pasadas intermedias de una cadena suelen desactivarlos —la era de confirmación se evalúa una
    sola vez y se reserva a la pasada final—, así que su ausencia es el caso normal y no un error:
    las columnas que dependen de él quedan vacías para esa fila.
    """
    rows = []
    for order, study_id in enumerate(study_ids, start=1):
        study = ROOT / "results" / "studies" / study_id
        summary = read_json(study / "evidence" / "summary.json")["summary"]
        attribution_path = study / "attribution.json"
        attribution = read_json(attribution_path) if attribution_path.exists() else {}
        winner = read_json(study / "winner.json")
        meta = read_json(study / "study.json")
        confirmation = summary.get("confirmation", {})
        rows.append(
            {
                "order": order,
                "study_id": study_id,
                "run_id": winner.get("winner_run_id"),
                "catalog_version": meta.get("catalog_version"),
                "mean_rank_ic": summary.get("mean_rank_ic"),
                "ic_ir": summary.get("ic_ir"),
                "newey_west_t": (
                    attribution.get("ic_significance", {}).get("selection", {}).get("newey_west_t")
                ),
                "positive_fraction": summary.get("rank_ic_positive_fraction"),
                "transfer_coefficient": summary.get("transfer_coefficient"),
                "information_ratio": summary.get("information_ratio"),
                "geometric_excess_return": summary.get("geometric_excess_return"),
                "beat_rate": summary.get("beat_rate"),
                "annualized_turnover": summary.get("annualized_turnover"),
                "configuration": winner.get("configuration", {}),
                "confirmation_rank_ic": attribution.get("confirmation_2025_2026", {}).get("mean_rank_ic"),
                "confirmation_excess": confirmation.get("geometric_excess_return"),
                "confirmation_ir": confirmation.get("information_ratio"),
                "confirmation_beat_rate": confirmation.get("beat_rate"),
            }
        )
    return pd.DataFrame(rows)


def chain_configuration_changes(chain: pd.DataFrame) -> list[dict]:
    """Variables en que cada pasada se aparta de la anterior.

    La primera fila se compara contra el baseline recomendado del catálogo, que es de donde partió
    la cadena; las siguientes, contra el ganador que heredaron. El recuento resultante es la
    evidencia de convergencia: si la cadena estuviera explorando al azar no decrecería.
    """
    # El script se invoca desde `latex/scripts/`, así que la raíz del repositorio no está en
    # `sys.path` y el catálogo no sería importable sin añadirla.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from module.studies.catalog import VARIABLES

    baseline = {variable.id: variable.recommended for variable in VARIABLES}
    changes = []
    previous = baseline
    for row in chain.itertuples():
        current = dict(row.configuration)
        differing = sorted(
            key for key, value in current.items()
            if key in previous and previous[key] != value
        )
        changes.append({"order": row.order, "study_id": row.study_id, "variables": differing})
        previous = current
    return changes


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig: plt.Figure, output: Path) -> None:
    fig.tight_layout()
    fig.savefig(output, format="png", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def comma_ticks(*axes: mpl.axis.Axis) -> None:
    """Coma decimal en las marcas de los ejes, como en el resto del documento.

    La mayoría de las figuras del manuscrito tienen marcas enteras y el problema no se ve; en
    cuanto un eje toma valores fraccionarios, matplotlib escribe «0.20» y la anotación de al lado
    «0,20». Es el mismo mojibake tipográfico que el proyecto persigue en el texto.
    """
    formatter = mpl.ticker.FuncFormatter(lambda value, _: f"{value:g}".replace(".", ","))
    for axis in axes:
        axis.set_major_formatter(formatter)


def legend_below(ax: plt.Axes, ncol: int | None = None, anchor: float = -0.16) -> None:
    """Leyenda horizontal centrada debajo del gráfico, para no solaparse con los datos."""
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=ncol or len(labels), loc="upper center", bbox_to_anchor=(0.5, anchor), frameon=False)


def table(path: Path, columns: list[str], rows: list[list[str]], align: str | None = None) -> None:
    alignment = align or ("l" + "r" * (len(columns) - 1))
    lines = [r"\begin{tabular}{" + alignment + "}", r"\toprule"]
    lines += [" & ".join(columns) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def longtable(path: Path, columns: list[str], rows: list[list[str]], caption: str, label: str, align: str | None = None) -> None:
    """Tabla que puede partirse entre páginas.

    A diferencia de ``table``, incorpora su propio caption y label porque
    ``longtable`` no admite ir dentro de un entorno ``table`` flotante.
    """
    alignment = align or ("l" + "r" * (len(columns) - 1))
    header = " & ".join(columns) + r" \\"
    lines = [
        r"\begin{longtable}{" + alignment + "}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{" + str(len(columns)) + r"}{r}{\footnotesize Continúa en la página siguiente}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\end{longtable}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tables_predictive(paths: Paths, diag: pd.DataFrame, features: pd.DataFrame) -> None:
    """Tablas sobre datos, features y capacidad predictiva por agente."""
    matrix = agent_era_matrix(diag)
    rows = [
        [tex(str(name).replace("_", " "))] + [num(matrix.loc[name, era]) for era in matrix.columns]
        for name in matrix.index
    ]
    table(paths.tables / "t06_rankic_era.tex", ["Señal"] + list(matrix.columns), rows)

    # El agregado por bloque se retiró del manuscrito: repetía el argumento de la
    # tabla de extremos —los Rank-IC univariantes son diminutos— y obligaba a tres
    # párrafos de advertencias sobre una asimetría entre catálogo y diagnósticos
    # que no aportaba al hilo. El reparto de bloques por agente se explica en el
    # capítulo de arquitectura.
    ranked = features.dropna(subset=["univariate_rank_ic"]).sort_values("univariate_rank_ic", ascending=False)
    selected = pd.concat([ranked.head(12), ranked.tail(5)])
    rows = []
    for position, row in enumerate(selected.itertuples()):
        if position == 12:
            rows.append([r"\multicolumn{5}{c}{\itshape · · ·}"] + [])
        rows.append(
            [
                tex(row.feature),
                tex(str(row.block).replace("_", " ")),
                tex(row.agents),
                pct(row.coverage, 1),
                num(row.univariate_rank_ic),
            ]
        )
    rows = [row for row in rows if len(row) == 5]
    table(
        paths.tables / "t03_features_top.tex",
        ["Feature", "Bloque", "Agentes", "Cobertura", "Rank-IC univ."],
        rows,
        "lllrr",
    )

# Descripción de cada motivo de exclusión. Se escriben aquí y no se leen de
# `universe_coverage.json` porque ese fichero conserva sus descripciones mal codificadas, y el
# verificador de activos rechaza cualquier mojibake que llegue al manuscrito.
RESOLUTION_LABELS = {
    "symbol_withdrawn": "El proveedor de precios no reconoce el símbolo: lo retiró de su API",
    "missing_price": "Sin serie de precios observable",
    "recycled_ticker": "Símbolo reutilizado después por otra empresa",
    "no_metric_period_match": "Sin fundamentales que casen con el periodo contable",
    "missing_reports": "Sin informes en la fuente de fundamentales",
    "missing_cik": "Sin identificador CIK con el que resolver la empresa",
    "download_failed": "La descarga falló por red o límite de peticiones: reintentable",
    "download_error": "La descarga terminó en error no recuperable",
    "missing_fundamentals": "Sin fundamentales descargados",
}


def write_tables_universe_resolution(paths: Paths) -> None:
    """Reparto del universo histórico por motivo de exclusión.

    Es la tabla que convierte una disculpa en una medición: el panel no calla qué empresas faltan,
    dice por qué falta cada una. Se lee de `data/raw/`, no de un estudio, porque describe el panel y
    no depende de qué modelo se entrene encima.
    """
    source = ROOT / "data" / "raw" / "universe_coverage.json"
    # `errors="replace"` porque el fichero trae descripciones mal codificadas; solo se usan los
    # campos numéricos y el identificador del motivo, que son ASCII.
    resolution = json.loads(source.read_text(encoding="utf-8", errors="replace"))["ticker_resolution"]
    excluded = int(resolution["excluded"])

    reasons = sorted(resolution["by_reason"], key=lambda item: -int(item["tickers"]))
    rows = [
        [
            tex(item["reason"]),
            RESOLUTION_LABELS.get(item["reason"], tex(item["reason"])),
            f"{int(item['tickers']):,}".replace(",", "."),
            pct(int(item["tickers"]) / excluded, 1),
        ]
        for item in reasons
        if int(item["tickers"]) > 0
    ]
    rows.append([
        r"\textbf{Total}",
        r"\textbf{Tickers del universo que no llegan al panel}",
        r"\textbf{" + f"{excluded:,}".replace(",", ".") + "}",
        r"\textbf{100,0\%}",
    ])
    table(
        paths.tables / "t03_resolucion_universo.tex",
        ["Motivo", "Qué significa", "Tickers", "\\% de excluidos"],
        rows,
        "l p{6.8cm} rr",
    )


def write_tables_robustness(
    paths: Paths, summary: dict, robustness: dict, attribution: dict, portfolio: dict | None = None,
) -> None:
    """Tablas de robustez, atribución factorial y ventanas de evaluación.

    ``portfolio`` redirige la tabla de las tres ventanas a la evidencia económica del ganador de la
    rejilla. Sin él, la tabla describiría la cartera del Model Study —la que el catálogo recomienda,
    no la que el documento adopta— y contradiría al resto del capítulo.
    """
    boot = robustness["bootstrap_and_era_exclusion"]
    rows = [
        [
            "Media de selección",
            num(boot["interval_90"]["mean"]),
            f"[{num(boot['interval_90']['ci_low'])}; {num(boot['interval_90']['ci_high'])}]",
            f"[{num(boot['interval_95']['ci_low'])}; {num(boot['interval_95']['ci_high'])}]",
            str(int(boot["interval_95"]["n_cohorts"])),
        ]
    ]
    windows = [("summary", "Selección 2015–2024"), ("confirmation", "Confirmación 2025–2026"), ("full_curve", "Curva completa")]
    metrics = [
        ("cagr_portfolio", "CAGR de cartera", "pct"),
        ("cagr_benchmark", "CAGR de SPY", "pct"),
        ("geometric_excess_return", "Exceso geométrico", "pct"),
        ("information_ratio", "Information ratio", "num3"),
        ("max_drawdown", "Máximo drawdown", "pct"),
        ("beat_rate", "Años por encima de SPY", "pct"),
        ("mean_annual_alpha", "Alfa anual media", "pct"),
        ("worst_year_alpha", "Peor alfa anual", "pct"),
        ("annualized_turnover", "Turnover anualizado", "pct"),
        ("mean_cash_weight", "Efectivo medio", "pct"),
        ("total_cost_drag", "Coste acumulado", "pct"),
        ("n_periods", "Periodos", "int"),
    ]
    # Las tres ventanas son económicas de principio a fin, así que describen la cartera adoptada y
    # no la del Model Study. `evidence_best_full/summary.json` tiene la misma forma que el summary
    # del modelo, de modo que el resto de la función no cambia.
    economic = summary
    if portfolio is not None:
        economic = read_json(portfolio["study"] / "evidence_best_full" / "summary.json")
    root = economic.get("summary", economic)
    blocks = {"summary": root, "confirmation": root["confirmation"], "full_curve": root["full_curve"]}
    rows = []
    for key, label, kind in metrics:
        cells = []
        for window, _ in windows:
            value = blocks[window].get(key)
            if value is None:
                cells.append("—")
            elif kind == "pct":
                cells.append(pct(value))
            elif kind == "num3":
                cells.append(num(value, 3))
            else:
                cells.append(str(int(value)))
        rows.append([label] + cells)
    table(paths.tables / "t07_selec_conf_full.tex", ["Métrica"] + [label for _, label in windows], rows)




def _write_tables_orders_and_tails(paths: Paths, orders: pd.DataFrame, _tails: pd.DataFrame) -> None:
    """Motivos de las órdenes; la cola se representa directamente como figura."""
    flows = orders.copy()
    flows["flow"] = (flows["weight_after"] - flows["weight_before"]).abs()
    grouped = flows.groupby("reason").agg(
        count=("flow", "size"),
        flow=("flow", "sum"),
        commission=("commission_amount", "sum"),
        slippage=("slippage_amount", "sum"),
    ).sort_values("flow", ascending=False)
    total_flow = grouped["flow"].sum()
    rows = [
        [
            tex(str(name)),
            str(int(row["count"])),
            pct(row["flow"] / total_flow, 1),
            num(row["commission"] + row["slippage"], 2),
        ]
        for name, row in grouped.iterrows()
    ]
    rows.append([r"\textbf{Total}", r"\textbf{" + str(int(grouped["count"].sum())) + "}", r"\textbf{100,0\%}", r"\textbf{" + num(grouped["commission"].sum() + grouped["slippage"].sum(), 2) + "}"])
    table(paths.tables / "t07_ordenes_motivo.tex", ["Motivo de la orden", "Órdenes", "\\% del flujo", "Coste"], rows, "lrrr")

def draw_tail_by_era(tails: pd.DataFrame, output: Path) -> None:
    """Resume visualmente si la cola superior cobra por era."""
    frame = tails.copy()
    frame["prediction_date"] = pd.to_datetime(frame["prediction_date"])
    frame["era"] = frame["prediction_date"].dt.year.map(era_of)
    grouped = frame.groupby("era", sort=False).agg(
        top10=("top_10_excess_mean", "mean"),
        top_decile=("top_decile_excess_mean", "mean"),
        spread=("top_minus_bottom", "mean"),
    )
    labels = list(grouped.index)
    positions = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    for offset, column, label, color in (
        (-width, "top10", "Top 10", NAVY),
        (0, "top_decile", "Decil superior", TEAL),
        (width, "spread", "Superior menos inferior", GOLD),
    ):
        ax.bar(positions + offset, 100 * grouped[column], width=width, label=label, color=color)
    ax.axhline(0, color=SLATE, linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set(title="El buen orden no garantiza que la cola superior pague en cada era",
           ylabel="Exceso medio (%)")
    comma_ticks(ax.yaxis)
    legend_below(ax, 3, anchor=-0.18)
    save(fig, output)


GRID_VARIABLES = (
    "target_size", "max_cash_weight", "sizing_mode",
    "minimum_holding_period", "coverage_percentile_floor", "rebalance_drift_tolerance",
)

GRID_LABELS = {
    "target_size": "Posiciones objetivo",
    "max_cash_weight": "Tope de efectivo",
    "sizing_mode": "Reparto de pesos",
    "minimum_holding_period": "Tenencia mínima",
    "coverage_percentile_floor": "Suelo de cobertura",
    "rebalance_drift_tolerance": "Tolerancia de deriva",
}


def load_agent_attribution(paths: "Paths") -> dict[str, pd.DataFrame]:
    """Resume la atribución local por acción en tres agregados pequeños.

    ``agent_local_attribution.parquet`` son 1,3 millones de filas —una por acción, fecha, agente y
    variable— y no cabe manipularlo repetidamente. Se lee **una sola vez**, se agrega aquí y el
    resto del script trabaja con los resúmenes.

    Se devuelven tres cosas distintas: cuánto pesa cada variable dentro de cada agente
    (``by_feature``), cuántas variables distintas llega a encabezar cada agente (``vocabulary``) y
    cómo cambia por año la variable dominante (``by_year``). La magnitud que se agrega es el valor
    absoluto de la contribución: contribuciones de signo opuesto se cancelarían en media y darían la
    impresión falsa de que una variable no interviene.
    """
    frame = pd.read_parquet(
        paths.evidence / "agent_local_attribution.parquet",
        columns=["snapshot_date", "agent", "feature", "local_contribution", "importance_rank"],
    )
    frame["abs_contribution"] = frame["local_contribution"].abs()

    by_feature = frame.groupby(["agent", "feature"], as_index=False).agg(
        mean_abs=("abs_contribution", "mean"),
        mean_signed=("local_contribution", "mean"),
        rows=("abs_contribution", "size"),
    )

    leaders = frame[frame["importance_rank"] == 1]
    vocabulary = leaders.groupby("agent", as_index=False).agg(
        distinct_features=("feature", "nunique"),
        rows=("feature", "size"),
    )

    leaders_by_year = leaders.copy()
    leaders_by_year["year"] = pd.to_datetime(leaders_by_year["snapshot_date"]).dt.year
    by_year = leaders_by_year.groupby(["agent", "year", "feature"], as_index=False).size()

    return {"by_feature": by_feature, "vocabulary": vocabulary, "by_year": by_year}


def draw_attribution_by_year(summary: dict[str, pd.DataFrame], agent: str, output: Path, top: int = 4) -> None:
    """Cuota anual de las variables que encabezan la atribución de un agente.

    Responde a si el agente mira siempre lo mismo. Cada banda es la fracción de acciones y fechas de
    ese año en las que la variable fue la primera por importancia.
    """
    frame = summary["by_year"]
    frame = frame[frame["agent"] == agent]
    pivot = frame.pivot_table(index="year", columns="feature", values="size", aggfunc="sum").fillna(0)
    # La base es el total de observaciones del agente en el año, no la suma de las cuatro variables
    # mostradas. Normalizar sobre el subconjunto inflaba las cuotas y hacía que la figura y el texto
    # hablasen de porcentajes distintos con el mismo nombre.
    totals = pivot.sum(axis=1)
    leaders = pivot.sum().nlargest(top).index
    shares = pivot.loc[:, leaders].div(totals.where(lambda total: total > 0), axis=0).fillna(0)
    shares["otras"] = (1 - shares.sum(axis=1)).clip(lower=0)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    labels = shares.index.astype(str)
    bottom = np.zeros(len(shares))
    for column, color in zip(shares.columns, [NAVY, TEAL, GOLD, RED, GREEN, SLATE]):
        values = shares[column].to_numpy()
        ax.bar(labels, values, bottom=bottom, color=color,
               label=column.replace("factor_", "").replace("_", " "))
        bottom += values
    # La era reservada se marca igual que en el resto del capítulo, para que no se lea como una
    # continuación de la ventana de selección.
    reserved = [position for position, year in enumerate(shares.index) if int(year) >= 2025]
    if reserved:
        separator = min(reserved) - 0.5
        ax.axvline(separator, color=GOLD, linewidth=2.2, zorder=5)
        ax.text(separator + 0.12, 1.02, "reserva", ha="left", va="bottom",
                fontsize=8, color=GOLD, fontweight="bold")
    ax.set(
        title=f"Qué variable encabeza la atribución de {agent}, año a año",
        ylabel="Cuota de las observaciones del año",
        ylim=(0, 1),
    )
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    legend_below(ax, 3, anchor=-0.20)
    save(fig, output)


def write_tables_attribution(paths: Paths, summary: dict[str, pd.DataFrame], top: int = 3) -> None:
    """Las variables principales de cada agente y la amplitud de su vocabulario.

    «Encabeza» es la fracción de observaciones en las que esa variable fue la primera por
    importancia; mide con qué frecuencia manda, frente a la contribución media, que mide cuánto pesa
    cuando interviene. No se tabula el signo medio porque se cancela entre acciones y no es
    interpretable.
    """
    by_feature, vocabulary = summary["by_feature"], summary["vocabulary"]
    leaders = summary["by_year"].groupby(["agent", "feature"], as_index=False)["size"].sum()
    rows = []
    for agent in AGENT_ORDER:
        best = by_feature[by_feature["agent"] == agent].nlargest(top, "mean_abs")
        vocab = vocabulary.loc[vocabulary["agent"] == agent, "distinct_features"]
        agent_leaders = leaders[leaders["agent"] == agent]
        total = agent_leaders["size"].sum()
        for position, row in enumerate(best.itertuples()):
            share = agent_leaders.loc[agent_leaders["feature"] == row.feature, "size"].sum()
            rows.append([
                tex(agent) if position == 0 else "",
                tex(row.feature.replace("factor_", "")),
                num(row.mean_abs),
                pct(share / total, 1) if total else "—",
                str(int(vocab.iloc[0])) if position == 0 and len(vocab) else "",
            ])
    table(
        paths.tables / "t06_atribucion.tex",
        ["Agente", "Variable", "Contrib. media", "Encabeza", "Variables distintas"],
        rows,
        "llrrr",
    )


AGENT_ORDER = ("risk", "value", "growth", "quality", "momentum")


def _grid_sort_key(value: object) -> tuple[int, float, str]:
    """Ordena los valores de una variable de rejilla de forma legible.

    Los valores llegan serializados como JSON, así que ``8`` y ``50`` son cadenas y un orden
    alfabético los colocaría como 12, 16, 25, 5, 50, 8. Los numéricos se ordenan por su valor y los
    categóricos alfabéticamente, detrás.
    """
    text = str(value).strip('"')
    try:
        return (0, float(text), "")
    except ValueError:
        return (1, 0.0, text)


def load_portfolio(study_id: str) -> dict:
    """Artefactos del Portfolio Study: rejilla completa, ganador y perfiles.

    La rejilla se midió sobre la serie recortada en 2024 —ninguna de las combinaciones pudo
    ver la era reservada—, mientras que ``portfolio_winner.json`` guarda además la confirmación del
    ganador sobre la serie completa. Son cifras de ventanas distintas y no deben mezclarse.
    """
    study = ROOT / "results" / "studies" / study_id
    winner_path = study / "portfolio_winner.json"
    grid_path = study / "portfolio_grid.parquet"
    if not winner_path.is_file() or not grid_path.is_file():
        raise FileNotFoundError(f"El Portfolio Study {study_id} no tiene artefactos completos.")
    profiles_path = study / "portfolio_profiles.parquet"
    profile_annual = {}
    if profiles_path.is_file():
        for profile in pd.read_parquet(profiles_path)["profile"].tolist():
            annual_path = study / "profiles" / str(profile) / "annual_metrics.parquet"
            if not annual_path.is_file():
                raise FileNotFoundError(f"Falta la serie anual del perfil {profile}: {annual_path}")
            profile_annual[str(profile)] = pd.read_parquet(annual_path)
    return {
        "study_id": study_id,
        "study": study,
        "grid": pd.read_parquet(grid_path),
        "winner": read_json(winner_path),
        "profiles": pd.read_parquet(profiles_path) if profiles_path.is_file() else None,
        "profile_annual": profile_annual,
        "cost": read_json(study / "cost_sensitivity.json"),
        "capacity": read_json(study / "capacity.json"),
    }


def draw_spiva_horizons(output: Path) -> None:
    """Fondos large-cap que no superan al S&P 500 según el Report 1a de SPIVA."""
    horizons = list(SPIVA_UNDERPERFORMANCE)
    values = list(SPIVA_UNDERPERFORMANCE.values())
    positions = np.arange(len(horizons))
    colors = [TEAL if horizon < 20 else GOLD for horizon in horizons]
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    bars = ax.bar(positions, values, color=colors, width=0.66)
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.2f}%".replace(".", ","),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8, color=NAVY,
        )
    ax.set(
        title="La mayoría de los fondos large-cap no supera al S&P 500",
        ylabel="Fondos por debajo del índice (%)", xlabel="Horizonte de evaluación",
        xticks=positions, ylim=(0, 100),
    )
    ax.set_xticklabels([f"{h} año" if h == 1 else f"{h} años" for h in horizons])
    ax.axhline(50, color=SLATE, linewidth=0.8, linestyle="--")
    ax.grid(axis="x", visible=False)
    save(fig, output)


def draw_universe_coverage(coverage: dict, output: Path) -> None:
    """Cobertura canónica del índice; no mezcla la calidad interna de las filas del panel."""
    frame = pd.DataFrame(coverage["years"])
    frame = frame.loc[frame["year"].between(2003, 2026)].copy()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.axvspan(2014.5, 2024.5, color=TEAL, alpha=0.09, label="Selección 2015–2024")
    ax.axvspan(2024.5, 2026.5, color=GOLD, alpha=0.17, label="Reserva 2025–2026")
    ax.plot(frame["year"], frame["coverage_pct"], color=NAVY, linewidth=2.2, marker="o", markersize=3)
    first, last = frame.iloc[0], frame.iloc[-1]
    for row in (first, last):
        ax.annotate(
            f"{row.coverage_pct:.1f}%".replace(".", ","),
            (row.year, row.coverage_pct), xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=8, fontweight="bold", color=NAVY,
        )
    ax.annotate(
        "Entrenamiento móvil: 8 años anteriores a cada cohorte",
        xy=(2014.0, 91.5), xytext=(2004.0, 94.0),
        arrowprops={"arrowstyle": "->", "color": SLATE, "linewidth": 1,
                    "connectionstyle": "arc3,rad=-0.08"},
        ha="left", va="center", fontsize=7.8, color=SLATE,
    )
    ax.set(
        title="Cobertura real del universo histórico",
        xlabel="Año", ylabel="Miembros del S&P 500 presentes en el panel (%)",
        xlim=(2002.5, 2026.5), ylim=(45, 102),
    )
    ax.set_xticks([2003, 2006, 2009, 2012, 2015, 2018, 2021, 2024, 2026])
    legend_below(ax, 2)
    ax.grid(axis="x", visible=False)
    save(fig, output)


def sample_summary(scores: pd.DataFrame) -> list[dict[str, object]]:
    """Resume la población que recibió una predicción fuera de muestra.

    No usa el panel preparado: la unidad que interesa al lector es una predicción realmente emitida
    por el ganador. La reserva contiene 18 snapshots puntuados, aunque solo seis cohortes tienen ya
    etiqueta cerrada; esa diferencia se conserva y se explica en el capítulo de datos.
    """
    frame = scores.copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
    windows = (
        ("Selección 2015--2024", frame.loc[frame["snapshot_date"].dt.year <= 2024]),
        ("Reserva 2025--2026", frame.loc[frame["snapshot_date"].dt.year >= 2025]),
        ("Total OOS", frame),
    )
    rows: list[dict[str, object]] = []
    for label, subset in windows:
        cohort_sizes = subset.groupby("snapshot_date").size()
        rows.append(
            {
                "window": label,
                "rows": int(len(subset)),
                "snapshots": int(subset["snapshot_date"].nunique()),
                "tickers": int(subset["ticker"].nunique()),
                "cohort_min": int(cohort_sizes.min()),
                "cohort_median": float(cohort_sizes.median()),
                "cohort_max": int(cohort_sizes.max()),
                "training_min": int(subset["training_rows"].min()),
                "training_median": float(
                    subset.groupby("snapshot_date")["training_rows"].first().median()
                ),
                "training_max": int(subset["training_rows"].max()),
            }
        )
    return rows


def draw_oos_sample(scores: pd.DataFrame, features: pd.DataFrame, output: Path) -> None:
    """Tamaño de las secciones transversales y cobertura de variables por bloque."""
    frame = scores.copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
    cohort_sizes = frame.groupby("snapshot_date").size()

    declared = features.loc[
        features["agents"].ne("—") & features["coverage"].notna()
    ].copy()
    declared["block_label"] = declared["block"].map(lambda value: tex_free(value))
    block_order = (
        declared.groupby("block_label")["coverage"].median().sort_values().index.tolist()
    )
    coverage_by_block = [
        100 * declared.loc[declared["block_label"] == block, "coverage"].to_numpy()
        for block in block_order
    ]

    fig, (ax_time, ax_cov) = plt.subplots(
        1, 2, figsize=(7.2, 4.5), gridspec_kw={"width_ratios": [1.12, 1]}
    )
    ax_time.plot(cohort_sizes.index, cohort_sizes.values, color=NAVY, linewidth=1.8)
    reserve_start = pd.Timestamp("2025-01-01")
    ax_time.axvspan(reserve_start, cohort_sizes.index.max(), color=GOLD, alpha=0.15, lw=0)
    ax_time.axvline(reserve_start, color=GOLD, linewidth=1.2, linestyle="--")
    ax_time.set_title("Acciones puntuadas por snapshot")
    ax_time.set_ylabel("Número de acciones")
    ax_time.set_xlabel("Fecha de predicción")
    ax_time.annotate(
        "reserva",
        xy=(pd.Timestamp("2025-03-01"), float(cohort_sizes.loc[cohort_sizes.index >= reserve_start].median())),
        color=GOLD,
        fontsize=8,
    )

    ax_cov.boxplot(
        coverage_by_block,
        vert=False,
        tick_labels=block_order,
        patch_artist=True,
        boxprops={"facecolor": LIGHT, "edgecolor": TEAL},
        medianprops={"color": NAVY, "linewidth": 1.4},
        whiskerprops={"color": TEAL},
        capprops={"color": TEAL},
        flierprops={"marker": ".", "markerfacecolor": RED, "markeredgecolor": RED},
    )
    ax_cov.axvline(80, color=RED, linestyle="--", linewidth=1.0, label="80 %")
    ax_cov.set_xlim(55, 101)
    ax_cov.set_xlabel("Filas con dato disponible (%)")
    ax_cov.set_title("Cobertura de variables por bloque")
    ax_cov.legend(loc="lower right", frameon=False)
    ax_cov.grid(axis="y", visible=False)
    fig.suptitle("Muestra realmente evaluada y disponibilidad de sus predictores", y=1.01)
    fig.tight_layout()
    save(fig, output)


def write_sample_table(paths: Paths, scores: pd.DataFrame) -> None:
    rows = []
    for item in sample_summary(scores):
        cohort = (
            f"{integer(item['cohort_min'])} / "
            f"{num(item['cohort_median'], 1)} / {integer(item['cohort_max'])}"
        )
        training = (
            f"{integer(item['training_min'])} / "
            f"{integer(round(float(item['training_median'])))} / {integer(item['training_max'])}"
        )
        rows.append(
            [
                str(item["window"]), integer(item["rows"]), integer(item["snapshots"]),
                integer(item["tickers"]), cohort, training,
            ]
        )
    table(
        paths.tables / "t03_muestra_oos.tex",
        ["Ventana", "Predicciones", "Snapshots", "Empresas", "Cohorte mín./med./máx.",
         "Filas train mín./med./máx."],
        rows,
        "lrrrrr",
    )


def calibration_state_rows(calibration: pd.DataFrame) -> list[dict[str, object]]:
    frame = calibration.copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
    by_date = frame.sort_values("snapshot_date").groupby("snapshot_date", as_index=False).first()
    labels = {"none": "Sin calibración", "fallback": "Salvaguarda", "era": "Era",
              "horizon": "Horizonte"}
    rows: list[dict[str, object]] = []
    for window, subset in (
        ("Selección 2015--2024", by_date.loc[by_date["snapshot_date"].dt.year <= 2024]),
        ("Reserva 2025--2026", by_date.loc[by_date["snapshot_date"].dt.year >= 2025]),
    ):
        counts = subset["alpha_curve_window"].fillna("none").value_counts()
        total = int(len(subset))
        for key in ("none", "fallback", "era", "horizon"):
            count = int(counts.get(key, 0))
            rows.append(
                {"window": window, "source": labels[key], "count": count,
                 "share": count / total if total else 0.0}
            )
    return rows


def draw_signal_calibration(
    calibration: pd.DataFrame, scores: pd.DataFrame, output: Path,
) -> None:
    """Fuente causal de la curva y traducción observada de percentil a alfa esperado."""
    frame = calibration.copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
    ranks = scores[["ticker", "snapshot_date", "meta_rank"]].copy()
    ranks["snapshot_date"] = pd.to_datetime(ranks["snapshot_date"])
    frame = frame.merge(ranks, on=["ticker", "snapshot_date"], how="left", validate="one_to_one")
    by_date = frame.sort_values("snapshot_date").groupby("snapshot_date", as_index=False).first()
    source_order = ["none", "fallback", "era", "horizon"]
    colors = {"none": SLATE, "fallback": RED, "era": GOLD, "horizon": TEAL}
    labels = {"none": "sin calibración", "fallback": "salvaguarda", "era": "era",
              "horizon": "horizonte"}

    fig, (ax_time, ax_curve) = plt.subplots(2, 1, figsize=(7.2, 5.1), height_ratios=[0.85, 2.15])
    for level, key in enumerate(source_order):
        points = by_date.loc[by_date["alpha_curve_window"].fillna("none") == key]
        ax_time.scatter(points["snapshot_date"], np.full(len(points), level), s=18,
                        color=colors[key], label=labels[key])
    ax_time.axvspan(pd.Timestamp("2025-01-01"), by_date["snapshot_date"].max(),
                    color=GOLD, alpha=0.10, lw=0)
    ax_time.set_yticks(range(len(source_order)), [labels[key] for key in source_order])
    ax_time.set_title("Qué historia pudo usar la calibración en cada fecha")
    ax_time.set_xlabel("")
    ax_time.grid(axis="y", visible=False)

    selection = frame.loc[frame["snapshot_date"].dt.year <= 2024].copy()
    selection["ventile"] = pd.cut(selection["meta_rank"], np.linspace(0, 1, 21), labels=False,
                                   include_lowest=True)
    for key in ("fallback", "era", "horizon"):
        subset = selection.loc[selection["alpha_curve_window"] == key]
        curve = subset.groupby("ventile", observed=True).agg(
            rank=("meta_rank", "median"), alpha=("expected_excess_return", "median")
        ).dropna()
        ax_curve.plot(100 * curve["rank"], 100 * curve["alpha"], marker="o", markersize=3,
                      linewidth=1.7, color=colors[key], label=labels[key])
    ax_curve.axhline(0, color=SLATE, linewidth=0.8)
    ax_curve.set_xlabel("Percentil del ranking final")
    ax_curve.set_ylabel("Exceso anual esperado (%)")
    ax_curve.set_title("Traducción observada del ranking en selección")
    ax_curve.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    save(fig, output)


def write_calibration_table(paths: Paths, calibration: pd.DataFrame) -> None:
    rows = [
        [str(item["window"]), str(item["source"]), integer(item["count"]), pct(item["share"], 1)]
        for item in calibration_state_rows(calibration)
    ]
    table(
        paths.tables / "t05_calibracion_estados.tex",
        ["Ventana", "Fuente de la curva", "Snapshots", "Peso en la ventana"],
        rows,
        "llrr",
    )


def draw_feature_stability(model_attribution: pd.DataFrame, output: Path) -> None:
    """Importancia relativa anual de las tres variables dominantes de cada agente."""
    frame = model_attribution.copy()
    frame["model_retrain_date"] = pd.to_datetime(frame["model_retrain_date"])
    frame["year"] = frame["model_retrain_date"].dt.year
    totals = frame.groupby(["agent", "model_retrain_date"])["coefficient"].transform("sum")
    frame["relative_importance"] = frame["coefficient"].div(totals.replace(0, np.nan))
    annual = frame.groupby(["agent", "feature", "year"], as_index=False)["relative_importance"].mean()
    top = (
        annual.groupby(["agent", "feature"])["relative_importance"].mean()
        .sort_values(ascending=False).groupby(level=0).head(3).reset_index()[["agent", "feature"]]
    )
    selected = annual.merge(top, on=["agent", "feature"], how="inner")
    selected["label"] = selected["agent"] + " · " + selected["feature"].map(tex_free)
    label_order = [
        label for agent in sorted(selected["agent"].unique())
        for label in selected.loc[selected["agent"] == agent]
            .groupby("label")["relative_importance"].mean().sort_values(ascending=False).index
    ]
    years = sorted(selected["year"].unique())
    matrix = selected.pivot(index="label", columns="year", values="relative_importance").reindex(
        index=label_order, columns=years
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.7))
    image = ax.imshow(100 * matrix.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0)
    ax.set_xticks(range(len(years)), years, rotation=45, ha="right")
    ax.set_yticks(range(len(label_order)), label_order)
    ax.set_title("Persistencia de las variables más usadas por cada agente")
    ax.set_xlabel("Año de reentrenamiento")
    ax.grid(False)
    bar = fig.colorbar(image, ax=ax, pad=0.02)
    bar.set_label("Importancia relativa dentro del agente (%)")
    fig.tight_layout()
    save(fig, output)


def draw_factor_attribution(attribution: dict, output: Path) -> None:
    """Cargas de estilo y señal que sobrevive a la neutralización, solo en selección."""
    regression = attribution["factor_regression"]["selection"]
    factors = attribution["factor_regression"]["factors"]
    loadings = np.asarray([regression["loadings"][name] for name in factors], dtype=float)
    errors = np.asarray(regression["standard_errors"][1:], dtype=float)
    neutral = attribution["neutralized_rank_ic"]

    fig, (ax_forest, ax_rank) = plt.subplots(
        1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1.5, 0.8]}
    )
    y = np.arange(len(factors))
    ax_forest.errorbar(loadings, y, xerr=1.96 * errors, fmt="o", color=NAVY, ecolor=TEAL,
                       capsize=3, linewidth=1.4)
    ax_forest.axvline(0, color=SLATE, linewidth=0.9)
    ax_forest.set_yticks(y, [tex_free(name) for name in factors])
    ax_forest.invert_yaxis()
    ax_forest.set_xlabel("Carga estimada (IC 95 % Newey--West)")
    ax_forest.set_title("Exposición a estilos")

    rank_values = [neutral["raw_mean_rank_ic"], neutral["neutralized_mean_rank_ic"]]
    bars = ax_rank.bar(["Bruto", "Neutralizado"], rank_values, color=[NAVY, GOLD], width=0.62)
    ax_rank.set_ylim(0, max(rank_values) * 1.32)
    ax_rank.set_ylabel("Rank-IC medio")
    ax_rank.set_title("Señal conservada")
    for bar, value in zip(bars, rank_values, strict=True):
        ax_rank.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.4f}".replace(".", ","),
                     ha="center", va="bottom", fontsize=8)
    ax_rank.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, output)


def write_feature_dictionary(paths: Paths, features: pd.DataFrame) -> None:
    declared = features.loc[features["origin"].eq("catálogo")].copy()
    declared = declared.sort_values(["agents", "block", "feature"])
    source_labels = {
        "finnhub_series": "Finnhub PIT", "price_series": "Precios PIT",
        "derived": "Derivada", "finnhub_metric": "Finnhub PIT",
    }
    rows = []
    for row in declared.itertuples():
        direction = "+" if float(row.direction or 0) > 0 else "−" if float(row.direction or 0) < 0 else "—"
        rows.append([
            tex(row.feature), tex(row.agents), tex(str(row.block).replace("_", " ")),
            tex(source_labels.get(str(row.source), tex_free(row.source))), direction,
            pct(row.coverage, 1), num(row.univariate_rank_ic),
        ])
    longtable(
        paths.tables / "tB_diccionario_features.tex",
        ["Variable", "Agente", "Bloque", "Fuente", "Dir.", "Cobertura", "Rank-IC"],
        rows,
        "Diccionario completo de las 68 variables declaradas.",
        "tab:diccionario-features",
        r"p{5.2cm}p{2.4cm}p{2.6cm}p{1.8cm}rrr",
    )


def draw_alpha_turnover_annual(annual: pd.DataFrame, output: Path) -> None:
    """Relación anual entre rotación y exceso, con la reserva fuera de selección."""
    frame = annual.copy()
    reserved = frame["year"].astype(int) >= 2025
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.scatter(frame.loc[~reserved, "turnover"], 100 * frame.loc[~reserved, "alpha"],
               color=TEAL, s=48, label="Selección 2015–2024", zorder=3)
    ax.scatter(frame.loc[reserved, "turnover"], 100 * frame.loc[reserved, "alpha"],
               color=GOLD, edgecolor=NAVY, linewidth=0.7, s=65,
               label="Reserva 2025–2026", zorder=4)
    label_offsets = {2017: (4, 10), 2019: (10, 4), 2020: (4, 7), 2022: (12, 4), 2025: (12, 9)}
    for row in frame.itertuples():
        ax.annotate(str(int(row.year)), (row.turnover, 100 * row.alpha),
                    xytext=label_offsets.get(int(row.year), (4, 4)),
                    textcoords="offset points", fontsize=7.4, color=NAVY)
    ax.axhline(0, color=SLATE, linewidth=0.9)
    ax.set(
        title="Rotar más no garantiza un mayor exceso anual",
        xlabel="Turnover anual (veces la cartera)", ylabel="Exceso anual frente a SPY (%)",
    )
    comma_ticks(ax.xaxis, ax.yaxis)
    legend_below(ax, 2)
    save(fig, output)


def draw_cost_ladder(cost: dict, output: Path) -> None:
    """Exceso geométrico de selección al variar el coste por operación."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    max_cost = float(max(cost["ladder_bps"]))
    styles = (("frozen_path", "Ruta congelada", NAVY, "o"),
              ("resimulated", "Cartera resimulada", TEAL, "s"))
    for key, label, color, marker in styles:
        frame = pd.DataFrame(cost[key])
        ax.plot(frame["cost_bps"], 100 * frame["selection_geometric_excess_return"],
                color=color, marker=marker, markersize=3.8, linewidth=1.8, label=label)
        point = cost["break_even"][key]["selection"]
        if point.get("available"):
            x = float(point["bps_per_trade"])
            ax.scatter([x], [0], color=color, s=35, zorder=5)
            # Una etiqueta por encima y otra por debajo del eje, alineadas hacia dentro cuando el
            # equilibrio cae cerca del extremo derecho: si no, «Equilibrio 447 pb» se sale del lienzo.
            dy = 26 if key == "resimulated" else -30
            ha = "right" if x > 0.78 * max_cost else "center"
            ax.annotate(f"Equilibrio {x:.0f} pb", (x, 0), xytext=(0, dy),
                        textcoords="offset points", ha=ha, fontsize=7.3, color=color,
                        bbox=dict(facecolor="white", edgecolor=color, linewidth=0.6,
                                  boxstyle="round,pad=0.25", alpha=0.92))
    ax.set_xlim(-max_cost * 0.04, max_cost * 1.06)
    adopted = float(cost["adopted_cost_bps"])
    ax.axvline(adopted, color=GOLD, linewidth=1.3, linestyle="--", label=f"Coste adoptado: {adopted:.0f} pb")
    ax.axhline(0, color=SLATE, linewidth=0.9)
    ax.set(
        title="Margen frente a costes en la ventana de selección",
        xlabel="Coste por operación (puntos básicos)", ylabel="Exceso geométrico frente a SPY (%)",
    )
    comma_ticks(ax.xaxis, ax.yaxis)
    legend_below(ax, 3, anchor=-0.20)
    save(fig, output)


def draw_capacity(capacity: dict, output: Path) -> None:
    """Capacidad en selección: participación P95 frente al patrimonio simulado."""
    window = capacity["windows"]["selection"]
    frame = pd.DataFrame(window["ladder"])
    frame = frame.loc[frame["aum_usd"] <= 1e8].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(frame["aum_usd"], 100 * frame["p95_participation"], color=NAVY,
            marker="o", markersize=4, linewidth=2)
    for threshold, color in ((5, TEAL), (10, GOLD)):
        aum = float(window["maximum_aum_usd"][f"{threshold}%"])
        ax.axhline(threshold, color=color, linestyle="--", linewidth=1)
        ax.scatter([aum], [threshold], color=color, s=45, zorder=4)
        offset = (8, -13) if threshold == 5 else (8, 7)
        ax.annotate(f"{threshold}% → {aum / 1e6:.1f} M USD".replace(".", ","),
                    (aum, threshold), xytext=offset, textcoords="offset points",
                    fontsize=7.6, color=color)
    ax.set_xscale("log")
    ax.set(
        title="Capacidad estimada de la cartera en selección",
        xlabel="Patrimonio de la cartera (USD, escala logarítmica)",
        ylabel="Participación P95 sobre volumen diario (%)",
        ylim=(-0.8, 27),
    )
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda value, _: f"{value / 1e9:g}B" if value >= 1e9 else f"{value / 1e6:g}M"
    ))
    comma_ticks(ax.yaxis)
    ax.text(0.01, 0.96, "Selección: 143 órdenes con volumen · reserva excluida (10 órdenes)",
            transform=ax.transAxes, va="top", fontsize=7.2, color=SLATE)
    save(fig, output)


def draw_profile_weights(output: Path) -> None:
    """Matriz firmada de pesos de los perfiles; balanced conserva el meta puro."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from module.evaluation.profiles import PROFILE_WEIGHTS

    columns = ("meta", "quality", "value", "growth", "momentum", "risk")
    profiles = tuple(PROFILE_WEIGHTS)
    values = np.array([
        [PROFILE_WEIGHTS[profile].get(f"{column}_rank", 0.0) for column in columns]
        for profile in profiles
    ])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    norm = mpl.colors.TwoSlopeNorm(vmin=-0.75, vcenter=0, vmax=1.0)
    image = ax.imshow(values, cmap=mpl.colors.LinearSegmentedColormap.from_list(
        "signed", [RED, "#F7F7F7", TEAL]), norm=norm, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = "—" if value == 0 else f"{value:+.0%}".replace("+", "")
            ax.text(column, row, label, ha="center", va="center", fontsize=8,
                    color="white" if abs(value) >= 0.45 else NAVY)
    ax.set(
        title="Cómo reordena cada perfil la señal aprendida",
        xticks=np.arange(len(columns)), yticks=np.arange(len(profiles)),
    )
    ax.set_xticklabels(columns)
    ax.set_yticklabels(profiles)
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Peso firmado")
    colorbar.ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    save(fig, output)


def draw_profile_results(profiles: pd.DataFrame, annual_by_profile: dict[str, pd.DataFrame], output: Path) -> None:
    """Resultado agregado y exceso anual de los ocho perfiles, sin convertirlo en selector."""
    order = profiles.sort_values("geometric_excess_return", ascending=False)["profile"].tolist()
    ordered = profiles.set_index("profile").loc[order]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 4.0), gridspec_kw={"width_ratios": [0.9, 1.6]})
    positions = np.arange(len(order))
    left.barh(positions + 0.18, 100 * ordered["geometric_excess_return"], height=0.34,
              color=TEAL, label="Selección")
    left.barh(positions - 0.18, 100 * ordered["confirmation_excess"], height=0.34,
              color=GOLD, label="Reserva")
    left.axvline(0, color=SLATE, linewidth=0.8)
    left.set(yticks=positions, xlabel="Exceso geométrico (%)", title="Resultado agregado")
    left.set_yticklabels(order)
    left.invert_yaxis()
    left.legend(frameon=False, loc="upper left", fontsize=7.2)
    comma_ticks(left.xaxis)

    years = sorted({int(year) for frame in annual_by_profile.values() for year in frame["year"]})
    matrix = np.array([
        [100 * float(annual_by_profile[profile].set_index("year").loc[year, "alpha"]) for year in years]
        for profile in order
    ])
    limit = max(5, float(np.nanpercentile(np.abs(matrix), 95)))
    image = right.imshow(matrix, cmap=mpl.colors.LinearSegmentedColormap.from_list(
        "alpha", [RED, "#F7F7F7", TEAL]), vmin=-limit, vmax=limit, aspect="auto")
    right.set(title="Exceso anual frente a SPY", xticks=np.arange(len(years)), yticks=np.arange(len(order)))
    right.set_xticklabels(years, rotation=45, ha="right", fontsize=7)
    right.set_yticklabels(order, fontsize=7)
    separator = years.index(2025) - 0.5
    right.axvline(separator, color=GOLD, linewidth=2)
    right.text(separator + 0.15, -0.7, "reserva", color=GOLD, fontsize=7.2, fontweight="bold")
    right.grid(False)
    colorbar = fig.colorbar(image, ax=right, fraction=0.05, pad=0.03)
    colorbar.set_label("Exceso anual (%)")
    comma_ticks(colorbar.ax.yaxis)
    fig.suptitle("Perfiles informativos: sensibilidad de la misma señal, no un ranking de estilos", fontsize=11)
    fig.text(0.5, 0.01, "La reserva contiene seis cohortes; los perfiles no participaron en la selección.",
             ha="center", fontsize=7.3, color=SLATE)
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    fig.savefig(output, format="png", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def draw_portfolio_grid(grid: pd.DataFrame, winner: dict, output: Path) -> None:
    """Nube de las combinaciones evaluadas: Information Ratio frente a rotación anual.

    Sustituye al barrido diagnóstico de una variable cada vez. Cada punto es una cartera completa
    —seis coordenadas simultáneas— y el color codifica el tope de efectivo, que es la variable que
    más separa la nube. La ganadora se marca aparte: interesa dónde cae respecto de la masa, no su
    valor aislado.
    """
    frame = grid.dropna(subset=["information_ratio", "annualized_turnover"]).copy()
    cash = frame["max_cash_weight"].astype(float)
    levels = sorted(cash.unique())
    palette = [NAVY, TEAL, GOLD, RED, GREEN][: len(levels)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for level, color in zip(levels, palette):
        mask = cash == level
        # Etiqueta para matplotlib, no para LaTeX: `pct` escaparía el símbolo de porcentaje.
        ax.scatter(
            frame.loc[mask, "annualized_turnover"], frame.loc[mask, "information_ratio"],
            s=9, alpha=0.55, color=color, linewidths=0, label=f"Efectivo máx. {level:.0%}",
        )
    best = winner["winner_summary"]
    ax.scatter(
        [best["annualized_turnover"]], [best["information_ratio"]],
        s=110, facecolor="none", edgecolor="black", linewidth=1.6, zorder=5, label="Cartera ganadora",
    )
    ax.set(
        title=f"Las {len(frame):,} carteras evaluadas: Information Ratio frente a rotación".replace(",", "."),
        xlabel="Rotación anualizada (veces la cartera al año)", ylabel="Information Ratio (2015-2024)",
    )
    legend_below(ax, min(len(levels) + 1, 4), anchor=-0.20)
    save(fig, output)


def portfolio_influence_rows(grid: pd.DataFrame) -> list[dict]:
    """Efecto marginal ordenado de las seis reglas de cartera."""
    frame = grid.dropna(subset=["information_ratio"])
    rows = []
    for variable in GRID_VARIABLES:
        medians = frame.groupby(frame[variable].astype(str))["information_ratio"].median().sort_values()
        if medians.empty:
            continue
        rows.append(
            {
                "variable": variable,
                "worst_value": medians.index[0],
                "worst": float(medians.iloc[0]),
                "best_value": medians.index[-1],
                "best": float(medians.iloc[-1]),
                "spread": float(medians.iloc[-1] - medians.iloc[0]),
            }
        )
    return sorted(rows, key=lambda item: item["spread"], reverse=True)


def draw_portfolio_influence(grid: pd.DataFrame, output: Path) -> None:
    """Rango de IR mediano de cada regla, más legible que seis boxplots pequeños."""
    rows = list(reversed(portfolio_influence_rows(grid)))
    positions = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for position, row in zip(positions, rows):
        ax.plot([row["worst"], row["best"]], [position, position], color=SLATE, linewidth=3)
        ax.scatter(row["worst"], position, color=RED, s=36, zorder=3)
        ax.scatter(row["best"], position, color=TEAL, s=36, zorder=3)
        ax.annotate(num(row["spread"], 3), (row["best"], position), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=8, color=NAVY)
    ax.set_yticks(positions)
    ax.set_yticklabels([GRID_LABELS[row["variable"]] for row in rows])
    ax.set(
        title="Qué reglas de cartera mueven el resultado",
        xlabel="Information Ratio mediano según el valor elegido",
    )
    ax.legend(
        handles=[
            mpl.lines.Line2D([], [], marker="o", linestyle="", color=RED, label="Peor valor"),
            mpl.lines.Line2D([], [], marker="o", linestyle="", color=TEAL, label="Mejor valor"),
        ],
        frameon=False,
        loc="lower right",
    )
    comma_ticks(ax.xaxis)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.16, y=0.16)
    save(fig, output)


def draw_selection_vs_reserved(chain: pd.DataFrame, output: Path) -> None:
    """El hallazgo central: la ventana de selección mejora y la era reservada se hunde.

    Enfrenta, pasada a pasada, el Information Ratio medido en la ventana donde se tomaron las
    decisiones contra el de la era reservada. La divergencia es el resultado: la tercera pasada es
    la mejor de la cadena en selección (0,339) y la peor con diferencia fuera de ella (-1,167).
    """
    labels = [f"Study {row.order}" for row in chain.itertuples()]
    selection = chain["information_ratio"].astype(float).tolist()
    reserved = chain["confirmation_ir"].astype(float).tolist()
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(positions - 0.2, selection, width=0.38, color=NAVY, label="Ventana de selección (2015-2024)")
    ax.bar(positions + 0.2, reserved, width=0.38, color=RED, label="Era reservada (2025-2026)")
    for index, (left, right) in enumerate(zip(selection, reserved)):
        ax.annotate(num(left, 3), (index - 0.2, left), xytext=(0, 4 if left >= 0 else -11), textcoords="offset points", ha="center", fontsize=7.5)
        ax.annotate(num(right, 3), (index + 0.2, right), xytext=(0, 4 if right >= 0 else -11), textcoords="offset points", ha="center", fontsize=7.5)
    ax.axhline(0, color=SLATE, linewidth=0.9)
    ax.set(title="Information Ratio dentro y fuera de la ventana de selección", ylabel="Information Ratio", xticks=positions)
    ax.set_xticklabels(labels)
    ax.margins(y=0.22)
    legend_below(ax, 2)
    save(fig, output)


def write_tables_chain(paths: Paths, chain: pd.DataFrame, changes: list[dict]) -> None:
    """Tabla comparativa de la cadena y tabla de qué cambió en cada pasada."""
    rows = []
    for row in chain.itertuples():
        rows.append([
            f"Study {row.order}",
            tex(row.run_id),
            num(row.mean_rank_ic),
            num(row.ic_ir, 3),
            num(row.newey_west_t, 2),
            num(row.transfer_coefficient, 3),
            num(row.information_ratio, 3),
            pct(row.geometric_excess_return),
            num(row.confirmation_ir, 3),
        ])
    table(
        paths.tables / "t06_cadena.tex",
        ["Pasada", "Run ganador", "Rank-IC", "IC-IR", "$t$ NW", "Transf.", "IR sel.", "Exceso sel.", "IR reserv."],
        rows,
        "llrrrrrrr",
    )

    change_rows = [
        [
            f"Study {item['order']}",
            str(len(item["variables"])),
            tex(", ".join(item["variables"])) if item["variables"] else "—",
        ]
        for item in changes
    ]
    table(
        paths.tables / "t06_cadena_config.tex",
        ["Pasada", "Cambios", "Variables modificadas respecto del punto de partida"],
        change_rows,
        "lrp{0.62\\linewidth}",
    )


# --- Relato de la cartera ---------------------------------------------------------------------
# `portfolio_narrative.json` responde a una pregunta que la curva de patrimonio no contesta: qué
# acciones tuvo la cartera, cuánto tiempo y con qué resultado. Es material descriptivo y posterior a
# la congelación del ganador —ninguna de estas cifras participó en ninguna selección—, y por eso el
# propio artefacto viaja con sus salvedades en `caveats`.

# Cuántos nombres entran en cada figura. Quince caben en el JSON; diez se leen en A4.
NARRATIVE_TOP_N = 10


def load_portfolio_narrative(study_id: str) -> dict | None:
    """Relato de la cartera ganadora, si el Portfolio Study llegó a producirlo.

    Devuelve ``None`` en vez de fallar: el bloque narrativo es un diagnóstico opcional y una pasada
    que no lo generó sigue siendo una pasada válida. Sus figuras simplemente no se emiten.
    """
    path = ROOT / "results" / "studies" / study_id / "portfolio_narrative.json"
    if not path.is_file():
        return None
    payload = read_json(path)
    windows = payload.get("windows", {})
    return payload if windows.get("selection", {}).get("available") else None


def draw_portfolio_narrative(narrative: dict, output: Path) -> None:
    """Una sola figura cuenta concentración, episodios y exposición descriptiva."""
    window = narrative["windows"]["selection"]
    confirmation = narrative["windows"]["confirmation"]
    contributors = sorted(window["best_contributors"][:6], key=lambda row: row["net_contribution"])
    trips = sorted(
        window["best_round_trips"][:3] + window["worst_round_trips"][:3],
        key=lambda row: row["entry_date"],
        reverse=True,
    )
    selection_sector = {
        row["sector"]: row["mean_portfolio_weight"]
        for row in window["sector_exposure"]["sectors"]
    }
    confirmation_sector = {
        row["sector"]: row["mean_portfolio_weight"]
        for row in confirmation["sector_exposure"]["sectors"]
    }
    # Los sectores se ordenan por el mayor de los dos pesos, no solo por el de selección: con el
    # criterio anterior «Media», que es el primer sector de la era reservada con un 33,9 %, no
    # llegaba a aparecer en la figura que el texto cita para justamente esa afirmación.
    weight = {
        name: max(selection_sector.get(name, 0.0), confirmation_sector.get(name, 0.0))
        for name in set(selection_sector) | set(confirmation_sector)
    }
    sectors = sorted(weight, key=weight.get, reverse=True)[:6]
    sectors.reverse()

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.7))
    ax = axes[0]
    values = [100 * row["net_contribution"] for row in contributors]
    ax.barh(np.arange(len(contributors)), values, color=NAVY)
    ax.set_yticks(np.arange(len(contributors)))
    ax.set_yticklabels([row["ticker"] for row in contributors])
    ax.set_title("Contribución concentrada")
    ax.set_xlabel("puntos porcentuales")
    comma_ticks(ax.xaxis)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    positions = np.arange(len(trips))
    for position, row in zip(positions, trips):
        start = np.datetime64(row["entry_date"])
        end = np.datetime64(row["snapshot_date"])
        gain = row["realized_pnl_pct"] >= 0
        ax.barh(position, (end - start).astype("timedelta64[D]").astype(int), left=start,
                color=GREEN if gain else RED, height=0.58)
    ax.set_yticks(positions)
    ax.set_yticklabels([row["ticker"] for row in trips])
    ax.set_title("Episodios extremos")
    # Una marca cada dos años: con una por año las etiquetas se solapaban hasta ser ilegibles.
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(axis="y", visible=False)

    ax = axes[2]
    positions = np.arange(len(sectors))
    ax.barh(positions - 0.18, [100 * selection_sector.get(s, 0) for s in sectors],
            height=0.34, color=NAVY, label="Selección")
    ax.barh(positions + 0.18, [100 * confirmation_sector.get(s, 0) for s in sectors],
            height=0.34, color=GOLD, label="Reservada")
    ax.set_yticks(positions)
    ax.set_yticklabels(sectors, fontsize=7)
    ax.set_title("Exposición sectorial*")
    ax.set_xlabel("peso medio (%)")
    comma_ticks(ax.xaxis)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", visible=False)
    fig.suptitle("Qué compró la cartera y dónde se concentró el resultado", fontsize=11)
    save(fig, output)


def write_tables_portfolio_narrative(paths: Paths, narrative: dict) -> None:
    """Las acciones más presentes en la cartera, con permanencia, peso y contribución.

    Se reportan los episodios además de los meses porque son cosas distintas: dos episodios
    separados por años significan que el sistema vendió y volvió a comprar el mismo nombre, que es
    justamente lo que una cartera con tenencia mínima puede hacer y una regla estática no.
    """
    window = narrative["windows"]["selection"]
    rows = [
        [
            tex(row["ticker"]),
            str(int(row["months_held_total"])),
            str(int(row["episodes"])),
            pct(row["mean_weight"], 1),
            pct(row["max_weight"], 1),
            # Dos decimales: con uno solo, una contribución nula se imprimiría como «-0,0 %».
            pct(row["net_contribution"], 2),
        ]
        for row in window["most_held"][:NARRATIVE_TOP_N]
    ]
    table(
        paths.tables / "t07_cartera_relato.tex",
        ["Acción", "Meses", "Episodios", "Peso medio", "Peso máx.", "Contrib. neta"],
        rows,
        "lrrrrr",
    )


def write_tables_portfolio_study(paths: Paths, portfolio: dict, baseline: dict) -> None:
    """Tabla compacta de la cartera ganadora frente a la política del modelo.

    ``baseline`` es el resumen del ganador del Model Study, es decir, la cartera que el manuscrito
    documentaba antes de optimizar. Aparece como fila de referencia para que la mejora se lea como
    diferencia y no como cifra suelta.

    """
    winner = portfolio["winner"]
    summary = winner["winner_summary"]
    confirmation = winner.get("winner_confirmation", {})

    def _delta(current: float | None, reference: float | None, formatter, digits: int) -> str:
        """Diferencia con signo, en el mismo formato que las dos columnas que compara."""
        if current is None or reference is None:
            return "—"
        return formatter(float(current) - float(reference), digits)

    metric_rows = [
        [
            "Information Ratio",
            num(baseline["information_ratio"], 3),
            num(summary["information_ratio"], 3),
            _delta(summary["information_ratio"], baseline["information_ratio"], num, 3),
            num(confirmation.get("information_ratio"), 3) if confirmation.get("information_ratio") is not None else "—",
        ],
        [
            "Exceso geométrico",
            pct(baseline["geometric_excess_return"]),
            pct(summary["geometric_excess_return"]),
            _delta(summary["geometric_excess_return"], baseline["geometric_excess_return"], pct, 2),
            pct(confirmation["geometric_excess_return"]) if confirmation.get("geometric_excess_return") is not None else "—",
        ],
        [
            "Rotación anualizada",
            num(baseline["annualized_turnover"], 2),
            num(summary["annualized_turnover"], 2),
            _delta(summary["annualized_turnover"], baseline["annualized_turnover"], num, 2),
            num(confirmation["annualized_turnover"], 2) if confirmation.get("annualized_turnover") is not None else "—",
        ],
        [
            "Máxima caída",
            pct(baseline["max_drawdown"]),
            pct(summary["max_drawdown"]),
            _delta(summary["max_drawdown"], baseline["max_drawdown"], pct, 2),
            pct(confirmation["max_drawdown"]) if confirmation.get("max_drawdown") is not None else "—",
        ],
        [
            "Años que baten",
            pct(baseline["beat_rate"], 0),
            pct(summary["beat_rate"], 0),
            _delta(summary["beat_rate"], baseline["beat_rate"], pct, 0),
            pct(confirmation["beat_rate"], 0) if confirmation.get("beat_rate") is not None else "—",
        ],
    ]
    table(
        paths.tables / "t07_cartera_ganadora.tex",
        ["Métrica", "Cartera del modelo", "Cartera ganadora", "Diferencia", "Era reservada"],
        metric_rows,
        "lrrrr",
    )

def write_tables_catalog(paths: Paths, catalog: dict, winner: dict, decisions: dict) -> None:
    """Catálogo cerrado completo y escalera de decisiones secuenciales."""
    configuration = winner["configuration"]
    stage_labels = {stage["id"]: stage["label"] for stage in catalog["stages"]}
    rows = []
    for variable in sorted(catalog["variables"], key=lambda item: (item["order"], item["id"])):
        values = ", ".join(tex_free(value) for value in variable["values"])
        rows.append(
            [
                tex(variable["id"]),
                tex(stage_labels.get(variable["stage"], variable["stage"])),
                "Sí" if variable["predictive"] else "No",
                tex(values),
                tex(tex_free(configuration.get(variable["id"], "—"))),
            ]
        )
    longtable(
        paths.tables / "tB_catalogo.tex",
        ["Variable", "Etapa", "Predictiva", "Valores del catálogo", "Ganador"],
        rows,
        "Catálogo cerrado completo: las 33 variables, su rejilla y el valor del ganador.",
        "tab:catalogo",
        # La tabla se incluye dentro de un entorno landscape, donde \textwidth
        # conserva el valor del formato vertical: hay que usar \linewidth para
        # aprovechar el ancho real de la página girada.
        "lllp{0.34\\linewidth}l",
    )


def write_tables_winner(paths: Paths, catalog: dict, winner: dict) -> None:
    """La configuración predictiva ganadora, entera y en una sola tabla.

    El anexo del catálogo ya la contiene, pero mezclada con la rejilla completa y en una página
    apaisada: el lector del capítulo de resultados tenía que saltar al final del documento para
    saber con qué configuración se obtuvieron las cifras que está leyendo. Aquí van solo las
    veintiuna variables predictivas y su valor final, en dos bloques para que quepan sin partir la
    página. Las de cartera se omiten a propósito: en esta fase conservaban el valor por defecto y no
    son las que el trabajo adopta.
    """
    configuration = winner["configuration"]
    stage_labels = {stage["id"]: stage["label"] for stage in catalog["stages"]}
    # Se ordena por etapa y no por el campo `order` global: el protocolo se recorre por fases, y
    # agrupar las variables como se decidieron es lo que permite leer la tabla de un vistazo.
    stage_rank = {stage: index for index, stage in enumerate(catalog["stage_order"])}
    predictive = sorted(
        (variable for variable in catalog["variables"] if variable["predictive"]),
        key=lambda item: (stage_rank.get(item["stage"], 99), item["order"], item["id"]),
    )
    entries = [
        (
            stage_labels.get(variable["stage"], variable["stage"]),
            tex(variable["id"]),
            tex(tex_free(configuration.get(variable["id"], "—"))),
        )
        for variable in predictive
    ]
    rows = []
    for index, (stage, name, value) in enumerate(entries):
        previous = entries[index - 1][0] if index else None
        rows.append([stage if stage != previous else "", name, value])
    table(
        paths.tables / "t06_ganador.tex",
        ["Etapa del protocolo", "Variable", "Valor del ganador"],
        rows,
        "lll",
    )


def draw_rank_ic(summary: dict, output: Path) -> None:
    frame = pd.DataFrame(summary["rank_ic_by_cohort"])
    frame["date"] = pd.to_datetime(frame["date"])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(frame["date"], frame["rank_ic"], color=SLATE, linewidth=0.8, alpha=0.72, label="Cohorte")
    ax.plot(frame["date"], frame["rank_ic"].rolling(12, min_periods=4).mean(), color=NAVY, linewidth=2, label="Media móvil de 12 cohortes")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(summary["summary"]["mean_rank_ic"], color=GOLD, linewidth=1.3, linestyle="--", label="Media de selección")
    ax.axvspan(pd.Timestamp("2025-01-01"), frame["date"].max(), color=GOLD, alpha=0.13, label="Era reservada")
    ax.set(title="Rank-IC por cohorte", ylabel="Rank-IC")
    legend_below(ax, ncol=4)
    save(fig, output)


def draw_order_vs_payoff(summary: dict, output: Path) -> None:
    """Rank-IC frente a Information Ratio, era a era: el orden mejora mientras el pago empeora.

    Es el hallazgo que gobierna la lectura del trabajo, y separarlo en dos cifras sueltas es
    justamente lo que lo haría invisible. Las dos series miden cosas distintas —la calidad del
    orden transversal y el pago ajustado por riesgo de operarlo con una cartera concreta— y por eso
    pueden divergir sin contradecirse.
    """
    eras = summary["eras"]
    labels = [row["era"] for row in eras]
    positions = np.arange(len(eras))

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(positions, [row["rank_ic"] for row in eras], width=0.5, color=NAVY, label="Rank-IC del meta")
    ax.set(ylabel="Rank-IC medio", xlabel="")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(row["rank_ic"] for row in eras) * 1.35)
    for position, row in zip(positions, eras):
        ax.annotate(
            num(row["rank_ic"]), (position, row["rank_ic"]), xytext=(0, 4),
            textcoords="offset points", ha="center", fontsize=7.5, color=NAVY,
        )

    twin = ax.twinx()
    twin.grid(visible=False)
    twin.spines["right"].set_visible(True)
    twin.plot(
        positions, [row["information_ratio"] for row in eras],
        color=RED, linewidth=2, marker="o", markersize=5, label="Information Ratio de la cartera",
    )
    twin.axhline(0, color=RED, linewidth=0.8, linestyle=":")
    twin.set_ylabel("Information Ratio")
    for position, row in zip(positions, eras):
        twin.annotate(
            num(row["information_ratio"], 3), (position, row["information_ratio"]), xytext=(0, 8),
            textcoords="offset points", ha="center", fontsize=7.5, color=RED,
        )

    comma_ticks(ax.yaxis, twin.yaxis)
    ax.set_title("El orden mejora mientras el pago empeora")
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels_all = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels_all, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.10), frameon=False)
    save(fig, output)


def draw_equity_and_drawdown(equity: pd.DataFrame, output: Path) -> None:
    """Patrimonio y riesgo en dos paneles alineados sobre el mismo calendario."""
    frame = equity.copy()
    date_column = "date" if "date" in frame.columns else "snapshot_date"
    frame[date_column] = pd.to_datetime(frame[date_column])
    portfolio_column = "portfolio_value" if "portfolio_value" in frame.columns else "portfolio_equity"
    benchmark_column = "benchmark_value" if "benchmark_value" in frame.columns else "benchmark_equity"
    portfolio = frame[portfolio_column].astype(float)
    benchmark = frame[benchmark_column].astype(float)
    portfolio_dd = portfolio / portfolio.cummax() - 1
    benchmark_dd = benchmark / benchmark.cummax() - 1

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.45, 1]})
    axes[0].plot(frame[date_column], portfolio, color=NAVY, linewidth=1.7, label="Cartera")
    axes[0].plot(frame[date_column], benchmark, color=SLATE, linewidth=1.3, label="SPY")
    axes[0].axvspan(pd.Timestamp("2025-01-01"), frame[date_column].max(), color=GOLD, alpha=0.13,
                    label="Era reservada")
    axes[0].set(title="La cartera elegida: crecimiento y caídas", ylabel="Patrimonio acumulado")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].fill_between(frame[date_column], 100 * portfolio_dd, 0, color=NAVY, alpha=0.35,
                         label="Cartera")
    axes[1].plot(frame[date_column], 100 * benchmark_dd, color=SLATE, linewidth=1.0, label="SPY")
    axes[1].axvspan(pd.Timestamp("2025-01-01"), frame[date_column].max(), color=GOLD, alpha=0.13)
    axes[1].set(ylabel="Drawdown (%)", xlabel="Fecha")
    comma_ticks(axes[1].yaxis)
    axes[1].legend(frameon=False, ncol=2)
    save(fig, output)


def era_of(year: int) -> str:
    """Etiqueta de era. Coincide con SELECTION_ERAS más la era reservada."""
    if year <= 2018:
        return "2015-2018"
    if year <= 2021:
        return "2019-2021"
    if year <= 2024:
        return "2022-2024"
    return "2025-2026"


def agent_era_matrix(diag: pd.DataFrame) -> pd.DataFrame:
    """Rank-IC medio por agente y era, ordenado por la media de selección."""
    frame = diag.copy()
    frame["prediction_date"] = pd.to_datetime(frame["prediction_date"])
    frame["era"] = frame["prediction_date"].dt.year.map(era_of)
    matrix = frame.pivot_table(index="agent", columns="era", values="rank_ic", aggfunc="mean")
    selection_eras = ["2015-2018", "2019-2021", "2022-2024"]
    order = matrix[selection_eras].mean(axis=1).sort_values(ascending=False).index
    return matrix.loc[order]


def draw_meta_weights_annual(weights: pd.DataFrame, output: Path) -> None:
    frame = weights.copy()
    frame["year"] = pd.to_datetime(frame["snapshot_date"]).dt.year
    wide = frame.pivot_table(index="year", columns="agent", values="weight", aggfunc="mean")
    order = [a for a in ["quality", "value", "growth", "momentum", "risk"] if a in wide]
    colors = [SLATE, GOLD, TEAL, RED, NAVY]
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    bottom = np.zeros(len(wide))
    for agent, color in zip(order, colors):
        ax.bar(wide.index.astype(str), wide[agent], bottom=bottom, color=color, label=agent, width=0.78)
        bottom += wide[agent].to_numpy()
    ax.set(title="Peso medio anual del meta-agente por agente", ylabel="Peso medio", ylim=(0, 1))
    legend_below(ax)
    save(fig, output)


def draw_bootstrap_forest(robustness: dict, output: Path) -> None:
    boot = robustness["bootstrap_and_era_exclusion"]
    rows: list[tuple[str, float, float | None, float | None]] = [
        ("Media de selección\n(IC 95 %)", boot["interval_95"]["mean"], boot["interval_95"]["ci_low"], boot["interval_95"]["ci_high"]),
        ("Media de selección\n(IC 90 %)", boot["interval_90"]["mean"], boot["interval_90"]["ci_low"], boot["interval_90"]["ci_high"]),
    ]
    for entry in boot["era_exclusions"]:
        rows.append((f"Excluyendo {entry['excluded_era']}\n({entry['n_cohorts']} cohortes)", entry["mean_rank_ic"], None, None))
    positions = range(len(rows))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for index, (_, mean, low, high) in zip(positions, rows):
        if low is not None:
            ax.plot([low, high], [index, index], color=SLATE, linewidth=2.4, solid_capstyle="round", zorder=2)
        ax.scatter(mean, index, color=NAVY, s=60, zorder=3)
    ax.axvline(0, color=RED, linewidth=1.2, linestyle="--")
    ax.set(title="Intervalos y estabilidad del Rank-IC", xlabel="Rank-IC medio", yticks=list(positions))
    ax.set_yticklabels([row[0] for row in rows], fontsize=8)
    ax.invert_yaxis()
    save(fig, output)


def build_robustness_rows(robustness: dict, attribution: dict) -> list[dict]:
    """Deriva los ocho contrastes de robustez desde los artefactos persistidos.

    Ninguna cifra ni veredicto se escribe a mano: tanto la tabla ``t06_robustez``
    como la figura ``f07_robustez`` consumen esta lista, de modo que si el study
    se regenera ambas salidas se actualizan solas. Los umbrales de veredicto son
    los declarados en ``docs/metodologia.md``.
    """
    boot = robustness["bootstrap_and_era_exclusion"]
    interval = boot["interval_95"]
    eras = [entry["mean_rank_ic"] for entry in boot["era_exclusions"]]
    placebos = [entry["summary"]["mean_rank_ic"] for entry in robustness["label_placebos"]]
    permutation = robustness["permutation"]
    dispersion = robustness["seed_dispersion"]["mean_rank_ic"]
    risk_matched = robustness["random_portfolios"]["risk_matched"]
    neutralized = attribution["neutralized_rank_ic"]
    deflated = attribution["deflated_sharpe"]
    replicas = f"{permutation['n_permutations']:,}".replace(",", ".")
    return [
        {
            "name": "Permutación",
            "detail": rf"$p={num(permutation['p_value'])}$; {replicas} réplicas",
            "passes": permutation["p_value"] < 0.05,
        },
        {
            "name": "Placebos",
            "detail": rf"Rank-IC entre ${num(min(placebos))}$ y ${num(max(placebos))}$",
            "passes": max(abs(value) for value in placebos) < 0.02,
        },
        {
            "name": "Bootstrap",
            "detail": rf"IC 95\% $[{num(interval['ci_low'])}; {num(interval['ci_high'])}]$",
            "passes": interval["ci_low"] > 0,
        },
        {
            "name": "Exclusión de eras",
            "detail": rf"Rank-IC entre ${num(min(eras))}$ y ${num(max(eras))}$",
            "passes": min(eras) > 0,
        },
        {
            "name": "Semillas",
            "detail": rf"Rango de Rank-IC ${num(dispersion['range'])}$",
            "passes": not dispersion["crosses_zero"],
        },
        {
            "name": "Aleatorias con riesgo",
            "detail": rf"Percentil ${num(100 * risk_matched['model_percentile'], 1)}$",
            "passes": risk_matched["model_percentile"] >= 0.95,
        },
        {
            "name": "Neutralización",
            "detail": "Retiene " + pct(neutralized["retained_fraction"]) + " de la señal",
            "passes": neutralized["retained_fraction"] > 0.5,
        },
        {
            "name": "Deflated Sharpe",
            "detail": rf"${num(deflated['deflated_sharpe_probability'], 3)} < 0,95$",
            "passes": deflated["deflated_sharpe_probability"] >= 0.95,
        },
    ]


def write_tables(paths: Paths, summary: dict, robustness: dict, attribution: dict, diag: pd.DataFrame, annual: pd.DataFrame, decisions: dict) -> None:
    selection = diag[pd.to_datetime(diag["prediction_date"]).dt.year.le(2024)]
    agents = selection.groupby("agent")["rank_ic"].agg(["mean", "std", "count", lambda s: (s > 0).mean()])
    agents.columns = ["mean", "std", "count", "positive"]
    rows = [[tex(i), num(r["mean"]), num(r["std"]), pct(r["positive"]), num(r["mean"] / r["std"], 3)] for i, r in agents.sort_values("mean", ascending=False).iterrows()]
    table(paths.tables / "t06_agentes.tex", ["Señal", "Rank-IC", "Desv.", "Positivas", "IC-IR"], rows)

    base = attribution["baselines"]["baselines"]
    rows = [[tex(v["baseline"]), num(v["mean_rank_ic"]), pct(v["positive_fraction"])] for v in base]
    rows.insert(0, ["Sistema (meta final)", num(summary["summary"]["mean_rank_ic"]), pct(summary["summary"]["rank_ic_positive_fraction"])])
    table(paths.tables / "t06_baselines.tex", ["Señal", "Rank-IC", "Cohortes positivas"], rows)

    robust_rows = [
        [row["name"], row["detail"], "Supera" if row["passes"] else "No supera"]
        for row in build_robustness_rows(robustness, attribution)
    ]
    table(paths.tables / "t06_robustez.tex", ["Contraste", "Resultado", "Veredicto"], robust_rows, "llc")

    annual_rows = [[str(int(r.year)), pct(r.portfolio_return), pct(r.benchmark_return), pct(r.alpha), pct(r.max_drawdown_year), num(r.information_ratio_year, 3), pct(r.mean_cash_weight), pct(r.turnover)] for r in annual.itertuples()]
    table(paths.tables / "t07_anual.tex", ["Año", "Cartera", "SPY", "Alfa", "MDD", "IR", "Efectivo", "Turnover"], annual_rows)

    # La tabla de perfiles se emite desde el Portfolio Study (`t08_perfiles_cartera`): con la
    # cartera del modelo el orden entre perfiles es distinto, y mantener las dos versiones dejaría
    # dos cifras contradictorias para el mismo perfil.

    write_tables_decisions(paths, decisions)


def write_tables_decisions(paths: Paths, decisions: dict) -> None:
    """Qué alternativa se rechazó en cada decisión y cuánto Rank-IC costaba.

    En la última pasada de una cadena convergida el Rank-IC del incumbente es constante —cada
    variable llega ya en su mejor valor— y una tabla de esa columna es literalmente una columna de
    ceros. Lo informativo es la alternativa descartada: cuánto habría costado adoptarla.

    El coste se define como el Rank-IC de la mejor alternativa menos el del ganador, de modo que un
    valor **positivo** significa que la alternativa medía mejor y que el ganador se decidió por otra
    vía. Eso ocurre en los empates técnicos, y por eso la regla se tabula junto al coste: sin ella,
    un coste positivo parecería un error.
    """
    rows = []
    for decision in decisions["decisions"]:
        candidates = decision["candidates"]
        winner = next(item for item in candidates if item["candidate_id"] == decision["winner_candidate_id"])
        others = [item for item in candidates if item["candidate_id"] != winner["candidate_id"]]
        if not others:
            continue
        runner_up = max(others, key=lambda item: item["mean_rank_ic"])
        rows.append({
            "variable": decision["variable_id"],
            "winner": decision["winner_value"],
            "alternative": runner_up["value"],
            "alternative_rank_ic": runner_up["mean_rank_ic"],
            "cost": runner_up["mean_rank_ic"] - winner["mean_rank_ic"],
            "rule": decision["selection_rule"],
        })
    rows.sort(key=lambda item: item["cost"])

    labels = {"robust_rank_ic": "Rank-IC robusto", "tie_simplicity": "Empate: simplicidad"}
    table_rows = [
        [
            tex(row["variable"]),
            tex(tex_free(row["winner"])),
            tex(tex_free(row["alternative"])),
            num(row["alternative_rank_ic"]),
            num(row["cost"]),
            tex(labels.get(row["rule"], row["rule"])),
        ]
        for row in rows
    ]
    table(
        paths.tables / "t05_decisiones.tex",
        ["Variable", "Ganador", "Mejor alternativa", "Su Rank-IC", "Coste", "Regla"],
        table_rows,
        "lllrrl",
    )


def build_result_macros(
    summary: dict,
    robustness: dict,
    attribution: dict,
    diag: pd.DataFrame,
    features: pd.DataFrame,
    catalog: dict,
    model_winner: dict,
    portfolio: dict,
    attribution_summary: dict[str, pd.DataFrame],
    cost: dict | None,
    narrative: dict | None,
) -> dict[str, str]:
    """Cifras decisivas compartidas por memoria y defensa.

    Los nombres son semánticos y no contienen identificadores ni rutas. La función recibe objetos
    ya leídos de los cuatro estudios adoptados, por lo que una misma cifra no se vuelve a copiar en
    prosa, tablas manuales o diapositivas.
    """
    selection = summary["summary"]
    confirmation = summary["confirmation_2025_2026"]
    significance = attribution["ic_significance"]["selection"]
    dsr = attribution["deflated_sharpe"]
    portfolio_winner = portfolio["winner"]
    winner_summary = portfolio_winner["winner_summary"]
    winner_confirmation = portfolio_winner["winner_confirmation"]
    portfolio_configuration = portfolio_winner["configuration"]
    baseline_confirmation = selection["confirmation"]

    selection_diag = diag[pd.to_datetime(diag["prediction_date"]).dt.year.le(2024)]
    risk_rank_ic = selection_diag.loc[selection_diag["agent"] == "risk", "rank_ic"].mean()
    leaders = attribution_summary["by_year"]
    risk_leaders = leaders[leaders["agent"] == "risk"].groupby("feature")["size"].sum()
    risk_total = risk_leaders.sum()
    gap_share = risk_leaders.get("factor_gap_21d", 0) / risk_total
    range_share = risk_leaders.get("factor_range_63d", 0) / risk_total
    influence = {row["variable"]: row for row in portfolio_influence_rows(portfolio["grid"])}

    values = {
        "NumPredictores": integer(sum(features["origin"].isin(["catálogo", "solo catálogo"]))),
        "NumParametrosCatalogo": integer(len(catalog["variables"])),
        "NumDiagnosticos": integer(len(features)),
        "RankICSeleccion": num(selection["mean_rank_ic"]),
        "RankICRisk": num(risk_rank_ic),
        "CohortesSeleccion": integer(significance["n_cohorts"]),
        "FraccionICPositiva": pct(selection["rank_ic_positive_fraction"], 1),
        "ICIRSeleccion": num(selection["ic_ir"], 3),
        "TNeweyWest": num(significance["newey_west_t"], 2),
        "PValorPermutacion": num(robustness["permutation"]["p_value"], 4),
        "NumPermutaciones": integer(robustness["permutation"]["n_permutations"]),
        "ProbabilidadDSR": num(dsr["deflated_sharpe_probability"], 3),
        "RankICReservado": num(confirmation["mean_rank_ic"]),
        "CohortesReservadas": integer(confirmation["n_cohorts"]),
        "FraccionICReservadaPositiva": pct(confirmation["rank_ic_positive_fraction"], 1),
        "GapShareRisk": pct(gap_share, 1),
        "RangeShareRisk": pct(range_share, 1),
        "DosVariablesRisk": pct(gap_share + range_share, 1),
        "NumCarteras": integer(len(portfolio["grid"])),
        "IRModelo": num(selection["information_ratio"], 3),
        "ExcesoModelo": pct(selection["geometric_excess_return"]),
        "TurnoverModelo": num(selection["annualized_turnover"], 2),
        "MDDModelo": pct(selection["max_drawdown"]),
        "IRModeloReservado": num(baseline_confirmation["information_ratio"], 3),
        "ExcesoModeloReservado": pct(baseline_confirmation["geometric_excess_return"]),
        "IRCartera": num(winner_summary["information_ratio"], 3),
        "ExcesoCartera": pct(winner_summary["geometric_excess_return"]),
        "TurnoverCartera": num(winner_summary["annualized_turnover"], 2),
        "MDDCartera": pct(winner_summary["max_drawdown"]),
        "BeatRateCartera": pct(winner_summary["beat_rate"], 0),
        "IRCarteraReservado": num(winner_confirmation["information_ratio"], 3),
        "ExcesoCarteraReservado": pct(winner_confirmation["geometric_excess_return"]),
        "TurnoverCarteraReservado": num(winner_confirmation["annualized_turnover"], 2),
        "MDDCarteraReservado": pct(winner_confirmation["max_drawdown"]),
        "BeatRateCarteraReservado": pct(winner_confirmation["beat_rate"], 0),
        "AnosCarteraReservada": num(winner_confirmation["years"], 2),
        "PosicionesCartera": integer(portfolio_configuration["target_size"]),
        "EfectivoModelo": pct(model_winner["configuration"]["max_cash_weight"], 0),
        "EfectivoCartera": pct(portfolio_configuration["max_cash_weight"], 0),
        "TenenciaModelo": tex(tex_free(model_winner["configuration"]["minimum_holding_period"])),
        "TenenciaCartera": tex(tex_free(portfolio_configuration["minimum_holding_period"])),
        "DerivaModelo": num(model_winner["configuration"]["rebalance_drift_tolerance"], 2),
        "DerivaCartera": num(portfolio_configuration["rebalance_drift_tolerance"], 2),
    }
    for variable, macro in (
        ("target_size", "InfluenciaPosiciones"),
        ("minimum_holding_period", "InfluenciaTenencia"),
        ("max_cash_weight", "InfluenciaEfectivo"),
        ("coverage_percentile_floor", "InfluenciaCobertura"),
        ("rebalance_drift_tolerance", "InfluenciaDeriva"),
        ("sizing_mode", "InfluenciaPesos"),
    ):
        values[macro] = num(influence[variable]["spread"], 3)
    if cost is not None:
        values["CosteAdoptado"] = integer(cost["adopted_cost_bps"])
        break_even = cost["break_even"]
        values["CosteEquilibrio"] = integer(
            break_even["resimulated"]["selection"]["bps_per_trade"]
        )
        values["CosteEquilibrioCongelado"] = integer(
            break_even["frozen_path"]["selection"]["bps_per_trade"]
        )
        # El equilibrio de la era reservada solo existe en la ruta congelada: al resimular, el
        # exceso ya es negativo con coste cero y no hay margen que agotar.
        reserved = break_even["frozen_path"]["confirmation"]
        if reserved.get("available"):
            values["CosteEquilibrioReservado"] = integer(reserved["bps_per_trade"])
        margins = cost["margin_over_adopted"]
        values["MargenCoste"] = num(margins["resimulated_selection"], 1)
        values["MargenCosteCongelado"] = num(margins["frozen_path_selection"], 1)
        values["MargenCosteReservado"] = num(margins["frozen_path_confirmation"], 1)
        zero = {row["family"]: row for row in cost["frozen_path"] + cost["resimulated"]
                if float(row["cost_bps"]) == 0.0}
        values["ExcesoBrutoCongelado"] = pct(
            zero["frozen_path"]["selection_geometric_excess_return"]
        )
        values["ExcesoBrutoResimulado"] = pct(
            zero["resimulated"]["selection_geometric_excess_return"]
        )
        values["ExcesoBrutoReservado"] = pct(
            zero["frozen_path"]["confirmation_geometric_excess_return"]
        )
    if narrative is not None:
        window = narrative["windows"]["selection"]
        values["NumAccionesCartera"] = integer(window["concentration"]["distinct_tickers_ever_held"])
        values["NumEpisodiosCartera"] = integer(window["holding_duration"]["episodes"])
        values["MedianaMesesTenencia"] = num(window["holding_duration"]["median_months"], 0)
        aapl = next(row for row in window["most_held"] if row["ticker"] == "AAPL")
        values["ContribucionAAPL"] = pct(aapl["net_contribution"], 1)
    return values


def assert_adopted_inputs(study_id: str, chain_ids: list[str], portfolio_study_id: str | None) -> None:
    if tuple(chain_ids) != ADOPTED_MODEL_STUDIES:
        raise ValueError(f"La cadena debe ser exactamente {ADOPTED_MODEL_STUDIES!r}.")
    if study_id != ADOPTED_MODEL_STUDIES[-1]:
        raise ValueError(f"El Model Study de referencia debe ser {ADOPTED_MODEL_STUDIES[-1]}.")
    if portfolio_study_id != ADOPTED_PORTFOLIO_STUDY:
        raise ValueError(f"El Portfolio Study adoptado debe ser {ADOPTED_PORTFOLIO_STUDY}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", required=True, help="Model Study de referencia (evidencia predictiva).")
    parser.add_argument(
        "--chain-study-id", action="append", default=None,
        help="Model Studies de la cadena, en orden de ejecución. Repetible. El último debe ser --study-id.",
    )
    parser.add_argument(
        "--portfolio-study-id", default=None,
        help="Portfolio Study cuyo ganador aporta la evidencia económica (rejilla, cartera y perfiles).",
    )
    parser.add_argument(
        "--audit", action="store_true",
        help="No escribe: recalcula las macros y comprueba que fuentes, activos e identificadores coinciden.",
    )
    args = parser.parse_args()
    assert_adopted_inputs(args.study_id, list(args.chain_study_id or []), args.portfolio_study_id)
    paths = load_paths(args.study_id)
    summary = read_json(paths.evidence / "summary.json")
    robustness = read_json(paths.study / "robustness.json")
    attribution = read_json(paths.study / "attribution.json")
    decisions = read_json(paths.study / "decisions.json")
    catalog = read_json(paths.study / "catalog_snapshot.json")
    winner = read_json(paths.study / "winner.json")
    weights = pd.read_parquet(paths.evidence / "meta_weights.parquet")
    diag = pd.read_parquet(paths.evidence / "rank_ic_diagnostics.parquet")
    tails = pd.read_parquet(paths.evidence / "rank_tail_diagnostics.parquet")
    scores = pd.read_parquet(paths.evidence / "agent_scores.parquet")
    calibration = pd.read_parquet(paths.evidence / "signal_calibration.parquet")
    model_attribution = pd.read_parquet(paths.evidence / "model_feature_attribution.parquet")
    features = load_features(paths)

    # La evidencia predictiva sale siempre del Model Study; la económica, del ganador del Portfolio
    # Study si se declara. Separarlas es lo que permite documentar el modelo del study 3 con la
    # cartera optimizada sin duplicar evidencia en disco ni mezclar procedencias en una figura.
    portfolio = load_portfolio(args.portfolio_study_id)
    economic = portfolio["study"] / "evidence_best_full"
    equity = pd.read_parquet(economic / "equity.parquet")
    annual = pd.read_parquet(economic / "annual_metrics.parquet")
    orders = pd.read_parquet(economic / "orders.parquet")
    coverage = read_json(ROOT / "data" / "raw" / "universe_coverage.json")

    chain = load_chain(list(args.chain_study_id))
    changes = chain_configuration_changes(chain)
    attribution_summary = load_agent_attribution(paths)
    cost = portfolio["cost"]
    narrative = load_portfolio_narrative(args.portfolio_study_id)
    macros = macro_content(
        build_result_macros(
            summary, robustness, attribution, diag, features, catalog, winner, portfolio,
            attribution_summary, cost, narrative,
        )
    )

    if args.audit:
        macro_path = paths.tables / "study_macros.tex"
        if not macro_path.is_file() or macro_path.read_text(encoding="utf-8") != macros:
            raise SystemExit("AUDITORÍA FALLIDA: study_macros.tex no coincide con los artefactos adoptados.")
        manifest = read_json(LATEX / "asset_manifest.json")
        expected_ids = {
            "study_id": args.study_id,
            "chain_study_ids": list(args.chain_study_id),
            "portfolio_study_id": args.portfolio_study_id,
        }
        for key, expected in expected_ids.items():
            if manifest.get(key) != expected:
                raise SystemExit(f"AUDITORÍA FALLIDA: {key}={manifest.get(key)!r}, esperado {expected!r}.")
        required_tables = {"t06_ganador.tex"}
        missing_tables = sorted(name for name in required_tables if not (paths.tables / name).is_file())
        if missing_tables:
            raise SystemExit(f"AUDITORÍA FALLIDA: faltan tablas {missing_tables!r}.")
        if not required_tables.issubset(set(manifest.get("tables", []))):
            raise SystemExit("AUDITORÍA FALLIDA: la tabla del ganador no está declarada.")
        required_assets = {
            "f01_spiva_horizontes.png", "f03_cobertura_anual.png",
            "f03_muestra_oos.png", "f06_estabilidad_features.png",
            "f05_calibracion_alfa.png", "f06_atribucion_factorial.png",
            "f06_atribucion_anual.png", "f06_pesos_anual.png", "f06_orden_vs_pago.png",
            "f07_alpha_turnover_anual.png", "f06_bootstrap.png", "f07_capacidad.png",
            "f07_cartera_narrativa.png", "f06_cola_eras.png", "f07_costes_escalera.png",
            "f07_equity_drawdown.png", "f07_perfiles_pesos.png", "f07_perfiles_resultados.png",
            "f06_rankic_serie.png",
            "f07_cartera_influencia.png", "f07_cartera_rejilla.png",
            "f07_seleccion_vs_reservada.png",
        }
        missing = sorted(name for name in required_assets if not (paths.figures / name).is_file())
        if missing:
            raise SystemExit(f"AUDITORÍA FALLIDA: faltan activos {missing!r}.")
        declared_figures = set(manifest.get("figures", []))
        if not required_assets.issubset(declared_figures):
            undeclared = sorted(required_assets - declared_figures)
            raise SystemExit(f"AUDITORÍA FALLIDA: activos no declarados {undeclared!r}.")
        required_profile_sources = {
            f"{args.portfolio_study_id}/profiles/{profile}/annual_metrics.parquet"
            for profile in portfolio["profile_annual"]
        }
        if not required_profile_sources.issubset(set(manifest.get("economic_sources", []))):
            raise SystemExit("AUDITORÍA FALLIDA: faltan las fuentes anuales de perfiles.")
        expected_spiva = json.loads(json.dumps(
            SPIVA_SOURCE | {"underperformance_pct": SPIVA_UNDERPERFORMANCE}
        ))
        if manifest.get("literature_sources", {}).get("spiva") != expected_spiva:
            raise SystemExit("AUDITORÍA FALLIDA: la procedencia SPIVA no coincide.")
        if manifest.get("panel_sources") != ["data/raw/universe_coverage.json"]:
            raise SystemExit("AUDITORÍA FALLIDA: la cobertura no declara su fuente canónica.")
        required_predictive_sources = {
            "evidence/agent_scores.parquet", "evidence/signal_calibration.parquet",
            "evidence/model_feature_attribution.parquet",
        }
        if not required_predictive_sources.issubset(set(manifest.get("sources", []))):
            raise SystemExit("AUDITORÍA FALLIDA: faltan fuentes de muestra, calibración o estabilidad.")
        print("Auditoría LaTeX correcta: macros, activos e identificadores coinciden.")
        return

    draw_spiva_horizons(paths.figures / "f01_spiva_horizontes.png")
    draw_universe_coverage(coverage, paths.figures / "f03_cobertura_anual.png")
    draw_oos_sample(scores, features, paths.figures / "f03_muestra_oos.png")
    write_sample_table(paths, scores)
    draw_signal_calibration(calibration, scores, paths.figures / "f05_calibracion_alfa.png")
    write_calibration_table(paths, calibration)
    draw_feature_stability(model_attribution, paths.figures / "f06_estabilidad_features.png")
    draw_factor_attribution(attribution, paths.figures / "f06_atribucion_factorial.png")
    write_feature_dictionary(paths, features)
    draw_rank_ic(summary, paths.figures / "f06_rankic_serie.png")
    draw_order_vs_payoff(summary, paths.figures / "f06_orden_vs_pago.png")
    draw_equity_and_drawdown(equity, paths.figures / "f07_equity_drawdown.png")
    draw_tail_by_era(tails, paths.figures / "f06_cola_eras.png")
    # Activos añadidos para la versión extendida del manuscrito.
    draw_meta_weights_annual(weights, paths.figures / "f06_pesos_anual.png")

    # Explicabilidad: qué variables mueven a cada agente y si eso cambia con el régimen. Se agrega
    # una sola vez porque el artefacto de origen tiene 1,3 millones de filas.
    draw_attribution_by_year(attribution_summary, "risk", paths.figures / "f06_atribucion_anual.png")
    write_tables_attribution(paths, attribution_summary)

    # Activos de la cadena de studies encadenados. Solo se generan si se declara la cadena: sin
    # ella el manuscrito documentaría un study suelto y estas figuras no tendrían nada que contar.
    draw_selection_vs_reserved(chain, paths.figures / "f07_seleccion_vs_reservada.png")
    write_tables_chain(paths, chain, changes)
    draw_bootstrap_forest(robustness, paths.figures / "f06_bootstrap.png")

    write_tables_winner(paths, catalog, winner)
    write_tables(paths, summary, robustness, attribution, diag, annual, decisions)
    write_tables_predictive(paths, diag, features)
    write_tables_universe_resolution(paths)
    write_tables_robustness(paths, summary, robustness, attribution, portfolio)
    _write_tables_orders_and_tails(paths, orders, tails)

    # Activos de la rejilla de cartera. Sustituyen al barrido diagnóstico de una variable cada vez:
    # el esquema de la rejilla tiene seis coordenadas simultáneas y el objetivo pasa a ser el IR.
    draw_portfolio_grid(portfolio["grid"], portfolio["winner"], paths.figures / "f07_cartera_rejilla.png")
    draw_portfolio_influence(portfolio["grid"], paths.figures / "f07_cartera_influencia.png")
    draw_alpha_turnover_annual(annual, paths.figures / "f07_alpha_turnover_anual.png")
    draw_cost_ladder(cost, paths.figures / "f07_costes_escalera.png")
    draw_capacity(portfolio["capacity"], paths.figures / "f07_capacidad.png")
    draw_profile_weights(paths.figures / "f07_perfiles_pesos.png")
    draw_profile_results(
        portfolio["profiles"], portfolio["profile_annual"],
        paths.figures / "f07_perfiles_resultados.png",
    )
    write_tables_portfolio_study(paths, portfolio, summary["summary"])
    if narrative is not None:
        draw_portfolio_narrative(narrative, paths.figures / "f07_cartera_narrativa.png")
        write_tables_portfolio_narrative(paths, narrative)
    write_tables_catalog(paths, catalog, winner, decisions)
    (paths.tables / "study_macros.tex").write_text(macros, encoding="utf-8")

    # El manifiesto declara de qué estudio sale cada familia de artefactos: lo predictivo del Model
    # Study y lo económico del ganador del Portfolio Study. Sin esa separación, una figura de
    # equity y una de Rank-IC parecerían tener el mismo origen y no lo tienen.
    manifest = {
        "study_id": args.study_id,
        "chain_study_ids": list(args.chain_study_id or []),
        "portfolio_study_id": args.portfolio_study_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [
            "catalog_snapshot.json", "winner.json",
            "evidence/summary.json", "robustness.json", "attribution.json", "decisions.json",
            "evidence/meta_weights.parquet", "evidence/rank_ic_diagnostics.parquet",
            "evidence/agent_scores.parquet", "evidence/signal_calibration.parquet",
            "evidence/model_feature_attribution.parquet",
            "profile_comparison.parquet",
            "evidence/rank_tail_diagnostics.parquet",
            "evidence/feature_catalog.json", "evidence/feature_diagnostics.parquet",
            "evidence/agent_local_attribution.parquet",
        ],
        "economic_sources": (
            [
                f"{args.portfolio_study_id}/portfolio_grid.parquet",
                f"{args.portfolio_study_id}/portfolio_winner.json",
                f"{args.portfolio_study_id}/portfolio_profiles.parquet",
                f"{args.portfolio_study_id}/evidence_best_full/equity.parquet",
                f"{args.portfolio_study_id}/evidence_best_full/annual_metrics.parquet",
                f"{args.portfolio_study_id}/evidence_best_full/orders.parquet",
                f"{args.portfolio_study_id}/cost_sensitivity.json",
                f"{args.portfolio_study_id}/capacity.json",
                f"{args.portfolio_study_id}/portfolio_narrative.json",
            ] + [
                f"{args.portfolio_study_id}/profiles/{profile}/annual_metrics.parquet"
                for profile in portfolio["profile_annual"]
            ]
            if portfolio
            else ["evidence/equity.parquet", "evidence/annual_metrics.parquet", "evidence/orders.parquet"]
        ),
        # Fuentes que describen el panel y no un estudio: viven en `data/raw/` y son las mismas
        # cualquiera que sea el modelo entrenado encima.
        "panel_sources": ["data/raw/universe_coverage.json"],
        "literature_sources": {
            "spiva": SPIVA_SOURCE | {"underperformance_pct": SPIVA_UNDERPERFORMANCE},
        },
        "figures": sorted(path.name for path in paths.figures.glob("f*.png")),
        "tables": sorted(path.name for path in paths.tables.glob("t*.tex")),
        "numeric_macros": "study_macros.tex",
    }
    (LATEX / "asset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
