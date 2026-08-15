"""Exporta activos de los estudios de referencia para el manuscrito XeLaTeX.

Uso desde la raíz del repositorio:
    python latex/scripts/export_study_assets.py \
        --study-id study-20260814-095144-5ec17b78 \
        --chain-study-id study-20260812-163136-1b104667 \
        --chain-study-id study-20260813-103456-aa733655 \
        --chain-study-id study-20260814-095144-5ec17b78 \
        --portfolio-study-id study-20260814-135754-fdbdf2c5

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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LATEX = ROOT / "latex"

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


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * float(value):.{digits}f}".replace(".", ",") + r"\%"


def num(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}".replace(".", ",")


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
    assets = LATEX / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    return Paths(study, evidence, assets, assets)


def load_chain(study_ids: list[str]) -> pd.DataFrame:
    """Métricas comparables de la cadena de Model Studies, en orden de ejecución.

    Cada pasada tomó como baseline al ganador de la anterior, así que la fila *i* no es un
    experimento independiente sino el punto de partida de la *i+1*. Todo se lee de los artefactos:
    ninguna cifra de la cadena se escribe a mano.

    ``newey_west_t`` no está en ``summary.json`` —vive en ``attribution.json``, bajo
    ``ic_significance/selection``— y las cifras de la era reservada salen de ``confirmation``, que
    solo existe porque el ganador se reevaluó sobre la serie completa.
    """
    rows = []
    for order, study_id in enumerate(study_ids, start=1):
        study = ROOT / "results" / "studies" / study_id
        summary = read_json(study / "evidence" / "summary.json")["summary"]
        attribution = read_json(study / "attribution.json")
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
                "newey_west_t": attribution["ic_significance"]["selection"]["newey_west_t"],
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


def write_tables_predictive(paths: Paths, diag: pd.DataFrame, features: pd.DataFrame, attribution: dict) -> None:
    """Tablas sobre datos, features y capacidad predictiva por agente."""
    matrix = agent_era_matrix(diag)
    rows = [
        [tex(str(name).replace("_", " "))] + [num(matrix.loc[name, era]) for era in matrix.columns]
        for name in matrix.index
    ]
    table(paths.tables / "t05_rankic_era.tex", ["Señal"] + list(matrix.columns), rows)

    grouped = (
        features.groupby("block")
        .agg(
            n=("feature", "size"),
            coverage=("coverage", "mean"),
            rank_ic=("univariate_rank_ic", "mean"),
            importance=("model_importance_mean", "mean"),
        )
        .sort_values("rank_ic", ascending=False)
    )
    agents_by_block = features.groupby("block")["agents"].agg(
        lambda values: ", ".join(sorted({item for value in values if value != "—" for item in str(value).split(", ")}))
    )
    rows = [
        [
            tex(str(name).replace("_", " ")),
            tex(agents_by_block.get(name, "—") or "—"),
            str(int(row.n)),
            pct(row.coverage, 1) if pd.notna(row.coverage) else "n/d",
            num(row.rank_ic) if pd.notna(row.rank_ic) else "n/d",
            num(row.importance, 1) if pd.notna(row.importance) else "n/d",
        ]
        for name, row in grouped.iterrows()
    ]
    table(
        paths.tables / "t03_features_bloque.tex",
        ["Bloque", "Agentes", "Vars.", "Cobertura", "Rank-IC univ.", "Importancia"],
        rows,
        "llrrrr",
    )

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

    coverage = pd.DataFrame(attribution["universe_coverage"])
    rows = [
        [str(int(row.year)), str(int(row.distinct_tickers)), f"{int(row.usable_rows):,}".replace(",", "."), pct(row.usable_fraction, 2)]
        for row in coverage.itertuples()
    ]
    longtable(
        paths.tables / "t03_cobertura_anual.tex",
        ["Año", "Tickers distintos", "Filas utilizables", "Fracción utilizable"],
        rows,
        "Cobertura anual del universo y calidad del panel.",
        "tab:cobertura-anual",
        "lrrr",
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




def _write_tables_orders_and_tails(paths: Paths, orders: pd.DataFrame, tails: pd.DataFrame) -> None:
    """Motivos de las órdenes y comportamiento de la cola de la señal.

    Se emiten con o sin Portfolio Study: describen cómo se ejecuta la cartera y cómo se comporta el
    extremo del ranking, no qué cartera se eligió.
    """
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

    frame = tails.copy()
    frame["prediction_date"] = pd.to_datetime(frame["prediction_date"])
    frame["era"] = frame["prediction_date"].dt.year.map(era_of)
    grouped = frame.groupby("era").agg(
        rank_ic=("rank_ic", "mean"),
        top10=("top_10_excess_mean", "mean"),
        top_decile=("top_decile_excess_mean", "mean"),
        universe=("universe_excess_mean", "mean"),
        spread=("top_minus_bottom", "mean"),
        cohorts=("rank_ic", "size"),
    )
    rows = [
        [
            tex(name),
            num(row.rank_ic),
            pct(row.top10),
            pct(row.top_decile),
            pct(row.universe),
            pct(row.spread),
            str(int(row.cohorts)),
        ]
        for name, row in grouped.iterrows()
    ]
    table(
        paths.tables / "t07_cola.tex",
        ["Era", "Rank-IC", "Top 10", "Decil sup.", "Universo", "Sup. menos inf.", "Cohortes"],
        rows,
    )


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
    pivot = pivot.loc[:, pivot.sum().nlargest(top).index]
    shares = pivot.div(pivot.sum(axis=1).where(lambda total: total > 0), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bottom = np.zeros(len(shares))
    for column, color in zip(shares.columns, [NAVY, TEAL, GOLD, RED, GREEN]):
        values = shares[column].to_numpy()
        ax.bar(shares.index.astype(str), values, bottom=bottom, color=color,
               label=column.replace("factor_", "").replace("_", " "))
        bottom += values
    ax.set(
        title=f"Qué variable encabeza la atribución de {agent}, año a año",
        ylabel="Cuota de las observaciones del año",
    )
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    legend_below(ax, 2, anchor=-0.20)
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
        paths.tables / "t05_atribucion.tex",
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

    La rejilla se midió sobre la serie recortada en 2024 —ninguna de las 1.728 combinaciones pudo
    ver la era reservada—, mientras que ``portfolio_winner.json`` guarda además la confirmación del
    ganador sobre la serie completa. Son cifras de ventanas distintas y no deben mezclarse.
    """
    study = ROOT / "results" / "studies" / study_id
    winner_path = study / "portfolio_winner.json"
    grid_path = study / "portfolio_grid.parquet"
    if not winner_path.is_file() or not grid_path.is_file():
        raise FileNotFoundError(f"El Portfolio Study {study_id} no tiene artefactos completos.")
    profiles_path = study / "portfolio_profiles.parquet"
    return {
        "study_id": study_id,
        "study": study,
        "grid": pd.read_parquet(grid_path),
        "winner": read_json(winner_path),
        "profiles": pd.read_parquet(profiles_path) if profiles_path.is_file() else None,
    }


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


def draw_portfolio_marginals(grid: pd.DataFrame, output: Path) -> None:
    """Efecto marginal de cada variable de cartera sobre el Information Ratio.

    Un panel por variable, con la distribución del IR de todas las combinaciones que comparten ese
    valor. Es la lectura honesta de un cartesiano: al fijar una coordenada, las otras cinco siguen
    variando, así que cada caja resume un conjunto de carteras y no una cartera concreta. Sirve
    para ver qué variables mueven el IR y cuáles son casi indiferentes.
    """
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.0))
    for ax, variable in zip(axes.ravel(), GRID_VARIABLES):
        values = sorted(grid[variable].dropna().unique(), key=_grid_sort_key)
        groups = [grid.loc[grid[variable] == value, "information_ratio"].dropna().to_numpy() for value in values]
        box = ax.boxplot(groups, patch_artist=True, widths=0.55, showfliers=False)
        for patch in box["boxes"]:
            patch.set(facecolor=LIGHT, edgecolor=NAVY, linewidth=0.9)
        for element in ("whiskers", "caps"):
            for item in box[element]:
                item.set(color=NAVY, linewidth=0.9)
        for median in box["medians"]:
            median.set(color=RED, linewidth=1.4)
        ax.set_title(GRID_LABELS[variable], fontsize=9)
        ax.set_xticks(range(1, len(values) + 1))
        ax.set_xticklabels([tex_free(value) for value in values], rotation=32, ha="right", fontsize=6.6)
        ax.tick_params(axis="y", labelsize=7)
    fig.supylabel("Information Ratio (2015-2024)", fontsize=9)
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


def write_tables_portfolio_study(paths: Paths, portfolio: dict, baseline: dict) -> None:
    """Tablas del Portfolio Study: mejores carteras, ganadora y perfiles.

    ``baseline`` es el resumen del ganador del Model Study, es decir, la cartera que el manuscrito
    documentaba antes de optimizar. Aparece como fila de referencia para que la mejora se lea como
    diferencia y no como cifra suelta.

    En lugar de un top-N —cuyas filas se diferencian en un decimal y repiten la misma cartera con
    variaciones menores— se tabula cuánto separa cada variable el Information Ratio. Responde a la
    pregunta útil: qué decisiones de cartera importan y cuáles son indiferentes.
    """
    grid = portfolio["grid"].dropna(subset=["information_ratio"])
    rows = []
    for variable in GRID_VARIABLES:
        medians = grid.groupby(grid[variable].astype(str))["information_ratio"].median().sort_values(ascending=False)
        if medians.empty:
            continue
        rows.append({
            "variable": variable,
            "best_value": medians.index[0],
            "best": medians.iloc[0],
            "worst_value": medians.index[-1],
            "worst": medians.iloc[-1],
            "spread": medians.iloc[0] - medians.iloc[-1],
        })
    rows.sort(key=lambda item: item["spread"], reverse=True)
    table(
        paths.tables / "t08_cartera_influencia.tex",
        ["Variable", "Mejor valor", "IR mediano", "Peor valor", "IR mediano", "Diferencia"],
        [
            [
                tex(GRID_LABELS[row["variable"]]),
                tex(tex_free(row["best_value"])),
                num(row["best"], 3),
                tex(tex_free(row["worst_value"])),
                num(row["worst"], 3),
                num(row["spread"], 3),
            ]
            for row in rows
        ],
        "llrlrr",
    )

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
        paths.tables / "t08_cartera_ganadora.tex",
        ["Métrica", "Cartera del modelo", "Cartera ganadora", "Diferencia", "Era reservada"],
        metric_rows,
        "lrrrr",
    )

    profiles = portfolio.get("profiles")
    if profiles is not None and not profiles.empty:
        applicable = profiles[profiles["applicable"]] if "applicable" in profiles else profiles
        ordered = applicable.sort_values("information_ratio", ascending=False)
        profile_rows = [
            [
                tex(row.profile),
                num(row.information_ratio, 3),
                pct(row.geometric_excess_return),
                num(row.annualized_turnover, 2),
                pct(row.mean_cash_weight),
                num(row.confirmation_information_ratio, 3) if pd.notna(row.confirmation_information_ratio) else "—",
                pct(row.confirmation_excess) if pd.notna(row.confirmation_excess) else "—",
            ]
            for row in ordered.itertuples()
        ]
        table(
            paths.tables / "t08_perfiles_cartera.tex",
            ["Perfil", "IR sel.", "Exceso sel.", "Turnover", "Efectivo", "IR reserv.", "Exceso reserv."],
            profile_rows,
            "lrrrrrr",
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
        paths.tables / "t06_catalogo.tex",
        ["Variable", "Etapa", "Predictiva", "Valores del catálogo", "Ganador"],
        rows,
        "Catálogo cerrado completo: las 33 variables, su rejilla y el valor del ganador.",
        "tab:catalogo",
        # La tabla se incluye dentro de un entorno landscape, donde \textwidth
        # conserva el valor del formato vertical: hay que usar \linewidth para
        # aprovechar el ancho real de la página girada.
        "lllp{0.34\\linewidth}l",
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


def draw_equity(equity: pd.DataFrame, output: Path) -> None:
    equity["snapshot_date"] = pd.to_datetime(equity["snapshot_date"])
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.plot(equity["snapshot_date"], equity["portfolio_value"], color=NAVY, linewidth=2, label="Cartera")
    ax.plot(equity["snapshot_date"], equity["benchmark_value"], color=SLATE, linewidth=1.7, label="SPY")
    ax.axvspan(pd.Timestamp("2025-01-01"), equity["snapshot_date"].max(), color=GOLD, alpha=0.13, label="Era reservada")
    ax.set(title="Evolución acumulada: cartera frente a SPY", ylabel="Valor de 100 unidades monetarias")
    legend_below(ax)
    save(fig, output)


def draw_drawdown(equity: pd.DataFrame, output: Path) -> None:
    equity["snapshot_date"] = pd.to_datetime(equity["snapshot_date"])
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    for col, color, label in [("portfolio_value", NAVY, "Cartera"), ("benchmark_value", SLATE, "SPY")]:
        drawdown = equity[col] / equity[col].cummax() - 1
        ax.fill_between(equity["snapshot_date"], drawdown, 0, color=color, alpha=0.22)
        ax.plot(equity["snapshot_date"], drawdown, color=color, linewidth=1.4, label=label)
    ax.set(title="Drawdown acumulado", ylabel="Drawdown")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    legend_below(ax)
    save(fig, output)


def draw_profile_tradeoff(profiles: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    for row in profiles.itertuples():
        color = NAVY if row.profile == "balanced" else TEAL
        ax.scatter(row.annualized_turnover, row.geometric_excess_return, color=color, s=46, zorder=3)
        ax.annotate(row.profile, (row.annualized_turnover, row.geometric_excess_return), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Rotación y exceso geométrico", xlabel="Turnover anualizado", ylabel="Exceso geométrico")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
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


def draw_permutation(robustness: dict, output: Path) -> None:
    permutation = robustness["permutation"]
    placebos = [entry["summary"]["mean_rank_ic"] for entry in robustness["label_placebos"]]
    observed = permutation["observed_mean_rank_ic"]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.axvspan(min(placebos), max(placebos), color=SLATE, alpha=0.30, label="Rango de los placebos de etiqueta")
    ax.scatter(placebos, [0] * len(placebos), color=SLATE, s=38, zorder=3)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.annotate(
        f"Observado {num(observed)}",
        xy=(observed, 0),
        xytext=(observed, 0.55),
        ha="center",
        fontsize=8.5,
        fontweight="bold",
        color=NAVY,
        arrowprops={"arrowstyle": "-|>", "color": NAVY, "linewidth": 1.6},
    )
    replicas = f"{permutation['n_permutations']:,}".replace(",", ".")
    ax.set(
        title=f"Ninguna de las {replicas} permutaciones alcanzó el Rank-IC observado ($p={num(permutation['p_value'])}$)",
        xlabel="Rank-IC medio",
        yticks=[],
        ylim=(-0.5, 1.05),
    )
    legend_below(ax)
    save(fig, output)


def draw_random_portfolios(robustness: dict, output: Path) -> None:
    scenarios = [("general", "Aleatorias generales"), ("risk_matched", "Riesgo emparejado")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=False)
    for ax, (key, title) in zip(axes, scenarios):
        data = robustness["random_portfolios"][key]
        labels = ["Mediana", "Media", "Percentil 95"]
        values = [data["random_median"], data["random_mean"], data["random_p95"]]
        ax.bar(labels, values, color=SLATE, width=0.62)
        ax.axhline(data["model_cagr"], color=NAVY, linewidth=2, label="Modelo")
        ax.set(title=f"{title}\npercentil {num(100 * data['model_percentile'], 1)}", ylabel="CAGR" if key == "general" else None)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
        ax.tick_params(axis="x", labelsize=7.5)
        ax.margins(y=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False)
    save(fig, output)


def draw_seed_dispersion(robustness: dict, output: Path) -> None:
    dispersion = robustness["seed_dispersion"]
    metrics = [
        ("mean_rank_ic", "Rank-IC medio", 1.0),
        ("geometric_excess_return", "Exceso geométrico", 100.0),
        ("information_ratio", "Information ratio", 1.0),
        ("cagr_portfolio", "CAGR de cartera", 100.0),
    ]
    available = [item for item in metrics if item[0] in dispersion]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for index, (key, label, scale) in enumerate(available):
        entry = dispersion[key]
        low, high = entry["min"] * scale, entry["max"] * scale
        ax.plot([low, high], [index, index], color=SLATE, linewidth=2.4, solid_capstyle="round", zorder=2)
        ax.scatter([low, high], [index, index], color=SLATE, s=34, zorder=3)
        ax.scatter(entry["median"] * scale, index, color=NAVY, s=70, zorder=4, label="Mediana" if index == 0 else None)
        ax.annotate(
            f"rango {num(entry['range'] * scale, 4 if scale == 1 else 2)}",
            xy=(high, index),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
        )
    ax.set(title="Dispersión entre semillas (42, 7 y 2026)", yticks=range(len(available)))
    ax.set_yticklabels([f"{label}{'  (%)' if scale != 1 else ''}" for _, label, scale in available], fontsize=8)
    ax.invert_yaxis()
    legend_below(ax)
    save(fig, output)


def draw_tail_spread(tails: pd.DataFrame, output: Path) -> None:
    frame = tails.copy()
    frame["prediction_date"] = pd.to_datetime(frame["prediction_date"])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(frame["prediction_date"], frame["top_decile_minus_universe"], width=22, color=SLATE, alpha=0.55, label="Decil superior menos universo")
    ax.plot(
        frame["prediction_date"],
        frame["top_decile_minus_universe"].rolling(12, min_periods=4).mean(),
        color=NAVY,
        linewidth=2,
        label="Media móvil de 12 cohortes",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvspan(pd.Timestamp("2025-01-01"), frame["prediction_date"].max(), color=GOLD, alpha=0.13, label="Era reservada")
    ax.set(title="Ventaja del decil superior sobre el universo", ylabel="Exceso medio")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    legend_below(ax)
    save(fig, output)


def draw_signal_health(health: pd.DataFrame, output: Path) -> None:
    frame = health.copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.plot(frame["snapshot_date"], frame["shrunk_rank_ic"], color=NAVY, linewidth=1.9, label="Memoria de 16 trimestres")
    ax.plot(frame["snapshot_date"], frame["shrunk_rank_ic_8q"], color=TEAL, linewidth=1.5, linestyle="--", label="Memoria de 8 trimestres")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvspan(pd.Timestamp("2025-01-01"), frame["snapshot_date"].max(), color=GOLD, alpha=0.13, label="Era reservada")
    ax.set(title="Salud de la señal observable en cada snapshot", ylabel="Rank-IC contraído")
    legend_below(ax)
    save(fig, output)


def build_robustness_rows(robustness: dict, attribution: dict) -> list[dict]:
    """Deriva los ocho contrastes de robustez desde los artefactos persistidos.

    Ninguna cifra ni veredicto se escribe a mano: tanto la tabla ``t07_robustez``
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
    table(paths.tables / "t05_agentes.tex", ["Señal", "Rank-IC", "Desv.", "Positivas", "IC-IR"], rows)

    base = attribution["baselines"]["baselines"]
    rows = [[tex(v["baseline"]), num(v["mean_rank_ic"]), pct(v["positive_fraction"])] for v in base]
    rows.insert(0, ["Sistema (meta final)", num(summary["summary"]["mean_rank_ic"]), pct(summary["summary"]["rank_ic_positive_fraction"])])
    table(paths.tables / "t07_baselines.tex", ["Señal", "Rank-IC", "Cohortes positivas"], rows)

    robust_rows = [
        [row["name"], row["detail"], "Supera" if row["passes"] else "No supera"]
        for row in build_robustness_rows(robustness, attribution)
    ]
    table(paths.tables / "t07_robustez.tex", ["Contraste", "Resultado", "Veredicto"], robust_rows, "llc")

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
        paths.tables / "t06_decisiones.tex",
        ["Variable", "Ganador", "Mejor alternativa", "Su Rank-IC", "Coste", "Regla"],
        table_rows,
        "lllrrl",
    )


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
    args = parser.parse_args()
    paths = load_paths(args.study_id)
    summary = read_json(paths.evidence / "summary.json")
    robustness = read_json(paths.study / "robustness.json")
    attribution = read_json(paths.study / "attribution.json")
    decisions = read_json(paths.study / "decisions.json")
    catalog = read_json(paths.study / "catalog_snapshot.json")
    winner = read_json(paths.study / "winner.json")
    weights = pd.read_parquet(paths.evidence / "meta_weights.parquet")
    diag = pd.read_parquet(paths.evidence / "rank_ic_diagnostics.parquet")
    profiles = pd.read_parquet(paths.study / "profile_comparison.parquet")
    tails = pd.read_parquet(paths.evidence / "rank_tail_diagnostics.parquet")
    health = pd.read_parquet(paths.evidence / "signal_health.parquet")
    features = load_features(paths)

    # La evidencia predictiva sale siempre del Model Study; la económica, del ganador del Portfolio
    # Study si se declara. Separarlas es lo que permite documentar el modelo del study 3 con la
    # cartera optimizada sin duplicar evidencia en disco ni mezclar procedencias en una figura.
    portfolio = load_portfolio(args.portfolio_study_id) if args.portfolio_study_id else None
    economic = (portfolio["study"] / "evidence_best_full") if portfolio else paths.evidence
    equity = pd.read_parquet(economic / "equity.parquet")
    annual = pd.read_parquet(economic / "annual_metrics.parquet")
    orders = pd.read_parquet(economic / "orders.parquet")

    draw_rank_ic(summary, paths.figures / "f07_rankic_serie.png")
    draw_equity(equity, paths.figures / "f07_equity.png")
    draw_drawdown(equity, paths.figures / "f07_drawdown.png")
    # Los perfiles se dibujan con la cartera adoptada cuando hay Portfolio Study: con la del modelo
    # el orden entre estilos cambia —gana `value` en vez de `balanced`— y las figuras contradirían
    # a la tabla del capítulo.
    profile_figures = portfolio["profiles"] if portfolio is not None and portfolio.get("profiles") is not None else profiles
    draw_profile_tradeoff(profile_figures, paths.figures / "f07_perfiles_tradeoff.png")

    # Activos añadidos para la versión extendida del manuscrito.
    draw_meta_weights_annual(weights, paths.figures / "f05_pesos_anual.png")

    # Explicabilidad: qué variables mueven a cada agente y si eso cambia con el régimen. Se agrega
    # una sola vez porque el artefacto de origen tiene 1,3 millones de filas.
    attribution_summary = load_agent_attribution(paths)
    draw_attribution_by_year(attribution_summary, "risk", paths.figures / "f05_atribucion_anual.png")
    write_tables_attribution(paths, attribution_summary)

    # Activos de la cadena de studies encadenados. Solo se generan si se declara la cadena: sin
    # ella el manuscrito documentaría un study suelto y estas figuras no tendrían nada que contar.
    if args.chain_study_id:
        chain = load_chain(args.chain_study_id)
        changes = chain_configuration_changes(chain)
        draw_selection_vs_reserved(chain, paths.figures / "f09_seleccion_vs_reservada.png")
        write_tables_chain(paths, chain, changes)
    draw_bootstrap_forest(robustness, paths.figures / "f07_bootstrap.png")
    draw_permutation(robustness, paths.figures / "f07_permutacion.png")
    draw_random_portfolios(robustness, paths.figures / "f07_aleatorias.png")
    draw_seed_dispersion(robustness, paths.figures / "f07_semillas.png")
    draw_tail_spread(tails, paths.figures / "f07_cola.png")
    draw_signal_health(health, paths.figures / "f07_salud.png")

    write_tables(paths, summary, robustness, attribution, diag, annual, decisions)
    write_tables_predictive(paths, diag, features, attribution)
    write_tables_robustness(paths, summary, robustness, attribution, portfolio)
    _write_tables_orders_and_tails(paths, orders, tails)

    # Activos de la rejilla de cartera. Sustituyen al barrido diagnóstico de una variable cada vez:
    # el esquema de la rejilla tiene seis coordenadas simultáneas y el objetivo pasa a ser el IR.
    if portfolio is not None:
        draw_portfolio_grid(portfolio["grid"], portfolio["winner"], paths.figures / "f08_cartera_rejilla.png")
        draw_portfolio_marginals(portfolio["grid"], paths.figures / "f08_cartera_marginales.png")
        write_tables_portfolio_study(paths, portfolio, summary["summary"])
    write_tables_catalog(paths, catalog, winner, decisions)

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
            "profile_comparison.parquet",
            "evidence/rank_tail_diagnostics.parquet", "evidence/signal_health.parquet",
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
            ]
            if portfolio
            else ["evidence/equity.parquet", "evidence/annual_metrics.parquet", "evidence/orders.parquet"]
        ),
        "figures": sorted(path.name for path in paths.figures.glob("f*.png")),
        "tables": sorted(path.name for path in paths.tables.glob("t*.tex")),
    }
    (LATEX / "asset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
