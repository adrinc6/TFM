"""Exporta activos del study de referencia para el manuscrito XeLaTeX.

Uso desde la raíz del repositorio:
    python latex/scripts/export_study_assets.py --study-id study-20260803-201234-b4d7a8d8

Los PNG se generan a 300 dpi, con fondo blanco y dimensiones de impresión, y están preparados
para incluirse directamente en Overleaf. Cada salida queda registrada en
``latex/asset_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
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


def decision_records(decisions: dict) -> list[dict]:
    """Traza de la optimización secuencial, decisión a decisión.

    ``decisions.json`` no persiste si el ganador desplazó al incumbente, así que
    se deriva comparando el candidato ganador con el marcado ``is_incumbent``.
    El Rank-IC devuelto es el del ganador, es decir, el incumbente con el que
    arranca la decisión siguiente: la serie forma la escalera acumulada.
    """
    records = []
    for decision in decisions["decisions"]:
        candidates = decision["candidates"]
        winner = next(item for item in candidates if item["candidate_id"] == decision["winner_candidate_id"])
        incumbent = next((item for item in candidates if item.get("is_incumbent")), None)
        paired = winner["paired_bootstrap_90"]
        records.append(
            {
                "variable": decision["variable_id"],
                "value": winner["value"],
                "rank_ic": winner["mean_rank_ic"],
                "advantage": winner["paired_advantage"],
                "rule": decision["selection_rule"],
                "changed": incumbent is not None and winner["candidate_id"] != incumbent["candidate_id"],
                "interval": (
                    f"[{num(paired['ci_low'])}; {num(paired['ci_high'])}]"
                    if paired.get("applicable")
                    else "No aplicable"
                ),
                "candidates": len(candidates),
            }
        )
    return records


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


def write_tables_predictive(paths: Paths, diag: pd.DataFrame, features: pd.DataFrame, attribution: dict, weights: pd.DataFrame) -> None:
    """Tablas sobre datos, features y capacidad predictiva por agente."""
    annual = weights.copy()
    annual["year"] = pd.to_datetime(annual["snapshot_date"]).dt.year
    wide = annual.pivot_table(index="year", columns="agent", values="weight", aggfunc="mean")
    order = [agent for agent in ["risk", "growth", "momentum", "quality", "value"] if agent in wide]
    status = annual.groupby("year")["weight_status"].agg(
        lambda values: "uniforme" if (values == "fallback_equal").all() else "aprendido"
    )
    rows = [
        [str(int(year))] + [num(wide.loc[year, agent], 3) for agent in order] + [status.loc[year]]
        for year in wide.index
    ]
    table(
        paths.tables / "t05_meta_pesos_anual.tex",
        ["Año"] + [tex(agent) for agent in order] + ["Estado"],
        rows,
        "l" + "r" * len(order) + "l",
    )

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


def write_tables_robustness(paths: Paths, summary: dict, robustness: dict, attribution: dict) -> None:
    """Tablas de robustez, atribución factorial y ventanas de evaluación."""
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
    for entry in boot["era_exclusions"]:
        rows.append([f"Excluyendo {tex(entry['excluded_era'])}", num(entry["mean_rank_ic"]), "—", "—", str(int(entry["n_cohorts"]))])
    dispersion = robustness["seed_dispersion"]
    for key, label, digits in [("mean_rank_ic", "Rank-IC entre semillas", 4), ("information_ratio", "IR entre semillas", 3)]:
        entry = dispersion[key]
        rows.append([label, num(entry["median"], digits), f"[{num(entry['min'], digits)}; {num(entry['max'], digits)}]", "—", str(int(dispersion["n_seeds"]))])
    table(
        paths.tables / "t07_eras_bootstrap.tex",
        ["Contraste", "Valor central", "Intervalo 90\\%", "Intervalo 95\\%", "Obs."],
        rows,
        "lrccr",
    )

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
    blocks = {"summary": summary["summary"], "confirmation": summary["summary"]["confirmation"], "full_curve": summary["summary"]["full_curve"]}
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

    rows = []
    for key, label in [("selection", "Selección"), ("confirmation", "Confirmación"), ("full", "Curva completa")]:
        window = attribution["factor_regression"][key]
        rows.append(
            [
                label,
                pct(window["alpha_per_period"]),
                num(window["alpha_t_stat"], 2),
                num(window["r_squared"], 3),
                str(int(window["n_observations"])),
            ]
            + [num(window["loadings"][name], 3) for name in window["loadings"]]
        )
    factor_names = list(attribution["factor_regression"]["selection"]["loadings"].keys())
    table(
        paths.tables / "t07_factores.tex",
        ["Ventana", "Alfa/periodo", "$t$", "$R^2$", "Obs."] + [tex(name.replace("_", " ")) for name in factor_names],
        rows,
    )

    neutralized = attribution["neutralized_rank_ic"]
    rows = [
        ["Rank-IC bruto", num(neutralized["raw_mean_rank_ic"])],
        ["Rank-IC neutralizado", num(neutralized["neutralized_mean_rank_ic"])],
        ["Fracción retenida", pct(neutralized["retained_fraction"])],
        ["Controles aplicados", str(len(neutralized["controls"]))],
        ["Cohortes", str(int(neutralized["n_cohorts"]))],
    ]
    table(paths.tables / "t07_neutralizacion.tex", ["Concepto", "Valor"], rows, "lr")


def write_tables_portfolio(paths: Paths, sweep: pd.DataFrame, baseline: dict, orders: pd.DataFrame, tails: pd.DataFrame) -> None:
    """Tablas del barrido diagnóstico de cartera y de la cola de la señal."""
    frame = sweep.sort_values("geometric_excess_return", ascending=False)
    rows = [
        [
            tex(row.variable_id),
            tex(tex_free(row.base_value)),
            tex(tex_free(row.diagnostic_value)),
            pct(row.geometric_excess_return),
            num(row.information_ratio, 3),
            num(row.annualized_turnover, 2),
            pct(row.max_drawdown),
            num(row.transfer_coefficient, 3),
        ]
        for row in frame.itertuples()
    ]
    baseline_row = [
        r"\textbf{Ganador congelado}",
        "—",
        "—",
        r"\textbf{" + pct(baseline["geometric_excess_return"]) + "}",
        r"\textbf{" + num(baseline["information_ratio"], 3) + "}",
        r"\textbf{" + num(baseline["annualized_turnover"], 2) + "}",
        r"\textbf{" + pct(baseline["max_drawdown"]) + "}",
        r"\textbf{" + num(baseline["transfer_coefficient"], 3) + "}",
    ]
    longtable(
        paths.tables / "t07_cartera_barrido.tex",
        ["Variable", "Base", "Diagnóstico", "Exceso", "IR", "Turnover", "MDD", "Transf."],
        [baseline_row] + rows,
        "Barrido diagnóstico de cartera sobre el ganador congelado, ordenado por exceso geométrico.",
        "tab:cartera-barrido",
        "lllrrrrr",
    )

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

    rows = []
    for index, record in enumerate(decision_records(decisions), start=1):
        rows.append(
            [
                str(index),
                tex(record["variable"]),
                tex(tex_free(record["value"])),
                num(record["rank_ic"]),
                num(record["advantage"]),
                record["interval"],
                "Sí" if record["changed"] else "—",
            ]
        )
    table(
        paths.tables / "t06_escalera.tex",
        ["\\#", "Variable", "Ganador", "Rank-IC", "Ventaja", "IC pareado 90\\%", "Cambió"],
        rows,
        "llrrrcc",
    )


def draw_meta_weights(weights: pd.DataFrame, output: Path) -> None:
    weights = weights.copy()
    weights["snapshot_date"] = pd.to_datetime(weights["snapshot_date"])
    wide = weights.pivot(index="snapshot_date", columns="agent", values="weight").sort_index()
    order = [a for a in ["quality", "value", "growth", "momentum", "risk"] if a in wide]
    colors = [SLATE, GOLD, TEAL, RED, NAVY]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.stackplot(wide.index, *[wide[a] for a in order], labels=order, colors=colors[: len(order)], alpha=0.93)
    ax.set(title="Evolución de los pesos del meta-agente", ylabel="Peso", ylim=(0, 1))
    legend_below(ax)
    save(fig, output)


def draw_agent_ic(diag: pd.DataFrame, output: Path) -> None:
    selection = diag[pd.to_datetime(diag["prediction_date"]).dt.year.le(2024)]
    summary = selection.groupby("agent")["rank_ic"].agg(["mean", "std", "count"]).sort_values("mean")
    names = [str(x).replace("_", " ") for x in summary.index]
    err = summary["std"] / np.sqrt(summary["count"])
    colors = [NAVY if a == "meta_final" else TEAL if a == "risk" else SLATE for a in summary.index]
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.barh(names, summary["mean"], xerr=err, color=colors, capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Capacidad predictiva por agente", xlabel="Rank-IC medio (2015–2024)")
    save(fig, output)


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


def draw_annual_alpha(annual: pd.DataFrame, output: Path) -> None:
    annual = annual.copy()
    colors = [GOLD if y >= 2025 else GREEN if x >= 0 else RED for y, x in zip(annual.year, annual.alpha)]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(annual["year"].astype(str), annual["alpha"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Alfa anual de la cartera frente a SPY", ylabel="Alfa anual")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    save(fig, output)


def draw_profiles(profiles: pd.DataFrame, output: Path) -> None:
    frame = profiles.sort_values("geometric_excess_return")
    colors = [NAVY if x == "balanced" else TEAL if v >= 0 else RED for x, v in zip(frame.profile, frame.geometric_excess_return)]
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.barh(frame["profile"], frame["geometric_excess_return"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Exceso geométrico por perfil", xlabel="Exceso geométrico anual")
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
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


def draw_placebos(robustness: dict, observed: float, output: Path) -> None:
    values = [entry["summary"]["mean_rank_ic"] for entry in robustness["label_placebos"]]
    labels = [str(entry["seed"]) for entry in robustness["label_placebos"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(labels, values, color=SLATE, label="Placebo")
    ax.axhline(observed, color=NAVY, linewidth=2, label="Sistema observado")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Placebos de etiqueta frente al Rank-IC observado", xlabel="Semilla del placebo", ylabel="Rank-IC medio")
    legend_below(ax)
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


def diverging_colors(values, vmax: float) -> list[str]:
    """Rojo para negativos, azul para positivos, con intensidad proporcional."""
    colors = []
    for value in values:
        weight = min(abs(float(value)) / vmax, 1.0) if vmax else 0.0
        base = np.array(mpl.colors.to_rgb(NAVY if value >= 0 else RED))
        colors.append(mpl.colors.to_hex(1 - weight * (1 - base)))
    return colors


def draw_rankic_era_heatmap(diag: pd.DataFrame, output: Path) -> None:
    matrix = agent_era_matrix(diag)
    limit = float(np.abs(matrix.to_numpy()).max())
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.imshow(matrix.to_numpy(), cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set(
        xticks=range(matrix.shape[1]),
        yticks=range(matrix.shape[0]),
        title="Rank-IC por señal y era",
    )
    ax.set_xticklabels(matrix.columns)
    ax.set_yticklabels([str(name).replace("_", " ") for name in matrix.index])
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iat[row, col]
            shade = "white" if abs(value) > 0.62 * limit else "black"
            ax.text(col, row, num(value, 4), ha="center", va="center", color=shade, fontsize=8)
    ax.grid(False)
    ax.axvline(2.5, color="black", linewidth=1.4)
    ax.text(3.0, -0.85, "Reservada", ha="center", fontsize=8, color=GOLD, fontweight="bold")
    save(fig, output)


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


def draw_meta_concentration(tails: pd.DataFrame, output: Path) -> None:
    frame = tails.copy()
    frame["prediction_date"] = pd.to_datetime(frame["prediction_date"])
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.plot(frame["prediction_date"], frame["meta_weight_concentration_hhi"], color=NAVY, linewidth=1.8, label="Concentración (HHI)")
    ax.plot(frame["prediction_date"], frame["meta_weight_max"], color=TEAL, linewidth=1.5, label="Peso máximo")
    ax.axhline(0.5, color=GOLD, linewidth=1.2, linestyle="--", label="Tope de la variante acotada")
    ax.set(title="Concentración de los pesos del meta-agente", ylabel="Proporción")
    legend_below(ax)
    save(fig, output)


def draw_decision_ladder(decisions: dict, output: Path) -> None:
    records = decision_records(decisions)
    labels = [record["variable"].replace("_", " ") for record in records]
    values = [record["rank_ic"] for record in records]
    changed = [record["changed"] for record in records]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.step(range(len(values)), values, where="post", color=NAVY, linewidth=1.8)
    ax.scatter(
        range(len(values)),
        values,
        s=[58 if flag else 26 for flag in changed],
        c=[GOLD if flag else SLATE for flag in changed],
        zorder=3,
    )
    for index, record in enumerate(records):
        if record["changed"]:
            ax.annotate(num(record["rank_ic"]), (index, record["rank_ic"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7.5)
    ax.set(
        title="Rank-IC acumulado tras cada decisión secuencial",
        ylabel="Rank-IC del incumbente",
        xticks=range(len(labels)),
    )
    ax.set_xticklabels(labels, rotation=62, ha="right", fontsize=7)
    ax.margins(y=0.16)   # deja aire para las anotaciones sobre los escalones
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


def draw_factor_loadings(attribution: dict, output: Path) -> None:
    windows = [("selection", "Selección 2015–2024"), ("confirmation", "Confirmación 2025–2026")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
    for ax, (key, title) in zip(axes, windows):
        window = attribution["factor_regression"][key]
        names = list(window["loadings"].keys())
        values = [window["loadings"][name] for name in names]
        stats = [window["loading_t_stats"][name] for name in names]
        colors = [NAVY if abs(stat) >= 2 else SLATE for stat in stats]
        ax.barh([name.replace("_", " ") for name in names], values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(title=f"{title}\n$R^2={num(window['r_squared'], 3)}$", xlabel="Carga factorial")
    axes[0].set_ylabel("Factor")
    fig.suptitle("Cargas factoriales (azul: $|t|\\geq2$)", fontsize=10)
    save(fig, output)


def draw_transfer_by_year(attribution: dict, output: Path) -> None:
    frame = pd.DataFrame(attribution["transfer"]["by_year"])
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    sizes = 40 + 5200 * frame["cost"]
    colors = [GREEN if value >= 0 else RED for value in frame["excess"]]
    ax.scatter(frame["turnover"], frame["excess"], s=sizes, c=colors, alpha=0.72, zorder=3)
    for row in frame.itertuples():
        ax.annotate(str(int(row.year)), (row.turnover, row.excess), xytext=(6, 5), textcoords="offset points", fontsize=7.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Rotación, exceso y coste por año (el área es el coste)", xlabel="Turnover anual", ylabel="Exceso sobre SPY")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    save(fig, output)


def draw_portfolio_sweep(sweep: pd.DataFrame, baseline: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.scatter(sweep["annualized_turnover"], sweep["geometric_excess_return"], color=SLATE, s=42, zorder=3, label="Variante diagnóstica")
    ax.scatter(
        baseline["annualized_turnover"],
        baseline["geometric_excess_return"],
        color=NAVY,
        s=125,
        marker="*",
        zorder=4,
        label="Ganador congelado",
    )
    best = sweep.loc[sweep["geometric_excess_return"].idxmax()]
    ax.annotate(
        f"{best.variable_id}\n→ {tex_free(best.diagnostic_value)}",
        (best.annualized_turnover, best.geometric_excess_return),
        xytext=(-8, -22),
        textcoords="offset points",
        ha="right",
        fontsize=7.5,
        color=GREEN,
    )
    ax.axhline(baseline["geometric_excess_return"], color=NAVY, linewidth=1, linestyle="--", alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Barrido diagnóstico de cartera sobre el mismo ganador", xlabel="Turnover anualizado", ylabel="Exceso geométrico")
    # El rango útil es de apenas tres puntos porcentuales: sin decimal, todas
    # las etiquetas colapsarían en el mismo «2 %».
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=1))
    ax.margins(y=0.14)
    legend_below(ax)
    save(fig, output)


def draw_order_reasons(orders: pd.DataFrame, output: Path) -> None:
    frame = orders.copy()
    frame["flow"] = (frame["weight_after"] - frame["weight_before"]).abs()
    grouped = frame.groupby("reason").agg(share=("flow", "sum"), count=("flow", "size")).sort_values("share")
    grouped["share"] /= grouped["share"].sum()
    labels = [str(name).replace("_", " ") for name in grouped.index]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.barh(labels, grouped["share"], color=[NAVY if value == grouped["share"].max() else SLATE for value in grouped["share"]])
    for index, (share, count) in enumerate(zip(grouped["share"], grouped["count"])):
        ax.text(share, index, f"  {pct(share, 1).replace(chr(92) + '%', ' %')} · {count} órdenes", va="center", fontsize=8)
    ax.set(title="Descomposición del turnover por motivo de orden", xlabel="Proporción del flujo total", xlim=(0, grouped["share"].max() * 1.32))
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
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


def draw_feature_blocks(features: pd.DataFrame, output: Path) -> None:
    grouped = (
        features.groupby("block")
        .agg(n=("feature", "size"), coverage=("coverage", "mean"), rank_ic=("univariate_rank_ic", "mean"))
        .sort_values("rank_ic")
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    colors = diverging_colors(grouped["rank_ic"], float(grouped["rank_ic"].abs().max()))
    ax.barh([str(name).replace("_", " ") for name in grouped.index], grouped["rank_ic"], color=colors)
    for index, (value, count, coverage) in enumerate(zip(grouped["rank_ic"], grouped["n"], grouped["coverage"])):
        label_x = value + 0.0006 if value >= 0 else 0.0006
        ax.text(label_x, index, f"{count} vars · cob. {100 * coverage:.0f} %".replace(".", ","), va="center", ha="left", fontsize=7.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Rank-IC univariante medio por bloque de features", xlabel="Rank-IC univariante medio")
    ax.margins(x=0.22)
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


def draw_robustness(rows: list[dict], output: Path) -> None:
    labels = [row["name"] for row in rows]
    verdict = [int(row["passes"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.scatter(range(len(labels)), verdict, s=150, c=[GREEN if x else RED for x in verdict], marker="o")
    for index, value in enumerate(verdict):
        ax.text(index, value, "✓" if value else "✕", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax.set(xticks=range(len(labels)), xticklabels=labels, ylim=(-0.35, 1.35), yticks=[0, 1], yticklabels=["No supera", "Supera"], title="Resumen de contrastes de robustez")
    ax.tick_params(axis="x", rotation=25)
    save(fig, output)


def draw_coverage(attribution: dict, output: Path) -> None:
    coverage = pd.DataFrame(attribution["universe_coverage"])
    fig, ax1 = plt.subplots(figsize=(7.2, 3.5))
    ax1.plot(coverage["year"], coverage["distinct_tickers"], color=NAVY, linewidth=2, label="Tickers distintos")
    ax1.set(ylabel="Tickers distintos", title="Cobertura histórica del universo")
    ax2 = ax1.twinx()
    ax2.plot(coverage["year"], coverage["usable_fraction"], color=TEAL, linewidth=1.8, label="Fracción utilizable")
    ax2.set_ylabel("Fracción utilizable")
    ax2.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    ax2.set_ylim(0.9, 1.005)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    save(fig, output)


def write_tables(paths: Paths, summary: dict, robustness: dict, attribution: dict, diag: pd.DataFrame, annual: pd.DataFrame, profiles: pd.DataFrame, decisions: dict) -> None:
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

    profile_rows = [[tex(r.profile), num(r.mean_rank_ic), pct(r.geometric_excess_return), num(r.information_ratio, 3), pct(r.annualized_turnover), pct(r.mean_cash_weight)] for r in profiles.sort_values("information_ratio", ascending=False).itertuples()]
    table(paths.tables / "t07_perfiles.tex", ["Perfil", "Rank-IC", "Exceso", "IR", "Turnover", "Efectivo"], profile_rows)

    decision_rows: list[list[str]] = []
    for decision in decisions["decisions"]:
        winner = next(candidate for candidate in decision["candidates"] if candidate["candidate_id"] == decision["winner_candidate_id"])
        paired = winner["paired_bootstrap_90"]
        if paired["applicable"]:
            interval = f"[{num(paired['ci_low'])}; {num(paired['ci_high'])}]"
        else:
            interval = "No aplicable"
        decision_rows.append([tex(decision["variable_id"]), tex(decision["winner_value"]), num(winner["paired_advantage"]), interval])
    table(paths.tables / "t06_decisiones.tex", ["Variable", "Ganador", "Ventaja", r"IC pareado 90\%"], decision_rows, "llrr")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", required=True)
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
    equity = pd.read_parquet(paths.evidence / "equity.parquet")
    annual = pd.read_parquet(paths.evidence / "annual_metrics.parquet")
    profiles = pd.read_parquet(paths.study / "profile_comparison.parquet")
    sweep = pd.read_parquet(paths.study / "portfolio_comparison.parquet")
    orders = pd.read_parquet(paths.evidence / "orders.parquet")
    tails = pd.read_parquet(paths.evidence / "rank_tail_diagnostics.parquet")
    health = pd.read_parquet(paths.evidence / "signal_health.parquet")
    features = load_features(paths)

    draw_meta_weights(weights, paths.figures / "f05_meta_pesos.png")
    draw_agent_ic(diag, paths.figures / "f05_agentes_rankic.png")
    draw_rank_ic(summary, paths.figures / "f07_rankic_serie.png")
    draw_equity(equity, paths.figures / "f07_equity.png")
    draw_drawdown(equity, paths.figures / "f07_drawdown.png")
    draw_annual_alpha(annual, paths.figures / "f07_alfa_anual.png")
    draw_profiles(profiles, paths.figures / "f07_perfiles_exceso.png")
    draw_profile_tradeoff(profiles, paths.figures / "f07_perfiles_tradeoff.png")
    draw_placebos(robustness, summary["summary"]["mean_rank_ic"], paths.figures / "f07_placebos.png")
    draw_robustness(build_robustness_rows(robustness, attribution), paths.figures / "f07_robustez.png")
    draw_coverage(attribution, paths.figures / "f03_cobertura.png")

    # Activos añadidos para la versión extendida del manuscrito.
    draw_feature_blocks(features, paths.figures / "f03_features_bloques.png")
    draw_rankic_era_heatmap(diag, paths.figures / "f05_rankic_era.png")
    draw_meta_weights_annual(weights, paths.figures / "f05_pesos_anual.png")
    draw_meta_concentration(tails, paths.figures / "f05_concentracion.png")
    draw_decision_ladder(decisions, paths.figures / "f06_escalera.png")
    draw_bootstrap_forest(robustness, paths.figures / "f07_bootstrap.png")
    draw_permutation(robustness, paths.figures / "f07_permutacion.png")
    draw_random_portfolios(robustness, paths.figures / "f07_aleatorias.png")
    draw_seed_dispersion(robustness, paths.figures / "f07_semillas.png")
    draw_factor_loadings(attribution, paths.figures / "f07_factores.png")
    draw_transfer_by_year(attribution, paths.figures / "f07_transferencia.png")
    draw_portfolio_sweep(sweep, summary["summary"], paths.figures / "f07_barrido_cartera.png")
    draw_order_reasons(orders, paths.figures / "f07_ordenes.png")
    draw_tail_spread(tails, paths.figures / "f07_cola.png")
    draw_signal_health(health, paths.figures / "f07_salud.png")

    write_tables(paths, summary, robustness, attribution, diag, annual, profiles, decisions)
    write_tables_predictive(paths, diag, features, attribution, weights)
    write_tables_robustness(paths, summary, robustness, attribution)
    write_tables_portfolio(paths, sweep, summary["summary"], orders, tails)
    write_tables_catalog(paths, catalog, winner, decisions)

    manifest = {
        "study_id": args.study_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [
            "catalog_snapshot.json", "winner.json",
            "evidence/summary.json", "robustness.json", "attribution.json", "decisions.json",
            "evidence/meta_weights.parquet", "evidence/rank_ic_diagnostics.parquet",
            "evidence/equity.parquet", "evidence/annual_metrics.parquet", "profile_comparison.parquet",
            "portfolio_comparison.parquet", "evidence/orders.parquet",
            "evidence/rank_tail_diagnostics.parquet", "evidence/signal_health.parquet",
            "evidence/feature_catalog.json", "evidence/feature_diagnostics.parquet",
        ],
        "figures": sorted(path.name for path in paths.figures.glob("f*.png")),
        "tables": sorted(path.name for path in paths.tables.glob("t*.tex")),
    }
    (LATEX / "asset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
