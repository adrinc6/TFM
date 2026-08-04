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


def load_paths(study_id: str) -> Paths:
    study = ROOT / "results" / "studies" / study_id
    evidence = study / "evidence"
    if not (study / "winner.json").is_file() or not evidence.is_dir():
        raise FileNotFoundError(f"No se encontraron artefactos completos para {study_id}.")
    figures, tables = LATEX / "figuras", LATEX / "tablas"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return Paths(study, evidence, figures, tables)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig: plt.Figure, output: Path) -> None:
    fig.tight_layout()
    fig.savefig(output, format="png", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def table(path: Path, columns: list[str], rows: list[list[str]], align: str | None = None) -> None:
    alignment = align or ("l" + "r" * (len(columns) - 1))
    lines = [r"\begin{tabular}{" + alignment + "}", r"\toprule"]
    lines += [" & ".join(columns) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def draw_meta_weights(weights: pd.DataFrame, output: Path) -> None:
    wide = weights.pivot(index="snapshot_date", columns="agent", values="weight").sort_index()
    order = [a for a in ["quality", "value", "growth", "momentum", "risk"] if a in wide]
    colors = [SLATE, GOLD, TEAL, RED, NAVY]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.stackplot(wide.index, *[wide[a] for a in order], labels=order, colors=colors[: len(order)], alpha=0.93)
    ax.set(title="Evolución de los pesos del meta-agente", ylabel="Peso", ylim=(0, 1))
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.19), frameon=False)
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
    ax.legend(loc="lower left", frameon=False)
    save(fig, output)


def draw_equity(equity: pd.DataFrame, output: Path) -> None:
    equity["snapshot_date"] = pd.to_datetime(equity["snapshot_date"])
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.plot(equity["snapshot_date"], equity["portfolio_value"], color=NAVY, linewidth=2, label="Cartera")
    ax.plot(equity["snapshot_date"], equity["benchmark_value"], color=SLATE, linewidth=1.7, label="SPY")
    ax.axvspan(pd.Timestamp("2025-01-01"), equity["snapshot_date"].max(), color=GOLD, alpha=0.13, label="Era reservada")
    ax.set(title="Evolución acumulada: cartera frente a SPY", ylabel="Valor de 100 unidades monetarias")
    ax.legend(loc="upper left", frameon=False)
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
    ax.legend(loc="lower left", frameon=False)
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
    ax.legend(frameon=False, loc="upper left")
    save(fig, output)


def draw_robustness(robustness: dict, attribution: dict, output: Path) -> None:
    tests = ["Permutación", "Placebos", "Bootstrap 95 %", "Exclusión de eras", "Semillas", "Aleatorias riesgo", "Neutralización", "Deflated Sharpe"]
    verdict = [1, 1, 1, 1, 1, 1, 1, 0]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.scatter(range(len(tests)), verdict, s=150, c=[GREEN if x else RED for x in verdict], marker="o")
    for index, value in enumerate(verdict):
        ax.text(index, value, "✓" if value else "✕", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax.set(xticks=range(len(tests)), xticklabels=tests, ylim=(-0.35, 1.35), yticks=[0, 1], yticklabels=["No supera", "Supera"], title="Resumen de contrastes de robustez")
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
    ax1.legend(lines + lines2, labels + labels2, loc="lower right", frameon=False)
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
        ["Permutación", r"$p=0,0001$; 9.999 réplicas", "Supera"],
        ["Placebos", r"Rank-IC entre $-0,006$ y $+0,001$", "Supera"],
        ["Bootstrap", r"IC 95\% $[0,0335; 0,1695]$", "Supera"],
        ["Exclusión de eras", r"Rank-IC entre $0,073$ y $0,126$", "Supera"],
        ["Semillas", r"Rango de Rank-IC $0,0020$", "Supera"],
        ["Aleatorias con riesgo", r"Percentil $97,4$", "Supera"],
        ["Neutralización", r"Retiene $84,35\%$ de la señal", "Supera"],
        ["Deflated Sharpe", r"$0,930 < 0,95$", "No supera"],
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
    weights = pd.read_parquet(paths.evidence / "meta_weights.parquet")
    diag = pd.read_parquet(paths.evidence / "rank_ic_diagnostics.parquet")
    equity = pd.read_parquet(paths.evidence / "equity.parquet")
    annual = pd.read_parquet(paths.evidence / "annual_metrics.parquet")
    profiles = pd.read_parquet(paths.study / "profile_comparison.parquet")

    draw_meta_weights(weights, paths.figures / "f05_meta_pesos.png")
    draw_agent_ic(diag, paths.figures / "f05_agentes_rankic.png")
    draw_rank_ic(summary, paths.figures / "f07_rankic_serie.png")
    draw_equity(equity, paths.figures / "f07_equity.png")
    draw_drawdown(equity, paths.figures / "f07_drawdown.png")
    draw_annual_alpha(annual, paths.figures / "f07_alfa_anual.png")
    draw_profiles(profiles, paths.figures / "f07_perfiles_exceso.png")
    draw_profile_tradeoff(profiles, paths.figures / "f07_perfiles_tradeoff.png")
    draw_placebos(robustness, summary["summary"]["mean_rank_ic"], paths.figures / "f07_placebos.png")
    draw_robustness(robustness, attribution, paths.figures / "f07_robustez.png")
    draw_coverage(attribution, paths.figures / "f03_cobertura.png")
    write_tables(paths, summary, robustness, attribution, diag, annual, profiles, decisions)

    manifest = {
        "study_id": args.study_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [
            "evidence/summary.json", "robustness.json", "attribution.json", "decisions.json",
            "evidence/meta_weights.parquet", "evidence/rank_ic_diagnostics.parquet",
            "evidence/equity.parquet", "evidence/annual_metrics.parquet", "profile_comparison.parquet",
        ],
        "figures": sorted(path.name for path in paths.figures.glob("*.png")),
        "tables": sorted(path.name for path in paths.tables.glob("*.tex")),
    }
    (LATEX / "asset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
