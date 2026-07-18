"""Contrato del HTML del barrido: 5 hojas, ranking correcto por metrica compuesta,
enlaces a los HTML de cada run."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from module.ui.reports import build_comparison_report


COMPARISON_SECTIONS = (
    'id="tab-resumen"',
    'id="tab-anual"',
    'id="tab-sensibilidad"',
    'id="tab-seleccion"',
    'id="tab-todos"',
)


def _write_scenario(scenarios_root: Path, name: str, summary: dict, annual_alpha: dict) -> Path:
    scenario_dir = scenarios_root / name
    run_dir = scenario_dir / "agents" / f"ridge-{name}"
    run_dir.mkdir(parents=True)
    (run_dir / "backtest_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "report.html").write_text("<html></html>", encoding="utf-8")
    annual = pd.DataFrame([
        {"year": year, "portfolio_return": alpha + 0.05, "benchmark_return": 0.05,
         "alpha": alpha, "beats_benchmark": alpha > 0, "max_drawdown_year": 0.05,
         "information_ratio_year": 0.5}
        for year, alpha in annual_alpha.items()
    ])
    annual.to_parquet(run_dir / "annual_metrics.parquet")
    (scenario_dir / "scenario_config.json").write_text(
        json.dumps({"name": name, "overrides": {}}), encoding="utf-8"
    )
    return scenario_dir


def test_comparison_report_ranks_scenarios_by_learning(tmp_path: Path) -> None:
    """El ganador es el que APRENDE (rank-IC), no el que mas alfa tiene."""
    scenarios_root = tmp_path / "escenarios"
    scenarios_root.mkdir()

    # high_alpha: alfa alto pero rank-IC nulo. Ruido afortunado, no aprende.
    _write_scenario(scenarios_root, "high_alpha", {
        "mean_rank_ic": 0.002, "rank_ic_positive_fraction": 0.49, "rank_ic_std": 0.14,
        "beat_rate": 0.6, "max_drawdown": 0.40, "annualized_alpha": 0.25,
        "median_alpha": 0.05, "worst_year_alpha": -0.20, "information_ratio": 0.6,
    }, {2000: 1.0, 2001: -0.10, 2002: 0.02, 2003: 0.03, 2004: -0.05})

    # learner: aprende de verdad (rank-IC positivo y consistente), alfa modesto.
    _write_scenario(scenarios_root, "learner", {
        "mean_rank_ic": 0.06, "rank_ic_positive_fraction": 0.68, "rank_ic_std": 0.09,
        "beat_rate": 0.6, "max_drawdown": 0.15, "annualized_alpha": 0.03,
        "median_alpha": 0.02, "worst_year_alpha": -0.03, "information_ratio": 0.5,
    }, {2000: 0.03, 2001: 0.02, 2002: 0.03, 2003: 0.025, 2004: 0.02})

    # middle.
    _write_scenario(scenarios_root, "middle", {
        "mean_rank_ic": 0.02, "rank_ic_positive_fraction": 0.55, "rank_ic_std": 0.12,
        "beat_rate": 0.6, "max_drawdown": 0.20, "annualized_alpha": 0.10,
        "median_alpha": 0.03, "worst_year_alpha": -0.04, "information_ratio": 0.4,
    }, {2000: 0.10, 2001: -0.04, 2002: 0.05, 2003: 0.08, 2004: 0.02})

    build_comparison_report(scenarios_root)

    comparison_html = (scenarios_root / "comparison.html").read_text(encoding="utf-8")
    for section in COMPARISON_SECTIONS:
        assert section in comparison_html, f"falta {section}"

    selection = json.loads((scenarios_root / "selection.json").read_text(encoding="utf-8"))
    assert selection["winner"] == "learner", (
        f"esperaba 'learner' como ganador; salio {selection['winner']!r}"
    )

    summary = pd.read_parquet(scenarios_root / "scenarios_summary.parquet")
    learner_rank = float(summary.loc[summary["scenario"] == "learner", "composite_rank_mean"].iloc[0])
    high_alpha_rank = float(summary.loc[summary["scenario"] == "high_alpha", "composite_rank_mean"].iloc[0])
    assert learner_rank < high_alpha_rank, "learner debe tener mejor (menor) rango medio"
