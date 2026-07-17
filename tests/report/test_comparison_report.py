"""Contrato del HTML del barrido: 5 hojas, ranking correcto por metrica compuesta,
enlaces a los HTML de cada run."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from module.report import build_comparison_report


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


def test_comparison_report_ranks_scenarios_by_composite_stability(tmp_path: Path) -> None:
    """El ganador es el mas estable, no el que mas alfa total tiene."""
    scenarios_root = tmp_path / "escenarios"
    scenarios_root.mkdir()

    # A: alfa media alta pero un ano brutal y otros mediocres (inestable).
    _write_scenario(scenarios_root, "spike_year", {
        "total_alpha": 1.20, "beat_rate": 0.5, "median_alpha": 0.01,
        "worst_year_alpha": -0.15, "max_drawdown": 0.40, "information_ratio": 0.6,
    }, {2000: 1.0, 2001: -0.10, 2002: 0.02, 2003: 0.03, 2004: -0.05})

    # B: alfa mediana baja pero todos los anos positivos, drawdown pequeno (estable).
    _write_scenario(scenarios_root, "steady", {
        "total_alpha": 0.15, "beat_rate": 1.0, "median_alpha": 0.03,
        "worst_year_alpha": 0.015, "max_drawdown": 0.08, "information_ratio": 0.9,
    }, {2000: 0.03, 2001: 0.02, 2002: 0.03, 2003: 0.025, 2004: 0.02})

    # C: intermedio.
    _write_scenario(scenarios_root, "middle", {
        "total_alpha": 0.30, "beat_rate": 0.6, "median_alpha": 0.02,
        "worst_year_alpha": -0.04, "max_drawdown": 0.20, "information_ratio": 0.4,
    }, {2000: 0.10, 2001: -0.04, 2002: 0.05, 2003: 0.08, 2004: 0.02})

    build_comparison_report(scenarios_root, dev_era=(1990, 2015))

    comparison_html = (scenarios_root / "comparison.html").read_text(encoding="utf-8")
    for section in COMPARISON_SECTIONS:
        assert section in comparison_html, f"falta {section}"

    selection = json.loads((scenarios_root / "selection.json").read_text(encoding="utf-8"))
    assert selection["winner"] == "steady", (
        f"esperaba 'steady' como ganador; salio {selection['winner']!r}"
    )

    summary = pd.read_parquet(scenarios_root / "scenarios_summary.parquet")
    steady_rank = int(summary.loc[summary["scenario"] == "steady", "composite_rank_mean"].iloc[0] * 1000)
    spike_rank = int(summary.loc[summary["scenario"] == "spike_year", "composite_rank_mean"].iloc[0] * 1000)
    assert steady_rank < spike_rank, "steady debe tener mejor (menor) rango medio que spike_year"
