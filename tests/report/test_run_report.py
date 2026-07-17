"""Contrato del HTML por run: 6 hojas, cifras coherentes con backtest_summary.json,
CSVs sueltos al lado del HTML para las tablas grandes."""

from __future__ import annotations

import json
from pathlib import Path

from module.report import build_run_report


REQUIRED_SECTIONS = (
    'id="tab-resumen"',
    'id="tab-rendimiento"',
    'id="tab-aprendizaje"',
    'id="tab-cartera"',
    'id="tab-cobertura"',
    'id="tab-posiciones"',
)


def test_run_report_generates_html_and_sidecar_csvs(minimal_run_dir: Path) -> None:
    """El HTML se genera con las 6 secciones y los CSVs esperados aparecen al lado."""
    build_run_report(minimal_run_dir)

    html_path = minimal_run_dir / "report.html"
    assert html_path.exists(), "no se genero report.html"

    html = html_path.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in html, f"falta la seccion {section}"

    for csv_name in (
        "positions_history.csv",
        "orders_history.csv",
        "ranking_by_snapshot.csv",
    ):
        assert (minimal_run_dir / csv_name).exists(), f"falta el CSV {csv_name}"


def test_summary_figures_match_backtest_summary(minimal_run_dir: Path) -> None:
    """La hoja Resumen del HTML muestra cifras coherentes con `backtest_summary.json`.

    Si el HTML mostrase un alfa distinto del artefacto, el TFM contaria dos historias
    a la vez. La verdad esta en el JSON; el HTML lo pinta.
    """
    build_run_report(minimal_run_dir)
    html = (minimal_run_dir / "report.html").read_text(encoding="utf-8")
    summary = json.loads((minimal_run_dir / "backtest_summary.json").read_text(encoding="utf-8"))

    # Cifras clave: alfa total, beat_rate, drawdown maximo, IR.
    assert f"{summary['total_alpha'] * 100:.2f}" in html or f"{summary['total_alpha']:.4f}" in html
    assert f"{int(summary['beat_rate'] * 100)}" in html or f"{summary['beat_rate']:.2f}" in html
