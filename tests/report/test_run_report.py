from __future__ import annotations

from module.ui.reports import build_run_report


def test_run_report_generates_html_and_sidecar_csvs(minimal_run_dir) -> None:
    report = build_run_report(minimal_run_dir)

    assert report.exists()
    assert "Rank-IC" in report.read_text(encoding="utf-8")
    assert (minimal_run_dir / "positions_history.csv").exists()
    assert (minimal_run_dir / "annual_metrics.csv").exists()
