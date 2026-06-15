"""Single-entry pipeline for the GARP AI portfolio system."""

from __future__ import annotations

import logging

from environment import Settings, ensure_directories
from module.backtest.engine import run_backtest
from module.common.logging import setup_logging
from module.data_download.pipeline import download_raw_data
from module.dataset_builder.master import build_master_dataset
from module.features.engineering import build_features
from module.ml.model import train_and_score
from module.research.openai_research import build_openai_research
from module.report.final_report import build_final_report
from module.viewer.html import build_viewer
from module.watchlist.engine import build_watchlist


log = logging.getLogger(__name__)


def main() -> None:
    ensure_directories()
    settings = Settings()
    setup_logging(settings.run_dir / "logs" / "pipeline.log")
    log.info(
        "Run mode=%s dev_mode=%s tickers=%s fundamental_frequency=%s price_frequency=%s portfolio_review_frequency=%s",
        settings.run_mode,
        settings.dev_mode,
        len(settings.tickers),
        settings.fundamental_review_frequency,
        settings.price_update_frequency,
        settings.review_frequency,
    )
    log.info(
        "Model/backtest settings walk_forward=%s label_horizon_months=%s min_train_years=%s max_train_years=%s transaction_cost_bps=%.2f slippage_bps=%.2f",
        settings.walk_forward_scoring,
        settings.walk_forward_label_horizon_months,
        settings.min_walk_forward_training_years,
        settings.max_walk_forward_training_years,
        settings.transaction_cost_bps,
        settings.slippage_bps,
    )

    if settings.run_mode in {"download", "full"}:
        download_raw_data(settings)
    if settings.run_mode in {"dataset", "full"}:
        build_master_dataset(settings)
    if settings.run_mode in {"features", "full"}:
        build_features()
    if settings.run_mode in {"ml", "full"}:
        train_and_score(settings)
    if settings.run_mode in {"watchlist", "full"}:
        build_watchlist(settings)
    if settings.run_mode in {"research_ai", "full"}:
        build_openai_research(settings)
    if settings.run_mode in {"backtest", "full"}:
        run_backtest(settings)
    if settings.run_mode in {"viewer", "full"}:
        viewer_dir = build_viewer(settings)
        log.info("Viewer ready: %s", viewer_dir / "index.html")
    if settings.run_mode in {"report", "full"}:
        report_path = build_final_report(settings)
        log.info("Final report ready: %s", report_path)


if __name__ == "__main__":
    main()
