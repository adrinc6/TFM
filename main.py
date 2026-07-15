"""Single-entry pipeline for the GARP AI portfolio system."""

from __future__ import annotations

import logging
import time

from environment import Settings, ensure_directories
from module.backtest import run_backtest
from module.ingest.pipeline import download_raw_data
from module.dataset import build_master_dataset
from module.features.pipeline import build_features
from module.ml import train_and_score
from module.research.ai import build_openai_research
from module.report import build_final_report
from module.strategy.selection import build_watchlist
from module.utils import setup_logging
from module.viewer import build_viewer


log = logging.getLogger(__name__)


def main() -> None:
    ensure_directories()
    settings = Settings()

    # Modo experimentos: en vez del pipeline normal, barre los escenarios del fichero indicado en
    # EXPERIMENTS_FILE (o "todos") y escribe la comparación en results/experiments/<fecha>/.
    if settings.run_mode == "experiments":
        from module.experiments import load_scenarios, run_experiment

        setup_logging()
        log.info("RUN_MODE=experiments: cargando escenarios de %s", settings.experiments_file)
        scenarios = load_scenarios(settings.experiments_file)
        exp_dir = run_experiment(scenarios, settings, resume=settings.experiments_resume)
        log.info("Experimentos completos. Informe: %s", exp_dir / "index.html")
        return

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

    pipeline_start = time.perf_counter()

    def _stage(name: str, enabled: bool, fn) -> None:
        if not enabled:
            return
        started = time.perf_counter()
        log.info("STAGE START: %s", name)
        fn()
        log.info("STAGE DONE: %s in %.1fs (total elapsed %.1fs)", name, time.perf_counter() - started, time.perf_counter() - pipeline_start)

    _stage("download", settings.run_mode in {"download", "full"}, lambda: download_raw_data(settings))
    _stage("dataset", settings.run_mode in {"dataset", "full"}, lambda: build_master_dataset(settings))
    _stage("features", settings.run_mode in {"features", "full"}, build_features)
    _stage("ml", settings.run_mode in {"ml", "full"}, lambda: train_and_score(settings))
    _stage("watchlist", settings.run_mode in {"watchlist", "full"}, lambda: build_watchlist(settings))
    _stage("research_ai", settings.run_mode in {"research_ai", "full"}, lambda: build_openai_research(settings))
    _stage("backtest", settings.run_mode in {"backtest", "full"}, lambda: run_backtest(settings))

    if settings.run_mode in {"viewer", "full"}:
        started = time.perf_counter()
        log.info("STAGE START: viewer")
        viewer_dir = build_viewer(settings)
        log.info("Viewer ready: %s", viewer_dir / "index.html")
        log.info("STAGE DONE: viewer in %.1fs (total elapsed %.1fs)", time.perf_counter() - started, time.perf_counter() - pipeline_start)
    if settings.run_mode in {"report", "full"}:
        started = time.perf_counter()
        log.info("STAGE START: report")
        report_path = build_final_report(settings)
        log.info("Final report ready: %s", report_path)
        log.info("STAGE DONE: report in %.1fs (total elapsed %.1fs)", time.perf_counter() - started, time.perf_counter() - pipeline_start)

    log.info("PIPELINE COMPLETE in %.1fs", time.perf_counter() - pipeline_start)


if __name__ == "__main__":
    main()
