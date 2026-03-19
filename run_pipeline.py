# =============================================================================
# run_pipeline.py — Script de ejecución mínima del pipeline ML
# =============================================================================
"""
Punto de entrada simplificado para ejecutar el pipeline completo.

Uso:
    python run_pipeline.py                  # ejecución normal
    python run_pipeline.py --skip-download  # reutiliza datos ya descargados
    python run_pipeline.py --skip-backtest  # solo genera features y fold live
    python run_pipeline.py --tickers AAPL MSFT NVDA  # subconjunto de tickers

Equivalente a ejecutar analyzer.py pero con soporte de argumentos CLI básicos.
Todos los parámetros principales se leen de environment.py; los flags de este
script sólo sobreescriben las variables de control (FORCE_DOWNLOAD, SKIP_BACKTEST).
"""

import argparse
import io as _io
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Importar configuración central ───────────────────────────────────────────
from environment import (
    DATA_DIR, RESULTS_DIR, AGENTS_RESULTS_DIR, BACKTEST_RESULTS_DIR, PLOTS_DIR,
    FORWARD_RETURN_DAYS, MIN_HISTORY_QUARTERS,
    WALKFORWARD_TRAIN_YEARS, WALKFORWARD_TEST_QUARTERS, RISK_FREE_RATE,
    RANDOM_SEED, START_DATE, END_DATE,
)

# ── Logging ───────────────────────────────────────────────────────────────────
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(_io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")),
        logging.FileHandler(f"{RESULTS_DIR}/pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Imports del módulo ────────────────────────────────────────────────────────
from module.data_router import DataRouter
from module.feature_engineering import (
    FundamentalFeatureBuilder,
    InsiderFeatureBuilder,
    TechnicalFeatureBuilder,
    ValuationFeatureBuilder,
)
from module.fetcher import fetch_tickers
from module.pipeline.data_ops import download_data, get_available_tickers
from module.pipeline.dataset_builder import build_master_dataset
from module.pipeline.evaluator import run_walkforward_pipeline
from module.pipeline.live_fold import run_live_fold
from module.visualizer import Visualizer


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent ML Stock Picker — Pipeline de ejecución"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="No re-descarga datos aunque sean antiguos (usa los que ya hay en disco).",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Omite el walk-forward backtest histórico; solo ejecuta el fold live.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        help="Subconjunto de tickers a usar (por defecto usa todos los de fetcher.py).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    force_download = not args.skip_download
    skip_backtest  = args.skip_backtest

    log.info("=" * 60)
    log.info("  INICIANDO PIPELINE ML MULTI-AGENTE STOCK PICKER")
    log.info("=" * 60)
    if args.tickers:
        log.info(f"  Modo: tickers seleccionados ({len(args.tickers)}): {args.tickers}")
    if skip_backtest:
        log.info("  Modo: --skip-backtest activado — solo fold live")

    # ── 1. Tickers
    tickers = args.tickers if args.tickers else fetch_tickers()

    # ── 2. Descargar datos
    download_data(
        tickers=tickers,
        start_date=START_DATE,
        end_date=END_DATE,
        data_dir=DATA_DIR,
        force_download=force_download,
    )

    # ── 3. DataRouter y filtrado
    router     = DataRouter(data_dir=DATA_DIR)
    sector_map = router.get_sector_map()
    tickers_ok = get_available_tickers(tickers, data_dir=DATA_DIR)
    log.info(f"Tickers con datos completos: {len(tickers_ok)} / {len(tickers)}")

    # ── 4. Builders de features
    fundamental_builder = FundamentalFeatureBuilder()
    technical_builder   = TechnicalFeatureBuilder()
    valuation_builder   = ValuationFeatureBuilder()
    insider_builder     = InsiderFeatureBuilder()

    # ── 5. Dataset maestro
    df = build_master_dataset(
        tickers=tickers_ok,
        router=router,
        fundamental_builder=fundamental_builder,
        technical_builder=technical_builder,
        valuation_builder=valuation_builder,
        insider_builder=insider_builder,
        min_history_quarters=MIN_HISTORY_QUARTERS,
        forward_return_days=FORWARD_RETURN_DAYS,
    )
    df.to_csv(f"{RESULTS_DIR}/master_dataset.csv")
    log.info(f"Dataset maestro: {len(df)} observaciones — {len(tickers_ok)} tickers")

    # ── 6. Diagnóstico sectorial
    visualizer = Visualizer(plots_dir=PLOTS_DIR)
    visualizer.plot_sector_performance(df.reset_index())

    summary = {}
    if not skip_backtest:
        # ── 7. Precios y benchmark
        import pandas as pd

        prices_dict = {}
        for ticker in tickers_ok:
            p = router.load_prices(ticker)
            if p is not None:
                prices_dict[ticker] = p

        benchmark = router.load_sp500_prices()
        if benchmark is None:
            log.warning("Sin S&P 500 — usando retorno cero como benchmark")
            benchmark = pd.Series(0.0, index=pd.date_range(START_DATE, END_DATE))
        benchmark_returns = benchmark.pct_change().dropna()

        # ── 8. Walk-Forward Pipeline
        summary = run_walkforward_pipeline(
            df=df,
            sector_map=sector_map,
            prices_dict=prices_dict,
            benchmark=benchmark_returns,
            agents_results_dir=AGENTS_RESULTS_DIR,
            backtest_results_dir=BACKTEST_RESULTS_DIR,
            plots_dir=PLOTS_DIR,
            start_date=START_DATE,
            end_date=END_DATE,
            walkforward_train_years=WALKFORWARD_TRAIN_YEARS,
            walkforward_test_quarters=WALKFORWARD_TEST_QUARTERS,
            risk_free_rate=RISK_FREE_RATE,
            random_seed=RANDOM_SEED,
        )
    else:
        log.info("--skip-backtest activado — saltando walk-forward backtest histórico")

    # ── 9. Fold live out-of-sample
    run_live_fold(
        df=df,
        sector_map=sector_map,
        tickers_ok=tickers_ok,
        as_of_date=END_DATE,
        router=router,
        fundamental_builder=fundamental_builder,
        technical_builder=technical_builder,
        valuation_builder=valuation_builder,
        insider_builder=insider_builder,
        results_dir=RESULTS_DIR,
        agents_results_dir=AGENTS_RESULTS_DIR,
        min_history_quarters=MIN_HISTORY_QUARTERS,
        random_seed=RANDOM_SEED,
    )

    # ── Resumen final
    log.info("\n" + "=" * 60)
    log.info("  PIPELINE COMPLETADO")
    log.info("=" * 60)
    log.info(f"  Tickers analizados:   {len(tickers_ok)}")
    if summary:
        log.info(f"  Alpha medio:          {summary.get('mean_alpha', 0):.2%}")
        log.info(f"  Folds con alpha > 0:  {summary.get('pct_folds_positive_alpha', 0):.0%}")
        log.info(f"  Sharpe Estrategia:    {summary.get('global_strategy_sharpe', 0):.3f}")
        log.info(f"  Sharpe Benchmark:     {summary.get('global_benchmark_sharpe', 0):.3f}")
        log.info(f"  Max DD Estrategia:    {summary.get('global_strategy_max_drawdown', 0):.2%}")
    log.info(f"  Resultados en:        {RESULTS_DIR}/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
