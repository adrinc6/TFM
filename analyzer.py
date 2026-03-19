# =============================================================================
# analyzer.py — Orquestador del pipeline Multi-Agente ML Stock Picker
# =============================================================================
"""
Punto de entrada principal del sistema. Coordina los pasos del pipeline:

  1. Descarga de datos (fetcher_finnhub → data_ops)
  2. Preparación / consolidación (FinnhubConsolidator → data_ops)
  3. Construcción del dataset maestro (dataset_builder)
  4. Walk-Forward Backtest histórico (evaluator)
  5. Fold live out-of-sample (live_fold)

Toda la lógica de negocio reside en module/pipeline/:
  - data_ops.py      : ETL (descarga, consolidación, filtrado de tickers)
  - dataset_builder.py: construcción de observaciones y normalización sectorial
  - trainer.py       : entrenamiento de agentes + OOF anti-leakage
  - evaluator.py     : walk-forward loop, SHAP, backtest, gráficos
  - live_fold.py     : predicción out-of-sample + evaluación con precios reales

Los parámetros globales están en environment.py.
"""
import sys
import logging
import warnings
import io as _io
from pathlib import Path

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from environment import (
    FINNHUB_DATA_DIR, FINNHUB_API_KEY,
    RESULTS_DIR, AGENTS_RESULTS_DIR, BACKTEST_RESULTS_DIR, PLOTS_DIR,
    FORWARD_RETURN_DAYS, MIN_HISTORY_QUARTERS,
    WALKFORWARD_TRAIN_YEARS, WALKFORWARD_TEST_QUARTERS, RISK_FREE_RATE,
    RANDOM_SEED, START_DATE, END_DATE,
    SKIP_BACKTEST, FORCE_DOWNLOAD, RETRY_MISSING_TICKERS,
)

from module.data_router import DataRouter
from module.feature_engineering import (
    FundamentalFeatureBuilder,
    TechnicalFeatureBuilder,
    ValuationFeatureBuilder,
    InsiderFeatureBuilder,
    SentimentFeatureBuilder,
)
from module.finnhub_processor import build_companies_df
from module.pipeline.data_ops import download_data, prepare_data, get_available_tickers, retry_missing_tickers
from module.pipeline.dataset_builder import build_master_dataset
from module.pipeline.evaluator import run_walkforward_pipeline
from module.pipeline.live_fold import run_live_fold
from module.visualizer import Visualizer

# ── Logging ───────────────────────────────────────────────────────────────────
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        # UTF-8 en consola: evita UnicodeEncodeError en Windows (cp1252)
        logging.StreamHandler(_io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")),
        logging.FileHandler(f"{RESULTS_DIR}/pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# Main
# =============================================================================

def main():
    log.info("=" * 60)
    log.info("  INICIANDO PIPELINE ML MULTI-AGENTE STOCK PICKER")
    log.info("=" * 60)

    # ── 1. Descargar datos
    companies_df = build_companies_df(FINNHUB_DATA_DIR)
    tickers = list(companies_df.index) if not companies_df.empty else []

    if not tickers:
        log.warning("No hay tickers en companies_df — usando lista hardcoded mínima.")
        # Fallback: carga de profiles existentes en data_finnhub/
        data_path = Path(FINNHUB_DATA_DIR)
        tickers = [d.name for d in data_path.iterdir()
                   if d.is_dir() and not d.name.startswith("_")]

    download_data(
        tickers=tickers,
        start_date=START_DATE,
        end_date=END_DATE,
        data_dir=FINNHUB_DATA_DIR,
        force_download=FORCE_DOWNLOAD,
        api_key=FINNHUB_API_KEY,
    )

    # ── 2. Consolidar datos
    prepare_data(tickers, data_dir=FINNHUB_DATA_DIR)

    # ── 3. DataRouter y filtrado de tickers disponibles
    router = DataRouter(data_dir=FINNHUB_DATA_DIR)
    sector_map = router.get_sector_map()
    tickers_ok, missing_detail = get_available_tickers(tickers, data_dir=FINNHUB_DATA_DIR)

    # ── 3b. Reintento para tickers con datos incompletos
    if RETRY_MISSING_TICKERS and missing_detail:
        recovered = retry_missing_tickers(
            missing_detail=missing_detail,
            start_date=START_DATE,
            end_date=END_DATE,
            data_dir=FINNHUB_DATA_DIR,
            api_key=FINNHUB_API_KEY,
        )
        tickers_ok = sorted(set(tickers_ok) | set(recovered))

    # ── 4. Builders de features
    fundamental_builder = FundamentalFeatureBuilder()
    technical_builder   = TechnicalFeatureBuilder()
    valuation_builder   = ValuationFeatureBuilder()
    insider_builder     = InsiderFeatureBuilder()
    sentiment_builder   = SentimentFeatureBuilder()

    # ── 5. Construir dataset maestro
    df = build_master_dataset(
        tickers=tickers_ok,
        router=router,
        fundamental_builder=fundamental_builder,
        technical_builder=technical_builder,
        valuation_builder=valuation_builder,
        insider_builder=insider_builder,
        sentiment_builder=sentiment_builder,
        min_history_quarters=MIN_HISTORY_QUARTERS,
        forward_return_days=FORWARD_RETURN_DAYS,
    )
    df.to_csv(f"{RESULTS_DIR}/master_dataset.csv")
    log.info(f"Dataset maestro: {len(df)} observaciones — {len(tickers_ok)} tickers")

    # ── 6. Diagnóstico sectorial
    visualizer = Visualizer(plots_dir=PLOTS_DIR)
    visualizer.plot_sector_performance(df.reset_index())

    summary = {}
    if not SKIP_BACKTEST:
        # ── 7. Precios y benchmark para el backtester
        prices_dict = {}
        for ticker in tickers_ok:
            p = router.load_prices(ticker)
            if p is not None:
                prices_dict[ticker] = p

        benchmark = router.load_sp500_prices()
        if benchmark is None:
            import pandas as pd
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
        log.info("SKIP_BACKTEST=True — saltando walk-forward backtest histórico")

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
        sentiment_builder=sentiment_builder,
        results_dir=RESULTS_DIR,
        agents_results_dir=AGENTS_RESULTS_DIR,
        min_history_quarters=MIN_HISTORY_QUARTERS,
        random_seed=RANDOM_SEED,
    )

    # ── Resultado final
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
