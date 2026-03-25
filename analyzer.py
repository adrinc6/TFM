# =============================================================================
# analyzer.py — Orquestador del pipeline Multi-Agente ML Stock Picker
# =============================================================================
"""
Punto de entrada principal del sistema. Coordina los pasos del pipeline:

    1. Descarga de datos (step_01_data)
    2. Preparación / consolidación (step_01_data)
    3. Construcción del dataset maestro (step_02_dataset)
    4. Walk-Forward Backtest histórico (step_04_evaluation)

Toda la lógica de negocio reside en module/steps/:
    - step_01_data/pipeline.py        : ETL (descarga, consolidación, filtrado de tickers)
    - step_02_dataset/dataset.py      : construcción de observaciones + features live
    - step_03_training/training.py    : entrenamiento de agentes + OOF anti-leakage
    - step_04_evaluation/evaluator.py : walk-forward loop, SHAP, backtest, gráficos

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
    TICKERS,
    FINNHUB_DATA_DIR, FINNHUB_API_KEY,
    RESULTS_DIR, AGENTS_RESULTS_DIR, BACKTEST_RESULTS_DIR, PLOTS_DIR,
    MIN_HISTORY_QUARTERS,
    WALKFORWARD_TRAIN_LOOKBACK_YEARS, WALKFORWARD_TEST_QUARTERS, RISK_FREE_RATE,
    RANDOM_SEED, DOWNLOAD_START_DATE,
    ANALYSIS_START_YEAR, ANALYSIS_START_QUARTER, ANALYSIS_END_YEAR, ANALYSIS_END_QUARTER,
    ANALYSIS_FREQUENCY, ANALYSIS_ANNUAL_START_DATE,
    SKIP_BACKTEST, FORCE_DOWNLOAD, RETRY_MISSING_TICKERS, UPDATE_PRICES_ONLY,
    TOP_N_STOCKS, SNAPSHOT_LAG_DAYS, HOLDING_PERIOD_MONTHS, TECHNICAL_LOOKBACK_DAYS,
)

from module.common.data_router import DataRouter
from module.steps.step_02_dataset.builders.fundamental import FundamentalFeatureBuilder
from module.steps.step_02_dataset.builders.insider import InsiderFeatureBuilder
from module.steps.step_02_dataset.builders.sentiment import SentimentFeatureBuilder
from module.steps.step_02_dataset.builders.technical import TechnicalFeatureBuilder
from module.steps.step_02_dataset.builders.valuation import ValuationFeatureBuilder

from module.steps.step_01_data.pipeline import download_data, prepare_data, get_available_tickers, retry_missing_tickers
from module.steps.step_02_dataset.dataset import build_master_dataset
from module.steps.step_04_evaluation.evaluator import run_walkforward_pipeline


def _quarter_end_date(year: int, quarter: int):
    import pandas as pd

    month = quarter * 3
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)

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
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
log = logging.getLogger(__name__)


# =============================================================================
# Main
# =============================================================================

def main():
    log.info("=" * 60)
    log.info("  INICIANDO PIPELINE ML MULTI-AGENTE STOCK PICKER")
    log.info("=" * 60)

    # Rango de análisis. En modo trimestral se usa ANALYSIS_START_QUARTER.
    # En modo anual se ignora ese quarter y se arranca desde el quarter de la fecha ancla anual.
    import pandas as pd
    end_date = _quarter_end_date(ANALYSIS_END_YEAR, ANALYSIS_END_QUARTER)
    analysis_frequency = str(ANALYSIS_FREQUENCY).strip().lower()
    if analysis_frequency not in {"quarterly", "annual"}:
        raise ValueError("ANALYSIS_FREQUENCY debe ser 'quarterly' o 'annual'")

    test_start_date = _quarter_end_date(ANALYSIS_START_YEAR, ANALYSIS_START_QUARTER)
    annual_anchor_date = None
    if analysis_frequency == "annual":
        if ANALYSIS_ANNUAL_START_DATE:
            annual_anchor_date = pd.Timestamp(ANALYSIS_ANNUAL_START_DATE).normalize()
        else:
            annual_anchor_date = (
                pd.Timestamp(year=int(ANALYSIS_START_YEAR), month=1, day=1)
                + pd.Timedelta(days=max(int(SNAPSHOT_LAG_DAYS), 0))
            ).normalize()

        # En anual, el backtester debe generar folds desde el quarter de la anchor,
        # no desde ANALYSIS_START_QUARTER.
        test_start_date = annual_anchor_date.to_period("Q").end_time.normalize()

        log.info(
            "Modo anual activado: anchor=%s | start_quarter=%sQ%s | holding=%s meses",
            annual_anchor_date.date(),
            test_start_date.year,
            test_start_date.quarter,
            12,
        )

    download_end_date = pd.Timestamp.today().normalize()
    end_date_str = download_end_date.strftime("%Y-%m-%d")

    # ── 1. Descargar datos
    tickers = list(dict.fromkeys(TICKERS))  # deduplica preservando orden
    download_data(
        tickers=tickers,
        start_date=DOWNLOAD_START_DATE,
        end_date=end_date_str,
        data_dir=FINNHUB_DATA_DIR,
        force_download=FORCE_DOWNLOAD,
        api_key=FINNHUB_API_KEY,
        prices_only=UPDATE_PRICES_ONLY,
    )

    if UPDATE_PRICES_ONLY:
        log.info("UPDATE_PRICES_ONLY=True — actualizando solo precios y macro, sin consolidacion ni backtest")
        return

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
            start_date=DOWNLOAD_START_DATE,
            end_date=end_date_str,
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
    if SNAPSHOT_LAG_DAYS is None:
        raise ValueError("SNAPSHOT_LAG_DAYS debe estar definido en environment.py")
    dataset_snapshot_lag_days = int(SNAPSHOT_LAG_DAYS)

    df = build_master_dataset(
        tickers=tickers_ok,
        router=router,
        fundamental_builder=fundamental_builder,
        technical_builder=technical_builder,
        valuation_builder=valuation_builder,
        insider_builder=insider_builder,
        sentiment_builder=sentiment_builder,
        min_history_quarters=MIN_HISTORY_QUARTERS,
        snapshot_lag_days=dataset_snapshot_lag_days,
        holding_period_months=HOLDING_PERIOD_MONTHS,
        technical_lookback_days=TECHNICAL_LOOKBACK_DAYS,
    )
    df.to_csv(f"{RESULTS_DIR}/master_dataset.csv")
    log.info(f"Dataset maestro: {len(df)} observaciones — {len(tickers_ok)} tickers")

    summary = {}
    if not SKIP_BACKTEST:
        effective_test_quarters = 4 if analysis_frequency == "annual" else WALKFORWARD_TEST_QUARTERS

        # ── 6. Precios y benchmark para el backtester
        prices_dict = {}
        for ticker in tickers_ok:
            p = router.load_prices(ticker)
            if p is not None:
                prices_dict[ticker] = p

        spy_prices = router.load_sp500_prices()
        if spy_prices is None:
            import pandas as pd
            log.warning("Sin S&P 500 — usando retorno cero como benchmark")
            benchmark_returns = pd.Series(0.0, index=pd.date_range(test_start_date, end_date))
        else:
            benchmark_returns = spy_prices.pct_change().dropna()

        # ── 7. Walk-Forward Pipeline
        summary = run_walkforward_pipeline(
            df=df,
            sector_map=sector_map,
            prices_dict=prices_dict,
            benchmark=benchmark_returns,
            spy_prices=spy_prices,
            agents_results_dir=AGENTS_RESULTS_DIR,
            backtest_results_dir=BACKTEST_RESULTS_DIR,
            plots_dir=PLOTS_DIR,
            start_date=test_start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            walkforward_train_years=WALKFORWARD_TRAIN_LOOKBACK_YEARS,
            walkforward_test_quarters=effective_test_quarters,
            risk_free_rate=RISK_FREE_RATE,
            top_n_stocks=TOP_N_STOCKS,
            random_seed=RANDOM_SEED,
            snapshot_lag_days=dataset_snapshot_lag_days,
            holding_period_months=12 if analysis_frequency == "annual" else HOLDING_PERIOD_MONTHS,
            finnhub_data_dir=FINNHUB_DATA_DIR,
            analysis_frequency=analysis_frequency,
            annual_anchor_date=annual_anchor_date,
        )
    else:
        log.info("SKIP_BACKTEST=True — saltando walk-forward backtest histórico")

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
