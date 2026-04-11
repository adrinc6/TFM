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
import json
import random
from datetime import datetime
import platform
import subprocess
from pathlib import Path

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from environment import (
    TICKERS,
    USE_DYNAMIC_SP500_UNIVERSE, SP500_HISTORIC_CSV_PATH,
    SP500_DYNAMIC_TOP_N,
    FINNHUB_DATA_DIR, FINNHUB_API_KEY,
    RESULTS_DIR, AGENTS_RESULTS_DIR, BACKTEST_RESULTS_DIR, PLOTS_DIR,
    MIN_HISTORY_QUARTERS,
    WALKFORWARD_TRAIN_LOOKBACK_YEARS, WALKFORWARD_TEST_QUARTERS, RISK_FREE_RATE,
    RANDOM_SEED, DOWNLOAD_START_DATE,
    ANALYSIS_START_YEAR, ANALYSIS_START_QUARTER, ANALYSIS_END_YEAR, ANALYSIS_END_QUARTER,
    ANALYSIS_FREQUENCY, ANALYSIS_ANNUAL_START_DATE,
    SKIP_BACKTEST, FORCE_DOWNLOAD, RETRY_MISSING_TICKERS, UPDATE_PRICES_ONLY,
    TOP_N_STOCKS, SNAPSHOT_LAG_DAYS, HOLDING_PERIOD_MONTHS, TECHNICAL_LOOKBACK_DAYS,
    INITIAL_CAPITAL_USD, TRANSACTION_FEE_USD, SLIPPAGE_PCT, USE_DOLLAR_BACKTEST,
    ALLOW_FRACTIONAL_SHARES, RUN_BASELINES, N_RANDOM_BASELINE_SIMS,
    BASELINE_MOMENTUM_LOOKBACK_DAYS, EXPORT_RUN_ARTIFACTS,
    ENABLE_CACHE, CACHE_DIR, CACHE_USE_MASTER_DATASET,
    CACHE_USE_ROUTER_DERIVED, CACHE_USE_WALKFORWARD_SUMMARY,
    CACHE_SCHEMA_VERSION,
    FUNDAMENTAL_FEATURE_COLUMNS, FUNDAMENTAL_FEATURE_EXCLUDE,
    VALUATION_FEATURE_COLUMNS, VALUATION_FEATURE_EXCLUDE,
    MOMENTUM_FEATURE_COLUMNS, MOMENTUM_FEATURE_EXCLUDE,
    BEAR_FEATURE_COLUMNS, BEAR_FEATURE_EXCLUDE,
    SENTIMENT_FEATURE_COLUMNS, SENTIMENT_FEATURE_EXCLUDE,
    SECTOR_ROTATION_FEATURE_COLUMNS, SECTOR_ROTATION_FEATURE_EXCLUDE,
    META_FEATURE_COLUMNS, META_FEATURE_EXCLUDE,
)

from module.common.cache import CacheManager
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


def _normalize_ticker_symbol(ticker: str) -> str:
    return str(ticker).strip().upper().replace(".", "-")


def _load_sp500_membership(csv_path: str):
    import pandas as pd

    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el CSV de histórico S&P 500: {csv_path}")

    df = pd.read_csv(p)
    required = {"ticker", "start_date", "end_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV S&P 500 inválido. Faltan columnas: {sorted(missing)}")

    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).map(_normalize_ticker_symbol)
    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce").dt.normalize()
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["ticker", "start_date"]).reset_index(drop=True)
    return out


def _membership_active_tickers(df_membership, start_date, end_date) -> list[str]:
    mask = (df_membership["start_date"] <= end_date) & (
        df_membership["end_date"].isna() | (df_membership["end_date"] >= start_date)
    )
    tickers = df_membership.loc[mask, "ticker"].dropna().astype(str).tolist()
    return sorted(set(tickers))


def _load_market_cap_panel(data_dir: str, tickers: list[str]) -> dict[str, object]:
    import pandas as pd

    panel: dict[str, object] = {}
    cons_dir = Path(data_dir) / "consolidated"

    for tk in tickers:
        fp = cons_dir / f"{tk}.csv"
        if not fp.exists():
            continue

        try:
            df = pd.read_csv(fp)
        except Exception:
            continue

        if "report_date" not in df.columns:
            continue

        mcap_col = None
        for c in ("market_cap", "bf_market_cap"):
            if c in df.columns:
                mcap_col = c
                break
        if mcap_col is None:
            continue

        d = pd.to_datetime(df["report_date"], errors="coerce").dt.normalize()
        m = pd.to_numeric(df[mcap_col], errors="coerce")
        md = pd.DataFrame({"date": d, "market_cap": m}).dropna()
        if md.empty:
            continue

        panel[tk] = md.sort_values("date").reset_index(drop=True)

    return panel


def _market_cap_asof(panel: dict[str, object], ticker: str, as_of_date):
    md = panel.get(ticker)
    if md is None or md.empty:
        return None

    hist = md.loc[md["date"] <= as_of_date]
    if hist.empty:
        return None

    return float(hist.iloc[-1]["market_cap"])


def _resolve_dynamic_universe(
    *,
    csv_path: str,
    data_dir: str,
    start_date,
    end_date,
    top_n,
) -> tuple[list[str], list[str], list[dict]]:
    import pandas as pd

    members = _load_sp500_membership(csv_path)
    candidates = _membership_active_tickers(members, start_date, end_date)
    if not candidates:
        return [], [], []

    top_n_enabled = False
    top_n_value = 0
    if isinstance(top_n, bool):
        top_n_enabled = False if top_n is False else True
        top_n_value = 0 if top_n is False else 1
    else:
        try:
            top_n_value = int(top_n)
            top_n_enabled = top_n_value > 0
        except Exception:
            top_n_enabled = False
            top_n_value = 0

    if not top_n_enabled:
        return candidates, candidates, []

    panel = _load_market_cap_panel(data_dir, candidates)
    yearly_details: list[dict] = []
    selected: set[str] = set()

    for year in range(int(start_date.year), int(end_date.year) + 1):
        y_start = pd.Timestamp(year=year, month=1, day=1)
        y_end = pd.Timestamp(year=year, month=12, day=31)
        if y_end < start_date or y_start > end_date:
            continue

        year_start = max(y_start, start_date)
        year_end = min(y_end, end_date)
        active_year = _membership_active_tickers(members, year_start, year_end)

        ranked = []
        missing_mcap = 0
        for tk in active_year:
            mcap = _market_cap_asof(panel, tk, year_end)
            if mcap is None or mcap <= 0:
                missing_mcap += 1
                continue
            ranked.append((tk, mcap))

        ranked.sort(key=lambda x: x[1], reverse=True)
        picked = [tk for tk, _ in ranked[: max(top_n_value, 1)]]
        if not picked:
            picked = active_year[: max(top_n_value, 1)]

        selected.update(picked)
        yearly_details.append(
            {
                "year": int(year),
                "active_members": int(len(active_year)),
                "ranked_with_mcap": int(len(ranked)),
                "missing_mcap": int(missing_mcap),
                "picked": int(len(picked)),
            }
        )

    selected_list = sorted(selected)
    if not selected_list:
        selected_list = candidates[: max(top_n_value, 1)]

    return candidates, selected_list, yearly_details

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


def _set_global_seeds(seed: int) -> None:
    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except Exception:
        pass


def _safe_git_commit_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        return out if out else None
    except Exception:
        return None


def _safe_version(module_name: str) -> str | None:
    try:
        mod = __import__(module_name)
        return str(getattr(mod, "__version__", None))
    except Exception:
        return None


def _export_run_config(
    *,
    results_dir: str,
    tickers_requested: list[str],
    tickers_ok: list[str],
    start_date: str,
    end_date: str,
    analysis_frequency: str,
    annual_anchor_date,
) -> None:
    payload = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "commit_hash": _safe_git_commit_hash(),
        "python_version": platform.python_version(),
        "library_versions": {
            "numpy": _safe_version("numpy"),
            "pandas": _safe_version("pandas"),
            "sklearn": _safe_version("sklearn"),
            "xgboost": _safe_version("xgboost"),
        },
        "flags": {
            "SKIP_BACKTEST": bool(SKIP_BACKTEST),
            "UPDATE_PRICES_ONLY": bool(UPDATE_PRICES_ONLY),
            "FORCE_DOWNLOAD": bool(FORCE_DOWNLOAD),
            "RETRY_MISSING_TICKERS": bool(RETRY_MISSING_TICKERS),
            "ENABLE_CACHE": bool(ENABLE_CACHE),
            "CACHE_USE_MASTER_DATASET": bool(CACHE_USE_MASTER_DATASET),
            "CACHE_USE_ROUTER_DERIVED": bool(CACHE_USE_ROUTER_DERIVED),
            "CACHE_USE_WALKFORWARD_SUMMARY": bool(CACHE_USE_WALKFORWARD_SUMMARY),
        },
        "parameters": {
            "TOP_N_STOCKS": TOP_N_STOCKS,
            "SNAPSHOT_LAG_DAYS": SNAPSHOT_LAG_DAYS,
            "HOLDING_PERIOD_MONTHS": HOLDING_PERIOD_MONTHS,
            "RANDOM_SEED": RANDOM_SEED,
            "TECHNICAL_LOOKBACK_DAYS": TECHNICAL_LOOKBACK_DAYS,
            "INITIAL_CAPITAL_USD": INITIAL_CAPITAL_USD,
            "TRANSACTION_FEE_USD": TRANSACTION_FEE_USD,
            "SLIPPAGE_PCT": SLIPPAGE_PCT,
            "USE_DOLLAR_BACKTEST": USE_DOLLAR_BACKTEST,
            "ALLOW_FRACTIONAL_SHARES": ALLOW_FRACTIONAL_SHARES,
            "RUN_BASELINES": RUN_BASELINES,
            "N_RANDOM_BASELINE_SIMS": N_RANDOM_BASELINE_SIMS,
            "BASELINE_MOMENTUM_LOOKBACK_DAYS": BASELINE_MOMENTUM_LOOKBACK_DAYS,
            "EXPORT_RUN_ARTIFACTS": EXPORT_RUN_ARTIFACTS,
            "WALKFORWARD_TRAIN_LOOKBACK_YEARS": WALKFORWARD_TRAIN_LOOKBACK_YEARS,
            "WALKFORWARD_TEST_QUARTERS": WALKFORWARD_TEST_QUARTERS,
            "RISK_FREE_RATE": RISK_FREE_RATE,
        },
        "universe": {
            "tickers_requested": tickers_requested,
            "tickers_ok": tickers_ok,
        },
        "time_range": {
            "start_date": start_date,
            "end_date": end_date,
            "analysis_frequency": analysis_frequency,
            "annual_anchor_date": None if annual_anchor_date is None else str(annual_anchor_date.date()),
        },
    }
    out = Path(results_dir) / "run_config.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _export_data_quality_report(
    *,
    tickers_requested: list[str],
    tickers_ok: list[str],
    router: DataRouter,
    master_df,
    out_path: Path,
) -> None:
    import pandas as pd

    if master_df is None:
        master_df = pd.DataFrame()

    group_cols = {
        "fundamental": [c for c in master_df.columns if c.startswith(("revenue", "net_income", "roe", "roa", "gross_margin", "operating_margin", "current_ratio", "piotroski"))],
        "technical": [c for c in master_df.columns if c.startswith(("rsi", "macd", "sma_", "bb_", "momentum_", "volatility_", "atr_", "price_vs_"))],
        "valuation": [c for c in master_df.columns if c.startswith(("pe", "pb", "ps", "ev_", "fcf_yield", "earnings_yield", "bf_"))],
        "insider": [c for c in master_df.columns if c.startswith(("insider_", "mspr_"))],
        "sentiment": [c for c in master_df.columns if c.startswith(("analyst_", "eps_", "beat_rate"))],
    }

    rows = []
    ok_set = set(tickers_ok)
    for tk in tickers_requested:
        prices = router.load_prices(tk)
        cons = router.load_consolidated(tk)
        per_ticker = master_df.loc[master_df.index.get_level_values("ticker") == tk] if not master_df.empty else pd.DataFrame()

        row = {
            "ticker": tk,
            "has_prices": bool(prices is not None and not prices.empty),
            "has_consolidated": bool(cons is not None and not cons.empty),
            "price_min_date": None,
            "price_max_date": None,
            "n_snapshots_master_dataset": int(len(per_ticker)),
            "ticker_ok": bool(tk in ok_set),
        }
        if prices is not None and not prices.empty:
            row["price_min_date"] = str(pd.Timestamp(prices.index.min()).date())
            row["price_max_date"] = str(pd.Timestamp(prices.index.max()).date())

        for grp, cols in group_cols.items():
            if len(cols) == 0 or per_ticker.empty:
                row[f"missing_pct_{grp}"] = None
            else:
                row[f"missing_pct_{grp}"] = float(per_ticker[cols].isna().mean().mean())

        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


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
    _set_global_seeds(RANDOM_SEED)
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

    analysis_window_start = test_start_date.normalize()
    analysis_window_end = end_date.normalize()

    # ── 1. Resolver universo solicitado y descargar datos
    if USE_DYNAMIC_SP500_UNIVERSE:
        try:
            candidates, _, _ = _resolve_dynamic_universe(
                csv_path=SP500_HISTORIC_CSV_PATH,
                data_dir=FINNHUB_DATA_DIR,
                start_date=analysis_window_start,
                end_date=analysis_window_end,
                top_n=SP500_DYNAMIC_TOP_N,
            )
        except Exception as ex:
            log.warning("No se pudo resolver universo dinámico S&P 500 (%s). Se usa TICKERS manual.", ex)
            candidates = []

        tickers_to_download = candidates if candidates else list(dict.fromkeys(TICKERS))
        log.info(
            "Universo dinámico: %s candidatos activos en [%s, %s]",
            len(tickers_to_download),
            analysis_window_start.date(),
            analysis_window_end.date(),
        )
    else:
        tickers_to_download = list(dict.fromkeys(TICKERS))
        log.info("Universo manual: %s tickers", len(tickers_to_download))

    download_data(
        tickers=tickers_to_download,
        start_date=DOWNLOAD_START_DATE,
        end_date=end_date_str,
        data_dir=FINNHUB_DATA_DIR,
        force_download=FORCE_DOWNLOAD,
        api_key=FINNHUB_API_KEY,
        prices_only=UPDATE_PRICES_ONLY,
        allow_retry_failed=RETRY_MISSING_TICKERS,
    )

    if UPDATE_PRICES_ONLY:
        log.info("UPDATE_PRICES_ONLY=True — actualizando solo precios y macro, sin consolidacion ni backtest")
        if EXPORT_RUN_ARTIFACTS:
            _export_run_config(
                results_dir=RESULTS_DIR,
                tickers_requested=tickers_to_download,
                tickers_ok=[],
                start_date=str(test_start_date.date()),
                end_date=str(end_date.date()),
                analysis_frequency=analysis_frequency,
                annual_anchor_date=annual_anchor_date,
            )
        return

    # ── 2. Consolidar datos
    prepare_data(tickers_to_download, data_dir=FINNHUB_DATA_DIR)

    # ── 2b. Selección final del universo para dataset/backtest
    if USE_DYNAMIC_SP500_UNIVERSE:
        candidates, tickers, yearly_details = _resolve_dynamic_universe(
            csv_path=SP500_HISTORIC_CSV_PATH,
            data_dir=FINNHUB_DATA_DIR,
            start_date=analysis_window_start,
            end_date=analysis_window_end,
            top_n=SP500_DYNAMIC_TOP_N,
        )
        if not tickers:
            tickers = list(dict.fromkeys(tickers_to_download))
        log.info(
            "Universo dinámico final: %s tickers seleccionados (candidatos=%s, top_n=%s)",
            len(tickers), len(candidates), SP500_DYNAMIC_TOP_N,
        )
        for row in yearly_details:
            log.info(
                "  Año %s | activos=%s | con_mcap=%s | sin_mcap=%s | seleccionados=%s",
                row["year"], row["active_members"], row["ranked_with_mcap"],
                row["missing_mcap"], row["picked"],
            )
    else:
        tickers = list(dict.fromkeys(tickers_to_download))

    cache = None
    cache_enabled = bool(ENABLE_CACHE) and not bool(FORCE_DOWNLOAD) and not bool(UPDATE_PRICES_ONLY)
    if cache_enabled:
        cache_context = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "tickers": list(dict.fromkeys(tickers)),
            "download_start_date": DOWNLOAD_START_DATE,
            "analysis_start_year": ANALYSIS_START_YEAR,
            "analysis_start_quarter": ANALYSIS_START_QUARTER,
            "analysis_end_year": ANALYSIS_END_YEAR,
            "analysis_end_quarter": ANALYSIS_END_QUARTER,
            "analysis_frequency": ANALYSIS_FREQUENCY,
            "analysis_annual_start_date": ANALYSIS_ANNUAL_START_DATE,
            "snapshot_lag_days": SNAPSHOT_LAG_DAYS,
            "holding_period_months": HOLDING_PERIOD_MONTHS,
            "technical_lookback_days": TECHNICAL_LOOKBACK_DAYS,
            "min_history_quarters": MIN_HISTORY_QUARTERS,
            "walkforward_train_years": WALKFORWARD_TRAIN_LOOKBACK_YEARS,
            "walkforward_test_quarters": WALKFORWARD_TEST_QUARTERS,
            "top_n_stocks": TOP_N_STOCKS,
            "risk_free_rate": RISK_FREE_RATE,
            "random_seed": RANDOM_SEED,
            "feature_controls": {
                "fundamental_include": FUNDAMENTAL_FEATURE_COLUMNS,
                "fundamental_exclude": FUNDAMENTAL_FEATURE_EXCLUDE,
                "valuation_include": VALUATION_FEATURE_COLUMNS,
                "valuation_exclude": VALUATION_FEATURE_EXCLUDE,
                "momentum_include": MOMENTUM_FEATURE_COLUMNS,
                "momentum_exclude": MOMENTUM_FEATURE_EXCLUDE,
                "bear_include": BEAR_FEATURE_COLUMNS,
                "bear_exclude": BEAR_FEATURE_EXCLUDE,
                "sentiment_include": SENTIMENT_FEATURE_COLUMNS,
                "sentiment_exclude": SENTIMENT_FEATURE_EXCLUDE,
                "sector_rotation_include": SECTOR_ROTATION_FEATURE_COLUMNS,
                "sector_rotation_exclude": SECTOR_ROTATION_FEATURE_EXCLUDE,
                "meta_include": META_FEATURE_COLUMNS,
                "meta_exclude": META_FEATURE_EXCLUDE,
            },
        }
        cache = CacheManager(CACHE_DIR, cache_context)
        log.info("Cache activa: key=%s dir=%s", cache.key, cache.run_dir)
    else:
        log.info("Cache desactivada para esta ejecución")

    # ── 3. DataRouter y filtrado de tickers disponibles
    router = DataRouter(data_dir=FINNHUB_DATA_DIR)
    sector_map = router.get_sector_map()
    tickers_ok = []
    missing_detail = {}
    loaded_ticker_availability_cache = False
    if cache is not None and CACHE_USE_ROUTER_DERIVED:
        cached_availability = cache.load_json("ticker_availability")
        if cached_availability:
            tickers_ok = list(cached_availability.get("tickers_ok", []))
            missing_detail = dict(cached_availability.get("missing_detail", {}))
            loaded_ticker_availability_cache = True
            log.info("Cache hit: ticker_availability (%s tickers OK)", len(tickers_ok))

    if not loaded_ticker_availability_cache:
        tickers_ok, missing_detail = get_available_tickers(tickers, data_dir=FINNHUB_DATA_DIR)
        if cache is not None and CACHE_USE_ROUTER_DERIVED:
            cache.save_json(
                "ticker_availability",
                {
                    "tickers_ok": tickers_ok,
                    "missing_detail": missing_detail,
                },
            )

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
        if cache is not None and CACHE_USE_ROUTER_DERIVED:
            cache.save_json(
                "ticker_availability",
                {
                    "tickers_ok": tickers_ok,
                    "missing_detail": missing_detail,
                },
            )

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

    df = None
    if cache is not None and CACHE_USE_MASTER_DATASET:
        cached_df = cache.load_pickle("master_dataset")
        if cached_df is not None:
            df = cached_df
            log.info("Cache hit: master_dataset (%s observaciones)", len(df))

    if df is None:
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
        if cache is not None and CACHE_USE_MASTER_DATASET:
            cache.save_pickle("master_dataset", df)
            log.info("Cache save: master_dataset")

    df.to_csv(f"{RESULTS_DIR}/master_dataset.csv")
    log.info(f"Dataset maestro: {len(df)} observaciones — {len(tickers_ok)} tickers")

    if EXPORT_RUN_ARTIFACTS:
        _export_run_config(
            results_dir=RESULTS_DIR,
            tickers_requested=tickers,
            tickers_ok=tickers_ok,
            start_date=str(test_start_date.date()),
            end_date=str(end_date.date()),
            analysis_frequency=analysis_frequency,
            annual_anchor_date=annual_anchor_date,
        )
        _export_data_quality_report(
            tickers_requested=tickers,
            tickers_ok=tickers_ok,
            router=router,
            master_df=df,
            out_path=Path(RESULTS_DIR) / "data_quality_report.csv",
        )

    summary = {}
    if not SKIP_BACKTEST:
        effective_test_quarters = 4 if analysis_frequency == "annual" else WALKFORWARD_TEST_QUARTERS

        # ── 6. Precios y benchmark para el backtester (cacheable)
        prices_dict = None
        spy_prices = None
        benchmark_returns = None

        if cache is not None and CACHE_USE_ROUTER_DERIVED:
            market_bundle = cache.load_pickle("market_bundle")
            if isinstance(market_bundle, dict):
                prices_dict = market_bundle.get("prices_dict")
                spy_prices = market_bundle.get("spy_prices")
                benchmark_returns = market_bundle.get("benchmark_returns")
                if prices_dict is not None and benchmark_returns is not None:
                    log.info("Cache hit: market_bundle")

        if prices_dict is None or benchmark_returns is None:
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

            if cache is not None and CACHE_USE_ROUTER_DERIVED:
                cache.save_pickle(
                    "market_bundle",
                    {
                        "prices_dict": prices_dict,
                        "spy_prices": spy_prices,
                        "benchmark_returns": benchmark_returns,
                    },
                )
                log.info("Cache save: market_bundle")

        # ── 7. Walk-Forward Pipeline
        if cache is not None and CACHE_USE_WALKFORWARD_SUMMARY:
            cached_summary = cache.load_json("walkforward_summary")
            if isinstance(cached_summary, dict) and cached_summary:
                summary = cached_summary
                log.info("Cache hit: walkforward_summary (se omite recomputo de backtest)")

        if not summary:
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
            if cache is not None:
                cache.save_json("walkforward_summary", summary if isinstance(summary, dict) else {})
                log.info("Cache save: walkforward_summary")
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
