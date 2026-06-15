# =============================================================================
# analyzer.py - Multi-Agent ML Stock Picker pipeline orchestrator
# =============================================================================
"""
Main system entry point. It coordinates the pipeline steps:

    1. Data download (step_01_data)
    2. Preparation / consolidation (step_01_data)
    3. Master dataset construction (step_02_dataset)
    4. Historical walk-forward backtest (step_04_evaluation)

All business logic lives in module/steps/:
    - step_01_data/pipeline.py        : ETL (download, consolidation, ticker filtering)
    - step_02_dataset/dataset.py      : observation construction + live features
    - step_03_training/training.py    : agent training + anti-leakage OOF
    - step_04_evaluation/evaluator.py : walk-forward loop, SHAP, backtest, plots

Global parameters are defined in environment.py.
"""
import sys
import logging
import warnings
import io as _io
import json
import random
import platform
import subprocess
import os
from pathlib import Path

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from environment import (
    TICKERS,
    USE_DYNAMIC_SP500_UNIVERSE, SP500_HISTORIC_CSV_PATH,
    SP500_DYNAMIC_TOP_N,
    FINNHUB_DATA_DIR, FINNHUB_API_KEY,
    RESULTS_DIR, AGENTS_RESULTS_DIR, AGENT_MODELS_RESULTS_DIR, BACKTEST_RESULTS_DIR, PLOTS_DIR,
    MIN_HISTORY_QUARTERS,
    WALKFORWARD_TRAIN_LOOKBACK_YEARS, WALKFORWARD_NUM_TESTS, RISK_FREE_RATE,
    RANDOM_SEED, DOWNLOAD_START_DATE,
    ANALYSIS_REFERENCE_DATE,
    SKIP_BACKTEST, FORCE_DOWNLOAD, RETRY_MISSING_TICKERS, UPDATE_PRICES_ONLY,
    DOWNLOAD_OPTIONAL_ENDPOINTS,
    HOLDING_PERIOD_MONTHS, TECHNICAL_LOOKBACK_DAYS,
    INITIAL_CAPITAL_USD, TRANSACTION_FEE_USD, SLIPPAGE_PCT, USE_DOLLAR_BACKTEST,
    ALLOW_FRACTIONAL_SHARES, RUN_BASELINES, N_RANDOM_BASELINE_SIMS,
    BASELINE_MOMENTUM_LOOKBACK_DAYS, EXPORT_RUN_ARTIFACTS,
    DEBUG_OUTPUT_PROFILE,
    EXPORT_TP_SL_UNIVERSE_MATRIX,
    EXPORT_GLOBAL_TP_SL_UNIVERSE_MATRIX,
    EXPORT_SNAPSHOT_AGENT_AUDITS,
    EXPORT_ALL_FOLDS_SCORES,
    EXPORT_DETAILED_TRADES_REPORT,
    REBUILD_MASTER_DATASET,
    SENTIMENT_FEATURE_COLUMNS, SENTIMENT_FEATURE_EXCLUDE,
    SECTOR_ROTATION_FEATURE_COLUMNS, SECTOR_ROTATION_FEATURE_EXCLUDE,
    META_FEATURE_COLUMNS, META_FEATURE_EXCLUDE,
    GARP_MAX_STOCKS, GARP_MIN_STOCKS,
)
import environment as env_module

from module.common.data_router import DataRouter
from module.steps.step_02_dataset.builders.fundamental import FundamentalFeatureBuilder
from module.steps.step_02_dataset.builders.insider import InsiderFeatureBuilder
from module.steps.step_02_dataset.builders.sentiment import SentimentFeatureBuilder
from module.steps.step_02_dataset.builders.technical import TechnicalFeatureBuilder
from module.steps.step_02_dataset.builders.valuation import ValuationFeatureBuilder

from module.steps.step_01_data.pipeline import download_data, prepare_data, get_available_tickers, retry_missing_tickers
from module.steps.step_02_dataset.dataset import build_master_dataset
from module.steps.step_04_evaluation.evaluator import run_walkforward_pipeline
from module.common.portfolio_intelligence import load_positions_csv, review_portfolio
from module.common.static_viewer import generate_static_viewer
from module.common.portfolio_evolution import run_portfolio_evolution


def _normalize_ticker_symbol(ticker: str) -> str:
    """Strip whitespace, uppercase, and replace dots with dashes in a ticker symbol.

    Args:
        ticker: Raw ticker string (e.g. ``" brk.b "``).

    Returns:
        str: Normalised ticker string (e.g. ``"BRK-B"``).
    """
    return str(ticker).strip().upper().replace(".", "-")


def _load_sp500_membership(csv_path: str):
    """Read the S&P 500 historical membership CSV and normalise its columns.

    The CSV must contain at least the columns ``ticker``, ``start_date``, and
    ``end_date``.  Ticker symbols are normalised via
    :func:`_normalize_ticker_symbol`; date columns are parsed with
    ``errors="coerce"`` and rows missing ``ticker`` or ``start_date`` are
    dropped.

    Args:
        csv_path: Absolute or relative path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned membership DataFrame with columns ``ticker``
            (str), ``start_date`` (Timestamp, tz-naive) and ``end_date``
            (Timestamp, tz-naive or NaT).

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist on disk.
        ValueError: If the CSV is missing one or more required columns.
    """
    import pandas as pd

    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Historical S&P 500 CSV does not exist: {csv_path}")

    df = pd.read_csv(p)
    required = {"ticker", "start_date", "end_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Invalid S&P 500 CSV. Missing columns: {sorted(missing)}")

    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).map(_normalize_ticker_symbol)
    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce").dt.normalize()
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["ticker", "start_date"]).reset_index(drop=True)
    return out


def _membership_active_tickers(df_membership, start_date, end_date) -> list[str]:
    """Return the sorted list of tickers whose membership overlapped [start_date, end_date].

    A ticker is considered active if its ``start_date`` is on or before
    ``end_date`` **and** its ``end_date`` is either NaT (still active) or on or
    after ``start_date``.

    Args:
        df_membership: DataFrame produced by :func:`_load_sp500_membership`
            with columns ``ticker``, ``start_date``, and ``end_date``.
        start_date: Beginning of the target window (inclusive, Timestamp-like).
        end_date: End of the target window (inclusive, Timestamp-like).

    Returns:
        list[str]: Alphabetically sorted list of unique normalised ticker
            symbols active in the given window.
    """
    mask = (df_membership["start_date"] <= end_date) & (
        df_membership["end_date"].isna() | (df_membership["end_date"] >= start_date)
    )
    tickers = df_membership.loc[mask, "ticker"].dropna().astype(str).tolist()
    return sorted(set(tickers))


def _load_market_cap_panel(data_dir: str, tickers: list[str]) -> dict[str, object]:
    """Build a per-ticker market-cap time-series dictionary from consolidated CSVs.

    For each ticker, reads ``<data_dir>/consolidated/<ticker>.csv`` and extracts
    the ``report_date`` column together with the first available market-cap column
    (``market_cap`` or ``bf_market_cap``).  Only rows with non-null values in
    both columns are kept.

    Args:
        data_dir: Path to the Finnhub data root directory.  The consolidated
            CSVs are expected under ``<data_dir>/consolidated/``.
        tickers: List of normalised ticker symbols to load.

    Returns:
        dict[str, pd.DataFrame]: Mapping from ticker to a two-column DataFrame
            with columns ``date`` (tz-naive Timestamp) and ``market_cap``
            (float), sorted by ``date`` ascending.  Tickers without a
            consolidated CSV or without a recognised market-cap column are
            omitted.
    """
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
    """Return the most recent market cap for a ticker up to (and including) as_of_date.

    Args:
        panel: Dict produced by :func:`_load_market_cap_panel` mapping ticker
            to a DataFrame with columns ``date`` and ``market_cap``.
        ticker: Normalised ticker symbol to look up.
        as_of_date: Cutoff date (Timestamp-like).  Only observations on or
            before this date are considered.

    Returns:
        float | None: The market cap of the last available observation on or
            before ``as_of_date``, or ``None`` if no such observation exists.
    """
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
    """Use S&P 500 membership and market-cap data to select the analysis universe.

    Loads the historical membership CSV, finds all tickers active during
    [start_date, end_date], and — when ``top_n`` is a positive integer —
    ranks them by market cap each calendar year and takes the top-N largest.
    The yearly details are returned for logging and audit purposes.

    Args:
        csv_path: Path to the historical S&P 500 membership CSV (passed to
            :func:`_load_sp500_membership`).
        data_dir: Finnhub data root directory (passed to
            :func:`_load_market_cap_panel`).
        start_date: Start of the analysis window (Timestamp-like, inclusive).
        end_date: End of the analysis window (Timestamp-like, inclusive).
        top_n: Maximum number of tickers to select per year.  If ``False`` or
            ``0``, all active candidates are returned without size filtering.

    Returns:
        tuple:
            - **all_candidates** (list[str]): All tickers active at any point
              in [start_date, end_date].
            - **selected** (list[str]): Tickers chosen after the top-N market-
              cap filter.  Equals *all_candidates* when ``top_n`` is disabled.
            - **yearly_details** (list[dict]): One dict per calendar year with
              keys ``year``, ``active_members``, ``ranked_with_mcap``,
              ``missing_mcap``, and ``picked``.
    """
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

# ── Logging (configured inside main() once the run slug is known) ───────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(_io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")),
    ],
)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def _run_slug(
    analysis_reference_date,
    holding_period_months: int,
    walkforward_num_tests: int,
) -> str:
    """Build a short human-readable identifier for a pipeline run.

    The slug is used as a subdirectory name under ``results/`` so that
    outputs from different parameter combinations don't overwrite each other.

    Example: ``anchor_20260503_h12_n8``
    """
    import pandas as pd

    anchor = pd.Timestamp(analysis_reference_date).normalize().strftime("%Y%m%d")
    h = max(int(holding_period_months), 1)
    n = max(int(walkforward_num_tests), 1)
    return f"anchor_{anchor}_h{h}_n{n}"


def _to_json_compatible(value):
    """Convert arbitrary Python objects into JSON-serializable values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out[str(k)] = _to_json_compatible(v)
        return out

    if isinstance(value, (list, tuple, set)):
        return [_to_json_compatible(v) for v in value]

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _collect_environment_snapshot() -> dict:
    """Capture all uppercase constants currently loaded from environment.py."""
    snapshot = {}
    for key, value in vars(env_module).items():
        if isinstance(key, str) and key.isupper():
            snapshot[key] = _to_json_compatible(value)
    return dict(sorted(snapshot.items(), key=lambda kv: kv[0]))


def _load_environment_snapshot(run_dir: Path) -> dict | None:
    """Load config/environment_snapshot.json for a run directory if present."""
    fp = run_dir / "config" / "environment_snapshot.json"
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None


def _diff_snapshots(base: dict | None, current: dict) -> dict:
    """Compute top-level key differences between two environment snapshots."""
    if base is None:
        return {
            "status": "no_v1_snapshot",
            "added": sorted(current.keys()),
            "removed": [],
            "changed": {},
        }

    base_keys = set(base.keys())
    current_keys = set(current.keys())
    added = sorted(current_keys - base_keys)
    removed = sorted(base_keys - current_keys)
    changed = {}
    for key in sorted(base_keys & current_keys):
        if base.get(key) != current.get(key):
            changed[key] = {
                "v1": base.get(key),
                "current": current.get(key),
            }
    return {
        "status": "ok",
        "added": added,
        "removed": removed,
        "changed": changed,
    }


def _resolve_versioned_run_dir(slug: str, env_snapshot: dict) -> tuple[Path, int]:
    """Resolve a non-colliding run directory and persist config snapshots.

    Behavior:
      - First config for a period -> results/<slug>
      - Different config same period -> results/<slug>_v2, _v3, ...
      - Same config as an existing version -> reuse that existing version dir
    """
    results_root = Path("results")
    results_root.mkdir(parents=True, exist_ok=True)

    index = 1
    while True:
        candidate = results_root / (slug if index == 1 else f"{slug}_v{index}")

        try:
            candidate.mkdir(parents=False, exist_ok=False)
            break
        except FileExistsError:
            existing_snapshot = _load_environment_snapshot(candidate)
            if existing_snapshot is not None and existing_snapshot == env_snapshot:
                return candidate, index
            index += 1

    config_dir = candidate / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = config_dir / "environment_snapshot.json"
    snapshot_path.write_text(json.dumps(env_snapshot, indent=2), encoding="utf-8")

    v1_dir = results_root / slug
    v1_snapshot = _load_environment_snapshot(v1_dir)
    diff = _diff_snapshots(v1_snapshot, env_snapshot)
    diff_payload = {
        "base_slug": slug,
        "current_version": f"v{index}",
        "current_dir": str(candidate).replace("\\", "/"),
        "compare_against": str(v1_dir).replace("\\", "/"),
        "diff": diff,
    }
    (config_dir / "changes_vs_v1.json").write_text(
        json.dumps(diff_payload, indent=2),
        encoding="utf-8",
    )

    return candidate, index


def _set_global_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across Python's ``random`` and NumPy.

    Args:
        seed: Integer seed value.  The same value is passed to both
            ``random.seed`` and ``numpy.random.seed``.
    """
    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except Exception:
        pass


def _safe_git_commit_hash() -> str | None:
    """Return the current HEAD git commit hash, or None on any error.

    Uses ``git rev-parse HEAD`` under the hood.  All exceptions (e.g. git not
    installed, not inside a repository) are swallowed silently.

    Returns:
        str | None: The 40-character commit SHA, or ``None`` if it cannot be
            determined.
    """
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
    """Return the ``__version__`` attribute of a module, or None if unavailable.

    Imports the module by name and reads its ``__version__`` attribute.  Any
    import error or missing attribute is swallowed silently.

    Args:
        module_name: Importable module name (e.g. ``"numpy"``).

    Returns:
        str | None: Version string, or ``None`` if the module cannot be
            imported or has no ``__version__``.
    """
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
    analysis_reference_date: str,
    walkforward_num_tests: int,
) -> None:
    """Serialise all run parameters and environment metadata to run_config.json.

    Writes a JSON file at ``<results_dir>/run_config.json`` containing the
    current timestamp, git commit hash, Python version, key library versions,
    feature-control flags, pipeline parameters, the ticker universe, and the
    analysis time range.

    Args:
        results_dir: Directory where ``run_config.json`` will be written.
        tickers_requested: Full list of tickers that were requested for
            download / analysis.
        tickers_ok: Subset of tickers for which all required data files are
            present.
        analysis_reference_date: ISO-8601 reference entry date.
        walkforward_num_tests: Number of historical test windows generated.
    """
    payload = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "commit_hash": _safe_git_commit_hash(),
        "python_version": platform.python_version(),
        "run_context": {
            "experiment_name": os.getenv("EXPERIMENT_NAME"),
            "batch_id": os.getenv("EXPERIMENT_BATCH_ID"),
            "env_overrides_json": os.getenv("ENV_OVERRIDES_JSON"),
        },
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
            "REBUILD_MASTER_DATASET": bool(REBUILD_MASTER_DATASET),
        },
        "parameters": {
            "GARP_MAX_STOCKS": GARP_MAX_STOCKS,
            "GARP_MIN_STOCKS": GARP_MIN_STOCKS,
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
            "WALKFORWARD_NUM_TESTS": WALKFORWARD_NUM_TESTS,
            "RISK_FREE_RATE": RISK_FREE_RATE,
            "DEBUG_OUTPUT_PROFILE": DEBUG_OUTPUT_PROFILE,
            "EXPORT_TP_SL_UNIVERSE_MATRIX": EXPORT_TP_SL_UNIVERSE_MATRIX,
            "EXPORT_GLOBAL_TP_SL_UNIVERSE_MATRIX": EXPORT_GLOBAL_TP_SL_UNIVERSE_MATRIX,
            "EXPORT_SNAPSHOT_AGENT_AUDITS": EXPORT_SNAPSHOT_AGENT_AUDITS,
            "EXPORT_ALL_FOLDS_SCORES": EXPORT_ALL_FOLDS_SCORES,
            "EXPORT_DETAILED_TRADES_REPORT": EXPORT_DETAILED_TRADES_REPORT,
        },
        "universe": {
            "tickers_requested": tickers_requested,
            "tickers_ok": tickers_ok,
        },
        "time_range": {
            "analysis_reference_date": analysis_reference_date,
            "walkforward_num_tests": int(walkforward_num_tests),
            "holding_period_months": int(HOLDING_PERIOD_MONTHS),
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
    """Build a per-ticker CSV report of data availability and feature missing rates.

    For every ticker in ``tickers_requested`` the report includes whether price
    and consolidated data exist, the min/max price date range, the number of
    observations in the master dataset, and the average missing-value rate for
    each of the following feature groups: fundamental, technical, valuation,
    insider, and sentiment.

    Args:
        tickers_requested: All tickers that were requested for the run.
        tickers_ok: Subset of tickers that passed the data-availability check.
        router: Configured :class:`~module.common.data_router.DataRouter`
            instance used to load prices and consolidated data.
        master_df: The master dataset DataFrame (multi-indexed by
            ``(ticker, date)``), or ``None`` if it was not built.
        out_path: File system path where the CSV report should be written.
            Parent directories are created if they do not exist.
    """
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


def _load_master_dataset_disk(path: Path):
    """Load master dataset from disk (Parquet)."""
    import pandas as pd

    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as ex:
            log.warning("Master dataset read failed (%s). Will rebuild.", ex)
    return None


# =============================================================================
# Main
# =============================================================================

def portfolio_review_main(argv: list[str] | None = None):
    """Run a lightweight thesis-based review without executing the full pipeline."""
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(
        prog="python analyzer.py portfolio_review",
        description="Review existing positions using point-in-time GARP thesis snapshots.",
    )
    parser.add_argument("--tickers", default="", help="Comma-separated tickers to review when --positions is not provided.")
    parser.add_argument("--positions", default="", help="CSV with ticker plus optional weight, purchase_date, avg_cost and snapshot_date.")
    parser.add_argument("--review-date", default=str(pd.Timestamp(ANALYSIS_REFERENCE_DATE).date()), help="Review date; latest snapshot on or before this date is used.")
    parser.add_argument("--output-dir", default=os.path.join(RESULTS_DIR, "portfolio_review"), help="Directory for portfolio-review CSV/JSON outputs.")
    parser.add_argument("--master-dataset", default=os.path.join(FINNHUB_DATA_DIR, "master_dataset.parquet"), help="Existing master dataset parquet/csv to reuse.")
    args = parser.parse_args(argv)

    dataset_path = Path(args.master_dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Master dataset not found: {dataset_path}. Build it with `python analyzer.py` before running portfolio_review."
        )
    if dataset_path.suffix.lower() == ".csv":
        snapshots = pd.read_csv(dataset_path)
    else:
        snapshots = pd.read_parquet(dataset_path)

    positions = load_positions_csv(args.positions) if args.positions else None
    tickers = [t.strip() for t in str(args.tickers).split(",") if t.strip()] if not args.positions else None
    review, summary = review_portfolio(
        snapshots=snapshots,
        positions=positions,
        tickers=tickers,
        review_date=args.review_date,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, default=str))
    print(f"Portfolio review exported to: {args.output_dir}")
    return review, summary


def portfolio_evolution_main(argv: list[str] | None = None):
    """Run a lightweight live-portfolio evolution simulation from existing snapshots."""
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(
        prog="python analyzer.py portfolio_evolution",
        description="Simulate a thesis-managed GARP portfolio through periodic reviews.",
    )
    parser.add_argument("--review-frequency", default="M", help="Review frequency: M, 2M or Q.")
    parser.add_argument("--output-dir", default=os.path.join(RESULTS_DIR, "portfolio_evolution"), help="Directory for evolution CSV/JSON outputs.")
    parser.add_argument("--master-dataset", default=os.path.join(FINNHUB_DATA_DIR, "master_dataset.parquet"), help="Existing master dataset parquet/csv to reuse.")
    args = parser.parse_args(argv)

    dataset_path = Path(args.master_dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Master dataset not found: {dataset_path}. Build it with `python analyzer.py` before running portfolio_evolution."
        )
    snapshots = pd.read_csv(dataset_path) if dataset_path.suffix.lower() == ".csv" else pd.read_parquet(dataset_path)
    evolution, transactions, holdings, summary = run_portfolio_evolution(
        snapshots=snapshots,
        review_frequency=args.review_frequency,
        output_dir=args.output_dir,
    )
    viewer_dir = generate_static_viewer(Path(args.output_dir).parent, viewer_dir=Path(args.output_dir).parent / "viewer")
    print(json.dumps(summary, indent=2, default=str))
    print(f"Portfolio evolution exported to: {args.output_dir}")
    print(f"Static viewer exported to: {viewer_dir}")
    return evolution, transactions, holdings, summary

def main():
    """Orchestrate the full multi-agent ML stock-picker pipeline.

    Execution steps:

    1. **Universe resolution** — loads the requested ticker list or, when
       ``USE_DYNAMIC_SP500_UNIVERSE`` is enabled, derives the universe from the
       historical S&P 500 membership CSV filtered by market-cap rank.
    2. **Data download** (Step 01) — calls Finnhub and Yahoo Finance to fetch
       OHLCV prices, fundamentals, earnings surprises, insider transactions, and
       analyst recommendations for every ticker in the universe.
    3. **Data consolidation** (Step 01) — normalises and merges all raw JSON
       files into per-ticker consolidated CSVs.
    4. **Master dataset** (Step 02) — builds the observation matrix with
       point-in-time features (fundamental, technical, valuation, sentiment,
       insider) for the full analysis window; result is optionally cached.
    5. **Walk-forward backtest** (Step 04) — runs the rolling train/test loop,
       trains all GARP agents (quality, growth, valuation, trend, catalyst, risk, technical guardrail, sector prior, meta-learner) on each fold, generates SHAP explanations,
       simulates portfolio returns, and exports summary artifacts.

    All parameters are read from ``environment.py``.  The function logs a
    final summary table with mean alpha, Sharpe ratios, and maximum drawdown.
    """
    import pandas as pd
    _set_global_seeds(RANDOM_SEED)

    anchor_entry_date = pd.Timestamp(ANALYSIS_REFERENCE_DATE).normalize()
    walkforward_num_tests = max(int(WALKFORWARD_NUM_TESTS), 1)
    holding_months = max(int(HOLDING_PERIOD_MONTHS), 1)

    scheduled_entry_dates = [
        (anchor_entry_date - pd.DateOffset(months=holding_months * k)).normalize()
        for k in range(walkforward_num_tests)
    ]
    earliest_entry_date = min(scheduled_entry_dates)
    latest_entry_date = max(scheduled_entry_dates)
    analysis_window_start = (earliest_entry_date - pd.DateOffset(years=max(int(WALKFORWARD_TRAIN_LOOKBACK_YEARS), 1))).normalize()
    analysis_window_end = (latest_entry_date + pd.DateOffset(months=holding_months)).normalize()

    # ── Run-specific result directories ──────────────────────────────────────
    # Compute a slug that captures the most important run parameters, then
    # shadow all result-dir variables so every output lands in the same
    # namespaced folder.  Changing any key parameter automatically produces
    # a fresh directory without touching previous runs.
    slug = _run_slug(
        analysis_reference_date=anchor_entry_date,
        holding_period_months=holding_months,
        walkforward_num_tests=walkforward_num_tests,
    )
    env_snapshot = _collect_environment_snapshot()
    RUN_RESULTS_DIR, run_version_idx = _resolve_versioned_run_dir(slug, env_snapshot)

    # Ensure config artifacts also exist when reusing an existing version folder.
    config_dir = RUN_RESULTS_DIR / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    snapshot_fp = config_dir / "environment_snapshot.json"
    if not snapshot_fp.exists():
        snapshot_fp.write_text(json.dumps(env_snapshot, indent=2), encoding="utf-8")

    v1_dir = Path("results") / slug
    changes_fp = config_dir / "changes_vs_v1.json"
    if not changes_fp.exists():
        v1_snapshot = _load_environment_snapshot(v1_dir)
        changes_payload = {
            "base_slug": slug,
            "current_version": f"v{run_version_idx}",
            "current_dir": str(RUN_RESULTS_DIR).replace("\\", "/"),
            "compare_against": str(v1_dir).replace("\\", "/"),
            "diff": _diff_snapshots(v1_snapshot, env_snapshot),
        }
        changes_fp.write_text(json.dumps(changes_payload, indent=2), encoding="utf-8")

    RESULTS_DIR              = str(RUN_RESULTS_DIR / "general")
    AGENTS_RESULTS_DIR       = str(RUN_RESULTS_DIR)
    AGENT_MODELS_RESULTS_DIR = str(RUN_RESULTS_DIR)
    BACKTEST_RESULTS_DIR     = str(RUN_RESULTS_DIR)
    PLOTS_DIR                = str(RUN_RESULTS_DIR)

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # Add a run-specific log file now that the directory exists.
    _file_handler = logging.FileHandler(f"{RESULTS_DIR}/pipeline.log", encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    logging.getLogger().addHandler(_file_handler)

    log.info("=" * 60)
    log.info("  STARTING MULTI-AGENT ML STOCK PICKER PIPELINE")
    log.info("  Run: %s (v%s)", slug, run_version_idx)
    log.info("=" * 60)

    log.info(
        "Scheduled tests: reference_entry=%s | holding=%s months | num_tests=%s | earliest_entry=%s",
        anchor_entry_date.date(),
        holding_months,
        walkforward_num_tests,
        earliest_entry_date.date(),
    )

    download_end_date = pd.Timestamp.today().normalize()
    end_date_str = download_end_date.strftime("%Y-%m-%d")

    # 1. Resolve requested universe and download data
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
            log.warning("Could not resolve dynamic S&P 500 universe (%s). Falling back to manual TICKERS.", ex)
            candidates = []

        tickers_to_download = candidates if candidates else list(dict.fromkeys(TICKERS))
        log.info(
            "Dynamic universe: %s active candidates in [%s, %s]",
            len(tickers_to_download),
            analysis_window_start.date(),
            analysis_window_end.date(),
        )
    else:
        tickers_to_download = list(dict.fromkeys(TICKERS))
        log.info("Manual universe: %s tickers", len(tickers_to_download))

    download_data(
        tickers=tickers_to_download,
        start_date=DOWNLOAD_START_DATE,
        end_date=end_date_str,
        data_dir=FINNHUB_DATA_DIR,
        force_download=FORCE_DOWNLOAD,
        api_key=FINNHUB_API_KEY,
        prices_only=UPDATE_PRICES_ONLY,
        allow_retry_failed=RETRY_MISSING_TICKERS,
        download_optional=DOWNLOAD_OPTIONAL_ENDPOINTS,
    )

    if UPDATE_PRICES_ONLY:
        log.info("UPDATE_PRICES_ONLY=True - updating prices and macro only, without consolidation or backtest")
        if EXPORT_RUN_ARTIFACTS:
            _export_run_config(
                results_dir=RESULTS_DIR,
                tickers_requested=tickers_to_download,
                tickers_ok=[],
                analysis_reference_date=str(anchor_entry_date.date()),
                walkforward_num_tests=walkforward_num_tests,
            )
        return

    # 2. Consolidate data
    prepare_data(tickers_to_download, data_dir=FINNHUB_DATA_DIR)

    # 2b. Final universe selection for dataset/backtest
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
            "Final dynamic universe: %s selected tickers (candidates=%s, top_n=%s)",
            len(tickers), len(candidates), SP500_DYNAMIC_TOP_N,
        )
        for row in yearly_details:
            log.info(
                "  Year %s | active=%s | with_mcap=%s | without_mcap=%s | selected=%s",
                row["year"], row["active_members"], row["ranked_with_mcap"],
                row["missing_mcap"], row["picked"],
            )
    else:
        tickers = list(dict.fromkeys(tickers_to_download))

    # 3. DataRouter and available ticker filtering
    router = DataRouter(data_dir=FINNHUB_DATA_DIR)
    sector_map = router.get_sector_map()
    tickers_ok, missing_detail = get_available_tickers(tickers, data_dir=FINNHUB_DATA_DIR)

    # 3b. Retry for tickers with incomplete data
    if RETRY_MISSING_TICKERS and missing_detail:
        recovered = retry_missing_tickers(
            missing_detail=missing_detail,
            start_date=DOWNLOAD_START_DATE,
            end_date=end_date_str,
            data_dir=FINNHUB_DATA_DIR,
            api_key=FINNHUB_API_KEY,
        )
        tickers_ok = sorted(set(tickers_ok) | set(recovered))

    # 4. Feature builders
    fundamental_builder = FundamentalFeatureBuilder()
    technical_builder   = TechnicalFeatureBuilder()
    valuation_builder   = ValuationFeatureBuilder()
    insider_builder     = InsiderFeatureBuilder()
    sentiment_builder   = SentimentFeatureBuilder()

    # 5. Build master dataset
    #    The master dataset uses ALL available data (all tickers, all quarters)
    #    and is independent of analysis parameters (start/end dates, top_n, etc.).
    #    It is stored in data_finnhub/master_dataset.parquet and only rebuilt
    #    when REBUILD_MASTER_DATASET=True or the file does not exist.

    master_dataset_path = Path(FINNHUB_DATA_DIR) / "master_dataset.parquet"
    df = None
    if not REBUILD_MASTER_DATASET:
        df = _load_master_dataset_disk(master_dataset_path)
        if df is not None:
            log.info("Master dataset loaded from disk (%s observations)", len(df))

    if df is None:
        log.info("Building master dataset from scratch...")
        df = build_master_dataset(
            tickers=tickers_ok,
            router=router,
            fundamental_builder=fundamental_builder,
            technical_builder=technical_builder,
            valuation_builder=valuation_builder,
            insider_builder=insider_builder,
            sentiment_builder=sentiment_builder,
            min_history_quarters=MIN_HISTORY_QUARTERS,
            holding_period_months=HOLDING_PERIOD_MONTHS,
            technical_lookback_days=TECHNICAL_LOOKBACK_DAYS,
        )
        try:
            df.to_parquet(master_dataset_path)
            log.info("Master dataset saved to %s", master_dataset_path)
        except Exception as ex:
            log.warning("Could not save master_dataset.parquet (%s)", ex)

    log.info("Master dataset: %d observations - %d tickers", len(df), len(tickers_ok))

    if EXPORT_RUN_ARTIFACTS:
        _export_run_config(
            results_dir=RESULTS_DIR,
            tickers_requested=tickers,
            tickers_ok=tickers_ok,
            analysis_reference_date=str(anchor_entry_date.date()),
            walkforward_num_tests=walkforward_num_tests,
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
        # 6. Prices and benchmark for the backtester
        prices_dict = {}
        for ticker in tickers_ok:
            p = router.load_prices(ticker)
            if p is not None:
                prices_dict[ticker] = p

        spy_prices = router.load_sp500_prices()
        if spy_prices is None:
            import pandas as pd

            log.warning("No S&P 500 data available - using zero return as benchmark")
            benchmark_returns = pd.Series(
                0.0,
                index=pd.date_range(analysis_window_start, analysis_window_end),
            )
        else:
            benchmark_returns = spy_prices.pct_change().dropna()

        # 7. Walk-forward pipeline
        summary = run_walkforward_pipeline(
            df=df,
            sector_map=sector_map,
            prices_dict=prices_dict,
            benchmark=benchmark_returns,
            spy_prices=spy_prices,
            agents_results_dir=AGENTS_RESULTS_DIR,
            agent_models_results_dir=AGENT_MODELS_RESULTS_DIR,
            backtest_results_dir=BACKTEST_RESULTS_DIR,
            plots_dir=PLOTS_DIR,
            walkforward_train_years=WALKFORWARD_TRAIN_LOOKBACK_YEARS,
            walkforward_num_tests=walkforward_num_tests,
            risk_free_rate=RISK_FREE_RATE,
            top_n_stocks=GARP_MAX_STOCKS,
            random_seed=RANDOM_SEED,
            holding_period_months=holding_months,
            finnhub_data_dir=FINNHUB_DATA_DIR,
            analysis_reference_date=anchor_entry_date,
        )
    else:
        log.info("SKIP_BACKTEST=True - skipping historical walk-forward backtest")

    # Final output
    log.info("\n" + "=" * 60)
    log.info("  PIPELINE COMPLETED")
    log.info("=" * 60)
    log.info(f"  Analyzed tickers:     {len(tickers_ok)}")
    if summary:
        log.info(f"  Mean alpha:            {summary.get('mean_alpha', 0):.2%}")
        log.info(f"  Folds with alpha > 0:  {summary.get('pct_folds_positive_alpha', 0):.0%}")
        log.info(f"  Sharpe Strategy:       {summary.get('global_strategy_sharpe', 0):.3f}")
        log.info(f"  Sharpe Benchmark:      {summary.get('global_benchmark_sharpe', 0):.3f}")
        log.info(f"  Max DD Strategy:       {summary.get('global_strategy_max_drawdown', 0):.2%}")
    log.info(f"  Results in:           {RESULTS_DIR}/")
    log.info("=" * 60)
    try:
        viewer_dir = generate_static_viewer(RUN_RESULTS_DIR)
        log.info("  Static viewer in:      %s/", viewer_dir)
    except Exception as ex:
        log.warning("Could not generate static viewer (%s)", ex)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "portfolio_review":
        portfolio_review_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "portfolio_evolution":
        portfolio_evolution_main(sys.argv[2:])
    else:
        main()
