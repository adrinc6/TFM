"""Walk-forward evaluation orchestration."""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from environment import (
    PORTFOLIO_MIN_SCORE,
    RUN_ABLATION_STUDY,
    WALKFORWARD_TRAIN_MIN_YEARS,
    MIN_TEST_TICKERS_PERCENT,
    ENABLE_FALLBACK_EXTRAPOLATION,
    FALLBACK_LOOK_BACK_QUARTERS,
    INITIAL_CAPITAL_USD,
    TRANSACTION_FEE_USD,
    SLIPPAGE_PCT,
    USE_DOLLAR_BACKTEST,
    ALLOW_FRACTIONAL_SHARES,
    RUN_BASELINES,
    N_RANDOM_BASELINE_SIMS,
    BASELINE_MOMENTUM_LOOKBACK_DAYS,
    USE_DYNAMIC_SP500_UNIVERSE,
    SP500_HISTORIC_CSV_PATH,
    PORTFOLIO_OPTIMIZER,
    TP_SL_BASE_TP,
    TP_SL_BASE_SL,
    TP_SL_MIN_TP,
    TP_SL_MAX_TP,
    TP_SL_MIN_SL,
    TP_SL_MAX_SL,
    TP_EDGE_ENABLE,
    TP_EDGE_PRIOR_STRENGTH,
    TP_EDGE_RELIABILITY_K,
    TP_EDGE_NONE_SCORE,
    TP_EDGE_CONFIDENCE_BLEND,
    TP_EDGE_TP_STRETCH_PENALTY,
    TP_EDGE_MIN_FEASIBILITY,
    TP_SL_RULE_SIGNAL_RBS_WEIGHT,
    TP_SL_ENABLE_STRATEGY_FINE_TUNING,
    TP_SL_FINE_TUNE_MAX_RELAX_STEPS,
    TP_SL_FINE_TUNE_MIN_HIT_RATE,
    TP_SL_FINE_TUNE_MIN_UTILITY,
    TP_SL_MIN_ACCEPTABLE_TP,
    TP_SL_SELECTION_CERTAINTY_WEIGHT,
    TP_SL_SELECTION_TP_QUALITY_WEIGHT,
    TP_SL_GRACE_PERIOD_FRACTION,
    TP_SL_TRAILING_REVIEW_DAYS,
    TP_SL_TRAILING_DRAWDOWN_QUANTILE,
    DEBUG_OUTPUT_PROFILE,
    EXPORT_TP_SL_UNIVERSE_MATRIX,
    EXPORT_GLOBAL_TP_SL_UNIVERSE_MATRIX,
    EXPORT_SNAPSHOT_AGENT_AUDITS,
    EXPORT_ALL_FOLDS_SCORES,
    EXPORT_DETAILED_TRADES_REPORT,
)
from module.common.asof import assert_no_future_data
from module.common.data_router import DataRouter
from module.common.cross_sectional_features import enrich_cross_sectional_features
from module.common.performance_metrics import sharpe_ratio
from module.common.target_engineering import infer_tp_sl_levels
from module.steps.step_04_evaluation.analysis import run_ablation_study, summarize_ablation
from module.steps.step_04_evaluation.backtesting import WalkForwardBacktester
from module.steps.step_04_evaluation.backtesting import (
    _get_close_column,
    compute_max_drawdown_from_equity,
    simulate_fold_usd,
    to_daily_returns_from_equity,
)
from module.steps.step_04_evaluation.reporting import (
    build_explanation_candidate_tickers,
    build_fold_scores_df,
    build_selection_audit_df,
    export_fold_scores,
    export_quarter_agent_feature_audit,
    export_quarter_snapshot_audit,
    export_selection_audit,
    export_ticker_explanations,
    generate_text_report,
)
from module.steps.step_04_evaluation.strategy import run_backtest
from module.steps.step_04_evaluation.strategy import simulate_tp_sl

from module.steps.step_03_training.training import train_fold
from module.steps.step_04_evaluation.visualization import Visualizer

log = logging.getLogger(__name__)


def _sample_tickers(tickers: set[str] | list[str], limit: int = 12) -> str:
    vals = sorted({str(t).upper() for t in tickers if str(t).strip()})
    if not vals:
        return "-"
    return ", ".join(vals[: max(int(limit), 1)])


def explain_top_tickers(
    agents: Dict,
    df_test: pd.DataFrame,
    scores: pd.Series,
    fold_id: int | str,
    agents_results_dir: str,
    selected_tickers: List[str] | None = None,
    audit_df: pd.DataFrame | None = None,
    top_n: int = 10,
) -> None:
    if scores.empty:
        return

    tickers_col = df_test.index.get_level_values("ticker")
    ticker_scores = pd.Series(scores.values, index=tickers_col).groupby(level=0).last()
    if selected_tickers is None:
        selected_tickers = ticker_scores.nlargest(top_n).index.tolist()

    audit = audit_df
    if audit is None or audit.empty:
        selected_for_audit = selected_tickers if selected_tickers else ticker_scores.nlargest(top_n).index.tolist()
        audit = build_selection_audit_df(
            df_scored=pd.DataFrame({"ticker": ticker_scores.index, "final_score": ticker_scores.values}),
            selected_tickers=selected_for_audit,
            score_col="final_score",
            threshold=PORTFOLIO_MIN_SCORE,
        )

    candidate_tickers = build_explanation_candidate_tickers(
        audit_df=audit,
        threshold=PORTFOLIO_MIN_SCORE,
        top_extra=max(top_n * 2, 10),
        near_margin=0.05,
        max_candidates=60,
    )

    export_ticker_explanations(
        agents=agents,
        df_test=df_test,
        scores=scores,
        fold_id=fold_id,
        agents_results_dir=agents_results_dir,
        candidate_tickers=candidate_tickers,
        audit_df=audit,
        explanation_top_n=6,
        prefix="quarter",
    )


def _safe_json_dump(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _concat_equity_parts(parts: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate a list of equity-curve DataFrames into a single time-ordered curve."""
    if not parts:
        return pd.DataFrame(columns=["date", "equity_usd", "cash_usd", "positions_value_usd"])
    return (
        pd.concat(parts, axis=0)
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )


def _prepare_fold_frames_by_filed_quarter(
    df: pd.DataFrame,
    train_start_date: pd.Timestamp,
    analysis_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if df.empty:


        return df.copy(), df.copy(), pd.Series(dtype="datetime64[ns]")

    train_start_date = pd.Timestamp(train_start_date).normalize()
    analysis_date = pd.Timestamp(analysis_date).normalize()

    if "snapshot_date" in df.columns:
        snapshot_dates = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.normalize()
    else:
        snapshot_dates = pd.Series(
            pd.to_datetime(df.index.get_level_values("date"), errors="coerce").normalize(),
            index=df.index,
        )

    train_mask = (snapshot_dates >= train_start_date) & (snapshot_dates < analysis_date)
    train_df = df.loc[train_mask.values].copy() if isinstance(train_mask, pd.Series) else df.loc[train_mask].copy()

    # Test snapshot per ticker = latest available snapshot at or before analysis_date.
    test_pool_mask = snapshot_dates <= analysis_date
    test_pool = df.loc[test_pool_mask.values].copy() if isinstance(test_pool_mask, pd.Series) else df.loc[test_pool_mask].copy()
    if test_pool.empty:
        return train_df.iloc[0:0], test_pool, pd.Series(dtype="datetime64[ns]")

    test_snaps = (
        pd.to_datetime(test_pool["snapshot_date"], errors="coerce").dt.normalize()
        if "snapshot_date" in test_pool.columns
        else pd.Series(pd.to_datetime(test_pool.index.get_level_values("date"), errors="coerce").normalize(), index=test_pool.index)
    )
    test_pool = test_pool.assign(__snapshot_date_norm=test_snaps.values)
    test_pool = test_pool.assign(__ticker=test_pool.index.get_level_values("ticker").astype(str).values)
    test_pool = test_pool.sort_values(["__ticker", "__snapshot_date_norm"])
    df_test = test_pool.groupby("__ticker", sort=False).tail(1).drop(columns=["__snapshot_date_norm", "__ticker"])

    train_df = train_df[~train_df.index.duplicated(keep="last")]
    df_test = df_test[~df_test.index.duplicated(keep="last")]

    if "snapshot_date" in df_test.columns:
        test_snapshot_dates = pd.to_datetime(df_test["snapshot_date"], errors="coerce")
        test_snapshot_dates.index = df_test.index
    else:
        test_snapshot_dates = pd.Series(df_test.index.get_level_values("date").values, index=df_test.index)
    return train_df, df_test, test_snapshot_dates


def _filter_fold_tickers_by_history_span(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    required_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Keep only test tickers with at least required_months of train-history span."""
    if df_train.empty or df_test.empty:
        return df_train.iloc[0:0], df_test.iloc[0:0], []

    train_tickers = pd.Index(df_train.index.get_level_values("ticker")).astype(str)
    test_tickers = pd.Index(df_test.index.get_level_values("ticker")).astype(str)

    if "year_quarter" in df_train.columns:
        train_quarters = pd.PeriodIndex(df_train["year_quarter"], freq="Q")
    else:
        train_quarters = pd.PeriodIndex(df_train.index.get_level_values("date"), freq="Q")

    temp = pd.DataFrame({"ticker": train_tickers.values, "q": train_quarters})
    span = temp.groupby("ticker")["q"].agg(["min", "max"])
    span_quarters = span["max"].astype(int) - span["min"].astype(int) + 1
    required_quarters = max(1, int(np.ceil(max(int(required_months), 1) / 3.0)))
    eligible_train = set(span_quarters[span_quarters >= required_quarters].index.astype(str))

    eligible_test = [
        tk for tk in pd.Index(test_tickers).unique().tolist()
        if tk in eligible_train
    ]
    eligible_set = set(eligible_test)

    train_mask = pd.Index(df_train.index.get_level_values("ticker")).astype(str).isin(eligible_set)
    test_mask = pd.Index(df_test.index.get_level_values("ticker")).astype(str).isin(eligible_set)
    return df_train.loc[train_mask].copy(), df_test.loc[test_mask].copy(), sorted(eligible_test)


def _load_sp500_membership(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=["ticker", "start_date", "end_date"])

    try:
        dfm = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame(columns=["ticker", "start_date", "end_date"])

    required = {"ticker", "start_date", "end_date"}
    if not required.issubset(set(dfm.columns)):
        return pd.DataFrame(columns=["ticker", "start_date", "end_date"])

    out = dfm[["ticker", "start_date", "end_date"]].copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce").dt.normalize()
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["ticker", "start_date"]).reset_index(drop=True)
    return out


def _active_sp500_tickers_on_date(membership_df: pd.DataFrame, as_of_date: pd.Timestamp) -> set[str]:
    if membership_df.empty:
        return set()
    d = pd.Timestamp(as_of_date).normalize()
    mask = (membership_df["start_date"] <= d) & (
        membership_df["end_date"].isna() | (membership_df["end_date"] >= d)
    )
    vals = membership_df.loc[mask, "ticker"].dropna().astype(str).tolist()
    return set(vals)


def _filter_test_by_sp500_membership(
    df_test: pd.DataFrame,
    active_tickers: set[str],
) -> tuple[pd.DataFrame, int]:
    if df_test.empty:
        return df_test, 0
    before = int(df_test.index.get_level_values("ticker").nunique())
    if not active_tickers:
        return df_test, before

    mask = pd.Index(df_test.index.get_level_values("ticker")).astype(str).isin(active_tickers)
    out = df_test.loc[mask].copy()
    return out, before


def _extrapolate_missing_snapshots(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    analysis_date: pd.Timestamp,
    lookback_quarters: int = 4,
) -> pd.DataFrame:
    """
    Extrapolates features for tickers that have no usable snapshot at analysis_date.
    
    If a ticker has at least `lookback_quarters` historical snapshots,
    the last snapshots prior to analysis_date are averaged
    and an "estimated" row is created to add to the test universe.
    
    Returns df_test updated with extrapolated snapshots.
    """
    if df.empty:
        return df_test
    
    analysis_date = pd.Timestamp(analysis_date).normalize()

    # Tickers already in the test for analysis_date
    test_tickers = set(df_test.index.get_level_values("ticker").unique())
    
    # Todos los tickers disponibles
    all_tickers = set(df.index.get_level_values("ticker").unique())
    
    # Tickers que NO tienen snapshot usable en analysis_date
    missing_tickers = all_tickers - test_tickers
    
    extrapolated_rows = []
    
    analysis_snapshot_date = analysis_date
    analysis_quarter = analysis_date.to_period("Q")

    for ticker in missing_tickers:
        # Obtain all historical snapshots for this ticker
        ticker_data = df.loc[df.index.get_level_values("ticker") == ticker].copy()

        # Evitar leakage: usar solo snapshots previos a la fecha analizada.
        if "snapshot_date" in ticker_data.columns:
            ticker_snapshot_dates = pd.to_datetime(ticker_data["snapshot_date"], errors="coerce").dt.normalize()
        else:
            ticker_snapshot_dates = pd.to_datetime(ticker_data.index.get_level_values("date"), errors="coerce").normalize()
        ticker_data = ticker_data.loc[ticker_snapshot_dates < analysis_date]
        
        if len(ticker_data) < lookback_quarters:
            # No hay suficiente historia
            continue
        
        # Ordenar por date
        ticker_data = ticker_data.sort_index()
        
        # Take the last `lookback_quarters` snapshots
        recent_snapshots = ticker_data.tail(lookback_quarters)
        
        if len(recent_snapshots) < lookback_quarters:
            continue
        
        # Detect numeric columns to average
        numeric_cols = recent_snapshots.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
        
        # Crear snapshot promediado
        aggregated = {}
        for col in numeric_cols:
            aggregated[col] = recent_snapshots[col].mean()
        
        # Copy non-numeric columns from the most recent snapshot
        last_row = recent_snapshots.iloc[-1]
        for col in recent_snapshots.columns:
            if col not in numeric_cols:
                aggregated[col] = last_row[col]
        
        # Create multi-index to add to df_test (ticker, date)
        new_index = (ticker, analysis_snapshot_date)

        # Forzar metadatos del quarter objetivo para no arrastrar valores del quarter previo.
        aggregated["year_quarter"] = f"{analysis_quarter.year}Q{analysis_quarter.quarter}"
        aggregated["snapshot_date"] = analysis_snapshot_date
        aggregated["is_fundamental_carry_forward"] = True

        # Keep traceability of the report actually used (the most recent historical one).
        if "report_end_date_used" not in aggregated or pd.isna(aggregated.get("report_end_date_used")):
            aggregated["report_end_date_used"] = last_row.get("report_end_date_used", last_row.name[1] if isinstance(last_row.name, tuple) else pd.NaT)
        if "report_filed_date_used" not in aggregated:
            aggregated["report_filed_date_used"] = last_row.get("report_filed_date_used", pd.NaT)
        
        # Agregar fila extrapolada
        extrapolated_rows.append((new_index, aggregated))
    
    if not extrapolated_rows:
        return df_test
    
    # Construir dataframe con los snapshots extrapolados
    extrapolated_indices = [idx for idx, _ in extrapolated_rows]
    extrapolated_data = [data for _, data in extrapolated_rows]
    
    df_extrapolated = pd.DataFrame(extrapolated_data)
    df_extrapolated.index = pd.MultiIndex.from_tuples(
        extrapolated_indices,
        names=["ticker", "date"]
    )
    
    # Agregar columnas faltantes que puedan estar en df_test
    for col in df_test.columns:
        if col not in df_extrapolated.columns:
            df_extrapolated[col] = None
    
    # Combinar df_test original con extrapolados
    df_test_extended = pd.concat([df_test, df_extrapolated], axis=0)
    df_test_extended = df_test_extended[~df_test_extended.index.duplicated(keep="first")]
    
    log.info(
        "[Fallback Extrapolation] Added %d estimated snapshots up to %s (last %d Q)",
        len(extrapolated_rows),
        analysis_snapshot_date.date(),
        int(lookback_quarters),
    )
    
    return df_test_extended


def _spy_quarterly_returns(spy_prices: pd.Series) -> Dict[str, float]:
    """
    Pre-computes the quarterly SPY return for each quarter present in spy_prices.

    For each quarter Q, computes: (last_price_day_Q / last_price_day_Q-1) - 1
    using only closing prices at the end of each quarter.

    Returns a dict {quarter_period_str: return_float}, e.g.:
        {"2024Q1": 0.107, "2024Q2": -0.032, ...}
    """
    spy = spy_prices.sort_index().dropna()
    quarterly = spy.resample("QE").last()   # last price of each quarter
    spy_returns: Dict[str, float] = {}
    for i in range(1, len(quarterly)):
        p0 = quarterly.iloc[i - 1]
        p1 = quarterly.iloc[i]
        period = quarterly.index[i].to_period("Q")
        if p0 > 0:
            spy_returns[str(period)] = float(p1 / p0 - 1)
    return spy_returns


def _compute_partial_forward_returns(
    prices_dict: Dict[str, pd.DataFrame],
    tickers: pd.Index,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
) -> Dict[str, float]:
    returns: Dict[str, float] = {}
    for ticker in tickers:
        prices = prices_dict.get(ticker)
        if prices is None or prices.empty:
            continue
        cc = _get_close_column(prices)
        period = prices.loc[entry_date:exit_date, cc]
        if len(period) < 2:
            continue
        p0 = float(period.iloc[0])
        p1 = float(period.iloc[-1])
        if p0 > 0 and not pd.isna(p0) and not pd.isna(p1):
            returns[ticker] = (p1 - p0) / p0
    return returns


def _compute_forward_return_from_prices(
    prices: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    lag_days: int,
    holding_period_months: int,
    entry_date_override: Optional[pd.Timestamp] = None,
) -> float | None:
    if prices is None or prices.empty:
        return None
    cc = _get_close_column(prices)
    entry_date = entry_date_override if entry_date_override is not None else (snapshot_date + pd.Timedelta(days=max(int(lag_days), 0)))
    exit_date = entry_date + pd.DateOffset(months=max(int(holding_period_months), 1))
    entry_window = prices.loc[prices.index <= entry_date, cc]
    exit_window = prices.loc[prices.index <= exit_date, cc]
    if entry_window.empty or exit_window.empty:
        return None
    p0 = float(entry_window.iloc[-1])
    p1 = float(exit_window.iloc[-1])
    if p0 <= 0 or pd.isna(p0) or pd.isna(p1):
        return None
    return (p1 - p0) / p0


def _recompute_forward_returns(
    df_part: pd.DataFrame,
    prices_dict: Dict[str, pd.DataFrame],
    lag_days: int,
    holding_period_months: int,
    filing_date_map: Optional[Dict[str, Dict[pd.Timestamp, pd.Timestamp]]] = None,
    post_filing_delay_days: int = 0,
) -> pd.DataFrame:
    if df_part.empty:
        return df_part
    df_out = df_part.copy()
    has_snapshot_date = "snapshot_date" in df_out.columns
    vals = []
    for (ticker, dt), row in df_out.iterrows():
        snapshot_dt = pd.Timestamp(dt)
        if has_snapshot_date and pd.notna(row.get("snapshot_date")):
            snapshot_dt = pd.Timestamp(row.get("snapshot_date"))

        effective_lag_days = 0 if has_snapshot_date else lag_days
        filed_dt = None
        if filing_date_map is not None and not has_snapshot_date:
            filed_dt = filing_date_map.get(str(ticker), {}).get(pd.Timestamp(dt).normalize())
            if filed_dt is not None:
                filed_dt = filed_dt + pd.Timedelta(days=max(int(post_filing_delay_days), 0))
        ret = _compute_forward_return_from_prices(
            prices=prices_dict.get(str(ticker)),
            snapshot_date=snapshot_dt,
            lag_days=effective_lag_days,
            holding_period_months=holding_period_months,
            entry_date_override=filed_dt,
        )
        vals.append(ret)
    df_out["forward_return"] = vals
    return df_out


def _prepare_fold_labels(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    spy_prices: Optional[pd.Series] = None,
    sector_map: Optional[Dict[str, str]] = None,
    prices_dict: Optional[Dict[str, pd.DataFrame]] = None,
    lag_days: int = 45,
    holding_period_months: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    if not prices_dict:
        raise ValueError("TP/SL target generation requires prices_dict for every fold.")

    log.info(
        "[TP/SL] Building adaptive labels for fold: train_rows=%d, test_rows=%d, strategies=%d",
        len(df_train),
        len(df_test),
        3,
    )

    strategy_candidates: List[Dict[str, object]] = []
    adaptive_profile_cache: Dict[tuple, Dict[str, object]] = {}

    base_strategies = ["conservative", "balanced", "aggressive"]
    strategy_names = list(base_strategies)
    max_relax = max(int(TP_SL_FINE_TUNE_MAX_RELAX_STEPS), 0)
    if bool(TP_SL_ENABLE_STRATEGY_FINE_TUNING) and max_relax > 0:
        for base in base_strategies:
            for step in range(1, max_relax + 1):
                strategy_names.append(f"{base}__relax{step}")

    for s_name in strategy_names:
        strategy_t0 = perf_counter()
        log.info("[TP/SL] Strategy=%s -> generating train adaptive targets...", s_name)
        train_targets = _build_adaptive_tp_sl_targets(
            df_target=df_train,
            history_source=df_train,
            prices_dict=prices_dict,
            lag_days=lag_days,
            holding_period_months=holding_period_months,
            strategy_name=s_name,
            profile_cache=adaptive_profile_cache,
        )
        log.info("[TP/SL] Strategy=%s -> generating test adaptive targets...", s_name)
        test_targets = _build_adaptive_tp_sl_targets(
            df_target=df_test,
            history_source=df_train,
            prices_dict=prices_dict,
            lag_days=lag_days,
            holding_period_months=holding_period_months,
            strategy_name=s_name,
            profile_cache=adaptive_profile_cache,
        )

        y_train_s = pd.to_numeric(train_targets["hit_label"], errors="coerce").dropna().astype(int)
        y_test_s = pd.to_numeric(test_targets["hit_label"], errors="coerce").dropna().astype(int)
        common_train = df_train.index.intersection(y_train_s.index)
        if len(common_train) == 0 or y_train_s.nunique() < 2 or len(y_test_s) == 0:
            log.info(
                "[TP/SL] Strategy=%s discarded (common_train=%d, unique_labels=%d, test_labels=%d) [%.1fs]",
                s_name,
                len(common_train),
                int(y_train_s.nunique()) if len(y_train_s) else 0,
                len(y_test_s),
                perf_counter() - strategy_t0,
            )
            continue

        tp_train = pd.to_numeric(train_targets["tp_level"].reindex(common_train), errors="coerce").fillna(float(TP_SL_BASE_TP))
        sl_train = pd.to_numeric(train_targets["sl_level"].reindex(common_train), errors="coerce").fillna(float(TP_SL_BASE_SL))
        y_train_al = y_train_s.reindex(common_train).fillna(0).astype(int)
        utility = float((y_train_al * tp_train - (1.0 - y_train_al) * sl_train).mean())
        hit_rate = float(y_train_al.mean())

        strategy_candidates.append(
            {
                "name": s_name,
                "train_targets": train_targets,
                "test_targets": test_targets,
                "y_train": y_train_s,
                "y_test": y_test_s,
                "utility": utility,
                "hit_rate": hit_rate,
            }
        )
        log.info(
            "[TP/SL] Strategy=%s ready: utility=%.4f, hit_rate=%.2f%%, train_labels=%d, test_labels=%d [%.1fs]",
            s_name,
            utility,
            100.0 * hit_rate,
            len(y_train_s),
            len(y_test_s),
            perf_counter() - strategy_t0,
        )

    if not strategy_candidates:
        raise ValueError("Fold has no valid TP/SL strategy labels after target generation.")

    base_candidates = [
        c for c in strategy_candidates if "__relax" not in str(c.get("name", ""))
    ]
    base_candidates = sorted(
        base_candidates,
        key=lambda x: (float(x.get("utility", -1e9)), float(x.get("hit_rate", -1e9))),
        reverse=True,
    )

    use_relaxed = False
    if base_candidates and bool(TP_SL_ENABLE_STRATEGY_FINE_TUNING):
        best_base = base_candidates[0]
        base_utility = float(best_base.get("utility", -1e9))
        base_hit = float(best_base.get("hit_rate", 0.0))
        if base_utility < float(TP_SL_FINE_TUNE_MIN_UTILITY) or base_hit < float(TP_SL_FINE_TUNE_MIN_HIT_RATE):
            use_relaxed = True
            log.info(
                "[TP/SL] Fine-tuning activated: base_best=%s utility=%.4f hit_rate=%.2f%% (min_utility=%.4f, min_hit=%.2f%%)",
                str(best_base.get("name", "")),
                base_utility,
                100.0 * base_hit,
                float(TP_SL_FINE_TUNE_MIN_UTILITY),
                100.0 * float(TP_SL_FINE_TUNE_MIN_HIT_RATE),
            )

    if not use_relaxed and base_candidates:
        strategy_candidates = base_candidates

    strategy_candidates = sorted(
        strategy_candidates,
        key=lambda x: (float(x.get("utility", -1e9)), float(x.get("hit_rate", -1e9))),
        reverse=True,
    )
    best = strategy_candidates[0]
    selected_strategy_variant = str(best["name"])
    selected_strategy = selected_strategy_variant.split("__", 1)[0]
    train_targets = best["train_targets"]
    test_targets = best["test_targets"]
    y_train = best["y_train"]
    y_test = best["y_test"]
    log.info(
        "[TP/SL] Selected training strategy=%s (variant=%s) | utility=%.4f | hit_rate=%.2f%%",
        selected_strategy,
        selected_strategy_variant,
        float(best["utility"]),
        100.0 * float(best["hit_rate"]),
    )

    df_train = df_train.loc[y_train.index]
    df_test  = df_test.loc[y_test.index]

    if y_train.empty or y_test.empty:
        raise ValueError("Fold has no valid TP/SL hit labels after target generation.")

    df_train["tp_level"] = pd.to_numeric(train_targets["tp_level"].reindex(df_train.index), errors="coerce")
    df_train["sl_level"] = pd.to_numeric(train_targets["sl_level"].reindex(df_train.index), errors="coerce")
    df_train["tp_sl_outcome"] = train_targets["outcome"].reindex(df_train.index)
    df_train["tp_sl_strategy"] = selected_strategy
    df_train["tp_sl_strategy_variant"] = selected_strategy_variant

    df_test["tp_level"] = pd.to_numeric(test_targets["tp_level"].reindex(df_test.index), errors="coerce")
    df_test["sl_level"] = pd.to_numeric(test_targets["sl_level"].reindex(df_test.index), errors="coerce")
    df_test["tp_sl_outcome"] = test_targets["outcome"].reindex(df_test.index)
    df_test["tp_sl_strategy"] = selected_strategy
    df_test["tp_sl_strategy_variant"] = selected_strategy_variant

    # Benchmark-relative alpha targets for ranking-sensitive meta training.
    def _benchmark_returns_for_rows(df_part: pd.DataFrame) -> pd.Series:
        idx = df_part.index
        has_snapshot_date = "snapshot_date" in df_part.columns
        if spy_prices is None or len(spy_prices) == 0:
            return pd.Series(0.0, index=idx, dtype=float)

        spy_series = pd.to_numeric(pd.Series(spy_prices), errors="coerce").dropna()
        if spy_series.empty:
            return pd.Series(0.0, index=idx, dtype=float)
        spy_series.index = pd.to_datetime(spy_series.index, errors="coerce")
        spy_series = spy_series[~spy_series.index.isna()].sort_index()
        if spy_series.empty:
            return pd.Series(0.0, index=idx, dtype=float)

        unique_snaps: set[pd.Timestamp] = set()
        if "snapshot_date" in df_part.columns:
            snap_vals = pd.to_datetime(df_part["snapshot_date"], errors="coerce").dropna()
            unique_snaps.update(pd.Timestamp(x).normalize() for x in snap_vals.tolist())
        if isinstance(idx, pd.MultiIndex) and "date" in idx.names:
            unique_snaps.update(pd.Timestamp(x).normalize() for x in idx.get_level_values("date").tolist())
        else:
            unique_snaps.update(pd.Timestamp(x).normalize() for x in idx.tolist())

        bench_map: Dict[pd.Timestamp, float] = {}
        for snap in unique_snaps:
            effective_lag_days = 0 if has_snapshot_date else max(int(lag_days), 0)
            entry = snap + pd.Timedelta(days=effective_lag_days)
            exit_dt = entry + pd.DateOffset(months=max(int(holding_period_months), 1))
            entry_window = spy_series.loc[spy_series.index <= entry]
            exit_window = spy_series.loc[spy_series.index <= exit_dt]
            if entry_window.empty or exit_window.empty:
                bench_map[snap] = float("nan")
                continue
            p0 = float(entry_window.iloc[-1])
            p1 = float(exit_window.iloc[-1])
            if not np.isfinite(p0) or p0 <= 0 or not np.isfinite(p1):
                bench_map[snap] = float("nan")
                continue
            bench_map[snap] = float((p1 - p0) / p0)

        if "snapshot_date" in df_part.columns:
            snap_series = pd.to_datetime(df_part["snapshot_date"], errors="coerce")
            bench = snap_series.map(lambda d: bench_map.get(pd.Timestamp(d).normalize(), np.nan) if pd.notna(d) else np.nan)
            bench.index = idx
        elif isinstance(idx, pd.MultiIndex) and "date" in idx.names:
            dvals = pd.to_datetime(idx.get_level_values("date"), errors="coerce")
            bench = pd.Series(
                [bench_map.get(pd.Timestamp(d).normalize(), np.nan) if pd.notna(d) else np.nan for d in dvals],
                index=idx,
                dtype=float,
            )
        else:
            dvals = pd.to_datetime(idx, errors="coerce")
            bench = pd.Series(
                [bench_map.get(pd.Timestamp(d).normalize(), np.nan) if pd.notna(d) else np.nan for d in dvals],
                index=idx,
                dtype=float,
            )

        map_vals = pd.Series(list(bench_map.values()), dtype=float)
        fallback = float(map_vals.dropna().median()) if not map_vals.dropna().empty else 0.0
        return pd.to_numeric(bench, errors="coerce").fillna(fallback)

    fr_train = pd.to_numeric(df_train.get("forward_return", pd.Series(index=df_train.index, dtype=float)), errors="coerce")
    fr_test = pd.to_numeric(df_test.get("forward_return", pd.Series(index=df_test.index, dtype=float)), errors="coerce")
    bench_train = _benchmark_returns_for_rows(df_train)
    bench_test = _benchmark_returns_for_rows(df_test)
    alpha_train = (fr_train.reindex(df_train.index) - bench_train.reindex(df_train.index)).clip(-1.5, 1.5)
    alpha_test = (fr_test.reindex(df_test.index) - bench_test.reindex(df_test.index)).clip(-1.5, 1.5)
    alpha_train = pd.to_numeric(alpha_train, errors="coerce").fillna(0.0)
    alpha_test = pd.to_numeric(alpha_test, errors="coerce").fillna(0.0)

    return (
        df_train,
        df_test,
        y_train,
        y_test,
        alpha_train,
        alpha_test,
    )


def _build_selection_df(preds_scored: pd.DataFrame, selected_tickers: List[str], ticker_weights: Dict[str, float]) -> pd.DataFrame:
    if not selected_tickers:
        return pd.DataFrame(columns=["ticker", "weight", "score", "rank"])
    score_map = (
        preds_scored[["ticker", "score"]]
        .drop_duplicates(subset=["ticker"], keep="last")
        .set_index("ticker")["score"]
        .to_dict()
    )
    rows = []
    for rank, t in enumerate(selected_tickers, start=1):
        rows.append({
            "ticker": t,
            "weight": float(ticker_weights.get(t, 0.0)),
            "score": float(score_map.get(t, np.nan)) if t in score_map else np.nan,
            "rank": rank,
        })
    return pd.DataFrame(rows)


def _strategy_quantiles(strategy_name: str) -> tuple[float, float]:
    name = str(strategy_name).strip().lower()
    base = name
    relax_step = 0
    if "__relax" in name:
        parts = name.split("__relax", 1)
        base = parts[0].strip() or "balanced"
        try:
            relax_step = max(int(parts[1]), 0)
        except Exception:
            relax_step = 1

    if base == "conservative":
        q_tp, q_sl = 0.55, 0.45
    elif base == "aggressive":
        q_tp, q_sl = 0.80, 0.75
    else:  # balanced
        q_tp, q_sl = 0.70, 0.60

    if relax_step > 0:
        q_tp = float(np.clip(q_tp - (0.05 * relax_step), 0.30, 0.85))
        q_sl = float(np.clip(q_sl + (0.03 * relax_step), 0.35, 0.90))
    return float(q_tp), float(q_sl)


def _compute_ticker_tp_edge(
    history_df: pd.DataFrame,
    *,
    prior_strength: float,
    reliability_k: float,
    none_score: float,
) -> pd.DataFrame:
    """Estimate ticker-level TP edge from train fold outcomes (no leakage)."""
    if history_df is None or history_df.empty or "tp_sl_outcome" not in history_df.columns:
        return pd.DataFrame(columns=["ticker", "historical_tp_prob", "historical_tp_edge", "historical_tp_obs"])

    idx_ticker = pd.Index(history_df.index.get_level_values("ticker")).astype(str)
    outcome = history_df["tp_sl_outcome"].astype(str).str.upper().fillna("NONE")
    score_map = {
        "TP": 1.0,
        "SL": 0.0,
        "NONE": float(none_score),
    }
    outcome_score = outcome.map(score_map).fillna(float(none_score))

    temp = pd.DataFrame(
        {
            "ticker": idx_ticker.values,
            "outcome_score": pd.to_numeric(outcome_score, errors="coerce").fillna(float(none_score)).values,
            "is_tp": (outcome == "TP").astype(float).values,
            "is_sl": (outcome == "SL").astype(float).values,
            "is_none": (outcome == "NONE").astype(float).values,
        }
    )

    grp = temp.groupby("ticker", dropna=False)
    stats = grp.agg(
        outcome_sum=("outcome_score", "sum"),
        historical_tp_obs=("outcome_score", "size"),
        tp_rate=("is_tp", "mean"),
        sl_rate=("is_sl", "mean"),
        none_rate=("is_none", "mean"),
    ).reset_index()

    n = pd.to_numeric(stats["historical_tp_obs"], errors="coerce").fillna(0.0)
    outcome_sum = pd.to_numeric(stats["outcome_sum"], errors="coerce").fillna(0.0)
    prior = max(float(prior_strength), 0.0)
    rel_k = max(float(reliability_k), 0.0)

    # Bayesian posterior mean around neutral prior 0.5.
    tp_prob = (outcome_sum + prior * 0.5) / (n + prior).replace(0.0, np.nan)
    tp_prob = pd.to_numeric(tp_prob, errors="coerce").fillna(0.5).clip(0.0, 1.0)
    reliability = np.sqrt(n / (n + rel_k).replace(0.0, np.nan)) if rel_k > 0 else pd.Series(1.0, index=stats.index)
    reliability = pd.to_numeric(reliability, errors="coerce").fillna(0.0).clip(0.0, 1.0)

    # Edge in [-1, 1], reliability-shrunk.
    edge = ((tp_prob - 0.5) * 2.0 * reliability).clip(-1.0, 1.0)

    out = stats[["ticker", "historical_tp_obs", "tp_rate", "sl_rate", "none_rate"]].copy()
    out["historical_tp_prob"] = tp_prob.values
    out["historical_tp_edge"] = edge.values
    return out


def _get_cached_ticker_snapshot_profile(
    *,
    ticker: str,
    snapshot_date: pd.Timestamp,
    history_df_ticker: pd.DataFrame,
    prices_dict: Dict[str, pd.DataFrame],
    lag_days: int,
    holding_period_months: int,
    profile_cache: Optional[Dict[tuple, Dict[str, object]]] = None,
) -> Dict[str, object]:
    snap_ts = pd.Timestamp(snapshot_date).normalize()
    cache_key = (str(ticker), snap_ts, int(lag_days), int(holding_period_months))
    if profile_cache is not None and cache_key in profile_cache:
        return profile_cache[cache_key]

    profile = _build_ticker_snapshot_profile(
        ticker=str(ticker),
        snapshot_date=snap_ts,
        history_df_ticker=history_df_ticker,
        prices_dict=prices_dict,
        lag_days=lag_days,
        holding_period_months=holding_period_months,
    )
    if profile_cache is not None:
        profile_cache[cache_key] = profile
    return profile


def _tp_feasibility_from_profile(
    *,
    tp_pct: float,
    strategy_name: str,
    profile: Dict[str, object],
) -> float:
    """Penalize TP targets stretched far above ticker's historical reachable TP."""
    if not bool(profile.get("has_signal", False)):
        return 1.0
    q_tp, _ = _strategy_quantiles(strategy_name)
    tp_map = profile.get("tp_quantiles", {}) or {}
    ref_tp = float(tp_map.get(float(q_tp), np.nan))
    if not np.isfinite(ref_tp) or ref_tp <= 0:
        return 1.0
    tp_val = float(tp_pct)
    if not np.isfinite(tp_val) or tp_val <= 0:
        return 1.0
    stretch = tp_val / ref_tp
    if stretch <= 1.0:
        return 1.0
    penalty = float(TP_EDGE_TP_STRETCH_PENALTY) * (stretch - 1.0)
    feasibility = 1.0 - penalty
    return float(np.clip(feasibility, float(TP_EDGE_MIN_FEASIBILITY), 1.0))


def _extract_close_from_prices_obj(price_obj) -> pd.Series:
    if price_obj is None:
        return pd.Series(dtype=float)
    if isinstance(price_obj, pd.Series):
        s = pd.to_numeric(price_obj, errors="coerce").dropna()
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    if isinstance(price_obj, pd.DataFrame) and not price_obj.empty:
        c = "Close" if "Close" in price_obj.columns else price_obj.columns[-1]
        s = pd.to_numeric(price_obj[c], errors="coerce").dropna()
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    return pd.Series(dtype=float)


def _compute_trailing_drawdown_pct(
    prices: pd.Series,
    *,
    snap_ts: pd.Timestamp,
    quantile: float = 0.65,
    rolling_window: int = 22,
    min_periods: int = 60,
) -> float:
    """Compute the typical pullback from rolling peak for this stock.

    Calculates a rolling 22-day drawdown from rolling 22-day high across the
    stock's full price history up to snap_ts. Returns the `quantile`-th
    percentile of absolute drawdown values, clipped to [0.04, 0.40].
    Used to calibrate the trailing stop distance per ticker.
    """
    if prices is None or prices.empty:
        return 0.12
    hist = prices.loc[prices.index <= snap_ts].dropna()
    if len(hist) < min_periods:
        return 0.12
    w = min(int(rolling_window), len(hist) // 4)
    if w < 5:
        return 0.12
    rolling_high = hist.rolling(window=w, min_periods=max(w // 2, 5)).max()
    drawdown = (hist / rolling_high) - 1.0  # values <= 0
    valid = drawdown.dropna()
    valid = valid[valid < 0.0]
    if len(valid) < 10:
        return 0.12
    return float(np.clip(np.abs(valid).quantile(quantile), 0.04, 0.40))


def _historical_quarter_path_stats(
    *,
    prices: pd.Series,
    snapshot_dates: List[pd.Timestamp],
    lag_days: int,
    holding_period_months: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if prices is None or prices.empty:
        return pd.DataFrame()
    for snap in snapshot_dates:
        snap_ts = pd.Timestamp(snap)
        entry_req = snap_ts + pd.Timedelta(days=max(int(lag_days), 0))
        entry_idx = prices.index[prices.index >= entry_req]
        if len(entry_idx) == 0:
            continue
        actual_entry = pd.Timestamp(entry_idx[0])
        entry_price = float(prices.loc[actual_entry])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue
        end_ts = actual_entry + pd.DateOffset(months=max(int(holding_period_months), 1))
        window = prices.loc[(prices.index > actual_entry) & (prices.index <= end_ts)]
        if window.empty:
            continue
        rel = (window / entry_price) - 1.0
        mfe = float(rel.max())
        mae = float(rel.min())
        peak_dt = pd.Timestamp(rel.idxmax())
        peak = float(rel.loc[peak_dt])
        tail = rel.loc[rel.index >= peak_dt]
        final_rel = float(rel.iloc[-1])
        fade = float(max(peak - final_rel, 0.0))
        rows.append(
            {
                "snapshot_date": snap_ts.normalize(),
                "mfe": mfe,
                "mae": mae,
                "mae_abs": abs(mae),
                "fade_after_peak": fade,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["snapshot_date"], keep="last")


def _build_ticker_snapshot_profile(
    *,
    ticker: str,
    snapshot_date: pd.Timestamp,
    history_df_ticker: pd.DataFrame,
    prices_dict: Dict[str, pd.DataFrame],
    lag_days: int,
    holding_period_months: int,
) -> Dict[str, object]:
    prices = _extract_close_from_prices_obj(prices_dict.get(str(ticker)))
    if prices.empty or history_df_ticker is None or history_df_ticker.empty:
        return {
            "tp_quantiles": {},
            "sl_quantiles": {},
            "fade": 0.0,
            "metric_adjust_tp": 0.0,
            "metric_adjust_sl": 0.0,
            "has_signal": False,
        }

    snap_ts = pd.Timestamp(snapshot_date).normalize()
    if isinstance(history_df_ticker.index, pd.MultiIndex):
        hist_dates = pd.to_datetime(history_df_ticker.index.get_level_values("date"), errors="coerce")
    else:
        hist_dates = pd.to_datetime(history_df_ticker.index, errors="coerce")

    hist_dates_s = pd.Series(hist_dates).dropna()
    prior_dates = sorted(hist_dates_s.loc[hist_dates_s < snap_ts].unique().tolist())
    if len(prior_dates) == 0:
        return {
            "tp_quantiles": {},
            "sl_quantiles": {},
            "fade": 0.0,
            "metric_adjust_tp": 0.0,
            "metric_adjust_sl": 0.0,
            "has_signal": False,
        }

    path_stats = _historical_quarter_path_stats(
        prices=prices,
        snapshot_dates=[pd.Timestamp(d) for d in prior_dates],
        lag_days=lag_days,
        holding_period_months=holding_period_months,
    )
    if path_stats.empty:
        return {
            "tp_quantiles": {},
            "sl_quantiles": {},
            "fade": 0.0,
            "metric_adjust_tp": 0.0,
            "metric_adjust_sl": 0.0,
            "has_signal": False,
        }

    mfe_pos = path_stats.loc[path_stats["mfe"] > 0.0, "mfe"]
    mae_abs = path_stats.loc[path_stats["mae_abs"] > 0.0, "mae_abs"]
    q_values = [0.40, 0.45, 0.55, 0.60, 0.70, 0.75, 0.80]

    tp_quantiles: Dict[float, float] = {}
    sl_quantiles: Dict[float, float] = {}
    for q in q_values:
        tp_q = float(mfe_pos.quantile(q)) if not mfe_pos.empty else float(path_stats["mfe"].clip(lower=0.0).median())
        sl_q = float(mae_abs.quantile(q)) if not mae_abs.empty else float(path_stats["mae_abs"].median())
        tp_quantiles[float(q)] = tp_q
        sl_quantiles[float(q)] = sl_q

    fade = float(path_stats["fade_after_peak"].quantile(0.60)) if "fade_after_peak" in path_stats.columns else 0.0

    metric_cols = [
        "momentum_3m",
        "momentum_6m",
        "rsi_14",
        "volatility_60d",
        "revenue_yoy_growth",
        "eps_surprise_pct",
    ]
    metric_adjust_tp = 0.0
    metric_adjust_sl = 0.0
    used = 0

    hist_features = history_df_ticker.copy()
    if isinstance(hist_features.index, pd.MultiIndex) and "date" in hist_features.index.names:
        hist_features = hist_features.reset_index().set_index("date")
    hist_features.index = pd.to_datetime(hist_features.index, errors="coerce")
    hist_features = hist_features[~hist_features.index.isna()].sort_index()

    path_stats_idx = path_stats.copy().set_index("snapshot_date")
    for mc in metric_cols:
        if mc not in hist_features.columns:
            continue
        m_hist = pd.to_numeric(hist_features[mc], errors="coerce")
        joined = pd.concat([m_hist, path_stats_idx[["mfe", "mae_abs"]]], axis=1, join="inner").dropna()
        if len(joined) < 8:
            continue
        corr_tp = float(joined[mc].corr(joined["mfe"])) if joined[mc].nunique() > 1 else 0.0
        corr_mae = float(joined[mc].corr(joined["mae_abs"])) if joined[mc].nunique() > 1 else 0.0
        cur_candidates = m_hist.loc[m_hist.index == snap_ts]
        if cur_candidates.empty:
            continue
        cur_val = float(cur_candidates.iloc[-1])
        pct = float((joined[mc] <= cur_val).mean())
        centered = pct - 0.5
        metric_adjust_tp += corr_tp * centered
        metric_adjust_sl += corr_mae * centered
        used += 1

    if used > 0:
        metric_adjust_tp /= used
        metric_adjust_sl /= used

    trailing_drawdown_pct = _compute_trailing_drawdown_pct(
        prices=prices,
        snap_ts=snap_ts,
        quantile=float(TP_SL_TRAILING_DRAWDOWN_QUANTILE),
    )

    return {
        "tp_quantiles": tp_quantiles,
        "sl_quantiles": sl_quantiles,
        "fade": float(fade),
        "metric_adjust_tp": float(metric_adjust_tp),
        "metric_adjust_sl": float(metric_adjust_sl),
        "trailing_drawdown_pct": float(trailing_drawdown_pct),
        "has_signal": True,
    }


def _adaptive_tp_sl_for_ticker_snapshot(
    *,
    ticker: str,
    snapshot_date: pd.Timestamp,
    strategy_name: str,
    history_df_ticker: pd.DataFrame,
    prices_dict: Dict[str, pd.DataFrame],
    lag_days: int,
    holding_period_months: int,
    profile_cache: Optional[Dict[tuple, Dict[str, object]]] = None,
) -> tuple[float, float]:
    profile = _get_cached_ticker_snapshot_profile(
        ticker=str(ticker),
        snapshot_date=pd.Timestamp(snapshot_date),
        history_df_ticker=history_df_ticker,
        prices_dict=prices_dict,
        lag_days=lag_days,
        holding_period_months=holding_period_months,
        profile_cache=profile_cache,
    )

    if not bool(profile.get("has_signal", False)):
        return float(TP_SL_BASE_TP), float(TP_SL_BASE_SL)

    q_tp, q_sl = _strategy_quantiles(strategy_name)
    tp_map = profile.get("tp_quantiles", {}) or {}
    sl_map = profile.get("sl_quantiles", {}) or {}
    tp = float(tp_map.get(float(q_tp), TP_SL_BASE_TP))
    sl = float(sl_map.get(float(q_sl), TP_SL_BASE_SL))

    if not np.isfinite(tp) or tp <= 0:
        tp = float(TP_SL_BASE_TP)
    if not np.isfinite(sl) or sl <= 0:
        sl = float(TP_SL_BASE_SL)

    fade = float(profile.get("fade", 0.0))
    tp *= (1.0 - min(max(fade, 0.0), 0.5) * 0.35)
    metric_adjust_tp = float(profile.get("metric_adjust_tp", 0.0))
    metric_adjust_sl = float(profile.get("metric_adjust_sl", 0.0))
    tp *= (1.0 + float(np.clip(metric_adjust_tp, -0.25, 0.25)))
    sl *= (1.0 - float(np.clip(metric_adjust_sl, -0.25, 0.25)))

    tp = float(np.clip(tp, float(TP_SL_MIN_TP), float(TP_SL_MAX_TP)))
    sl = float(np.clip(sl, float(TP_SL_MIN_SL), float(TP_SL_MAX_SL)))
    return tp, sl


def _build_adaptive_tp_sl_targets(
    *,
    df_target: pd.DataFrame,
    history_source: pd.DataFrame,
    prices_dict: Dict[str, pd.DataFrame],
    lag_days: int,
    holding_period_months: int,
    strategy_name: str,
    profile_cache: Optional[Dict[tuple, Dict[str, object]]] = None,
) -> Dict[str, pd.Series]:
    idx = df_target.index
    tp_level = pd.Series(np.nan, index=idx, dtype=float)
    sl_level = pd.Series(np.nan, index=idx, dtype=float)
    outcome = pd.Series(index=idx, dtype="object")
    hit_label = pd.Series(np.nan, index=idx, dtype=float)

    base_ts = pd.Timestamp("2000-01-01")
    max_holding_days = int((base_ts + pd.DateOffset(months=max(int(holding_period_months), 1)) - base_ts).days)

    by_ticker_history: Dict[str, pd.DataFrame] = {}
    if isinstance(history_source.index, pd.MultiIndex):
        for tk in history_source.index.get_level_values("ticker").unique().tolist():
            by_ticker_history[str(tk)] = history_source.xs(tk, level="ticker", drop_level=False).copy()

    total_rows = len(df_target)
    if total_rows == 0:
        return {
            "tp_level": tp_level,
            "sl_level": sl_level,
            "outcome": outcome,
            "hit_label": hit_label,
            "max_holding_days": pd.Series(float(max_holding_days), index=idx, dtype=float),
        }

    # Roughly 20 progress updates per pass, with a sane floor to avoid noise.
    report_every = max(250, total_rows // 20)
    loop_t0 = perf_counter()
    tp_count = 0
    sl_count = 0
    none_count = 0

    log.info(
        "[TP/SL][%s] Adaptive target pass start: rows=%d, lag_days=%d, holding_months=%d",
        strategy_name,
        total_rows,
        int(lag_days),
        int(holding_period_months),
    )

    for i, ((ticker, dt), row) in enumerate(df_target.iterrows(), start=1):
        tk = str(ticker)
        hist_tk = by_ticker_history.get(tk, pd.DataFrame())
        has_snapshot = "snapshot_date" in row and pd.notna(row.get("snapshot_date"))
        snap_dt = pd.Timestamp(row.get("snapshot_date")) if has_snapshot else pd.Timestamp(dt)

        profile_for_ticker = _get_cached_ticker_snapshot_profile(
            ticker=tk,
            snapshot_date=snap_dt,
            history_df_ticker=hist_tk,
            prices_dict=prices_dict,
            lag_days=lag_days,
            holding_period_months=holding_period_months,
            profile_cache=profile_cache,
        )
        tp, sl = _adaptive_tp_sl_for_ticker_snapshot(
            ticker=tk,
            snapshot_date=snap_dt,
            strategy_name=strategy_name,
            history_df_ticker=hist_tk,
            prices_dict=prices_dict,
            lag_days=lag_days,
            holding_period_months=holding_period_months,
            profile_cache=profile_cache,
        )
        tp_level.loc[(ticker, dt)] = tp
        sl_level.loc[(ticker, dt)] = sl

        trailing_stop_pct = float(profile_for_ticker.get("trailing_drawdown_pct", 0.12))
        prices = _extract_close_from_prices_obj(prices_dict.get(tk))
        entry_date = pd.Timestamp(snap_dt) + pd.Timedelta(days=max(int(lag_days), 0))
        grace_days = int(max_holding_days * float(TP_SL_GRACE_PERIOD_FRACTION))
        sim = simulate_tp_sl(
            ticker=tk,
            prices=prices,
            entry_date=entry_date,
            tp_pct=float(tp),
            sl_pct=float(sl),
            max_holding_days=max_holding_days,
            min_holding_days=grace_days,
            trailing_stop_pct=trailing_stop_pct,
            trailing_review_days=int(TP_SL_TRAILING_REVIEW_DAYS),
        )
        out = str(sim.get("outcome", "NONE")).upper()
        outcome.loc[(ticker, dt)] = out
        hit_label.loc[(ticker, dt)] = 1.0 if out == "TP" else 0.0

        if out == "TP":
            tp_count += 1
        elif out == "SL":
            sl_count += 1
        else:
            none_count += 1

        if i % report_every == 0 or i == total_rows:
            elapsed = perf_counter() - loop_t0
            rows_per_sec = i / elapsed if elapsed > 0 else 0.0
            log.info(
                "[TP/SL][%s] progress: %d/%d (%.1f%%) | TP=%d SL=%d NONE=%d | %.1fs elapsed (%.1f rows/s)",
                strategy_name,
                i,
                total_rows,
                100.0 * i / total_rows,
                tp_count,
                sl_count,
                none_count,
                elapsed,
                rows_per_sec,
            )

    elapsed_total = perf_counter() - loop_t0
    log.info(
        "[TP/SL][%s] pass complete: rows=%d | TP=%d SL=%d NONE=%d | %.1fs total",
        strategy_name,
        total_rows,
        tp_count,
        sl_count,
        none_count,
        elapsed_total,
    )

    return {
        "tp_level": tp_level,
        "sl_level": sl_level,
        "outcome": outcome,
        "hit_label": hit_label,
        "max_holding_days": pd.Series(float(max_holding_days), index=idx, dtype=float),
    }


def _build_tp_sl_strategy_universe_matrix(
    *,
    preds_df: pd.DataFrame,
    history_source_df: pd.DataFrame,
    prices_dict: Dict[str, pd.DataFrame],
    entry_date: pd.Timestamp,
    lag_days: int,
    holding_period_months: int,
) -> pd.DataFrame:
    """Build universe x strategy TP/SL matrix with realized outcomes."""
    if preds_df is None or preds_df.empty:
        return pd.DataFrame()

    base = preds_df.copy()
    if "ticker" not in base.columns:
        return pd.DataFrame()

    base["confidence"] = pd.to_numeric(base.get("confidence", 0.5), errors="coerce").fillna(0.5).clip(0.0, 1.0)
    base["tp_level"] = pd.to_numeric(base.get("tp_level", TP_SL_BASE_TP), errors="coerce").fillna(float(TP_SL_BASE_TP))
    base["sl_level"] = pd.to_numeric(base.get("sl_level", TP_SL_BASE_SL), errors="coerce").fillna(float(TP_SL_BASE_SL))

    base_ts = pd.Timestamp("2000-01-01")
    period_holding_days = int((base_ts + pd.DateOffset(months=max(int(holding_period_months), 1)) - base_ts).days)

    by_ticker_history: Dict[str, pd.DataFrame] = {}
    if isinstance(history_source_df.index, pd.MultiIndex):
        for tk in history_source_df.index.get_level_values("ticker").unique().tolist():
            by_ticker_history[str(tk)] = history_source_df.xs(tk, level="ticker", drop_level=False).copy()

    profile_cache: Dict[tuple, Dict[str, object]] = {}

    frames: List[pd.DataFrame] = []
    for s_name in ["conservative", "balanced", "aggressive"]:

        s_df = base.copy()
        s_df["strategy"] = s_name

        tp_vals = []
        sl_vals = []
        tp_feasibility_vals = []
        trailing_vals: List[float] = []
        for _, r in s_df.iterrows():
            tk = str(r.get("ticker"))
            snap_dt = pd.Timestamp(r.get("date", entry_date))
            profile = _get_cached_ticker_snapshot_profile(
                ticker=tk,
                snapshot_date=snap_dt,
                history_df_ticker=by_ticker_history.get(tk, pd.DataFrame()),
                prices_dict=prices_dict,
                lag_days=lag_days,
                holding_period_months=holding_period_months,
                profile_cache=profile_cache,
            )
            tp, sl = _adaptive_tp_sl_for_ticker_snapshot(
                ticker=tk,
                snapshot_date=snap_dt,
                strategy_name=s_name,
                history_df_ticker=by_ticker_history.get(tk, pd.DataFrame()),
                prices_dict=prices_dict,
                lag_days=lag_days,
                holding_period_months=holding_period_months,
                profile_cache=profile_cache,
            )
            tp_vals.append(tp)
            sl_vals.append(sl)
            tp_feasibility_vals.append(
                _tp_feasibility_from_profile(
                    tp_pct=float(tp),
                    strategy_name=s_name,
                    profile=profile,
                )
            )
            trailing_vals.append(float(profile.get("trailing_drawdown_pct", 0.12)))
        s_df["tp_pct"] = pd.Series(tp_vals, index=s_df.index, dtype=float)
        s_df["sl_pct"] = pd.Series(sl_vals, index=s_df.index, dtype=float)
        s_df["trailing_stop_pct"] = pd.Series(trailing_vals, index=s_df.index, dtype=float)
        s_df["tp_feasibility"] = pd.Series(tp_feasibility_vals, index=s_df.index, dtype=float).clip(0.0, 1.0)
        s_df["max_holding_days"] = period_holding_days
        s_df["rr_ratio"] = (s_df["tp_pct"] / s_df["sl_pct"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        s_df["ev"] = s_df["confidence"] * s_df["tp_pct"] - (1.0 - s_df["confidence"]) * s_df["sl_pct"]
        s_df["risk_benefit_score"] = (
            s_df["ev"]
            * (1.0 + s_df["rr_ratio"].clip(lower=0.0, upper=4.0) / 4.0)
            * s_df["tp_feasibility"].fillna(1.0)
        )

        signals = s_df[["ticker", "tp_pct", "sl_pct", "trailing_stop_pct"]].drop_duplicates(subset=["ticker"], keep="last")
        bt = run_backtest(
            signals=signals,
            prices_dict=prices_dict,
            entry_date=entry_date,
            max_holding_days=period_holding_days,
        )
        bt = bt.rename(columns={
            "entry_date": "sim_entry_date",
            "actual_entry_date": "sim_actual_entry_date",
            "outcome": "sim_outcome",
            "days_to_outcome": "sim_days_to_outcome",
            "outcome_date": "sim_outcome_date",
        })
        bt_cols = [c for c in [
            "ticker",
            "sim_entry_date",
            "sim_actual_entry_date",
            "entry_price",
            "tp_price",
            "sl_price",
            "sim_outcome",
            "sim_days_to_outcome",
            "sim_outcome_date",
        ] if c in bt.columns]
        if bt_cols:
            s_df = s_df.merge(bt[bt_cols], on="ticker", how="left")

        frames.append(s_df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def _pick_best_strategy_per_ticker(tp_sl_matrix: pd.DataFrame) -> pd.DataFrame:
    """Choose the best strategy row per ticker by risk-benefit score."""
    if tp_sl_matrix is None or tp_sl_matrix.empty:
        return pd.DataFrame()
    df = tp_sl_matrix.copy()
    df["risk_benefit_score"] = pd.to_numeric(df.get("risk_benefit_score", 0.0), errors="coerce").fillna(0.0)
    df["ev"] = pd.to_numeric(df.get("ev", 0.0), errors="coerce").fillna(0.0)
    df["confidence"] = pd.to_numeric(df.get("confidence", 0.5), errors="coerce").fillna(0.5)
    df["tp_feasibility"] = pd.to_numeric(df.get("tp_feasibility", 1.0), errors="coerce").fillna(1.0)
    df["tp_pct"] = pd.to_numeric(df.get("tp_pct", TP_SL_BASE_TP), errors="coerce").fillna(float(TP_SL_BASE_TP))
    df["historical_tp_prob"] = pd.to_numeric(df.get("historical_tp_prob", 0.5), errors="coerce").fillna(0.5).clip(0.0, 1.0)
    # Rule consensus signal in [-1, 1]: positive means rules favour TP, negative means SL risk.
    # Used as a multiplicative boost over risk_benefit_score.
    rule_signal = pd.to_numeric(df.get("rules_consensus_signal", 0.0), errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    df["rule_adjusted_rbs"] = df["risk_benefit_score"] * (1.0 + float(TP_SL_RULE_SIGNAL_RBS_WEIGHT) * rule_signal)

    # Selection score: reward certainty and feasible/acceptable TP levels,
    # while keeping EV/rule-adjusted economics as primary driver.
    certainty = (0.55 * df["confidence"] + 0.45 * df["historical_tp_prob"]).clip(0.0, 1.0)
    certainty *= df["tp_feasibility"].clip(0.0, 1.0)
    tp_floor = float(max(TP_SL_MIN_ACCEPTABLE_TP, 0.0))
    tp_quality = ((df["tp_pct"] - tp_floor) / max(float(TP_SL_MAX_TP) - tp_floor, 1e-6)).clip(0.0, 1.0)
    below_floor_penalty = np.where(df["tp_pct"] < tp_floor, 0.75, 1.0)

    df["certainty_component"] = certainty
    df["tp_quality_component"] = tp_quality
    df["tp_floor_penalty_component"] = below_floor_penalty

    certainty_weight = float(np.clip(TP_SL_SELECTION_CERTAINTY_WEIGHT, 0.0, 1.0))
    tp_quality_weight = float(np.clip(TP_SL_SELECTION_TP_QUALITY_WEIGHT, 0.0, 1.0))
    df["selection_score"] = (
        df["rule_adjusted_rbs"]
        * (1.0 + certainty_weight * (certainty - 0.5))
        * (1.0 + tp_quality_weight * (tp_quality - 0.5))
        * below_floor_penalty
    )

    rank_cols = ["selection_score", "rule_adjusted_rbs", "risk_benefit_score", "tp_feasibility", "ev", "confidence"]
    chosen = (
        df.sort_values(rank_cols, ascending=False)
        .drop_duplicates(subset=["ticker"], keep="first")
        .reset_index(drop=True)
    )
    return chosen


def _build_tp_sl_decision_audit(
    tp_sl_matrix: pd.DataFrame,
    best_per_ticker: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact per-ticker strategy decision trace with reject reasons."""
    if tp_sl_matrix is None or tp_sl_matrix.empty:
        return pd.DataFrame()

    full = tp_sl_matrix.copy()
    best = best_per_ticker.copy() if best_per_ticker is not None else pd.DataFrame()
    best_map = {}
    if not best.empty and "ticker" in best.columns and "strategy" in best.columns:
        best_map = best.drop_duplicates(subset=["ticker"], keep="first").set_index("ticker")["strategy"].to_dict()

    rows: List[Dict[str, object]] = []
    for _, row in full.iterrows():
        ticker = str(row.get("ticker", ""))
        strategy = str(row.get("strategy", ""))
        selected_strategy = str(best_map.get(ticker, ""))
        selected = strategy == selected_strategy and selected_strategy != ""

        tp_pct = float(pd.to_numeric(row.get("tp_pct", np.nan), errors="coerce"))
        tp_feas = float(pd.to_numeric(row.get("tp_feasibility", np.nan), errors="coerce"))
        certainty = float(pd.to_numeric(row.get("certainty_component", np.nan), errors="coerce"))

        if selected:
            reject_reason = "selected"
        elif np.isfinite(tp_pct) and tp_pct < float(TP_SL_MIN_ACCEPTABLE_TP):
            reject_reason = "tp_below_min_acceptable"
        elif np.isfinite(tp_feas) and tp_feas < 0.60:
            reject_reason = "low_tp_feasibility"
        elif np.isfinite(certainty) and certainty < 0.45:
            reject_reason = "low_certainty"
        else:
            reject_reason = "lower_selection_score"

        rows.append(
            {
                "ticker": ticker,
                "strategy": strategy,
                "selected": bool(selected),
                "selected_strategy": selected_strategy,
                "reject_reason": reject_reason,
                "selection_score": pd.to_numeric(row.get("selection_score", np.nan), errors="coerce"),
                "rule_adjusted_rbs": pd.to_numeric(row.get("rule_adjusted_rbs", np.nan), errors="coerce"),
                "risk_benefit_score": pd.to_numeric(row.get("risk_benefit_score", np.nan), errors="coerce"),
                "certainty_component": pd.to_numeric(row.get("certainty_component", np.nan), errors="coerce"),
                "tp_quality_component": pd.to_numeric(row.get("tp_quality_component", np.nan), errors="coerce"),
                "tp_floor_penalty_component": pd.to_numeric(row.get("tp_floor_penalty_component", np.nan), errors="coerce"),
                "tp_pct": pd.to_numeric(row.get("tp_pct", np.nan), errors="coerce"),
                "sl_pct": pd.to_numeric(row.get("sl_pct", np.nan), errors="coerce"),
                "tp_feasibility": pd.to_numeric(row.get("tp_feasibility", np.nan), errors="coerce"),
                "confidence": pd.to_numeric(row.get("confidence", np.nan), errors="coerce"),
                "historical_tp_prob": pd.to_numeric(row.get("historical_tp_prob", np.nan), errors="coerce"),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _export_fold_usd_artifacts(
    *,
    fold_backtest_dir: str,
    sim_out: Dict,
    selection_df: pd.DataFrame,
    metrics_payload: Dict,
    prices_dict: Optional[Dict[str, object]] = None,
    export_detailed_trades: bool = False,
) -> None:
    fold_dir = Path(fold_backtest_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)

    trades_df = sim_out.get("trades_df", pd.DataFrame())
    equity_df = sim_out.get("equity_curve_df", pd.DataFrame())
    trail_events = sim_out.get("trail_events", [])
    summary = sim_out.get("fold_summary", {})

    if not trades_df.empty:
        trades_df.to_csv(fold_dir / "trades.csv", index=False)
    else:
        pd.DataFrame(columns=[
            "fold_id", "datetime", "action", "ticker", "raw_price", "exec_price", "shares",
            "notional_usd", "fee_usd", "slippage_pct", "entry_date_requested", "entry_date_used",
            "exit_date_requested", "exit_date_used", "reason", "tp_pct", "sl_pct", "tp_sl_outcome", "days_to_outcome",
        ]).to_csv(fold_dir / "trades.csv", index=False)

    if not equity_df.empty:
        equity_df.to_csv(fold_dir / "equity_curve.csv", index=False)
    else:
        pd.DataFrame(columns=["date", "equity_usd", "cash_usd", "positions_value_usd"]).to_csv(
            fold_dir / "equity_curve.csv", index=False
        )

    if selection_df is not None and not selection_df.empty:
        selection_df.to_csv(fold_dir / "selection.csv", index=False)
    else:
        pd.DataFrame(columns=["ticker", "weight", "score", "rank"]).to_csv(fold_dir / "selection.csv", index=False)

    # Export trailing stop evolution events
    if trail_events:
        trail_df = pd.DataFrame(trail_events)
        trail_df.to_csv(fold_dir / "trailing_stop_evolution.csv", index=False)
        log.info("[Trail] Exported %d trailing stop events to %s", len(trail_df), (fold_dir / "trailing_stop_evolution.csv").name)
        _export_trailing_stop_stock_charts(
            fold_dir=fold_dir,
            trail_df=trail_df,
            trades_df=trades_df,
            prices_dict=prices_dict,
        )

    _safe_json_dump(fold_dir / "portfolio_summary.json", summary)
    _safe_json_dump(fold_dir / "metrics.json", metrics_payload)

    # Generate detailed trades report showing compra/venta side-by-side with USD values
    if export_detailed_trades and not trades_df.empty:
        _generate_detailed_trades_report(trades_df, fold_dir)


def _export_trailing_stop_stock_charts(
    *,
    fold_dir: Path,
    trail_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    prices_dict: Optional[Dict[str, object]] = None,
) -> None:
    """Export one advanced TP/SL/trailing chart per ticker into Backtest/Stocks."""
    stocks_dir = fold_dir / "Stocks"
    stocks_dir.mkdir(parents=True, exist_ok=True)

    if trail_df is None or trail_df.empty:
        return

    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency protection
        log.warning("[Trail] matplotlib unavailable, skipping stock charts: %s", exc)
        return

    tdf = trail_df.copy()
    tdf["ticker"] = tdf.get("ticker", "").astype(str).str.upper().str.strip()
    tdf = tdf[tdf["ticker"] != ""]
    tdf["event_date"] = pd.to_datetime(tdf.get("event_date"), errors="coerce")
    tdf["entry_date"] = pd.to_datetime(tdf.get("entry_date"), errors="coerce")
    for col in ["price", "trailing_stop", "entry_price", "tp_price", "sl_price_original"]:
        tdf[col] = pd.to_numeric(tdf.get(col), errors="coerce")
    tdf = tdf.dropna(subset=["event_date", "price"]).sort_values(["ticker", "event_date"])
    if tdf.empty:
        return

    def _extract_close_series(raw_obj: object) -> pd.Series:
        if raw_obj is None:
            return pd.Series(dtype=float)
        if isinstance(raw_obj, pd.Series):
            s = pd.to_numeric(raw_obj, errors="coerce").dropna()
            s.index = pd.to_datetime(s.index, errors="coerce")
            return s.sort_index()
        if isinstance(raw_obj, pd.DataFrame):
            col_candidates = ["close", "Close", "adj_close", "Adj Close", "adjclose"]
            for col in col_candidates:
                if col in raw_obj.columns:
                    s = pd.to_numeric(raw_obj[col], errors="coerce").dropna()
                    s.index = pd.to_datetime(raw_obj.loc[s.index].index, errors="coerce")
                    return s.sort_index()
        return pd.Series(dtype=float)

    local_trades = trades_df.copy() if trades_df is not None else pd.DataFrame()
    if not local_trades.empty:
        local_trades["ticker"] = local_trades.get("ticker", "").astype(str).str.upper().str.strip()
        local_trades["datetime"] = pd.to_datetime(local_trades.get("datetime"), errors="coerce")
        local_trades["exec_price"] = pd.to_numeric(local_trades.get("exec_price"), errors="coerce")

    event_styles = {
        "TRAILING_ACTIVATED": {"color": "#2ca02c", "marker": "^"},
        "REVIEW": {"color": "#ff7f0e", "marker": "o"},
        "EXIT_TP_TRAIL": {"color": "#1f77b4", "marker": "*"},
        "EXIT_TP": {"color": "#1f77b4", "marker": "*"},
        "EXIT_SL": {"color": "#d62728", "marker": "x"},
        "EXIT_TIME": {"color": "#7f7f7f", "marker": "s"},
    }

    for ticker, tk_df in tdf.groupby("ticker", sort=True):
        tk_df = tk_df.sort_values("event_date")
        if tk_df.empty:
            continue

        local_entry = pd.to_datetime(tk_df["entry_date"], errors="coerce").dropna()
        entry_ts = pd.Timestamp(local_entry.iloc[0]) if not local_entry.empty else pd.Timestamp(tk_df["event_date"].min())
        exit_ts = pd.Timestamp(tk_df["event_date"].max())

        if not local_trades.empty:
            tk_trades = local_trades[local_trades["ticker"] == ticker].dropna(subset=["datetime", "exec_price"])
            if not tk_trades.empty:
                buy_ts = pd.to_datetime(tk_trades.loc[tk_trades["action"].astype(str).str.upper() == "BUY", "datetime"], errors="coerce").dropna()
                sell_ts = pd.to_datetime(tk_trades.loc[tk_trades["action"].astype(str).str.upper() == "SELL", "datetime"], errors="coerce").dropna()
                if not buy_ts.empty:
                    entry_ts = pd.Timestamp(min(entry_ts, buy_ts.min()))
                if not sell_ts.empty:
                    exit_ts = pd.Timestamp(max(exit_ts, sell_ts.max()))
        else:
            tk_trades = pd.DataFrame()

        close_s = pd.Series(dtype=float)
        if prices_dict is not None:
            close_s = _extract_close_series(prices_dict.get(ticker))
        if not close_s.empty:
            close_s = close_s.loc[(close_s.index >= entry_ts) & (close_s.index <= exit_ts)].dropna()

        fig, ax = plt.subplots(figsize=(12, 6))
        if not close_s.empty:
            ax.plot(
                close_s.index,
                close_s.values,
                color="#111111",
                linewidth=1.8,
                label="Close price",
                zorder=2,
            )
            running_max = close_s.cummax()
            prev_max = running_max.shift(1).fillna(-np.inf)
            peak_mask = close_s > prev_max
            peak_s = close_s[peak_mask]
            if not peak_s.empty:
                ax.scatter(
                    peak_s.index,
                    peak_s.values,
                    color="#6a3d9a",
                    marker="D",
                    s=24,
                    alpha=0.7,
                    label="New peak",
                    zorder=3,
                )
        else:
            ax.plot(
                tk_df["event_date"],
                tk_df["price"],
                color="#111111",
                linewidth=1.8,
                marker=".",
                markersize=4,
                label="Event price",
                zorder=2,
            )

        first = tk_df.iloc[0]
        entry_price = pd.to_numeric(first.get("entry_price"), errors="coerce")
        tp_price = pd.to_numeric(first.get("tp_price"), errors="coerce")
        sl_price = pd.to_numeric(first.get("sl_price_original"), errors="coerce")

        ev_upper = tk_df["event_type"].astype(str).str.upper()
        act_mask = ev_upper.eq("TRAILING_ACTIVATED")
        exit_mask = ev_upper.str.startswith("EXIT_")
        activation_ts = pd.Timestamp(tk_df.loc[act_mask, "event_date"].iloc[0]) if act_mask.any() else None
        first_exit_ts = pd.Timestamp(tk_df.loc[exit_mask, "event_date"].iloc[0]) if exit_mask.any() else exit_ts
        tp_sl_active_end = activation_ts if activation_ts is not None else first_exit_ts

        if np.isfinite(entry_price):
            ax.hlines(
                y=float(entry_price),
                xmin=entry_ts,
                xmax=first_exit_ts,
                color="#333333",
                linestyle=":",
                linewidth=1.0,
                label="Entry price (active window)",
                zorder=1,
            )
        if np.isfinite(tp_price):
            ax.hlines(
                y=float(tp_price),
                xmin=entry_ts,
                xmax=tp_sl_active_end,
                color="#2ca02c",
                linestyle=":",
                linewidth=1.3,
                label="TP (active)",
                zorder=1,
            )
        if np.isfinite(sl_price):
            ax.hlines(
                y=float(sl_price),
                xmin=entry_ts,
                xmax=tp_sl_active_end,
                color="#d62728",
                linestyle=":",
                linewidth=1.3,
                label="Initial SL (active)",
                zorder=1,
            )

        stop_state_types = ["TRAILING_ACTIVATED", "REVIEW", "EXIT_TP_TRAIL", "EXIT_TIME", "EXIT_SL", "EXIT_TP"]
        stop_df = tk_df[ev_upper.isin(stop_state_types)].copy()
        stop_df = stop_df.dropna(subset=["trailing_stop"]).sort_values("event_date")
        if not stop_df.empty and (activation_ts is not None):
            stop_dates = stop_df["event_date"].tolist()
            stop_vals = stop_df["trailing_stop"].tolist()
            for i in range(len(stop_dates) - 1):
                x0 = pd.Timestamp(stop_dates[i])
                x1 = pd.Timestamp(stop_dates[i + 1])
                y = float(stop_vals[i])
                ax.hlines(
                    y=y,
                    xmin=x0,
                    xmax=x1,
                    color="#c2185b",
                    linewidth=2.0,
                    linestyle="-",
                    label="Trailing stop segment" if i == 0 else None,
                    zorder=2,
                )

        for event_type, ev_df in tk_df.groupby(tk_df["event_type"].astype(str).str.upper()):
            style = event_styles.get(event_type, {"color": "#9467bd", "marker": "o"})
            ax.scatter(
                ev_df["event_date"],
                ev_df["price"],
                color=style["color"],
                marker=style["marker"],
                s=65,
                label=event_type,
                zorder=4,
            )

        if not tk_trades.empty:
            buys = tk_trades[tk_trades["action"].astype(str).str.upper() == "BUY"]
            sells = tk_trades[tk_trades["action"].astype(str).str.upper() == "SELL"]
            if not buys.empty:
                ax.scatter(
                    buys["datetime"],
                    buys["exec_price"],
                    color="#1565c0",
                    marker="^",
                    s=88,
                    label="BUY",
                    zorder=5,
                )
            if not sells.empty:
                ax.scatter(
                    sells["datetime"],
                    sells["exec_price"],
                    color="#000000",
                    marker="v",
                    s=88,
                    label="SELL",
                    zorder=5,
                )

        ax.set_title(f"{ticker} | TP/SL Trailing Evolution")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uniq_h = []
        uniq_l = []
        for h, l in zip(handles, labels):
            if l not in seen:
                uniq_h.append(h)
                uniq_l.append(l)
                seen.add(l)
        ax.legend(uniq_h, uniq_l, fontsize=8, loc="best")
        fig.tight_layout()

        safe_ticker = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in ticker)
        plt.close(fig)

        # ── Panel doble: precio+niveles (top) + retorno acumulado (bottom) ──
        fig2, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(13, 8),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
            sharex=True,
        )

        # ─ Panel superior: replica curva de precio y niveles ─
        price_src = close_s if not close_s.empty else pd.Series(
            tk_df["price"].values, index=pd.to_datetime(tk_df["event_date"]), dtype=float
        )
        if not price_src.empty:
            ax_top.plot(price_src.index, price_src.values, color="#111111", linewidth=1.8,
                        label="Close price", zorder=2)
            running_max2 = price_src.cummax()
            prev_max2 = running_max2.shift(1).fillna(-np.inf)
            peak_s2 = price_src[price_src > prev_max2]
            if not peak_s2.empty:
                ax_top.scatter(peak_s2.index, peak_s2.values, color="#6a3d9a", marker="D",
                               s=24, alpha=0.7, label="New peak", zorder=3)

        if np.isfinite(entry_price):
            ax_top.hlines(float(entry_price), entry_ts, first_exit_ts,
                          color="#333333", linestyle=":", linewidth=1.0, label="Entry price")
        if np.isfinite(tp_price):
            ax_top.hlines(float(tp_price), entry_ts, tp_sl_active_end,
                          color="#2ca02c", linestyle=":", linewidth=1.3, label="TP (active)")
        if np.isfinite(sl_price):
            ax_top.hlines(float(sl_price), entry_ts, tp_sl_active_end,
                          color="#d62728", linestyle=":", linewidth=1.3, label="Initial SL (active)")
        if not stop_df.empty and activation_ts is not None:
            for i in range(len(stop_dates) - 1):
                ax_top.hlines(float(stop_vals[i]),
                              pd.Timestamp(stop_dates[i]), pd.Timestamp(stop_dates[i + 1]),
                              color="#c2185b", linewidth=2.0, linestyle="-",
                              label="Trailing stop" if i == 0 else None, zorder=2)
        for event_type, ev_df2 in tk_df.groupby(tk_df["event_type"].astype(str).str.upper()):
            style2 = event_styles.get(event_type, {"color": "#9467bd", "marker": "o"})
            ax_top.scatter(ev_df2["event_date"], ev_df2["price"],
                           color=style2["color"], marker=style2["marker"],
                           s=65, label=event_type, zorder=4)
        if not tk_trades.empty:
            buys2 = tk_trades[tk_trades["action"].astype(str).str.upper() == "BUY"]
            sells2 = tk_trades[tk_trades["action"].astype(str).str.upper() == "SELL"]
            if not buys2.empty:
                ax_top.scatter(buys2["datetime"], buys2["exec_price"],
                               color="#1565c0", marker="^", s=88, label="BUY", zorder=5)
            if not sells2.empty:
                ax_top.scatter(sells2["datetime"], sells2["exec_price"],
                               color="#000000", marker="v", s=88, label="SELL", zorder=5)

        ax_top.set_title(f"{ticker} | TP/SL Trailing Evolution + Retorno acumulado", fontsize=11)
        ax_top.set_ylabel("Price")
        ax_top.grid(True, alpha=0.25)
        handles2, labels2 = ax_top.get_legend_handles_labels()
        seen2: set = set()
        uniq_h2, uniq_l2 = [], []
        for h, l in zip(handles2, labels2):
            if l not in seen2:
                uniq_h2.append(h); uniq_l2.append(l); seen2.add(l)
        ax_top.legend(uniq_h2, uniq_l2, fontsize=7, loc="best", ncol=2)

        # ─ Panel inferior: retorno acumulado desde entrada ─
        if not price_src.empty and np.isfinite(entry_price) and float(entry_price) > 0:
            ret_pct = (price_src / float(entry_price) - 1.0) * 100.0
            ax_bot.plot(ret_pct.index, ret_pct.values, color="#1565c0", linewidth=1.6,
                        label="Return %", zorder=2)
            ax_bot.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--")
            if np.isfinite(tp_price) and float(entry_price) > 0:
                ax_bot.axhline(
                    (float(tp_price) / float(entry_price) - 1.0) * 100.0,
                    xmin=0, xmax=1, color="#2ca02c", linewidth=1.0, linestyle=":",
                    label="TP %",
                )
            if np.isfinite(sl_price) and float(entry_price) > 0:
                ax_bot.axhline(
                    (float(sl_price) / float(entry_price) - 1.0) * 100.0,
                    xmin=0, xmax=1, color="#d62728", linewidth=1.0, linestyle=":",
                    label="SL %",
                )
            ax_bot.fill_between(ret_pct.index, ret_pct.values, 0,
                                where=ret_pct.values >= 0, alpha=0.15, color="#2ca02c")
            ax_bot.fill_between(ret_pct.index, ret_pct.values, 0,
                                where=ret_pct.values < 0, alpha=0.15, color="#d62728")
            ax_bot.set_ylabel("Return (%)")
            ax_bot.grid(True, alpha=0.2)
            ax_bot.legend(fontsize=7, loc="best")

        ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig2.autofmt_xdate()
        fig2.tight_layout()
        out_path2 = stocks_dir / f"{safe_ticker}.png"
        fig2.savefig(out_path2, dpi=140)
        plt.close(fig2)


def _generate_detailed_trades_report(trades_df: pd.DataFrame, fold_dir: Path) -> None:
    """
    Generate a detailed trades report showing BUY and SELL side-by-side with USD values.
    Shows clearly how much USD was spent buying and how much was received selling.
    """
    if trades_df.empty:
        return

    # Group trades by ticker into BUY and SELL
    trades_df = trades_df.copy()
    trades_df['datetime'] = pd.to_datetime(trades_df['datetime'], errors='coerce')
    
    buy_trades = trades_df[trades_df['action'] == 'BUY'].sort_values('datetime')
    sell_trades = trades_df[trades_df['action'] == 'SELL'].sort_values('datetime')
    
    # Group by ticker to match buy/sell pairs
    report_rows = []
    
    for ticker in trades_df['ticker'].unique():
        ticker_buys = buy_trades[buy_trades['ticker'] == ticker]
        ticker_sells = sell_trades[sell_trades['ticker'] == ticker]
        
        # Match buy and sell trades in order
        # If multiple buys/sells for same ticker in a fold, process in chronological order
        for idx in range(max(len(ticker_buys), len(ticker_sells))):
            row_data = {
                'ticker': ticker,
                'buy_date': None,
                'buy_price': None,
                'buy_shares': None,
                'buy_notional_usd': None,
                'buy_fees_usd': None,
                'buy_total_cost_usd': None,
                'sell_date': None,
                'sell_price': None,
                'sell_shares': None,
                'sell_notional_usd': None,
                'sell_fees_usd': None,
                'sell_total_received_usd': None,
                'pnl_usd': None,
                'pnl_pct': None,
                'hold_days': None,
            }
            
            if idx < len(ticker_buys):
                buy = ticker_buys.iloc[idx]
                row_data['buy_date'] = pd.Timestamp(buy['datetime']).strftime('%Y-%m-%d')
                row_data['buy_price'] = float(buy['exec_price'])
                row_data['buy_shares'] = float(buy['shares'])
                row_data['buy_notional_usd'] = float(buy['notional_usd'])
                row_data['buy_fees_usd'] = float(buy['fee_usd'])
                row_data['buy_total_cost_usd'] = float(buy['notional_usd']) + float(buy['fee_usd'])
            
            if idx < len(ticker_sells):
                sell = ticker_sells.iloc[idx]
                row_data['sell_date'] = pd.Timestamp(sell['datetime']).strftime('%Y-%m-%d')
                row_data['sell_price'] = float(sell['exec_price'])
                row_data['sell_shares'] = float(sell['shares'])
                row_data['sell_notional_usd'] = float(sell['notional_usd'])
                row_data['sell_fees_usd'] = float(sell['fee_usd'])
                row_data['sell_total_received_usd'] = float(sell['notional_usd']) - float(sell['fee_usd'])
                
                # Calculate PnL if both buy and sell exist
                if row_data['buy_total_cost_usd'] is not None and row_data['sell_total_received_usd'] is not None:
                    pnl = row_data['sell_total_received_usd'] - row_data['buy_total_cost_usd']
                    pnl_pct = (pnl / row_data['buy_total_cost_usd']) * 100.0 if row_data['buy_total_cost_usd'] != 0 else 0.0
                    row_data['pnl_usd'] = pnl
                    row_data['pnl_pct'] = pnl_pct
                    
                    # Calculate holding period
                    buy_dt = pd.Timestamp(row_data['buy_date'])
                    sell_dt = pd.Timestamp(row_data['sell_date'])
                    row_data['hold_days'] = (sell_dt - buy_dt).days
            
            report_rows.append(row_data)
    
    if report_rows:
        report_df = pd.DataFrame(report_rows)
        report_df.to_csv(fold_dir / "trades_detailed.csv", index=False)


def _audit_fold_leakage(
    *,
    router: Optional[DataRouter],
    tickers: List[str],
    as_of: pd.Timestamp,
    fold_id: str,
) -> List[Dict]:
    if router is None:
        return []

    rows: List[Dict] = []
    for ticker in tickers:
        rec_df = router.load_recommendation_trends(ticker)
        mspr_df = router.load_insider_sentiment(ticker)
        ins_df = router.load_insider_transactions(ticker)
        eps_df = router.load_eps_surprises(ticker)
        prices_df = router.load_prices(ticker)
        cons_df = router.load_consolidated(ticker)

        checks = [
            ("sentiment", rec_df if rec_df is not None else pd.DataFrame(), None, f"rec_df:{ticker}"),
            ("sentiment", mspr_df if mspr_df is not None else pd.DataFrame(), None, f"mspr_df:{ticker}"),
            ("insider", ins_df if ins_df is not None else pd.DataFrame(), "date", f"insider_df:{ticker}"),
            ("sentiment", eps_df if eps_df is not None else pd.DataFrame(), None, f"eps_df:{ticker}"),
            ("technical", prices_df if prices_df is not None else pd.DataFrame(), None, f"prices_df:{ticker}"),
            ("fundamental", cons_df if cons_df is not None else pd.DataFrame(), None, f"consolidated_df:{ticker}"),
            ("valuation", cons_df if cons_df is not None else pd.DataFrame(), None, f"valuation_input:{ticker}"),
        ]

        for feature_group, frame, date_col, context in checks:
            detail = assert_no_future_data(frame, as_of=as_of, context=context, date_col=date_col)
            rows.append({
                "fold_id": fold_id,
                "ticker": ticker,
                "feature_group": feature_group,
                "n_rows_future_detected": int(detail["n_rows_future_detected"]),
                "max_future_date_detected": detail["max_future_date_detected"],
                "context": detail["context"],
            })
    return rows


def _summary_row_from_equity(name: str, equity_df: pd.DataFrame, total_fees: float, availability_flag: bool = True) -> Dict:
    if equity_df is None or equity_df.empty:
        return {
            "name": name,
            "final_value_usd": np.nan,
            "total_return_pct": np.nan,
            "total_fees_usd": float(total_fees),
            "max_drawdown": np.nan,
            "sharpe": np.nan,
            "availability_flag": bool(availability_flag),
        }
    start_val = float(equity_df["equity_usd"].iloc[0]) if len(equity_df) else np.nan
    end_val = float(equity_df["equity_usd"].iloc[-1]) if len(equity_df) else np.nan
    daily_ret = to_daily_returns_from_equity(equity_df)
    return {
        "name": name,
        "final_value_usd": float(end_val),
        "total_return_pct": float((end_val / start_val - 1.0) if start_val and np.isfinite(start_val) and start_val > 0 else np.nan),
        "total_fees_usd": float(total_fees),
        "max_drawdown": float(compute_max_drawdown_from_equity(equity_df)),
        "sharpe": float(sharpe_ratio(daily_ret)) if not daily_ret.empty else np.nan,
        "availability_flag": bool(availability_flag),
    }


def run_walkforward_pipeline(
    df: pd.DataFrame,
    sector_map: Dict[str, str],
    prices_dict: Dict[str, pd.DataFrame],
    benchmark: pd.Series,
    agents_results_dir: str,
    agent_models_results_dir: str,
    backtest_results_dir: str,
    plots_dir: str,
    walkforward_train_years: int,
    walkforward_num_tests: int,
    risk_free_rate: float,
    top_n_stocks: int = 10,
    random_seed: int = 42,
    spy_prices: Optional[pd.Series] = None,
    holding_period_months: int = 3,
    finnhub_data_dir: str = "data_finnhub",
    analysis_reference_date: Optional[pd.Timestamp] = None,
) -> Dict:
    run_root_candidate = Path(backtest_results_dir)
    run_root = run_root_candidate.parent if run_root_candidate.name.lower() == "backtest" else run_root_candidate
    run_root.mkdir(parents=True, exist_ok=True)
    strategy_dir = run_root / "strategy"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    baselines_dir = strategy_dir / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    backtester = WalkForwardBacktester(
        train_years=walkforward_train_years,
        test_quarters=1,
        risk_free=risk_free_rate,
        results_dir=strategy_dir.as_posix(),
        strategy_dir=strategy_dir.as_posix(),
        top_n_stocks=top_n_stocks,
        portfolio_optimizer=PORTFOLIO_OPTIMIZER,
    )
    global_visualizer = Visualizer(plots_dir=strategy_dir.as_posix())

    agent_diag_history: Dict[str, List] = {
        "fundamental": [], "valuation": [], "momentum": [], "bear": [],
        "sentiment": [], "sector_rotation": [], "meta_learner": [],
    }
    ablation_results: List[Dict] = []
    data_router = DataRouter(data_dir=finnhub_data_dir)

    leakage_rows: List[Dict] = []
    missing_prices_rows: List[Dict] = []
    tp_sl_universe_rows: List[pd.DataFrame] = []
    usd_fold_rows: List[Dict] = []
    usd_fold_contexts: List[Dict] = []
    strategy_equity_parts: List[pd.DataFrame] = []
    chained_cash_usd = float(INITIAL_CAPITAL_USD)
    total_strategy_fees_usd = 0.0

    all_master_tickers = sorted(set(pd.Index(df.index.get_level_values("ticker")).astype(str).tolist()))
    total_universe_tickers = int(len(all_master_tickers))
    max_train_years = int(walkforward_train_years)
    min_train_years = int(WALKFORWARD_TRAIN_MIN_YEARS)
    if max_train_years < min_train_years:
        max_train_years, min_train_years = min_train_years, max_train_years
    max_train_months = max(1, max_train_years * 12)
    min_train_months = max(1, min_train_years * 12)

    log.info(
        "[WalkForward] Dynamic train window enabled: max=%sM -> min=%sM (step=%sM) | minimum test=%s%%",
        max_train_months,
        min_train_months,
        max(int(holding_period_months), 1),
        MIN_TEST_TICKERS_PERCENT,
    )

    if analysis_reference_date is None:
        analysis_reference_date = pd.Timestamp.today().normalize()
    anchor_entry_date = pd.Timestamp(analysis_reference_date).normalize()
    n_tests = max(int(walkforward_num_tests), 1)
    holding_months = max(int(holding_period_months), 1)

    scheduled_entry_dates = sorted([
        (anchor_entry_date - pd.DateOffset(months=holding_months * k)).normalize()
        for k in range(n_tests)
    ])

    fold_plan: List[Dict[str, object]] = []
    for idx, entry_date in enumerate(scheduled_entry_dates, start=1):
        analysis_date = pd.Timestamp(entry_date).normalize()
        analysis_quarter = analysis_date.to_period("Q")
        train_end = analysis_date
        run_id = f"T{idx:02d}_{entry_date.strftime('%Y%m%d')}"
        fold_plan.append(
            {
                "fold_id": idx,
                "run_id": run_id,
                "entry_date": entry_date,
                "analysis_date": analysis_date,
                "analysis_quarter": analysis_quarter,
                "analysis_quarter_label": f"{analysis_quarter.year}Q{analysis_quarter.quarter}",
                "train_end": train_end,
            }
        )

    log.info(
        "[WalkForward] Scheduled %d tests from anchor %s (holding=%sM)",
        len(fold_plan),
        anchor_entry_date.date(),
        holding_months,
    )

    debug_profile = str(DEBUG_OUTPUT_PROFILE).strip().lower()
    full_debug = debug_profile == "full"
    export_tp_sl_universe = bool(EXPORT_TP_SL_UNIVERSE_MATRIX or full_debug)
    export_global_tp_sl_universe = bool(EXPORT_GLOBAL_TP_SL_UNIVERSE_MATRIX or full_debug)
    export_snapshot_agent_audits = bool(EXPORT_SNAPSHOT_AGENT_AUDITS or full_debug)
    export_all_folds_scores = bool(EXPORT_ALL_FOLDS_SCORES or full_debug)
    export_detailed_trades = bool(EXPORT_DETAILED_TRADES_REPORT or full_debug)

    membership_path = Path(SP500_HISTORIC_CSV_PATH)
    if not membership_path.is_absolute():
        membership_path = Path.cwd() / membership_path
    sp500_membership_df = _load_sp500_membership(membership_path) if USE_DYNAMIC_SP500_UNIVERSE else pd.DataFrame()
    if USE_DYNAMIC_SP500_UNIVERSE:
        if sp500_membership_df.empty:
            log.warning("[WalkForward] Dynamic universe active but no usable SP500 membership (%s)", membership_path)
        else:
            log.info("[WalkForward] Dynamic SP500 membership loaded: %s rows (%s)", len(sp500_membership_df), membership_path)

    prev_membership_tickers: Optional[set[str]] = None
    prev_membership_entry_date: Optional[pd.Timestamp] = None


    def _has_price_coverage(entry_date: pd.Timestamp, actual_end: pd.Timestamp) -> bool:
        if actual_end <= entry_date:
            return False
        for prices in prices_dict.values():
            if prices is None or prices.empty:
                continue
            cc = _get_close_column(prices)
            period = prices.loc[entry_date:actual_end, cc]
            if len(period) >= 2:
                return True
        return False

    try:
        from tqdm import tqdm
        fold_iter = tqdm(fold_plan, desc="Walk-forward tests", unit="test")
    except ImportError:
        fold_iter = fold_plan

    for plan in fold_iter:
        fold_id = int(plan["fold_id"])
        run_id = str(plan["run_id"])
        entry_date = pd.Timestamp(plan["entry_date"]).normalize()
        analysis_date = pd.Timestamp(plan["analysis_date"]).normalize()
        analysis_quarter = plan["analysis_quarter"]
        analysis_quarter_label = str(plan["analysis_quarter_label"])
        train_end = pd.Timestamp(plan["train_end"]).normalize()
        lag_days = 0
        fold_root_dir = run_root / str(run_id)
        fold_general_dir = fold_root_dir / "general"
        fold_agents_dir = fold_root_dir / "agents"
        fold_backtest_dir = fold_root_dir / "backtest"
        fold_plots_dir = fold_root_dir / "plots"
        fold_general_dir.mkdir(parents=True, exist_ok=True)
        fold_agents_dir.mkdir(parents=True, exist_ok=True)
        fold_backtest_dir.mkdir(parents=True, exist_ok=True)
        fold_plots_dir.mkdir(parents=True, exist_ok=True)
        fold_visualizer = Visualizer(plots_dir=fold_plots_dir.as_posix())

        selected_train_years: Optional[int] = None
        selected_train_start: Optional[pd.Timestamp] = None
        selected_df_train: Optional[pd.DataFrame] = None
        selected_df_test: Optional[pd.DataFrame] = None
        selected_eligibility_rows: Optional[list[dict]] = None
        selected_eligibility_train_years: Optional[int] = None
        last_eligibility_rows: list[dict] = []
        last_eligibility_train_years: Optional[int] = None

        active_tickers_on_entry = _active_sp500_tickers_on_date(sp500_membership_df, entry_date)
        fold_universe_tickers = len(active_tickers_on_entry) if (USE_DYNAMIC_SP500_UNIVERSE and active_tickers_on_entry) else total_universe_tickers
        min_test_tickers_required = int(np.ceil(fold_universe_tickers * MIN_TEST_TICKERS_PERCENT / 100.0))
        if USE_DYNAMIC_SP500_UNIVERSE and active_tickers_on_entry:
            log.info(
                f"[{run_id}] SP500 dynamic @entry {entry_date.date()}: {len(active_tickers_on_entry)} active members"
            )
            if prev_membership_tickers is not None and prev_membership_entry_date is not None:
                entered = sorted(active_tickers_on_entry - prev_membership_tickers)
                exited = sorted(prev_membership_tickers - active_tickers_on_entry)
                sample_n = 12
                entered_sample = ", ".join(entered[:sample_n]) if entered else "-"
                exited_sample = ", ".join(exited[:sample_n]) if exited else "-"
                log.info(
                    "[%s] Cambios vs fold previo (%s -> %s): +%s entran, -%s salen",
                    run_id,
                    prev_membership_entry_date.date(),
                    entry_date.date(),
                    len(entered),
                    len(exited),
                )
                log.info("[%s]   Entran (muestra): %s", run_id, entered_sample)
                log.info("[%s]   Salen  (muestra): %s", run_id, exited_sample)

            prev_membership_tickers = set(active_tickers_on_entry)
            prev_membership_entry_date = pd.Timestamp(entry_date)

        candidate_train_months_grid = list(range(max_train_months, min_train_months - 1, -holding_months))
        if min_train_months not in candidate_train_months_grid:
            candidate_train_months_grid.append(min_train_months)
        candidate_train_months_grid = sorted(set(candidate_train_months_grid), reverse=True)

        for candidate_train_months in candidate_train_months_grid:
            candidate_train_start = train_end - pd.DateOffset(months=int(candidate_train_months))

            cand_train, cand_test, _ = _prepare_fold_frames_by_filed_quarter(
                df=df,
                train_start_date=candidate_train_start,
                analysis_date=analysis_date,
            )

            if ENABLE_FALLBACK_EXTRAPOLATION:
                cand_test = _extrapolate_missing_snapshots(
                    df=df,
                    df_test=cand_test,
                    analysis_date=analysis_date,
                    lookback_quarters=FALLBACK_LOOK_BACK_QUARTERS,
                )

            test_tickers_pre_sp500 = (
                set(pd.Index(cand_test.index.get_level_values("ticker")).astype(str).tolist())
                if not cand_test.empty
                else set()
            )

            cand_test, test_before_sp500 = _filter_test_by_sp500_membership(
                df_test=cand_test,
                active_tickers=active_tickers_on_entry,
            )
            if USE_DYNAMIC_SP500_UNIVERSE:
                test_after_sp500 = int(cand_test.index.get_level_values("ticker").nunique()) if not cand_test.empty else 0
                test_tickers_post_sp500 = (
                    set(pd.Index(cand_test.index.get_level_values("ticker")).astype(str).tolist())
                    if not cand_test.empty
                    else set()
                )
                dropped_not_member = test_tickers_pre_sp500 - test_tickers_post_sp500
                active_without_snapshot = set(active_tickers_on_entry) - test_tickers_pre_sp500
                log.info(
                    f"[{run_id}] Cobertura test pre-filtro: {test_before_sp500} tickers con snapshot/relleno"
                )
                log.info(
                    f"[{run_id}] Filtro SP500 @entry: {test_before_sp500} -> {test_after_sp500} tickers "
                    f"(excluidos por no ser miembro activo: {len(dropped_not_member)})"
                )
                if dropped_not_member:
                    log.info(
                        f"[{run_id}]   No miembro SP500 @entry (muestra): {_sample_tickers(dropped_not_member)}"
                    )
                if active_without_snapshot:
                    log.info(
                        f"[{run_id}]   Miembros SP500 sin snapshot usable en quarter (muestra): "
                        f"{_sample_tickers(active_without_snapshot)}"
                    )
            else:
                test_tickers_post_sp500 = test_tickers_pre_sp500
                dropped_not_member = set()
                active_without_snapshot = set()

            test_tickers_before_history = (
                set(pd.Index(cand_test.index.get_level_values("ticker")).astype(str).tolist())
                if not cand_test.empty
                else set()
            )

            cand_train, cand_test, eligible_tickers = _filter_fold_tickers_by_history_span(
                df_train=cand_train,
                df_test=cand_test,
                required_months=int(candidate_train_months),
            )
            eligible_set = set(str(tk) for tk in eligible_tickers)
            dropped_insufficient_history = test_tickers_before_history - eligible_set

            if USE_DYNAMIC_SP500_UNIVERSE and active_tickers_on_entry:
                fold_base_universe = set(active_tickers_on_entry)
            else:
                fold_base_universe = set(all_master_tickers)

            # Traced universe for full fold audit: baseline + candidates observed in test.
            fold_trace_universe = set(fold_base_universe) | set(test_tickers_pre_sp500) | set(all_master_tickers)

            if USE_DYNAMIC_SP500_UNIVERSE:
                outside_base_universe = fold_trace_universe - set(fold_base_universe)
                log.info(
                    f"[{run_id}] Universo base fold (SP500 @entry): {len(fold_base_universe)} tickers"
                )
                if outside_base_universe:
                    log.info(
                        f"[{run_id}]   Fuera del universo base @entry (muestra): {_sample_tickers(outside_base_universe)}"
                    )
            else:
                log.info(
                    f"[{run_id}] Universo base fold (master dataset): {len(fold_base_universe)} tickers"
                )

            current_eligibility_rows: list[dict] = []
            for tk in sorted(fold_trace_universe):
                in_fold_base_universe = tk in fold_base_universe
                has_snapshot = tk in test_tickers_pre_sp500
                passed_sp500 = tk in test_tickers_post_sp500
                passed_history = tk in eligible_set
                is_eligible = tk in eligible_set

                if is_eligible:
                    reason = "eligible"
                elif tk in dropped_not_member:
                    reason = "not_member_sp500_on_entry"
                elif not in_fold_base_universe:
                    reason = "out_of_fold_base_universe"
                elif not has_snapshot:
                    reason = "no_snapshot_in_analysis_quarter"
                elif tk in dropped_insufficient_history:
                    reason = "insufficient_train_history"
                else:
                    reason = "other_filter"

                current_eligibility_rows.append(
                    {
                        "run_id": run_id,
                        "entry_date": str(entry_date.date()),
                        "candidate_train_months": int(candidate_train_months),
                        "ticker": tk,
                        "eligible": bool(is_eligible),
                        "reason": reason,
                        "in_fold_base_universe": bool(in_fold_base_universe),
                        "has_snapshot_in_analysis_quarter": bool(has_snapshot),
                        "passes_sp500_entry_filter": bool(passed_sp500),
                        "passes_history_filter": bool(passed_history),
                    }
                )

            last_eligibility_rows = current_eligibility_rows
            last_eligibility_train_years = int(round(candidate_train_months / 12.0))

            test_tickers_count = int(cand_test.index.get_level_values("ticker").nunique()) if not cand_test.empty else 0
            test_tickers_pct = (100.0 * test_tickers_count / fold_universe_tickers) if fold_universe_tickers > 0 else 0.0
            if dropped_insufficient_history:
                log.info(
                    f"[{run_id}]   Excluidos por historial insuficiente (<{candidate_train_months}M train): "
                    f"{len(dropped_insufficient_history)} | muestra: {_sample_tickers(dropped_insufficient_history)}"
                )
            log.info(
                f"[{run_id}] Intento train={candidate_train_months}M -> test elegible {test_tickers_count}/{fold_universe_tickers} "
                f"({test_tickers_pct:.1f}%) | min requerido={min_test_tickers_required}"
            )

            if test_tickers_count >= min_test_tickers_required:
                selected_train_years = int(round(candidate_train_months / 12.0))
                selected_train_start = pd.Timestamp(candidate_train_start)
                selected_df_train = cand_train
                selected_df_test = cand_test
                selected_eligibility_rows = current_eligibility_rows
                selected_eligibility_train_years = int(round(candidate_train_months / 12.0))
                log.info(
                    f"[{run_id}] Ventana seleccionada: {candidate_train_months}M "
                    f"({selected_train_start.date()} -> {train_end.date()}) | tickers elegibles={len(eligible_tickers)}"
                )
                break

        rows_to_export = selected_eligibility_rows if selected_eligibility_rows is not None else last_eligibility_rows
        years_to_export = (
            selected_eligibility_train_years if selected_eligibility_train_years is not None else last_eligibility_train_years
        )
        if rows_to_export:
            period_dir = fold_general_dir
            period_dir.mkdir(parents=True, exist_ok=True)
            eligibility_path = period_dir / "ticker_eligibility_audit.csv"
            eligibility_path.parent.mkdir(parents=True, exist_ok=True)
            eligibility_df = pd.DataFrame(rows_to_export)
            eligibility_df.to_csv(eligibility_path, index=False)

            summary_path = period_dir / "eligibility_reason_summary.csv"
            summary_df = (
                eligibility_df.groupby(
                    [
                        "candidate_train_months",
                        "reason",
                        "in_fold_base_universe",
                        "has_snapshot_in_analysis_quarter",
                        "passes_sp500_entry_filter",
                        "passes_history_filter",
                        "eligible",
                    ],
                    dropna=False,
                )
                .size()
                .reset_index(name="count")
                .sort_values(["candidate_train_months", "count", "reason"], ascending=[True, False, True])
            )
            summary_df.to_csv(summary_path, index=False)
            if years_to_export is not None:
                log.info(
                    "[%s] Eligibility audit (%sY) -> %s",
                    run_id,
                    years_to_export,
                    eligibility_path.name,
                )
            else:
                log.info("[%s] Eligibility audit -> %s", run_id, eligibility_path.name)
            log.info("[%s] Eligibility reason summary -> %s", run_id, summary_path.name)

        if selected_df_train is None or selected_df_test is None or selected_train_years is None or selected_train_start is None:
            log.warning(
                f"[{run_id}] Minimum test coverage ({MIN_TEST_TICKERS_PERCENT}%) not reached "
                f"even after reducing train to {min_train_years}Y — fold skipped."
            )
            continue

        train_start = selected_train_start
        _train_years = selected_train_years
        df_train = selected_df_train
        df_test = selected_df_test

        log.info(f"\n{'='*60}")
        log.info(f"  ANALYSIS {run_id}")
        log.info(f"  Train : {train_start.date()} → {train_end.date()}  (~{_train_years} years)")
        log.info(f"  Mode  : anchored-entry schedule")
        log.info(
            f"  Test  : snapshot simulado en {entry_date.date()} (as-of exacto, sin lag)"
        )
        log.info(f"{'='*60}")

        exit_date = entry_date + pd.DateOffset(months=max(int(holding_period_months), 1))
        bench_window = benchmark.loc[entry_date:exit_date].dropna()
        if len(bench_window) < 2:
            log.warning(
                f"[{run_id}] Benchmark sin precios suficientes desde {entry_date.date()} "
                "— fold omitido (sin entrenamiento)"
            )
            continue
        actual_end = bench_window.index.max()
        price_days = int(len(bench_window.loc[entry_date:actual_end]))
        log.info(
            f"[{run_id}] Ventana real de precios: {entry_date.date()} -> {actual_end.date()} "
            f"({price_days} dias)"
        )
        if not _has_price_coverage(entry_date, actual_end):
            log.warning(
                f"[{run_id}] Sin precios suficientes despues de {entry_date.date()} "
                "— fold omitido (sin entrenamiento)"
            )
            continue

        df_train = _recompute_forward_returns(
            df_part=df_train,
            prices_dict=prices_dict,
            lag_days=lag_days,
            holding_period_months=holding_period_months,
            filing_date_map=None,
            post_filing_delay_days=0,
        )
        df_test = _recompute_forward_returns(
            df_part=df_test,
            prices_dict=prices_dict,
            lag_days=lag_days,
            holding_period_months=holding_period_months,
            filing_date_map=None,
            post_filing_delay_days=0,
        )

        # Cross-sectional normalized and interaction features for ranking models.
        df_train = enrich_cross_sectional_features(df_train)
        df_test = enrich_cross_sectional_features(df_test)

        if len(df_train) < 100:
            log.warning(f"[{run_id}] Insufficient train ({len(df_train)} observations, minimum 100) — fold skipped.")
            continue

        df_train, df_test = df_train[~df_train.index.duplicated(keep="last")], df_test[~df_test.index.duplicated(keep="last")]

        df_train, df_test, y_train, y_test, alpha_train, alpha_test = _prepare_fold_labels(
            df_train=df_train,
            df_test=df_test,
            spy_prices=spy_prices,
            sector_map=sector_map,
            prices_dict=prices_dict,
            lag_days=lag_days,
            holding_period_months=holding_period_months,
        )
        if df_test.empty or y_test.empty:
            log.warning(f"[{run_id}] Empty test after preparing labels — fold skipped.")
            continue

        try:
            agents, df_test_scored, df_train_with_oof = train_fold(
                df_train_norm=df_train,
                df_test_norm=df_test,
                y_train=y_train,
                y_test=y_test,
                # Feed realized alpha targets to improve cross-sectional ranking quality.
                target_alpha_train=alpha_train,
                target_alpha_test=alpha_test,
                fold_id=run_id,
                agent_models_results_dir=fold_agents_dir.as_posix(),
                agents_results_dir=fold_agents_dir.as_posix(),
                random_seed=random_seed,
                sector_map=sector_map,
                spy_prices=spy_prices,
            )

            meta = agents["meta_learner"]
            eval_metrics = meta.evaluate(df_test_scored, y_test, fold=run_id, target_alpha=alpha_test)

            fold_visualizer.plot_score_distribution(df_test_scored, fold=run_id)

            keep_cols = [
                c for c in [
                    "final_score",
                    "label",
                    "regime_adjusted_score",
                    "tp_level",
                    "sl_level",
                    "tp_sl_outcome",
                    "rules_consensus_signal",
                    "rules_consensus_confidence",
                ]
                if c in df_test_scored.columns
            ]
            preds_df = df_test_scored[keep_cols].copy()
            preds_df["confidence"] = pd.to_numeric(
                preds_df.get("final_score", preds_df.get("regime_adjusted_score", 0.5)),
                errors="coerce",
            ).fillna(0.5).clip(0.0, 1.0)
            if "tp_level" not in preds_df.columns or preds_df["tp_level"].isna().all():
                inferred_tp, inferred_sl = infer_tp_sl_levels(
                    df_test_scored,
                    tp_default=float(TP_SL_BASE_TP),
                    sl_default=float(TP_SL_BASE_SL),
                    volatility_col="volatility_60d",
                )
                preds_df["tp_level"] = pd.to_numeric(inferred_tp.reindex(preds_df.index), errors="coerce").clip(
                    float(TP_SL_MIN_TP), float(TP_SL_MAX_TP)
                )
                preds_df["sl_level"] = pd.to_numeric(inferred_sl.reindex(preds_df.index), errors="coerce").clip(
                    float(TP_SL_MIN_SL), float(TP_SL_MAX_SL)
                )
            preds_df["ev"] = (
                preds_df["confidence"] * pd.to_numeric(preds_df["tp_level"], errors="coerce").fillna(float(TP_SL_BASE_TP))
                - (1.0 - preds_df["confidence"]) * pd.to_numeric(preds_df["sl_level"], errors="coerce").fillna(float(TP_SL_BASE_SL))
            )
            preds_df["ticker"] = preds_df.index.get_level_values("ticker")
            preds_df["date"] = preds_df.index.get_level_values("date")
            if "sector" in df_test_scored.columns:
                preds_df["sector"] = df_test_scored["sector"].values
            preds_df = preds_df.reset_index(drop=True)

            if bool(TP_EDGE_ENABLE):
                hist_edge = _compute_ticker_tp_edge(
                    history_df=df_train,
                    prior_strength=float(TP_EDGE_PRIOR_STRENGTH),
                    reliability_k=float(TP_EDGE_RELIABILITY_K),
                    none_score=float(TP_EDGE_NONE_SCORE),
                )
                if not hist_edge.empty:
                    preds_df = preds_df.merge(
                        hist_edge[["ticker", "historical_tp_prob", "historical_tp_edge", "historical_tp_obs"]],
                        on="ticker",
                        how="left",
                    )
                    preds_df["historical_tp_prob"] = pd.to_numeric(
                        preds_df.get("historical_tp_prob", 0.5), errors="coerce"
                    ).fillna(0.5).clip(0.0, 1.0)
                    preds_df["historical_tp_edge"] = pd.to_numeric(
                        preds_df.get("historical_tp_edge", 0.0), errors="coerce"
                    ).fillna(0.0).clip(-1.0, 1.0)
                    preds_df["historical_tp_obs"] = pd.to_numeric(
                        preds_df.get("historical_tp_obs", 0.0), errors="coerce"
                    ).fillna(0.0)
                    preds_df["confidence_raw"] = preds_df["confidence"].astype(float)
                    preds_df["confidence"] = (
                        preds_df["confidence_raw"]
                        + float(TP_EDGE_CONFIDENCE_BLEND) * preds_df["historical_tp_edge"]
                    ).clip(0.0, 1.0)

            # Build universe x strategy TP/SL matrix and choose best strategy per ticker.
            tp_sl_matrix = _build_tp_sl_strategy_universe_matrix(
                preds_df=preds_df,
                history_source_df=df_train,
                prices_dict=prices_dict,
                entry_date=entry_date,
                lag_days=lag_days,
                holding_period_months=holding_period_months,
            )
            if tp_sl_matrix.empty:
                log.warning("[%s] Empty TP/SL strategy matrix — fold skipped.", run_id)
                continue
            tp_sl_matrix.insert(0, "fold_id", str(run_id))
            if export_global_tp_sl_universe:
                tp_sl_universe_rows.append(tp_sl_matrix.copy())
            if export_tp_sl_universe:
                tp_sl_matrix.to_csv(fold_backtest_dir / "tp_sl_universe_matrix.csv", index=False)

            best_per_ticker = _pick_best_strategy_per_ticker(tp_sl_matrix)
            if best_per_ticker.empty:
                log.warning("[%s] No strategy-selected tickers available — fold skipped.", run_id)
                continue

            tp_sl_decision_audit = _build_tp_sl_decision_audit(
                tp_sl_matrix=tp_sl_matrix,
                best_per_ticker=best_per_ticker,
            )
            if not tp_sl_decision_audit.empty:
                tp_sl_decision_audit.to_csv(
                    fold_backtest_dir / "tp_sl_strategy_decision_audit.csv",
                    index=False,
                )

            # Broadcast per-ticker TP/SL confidence diagnostics to the full test
            # frame so fold reports include "up" and "TP-before-SL" confidence.
            aux_cols = [
                c for c in [
                    "ticker",
                    "confidence",
                    "confidence_raw",
                    "historical_tp_prob",
                    "historical_tp_edge",
                    "historical_tp_obs",
                    "strategy",
                    "tp_pct",
                    "sl_pct",
                    "ev",
                    "risk_benefit_score",
                    "rule_adjusted_rbs",
                    "selection_score",
                ]
                if c in best_per_ticker.columns
            ]
            if "ticker" in aux_cols:
                aux_map = best_per_ticker[aux_cols].drop_duplicates(subset=["ticker"], keep="first").set_index("ticker")
                idx_ticker = pd.Index(df_test_scored.index.get_level_values("ticker")).astype(str)
                for c in [x for x in aux_cols if x != "ticker"]:
                    df_test_scored[c] = idx_ticker.map(aux_map[c])
                if "confidence" in df_test_scored.columns:
                    df_test_scored["confidence_up"] = pd.to_numeric(df_test_scored["confidence"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
                else:
                    df_test_scored["confidence_up"] = pd.to_numeric(df_test_scored.get("final_score", 0.5), errors="coerce").fillna(0.5).clip(0.0, 1.0)
                if "historical_tp_prob" in df_test_scored.columns:
                    df_test_scored["confidence_tp_vs_sl"] = pd.to_numeric(df_test_scored["historical_tp_prob"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
                else:
                    df_test_scored["confidence_tp_vs_sl"] = df_test_scored["confidence_up"]

            fold_result = backtester.simulate_portfolio(
                predictions_df=best_per_ticker.rename(columns={"final_score": "score"}),
                prices_dict=prices_dict,
                benchmark=benchmark,
                fold_id=run_id,
                test_start=entry_date,
                test_end=exit_date,
                train_start=train_start,
                train_years_int=_train_years,
                analysis_quarter=analysis_quarter_label,
            )
            if not fold_result:
                log.warning(f"[{run_id}] Sin retornos suficientes — fold omitido tras simulacion")
                continue
            fold_result.update(eval_metrics)
            backtester.fold_results.append(fold_result)

            # PIT leakage audit on as-of sources used by the fold.
            fold_test_tickers = df_test_scored.index.get_level_values("ticker").unique().tolist()
            fold_leak_rows = _audit_fold_leakage(
                router=data_router,
                tickers=fold_test_tickers,
                as_of=entry_date,
                fold_id=run_id,
            )
            leakage_rows.extend(fold_leak_rows)
            n_leak_audit = int(sum(1 for r in fold_leak_rows if int(r.get("n_rows_future_detected", 0)) > 0))
            if n_leak_audit > 0:
                log.warning("[%s] Source audit detected %s incidents (diagnostic-only).", run_id, n_leak_audit)

            # Strict leakage flag must be based on the actual scored fold frame
            # used for decisions, not on raw source tables that naturally contain
            # observations after the as-of date.
            strict_detail = assert_no_future_data(
                df_test_scored.reset_index(),
                as_of=entry_date,
                context=f"fold_scored:{run_id}",
                date_col="date",
            )
            n_leak_fold = int(strict_detail.get("n_rows_future_detected", 0))
            leakage_rows.append({
                "fold_id": run_id,
                "ticker": "__FOLD_FRAME__",
                "feature_group": "strict_fold_frame",
                "n_rows_future_detected": n_leak_fold,
                "max_future_date_detected": strict_detail.get("max_future_date_detected"),
                "context": strict_detail.get("context"),
            })
            if n_leak_fold > 0:
                log.warning("[%s] Strict fold leakage detected: %s rows after as-of.", run_id, n_leak_fold)

            # USD monetary mode (without replacing current historical metrics).
            if USE_DOLLAR_BACKTEST:
                selected_tickers = list(fold_result.get("selected_tickers", []))
                ticker_weights = dict(fold_result.get("ticker_weights", {}))
                selected_plan = (
                    best_per_ticker.loc[best_per_ticker["ticker"].isin(selected_tickers), ["ticker", "tp_pct", "sl_pct", "max_holding_days"]]
                    .drop_duplicates(subset=["ticker"], keep="last")
                )
                base_ts = pd.Timestamp("2000-01-01")
                period_holding_days = int((base_ts + pd.DateOffset(months=max(int(holding_period_months), 1)) - base_ts).days)
                tp_sl_plan_by_ticker = {
                    str(r["ticker"]): {
                        "tp_pct": float(r["tp_pct"]),
                        "sl_pct": float(r["sl_pct"]),
                        "max_holding_days": int(period_holding_days),
                    }
                    for _, r in selected_plan.iterrows()
                }
                fold_trail_events: List[Dict] = []
                sim_out = simulate_fold_usd(
                    fold_id=str(run_id),
                    prices_dict=prices_dict,
                    selected_tickers=selected_tickers,
                    weights=ticker_weights,
                    entry_date_requested=entry_date,
                    exit_date_requested=exit_date,
                    starting_cash_usd=float(chained_cash_usd),
                    transaction_fee_usd=float(TRANSACTION_FEE_USD),
                    slippage_pct=float(SLIPPAGE_PCT),
                    allow_fractional_shares=bool(ALLOW_FRACTIONAL_SHARES),
                    tp_sl_plan_by_ticker=tp_sl_plan_by_ticker,
                    trail_events=fold_trail_events,
                )
                sim_out["fold_summary"]["leakage_tainted"] = bool(n_leak_fold > 0)
                chained_cash_usd = float(sim_out["fold_summary"].get("ending_capital_usd", chained_cash_usd))
                total_strategy_fees_usd += float(sim_out["fold_summary"].get("total_fees_usd", 0.0))

                fold_equity = sim_out.get("equity_curve_df", pd.DataFrame())
                if not fold_equity.empty:
                    strategy_equity_parts.append(fold_equity.copy())

                usd_fold_rows.append({
                    "fold_id": run_id,
                    "starting_capital_usd": float(sim_out["fold_summary"].get("starting_capital_usd", np.nan)),
                    "ending_capital_usd": float(sim_out["fold_summary"].get("ending_capital_usd", np.nan)),
                    "pnl_usd": float(sim_out["fold_summary"].get("pnl_usd", 0.0)),
                    "pnl_pct": float(sim_out["fold_summary"].get("pnl_pct", 0.0)),
                    "fees_usd": float(sim_out["fold_summary"].get("total_fees_usd", 0.0)),
                    "n_tickers": int(sim_out["fold_summary"].get("n_selected_tickers", 0)),
                    "leakage_tainted": bool(sim_out["fold_summary"].get("leakage_tainted", False)),
                })

                for miss_tk in sim_out.get("missing_tickers", []):
                    missing_prices_rows.append({
                        "fold_id": run_id,
                        "ticker": miss_tk,
                        "start_date": str(pd.Timestamp(entry_date).date()),
                        "end_date": str(pd.Timestamp(exit_date).date()),
                        "reason": sim_out.get("missing_reasons", {}).get(miss_tk, "missing_price_data"),
                    })

                preds_scored_for_selection = best_per_ticker.rename(columns={"final_score": "score"})
                selection_df = _build_selection_df(
                    preds_scored=preds_scored_for_selection,
                    selected_tickers=sim_out.get("selected_tickers_used", selected_tickers),
                    ticker_weights=sim_out.get("weights_used", ticker_weights),
                )
                _export_fold_usd_artifacts(
                    fold_backtest_dir=fold_backtest_dir.as_posix(),
                    sim_out=sim_out,
                    selection_df=selection_df,
                    metrics_payload={
                        "fold_id": run_id,
                        "classification_metrics": eval_metrics,
                        "return_metrics": {k: v for k, v in fold_result.items() if not str(k).startswith("_")},
                        "usd_summary": sim_out.get("fold_summary", {}),
                    },
                    prices_dict=prices_dict,
                    export_detailed_trades=export_detailed_trades,
                )

                usd_fold_contexts.append({
                    "fold_id": run_id,
                    "fold_idx": fold_id,
                    "entry_date": pd.Timestamp(entry_date),
                    "exit_date": pd.Timestamp(exit_date),
                    "preds_df": preds_df.rename(columns={"final_score": "score"}).copy(),
                    "df_test_scored": df_test_scored.reset_index().copy(),
                    "eligible_tickers": df_test_scored.index.get_level_values("ticker").unique().tolist(),
                })

            fold_visualizer.plot_fold_performance(fold_result, fold_id=run_id)

            audit_df = build_selection_audit_df(
                df_scored=df_test_scored.reset_index()[
                    ["ticker", "final_score", "label"]
                    + [c for c in ["fundamental_score", "valuation_score", "momentum_score",
                                   "bear_score", "sentiment_score", "sector_score"]
                       if c in df_test_scored.columns]
                ],
                selected_tickers=fold_result.get("selected_tickers", []),
                score_col="final_score",
                threshold=PORTFOLIO_MIN_SCORE,
            )
            period_dir = fold_agents_dir
            period_dir.mkdir(parents=True, exist_ok=True)
            export_selection_audit(audit_df, period_dir.as_posix(), fold_id=analysis_quarter_label, prefix="quarter")

            # CSV de scores con explicaciones legibles por agente
            ticker_returns = fold_result.get("ticker_returns", {})
            bench_ret = fold_result.get("benchmark_cumulative_return")

            fold_scores_df = build_fold_scores_df(
                df_test_scored=df_test_scored,
                y_test=y_test,
                fold_id=run_id,
                year_quarter=analysis_quarter_label,
                agents=agents,
                audit_df=audit_df,
                actual_returns=ticker_returns if ticker_returns else None,
                benchmark_return=bench_ret,
                ticker_weights=fold_result.get("ticker_weights"),
            )
            export_fold_scores(fold_scores_df, period_dir.as_posix(), fold_id=analysis_quarter_label)

            # Auditoria completa por quarter: snapshot por ticker + detalle agente-feature.
            if export_snapshot_agent_audits:
                export_quarter_snapshot_audit(
                    df_test_scored=df_test_scored,
                    year_quarter=analysis_quarter_label,
                    agents_results_dir=period_dir.as_posix(),
                )
                export_quarter_agent_feature_audit(
                    df_test_scored=df_test_scored,
                    agents=agents,
                    year_quarter=analysis_quarter_label,
                    agents_results_dir=period_dir.as_posix(),
                )

            for ag_name, ag in agents.items():
                agent_diag_history[ag_name].append(ag._diagnostics.copy())

            for ag_name in ["fundamental", "valuation", "momentum", "bear", "sector_rotation"]:
                ag = agents[ag_name]
                if hasattr(ag, "_feature_cols") and ag.is_trained:
                    try:
                        imp_path = (
                            fold_agents_dir / ag_name
                            / f"feature_importances_{run_id}.csv"
                        )
                        if imp_path.exists():
                            imp = pd.read_csv(imp_path, index_col=0)["importance"]
                            fold_visualizer.plot_feature_importances(imp, ag_name, fold=run_id)
                    except Exception as plot_exc:
                        log.debug(f"[Visualizer] plot_feature_importances {ag_name} {run_id}: {plot_exc}")

            explain_top_tickers(
                agents=agents,
                df_test=df_test_scored,
                scores=df_test_scored["final_score"],
                fold_id=analysis_quarter_label,
                agents_results_dir=period_dir.as_posix(),
                selected_tickers=fold_result.get("selected_tickers", []),
                audit_df=audit_df,
            )

            if RUN_ABLATION_STUDY:
                abl = run_ablation_study(
                    df_test_scored=df_test_scored,
                    y_test=y_test,
                    df_train_norm=df_train_with_oof,
                    y_train=y_train,
                    agents_results_dir=fold_agents_dir.as_posix(),
                    fold_id=run_id,
                    random_seed=random_seed,
                    fold_result=fold_result,
                )
                if abl:
                    ablation_results.append(abl)

        except Exception as e:
            log.error(f"Analysis {run_id} failed: {e}", exc_info=True)
            continue

    summary = backtester.summarize()
    backtester.save_folds_summary(plots_dir=strategy_dir.as_posix())

    # CSV consolidado de todos los folds: una fila por ticker-quarter con
    # scores, interpretaciones y explicaciones de cada agente.
    all_fold_score_files = sorted(run_root.glob("*/agents/scores.csv"))
    if export_all_folds_scores and all_fold_score_files:
        all_scores = []
        for p in all_fold_score_files:
            try:
                all_scores.append(pd.read_csv(p))
            except Exception as ex:
                log.warning("[FoldReport] Could not read %s (%s)", p, ex)
        if all_scores:
            pd.concat(all_scores, ignore_index=True).to_csv(strategy_dir / "all_folds_scores.csv", index=False)

    if export_global_tp_sl_universe and tp_sl_universe_rows:
        pd.concat(tp_sl_universe_rows, ignore_index=True).to_csv(
            strategy_dir / "all_folds_tp_sl_universe_matrix.csv",
            index=False,
        )

    diag_path = strategy_dir / "all_folds_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(agent_diag_history, f, indent=2, default=str)

    last_agent_diag = {k: v[-1] if v else {} for k, v in agent_diag_history.items()}

    # Build the global USD equity curve early so the plot uses the same return
    # series as the JSON summary (i.e. net-of-fees returns from simulate_fold_usd
    # rather than the gross daily returns from backtester.simulate_portfolio).
    _pre_equity_global = pd.DataFrame(columns=["date", "equity_usd", "cash_usd", "positions_value_usd"])
    if USE_DOLLAR_BACKTEST and strategy_equity_parts:
        _pre_equity_global = _concat_equity_parts(strategy_equity_parts)

    _plot_strategy_returns = backtester.all_strategy_returns
    _plot_benchmark_returns = backtester.all_benchmark_returns
    if USE_DOLLAR_BACKTEST and not _pre_equity_global.empty:
        # Derive plot returns from the USD equity curve so the wealth-curve
        # annotation (total return %) is consistent with final_summary.json.
        _usd_daily = to_daily_returns_from_equity(_pre_equity_global)
        if not _usd_daily.empty:
            _plot_strategy_returns = _usd_daily

    global_visualizer.plot_full_report(
        strategy_returns=_plot_strategy_returns,
        benchmark_returns=_plot_benchmark_returns,
        fold_results=backtester.fold_results,
        agent_diagnostics=last_agent_diag,
    )

    if RUN_ABLATION_STUDY and ablation_results:
        summarize_ablation(ablation_results, agents_results_dir=strategy_dir.as_posix())

    generate_text_report(
        summary=summary,
        fold_results=backtester.fold_results,
        agent_diag_history=agent_diag_history,
        backtest_results_dir=strategy_dir.as_posix(),
    )

    # Mandatory audit/leakage and missing-prices reports.
    leakage_df = pd.DataFrame(leakage_rows)
    if leakage_df.empty:
        leakage_df = pd.DataFrame(columns=[
            "fold_id", "ticker", "feature_group", "n_rows_future_detected",
            "max_future_date_detected", "context",
        ])
    leakage_df.to_csv(strategy_dir / "leakage_audit.csv", index=False)

    missing_prices_df = pd.DataFrame(missing_prices_rows)
    if missing_prices_df.empty:
        missing_prices_df = pd.DataFrame(columns=["fold_id", "ticker", "start_date", "end_date", "reason"])
    missing_prices_df.to_csv(strategy_dir / "missing_prices_report.csv", index=False)

    baselines_rows = []
    value_availability_rows = []
    value_selection_rows = []
    baseline_fold_compare_rows: List[Dict] = []

    strategy_equity_global = pd.DataFrame(columns=["date", "equity_usd", "cash_usd", "positions_value_usd"])
    if USE_DOLLAR_BACKTEST and strategy_equity_parts:
        # Reuse the equity curve already built for the plot above.
        strategy_equity_global = _pre_equity_global
        strategy_equity_global.to_csv(strategy_dir / "strategy_equity_curve.csv", index=False)

        strategy_summary = _summary_row_from_equity(
            "strategy_main",
            strategy_equity_global,
            total_strategy_fees_usd,
            availability_flag=not strategy_equity_global.empty,
        )
        baselines_rows.append(strategy_summary)

        # Benchmark principal SPY buy&hold en USD.
        benchmark_equity = pd.DataFrame(columns=["date", "equity_usd", "cash_usd", "positions_value_usd"])
        benchmark_summary = {
            "final_value_usd": np.nan,
            "return_pct": np.nan,
            "fees_usd": 0.0,
            "availability_flag": False,
        }
        if spy_prices is not None and not spy_prices.empty and usd_fold_contexts:
            first_entry = usd_fold_contexts[0]["entry_date"]
            last_exit_requested = usd_fold_contexts[-1]["exit_date"]
            last_spy_date = pd.Timestamp(pd.to_datetime(spy_prices.index).max())
            last_exit = min(pd.Timestamp(last_exit_requested), last_spy_date)
            if last_exit < pd.Timestamp(last_exit_requested):
                log.warning(
                    "[Benchmark USD] Salida truncada por falta de datos: requested=%s, used=%s",
                    pd.Timestamp(last_exit_requested).date(),
                    pd.Timestamp(last_exit).date(),
                )
            bench_sim = simulate_fold_usd(
                fold_id="benchmark",
                prices_dict={"SPY": spy_prices},
                selected_tickers=["SPY"],
                weights={"SPY": 1.0},
                entry_date_requested=first_entry,
                exit_date_requested=last_exit,
                starting_cash_usd=float(INITIAL_CAPITAL_USD),
                transaction_fee_usd=float(TRANSACTION_FEE_USD),
                slippage_pct=float(SLIPPAGE_PCT),
                allow_fractional_shares=True,
            )
            benchmark_equity = bench_sim.get("equity_curve_df", benchmark_equity)
            if not benchmark_equity.empty:
                benchmark_equity.to_csv(strategy_dir / "benchmark_equity_curve.csv", index=False)
            bench_available = not benchmark_equity.empty
            benchmark_summary = {
                "final_value_usd": float(bench_sim.get("fold_summary", {}).get("ending_capital_usd", np.nan)),
                "return_pct": float(bench_sim.get("fold_summary", {}).get("pnl_pct", np.nan)),
                "fees_usd": float(bench_sim.get("fold_summary", {}).get("total_fees_usd", 0.0)),
                "availability_flag": bool(bench_available),
            }
            _safe_json_dump(strategy_dir / "benchmark_summary.json", benchmark_summary)

            baselines_rows.append(_summary_row_from_equity(
                "benchmark",
                benchmark_equity,
                total_fees=float(benchmark_summary["fees_usd"]),
                availability_flag=bool(bench_available),
            ))

        # Baseline EW-UNIVERSE, MOMENTUM-12M, RANDOM-TOPN, VALUE-COMBINED.
        if RUN_BASELINES and usd_fold_contexts:
            ew_parts = []
            mom_parts = []
            random_curves: List[pd.DataFrame] = []
            value_parts = []
            random_fold_pnl_map: Dict[str, List[float]] = {}
            ew_selection_rows: List[Dict] = []
            mom_selection_rows: List[Dict] = []
            random_selection_rows: List[Dict] = []

            for sim_idx in range(int(N_RANDOM_BASELINE_SIMS)):
                sim_cash = float(INITIAL_CAPITAL_USD)
                sim_parts: List[pd.DataFrame] = []
                rng = random.Random(int(random_seed) + sim_idx)
                for ctx in usd_fold_contexts:
                    eligible = [t for t in ctx["eligible_tickers"] if t in prices_dict]
                    if not eligible:
                        continue
                    k = min(int(top_n_stocks), len(eligible))
                    picked = rng.sample(eligible, k=k)
                    for rank, tk in enumerate(picked, start=1):
                        random_selection_rows.append({
                            "simulation_id": sim_idx,
                            "fold_id": ctx["fold_id"],
                            "ticker": tk,
                            "weight": float(1.0 / len(picked)),
                            "rank": rank,
                        })
                    r_out = simulate_fold_usd(
                        fold_id=f"random_{sim_idx}_{ctx['fold_id']}",
                        prices_dict=prices_dict,
                        selected_tickers=picked,
                        weights={t: 1.0 / len(picked) for t in picked},
                        entry_date_requested=ctx["entry_date"],
                        exit_date_requested=ctx["exit_date"],
                        starting_cash_usd=sim_cash,
                        transaction_fee_usd=float(TRANSACTION_FEE_USD),
                        slippage_pct=float(SLIPPAGE_PCT),
                        allow_fractional_shares=True,
                    )
                    random_fold_pnl_map.setdefault(str(ctx["fold_id"]), []).append(
                        float(r_out.get("fold_summary", {}).get("pnl_pct", np.nan))
                    )
                    sim_cash = float(r_out.get("fold_summary", {}).get("ending_capital_usd", sim_cash))
                    if not r_out.get("equity_curve_df", pd.DataFrame()).empty:
                        sim_parts.append(r_out["equity_curve_df"])
                if sim_parts:
                    random_curves.append(_concat_equity_parts(sim_parts))

            ew_cash = float(INITIAL_CAPITAL_USD)
            mom_cash = float(INITIAL_CAPITAL_USD)
            value_cash = float(INITIAL_CAPITAL_USD)
            ew_fees = 0.0
            mom_fees = 0.0
            value_fees = 0.0

            for ctx in usd_fold_contexts:
                eligible = [t for t in ctx["eligible_tickers"] if t in prices_dict]
                if not eligible:
                    continue

                # EW universe.
                ew_weights = {t: 1.0 / len(eligible) for t in eligible}
                for rank, tk in enumerate(sorted(eligible), start=1):
                    ew_selection_rows.append({
                        "fold_id": ctx["fold_id"],
                        "ticker": tk,
                        "weight": float(ew_weights[tk]),
                        "rank": rank,
                    })
                ew_out = simulate_fold_usd(
                    fold_id=f"ew_{ctx['fold_id']}",
                    prices_dict=prices_dict,
                    selected_tickers=eligible,
                    weights=ew_weights,
                    entry_date_requested=ctx["entry_date"],
                    exit_date_requested=ctx["exit_date"],
                    starting_cash_usd=ew_cash,
                    transaction_fee_usd=float(TRANSACTION_FEE_USD),
                    slippage_pct=float(SLIPPAGE_PCT),
                    allow_fractional_shares=True,
                )
                baseline_fold_compare_rows.append({
                    "fold_id": str(ctx["fold_id"]),
                    "strategy_name": "ew_universe",
                    "pnl_pct": float(ew_out.get("fold_summary", {}).get("pnl_pct", np.nan)),
                    "ending_capital_usd": float(ew_out.get("fold_summary", {}).get("ending_capital_usd", np.nan)),
                })
                ew_cash = float(ew_out.get("fold_summary", {}).get("ending_capital_usd", ew_cash))
                ew_fees += float(ew_out.get("fold_summary", {}).get("total_fees_usd", 0.0))
                if not ew_out.get("equity_curve_df", pd.DataFrame()).empty:
                    ew_parts.append(ew_out["equity_curve_df"])

                # Momentum 12m baseline.
                mom_scores = []
                for tk in eligible:
                    s = prices_dict.get(tk)
                    if s is None or s.empty:
                        continue
                    close = s["Close"] if "Close" in s.columns else s.iloc[:, 0]
                    close = close.sort_index()
                    p_now = close.loc[close.index <= pd.Timestamp(ctx["entry_date"])]
                    p_old = close.loc[close.index <= (pd.Timestamp(ctx["entry_date"]) - pd.Timedelta(days=int(BASELINE_MOMENTUM_LOOKBACK_DAYS)))]
                    if p_now.empty or p_old.empty:
                        continue
                    v0 = float(p_old.iloc[-1])
                    v1 = float(p_now.iloc[-1])
                    if v0 > 0:
                        mom_scores.append((tk, v1 / v0 - 1.0))
                mom_scores = sorted(mom_scores, key=lambda x: x[1], reverse=True)
                mom_pick = [t for t, _ in mom_scores[: min(int(top_n_stocks), len(mom_scores))]]
                for rank, (tk, mom_ret) in enumerate(mom_scores, start=1):
                    mom_selection_rows.append({
                        "fold_id": ctx["fold_id"],
                        "ticker": tk,
                        "momentum_12m": float(mom_ret),
                        "rank": rank,
                        "selected_flag": int(tk in mom_pick),
                    })
                if mom_pick:
                    mom_out = simulate_fold_usd(
                        fold_id=f"mom_{ctx['fold_id']}",
                        prices_dict=prices_dict,
                        selected_tickers=mom_pick,
                        weights={t: 1.0 / len(mom_pick) for t in mom_pick},
                        entry_date_requested=ctx["entry_date"],
                        exit_date_requested=ctx["exit_date"],
                        starting_cash_usd=mom_cash,
                        transaction_fee_usd=float(TRANSACTION_FEE_USD),
                        slippage_pct=float(SLIPPAGE_PCT),
                        allow_fractional_shares=True,
                    )
                    baseline_fold_compare_rows.append({
                        "fold_id": str(ctx["fold_id"]),
                        "strategy_name": "momentum_12m",
                        "pnl_pct": float(mom_out.get("fold_summary", {}).get("pnl_pct", np.nan)),
                        "ending_capital_usd": float(mom_out.get("fold_summary", {}).get("ending_capital_usd", np.nan)),
                    })
                    mom_cash = float(mom_out.get("fold_summary", {}).get("ending_capital_usd", mom_cash))
                    mom_fees += float(mom_out.get("fold_summary", {}).get("total_fees_usd", 0.0))
                    if not mom_out.get("equity_curve_df", pd.DataFrame()).empty:
                        mom_parts.append(mom_out["equity_curve_df"])

                # Value combined fixed (P/E + EV/EBITDA).
                scored_df = ctx.get("df_test_scored", pd.DataFrame())
                pe_candidates = ["pe_ttm", "pe", "p_e", "peRatio", "pe_ratio"]
                ev_candidates = ["ev_ebitda", "evToEbitda", "ev_to_ebitda", "enterprise_value_to_ebitda"]
                pe_col = next((c for c in pe_candidates if c in scored_df.columns), None)
                ev_col = next((c for c in ev_candidates if c in scored_df.columns), None)
                if pe_col is None or ev_col is None:
                    value_availability_rows.append({"fold_id": ctx["fold_id"], "available": False, "reason": "missing_columns"})
                else:
                    vv = scored_df[["ticker", pe_col, ev_col]].copy()
                    vv = vv.rename(columns={pe_col: "pe_value", ev_col: "ev_ebitda_value"})
                    vv = vv.dropna(subset=["pe_value", "ev_ebitda_value"])
                    vv = vv[(vv["pe_value"] > 0) & (vv["ev_ebitda_value"] > 0)]
                    vv = vv[(vv["pe_value"] <= 300) & (vv["ev_ebitda_value"] <= 200)]
                    vv = vv.drop_duplicates(subset=["ticker"], keep="last")
                    if len(vv) < 5:
                        value_availability_rows.append({"fold_id": ctx["fold_id"], "available": False, "reason": "too_few_valid_tickers"})
                    else:
                        vv["rank_pe"] = vv["pe_value"].rank(method="average", ascending=True)
                        vv["rank_ev_ebitda"] = vv["ev_ebitda_value"].rank(method="average", ascending=True)
                        vv["rank_combined"] = (vv["rank_pe"] + vv["rank_ev_ebitda"]) / 2.0
                        vv = vv.sort_values("rank_combined", ascending=True)
                        vv["selected_flag"] = 0
                        value_pick = vv.head(min(int(top_n_stocks), len(vv)))["ticker"].tolist()
                        vv.loc[vv["ticker"].isin(value_pick), "selected_flag"] = 1
                        vv["fold_id"] = ctx["fold_id"]
                        value_selection_rows.append(vv[[
                            "fold_id", "ticker", "pe_value", "ev_ebitda_value", "rank_pe", "rank_ev_ebitda", "rank_combined", "selected_flag"
                        ]])
                        value_availability_rows.append({"fold_id": ctx["fold_id"], "available": True, "reason": "ok"})

                        value_out = simulate_fold_usd(
                            fold_id=f"value_{ctx['fold_id']}",
                            prices_dict=prices_dict,
                            selected_tickers=value_pick,
                            weights={t: 1.0 / len(value_pick) for t in value_pick},
                            entry_date_requested=ctx["entry_date"],
                            exit_date_requested=ctx["exit_date"],
                            starting_cash_usd=value_cash,
                            transaction_fee_usd=float(TRANSACTION_FEE_USD),
                            slippage_pct=float(SLIPPAGE_PCT),
                            allow_fractional_shares=True,
                        )
                        baseline_fold_compare_rows.append({
                            "fold_id": str(ctx["fold_id"]),
                            "strategy_name": "value_combined",
                            "pnl_pct": float(value_out.get("fold_summary", {}).get("pnl_pct", np.nan)),
                            "ending_capital_usd": float(value_out.get("fold_summary", {}).get("ending_capital_usd", np.nan)),
                        })
                        value_cash = float(value_out.get("fold_summary", {}).get("ending_capital_usd", value_cash))
                        value_fees += float(value_out.get("fold_summary", {}).get("total_fees_usd", 0.0))
                        if not value_out.get("equity_curve_df", pd.DataFrame()).empty:
                            value_parts.append(value_out["equity_curve_df"])

            ew_curve = _concat_equity_parts(ew_parts)
            mom_curve = _concat_equity_parts(mom_parts)
            value_curve = _concat_equity_parts(value_parts)
            ew_curve.to_csv(baselines_dir / "ew_universe_equity_curve.csv", index=False)
            mom_curve.to_csv(baselines_dir / "momentum_12m_equity_curve.csv", index=False)
            value_curve.to_csv(baselines_dir / "value_combined_equity_curve.csv", index=False)

            _safe_json_dump(baselines_dir / "ew_universe_summary.json", _summary_row_from_equity("ew_universe", ew_curve, ew_fees, True))
            _safe_json_dump(baselines_dir / "momentum_12m_summary.json", _summary_row_from_equity("momentum_12m", mom_curve, mom_fees, True))
            _safe_json_dump(baselines_dir / "value_combined_summary.json", _summary_row_from_equity("value_combined", value_curve, value_fees, len(value_parts) > 0))

            pd.DataFrame(value_availability_rows).to_csv(baselines_dir / "value_availability_report.csv", index=False)
            if value_selection_rows:
                pd.concat(value_selection_rows, axis=0).to_csv(baselines_dir / "value_combined_selection_by_fold.csv", index=False)
            else:
                pd.DataFrame(columns=[
                    "fold_id", "ticker", "pe_value", "ev_ebitda_value", "rank_pe", "rank_ev_ebitda", "rank_combined", "selected_flag"
                ]).to_csv(baselines_dir / "value_combined_selection_by_fold.csv", index=False)

            pd.DataFrame(ew_selection_rows).to_csv(baselines_dir / "ew_universe_selection_by_fold.csv", index=False)
            pd.DataFrame(mom_selection_rows).to_csv(baselines_dir / "momentum_12m_selection_by_fold.csv", index=False)
            pd.DataFrame(random_selection_rows).to_csv(baselines_dir / "random_topn_selection_by_sim.csv", index=False)

            if random_curves:
                merged = pd.concat([
                    c.set_index("date")["equity_usd"].rename(f"sim_{i}")
                    for i, c in enumerate(random_curves)
                ], axis=1).sort_index().ffill()
                random_mean = merged.mean(axis=1)
                random_p05 = merged.quantile(0.05, axis=1)
                random_p95 = merged.quantile(0.95, axis=1)
                random_df = pd.DataFrame({
                    "date": random_mean.index,
                    "equity_usd_mean": random_mean.values,
                    "equity_usd_p05": random_p05.values,
                    "equity_usd_p95": random_p95.values,
                })
                random_df.to_csv(baselines_dir / "random_topn_equity_curve_mean.csv", index=False)
                random_curve_for_metrics = pd.DataFrame({
                    "date": random_mean.index,
                    "equity_usd": random_mean.values,
                    "cash_usd": np.nan,
                    "positions_value_usd": np.nan,
                })
                _safe_json_dump(
                    baselines_dir / "random_topn_summary.json",
                    _summary_row_from_equity("random_topn_mean", random_curve_for_metrics, 0.0, True),
                )
            random_fold_rows: List[Dict] = []
            for fold_key in sorted(random_fold_pnl_map.keys()):
                vals = [v for v in random_fold_pnl_map.get(fold_key, []) if np.isfinite(v)]
                if not vals:
                    continue
                random_fold_rows.append({
                    "fold_id": str(fold_key),
                    "strategy_name": "random_topn_mean",
                    "pnl_pct": float(np.mean(vals)),
                    "pnl_pct_p05": float(np.quantile(vals, 0.05)),
                    "pnl_pct_p95": float(np.quantile(vals, 0.95)),
                    "n_sims": int(len(vals)),
                })
            if random_fold_rows:
                pd.DataFrame(random_fold_rows).to_csv(
                    baselines_dir / "random_topn_fold_summary.csv", index=False
                )
                baseline_fold_compare_rows.extend(random_fold_rows)

            baselines_rows.extend([
                _summary_row_from_equity("ew_universe", ew_curve, ew_fees, True),
                _summary_row_from_equity("momentum_12m", mom_curve, mom_fees, True),
                _summary_row_from_equity("value_combined", value_curve, value_fees, len(value_parts) > 0),
            ])

            if (baselines_dir / "random_topn_equity_curve_mean.csv").exists():
                rnd = pd.read_csv(baselines_dir / "random_topn_equity_curve_mean.csv")
                rnd_curve = pd.DataFrame({"date": rnd["date"], "equity_usd": rnd["equity_usd_mean"]})
                baselines_rows.append(_summary_row_from_equity("random_topn_mean", rnd_curve, 0.0, True))

        # Resumen consolidado de baselines/benchmark/estrategia.
        bs_df = pd.DataFrame(baselines_rows)
        if not bs_df.empty:
            bs_df = bs_df.rename(columns={"name": "strategy_name"})
            pct_main = float((pd.DataFrame(usd_fold_rows).get("pnl_pct", pd.Series(dtype=float)) > 0).mean()) if usd_fold_rows else np.nan
            bs_df["pct_folds_positive"] = np.nan
            bs_df.loc[bs_df["strategy_name"] == "strategy_main", "pct_folds_positive"] = pct_main
            bs_df.to_csv(strategy_dir / "baselines_summary.csv", index=False)

        final_value_payload = {
            "initial_capital_usd": float(INITIAL_CAPITAL_USD),
            "final_strategy_value_usd": float(strategy_summary.get("final_value_usd", np.nan)),
            "strategy_total_return_pct": float(strategy_summary.get("total_return_pct", np.nan)),
            "strategy_total_fees_usd": float(strategy_summary.get("total_fees_usd", 0.0)),
        }
        bench_row = next((r for r in baselines_rows if r.get("name") == "benchmark"), None)
        if bench_row is not None:
            b_final = bench_row.get("final_value_usd", np.nan)
            b_ret = bench_row.get("total_return_pct", np.nan)
            b_fees = bench_row.get("total_fees_usd", 0.0)
            final_value_payload.update({
                "benchmark_final_value_usd": None if not np.isfinite(float(b_final)) else float(b_final),
                "benchmark_total_return_pct": None if not np.isfinite(float(b_ret)) else float(b_ret),
                "benchmark_total_fees_usd": float(b_fees),
            })
        _safe_json_dump(strategy_dir / "final_portfolio_value.json", final_value_payload)

        # Final summary JSON/CSV.
        fold_usd_df = pd.DataFrame(usd_fold_rows)
        if not fold_usd_df.empty:
            for _, r in fold_usd_df.iterrows():
                baseline_fold_compare_rows.append({
                    "fold_id": str(r.get("fold_id")),
                    "strategy_name": "strategy_main",
                    "pnl_pct": float(r.get("pnl_pct", np.nan)),
                    "ending_capital_usd": float(r.get("ending_capital_usd", np.nan)),
                })
        if bench_row is not None and spy_prices is not None and not spy_prices.empty and usd_fold_contexts:
            for ctx in usd_fold_contexts:
                b_fold = simulate_fold_usd(
                    fold_id=f"benchmark_{ctx['fold_id']}",
                    prices_dict={"SPY": spy_prices},
                    selected_tickers=["SPY"],
                    weights={"SPY": 1.0},
                    entry_date_requested=ctx["entry_date"],
                    exit_date_requested=ctx["exit_date"],
                    starting_cash_usd=float(INITIAL_CAPITAL_USD),
                    transaction_fee_usd=float(TRANSACTION_FEE_USD),
                    slippage_pct=float(SLIPPAGE_PCT),
                    allow_fractional_shares=True,
                )
                baseline_fold_compare_rows.append({
                    "fold_id": str(ctx["fold_id"]),
                    "strategy_name": "benchmark",
                    "pnl_pct": float(b_fold.get("fold_summary", {}).get("pnl_pct", np.nan)),
                    "ending_capital_usd": float(b_fold.get("fold_summary", {}).get("ending_capital_usd", np.nan)),
                })

        fold_compare_df = pd.DataFrame(baseline_fold_compare_rows)
        if not fold_compare_df.empty:
            fold_compare_df.to_csv(strategy_dir / "fold_comparison_summary.csv", index=False)

        summary_payload = {
            "timestamp": datetime.now().isoformat(),
            "legacy_summary": summary,
            "usd_strategy": strategy_summary,
            "total_leakage_incidents": int((leakage_df["n_rows_future_detected"] > 0).sum()) if not leakage_df.empty else 0,
            "folds_usd": fold_usd_df.to_dict(orient="records"),
            "baselines": pd.DataFrame(baselines_rows).to_dict(orient="records"),
        }
        _safe_json_dump(strategy_dir / "final_summary.json", summary_payload)

        final_summary_csv_rows = []
        final_summary_csv_rows.append({"name": "strategy_main", **strategy_summary})
        for row in baselines_rows:
            if row.get("name") != "strategy_main":
                final_summary_csv_rows.append(row)
        pd.DataFrame(final_summary_csv_rows).to_csv(strategy_dir / "final_summary.csv", index=False)

        # Plots requeridos para memoria TFM.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_path = Path(plots_dir)
        plots_path.mkdir(parents=True, exist_ok=True)

        if not strategy_equity_global.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(pd.to_datetime(strategy_equity_global["date"]), strategy_equity_global["equity_usd"], label="strategy_main", lw=2)
            if (strategy_dir / "benchmark_equity_curve.csv").exists():
                bdf = pd.read_csv(strategy_dir / "benchmark_equity_curve.csv")
                plt.plot(pd.to_datetime(bdf["date"]), bdf["equity_usd"], label="benchmark", lw=1.8)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.title("Equity Curve USD")
            plt.tight_layout()
            plt.savefig(plots_path / "equity_curve_usd.png", dpi=140)
            plt.close()

            plt.figure(figsize=(12, 6))
            plt.plot(pd.to_datetime(strategy_equity_global["date"]), strategy_equity_global["equity_usd"], label="strategy_main", lw=2)
            if (strategy_dir / "benchmark_equity_curve.csv").exists():
                bdf = pd.read_csv(strategy_dir / "benchmark_equity_curve.csv")
                plt.plot(pd.to_datetime(bdf["date"]), bdf["equity_usd"], label="benchmark", lw=1.6)
            for name, fname in [
                ("ew_universe", "ew_universe_equity_curve.csv"),
                ("momentum_12m", "momentum_12m_equity_curve.csv"),
                ("value_combined", "value_combined_equity_curve.csv"),
            ]:
                p = baselines_dir / fname
                if p.exists():
                    cdf = pd.read_csv(p)
                    if not cdf.empty and "equity_usd" in cdf.columns:
                        plt.plot(pd.to_datetime(cdf["date"]), cdf["equity_usd"], label=name, alpha=0.9)
            p_rand = baselines_dir / "random_topn_equity_curve_mean.csv"
            if p_rand.exists():
                rdf = pd.read_csv(p_rand)
                plt.plot(pd.to_datetime(rdf["date"]), rdf["equity_usd_mean"], label="random_topn_mean", alpha=0.9)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.title("Equity Curve USD with Baselines")
            plt.tight_layout()
            plt.savefig(plots_path / "equity_curve_usd_with_baselines.png", dpi=140)
            plt.close()

            eq = pd.Series(strategy_equity_global["equity_usd"].values, index=pd.to_datetime(strategy_equity_global["date"]))
            dd = (eq / eq.cummax()) - 1.0
            plt.figure(figsize=(12, 4.8))
            plt.fill_between(dd.index, dd.values * 100.0, 0, alpha=0.4)
            plt.title("Drawdown USD (%)")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_path / "drawdown_usd.png", dpi=140)
            plt.close()

        fold_usd_df = pd.DataFrame(usd_fold_rows)
        if not fold_usd_df.empty:
            plt.figure(figsize=(10, 4.8))
            plt.bar(fold_usd_df["fold_id"].astype(str), fold_usd_df["ending_capital_usd"].astype(float))
            plt.title("Ending Capital by Fold")
            plt.xticks(rotation=45, ha="right")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_path / "capital_by_fold.png", dpi=140)
            plt.close()

            plt.figure(figsize=(10, 4.8))
            plt.bar(fold_usd_df["fold_id"].astype(str), fold_usd_df["pnl_pct"].astype(float) * 100.0)
            plt.title("PnL % by Fold")
            plt.xticks(rotation=45, ha="right")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_path / "pnl_pct_by_fold.png", dpi=140)
            plt.close()

        # Plot comparativo por fold: estrategia vs benchmark vs baselines.
        fold_compare_plot_df = pd.DataFrame(baseline_fold_compare_rows)
        if not fold_compare_plot_df.empty and {"fold_id", "strategy_name", "pnl_pct"}.issubset(fold_compare_plot_df.columns):
            cmp = fold_compare_plot_df[["fold_id", "strategy_name", "pnl_pct"]].copy()
            cmp = cmp.dropna(subset=["fold_id", "strategy_name", "pnl_pct"])
            if not cmp.empty:
                pivot = cmp.pivot_table(index="fold_id", columns="strategy_name", values="pnl_pct", aggfunc="mean")
                preferred_order = [
                    "strategy_main", "benchmark", "ew_universe", "momentum_12m", "value_combined", "random_topn_mean"
                ]
                ordered_cols = [c for c in preferred_order if c in pivot.columns] + [c for c in pivot.columns if c not in preferred_order]
                pivot = pivot[ordered_cols]

                ax = pivot.mul(100.0).plot(kind="bar", figsize=(12, 5.2), width=0.82)
                ax.set_title("PnL % por Fold - Estrategia vs Benchmark y Baselines")
                ax.set_xlabel("Fold")
                ax.set_ylabel("PnL %")
                ax.grid(axis="y", alpha=0.3)
                ax.legend(title="Serie", fontsize=8)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(plots_path / "fold_pnl_comparison_with_baselines.png", dpi=140)
                plt.close()

        # Plot comparativo anual (anos calendario): estrategia vs benchmark vs baselines.
        def _yearly_return_from_curve(df_curve: pd.DataFrame, value_col: str) -> pd.Series:
            if df_curve is None or df_curve.empty or value_col not in df_curve.columns or "date" not in df_curve.columns:
                return pd.Series(dtype=float)
            s = df_curve[["date", value_col]].copy()
            s["date"] = pd.to_datetime(s["date"], errors="coerce")
            s[value_col] = pd.to_numeric(s[value_col], errors="coerce")
            s = s.dropna(subset=["date", value_col]).sort_values("date")
            if s.empty:
                return pd.Series(dtype=float)
            by_year = s.groupby(s["date"].dt.year)[value_col]
            out = (by_year.last() / by_year.first()) - 1.0
            out.index = out.index.astype(str)
            return out

        annual_series: Dict[str, pd.Series] = {}
        if not strategy_equity_global.empty:
            annual_series["strategy_main"] = _yearly_return_from_curve(strategy_equity_global, "equity_usd")

        p_bench = strategy_dir / "benchmark_equity_curve.csv"
        if p_bench.exists():
            bdf = pd.read_csv(p_bench)
            annual_series["benchmark"] = _yearly_return_from_curve(bdf, "equity_usd")

        for name, fname, col in [
            ("ew_universe", "ew_universe_equity_curve.csv", "equity_usd"),
            ("momentum_12m", "momentum_12m_equity_curve.csv", "equity_usd"),
            ("value_combined", "value_combined_equity_curve.csv", "equity_usd"),
            ("random_topn_mean", "random_topn_equity_curve_mean.csv", "equity_usd_mean"),
        ]:
            p = baselines_dir / fname
            if p.exists():
                cdf = pd.read_csv(p)
                annual_series[name] = _yearly_return_from_curve(cdf, col)

        annual_series = {k: v for k, v in annual_series.items() if v is not None and not v.empty}
        if annual_series:
            years = sorted(set().union(*[set(s.index.tolist()) for s in annual_series.values()]))
            ann_df = pd.DataFrame(index=years)
            preferred_cols = [
                "strategy_main", "benchmark", "ew_universe", "momentum_12m", "value_combined", "random_topn_mean"
            ]
            for name, s in annual_series.items():
                ann_df[name] = s
            ordered_cols = [c for c in preferred_cols if c in ann_df.columns] + [c for c in ann_df.columns if c not in preferred_cols]
            ann_df = ann_df[ordered_cols]
            ann_df.to_csv(strategy_dir / "annual_return_comparison.csv", index_label="year")

            ax = ann_df.mul(100.0).plot(kind="bar", figsize=(12.5, 5.4), width=0.84)
            ax.set_title("Annual Return Comparison - Strategy vs Benchmark and Baselines")
            ax.set_xlabel("Calendar year")
            ax.set_ylabel("Return %")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(title="Series", fontsize=8)
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(plots_path / "annual_return_comparison_with_baselines.png", dpi=140)
            plt.close()

    # ── Portfolio Size vs Benchmark Alpha Analysis ──
    if usd_fold_contexts:
        from module.steps.step_04_evaluation.analysis import run_portfolio_size_analysis
        run_portfolio_size_analysis(
            fold_contexts=usd_fold_contexts,
            prices_dict=prices_dict,
            benchmark_prices=spy_prices,
            output_dir=strategy_dir,
        )

    return summary
