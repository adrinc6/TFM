"""Walk-forward evaluation orchestration."""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from environment import (
    BASE_AGENTS_LABEL_MODE,
    BASE_LABEL_SECTOR_MIN_PEERS,
    PORTFOLIO_MIN_SCORE,
    RUN_ABLATION_STUDY,
    SECTOR_ZSCORE_MIN_PEERS,
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
)
from module.common.asof import assert_no_future_data, filter_asof
from module.common.data_router import DataRouter
from module.steps.step_02_dataset.builders.sector import SectorNormalizer
from module.steps.step_02_dataset.normalization import apply_sector_normalization
from module.steps.step_03_training.training import train_fold
from module.steps.step_04_evaluation.ablation import run_ablation_study, summarize_ablation
from module.steps.step_04_evaluation.backtester import WalkForwardBacktester
from module.steps.step_04_evaluation.fold_report import (
    build_fold_scores_df,
    export_fold_scores,
    export_all_folds_scores,
    export_quarter_snapshot_audit,
    export_quarter_agent_feature_audit,
)

from module.steps.step_04_evaluation.reports import generate_text_report
from module.steps.step_04_evaluation.metrics import compute_all_metrics, max_drawdown, sharpe_ratio
from module.steps.step_04_evaluation.portfolio_simulator import (
    compute_max_drawdown_from_equity,
    simulate_fold_usd,
    to_daily_returns_from_equity,
)
from module.steps.step_04_evaluation.selection_reports import (
    build_explanation_candidate_tickers,
    build_selection_audit_df,
    export_selection_audit,
    export_ticker_explanations,
)
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


def _prepare_fold_frames(


    df: pd.DataFrame,
    dates: pd.Index,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = (dates >= train_start) & (dates < train_end)
    test_mask = (dates >= train_end) & (dates < test_end)

    df_train = df.loc[train_mask]
    df_test = df.loc[test_mask]
    df_train = df_train[~df_train.index.duplicated(keep="last")]
    df_test = df_test[~df_test.index.duplicated(keep="last")]
    return df_train, df_test


def _build_filing_date_map(
    finnhub_data_dir: str,


    tickers: List[str],
) -> Dict[str, Dict[pd.Timestamp, pd.Timestamp]]:
    out: Dict[str, Dict[pd.Timestamp, pd.Timestamp]] = {}
    for ticker in tickers:
        tk = str(ticker)
        per_ticker: Dict[pd.Timestamp, pd.Timestamp] = {}
        for file_name in ["financials_reported_quarterly.json", "financials_reported_annual.json"]:
            path = Path(finnhub_data_dir) / tk / file_name
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue
            for item in payload.get("data", []):
                end_date = item.get("endDate")
                filed_date = item.get("filedDate") or item.get("acceptedDate")
                if not end_date or not filed_date:
                    continue
                try:
                    end_ts = pd.Timestamp(end_date).normalize()
                    filed_ts = pd.Timestamp(filed_date).normalize()
                except Exception:
                    continue
                old = per_ticker.get(end_ts)
                if old is None or filed_ts > old:
                    per_ticker[end_ts] = filed_ts
        out[tk] = per_ticker
    return out


def _prepare_fold_frames_by_filed_quarter(
    df: pd.DataFrame,
    filing_date_map: Dict[str, Dict[pd.Timestamp, pd.Timestamp]],
    train_start_quarter: pd.Period,
    analysis_quarter: pd.Period,
    fallback_lag_days: int = 45,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if df.empty:


        return df.copy(), df.copy(), pd.Series(dtype="datetime64[ns]")

    # Nuevo modo preferido: panel trimestral continuo por ticker.
    # Si el dataset trae year_quarter, el split train/test se hace por este campo.
    if "year_quarter" in df.columns:
        quarter_series = pd.PeriodIndex(df["year_quarter"], freq="Q")
        train_mask = (quarter_series >= train_start_quarter) & (quarter_series < analysis_quarter)
        test_mask = quarter_series == analysis_quarter

        df_train = df.loc[train_mask]
        df_test = df.loc[test_mask]
        df_train = df_train[~df_train.index.duplicated(keep="last")]
        df_test = df_test[~df_test.index.duplicated(keep="last")]

        # Serie informativa solo para logging/compatibilidad.
        if "snapshot_date" in df_test.columns:
            test_snapshot_dates = pd.to_datetime(df_test["snapshot_date"]).copy()
            test_snapshot_dates.index = df_test.index
        else:
            test_snapshot_dates = pd.Series(df_test.index.get_level_values("date").values, index=df_test.index)
        return df_train, df_test, test_snapshot_dates

    meta = df.reset_index()[["ticker", "date"]].copy()

    def _filed_date(row: pd.Series) -> pd.Timestamp:
        tk = str(row["ticker"])
        dt = pd.Timestamp(row["date"]).normalize()
        filed_dt = filing_date_map.get(tk, {}).get(dt)
        if filed_dt is not None:
            return filed_dt
        return dt + pd.Timedelta(days=max(int(fallback_lag_days), 0))

    filed_dates = meta.apply(_filed_date, axis=1)
    filed_quarters = filed_dates.dt.to_period("Q")

    train_mask = (filed_quarters >= train_start_quarter) & (filed_quarters < analysis_quarter)
    test_mask = filed_quarters == analysis_quarter

    df_train = df.loc[train_mask.values]
    df_test = df.loc[test_mask.values]
    df_train = df_train[~df_train.index.duplicated(keep="last")]
    df_test = df_test[~df_test.index.duplicated(keep="last")]

    test_filed_dates = pd.Series(filed_dates[test_mask].values, index=df_test.index)
    return df_train, df_test, test_filed_dates


def _filter_fold_tickers_by_history_span(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    required_years: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Keep only test tickers with at least required_years of train-history span."""
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
    required_quarters = max(1, int(required_years) * 4)
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
    analysis_quarter: pd.Period,
    filing_date_map: Dict[str, Dict[pd.Timestamp, pd.Timestamp]],
    lookback_quarters: int = 4,
    snapshot_lag_days: int = 0,
) -> pd.DataFrame:
    """
    Extraola features para tickers que no tienen reporte en analysis_quarter.
    
    Si un ticker tiene al menos `lookback_quarters` snapshots históricos,
    se promedian los últimos snapshots anteriores a analysis_quarter
    y se crea una fila "estimada" para agregar al test universe.
    
    Retorna df_test actualizado con snapshots extrapolados.
    """
    if df.empty:
        return df_test
    
    # Tickers que ya están en el test para analysis_quarter
    test_tickers = set(df_test.index.get_level_values("ticker").unique())
    
    # Todos los tickers disponibles
    all_tickers = set(df.index.get_level_values("ticker").unique())
    
    # Tickers que NO tienen snapshot en analysis_quarter
    missing_tickers = all_tickers - test_tickers
    
    extrapolated_rows = []
    
    analysis_quarter_end = analysis_quarter.end_time.normalize()
    analysis_quarter_start = analysis_quarter.start_time.normalize()
    analysis_snapshot_date = analysis_quarter_start + pd.Timedelta(days=max(int(snapshot_lag_days), 0))

    for ticker in missing_tickers:
        # Obtener todos los snapshots históricos de este ticker
        ticker_data = df.loc[df.index.get_level_values("ticker") == ticker].copy()

        # Evitar leakage: solo quarters previos al quarter analizado.
        if "year_quarter" in ticker_data.columns:
            ticker_quarters = pd.PeriodIndex(ticker_data["year_quarter"], freq="Q")
            ticker_data = ticker_data.loc[ticker_quarters < analysis_quarter]
        
        if len(ticker_data) < lookback_quarters:
            # No hay suficiente historia
            continue
        
        # Ordenar por date
        ticker_data = ticker_data.sort_index()
        
        # Tomar los últimos `lookback_quarters` snapshots
        recent_snapshots = ticker_data.tail(lookback_quarters)
        
        if len(recent_snapshots) < lookback_quarters:
            continue
        
        # Detectar columnas numéricas para promediar
        numeric_cols = recent_snapshots.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
        
        # Crear snapshot promediado
        aggregated = {}
        for col in numeric_cols:
            aggregated[col] = recent_snapshots[col].mean()
        
        # Copiar columnas no numéricas del snapshot más reciente
        last_row = recent_snapshots.iloc[-1]
        for col in recent_snapshots.columns:
            if col not in numeric_cols:
                aggregated[col] = last_row[col]
        
        # Crear índice multi para agregar al df_test (ticker, date)
        new_index = (ticker, analysis_quarter_end)

        # Forzar metadatos del quarter objetivo para no arrastrar valores del quarter previo.
        aggregated["year_quarter"] = f"{analysis_quarter.year}Q{analysis_quarter.quarter}"
        aggregated["snapshot_date"] = analysis_snapshot_date
        aggregated["is_fundamental_carry_forward"] = True

        # Mantener trazabilidad del reporte realmente usado (el más reciente histórico).
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
    
    log.info(f"[Fallback Extrapolation] Agregados {len(extrapolated_rows)} snapshots estimados (últimos {lookback_quarters} Q)")
    
    return df_test_extended


def _spy_quarterly_returns(spy_prices: pd.Series) -> Dict[str, float]:
    """
    Precalcula el retorno trimestral del SPY para cada quarter presente en spy_prices.

    Para cada quarter Q, calcula: (precio_último_día_Q / precio_último_día_Q-1) - 1
    usando únicamente precios de cierre al final de cada quarter.

    Devuelve un dict {periodo_quarter_str: retorno_float}, p. ej.:
        {"2024Q1": 0.107, "2024Q2": -0.032, ...}
    """
    spy = spy_prices.sort_index().dropna()
    quarterly = spy.resample("QE").last()   # último precio de cada quarter
    spy_returns: Dict[str, float] = {}
    for i in range(1, len(quarterly)):
        p0 = quarterly.iloc[i - 1]
        p1 = quarterly.iloc[i]
        period = quarterly.index[i].to_period("Q")
        if p0 > 0:
            spy_returns[str(period)] = float(p1 / p0 - 1)
    return spy_returns


def _excess_return_label(
    df: pd.DataFrame,
    spy_prices: Optional[pd.Series] = None,
    sector_map: Optional[Dict[str, str]] = None,
) -> pd.Series:
    """
        Label de outperformance sectorial por snapshot: 1 si el ticker superó la
        mediana de su sector usando el forward_return definido para ese snapshot.

        El agrupado se hace por quarter del snapshot (no por quarter calendario
        de salida), porque todos los tickers del mismo snapshot comparten la misma
        regla de entrada/salida (lag + holding) y deben compararse entre sí.
    """
    dates = df.index.get_level_values("date")
    tickers = df.index.get_level_values("ticker")
    snapshot_quarters = dates.to_period("Q")
    forward_return = df["forward_return"]
    valid_mask = forward_return.notna()

    quarter_median = df.groupby(snapshot_quarters)["forward_return"].transform("median")

    want_vs_sector = str(BASE_AGENTS_LABEL_MODE).lower().strip() == "vs_sector"
    if want_vs_sector and sector_map is not None and len(sector_map) > 0:
        sector_series = pd.Series(tickers, index=df.index).map(sector_map).fillna("Unknown")
        temp = pd.DataFrame(
            {
                "forward_return": forward_return.values,
                "sector": sector_series.values,
                "snapshot_quarter": snapshot_quarters.astype(str),
            },
            index=df.index,
        )

        grp = ["sector", "snapshot_quarter"]
        sector_quarter_median = temp.groupby(grp)["forward_return"].transform("median")
        sector_quarter_count = temp.groupby(grp)["forward_return"].transform("count")

        enough_peers = (temp["sector"] != "Unknown") & (sector_quarter_count >= int(BASE_LABEL_SECTOR_MIN_PEERS))
        benchmark = pd.Series(np.where(enough_peers, sector_quarter_median, quarter_median), index=df.index)

        n_with_sector = int((temp["sector"] != "Unknown").sum())
        n_sector_label = int(enough_peers.sum())
        n_fallback = int(len(df) - n_sector_label)
        log.debug(
            "[Label] mode=vs_sector | sector conocidos=%d/%d | con peers suficientes=%d | fallback_universo=%d",
            n_with_sector,
            len(df),
            n_sector_label,
            n_fallback,
        )
        labels = (forward_return > benchmark).astype(float)
        return labels.where(valid_mask)

    # Fallback explícito: mediana del universo por snapshot quarter.
    log.debug("[Label] mode=vs_universe (o sin sector_map) — usando mediana del universo por snapshot quarter")
    labels = (forward_return > quarter_median).astype(float)
    return labels.where(valid_mask)


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
        cc = "Close" if "Close" in prices.columns else prices.columns[3]
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
    cc = "Close" if "Close" in prices.columns else prices.columns[3]
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
    df_train_norm: pd.DataFrame,
    df_test_norm: pd.DataFrame,
    spy_prices: Optional[pd.Series] = None,
    sector_map: Optional[Dict[str, str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Label configurable para agentes base:
    # - vs_sector (default): y=1 si supera mediana sectorial por snapshot quarter.
    # - vs_universe: y=1 si supera mediana del universo por snapshot quarter.
    forward_train = _excess_return_label(df_train, spy_prices, sector_map).reindex(df_train_norm.index)
    forward_test  = _excess_return_label(df_test,  spy_prices, sector_map).reindex(df_test_norm.index)
    y_train = forward_train.dropna().astype(int)
    y_test  = forward_test.dropna().astype(int)

    df_train_norm = df_train_norm.loc[y_train.index]
    df_test_norm  = df_test_norm.loc[y_test.index]
    return df_train_norm, df_test_norm, y_train, y_test


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


def _export_fold_usd_artifacts(
    *,
    backtest_results_dir: str,
    fold_id_num: int,
    sim_out: Dict,
    selection_df: pd.DataFrame,
    metrics_payload: Dict,
) -> None:
    fold_dir = Path(backtest_results_dir) / f"fold_{fold_id_num}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    trades_df = sim_out.get("trades_df", pd.DataFrame())
    equity_df = sim_out.get("equity_curve_df", pd.DataFrame())
    summary = sim_out.get("fold_summary", {})

    if not trades_df.empty:
        trades_df.to_csv(fold_dir / "trades.csv", index=False)
    else:
        pd.DataFrame(columns=[
            "fold_id", "datetime", "action", "ticker", "raw_price", "exec_price", "shares",
            "notional_usd", "fee_usd", "slippage_pct", "entry_date_requested", "entry_date_used",
            "exit_date_requested", "exit_date_used", "reason",
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

    _safe_json_dump(fold_dir / "portfolio_summary.json", summary)
    _safe_json_dump(fold_dir / "metrics.json", metrics_payload)

    # Generate detailed trades report showing compra/venta side-by-side with USD values
    if not trades_df.empty:
        _generate_detailed_trades_report(trades_df, fold_dir)


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
            ("sentiment", filter_asof(rec_df, as_of=as_of) if rec_df is not None else pd.DataFrame(), None, f"rec_df:{ticker}"),
            ("sentiment", filter_asof(mspr_df, as_of=as_of) if mspr_df is not None else pd.DataFrame(), None, f"mspr_df:{ticker}"),
            ("insider", filter_asof(ins_df, as_of=as_of, date_col="date") if ins_df is not None else pd.DataFrame(), "date", f"insider_df:{ticker}"),
            ("sentiment", filter_asof(eps_df, as_of=as_of) if eps_df is not None else pd.DataFrame(), None, f"eps_df:{ticker}"),
            ("technical", filter_asof(prices_df, as_of=as_of) if prices_df is not None else pd.DataFrame(), None, f"prices_df:{ticker}"),
            ("fundamental", filter_asof(cons_df, as_of=as_of) if cons_df is not None else pd.DataFrame(), None, f"consolidated_df:{ticker}"),
            ("valuation", filter_asof(cons_df, as_of=as_of) if cons_df is not None else pd.DataFrame(), None, f"valuation_input:{ticker}"),
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
    backtest_results_dir: str,
    plots_dir: str,
    start_date: str,
    end_date: str,
    walkforward_train_years: int,
    walkforward_test_quarters: int,
    risk_free_rate: float,
    top_n_stocks: int = 10,
    random_seed: int = 42,
    spy_prices: Optional[pd.Series] = None,
    snapshot_lag_days: Optional[int] = 45,
    holding_period_months: int = 3,
    finnhub_data_dir: str = "data_finnhub",
    analysis_frequency: str = "quarterly",
    annual_anchor_date: Optional[pd.Timestamp] = None,
) -> Dict:
    Path(agents_results_dir).mkdir(parents=True, exist_ok=True)

    backtester = WalkForwardBacktester(
        train_years=walkforward_train_years,
        test_quarters=walkforward_test_quarters,
        risk_free=risk_free_rate,
        results_dir=backtest_results_dir,
        top_n_stocks=top_n_stocks,
    )
    visualizer = Visualizer(plots_dir=plots_dir)
    normalizer = SectorNormalizer(min_peers=SECTOR_ZSCORE_MIN_PEERS)

    folds = backtester.generate_folds(start_date, end_date)
    agent_diag_history: Dict[str, List] = {
        "fundamental": [], "valuation": [], "momentum": [], "bear": [],
        "sentiment": [], "sector_rotation": [], "meta_learner": [],
    }
    ablation_results: List[Dict] = []
    data_router = DataRouter(data_dir=finnhub_data_dir)

    leakage_rows: List[Dict] = []
    missing_prices_rows: List[Dict] = []
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

    log.info(
        "[WalkForward] Ventana train dinámica activada: max=%sY -> min=%sY | mínimo test=%s%%",
        max_train_years,
        min_train_years,
        MIN_TEST_TICKERS_PERCENT,
    )

    filing_date_map = _build_filing_date_map(
        finnhub_data_dir=finnhub_data_dir,
        tickers=df.index.get_level_values("ticker").unique().tolist(),
    )
    resolved_snapshot_lag_days = int(snapshot_lag_days) if snapshot_lag_days is not None else 0
    analysis_frequency = str(analysis_frequency).strip().lower()
    if analysis_frequency not in {"quarterly", "annual"}:
        raise ValueError("analysis_frequency debe ser 'quarterly' o 'annual'")

    if analysis_frequency == "annual" and annual_anchor_date is None:
        annual_anchor_date = pd.Timestamp(start_date).normalize() + pd.Timedelta(days=max(int(resolved_snapshot_lag_days), 0))
    if annual_anchor_date is not None:
        annual_anchor_date = pd.Timestamp(annual_anchor_date).normalize()

    membership_path = Path(SP500_HISTORIC_CSV_PATH)
    if not membership_path.is_absolute():
        membership_path = Path.cwd() / membership_path
    sp500_membership_df = _load_sp500_membership(membership_path) if USE_DYNAMIC_SP500_UNIVERSE else pd.DataFrame()
    if USE_DYNAMIC_SP500_UNIVERSE:
        if sp500_membership_df.empty:
            log.warning("[WalkForward] Universo dinámico activo pero sin membresía SP500 utilizable (%s)", membership_path)
        else:
            log.info("[WalkForward] Membresía SP500 dinámica cargada: %s filas (%s)", len(sp500_membership_df), membership_path)

    prev_membership_tickers: Optional[set[str]] = None
    prev_membership_entry_date: Optional[pd.Timestamp] = None


    def _has_price_coverage(entry_date: pd.Timestamp, actual_end: pd.Timestamp) -> bool:
        if actual_end <= entry_date:
            return False
        for prices in prices_dict.values():
            if prices is None or prices.empty:
                continue
            cc = "Close" if "Close" in prices.columns else prices.columns[3]
            period = prices.loc[entry_date:actual_end, cc]
            if len(period) >= 2:
                return True
        return False

    for fold_id, (_fold_train_start, train_end, _test_end, _fold_train_years) in enumerate(folds, 1):
        analysis_quarter = train_end.to_period("Q")

        q_start = analysis_quarter.start_time.normalize()
        lag_days = max(int(resolved_snapshot_lag_days), 0)
        entry_date = q_start + pd.Timedelta(days=lag_days)

        if analysis_frequency == "annual":
            if annual_anchor_date is not None and entry_date < annual_anchor_date:
                log.debug(
                    "[Fold %s] Skip anual: entry %s < anchor %s",
                    fold_id,
                    entry_date.date(),
                    annual_anchor_date.date(),
                )
                continue
            # Formato para análisis anual: "2026YQ2" = año 2026, yearly, reporte Q2
            analysis_quarter_label = f"{train_end.year}YQ{train_end.quarter}"
        else:
            # Formato trimestral: "2026Q2"
            analysis_quarter_label = f"{train_end.year}Q{train_end.quarter}"

        run_id = analysis_quarter_label

        selected_train_years: Optional[int] = None
        selected_train_start: Optional[pd.Timestamp] = None
        selected_df_train: Optional[pd.DataFrame] = None
        selected_df_test: Optional[pd.DataFrame] = None
        selected_test_filed_dates: Optional[pd.Series] = None
        selected_eligibility_rows: Optional[list[dict]] = None
        selected_eligibility_train_years: Optional[int] = None
        last_eligibility_rows: list[dict] = []
        last_eligibility_train_years: Optional[int] = None

        active_tickers_on_entry = _active_sp500_tickers_on_date(sp500_membership_df, entry_date)
        fold_universe_tickers = len(active_tickers_on_entry) if (USE_DYNAMIC_SP500_UNIVERSE and active_tickers_on_entry) else total_universe_tickers
        min_test_tickers_required = int(np.ceil(fold_universe_tickers * MIN_TEST_TICKERS_PERCENT / 100.0))
        if USE_DYNAMIC_SP500_UNIVERSE and active_tickers_on_entry:
            log.info(
                f"[{run_id}] SP500 dinámico @entry {entry_date.date()}: {len(active_tickers_on_entry)} miembros activos"
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

        for candidate_years in range(max_train_years, min_train_years - 1, -1):
            candidate_train_start = train_end - pd.DateOffset(years=int(candidate_years))
            candidate_train_start_quarter = candidate_train_start.to_period("Q")

            cand_train, cand_test, cand_test_filed_dates = _prepare_fold_frames_by_filed_quarter(
                df=df,
                filing_date_map=filing_date_map,
                train_start_quarter=candidate_train_start_quarter,
                analysis_quarter=analysis_quarter,
                fallback_lag_days=45,
            )

            if ENABLE_FALLBACK_EXTRAPOLATION:
                cand_test = _extrapolate_missing_snapshots(
                    df=df,
                    df_test=cand_test,
                    analysis_quarter=analysis_quarter,
                    filing_date_map=filing_date_map,
                    lookback_quarters=FALLBACK_LOOK_BACK_QUARTERS,
                    snapshot_lag_days=resolved_snapshot_lag_days,
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
                required_years=int(candidate_years),
            )
            eligible_set = set(str(tk) for tk in eligible_tickers)
            dropped_insufficient_history = test_tickers_before_history - eligible_set

            if USE_DYNAMIC_SP500_UNIVERSE and active_tickers_on_entry:
                fold_base_universe = set(active_tickers_on_entry)
            else:
                fold_base_universe = set(all_master_tickers)

            # Universo trazado para auditoría completa del fold: arranque + candidatos observados en test.
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
                        "candidate_train_years": int(candidate_years),
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
            last_eligibility_train_years = int(candidate_years)

            test_tickers_count = int(cand_test.index.get_level_values("ticker").nunique()) if not cand_test.empty else 0
            test_tickers_pct = (100.0 * test_tickers_count / fold_universe_tickers) if fold_universe_tickers > 0 else 0.0
            if dropped_insufficient_history:
                log.info(
                    f"[{run_id}]   Excluidos por historial insuficiente (<{candidate_years}Y train): "
                    f"{len(dropped_insufficient_history)} | muestra: {_sample_tickers(dropped_insufficient_history)}"
                )
            log.info(
                f"[{run_id}] Intento train={candidate_years}Y -> test elegible {test_tickers_count}/{fold_universe_tickers} "
                f"({test_tickers_pct:.1f}%) | min requerido={min_test_tickers_required}"
            )

            if test_tickers_count >= min_test_tickers_required:
                selected_train_years = int(candidate_years)
                selected_train_start = pd.Timestamp(candidate_train_start)
                selected_df_train = cand_train
                selected_df_test = cand_test
                selected_test_filed_dates = cand_test_filed_dates
                selected_eligibility_rows = current_eligibility_rows
                selected_eligibility_train_years = int(candidate_years)
                log.info(
                    f"[{run_id}] Ventana seleccionada: {selected_train_years}Y "
                    f"({selected_train_start.date()} -> {train_end.date()}) | tickers elegibles={len(eligible_tickers)}"
                )
                break

        rows_to_export = selected_eligibility_rows if selected_eligibility_rows is not None else last_eligibility_rows
        years_to_export = (
            selected_eligibility_train_years if selected_eligibility_train_years is not None else last_eligibility_train_years
        )
        if rows_to_export:
            eligibility_path = Path(agents_results_dir) / f"quarter_{run_id}_ticker_eligibility_audit.csv"
            eligibility_path.parent.mkdir(parents=True, exist_ok=True)
            eligibility_df = pd.DataFrame(rows_to_export)
            eligibility_df.to_csv(eligibility_path, index=False)

            summary_path = Path(agents_results_dir) / f"quarter_{run_id}_eligibility_reason_summary.csv"
            summary_df = (
                eligibility_df.groupby(
                    [
                        "candidate_train_years",
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
                .sort_values(["candidate_train_years", "count", "reason"], ascending=[True, False, True])
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
                f"[{run_id}] No se alcanzó cobertura mínima de test ({MIN_TEST_TICKERS_PERCENT}%) "
                f"ni reduciendo train hasta {min_train_years}Y — fold omitido."
            )
            continue

        train_start = selected_train_start
        _train_years = selected_train_years
        df_train = selected_df_train
        df_test = selected_df_test
        test_filed_dates = selected_test_filed_dates

        log.info(f"\n{'='*60}")
        log.info(f"  ANALISIS {run_id}")
        log.info(f"  Train : {train_start.date()} → {train_end.date()}  ({_train_years} años)")
        frequency_mode = "anual (Y)" if analysis_frequency == "annual" else "trimestral (Q)"
        log.info(f"  Mode  : {frequency_mode}")
        log.info(
            f"  Test  : snapshot simulado en {entry_date.date()} "
            f"(Q{analysis_quarter.quarter} start + {lag_days}d)"
        )
        log.info(f"{'='*60}")

        # Calcular entry_date: desde el primer día del quarter + lag configurado
        
        log.info(
            f"[{run_id}] Snapshot lag empieza en: {q_start.date()} (primer día del Q{analysis_quarter.quarter}) "
            f"+ {lag_days} días máximo = fecha mínima de precios requerida: {entry_date.date()}"
        )

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

        if len(df_train) < 100:
            log.warning(f"[{run_id}] Train insuficiente ({len(df_train)} observaciones, mínimo 100) — fold omitido.")
            continue

        df_train_norm = apply_sector_normalization(df_train, sector_map, normalizer, fit=True)
        df_test_norm = apply_sector_normalization(df_test, sector_map, normalizer, fit=False)
        df_train_norm = df_train_norm[~df_train_norm.index.duplicated(keep="last")]
        df_test_norm = df_test_norm[~df_test_norm.index.duplicated(keep="last")]

        df_train_norm, df_test_norm, y_train, y_test = _prepare_fold_labels(
            df_train=df_train,
            df_test=df_test,
            df_train_norm=df_train_norm,
            df_test_norm=df_test_norm,
            spy_prices=spy_prices,
            sector_map=sector_map,
        )
        if (df_test_norm.empty or y_test.empty) and not df_test_norm.empty:
            partial_returns = _compute_partial_forward_returns(
                prices_dict=prices_dict,
                tickers=df_test_norm.index.get_level_values("ticker"),
                entry_date=entry_date,
                exit_date=actual_end,
            )
            if partial_returns:
                df_test_partial = df_test_norm.copy()
                df_test_partial["forward_return"] = (
                    df_test_partial.index.get_level_values("ticker").map(partial_returns)
                )
                forward_test = _excess_return_label(df_test_partial, spy_prices, sector_map).reindex(df_test_norm.index)
                y_test = forward_test.dropna().astype(int)
                df_test_norm = df_test_norm.loc[y_test.index]
                log.info(
                    f"[{run_id}] Labels parciales con precios hasta {actual_end.date()} "
                    f"— {len(y_test)} observaciones"
                )
        if df_test_norm.empty or y_test.empty:
            log.warning(f"[{run_id}] Test vacio tras preparar labels — fold omitido.")
            continue

        try:
            agents, df_test_scored, df_train_with_oof = train_fold(
                df_train_norm=df_train_norm,
                df_test_norm=df_test_norm,
                y_train=y_train,
                y_test=y_test,
                fold_id=run_id,
                agents_results_dir=agents_results_dir,
                random_seed=random_seed,
                sector_map=sector_map,
                spy_prices=spy_prices,
            )

            meta = agents["meta_learner"]
            eval_metrics = meta.evaluate(df_test_scored, y_test, fold=run_id)

            visualizer.plot_score_distribution(df_test_scored, fold=run_id)

            preds_df = df_test_scored[["final_score", "label"]].copy()
            preds_df["ticker"] = preds_df.index.get_level_values("ticker")
            preds_df["date"] = preds_df.index.get_level_values("date")
            preds_df = preds_df.reset_index(drop=True)

            fold_result = backtester.simulate_portfolio(
                predictions_df=preds_df.rename(columns={"final_score": "score"}),
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

            # Auditoría PIT leakage sobre fuentes as-of usadas por el fold.
            fold_test_tickers = df_test_scored.index.get_level_values("ticker").unique().tolist()
            fold_leak_rows = _audit_fold_leakage(
                router=data_router,
                tickers=fold_test_tickers,
                as_of=entry_date,
                fold_id=run_id,
            )
            leakage_rows.extend(fold_leak_rows)
            n_leak_fold = int(sum(1 for r in fold_leak_rows if int(r.get("n_rows_future_detected", 0)) > 0))
            if n_leak_fold > 0:
                log.warning("[%s] Leakage audit detectó %s incidencias en fuentes filtradas.", run_id, n_leak_fold)

            # Modo monetario USD (sin reemplazar métricas históricas actuales).
            if USE_DOLLAR_BACKTEST:
                selected_tickers = list(fold_result.get("selected_tickers", []))
                ticker_weights = dict(fold_result.get("ticker_weights", {}))
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

                preds_scored_for_selection = preds_df.rename(columns={"final_score": "score"})
                selection_df = _build_selection_df(
                    preds_scored=preds_scored_for_selection,
                    selected_tickers=sim_out.get("selected_tickers_used", selected_tickers),
                    ticker_weights=sim_out.get("weights_used", ticker_weights),
                )
                _export_fold_usd_artifacts(
                    backtest_results_dir=backtest_results_dir,
                    fold_id_num=fold_id,
                    sim_out=sim_out,
                    selection_df=selection_df,
                    metrics_payload={
                        "fold_id": run_id,
                        "classification_metrics": eval_metrics,
                        "legacy_return_metrics": {k: v for k, v in fold_result.items() if not str(k).startswith("_")},
                        "usd_summary": sim_out.get("fold_summary", {}),
                    },
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

            visualizer.plot_fold_performance(fold_result, fold_id=run_id)

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
            export_selection_audit(audit_df, agents_results_dir, fold_id=analysis_quarter_label, prefix="quarter")

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
            export_fold_scores(fold_scores_df, agents_results_dir, fold_id=analysis_quarter_label)

            # Auditoria completa por quarter: snapshot por ticker + detalle agente-feature.
            export_quarter_snapshot_audit(
                df_test_scored=df_test_scored,
                year_quarter=analysis_quarter_label,
                agents_results_dir=agents_results_dir,
            )
            export_quarter_agent_feature_audit(
                df_test_scored=df_test_scored,
                agents=agents,
                year_quarter=analysis_quarter_label,
                agents_results_dir=agents_results_dir,
            )

            for ag_name, ag in agents.items():
                agent_diag_history[ag_name].append(ag._diagnostics.copy())

            for ag_name in ["fundamental", "valuation", "momentum", "bear", "sector_rotation"]:
                ag = agents[ag_name]
                if hasattr(ag, "_feature_cols") and ag.is_trained:
                    try:
                        imp_path = (
                            Path(agents_results_dir) / ag_name
                            / f"feature_importances_{run_id}.csv"
                        )
                        if imp_path.exists():
                            imp = pd.read_csv(imp_path, index_col=0)["importance"]
                            visualizer.plot_feature_importances(imp, ag_name, fold=run_id)
                    except Exception as plot_exc:
                        log.debug(f"[Visualizer] plot_feature_importances {ag_name} {run_id}: {plot_exc}")

            explain_top_tickers(
                agents=agents,
                df_test=df_test_scored,
                scores=df_test_scored["final_score"],
                fold_id=analysis_quarter_label,
                agents_results_dir=agents_results_dir,
                selected_tickers=fold_result.get("selected_tickers", []),
                audit_df=audit_df,
            )

            if RUN_ABLATION_STUDY:
                abl = run_ablation_study(
                    df_test_scored=df_test_scored,
                    y_test=y_test,
                    df_train_norm=df_train_with_oof,
                    y_train=y_train,
                    agents_results_dir=agents_results_dir,
                    fold_id=run_id,
                    random_seed=random_seed,
                )
                if abl:
                    ablation_results.append(abl)

        except Exception as e:
            log.error(f"Análisis {run_id} falló: {e}", exc_info=True)
            continue

    summary = backtester.summarize()
    backtester.save_folds_summary(plots_dir=plots_dir)

    # CSV consolidado de todos los folds: una fila por ticker-quarter con
    # scores, interpretaciones y explicaciones de cada agente.
    export_all_folds_scores(agents_results_dir)

    diag_path = Path(agents_results_dir) / "all_folds_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(agent_diag_history, f, indent=2, default=str)

    last_agent_diag = {k: v[-1] if v else {} for k, v in agent_diag_history.items()}
    visualizer.plot_full_report(
        strategy_returns=backtester.all_strategy_returns,
        benchmark_returns=backtester.all_benchmark_returns,
        fold_results=backtester.fold_results,
        agent_diagnostics=last_agent_diag,
    )

    if RUN_ABLATION_STUDY and ablation_results:
        summarize_ablation(ablation_results, agents_results_dir=agents_results_dir)

    generate_text_report(
        summary=summary,
        fold_results=backtester.fold_results,
        agent_diag_history=agent_diag_history,
        backtest_results_dir=backtest_results_dir,
    )

    results_root = Path(backtest_results_dir).parent
    backtest_root = Path(backtest_results_dir)
    backtest_root.mkdir(parents=True, exist_ok=True)
    baselines_dir = backtest_root / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    # Reportes obligatorios de auditoría/leakage y precios faltantes.
    leakage_df = pd.DataFrame(leakage_rows)
    if leakage_df.empty:
        leakage_df = pd.DataFrame(columns=[
            "fold_id", "ticker", "feature_group", "n_rows_future_detected",
            "max_future_date_detected", "context",
        ])
    leakage_df.to_csv(results_root / "leakage_audit.csv", index=False)

    missing_prices_df = pd.DataFrame(missing_prices_rows)
    if missing_prices_df.empty:
        missing_prices_df = pd.DataFrame(columns=["fold_id", "ticker", "start_date", "end_date", "reason"])
    missing_prices_df.to_csv(backtest_root / "missing_prices_report.csv", index=False)

    final_rows = []
    baselines_rows = []
    value_availability_rows = []
    value_selection_rows = []
    baseline_fold_compare_rows: List[Dict] = []

    strategy_equity_global = pd.DataFrame(columns=["date", "equity_usd", "cash_usd", "positions_value_usd"])
    if USE_DOLLAR_BACKTEST and strategy_equity_parts:
        strategy_equity_global = (
            pd.concat(strategy_equity_parts, axis=0)
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
        strategy_equity_global.to_csv(backtest_root / "strategy_equity_curve.csv", index=False)

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
                benchmark_equity.to_csv(backtest_root / "benchmark_equity_curve.csv", index=False)
            bench_available = not benchmark_equity.empty
            benchmark_summary = {
                "final_value_usd": float(bench_sim.get("fold_summary", {}).get("ending_capital_usd", np.nan)),
                "return_pct": float(bench_sim.get("fold_summary", {}).get("pnl_pct", np.nan)),
                "fees_usd": float(bench_sim.get("fold_summary", {}).get("total_fees_usd", 0.0)),
                "availability_flag": bool(bench_available),
            }
            _safe_json_dump(backtest_root / "benchmark_summary.json", benchmark_summary)

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
                    random_curves.append(
                        pd.concat(sim_parts).sort_values("date").drop_duplicates(subset=["date"], keep="last")
                    )

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

            def _concat_parts(parts: List[pd.DataFrame]) -> pd.DataFrame:
                if not parts:
                    return pd.DataFrame(columns=["date", "equity_usd", "cash_usd", "positions_value_usd"])
                return pd.concat(parts).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

            ew_curve = _concat_parts(ew_parts)
            mom_curve = _concat_parts(mom_parts)
            value_curve = _concat_parts(value_parts)
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
            bs_df.to_csv(results_root / "baselines_summary.csv", index=False)

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
        _safe_json_dump(results_root / "final_portfolio_value.json", final_value_payload)

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
            fold_compare_df.to_csv(backtest_root / "fold_comparison_summary.csv", index=False)

        summary_payload = {
            "timestamp": datetime.now().isoformat(),
            "legacy_summary": summary,
            "usd_strategy": strategy_summary,
            "total_leakage_incidents": int((leakage_df["n_rows_future_detected"] > 0).sum()) if not leakage_df.empty else 0,
            "folds_usd": fold_usd_df.to_dict(orient="records"),
            "baselines": pd.DataFrame(baselines_rows).to_dict(orient="records"),
        }
        _safe_json_dump(results_root / "final_summary.json", summary_payload)

        final_summary_csv_rows = []
        final_summary_csv_rows.append({"name": "strategy_main", **strategy_summary})
        for row in baselines_rows:
            if row.get("name") != "strategy_main":
                final_summary_csv_rows.append(row)
        pd.DataFrame(final_summary_csv_rows).to_csv(results_root / "final_summary.csv", index=False)

        # Plots requeridos para memoria TFM.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_path = Path(plots_dir)
        plots_path.mkdir(parents=True, exist_ok=True)

        if not strategy_equity_global.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(pd.to_datetime(strategy_equity_global["date"]), strategy_equity_global["equity_usd"], label="strategy_main", lw=2)
            if (backtest_root / "benchmark_equity_curve.csv").exists():
                bdf = pd.read_csv(backtest_root / "benchmark_equity_curve.csv")
                plt.plot(pd.to_datetime(bdf["date"]), bdf["equity_usd"], label="benchmark", lw=1.8)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.title("Equity Curve USD")
            plt.tight_layout()
            plt.savefig(plots_path / "equity_curve_usd.png", dpi=140)
            plt.close()

            plt.figure(figsize=(12, 6))
            plt.plot(pd.to_datetime(strategy_equity_global["date"]), strategy_equity_global["equity_usd"], label="strategy_main", lw=2)
            if (backtest_root / "benchmark_equity_curve.csv").exists():
                bdf = pd.read_csv(backtest_root / "benchmark_equity_curve.csv")
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

        p_bench = backtest_root / "benchmark_equity_curve.csv"
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
            ann_df.to_csv(backtest_root / "annual_return_comparison.csv", index_label="year")

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

    return summary
