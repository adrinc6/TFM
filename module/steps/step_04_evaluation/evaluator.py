"""Walk-forward evaluation orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from environment import (
    PORTFOLIO_MIN_SCORE,
    RUN_ABLATION_STUDY,
    SECTOR_ZSCORE_MIN_PEERS,
    MIN_TEST_TICKERS_PERCENT,
    ENABLE_FALLBACK_EXTRAPOLATION,
    FALLBACK_LOOK_BACK_QUARTERS,
)
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
from module.steps.step_04_evaluation.selection_reports import (
    build_explanation_candidate_tickers,
    build_selection_audit_df,
    export_selection_audit,
    export_ticker_explanations,
)
from module.steps.step_04_evaluation.visualization import Visualizer

log = logging.getLogger(__name__)


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

    if sector_map is not None and len(sector_map) > 0:
        # Añadir sector al DataFrame temporal para calcular mediana sectorial
        sector_series = pd.Series(tickers, index=df.index).map(sector_map).fillna("Unknown")
        temp = pd.DataFrame({
            "forward_return": forward_return.values,
            "sector": sector_series.values,
            "snapshot_quarter": snapshot_quarters.astype(str),
        }, index=df.index)

        # Mediana del sector × snapshot_quarter como benchmark
        sector_quarter_median = temp.groupby(["sector", "snapshot_quarter"])["forward_return"].transform("median")
        n_with_sector = (sector_series != "Unknown").sum()
        log.debug(f"[Label] Usando mediana sectorial por snapshot quarter — {n_with_sector}/{len(df)} tickers con sector conocido")
        labels = (forward_return > sector_quarter_median).astype(float)
        return labels.where(valid_mask)

    # Fallback: mediana del universo completo por snapshot quarter
    log.debug("[Label] sector_map no disponible — usando mediana del universo por snapshot quarter")
    quarter_median = df.groupby(snapshot_quarters)["forward_return"].transform("median")
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
    # Label: outperformance relativa a la mediana del sector en cada quarter.
    # y=1 si el forward_return del ticker supera la mediana de su sector en ese período.
    forward_train = _excess_return_label(df_train, spy_prices, sector_map).reindex(df_train_norm.index)
    forward_test  = _excess_return_label(df_test,  spy_prices, sector_map).reindex(df_test_norm.index)
    y_train = forward_train.dropna().astype(int)
    y_test  = forward_test.dropna().astype(int)

    df_train_norm = df_train_norm.loc[y_train.index]
    df_test_norm  = df_test_norm.loc[y_test.index]
    return df_train_norm, df_test_norm, y_train, y_test


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

    total_universe_tickers = int(df.index.get_level_values("ticker").nunique())
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

    for fold_id, (train_start, train_end, test_end, _train_years) in enumerate(folds, 1):
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

        train_start_quarter = train_start.to_period("Q")
        df_train, df_test, test_filed_dates = _prepare_fold_frames_by_filed_quarter(
            df=df,
            filing_date_map=filing_date_map,
            train_start_quarter=train_start_quarter,
            analysis_quarter=analysis_quarter,
            fallback_lag_days=45,
        )
        
        # Aplicar extrapolación de features si está habilitada
        if ENABLE_FALLBACK_EXTRAPOLATION:
            df_test = _extrapolate_missing_snapshots(
                df=df,
                df_test=df_test,
                analysis_quarter=analysis_quarter,
                filing_date_map=filing_date_map,
                lookback_quarters=FALLBACK_LOOK_BACK_QUARTERS,
                snapshot_lag_days=resolved_snapshot_lag_days,
            )
        test_tickers_count = int(df_test.index.get_level_values("ticker").nunique()) if not df_test.empty else 0
        test_tickers_pct = (100.0 * test_tickers_count / total_universe_tickers) if total_universe_tickers > 0 else 0.0
        log.info(
            f"[{run_id}] Universo test: {test_tickers_count}/{total_universe_tickers} tickers "
            f"({test_tickers_pct:.1f}%)"
        )
        # Calcular dinámicamente el mínimo requerido basado en porcentaje del universo total
        min_test_tickers_required = int(total_universe_tickers * MIN_TEST_TICKERS_PERCENT / 100.0)
        if test_tickers_count < min_test_tickers_required:
            log.warning(
                f"[{run_id}] Test insuficiente: {test_tickers_count} < {min_test_tickers_required} "
                f"(mínimo {MIN_TEST_TICKERS_PERCENT}% de {total_universe_tickers}) — fold omitido."
            )
            continue

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

    return summary
