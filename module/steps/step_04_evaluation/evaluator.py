"""Walk-forward evaluation orchestration."""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from environment import PORTFOLIO_MIN_SCORE, RUN_ABLATION_STUDY, SECTOR_ZSCORE_MIN_PEERS
from module.steps.step_02_dataset.builders.sector import SectorNormalizer
from module.steps.step_02_dataset.normalization import apply_sector_normalization
from module.steps.step_03_training.training import train_fold
from module.steps.step_04_evaluation.ablation import run_ablation_study, summarize_ablation
from module.steps.step_04_evaluation.backtester import WalkForwardBacktester
from module.steps.step_04_evaluation.fold_report import (
    build_fold_scores_df,
    export_fold_scores,
    export_all_folds_scores,
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
    fold_id: int,
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
    Label de outperformance sectorial: 1 si el ticker superó la mediana de su
    sector durante el período FORWARD (el quarter siguiente al snapshot).

    La agrupación usa el quarter FORWARD — el período cuyo retorno se mide —
    no el quarter del snapshot. Esto es correcto porque:
      - El snapshot de Q2 (tomado ~30 días antes del inicio de Q3) tiene
        forward_return = retorno durante Q3.
      - La mediana sectorial debe comparar retornos del MISMO período (Q3),
        no del período del snapshot (Q2).

    Con DAYS_BEFORE_QUARTER_START=30: as_of ~ Dec 1 → snapshot en Q4,
    pero forward_return mide Q1. El forward quarter = next_quarter_end(as_of).

    Fallback: si sector_map no está disponible o un ticker no tiene sector,
    compara contra la mediana del universo completo en ese forward quarter.
    """
    dates = df.index.get_level_values("date")
    tickers = df.index.get_level_values("ticker")
    # Forward quarter: el quarter cuyo retorno medimos (no el del snapshot)
    forward_quarters = (dates + pd.offsets.QuarterEnd(1)).to_period("Q")

    if sector_map is not None and len(sector_map) > 0:
        # Añadir sector al DataFrame temporal para calcular mediana sectorial
        sector_series = pd.Series(tickers, index=df.index).map(sector_map).fillna("Unknown")
        temp = pd.DataFrame({
            "forward_return": df["forward_return"].values,
            "sector": sector_series.values,
            "forward_quarter": forward_quarters.astype(str),
        }, index=df.index)

        # Mediana del sector × forward_quarter como benchmark
        sector_quarter_median = temp.groupby(["sector", "forward_quarter"])["forward_return"].transform("median")
        n_with_sector = (sector_series != "Unknown").sum()
        log.debug(f"[Label] Usando mediana sectorial (forward quarter) como benchmark — {n_with_sector}/{len(df)} tickers con sector conocido")
        return (df["forward_return"] > sector_quarter_median).astype(int)

    # Fallback: mediana del universo completo por forward quarter
    log.debug("[Label] sector_map no disponible — usando mediana del universo (forward quarter) como benchmark")
    quarter_median = df.groupby(forward_quarters)["forward_return"].transform("median")
    return (df["forward_return"] > quarter_median).astype(int)


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
    days_before_quarter_start: int = 0,
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

    dates = df.index.get_level_values("date")

    def _has_price_coverage(entry_date: pd.Timestamp, exit_date: pd.Timestamp) -> bool:
        bench_window = benchmark.loc[entry_date:exit_date].dropna()
        if len(bench_window) < 2:
            return False
        for prices in prices_dict.values():
            if prices is None or prices.empty:
                continue
            cc = "Close" if "Close" in prices.columns else prices.columns[3]
            period = prices.loc[entry_date:exit_date, cc]
            if len(period) >= 2:
                return True
        return False

    for fold_id, (train_start, train_end, test_end, _train_years) in enumerate(folds, 1):
        # Identificar el quarter de test: es el quarter siguiente al de train_end
        test_quarter_ts = train_end + pd.offsets.QuarterEnd(1)
        test_quarter_label = f"{test_quarter_ts.year}Q{test_quarter_ts.quarter}"
        log.info(f"\n{'='*60}")
        log.info(f"  FOLD {fold_id}  —  Predicción para {test_quarter_label}")
        log.info(f"  Train : {train_start.date()} → {train_end.date()}  ({_train_years} años)")
        log.info(f"  Test  : {train_end.date()} → {test_end.date()}  (1 quarter)")
        log.info(f"{'='*60}")

        offset = pd.Timedelta(days=days_before_quarter_start)
        entry_date = train_end - offset
        exit_date = test_end - offset
        if not _has_price_coverage(entry_date, exit_date):
            log.warning(
                f"[Fold {fold_id}] Sin precios suficientes despues de {entry_date.date()} "
                "— fold omitido (sin entrenamiento)"
            )
            continue

        df_train, df_test = _prepare_fold_frames(
            df=df,
            dates=dates,
            train_start=train_start,
            train_end=train_end,
            test_end=test_end,
        )

        if len(df_train) < 100:
            log.warning(f"[Fold {fold_id}] Train insuficiente ({len(df_train)} observaciones, mínimo 100) — fold omitido.")
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
        if df_test_norm.empty or y_test.empty:
            log.warning(f"[Fold {fold_id}] Test vacio tras preparar labels — fold omitido.")
            continue

        try:
            agents, df_test_scored, df_train_with_oof = train_fold(
                df_train_norm=df_train_norm,
                df_test_norm=df_test_norm,
                y_train=y_train,
                y_test=y_test,
                fold_id=fold_id,
                agents_results_dir=agents_results_dir,
                random_seed=random_seed,
                sector_map=sector_map,
                spy_prices=spy_prices,
            )

            meta = agents["meta_learner"]
            eval_metrics = meta.evaluate(df_test_scored, y_test, fold=fold_id)

            visualizer.plot_score_distribution(df_test_scored, fold=fold_id)

            preds_df = df_test_scored[["final_score", "label"]].copy()
            preds_df["ticker"] = preds_df.index.get_level_values("ticker")
            preds_df["date"] = preds_df.index.get_level_values("date")
            preds_df = preds_df.reset_index(drop=True)

            fold_result = backtester.simulate_portfolio(
                predictions_df=preds_df.rename(columns={"final_score": "score"}),
                prices_dict=prices_dict,
                benchmark=benchmark,
                fold_id=fold_id,
                test_start=entry_date,
                test_end=exit_date,
                train_start=train_start,
                train_years_int=_train_years,
            )
            if not fold_result:
                log.warning(f"[Fold {fold_id}] Sin retornos suficientes — fold omitido tras simulacion")
                continue
            fold_result.update(eval_metrics)
            backtester.fold_results.append(fold_result)

            visualizer.plot_fold_performance(fold_result, fold_id=fold_id)

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
            export_selection_audit(audit_df, agents_results_dir, fold_id=fold_id, prefix="fold")

            # CSV de scores con explicaciones legibles por agente
            ticker_returns = fold_result.get("ticker_returns", {})
            bench_ret = fold_result.get("benchmark_cumulative_return")

            fold_scores_df = build_fold_scores_df(
                df_test_scored=df_test_scored,
                y_test=y_test,
                fold_id=fold_id,
                year_quarter=test_quarter_label,
                agents=agents,
                audit_df=audit_df,
                actual_returns=ticker_returns if ticker_returns else None,
                benchmark_return=bench_ret,
                ticker_weights=fold_result.get("ticker_weights"),
            )
            export_fold_scores(fold_scores_df, agents_results_dir, fold_id=fold_id)

            for ag_name, ag in agents.items():
                agent_diag_history[ag_name].append(ag._diagnostics.copy())

            for ag_name in ["fundamental", "valuation", "momentum", "bear", "sector_rotation"]:
                ag = agents[ag_name]
                if hasattr(ag, "_feature_cols") and ag.is_trained:
                    try:
                        imp_path = (
                            Path(agents_results_dir) / ag_name
                            / f"feature_importances_fold{fold_id}.csv"
                        )
                        if imp_path.exists():
                            imp = pd.read_csv(imp_path, index_col=0)["importance"]
                            visualizer.plot_feature_importances(imp, ag_name, fold=fold_id)
                    except Exception as plot_exc:
                        log.debug(f"[Visualizer] plot_feature_importances {ag_name} fold {fold_id}: {plot_exc}")

            explain_top_tickers(
                agents=agents,
                df_test=df_test_scored,
                scores=df_test_scored["final_score"],
                fold_id=fold_id,
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
                    fold_id=fold_id,
                    random_seed=random_seed,
                )
                if abl:
                    ablation_results.append(abl)

        except Exception as e:
            log.error(f"Fold {fold_id} fallo: {e}", exc_info=True)
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
