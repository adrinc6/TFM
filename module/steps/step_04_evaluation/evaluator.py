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


def _excess_return_label(df: pd.DataFrame, spy_prices: Optional[pd.Series] = None) -> pd.Series:
    """
    Label de outperformance: 1 si el ticker superó al SPY en ese mismo quarter.

    Si spy_prices no está disponible (None o vacío), cae a comparar contra la
    media del universo como comportamiento de seguridad.

    El retorno del SPY se calcula como retorno precio-a-precio entre el último día
    del quarter anterior y el último día del quarter de la observación, usando la
    misma ventana temporal que el forward_return del ticker.
    """
    dates = df.index.get_level_values("date")

    if spy_prices is not None and not spy_prices.empty:
        spy_ret_by_q = _spy_quarterly_returns(spy_prices)
        quarters = dates.to_period("Q").astype(str)
        spy_threshold = pd.Series(quarters, index=df.index).map(spy_ret_by_q)
        n_matched = spy_threshold.notna().sum()
        if n_matched > 0:
            log.debug(f"[Label] SPY threshold disponible para {n_matched}/{len(df)} observaciones")
            return (df["forward_return"] > spy_threshold).astype(int)
        log.warning("[Label] SPY sin cobertura para este período — usando media del universo como fallback")

    # Fallback: media del universo por quarter
    quarter_mean = df.groupby(dates.to_period("Q"))["forward_return"].transform("mean")
    return (df["forward_return"] > quarter_mean).astype(int)


def _prepare_fold_labels(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_train_norm: pd.DataFrame,
    df_test_norm: pd.DataFrame,
    spy_prices: Optional[pd.Series] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Label: outperformance relativa al SPY en cada quarter.
    # y=1 si el forward_return del ticker supera el retorno del SPY en ese mismo período.
    forward_train = _excess_return_label(df_train, spy_prices).reindex(df_train_norm.index)
    forward_test  = _excess_return_label(df_test,  spy_prices).reindex(df_test_norm.index)
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
        "fundamental": [], "valuation": [], "momentum": [], "bear": [], "sentiment": [], "meta_learner": []
    }
    ablation_results: List[Dict] = []

    dates = df.index.get_level_values("date")

    for fold_id, (train_start, train_end, test_end, _train_years) in enumerate(folds, 1):
        # Identificar el quarter de test: es el quarter siguiente al de train_end
        test_quarter_ts = train_end + pd.offsets.QuarterEnd(1)
        test_quarter_label = f"{test_quarter_ts.year}Q{test_quarter_ts.quarter}"
        log.info(f"\n{'='*60}")
        log.info(
            f"  FOLD {fold_id} [{test_quarter_label}] | "
            f"Train: {train_start.date()} -> {train_end.date()} "
            f"| Test: {train_end.date()} -> {test_end.date()}"
        )
        log.info(f"{'='*60}")

        df_train, df_test = _prepare_fold_frames(
            df=df,
            dates=dates,
            train_start=train_start,
            train_end=train_end,
            test_end=test_end,
        )

        if len(df_train) < 100:
            log.warning(f"Fold {fold_id}: Train insuficiente ({len(df_train)} obs). Skip.")
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
        )

        try:
            agents, df_test_scored, df_train_with_oof = train_fold(
                df_train_norm=df_train_norm,
                df_test_norm=df_test_norm,
                y_train=y_train,
                y_test=y_test,
                fold_id=fold_id,
                agents_results_dir=agents_results_dir,
                random_seed=random_seed,
            )

            meta = agents["meta_learner"]
            eval_metrics = meta.evaluate(df_test_scored, y_test, fold=fold_id)

            visualizer.plot_score_distribution(df_test_scored, fold=fold_id)

            preds_df = df_test_scored[["final_score", "label"]].copy()
            preds_df["ticker"] = preds_df.index.get_level_values("ticker")
            preds_df["date"] = preds_df.index.get_level_values("date")
            preds_df = preds_df.reset_index(drop=True)

            offset = pd.Timedelta(days=days_before_quarter_start)
            entry_date = train_end - offset
            exit_date  = test_end  - offset
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
            fold_result.update(eval_metrics)
            backtester.fold_results.append(fold_result)

            visualizer.plot_fold_performance(fold_result, fold_id=fold_id)

            audit_df = build_selection_audit_df(
                df_scored=df_test_scored.reset_index()[
                    ["ticker", "final_score", "label"]
                    + [c for c in ["fundamental_score", "valuation_score", "momentum_score", "bear_score", "sentiment_score"]
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

            for ag_name in ["fundamental", "valuation", "momentum", "bear"]:
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
