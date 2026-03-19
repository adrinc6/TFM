"""Walk-forward evaluation orchestration."""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from environment import FORWARD_RETURN_DAYS, SECTOR_ZSCORE_MIN_PEERS
from module.steps.step_02_dataset.builders.sector import SectorNormalizer
from module.steps.step_02_dataset.normalization import apply_sector_normalization
from module.steps.step_03_training.training import train_fold
from module.steps.step_04_evaluation.ablation import run_ablation_study, summarize_ablation
from module.steps.step_04_evaluation.backtester import WalkForwardBacktester
from module.steps.step_04_evaluation.reports import generate_text_report
from module.steps.step_04_evaluation.visualization import Visualizer

log = logging.getLogger(__name__)


def explain_top_tickers(
    agents: Dict,
    df_test: pd.DataFrame,
    scores: pd.Series,
    fold_id: int,
    agents_results_dir: str,
    top_n: int = 10,
) -> None:
    if scores.empty:
        return

    tickers_col = df_test.index.get_level_values("ticker")
    ticker_scores = pd.Series(scores.values, index=tickers_col).groupby(level=0).last()

    top_bulls = ticker_scores.nlargest(top_n).index.tolist()
    top_bears = ticker_scores.nsmallest(top_n).index.tolist()
    selected = list(dict.fromkeys(top_bulls + top_bears))

    results_path = Path(agents_results_dir) / f"fold_{fold_id}_ticker_explanations.json"
    all_explanations = {"fold": fold_id, "tickers": {}}

    for ticker in selected:
        mask = tickers_col == ticker
        if not mask.any():
            continue
        row = df_test.loc[mask].iloc[-1]
        score = float(ticker_scores.get(ticker, 0.5))

        ticker_exp = {
            "score": round(score, 4),
            "label": "Outperform" if score >= 0.5 else "Underperform",
            "agents": {},
        }

        for ag_name, ag in agents.items():
            if ag._explainer is None:
                continue
            try:
                exp = ag._explainer.explain_prediction(row, ticker, score, top_n=6, fold=fold_id)
                ticker_exp["agents"][ag_name] = {
                    "text": exp.get("text", ""),
                    "top_drivers": exp.get("top_drivers", [])[:6],
                }
            except Exception as ex:
                log.debug(f"Explain {ag_name}/{ticker}: {ex}")

        all_explanations["tickers"][ticker] = ticker_exp
        log.info(
            f"  [{'Outperform' if score >= 0.5 else 'Underperform':11s}] "
            f"{ticker:<6} score={score:.3f}"
        )

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_explanations, f, indent=2, ensure_ascii=False, default=str)

    log.info(
        f"[Explainer] Fold {fold_id}: explicaciones de {len(selected)} tickers "
        f"-> {results_path.name}"
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


def _prepare_fold_labels(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_train_norm: pd.DataFrame,
    df_test_norm: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    median_ret = df_train["forward_return"].median()
    forward_train = df_train["forward_return"].reindex(df_train_norm.index)
    forward_test = df_test["forward_return"].reindex(df_test_norm.index)
    y_train = (forward_train > median_ret).astype(int).dropna()
    y_test = (forward_test > median_ret).astype(int).dropna()

    df_train_norm = df_train_norm.loc[y_train.index]
    df_test_norm = df_test_norm.loc[y_test.index]
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

    folds = backtester.generate_folds(start_date, end_date, forward_return_days=FORWARD_RETURN_DAYS)
    agent_diag_history: Dict[str, List] = {
        "fundamental": [], "valuation": [], "momentum": [], "bear": [], "sentiment": [], "meta_learner": []
    }
    ablation_results: List[Dict] = []

    dates = df.index.get_level_values("date")

    for fold_id, (train_start, train_end, test_end, _train_years) in enumerate(folds, 1):
        log.info(f"\n{'='*60}")
        log.info(
            f"  FOLD {fold_id} | Train: {train_start.date()} -> {train_end.date()} "
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

            fold_result = backtester.simulate_portfolio(
                predictions_df=preds_df.rename(columns={"final_score": "score"}),
                prices_dict=prices_dict,
                benchmark=benchmark,
                fold_id=fold_id,
                test_start=train_end,
                test_end=test_end,
                train_start=train_start,
                train_years_int=_train_years,
            )
            fold_result.update(eval_metrics)
            backtester.fold_results.append(fold_result)

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
            )

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

    if ablation_results:
        summarize_ablation(ablation_results, agents_results_dir=agents_results_dir)

    generate_text_report(
        summary=summary,
        fold_results=backtester.fold_results,
        agent_diag_history=agent_diag_history,
        backtest_results_dir=backtest_results_dir,
    )

    return summary
