from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from module.common.feature_engineering import generate_strategy_targets, primary_label_column, ratio_feature_candidates
from module.common.metrics import summarize_strategy_metrics
from module.common.utils import load_master_dataset, load_price_cache
from module.steps.trainer.model_training import FoldTrainingResult, train_fold_models
from module.steps.trainer.walk_forward import build_walk_forward_folds


@dataclass
class TrainingArtifacts:
    training_frame: pd.DataFrame
    strategy_truth: pd.DataFrame
    diagnostics: pd.DataFrame
    strategy_diagnostics: pd.DataFrame
    model_performance: pd.DataFrame
    strategy_performance: pd.DataFrame


def run_training_pipeline() -> TrainingArtifacts:
    master_df = load_master_dataset()
    price_cache = load_price_cache(master_df.index.get_level_values("ticker"))

    labeled_df, strategy_truth = generate_strategy_targets(master_df, price_cache)
    y_col = primary_label_column()
    if y_col not in labeled_df.columns:
        raise ValueError(f"Missing required target column: {y_col}")

    numeric_candidates = ratio_feature_candidates(labeled_df)
    base_cols = [
        "snapshot_date",
        "year_quarter",
        "sector",
        "industry",
        y_col,
    ]

    strategy_label_cols = [c for c in labeled_df.columns if c.startswith("label_")]
    strategy_meta_cols = [
        c
        for c in labeled_df.columns
        if c.startswith("outcome_")
        or c.startswith("days_to_event_")
        or c.startswith("entry_price_")
        or c.startswith("tp_level_")
        or c.startswith("sl_level_")
    ]

    use_cols = list(dict.fromkeys(base_cols + strategy_label_cols + strategy_meta_cols + numeric_candidates))
    train_frame = labeled_df[use_cols].copy()

    folds = build_walk_forward_folds(train_frame, min_train_periods=8)
    if not folds:
        raise RuntimeError("No valid walk-forward folds could be generated")

    diagnostics_parts: list[pd.DataFrame] = []
    strategy_parts: list[pd.DataFrame] = []
    perf_parts: list[pd.DataFrame] = []

    for fold_no, (train_idx, test_idx) in enumerate(folds, start=1):
        fold_train = train_frame.loc[train_idx].copy()
        fold_test = train_frame.loc[test_idx].copy()
        if fold_train.empty or fold_test.empty:
            continue

        result: FoldTrainingResult = train_fold_models(
            fold_number=fold_no,
            train_df=fold_train,
            test_df=fold_test,
            y_col=y_col,
        )

        diagnostics_parts.append(result.predictions)
        perf_parts.append(result.model_performance)

        fold_strategy = strategy_truth.loc[fold_test.index].copy()
        fold_strategy = fold_strategy.join(result.predictions.filter(regex="_score$|meta_score|fold"), how="left")
        strategy_parts.append(fold_strategy)

    diagnostics = pd.concat(diagnostics_parts, axis=0).sort_index() if diagnostics_parts else pd.DataFrame()
    strategy_diagnostics = pd.concat(strategy_parts, axis=0).sort_index() if strategy_parts else pd.DataFrame()
    model_performance = pd.concat(perf_parts, axis=0).reset_index(drop=True) if perf_parts else pd.DataFrame()
    strategy_performance = summarize_strategy_metrics(strategy_diagnostics, prob_col="meta_score")

    return TrainingArtifacts(
        training_frame=train_frame,
        strategy_truth=strategy_truth,
        diagnostics=diagnostics,
        strategy_diagnostics=strategy_diagnostics,
        model_performance=model_performance,
        strategy_performance=strategy_performance,
    )
