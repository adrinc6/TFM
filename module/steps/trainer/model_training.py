from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from module.agents import BaseAgent, build_agents
from module.common.meta_model import MetaModel
from module.common.utils import split_train_validation_by_time


@dataclass
class FoldTrainingResult:
    fold_number: int
    predictions: pd.DataFrame
    model_performance: pd.DataFrame


def train_fold_models(
    fold_number: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_col: str,
) -> FoldTrainingResult:
    train_split, val_split = split_train_validation_by_time(train_df.reset_index())
    train_split = train_split.set_index(["ticker", "date"]).sort_index()
    val_split = val_split.set_index(["ticker", "date"]).sort_index()

    if train_split.empty or val_split.empty:
        split_at = max(int(len(train_df) * 0.8), 1)
        train_split = train_df.iloc[:split_at].copy()
        val_split = train_df.iloc[split_at:].copy()

    y_train = pd.to_numeric(train_split[y_col], errors="coerce").fillna(0).astype(int)
    y_val = pd.to_numeric(val_split[y_col], errors="coerce").fillna(0).astype(int)

    x_train = train_split.drop(columns=[c for c in train_split.columns if c.startswith("label_")], errors="ignore")
    x_val = val_split.drop(columns=[c for c in val_split.columns if c.startswith("label_")], errors="ignore")
    x_test = test_df.drop(columns=[c for c in test_df.columns if c.startswith("label_")], errors="ignore")

    agents: list[BaseAgent] = build_agents()
    pred_test = pd.DataFrame(index=test_df.index)
    pred_val = pd.DataFrame(index=val_split.index)
    perf_rows: list[dict] = []

    for agent in agents:
        agent.fit(x_train, y_train, x_val, y_val)
        pred_test[f"{agent.name}_score"] = agent.predict_proba(x_test)
        pred_val[f"{agent.name}_score"] = agent.predict_proba(x_val)

        row = {
            "fold": fold_number,
            "model": agent.name,
            "best_model": agent.performance.best_model,
            "selected_features": ",".join(agent.selected_features),
            "selected_feature_count": len(agent.selected_features),
        }
        for model_name, score in agent.performance.model_scores.items():
            row[f"auc_{model_name}"] = float(score)
        perf_rows.append(row)

    y_val_meta = pd.to_numeric(val_split[y_col], errors="coerce").fillna(0).astype(int)
    meta_train_df, meta_val_df = split_train_validation_by_time(pred_val.reset_index())
    meta_train_df = meta_train_df.set_index(["ticker", "date"]).sort_index()
    meta_val_df = meta_val_df.set_index(["ticker", "date"]).sort_index()

    if meta_train_df.empty or meta_val_df.empty:
        meta_train_df = pred_val.copy()
        meta_val_df = pred_val.copy()

    y_meta_train = y_val_meta.loc[meta_train_df.index]
    y_meta_val = y_val_meta.loc[meta_val_df.index]

    meta_model = MetaModel()
    meta_model.fit(meta_train_df, y_meta_train, meta_val_df, y_meta_val)
    pred_test["meta_score"] = meta_model.predict_proba(pred_test)

    perf_rows.append(
        {
            "fold": fold_number,
            "model": "meta_model",
            "best_model": meta_model.performance.best_model,
            "selected_features": ",".join(meta_model.features),
            "selected_feature_count": len(meta_model.features),
            **{f"auc_{k}": float(v) for k, v in meta_model.performance.scores.items()},
            "validation_auc": float(meta_model.performance.validation_auc),
        }
    )

    predictions = test_df[["snapshot_date", "year_quarter", "sector"]].join(pred_test, how="left")
    predictions["label"] = pd.to_numeric(test_df[y_col], errors="coerce").fillna(0).astype(int)
    predictions["fold"] = int(fold_number)

    return FoldTrainingResult(
        fold_number=fold_number,
        predictions=predictions,
        model_performance=pd.DataFrame(perf_rows),
    )
