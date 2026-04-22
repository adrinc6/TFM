from __future__ import annotations

import pandas as pd

from module.common.metrics import expected_value_from_probability, model_classification_metrics


def evaluate_per_model_per_strategy(strategy_df: pd.DataFrame) -> pd.DataFrame:
    if strategy_df is None or strategy_df.empty:
        return pd.DataFrame(columns=["strategy", "model", "auc", "brier", "hit_rate", "expected_value", "n_obs"])

    score_cols = [c for c in strategy_df.columns if c.endswith("_score")]
    rows: list[dict] = []

    for strategy, group in strategy_df.groupby("strategy"):
        for model_col in score_cols:
            metrics = model_classification_metrics(group["label"], group[model_col])
            ev = expected_value_from_probability(group[model_col], group["tp_pct"], group["sl_pct"])
            rows.append(
                {
                    "strategy": strategy,
                    "model": model_col,
                    "auc": float(metrics["auc"]),
                    "brier": float(metrics["brier"]),
                    "hit_rate": float(metrics["hit_rate"]),
                    "expected_value": float(ev.mean()),
                    "n_obs": int(len(group)),
                }
            )

    return pd.DataFrame(rows).sort_values(["strategy", "expected_value"], ascending=[True, False]).reset_index(drop=True)
