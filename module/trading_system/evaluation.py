from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def evaluate_model_predictions(strategy_diagnostics: pd.DataFrame) -> pd.DataFrame:
    if strategy_diagnostics is None or strategy_diagnostics.empty:
        return pd.DataFrame(columns=["model", "strategy", "auc", "brier", "hit_rate", "expected_value", "avg_days_to_event", "n_obs"])

    model_cols = [c for c in strategy_diagnostics.columns if c.endswith("_score")]
    rows = []

    for strategy, g in strategy_diagnostics.groupby("strategy"):
        y = pd.to_numeric(g["label"], errors="coerce").fillna(0).astype(int)
        tp_pct = pd.to_numeric(g["tp_pct"], errors="coerce").fillna(0.0)
        sl_pct = pd.to_numeric(g["sl_pct"], errors="coerce").fillna(0.0)
        days = pd.to_numeric(g["days_to_event"], errors="coerce")

        for col in model_cols:
            pred = pd.to_numeric(g[col], errors="coerce").fillna(0.5).clip(0.0, 1.0)
            if y.nunique() > 1:
                auc = float(roc_auc_score(y, pred))
            else:
                auc = 0.5
            brier = float(brier_score_loss(y, pred))
            hit_rate = float(y.mean())
            expected_value = float((pred * tp_pct - (1.0 - pred) * sl_pct).mean())
            rows.append(
                {
                    "model": col,
                    "strategy": strategy,
                    "auc": auc,
                    "brier": brier,
                    "hit_rate": hit_rate,
                    "expected_value": expected_value,
                    "avg_days_to_event": float(days.mean()),
                    "n_obs": int(len(g)),
                }
            )

    return pd.DataFrame(rows).sort_values(["strategy", "expected_value"], ascending=[True, False]).reset_index(drop=True)


def choose_best_strategy_per_stock(strategy_diagnostics: pd.DataFrame, score_col: str = "meta_score") -> pd.DataFrame:
    if strategy_diagnostics is None or strategy_diagnostics.empty:
        return pd.DataFrame()

    df = strategy_diagnostics.copy()
    score = pd.to_numeric(df[score_col], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    tp = pd.to_numeric(df["tp_pct"], errors="coerce").fillna(0.0)
    sl = pd.to_numeric(df["sl_pct"], errors="coerce").fillna(0.0)
    df["expected_value_model"] = score * tp - (1.0 - score) * sl
    df["risk_reward"] = tp / sl.replace(0, np.nan)

    group_cols = ["ticker", "date"]
    idx = df.groupby(group_cols)["expected_value_model"].idxmax()
    best = df.loc[idx].copy()
    best = best.sort_values(["snapshot_date", "expected_value_model"], ascending=[True, False])
    return best
