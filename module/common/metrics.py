from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def hit_rate(y_true: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce").fillna(0.0)
    return float(y.mean()) if len(y) else np.nan


def expected_value_from_probability(prob: pd.Series, tp_pct: pd.Series, sl_pct: pd.Series) -> pd.Series:
    p = pd.to_numeric(prob, errors="coerce").fillna(0.5).clip(0.0, 1.0)
    tp = pd.to_numeric(tp_pct, errors="coerce").fillna(0.0)
    sl = pd.to_numeric(sl_pct, errors="coerce").fillna(0.0)
    return p * tp - (1.0 - p) * sl


def model_classification_metrics(y_true: pd.Series, prob: pd.Series) -> dict[str, float]:
    y = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)
    p = pd.to_numeric(prob, errors="coerce").fillna(0.5).clip(0.0, 1.0)
    auc = float(roc_auc_score(y, p)) if y.nunique() > 1 else 0.5
    brier = float(brier_score_loss(y, p))
    return {"auc": auc, "brier": brier, "hit_rate": float(y.mean())}


def summarize_strategy_metrics(df: pd.DataFrame, prob_col: str = "meta_score") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["strategy", "n_obs", "hit_rate", "expected_value", "avg_days_to_event"])

    rows = []
    for strategy, group in df.groupby("strategy"):
        y = pd.to_numeric(group["label"], errors="coerce").fillna(0)
        ev = expected_value_from_probability(group[prob_col], group["tp_pct"], group["sl_pct"])
        rows.append(
            {
                "strategy": strategy,
                "n_obs": int(len(group)),
                "hit_rate": float(y.mean()),
                "expected_value": float(ev.mean()),
                "avg_days_to_event": float(pd.to_numeric(group["days_to_event"], errors="coerce").mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("expected_value", ascending=False).reset_index(drop=True)
