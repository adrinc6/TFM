"""Ablation study utilities for meta-learner."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ABLATION_AGENTS = ["fundamental", "valuation", "momentum", "bear"]


def run_ablation_study(
    df_test_scored: pd.DataFrame,
    y_test: pd.Series,
    df_train_norm: pd.DataFrame,
    y_train: pd.Series,
    agents_results_dir: str,
    fold_id: int,
    random_seed: int = 42,
) -> Dict:
    try:
        from sklearn.metrics import roc_auc_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        log.error("[Ablation] scikit-learn no disponible.")
        return {}

    score_cols = [f"{ag}_score" for ag in ABLATION_AGENTS]
    available_score_cols = [c for c in score_cols if c in df_test_scored.columns]

    if len(available_score_cols) < 2:
        log.warning(f"[Ablation] Fold {fold_id}: pocas columnas de score disponibles — omitido.")
        return {}

    X_train = df_train_norm[[c for c in available_score_cols if c in df_train_norm.columns]].copy()
    X_test = df_test_scored[[c for c in available_score_cols if c in df_test_scored.columns]].copy()

    y_tr = y_train.reindex(X_train.index).dropna()
    X_train = X_train.loc[y_tr.index].fillna(0.5)
    y_te = y_test.reindex(X_test.index).dropna()
    X_test = X_test.loc[y_te.index].fillna(0.5)

    if len(y_tr) < 20 or len(y_te) < 5:
        log.warning(f"[Ablation] Fold {fold_id}: insufficient data for ablation.")
        return {}

    def _auc(X_tr, y_tr, X_te, y_te):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.5, class_weight="balanced", max_iter=500,
                random_state=random_seed, solver="lbfgs",
            )),
        ])
        pipe.fit(X_tr, y_tr)
        if y_te.nunique() < 2:
            return float("nan")
        p = pipe.predict_proba(X_te)[:, 1]
        return float(roc_auc_score(y_te, p))

    baseline_auc = _auc(X_train, y_tr, X_test, y_te)

    ablation_results = {}
    for ag in ABLATION_AGENTS:
        col = f"{ag}_score"
        if col not in X_train.columns:
            continue
        X_tr_ab = X_train.drop(columns=[col])
        X_te_ab = X_test.drop(columns=[col])
        if X_tr_ab.shape[1] == 0:
            continue
        auc_ab = _auc(X_tr_ab, y_tr, X_te_ab, y_te)
        contribution = (baseline_auc - auc_ab) if not np.isnan(auc_ab) else float("nan")
        ablation_results[ag] = {
            "auc_without": round(auc_ab, 4),
            "auc_baseline": round(baseline_auc, 4),
            "marginal_contribution": round(contribution, 4),
        }
        log.info(
            f"[Ablation] Fold {fold_id} | sin {ag:<12} "
            f"AUC={auc_ab:.4f} (baseline={baseline_auc:.4f}, d={contribution:+.4f})"
        )

    result = {
        "fold": fold_id,
        "baseline_auc": round(baseline_auc, 4),
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "agents": ablation_results,
    }

    out_path = Path(agents_results_dir) / f"ablation_fold{fold_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log.info(f"[Ablation] Fold {fold_id} -> {out_path.name}")

    return result


def summarize_ablation(
    ablation_results: List[Dict],
    agents_results_dir: str,
) -> None:
    if not ablation_results:
        return

    per_agent: Dict[str, List[float]] = {ag: [] for ag in ABLATION_AGENTS}
    baseline_aucs: List[float] = []

    for res in ablation_results:
        if not res:
            continue
        baseline_aucs.append(res.get("baseline_auc", float("nan")))
        for ag, stats in res.get("agents", {}).items():
            contrib = stats.get("marginal_contribution", float("nan"))
            if not np.isnan(contrib):
                per_agent[ag].append(contrib)

    summary = {
        "n_folds": len(ablation_results),
        "mean_baseline_auc": float(np.nanmean(baseline_aucs)) if baseline_aucs else float("nan"),
        "agents": {},
    }

    rows = []
    for ag in ABLATION_AGENTS:
        contribs = per_agent[ag]
        if not contribs:
            continue
        mean_c = float(np.mean(contribs))
        std_c = float(np.std(contribs))
        n_pos = int(sum(1 for c in contribs if c > 0))
        summary["agents"][ag] = {
            "mean_contribution": round(mean_c, 4),
            "std_contribution": round(std_c, 4),
            "pct_folds_positive": round(n_pos / len(contribs), 3),
            "n_folds": len(contribs),
        }
        rows.append({
            "agent": ag,
            "mean_contribution": mean_c,
            "std_contribution": std_c,
            "pct_folds_positive": n_pos / len(contribs),
            "n_folds": len(contribs),
        })

    out_dir = Path(agents_results_dir)
    with open(out_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if rows:
        df = pd.DataFrame(rows).sort_values("mean_contribution", ascending=False)
        df.to_csv(out_dir / "ablation_summary.csv", index=False, float_format="%.4f")
        log.info("[Ablation] Ablation summary:")
        log.info(f"  Mean baseline AUC: {summary['mean_baseline_auc']:.4f}")
        for _, row in df.iterrows():
            log.info(
                f"  {row['agent']:<15}  dAuC medio={row['mean_contribution']:+.4f} "
                f"± {row['std_contribution']:.4f}  "
                f"(positivo en {row['pct_folds_positive']:.0%} de folds)"
            )

    log.info(f"[Ablation] Resumen guardado en {out_dir}/ablation_summary.{{json,csv}}")
