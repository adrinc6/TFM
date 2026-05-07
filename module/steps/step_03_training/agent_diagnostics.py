"""Fold-level diagnostics for agent redundancy, consistency, and rule quality.

These utilities are designed to support an iterative research loop:
- diagnose where agents overlap too much,
- enforce score diversity when overlap is high,
- quantify cross-agent contradiction,
- and audit out-of-fold rule quality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


def _numeric_scores(df: pd.DataFrame, score_cols: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in score_cols if c in df.columns]
    if not cols:
        return pd.DataFrame(index=df.index)
    out = df[cols].apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.5)


def compute_agent_redundancy(
    df: pd.DataFrame,
    score_cols: list[str],
    *,
    corr_floor: float = 0.35,
    corr_cap: float = 0.85,
    max_penalty: float = 0.35,
) -> Dict[str, Any]:
    """Estimate overlap between agents and derive diversity multipliers.

    Returns a dict with:
    - multipliers: per-score shrink factors in [0.5, 1.0]
    - matrix: absolute correlation matrix
    - summary: per-agent redundancy stats
    - high_corr_pairs: pairs above corr_cap
    """
    scores = _numeric_scores(df, score_cols)
    if scores.shape[1] == 0:
        return {
            "multipliers": {},
            "matrix": pd.DataFrame(),
            "summary": pd.DataFrame(),
            "high_corr_pairs": pd.DataFrame(),
        }

    if scores.shape[1] == 1:
        col = scores.columns[0]
        summary = pd.DataFrame(
            [{"score_col": col, "mean_abs_corr": 0.0, "multiplier": 1.0, "redundancy_index": 0.0}]
        )
        return {
            "multipliers": {col: 1.0},
            "matrix": pd.DataFrame([[1.0]], index=[col], columns=[col]),
            "summary": summary,
            "high_corr_pairs": pd.DataFrame(columns=["left", "right", "abs_corr"]),
        }

    corr = scores.corr(method="spearman").abs().fillna(0.0)
    multipliers: Dict[str, float] = {}
    rows: list[Dict[str, Any]] = []

    denom = max(float(corr_cap) - float(corr_floor), 1e-6)
    for col in corr.columns:
        others = [c for c in corr.columns if c != col]
        mean_abs_corr = float(corr.loc[col, others].mean()) if others else 0.0
        redundancy_index = float(np.clip((mean_abs_corr - float(corr_floor)) / denom, 0.0, 1.0))
        multiplier = float(np.clip(1.0 - float(max_penalty) * redundancy_index, 0.5, 1.0))
        multipliers[col] = multiplier
        rows.append(
            {
                "score_col": col,
                "mean_abs_corr": mean_abs_corr,
                "redundancy_index": redundancy_index,
                "multiplier": multiplier,
            }
        )

    pair_rows: list[Dict[str, Any]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = float(corr.loc[cols[i], cols[j]])
            if c >= float(corr_cap):
                pair_rows.append({"left": cols[i], "right": cols[j], "abs_corr": c})

    high_corr_pairs = pd.DataFrame(pair_rows)
    if high_corr_pairs.empty:
        high_corr_pairs = pd.DataFrame(columns=["left", "right", "abs_corr"])
    else:
        high_corr_pairs = high_corr_pairs.sort_values("abs_corr", ascending=False)

    return {
        "multipliers": multipliers,
        "matrix": corr,
        "summary": pd.DataFrame(rows).sort_values("mean_abs_corr", ascending=False),
        "high_corr_pairs": high_corr_pairs,
    }


def apply_diversity_multipliers(df: pd.DataFrame, multipliers: Dict[str, float]) -> pd.DataFrame:
    """Shrink overlapping agent scores toward neutral 0.5."""
    if df.empty or not multipliers:
        return df

    out = df.copy()
    for score_col, mult in multipliers.items():
        if score_col not in out.columns:
            continue
        m = float(np.clip(mult, 0.5, 1.0))
        s = pd.to_numeric(out[score_col], errors="coerce").fillna(0.5)
        out[score_col] = (0.5 + (s - 0.5) * m).clip(0.0, 1.0)
        out[f"{score_col}_diversity_multiplier"] = m
    return out


def compute_agent_behavior_features(
    df: pd.DataFrame,
    score_cols: list[str],
    *,
    bullish_threshold: float = 0.60,
    bearish_threshold: float = 0.40,
) -> pd.DataFrame:
    """Build cross-agent consistency features for the meta learner."""
    scores = _numeric_scores(df, score_cols)
    if scores.empty:
        return pd.DataFrame(index=df.index)

    out = pd.DataFrame(index=scores.index)
    out["agent_score_mean"] = scores.mean(axis=1)
    out["agent_score_std"] = scores.std(axis=1).fillna(0.0)

    arr = scores.to_numpy(dtype=float)
    n_agents = arr.shape[1]
    if n_agents >= 2:
        sum_abs = np.zeros(arr.shape[0], dtype=float)
        n_pairs = 0
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                sum_abs += np.abs(arr[:, i] - arr[:, j])
                n_pairs += 1
        out["agent_disagreement"] = np.clip(sum_abs / max(n_pairs, 1), 0.0, 1.0)
    else:
        out["agent_disagreement"] = 0.0

    bullish = (scores >= float(bullish_threshold)).sum(axis=1).astype(float)
    bearish = (scores <= float(bearish_threshold)).sum(axis=1).astype(float)
    out["bullish_agents"] = bullish
    out["bearish_agents"] = bearish
    out["agent_contradiction_flag"] = ((bullish > 0.0) & (bearish > 0.0)).astype(float)
    return out


def compute_rule_quality(df_scores: pd.DataFrame, y_true: pd.Series) -> pd.DataFrame:
    """Compute OOF-quality diagnostics for per-agent rule signals."""
    if df_scores.empty:
        return pd.DataFrame()

    rule_cols = [c for c in df_scores.columns if c.endswith("_rule_signal")]
    if not rule_cols:
        return pd.DataFrame()

    y = pd.to_numeric(y_true.reindex(df_scores.index), errors="coerce")
    rows: list[Dict[str, Any]] = []

    for col in rule_cols:
        s = pd.to_numeric(df_scores[col], errors="coerce")
        valid = y.notna() & s.notna()
        n = int(valid.sum())
        if n < 30:
            rows.append(
                {
                    "rule_col": col,
                    "n": n,
                    "spearman_ic": float("nan"),
                    "top_decile_hit_rate": float("nan"),
                    "bottom_decile_hit_rate": float("nan"),
                    "top_bottom_spread": float("nan"),
                    "quarter_ic_std": float("nan"),
                    "stability": float("nan"),
                }
            )
            continue

        yv = y[valid]
        sv = s[valid]
        ic = float(yv.corr(sv, method="spearman"))

        q_top = float(np.nanquantile(sv.to_numpy(dtype=float), 0.90))
        q_bot = float(np.nanquantile(sv.to_numpy(dtype=float), 0.10))
        top_hit = float(yv[sv >= q_top].mean()) if bool((sv >= q_top).any()) else float("nan")
        bot_hit = float(yv[sv <= q_bot].mean()) if bool((sv <= q_bot).any()) else float("nan")
        spread = float(top_hit - bot_hit) if np.isfinite(top_hit) and np.isfinite(bot_hit) else float("nan")

        q_std = float("nan")
        stability = float("nan")
        if isinstance(df_scores.index, pd.MultiIndex) and "date" in df_scores.index.names:
            q = pd.PeriodIndex(
                pd.to_datetime(df_scores.index.get_level_values("date"), errors="coerce"),
                freq="Q",
            ).astype(str)
            tmp = pd.DataFrame({"q": q, "y": y, "s": s}).dropna()
            quarter_ics: list[float] = []
            for _, grp in tmp.groupby("q", dropna=True):
                if len(grp) < 20:
                    continue
                ic_q = float(grp["y"].corr(grp["s"], method="spearman"))
                if np.isfinite(ic_q):
                    quarter_ics.append(ic_q)
            if quarter_ics:
                q_std = float(np.std(quarter_ics))
                stability = float(np.clip(1.0 - (q_std / 0.20), 0.0, 1.0))

        rows.append(
            {
                "rule_col": col,
                "n": n,
                "spearman_ic": ic,
                "top_decile_hit_rate": top_hit,
                "bottom_decile_hit_rate": bot_hit,
                "top_bottom_spread": spread,
                "quarter_ic_std": q_std,
                "stability": stability,
            }
        )

    return pd.DataFrame(rows).sort_values("rule_col").reset_index(drop=True)


def build_research_actions(
    *,
    redundancy: Dict[str, Any],
    rule_quality: pd.DataFrame,
    behavior_features: pd.DataFrame,
) -> list[Dict[str, Any]]:
    """Generate actionable recommendations for the next train-test-adjust cycle."""
    actions: list[Dict[str, Any]] = []

    pair_df = redundancy.get("high_corr_pairs", pd.DataFrame())
    if pair_df is not None and not pair_df.empty:
        top_pairs = pair_df.head(5)
        pairs_txt = [f"{r.left}~{r.right}:{r.abs_corr:.2f}" for r in top_pairs.itertuples(index=False)]
        actions.append(
            {
                "priority": "high",
                "type": "diversity",
                "issue": "high_agent_overlap",
                "details": "; ".join(pairs_txt),
                "recommendation": "Orthogonalize overlapping feature sets or tighten include/exclude lists for these agents.",
            }
        )

    if rule_quality is not None and not rule_quality.empty:
        weak = rule_quality[
            (pd.to_numeric(rule_quality["spearman_ic"], errors="coerce") < 0.01)
            | (pd.to_numeric(rule_quality["top_bottom_spread"], errors="coerce") <= 0.0)
        ]
        for row in weak.itertuples(index=False):
            actions.append(
                {
                    "priority": "medium",
                    "type": "rule_engine",
                    "issue": "weak_rule_signal",
                    "rule_col": str(row.rule_col),
                    "details": f"ic={float(row.spearman_ic):.4f}, spread={float(row.top_bottom_spread):.4f}",
                    "recommendation": "Increase min support/edge or trim unstable candidate features in this agent rulebook.",
                }
            )

    if behavior_features is not None and not behavior_features.empty:
        contradiction_rate = float(pd.to_numeric(behavior_features["agent_contradiction_flag"], errors="coerce").fillna(0.0).mean())
        avg_disagreement = float(pd.to_numeric(behavior_features["agent_disagreement"], errors="coerce").fillna(0.0).mean())
        if contradiction_rate > 0.20:
            actions.append(
                {
                    "priority": "medium",
                    "type": "consistency",
                    "issue": "frequent_agent_contradictions",
                    "details": f"contradiction_rate={contradiction_rate:.2%}, disagreement={avg_disagreement:.3f}",
                    "recommendation": "Audit conflicting agent regimes and add consistency-aware features/constraints in meta training.",
                }
            )

    if not actions:
        actions.append(
            {
                "priority": "low",
                "type": "status",
                "issue": "no_critical_diagnostics",
                "details": "No severe overlap or rule-quality issues detected in this fold.",
                "recommendation": "Proceed with current configuration and monitor drift in next fold.",
            }
        )
    return actions


def export_fold_diagnostics(
    *,
    output_dir: str,
    fold_id: int | str,
    redundancy: Dict[str, Any],
    rule_quality: pd.DataFrame,
    behavior_features: pd.DataFrame,
    actions: list[Dict[str, Any]],
) -> None:
    """Persist diagnostics artifacts for iterative model refinement."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"{fold_id}"

    summary = redundancy.get("summary", pd.DataFrame())
    matrix = redundancy.get("matrix", pd.DataFrame())
    pairs = redundancy.get("high_corr_pairs", pd.DataFrame())

    if summary is not None and not summary.empty:
        summary.to_csv(out / f"agent_redundancy_summary_{suffix}.csv", index=False, encoding="utf-8")
    if matrix is not None and not matrix.empty:
        matrix.to_csv(out / f"agent_redundancy_matrix_{suffix}.csv", encoding="utf-8")
    if pairs is not None and not pairs.empty:
        pairs.to_csv(out / f"agent_high_corr_pairs_{suffix}.csv", index=False, encoding="utf-8")
    if rule_quality is not None and not rule_quality.empty:
        rule_quality.to_csv(out / f"rule_quality_{suffix}.csv", index=False, encoding="utf-8")

    behavior_summary = {}
    if behavior_features is not None and not behavior_features.empty:
        for col in [
            "agent_score_mean",
            "agent_score_std",
            "agent_disagreement",
            "bullish_agents",
            "bearish_agents",
            "agent_contradiction_flag",
        ]:
            if col in behavior_features.columns:
                behavior_summary[col] = float(pd.to_numeric(behavior_features[col], errors="coerce").fillna(0.0).mean())

    payload = {
        "fold": str(fold_id),
        "behavior_summary": behavior_summary,
        "actions": actions,
    }
    with open(out / f"iterative_research_actions_{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
