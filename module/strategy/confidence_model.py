"""Confidence model: probability that TP is reached before SL.

The confidence score combines:
  1. The aggregate agent score (a model-based prior).
  2. A calibration factor derived from each agent's historical hit rate.

Formula
-------
    raw_confidence = weighted_mean(agent_scores)

    calibration    = weighted_mean(agent_hit_rates)   # historical accuracy

    confidence     = 0.5 * raw_confidence
                   + 0.5 * calibration

Both components are clipped to [0, 1] and the final score is clipped to
[MIN_CONFIDENCE, MAX_CONFIDENCE] to avoid degenerate cases.

When no historical performance data are available, calibration defaults to
0.5 (maximum uncertainty), so confidence degrades gracefully to the raw
model score.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


MIN_CONFIDENCE = 0.05
MAX_CONFIDENCE = 0.95


def _safe_mean(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").dropna()
    return float(v.mean()) if not v.empty else 0.5


def compute_confidence(
    agent_scores_df: pd.DataFrame,
    *,
    agent_weights: Optional[Dict[str, float]] = None,
    agent_hit_rates: Optional[Dict[str, float]] = None,
    score_weight: float = 0.5,
    calibration_weight: float = 0.5,
) -> pd.Series:
    """Compute per-ticker confidence scores.

    Parameters
    ----------
    agent_scores_df:
        DataFrame with a ``ticker`` column and one column per agent
        (e.g. ``fundamental_score``, ``momentum_score``, …).
        Scores must be in [0, 1].
    agent_weights:
        Optional mapping agent_col → weight.  Equal weights used when ``None``.
    agent_hit_rates:
        Optional mapping agent_col → historical TP-hit rate in [0, 1].
        Defaults to 0.5 for unknown agents (maximum uncertainty).
    score_weight:
        How much the raw model score contributes to the final confidence.
    calibration_weight:
        How much the historical calibration contributes to the final confidence.

    Returns
    -------
    pd.Series indexed by ticker with confidence values in
    [MIN_CONFIDENCE, MAX_CONFIDENCE].
    """
    df = agent_scores_df.copy()
    score_cols = [c for c in df.columns if c.endswith("_score")]
    if not score_cols and "score" in df.columns:
        score_cols = ["score"]
    if not score_cols:
        raise ValueError(
            "agent_scores_df must contain columns ending in '_score' or a 'score' column."
        )

    tickers = df["ticker"].values if "ticker" in df.columns else df.index.values

    # --- Raw aggregate score ------------------------------------------------
    if agent_weights:
        total_w = sum(float(agent_weights.get(c, 0.0)) for c in score_cols) or 1.0
        w_norm = {c: float(agent_weights.get(c, 0.0)) / total_w for c in score_cols}
        raw_score = sum(
            df[col].fillna(0.5) * w_norm.get(col, 0.0) for col in score_cols
        )
    else:
        raw_score = (
            df[score_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.5)
            .mean(axis=1)
        )

    # --- Historical calibration factor -------------------------------------
    hit_rates = agent_hit_rates or {}
    if agent_weights:
        total_hw = sum(float(agent_weights.get(c, 0.0)) for c in score_cols) or 1.0
        w_h = {c: float(agent_weights.get(c, 0.0)) / total_hw for c in score_cols}
        calibration = sum(
            float(hit_rates.get(col, 0.5)) * w_h.get(col, 0.0)
            for col in score_cols
        )
    else:
        calibration = np.mean(
            [float(hit_rates.get(col, 0.5)) for col in score_cols]
        )

    # --- Combine -----------------------------------------------------------
    sw = float(score_weight)
    cw = float(calibration_weight)
    total = sw + cw or 1.0
    sw /= total
    cw /= total

    combined = sw * np.array(raw_score, dtype=float) + cw * float(calibration)
    combined = np.clip(combined, MIN_CONFIDENCE, MAX_CONFIDENCE)

    return pd.Series(combined, index=tickers, name="confidence")


def attach_confidence(
    signals: pd.DataFrame,
    *,
    agent_weights: Optional[Dict[str, float]] = None,
    agent_hit_rates: Optional[Dict[str, float]] = None,
    score_weight: float = 0.5,
    calibration_weight: float = 0.5,
) -> pd.DataFrame:
    """Convenience wrapper: attach a ``confidence`` column to *signals*.

    Parameters
    ----------
    signals:
        Output of :func:`~module.strategy.signal_generation.build_signals`.
        Must contain a ``ticker`` column and agent score columns.
    agent_weights, agent_hit_rates, score_weight, calibration_weight:
        Forwarded to :func:`compute_confidence`.

    Returns
    -------
    *signals* with an additional ``confidence`` column.
    """
    conf = compute_confidence(
        signals,
        agent_weights=agent_weights,
        agent_hit_rates=agent_hit_rates,
        score_weight=score_weight,
        calibration_weight=calibration_weight,
    )
    out = signals.copy()
    # Map by ticker in case index differs
    if "ticker" in out.columns:
        out["confidence"] = out["ticker"].map(conf).fillna(0.5)
    else:
        out["confidence"] = conf.values
    return out
