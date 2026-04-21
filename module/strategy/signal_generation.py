"""Take-Profit / Stop-Loss signal generation from multi-agent scores.

Each agent contributes a score in [0, 1] where 1 = strongly bullish.
These scores are translated into:
  - tp_pct : expected upside percentage (take-profit level)
  - sl_pct : expected downside risk percentage (stop-loss level)

Design rationale
----------------
* Higher aggregate score  → agents are confident the stock will rise
  → we can afford a more ambitious TP while the risk of hitting SL is lower.
* Lower aggregate score   → agents are uncertain or bearish
  → we widen the SL to give the position time to recover, but TP shrinks.

The mapping is intentionally simple and interpretable:

    tp_pct = BASE_TP  + TP_SENSITIVITY  * (score - 0.5)
    sl_pct = BASE_SL  - SL_SENSITIVITY  * (score - 0.5)

Both values are clipped to [MIN_TP, MAX_TP] and [MIN_SL, MAX_SL] respectively.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default hyper-parameters (all overridable via kwargs)
# ---------------------------------------------------------------------------
_DEFAULTS: Dict[str, float] = {
    "base_tp": 0.08,   # 8 % base take-profit
    "base_sl": 0.05,   # 5 % base stop-loss
    "tp_sensitivity": 0.10,  # max TP shift ± from base
    "sl_sensitivity": 0.04,  # max SL shift ± from base
    "min_tp": 0.02,
    "max_tp": 0.25,
    "min_sl": 0.01,
    "max_sl": 0.15,
}


def compute_tp_sl(
    scores: pd.Series,
    *,
    base_tp: float = _DEFAULTS["base_tp"],
    base_sl: float = _DEFAULTS["base_sl"],
    tp_sensitivity: float = _DEFAULTS["tp_sensitivity"],
    sl_sensitivity: float = _DEFAULTS["sl_sensitivity"],
    min_tp: float = _DEFAULTS["min_tp"],
    max_tp: float = _DEFAULTS["max_tp"],
    min_sl: float = _DEFAULTS["min_sl"],
    max_sl: float = _DEFAULTS["max_sl"],
) -> pd.DataFrame:
    """Map aggregate agent scores to TP/SL percentages.

    Parameters
    ----------
    scores:
        Series indexed by ticker with values in [0, 1].
    base_tp, base_sl:
        Baseline TP/SL percentages (applied when score == 0.5).
    tp_sensitivity, sl_sensitivity:
        Maximum shift from the baseline as score moves from 0.5 to 1.0.
    min_tp, max_tp, min_sl, max_sl:
        Hard bounds on the output percentages.

    Returns
    -------
    pd.DataFrame with columns ``ticker``, ``score``, ``tp_pct``, ``sl_pct``.
    """
    if scores is None or scores.empty:
        return pd.DataFrame(columns=["ticker", "score", "tp_pct", "sl_pct"])

    s = pd.to_numeric(scores, errors="coerce").fillna(0.5).clip(0.0, 1.0)

    deviation = s - 0.5  # range [-0.5, +0.5]

    tp = (base_tp + tp_sensitivity * deviation).clip(min_tp, max_tp)
    sl = (base_sl - sl_sensitivity * deviation).clip(min_sl, max_sl)

    result = pd.DataFrame({
        "ticker": s.index,
        "score": s.values,
        "tp_pct": tp.values,
        "sl_pct": sl.values,
    })
    return result.reset_index(drop=True)


def add_price_levels(
    signals: pd.DataFrame,
    entry_prices: pd.Series,
) -> pd.DataFrame:
    """Augment a signals DataFrame with absolute TP/SL price levels.

    Parameters
    ----------
    signals:
        Output of :func:`compute_tp_sl` (must have columns ticker, tp_pct, sl_pct).
    entry_prices:
        Series indexed by ticker with the entry (current) price.

    Returns
    -------
    signals with additional columns ``entry_price``, ``tp_price``, ``sl_price``.
    """
    out = signals.copy()
    out["entry_price"] = out["ticker"].map(entry_prices)
    ep = pd.to_numeric(out["entry_price"], errors="coerce")
    out["tp_price"] = ep * (1.0 + pd.to_numeric(out["tp_pct"], errors="coerce"))
    out["sl_price"] = ep * (1.0 - pd.to_numeric(out["sl_pct"], errors="coerce"))
    return out


def build_signals(
    agent_scores_df: pd.DataFrame,
    *,
    agent_weights: Optional[Dict[str, float]] = None,
    entry_prices: Optional[pd.Series] = None,
    **tp_sl_kwargs,
) -> pd.DataFrame:
    """End-to-end signal generation from per-agent score columns.

    Parameters
    ----------
    agent_scores_df:
        DataFrame with a ``ticker`` column and one column per agent
        (e.g. ``fundamental_score``, ``momentum_score``, …).
    agent_weights:
        Optional mapping of agent_name → weight.  Weights are normalised
        internally.  Unweighted (equal) average is used when ``None``.
    entry_prices:
        Series indexed by ticker; when provided, absolute TP/SL price
        levels are appended to the result.
    **tp_sl_kwargs:
        Forwarded to :func:`compute_tp_sl`.

    Returns
    -------
    DataFrame with columns: ticker, score, tp_pct, sl_pct
    (and optionally entry_price, tp_price, sl_price).
    """
    df = agent_scores_df.copy()
    score_cols = [c for c in df.columns if c.endswith("_score")]

    if not score_cols:
        # Fall back to a single 'score' column if present
        if "score" in df.columns:
            score_cols = ["score"]
        else:
            raise ValueError(
                "agent_scores_df must contain at least one column ending in '_score' "
                "or a column named 'score'."
            )

    if agent_weights:
        # Normalise provided weights to sum to 1
        w = {k: float(v) for k, v in agent_weights.items() if k in score_cols}
        total_w = sum(w.values()) or 1.0
        w = {k: v / total_w for k, v in w.items()}
        weighted = sum(
            df[col].fillna(0.5) * w.get(col, 0.0) for col in score_cols
        )
        unweighted_cols = [c for c in score_cols if c not in w]
        if unweighted_cols:
            remaining_w = 1.0 - sum(w.values())
            eq = remaining_w / len(unweighted_cols) if unweighted_cols else 0.0
            weighted += sum(df[col].fillna(0.5) * eq for col in unweighted_cols)
        agg_score = weighted
    else:
        agg_score = df[score_cols].apply(pd.to_numeric, errors="coerce").fillna(0.5).mean(axis=1)

    agg_score.index = df["ticker"].values if "ticker" in df.columns else df.index
    signals = compute_tp_sl(agg_score, **tp_sl_kwargs)

    # Carry forward any per-agent score columns for transparency
    if "ticker" in df.columns:
        for col in score_cols:
            merge_src = df.set_index("ticker")[col]
            signals[col] = signals["ticker"].map(merge_src)

    if entry_prices is not None:
        signals = add_price_levels(signals, entry_prices)

    return signals
