"""CSV reporter for the TP/SL + confidence strategy pipeline.

Generates a single, self-contained CSV that covers **all** stocks in the
universe (not only those selected for investment), including:

  * Per-agent TP/SL predictions and confidence scores
  * Portfolio selection flag (selected / not selected)
  * Backtest outcome (TP / SL / NONE) and days to outcome
  * Agent performance metrics (EWMA hit rate, dynamic weight)

The CSV is intended to serve as both the primary output of the strategy
and a debug / learning dataset for ongoing model improvement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


_COLUMN_ORDER = [
    # Identification
    "ticker",
    "fold_id",
    "entry_date",
    "sector",
    # Aggregate signal
    "score",
    "confidence",
    "ev",
    # TP/SL percentages
    "tp_pct",
    "sl_pct",
    # TP/SL absolute prices
    "entry_price",
    "tp_price",
    "sl_price",
    # Portfolio decision
    "selected",
    # Backtest outcome
    "outcome",
    "days_to_outcome",
]


def build_strategy_csv(
    signals: pd.DataFrame,
    *,
    fold_id: str = "",
    agent_weights: Optional[Dict[str, float]] = None,
    agent_hit_rates: Optional[Dict[str, float]] = None,
    extra_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Assemble the full strategy output DataFrame (all stocks).

    Parameters
    ----------
    signals:
        Output of the full strategy pipeline (signal_generation →
        confidence_model → portfolio_selection → backtesting_engine).
        Must contain ``ticker``.
    fold_id:
        Identifier for the current fold (e.g. ``"2023Q4"``).
    agent_weights:
        Mapping agent_col → dynamic weight.  Added as ``weight_<agent>``
        columns for the debug dataset.
    agent_hit_rates:
        Mapping agent_col → EWMA hit rate.  Added as ``hit_rate_<agent>``
        columns.
    extra_cols:
        Additional column names from *signals* to include verbatim.

    Returns
    -------
    pd.DataFrame ready to be written with :meth:`pandas.DataFrame.to_csv`.
    """
    df = signals.copy()

    # Ensure fold_id column
    df["fold_id"] = str(fold_id)

    # Agent weight & hit-rate columns (debug layer) — skip if already present
    if agent_weights:
        for col, w in agent_weights.items():
            dest = f"weight_{col}"
            if dest not in df.columns:
                df[dest] = round(float(w), 6)
    if agent_hit_rates:
        for col, hr in agent_hit_rates.items():
            dest = f"hit_rate_{col}"
            if dest not in df.columns:
                df[dest] = round(float(hr), 6)

    # Determine column order
    base_cols = [c for c in _COLUMN_ORDER if c in df.columns]

    # Agent score columns (e.g. fundamental_score, momentum_score, …)
    score_cols = sorted([c for c in df.columns if c.endswith("_score")])

    # Agent weight / hit-rate columns
    metric_cols = sorted(
        [c for c in df.columns if c.startswith("weight_") or c.startswith("hit_rate_")]
    )

    # Any user-supplied extra columns that aren't already included
    extra = [c for c in (extra_cols or []) if c in df.columns and c not in base_cols + score_cols + metric_cols]

    ordered_cols = base_cols + score_cols + metric_cols + extra
    # Add any remaining columns not explicitly ordered
    ordered_set = set(ordered_cols)
    remaining = [c for c in df.columns if c not in ordered_set]
    final_cols = ordered_cols + remaining

    # Deduplicate while preserving order (in case df.columns has dupes)
    seen: set = set()
    deduped = []
    for c in final_cols:
        if c not in seen:
            seen.add(c)
            deduped.append(c)

    return df[[c for c in deduped if c in df.columns]].reset_index(drop=True)


def export_strategy_csv(
    signals: pd.DataFrame,
    output_path: str | Path,
    *,
    fold_id: str = "",
    agent_weights: Optional[Dict[str, float]] = None,
    agent_hit_rates: Optional[Dict[str, float]] = None,
    extra_cols: Optional[List[str]] = None,
) -> Path:
    """Write the strategy output to a CSV file and return the path.

    Parameters
    ----------
    signals:
        Strategy pipeline output (all stocks).
    output_path:
        Destination file path (e.g. ``results/strategy/2023Q4_output.csv``).
    fold_id, agent_weights, agent_hit_rates, extra_cols:
        Forwarded to :func:`build_strategy_csv`.

    Returns
    -------
    Resolved :class:`pathlib.Path` of the written file.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_df = build_strategy_csv(
        signals,
        fold_id=fold_id,
        agent_weights=agent_weights,
        agent_hit_rates=agent_hit_rates,
        extra_cols=extra_cols,
    )
    csv_df.to_csv(out_path, index=False)
    return out_path
