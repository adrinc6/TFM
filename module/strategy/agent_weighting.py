"""Dynamic agent weighting based on historical TP/SL hit accuracy.

Each agent receives a weight proportional to its empirical TP-hit rate
over previous folds.  Agents that correctly identify stocks that reach TP
get higher weights in the meta-ensemble.

Algorithm
---------
1. After each fold, record per-agent outcomes (TP hit or not) for their
   top-scored stocks.
2. Maintain an exponentially-weighted moving average (EWMA) of the hit rate.
3. Normalise to obtain a weight vector that sums to 1.

Persistence
-----------
History is stored as a JSON-serialisable list so it survives restarts when
saved to disk (e.g. via ``agent_weights.json``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_DEFAULT_DECAY = 0.85      # EWMA decay factor  (higher = slower to forget)
_DEFAULT_PRIOR = 0.50      # Prior hit-rate before any data (max uncertainty)
_MIN_WEIGHT = 0.05         # Floor weight to keep all agents active


class AgentWeightTracker:
    """Track per-agent TP-hit accuracy and compute dynamic weights.

    Parameters
    ----------
    agent_names:
        List of agent identifiers (e.g. ``["fundamental_score", ...]``).
    decay:
        EWMA decay factor in (0, 1).  Higher values retain older history
        longer (slower adaptation).
    prior_hit_rate:
        Initial hit-rate used before any fold data are available.
    min_weight:
        Minimum normalised weight assigned to any agent.
    """

    def __init__(
        self,
        agent_names: List[str],
        *,
        decay: float = _DEFAULT_DECAY,
        prior_hit_rate: float = _DEFAULT_PRIOR,
        min_weight: float = _MIN_WEIGHT,
    ) -> None:
        self.agent_names = list(agent_names)
        self.decay = float(decay)
        self.prior = float(prior_hit_rate)
        self.min_weight = float(min_weight)
        # EWMA hit-rate per agent
        self._ewma: Dict[str, float] = {a: self.prior for a in self.agent_names}
        # Full fold-level history for auditing
        self._history: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        fold_id: str,
        outcomes: pd.DataFrame,
        agent_scores_df: pd.DataFrame,
        top_n: int = 10,
    ) -> None:
        """Record outcomes for one fold and update EWMA weights.

        Parameters
        ----------
        fold_id:
            Identifier for the fold (e.g. ``"2023Q4"``).
        outcomes:
            DataFrame with columns ``ticker`` and ``outcome`` (TP/SL/NONE).
        agent_scores_df:
            DataFrame with ``ticker`` and per-agent score columns.
        top_n:
            Number of top-ranked stocks to attribute to each agent.
        """
        if outcomes.empty or agent_scores_df.empty:
            return

        outcome_map = dict(
            zip(
                outcomes["ticker"].astype(str),
                outcomes["outcome"].astype(str),
            )
        )

        score_cols = [c for c in agent_scores_df.columns if c.endswith("_score")]
        fold_record: Dict[str, object] = {"fold_id": fold_id, "agents": {}}

        for col in score_cols:
            if col not in self.agent_names:
                self.agent_names.append(col)
                self._ewma[col] = self.prior

            # Top-N tickers according to this agent
            src = agent_scores_df[["ticker", col]].copy()
            src[col] = pd.to_numeric(src[col], errors="coerce")
            top_tickers = (
                src.nlargest(top_n, col)["ticker"].astype(str).tolist()
            )
            tp_hits = sum(1 for t in top_tickers if outcome_map.get(t) == "TP")
            hit_rate = tp_hits / len(top_tickers) if top_tickers else self.prior

            # EWMA update
            self._ewma[col] = (
                self.decay * self._ewma[col] + (1.0 - self.decay) * hit_rate
            )

            fold_record["agents"][col] = {  # type: ignore[index]
                "top_tickers": top_tickers,
                "tp_hits": tp_hits,
                "hit_rate": round(hit_rate, 4),
                "ewma_hit_rate": round(self._ewma[col], 4),
            }

        self._history.append(fold_record)
        log.info(
            "[AgentWeightTracker] Fold %s updated. EWMA hit-rates: %s",
            fold_id,
            {k: round(v, 3) for k, v in self._ewma.items()},
        )

    def get_weights(self) -> Dict[str, float]:
        """Return normalised dynamic weights for all tracked agents.

        Returns
        -------
        Dict mapping agent_col → weight (sums to 1, each >= min_weight).
        """
        raw = {a: max(self._ewma.get(a, self.prior), self.min_weight)
               for a in self.agent_names}
        total = sum(raw.values()) or 1.0
        return {a: v / total for a, v in raw.items()}

    def get_hit_rates(self) -> Dict[str, float]:
        """Return the current EWMA hit-rate per agent."""
        return {a: round(self._ewma.get(a, self.prior), 4)
                for a in self.agent_names}

    def get_history(self) -> List[Dict]:
        """Return the full fold-level history list."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save tracker state to a JSON file."""
        state = {
            "agent_names": self.agent_names,
            "decay": self.decay,
            "prior": self.prior,
            "min_weight": self.min_weight,
            "ewma": self._ewma,
            "history": self._history,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, default=str)
        log.info("[AgentWeightTracker] State saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "AgentWeightTracker":
        """Restore a tracker from a JSON file previously created by :meth:`save`."""
        with open(path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        tracker = cls(
            agent_names=state.get("agent_names", []),
            decay=state.get("decay", _DEFAULT_DECAY),
            prior_hit_rate=state.get("prior", _DEFAULT_PRIOR),
            min_weight=state.get("min_weight", _MIN_WEIGHT),
        )
        tracker._ewma = state.get("ewma", {})
        tracker._history = state.get("history", [])
        log.info("[AgentWeightTracker] State loaded ← %s", path)
        return tracker
