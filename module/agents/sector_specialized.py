"""Sector-specialized wrapper that trains one independent model per sector."""

from __future__ import annotations

import inspect
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Type

import pandas as pd

from module.agents.base import BaseAgent

log = logging.getLogger(__name__)


class SectorSpecializedAgent(BaseAgent):
    """Wraps a base agent class and trains one model per sector.

    This class enforces a strict specialization policy:
      - no global model trained across all sectors,
      - one independent model per sector,
      - neutral fallback score for unseen or under-sampled sectors.
    """

    def __init__(
        self,
        name: str,
        agent_cls: Type[BaseAgent],
        results_dir: str,
        random_seed: int = 42,
        agent_kwargs: Optional[Dict[str, Any]] = None,
        min_samples_per_sector: int = 40,
        neutral_score: float = 0.5,
        save_artifacts: bool = True,
    ):
        super().__init__(name=name, results_dir=results_dir, random_seed=random_seed, save_artifacts=save_artifacts)
        self._agent_cls = agent_cls
        self._agent_kwargs = dict(agent_kwargs or {})
        self._min_samples_per_sector = max(int(min_samples_per_sector), 1)
        self._neutral_score = float(neutral_score)
        self._sector_agents: Dict[str, BaseAgent] = {}
        self._sector_sample_count: Dict[str, int] = {}
        self._feature_cols: list[str] = []

    @staticmethod
    def _sanitize_sector_name(sector: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(sector).strip())
        return safe or "Unknown"

    @staticmethod
    def _supports_arg(callable_obj: Any, arg_name: str) -> bool:
        try:
            sig = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return False
        return arg_name in sig.parameters

    def _instantiate_child_agent(self, sector: str) -> BaseAgent:
        sector_safe = self._sanitize_sector_name(sector)
        sector_results_dir = Path(self.results_dir) / "sectors" / sector_safe
        kwargs = dict(self._agent_kwargs)
        kwargs.setdefault("results_dir", sector_results_dir.as_posix())
        kwargs.setdefault("random_seed", self.random_seed)
        kwargs.setdefault("save_artifacts", self.save_artifacts)
        child = self._agent_cls(**kwargs)
        child.results_dir = sector_results_dir
        child.results_dir.mkdir(parents=True, exist_ok=True)
        return child

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fold: Optional[int | str] = None,
        sector_col: str = "sector",
    ) -> "SectorSpecializedAgent":
        self._sector_agents = {}
        self._sector_sample_count = {}
        self._feature_cols = []

        if sector_col not in X.columns:
            log.warning("[%s] Missing sector column '%s' — no sector models trained.", self.name, sector_col)
            self.is_trained = False
            return self

        y_aligned = y.reindex(X.index)
        if y_aligned.isna().all() and len(y) == len(X):
            y_aligned = pd.Series(y.to_numpy(), index=X.index)

        sectors = X[sector_col].fillna("Unknown").astype(str)
        trained = 0
        skipped = 0

        for sector in sorted(sectors.unique().tolist()):
            if sector == "Unknown":
                continue
            mask = sectors == sector
            X_sector = X.loc[mask].copy()
            y_sector = y_aligned.loc[X_sector.index].dropna()
            X_sector = X_sector.loc[y_sector.index]

            n_obs = int(len(X_sector))
            self._sector_sample_count[sector] = n_obs

            if n_obs < self._min_samples_per_sector:
                skipped += 1
                continue
            if y_sector.nunique() < 2:
                skipped += 1
                continue

            child = self._instantiate_child_agent(sector)
            fit_kwargs: Dict[str, Any] = {}
            if self._supports_arg(child.fit, "fold"):
                fit_kwargs["fold"] = f"{fold}_{self._sanitize_sector_name(sector)}" if fold is not None else sector
            if self._supports_arg(child.fit, "sector_col"):
                fit_kwargs["sector_col"] = sector_col

            child.fit(X_sector, y_sector, **fit_kwargs)
            if getattr(child, "is_trained", False):
                self._sector_agents[sector] = child
                trained += 1
                if not self._feature_cols:
                    self._feature_cols = list(getattr(child, "_feature_cols", []) or [])
            else:
                skipped += 1

        self.is_trained = trained > 0
        self._diagnostics = {
            "n_sectors_seen": int(len([s for s in sectors.unique().tolist() if s != "Unknown"])),
            "n_sectors_trained": int(trained),
            "n_sectors_skipped": int(skipped),
            "min_samples_per_sector": int(self._min_samples_per_sector),
            "neutral_score": float(self._neutral_score),
            "sector_sample_count": self._sector_sample_count,
            "trained_sectors": sorted(self._sector_agents.keys()),
            "wrapped_agent_class": self._agent_cls.__name__,
        }
        self.save_diagnostics(fold)
        return self

    def predict_score(self, X: pd.DataFrame, sector_col: str = "sector") -> pd.Series:
        scores = pd.Series(self._neutral_score, index=X.index, dtype=float, name=f"{self.name}_score")
        if not self.is_trained:
            return scores
        if sector_col not in X.columns:
            return scores

        sectors = X[sector_col].fillna("Unknown").astype(str)
        for sector in sorted(sectors.unique().tolist()):
            mask = sectors == sector
            idx = X.index[mask]
            if len(idx) == 0:
                continue

            child = self._sector_agents.get(sector)
            if child is None:
                continue

            X_sector = X.loc[idx]
            pred_kwargs: Dict[str, Any] = {}
            if self._supports_arg(child.predict_score, "sector_col"):
                pred_kwargs["sector_col"] = sector_col
            pred = child.predict_score(X_sector, **pred_kwargs)
            pred = pd.Series(pred, index=X_sector.index, dtype=float)
            scores.loc[idx] = pred.reindex(idx).fillna(self._neutral_score).to_numpy()

        return scores
