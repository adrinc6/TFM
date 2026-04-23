"""Tests for sector-specialized modeling wrappers."""

import numpy as np
import pandas as pd
import pytest

from module.agents.base import BaseAgent
from module.agents.fundamental import FundamentalAgent
from module.agents.sector_specialized import SectorSpecializedAgent


class _DummyBaseAgent(BaseAgent):
    """Small deterministic agent used to validate sector wrapper behavior."""

    def __init__(self, results_dir: str, random_seed: int = 42, save_artifacts: bool = True):
        super().__init__("dummy", results_dir, random_seed, save_artifacts)
        self._mean = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "_DummyBaseAgent":
        self._mean = float(y.mean()) if len(y) else 0.5
        self.is_trained = True
        return self

    def predict_score(self, X: pd.DataFrame, **kwargs) -> pd.Series:
        return pd.Series(self._mean, index=X.index, name="dummy_score")


def _build_xy() -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.MultiIndex.from_tuples(
        [
            ("A1", pd.Timestamp("2024-01-01")),
            ("A2", pd.Timestamp("2024-01-01")),
            ("A3", pd.Timestamp("2024-01-01")),
            ("B1", pd.Timestamp("2024-01-01")),
            ("B2", pd.Timestamp("2024-01-01")),
            ("B3", pd.Timestamp("2024-01-01")),
        ],
        names=["ticker", "date"],
    )
    X = pd.DataFrame(
        {
            "sector": ["Tech", "Tech", "Tech", "Utilities", "Utilities", "Utilities"],
            "feature": [1.0, 2.0, 3.0, 1.0, 0.5, 0.25],
        },
        index=idx,
    )
    y = pd.Series([1, 1, 0, 0, 0, 1], index=idx)
    return X, y


def test_sector_wrapper_trains_independent_models(tmp_path):
    X, y = _build_xy()

    agent = SectorSpecializedAgent(
        name="dummy",
        agent_cls=_DummyBaseAgent,
        results_dir=str(tmp_path),
        min_samples_per_sector=2,
        save_artifacts=False,
    )
    agent.fit(X, y, sector_col="sector")

    preds = agent.predict_score(X, sector_col="sector")
    tech_mean = float(preds.loc[preds.index.get_level_values("ticker").str.startswith("A")].mean())
    util_mean = float(preds.loc[preds.index.get_level_values("ticker").str.startswith("B")].mean())

    assert agent.is_trained is True
    assert tech_mean > util_mean


def test_sector_wrapper_unseen_sector_gets_neutral_score(tmp_path):
    X, y = _build_xy()

    agent = SectorSpecializedAgent(
        name="dummy",
        agent_cls=_DummyBaseAgent,
        results_dir=str(tmp_path),
        min_samples_per_sector=2,
        save_artifacts=False,
    )
    agent.fit(X, y, sector_col="sector")

    idx_new = pd.MultiIndex.from_tuples(
        [("C1", pd.Timestamp("2024-04-01"))],
        names=["ticker", "date"],
    )
    X_new = pd.DataFrame({"sector": ["Energy"], "feature": [2.0]}, index=idx_new)
    pred = agent.predict_score(X_new, sector_col="sector")

    assert float(pred.iloc[0]) == 0.5


def test_sector_wrapper_skips_under_sampled_sectors(tmp_path):
    X, y = _build_xy()

    # Force skip: each sector has only 3 rows, but min required is 4.
    agent = SectorSpecializedAgent(
        name="dummy",
        agent_cls=_DummyBaseAgent,
        results_dir=str(tmp_path),
        min_samples_per_sector=4,
        save_artifacts=False,
    )
    agent.fit(X, y, sector_col="sector")
    preds = agent.predict_score(X, sector_col="sector")

    assert agent.is_trained is False
    assert (preds == 0.5).all()


def test_fundamental_cv_skips_single_class_train_splits(tmp_path, monkeypatch):
    class _DummyXGBClassifier:
        def __init__(self, *args, **kwargs):
            self._proba = 0.5

        def fit(self, X, y, eval_set=None, verbose=False):
            if pd.Series(y).nunique() < 2:
                raise AssertionError("CV should skip one-class train splits")
            self._proba = float(pd.Series(y).mean())
            return self

        def predict_proba(self, X):
            proba = np.full(len(X), self._proba, dtype=float)
            return np.column_stack([1.0 - proba, proba])

    monkeypatch.setattr("module.agents.fundamental.xgb.XGBClassifier", _DummyXGBClassifier)

    X = pd.DataFrame({"f1": np.linspace(0.0, 1.0, 12)})
    y = pd.Series([1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1])

    agent = FundamentalAgent(results_dir=str(tmp_path), save_artifacts=False)
    cv = agent._cv(X, y, spw=1.0)

    assert cv["mean_acc"] >= 0.0
    assert cv["mean_f1"] >= 0.0
    assert cv["std_auc"] >= 0.0
