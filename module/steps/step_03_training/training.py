"""Training workflows for base agents and the meta learner."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from environment import OOF_N_SPLITS
from module.agents.meta_learner import MetaLearner
from module.steps.step_03_training.agent_config import build_agents_config
from module.steps.step_03_training.oof import generate_oof_scores

log = logging.getLogger(__name__)


def _instantiate_base_agents(agents_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        ag_name: cfg["cls"](**cfg["kwargs"])
        for ag_name, cfg in agents_config.items()
    }


def _fit_base_agents(
    agents: Dict[str, Any],
    agents_config: Dict[str, Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    fold: int,
) -> None:
    for ag_name, agent in agents.items():
        cfg = agents_config[ag_name]
        y_fit = (1 - y) if cfg.get("invert_y") else y
        sector_col = cfg.get("sector_col")
        if sector_col:
            agent.fit(X, y_fit, fold=fold, sector_col=sector_col)
        else:
            agent.fit(X, y_fit, fold=fold)


def _predict_base_scores(
    agents: Dict[str, Any],
    agents_config: Dict[str, Dict[str, Any]],
    X: pd.DataFrame,
) -> pd.DataFrame:
    out = X.copy()
    for ag_name, agent in agents.items():
        sector_col = agents_config[ag_name].get("sector_col")
        if sector_col:
            scores = agent.predict_score(out, sector_col)
        else:
            scores = agent.predict_score(out)
        out[f"{ag_name}_score"] = scores.values
    return out


def train_fold(
    df_train_norm: pd.DataFrame,
    df_test_norm: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    fold_id: int,
    agents_results_dir: str,
    random_seed: int = 42,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    agents_config = build_agents_config(agents_results_dir=agents_results_dir, random_seed=random_seed)
    base_agents = _instantiate_base_agents(agents_config)
    meta = MetaLearner(results_dir=agents_results_dir, random_seed=random_seed)

    _fit_base_agents(base_agents, agents_config, df_train_norm, y_train, fold=fold_id)

    oof_scores = generate_oof_scores(
        df_train_norm,
        y_train,
        agents_config=agents_config,
        n_splits=OOF_N_SPLITS,
        random_seed=random_seed,
    )
    df_train_with_oof = df_train_norm.copy()
    for col_name, scores_series in oof_scores.items():
        df_train_with_oof[col_name] = scores_series

    meta.fit(df_train_with_oof, y_train, fold=fold_id, sector_col="sector")

    df_test = _predict_base_scores(base_agents, agents_config, df_test_norm)
    df_test["final_score"] = meta.predict_score(df_test, "sector").values
    df_test["label"] = y_test.values

    agents_dict = {**base_agents, "meta_learner": meta}
    return agents_dict, df_test, df_train_with_oof


def train_full_history(
    df_norm: pd.DataFrame,
    y: pd.Series,
    agents_results_dir: str,
    random_seed: int = 42,
) -> Tuple[Dict, pd.DataFrame]:
    agents_config = build_agents_config(agents_results_dir=agents_results_dir, random_seed=random_seed)
    base_agents = _instantiate_base_agents(agents_config)
    meta = MetaLearner(results_dir=agents_results_dir, random_seed=random_seed)

    _fit_base_agents(base_agents, agents_config, df_norm, y, fold=0)

    df_with_scores = _predict_base_scores(base_agents, agents_config, df_norm)
    meta.fit(df_with_scores, y, fold=0, sector_col="sector")

    agents_dict = {**base_agents, "meta_learner": meta}
    return agents_dict, df_with_scores
