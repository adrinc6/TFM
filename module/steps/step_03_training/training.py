"""Training workflows for base agents and the meta learner."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from environment import (
    OOF_N_SPLITS,
    SCORE_DISPERSION_MIN_STD,
    SECTOR_CONFIDENCE_PEERS,
    SECTOR_SCORE_PRIOR_BASE,
    SECTOR_SCORE_PRIOR_WEIGHT,
)
from module.agents.meta_learner import MetaLearner
from module.steps.step_03_training.agent_config import build_agents_config, build_sector_rotation_agent
from module.steps.step_03_training.oof import generate_oof_scores

log = logging.getLogger(__name__)


def _compute_dispersion_scales(df: pd.DataFrame, score_cols: list[str]) -> Dict[str, float]:
    scales: Dict[str, float] = {}
    for col in score_cols:
        if col not in df.columns:
            continue
        std = float(df[col].std()) if SCORE_DISPERSION_MIN_STD > 0 else 0.0
        scale = 1.0 if std >= SCORE_DISPERSION_MIN_STD else (std / SCORE_DISPERSION_MIN_STD)
        scales[col] = max(0.0, min(1.0, scale))
    return scales


def _apply_dispersion_shrink(df: pd.DataFrame, scales: Dict[str, float]) -> pd.DataFrame:
    for col, scale in scales.items():
        if col not in df.columns or scale >= 1.0:
            continue
        df[col] = 0.5 + (df[col] - 0.5) * scale
    return df


def _apply_sector_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    if "sector" not in df.columns or "final_score" not in df.columns:
        return df
    sector_counts = (
        df.reset_index()[["ticker", "sector"]]
        .drop_duplicates(subset="ticker")
        .groupby("sector")["ticker"]
        .nunique()
    )
    df["sector_peer_count"] = df["sector"].map(sector_counts).fillna(0).astype(int)
    if SECTOR_CONFIDENCE_PEERS > 0:
        df["sector_confidence"] = np.sqrt(df["sector_peer_count"] / SECTOR_CONFIDENCE_PEERS).clip(upper=1.0)
    else:
        df["sector_confidence"] = 1.0
    if "sector_score" in df.columns:
        sector_score = df["sector_score"].fillna(0.5)
    else:
        sector_score = 0.5
    sector_prior = SECTOR_SCORE_PRIOR_BASE + SECTOR_SCORE_PRIOR_WEIGHT * sector_score
    df["final_score_raw"] = df["final_score"]
    df["final_score"] = (df["final_score"] * sector_prior * df["sector_confidence"]).clip(0.0, 1.0)
    return df


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
        if not getattr(agent, "is_trained", False):
            scores = pd.Series(0.5, index=out.index)
        elif sector_col:
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
    sector_map: Optional[Dict[str, str]] = None,
    spy_prices: Optional[pd.Series] = None,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    agents_config = build_agents_config(agents_results_dir=agents_results_dir, random_seed=random_seed)
    base_agents = _instantiate_base_agents(agents_config)
    meta = MetaLearner(results_dir=agents_results_dir, random_seed=random_seed)

    log.info(f"[Fold {fold_id}] 1/3 — Entrenando 6 agentes base con datos de entrenamiento del fold...")
    _fit_base_agents(base_agents, agents_config, df_train_norm, y_train, fold=fold_id)

    # Entrenar SectorRotationAgent (opera a nivel sector, no ticker)
    sector_agent = build_sector_rotation_agent(agents_results_dir=agents_results_dir, random_seed=random_seed)
    if sector_map is not None and "forward_return" in df_train_norm.columns:
        log.info(f"[Fold {fold_id}] 1/3 — Entrenando SectorRotationAgent (top-down, nivel sector)...")
        sector_agent.fit(df_train_norm, sector_map=sector_map, spy_prices=spy_prices, fold=fold_id)
    else:
        log.warning(f"[Fold {fold_id}] SectorRotationAgent: sin sector_map o forward_return — score neutro 0.5")

    log.info(f"[Fold {fold_id}] 2/3 — Generando scores OOF ({OOF_N_SPLITS} splits temporales) como input del MetaLearner...")
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

    # Añadir sector_score al train OOF usando el agente ya entrenado
    if sector_agent.is_trained and sector_map is not None:
        sector_scores_train = sector_agent.predict_sector_scores(df_train_with_oof, sector_map)
        df_train_with_oof["sector_score"] = sector_agent.map_to_tickers(
            df_train_with_oof, sector_map, sector_scores_train
        ).values

    score_cols = [f"{ag_name}_score" for ag_name in agents_config.keys()]
    if "sector_score" in df_train_with_oof.columns:
        score_cols.append("sector_score")
    dispersion_scales = _compute_dispersion_scales(df_train_with_oof, score_cols)
    df_train_with_oof = _apply_dispersion_shrink(df_train_with_oof, dispersion_scales)

    log.info(f"[Fold {fold_id}] 2/3 — Entrenando MetaLearner sobre scores OOF (anti-leakage)...")
    meta.fit(df_train_with_oof, y_train, fold=fold_id, sector_col="sector")

    log.info(f"[Fold {fold_id}] 3/3 — Generando predicciones sobre el quarter de test ({len(df_test_norm)} tickers)...")
    df_test = _predict_base_scores(base_agents, agents_config, df_test_norm)

    # Añadir sector_score al test
    if sector_agent.is_trained and sector_map is not None:
        sector_scores_test = sector_agent.predict_sector_scores(df_test, sector_map)
        df_test["sector_score"] = sector_agent.map_to_tickers(
            df_test, sector_map, sector_scores_test
        ).values
    else:
        df_test["sector_score"] = 0.5

    df_test = _apply_dispersion_shrink(df_test, dispersion_scales)
    df_test["final_score"] = meta.predict_score(df_test, "sector").values
    df_test = _apply_sector_adjustments(df_test)
    df_test["label"] = y_test.values
    log.info(f"[Fold {fold_id}] 3/3 — Predicciones listas. Scores en rango [{df_test['final_score'].min():.3f}, {df_test['final_score'].max():.3f}]")

    agents_dict = {**base_agents, "sector_rotation": sector_agent, "meta_learner": meta}
    return agents_dict, df_test, df_train_with_oof


def train_full_history(
    df_norm: pd.DataFrame,
    y: pd.Series,
    agents_results_dir: str,
    random_seed: int = 42,
    sector_map: Optional[Dict[str, str]] = None,
    spy_prices: Optional[pd.Series] = None,
) -> Tuple[Dict, pd.DataFrame, Dict[str, float]]:
    agents_config = build_agents_config(agents_results_dir=agents_results_dir, random_seed=random_seed)
    base_agents = _instantiate_base_agents(agents_config)
    meta = MetaLearner(results_dir=agents_results_dir, random_seed=random_seed)

    _fit_base_agents(base_agents, agents_config, df_norm, y, fold=0)

    sector_agent = build_sector_rotation_agent(agents_results_dir=agents_results_dir, random_seed=random_seed)
    if sector_map is not None and "forward_return" in df_norm.columns:
        sector_agent.fit(df_norm, sector_map=sector_map, spy_prices=spy_prices, fold=0)

    df_with_scores = _predict_base_scores(base_agents, agents_config, df_norm)

    if sector_agent.is_trained and sector_map is not None:
        sector_scores = sector_agent.predict_sector_scores(df_with_scores, sector_map)
        df_with_scores["sector_score"] = sector_agent.map_to_tickers(
            df_with_scores, sector_map, sector_scores
        ).values
    else:
        df_with_scores["sector_score"] = 0.5

    score_cols = [f"{ag_name}_score" for ag_name in agents_config.keys()]
    if "sector_score" in df_with_scores.columns:
        score_cols.append("sector_score")
    dispersion_scales = _compute_dispersion_scales(df_with_scores, score_cols)
    df_with_scores = _apply_dispersion_shrink(df_with_scores, dispersion_scales)

    meta.fit(df_with_scores, y, fold=0, sector_col="sector")

    agents_dict = {**base_agents, "sector_rotation": sector_agent, "meta_learner": meta}
    return agents_dict, df_with_scores, dispersion_scales
