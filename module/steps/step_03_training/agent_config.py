"""Agent configuration factory for training."""

from __future__ import annotations

from typing import Any, Dict

from module.agents.sector_rotation import SectorRotationAgent
from module.agents.universe_tp_sl import UniversalTpSlAgent
from environment import (
    ENABLE_SENTIMENT_AGENT,
    SECTOR_SPECIALIST_LONG_FALLBACK_SCORE,
    FUNDAMENTAL_FEATURE_COLUMNS,
    FUNDAMENTAL_FEATURE_EXCLUDE,
    VALUATION_FEATURE_COLUMNS,
    VALUATION_FEATURE_EXCLUDE,
    MOMENTUM_FEATURE_COLUMNS,
    MOMENTUM_FEATURE_EXCLUDE,
    BEAR_FEATURE_COLUMNS,
    BEAR_FEATURE_EXCLUDE,
    SENTIMENT_FEATURE_COLUMNS,
    SENTIMENT_FEATURE_EXCLUDE,
)


def build_agents_config(agent_models_results_dir: str, random_seed: int) -> Dict[str, Dict[str, Any]]:
    """Builds the declarative configuration for all base agents.

    Agents are trained on the full universe (single model per agent), without
    per-sector model splitting. Sector is used only to compute percentile
    features inside each snapshot.

    Args:
        agent_models_results_dir (str): Root directory for agent model artefact output.
        random_seed (int): Random seed forwarded to every agent.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping of agent name to a configuration
            dictionary with keys:
            - cls: Agent class.
            - kwargs: Constructor keyword arguments.
            - sector_col: Column name for sector (None if unused).
            - invert_y: Whether to invert the target label before training.
    """
    config = {
        "fundamental": {
            "cls": UniversalTpSlAgent,
            "kwargs": {
                "name": "fundamental",
                "results_dir": agent_models_results_dir,
                "random_seed": random_seed,
                "include_features": list(FUNDAMENTAL_FEATURE_COLUMNS),
                "exclude_features": list(FUNDAMENTAL_FEATURE_EXCLUDE),
                "neutral_score": SECTOR_SPECIALIST_LONG_FALLBACK_SCORE,
            },
            "sector_col": "sector",
            "invert_y": False,
        },
        "valuation": {
            "cls": UniversalTpSlAgent,
            "kwargs": {
                "name": "valuation",
                "results_dir": agent_models_results_dir,
                "random_seed": random_seed,
                "include_features": list(VALUATION_FEATURE_COLUMNS),
                "exclude_features": list(VALUATION_FEATURE_EXCLUDE),
                "neutral_score": SECTOR_SPECIALIST_LONG_FALLBACK_SCORE,
            },
            "sector_col": "sector",
            "invert_y": False,
        },
        "momentum": {
            "cls": UniversalTpSlAgent,
            "kwargs": {
                "name": "momentum",
                "results_dir": agent_models_results_dir,
                "random_seed": random_seed,
                "include_features": list(MOMENTUM_FEATURE_COLUMNS),
                "exclude_features": list(MOMENTUM_FEATURE_EXCLUDE),
                "neutral_score": SECTOR_SPECIALIST_LONG_FALLBACK_SCORE,
            },
            "sector_col": "sector",
            "invert_y": False,
        },
        "bear": {
            "cls": UniversalTpSlAgent,
            "kwargs": {
                "name": "bear",
                "results_dir": agent_models_results_dir,
                "random_seed": random_seed,
                "include_features": list(BEAR_FEATURE_COLUMNS),
                "exclude_features": list(BEAR_FEATURE_EXCLUDE),
                "neutral_score": 0.5,
            },
            "sector_col": "sector",
            "invert_y": True,
        },
    }
    if ENABLE_SENTIMENT_AGENT:
        config["sentiment"] = {
            "cls": UniversalTpSlAgent,
            "kwargs": {
                "name": "sentiment",
                "results_dir": agent_models_results_dir,
                "random_seed": random_seed,
                "include_features": list(SENTIMENT_FEATURE_COLUMNS),
                "exclude_features": list(SENTIMENT_FEATURE_EXCLUDE),
                # When sentiment data is sparse, use neutral fallback instead of
                # bearish bias to avoid dragging the ensemble unfairly.
                "neutral_score": 0.5,
            },
            "sector_col": "sector",
            "invert_y": False,
        }
    return config


def build_sector_rotation_agent(agent_models_results_dir: str, random_seed: int) -> SectorRotationAgent:
    """Instantiates the SectorRotationAgent (trained separately in training.py).

    Args:
        agent_models_results_dir (str): Root directory for agent model artefact output.
        random_seed (int): Random seed for the agent.

    Returns:
        SectorRotationAgent: Sector agent instance.
    """
    return SectorRotationAgent(results_dir=agent_models_results_dir, random_seed=random_seed)
