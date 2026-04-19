"""Agent configuration factory for training."""

from __future__ import annotations

from typing import Any, Dict

from module.agents.fundamental import FundamentalAgent
from module.agents.valuation import ValuationAgent
from module.agents.momentum import MomentumAgent
from module.agents.bear import BearAgent
from module.agents.sentiment import SentimentAgent
from module.agents.sector_specialized import SectorSpecializedAgent
from module.agents.sector_rotation import SectorRotationAgent
from environment import SECTOR_SPECIALIST_MIN_SAMPLES


def build_agents_config(agent_models_results_dir: str, random_seed: int) -> Dict[str, Dict[str, Any]]:
    """Builds the declarative configuration for all base agents.

    The SectorRotationAgent is instantiated here but trained separately in
    training.py because it operates at the sector level (not ticker level) and
    does not share the standard ``fit(X, y)`` signature.

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
    return {
        "fundamental": {
            "cls": SectorSpecializedAgent,
            "kwargs": {
                "name": "fundamental",
                "agent_cls": FundamentalAgent,
                "results_dir": agent_models_results_dir,
                "random_seed": random_seed,
                "agent_kwargs": {},
                "min_samples_per_sector": SECTOR_SPECIALIST_MIN_SAMPLES,
            },
            "sector_col": "sector",
            "invert_y": False,
        },
        "valuation": {
            "cls": SectorSpecializedAgent,
            "kwargs": {
                "name": "valuation",
                "agent_cls": ValuationAgent,
                "results_dir": agent_models_results_dir,
                "random_seed": random_seed,
                "agent_kwargs": {},
                "min_samples_per_sector": SECTOR_SPECIALIST_MIN_SAMPLES,
            },
            "sector_col": "sector",
            "invert_y": False,
        },
        "momentum": {
            "cls": SectorSpecializedAgent,
            "kwargs": {
                "name": "momentum",
                "agent_cls": MomentumAgent,
                "results_dir": agent_models_results_dir,
                "random_seed": random_seed,
                "agent_kwargs": {},
                "min_samples_per_sector": SECTOR_SPECIALIST_MIN_SAMPLES,
            },
            "sector_col": "sector",
            "invert_y": False,
        },
        "bear": {
            "cls": SectorSpecializedAgent,
            "kwargs": {
                "name": "bear",
                "agent_cls": BearAgent,
                "results_dir": agent_models_results_dir,
                "random_seed": random_seed,
                "agent_kwargs": {},
                "min_samples_per_sector": SECTOR_SPECIALIST_MIN_SAMPLES,
            },
            "sector_col": "sector",
            "invert_y": True,
        },
        "sentiment": {
            "cls": SectorSpecializedAgent,
            "kwargs": {
                "name": "sentiment",
                "agent_cls": SentimentAgent,
                "results_dir": agent_models_results_dir,
                "random_seed": random_seed,
                "agent_kwargs": {},
                "min_samples_per_sector": SECTOR_SPECIALIST_MIN_SAMPLES,
            },
            "sector_col": "sector",
            "invert_y": False,
        },
    }


def build_sector_rotation_agent(agent_models_results_dir: str, random_seed: int) -> SectorRotationAgent:
    """Instantiates the SectorRotationAgent (trained separately in training.py).

    Args:
        agent_models_results_dir (str): Root directory for agent model artefact output.
        random_seed (int): Random seed for the agent.

    Returns:
        SectorRotationAgent: An untrained SectorRotationAgent instance.
    """
    return SectorRotationAgent(results_dir=agent_models_results_dir, random_seed=random_seed)
