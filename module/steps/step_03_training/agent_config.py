"""Agent configuration factory for training."""

from __future__ import annotations

from typing import Any, Dict

from module.agents.fundamental import FundamentalAgent
from module.agents.valuation import ValuationAgent
from module.agents.momentum import MomentumAgent
from module.agents.bear import BearAgent
from module.agents.sentiment import SentimentAgent
from module.agents.sector_rotation import SectorRotationAgent


def build_agents_config(agents_results_dir: str, random_seed: int) -> Dict[str, Dict[str, Any]]:
    """Configuracion declarativa de agentes base y su contrato de entrenamiento.

    El SectorRotationAgent se instancia aquí pero se entrena aparte en training.py
    porque opera a nivel sector (no ticker) y no sigue la misma firma fit(X, y).
    """
    return {
        "fundamental": {
            "cls": FundamentalAgent,
            "kwargs": {"results_dir": agents_results_dir, "random_seed": random_seed},
            "sector_col": "sector",
            "invert_y": False,
        },
        "valuation": {
            "cls": ValuationAgent,
            "kwargs": {"results_dir": agents_results_dir, "random_seed": random_seed},
            "sector_col": "sector",
            "invert_y": False,
        },
        "momentum": {
            "cls": MomentumAgent,
            "kwargs": {"results_dir": agents_results_dir, "random_seed": random_seed},
            "sector_col": None,
            "invert_y": False,
        },
        "bear": {
            "cls": BearAgent,
            "kwargs": {"results_dir": agents_results_dir, "random_seed": random_seed},
            "sector_col": None,
            "invert_y": True,
        },
        "sentiment": {
            "cls": SentimentAgent,
            "kwargs": {"results_dir": agents_results_dir, "random_seed": random_seed},
            "sector_col": None,
            "invert_y": False,
        },
    }


def build_sector_rotation_agent(agents_results_dir: str, random_seed: int) -> SectorRotationAgent:
    """Instancia el SectorRotationAgent (se entrena aparte en training.py)."""
    return SectorRotationAgent(results_dir=agents_results_dir, random_seed=random_seed)
