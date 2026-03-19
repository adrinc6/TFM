"""Agent configuration factory for training."""

from __future__ import annotations

from typing import Any, Dict

from module.agents.fundamental import FundamentalAgent
from module.agents.valuation import ValuationAgent
from module.agents.momentum import MomentumAgent
from module.agents.bear import BearAgent
from module.agents.sentiment import SentimentAgent


def build_agents_config(agents_results_dir: str, random_seed: int) -> Dict[str, Dict[str, Any]]:
	"""Configuracion declarativa de agentes base y su contrato de entrenamiento."""
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
