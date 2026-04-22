from __future__ import annotations

from environment import VALUATION_FEATURE_COLUMNS
from module.agents.base import BaseAgent


class ValuationAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="valuation", feature_pool=list(VALUATION_FEATURE_COLUMNS), min_features=8, max_features=12)
