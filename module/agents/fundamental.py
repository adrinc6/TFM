from __future__ import annotations

from environment import FUNDAMENTAL_FEATURE_COLUMNS
from module.agents.base import BaseAgent


class FundamentalAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="fundamental", feature_pool=list(FUNDAMENTAL_FEATURE_COLUMNS), min_features=8, max_features=12)
