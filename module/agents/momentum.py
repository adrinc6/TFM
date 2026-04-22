from __future__ import annotations

from environment import MOMENTUM_FEATURE_COLUMNS
from module.agents.base import BaseAgent


class MomentumAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="momentum", feature_pool=list(MOMENTUM_FEATURE_COLUMNS), min_features=8, max_features=12)
