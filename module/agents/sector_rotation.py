from __future__ import annotations

from environment import SECTOR_ROTATION_FEATURE_COLUMNS
from module.agents.base import BaseAgent


class SectorRotationAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="sector_rotation", feature_pool=list(SECTOR_ROTATION_FEATURE_COLUMNS), min_features=8, max_features=12)
