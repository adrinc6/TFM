from __future__ import annotations

from environment import SENTIMENT_FEATURE_COLUMNS
from module.agents.base import BaseAgent


class SentimentAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="sentiment", feature_pool=list(SENTIMENT_FEATURE_COLUMNS), min_features=8, max_features=12)
