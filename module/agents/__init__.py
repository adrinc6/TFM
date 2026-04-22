from module.agents.base import AgentPerformance, BaseAgent
from module.agents.fundamental import FundamentalAgent
from module.agents.momentum import MomentumAgent
from module.agents.sector_rotation import SectorRotationAgent
from module.agents.sentiment import SentimentAgent
from module.agents.technical import TechnicalAgent
from module.agents.valuation import ValuationAgent


def build_agents() -> list[BaseAgent]:
    return [
        FundamentalAgent(),
        ValuationAgent(),
        TechnicalAgent(),
        MomentumAgent(),
        SectorRotationAgent(),
        SentimentAgent(),
    ]


__all__ = [
    "AgentPerformance",
    "BaseAgent",
    "FundamentalAgent",
    "ValuationAgent",
    "TechnicalAgent",
    "MomentumAgent",
    "SectorRotationAgent",
    "SentimentAgent",
    "build_agents",
]
