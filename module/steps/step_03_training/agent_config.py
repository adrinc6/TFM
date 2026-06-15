"""Agent configuration factory for GARP/value-growth training."""

from __future__ import annotations

from typing import Any, Dict

from module.agents.sector_rotation import SectorRotationAgent
from module.agents.garp_domain_agent import GarpDomainAgent
from module.common.garp_validation import validate_garp_runtime_config
from environment import (
    ENABLE_SENTIMENT_AGENT,
    SECTOR_SPECIALIST_LONG_FALLBACK_SCORE,
    QUALITY_FEATURE_COLUMNS,
    QUALITY_FEATURE_EXCLUDE,
    GROWTH_FEATURE_COLUMNS,
    GROWTH_FEATURE_EXCLUDE,
    GARP_VALUATION_FEATURE_COLUMNS,
    GARP_VALUATION_FEATURE_EXCLUDE,
    FUNDAMENTAL_TREND_FEATURE_COLUMNS,
    FUNDAMENTAL_TREND_FEATURE_EXCLUDE,
    CATALYST_FEATURE_COLUMNS,
    CATALYST_FEATURE_EXCLUDE,
    RISK_BEAR_FEATURE_COLUMNS,
    RISK_BEAR_FEATURE_EXCLUDE,
    TECHNICAL_GUARDRAIL_FEATURE_COLUMNS,
    TECHNICAL_GUARDRAIL_FEATURE_EXCLUDE,
    SENTIMENT_FEATURE_COLUMNS,
    SENTIMENT_FEATURE_EXCLUDE,
)


def _agent(name: str, include: list[str], exclude: list[str], *, invert_y: bool = False, neutral_score: float | None = None) -> Dict[str, Any]:
    return {
        "cls": GarpDomainAgent,
        "kwargs": {
            "name": name,
            "results_dir": "",  # overwritten in build_agents_config
            "random_seed": 42,    # overwritten in build_agents_config
            "include_features": list(include),
            "exclude_features": list(exclude),
            "neutral_score": SECTOR_SPECIALIST_LONG_FALLBACK_SCORE if neutral_score is None else neutral_score,
        },
        "sector_col": "sector",
        "invert_y": bool(invert_y),
    }


def build_agents_config(agent_models_results_dir: str, random_seed: int) -> Dict[str, Dict[str, Any]]:
    """Build the GARP/value-growth base-agent stack.

    The stack de-emphasises pure price momentum. Technical information is now a
    guardrail/risk input, while quality, growth, valuation and fundamental trend
    are the primary learned signals.
    """
    config: Dict[str, Dict[str, Any]] = {
        "quality": _agent("quality", QUALITY_FEATURE_COLUMNS, QUALITY_FEATURE_EXCLUDE),
        "growth": _agent("growth", GROWTH_FEATURE_COLUMNS, GROWTH_FEATURE_EXCLUDE),
        "valuation": _agent("valuation", GARP_VALUATION_FEATURE_COLUMNS, GARP_VALUATION_FEATURE_EXCLUDE),
        "fundamental_trend": _agent("fundamental_trend", FUNDAMENTAL_TREND_FEATURE_COLUMNS, FUNDAMENTAL_TREND_FEATURE_EXCLUDE),
        "catalyst": _agent("catalyst", CATALYST_FEATURE_COLUMNS, CATALYST_FEATURE_EXCLUDE),
        # Inverted label: model predicts adverse outcomes; training converts it
        # to a safety score downstream, using a high-is-safer risk_bear_score convention.
        "risk_bear": _agent("risk_bear", RISK_BEAR_FEATURE_COLUMNS, RISK_BEAR_FEATURE_EXCLUDE, invert_y=True, neutral_score=0.5),
        # Technical guardrail is long-oriented only in the sense of "entry is not dangerous".
        "technical_guardrail": _agent("technical_guardrail", TECHNICAL_GUARDRAIL_FEATURE_COLUMNS, TECHNICAL_GUARDRAIL_FEATURE_EXCLUDE, neutral_score=0.45),
    }
    for cfg in config.values():
        cfg["kwargs"]["results_dir"] = agent_models_results_dir
        cfg["kwargs"]["random_seed"] = random_seed

    if ENABLE_SENTIMENT_AGENT:
        config["sentiment"] = _agent("sentiment", SENTIMENT_FEATURE_COLUMNS, SENTIMENT_FEATURE_EXCLUDE)
        config["sentiment"]["kwargs"]["results_dir"] = agent_models_results_dir
        config["sentiment"]["kwargs"]["random_seed"] = random_seed
    validate_garp_runtime_config(config)
    return config


def build_sector_rotation_agent(agent_models_results_dir: str, random_seed: int) -> SectorRotationAgent:
    """Instantiates the sector context model used as a top-down prior only."""
    return SectorRotationAgent(results_dir=agent_models_results_dir, random_seed=random_seed)
