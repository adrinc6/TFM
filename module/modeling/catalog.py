"""Catálogo declarativo de factores point-in-time y sus bloques.

El catálogo es la fuente única para documentar qué mide cada columna y para
construir los agentes sin listas de features duplicadas en varios módulos.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    block: str
    agents: tuple[str, ...]
    direction: int = 1
    source: str = "finnhub_series"
    description: str = ""


def _specs(names: tuple[str, ...], block: str, agents: tuple[str, ...], direction: int = 1,
           source: str = "finnhub_series") -> list[FeatureSpec]:
    return [FeatureSpec(f"factor_{name}", block, agents, direction, source) for name in names]


FEATURE_CATALOG: tuple[FeatureSpec, ...] = tuple(
    _specs(("roe", "roic", "roa", "rotc", "net_margin", "operating_margin", "gross_margin",
            "pretax_margin", "fcf_margin"), "quality_core", ("quality",))
    + _specs(("asset_turnover", "inventory_turnover", "receivables_turnover"),
             "quality_efficiency", ("quality",))
    + _specs(("current_ratio", "quick_ratio", "cash_ratio"), "quality_efficiency", ("quality",))
    + _specs(("debt_equity", "debt_assets", "debt_capital", "long_debt_equity", "long_debt_assets",
              "long_debt_capital", "net_debt_equity", "net_debt_capital", "sga_to_sales"),
             "financial_strength", ("quality",), -1)
    + _specs(("pe", "pb", "ptbv", "ps", "pfcf", "ev_ebitda", "ev_revenue"),
             "value_core", ("value",), -1)
    + _specs(("earnings_yield", "fcf_yield"), "value_cashflow", ("value",))
    + _specs(("eps_growth_yoy", "sales_per_share_growth_yoy", "ebitda_growth_yoy", "fcf_per_share_growth_yoy",
              "eps_growth_acceleration", "sales_growth_acceleration", "ebitda_growth_acceleration",
              "fcf_growth_acceleration"), "growth_acceleration", ("growth",))
    + _specs(("roe_trend", "roic_trend", "margin_stability", "roe_stability", "cash_conversion"),
             "fundamental_stability", ("growth",))
    + _specs(("relative_return_3m", "relative_return_6m", "relative_return_12m", "momentum_12_1",
              "mom_acceleration", "mom_reversal_1m"), "momentum_core", ("momentum",))
    # `mom_volatility` vivía fuera de todo bloque: se inyectaba siempre en el agente momentum pero
    # no aparecía en ninguna ablación ni en `feature_catalog.json`. Declararlo aquí lo hace visible
    # y lo somete a la misma activación por bloques que el resto.
    + _specs(("ma_price_vs_sma6", "ma_price_vs_sma12", "ma_distance_to_high12", "mom_volatility"),
             "momentum_trend", ("momentum",))
    + _specs(("realized_vol_63d", "realized_vol_126d", "downside_vol_63d", "beta_252d",
              "drawdown_252d", "max_drawdown_252d", "range_21d", "range_63d"),
             "price_risk", ("risk",), -1, "ohlcv")
    + _specs(("volume_relative", "volume_volatility", "amihud_illiquidity", "gap_21d"),
             "market_liquidity", ("risk",), -1, "ohlcv")
)

AGENT_NAMES = ("quality", "value", "growth", "momentum", "risk")


def catalog_by_agent(enabled_blocks: tuple[str, ...], enabled_agents: tuple[str, ...]) -> dict[str, list[str]]:
    return {
        agent: [spec.name for spec in FEATURE_CATALOG if spec.block in enabled_blocks and agent in spec.agents]
        for agent in enabled_agents
    }
