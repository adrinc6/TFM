"""Baselines deterministas de Fase 2; producen scores, no rentabilidad."""

from __future__ import annotations

import math

import pandas as pd


QUALITY_FACTORS = [
    "factor_roe",
    "factor_roic",
    "factor_net_margin",
    "factor_operating_margin",
    "factor_gross_margin",
    "factor_fcf_margin",
    "factor_debt_equity",
    "factor_current_ratio",
]
GROWTH_FACTORS = ["factor_eps_growth_yoy", "factor_sales_per_share_growth_yoy"]
VALUE_FACTORS = ["factor_pe", "factor_pb", "factor_ps", "factor_ev_ebitda"]
MOMENTUM_FACTORS = ["factor_relative_return_3m", "factor_relative_return_6m", "factor_relative_return_12m"]


def build_baseline_scores(features: pd.DataFrame) -> pd.DataFrame:
    """Calcula GARP equilibrado y momentum puro con cobertura explícita."""
    frame = features.copy()
    result = frame[["ticker", "snapshot_date"]].copy()
    eligible = frame["is_price_fresh"].fillna(False).astype(bool)

    result["quality_components"] = frame[QUALITY_FACTORS].notna().sum(axis=1)
    result["growth_components"] = frame[GROWTH_FACTORS].notna().sum(axis=1)
    result["value_components"] = frame[VALUE_FACTORS].notna().sum(axis=1)
    result["momentum_components"] = frame[MOMENTUM_FACTORS].notna().sum(axis=1)

    result["quality_score"] = _mean_with_minimum(
        frame[QUALITY_FACTORS], math.ceil(len(QUALITY_FACTORS) / 2), eligible
    )
    result["growth_score"] = _mean_with_minimum(
        frame[GROWTH_FACTORS], math.ceil(len(GROWTH_FACTORS) / 2), eligible
    )
    result["value_score"] = _mean_with_minimum(
        frame[VALUE_FACTORS], math.ceil(len(VALUE_FACTORS) / 2), eligible
    )
    result["momentum_score"] = _mean_with_minimum(frame[MOMENTUM_FACTORS], 2, eligible)

    result["garp_score"] = result[["quality_score", "growth_score", "value_score"]].mean(axis=1)
    result.loc[
        result[["quality_score", "growth_score", "value_score"]].isna().any(axis=1), "garp_score"
    ] = float("nan")

    for score in ("quality_score", "growth_score", "value_score", "momentum_score", "garp_score"):
        result[f"{score}_rank"] = result.groupby("snapshot_date")[score].rank(
            method="average", pct=True
        )
    return result


def _mean_with_minimum(values: pd.DataFrame, minimum: int, eligible: pd.Series) -> pd.Series:
    result = values.mean(axis=1, skipna=True)
    return result.where(eligible & values.notna().sum(axis=1).ge(minimum))
