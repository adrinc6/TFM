"""Market regime detection and dynamic score weighting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


REGIME_RISK_ON = "Risk-On"
REGIME_NEUTRAL = "Neutral"
REGIME_RISK_OFF = "Risk-Off"


@dataclass
class MarketRegimeModel:
    """Deterministic regime detector using macro and volatility context."""

    on_threshold: float = 0.75
    off_threshold: float = -0.75

    def __post_init__(self) -> None:
        self._mu: Dict[str, float] = {}
        self._sigma: Dict[str, float] = {}

    @staticmethod
    def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
        cols = {
            "vix": 0.0,
            "sp500_momentum_3m": 0.0,
            "sp500_momentum_6m": 0.0,
            "sp500_momentum_12m": 0.0,
            "yield_curve": 0.0,
            "volatility_60d": 0.0,
        }
        out = pd.DataFrame(index=df.index)
        for c, default in cols.items():
            if c in df.columns:
                out[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                out[c] = default
        out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        return out

    def fit(self, df: pd.DataFrame) -> "MarketRegimeModel":
        X = self._feature_frame(df)
        for c in X.columns:
            mu = float(X[c].median())
            # MAD (mean absolute deviation) from median
            sigma = float(np.abs(X[c] - mu).mean())
            if not np.isfinite(sigma) or sigma < 1e-6:
                sigma = float(X[c].std())
            if not np.isfinite(sigma) or sigma < 1e-6:
                sigma = 1.0
            self._mu[c] = mu
            self._sigma[c] = sigma
        return self

    def score(self, df: pd.DataFrame) -> pd.Series:
        X = self._feature_frame(df)
        z = pd.DataFrame(index=X.index)
        for c in X.columns:
            mu = self._mu.get(c, 0.0)
            sigma = self._sigma.get(c, 1.0)
            z[c] = (X[c] - mu) / sigma

        score = (
            0.30 * z["sp500_momentum_3m"]
            + 0.25 * z["sp500_momentum_6m"]
            + 0.20 * z["sp500_momentum_12m"]
            + 0.15 * z["yield_curve"]
            - 0.20 * z["vix"]
            - 0.10 * z["volatility_60d"]
        )
        return score.clip(-4.0, 4.0)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        s = self.score(df)
        regime = pd.Series(REGIME_NEUTRAL, index=s.index, dtype=object)
        regime.loc[s >= self.on_threshold] = REGIME_RISK_ON
        regime.loc[s <= self.off_threshold] = REGIME_RISK_OFF
        return regime


def apply_regime_weighting(
    df: pd.DataFrame,
    regime_col: str = "regime_state",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply dynamic agent reweighting based on detected regime."""
    out = df.copy()
    if regime_col not in out.columns:
        out[regime_col] = REGIME_NEUTRAL

    multipliers = {
        REGIME_RISK_ON: {
            "momentum_score": 1.20,
            "sentiment_score": 1.15,
            "bear_score": 0.95,
            "valuation_score": 0.95,
            "sector_score": 1.10,
        },
        REGIME_RISK_OFF: {
            "momentum_score": 0.80,
            "sentiment_score": 0.90,
            "bear_score": 1.20,
            "valuation_score": 1.15,
            "sector_score": 0.90,
        },
        REGIME_NEUTRAL: {
            "momentum_score": 1.0,
            "sentiment_score": 1.0,
            "bear_score": 1.0,
            "valuation_score": 1.0,
            "sector_score": 1.0,
        },
    }

    for c in ["fundamental_score", "valuation_score", "momentum_score", "bear_score", "sentiment_score", "sector_score"]:
        if c not in out.columns:
            continue
        vals = pd.to_numeric(out[c], errors="coerce").fillna(0.5)
        m = out[regime_col].map(lambda r: multipliers.get(str(r), multipliers[REGIME_NEUTRAL]).get(c, 1.0)).astype(float)
        out[c] = (vals * m).clip(0.0, 1.0)

    score_cols = [c for c in ["fundamental_score", "valuation_score", "momentum_score", "bear_score", "sentiment_score", "sector_score"] if c in out.columns]
    if score_cols:
        regime_adjusted = out[score_cols].mean(axis=1)
    else:
        regime_adjusted = pd.Series(0.5, index=out.index)
    out["regime_adjusted_score"] = regime_adjusted
    return out, regime_adjusted
