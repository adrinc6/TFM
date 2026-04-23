"""Integrated strategy helpers used by step_04_evaluation.

This module consolidates the old ``module.strategy`` package into one place
because TP/SL simulation, confidence scoring, EV ranking, and dynamic agent
weighting are evaluation concerns in this project.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


log = logging.getLogger(__name__)


MIN_CONFIDENCE = 0.05
MAX_CONFIDENCE = 0.95
_DEFAULT_MIN_STOCKS = 4
_DEFAULT_MAX_STOCKS = 8
_DEFAULT_SECTOR_CAP = 3
_DEFAULT_MAX_HOLDING_DAYS = 90
_DEFAULT_DECAY = 0.85
_DEFAULT_PRIOR = 0.50
_MIN_WEIGHT = 0.05


def compute_confidence(
    agent_scores_df: pd.DataFrame,
    *,
    agent_weights: Optional[Dict[str, float]] = None,
    agent_hit_rates: Optional[Dict[str, float]] = None,
    score_weight: float = 0.5,
    calibration_weight: float = 0.5,
) -> pd.Series:
    """Compute per-ticker confidence scores in [MIN_CONFIDENCE, MAX_CONFIDENCE]."""
    df = agent_scores_df.copy()
    score_cols = [column for column in df.columns if column.endswith("_score")]
    if not score_cols and "score" in df.columns:
        score_cols = ["score"]
    if not score_cols:
        raise ValueError(
            "agent_scores_df must contain columns ending in '_score' or a 'score' column."
        )

    tickers = df["ticker"].values if "ticker" in df.columns else df.index.values

    if agent_weights:
        total_weight = sum(float(agent_weights.get(column, 0.0)) for column in score_cols) or 1.0
        normalized_weights = {
            column: float(agent_weights.get(column, 0.0)) / total_weight for column in score_cols
        }
        raw_score = sum(
            df[column].fillna(0.5) * normalized_weights.get(column, 0.0)
            for column in score_cols
        )
    else:
        raw_score = (
            df[score_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.5)
            .mean(axis=1)
        )

    hit_rates = agent_hit_rates or {}
    if agent_weights:
        total_weight = sum(float(agent_weights.get(column, 0.0)) for column in score_cols) or 1.0
        normalized_weights = {
            column: float(agent_weights.get(column, 0.0)) / total_weight for column in score_cols
        }
        calibration = sum(
            float(hit_rates.get(column, 0.5)) * normalized_weights.get(column, 0.0)
            for column in score_cols
        )
    else:
        calibration = np.mean([float(hit_rates.get(column, 0.5)) for column in score_cols])

    score_weight = float(score_weight)
    calibration_weight = float(calibration_weight)
    total = score_weight + calibration_weight or 1.0
    score_weight /= total
    calibration_weight /= total

    combined = score_weight * np.array(raw_score, dtype=float) + calibration_weight * float(calibration)
    combined = np.clip(combined, MIN_CONFIDENCE, MAX_CONFIDENCE)
    return pd.Series(combined, index=tickers, name="confidence")


def attach_confidence(
    signals: pd.DataFrame,
    *,
    agent_weights: Optional[Dict[str, float]] = None,
    agent_hit_rates: Optional[Dict[str, float]] = None,
    score_weight: float = 0.5,
    calibration_weight: float = 0.5,
) -> pd.DataFrame:
    """Attach a confidence column to a signal DataFrame."""
    confidence = compute_confidence(
        signals,
        agent_weights=agent_weights,
        agent_hit_rates=agent_hit_rates,
        score_weight=score_weight,
        calibration_weight=calibration_weight,
    )
    output = signals.copy()
    if "ticker" in output.columns:
        output["confidence"] = output["ticker"].map(confidence).fillna(0.5)
    else:
        output["confidence"] = confidence.values
    return output


def compute_expected_value(signals: pd.DataFrame) -> pd.Series:
    """Compute EV = confidence * tp_pct - (1 - confidence) * sl_pct."""
    confidence = pd.to_numeric(signals["confidence"], errors="coerce").clip(0.0, 1.0)
    take_profit = pd.to_numeric(signals["tp_pct"], errors="coerce")
    stop_loss = pd.to_numeric(signals["sl_pct"], errors="coerce")
    return confidence * take_profit - (1.0 - confidence) * stop_loss


def select_portfolio(
    signals: pd.DataFrame,
    *,
    min_stocks: int = _DEFAULT_MIN_STOCKS,
    max_stocks: int = _DEFAULT_MAX_STOCKS,
    sector_cap: int = _DEFAULT_SECTOR_CAP,
    ev_threshold: float = 0.0,
    sector_col: Optional[str] = "sector",
) -> pd.DataFrame:
    """Rank candidates by EV and mark the selected portfolio."""
    df = signals.copy()
    df["ev"] = compute_expected_value(df)
    eligible = df[df["ev"] >= ev_threshold].sort_values("ev", ascending=False)

    if eligible.empty or len(eligible) < min_stocks:
        return df.assign(selected=False)

    selected_rows = []
    sector_counts: dict[str, int] = {}

    for _, row in eligible.iterrows():
        if len(selected_rows) >= max_stocks:
            break
        sector = str(row.get(sector_col, "Unknown")) if sector_col else "Unknown"
        if sector_cap > 0 and sector_counts.get(sector, 0) >= sector_cap:
            continue
        selected_rows.append(row)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    if len(selected_rows) < min_stocks:
        existing_tickers = {str(row["ticker"]) for row in selected_rows}
        for _, row in eligible.iterrows():
            if len(selected_rows) >= min_stocks:
                break
            ticker = str(row["ticker"])
            if ticker not in existing_tickers:
                selected_rows.append(row)
                existing_tickers.add(ticker)

    if len(selected_rows) < min_stocks:
        return df.assign(selected=False)

    selected_df = pd.DataFrame(selected_rows)
    selected_tickers = set(selected_df["ticker"].astype(str))
    df["selected"] = df["ticker"].astype(str).isin(selected_tickers)
    return df


def get_portfolio_weights(
    signals: pd.DataFrame,
    selected_only: bool = True,
    weight_by: str = "ev",
) -> pd.Series:
    """Compute normalized portfolio weights."""
    df = signals.copy()
    if selected_only and "selected" in df.columns:
        df = df[df["selected"].astype(bool)]

    if df.empty:
        return pd.Series(dtype=float)

    if weight_by in df.columns:
        raw = pd.to_numeric(df[weight_by], errors="coerce").clip(0.0, None)
        total = float(raw.sum())
        if total > 0:
            weights = raw / total
        else:
            weights = pd.Series(1.0 / len(df), index=df.index)
    else:
        weights = pd.Series(1.0 / len(df), index=df.index)

    weights.index = df["ticker"].values
    return weights


def simulate_tp_sl(
    ticker: str,
    prices: pd.Series,
    entry_date: pd.Timestamp,
    tp_pct: float,
    sl_pct: float,
    *,
    max_holding_days: int = _DEFAULT_MAX_HOLDING_DAYS,
) -> Dict[str, object]:
    """Simulate whether TP or SL is hit first from entry_date onward."""
    result: Dict[str, object] = {
        "ticker": ticker,
        "entry_date": entry_date,
        "actual_entry_date": pd.NaT,
        "entry_price": np.nan,
        "tp_pct": float(tp_pct),
        "sl_pct": float(sl_pct),
        "tp_price": np.nan,
        "sl_price": np.nan,
        "outcome": "NONE",
        "days_to_outcome": int(max_holding_days),
        "outcome_date": pd.NaT,
    }

    if prices is None or prices.empty:
        return result

    prices = pd.to_numeric(prices, errors="coerce").dropna().sort_index()
    prices.index = pd.to_datetime(prices.index)

    entry_ts = pd.Timestamp(entry_date)
    expiry_ts = entry_ts + pd.Timedelta(days=int(max_holding_days))
    entry_candidates = prices.index[prices.index >= entry_ts]
    if len(entry_candidates) == 0:
        return result

    actual_entry = entry_candidates[0]
    entry_price = float(prices.loc[actual_entry])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return result

    result["actual_entry_date"] = pd.Timestamp(actual_entry)
    result["entry_price"] = entry_price
    result["tp_price"] = entry_price * (1.0 + float(tp_pct))
    result["sl_price"] = entry_price * (1.0 - float(sl_pct))

    window = prices.loc[(prices.index > actual_entry) & (prices.index <= expiry_ts)]
    for dt, px in window.items():
        price = float(px)
        days_elapsed = int((pd.Timestamp(dt) - actual_entry).days)
        if price >= result["tp_price"]:
            result["outcome"] = "TP"
            result["days_to_outcome"] = days_elapsed
            result["outcome_date"] = pd.Timestamp(dt)
            return result
        if price <= result["sl_price"]:
            result["outcome"] = "SL"
            result["days_to_outcome"] = days_elapsed
            result["outcome_date"] = pd.Timestamp(dt)
            return result

    if not window.empty:
        result["days_to_outcome"] = int((pd.Timestamp(window.index[-1]) - actual_entry).days)
        result["outcome_date"] = pd.Timestamp(window.index[-1])
    else:
        result["outcome_date"] = pd.Timestamp(actual_entry)
    return result


def run_backtest(
    signals: pd.DataFrame,
    prices_dict: Dict[str, object],
    entry_date: pd.Timestamp,
    *,
    max_holding_days: int = _DEFAULT_MAX_HOLDING_DAYS,
) -> pd.DataFrame:
    """Run TP/SL backtest for all rows in the signals DataFrame."""
    rows = []
    for _, row in signals.iterrows():
        ticker = str(row["ticker"])
        tp_pct = float(row.get("tp_pct", 0.08))
        sl_pct = float(row.get("sl_pct", 0.05))
        prices = _extract_close(prices_dict.get(ticker))
        rows.append(
            simulate_tp_sl(
                ticker=ticker,
                prices=prices,
                entry_date=entry_date,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                max_holding_days=max_holding_days,
            )
        )

    backtest_df = pd.DataFrame(rows)
    if backtest_df.empty:
        return signals.copy()

    merge_cols = [column for column in backtest_df.columns if column != "ticker"]
    merge_cols = [column for column in merge_cols if column not in signals.columns or column in ("entry_date",)]
    output = signals.copy()
    indexed_backtest = backtest_df.set_index("ticker")
    for column in merge_cols:
        output[column] = output["ticker"].map(indexed_backtest[column])
    for column in ("outcome", "days_to_outcome", "entry_price", "tp_price", "sl_price", "entry_date"):
        if column in backtest_df.columns:
            output[column] = output["ticker"].map(indexed_backtest[column])
    return output


def _extract_close(price_obj) -> pd.Series:
    if price_obj is None:
        return pd.Series(dtype=float)
    if isinstance(price_obj, pd.Series):
        return pd.to_numeric(price_obj, errors="coerce").dropna().sort_index()
    if isinstance(price_obj, pd.DataFrame) and not price_obj.empty:
        column = "Close" if "Close" in price_obj.columns else price_obj.columns[-1]
        return pd.to_numeric(price_obj[column], errors="coerce").dropna().sort_index()
    return pd.Series(dtype=float)


class AgentWeightTracker:
    """Track per-agent TP-hit accuracy and derive dynamic weights."""

    def __init__(
        self,
        agent_names: List[str],
        *,
        decay: float = _DEFAULT_DECAY,
        prior_hit_rate: float = _DEFAULT_PRIOR,
        min_weight: float = _MIN_WEIGHT,
    ) -> None:
        self.agent_names = list(agent_names)
        self.decay = float(decay)
        self.prior = float(prior_hit_rate)
        self.min_weight = float(min_weight)
        self._ewma: Dict[str, float] = {agent: self.prior for agent in self.agent_names}
        self._history: List[Dict] = []

    def update(
        self,
        fold_id: str,
        outcomes: pd.DataFrame,
        agent_scores_df: pd.DataFrame,
        top_n: int = 10,
    ) -> None:
        if outcomes.empty or agent_scores_df.empty:
            return

        outcome_map = dict(zip(outcomes["ticker"].astype(str), outcomes["outcome"].astype(str)))
        score_cols = [column for column in agent_scores_df.columns if column.endswith("_score")]
        fold_record: Dict[str, object] = {"fold_id": fold_id, "agents": {}}

        for column in score_cols:
            if column not in self.agent_names:
                self.agent_names.append(column)
                self._ewma[column] = self.prior

            source = agent_scores_df[["ticker", column]].copy()
            source[column] = pd.to_numeric(source[column], errors="coerce")
            top_tickers = source.nlargest(top_n, column)["ticker"].astype(str).tolist()
            tp_hits = sum(1 for ticker in top_tickers if outcome_map.get(ticker) == "TP")
            hit_rate = tp_hits / len(top_tickers) if top_tickers else self.prior
            self._ewma[column] = self.decay * self._ewma[column] + (1.0 - self.decay) * hit_rate

            fold_record["agents"][column] = {
                "top_tickers": top_tickers,
                "tp_hits": tp_hits,
                "hit_rate": round(hit_rate, 4),
                "ewma_hit_rate": round(self._ewma[column], 4),
            }

        self._history.append(fold_record)
        log.info(
            "[AgentWeightTracker] Fold %s updated. EWMA hit-rates: %s",
            fold_id,
            {key: round(value, 3) for key, value in self._ewma.items()},
        )

    def get_weights(self) -> Dict[str, float]:
        raw = {
            agent: max(self._ewma.get(agent, self.prior), self.min_weight)
            for agent in self.agent_names
        }
        total = sum(raw.values()) or 1.0
        return {agent: value / total for agent, value in raw.items()}

    def get_hit_rates(self) -> Dict[str, float]:
        return {agent: round(self._ewma.get(agent, self.prior), 4) for agent in self.agent_names}

    def get_history(self) -> List[Dict]:
        return list(self._history)

    def save(self, path: str | Path) -> None:
        state = {
            "agent_names": self.agent_names,
            "decay": self.decay,
            "prior": self.prior,
            "min_weight": self.min_weight,
            "ewma": self._ewma,
            "history": self._history,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, default=str)
        log.info("[AgentWeightTracker] State saved -> %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "AgentWeightTracker":
        with open(path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        tracker = cls(
            agent_names=state.get("agent_names", []),
            decay=state.get("decay", _DEFAULT_DECAY),
            prior_hit_rate=state.get("prior", _DEFAULT_PRIOR),
            min_weight=state.get("min_weight", _MIN_WEIGHT),
        )
        tracker._ewma = state.get("ewma", {})
        tracker._history = state.get("history", [])
        log.info("[AgentWeightTracker] State loaded <- %s", path)
        return tracker


__all__ = [
    "AgentWeightTracker",
    "attach_confidence",
    "compute_confidence",
    "compute_expected_value",
    "get_portfolio_weights",
    "run_backtest",
    "select_portfolio",
    "simulate_tp_sl",
]
