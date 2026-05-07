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
    weight_by: str = "confidence",
    max_weight: float = 1.0,
) -> pd.Series:
    """Compute normalized portfolio weights proportional to confidence.

    Weights are computed as ``w_i = P_i / sum(P_all)`` when ``weight_by`` is
    ``"confidence"``, implementing the probabilistic portfolio sizing rule from
    the spec.  Optional ``max_weight`` cap prevents excessive concentration.

    Args:
        signals (pd.DataFrame): Signal DataFrame with a ``"ticker"`` column and
            a column matching ``weight_by``.
        selected_only (bool): If True, restricts to rows where ``selected``
            is ``True``.
        weight_by (str): Column to use for weight proportionality.  Defaults to
            ``"confidence"`` (pure P_i-proportional weighting).  Falls back to
            equal weights when the column is absent or all-zero.
        max_weight (float): Maximum weight per position (0, 1].  Positions
            exceeding this cap are truncated and the surplus is redistributed
            proportionally.  Defaults to 1.0 (cap disabled).

    Returns:
        pd.Series: Normalised weights indexed by ticker.
    """
    df = signals.copy()
    if selected_only and "selected" in df.columns:
        df = df[df["selected"].astype(bool)]

    if df.empty:
        return pd.Series(dtype=float)

    if weight_by in df.columns:
        raw = pd.to_numeric(df[weight_by], errors="coerce").clip(0.0, None)
        total = float(raw.sum())
        if total > 0:
            weights = np.asarray(raw / total, dtype=float).copy()
        else:
            weights = np.full(len(df), 1.0 / len(df), dtype=float)
    else:
        weights = np.full(len(df), 1.0 / len(df), dtype=float)

    # Apply max-weight cap iteratively (water-filling redistribution).
    # At most len(weights) rounds are needed because each round converts at
    # least one position from "over" to "capped", reducing remaining iterations.
    # In practice the loop terminates in 1–3 rounds for typical portfolios.
    max_w = float(np.clip(max_weight, 1e-6, 1.0))
    for _ in range(len(weights)):
        over = weights > max_w
        if not over.any():
            break
        excess = float((weights[over] - max_w).sum())
        weights[over] = max_w
        under = ~over
        if under.any():
            under_sum = float(weights[under].sum())
            if under_sum > 0:
                weights[under] += excess * weights[under] / under_sum
        else:
            break

    total_after = float(weights.sum())
    if total_after > 0:
        weights = weights / total_after
    else:
        weights = np.full(len(df), 1.0 / len(df))

    result = pd.Series(weights, index=df["ticker"].values, name="weight")
    return result


def apply_regime_exposure(
    weights: pd.Series,
    regime: str,
    *,
    cash_ticker: str = "_CASH",
) -> pd.Series:
    """Scale portfolio weights by the regime-specific exposure multiplier.

    In a BEAR (Risk-Off) regime the system deploys only a fraction of capital,
    effectively parking the remainder in cash.  In BULL (Risk-On) the full
    weight allocation is used.

    Args:
        weights (pd.Series): Normalised position weights indexed by ticker.
            Must sum to approximately 1.0.
        regime (str): Current market regime label. One of ``REGIME_RISK_ON``
            (BULL), ``REGIME_NEUTRAL``, or ``REGIME_RISK_OFF`` (BEAR).
        cash_ticker (str): Placeholder label used for the uninvested cash
            residual in the returned series. Defaults to ``"_CASH"``.

    Returns:
        pd.Series: Adjusted weights where the invested fraction equals the
            regime exposure multiplier and the remainder is labelled
            ``cash_ticker``.  The series always sums to 1.0.
    """
    from module.common.regime import get_regime_exposure_multiplier
    exposure = get_regime_exposure_multiplier(str(regime))
    invested = weights * float(exposure)
    cash_residual = max(0.0, 1.0 - float(invested.sum()))
    if cash_residual > 1e-6:
        cash_entry = pd.Series({cash_ticker: cash_residual})
        return pd.concat([invested, cash_entry])
    return invested


def simulate_tp_sl(
    ticker: str,
    prices: pd.Series,
    entry_date: pd.Timestamp,
    tp_pct: float,
    sl_pct: float,
    *,
    max_holding_days: int = _DEFAULT_MAX_HOLDING_DAYS,
    min_holding_days: int = 0,
    trailing_stop_pct: float = 0.0,
    trailing_review_days: int = 30,
    trail_events: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    """Simulate whether TP or SL is hit first from entry_date onward.

    ``min_holding_days`` is a grace period: TP/SL checks are skipped for the
    first *min_holding_days* calendar days after entry, preventing early exits
    caused by normal short-term volatility.

    When ``trailing_stop_pct > 0``, reaching the TP level does not immediately
    close the position. Instead the TP becomes a hard floor for a trailing stop
    that ratchets up every ``trailing_review_days`` calendar days based on the
    running peak price. The trailing stop never moves down. Exit outcome is
    reported as "TP" (profitable exit above the original TP level).
    """
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

    use_trailing = float(trailing_stop_pct) > 0.0
    trailing_active = False
    trailing_stop_price = float(result["sl_price"])  # initialised; set properly at activation
    peak_price = entry_price
    last_review_day = 0

    def _record(event_type: str, dt: pd.Timestamp, price: float, stop: float) -> None:
        if trail_events is None:
            return
        trail_events.append({
            "ticker": ticker,
            "entry_date": pd.Timestamp(actual_entry).date(),
            "entry_price": round(entry_price, 4),
            "tp_price": round(float(result["tp_price"]), 4),
            "sl_price_original": round(float(result["sl_price"]), 4),
            "event_type": event_type,
            "event_date": pd.Timestamp(dt).date(),
            "days_from_entry": int((pd.Timestamp(dt) - actual_entry).days),
            "price": round(price, 4),
            "trailing_stop": round(stop, 4),
            "peak_price": round(peak_price, 4),
            "return_pct": round((price / entry_price - 1.0) * 100.0, 2),
        })

    window = prices.loc[(prices.index > actual_entry) & (prices.index <= expiry_ts)]
    grace = int(max(min_holding_days, 0))
    for dt, px in window.items():
        price = float(px)
        days_elapsed = int((pd.Timestamp(dt) - actual_entry).days)
        if days_elapsed < grace:
            # Track peak even during grace (trailing activates immediately after)
            if price > peak_price:
                peak_price = price
            continue

        if not trailing_active:
            # Normal TP/SL mode
            if price >= result["tp_price"]:
                if use_trailing:
                    # TP crossed: switch to trailing mode.
                    # New SL = price × (1 - trailing_pct) — gives breathing room so a
                    # tiny 0.5% retrace after crossing TP does NOT trigger exit.
                    # Hard floor: never below the original SL price.
                    trailing_active = True
                    peak_price = price
                    last_review_day = days_elapsed
                    initial_stop = price * (1.0 - float(trailing_stop_pct))
                    trailing_stop_price = max(initial_stop, float(result["sl_price"]))
                    _record("TRAILING_ACTIVATED", dt, price, trailing_stop_price)
                else:
                    result["outcome"] = "TP"
                    result["days_to_outcome"] = days_elapsed
                    result["outcome_date"] = pd.Timestamp(dt)
                    _record("EXIT_TP", dt, price, trailing_stop_price)
                    return result
            elif price <= result["sl_price"]:
                result["outcome"] = "SL"
                result["days_to_outcome"] = days_elapsed
                result["outcome_date"] = pd.Timestamp(dt)
                _record("EXIT_SL", dt, price, trailing_stop_price)
                return result
        else:
            # Trailing stop mode: ratchet up every review interval, never down.
            if price > peak_price:
                peak_price = price

            # Monthly review: ratchet trailing stop upward (never down)
            if (days_elapsed - last_review_day) >= int(trailing_review_days):
                new_stop = peak_price * (1.0 - float(trailing_stop_pct))
                # Only ratchet up if price is currently above the new level.
                # If the price has already pulled back below the new calculated stop,
                # skip the update: the old stop stays active and the position gets a
                # chance to recover. A forced exit caused by a calendar event (not a
                # market move) is an artefact we want to avoid.
                if new_stop <= price:
                    trailing_stop_price = max(trailing_stop_price, new_stop, float(result["sl_price"]))
                last_review_day = days_elapsed
                _record("REVIEW", dt, price, trailing_stop_price)

            # Daily check: did price fall below current trailing stop?
            if price <= trailing_stop_price:
                result["outcome"] = "TP"
                result["days_to_outcome"] = days_elapsed
                result["outcome_date"] = pd.Timestamp(dt)
                _record("EXIT_TP_TRAIL", dt, price, trailing_stop_price)
                return result

    if not window.empty:
        result["days_to_outcome"] = int((pd.Timestamp(window.index[-1]) - actual_entry).days)
        result["outcome_date"] = pd.Timestamp(window.index[-1])
        _record("EXIT_TIME", window.index[-1], float(window.iloc[-1]), trailing_stop_price)
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
        trailing_stop_pct = float(row.get("trailing_stop_pct", 0.0))
        prices = _extract_close(prices_dict.get(ticker))
        rows.append(
            simulate_tp_sl(
                ticker=ticker,
                prices=prices,
                entry_date=entry_date,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                max_holding_days=max_holding_days,
                trailing_stop_pct=trailing_stop_pct,
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
    "apply_regime_exposure",
    "attach_confidence",
    "compute_confidence",
    "compute_expected_value",
    "get_portfolio_weights",
    "run_backtest",
    "select_portfolio",
    "simulate_tp_sl",
]
