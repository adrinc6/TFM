"""Concentrated portfolio construction and review policy."""

from __future__ import annotations

import pandas as pd

from environment import MAX_PORTFOLIO_SIZE, MIN_PORTFOLIO_SIZE, MIN_SCORE_ADVANTAGE_TO_REPLACE

INVESTABLE_STATES = {"Improving", "Intact"}
HOLDABLE_STATES = {"Improving", "Intact", "Maturing"}
BLOCKED_OPPORTUNITIES = {"Avoid", "Value Trap"}
MIN_ENTRY_SCORE = 0.56
MIN_ENTRY_QUALITY = 0.48
MIN_HOLD_SCORE = 0.46
MIN_BUY_TODAY_SCORE = 0.54
MIN_HOLD_MONTHS_BEFORE_ROTATION = 4
REVIEW_TOP_N = max(MAX_PORTFOLIO_SIZE * 3, 20)


def initial_portfolio(universe: pd.DataFrame, snapshot_date: str) -> dict[str, dict]:
    ranked = _rank(universe, snapshot_date)
    candidates = _entry_candidates(ranked).head(MAX_PORTFOLIO_SIZE)
    if len(candidates) < MIN_PORTFOLIO_SIZE:
        raise RuntimeError(f"Not enough investable candidates at {snapshot_date} to build portfolio.")
    return {row.ticker: _position_from_row(row, snapshot_date) for row in candidates.itertuples(index=False)}


def review_portfolio(current: dict[str, dict], universe: pd.DataFrame, snapshot_date: str) -> tuple[dict[str, dict], list[dict]]:
    today = _rank(universe, snapshot_date)
    by_ticker = today.set_index("ticker")
    transactions: list[dict] = []
    updated = dict(current)

    for ticker, position in list(current.items()):
        if ticker not in by_ticker.index:
            transactions.append(_sell_missing(ticker, snapshot_date, position))
            updated.pop(ticker, None)
            continue
        row = by_ticker.loc[ticker]
        _refresh_position(position, row, snapshot_date)
        exit_reason = _exit_reason(row, position)
        if exit_reason:
            transactions.append(_sell(ticker, snapshot_date, row, exit_reason))
            updated.pop(ticker, None)

    for row in _entry_candidates(today).head(REVIEW_TOP_N).itertuples(index=False):
        if row.ticker in updated:
            continue
        if len(updated) >= MAX_PORTFOLIO_SIZE:
            replacement = _replacement_target(updated, row)
            if replacement is None:
                continue
            sell_ticker, sell_position = replacement
            transactions.append(_sell(sell_ticker, snapshot_date, by_ticker.loc[sell_ticker], _replacement_reason(row, sell_position)))
            updated.pop(sell_ticker, None)
        updated[row.ticker] = _position_from_row(row, snapshot_date)
        transactions.append(_buy(row, snapshot_date))
        if len(updated) >= MAX_PORTFOLIO_SIZE:
            break

    return updated, transactions


def manager_score(row) -> float:
    """Live manager score.

    The ML block is `final_score`, the output of the learned meta-agent (the four specialist
    agents combined with weights learned walk-forward from realized forward alpha, see
    module/ml.py::_meta_agent_scores). Weight raised from 0.40 to 0.45 (2026-07 diagnostic round):
    on the full-universe run, the underlying alpha_probability agent's OOS rank-IC was positive in
    6 of 8 calendar years (2019-2020, 2022-2025), negative only in 2021 (a sharp growth-to-value
    rotation) and 2026 (partial year, 5 obs) — consistent enough to lean on it a bit more, but the
    2021 miss argues against going further than 0.45 with ~40 months of live-portfolio data. The
    remaining factors are manager overlays (timing/valuation/risk) not captured by the ML score.
    """
    learned_ml = getattr(row, "final_score", 0.5)
    return float((
        0.45 * learned_ml
        + 0.13 * row.thesis_rank_score
        + 0.11 * row.buy_today_score
        + 0.09 * getattr(row, "momentum_score", 0.5)
        + 0.08 * getattr(row, "price_adjusted_valuation_score", row.valuation_score)
        + 0.05 * row.positive_expectation_gap
        + 0.05 * row.moat_score
        + 0.04 * row.risk_score
    ))


def _rank(universe: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    today = universe[universe["snapshot_date"] == snapshot_date].copy()
    if today.empty:
        return today
    today["manager_score"] = today.apply(manager_score, axis=1)
    return today.sort_values(["manager_score", "thesis_rank_score", "buy_today_score"], ascending=False)


def _entry_candidates(universe: pd.DataFrame) -> pd.DataFrame:
    if universe.empty:
        return universe
    return universe[
        universe["thesis_state"].isin(INVESTABLE_STATES)
        & ~universe["opportunity_type"].isin(BLOCKED_OPPORTUNITIES)
        & (universe["business_quality_score"] >= MIN_ENTRY_QUALITY)
        & (universe["buy_today_score"] >= MIN_BUY_TODAY_SCORE)
        & (universe["manager_score"] >= MIN_ENTRY_SCORE)
        & (
            (universe.get("momentum_score", pd.Series(0.5, index=universe.index)) >= 0.35)
            | (
                (universe.get("price_adjusted_valuation_score", universe["valuation_score"]) >= 0.72)
                & (universe["business_quality_score"] >= 0.58)
            )
        )
    ]


def _refresh_position(position: dict, row: pd.Series, snapshot_date: str) -> None:
    position["months_since_entry"] += 1
    position["last_snapshot_date"] = snapshot_date
    fields = {
        "current_thesis_state": row["thesis_state"],
        "current_conviction_score": float(row["conviction_score"]),
        "current_thesis_rank_score": float(row["thesis_rank_score"]),
        "current_manager_score": float(row["manager_score"]),
        "current_would_buy_today": bool(row["would_buy_today"]),
        "current_buy_today_score": float(row["buy_today_score"]),
        "current_best_alternative_ticker": row["best_alternative_ticker"],
        "current_opportunity_cost_score": float(row["opportunity_cost_score"]),
        "current_business_quality_score": float(row["business_quality_score"]),
        "current_risk_score": float(row["risk_score"]),
        "current_valuation_score": float(row["valuation_score"]),
        "current_price_adjusted_valuation_score": float(row.get("price_adjusted_valuation_score", row["valuation_score"])),
        "current_momentum_score": float(row.get("momentum_score", 0.5)),
        "current_price_return_3m": float(row.get("price_return_3m", 0)),
        "current_price_return_6m": float(row.get("price_return_6m", 0)),
        "current_price_return_12m": float(row.get("price_return_12m", 0)),
        "current_price_return_since_fundamental": float(row.get("price_return_since_fundamental", 0)),
        "current_stale_fundamental_months": float(row.get("stale_fundamental_months", 0)),
        "current_expectation_gap": float(row["expectation_gap"]),
        "current_positive_expectation_gap": float(row["positive_expectation_gap"]),
        "current_thesis": row["investment_thesis"],
        "current_exit_thesis": row["exit_thesis"],
        "current_catalyst": row["catalyst"],
        "sector": row.get("sector", "Unknown"),
    }
    position.update(fields)
    if row["thesis_state"] in HOLDABLE_STATES:
        position["months_thesis_intact"] += 1
        position["thesis_improvement_count"] += int(row["thesis_state"] == "Improving")
    else:
        position["thesis_deterioration_count"] += 1
    if not bool(row["would_buy_today"]):
        position["not_buy_today_count"] += 1
    else:
        position["not_buy_today_count"] = 0
    position["thesis_persistence_score"] = position["months_thesis_intact"] / max(position["months_since_entry"], 1)


def _exit_reason(row: pd.Series, position: dict) -> str | None:
    adjusted_valuation = float(row.get("price_adjusted_valuation_score", row["valuation_score"]))
    momentum = float(row.get("momentum_score", 0.5))
    if row["thesis_state"] == "Broken":
        return "Thesis Broken"
    if row["exit_score"] >= 0.66:
        return "Exit Score Trigger"
    if adjusted_valuation < 0.20 and not bool(row["would_buy_today"]):
        return "Price Adjusted Valuation No Longer Attractive"
    if momentum < 0.35 and row["thesis_state"] == "Weakening":
        # Threshold raised from 0.20 to 0.35 (2026-07 diagnostic round): this trigger had by far
        # the worst mean excess return per sale (-26.4%) of any exit reason on the full-universe
        # run, indicating it fired only after most of the relative damage was already priced in.
        # Reacting at the first sign of weakening thesis + fading momentum (rather than waiting for
        # momentum to collapse below 0.20) should cut losses earlier.
        return "Momentum And Thesis Deterioration"
    if row["manager_score"] < MIN_HOLD_SCORE and not bool(row["would_buy_today"]):
        return "Manager Score Below Hold Hurdle"
    if position["not_buy_today_count"] >= 4 and row["opportunity_cost_score"] >= 0.08:
        return "Persistent Better Use Of Capital"
    if row["thesis_state"] == "Weakening" and position["thesis_deterioration_count"] >= 3:
        return "Repeated Thesis Deterioration"
    return None


def _replacement_target(updated: dict[str, dict], candidate) -> tuple[str, dict] | None:
    eligible = {
        ticker: position
        for ticker, position in updated.items()
        if position.get("months_since_entry", 0) >= MIN_HOLD_MONTHS_BEFORE_ROTATION
        or position.get("current_thesis_state") in {"Broken", "Weakening"}
    }
    if not eligible:
        return None
    weakest = min(eligible.items(), key=lambda item: item[1]["current_manager_score"])
    score_advantage = float(candidate.manager_score) - weakest[1]["current_manager_score"]
    conviction_advantage = float(candidate.conviction_score) - weakest[1]["current_conviction_score"]
    momentum_advantage = float(getattr(candidate, "momentum_score", 0.5)) - weakest[1].get("current_momentum_score", 0.5)
    better_capital_use = score_advantage >= max(MIN_SCORE_ADVANTAGE_TO_REPLACE, 0.09) or (
        score_advantage >= 0.06 and conviction_advantage >= 0.04 and momentum_advantage >= 0.05 and bool(candidate.would_buy_today)
    )
    if better_capital_use:
        return weakest
    return None


def _replacement_reason(candidate, weakest_position: dict) -> str:
    advantage = float(candidate.manager_score) - weakest_position["current_manager_score"]
    return f"Opportunity Cost: {candidate.ticker} manager_score advantage {advantage:.2f}"


def _position_from_row(row, snapshot_date: str) -> dict:
    return {
        "ticker": row.ticker,
        "entry_date": snapshot_date,
        "original_snapshot_date": snapshot_date,
        "original_thesis": row.thesis_state,
        "investment_thesis": row.investment_thesis,
        "bull_thesis": row.bull_thesis,
        "bear_thesis": row.bear_thesis,
        "catalyst": row.catalyst,
        "sector": getattr(row, "sector", "Unknown"),
        "moat_analysis": row.moat_analysis,
        "exit_thesis": row.exit_thesis,
        "entry_trigger": row.entry_trigger,
        "would_buy_today": bool(row.would_buy_today),
        "buy_today_score": float(row.buy_today_score),
        "manager_score": float(row.manager_score),
        "best_alternative_ticker": row.best_alternative_ticker,
        "opportunity_cost_score": float(row.opportunity_cost_score),
        "original_scores": {
            "final_score": float(row.final_score),
            "thesis_score": float(row.thesis_score),
            "conviction_score": float(row.conviction_score),
            "business_quality_score": float(row.business_quality_score),
            "manager_score": float(row.manager_score),
            "price_adjusted_valuation_score": float(getattr(row, "price_adjusted_valuation_score", row.valuation_score)),
        },
        "opportunity_type_original": row.opportunity_type,
        "buy_reason": row.buy_reason,
        "months_since_entry": 0,
        "months_thesis_intact": 0,
        "thesis_persistence_score": 1.0,
        "thesis_improvement_count": 0,
        "thesis_deterioration_count": 0,
        "not_buy_today_count": 0,
        "current_thesis_state": row.thesis_state,
        "current_conviction_score": float(row.conviction_score),
        "current_thesis_rank_score": float(row.thesis_rank_score),
        "current_manager_score": float(row.manager_score),
        "current_would_buy_today": bool(row.would_buy_today),
        "current_buy_today_score": float(row.buy_today_score),
        "current_best_alternative_ticker": row.best_alternative_ticker,
        "current_opportunity_cost_score": float(row.opportunity_cost_score),
        "current_business_quality_score": float(row.business_quality_score),
        "current_risk_score": float(row.risk_score),
        "current_valuation_score": float(row.valuation_score),
        "current_price_adjusted_valuation_score": float(getattr(row, "price_adjusted_valuation_score", row.valuation_score)),
        "current_momentum_score": float(getattr(row, "momentum_score", 0.5)),
        "current_price_return_3m": float(getattr(row, "price_return_3m", 0)),
        "current_price_return_6m": float(getattr(row, "price_return_6m", 0)),
        "current_price_return_12m": float(getattr(row, "price_return_12m", 0)),
        "current_price_return_since_fundamental": float(getattr(row, "price_return_since_fundamental", 0)),
        "current_stale_fundamental_months": float(getattr(row, "stale_fundamental_months", 0)),
        "current_expectation_gap": float(row.expectation_gap),
        "current_positive_expectation_gap": float(row.positive_expectation_gap),
        "current_thesis": row.investment_thesis,
        "current_exit_thesis": row.exit_thesis,
        "current_catalyst": row.catalyst,
    }


def _buy(row, snapshot_date: str) -> dict:
    return {
        "date": snapshot_date,
        "ticker": row.ticker,
        "action": "BUY",
        "reason": row.buy_reason,
        "thesis": row.investment_thesis,
        "catalyst": row.catalyst,
        "exit_thesis": row.exit_thesis,
        "would_buy_today": bool(row.would_buy_today),
        "buy_today_score": float(row.buy_today_score),
        "manager_score": float(row.manager_score),
        "best_alternative_ticker": row.best_alternative_ticker,
        "opportunity_cost_score": float(row.opportunity_cost_score),
    }


def _sell(ticker: str, snapshot_date: str, row, reason: str) -> dict:
    return {
        "date": snapshot_date,
        "ticker": ticker,
        "action": "SELL",
        "reason": f"{reason}: state={row['thesis_state']}, exit={row['exit_score']:.2f}, manager={row['manager_score']:.2f}",
        "thesis": row["investment_thesis"],
        "exit_thesis": row["exit_thesis"],
        "catalyst": row["catalyst"],
    }


def _sell_missing(ticker: str, snapshot_date: str, position: dict) -> dict:
    return {
        "date": snapshot_date,
        "ticker": ticker,
        "action": "SELL",
        "reason": "No current data available for review",
        "thesis": position.get("current_thesis", ""),
        "exit_thesis": position.get("current_exit_thesis", ""),
        "catalyst": position.get("current_catalyst", ""),
    }
