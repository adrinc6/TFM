"""Review, rebalance and candidate selection helpers."""

from __future__ import annotations

import pandas as pd

from module.strategy.portfolio import manager_score


def universe_review_rows(date: str, universe: pd.DataFrame) -> list[dict]:
    today = universe[universe["snapshot_date"] == date].copy()
    if today.empty:
        return []
    today["manager_score"] = today.apply(manager_score, axis=1)
    today = today.sort_values(["manager_score", "thesis_rank_score", "buy_today_score"], ascending=False)
    rows = []
    for rank, row in enumerate(today.itertuples(index=False), start=1):
        classification, reason = universe_action(row)
        rows.append({
            "date": date,
            "review_type": getattr(row, "review_type", ""),
            "rank": rank,
            "ticker": row.ticker,
            "universe_action": classification,
            "action_reason": reason,
            "manager_score": float(row.manager_score),
            "thesis_rank_score": float(row.thesis_rank_score),
            "conviction_score": float(row.conviction_score),
            "business_quality_score": float(row.business_quality_score),
            "alpha_probability": float(getattr(row, "alpha_probability", row.final_score)),
            "valuation_score": float(row.valuation_score),
            "price_adjusted_valuation_score": float(getattr(row, "price_adjusted_valuation_score", row.valuation_score)),
            "momentum_score": float(getattr(row, "momentum_score", 0.5)),
            "price_return_3m": float(getattr(row, "price_return_3m", 0)),
            "price_return_6m": float(getattr(row, "price_return_6m", 0)),
            "price_return_12m": float(getattr(row, "price_return_12m", 0)),
            "price": float(row.price) if pd.notna(row.price) else None,
            "price_return_since_fundamental": float(getattr(row, "price_return_since_fundamental", 0)),
            "fundamental_asof_date": getattr(row, "fundamental_asof_date", ""),
            "stale_fundamental_months": float(getattr(row, "stale_fundamental_months", 0)),
            "thesis_state": row.thesis_state,
            "would_buy_today": bool(row.would_buy_today),
            "buy_today_score": float(row.buy_today_score),
            "exit_score": float(row.exit_score),
            "opportunity_type": row.opportunity_type,
            "best_alternative_ticker": row.best_alternative_ticker,
            "opportunity_cost_score": float(row.opportunity_cost_score),
            "investment_thesis": row.investment_thesis,
            "exit_thesis": row.exit_thesis,
        })
    return rows


def universe_action(row) -> tuple[str, str]:
    adjusted_valuation = float(getattr(row, "price_adjusted_valuation_score", row.valuation_score))
    momentum = float(getattr(row, "momentum_score", 0.5))
    if row.thesis_state == "Broken" or row.exit_score >= 0.66:
        return "SELL", "Thesis or exit score says capital should leave"
    if momentum < 0.20 and row.thesis_state == "Weakening":
        return "SELL", "Momentum and thesis are both deteriorating"
    if adjusted_valuation < 0.20 and not bool(row.would_buy_today):
        return "SELL", "Price-adjusted valuation is no longer attractive"
    if bool(row.would_buy_today) and row.manager_score >= 0.62 and row.thesis_state in {"Improving", "Intact"}:
        return "BUY", "Independent universe review marks it as a fresh opportunity"
    if row.thesis_state in {"Improving", "Intact", "Maturing"} and row.manager_score >= 0.48:
        return "HOLD", "Business remains holdable but not a fresh buy"
    return "AVOID", "Insufficient quality, valuation, thesis health or opportunity cost"


def decision_rows(date: str, portfolio: dict[str, dict]) -> list[dict]:
    rows = []
    for ticker, position in portfolio.items():
        decision, reason = manager_decision(position)
        rows.append({
            "date": date,
            "ticker": ticker,
            "state": position["current_thesis_state"],
            "conviction_score": position["current_conviction_score"],
            "manager_score": position.get("current_manager_score"),
            "business_quality_score": position.get("current_business_quality_score"),
            "risk_score": position.get("current_risk_score"),
            "expectation_gap": position.get("current_expectation_gap"),
            "thesis_persistence_score": position["thesis_persistence_score"],
            "decision": decision,
            "reason": reason,
            "would_buy_today": position["current_would_buy_today"],
            "buy_today_score": position["current_buy_today_score"],
            "best_alternative_ticker": position["current_best_alternative_ticker"],
            "opportunity_cost_score": position["current_opportunity_cost_score"],
            "momentum_score": position.get("current_momentum_score"),
            "thesis": position["current_thesis"],
            "exit_thesis": position["current_exit_thesis"],
            "catalyst": position["current_catalyst"],
        })
    return rows


def review_diagnostics(date: str, portfolio: dict[str, dict], universe: pd.DataFrame) -> list[dict]:
    today = universe[universe["snapshot_date"] == date].copy()
    if not today.empty and "manager_score" not in today.columns:
        today["manager_score"] = today.apply(manager_score, axis=1)
    today = today.sort_values(["manager_score", "thesis_rank_score"], ascending=False)
    if today.empty or not portfolio:
        return []
    weakest_ticker, weakest_position = min(
        portfolio.items(),
        key=lambda item: item[1].get("current_manager_score", item[1]["current_thesis_rank_score"]),
    )
    rows = []
    for rank, row in enumerate(today.head(20).itertuples(index=False), start=1):
        in_portfolio = row.ticker in portfolio
        weakest_score = float(weakest_position.get("current_manager_score", weakest_position["current_thesis_rank_score"]))
        advantage = float(getattr(row, "manager_score", row.thesis_rank_score)) - weakest_score
        rows.append({
            "date": date,
            "rank": rank,
            "ticker": row.ticker,
            "in_portfolio": in_portfolio,
            "thesis_rank_score": float(row.thesis_rank_score),
            "manager_score": float(getattr(row, "manager_score", row.thesis_rank_score)),
            "conviction_score": float(row.conviction_score),
            "would_buy_today": bool(row.would_buy_today),
            "buy_today_score": float(row.buy_today_score),
            "opportunity_type": row.opportunity_type,
            "weakest_holding": weakest_ticker,
            "weakest_holding_score": weakest_score,
            "score_advantage_vs_weakest": advantage,
            "replacement_candidate": (not in_portfolio) and advantage >= 0.06 and bool(row.would_buy_today),
            "reason": review_reason(in_portfolio, advantage),
        })
    return rows


def review_reason(in_portfolio: bool, advantage: float) -> str:
    if in_portfolio:
        return "Already held"
    if advantage >= 0.06:
        return "Materially better than weakest holding"
    return "Not enough advantage to justify rotation"


def top_candidates(universe_scores: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    if universe_scores.empty:
        return universe_scores
    return (
        universe_scores.sort_values(["date", "manager_score", "thesis_rank_score"], ascending=[True, False, False])
        .groupby("date", as_index=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def rebalance_report(transactions: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tx in transactions.itertuples(index=False):
        rows.append({
            "date": tx.date,
            "section": "ADD" if tx.action == "BUY" else "SELL",
            "ticker": tx.ticker,
            "reason": tx.reason,
            "thesis": getattr(tx, "thesis", ""),
            "exit_thesis": getattr(tx, "exit_thesis", ""),
            "catalyst": getattr(tx, "catalyst", ""),
            "would_buy_today": getattr(tx, "would_buy_today", ""),
            "buy_today_score": getattr(tx, "buy_today_score", ""),
        })
    for hold in decisions.itertuples(index=False):
        rows.append({
            "date": hold.date,
            "section": hold.decision,
            "ticker": hold.ticker,
            "reason": hold.reason,
            "thesis": getattr(hold, "thesis", ""),
            "exit_thesis": getattr(hold, "exit_thesis", ""),
            "catalyst": getattr(hold, "catalyst", ""),
            "would_buy_today": getattr(hold, "would_buy_today", ""),
            "buy_today_score": getattr(hold, "buy_today_score", ""),
        })
    return pd.DataFrame(rows)


def manager_decision(position: dict) -> tuple[str, str]:
    if position["current_thesis_state"] == "Improving" and position.get("current_would_buy_today") and position.get("current_manager_score", 0) >= 0.65:
        return "ADD", "Thesis is improving and still deserves fresh capital"
    if (
        position["current_thesis_state"] == "Maturing"
        or position["current_conviction_score"] < 0.50
        or position.get("current_manager_score", 1) < 0.50
        or position["thesis_deterioration_count"] >= 2
        or position.get("not_buy_today_count", 0) >= 2
    ):
        return "REDUCE", "Risk, opportunity cost, overvaluation or thesis maturity requires lower exposure"
    if not position.get("current_would_buy_today"):
        return "WATCH", "Existing thesis is holdable, but the manager would not add new capital today"
    return "HOLD", "Thesis remains investable and competitive versus alternatives"
