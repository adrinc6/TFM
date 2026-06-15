"""Concentrated portfolio construction and review policy."""

from __future__ import annotations

import pandas as pd

from environment import MAX_PORTFOLIO_SIZE, MIN_PORTFOLIO_SIZE, MIN_SCORE_ADVANTAGE_TO_REPLACE


def initial_portfolio(universe: pd.DataFrame, snapshot_date: str) -> dict[str, dict]:
    ranked = _rank(universe, snapshot_date)
    candidates = _investable(ranked).head(MAX_PORTFOLIO_SIZE)
    if len(candidates) < MIN_PORTFOLIO_SIZE:
        fillers = ranked[
            ~ranked["ticker"].isin(candidates["ticker"])
            & ~ranked["opportunity_type"].isin({"Avoid", "Value Trap"})
            & (ranked["business_quality_score"] >= 0.40)
        ]
        candidates = pd.concat([candidates, fillers]).head(MAX_PORTFOLIO_SIZE)
    if len(candidates) < MIN_PORTFOLIO_SIZE:
        research_candidates = ranked[
            ~ranked["ticker"].isin(candidates["ticker"])
            & ~ranked["opportunity_type"].isin({"Avoid", "Value Trap"})
            & (ranked["business_quality_score"] >= 0.30)
        ]
        candidates = pd.concat([candidates, research_candidates]).head(MAX_PORTFOLIO_SIZE)
    if len(candidates) < MIN_PORTFOLIO_SIZE:
        raise RuntimeError(f"Not enough candidates at {snapshot_date} to build portfolio.")
    return {row.ticker: _position_from_row(row, snapshot_date) for row in candidates.itertuples(index=False)}


def review_portfolio(current: dict[str, dict], universe: pd.DataFrame, snapshot_date: str) -> tuple[dict[str, dict], list[dict]]:
    today = _rank(universe, snapshot_date)
    by_ticker = today.set_index("ticker")
    transactions: list[dict] = []
    updated = dict(current)

    for ticker, position in list(current.items()):
        if ticker not in by_ticker.index:
            continue
        row = by_ticker.loc[ticker]
        position["months_since_entry"] += 3
        position["last_snapshot_date"] = snapshot_date
        position["current_thesis_state"] = row["thesis_state"]
        position["current_conviction_score"] = float(row["conviction_score"])
        position["current_thesis_rank_score"] = float(row["thesis_rank_score"])
        position["current_would_buy_today"] = bool(row["would_buy_today"])
        position["current_buy_today_score"] = float(row["buy_today_score"])
        position["current_best_alternative_ticker"] = row["best_alternative_ticker"]
        position["current_opportunity_cost_score"] = float(row["opportunity_cost_score"])
        position["current_thesis"] = row["investment_thesis"]
        position["current_catalyst"] = row["catalyst"]
        if row["thesis_state"] in {"Improving", "Intact", "Maturing"}:
            position["months_thesis_intact"] += 3
            position["thesis_improvement_count"] += int(row["thesis_state"] == "Improving")
        else:
            position["thesis_deterioration_count"] += 1
        position["thesis_persistence_score"] = position["months_thesis_intact"] / max(position["months_since_entry"], 1)
        if row["thesis_state"] == "Broken" or row["exit_score"] >= 0.70:
            transactions.append(_sell(ticker, snapshot_date, row, "Thesis Broken" if row["thesis_state"] == "Broken" else "Thesis Deterioration"))
            updated.pop(ticker, None)

    for row in today.itertuples(index=False):
        if len(updated) >= MAX_PORTFOLIO_SIZE:
            weakest = min(updated.items(), key=lambda item: item[1]["current_conviction_score"])
            advantage = float(row.thesis_rank_score) - weakest[1]["current_thesis_rank_score"]
            if row.ticker not in updated and advantage >= MIN_SCORE_ADVANTAGE_TO_REPLACE and row.business_quality_score >= 0.55:
                transactions.append(_sell(weakest[0], snapshot_date, by_ticker.loc[weakest[0]], "Opportunity Cost"))
                updated.pop(weakest[0], None)
            else:
                continue
        if (
            row.ticker not in updated
            and len(updated) < MAX_PORTFOLIO_SIZE
            and row.thesis_state in {"Improving", "Intact"}
            and row.business_quality_score >= 0.55
        ):
            updated[row.ticker] = _position_from_row(row, snapshot_date)
            transactions.append(_buy(row, snapshot_date))
        if len(updated) >= MAX_PORTFOLIO_SIZE:
            break

    return updated, transactions


def _rank(universe: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    return universe[universe["snapshot_date"] == snapshot_date].sort_values("thesis_rank_score", ascending=False)


def _investable(universe: pd.DataFrame) -> pd.DataFrame:
    return universe[
        universe["thesis_state"].isin({"Improving", "Intact", "Maturing"})
        & ~universe["opportunity_type"].isin({"Avoid", "Value Trap"})
        & (universe["business_quality_score"] >= 0.50)
    ]


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
        "moat_analysis": row.moat_analysis,
        "exit_thesis": row.exit_thesis,
        "entry_trigger": row.entry_trigger,
        "would_buy_today": bool(row.would_buy_today),
        "buy_today_score": float(row.buy_today_score),
        "best_alternative_ticker": row.best_alternative_ticker,
        "opportunity_cost_score": float(row.opportunity_cost_score),
        "original_scores": {
            "final_score": float(row.final_score),
            "thesis_score": float(row.thesis_score),
            "conviction_score": float(row.conviction_score),
            "business_quality_score": float(row.business_quality_score),
        },
        "opportunity_type_original": row.opportunity_type,
        "buy_reason": row.buy_reason,
        "months_since_entry": 0,
        "months_thesis_intact": 0,
        "thesis_persistence_score": 1.0,
        "thesis_improvement_count": 0,
        "thesis_deterioration_count": 0,
        "current_thesis_state": row.thesis_state,
        "current_conviction_score": float(row.conviction_score),
        "current_thesis_rank_score": float(row.thesis_rank_score),
        "current_would_buy_today": bool(row.would_buy_today),
        "current_buy_today_score": float(row.buy_today_score),
        "current_best_alternative_ticker": row.best_alternative_ticker,
        "current_opportunity_cost_score": float(row.opportunity_cost_score),
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
        "best_alternative_ticker": row.best_alternative_ticker,
        "opportunity_cost_score": float(row.opportunity_cost_score),
    }


def _sell(ticker: str, snapshot_date: str, row, reason: str) -> dict:
    return {
        "date": snapshot_date,
        "ticker": ticker,
        "action": "SELL",
        "reason": f"{reason}: state={row['thesis_state']}, exit={row['exit_score']:.2f}",
        "thesis": row["investment_thesis"],
        "exit_thesis": row["exit_thesis"],
        "catalyst": row["catalyst"],
    }
