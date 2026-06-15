"""Live portfolio simulation against benchmark."""

from __future__ import annotations

import logging

import pandas as pd

from environment import PROCESSED_DIR, RAW_DIR, Settings
from module.common.io import read_parquet, write_json
from module.portfolio.construction import initial_portfolio, manager_score, review_portfolio
from module.portfolio.sizing import add_position_sizing
from module.thesis.intelligence import enrich_with_thesis_scores

log = logging.getLogger(__name__)


def run_backtest(settings: Settings) -> dict[str, pd.DataFrame]:
    scored = enrich_with_thesis_scores(read_parquet(PROCESSED_DIR / "scored_universe.parquet"))
    prices = read_parquet(RAW_DIR / "prices.parquet")
    dates = [date for date in sorted(scored["snapshot_date"].unique()) if date >= settings.start_date]
    if not dates:
        raise RuntimeError("No snapshot dates available for backtest.")

    portfolio = initial_portfolio(scored, dates[0])
    transactions = [
        {
            "date": dates[0],
            "ticker": ticker,
            "action": "BUY",
            "reason": pos["buy_reason"],
            "thesis": pos["investment_thesis"],
            "exit_thesis": pos["exit_thesis"],
            "catalyst": pos["catalyst"],
            "would_buy_today": pos["would_buy_today"],
            "buy_today_score": pos["buy_today_score"],
            "manager_score": pos["manager_score"],
        }
        for ticker, pos in portfolio.items()
    ]
    evolution = []
    holdings = []
    decisions = []
    review_diagnostics = []

    for date in dates:
        if date != dates[0]:
            portfolio, tx = review_portfolio(portfolio, scored, date)
            transactions.extend(tx)
        tickers = sorted(portfolio)
        for ticker, position in portfolio.items():
            holdings.append({"date": date, **position})
        evolution.append({"date": date, "holdings": len(tickers), "tickers": ", ".join(tickers)})
        decisions.extend(_decision_rows(date, portfolio))
        review_diagnostics.extend(_review_diagnostics(date, portfolio, scored))

    run_dir = settings.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "portfolio_evolution": pd.DataFrame(evolution),
        "portfolio_transactions": pd.DataFrame(transactions),
        "portfolio_monthly_holdings": add_position_sizing(pd.DataFrame(holdings)),
        "portfolio_decision_log": pd.DataFrame(decisions),
        "portfolio_review_diagnostics": pd.DataFrame(review_diagnostics),
    }
    allocation_columns = [
        "date", "ticker", "equal_weight", "conviction_weight", "risk_adjusted_weight", "hybrid_weight",
        "sizing_score", "position_action", "current_conviction_score", "current_manager_score",
        "current_buy_today_score", "current_opportunity_cost_score",
    ]
    outputs["portfolio_allocation"] = outputs["portfolio_monthly_holdings"][[
        col for col in allocation_columns if col in outputs["portfolio_monthly_holdings"].columns
    ]]
    outputs["rebalance_report"] = _rebalance_report(outputs["portfolio_transactions"], outputs["portfolio_decision_log"])
    outputs["portfolio_vs_benchmark"] = _portfolio_vs_benchmark(outputs["portfolio_evolution"], prices, settings.benchmark_ticker)
    outputs["portfolio_turnover"] = _turnover(outputs["portfolio_transactions"], len(dates))
    outputs["portfolio_monthly_summary"] = _summary(outputs)

    for name, df in outputs.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(run_dir / f"{name}.csv", index=False)
    write_json(outputs["portfolio_monthly_summary"], run_dir / "portfolio_monthly_summary.json")
    log.info("Backtest outputs written to %s", run_dir)
    return outputs


def _decision_rows(date: str, portfolio: dict[str, dict]) -> list[dict]:
    rows = []
    for ticker, pos in portfolio.items():
        decision, reason = _manager_decision(pos)
        rows.append({
            "date": date,
            "ticker": ticker,
            "state": pos["current_thesis_state"],
            "conviction_score": pos["current_conviction_score"],
            "manager_score": pos.get("current_manager_score"),
            "business_quality_score": pos.get("current_business_quality_score"),
            "risk_score": pos.get("current_risk_score"),
            "expectation_gap": pos.get("current_expectation_gap"),
            "thesis_persistence_score": pos["thesis_persistence_score"],
            "decision": decision,
            "reason": reason,
            "would_buy_today": pos["current_would_buy_today"],
            "buy_today_score": pos["current_buy_today_score"],
            "best_alternative_ticker": pos["current_best_alternative_ticker"],
            "opportunity_cost_score": pos["current_opportunity_cost_score"],
            "thesis": pos["current_thesis"],
            "exit_thesis": pos["current_exit_thesis"],
            "catalyst": pos["current_catalyst"],
        })
    return rows


def _review_diagnostics(date: str, portfolio: dict[str, dict], universe: pd.DataFrame) -> list[dict]:
    today = universe[universe["snapshot_date"] == date].copy()
    if not today.empty and "manager_score" not in today.columns:
        today["manager_score"] = today.apply(manager_score, axis=1)
    today = today.sort_values(["manager_score", "thesis_rank_score"], ascending=False)
    if today.empty or not portfolio:
        return []
    weakest_ticker, weakest_position = min(portfolio.items(), key=lambda item: item[1].get("current_manager_score", item[1]["current_thesis_rank_score"]))
    rows = []
    for rank, row in enumerate(today.head(20).itertuples(index=False), start=1):
        in_portfolio = row.ticker in portfolio
        advantage = float(getattr(row, "manager_score", row.thesis_rank_score)) - float(weakest_position.get("current_manager_score", weakest_position["current_thesis_rank_score"]))
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
            "weakest_holding_score": float(weakest_position.get("current_manager_score", weakest_position["current_thesis_rank_score"])),
            "score_advantage_vs_weakest": advantage,
            "replacement_candidate": (not in_portfolio) and advantage >= 0.06 and bool(row.would_buy_today),
            "reason": _review_reason(in_portfolio, advantage),
        })
    return rows


def _review_reason(in_portfolio: bool, advantage: float) -> str:
    if in_portfolio:
        return "Already held"
    if advantage >= 0.06:
        return "Materially better than weakest holding"
    return "Not enough advantage to justify rotation"


def _turnover(transactions: pd.DataFrame, periods: int) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame([{"monthly_turnover": 0, "annual_turnover": 0, "average_holding_period": None, "median_holding_period": None, "churn_by_reason": ""}])
    sells = transactions[transactions["action"] == "SELL"]
    monthly = len(sells) / max(periods, 1)
    return pd.DataFrame([{
        "monthly_turnover": monthly,
        "annual_turnover": monthly * 12,
        "average_holding_period": None,
        "median_holding_period": None,
        "churn_by_reason": "; ".join(sells["reason"].value_counts().astype(str).to_dict().keys()),
    }])


def _portfolio_vs_benchmark(evolution: pd.DataFrame, prices: pd.DataFrame, benchmark_ticker: str) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    rows = []
    previous_value = 1.0
    previous_benchmark_value = 1.0
    previous_benchmark = None
    previous_date = None
    for row in evolution.itertuples(index=False):
        date = pd.to_datetime(row.date)
        tickers = [ticker.strip() for ticker in row.tickers.split(",") if ticker.strip()]
        portfolio_return = 0.0 if previous_date is None else _basket_return(prices, tickers, previous_date, date)
        benchmark_price = _last_price(prices, benchmark_ticker, date)
        benchmark_return = 0.0 if previous_benchmark is None or benchmark_price is None else benchmark_price / previous_benchmark - 1
        previous_benchmark = benchmark_price or previous_benchmark
        previous_value *= 1 + portfolio_return
        previous_benchmark_value *= 1 + benchmark_return
        previous_date = date
        rows.append({
            "date": row.date,
            "portfolio_period_return": portfolio_return,
            "benchmark_period_return": benchmark_return,
            "period_alpha": portfolio_return - benchmark_return,
            "portfolio_value": previous_value,
            "benchmark_value": previous_benchmark_value,
        })
    return pd.DataFrame(rows)


def _rebalance_report(transactions: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
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


def _manager_decision(position: dict) -> tuple[str, str]:
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


def _basket_return(prices: pd.DataFrame, tickers: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    if not tickers:
        return 0.0
    returns = []
    for ticker in tickers:
        start_price = _last_price(prices, ticker, start_date)
        end_price = _last_price(prices, ticker, end_date)
        if start_price and end_price and start_price > 0:
            returns.append(end_price / start_price - 1)
    return float(pd.Series(returns).mean()) if returns else 0.0


def _last_price(prices: pd.DataFrame, ticker: str, date: pd.Timestamp) -> float | None:
    rows = prices[(prices["ticker"] == ticker) & (prices["date"] <= date)].sort_values("date")
    if rows.empty:
        return None
    return float(rows.iloc[-1]["adj_close"])


def _summary(outputs: dict) -> dict:
    transactions = outputs["portfolio_transactions"]
    vs_benchmark = outputs["portfolio_vs_benchmark"]
    return {
        "initial_holdings": int(outputs["portfolio_evolution"].iloc[0]["holdings"]),
        "final_holdings": int(outputs["portfolio_evolution"].iloc[-1]["holdings"]),
        "transactions": int(len(transactions)),
        "buys": int((transactions["action"] == "BUY").sum()),
        "sells": int((transactions["action"] == "SELL").sum()),
        "cumulative_alpha": float(vs_benchmark["period_alpha"].sum()) if not vs_benchmark.empty else 0.0,
    }
