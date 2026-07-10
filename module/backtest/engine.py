"""Live portfolio simulation against benchmark."""

from __future__ import annotations

import logging

import pandas as pd

from environment import PROCESSED_DIR, RAW_DIR, Settings
from module.research.thesis import enrich_with_thesis_scores
from module.strategy.portfolio import initial_portfolio, review_portfolio
from module.strategy.sizing import add_position_sizing
from module.utils import read_parquet, write_json

from .artifacts import (
    action_journal,
    buy_rationale,
    current_portfolio,
    executive_summary_table,
    improvement_backlog,
    latest_top_opportunities,
    sector_exposure,
    sell_reasons_summary,
    strategy_learning_log,
    summary_metrics,
    tracking_dashboard,
)
from .performance import portfolio_vs_benchmark, position_performance, turnover
from .reviews import decision_rows, rebalance_report, review_diagnostics, top_candidates, universe_review_rows
from .robustness import build_robustness

log = logging.getLogger(__name__)


AUDIT_OUTPUTS = {
    "portfolio_allocation",
    "portfolio_decision_log",
    "portfolio_evolution",
    "portfolio_monthly_holdings",
    "portfolio_review_diagnostics",
    "portfolio_transactions",
    "portfolio_turnover",
    "rebalance_report",
    "universe_monthly_scores",
    "universe_monthly_price_update",
    "universe_quarterly_fundamental_review",
    "universe_top_candidates",
    "robustness_bootstrap_distribution",
}


def run_backtest(settings: Settings) -> dict[str, pd.DataFrame]:
    scored = enrich_with_thesis_scores(read_parquet(PROCESSED_DIR / "scored_universe.parquet"))
    prices = read_parquet(RAW_DIR / "prices.parquet")
    dates = [date for date in sorted(scored["snapshot_date"].unique()) if date >= settings.start_date]
    if not dates:
        raise RuntimeError("No snapshot dates available for backtest.")
    log.info("Backtest starting snapshots=%s start=%s end=%s universe_rows=%s", len(dates), dates[0], dates[-1], len(scored))
    portfolio = initial_portfolio(scored, dates[0])
    transactions = [{
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
    } for ticker, pos in portfolio.items()]
    evolution: list[dict] = []
    holdings: list[dict] = []
    decisions: list[dict] = []
    review_rows: list[dict] = []
    universe_reviews: list[dict] = []
    for date in dates:
        universe_reviews.extend(universe_review_rows(date, scored))
        if date != dates[0]:
            portfolio, tx = review_portfolio(portfolio, scored, date)
            transactions.extend(tx)
        evolution.append({"date": date, "holdings": len(portfolio), "tickers": ", ".join(sorted(portfolio))})
        holdings.extend({"date": date, **position} for position in portfolio.values())
        decisions.extend(decision_rows(date, portfolio))
        review_rows.extend(review_diagnostics(date, portfolio, scored))
    run_dir = settings.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    clean_managed_outputs(run_dir)
    outputs = {
        "portfolio_evolution": pd.DataFrame(evolution),
        "portfolio_transactions": pd.DataFrame(transactions),
        "portfolio_monthly_holdings": add_position_sizing(pd.DataFrame(holdings)),
        "portfolio_decision_log": pd.DataFrame(decisions),
        "portfolio_review_diagnostics": pd.DataFrame(review_rows),
        "universe_monthly_scores": pd.DataFrame(universe_reviews),
    }
    allocation_columns = [
        "date", "ticker", "equal_weight", "conviction_weight", "risk_adjusted_weight", "hybrid_weight",
        "sizing_score", "position_action", "current_conviction_score", "current_manager_score",
        "current_buy_today_score", "current_opportunity_cost_score", "current_price_adjusted_valuation_score",
        "current_price_return_since_fundamental", "current_stale_fundamental_months", "current_momentum_score",
        "current_price_return_3m", "current_price_return_6m", "current_price_return_12m",
    ]
    outputs["portfolio_allocation"] = outputs["portfolio_monthly_holdings"][[col for col in allocation_columns if col in outputs["portfolio_monthly_holdings"].columns]]
    outputs["position_performance"] = position_performance(outputs["portfolio_transactions"], outputs["portfolio_evolution"], prices, settings.benchmark_ticker)
    outputs["rebalance_report"] = rebalance_report(outputs["portfolio_transactions"], outputs["portfolio_decision_log"])
    outputs["universe_top_candidates"] = top_candidates(outputs["universe_monthly_scores"])
    outputs["universe_quarterly_fundamental_review"] = outputs["universe_monthly_scores"][outputs["universe_monthly_scores"].get("review_type", pd.Series(dtype=str)) == "quarterly_fundamental_review"].copy()
    outputs["universe_monthly_price_update"] = outputs["universe_monthly_scores"][outputs["universe_monthly_scores"].get("review_type", pd.Series(dtype=str)) == "monthly_price_update"].copy()
    outputs["portfolio_vs_benchmark"] = portfolio_vs_benchmark(
        outputs["portfolio_evolution"],
        prices,
        settings.benchmark_ticker,
        outputs["portfolio_transactions"],
        settings,
        weights=outputs["portfolio_monthly_holdings"],
    )
    outputs["portfolio_turnover"] = turnover(outputs["portfolio_transactions"], len(dates), outputs["position_performance"])
    outputs["tracking_dashboard"] = tracking_dashboard(outputs)
    outputs["buy_rationale"] = buy_rationale(outputs["portfolio_transactions"], outputs["universe_monthly_scores"])
    outputs["action_journal"] = action_journal(outputs["portfolio_transactions"], outputs["position_performance"], outputs["buy_rationale"])
    outputs["sell_reasons_summary"] = sell_reasons_summary(outputs["portfolio_transactions"])
    outputs["sector_exposure"] = sector_exposure(outputs["portfolio_monthly_holdings"])
    outputs["portfolio_monthly_summary"] = summary_metrics(outputs)
    outputs["current_portfolio"] = current_portfolio(outputs["portfolio_monthly_holdings"])
    outputs["top_opportunities_latest"] = latest_top_opportunities(outputs["universe_top_candidates"])
    outputs["executive_summary"] = executive_summary_table(outputs)
    outputs["strategy_learning_log"] = strategy_learning_log(outputs)
    outputs["improvement_backlog"] = improvement_backlog(outputs)
    # Statistical robustness (bootstrap CI, cost-sensitivity breakeven) computed from the finished
    # tables above — no re-scoring or re-selection; see module/backtest/robustness.py.
    outputs.update(build_robustness(outputs, settings))
    for name, df in outputs.items():
        if isinstance(df, pd.DataFrame):
            output_dir = run_dir / "audit" if name in AUDIT_OUTPUTS else run_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_dir / f"{name}.csv", index=False)
    write_json(outputs["portfolio_monthly_summary"], run_dir / "portfolio_monthly_summary.json")
    return outputs


def clean_managed_outputs(run_dir) -> None:
    managed_names = {
        "action_journal", "buy_rationale", "current_portfolio", "executive_summary", "portfolio_allocation",
        "portfolio_decision_log", "portfolio_evolution", "portfolio_monthly_holdings", "portfolio_review_diagnostics",
        "portfolio_transactions", "portfolio_turnover", "portfolio_vs_benchmark", "position_performance",
        "rebalance_report", "sector_exposure", "sell_reasons_summary", "strategy_learning_log", "improvement_backlog",
        "top_opportunities_latest", "tracking_dashboard", "universe_monthly_price_update", "universe_monthly_scores",
        "universe_quarterly_fundamental_review", "universe_top_candidates",
        "robustness_bootstrap", "robustness_bootstrap_distribution", "robustness_cost_sensitivity", "robustness_summary",
    }
    for name in managed_names:
        for path in (run_dir / f"{name}.csv", run_dir / "audit" / f"{name}.csv"):
            if path.exists():
                path.unlink()
