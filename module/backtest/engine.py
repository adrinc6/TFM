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
}


def run_backtest(settings: Settings) -> dict[str, pd.DataFrame]:
    scored = enrich_with_thesis_scores(read_parquet(PROCESSED_DIR / "scored_universe.parquet"))
    prices = read_parquet(RAW_DIR / "prices.parquet")
    dates = [date for date in sorted(scored["snapshot_date"].unique()) if date >= settings.start_date]
    if not dates:
        raise RuntimeError("No snapshot dates available for backtest.")

    log.info("Backtest starting snapshots=%s start=%s end=%s universe_rows=%s", len(dates), dates[0], dates[-1], len(scored))
    portfolio = initial_portfolio(scored, dates[0])
    log.info("Initial portfolio at %s: %s", dates[0], ", ".join(sorted(portfolio)))
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
    universe_reviews = []

    for date in dates:
        today_review = _universe_review_rows(date, scored)
        universe_reviews.extend(today_review)
        if date != dates[0]:
            before = set(portfolio)
            portfolio, tx = review_portfolio(portfolio, scored, date)
            transactions.extend(tx)
            after = set(portfolio)
            log.info(
                "Review %s tx=%s buys=%s sells=%s unchanged=%s",
                date,
                len(tx),
                sorted(after - before),
                sorted(before - after),
                sorted(before & after),
            )
        tickers = sorted(portfolio)
        for ticker, position in portfolio.items():
            holdings.append({"date": date, **position})
        evolution.append({"date": date, "holdings": len(tickers), "tickers": ", ".join(tickers)})
        decisions.extend(_decision_rows(date, portfolio))
        review_diagnostics.extend(_review_diagnostics(date, portfolio, scored))

    run_dir = settings.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    _clean_managed_outputs(run_dir)
    outputs = {
        "portfolio_evolution": pd.DataFrame(evolution),
        "portfolio_transactions": pd.DataFrame(transactions),
        "portfolio_monthly_holdings": add_position_sizing(pd.DataFrame(holdings)),
        "portfolio_decision_log": pd.DataFrame(decisions),
        "portfolio_review_diagnostics": pd.DataFrame(review_diagnostics),
        "universe_monthly_scores": pd.DataFrame(universe_reviews),
    }
    allocation_columns = [
        "date", "ticker", "equal_weight", "conviction_weight", "risk_adjusted_weight", "hybrid_weight",
        "sizing_score", "position_action", "current_conviction_score", "current_manager_score",
        "current_buy_today_score", "current_opportunity_cost_score",
        "current_price_adjusted_valuation_score", "current_price_return_since_fundamental",
        "current_stale_fundamental_months", "current_momentum_score",
        "current_price_return_3m", "current_price_return_6m", "current_price_return_12m",
    ]
    outputs["portfolio_allocation"] = outputs["portfolio_monthly_holdings"][[
        col for col in allocation_columns if col in outputs["portfolio_monthly_holdings"].columns
    ]]
    outputs["position_performance"] = _position_performance(
        outputs["portfolio_transactions"],
        outputs["portfolio_evolution"],
        prices,
        settings.benchmark_ticker,
    )
    outputs["rebalance_report"] = _rebalance_report(outputs["portfolio_transactions"], outputs["portfolio_decision_log"])
    outputs["universe_top_candidates"] = _top_candidates(outputs["universe_monthly_scores"])
    outputs["universe_quarterly_fundamental_review"] = outputs["universe_monthly_scores"][
        outputs["universe_monthly_scores"].get("review_type", pd.Series(dtype=str)) == "quarterly_fundamental_review"
    ].copy()
    outputs["universe_monthly_price_update"] = outputs["universe_monthly_scores"][
        outputs["universe_monthly_scores"].get("review_type", pd.Series(dtype=str)) == "monthly_price_update"
    ].copy()
    outputs["portfolio_vs_benchmark"] = _portfolio_vs_benchmark(
        outputs["portfolio_evolution"],
        prices,
        settings.benchmark_ticker,
        outputs["portfolio_transactions"],
        settings,
    )
    outputs["portfolio_turnover"] = _turnover(outputs["portfolio_transactions"], len(dates), outputs["position_performance"])
    outputs["tracking_dashboard"] = _tracking_dashboard(outputs)
    outputs["buy_rationale"] = _buy_rationale(outputs["portfolio_transactions"], outputs["universe_monthly_scores"])
    outputs["action_journal"] = _action_journal(
        outputs["portfolio_transactions"],
        outputs["position_performance"],
        outputs["buy_rationale"],
    )
    outputs["sell_reasons_summary"] = _sell_reasons_summary(outputs["portfolio_transactions"])
    outputs["sector_exposure"] = _sector_exposure(outputs["portfolio_monthly_holdings"])
    outputs["portfolio_monthly_summary"] = _summary(outputs)
    outputs["current_portfolio"] = _current_portfolio(outputs["portfolio_monthly_holdings"])
    outputs["top_opportunities_latest"] = _latest_top_opportunities(outputs["universe_top_candidates"])
    outputs["executive_summary"] = _executive_summary_table(outputs)
    outputs["strategy_learning_log"] = _strategy_learning_log(outputs)
    outputs["improvement_backlog"] = _improvement_backlog(outputs)

    for name, df in outputs.items():
        if isinstance(df, pd.DataFrame):
            output_dir = run_dir / "audit" if name in AUDIT_OUTPUTS else run_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_dir / f"{name}.csv", index=False)
    write_json(outputs["portfolio_monthly_summary"], run_dir / "portfolio_monthly_summary.json")
    log.info(
        "Backtest outputs written to %s transactions=%s sells=%s universe_review_rows=%s",
        run_dir,
        len(outputs["portfolio_transactions"]),
        int((outputs["portfolio_transactions"]["action"] == "SELL").sum()),
        len(outputs["universe_monthly_scores"]),
    )
    return outputs


def _clean_managed_outputs(run_dir) -> None:
    managed_names = {
        "action_journal",
        "buy_rationale",
        "current_portfolio",
        "executive_summary",
        "portfolio_allocation",
        "portfolio_decision_log",
        "portfolio_evolution",
        "portfolio_monthly_holdings",
        "portfolio_review_diagnostics",
        "portfolio_transactions",
        "portfolio_turnover",
        "portfolio_vs_benchmark",
        "position_performance",
        "rebalance_report",
        "sector_exposure",
        "sell_reasons_summary",
        "strategy_learning_log",
        "improvement_backlog",
        "top_opportunities_latest",
        "tracking_dashboard",
        "universe_monthly_price_update",
        "universe_monthly_scores",
        "universe_quarterly_fundamental_review",
        "universe_top_candidates",
    }
    for name in managed_names:
        for path in (run_dir / f"{name}.csv", run_dir / "audit" / f"{name}.csv"):
            if path.exists():
                path.unlink()


def _universe_review_rows(date: str, universe: pd.DataFrame) -> list[dict]:
    today = universe[universe["snapshot_date"] == date].copy()
    if today.empty:
        return []
    today["manager_score"] = today.apply(manager_score, axis=1)
    today = today.sort_values(["manager_score", "thesis_rank_score", "buy_today_score"], ascending=False)
    rows = []
    for rank, row in enumerate(today.itertuples(index=False), start=1):
        classification, reason = _universe_action(row)
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


def _universe_action(row) -> tuple[str, str]:
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
            "momentum_score": pos.get("current_momentum_score"),
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


def _top_candidates(universe_scores: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    if universe_scores.empty:
        return universe_scores
    return (
        universe_scores.sort_values(["date", "manager_score", "thesis_rank_score"], ascending=[True, False, False])
        .groupby("date", as_index=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def _position_performance(
    transactions: pd.DataFrame,
    evolution: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_ticker: str,
) -> pd.DataFrame:
    columns = [
        "lot_id", "ticker", "entry_date", "exit_date", "closed", "holding_days",
        "entry_price", "exit_price", "total_return", "annualized_return",
        "benchmark_total_return", "benchmark_annualized_return", "excess_total_return",
        "entry_manager_score", "entry_buy_today_score", "entry_reason", "exit_reason",
        "exit_reason_category",
    ]
    if transactions.empty or evolution.empty:
        return pd.DataFrame(columns=columns)

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    tx = transactions.copy()
    tx["date"] = pd.to_datetime(tx["date"])
    tx["_action_order"] = tx["action"].map({"SELL": 0, "BUY": 1}).fillna(2)
    tx = tx.sort_values(["date", "_action_order", "ticker"])
    end_date = pd.to_datetime(evolution["date"]).max()

    open_lots: dict[str, list[dict]] = {}
    rows = []
    lot_id = 1
    for row in tx.itertuples(index=False):
        if row.action == "BUY":
            lot = {
                "lot_id": lot_id,
                "ticker": row.ticker,
                "entry_date": row.date,
                "entry_price": _last_price(prices, row.ticker, row.date),
                "entry_reason": getattr(row, "reason", ""),
                "entry_manager_score": getattr(row, "manager_score", None),
                "entry_buy_today_score": getattr(row, "buy_today_score", None),
            }
            open_lots.setdefault(row.ticker, []).append(lot)
            lot_id += 1
        elif row.action == "SELL":
            lot = open_lots.get(row.ticker, []).pop(0) if open_lots.get(row.ticker) else None
            if lot:
                rows.append(_performance_row(lot, row.date, prices, benchmark_ticker, getattr(row, "reason", ""), True))

    for lots in open_lots.values():
        for lot in lots:
            rows.append(_performance_row(lot, end_date, prices, benchmark_ticker, "Open", False))

    return pd.DataFrame(rows, columns=columns)


def _performance_row(
    lot: dict,
    exit_date: pd.Timestamp,
    prices: pd.DataFrame,
    benchmark_ticker: str,
    exit_reason: str,
    closed: bool,
) -> dict:
    entry_date = lot["entry_date"]
    entry_price = lot["entry_price"]
    exit_price = _last_price(prices, lot["ticker"], exit_date)
    benchmark_entry = _last_price(prices, benchmark_ticker, entry_date)
    benchmark_exit = _last_price(prices, benchmark_ticker, exit_date)
    holding_days = max((exit_date - entry_date).days, 1)
    total_return = _safe_return(entry_price, exit_price)
    benchmark_total_return = _safe_return(benchmark_entry, benchmark_exit)
    return {
        "lot_id": lot["lot_id"],
        "ticker": lot["ticker"],
        "entry_date": entry_date.date().isoformat(),
        "exit_date": exit_date.date().isoformat(),
        "closed": bool(closed),
        "holding_days": holding_days,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "total_return": total_return,
        "annualized_return": _annualize(total_return, holding_days),
        "benchmark_total_return": benchmark_total_return,
        "benchmark_annualized_return": _annualize(benchmark_total_return, holding_days),
        "excess_total_return": total_return - benchmark_total_return,
        "entry_manager_score": lot.get("entry_manager_score"),
        "entry_buy_today_score": lot.get("entry_buy_today_score"),
        "entry_reason": lot.get("entry_reason", ""),
        "exit_reason": exit_reason,
        "exit_reason_category": _reason_category(exit_reason),
    }


def _safe_return(start_price: float | None, end_price: float | None) -> float:
    if start_price is None or end_price is None or start_price <= 0:
        return 0.0
    return float(end_price / start_price - 1)


def _annualize(total_return: float, holding_days: int) -> float:
    if holding_days <= 0 or total_return <= -1:
        return 0.0
    return float((1 + total_return) ** (365.25 / holding_days) - 1)


def _reason_category(reason: str | None) -> str:
    if not reason:
        return "Unknown"
    text = str(reason)
    return text.split(":", 1)[0].strip() or "Unknown"


def _turnover(transactions: pd.DataFrame, periods: int, performance: pd.DataFrame | None = None) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame([{
            "monthly_turnover": 0,
            "annual_turnover": 0,
            "average_holding_days": None,
            "median_holding_days": None,
            "buys": 0,
            "sells": 0,
            "sell_reason_mix": "",
        }])
    sells = transactions[transactions["action"] == "SELL"]
    buys = transactions[transactions["action"] == "BUY"]
    monthly = len(sells) / max(periods, 1)
    closed = (
        performance[performance["closed"]]
        if performance is not None and not performance.empty and "closed" in performance.columns
        else pd.DataFrame()
    )
    reason_mix = sells["reason"].map(_reason_category).value_counts()
    return pd.DataFrame([{
        "monthly_turnover": monthly,
        "annual_turnover": monthly * 12,
        "average_holding_days": float(closed["holding_days"].mean()) if not closed.empty else None,
        "median_holding_days": float(closed["holding_days"].median()) if not closed.empty else None,
        "buys": int(len(buys)),
        "sells": int(len(sells)),
        "sell_reason_mix": "; ".join(f"{reason}={count}" for reason, count in reason_mix.items()),
    }])


def _portfolio_vs_benchmark(
    evolution: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_ticker: str,
    transactions: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    tx = transactions.copy()
    if not tx.empty:
        tx["date"] = pd.to_datetime(tx["date"])
    cost_rate = (settings.transaction_cost_bps + settings.slippage_bps) / 10000
    rows = []
    previous_gross_value = 1.0
    previous_net_value = 1.0
    previous_benchmark_value = 1.0
    previous_benchmark = None
    previous_date = None
    previous_tickers: list[str] = []
    for row in evolution.itertuples(index=False):
        date = pd.to_datetime(row.date)
        tickers = [ticker.strip() for ticker in row.tickers.split(",") if ticker.strip()]
        gross_return = 0.0 if previous_date is None else _basket_return(prices, previous_tickers, previous_date, date)
        transaction_cost = _period_transaction_cost(tx, date, len(tickers), cost_rate)
        net_return = gross_return - transaction_cost
        benchmark_price = _last_price(prices, benchmark_ticker, date)
        benchmark_return = 0.0 if previous_benchmark is None or benchmark_price is None else benchmark_price / previous_benchmark - 1
        previous_benchmark = benchmark_price or previous_benchmark
        previous_gross_value *= 1 + gross_return
        previous_net_value *= 1 + net_return
        previous_benchmark_value *= 1 + benchmark_return
        previous_date = date
        previous_tickers = tickers
        rows.append({
            "date": row.date,
            "portfolio_gross_period_return": gross_return,
            "transaction_cost_drag": transaction_cost,
            "portfolio_period_return": net_return,
            "benchmark_period_return": benchmark_return,
            "period_alpha": net_return - benchmark_return,
            "portfolio_gross_value": previous_gross_value,
            "portfolio_value": previous_net_value,
            "benchmark_value": previous_benchmark_value,
        })
    return pd.DataFrame(rows)


def _period_transaction_cost(transactions: pd.DataFrame, date: pd.Timestamp, holding_count: int, cost_rate: float) -> float:
    if transactions.empty or cost_rate <= 0:
        return 0.0
    count = int((transactions["date"] == date).sum())
    if count == 0:
        return 0.0
    traded_weight = count / max(holding_count, 1)
    return float(traded_weight * cost_rate)


def _tracking_dashboard(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    vs = outputs.get("portfolio_vs_benchmark", pd.DataFrame()).copy()
    evolution = outputs.get("portfolio_evolution", pd.DataFrame()).copy()
    transactions = outputs.get("portfolio_transactions", pd.DataFrame()).copy()
    if vs.empty or evolution.empty:
        return pd.DataFrame()
    tx_counts = pd.DataFrame()
    if not transactions.empty:
        tx_counts = (
            transactions
            .pivot_table(index="date", columns="action", values="ticker", aggfunc="count", fill_value=0)
            .reset_index()
            .rename(columns={"BUY": "buys", "SELL": "sells"})
        )
    dashboard = vs.merge(evolution[["date", "holdings", "tickers"]], on="date", how="left")
    if not tx_counts.empty:
        dashboard = dashboard.merge(tx_counts, on="date", how="left")
    for column in ["buys", "sells"]:
        if column not in dashboard.columns:
            dashboard[column] = 0
        dashboard[column] = dashboard[column].fillna(0).astype(int)
    dashboard["cumulative_alpha"] = dashboard["period_alpha"].cumsum()
    columns = [
        "date", "portfolio_value", "portfolio_gross_value", "benchmark_value", "portfolio_period_return",
        "portfolio_gross_period_return", "transaction_cost_drag", "benchmark_period_return",
        "period_alpha", "cumulative_alpha", "holdings",
        "buys", "sells", "tickers",
    ]
    return dashboard[[column for column in columns if column in dashboard.columns]]


def _buy_rationale(transactions: pd.DataFrame, universe_scores: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()
    buys = transactions[transactions["action"] == "BUY"].copy()
    if buys.empty:
        return pd.DataFrame()
    cols = [
        "date", "ticker", "rank", "review_type", "universe_action", "manager_score",
        "thesis_rank_score", "conviction_score", "business_quality_score",
        "price_adjusted_valuation_score", "momentum_score", "alpha_probability",
        "buy_today_score", "opportunity_type", "best_alternative_ticker",
        "opportunity_cost_score", "investment_thesis",
    ]
    scores = _select_columns(universe_scores, cols)
    if scores.empty:
        return buys
    merged = buys.merge(scores, on=["date", "ticker"], how="left", suffixes=("_tx", ""))
    output_cols = [
        "date", "ticker", "rank", "review_type", "manager_score", "buy_today_score",
        "thesis_rank_score", "conviction_score", "business_quality_score",
        "price_adjusted_valuation_score", "momentum_score", "alpha_probability",
        "opportunity_type", "best_alternative_ticker", "opportunity_cost_score",
        "reason", "investment_thesis",
    ]
    return _select_columns(merged, output_cols)


def _action_journal(transactions: pd.DataFrame, performance: pd.DataFrame, buy_rationale: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()
    tx = transactions.copy()
    tx["reason_category"] = tx["reason"].map(_reason_category)
    if not buy_rationale.empty:
        buy_cols = [
            "date", "ticker", "rank", "manager_score", "buy_today_score",
            "business_quality_score", "price_adjusted_valuation_score",
            "momentum_score", "alpha_probability", "opportunity_type",
        ]
        tx = tx.merge(_select_columns(buy_rationale, buy_cols), on=["date", "ticker"], how="left", suffixes=("", "_buy"))
    if not performance.empty:
        perf_cols = [
            "ticker", "entry_date", "exit_date", "closed", "holding_days",
            "total_return", "annualized_return", "benchmark_total_return",
            "benchmark_annualized_return", "excess_total_return", "exit_reason_category",
        ]
        closed_perf = performance[performance["closed"]].copy() if "closed" in performance.columns else pd.DataFrame()
        if not closed_perf.empty:
            tx = tx.merge(
                _select_columns(closed_perf, perf_cols),
                left_on=["date", "ticker", "reason_category"],
                right_on=["exit_date", "ticker", "exit_reason_category"],
                how="left",
            )
    columns = [
        "date", "ticker", "action", "reason_category", "reason", "rank",
        "manager_score", "buy_today_score", "business_quality_score",
        "price_adjusted_valuation_score", "momentum_score", "alpha_probability",
        "opportunity_type", "holding_days", "total_return", "benchmark_total_return",
        "excess_total_return", "thesis", "exit_thesis", "catalyst",
    ]
    return _select_columns(tx, columns)


def _sell_reasons_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()
    sells = transactions[transactions["action"] == "SELL"].copy()
    if sells.empty:
        return pd.DataFrame()
    sells["reason_category"] = sells["reason"].map(_reason_category)
    return (
        sells.groupby("reason_category", as_index=False)
        .agg(sells=("ticker", "count"), tickers=("ticker", lambda values: ", ".join(sorted(set(values)))))
        .sort_values("sells", ascending=False)
    )


def _sector_exposure(holdings: pd.DataFrame) -> pd.DataFrame:
    if holdings.empty or "sector" not in holdings.columns:
        return pd.DataFrame()
    weight_col = "hybrid_weight" if "hybrid_weight" in holdings.columns else None
    rows = []
    for (date, sector), group in holdings.groupby(["date", "sector"]):
        rows.append({
            "date": date,
            "sector": sector,
            "positions": int(group["ticker"].nunique()),
            "weight": float(group[weight_col].sum()) if weight_col else float(len(group) / max(holdings[holdings["date"] == date]["ticker"].nunique(), 1)),
            "avg_manager_score": float(group.get("current_manager_score", pd.Series(dtype=float)).mean()),
            "tickers": ", ".join(sorted(group["ticker"].astype(str).unique())),
        })
    return pd.DataFrame(rows).sort_values(["date", "weight"], ascending=[True, False])


def _current_portfolio(holdings: pd.DataFrame) -> pd.DataFrame:
    if holdings.empty or "date" not in holdings.columns:
        return pd.DataFrame()
    latest_date = holdings["date"].max()
    cols = [
        "date", "ticker", "sector", "hybrid_weight", "position_action",
        "current_manager_score", "current_conviction_score", "current_buy_today_score",
        "current_would_buy_today", "current_thesis_state",
        "current_price_adjusted_valuation_score", "current_momentum_score",
        "thesis_persistence_score", "months_since_entry", "investment_thesis",
        "current_exit_thesis",
    ]
    latest = holdings[holdings["date"] == latest_date].copy()
    return _select_columns(latest.sort_values("hybrid_weight", ascending=False), cols)


def _latest_top_opportunities(universe_top: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    if universe_top.empty or "date" not in universe_top.columns:
        return pd.DataFrame()
    latest_date = universe_top["date"].max()
    cols = [
        "date", "rank", "ticker", "universe_action", "manager_score",
        "buy_today_score", "thesis_rank_score", "conviction_score",
        "business_quality_score", "price_adjusted_valuation_score",
        "momentum_score", "alpha_probability", "opportunity_type",
        "best_alternative_ticker", "opportunity_cost_score", "investment_thesis",
    ]
    latest = universe_top[universe_top["date"] == latest_date].copy().head(top_n)
    return _select_columns(latest, cols)


def _executive_summary_table(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary = _summary(outputs)
    vs = outputs.get("portfolio_vs_benchmark", pd.DataFrame())
    turnover = outputs.get("portfolio_turnover", pd.DataFrame())
    latest = vs.iloc[-1].to_dict() if not vs.empty else {}
    row = {
        **summary,
        "portfolio_value": latest.get("portfolio_value", 0),
        "portfolio_gross_value": latest.get("portfolio_gross_value", 0),
        "benchmark_value": latest.get("benchmark_value", 0),
        "total_cost_drag": float(vs.get("transaction_cost_drag", pd.Series([0])).sum()) if not vs.empty else 0,
    }
    if not turnover.empty:
        row.update({
            "annual_turnover": turnover.iloc[0].get("annual_turnover"),
            "average_holding_days": turnover.iloc[0].get("average_holding_days"),
            "sell_reason_mix": turnover.iloc[0].get("sell_reason_mix"),
        })
    return pd.DataFrame([row])


def _strategy_learning_log(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    performance = outputs.get("position_performance", pd.DataFrame())
    action_journal = outputs.get("action_journal", pd.DataFrame())
    rows = []
    if not performance.empty:
        closed = performance[performance["closed"]].copy() if "closed" in performance.columns else performance.copy()
        if not closed.empty:
            for category, group in closed.groupby("exit_reason_category"):
                rows.append({
                    "area": "exit_reason",
                    "segment": category,
                    "observations": int(len(group)),
                    "win_rate_vs_benchmark": float((group["excess_total_return"] > 0).mean()),
                    "avg_total_return": float(group["total_return"].mean()),
                    "avg_excess_return": float(group["excess_total_return"].mean()),
                    "avg_holding_days": float(group["holding_days"].mean()),
                    "hint": _learning_hint("exit_reason", category, group),
                })
    if not action_journal.empty and "opportunity_type" in action_journal.columns:
        buys = action_journal[action_journal["action"] == "BUY"].copy()
        if not buys.empty:
            for opportunity_type, group in buys.groupby("opportunity_type", dropna=False):
                closed = group.dropna(subset=["excess_total_return"]) if "excess_total_return" in group.columns else pd.DataFrame()
                rows.append({
                    "area": "entry_opportunity_type",
                    "segment": str(opportunity_type),
                    "observations": int(len(group)),
                    "win_rate_vs_benchmark": float((closed["excess_total_return"] > 0).mean()) if not closed.empty else None,
                    "avg_total_return": float(closed["total_return"].mean()) if not closed.empty else None,
                    "avg_excess_return": float(closed["excess_total_return"].mean()) if not closed.empty else None,
                    "avg_holding_days": float(closed["holding_days"].mean()) if not closed.empty else None,
                    "hint": _learning_hint("entry_opportunity_type", str(opportunity_type), closed),
                })
    return pd.DataFrame(rows)


def _learning_hint(area: str, segment: str, group: pd.DataFrame) -> str:
    if group.empty or "excess_total_return" not in group.columns:
        return "Collect more closed observations before changing rules."
    avg_excess = float(group["excess_total_return"].mean())
    win_rate = float((group["excess_total_return"] > 0).mean())
    if avg_excess < 0 and win_rate < 0.45:
        return f"Review {area}={segment}; this segment has weak excess return and low win rate."
    if avg_excess > 0.05 and win_rate > 0.55:
        return f"Protect or expand {area}={segment}; this segment has strong benchmark-relative evidence."
    return f"Monitor {area}={segment}; evidence is mixed."


def _improvement_backlog(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    learning = _strategy_learning_log(outputs)
    summary = _executive_summary_table(outputs)
    rows = []
    if not summary.empty:
        row = summary.iloc[0]
        if float(row.get("annual_turnover", 0) or 0) > 8:
            rows.append({
                "priority": "high",
                "theme": "turnover",
                "evidence": f"annual_turnover={row.get('annual_turnover')}",
                "suggested_next_step": "Test stricter replacement thresholds or longer minimum holding period.",
            })
        if float(row.get("closed_position_win_rate_vs_benchmark", 0) or 0) < 0.5:
            rows.append({
                "priority": "high",
                "theme": "entry_quality",
                "evidence": f"win_rate_vs_benchmark={row.get('closed_position_win_rate_vs_benchmark')}",
                "suggested_next_step": "Inspect buy_rationale and increase hurdles for weak opportunity types.",
            })
    if not learning.empty:
        weak = learning[(learning["avg_excess_return"].fillna(0) < 0) & (learning["observations"] >= 3)]
        for row in weak.itertuples(index=False):
            rows.append({
                "priority": "medium",
                "theme": row.area,
                "evidence": f"{row.segment}: avg_excess={row.avg_excess_return:.3f}, observations={row.observations}",
                "suggested_next_step": row.hint,
            })
        strong = learning[(learning["avg_excess_return"].fillna(0) > 0.05) & (learning["observations"] >= 3)]
        for row in strong.head(5).itertuples(index=False):
            rows.append({
                "priority": "low",
                "theme": row.area,
                "evidence": f"{row.segment}: avg_excess={row.avg_excess_return:.3f}, observations={row.observations}",
                "suggested_next_step": row.hint,
            })
    if not rows:
        rows.append({
            "priority": "low",
            "theme": "monitoring",
            "evidence": "No obvious automatic warning from current diagnostics.",
            "suggested_next_step": "Review position_performance and model_walk_forward_diagnostics after the long-history run.",
        })
    return pd.DataFrame(rows)


def _select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df[[column for column in columns if column in df.columns]]


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
    performance = outputs.get("position_performance", pd.DataFrame())
    closed = (
        performance[performance["closed"]]
        if not performance.empty and "closed" in performance.columns
        else pd.DataFrame()
    )
    return {
        "initial_holdings": int(outputs["portfolio_evolution"].iloc[0]["holdings"]),
        "final_holdings": int(outputs["portfolio_evolution"].iloc[-1]["holdings"]),
        "transactions": int(len(transactions)),
        "buys": int((transactions["action"] == "BUY").sum()),
        "sells": int((transactions["action"] == "SELL").sum()),
        "cumulative_alpha": float(vs_benchmark["period_alpha"].sum()) if not vs_benchmark.empty else 0.0,
        "closed_positions": int(len(closed)),
        "closed_position_win_rate_vs_benchmark": float((closed["excess_total_return"] > 0).mean()) if not closed.empty else 0.0,
        "average_closed_total_return": float(closed["total_return"].mean()) if not closed.empty else 0.0,
        "average_closed_excess_return": float(closed["excess_total_return"].mean()) if not closed.empty else 0.0,
    }
