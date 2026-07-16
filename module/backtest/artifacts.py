"""Backtest output tables and summary helpers."""

from __future__ import annotations

import math

import pandas as pd

from .performance import compound_alpha, reason_category

SMALL_SAMPLE_CAVEAT = (
    "With only a few dozen monthly observations, the t-stat below has limited statistical power; "
    "treat it as a directional signal, not proof of a significant edge."
)


def tracking_dashboard(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    vs = outputs.get("portfolio_vs_benchmark", pd.DataFrame()).copy()
    evolution = outputs.get("portfolio_evolution", pd.DataFrame()).copy()
    transactions = outputs.get("portfolio_transactions", pd.DataFrame()).copy()
    if vs.empty or evolution.empty:
        return pd.DataFrame()
    tx_counts = pd.DataFrame()
    if not transactions.empty:
        tx_counts = (
            transactions.pivot_table(index="date", columns="action", values="ticker", aggfunc="count", fill_value=0)
            .reset_index()
            .rename(columns={"BUY": "buys", "SELL": "sells"})
        )
    dashboard = vs.merge(evolution[["date", "holdings", "tickers"]], on="date", how="left")
    if not tx_counts.empty:
        dashboard = dashboard.merge(tx_counts, on="date", how="left")
    for column in ["buys", "sells"]:
        dashboard[column] = dashboard.get(column, 0)
        dashboard[column] = dashboard[column].fillna(0).astype(int)
    dashboard["cumulative_alpha"] = dashboard["period_alpha"].cumsum()
    columns = [
        "date", "portfolio_value", "portfolio_gross_value", "benchmark_value", "portfolio_period_return",
        "portfolio_gross_period_return", "transaction_cost_drag", "benchmark_period_return",
        "period_alpha", "cumulative_alpha", "holdings", "buys", "sells", "tickers",
    ]
    return dashboard[[column for column in columns if column in dashboard.columns]]


def buy_rationale(transactions: pd.DataFrame, universe_scores: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()
    buys = transactions[transactions["action"] == "BUY"].copy()
    if buys.empty:
        return pd.DataFrame()
    cols = [
        "date", "ticker", "rank", "review_type", "universe_action", "manager_score",
        "thesis_rank_score", "conviction_score", "business_quality_score",
        "price_adjusted_valuation_score", "momentum_score", "timing_probability",
        "buy_today_score", "opportunity_type", "best_alternative_ticker",
        "opportunity_cost_score", "investment_thesis",
    ]
    scores = select_columns(universe_scores, cols)
    if scores.empty:
        return buys
    merged = buys.merge(scores, on=["date", "ticker"], how="left", suffixes=("_tx", ""))
    output_cols = [
        "date", "ticker", "rank", "review_type", "manager_score", "buy_today_score",
        "thesis_rank_score", "conviction_score", "business_quality_score",
        "price_adjusted_valuation_score", "momentum_score", "timing_probability",
        "opportunity_type", "best_alternative_ticker", "opportunity_cost_score",
        "reason", "investment_thesis",
    ]
    return select_columns(merged, output_cols)


def action_journal(transactions: pd.DataFrame, performance: pd.DataFrame, rationale: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()
    tx = transactions.copy()
    tx["reason_category"] = tx["reason"].map(reason_category)
    if not rationale.empty:
        buy_cols = [
            "date", "ticker", "rank", "manager_score", "buy_today_score",
            "business_quality_score", "price_adjusted_valuation_score",
            "momentum_score", "timing_probability", "opportunity_type",
        ]
        tx = tx.merge(select_columns(rationale, buy_cols), on=["date", "ticker"], how="left", suffixes=("", "_buy"))
    if not performance.empty:
        perf_cols = [
            "ticker", "entry_date", "exit_date", "closed", "holding_days",
            "total_return", "annualized_return", "benchmark_total_return",
            "benchmark_annualized_return", "excess_total_return", "exit_reason_category",
        ]
        closed_perf = performance[performance["closed"]].copy() if "closed" in performance.columns else pd.DataFrame()
        if not closed_perf.empty:
            tx = tx.merge(
                select_columns(closed_perf, perf_cols),
                left_on=["date", "ticker", "reason_category"],
                right_on=["exit_date", "ticker", "exit_reason_category"],
                how="left",
            )
    columns = [
        "date", "ticker", "action", "reason_category", "reason", "rank",
        "manager_score", "buy_today_score", "business_quality_score",
        "price_adjusted_valuation_score", "momentum_score", "timing_probability",
        "opportunity_type", "holding_days", "total_return", "benchmark_total_return",
        "excess_total_return", "thesis", "exit_thesis", "catalyst",
        "manager_score_breakdown", "vs_best_alternative", "entry_vs_current",
    ]
    return select_columns(tx, columns)


def sell_reasons_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()
    sells = transactions[transactions["action"] == "SELL"].copy()
    if sells.empty:
        return pd.DataFrame()
    sells["reason_category"] = sells["reason"].map(reason_category)
    return (
        sells.groupby("reason_category", as_index=False)
        .agg(sells=("ticker", "count"), tickers=("ticker", lambda values: ", ".join(sorted(set(values)))))
        .sort_values("sells", ascending=False)
    )


def sector_exposure(holdings: pd.DataFrame) -> pd.DataFrame:
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


def current_portfolio(holdings: pd.DataFrame) -> pd.DataFrame:
    if holdings.empty or "date" not in holdings.columns:
        return pd.DataFrame()
    latest = holdings[holdings["date"] == holdings["date"].max()].copy()
    cols = [
        "date", "ticker", "sector", "hybrid_weight", "position_action",
        "current_manager_score", "current_conviction_score", "current_buy_today_score",
        "current_would_buy_today", "current_thesis_state", "current_price_adjusted_valuation_score",
        "current_momentum_score", "thesis_persistence_score", "months_since_entry",
        "investment_thesis", "current_exit_thesis", "manager_score_breakdown_at_entry",
        "vs_best_alternative_at_entry", "entry_vs_current",
    ]
    return select_columns(latest.sort_values("hybrid_weight", ascending=False), cols)


def latest_top_opportunities(universe_top: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    if universe_top.empty or "date" not in universe_top.columns:
        return pd.DataFrame()
    latest = universe_top[universe_top["date"] == universe_top["date"].max()].copy().head(top_n)
    cols = [
        "date", "rank", "ticker", "universe_action", "manager_score", "buy_today_score",
        "thesis_rank_score", "conviction_score", "business_quality_score", "price_adjusted_valuation_score",
        "momentum_score", "timing_probability", "opportunity_type", "best_alternative_ticker",
        "opportunity_cost_score", "investment_thesis",
    ]
    return select_columns(latest, cols)


def executive_summary_table(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary = summary_metrics(outputs)
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


def strategy_learning_log(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    performance = outputs.get("position_performance", pd.DataFrame())
    journal = outputs.get("action_journal", pd.DataFrame())
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
                    "hint": learning_hint("exit_reason", category, group),
                })
    if not journal.empty and "opportunity_type" in journal.columns:
        buys = journal[journal["action"] == "BUY"].copy()
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
                    "hint": learning_hint("entry_opportunity_type", str(opportunity_type), closed),
                })
    return pd.DataFrame(rows)


def improvement_backlog(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    learning = strategy_learning_log(outputs)
    summary = executive_summary_table(outputs)
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


def select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df[[column for column in columns if column in df.columns]]


def learning_hint(area: str, segment: str, group: pd.DataFrame) -> str:
    if group.empty or "excess_total_return" not in group.columns:
        return "Collect more closed observations before changing rules."
    avg_excess = float(group["excess_total_return"].mean())
    win_rate = float((group["excess_total_return"] > 0).mean())
    if avg_excess < 0 and win_rate < 0.45:
        return f"Review {area}={segment}; this segment has weak excess return and low win rate."
    if avg_excess > 0.05 and win_rate > 0.55:
        return f"Protect or expand {area}={segment}; this segment has strong benchmark-relative evidence."
    return f"Monitor {area}={segment}; evidence is mixed."


def summary_metrics(outputs: dict) -> dict:
    transactions = outputs["portfolio_transactions"]
    vs_benchmark = outputs["portfolio_vs_benchmark"]
    performance = outputs.get("position_performance", pd.DataFrame())
    closed = performance[performance["closed"]] if not performance.empty and "closed" in performance.columns else pd.DataFrame()
    return {
        "initial_holdings": int(outputs["portfolio_evolution"].iloc[0]["holdings"]),
        "final_holdings": int(outputs["portfolio_evolution"].iloc[-1]["holdings"]),
        "transactions": int(len(transactions)),
        "buys": int((transactions["action"] == "BUY").sum()),
        "sells": int((transactions["action"] == "SELL").sum()),
        "cumulative_alpha": compound_alpha(vs_benchmark),
        "sum_period_alpha": float(vs_benchmark["period_alpha"].sum()) if not vs_benchmark.empty else 0.0,
        "closed_positions": int(len(closed)),
        "closed_position_win_rate_vs_benchmark": float((closed["excess_total_return"] > 0).mean()) if not closed.empty else 0.0,
        "average_closed_total_return": float(closed["total_return"].mean()) if not closed.empty else 0.0,
        "average_closed_excess_return": float(closed["excess_total_return"].mean()) if not closed.empty else 0.0,
        **excess_return_statistics(vs_benchmark),
    }


def edge_attribution(position_performance: pd.DataFrame, walk_forward_diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Where does the alpha actually come from — the ML ranking, or something else?

    The rolling OOS rank-IC of `final_score` can be persistently low or negative (the ML signal
    fails to rank forward alpha correctly even while relearning every quarter) even while
    cumulative alpha is positive. This decomposes the closed-position record into the pieces needed
    to judge that: the correlation between the entry manager_score/buy_today_score and the realized
    excess return (does a HIGHER score at entry actually predict a BETTER outcome, independent of
    rank-IC), and the winner/loser skew (a few large winners carrying a losing-on-average book vs. a
    broadly positive edge). Returns one row per metric so it prints as a simple table — no forced
    narrative, just numbers.
    """
    rows: list[dict] = []
    closed = (
        position_performance[position_performance["closed"]].copy()
        if not position_performance.empty and "closed" in position_performance.columns
        else pd.DataFrame()
    )
    if not closed.empty and "excess_total_return" in closed.columns:
        excess = pd.to_numeric(closed["excess_total_return"], errors="coerce")
        winners = excess[excess > 0]
        losers = excess[excess <= 0]
        for score_col, label in (
            ("entry_manager_score", "manager_score"),
            ("entry_buy_today_score", "buy_today_score"),
        ):
            if score_col in closed.columns:
                pair = pd.DataFrame({
                    "score": pd.to_numeric(closed[score_col], errors="coerce"),
                    "excess": excess,
                }).dropna()
                corr = pair["score"].corr(pair["excess"], method="spearman") if len(pair) >= 5 else None
                rows.append({
                    "metrica": f"Correlación entrada ({label}) vs. exceso realizado",
                    "valor": corr,
                    "n": int(len(pair)),
                })
        rows.append({
            "metrica": "Win rate (posiciones cerradas, exceso > 0)",
            "valor": float((excess > 0).mean()) if not excess.empty else None,
            "n": int(len(excess)),
        })
        rows.append({
            "metrica": "Retorno medio en exceso, ganadoras",
            "valor": float(winners.mean()) if not winners.empty else None,
            "n": int(len(winners)),
        })
        rows.append({
            "metrica": "Retorno medio en exceso, perdedoras",
            "valor": float(losers.mean()) if not losers.empty else None,
            "n": int(len(losers)),
        })
        rows.append({
            "metrica": "Suma total de exceso (ganadoras)",
            "valor": float(winners.sum()) if not winners.empty else None,
            "n": int(len(winners)),
        })
        rows.append({
            "metrica": "Suma total de exceso (perdedoras)",
            "valor": float(losers.sum()) if not losers.empty else None,
            "n": int(len(losers)),
        })
    if not walk_forward_diagnostics.empty and "rank_ic_final" in walk_forward_diagnostics.columns:
        ic = pd.to_numeric(walk_forward_diagnostics["rank_ic_final"], errors="coerce").dropna()
        rows.append({
            "metrica": "Rank-IC medio de final_score (toda la ventana, rolling)",
            "valor": float(ic.mean()) if not ic.empty else None,
            "n": int(len(ic)),
        })
    return pd.DataFrame(rows)


def excess_return_statistics(vs_benchmark: pd.DataFrame) -> dict:
    if vs_benchmark.empty or "portfolio_period_return" not in vs_benchmark.columns:
        return {
            "information_ratio": 0.0,
            "tracking_error_annualized": 0.0,
            "excess_return_t_stat": 0.0,
            "periods_n": 0,
            "small_sample_caveat": SMALL_SAMPLE_CAVEAT,
        }
    excess = (vs_benchmark["portfolio_period_return"].fillna(0) - vs_benchmark["benchmark_period_return"].fillna(0))
    n = int(len(excess))
    std = float(excess.std(ddof=1)) if n > 1 else 0.0
    tracking_error = std * math.sqrt(12)
    mean_excess = float(excess.mean())
    information_ratio = (mean_excess * 12) / tracking_error if tracking_error > 0 else 0.0
    t_stat = mean_excess / (std / math.sqrt(n)) if std > 0 and n > 1 else 0.0
    return {
        "information_ratio": information_ratio,
        "tracking_error_annualized": tracking_error,
        "excess_return_t_stat": t_stat,
        "periods_n": n,
        "small_sample_caveat": SMALL_SAMPLE_CAVEAT,
    }
