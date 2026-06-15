"""Live-portfolio evolution simulator for GARP / Value-Growth thesis management.

This module does not train models and does not alter the GARP agent stack.  It
reuses Portfolio Intelligence to simulate a concentrated, thesis-managed
portfolio with memory, explicit sell discipline and turnover reporting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from environment import (
    GARP_MAX_STOCKS,
    GARP_MIN_STOCKS,
    HOLD_WINNER_BONUS,
    MIN_CONVICTION_ADVANTAGE,
    MIN_OPPORTUNITY_COST_THRESHOLD,
    MIN_ROTATION_ADVANTAGE,
    MIN_SCORE_ADVANTAGE_TO_REPLACE,
    PORTFOLIO_REVIEW_FREQUENCY,
    PORTFOLIO_WEIGHTING_MODE,
    THESIS_INTACT_HOLD_PREFERENCE,
)
from module.common.portfolio_intelligence import add_portfolio_review_scores, review_portfolio


def _normalize_freq(freq: str) -> str:
    value = str(freq or PORTFOLIO_REVIEW_FREQUENCY).strip().upper()
    if value in {"M", "MONTH", "MONTHLY"}:
        return "M"
    if value in {"2M", "BIMONTHLY", "BI_MONTHLY"}:
        return "2M"
    if value in {"Q", "QUARTER", "QUARTERLY"}:
        return "Q"
    return value


def _review_dates(scored: pd.DataFrame, frequency: str) -> list[pd.Timestamp]:
    dates = pd.to_datetime(scored["date"].dropna().unique())
    if len(dates) == 0:
        return []
    frame = pd.DataFrame({"date": sorted(pd.Timestamp(d).normalize() for d in dates)})
    period = frame["date"].dt.to_period(_normalize_freq(frequency))
    return [pd.Timestamp(x).normalize() for x in frame.groupby(period)["date"].max().tolist()]


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    if pd.isna(start) or pd.isna(end):
        return 0
    s = pd.Timestamp(start); e = pd.Timestamp(end)
    return max(0, (e.year - s.year) * 12 + (e.month - s.month))


def _top_candidates(snapshot: pd.DataFrame, held: set[str], n: int | None = None) -> pd.DataFrame:
    pool = snapshot[~snapshot["ticker"].astype(str).isin(held)].copy()
    if pool.empty:
        return pool
    pool = pool.sort_values(["thesis_score", "expectation_gap_score", "quality_score"], ascending=False)
    return pool.head(n) if n is not None else pool


def _interval_return(prices_dict: Mapping[str, pd.DataFrame] | None, holdings: dict[str, float], start: pd.Timestamp, end: pd.Timestamp) -> float:
    if not prices_dict or not holdings:
        return 0.0
    total = 0.0; used = 0.0
    for ticker, weight in holdings.items():
        prices = prices_dict.get(ticker)
        if prices is None or prices.empty or "Close" not in prices.columns:
            continue
        close = pd.to_numeric(prices.loc[start:end, "Close"], errors="coerce").dropna()
        if len(close) < 2:
            continue
        total += float(weight) * float(close.iloc[-1] / close.iloc[0] - 1.0)
        used += float(weight)
    return total / used if used > 0 else 0.0


def _latest_price(prices_dict: Mapping[str, pd.DataFrame] | None, ticker: str, date: pd.Timestamp) -> float:
    if not prices_dict or ticker not in prices_dict:
        return np.nan
    prices = prices_dict[ticker]
    if prices is None or prices.empty or "Close" not in prices.columns:
        return np.nan
    close = pd.to_numeric(prices.loc[:date, "Close"], errors="coerce").dropna()
    return float(close.iloc[-1]) if len(close) else np.nan


def _cumulative_return(prices_dict: Mapping[str, pd.DataFrame] | None, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if not prices_dict or ticker not in prices_dict:
        return np.nan
    prices = prices_dict[ticker]
    if prices is None or prices.empty or "Close" not in prices.columns:
        return np.nan
    close = pd.to_numeric(prices.loc[start:end, "Close"], errors="coerce").dropna()
    if len(close) < 2:
        return np.nan
    return float(close.iloc[-1] / close.iloc[0] - 1.0)


def _opportunity_type(row: pd.Series) -> str:
    q = float(row.get("quality_score", 0.5)); g = float(row.get("growth_score", 0.5)); v = float(row.get("valuation_score", 0.5))
    t = float(row.get("fundamental_trend_score", 0.5)); c = float(row.get("catalyst_score", 0.5)); r = float(row.get("risk_score", 0.5))
    if v >= 0.68 and g >= 0.62 and q >= 0.55 and r >= 0.50:
        return "Growth infravalorado"
    if q >= 0.70 and g >= 0.62 and v >= 0.45 and r >= 0.55:
        return "Quality Growth razonable"
    if v >= 0.68 and c >= 0.60 and t >= 0.52 and r >= 0.50:
        return "Value con catalizador"
    if q >= 0.72 and t >= 0.62 and v >= 0.50 and r >= 0.58:
        return "Compounder a precio razonable"
    return "GARP candidate"


def _original_scores(row: pd.Series) -> dict[str, float]:
    cols = ["quality_score", "growth_score", "valuation_score", "fundamental_trend_score", "catalyst_score", "risk_score", "moat_proxy_score", "expectation_gap_score", "thesis_score"]
    return {c: round(float(row.get(c, np.nan)), 4) for c in cols if c in row.index and pd.notna(row.get(c))}


def _hold_preference(row: pd.Series, memory: dict[str, object]) -> float:
    status = str(row.get("thesis_status", ""))
    bonus = 0.0
    if status in {"Improving", "Intact"}:
        bonus += float(THESIS_INTACT_HOLD_PREFERENCE)
    if float(row.get("quality_score", 0.5)) >= 0.70 and float(row.get("moat_proxy_score", 0.5)) >= 0.65:
        bonus += float(HOLD_WINNER_BONUS)
    months_intact = float(memory.get("months_thesis_intact", 0))
    bonus += min(0.08, months_intact / 60.0)
    return bonus


def _weight_holdings(holdings: dict[str, dict[str, object]]) -> None:
    if not holdings:
        return
    if str(PORTFOLIO_WEIGHTING_MODE).lower() == "conviction":
        raw = {t: max(0.2, float(h.get("latest_conviction_score", 50.0)) / 100.0) for t, h in holdings.items()}
        denom = sum(raw.values()) or 1.0
        for t, h in holdings.items():
            h["weight"] = raw[t] / denom
    else:
        weight = 1.0 / len(holdings)
        for h in holdings.values():
            h["weight"] = weight


def _decision_row(date: pd.Timestamp, ticker: str, action: str, previous_weight: float, new_weight: float, reason: str, row: pd.Series | dict, memory: dict[str, object], replacement: pd.Series | None = None) -> dict[str, object]:
    return {
        "date": str(date.date()),
        "ticker": ticker,
        "action": action,
        "previous_weight": float(previous_weight),
        "new_weight": float(new_weight),
        "reason": reason,
        "thesis_status": row.get("thesis_status", ""),
        "thesis_score": row.get("thesis_score", np.nan),
        "conviction_score": row.get("conviction_score", row.get("latest_conviction_score", np.nan)),
        "exit_score": row.get("exit_score", np.nan),
        "opportunity_cost": row.get("opportunity_cost_flag", False),
        "replacement_ticker": "" if replacement is None else replacement.get("ticker", ""),
        "replacement_score": np.nan if replacement is None else replacement.get("thesis_score", np.nan),
        "original_thesis_summary": memory.get("original_buy_reason", ""),
        "current_thesis_summary": row.get("current_vs_original_summary", row.get("reason", "")),
        "key_positive_drivers": row.get("top_positive_drivers", ""),
        "key_negative_drivers": row.get("top_risks", ""),
        "months_since_entry": int(memory.get("months_since_entry", 0)),
        "months_thesis_intact": int(memory.get("months_thesis_intact", 0)),
        "thesis_persistence_score": float(memory.get("thesis_persistence_score", np.nan)),
    }


def run_portfolio_evolution(
    snapshots: pd.DataFrame,
    *,
    prices_dict: Mapping[str, pd.DataFrame] | None = None,
    benchmark_returns: pd.Series | None = None,
    review_frequency: str | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    min_positions: int = GARP_MIN_STOCKS,
    max_positions: int = GARP_MAX_STOCKS,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Simulate a live thesis-managed portfolio through periodic reviews."""
    scored = add_portfolio_review_scores(snapshots)
    if start_date is not None:
        scored = scored[pd.to_datetime(scored["date"]) >= pd.Timestamp(start_date).normalize()].copy()
    if end_date is not None:
        scored = scored[pd.to_datetime(scored["date"]) <= pd.Timestamp(end_date).normalize()].copy()
    dates = _review_dates(scored, review_frequency or PORTFOLIO_REVIEW_FREQUENCY)
    holdings: dict[str, dict[str, object]] = {}
    evolution_rows: list[dict[str, object]] = []
    transaction_rows: list[dict[str, object]] = []
    holdings_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    closed_holding_months: list[int] = []
    equity = 1.0; benchmark_equity = 1.0; prev_weights: dict[str, float] = {}

    for i, review_date in enumerate(dates):
        snapshot = scored[scored["date"] == review_date].copy()
        if snapshot.empty:
            continue
        next_date = dates[i + 1] if i + 1 < len(dates) else review_date
        best_candidate = _top_candidates(snapshot, set(holdings), 1)
        best_replacement = best_candidate.iloc[0] if not best_candidate.empty else None

        positions_df = pd.DataFrame([
            {"ticker": t, "weight": h.get("weight", 0.0), "purchase_date": h.get("entry_date"), "snapshot_date": h.get("snapshot_date")}
            for t, h in holdings.items()
        ])
        review = pd.DataFrame()
        if not positions_df.empty:
            review, _ = review_portfolio(scored, positions=positions_df, review_date=review_date)
            for _, row in review.iterrows():
                ticker = str(row["ticker"]); memory = holdings.get(ticker, {})
                previous_weight = float(memory.get("weight", 0.0))
                months = _months_between(pd.Timestamp(memory.get("entry_date")), review_date)
                memory["months_since_entry"] = months
                status = str(row.get("thesis_status", ""))
                if status in {"Improving", "Intact"}:
                    memory["months_thesis_intact"] = int(memory.get("months_thesis_intact", 0)) + max(1, _months_between(pd.Timestamp(memory.get("last_review_date", memory.get("entry_date"))), review_date))
                if status == "Improving":
                    memory["thesis_improvement_count"] = int(memory.get("thesis_improvement_count", 0)) + 1
                if status in {"Weakening", "Broken"}:
                    memory["thesis_deterioration_count"] = int(memory.get("thesis_deterioration_count", 0)) + 1
                memory["last_review_date"] = review_date
                memory["latest_conviction_score"] = float(row.get("conviction_score", 50.0))
                memory["thesis_persistence_score"] = max(0.0, min(100.0, 55 + 2.0 * int(memory.get("months_thesis_intact", 0)) - 12.0 * int(memory.get("thesis_deterioration_count", 0)) + 6.0 * int(memory.get("thesis_improvement_count", 0))))

                replacement_clear = False
                if best_replacement is not None:
                    score_adv = float(best_replacement.get("thesis_score", 0.0)) - float(row.get("thesis_score", 0.0))
                    conviction_adv = float(best_replacement.get("thesis_score", 0.0)) * 100 - float(row.get("conviction_score", 50.0))
                    replacement_clear = score_adv >= max(float(MIN_SCORE_ADVANTAGE_TO_REPLACE), float(MIN_ROTATION_ADVANTAGE), float(MIN_OPPORTUNITY_COST_THRESHOLD)) and conviction_adv >= float(MIN_CONVICTION_ADVANTAGE)
                action = "HOLD"; reason = row.get("current_vs_original_summary", row.get("exit_reason", "Thesis intact"))
                hard_sell = row.get("buy_hold_sell_rating") == "Sell" or status == "Broken"
                persistent_weakening = status == "Weakening" and int(memory.get("thesis_deterioration_count", 0)) >= 2
                overvalued_exit = str(row.get("valuation_status", "")) in {"Overvalued", "Extremely Overvalued"} and float(row.get("exit_score", 0)) >= 45
                better_opp_exit = replacement_clear and (persistent_weakening or overvalued_exit or float(row.get("exit_score", 0)) >= 55)
                hold_preference = _hold_preference(row, memory)
                if hard_sell or persistent_weakening or overvalued_exit or better_opp_exit:
                    action = "SELL" if not overvalued_exit else "REDUCE"
                    if status in {"Improving", "Intact"} and not hard_sell and not overvalued_exit:
                        action = "HOLD"; reason = f"Held by thesis persistence; replacement advantage not sufficient after hold bonus={hold_preference:.2f}."
                    else:
                        reason = row.get("exit_reason", "Thesis-based exit")
                        closed_holding_months.append(months)
                        holdings.pop(ticker, None)
                transaction_rows.append({"date": str(review_date.date()), "ticker": ticker, "action": action, "previous_weight": previous_weight, "new_weight": 0.0 if action in {"SELL", "REDUCE"} else previous_weight, "reason": reason})
                decision_rows.append(_decision_row(review_date, ticker, action, previous_weight, 0.0 if action in {"SELL", "REDUCE"} else previous_weight, reason, row, memory, best_replacement if better_opp_exit else None))

        target_n = int(max(min_positions, max_positions))
        needed = max(0, target_n - len(holdings))
        for _, row in _top_candidates(snapshot, set(holdings), needed).iterrows():
            ticker = str(row["ticker"])
            memory = {
                "entry_date": review_date,
                "snapshot_date": review_date,
                "weight": 0.0,
                "original_thesis_score": float(row.get("thesis_score", 0.5)),
                "original_opportunity_type": _opportunity_type(row),
                "original_scores": _original_scores(row),
                "original_buy_reason": f"ADD: {_opportunity_type(row)} with thesis_score={float(row.get('thesis_score', 0.5)):.2f}, quality={float(row.get('quality_score', 0.5)):.2f}, growth={float(row.get('growth_score', 0.5)):.2f}, valuation={float(row.get('valuation_score', 0.5)):.2f}.",
                "months_since_entry": 0,
                "months_thesis_intact": 0,
                "thesis_deterioration_count": 0,
                "thesis_improvement_count": 0,
                "thesis_persistence_score": 55.0,
                "latest_conviction_score": float(row.get("thesis_score", 0.5)) * 100,
                "entry_price": _latest_price(prices_dict, ticker, review_date),
            }
            holdings[ticker] = memory
            reason = memory["original_buy_reason"]
            transaction_rows.append({"date": str(review_date.date()), "ticker": ticker, "action": "ADD", "previous_weight": 0.0, "new_weight": 0.0, "reason": reason})
            decision_rows.append(_decision_row(review_date, ticker, "ADD", 0.0, 0.0, reason, {"thesis_status": "New", "thesis_score": row.get("thesis_score"), "conviction_score": memory["latest_conviction_score"], "exit_score": 0, "reason": reason}, memory))

        _weight_holdings(holdings)
        interval_ret = _interval_return(prices_dict, {t: float(h["weight"]) for t, h in holdings.items()}, review_date, next_date)
        equity *= (1.0 + interval_ret)
        if benchmark_returns is not None and len(benchmark_returns):
            bench_slice = benchmark_returns.loc[review_date:next_date].dropna()
            benchmark_equity *= float((1.0 + bench_slice).prod()) if len(bench_slice) else 1.0
        held_scores = snapshot[snapshot["ticker"].astype(str).isin(holdings.keys())]
        status_counts = review["thesis_status"].value_counts().to_dict() if not review.empty and "thesis_status" in review else {}
        new_weights = {t: float(h["weight"]) for t, h in holdings.items()}
        turnover = 0.5 * sum(abs(new_weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in set(new_weights) | set(prev_weights))
        prev_weights = dict(new_weights)
        evolution_rows.append({
            "date": str(review_date.date()), "n_positions": int(len(holdings)),
            "avg_thesis_score": float(pd.to_numeric(held_scores.get("thesis_score", pd.Series(dtype=float)), errors="coerce").mean()) if not held_scores.empty else np.nan,
            "avg_conviction_score": float(np.mean([h.get("latest_conviction_score", np.nan) for h in holdings.values()])) if holdings else np.nan,
            "avg_position_health_score": float(pd.to_numeric(review.get("position_health_score", pd.Series(dtype=float)), errors="coerce").mean()) if not review.empty else np.nan,
            "avg_thesis_persistence_score": float(np.mean([h.get("thesis_persistence_score", np.nan) for h in holdings.values()])) if holdings else np.nan,
            "n_improving": int(status_counts.get("Improving", 0)), "n_intact": int(status_counts.get("Intact", 0)), "n_maturing": int(status_counts.get("Maturing", 0)), "n_weakening": int(status_counts.get("Weakening", 0)), "n_broken": int(status_counts.get("Broken", 0)),
            "monthly_turnover": float(turnover), "portfolio_equity": float(equity), "benchmark_equity": float(benchmark_equity), "alpha_equity": float(equity - benchmark_equity),
        })
        for ticker, h in holdings.items():
            cum_return = _cumulative_return(prices_dict, ticker, pd.Timestamp(h["entry_date"]), review_date)
            holdings_rows.append({"date": str(review_date.date()), "ticker": ticker, "weight": float(h["weight"]), "entry_date": str(pd.Timestamp(h["entry_date"]).date()), "months_since_entry": int(h.get("months_since_entry", 0)), "months_thesis_intact": int(h.get("months_thesis_intact", 0)), "thesis_persistence_score": float(h.get("thesis_persistence_score", np.nan)), "cumulative_return": cum_return, "original_opportunity_type": h.get("original_opportunity_type", ""), "original_buy_reason": h.get("original_buy_reason", "")})

    evolution = pd.DataFrame(evolution_rows); transactions = pd.DataFrame(transaction_rows); monthly_holdings = pd.DataFrame(holdings_rows); decision_log = pd.DataFrame(decision_rows)
    turnover = _build_turnover(evolution, transactions, closed_holding_months, monthly_holdings)
    summary = {
        "review_frequency": _normalize_freq(review_frequency or PORTFOLIO_REVIEW_FREQUENCY),
        "start_date": evolution["date"].iloc[0] if not evolution.empty else "", "end_date": evolution["date"].iloc[-1] if not evolution.empty else "",
        "ending_positions": int(evolution["n_positions"].iloc[-1]) if not evolution.empty else 0,
        "portfolio_equity_final": float(evolution["portfolio_equity"].iloc[-1]) if not evolution.empty else 1.0,
        "benchmark_equity_final": float(evolution["benchmark_equity"].iloc[-1]) if not evolution.empty else 1.0,
        "alpha_equity_final": float(evolution["alpha_equity"].iloc[-1]) if not evolution.empty else 0.0,
        "average_holding_period_months": float(turnover["average_holding_period"].dropna().iloc[-1]) if not turnover.empty and turnover["average_holding_period"].notna().any() else np.nan,
        "annual_turnover_latest": float(turnover["annual_turnover"].dropna().iloc[-1]) if not turnover.empty and turnover["annual_turnover"].notna().any() else np.nan,
    }
    if output_dir is not None:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        evolution.to_csv(out / "portfolio_evolution.csv", index=False); transactions.to_csv(out / "portfolio_transactions.csv", index=False); monthly_holdings.to_csv(out / "portfolio_monthly_holdings.csv", index=False); decision_log.to_csv(out / "portfolio_decision_log.csv", index=False); turnover.to_csv(out / "portfolio_turnover.csv", index=False)
        (out / "portfolio_monthly_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return evolution, transactions, monthly_holdings, summary


def _build_turnover(evolution: pd.DataFrame, transactions: pd.DataFrame, closed_holding_months: list[int], holdings: pd.DataFrame) -> pd.DataFrame:
    if evolution.empty:
        return pd.DataFrame(columns=["date", "monthly_turnover", "annual_turnover", "buys_per_year", "sells_per_year", "average_holding_period", "median_holding_period", "churn_by_reason"])
    out = evolution[["date", "monthly_turnover"]].copy()
    out["year"] = pd.to_datetime(out["date"]).dt.year
    annual = out.groupby("year")["monthly_turnover"].sum().rename("annual_turnover")
    out = out.merge(annual, on="year", how="left")
    if not transactions.empty:
        tx = transactions.copy(); tx["year"] = pd.to_datetime(tx["date"]).dt.year
        buys = tx[tx["action"].eq("ADD")].groupby("year").size().rename("buys_per_year")
        sells = tx[tx["action"].isin(["SELL", "REDUCE"])].groupby("year").size().rename("sells_per_year")
        reason = tx[tx["action"].isin(["SELL", "REDUCE"])] ["reason"].astype(str).value_counts().head(5).to_dict()
        out = out.merge(buys, on="year", how="left").merge(sells, on="year", how="left")
        out["churn_by_reason"] = json.dumps(reason, default=str)
    else:
        out["buys_per_year"] = 0; out["sells_per_year"] = 0; out["churn_by_reason"] = "{}"
    periods = list(closed_holding_months)
    if not holdings.empty and "months_since_entry" in holdings:
        periods += pd.to_numeric(holdings.groupby("ticker")["months_since_entry"].max(), errors="coerce").dropna().astype(int).tolist()
    out["average_holding_period"] = float(np.mean(periods)) if periods else np.nan
    out["median_holding_period"] = float(np.median(periods)) if periods else np.nan
    return out
