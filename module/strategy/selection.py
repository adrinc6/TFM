"""Entry selection, buy-today logic and watchlist generation."""

from __future__ import annotations

import logging

import pandas as pd

from environment import PROCESSED_DIR, Settings
from module.utils import read_parquet, write_parquet

log = logging.getLogger(__name__)


WATCHLIST_COLUMNS = [
    "snapshot_date",
    "ticker",
    "opportunity_type",
    "investment_thesis",
    "catalyst",
    "moat_analysis",
    "valuation_score",
    "conviction_score",
    "entry_trigger",
]


def add_buy_today_decision(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rows = []
    for _, group in df.groupby("snapshot_date", sort=True):
        ranked = group.sort_values("thesis_rank_score", ascending=False)
        best_by_ticker = _best_alternatives(ranked)
        group = group.copy()
        group["best_alternative_ticker"] = group["ticker"].map(lambda ticker: best_by_ticker[ticker]["ticker"])
        group["best_alternative_score"] = group["ticker"].map(lambda ticker: best_by_ticker[ticker]["score"])
        group["opportunity_cost_score"] = (group["best_alternative_score"] - group["thesis_rank_score"]).clip(lower=0)
        valuation = group.get("price_adjusted_valuation_score", group["valuation_score"])
        momentum = group.get("momentum_score", pd.Series(0.5, index=group.index))
        group["buy_today_score"] = (
            0.28 * group["thesis_score"]
            + 0.24 * group["business_quality_score"]
            + 0.17 * group["positive_expectation_gap"]
            + 0.15 * valuation
            + 0.08 * momentum
            + 0.08 * (1 - group["opportunity_cost_score"].clip(0, 1))
        ).clip(0, 1)
        group["would_buy_today"] = (
            (group["buy_today_score"] >= 0.60)
            & ((momentum >= 0.35) | ((valuation >= 0.72) & (group["business_quality_score"] >= 0.58)))
            & group["thesis_state"].isin({"Improving", "Intact"})
            & ~group["opportunity_type"].isin({"Avoid", "Value Trap"})
        )
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def build_watchlist(settings: Settings) -> pd.DataFrame:
    from module.research.thesis import enrich_with_thesis_scores

    scored = read_parquet(PROCESSED_DIR / "scored_universe.parquet")
    log.info("Building watchlist from scored universe rows=%s", len(scored))
    research = enrich_with_thesis_scores(scored)
    full_watchlist = (
        research[
            (~research["opportunity_type"].isin({"Avoid", "Value Trap"}))
            & (research["conviction_score"] >= 0.35)
            & (research["business_quality_score"] >= 0.40)
        ]
        .sort_values(["snapshot_date", "business_quality_score", "conviction_score"], ascending=[True, False, False])
        .groupby("snapshot_date", as_index=False)
        .head(200)
    )
    latest_date = sorted(research["snapshot_date"].unique())[-1]
    watchlist = full_watchlist[full_watchlist["snapshot_date"] == latest_date].copy()
    if watchlist.empty:
        latest = research[research["snapshot_date"] == latest_date].copy()
        watchlist = latest.sort_values("business_quality_score", ascending=False).head(10)
        full_watchlist = pd.concat([full_watchlist, watchlist], ignore_index=True)
    watchlist = watchlist[WATCHLIST_COLUMNS]
    full_watchlist = full_watchlist[WATCHLIST_COLUMNS]
    write_parquet(watchlist, PROCESSED_DIR / "watchlist.parquet")
    settings.run_dir.mkdir(parents=True, exist_ok=True)
    stale_history = settings.run_dir / "watchlist_history.csv"
    if stale_history.exists():
        stale_history.unlink()
    watchlist.to_csv(settings.run_dir / "watchlist.csv", index=False)
    audit_dir = settings.run_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    full_watchlist.to_csv(audit_dir / "watchlist_history.csv", index=False)
    return watchlist


def _best_alternatives(group: pd.DataFrame) -> dict[str, dict]:
    alternatives = {}
    records = group[["ticker", "thesis_rank_score"]].to_dict("records")
    for record in records:
        other = [candidate for candidate in records if candidate["ticker"] != record["ticker"]]
        if not other:
            alternatives[record["ticker"]] = {"ticker": "", "score": float(record["thesis_rank_score"])}
            continue
        best = max(other, key=lambda item: item["thesis_rank_score"])
        alternatives[record["ticker"]] = {"ticker": best["ticker"], "score": float(best["thesis_rank_score"])}
    return alternatives
