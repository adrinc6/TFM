"""Build and persist the strategy watchlist."""

from __future__ import annotations

import logging

import pandas as pd

from environment import PROCESSED_DIR, Settings
from module.common.io import read_parquet, write_parquet
from module.thesis.intelligence import enrich_with_thesis_scores

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


def build_watchlist(settings: Settings) -> pd.DataFrame:
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
    log.info(
        "Watchlist latest_date=%s latest_rows=%s history_rows=%s history_dates=%s",
        latest_date,
        len(watchlist),
        len(full_watchlist),
        full_watchlist["snapshot_date"].nunique(),
    )
    return watchlist
