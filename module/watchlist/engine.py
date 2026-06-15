"""Build and persist the strategy watchlist."""

from __future__ import annotations

import logging

import pandas as pd

from environment import PROCESSED_DIR, Settings
from module.common.io import read_parquet, write_parquet
from module.research.thesis_generator import generate_research
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
    research = generate_research(enrich_with_thesis_scores(scored))
    latest_date = sorted(research["snapshot_date"].unique())[-1]
    latest = research[research["snapshot_date"] == latest_date].copy()
    watchlist = latest[
        (~latest["opportunity_type"].isin({"Avoid", "Value Trap"}))
        & (latest["conviction_score"] >= 0.35)
        & (latest["business_quality_score"] >= 0.40)
    ].sort_values(["business_quality_score", "conviction_score"], ascending=False)
    if watchlist.empty:
        watchlist = latest.sort_values("business_quality_score", ascending=False).head(10)
    watchlist = watchlist[WATCHLIST_COLUMNS]
    write_parquet(watchlist, PROCESSED_DIR / "watchlist.parquet")
    settings.run_dir.mkdir(parents=True, exist_ok=True)
    watchlist.to_csv(settings.run_dir / "watchlist.csv", index=False)
    log.info("Watchlist rows: %s", len(watchlist))
    return watchlist
