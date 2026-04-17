"""Step 01: data download and consolidation entry points."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from module.steps.step_01_data.consolidation import FinnhubConsolidator
from module.steps.step_01_data.downloaders import run_download
from module.steps.step_01_data.registry import Registry

log = logging.getLogger(__name__)


def download_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    data_dir: str,
    force_download: bool = False,
    api_key: str = "",
    prices_only: bool = False,
    allow_retry_failed: bool = False,
) -> List[str]:
    log.info("=" * 60)
    log.info("  STEP 1 — DATA DOWNLOAD")
    log.info(f"  Tickers : {len(tickers)} | Period: {start_date} → {end_date}")
    log.info(f"  Destination : {data_dir}")
    log.info("=" * 60)

    if not api_key:
        raise ValueError(
            "FINNHUB_API_KEY not configured. "
            "Set the FINNHUB_API_KEY environment variable before running the pipeline."
        )

    run_download(
        api_key=api_key,
        tickers=tickers,
        start=start_date,
        end=end_date,
        base_dir=data_dir,
        force=force_download,
        prices_only=prices_only,
        allow_retry_failed=allow_retry_failed,
    )

    log.info("  Download completed. Proceeding to consolidation...")
    return tickers


def prepare_data(tickers: List[str], data_dir: str) -> None:
    log.info("=" * 60)
    log.info("  STEP 2 — DATA CONSOLIDATION")
    log.info(f"  Normalising and merging Finnhub files for {len(tickers)} tickers...")
    log.info("=" * 60)
    consolidator = FinnhubConsolidator(finnhub_data_dir=data_dir)
    consolidator.process_all_tickers(tickers)
    log.info("  Consolidation completed.")


def get_available_tickers(
    tickers: List[str],
    data_dir: str,
) -> Tuple[List[str], Dict]:
    data_path = Path(data_dir)
    consolidated_dir = data_path / "consolidated"

    available = []
    missing_detail: Dict = {}

    for t in tickers:
        missing = []
        if not (data_path / t / "prices.json").exists():
            missing.append("prices")
        if not (consolidated_dir / f"{t}.csv").exists():
            missing.append("consolidated")
        if missing:
            missing_detail[t] = missing
        else:
            available.append(t)

    available = sorted(available)

    if missing_detail:
        log.info(f"  Tickers without complete data: {len(missing_detail)} (missing prices or consolidated)")
        for ticker, missing in sorted(missing_detail.items()):
            log.debug(f"    {ticker}: missing {missing}")
    log.info(f"  Tickers ready for pipeline: {len(available)} / {len(tickers)}")
    return available, missing_detail


def retry_missing_tickers(
    missing_detail: Dict,
    start_date: str,
    end_date: str,
    data_dir: str,
    api_key: str = "",
) -> List[str]:
    if not missing_detail:
        return []

    tickers_missing = list(missing_detail.keys())
    log.info(f"  Retrying download for {len(tickers_missing)} incomplete tickers...")

    # Avoids deleting the global registry: only clears entries for the tickers to retry.
    # This preserves the rest of the universe state and avoids triggering a mass re-download.
    registry = Registry(Path(data_dir))
    for ticker in tickers_missing:
        registry.clear(group=ticker)

    run_download(
        api_key=api_key,
        tickers=tickers_missing,
        start=start_date,
        end=end_date,
        base_dir=data_dir,
        force=False,
        allow_retry_failed=True,
    )

    need_consolidated = [t for t, m in missing_detail.items() if "consolidated" in m]
    if need_consolidated:
        consolidator = FinnhubConsolidator(finnhub_data_dir=data_dir)
        consolidator.process_all_tickers(need_consolidated)

    data_path = Path(data_dir)
    consolidated_dir = data_path / "consolidated"
    recovered = [
        t
        for t in tickers_missing
        if (data_path / t / "prices.json").exists()
        and (consolidated_dir / f"{t}.csv").exists()
    ]
    still_missing = set(tickers_missing) - set(recovered)

    log.info(f"  Recovered after retry: {len(recovered)} tickers  {recovered}")
    if still_missing:
        log.warning(f"  Still missing after retry ({len(still_missing)}): {sorted(still_missing)}")

    return recovered
