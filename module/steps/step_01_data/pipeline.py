"""Step 01: data download and consolidation entry points."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from module.steps.step_01_data.consolidation import FinnhubConsolidator
from module.steps.step_01_data.downloaders import fetch_all_finnhub

log = logging.getLogger(__name__)


def download_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    data_dir: str,
    force_download: bool = False,
    api_key: str = "",
) -> List[str]:
    log.info("PASO 1 — DESCARGANDO DATOS (Finnhub + Yahoo Finance)")
    log.info(f"  Tickers: {len(tickers)} | {start_date} -> {end_date}")
    log.info(f"  Destino: {data_dir}")

    fetch_all_finnhub(
        tickers=tickers,
        start=start_date,
        end=end_date,
        base_dir=data_dir,
        api_key=api_key,
        force=force_download,
    )

    log.info("  Descarga completada.")
    return tickers


def prepare_data(tickers: List[str], data_dir: str) -> None:
    log.info("PASO 2 — CONSOLIDANDO DATOS (FinnhubConsolidator)")
    consolidator = FinnhubConsolidator(finnhub_data_dir=data_dir)
    consolidator.process_all_tickers(tickers)
    log.info("Consolidacion completada.")


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
        log.info(f"Tickers sin datos completos: {len(missing_detail)}")
        for ticker, missing in sorted(missing_detail.items()):
            log.debug(f"  {ticker}: falta {missing}")
    log.info(f"Tickers con datos completos: {len(available)} / {len(tickers)}")
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
    log.info(f"REINTENTANDO {len(tickers_missing)} tickers con datos incompletos...")

    fetch_all_finnhub(
        tickers=tickers_missing,
        start=start_date,
        end=end_date,
        base_dir=data_dir,
        api_key=api_key,
        force=True,
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

    log.info(f"  Recuperados: {recovered}")
    if still_missing:
        log.info(f"  Siguen sin datos: {still_missing}")

    return recovered
