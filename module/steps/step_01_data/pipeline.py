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
    log.info("  PASO 1 — DESCARGA DE DATOS")
    log.info(f"  Tickers : {len(tickers)} | Periodo: {start_date} → {end_date}")
    log.info(f"  Destino : {data_dir}")
    log.info("=" * 60)

    if not api_key:
        raise ValueError(
            "FINNHUB_API_KEY no configurada. "
            "Establece la variable de entorno FINNHUB_API_KEY antes de ejecutar el pipeline."
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

    log.info("  Descarga completada. Continuando con consolidación...")
    return tickers


def prepare_data(tickers: List[str], data_dir: str) -> None:
    log.info("=" * 60)
    log.info("  PASO 2 — CONSOLIDACIÓN DE DATOS")
    log.info(f"  Normalizando y mergeando ficheros Finnhub para {len(tickers)} tickers...")
    log.info("=" * 60)
    consolidator = FinnhubConsolidator(finnhub_data_dir=data_dir)
    consolidator.process_all_tickers(tickers)
    log.info("  Consolidación completada.")


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
        log.info(f"  Tickers sin datos completos: {len(missing_detail)} (sin precios o consolidado)")
        for ticker, missing in sorted(missing_detail.items()):
            log.debug(f"    {ticker}: falta {missing}")
    log.info(f"  Tickers listos para el pipeline: {len(available)} / {len(tickers)}")
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
    log.info(f"  Reintentando descarga para {len(tickers_missing)} tickers incompletos...")

    # Evita borrar el registry global: limpia solo las entradas de tickers a reintentar.
    # Asi se preserva el estado del resto del universo y no se dispara re-descarga masiva.
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

    log.info(f"  Recuperados tras reintento: {len(recovered)} tickers  {recovered}")
    if still_missing:
        log.warning(f"  Siguen sin datos tras reintento ({len(still_missing)}): {sorted(still_missing)}")

    return recovered
