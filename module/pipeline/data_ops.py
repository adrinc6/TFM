# =============================================================================
# module/pipeline/data_ops.py
# Descarga y preparación de datos usando Finnhub + Yahoo Finance HTTP.
# =============================================================================
"""
Responsabilidades:
  - download_data   : descarga todos los datos via Finnhub + Yahoo Finance HTTP.
  - prepare_data    : consolida datos SEC crudos en data_finnhub/consolidated/.
  - get_available_tickers: filtra tickers con datos completos en disco.
"""
import logging
from pathlib import Path
from typing import List, Tuple, Dict

log = logging.getLogger(__name__)


def download_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    data_dir: str,
    force_download: bool = False,
    api_key: str = "",
) -> List[str]:
    """
    Descarga todos los datos necesarios via Finnhub + Yahoo Finance HTTP.

    Args:
        tickers:        Lista de tickers a descargar.
        start_date:     Fecha de inicio (YYYY-MM-DD).
        end_date:       Fecha de fin (YYYY-MM-DD).
        data_dir:       Directorio raíz Finnhub (data_finnhub/).
        force_download: Si True, fuerza re-descarga ignorando el registro.
        api_key:        API key de Finnhub.

    Returns:
        Lista de tickers (igual que la entrada).
    """
    from module.fetcher_finnhub import fetch_all_finnhub

    log.info("PASO 1 — DESCARGANDO DATOS (Finnhub + Yahoo Finance)")
    log.info(f"  Tickers: {len(tickers)} | {start_date} → {end_date}")
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
    """
    Consolida los datos SEC crudos en data_finnhub/consolidated/{TICKER}.csv.

    Procesa cada ticker generando un CSV con columnas estandarizadas
    listo para ser consumido por DataRouter y los builders de features.

    Args:
        tickers:  Lista de tickers a consolidar.
        data_dir: Directorio raíz Finnhub (data_finnhub/).
    """
    from module.finnhub_processor import FinnhubConsolidator

    log.info("PASO 2 — CONSOLIDANDO DATOS (FinnhubConsolidator)")
    consolidator = FinnhubConsolidator(finnhub_data_dir=data_dir)
    consolidator.process_all_tickers(tickers)
    log.info("Consolidación completada.")


def get_available_tickers(
    tickers: List[str],
    data_dir: str,
) -> Tuple[List[str], Dict]:
    """
    Filtra la lista de tickers a los que tienen datos completos en disco:
      - data_finnhub/{TICKER}/prices.json
      - data_finnhub/consolidated/{TICKER}.csv

    Args:
        tickers:  Lista completa de tickers del universo.
        data_dir: Directorio raíz Finnhub (data_finnhub/).

    Returns:
        Tupla (available, missing_detail) donde:
          - available:      lista de tickers con datos completos, ordenada.
          - missing_detail: dict {ticker: lista de archivos que faltan}.
    """
    data_path         = Path(data_dir)
    consolidated_dir  = data_path / "consolidated"

    available      = []
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
    """
    Reintenta descargar y consolidar datos para los tickers con datos faltantes.

    Args:
        missing_detail: Dict {ticker: ['prices', 'consolidated']}.
        start_date:     Fecha de inicio (YYYY-MM-DD).
        end_date:       Fecha de fin (YYYY-MM-DD).
        data_dir:       Directorio raíz Finnhub.
        api_key:        API key de Finnhub.

    Returns:
        Lista de tickers recuperados exitosamente.
    """
    from module.fetcher_finnhub import fetch_all_finnhub
    from module.finnhub_processor import FinnhubConsolidator

    if not missing_detail:
        return []

    tickers_missing = list(missing_detail.keys())
    log.info(f"REINTENTANDO {len(tickers_missing)} tickers con datos incompletos...")

    # Re-descarga (force=True solo para los que faltan)
    fetch_all_finnhub(
        tickers=tickers_missing,
        start=start_date,
        end=end_date,
        base_dir=data_dir,
        api_key=api_key,
        force=True,
    )

    # Re-consolidar los que necesitan consolidado
    need_consolidated = [t for t, m in missing_detail.items() if "consolidated" in m]
    if need_consolidated:
        consolidator = FinnhubConsolidator(finnhub_data_dir=data_dir)
        consolidator.process_all_tickers(need_consolidated)

    # Verificar cuáles se han recuperado
    data_path        = Path(data_dir)
    consolidated_dir = data_path / "consolidated"
    recovered = [
        t for t in tickers_missing
        if (data_path / t / "prices.json").exists()
        and (consolidated_dir / f"{t}.csv").exists()
    ]
    still_missing = set(tickers_missing) - set(recovered)

    log.info(f"  Recuperados: {recovered}")
    if still_missing:
        log.info(f"  Siguen sin datos: {still_missing}")

    return recovered
