"""Entrada única del proyecto: descarga de datos crudos.

Tras el reinicio en limpio, lo único que existe es la descarga. Las siguientes
etapas (dataset point-in-time, features, agentes ML, cartera, backtest,
experimentos e informes HTML) se reconstruirán siguiendo el plan maestro en
`docs/doc.md`.

Ejecutar con:
    python main.py
"""

from __future__ import annotations

import logging

from environment import Settings, ensure_directories
from module.ingest.pipeline import download_raw_data
from module.utils import setup_logging


log = logging.getLogger(__name__)


def main() -> None:
    ensure_directories()
    settings = Settings()
    setup_logging()
    log.info(
        "Descarga de datos crudos: tickers=%s rango=%s..%s dev_mode=%s",
        len(settings.tickers),
        settings.data_start_date,
        settings.end_date,
        settings.dev_mode,
    )
    download_raw_data(settings)
    log.info("Descarga completa.")


if __name__ == "__main__":
    main()
