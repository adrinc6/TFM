"""Entrada única del proyecto.

Dos subcomandos, sin framework de CLI:

- ``python main.py ingest``: descarga y consolida los datos crudos en ``data/raw``. Es el paso que
  permite regenerar el proyecto desde cero; sin esta entrada, el subárbol de ingesta no tenía ningún
  consumidor y el trabajo no podía reproducir sus propios datos.
- ``python main.py serve`` (por defecto): levanta la API y el dashboard.
"""

import sys

from module.common.utils import setup_logging


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "serve"
    setup_logging()
    if command == "ingest":
        from environment import Settings
        from module.data.ingest.pipeline import download_raw_data

        download_raw_data(Settings())
        return 0
    if command == "serve":
        from module.web.api import serve

        serve()
        return 0
    print(f"Comando desconocido: {command!r}. Usa 'ingest' o 'serve'.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
