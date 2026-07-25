"""Entrada única del proyecto."""

from module.common.utils import setup_logging
from module.web.api import serve


if __name__ == "__main__":
    setup_logging()
    serve()
