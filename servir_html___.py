"""Levanta un servidor HTTP local para ver los informes HTML con sus CSVs.

Los informes por run cargan tablas grandes desde CSVs sueltos mediante `fetch`, que los
navegadores bloquean al abrir un fichero local directamente (file://). Este script sirve el
directorio de resultados por HTTP para que las tablas carguen.

Uso:
    python servir_html.py                      # sirve data/processed y results en el puerto 8000
    python servir_html.py results/escenarios   # sirve un directorio concreto
    python servir_html.py . 8080               # directorio y puerto a medida

Luego abre en el navegador, por ejemplo:
    http://localhost:8000/results/escenarios/comparison.html
"""

from __future__ import annotations

import http.server
import socketserver
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    directory = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT)
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(
        *args, directory=directory, **kwargs
    )
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"Sirviendo {directory} en http://localhost:{port}")
        print("Ctrl+C para parar. Abre p. ej. http://localhost:%d/results/escenarios/comparison.html" % port)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServidor detenido.")


if __name__ == "__main__":
    main()
