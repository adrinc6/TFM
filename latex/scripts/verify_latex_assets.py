"""Verifica que el proyecto LaTeX sea portable a Overleaf.

Ejecutar desde la raíz del repositorio::

    python latex/scripts/verify_latex_assets.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LATEX = ROOT / "latex"
INCLUDE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
TABLE = re.compile(r"\\input\{(tablas/[^}]+)\}")
CITE = re.compile(r"\\(?:textcite|parencite|cite)\{([^}]+)\}")
ENTRY = re.compile(r"@\w+\{([^,]+),")
MOJIBAKE = ("Ã", "Â", "�")


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def main() -> int:
    errors: list[str] = []
    tex_files = [LATEX / "main.tex", *sorted((LATEX / "caps").glob("*.tex"))]
    bibliography = (LATEX / "referencias.bib").read_text(encoding="utf-8")
    keys = set(ENTRY.findall(bibliography))
    for path in tex_files:
        content = path.read_text(encoding="utf-8")
        if any(marker in content for marker in MOJIBAKE):
            fail(f"Mojibake en {path.relative_to(ROOT)}", errors)
        for graphic in INCLUDE.findall(content):
            if not graphic.startswith("figuras/"):
                fail(f"Ruta de figura no explícita en {path.name}: {graphic}", errors)
            if Path(graphic).is_absolute() or ":" in graphic:
                fail(f"Ruta absoluta de figura en {path.name}: {graphic}", errors)
            if not (LATEX / graphic).is_file():
                fail(f"Figura inexistente: {graphic}", errors)
        for table in TABLE.findall(content):
            if not (LATEX / table).is_file():
                fail(f"Tabla inexistente: {table}", errors)
        for citation_group in CITE.findall(content):
            for key in (item.strip() for item in citation_group.split(",")):
                if key and key not in keys:
                    fail(f"Cita sin bibliografía en {path.name}: {key}", errors)
    pdfs = list((LATEX / "figuras").glob("*.pdf"))
    if pdfs:
        fail("Hay figuras PDF: " + ", ".join(path.name for path in pdfs), errors)
    if errors:
        print("VALIDACIÓN FALLIDA")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("VALIDACIÓN CORRECTA: rutas explícitas, recursos existentes, UTF-8 y citas verificadas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
