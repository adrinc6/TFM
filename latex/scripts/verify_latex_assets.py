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
ASSETS = LATEX / "assets"
# Documentos maestros: la memoria y la presentación de defensa. Ambos viven en
# latex/ y comparten las figuras de assets/, así que se validan igual.
MAESTROS = ("main.tex", "presentacion.tex")
INCLUDE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
TABLE = re.compile(r"\\input\{(assets/t[^}]+\.tex)\}")
CHAPTER = re.compile(r"\\input\{(assets/[^}]+)\}")
# El documento no usa biblatex: la bibliografía es un capítulo manual. Si
# reaparece cualquiera de estos comandos, la compilación vuelve a romperse.
BIB_COMMAND = re.compile(r"\\(?:textcite|parencite|autocite|cite|printbibliography|addbibresource)\b")
LABEL = re.compile(r"\\label\{([^}]+)\}")
REF = re.compile(r"\\(?:ref|eqref|autoref)\{([^}]+)\}")
MOJIBAKE = ("Ã", "Â", "�")


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def main() -> int:
    errors: list[str] = []
    # Capítulos, tablas y figuras conviven sueltos en assets/, sin subcarpetas
    # dentro de assets/. Los documentos maestros viven un nivel por encima, en
    # latex/, así que todo \input y \includegraphics debe llevar el prefijo
    # "assets/". La presentación se coloca junto a main.tex precisamente para
    # poder reutilizar esas mismas rutas sin duplicar ninguna figura.
    tex_files = [LATEX / nombre for nombre in MAESTROS if (LATEX / nombre).is_file()]
    tex_files.extend(sorted(ASSETS.glob("*.tex")))
    used_graphics: set[str] = set()
    used_tables: set[str] = set()
    labels: set[str] = set()
    references: list[tuple[str, str]] = []
    for path in tex_files:
        content = path.read_text(encoding="utf-8")
        if any(marker in content for marker in MOJIBAKE):
            fail(f"Mojibake en {path.relative_to(ROOT)}", errors)
        if BIB_COMMAND.search(content):
            fail(
                f"Comando de bibliografía en {path.name}: el documento no usa biblatex, "
                "las citas van en texto plano autor-año",
                errors,
            )
        for graphic in INCLUDE.findall(content):
            used_graphics.add(graphic)
            if not graphic.startswith("assets/"):
                fail(f"Ruta de figura sin prefijo assets/ en {path.name}: {graphic}", errors)
            if Path(graphic).is_absolute() or ":" in graphic:
                fail(f"Ruta absoluta de figura en {path.name}: {graphic}", errors)
            if not (LATEX / graphic).is_file():
                fail(f"Figura inexistente: {graphic}", errors)
        for table in TABLE.findall(content):
            used_tables.add(table)
            if not (LATEX / table).is_file():
                fail(f"Tabla inexistente: {table}", errors)
        for chapter in CHAPTER.findall(content):
            target = LATEX / chapter
            if target.suffix != ".tex":
                target = target.with_suffix(".tex")
            if not target.is_file():
                fail(f"Capítulo inexistente: {chapter}", errors)
        references.extend((path.name, key) for key in REF.findall(content))

    for path in tex_files:
        labels.update(LABEL.findall(path.read_text(encoding="utf-8")))

    # Una \ref sin \label compila pero imprime «??» en el PDF. Sin compilador
    # local es el fallo más fácil de dejar escapar tras renumerar capítulos.
    for origin, key in references:
        if key not in labels:
            fail(f"Referencia sin destino en {origin}: \\ref{{{key}}}", errors)

    # Activos generados pero no insertados en ningún capítulo.
    for figure in sorted(ASSETS.glob("f*.png")):
        if f"assets/{figure.name}" not in used_graphics:
            fail(f"Figura huérfana (generada pero no insertada): {figure.name}", errors)
    for table in sorted(ASSETS.glob("t*.tex")):
        if f"assets/{table.name}" not in used_tables:
            fail(f"Tabla huérfana (generada pero no insertada): {table.name}", errors)

    pdfs = list(ASSETS.glob("*.pdf"))
    if pdfs:
        fail("Hay figuras PDF: " + ", ".join(path.name for path in pdfs), errors)
    if errors:
        print("VALIDACIÓN FALLIDA")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(
        "VALIDACIÓN CORRECTA: rutas explícitas, recursos existentes, UTF-8, "
        "referencias cruzadas resueltas y sin comandos de bibliografía."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
