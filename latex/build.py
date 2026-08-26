"""Punto único de entrada para todo lo que hay que hacer con el LaTeX del TFM.

Se ejecuta desde cualquier sitio::

    python latex/build.py

Qué hace en cada ejecución se decide con los interruptores `True`/`False` del bloque
CONFIGURACIÓN, justo debajo. No hay que recordar comandos ni identificadores de estudio: los
identificadores se leen de `latex/asset_manifest.json`, de modo que este fichero no es una copia
más de esa información.

Los interruptores se pueden sobrescribir sin editar el fichero, para una ejecución suelta::

    python latex/build.py --solo-memoria      # compila TFM.tex y nada más
    python latex/build.py --solo-defensa      # compila TFM_ppt.tex y nada más
    python latex/build.py --activos           # regenera figuras, tablas y macros, y verifica
    python latex/build.py --todo              # regenera, verifica y compila las dos
    python latex/build.py --notas             # la defensa con las notas del ponente
    python latex/build.py --limpiar           # borra los auxiliares y termina

Los pasos se ejecutan en el orden en que aparecen abajo y la ejecución se detiene en el primer
fallo, porque encadenarlos con un paso roto produce un PDF que parece correcto y no lo es.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# =====================================================================
# CONFIGURACIÓN · pon a True lo que quieras que haga esta ejecución
# =====================================================================

# --- Activos: figuras, cuerpos de tabla y macros numéricas -----------
# Regenerarlos requiere los .parquet de la evidencia, que no están versionados: solo funciona en
# la instalación donde corrieron los estudios. Déjalo en False si solo quieres el PDF.
REGENERAR_ACTIVOS = False

# Recalcula las macros en memoria y comprueba que macros, manifiesto y activos coinciden con los
# estudios adoptados. No escribe nada. Es barato y detecta que alguien tocó un activo a mano.
AUDITAR_ACTIVOS = False

# Rutas relativas, recursos existentes, UTF-8 sin mojibake, referencias cruzadas con destino y
# ningún activo huérfano. También es barato.
VERIFICAR_PROYECTO = True

# --- Compilación ------------------------------------------------------
COMPILAR_MEMORIA = True   # TFM.tex      -> TFM.pdf
COMPILAR_DEFENSA = True   # TFM_ppt.tex  -> TFM_ppt.pdf

# Compila además una versión de la defensa con el guion del ponente intercalado, pensada para la
# segunda pantalla. Sale como TFM_ppt_notes.pdf y no sustituye a la normal.
DEFENSA_CON_NOTAS = True

# --- Salida -----------------------------------------------------------
# Copia los PDF resultantes a latex/TFM.pdf, latex/TFM_ppt.pdf y latex/TFM_ppt_notes.pdf, que es
# donde se buscan y donde el repositorio los versiona: `.gitignore` solo excluye la carpeta de
# trabajo `latex/build/`. Con False se quedan solo en esa carpeta de trabajo.
COPIAR_PDF_AL_REPO = True

# Borra .aux, .log, .toc, .fls y compañía al terminar. La carpeta de trabajo se conserva igualmente
# porque contiene los registros, que hacen falta para diagnosticar.
LIMPIAR_AUXILIARES = True

# Muestra la salida completa del compilador en lugar del resumen. Útil cuando algo falla y el
# registro no basta.
SALIDA_DETALLADA = False

# =====================================================================
# A partir de aquí no hace falta tocar nada
# =====================================================================

LATEX = Path(__file__).resolve().parent
ROOT = LATEX.parent
TRABAJO = LATEX / "build"
MANIFIESTO = LATEX / "asset_manifest.json"

VERDE, ROJO, AZUL, GRIS, FIN = "\033[92m", "\033[91m", "\033[94m", "\033[90m", "\033[0m"


def rotulo(texto: str) -> None:
    print(f"\n{AZUL}>> {texto}{FIN}")


def bien(texto: str) -> None:
    print(f"  {VERDE}OK{FIN} {texto}")


def mal(texto: str) -> None:
    print(f"  {ROJO}ERROR: {texto}{FIN}")


def aborta(motivo: str, pista: str = "") -> None:
    mal(motivo)
    if pista:
        print(f"    {GRIS}{pista}{FIN}")
    sys.exit(1)


def corre(orden: list[str], cwd: Path, detallada: bool) -> subprocess.CompletedProcess[str]:
    """Ejecuta un comando capturando su salida salvo que se pida verla entera."""
    if detallada:
        return subprocess.run(orden, cwd=cwd, text=True)
    return subprocess.run(orden, cwd=cwd, text=True, capture_output=True)


def identificadores() -> list[str]:
    """Los estudios adoptados, leídos del manifiesto en vez de copiados aquí.

    El manifiesto lo escribe el propio exportador, así que no puede desincronizarse: si alguien
    cambia de cadena, el siguiente export lo actualiza y este script lo sigue sin tocar nada.
    """
    if not MANIFIESTO.is_file():
        aborta(
            "No existe latex/asset_manifest.json",
            "Sin él no se sabe qué estudios alimentan el manuscrito. Ver COMO_COMPILAR.md §3.",
        )
    datos = json.loads(MANIFIESTO.read_text(encoding="utf-8"))
    orden = ["--study-id", datos["study_id"]]
    for study in datos["chain_study_ids"]:
        orden += ["--chain-study-id", study]
    orden += ["--portfolio-study-id", datos["portfolio_study_id"]]
    return orden


def hay_latexmk() -> bool:
    """¿Se puede usar latexmk? Solo si además hay un Perl que lo ejecute.

    latexmk es un script de Perl, no un binario. MiKTeX lo instala igualmente, de modo que
    `shutil.which("latexmk")` lo encuentra y aun así falla al ejecutarse con «MiKTeX could not find
    the script engine 'perl'». Git Bash trae su propio perl y PowerShell no, así que el mismo
    comando funciona en una consola y falla en la otra. Por eso se comprueban los dos.
    """
    return shutil.which("latexmk") is not None and shutil.which("perl") is not None


def exige_motor() -> None:
    if shutil.which("xelatex") is None:
        aborta(
            "No se encuentra xelatex en el PATH",
            "Hace falta una distribución LaTeX con XeLaTeX. Ver COMO_COMPILAR.md §1.",
        )


def exporta(auditar: bool, detallada: bool) -> None:
    """Regenera o audita los activos llamando al exportador con los estudios del manifiesto."""
    orden = [sys.executable, "latex/scripts/export_study_assets.py", *identificadores()]
    if auditar:
        orden.append("--audit")
    proceso = corre(orden, ROOT, detallada)
    if proceso.returncode:
        salida = (proceso.stderr or proceso.stdout or "") if not detallada else ""
        pista = "Faltan los .parquet de la evidencia: ver el aviso de COMO_COMPILAR.md §3." \
            if "parquet" in salida.lower() or "FileNotFoundError" in salida else salida.strip()[-600:]
        aborta("auditoría de activos fallida" if auditar else "regeneración de activos fallida", pista)
    bien("macros, manifiesto y activos coinciden" if auditar else "figuras, tablas y macros regeneradas")


def verifica(detallada: bool) -> None:
    proceso = corre([sys.executable, "latex/scripts/verify_latex_assets.py"], ROOT, detallada)
    if proceso.returncode:
        aborta("verificación fallida", (proceso.stdout or "").strip())
    bien("rutas, UTF-8, referencias cruzadas y activos sin huérfanos")


def prepara_defensa_con_notas() -> Path:
    """Deriva la defensa con notas del ponente a partir de la normal, sin tocar el original.

    Las dos versiones se diferencian en una sola línea —la opción de beamer que manda el guion a
    la segunda pantalla—, de modo que TFM_ppt_notes.tex no se escribe a mano: se regenera aquí
    desde TFM_ppt.tex cada vez que se compila, y así no puede quedarse desincronizado. Vive en
    latex/ junto al original, con lo que las rutas `figures/` y `tables/` siguen resolviendo.
    """
    original = (LATEX / "TFM_ppt.tex").read_text(encoding="utf-8")
    marca = "% \\setbeameroption{show notes on second screen=right}"
    if marca not in original:
        aborta(
            "TFM_ppt.tex no contiene la línea comentada de las notas",
            f"Se esperaba encontrar: {marca}",
        )
    destino = LATEX / "TFM_ppt_notes.tex"
    destino.write_text(original.replace(marca, marca[2:]), encoding="utf-8")
    return destino


def _pasadas_necesarias(registro: Path) -> bool:
    """¿Hace falta otra pasada? Lo dice el propio LaTeX en su registro."""
    if not registro.is_file():
        return True
    texto = registro.read_text(encoding="utf-8", errors="replace")
    return ("Rerun to get" in texto or "Rerun LaTeX" in texto
            or "There were undefined references" in texto)


def compila(fuente: Path, etiqueta: str, detallada: bool) -> Path:
    """Compila el documento y resume el resultado: errores, referencias rotas y páginas.

    Usa latexmk cuando está disponible con su Perl, porque decide él solo cuántas pasadas hacen
    falta. Cuando no lo está —el caso de PowerShell con MiKTeX— llama a xelatex directamente y
    repite mientras el registro pida otra pasada, hasta un máximo de cuatro. El documento necesita
    al menos dos: la primera escribe índice, listas y etiquetas, y la segunda las coloca.
    """
    TRABAJO.mkdir(parents=True, exist_ok=True)
    relativa = fuente.relative_to(LATEX).as_posix()
    registro = TRABAJO / f"{fuente.stem}.log"
    inicio = time.perf_counter()

    if hay_latexmk():
        corre(
            ["latexmk", "-xelatex", "-interaction=nonstopmode", "-f",
             f"-outdir={TRABAJO.as_posix()}", relativa],
            LATEX, detallada,
        )
    else:
        orden = ["xelatex", "-interaction=nonstopmode", "-halt-on-error=0",
                 f"-output-directory={TRABAJO.as_posix()}", relativa]
        for pasada in range(4):
            corre(orden, LATEX, detallada)
            if pasada >= 1 and not _pasadas_necesarias(registro):
                break
    segundos = time.perf_counter() - inicio

    if not registro.is_file():
        aborta(f"{etiqueta}: la compilación no llegó a producir un registro")
    texto = registro.read_text(encoding="utf-8", errors="replace")
    errores = [linea for linea in texto.splitlines() if linea.startswith("!")]
    if errores:
        mal(f"{etiqueta}: {len(errores)} error(es) de LaTeX")
        for linea in errores[:5]:
            print(f"    {GRIS}{linea}{FIN}")
        aborta(f"revisa {registro}", "COMO_COMPILAR.md §5 recoge los fallos más frecuentes.")

    pdf = TRABAJO / f"{fuente.stem}.pdf"
    if not pdf.is_file():
        aborta(f"{etiqueta}: no se generó el PDF", f"revisa {registro}")

    sueltas = texto.count("LaTeX Warning: Reference")
    resumen = f"{etiqueta}: {_paginas(texto)} en {segundos:.0f} s"
    bien(resumen if not sueltas else f"{resumen} · {ROJO}{sueltas} referencia(s) sin destino{FIN}")

    desbordes = [linea for linea in texto.splitlines() if "Overfull" in linea]
    graves = [linea for linea in desbordes if _puntos(linea) >= 5.0]
    if graves:
        print(f"    {GRIS}{len(desbordes)} aviso(s) de desbordamiento, {len(graves)} por encima de 5 pt{FIN}")
    return pdf


def _paginas(registro: str) -> str:
    """Cuántas páginas salieron, leídas del aviso final de xdvipdfmx.

    LaTeX parte las líneas del registro a unos ochenta caracteres sin avisar, de modo que
    «(102 pages» puede quedar cortado en dos. Se quitan los saltos antes de buscar; el registro no
    usa guiones de partición, así que unir las líneas reconstruye el texto original.
    """
    plano = registro.replace("\n", "")
    encontrado = re.search(r"Output written on.*?\((\d+) pages", plano)
    return f"{encontrado.group(1)} páginas" if encontrado else "PDF generado"


def _puntos(linea: str) -> float:
    """Cuántos puntos se sale una caja, para separar lo invisible de lo que se ve."""
    try:
        return float(linea.split("(")[1].split("pt")[0])
    except (IndexError, ValueError):
        return 0.0


def limpia() -> None:
    basura = (".aux", ".log", ".toc", ".lof", ".lot", ".out", ".fls",
              ".fdb_latexmk", ".xdv", ".nav", ".snm", ".vrb", ".synctex.gz")
    borrados = 0
    for fichero in TRABAJO.glob("*"):
        if not fichero.is_file():
            continue
        if fichero.suffix in basura or fichero.name.endswith(".synctex.gz"):
            fichero.unlink()
            borrados += 1
    bien(f"{borrados} fichero(s) auxiliar(es) borrado(s); los PDF se conservan")


def opciones() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compila y verifica el LaTeX del TFM. Sin argumentos usa la configuración del fichero.",
    )
    parser.add_argument("--todo", action="store_true", help="regenerar activos, verificar y compilar las dos")
    parser.add_argument("--activos", action="store_true", help="regenerar activos y verificar, sin compilar")
    parser.add_argument("--solo-memoria", action="store_true", help="compilar solo TFM.tex")
    parser.add_argument("--solo-defensa", action="store_true", help="compilar solo TFM_ppt.tex")
    parser.add_argument("--notas", action="store_true", help="añadir la defensa con notas del ponente")
    parser.add_argument("--limpiar", action="store_true", help="borrar auxiliares y terminar")
    parser.add_argument("--detallada", action="store_true", help="mostrar la salida completa de cada paso")
    return parser.parse_args()


def plan(args: argparse.Namespace) -> dict[str, bool]:
    """Los interruptores del fichero, con los argumentos de línea de comandos por encima."""
    actual = {
        "regenerar": REGENERAR_ACTIVOS,
        "auditar": AUDITAR_ACTIVOS,
        "verificar": VERIFICAR_PROYECTO,
        "memoria": COMPILAR_MEMORIA,
        "defensa": COMPILAR_DEFENSA,
        "notas": DEFENSA_CON_NOTAS,
        "copiar": COPIAR_PDF_AL_REPO,
        "limpiar": LIMPIAR_AUXILIARES,
        "detallada": SALIDA_DETALLADA or args.detallada,
    }
    if args.todo:
        actual |= {"regenerar": True, "auditar": True, "verificar": True,
                   "memoria": True, "defensa": True}
    if args.activos:
        actual |= {"regenerar": True, "auditar": True, "verificar": True,
                   "memoria": False, "defensa": False, "notas": False}
    if args.solo_memoria:
        actual |= {"regenerar": False, "memoria": True, "defensa": False, "notas": False}
    if args.solo_defensa:
        actual |= {"regenerar": False, "memoria": False, "defensa": True}
    if args.notas:
        actual["notas"] = True
    if args.limpiar:
        actual = dict.fromkeys(actual, False) | {"limpiar": True, "detallada": actual["detallada"]}
    return actual


def main() -> int:
    args = opciones()
    tareas = plan(args)

    print(f"{AZUL}TFM - LaTeX{FIN}")
    activas = [nombre for nombre, valor in tareas.items()
               if valor and nombre not in {"detallada", "copiar"}]
    print(f"{GRIS}Pasos: {', '.join(activas) if activas else 'ninguno'}{FIN}")
    if not activas:
        print(f"{GRIS}Todo en False. Edita el bloque CONFIGURACIÓN o usa --todo.{FIN}")
        return 0

    if tareas["regenerar"]:
        rotulo("Regenerando figuras, tablas y macros")
        exporta(auditar=False, detallada=tareas["detallada"])
    if tareas["auditar"]:
        rotulo("Auditando activos contra los estudios adoptados")
        exporta(auditar=True, detallada=tareas["detallada"])
    if tareas["verificar"]:
        rotulo("Verificando el proyecto")
        verifica(tareas["detallada"])

    producidos: list[tuple[Path, str]] = []
    if tareas["memoria"] or tareas["defensa"] or tareas["notas"]:
        exige_motor()
    if tareas["memoria"]:
        rotulo("Compilando la memoria")
        producidos.append((compila(LATEX / "TFM.tex", "memoria", tareas["detallada"]), "TFM.pdf"))
    if tareas["defensa"]:
        rotulo("Compilando la defensa")
        producidos.append(
            (compila(LATEX / "TFM_ppt.tex", "defensa", tareas["detallada"]), "TFM_ppt.pdf")
        )
    if tareas["notas"]:
        rotulo("Compilando la defensa con notas del ponente")
        fuente = prepara_defensa_con_notas()
        producidos.append(
            (compila(fuente, "defensa con notas", tareas["detallada"]), "TFM_ppt_notes.pdf")
        )

    if tareas["copiar"] and producidos:
        rotulo("Copiando los PDF al repositorio")
        for pdf, nombre in producidos:
            shutil.copy2(pdf, LATEX / nombre)
            bien(f"latex/{nombre} actualizado")

    if tareas["limpiar"]:
        rotulo("Limpiando auxiliares")
        limpia()

    print(f"\n{VERDE}Listo.{FIN} {GRIS}Salidas y registros en {TRABAJO}{FIN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
