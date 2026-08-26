# Cómo compilar el TFM

Guía práctica para generar `TFM.pdf` y `TFM_ppt.pdf` desde cero, regenerar las figuras y las
tablas, y desinstalarlo todo cuando el proyecto termine.

El documento se compila con **XeLaTeX**, no con pdfLaTeX. No es opcional: el preámbulo usa
`fontspec` para cargar fuentes OpenType y lee UTF-8 de forma nativa, sin `inputenc`. Con pdfLaTeX no
compila.

---

## 0. La vía rápida: `latex/build.py`

Hay un único script que hace todo lo que este documento explica por partes. Desde la raíz del
repositorio:

```powershell
python latex/build.py
```

Qué hace en cada ejecución se elige con los interruptores `True`/`False` del bloque
**CONFIGURACIÓN**, al principio del fichero:

| Interruptor | Qué activa |
|---|---|
| `REGENERAR_ACTIVOS` | Rehace figuras, cuerpos de tabla y macros desde los estudios. Necesita los `.parquet` (ver §3) |
| `AUDITAR_ACTIVOS` | Comprueba que macros, manifiesto y activos coinciden con los estudios adoptados. No escribe |
| `VERIFICAR_PROYECTO` | Rutas, UTF-8, referencias cruzadas y activos huérfanos |
| `COMPILAR_MEMORIA` | `TFM.tex` → `TFM.pdf` |
| `COMPILAR_DEFENSA` | `TFM_ppt.tex` → `TFM_ppt.pdf` |
| `DEFENSA_CON_NOTAS` | Además, `TFM_ppt_notes.tex` → `TFM_ppt_notes.pdf`, con el guion del ponente para la segunda pantalla |
| `COPIAR_PDF_AL_REPO` | Deja los PDF en `latex/`; con `False` se quedan en `latex/build/` |
| `LIMPIAR_AUXILIARES` | Borra `.aux`, `.log`, `.toc`… al terminar |
| `SALIDA_DETALLADA` | Muestra la salida completa de cada paso en vez del resumen |

Para una ejecución suelta sin editar el fichero:

```powershell
python latex/build.py --todo            # regenera, verifica y compila las dos
python latex/build.py --solo-memoria    # solo TFM.tex
python latex/build.py --solo-defensa    # solo TFM_ppt.tex
python latex/build.py --activos         # regenera y verifica, sin compilar
python latex/build.py --notas           # añade la defensa con notas del ponente
python latex/build.py --limpiar         # borra auxiliares y termina
python latex/build.py --detallada       # con la salida completa de cada paso
```

Los pasos van en ese orden y **se detiene en el primero que falle**: encadenarlos con un paso roto
produce un PDF que parece correcto y no lo es. Los identificadores de estudio se leen de
`asset_manifest.json`, así que el script no es una copia más de esa información.

El resto de este documento explica cada paso por separado, por si hace falta ejecutarlo a mano o
diagnosticar un fallo.

---

## 1. Qué hay que tener instalado

| Para | Herramienta | ¿Hace falta? |
|---|---|---|
| Compilar el PDF | Una distribución LaTeX con XeLaTeX | Sí |
| Repetir pasadas solo | `latexmk` **y** un `perl` que lo ejecute | No: `build.py` lo suple |
| Regenerar figuras y tablas | Python con `pandas`, `pyarrow` y `matplotlib` | Solo si cambian los estudios |
| Verificar el proyecto | Python (sin dependencias extra) | Recomendado antes de subir nada |

### En esta máquina ya está

Hay **MiKTeX 25.12** instalado en el perfil del usuario, no en `Archivos de programa`:

```
%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\
```

Incluye XeTeX 4.16, `latexmk`, las fuentes Latin Modern OpenType que pide el preámbulo y los
paquetes `siunitx`, `pdflscape`, `enumitem` y `makecell`. Para comprobarlo:

```powershell
xelatex --version
kpsewhich lmroman10-regular.otf
```

Si el segundo comando no devuelve una ruta, faltan las fuentes Latin Modern y hay que instalarlas
desde la consola de MiKTeX (`MiKTeX Console` → *Packages* → buscar `lm`).

### Instalar desde cero en otra máquina

**Opción A · MiKTeX (Windows), la que usa este proyecto.**

1. Descargar el instalador de <https://miktex.org/download> y elegir la instalación **solo para mí**
   (no requiere permisos de administrador).
2. En la primera pantalla de configuración, dejar `Install missing packages on-the-fly` en **Yes**.
   Con esa opción MiKTeX descarga por sí solo cualquier paquete que falte durante la primera
   compilación, y por eso la primera pasada tarda bastante más que las siguientes.
3. Abrir `MiKTeX Console` → *Updates* → *Check for updates* antes de compilar por primera vez.

**Opción B · TeX Live (Linux, macOS o Windows).** Instalar el esquema completo
(`scheme-full`) o, como mínimo, `scheme-medium` más `collection-fontsrecommended` y
`collection-langspanish`. TeX Live no instala paquetes bajo demanda: si falta uno, la compilación
falla y hay que añadirlo con `tlmgr install <paquete>`.

**Opción C · Overleaf, sin instalar nada.** Es el destino para el que está preparado el proyecto.
Subir la carpeta `latex/` completa y configurar:

- *Menu* → *Compiler*: **XeLaTeX**
- *Menu* → *Main document*: `TFM.tex` para la memoria, `TFM_ppt.tex` para la defensa

Las rutas de los recursos son relativas (`chapters/`, `figures/`, `tables/`), de modo que funcionan
igual en local y en Overleaf sin tocar nada.

---

## 2. Compilar

Lo normal es no hacerlo a mano: `python latex/build.py` (§0) se encarga. Lo que sigue es el
equivalente manual, por si hace falta diagnosticar.

El documento necesita **al menos dos pasadas**: la primera escribe el índice, la lista de figuras,
la de tablas y las etiquetas de referencia cruzada en `TFM.aux`, y la segunda las coloca. Con una
sola pasada el índice sale vacío y las referencias aparecen como `??`.

```powershell
cd latex
xelatex -interaction=nonstopmode TFM.tex
xelatex -interaction=nonstopmode TFM.tex
```

### Cuidado con `latexmk` en PowerShell

`latexmk` repite las pasadas por su cuenta y es la opción cómoda, pero **es un script de Perl, no un
binario**. MiKTeX lo instala igualmente, así que `Get-Command latexmk` lo encuentra y aun así falla
al ejecutarse:

```
MiKTeX could not find the script engine 'perl' which is required to execute 'latexmk'
```

Git Bash trae su propio `perl` y PowerShell no, de modo que **el mismo comando funciona en una
consola y falla en la otra**. Por eso `build.py` comprueba las dos cosas y, si no hay Perl, llama a
`xelatex` directamente repitiendo mientras el registro pida otra pasada. No hay que instalar nada.

Si aun así se prefiere `latexmk`, basta con instalar Strawberry Perl y reabrir la consola.

### Notas del ponente en la defensa

Cada diapositiva lleva su guion en un `\note{}`. Para proyectar con las notas en una segunda
pantalla:

```powershell
python latex/build.py --solo-defensa --notas
```

Sale `latex/TFM_ppt_notes.pdf`, con cada página al doble de ancho: la diapositiva a la izquierda y
el guion a la derecha. El script no edita la defensa: **regenera `TFM_ppt_notes.tex` desde
`TFM_ppt.tex`** con la opción de beamer descomentada, de modo que `TFM_ppt.tex` se queda intacto y
la versión con notas no puede desincronizarse. A mano, el equivalente es descomentar esta línea del
preámbulo:

```latex
\setbeameroption{show notes on second screen=right}
```

---

## 3. Regenerar figuras y tablas

Ningún PNG ni cuerpo de tabla se dibuja a mano: todos salen de los artefactos de los estudios
adoptados. El comando completo, desde la **raíz del repositorio** (no desde `latex/`):

```powershell
python latex/scripts/export_study_assets.py `
  --study-id study-20260817-094411-568bd37e `
  --chain-study-id study-20260816-182345-3cc1a5fb `
  --chain-study-id study-20260817-021135-b5926b62 `
  --chain-study-id study-20260817-094411-568bd37e `
  --portfolio-study-id study-20260817-212856-f86ca822
```

Los `--chain-study-id` van en orden de ejecución y el último debe coincidir con `--study-id`. El
script rechaza cualquier otro identificador: la cadena adoptada está fijada en el código
precisamente para que no se cuele evidencia de un estudio distinto.

Después conviene pasar las dos comprobaciones:

```powershell
# No escribe nada: recalcula las macros y comprueba que macros, manifiesto y activos coinciden
python latex/scripts/export_study_assets.py [mismos identificadores] --audit

# Rutas relativas, recursos existentes, UTF-8, referencias con destino y sin activos huérfanos
python latex/scripts/verify_latex_assets.py
```

> **Aviso.** Los `.parquet` de la evidencia no están versionados, por tamaño. En un clon del
> repositorio el exportador fallará al leerlos: la regeneración completa solo funciona en la
> instalación donde corrieron los estudios. Las figuras y tablas ya generadas sí viajan con el
> repositorio, así que el PDF se puede compilar en cualquier sitio.

---

## 4. Estructura de la carpeta

```
latex/
  build.py              hace todo: compila, regenera y verifica. Empieza por aquí
  COMO_COMPILAR.md      este documento
  guion_defensa.md      tiempos y mapa de preguntas de la defensa
  TFM.tex               memoria
  TFM_ppt.tex           defensa (Beamer 16:9)
  TFM_ppt_notes.tex     defensa con las notas del ponente. Lo regenera build.py --notas
  chapters/             un .tex por capítulo y anexo. Escritos a mano
  figures/              solo PNG. Generados
  tables/               solo .tex: cuerpos de tabla y study_macros.tex. Generados
  scripts/              exportador y verificador
  build/                carpeta de trabajo: PDF y registros. No se versiona
  asset_manifest.json   qué artefacto alimenta cada figura, tabla y macro
```

El prefijo del nombre dice en qué capítulo se usa: `f06_*` en el capítulo 6, `t07_*` en el 7,
`tB_*` en el anexo B. Los dos documentos maestros viven en `latex/` y comparten `figures/`, de modo
que ninguna imagen está duplicada.

---

## 5. Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| El PDF muestra `??` donde debería haber un número de figura o de capítulo | Falta una pasada. Usar `python latex/build.py`, que repite las que hagan falta, en lugar de invocar `xelatex` una sola vez. |
| `MiKTeX could not find the script engine 'perl'` | `latexmk` es un script de Perl y PowerShell no trae intérprete. Usar `python latex/build.py`, que no lo necesita (ver §2). |
| El script termina bien pero el PDF que abro es el de antes | `COPIAR_PDF_AL_REPO = False`: los PDF quedan en `latex/build/` y los de `latex/` se quedan viejos. Ponerlo a `True` o abrir los de `build/`. |
| `Font ... not found` o el PDF sale en una fuente distinta | Faltan las fuentes Latin Modern OpenType. Comprobar con `kpsewhich lmroman10-regular.otf` e instalarlas desde MiKTeX Console. |
| La compilación se detiene pidiendo instalar un paquete | Es MiKTeX con la instalación bajo demanda en modo pregunta. Aceptar, o cambiarlo a automático en MiKTeX Console → *Settings* → *Package installation*. |
| `Package pgfkeys Error: The key '/tikz/...' requires a value` | Un estilo de TikZ usa un nombre reservado (`cap`, `draw`, `fill`…). Renombrar el estilo. |
| `Overfull \hbox` en el registro | Aviso, no error: algo sobresale del margen. Por debajo de unos 5 pt es invisible; por encima, suele ser una tabla ancha y se arregla con `\resizebox{\textwidth}{!}{...}`. |
| El exportador falla con `FileNotFoundError` en un `.parquet` | Es un clon sin la evidencia completa. Ver el aviso de la sección 3. |
| `VALIDACIÓN FALLIDA: Figura huérfana` | Se generó un PNG que ningún capítulo incluye. O se cita, o se borra. |

Para ver los errores reales de una compilación, buscar las líneas que empiezan por `!` en el
registro:

```powershell
Select-String -Path build\main.log -Pattern '^!' -Context 0,4
```

---

## 6. Limpiar y desinstalar

### Borrar los ficheros auxiliares

```powershell
python latex/build.py --limpiar   # borra los auxiliares de latex/build/ y conserva los PDF
```

O, a mano, borrando la carpeta `latex/build/` entera.

### Desinstalar MiKTeX por completo

1. **Aplicaciones instaladas** → buscar *MiKTeX* → *Desinstalar*.
2. La desinstalación deja atrás la configuración y la caché de paquetes. Borrar a mano, si no se va
   a volver a usar:

   ```powershell
   Remove-Item -Recurse -Force "$env:LOCALAPPDATA\MiKTeX"
   Remove-Item -Recurse -Force "$env:APPDATA\MiKTeX"
   Remove-Item -Recurse -Force "$env:LOCALAPPDATA\Programs\MiKTeX"
   ```

   La tercera solo hace falta si la desinstalación no vació la carpeta de programa.
3. Comprobar que la variable `PATH` del usuario ya no apunta a
   `...\Programs\MiKTeX\miktex\bin\x64`. Se revisa en *Editar las variables de entorno de tu
   cuenta*.

Con TeX Live, el equivalente es ejecutar `tlmgr uninstall --all` o borrar el directorio de la
instalación (`C:\texlive\<año>` o `/usr/local/texlive/<año>`) y limpiar el `PATH`.

Nada de esto afecta al repositorio: los PDF y los activos generados siguen ahí, y basta con volver a
instalar una distribución LaTeX para poder recompilar.

---

## 7. Convenciones del manuscrito

Estas reglas gobiernan cómo se escribe, no cómo se compila. Van aquí porque son la única parte del
antiguo `plan_tfm.md` que no estaba ya en otro sitio.

### Decisiones de formato

| Tema | Decisión |
|---|---|
| Plantilla | Estructura académica libre; el máster no impone una. |
| Idioma | Español. |
| Motor | **XeLaTeX**, sin `biber`. UTF-8 nativo (sin `inputenc`), fuentes con `fontspec`. |
| Granularidad | Capitulado clásico: 9 capítulos, bibliografía y tres anexos. |
| Referencias | **Autor-año escritas a mano** en `chapters/10_bibliografia.tex`. No se usa `biblatex` ni `biber`: esa cadena impedía compilar. El proyecto no contiene ningún `.bib`. |
| Figuras y tablas | Generadas desde los artefactos de los estudios, nunca dibujadas a mano. |
| Defensa | `TFM_ppt.tex`, Beamer 16:9. Vive junto a `TFM.tex` para reutilizar `figures/` sin duplicar ningún PNG; en Overleaf sólo se cambia *Main document*. |

Queda pendiente de decidir, sin que bloquee nada: la portada oficial de la universidad (hay una
provisional en `TFM.tex`).

### Convenciones de escritura

- **Toda cifra citada debe poder trazarse a un artefacto real del repositorio** (parquet, json,
  figura). Igual que el código: nada sin fuente verificable. Cuando la cifra existe como macro en
  `tables/study_macros.tex`, se usa la macro y no el número: es la única forma de que se refresque
  sola al cambiar de estudio.
- Notación consistente entre capítulos: snapshot \(\tsnap\), publicación \(\tfiled\), horizonte
  \(\hlabel\), coeficiente de información por rango \(\rankic\). Se fija en el preámbulo de
  `TFM.tex` y se reutiliza sin redefinir.
- Términos en inglés sin traducción asentada (*lookahead bias*, *walk-forward*, *rank-IC*) en
  cursiva la primera vez y así el resto del documento.
- **Cada resultado positivo va acompañado de su matiz** en el mismo párrafo o en el inmediatamente
  siguiente. El valor del trabajo está en la honestidad de la medición, no en el tamaño del alfa: un
  tribunal premia el matiz declarado y penaliza el matiz encontrado.
- Distinguir siempre y explícitamente el papel de cada cifra: **selección**, **confirmación fuera de
  muestra** o **diagnóstico**. Nunca mezclar los tres en una misma tabla sin etiquetarlos.
- Los decimales en español con coma; los identificadores, rutas y nombres de variables en
  `\texttt{}` y sin traducir.
