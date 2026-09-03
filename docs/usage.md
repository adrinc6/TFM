# Guía de uso

Cómo instalar, configurar y ejecutar el proyecto de cero. Para entender qué hace cada pieza por
dentro, ver [architecture.md](architecture.md); para saber qué produce, [results.md](results.md).

---

## 1. Requisitos

- **Python 3.12** (desarrollado y probado en esa versión).
- **XeLaTeX** únicamente si se va a compilar el manuscrito. No hace falta para ejecutar el sistema.
- **Node** únicamente para `node --check app/js/app.js`. No hay `npm install`: el dashboard no tiene
  cadena de construcción.

```powershell
python -m pip install -r requirements.txt
```

Las dependencias son deliberadamente pocas: pandas, numpy, pyarrow, lightgbm, scikit-learn,
matplotlib, requests, psutil, pytest y ruff. No hay framework web (se usa la biblioteca estándar),
ni librería de dotenv, ni framework de CLI.

---

## 2. Configuración

Toda la configuración vive en `environment.py` y en un fichero `.env` en la raíz. No hay YAML ni
TOML. Copiar `.env.example` a `.env` y rellenarlo:

```powershell
Copy-Item .env.example .env
```

| Variable | Obligatoria | Por defecto | Para qué |
|---|---|---|---|
| `FINNHUB_API_KEY` | **Sí**, para ingerir datos | vacío | Perfiles y métricas fundamentales de Finnhub. Hay plan gratuito. |
| `EDGAR_USER_AGENT` | **Sí**, cámbialo | correo del autor | La SEC exige un User-Agent que te identifique. **Pon tu propio nombre y correo.** |
| `RUN_SCOPE` | No | `full` | `dev` limita el universo a cinco tickers y aísla la salida. |
| `PORTFOLIO_GRID_WORKERS` | No | `6` | Procesos paralelos de la rejilla del Portfolio Study. |

Dos advertencias que ahorran tiempo:

- **`EDGAR_USER_AGENT` trae por defecto el correo del autor de este TFM.** Es un valor heredado del
  desarrollo, no una plantilla: si lo dejas, tus peticiones a la SEC se identifican como suyas.
  Cámbialo antes de la primera ingesta.
- **`PORTFOLIO_GRID_WORKERS` lo limita la RAM, no la CPU** (unos 0,2–0,4 GB por worker). Con poca
  memoria libre, bajarlo. `1` da ejecución secuencial exacta.

Sin `FINNHUB_API_KEY` la ingesta no falla ruidosamente: `download_raw_data` deja el cliente a `None`
y degrada a un panel sin fundamentales, que es un panel inservible. Si el panel sale vacío, ese es el
primer sitio donde mirar.

---

## 3. Primer arranque

```powershell
# 1. Dependencias
python -m pip install -r requirements.txt

# 2. Credenciales
Copy-Item .env.example .env      # y editarlo

# 3. Descargar y consolidar los datos crudos  (horas: ~1.200 tickers)
python main.py ingest

# 4. Levantar API y dashboard
python main.py serve
```

Abrir **`http://127.0.0.1:8765/`**.

Detener con `Ctrl+C`: el servidor termina los workers que haya creado y no deben quedar procesos
Python huérfanos.

### Antes de esperar horas: los dos caminos cortos

- **`http://127.0.0.1:8765/dev`** — modo visual con fixtures. No entrena nada y no necesita datos.
  Sirve para ver la interfaz y comprobar que el servidor funciona.
- **`RUN_SCOPE=dev`** — camino de humo **real**: ingiere y entrena de verdad, pero solo con
  `AAPL, MSFT, NVDA, JPM, XOM`, escribiendo en `data/raw/dev` y `data/prepared/standalone-dev` para
  no contaminar la ejecución completa. Es la forma de verificar el flujo entero en minutos.

  ```powershell
  $env:RUN_SCOPE = "dev"; python main.py ingest
  ```

  Un Study en modo dev demuestra que **el software funciona, no que exista señal económica**. Sus
  cifras no son evidencia de nada.

`main.py` solo acepta `ingest` y `serve`; sin argumento hace `serve`. Los estudios **no se lanzan
desde la línea de comandos**, sino desde el dashboard.

---

## 4. Lanzar un Model Study

En la pestaña **Inicio** se marcan valores por variable, directamente sobre el catálogo cerrado:

- **un solo valor** → esa variable queda fija;
- **dos o más valores** de una variable predictiva → se optimiza;
- **dos o más valores** de una variable de cartera → comparación diagnóstica (no elige nada).

La configuración recomendada llega ya marcada. Cada vez que se añade o quita un valor, el presupuesto
se recalcula: número de runs, minutos estimados y disco estimado. **No hay límite artificial de
runs** — el preflight informa del coste, no lo prohíbe.

### Por dónde empezar

Deja el baseline `recommended` tal cual y optimiza **una sola variable** la primera vez. El baseline
no es un valor arbitrario: se eligió por lógica de dominio con dos criterios —preferir lo simple y
regularizado, y preferir lo estrictamente point-in-time— precisamente para que sea un punto de
partida defendible sin haber mirado ningún resultado. La tabla completa con la razón de cada valor
está en [architecture.md](architecture.md#el-baseline-recommended).

Un Study completo tarda horas. Antes de lanzar uno largo, conviene haber hecho el recorrido con
`RUN_SCOPE=dev`.

### Mientras corre

El estudio se ejecuta en un **worker hijo**. La pestaña **Resultados** muestra la tabla de Studies;
al abrir uno aparece una cabecera estable con sus métricas y los botones Runs, Consola, Robustez y
Perfiles, más la acción Pausar o Reanudar que corresponda.

La Consola es la única vista con scroll propio (viewport de veinte líneas). La zona inferior no se
refresca sola, para no desplazar el scroll mientras se lee.

Al abrir un run se llega a una segunda página con vistas de resumen, rendimiento, aprendizaje,
cartera y acciones. **Cartera** permite elegir snapshot y ver posiciones y órdenes de esa fecha;
**Acciones** conserva la fecha y permite consultar la situación en cartera, los agentes, las
puntuaciones por parámetro y los valores PIT.

Un candidato descartado solo muestra su evidencia compacta. El ganador y el baseline
(`predictive:baseline`) son los únicos con evidencia completa, salvo que se active el conmutador
«Ruta científica completa» al lanzar, que la retiene para todos los runs a cambio de disco y de
recalcular en vez de reutilizar la caché.

### Cancelar, pausar y reanudar

- **Pausar / Reanudar / Cancelar** desde la cabecera del Study.
- **Reanudar no repite runs terminados** ni duplica el ledger: solo reinicia los incompletos.
- Un **artefacto parcial nunca es caché válida**; la caché solo se publica completa.
- Si el servidor se cae, al arrancar de nuevo marca como `interrupted` cualquier Study cuyo proceso
  ya no exista. No quedan estudios «corriendo» de mentira.

---

## 5. Lanzar un Portfolio Study

Requiere un Model Study ya terminado, que se indica como `source_study_id`. **No reentrena nada**:
reutiliza los scores congelados del ganador y solo rehace el backtest, así que cada combinación
cuesta segundos en vez de minutos.

Recorre una **rejilla cartesiana completa** de seis variables de cartera y elige por **Information
Ratio**. Su preflight declara el número de combinaciones y el tiempo estimado antes de lanzar.

Dos cosas que conviene saber antes de mirar sus cifras:

- Durante la rejilla, el backtest **se corta en 2024**: 2025–2026 no se calcula para ninguna
  combinación. Solo la ganadora se reevalúa después sobre la serie completa.
- La cartera adoptada es la mejor de la rejilla, así que sus cifras **dentro** de la ventana de
  selección son una cota superior optimista, no una estimación insesgada.

Al terminar calcula solos tres diagnósticos: sensibilidad a costes, capacidad y narrativa de cartera.

---

## 6. Compilar el manuscrito

Un único comando, con interruptores al principio del fichero:

```powershell
python latex/build.py
```

| Interruptor | Por defecto | Qué hace |
|---|---|---|
| `REGENERAR_ACTIVOS` | `False` | Regenera figuras, tablas y macros numéricas desde los artefactos. |
| `AUDITAR_ACTIVOS` | `False` | Recalcula las macros en memoria y comprueba que cuadran. No escribe nada. |
| `VERIFICAR_PROYECTO` | `True` | Rutas relativas, recursos existentes, UTF-8 sin mojibake, referencias cruzadas. |
| `COMPILAR_MEMORIA` | `True` | `TFM.tex` → `TFM.pdf` |
| `COMPILAR_DEFENSA` | `True` | `TFM_ppt.tex` → `TFM_ppt.pdf` |
| `DEFENSA_CON_NOTAS` | `True` | Añade la presentación con el guion intercalado, para la segunda pantalla. |
| `COPIAR_PDF_AL_REPO` | `True` | Copia los PDF a `latex/`, que es donde se versionan. |
| `LIMPIAR_AUXILIARES` | `True` | Borra `.aux`, `.log`, `.toc`, `.fls` al terminar. |
| `SALIDA_DETALLADA` | `False` | Muestra la salida completa del compilador. |

Sin editar el fichero, para una sola ejecución:

```powershell
python latex/build.py --todo            # regenerar activos, verificar y compilar todo
python latex/build.py --activos         # regenerar y verificar, sin compilar
python latex/build.py --solo-memoria    # solo TFM.tex
python latex/build.py --solo-defensa    # solo TFM_ppt.tex
python latex/build.py --notas           # añadir la presentación con notas
python latex/build.py --limpiar         # borrar auxiliares y salir
python latex/build.py --detallada       # salida completa de cada paso
```

Los pasos se ejecutan en orden y **se detienen en el primer fallo**: encadenar sobre un paso roto
produce un PDF que parece correcto y no lo es.

Tres avisos:

- **`REGENERAR_ACTIVOS=True` no funciona en un clon.** Necesita los `.parquet` de evidencia, que
  están en `.gitignore`. Solo funciona en la máquina donde se ejecutaron los estudios. Para leer el
  trabajo no hace falta: `latex/TFM.pdf` está versionado.
- **El motor es XeLaTeX, no pdfLaTeX.** El preámbulo usa `fontspec` y UTF-8 nativo; `build.py` lo
  exige explícitamente.
- Los `study_id` se leen de `latex/asset_manifest.json`, nunca se escriben a mano en el script.

El detalle de cada paso, las convenciones de escritura y las decisiones de formato están en
[../latex/COMO_COMPILAR.md](../latex/COMO_COMPILAR.md).

---

## 7. Verificación

```powershell
python -m pytest -q
python -m ruff check .
node --check app/js/app.js
```

Los tests de `tests/` son **tests de contrato**: comprueban las identidades que el sistema promete
—que la suma de contribuciones es exactamente el retorno bruto, que el catálogo es coherente, que la
cartera no rompe sus propios suelos— y **no necesitan datos reales ni credenciales**. Se ejecutan en
segundos.

---

## 8. Problemas frecuentes

**El panel sale vacío o sin fundamentales.**
Falta `FINNHUB_API_KEY`. La ingesta no aborta: degrada silenciosamente a un panel sin fundamentales.

**La SEC devuelve 403 o corta las peticiones.**
`EDGAR_USER_AGENT` sigue con el valor por defecto o no identifica a nadie. La SEC exige un
User-Agent con nombre y correo reales, y limita a ~10 peticiones por segundo.

**`latexmk` funciona en Git Bash y falla en PowerShell.**
`latexmk` es un script de Perl. MiKTeX lo instala siempre, pero Perl solo viene con Git Bash.
`build.py` comprueba **ambos** y, si falta Perl, cae a pasadas manuales decididas leyendo el log. No
es un error, solo es más lento.

**Quedan procesos Python después de cerrar.**
No debería: `stop_all()` está registrado con `atexit`. Si ocurre, al volver a arrancar el servidor
marca esos estudios como `interrupted` y se pueden reanudar.

**La ingesta falla al arrancar.**
Es intencionado: se comprueba que las fuentes respondan antes de entrar en el bucle de ~1.200
tickers, para no descubrir a las tres horas que faltaba una credencial.

**Un Study aparece como `interrupted`.**
Su proceso desapareció (cierre del servidor, reinicio, falta de memoria). Reanudarlo conserva los
runs ya terminados y solo repite los incompletos.
