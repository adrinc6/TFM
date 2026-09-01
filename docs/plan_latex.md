# Plan del manuscrito LaTeX

> **Este documento es un plan vivo, no un encargo de un solo uso.** Cumple dos funciones:
>
> 1. Es el **plan completo** para dejar el manuscrito al día con la situación actual del proyecto.
> 2. Es el **destino de toda deuda futura con el manuscrito**: cualquier cambio de código que afecte
>    al LaTeX añade aquí qué habrá que escribir, en qué capítulo y con qué artefacto detrás
>    (sección «Deuda abierta», al final).
>
> **Aquí no se copian cifras de resultados.** Viven en `results/studies/<study_id>/` y se leen de
> ahí. Sí se recogen las cifras del panel, porque son mediciones cerradas sobre `data/raw/` que no
> dependen de ningún estudio.
>
> Regla de oro: **ninguna cifra entra en un `.tex` sin poder señalar el artefacto exacto** que la
> respalda (`study_id` y ruta).

El manuscrito (`latex/TFM.tex`, `latex/TFM_ppt.tex`, `latex/assets/*.tex`) está **congelado**:
los cambios de código no lo editan. `latex/scripts/*.py`, `latex/build.py` y
`latex/COMO_COMPILAR.md` sí son editables, pero
el exportador no se ejecuta como parte de un cambio corriente.

**Última actualización del manuscrito: 2026-09-01**, con la revisión editorial de las 18
anotaciones: corrección de la explicación del retardo de observación, eliminación del Abstract,
glosario ampliado, reorganización de los capítulos 6 y 7, caso Apple, Anexo C nuevo, páginas
apaisadas eliminadas y recorte a 89 páginas. Desde entonces vuelve a estar congelado y la deuda se
acumula en §6.

**Restricción de extensión, ahora dura.** Cuerpo (capítulos 1 a 9) ≤ **60 páginas**, anexos ≤ **15**.
Si entra contenido nuevo, sale otro. El reparto por capítulo sólo se ve compilando: no hay forma de
comprobarlo desde los `.tex`.

Dos avisos de alcance que este documento no traía y que costaron tiempo:

- **`latex/TFM_ppt.tex` cuenta como manuscrito.** Tiene sus propias cifras, y varias están
  escritas **con letra** en el guion hablado («mil cuatrocientas cuarenta», «diez puntos básicos de
  comisión»), donde ningún `grep` numérico las encuentra.
- **Los identificadores de estudio viven en tres sitios** que se desincronizan solos: las macros de
  `TFM.tex`, el bloque literal de comando de `a_reproducibilidad.tex` —que el exportador **no**
  toca— y `asset_manifest.json`, que sí se regenera. El `dataset_hash` es un literal más, sin macro.

Este plan cubre **qué escribir y dónde**. El **cómo** del formato —notación, estructura de carpetas,
convenciones de redacción, criterio de poda de activos— está en `latex/COMO_COMPILAR.md` y no se
duplica
aquí.

---

## 1. Estado del manuscrito

Desde el 2026-08-24 `latex/` está separado por tipo, y el prefijo del nombre indica **el capítulo que
cita el activo**, no el capítulo en el que se generó:

```
latex/chapters/   15 ficheros .tex, escritos a mano
latex/figures/    22 PNG, todos generados
latex/tables/     22 cuerpos de tabla + study_macros.tex, generados
```

Antes convivían los tres tipos sueltos en `assets/` y con prefijos desalineados —`f07_*` citado en
el capítulo 6, `t08_*` en el 7—: fue la causa mecánica de varias cifras obsoletas, porque al corregir
un capítulo era fácil no tocar su activo. `verify_latex_assets.py` comprueba ahora también que cada
carpeta contenga un solo tipo de fichero.

La distinción que más errores evita, y que no está recogida en ningún otro sitio:

**20 tablas se generan solas** al ejecutar `export_study_assets.py` y se refrescan sin intervención.
**2 se escriben a mano** y no las actualiza nadie por ti. Son las que hay que releer cada vez que
cambie un estudio.

| Tabla manual | De dónde sale su contenido |
|---|---|
| `tD_defectos_validez.tex` | Transcrito de `docs/bitacora.md` |
| `t08_limitaciones.tex` | Compuesto a mano desde los artefactos citados en el capítulo 8 |
| Caso Apple (`tab:caso-apple`, inline en el cap. 7) | Escrita a mano en el capítulo desde `agent_scores.parquet` y `meta_weights.parquet` del Model Study 3, fila AAPL / 2019-10-30 |

`latex/asset_manifest.json` las lista todas juntas porque se construye con un glob de `t*.tex`, no
rastreando autoría: **no sirve** para saber cuáles son generadas. Y como el glob describe el disco en
el momento del export, una tabla borrada a mano sobrevive en el manifiesto hasta la siguiente
ejecución.

`latex/scripts/verify_latex_assets.py` exige que toda tabla esté referenciada con `\input` y toda
figura citada. Un activo huérfano hace fallar la comprobación.

**Lo que ningún script comprueba, y por eso hay que hacerlo a mano: la prosa.** Los dos verificadores
pasaban limpios el 2026-08-24 mientras el texto contenía veinticinco cifras incorrectas. Las macros
de `study_macros.tex` sí estaban bien —se generan y se auditan—; el problema estaba en los números
escritos a mano en los párrafos, que ninguna herramienta contrasta contra su artefacto.

---

## 2. Qué dice el panel hoy

Mediciones cerradas, leídas de `data/raw/universe_coverage.json` y `data/raw/ticker_diagnostics.csv`.
No dependen de ningún estudio, así que este bloque ya no va a cambiar.

| Hecho | Dato |
|---|---|
| Universo histórico | 1206 tickers |
| En panel | **643** |
| Miembros actuales del índice en panel | **502 de 503** |
| Excluidos con marcador `Q` de quiebra | **5,5 %** (31 de 563) |
| Excluidos con fundamentales descargados | **563 de 563** |

Reparto por motivo de exclusión: `symbol_withdrawn` 207, `missing_price` 168, `recycled_ticker` 153,
`no_metric_period_match` 31, `missing_reports` 2, `missing_cik` 2.

**Exclusión según antigüedad de la salida del índice** — es la tabla que explica el mecanismo:

| Antigüedad de la salida | Tickers | Fuera del panel |
|---|---|---|
| Aún en el índice | 516 | 1,2 % |
| < 2 años | 41 | 39 % |
| 2-5 años | 62 | 61 % |
| 5-10 años | 122 | 71 % |
| 10-20 años | 216 | 84 % |
| > 20 años | 249 | 94,4 % |

**Dirección y magnitud del sesgo** — supervivencia hasta hoy de los miembros incluidos frente a los
del índice real:

| Año | Cobertura | Siguen hoy (incluidos) | Siguen hoy (índice real) | Exceso |
|---|---|---|---|---|
| 1998 | 39,0 % | 77,7 % | 35,5 % | +42,2 pp |
| 2003 | 49,0 % | 79,5 % | 44,1 % | +35,4 pp |
| 2008 | 59,6 % | 79,3 % | 50,3 % | +29,0 pp |
| 2013 | 69,4 % | 83,9 % | 61,0 % | +22,9 pp |
| 2018 | 82,2 % | 87,3 % | 71,9 % | +15,3 pp |
| 2026 | 99,2 % | 100 % | 100 % | 0 pp |

Como el entrenamiento es rolling, el exceso decae de **26,2 pp** (evaluando 2015) a **8,0 pp**
(evaluando 2026).

La causa está verificada contra la API: el proveedor de precios **retira los símbolos que dejan de
cotizar**. Su buscador no encuentra `BK`, `EA`, `MMC`, `GPS`, `WBA`, `HOLX` ni `CMA` como equity
mientras `AAPL` sí resuelve; no es el rango de fechas, ni autenticación, ni la longitud del símbolo.

---

## 3. Dónde vive cada cifra que caduca

Inventario por concepto, para poder ir a por ellas sin releer el manuscrito entero. Todas son cifras
**escritas a mano en la prosa**: las tablas generadas se refrescan solas.

| Concepto | Dónde aparece |
|---|---|
| Nº de carteras de la rejilla | `00_resumen` (×4), `07_resultados_economicos` (×3), `09_conclusiones` (×3), `t09_limitaciones` (×2), `a_reproducibilidad` (×2), `b_catalogo_protocolo`, `c_evidencia_complementaria` |
| Nº de cohortes de selección | `03_datos_y_universo`, `06_resultados_predictivos` (×5), `09_conclusiones`, `a_reproducibilidad`, `t09_limitaciones`, `t01_afirmaciones` |
| Rank-IC del meta | `00_resumen` (×2), `03`, `05`, `06` (×7), `07`, `09`, `t01` (×2), `t09` |
| Deflated Sharpe y su umbral | `00_resumen` (×2), `05`, `06` (×4), `09` (×3), `t09`, `b_catalogo_protocolo` |
| IR y exceso de la cartera | `00_resumen` (×4), `07` (×4), `09`, `t01` |
| Era reservada (cohortes, años, IR) | `00_resumen` (×3), `06` (×2), `07` (×3), `09` (×2), `t01`, `t09` |
| Nº de tickers y cobertura | `03` (×2), `t09` |
| `study_id` | Macros en `TFM.tex` **y** bloque literal en `a_reproducibilidad.tex:46-50` |
| `dataset_hash` | `03_datos_y_universo` |

El bloque de `a_reproducibilidad.tex:46-50` reproduce la línea de comandos con los identificadores
escritos literalmente: **no lo actualiza el exportador** y se edita a mano, además de las macros.

---

## 4. Comandos y verificación

Invocación **completa**: sin la cadena y el Portfolio Study no se generan las tablas de cadena,
cartera y perfiles.

```powershell
python latex/scripts/export_study_assets.py `
  --study-id <MODEL_STUDY_3> `
  --chain-study-id <MODEL_STUDY_1> `
  --chain-study-id <MODEL_STUDY_2> `
  --chain-study-id <MODEL_STUDY_3> `
  --portfolio-study-id <PORTFOLIO_STUDY>
python latex/scripts/verify_latex_assets.py
```

La guía completa —instalar XeLaTeX, compilar, regenerar, resolver errores y desinstalar— está en
`latex/COMO_COMPILAR.md`.

`--chain-study-id` es repetible y **el último debe coincidir con `--study-id`**.

Comprobaciones antes de dar el manuscrito por bueno:

- `verify_latex_assets.py` sin huérfanos ni faltantes. Los activos sobrantes se borran a mano.
- Ninguna cifra sin `study_id` y ruta de artefacto detrás.
- Las cinco afirmaciones de `t01_afirmaciones.tex`, revalidadas una a una: es una tabla manual y cada
  una lleva su matiz obligatorio.
- Ninguna cifra huérfana de estudios anteriores: buscar los identificadores viejos en `latex/`.

### Resultados

**Cadena adoptada** (2026-08-17). Las cifras viven en los artefactos; aquí solo los identificadores y
lo que se concluyó.

| Rol | `study_id` |
|---|---|
| Model Study 1 | `study-20260816-182345-3cc1a5fb` |
| Model Study 2 | `study-20260817-021135-b5926b62` |
| Model Study 3 — referencia predictiva | `study-20260817-094411-568bd37e` |
| Portfolio Study — referencia económica | `study-20260817-212856-f86ca822` |

Las siete lecturas del Bloque B, resueltas:

1. **Identificadores.** Macros de `TFM.tex`, bloque literal de `a_reproducibilidad.tex` y
   `dataset_hash` de `03_datos_y_universo.tex`, los tres actualizados.
2. **Ganador y cadena.** La cadena converge (9 → 2 → 2 variables) y mejora por milésimas: esa
   pequeñez es el argumento para detenerla, y así se enuncia.
3. **Rank-IC por era.** Ni estable ni concentrado en 2015-2018: **sube** hacia la era menos sesgada
   mientras el Information Ratio cae y se vuelve negativo. Descarta que el sesgo de cobertura
   fabrique la capacidad predictiva; no permite extender esa conclusión al resultado económico.
   Tiene subsección propia en el capítulo 6 y fila propia en limitaciones.
4. **Versión de catálogo.** Única en toda la cadena → la limitación **desaparece**, y por decisión
   del usuario no se menciona el versionado en ninguna parte. Tabla `t08_versiones_catalogo.tex`
   eliminada.
5. **Deflated Sharpe.** Cambia de base, no solo de valor: penaliza por 46 configuraciones
   predictivas, no por la rejilla de cartera. El argumento se rederivó, distinguiendo las dos
   búsquedas en vez de sumarlas.
6. **Dominio de un solo agente.** Sigue: `risk` ordena mejor que el meta y concentra el grueso del
   peso. Se mantiene como limitación y como matiz de la primera afirmación.
7. **Diagnósticos económicos.** `cost_sensitivity` y `capacity` entran como secciones nuevas del
   capítulo 7, con sus dos salvedades obligatorias: el equilibrio de la era reservada **no existe**
   —el exceso ya es negativo con coste cero— y la capacidad acota el trabajo a un patrimonio
   pequeño.

**Dos hallazgos que el plan no anticipaba.** La cartera ganadora mejora **operando menos** (rotación
a la mitad, tenencia mínima máxima, deriva relajada), de modo que el capítulo 7 pasa de justificar
una rotación alta a explicar por qué conviene una baja. Y la estabilidad económica entre semillas es
ahora **verdadera**, al contrario que en la cadena anterior.

---

## 5. Descartado, y por qué

- **Acortar la ventana de entrenamiento** para reducir el sesgo. De 8 a 4 años se pierde la mitad de
  las filas para quitar 2,3 pp: el sesgo no está concentrado en los años que se recortarían.
- **Mover el ancla OOS** de 2015 a 2019. Reduciría el sesgo de 26 a ~14 pp, pero dejaría 7 años de
  evaluación en vez de 11 con 2025-2026 ya reservados como estrés. Además ancla y lookback están
  **pre-registrados**: cambiarlos tras ver la cobertura sería elegir la ventana en función de lo
  observado.
- **Un proveedor de precios de pago**. Se evaluó Tiingo, que cubría 185 de los símbolos ausentes, y
  se descartó por su límite de 50 símbolos/hora. El manuscrito debe decir que la cobertura completa
  del histórico lo exigiría.
- **Reconstruir precios desde los ratios de fundamentales**. Medido sobre 589 tickers: correlación
  0,97 en nivel pero **0,73 en retornos**, y serie trimestral. Todo lo predictivo son retornos.
- **Ablación de agentes**: con un meta sin pesos mínimos, el propio meta ya puede anular agentes y
  los pesos aprendidos muestran esa información.
- **Ampliar el catálogo o la rejilla de carteras**: cada configuración adicional empeora el Deflated
  Sharpe, que ya es la limitación de severidad más alta.
- **Extender el universo fuera del S&P 500**: es otro trabajo, no una corrección de este.
- **Optimizar comisión o slippage**: sería elegir el mundo en el que la estrategia luce mejor. Por
  eso la sensibilidad a costes es un diagnóstico y nunca un criterio de selección.
- **Reejecutar el ganador al cerrar el Portfolio Study**: ese estudio no reentrena, así que sus
  artefactos de modelo son los del Model Study de origen y se enlazan.

---

## 6. Deuda abierta

Cada cambio de código que afecte al manuscrito añade aquí una entrada: qué cambió, a qué capítulo,
tabla o figura afecta y qué artefacto lo respalda. **Sin cifras.** Cuando una entrada se resuelve, se
borra de aquí y su historia queda en `docs/bitacora.md`: este fichero dice qué falta, no qué se hizo.

A 2026-08-24 no hay deuda de contenido pendiente. Queda un aviso permanente, dos decisiones
editoriales abiertas y un cambio de nombres de fichero sin efecto sobre el contenido.

### Resuelta (2026-09-01) — revisión editorial de las 18 anotaciones

Se aplicaron las dieciocho y se cerró la deuda de contenido. Lo que conviene recordar de esa pasada:

- **El manuscrito describía un mecanismo que el código no implementa.** Ver la entrada del
  2026-09-01 en `docs/bitacora.md`. Corregido en el capítulo 3.
- **Seis figuras salieron de la memoria** porque su conclusión cabía en una o dos cifras. Tres de
  ellas —`f06_bootstrap`, `f07_capacidad`, `f07_perfiles_pesos`— **las sigue usando `TFM_ppt.tex`**,
  así que siguen en `figures/` y el exportador las sigue generando: sólo dejó de citarlas la
  memoria. Las otras cuatro —`f06_estabilidad_features`, `f06_cola_eras`, `f07_alpha_turnover_anual`
  y `f05_calibracion_alfa`— se borraron del disco y del exportador, con sus funciones de dibujo.
- **Las dos tablas apaisadas pasaron a vertical** cambiando los anchos en el exportador
  (`export_study_assets.py`, funciones `write_tables_catalog` y `write_feature_dictionary`) y
  reformateando a mano los `.tex` ya generados, porque el exportador **no se ejecutó**. Si se vuelve
  a exportar, saldrán ya con los anchos nuevos. El diccionario perdió la columna «Fuente».

### Deuda nueva — el nombre `execution_lag_days` induce a error

`module/studies/catalog.py:142` describe la variable como «días exigidos entre el cierre fiscal y la
disponibilidad operativa de fundamentales», y sus descripciones por valor hablan de riesgo de
*lookahead*. El código no hace eso: `module/data/dataset.py:223-235` suma el lag al **fin de mes
calendario** para colocar la rejilla de observación, y quien impide el *lookahead* es el filtro por
`filed_date` de `dataset.py:450`, que no usa el lag.

El manuscrito ya está corregido y describe los dos mecanismos por separado. Lo que queda pendiente
—y **no** es deuda del LaTeX sino del código— es decidir si el parámetro se renombra (a algo como
`observation_lag_days`) y si su descripción del catálogo se reescribe. No se tocó aquí porque el
nombre entra en la clave de caché y en los artefactos ya persistidos: cambiarlo invalidaría los
estudios adoptados. Si algún día se renombra, hay que revisar el capítulo 3 y el Anexo C.

### Deuda nueva — la defensa quedó desalineada con la memoria

`TFM_ppt.tex` **no se tocó** en esta pasada. Los puntos donde ahora difiere del informe:

- La memoria ya no tiene Abstract.
- El capítulo 3 explica el retardo de observación de otra forma; si la defensa lo menciona, dice lo
  que el informe ya no dice.
- Los capítulos 6 y 7 están reorganizados y el 7 tiene una sección nueva (caso Apple) que la
  defensa podría aprovechar.
- Existe un Anexo C nuevo.
- Los identificadores de código ya no aparecen en la prosa de la memoria.

### Sin deuda de contenido — los ficheros del manuscrito cambiaron de nombre (2026-08-26)

`main.tex` pasa a `TFM.tex`, `presentacion.tex` a `TFM_ppt.tex` y `presentacion_notas.tex` a
`TFM_ppt_notes.tex`, con sus PDF a juego (`TFM.pdf`, `TFM_ppt.pdf`, `TFM_ppt_notes.pdf`); el
duplicado `main.pdf` desaparece. **No cambia ni una línea de prosa, ni una cifra, ni una figura,
ni una tabla**: solo los nombres de los dos documentos maestros y los comentarios que se citaban a
sí mismos. No hay nada que reescribir en la próxima actualización del manuscrito; la entrada está
aquí para que quien vuelva sepa por qué las rutas de este plan ya no dicen `main.tex`.

### Aviso permanente — existe un Portfolio Study posterior que no es el del manuscrito

`results/studies/study-20260817-231905-9ac87639` es un Portfolio Study ejecutado **después** del
adoptado, sobre el mismo `source_study_id`, con 640 combinaciones en vez de 1.440 y un ganador
distinto (`sizing_mode: equal`, `coverage_percentile_floor: 60`). Su `portfolio_narrative.json` es
más rico.

El manuscrito está anclado a `study-20260817-212856-f86ca822` y así debe seguir salvo decisión
expresa. Queda escrito para que nadie mezcle cifras de los dos: son carteras diferentes y sus
relatos no son comparables. El exportador rechaza por construcción cualquier identificador que no
sea el adoptado, de modo que el aviso protege sobre todo contra la lectura manual de artefactos.

### Decisión editorial cerrada (2026-09-01) — las dos figuras salieron

Las dos que quedaban anotadas aquí se retiraron de la memoria al aplicar el límite de extensión, y
por el motivo que ya estaba escrito: su conclusión cabía en el texto.

- `f07_capacidad.png` sostenía dos cifras que ahora van en una frase. **Sigue existiendo** porque la
  usa `TFM_ppt.tex`, donde la escala logarítmica sí aporta.
- `f05_calibracion_alfa.png` solapaba con `t05_calibracion_estados.tex`, que ya daba el recuento de
  estados. **Borrada del disco y del exportador**: no la usa nadie.
