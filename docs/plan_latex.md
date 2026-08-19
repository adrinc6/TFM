# Plan del manuscrito LaTeX

> **Este documento es un plan vivo, no un encargo de un solo uso.** Cumple dos funciones:
>
> 1. Es el **plan completo** para dejar el manuscrito al día con la situación actual del proyecto.
> 2. Es el **destino de toda deuda futura con el manuscrito**: cualquier cambio de código que afecte
>    al LaTeX añade aquí qué habrá que escribir, en qué capítulo y con qué artefacto detrás
>    (sección «Deuda nueva», al final).
>
> **Aquí no se copian cifras de resultados.** Viven en `results/studies/<study_id>/` y se leen de
> ahí. Sí se recogen las cifras del panel, porque son mediciones cerradas sobre `data/raw/` que no
> dependen de ningún estudio.
>
> Regla de oro: **ninguna cifra entra en un `.tex` sin poder señalar el artefacto exacto** que la
> respalda (`study_id` y ruta).

El manuscrito (`latex/main.tex`, `latex/presentacion.tex`, `latex/assets/*.tex`) está **congelado**:
los cambios de código no lo editan. `latex/scripts/*.py` y `latex/plan_tfm.md` sí son editables, pero
el exportador no se ejecuta como parte de un cambio corriente.

**Última actualización del manuscrito: 2026-08-19**, con la migración visual del informe y la
defensa. Desde entonces vuelve a estar congelado y la deuda se acumula en §10.

Dos avisos de alcance que este documento no traía y que costaron tiempo:

- **`latex/presentacion.tex` cuenta como manuscrito.** Tiene sus propias cifras, y varias están
  escritas **con letra** en el guion hablado («mil cuatrocientas cuarenta», «diez puntos básicos de
  comisión»), donde ningún `grep` numérico las encuentra.
- **Los identificadores de estudio viven en tres sitios** que se desincronizan solos: las macros de
  `main.tex`, el bloque literal de comando de `a_reproducibilidad.tex` —que el exportador **no**
  toca— y `asset_manifest.json`, que sí se regenera. El `dataset_hash` es un literal más, sin macro.

Este plan cubre **qué escribir y dónde**. El **cómo** del formato —notación, estructura de carpetas,
convenciones de redacción, criterio de poda de activos— está en `latex/plan_tfm.md` y no se duplica
aquí.

---

## 1. Estado del manuscrito

15 capítulos y anexos en `latex/assets/`, **18 tablas** `t*.tex` y **18 figuras** `f*.png` tras la
actualización de 2026-08-19. La tabla anual de cobertura fue sustituida por una figura leída de la
fuente canónica; el manifiesto declara además SPIVA y las ocho series anuales de perfiles.

La distinción que más errores evita, y que no está recogida en ningún otro sitio:

**13 tablas se generan solas** al ejecutar `export_study_assets.py` y se refrescan sin intervención.
**5 se escriben a mano** y no las actualiza nadie por ti. La auditoría de 2026-08-18 encontró que las
cinco manuales llevaban cifras que contradecían al capítulo que las cita: son las que hay que releer
cada vez que cambie un estudio.

| Tabla manual | De dónde sale su contenido |
|---|---|
| `t01_afirmaciones.tex` | Las cinco afirmaciones vertebradoras: combina los cuatro estudios más juicio cualitativo |
| `t04_tests_contrato.tex` | Recuento de funciones `test_*` en `tests/` |
| `t08_defectos_validez.tex` | Transcrito de `docs/bitacora.md` |
| `t08_perfiles_def.tex` | Transcrito de `PROFILE_WEIGHTS` (`module/evaluation/profiles.py`) |
| `t09_limitaciones.tex` | Compuesto a mano desde los artefactos citados en el capítulo 8 |

`latex/asset_manifest.json` las lista todas juntas porque se construye con un glob de `t*.tex`, no
rastreando autoría: **no sirve** para saber cuáles son generadas. Y como el glob describe el disco en
el momento del export, una tabla borrada a mano sobrevive en el manifiesto hasta la siguiente
ejecución.

`latex/scripts/verify_latex_assets.py` exige que toda tabla esté referenciada con `\input` y toda
figura citada. Un activo huérfano hace fallar la comprobación.

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

## 3. Bloque A — Lo que ya está decidido

> **Ejecutado el 2026-08-17.** Los tres pasajes se reescribieron y la tabla nueva existe. Se conserva
> la descripción porque documenta *por qué* el manuscrito dice lo que dice, que es lo que hará falta
> si alguna de estas afirmaciones vuelve a cuestionarse.

Su contenido no depende de ningún estudio: son mediciones cerradas sobre `data/raw/`.

### 3.1 Cobertura del universo · `03_datos_y_universo.tex:90-103`

Tres cambios en el mismo pasaje:

1. **La afirmación de la línea 103 ya no es cierta.** Dice que la ausencia de empresas «no es
   observable desde el propio panel». `ticker_diagnostics.csv` da una fila por cada uno de los 1206
   tickers con su motivo de exclusión, así que **sí es observable y está medida**. Es una mejora del
   trabajo, no una rebaja: se pasa de una disculpa epistemológica a una medición.
2. **Corregir las cifras de las líneas 95-99** con la cobertura real (2003: 49,0 %; 2026: 99,2 %).
3. **Explicar las dos columnas de la tabla de cobertura**: una mide calidad de las filas construidas
   (no baja del 99,4 %) y otra el alcance sobre el índice. Solo la segunda habla del sesgo, y
   confundirlas es lo que hacía que la tabla pareciera decir que todo estaba bien.

### 3.2 Error factual sobre el tamaño del índice · `t09_limitaciones.tex:47-49`

La fila «Universo restringido» dice hoy *«S&P 500 estadounidense; 278 tickers en 2003 frente a más de
500 en años recientes»*, y `08_limitaciones.tex:53-57` lo repite en prosa. Insinúa que el índice
creció, y **el S&P 500 tiene ~500 miembros desde 1957**: la cifra describe cobertura perdida del
panel, no el tamaño del índice.

La fila pasa a llamarse **«Cobertura incompleta del universo»** y se enuncia como defecto medido, con
la cobertura efectiva por año. Es la corrección más urgente de la lista: un tribunal que conozca el
índice detecta el error de inmediato.

### 3.3 El sesgo de supervivencia, medido · `08_limitaciones.tex:53-57`

Hoy se enuncia como posibilidad vaga («las fuentes gratuitas pueden no ofrecer la misma
profundidad»). Pasa a declararse con las tres cosas que ahora se saben: **causa** (retirada de
símbolos por el proveedor, no mortalidad: solo el 5,5 % son quiebras), **dirección** (optimista:
infla) y **decaimiento** (de 26 a 8 pp a lo largo del OOS).

Reetiquetado: de «sesgo de supervivencia» a **«sesgo de supervivencia por cobertura del proveedor»**
—conserva la dirección, precisa el mecanismo—.

Tres avisos de redacción obligatorios:

- **No usar el argumento** «como casi todas las ausentes fueron adquiridas con prima, el sesgo sería
  conservador». Es tentador y **falso**: cuenta cabezas en vez de permanencia. Una adquirida rinde
  una vez y desaparece; una superviviente compone durante todo el periodo.
- **Límite de lo demostrable**: se ha medido la **composición**, no el **retorno** de las excluidas
  (solo 35 de 563 tienen alguna fila de precio, verificado en `ticker_diagnostics.csv` contando
  `price_first` no nulo). El sesgo **no se cuantifica en puntos de
  rentabilidad** y el manuscrito no debe insinuarlo.
- **Dato que refuerza la validez**: los 563 excluidos tienen fundamentales. El cuello de botella es
  exclusivamente el precio, así que la exclusión no señala empresas sin información contable.

### 3.4 Guarda de reciclaje y ventana de descarga · `03_datos_y_universo.tex:53-58`

Actualizar dos hechos: la serie de un símbolo reciclado se **trunca al periodo de pertenencia** en
vez de descartarse entera, y la regla lleva dos salvaguardas (no aplica si el ticker salió antes del
inicio de la ventana; exige margen de 30 días). Añadir que la **ventana de descarga (1990) es
distinta del inicio del panel (2003)**: se baja más historia de la que el panel usa, para resolver el
universo y alimentar medias móviles y momentum, sin mover el periodo evaluado.

### 3.5 Tabla nueva: resolución del universo

Una tabla con el reparto por motivo de exclusión, en `03_datos_y_universo.tex`. Sustituye una
disculpa por una medición e impide presentar los símbolos no resueltos como mortalidad.

Requiere **una función nueva en `latex/scripts/export_study_assets.py`** con prefijo `tNN_` y su
`\input` correspondiente, o `verify_latex_assets.py` la marcará huérfana. Fuente:
`universe_coverage.json` → `ticker_resolution` y `ticker_diagnostics.csv`.

---

## 4. Bloque B — Qué recoger cuando terminen los estudios

> **Ejecutado el 2026-08-17.** Las siete lecturas están resueltas y resumidas en «Resultados» (§8).
> La tabla se conserva como guion de qué mirar cuando vuelva a correrse la cadena.

Lecturas, con su artefacto y **qué decide cada una**.

| # | Qué leer | Dónde | Qué decide |
|---|---|---|---|
| 1 | `study_id` de los Model Studies y del Portfolio Study, `dataset_hash` | `study.json`, `winner.json` | Macros de `main.tex` y bloque de `a_reproducibilidad.tex` |
| 2 | Ganador y cadena: configuración y fase que la decidió | `decisions.json`, `winner.json` | Capítulo 6 entero |
| 3 | **Rank-IC por era** | `t05_rankic_era.tex` (`agent_era_matrix`) | **La respuesta al sesgo de cobertura** |
| 4 | ¿Corrió toda la cadena bajo la misma versión de catálogo? | `catalog_snapshot.json` | Puede **eliminar** una limitación |
| 5 | Deflated Sharpe y nº de configuraciones | `robustness.json` | Fila 1 de limitaciones, la de severidad más alta |
| 6 | ¿Sigue dominando un solo agente? | `evidence/summary.json` | Fila 5 de limitaciones |
| 7 | Diagnósticos económicos | `cost_sensitivity.json`, `capacity.json`, `portfolio_narrative.json` | Capítulo 7 y dos filas de limitaciones |

Notas sobre cuatro de ellas:

**(3) La lectura por eras es la que responde al sesgo.** El sesgo vale ~26 pp en 2015-2018 y ~8 pp en
2022-2024: si el Rank-IC es **estable entre eras**, el sesgo no está impulsando el resultado y es un
argumento medido a favor; si la ventaja se **concentra en 2015-2018**, la limitación es real y queda
localizada. Sin esta lectura explícita, la cautela del capítulo 8 es retórica.

No confundir con `bootstrap_and_eras` (`module/research/robustness.py`), que recalcula el Rank-IC
**excluyendo** cada era —si el resultado depende de una sola— que es una pregunta distinta y
complementaria.

**(4) Si la versión de catálogo es única, la limitación desaparece** en vez de reescribirse. Ojo, que
son **cinco** sitios y no tres, y encontrarlos costó: la fila de `t09_limitaciones.tex`, la sección de
`08_limitaciones.tex`, la mención en la lista de estudios de `a_reproducibilidad.tex`, la sección «El
libro de versiones» de `d_auditoria_desarrollo.tex` —que es donde vive el `\input`, no en el capítulo
8— y la propia tabla `t08_versiones_catalogo.tex`, que hay que **borrar**: dejarla en disco sin
`\input` la convierte en huérfana y hace fallar la verificación. Hay además **dos** `\ref{tab:versiones}`,
y borrar la tabla sin quitar ambas deja una referencia sin destino.

**(5) El número de configuraciones cambia**: la rejilla del Model Study ya no gasta runs en variables
de cartera, que ahora quedan fijas y se optimizan solo en el Portfolio Study.

**(7) `cost_sensitivity.json` puede bajar la severidad** de la fila de costes: hoy figura como Media
y **sin ninguna cifra que la acote**. Con el equilibrio medido pasa a declarar el margen. Lecturas
obligadas: el resimulado debe salir mayor o igual que el congelado; si aparece `beyond_ladder`, el
equilibrio cae fuera de la escalera medida y **no se cita ni se extrapola**. Para `capacity.json`,
comprobar `volume_coverage` antes que nada: con cobertura baja el límite se calcula sobre pocas
órdenes y no se cita.

---

## 5. Bloque C — Capítulo de cartera (material nuevo)

> **Ejecutado en parte el 2026-08-17.** Entraron sensibilidad a costes y capacidad como secciones
> propias, y el capítulo se reescribió sobre el perfil `balanced` —que resultó ser la propia cartera
> ganadora—. **Queda pendiente** el material narrativo de `portfolio_narrative.json`: mapa de
> posiciones, exposición sectorial y permanencia. Se dejó fuera porque cada figura debe responder una
> pregunta que el texto plantee, y esas preguntas aún no están escritas.

El mayor hueco del manuscrito: hoy la cartera se reporta como una curva y unas métricas agregadas, y
no dice **qué hizo**. Va en el capítulo 7. Respaldo: `portfolio_narrative.json`,
`portfolio_narrative_holdings.parquet`, `evidence_best_full/positions.parquet`, `orders.parquet` y
`contributions.parquet`.

**Figuras**

| Figura | Qué muestra | Fuente |
|---|---|---|
| Mapa de posiciones | Qué acciones tuvo en cada snapshot y con qué peso | `positions.parquet` |
| Exposición sectorial | Peso por sector a lo largo del tiempo | `portfolio_narrative.json` → `sector_exposure` |
| Distribución de permanencia | Cuánto dura una posición | `holding_duration` |
| Exceso frente al coste | Las dos familias con `c*` marcado | `cost_sensitivity.json` |

**Tablas**: las más presentes (meses, episodios, peso medio, contribución neta); mayores y menores
contribuciones; mejores y peores operaciones cerradas; ventas que luego subieron; capacidad por
umbral.

**Cuatro salvedades obligatorias**, que no se omiten por brevedad:

1. **El sector no es point-in-time**: procede de una foto actual de Finnhub y solo agrupa.
2. **El nocional diario es aproximado**: precio ajustado por splits y dividendos, volumen solo por
   splits.
3. **La cartera es la mejor de la rejilla**, así que sus cifras dentro de la ventana de selección son
   una cota superior optimista.
4. **Las peores decisiones se leen con el resultado ya conocido**: `sold_and_recovered` señala dónde
   falló la doctrina de umbrales, **no** una regla que se pueda añadir sin volver a ajustar sobre el
   resultado.

Otros dos añadidos menores:

- **Columna de volumen en el panel** · `03_datos_y_universo.tex`: describir
  `median_dollar_volume_21d`, para qué está (capacidad, nunca señal) y su salvedad de ajuste.
- **Anexo de reproducibilidad** · `a_reproducibilidad.tex`: añadir a la lista de artefactos citables
  `contributions.parquet`, `cost_sensitivity.json`, `capacity.json`, `portfolio_narrative.json`,
  `portfolio_narrative_holdings.parquet` y el `report.md` del Portfolio Study.

---

## 6. Orden de edición

El orden no es arbitrario: cada capítulo fija hechos que el siguiente cita, y saltárselo obliga a
reescribir.

| # | Capítulo | Qué entra |
|---|---|---|
| 1 | `03_datos_y_universo.tex` | Panel, cobertura, resolución de tickers (Bloque A) |
| 2 | `06_resultados_predictivos.tex` | Rank-IC, ganador, cadena, **lectura por eras** |
| 3 | `07_resultados_economicos.tex` | Cartera, costes, capacidad (Bloque C) |
| 4 | `08_limitaciones.tex` + `t09_limitaciones.tex` | **Al final**: cada fila cita cifras de 1-3 |
| 5 | `00_resumen.tex`, `09_conclusiones.tex`, `t01_afirmaciones.tex` | Lo último: resumen de lo escrito |

---

## 7. Dónde vive cada cifra que caduca

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
| `study_id` | Macros en `main.tex` **y** bloque literal en `a_reproducibilidad.tex:46-50` |
| `dataset_hash` | `03_datos_y_universo` |

El bloque de `a_reproducibilidad.tex:46-50` reproduce la línea de comandos con los identificadores
escritos literalmente: **no lo actualiza el exportador** y se edita a mano, además de las macros.

---

## 8. Comandos y verificación

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

1. **Identificadores.** Macros de `main.tex`, bloque literal de `a_reproducibilidad.tex` y
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

## 9. Descartado, y por qué

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

## 10. Deuda nueva

### 2026-08-17 — No todas las pasadas publican robustez, perfiles ni atribución

**Qué cambió.** El lanzamiento admite `post_winner_diagnostics` (activo por defecto). Al
desactivarlo, el Model Study termina al congelar el ganador y no genera `profile_comparison.parquet`,
`robustness.json`, `portfolio_comparison.parquet` ni `attribution.json`. Las pasadas intermedias de
una cadena lo usan; el ganador final y el Portfolio Study, no.

**A qué afecta.** El capítulo de metodología debe explicar que los diagnósticos posteriores al
ganador son opcionales por pasada, y **por qué**: la confirmación 2025–2026 se evalúa exactamente
una vez y se reserva para la última pasada, de modo que ejecutarla en las intermedias la gastaría
sobre configuraciones descartadas. Donde el manuscrito describa el recorrido del Study como una
secuencia única e invariable, hay que matizarlo.

Afecta también a cómo se citan las tablas de robustez, perfiles y atribución: debe quedar claro de
**qué `study_id`** salen, porque no todos los studies de la cadena las tienen. Un `study_id`
intermedio no puede respaldar una tabla de robustez que nunca produjo.

**Artefacto detrás.** `docs/metodologia.md`, sección «5 bis → Los diagnósticos posteriores al ganador
son opcionales por pasada»; código en `module/studies/runner.py` y `module/studies/config.py`
(`post_winner_budget`). Sin cifras.

### 2026-08-17 — Saldada: la deuda anterior está en el manuscrito

La entrada de arriba queda **resuelta**. El exportador tolera ahora la ausencia de
`attribution.json` en las pasadas intermedias —antes crasheaba— y el capítulo 6 explica por qué esas
celdas van vacías, en vez de disimularlo. La regla de procedencia por `study_id` está en
`a_reproducibilidad.tex` y en el anexo de evidencia.

### 2026-08-18 — Saldada: el relato de la cartera está en el manuscrito

Lo único del Bloque C que faltaba. El capítulo 7 tiene ahora una sección propia, «Qué compró la
cartera», entre la curva de patrimonio y el análisis de rotación, con tres figuras y una tabla
generadas por el exportador desde `portfolio_narrative.json`. Las cuatro salvedades obligatorias de
§5 están en el texto, y se añadió una quinta que el material hizo evidente: la concentración del
resultado en un solo nombre, que además entra como fila propia en la tabla de limitaciones.

El criterio que había mantenido esto fuera —que cada figura responda a una pregunta que el texto
plantee— se cumplió escribiendo primero el argumento: cuántos nombres pasaron por la cartera y
cuánto duraron, de dónde salió el resultado, qué dicen las dos operaciones extremas sobre la
doctrina de umbrales, y dónde estuvo invertida.

### 2026-08-18 — Deuda nueva: la poda que compensa las figuras nuevas

**Qué falta.** El recuento de activos sube de 24+12 a 25+17. Las dos podas que lo compensarían no
pudieron ejecutarse porque necesitan `.parquet` que el clon no tiene:

- Fusionar `f07_equity` y `f07_drawdown` en una figura de dos paneles con eje X compartido. Son
  anverso y reverso de la misma serie y el capítulo 7 las comenta seguidas. Requiere
  `evidence_best_full/equity.parquet`.
- Convertir `t07_cola.tex` en figura de barras por era. Es una tabla que se lee como forma: lo que
  importa es que el decil superior bate al universo en las tres eras de selección y deja de hacerlo
  en la reservada. Requiere `evidence/rank_tail_diagnostics.parquet`.

**Además**, al compilar conviene mirar la densidad del capítulo 7: si queda demasiado cargado,
`f07_costes_escalera` es la primera que se retira, porque su sección se sostiene en prosa.

### 2026-08-18 — Deuda nueva: el reparto de la salvaguarda de la curva de alfa

**Qué cambió.** `08_limitaciones.tex` afirmaba que la salvaguarda «no se activa en ninguna fila»
(calibración 80 % horizonte / 20 % era) y `t09_limitaciones.tex` que se activa en el 4,1 %
(68,2 / 18,0 / 9,6). Son incompatibles y **ningún artefacto JSON publicado los dirime**: el
diagnóstico de calibración vive en un `.parquet` de la evidencia.

**Qué se hizo.** Reformular ambas menciones sobre lo que sí es comprobable —el mecanismo y su
carácter *a priori*— en vez de elegir una de las dos cifras sin poder verificarla.

**Qué falta.** Al volver a ejecutar el exportador con la evidencia completa, medir el reparto real y
decidir si merece volver a la tabla. Si el reparto entra, tiene que entrar en los dos sitios.

### 2026-08-18 — Aviso: existe un Portfolio Study posterior que no es el del manuscrito

`results/studies/study-20260817-231905-9ac87639` es un Portfolio Study ejecutado **después** del
adoptado, sobre el mismo `source_study_id`, con 640 combinaciones en vez de 1.440 y un ganador
distinto (`sizing_mode: equal`, `coverage_percentile_floor: 60`). Su `portfolio_narrative.json` es
más rico —50 acciones distintas en selección, 13 en confirmación—.

El manuscrito está anclado a `study-20260817-212856-f86ca822` y así debe seguir salvo decisión
expresa. Queda escrito para que nadie mezcle cifras de los dos: son carteras diferentes y sus
relatos no son comparables.

---

Cada cambio de código posterior que afecte al manuscrito añade aquí una entrada: qué cambió, a qué
capítulo, tabla o figura afecta y qué artefacto lo respalda. Sin cifras.

### 2026-08-18 — Saldada: migración editorial, visual y de la defensa

**Qué se cerró.** La migración autorizada ya está incorporada a `main.tex`, `presentacion.tex` y
los activos. Se mantuvo la profundidad de los capítulos técnicos y se podaron únicamente
repeticiones, inventarios y visuales que contaban dos veces la misma historia. Resumen e
introducción presentan una sola taxonomía de objetivos; resultados predictivos y económicos
conservan el razonamiento completo, pero distinguen con más precisión evidencia, interpretación y
límite; conclusiones separa capacidad de ordenación, sensibilidad a la cartera y generalización
económica.

**Deudas visuales saldadas.** Patrimonio y drawdown forman una figura de dos paneles; la cola por
eras es un gráfico; la influencia de cartera se resume en un visual ordenado; contribución,
operaciones y sector forman una narración conjunta; se retiraron la escalera de costes y los
gráficos individuales de semillas y carteras aleatorias. El inventario de evidencia se integró en
reproducibilidad y la auditoría del desarrollo quedó reducida a defectos con efecto sobre la validez
y tres lecciones reutilizables.

**Control interno.** `latex/assets/study_macros.tex` se genera desde los tres Model Studies
adoptados y el Portfolio Study adoptado. El modo `--audit` rechaza identificadores distintos,
recalcula las macros sin escribir y contrasta macros, manifiesto y activos. El cuerpo del manuscrito
usa esas macros sin mostrar al lector rutas de artefactos junto a cada cifra. La evidencia procede
de `winner.json`, `evidence/summary.json`, `robustness.json`, `attribution.json`, `decisions.json`,
`portfolio_grid.parquet` y `portfolio_narrative.json` bajo
`results/studies/study-20260817-094411-568bd37e/` y
`results/studies/study-20260817-212856-f86ca822/`; la cadena se completa con
`study-20260816-182345-3cc1a5fb` y `study-20260817-021135-b5926b62`.

**Defensa.** La presentación queda cerrada en veinte diapositivas narradas, una reserva sin numerar
y un guion de 18:15. Los títulos
formulan conclusiones defendibles, la métrica exacta de la cartera es la única tabla y los gráficos
de eras, cadena, sensibilidad y composición sostienen la explicación oral sin texto esencial en
tamaños reducidos.

### 2026-08-19 — Saldada: contexto competitivo y comparaciones visuales

**Qué se incorporó.** La introducción presenta antes de los objetivos el porcentaje de fondos
large-cap que no supera al S\&P 500 a varios horizontes, con procedencia SPIVA declarada. El capítulo
7 conserva la aritmética de Sharpe y remite a esa figura; ya no afirma una progresión monótona.

La cobertura anual se lee exclusivamente de `data/raw/universe_coverage.json` y sustituye a
`t03_cobertura_anual.tex`, cuya fuente mezclaba poblaciones y podía producir coberturas superiores
al 100 %. En resultados económicos entran relaciones visuales que la prosa no permitía reconocer:
turnover frente a exceso anual, escalera de costes, capacidad, pesos firmados de perfiles y
resultados agregados/anuales de perfiles.

**Procedencia.** El Portfolio Study adoptado aporta `evidence_best_full/annual_metrics.parquet`,
`cost_sensitivity.json`, `capacity.json`, `portfolio_profiles.parquet` y las ocho rutas
`profiles/<perfil>/annual_metrics.parquet`. `PROFILE_WEIGHTS` se importa del código vigente, no se
duplica. `asset_manifest.json` añade `literature_sources.spiva`, la fuente canónica del panel y cada
serie anual de perfil; `--audit` exige todas ellas y los dieciocho gráficos.

**Defensa.** Queda congelada en veinte diapositivas narradas y una reserva sin numerar. SPIVA aparece
después de la portada, perfiles después del resultado reservado y la transición de cartera se
elimina; las dos conclusiones anteriores se funden en una síntesis. El guion objetivo es 18:15.

**Estado de la deuda.** Esta migración no deja deuda nueva: cualquier cambio posterior vuelve a
anotarse aquí sin editar el manuscrito hasta la siguiente migración autorizada.

### 2026-08-19 — Saldada: ampliación explicativa y trazabilidad de la muestra

**Criterio editorial.** Se amplía lo que aportaba valor probatorio y se conserva la frontera entre
capítulos: muestra efectiva en datos; objetivo, relojes y meta en arquitectura; calibración,
precedencia y multiplicidad en protocolo; estabilidad interna y factores en resultados. El capítulo
económico no repite esos mecanismos. La bibliografía se refuerza con trabajos primarios sobre
factores, aprendizaje estadístico, sesgo de selección, liquidez y costes.

**Activos nuevos.** `agent_scores.parquet`, `signal_calibration.parquet` y
`model_feature_attribution.parquet` alimentan `f03_muestra_oos.png`,
`f05_calibracion_alfa.png`, `f06_estabilidad_features.png`, `f06_atribucion_factorial.png` y tres
tablas generadas. El diccionario de 68 variables procede directamente de `feature_catalog.json`.
El manifiesto y `--audit` exigen estas fuentes y salidas.

**Valor del proyecto que se hace visible.** Los anexos explican el ciclo de vida de Study y run, la
cadena de custodia de la evidencia y cuatro familias de defectos materiales que originaron contratos
automáticos. Esta ingeniería se presenta como garantía de validez, no como resultado económico.

**Límite de esta intervención.** Se regeneraron sólo PNG, cuerpos de tabla, macros y manifiesto. Por
indicación expresa no se ejecutó XeLaTeX, no se creó ningún PDF y no se hizo inspección paginada; la
validación se limita a fuentes, activos, auditoría estática y pruebas del proyecto.
