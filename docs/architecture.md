# Arquitectura del sistema

Este documento explica **qué hace cada pieza del código y por qué está decidida así**. Cada módulo
se presenta junto a la justificación metodológica que lo gobierna: separarlas produciría dos
verdades que se desincronizan.

Para instalarlo y ejecutarlo, ver [usage.md](usage.md). Para saber qué produce y cómo leerlo, ver
[results.md](results.md).

---

## 1. La pregunta de investigación

El objetivo principal no es encontrar retrospectivamente la cartera con mayor rentabilidad. La
pregunta es si un sistema de agentes puede aprender una ordenación transversal de acciones que
mantenga capacidad predictiva fuera de muestra y a través de regímenes diferentes.

La variable central es **Rank-IC**: correlación de Spearman entre la puntuación producida en una
fecha y el retorno futuro observado cuando la etiqueta queda cerrada. Rentabilidad y alfa son
resultados económicos posteriores. Separar ambas capas evita seleccionar reglas de cartera porque
casualmente funcionaron mejor en la historia conocida.

Esa separación es la que explica la forma del código: hay un plano que **decide** (Rank-IC, en
`module/studies`) y un plano que **describe** (economía, en `module/evaluation` y
`module/research`). Ninguna pieza del segundo puede mover al ganador del primero.

## 2. Mapa del repositorio

```text
main.py                  Único punto de entrada: ingest | serve
environment.py           Constantes científicas y Settings congelado
module/
  common/                Utilidades de ficheros y logging
  data/                  Ingesta, universo histórico y panel point-in-time
    ingest/              Clientes de Finnhub, Yahoo Finance y SEC EDGAR
  modeling/              Features, cinco agentes especialistas y meta-agente
  evaluation/            Cartera, backtest, perfiles y estadística
  research/              Diagnósticos post-hoc: robustez, atribución, capacidad, costes
  storage/               Datasets preparados, caché, evidencia y persistencia de estudios
  studies/               Catálogo cerrado, runner, selección y Portfolio Study
  web/                   API HTTP y consultas del dashboard
app/                     Dashboard: HTML, CSS y un único fichero JS
tests/                   Tests de contrato
latex/                   Manuscrito, exportador de activos y build
results/studies/<id>/    Artefactos de cada estudio
data/                    Datos crudos, preparados y caché (no versionados)
```

El flujo completo, de extremo a extremo:

```text
main.py ingest
   │  Finnhub (fundamentales) + Yahoo (precios) + EDGAR (fechas reales de publicación)
   ▼
data/raw/                        agregados consolidados
   │  universo histórico del S&P 500 + lag de publicación
   ▼
data/prepared/<hash>/            panel point-in-time inmutable y compartido
   │  features por agente
   ▼
cinco agentes (quality, value, growth, momentum, risk)
   │  combinación causal sobre cohortes cerradas
   ▼
meta-score  ──►  Rank-IC  ──►  selección secuencial  ──►  ganador congelado
   │                                                          │
   │                                                          ├─ robustez
   │                                                          ├─ atribución
   ▼                                                          └─ perfiles
Portfolio Study (rejilla cartesiana por Information Ratio)
   │
   ▼
results/studies/<study_id>/      evidencia, informe y artefactos
```

## 3. `environment.py`: la fuente única de constantes

Todos los parámetros de infraestructura y las constantes científicas viven en un solo fichero. No
hay YAML, TOML ni JSON de configuración. `Settings` es un `@dataclass(frozen=True)` que valida cada
campo en `__post_init__` y lanza `ValueError` ante cualquier valor inválido: una configuración
imposible falla al construirse, no a mitad de un estudio de horas.

Las constantes que marcan la ciencia:

| Constante | Valor | Por qué |
|---|---|---|
| `DATA_START_DATE` | 1990-01-01 | Ventana de **descarga** |
| `PANEL_START_DATE` | 2003-01-01 | Inicio del **panel evaluado** |
| `DATA_END_DATE` | 2026-07-15 | Fin de la descarga |
| `EXECUTION_YEAR` / `QUARTER` | 2015 / 1 | Ancla fuera de muestra fija; nunca se barre |
| `EXECUTION_LAG_DAYS` | 45 | Días entre la publicación real y el uso del dato |
| `TARGET_HORIZON_MONTHS` | 6 | Horizonte del retorno que aprende el modelo |
| `TRAIN_LOOKBACK_YEARS` | 8 | Historia de cada ajuste walk-forward |
| `SELECTION_UNTIL_YEAR` | 2024 | Frontera entre lo que decide y lo que solo confirma |
| `SEED_ENSEMBLE` | 5 | Réplicas por agente que se promedian |
| `RANDOM_SEED` | 42 | Semilla base |
| `BENCHMARK_TICKER` | SPY | Solo benchmark; **nunca** una posición |

**La ventana de descarga (1990) es deliberadamente distinta del inicio del panel (2003)**: se baja
más historia de la que el panel usa, para resolver el universo y alimentar medias móviles y
momentum a 12 meses ya desde el primer snapshot, sin mover el periodo evaluado. Ambas fechas entran
en la huella del dataset, de modo que ampliar la descarga no puede cambiar el ganador en silencio.

Tres campos de versión de código invalidan la caché cuando cambia la ciencia:
`dataset_code_version`, `features_code_version` y `agents_fit_code_version`. Subirlos obliga a
recalcular en lugar de reutilizar un resultado producido por otra implementación.

`_load_dotenv()` (ocho líneas, sin dependencias) lee `.env` de la raíz y hace `os.environ.setdefault`,
de modo que una variable ya presente en el entorno gana sobre el fichero.

---

## 4. `module/data`: ingesta y panel point-in-time

### Las tres fuentes, y por qué son tres

| Fuente | Módulo | Clave | Aporta |
|---|---|---|---|
| **Finnhub** | `ingest/clients.py` | `FINNHUB_API_KEY` | Perfiles de empresa y series de métricas fundamentales |
| **Yahoo Finance** | `ingest/clients.py` | No | Series diarias de precios |
| **SEC EDGAR** | `ingest/edgar.py` | No, pero exige User-Agent identificativo | **Fechas reales de publicación** (`filingDate`) desde 1993 |

EDGAR no es un lujo, es la columna vertebral anti-lookahead. Finnhub fecha los fundamentales por el
`period` fiscal al que se refieren, que es precisamente la fecha en la que el dato **todavía no era
público**; y solo ofrece fechas de presentación desde 2010. Usar el `period` como si fuera la fecha
de disponibilidad introduciría lookahead en todo el panel anterior a 2010. El `filingDate` de EDGAR
es la fecha en la que el informe se presentó de verdad, es gratuita y cubre desde 1993.

`ingest/pipeline.py::download_raw_data()` orquesta la descarga. Comprueba primero que las fuentes
respondan (`_require_sources_reachable()`) antes de entrar en un bucle de ~1.200 tickers: fallar
rápido evita descubrir a las tres horas que faltaba una credencial.

Los fallos de precio se clasifican con una taxonomía explícita (`PRICE_FAILURE_REASONS`:
`not_found`, `bad_request`, `rate_limited`, `http_error`) para que **una avería de red no se
confunda con la desaparición de una empresa**. Esa distinción es la que sostiene el análisis de
cobertura de la sección siguiente.

### El universo dinámico

`universe.py` reconstruye la composición histórica del S&P 500 a partir del CSV de componentes y
cambios históricos que vive en `data/`, con una instantánea por día de mercado desde 1996. Expone
`historical_universe()`, `members_at()` y `first_membership_date()`.

Una empresa solo puede participar en las fechas en las que pertenecía al índice: una señal fechada
en 2000 solo ve los miembros de 2000. Los fundamentales se incorporan según su fecha de publicación
más el lag de ejecución; precios y SPY se alinean con cada snapshot.

### Cobertura del universo: dónde entra realmente el sesgo de supervivencia

La composición del índice es point-in-time y se intenta descargar **todo** ticker que perteneció al
S&P 500 en algún momento, no solo los vivos (`historical_universe`), con una guarda contra símbolos
reciclados: si el primer precio disponible es posterior a la última fecha en el índice, esos datos
son de otra empresa que reutilizó el símbolo (`is_recycled_ticker`). La serie se **trunca al periodo
de pertenencia** en vez de descartarse entera, porque un símbolo reciclado conserva historia legítima
mientras la empresa estuvo en el índice, y es justo la que el panel necesita.

La regla compara una fecha de *pertenencia* con una de *disponibilidad*, así que lleva dos
salvaguardas contra el falso positivo: no se aplica cuando el ticker salió del índice antes de que
empiece la ventana de descarga —ahí el primer precio observable es tardío por construcción, no por
reciclaje— y exige un margen de 30 días, porque reasignar un símbolo lleva meses y un hueco de
semanas es historia truncada por el proveedor.

Eso elimina el sesgo de supervivencia **de la composición**, pero no el de la **cobertura de datos**,
que es donde de verdad vive. Una empresa entra en el panel solo si tiene precio observable y un
periodo fundamental casado con un informe publicado; las que quebraron, fueron absorbidas o
cambiaron de símbolo tienden a fallar ese requisito, de modo que el panel de los años tempranos es
más pequeño que el índice de esos años. **El índice ha tenido ~500 miembros durante todo el periodo
estudiado**: cualquier diferencia entre eso y el número de tickers del panel es cobertura perdida, no
un índice más pequeño, y confundir ambas cosas convierte una limitación honesta en un error.

Por eso la cobertura se **mide y se publica**, no se declara: `universe_coverage.json` registra, por
año, los miembros del índice, cuántos son elegibles para el panel y el motivo de exclusión de cada
uno de los demás, y `ticker_diagnostics.csv` baja al detalle con **una fila por cada ticker del
universo** (estado del precio, cobertura de fundamentales, CIK, informes y motivo de exclusión). Un
ticker que no resuelve **no es prueba de que la empresa muriera**: puede haber cambiado de símbolo,
presentar formularios de emisor extranjero o que el proveedor de precios haya retirado el símbolo.
Esas causas son separables y se cuentan por separado, precisamente para que la mortalidad real no se
sobreestime — de ahí que `symbol_withdrawn` (el proveedor no sirve el símbolo) y `download_failed`
(avería reintentable) no se confundan con `missing_price` (sin serie observable).

#### Qué dijo la medición, y por qué importa para leer los resultados

La causa dominante de exclusión **no es la mortalidad**: solo el 5,5 % de los excluidos llevan el
marcador `Q` de quiebra. Es la retirada de símbolos por el proveedor, que purga lo que deja de
cotizar. Por eso la exclusión depende de la **antigüedad de la salida del índice**: 1,2 % entre los
que siguen en él y 94,4 % entre los que salieron hace más de veinte años. Los 503 miembros actuales
están casi todos en el panel; el agujero son los tickers históricos.

La consecuencia es un **sesgo de supervivencia que infla los resultados y decae con el tiempo**: en
1998 los miembros incluidos sobreviven hasta hoy un 42 pp más que los del índice real, y en 2026 la
diferencia es nula. Como el entrenamiento es rolling, el exceso baja de 26 pp (evaluando 2015) a
8 pp (evaluando 2026).

Dos cautelas que se derivan y gobiernan la lectura:

1. **No vale el argumento «casi todas las ausentes fueron adquiridas con prima, luego el sesgo es
   conservador».** Cuenta cabezas en lugar de permanencia: una adquirida rinde una vez y desaparece,
   una superviviente compone durante todo el periodo. Se comprobó y es falso.
2. **Lo medido es la composición, no el retorno de las excluidas** (solo 35 de 563 tienen alguna
   fila de precio), así que el sesgo **no se cuantifica en puntos de rentabilidad**.

Por eso la limitación no se responde acortando la ventana de entrenamiento —de 8 a 4 años se pierde
la mitad de las filas para quitar 2,3 pp— sino **leyendo el rendimiento era por era**, que ya se
calcula y no requiere código nuevo:

- `SELECTION_ERAS = ((2015,2018), (2019,2021), (2022,2024))` en el catálogo, pre-registradas.
- `agent_era_matrix()` (exportador LaTeX) da el **Rank-IC medio por agente y era**. Es la lectura
  que responde a esta pregunta: el sesgo vale ~26 pp en 2015-2018 y ~8 pp en 2022-2024, así que un
  Rank-IC estable entre eras indica que el sesgo **no** está impulsando el resultado, y una ventaja
  concentrada en 2015-2018 lo delataría.
- `bootstrap_and_eras()` (`robustness.py`) responde una pregunta **distinta y complementaria**:
  recalcula el Rank-IC **excluyendo** cada era, para ver si el resultado depende de una sola. No
  sustituye a la lectura anterior; conviene no confundirlas al redactar.

### El panel y su identidad

`dataset.py::build_point_in_time_dataset()` construye el panel a partir de los agregados crudos y
define `PANEL_COLUMNS`. Aplica el lag de publicación de modo que ningún fundamental sea visible
antes de su `filingDate` más el lag.

La identidad del dataset incluye fuentes, fechas, universo, cadencia, horizonte, lag y versiones de
transformación. Una identidad igual reutiliza `data/prepared/<dataset_hash>/`; una identidad
distinta crea otra materialización. Los estudios guardan **referencias, no copias**.

**La identidad de una evaluación incluye el hash del dataset.** Sin él, dos configuraciones
idénticas evaluadas sobre datos distintos compartían clave de caché y la segunda leía el resultado
de la primera: en artefactos anteriores se observa la misma `evaluation_key` asociada a dos CAGR
distintos. Es un fallo de corrección, no de reporte, y por eso la clave se calcula a partir de la
identidad del dataset resuelta antes de materializarlo.

Controles de ausencia de lookahead:

1. La fecha efectiva del fundamental no supera el snapshot.
2. El target se calcula hacia delante, pero solo se usa para evaluar o entrenar meta cuando su
   `label_end_date` ya ha pasado.
3. El meta solo consume cohortes OOS trimestrales cerradas.
4. 2025–2026 se separa antes de cualquier selector.

### Volumen negociado de referencia

El artefacto de precios publica `median_dollar_volume_21d`: la mediana de `precio × volumen` en las
21 sesiones anteriores o iguales al snapshot, con la misma disciplina hacia atrás que el resto del
panel —una sesión posterior no puede alterarlo—. Existe para dimensionar capacidad y no como
variable predictiva: **ningún agente la ve**.

Mediana y no media porque un único día de volumen extraordinario —una entrada en el índice, una
fusión— inflaría la liquidez estimada justo donde más engaña. **Salvedad obligatoria**: el precio
está ajustado por splits y dividendos y el volumen solo por splits, así que el nocional es una
aproximación. Sirve para saber si una orden cabe en el mercado, no como dato de mercado citable.

### `baselines.py`

Baselines factoriales deterministas (listas de factores QUALITY, GROWTH, VALUE, MOMENTUM) que
producen *puntuaciones*, no retornos. Sirven de referencia contra la que medir si el aparato de
aprendizaje se justifica.

---

## 5. `module/modeling`: features, agentes y meta-agente

### El catálogo de features

`modeling/catalog.py` es un catálogo declarativo de `FeatureSpec`: qué factores point-in-time
existen, a qué bloque pertenecen y qué agente los recibe. Es la **fuente única** de `AGENT_NAMES`,
de modo que la lista de agentes no se declara en dos sitios.

`features.py::build_features()` construye las features PIT, las etiquetas forward separadas y los
baselines, con `FACTOR_SOURCES` como mapa de procedencia. Aquí viven la neutralización sectorial, el
momentum fundamental y el régimen de mercado.

`artifacts.py` calcula transformaciones que están **siempre** disponibles y quedan fuera del
catálogo cerrado —momentum de precio a varios horizontes, medias móviles y tendencia, riesgo técnico
y liquidez diaria—, cada una expuesta como `add_<nombre>(frame, ...)`.

`targets.py` fija el contrato del objetivo, neutro al horizonte: un único artefacto
`targets_forward.parquet` con `forward_return`, `forward_benchmark_return` y
`forward_excess_return`.

### Los cinco agentes

`agents.py::build_agent_scores()` entrena los cinco especialistas —quality, value, growth, momentum
y risk— con reentrenamiento walk-forward. Cada agente ve **features disjuntas** de su especialidad.

El objetivo es `rank_regression` (regresión sobre el percentil transversal de retorno) o `ranking`
(LGBMRanker con lambdarank, agrupado por snapshot). Cada agente entrena `SEED_ENSEMBLE = 5` réplicas
que solo difieren en la semilla y promedia sus scores; el porqué está en la sección de robustez.

### El meta-agente

`meta.py::combine_agent_scores()` devuelve el meta-score, los pesos secuenciales y los diagnósticos
fuera de muestra. Admite peso igual y **stacking Ridge no negativo causal** sobre cohortes
trimestrales **ya cerradas**, con tope por peso y encogimiento hacia el peso igual.

- **Equal:** 20 % exacto por agente.
- **Rolling free:** Ridge positivo, pesos 0–100 %.
- **Rolling bounded:** Ridge positivo, pesos 10–50 %.

El stacker convierte scores y retornos en rangos transversales. Se ajusta en cada fecha con las
últimas 8 o 16 cohortes trimestrales cerradas. Si no existe evidencia suficiente, usa equal. La
causalidad no es un detalle de implementación: un stacker que viera cohortes abiertas estaría
ponderando agentes con información del futuro.

### Qué hipótesis prueba cada etapa

**Temporal.**

- **Cadencia 1/3/6/12 meses:** frecuencia de observación y puntuación.
- **Horizonte 3/6/12 meses:** retorno futuro que aprende el modelo.
- **Lookback 4/8/12 años:** historia disponible en cada fit walk-forward.
- **Lag 30/45/60 días:** prudencia sobre disponibilidad real de fundamentales.
- **Recencia off/lineal/exponencial:** peso temporal de observaciones de entrenamiento.
- **Objetivo:** regresión de ranking o ranking directo.

Un lag menor puede mejorar actualidad, pero exige una hipótesis más fuerte sobre publicación. Por
eso el informe debe mostrar su sensibilidad explícitamente.

**Representación.** Los presets `core` y `all` seleccionan bloques cerrados. **Ambos alimentan a los
cinco agentes**: un preset que deja a un agente sin ningún bloque activo lo elimina de hecho del
sistema, y entonces la comparación deja de medir qué información necesita cada agente para medir qué
pasa al amputar parte de la arquitectura. `core` da a cada agente su bloque esencial; `all` le da
toda la profundidad disponible de su especialidad. También pueden compararse momentum fundamental,
régimen de mercado, neutralización sectorial, winsorización, máximo de features y poda por
estabilidad OOS. No se admiten listas manuales de features.

**Modelo.** Cada agente ajusta LightGBM o Elastic Net. LightGBM permite comparar profundidad,
estimadores, learning rate y mínimo por hoja. Los parámetros incompatibles quedan ocultos e
inactivos.

---

## 6. `module/evaluation`: cartera, backtest y perfiles

### `backtest.py` es contabilidad, no ciencia

`run_backtest()` devuelve un `BacktestResult` y **no calcula Rank-IC**, deliberadamente: la métrica
que decide vive en el plano predictivo y no puede contaminarse con supuestos de cartera. Las
métricas económicas se devuelven **segmentadas**: la ventana de selección (hasta
`SELECTION_UNTIL_YEAR = 2024`) y 2025–2026 se reportan por separado, como confirmación fuera de
muestra.

### La doctrina de cartera

**El propósito de esta etapa completa es la estabilidad, no más alfa.** El ganador (temporal,
representación, modelo, meta) ya está congelado por Rank-IC antes de tocar cartera. Barrer
`max_cash_weight`, `target_size`, `sizing_mode`, `minimum_holding_period`,
`coverage_percentile_floor` o `price_only_sell_only` no es una búsqueda de qué combinación da más
rentabilidad —sería la misma fuga de validez que ya se corrigió una vez, elegir retrospectivamente
por rentabilidad—, sino una exploración de cómo aprovechar de forma **estable y sostenible en el
tiempo** la información que el modelo ya congelado produce: menos rotación innecesaria, menos operar
sobre ruido de precio sin confirmación fundamental, sin comprar a ciegas cuando no hay datos nuevos
que lo justifiquen. `portfolio_comparison.parquet` reporta turnover, coste y dispersión de alfa
entre configuraciones de cartera como diagnóstico de estabilidad; ninguna de esas comparaciones
elige nada, y una opción con más alfa pero más varianza entre semillas o más rotación no se prefiere
solo por el alfa.

SPY es únicamente benchmark y nunca una posición. Los umbrales de la cartera son **económicos, en
puntos básicos de alfa esperado ANUALES**, no percentiles del ranking: un percentil no dice cuánto se
espera ganar y por tanto no puede compararse contra lo que cuesta operar. Definirlos en anual —en vez
de directamente sobre el horizonte del modelo— los hace comparables entre configuraciones con
distinto `target_horizon_months`: 250 pb no pueden significar un 10 %/año con horizonte de 3 meses y
un 2,5 %/año con horizonte de 12.

**Toda la comparación ocurre en base anual.** El alfa esperado ya se estima anualizado (ver abajo) y
los umbrales del catálogo son anuales, así que ninguno de los dos necesita conversión. El que sí la
necesita es el **coste**: comisión y slippage se pagan una vez por operación, no cada año, y
compararlos contra un alfa anual los infravaloraría en cuanto el horizonte baja de 12 meses. Se
anualiza **geométricamente**:

```text
coste_anual = (1 + coste_horizonte)^(12 / horizonte_meses) − 1
```

Compuesto, no lineal —igual que el resto del proyecto anualiza CAGR e IR—: multiplicar el coste por
las vueltas que caben en un año es el mismo atajo aritmético que ya se corrigió en otros sitios. Con
horizonte de 12 meses la conversión es la identidad; con 6 meses, 30 pb de ida y vuelta equivalen a
`(1,003)^2 − 1 ≈ 60,09` pb anuales, no a 60 pb exactos.

#### Alfa esperado: curva percentil → retorno real, con cascada de ventanas

El alfa esperado (`signal_calibration.parquet`) se estima ajustando, sobre cohortes **ya cerradas**,
una recta que relaciona el percentil de `meta_rank` con el **retorno excedente real anualizado** que
obtuvieron las acciones de ese tramo. Las cohortes recientes pesan más que las antiguas (decaimiento
exponencial), porque una relación percentil→alfa de hace cuatro años no describe el régimen actual.

**Estimación y evaluación usan granularidades distintas, a propósito.** La recta se *estima*
agrupando en **20 ventiles** (tramos de 5 puntos de percentil): con un universo de ~500 valores eso
deja ~25 acciones por punto, suficiente para que cada media signifique algo. Agrupar en 100
percentiles dejaría ~5 acciones por punto y la recta se ajustaría sobre ruido, sobre todo en la
ventana más corta. Pero la recta se *evalúa* en el **rank continuo** de cada acción, no en su ventil:
así un p99 recibe estrictamente más alfa esperado que un p88, en vez de compartir el valor de su
tramo, y no aparecen saltos artificiales en las fronteras entre ventiles —justo donde la cartera
decide a quién desplaza—.

La pendiente de esa recta es la condición económica que la cartera necesita: **solo si es creciente**
tiene sentido ordenar por `meta_rank`, porque solo entonces mejor percentil se tradujo en más alfa.
Cuando no lo es, la ventana se descarta y se amplía la evidencia, en cascada:

```text
horizonte objetivo → era (16 trimestres) → todo el histórico → salvaguarda
```

La **salvaguarda** es una recta impuesta de −10 % anual en el peor percentil a +10 % en el mejor. No se
estima de los datos: es un supuesto a priori que se activa justo cuando las tres ventanas dicen que
el ranking no discrimina a favor. Queda registrada por fila en `alpha_curve_window`, de modo que
siempre puede contarse en qué snapshots la cartera operó sobre evidencia y en cuáles sobre supuesto.

Esta formulación sustituye a la calibración isotónica anterior. La isotónica forzaba monotonía
creciente (`increasing=True`), y cuando la relación real es decreciente la única curva creciente que
minimiza el error es **una constante**: el alfa colapsaba al mismo valor para todo el universo y la
cartera dejaba de poder discriminar entre posiciones, congelándolas.

Mientras no hay cohortes cerradas suficientes el valor es `NaN`, no cero: son cosas distintas y la
cartera las trata distinto. Un `NaN` nunca dispara una venta ni bloquea una compra —la regla es
actuar solo ante evidencia económica—, de modo que durante el arranque manda la ordenación y, en
cuanto hay evidencia, mandan los umbrales.

Una ausencia de `meta_rank` es distinta de un alfa sin calibrar: significa que una posición ya no
está en el universo scoreable del snapshot actual (por ejemplo, porque salió del S&P 500 o perdió
cobertura de datos). Se vende de inmediato con el motivo `missing_current_score`, sin respetar el
mínimo de tenencia y sin permitir su recompra en el mismo snapshot. Si existe una candidata, las
reglas normales de llenado deciden el reemplazo; si no, queda efectivo. Esta salida es una regla de
integridad de cobertura, no una decisión de alfa ni un umbral de percentil.

#### Las dos vías de venta

El principio que gobierna todas las órdenes: **una venta solo se emite si el destino del dinero es
mejor que la posición después de costes**. Hay exactamente dos destinos posibles y cada uno tiene su
regla. En cada snapshot se marcan posiciones a mercado y:

- **Rotación (destino: otra acción).** Un outsider desplaza a la peor posición solo si

  ```text
  alfa_esperado(outsider) − alfa_esperado(peor) > 2·(comisión + slippage) + rotation_edge_bps
  ```

  (con `alfa_esperado`, `rotation_edge_bps` y el coste ya todos en base anual), es decir, **la rotación
  paga su propio coste de ida y vuelta** antes de autorizarse. Este es el mecanismo que faltaba: con
  877 % de rotación anual a 15 pb por operación, el coste drenaba en torno a 1,3 puntos porcentuales
  al año contra una ventaja bruta de unos 3,1. Es la **única** vía de venta con `max_cash_weight = 0`
  y sin `price_only_sell_only`: vender por umbral con la obligación de recomprar en el mismo snapshot
  pagaría una ida y vuelta para quedar igual.
- **Venta a efectivo (destino: efectivo).** Con `max_cash_weight > 0`, una posición sale si su alfa
  esperado cae por debajo de `exit_expected_alpha_bps` **y** su plaza puede quedar en efectivo sin
  violar el suelo de diversificación ni el propio tope. Bajo `price_only_sell_only` en un snapshot
  sin fundamentales nuevos, esta vía se abre también con tope 0, sin suelo propio.
- **Compra con histéresis.** Una entrada nueva exige `exit_expected_alpha_bps` **más el coste de ida
  y vuelta de la propia operación**; mantener una posición ya en cartera exige solo el umbral de
  salida. Sin esa banda, una acción oscilando alrededor del umbral se compraría y vendería en
  snapshots consecutivos pagando costes con ventaja esperada nula.
- Las posiciones dentro de la tolerancia mantienen sus unidades y el presupuesto restante se reparte
  respetando las relaciones objetivo.

Por encima de todo lo anterior, **el mínimo de tenencia** (`minimum_holding_period`) bloquea
cualquier venta —por caída de alfa o por rotación— mientras la posición no lleve un mínimo de meses
en cartera, expresado como fracción del horizonte del modelo: `none` (sin mínimo), `quarter_horizon`
(`ceil(horizonte / 4)`), `half_horizon` (`ceil(horizonte / 2)`) o `full_horizon` (el horizonte
completo). Es el único freno que no depende de ninguna magnitud económica, solo del tiempo, y no
busca más alfa sino menos rotación de alta frecuencia sobre el mismo modelo ya congelado.

La única excepción al mínimo es `missing_current_score`: no se puede proteger una posición que el
snapshot actual ya no puede evaluar.

**El suelo de cobertura** (`coverage_percentile_floor`, 0 lo desactiva) vende entera una posición que
cae por debajo del percentil configurado del ranking, aunque su alfa esperado no active ninguna otra
regla. Es la generalización de `missing_current_score` —el percentil ausente sustituido por un
percentil demasiado bajo— y, como aquella, **no se compara contra ningún coste**: no decide si una
operación es rentable, sino si la acción sigue perteneciendo al universo invertible. Por eso convive
con la doctrina de umbrales económicos sin contradecirla. A diferencia de la pérdida de cobertura,
**sí respeta el mínimo de tenencia**, porque la posición sigue siendo evaluable y su exclusión es una
preferencia declarada, no una imposibilidad. Nunca rompe el suelo de diversificación. Debe conocerse
una consecuencia: con `max_cash_weight = 0` el relleno obligatorio recompra en el mismo snapshot sin
aplicar umbrales, así que la regla fuerza una rotación que el bucle económico habría rechazado y
**aumenta** la rotación de la cartera; con tope de efectivo la plaza puede quedarse vacía.

El papel de cada insumo es explícito: el **ranking del meta** decide el orden de preferencia y los
desempates (la confianza de los agentes), el **alfa calibrado** decide si cada operación se paga a sí
misma (la magnitud económica), y los **costes** son el listón que toda operación debe superar.

`price_only_strictness_multiplier` endurece los umbrales en los snapshots que solo traen precio
nuevo (sin fundamentales publicados): baja el umbral de salida y sube tanto el umbral de entrada
como la ventaja exigida para rotar —la banda de histéresis se ensancha—, de modo que la cartera no se
mueve por ruido de precio sin confirmación fundamental.

`price_only_sell_only` va un paso más allá: en esos mismos snapshots, si está activo, se puede
**vender** una posición cuyo alfa esperado ya no cumple, pero se **prohíbe comprar cualquier
reemplazo** —ni compra nueva, ni relleno obligatorio, ni rotación— porque no hay información nueva
que justifique elegir una acción distinta a la ya elegida con datos reales. La plaza vendida queda en
efectivo, sin el tope ni el suelo habituales (es un estado transitorio, no una asignación deliberada
de cartera) y **sin repartirse entre las supervivientes**, porque concentrar la cartera en lo que
queda sería actuar justo cuando se ha decidido no actuar sin información nueva. Dura hasta el
siguiente snapshot con fundamentales frescos. Con esta variable activa, un tope de 0 puede por tanto
mostrar efectivo temporal: el relleno obligatorio vuelve a completar la cartera en el siguiente
snapshot con datos reales si hay candidatas por encima del umbral de entrada.

#### Efectivo

`max_cash_weight` es la **única** variable que gobierna la exposición: fija el peso máximo que puede
quedar sin invertir cuando ninguna candidata supera el umbral (máximo del catálogo: 25 %), y con
valor **0 significa «siempre invertida al 100 %»**, que es la referencia. Existió además una
`cash_policy` con los valores `fully_invested`/`opportunity_cash`, pero era redundante —el catálogo
ya forzaba el tope a 0 bajo `fully_invested` y el suelo de diversificación se deriva del tope—, así
que se eliminó: dos variables para una misma decisión solo invitan a combinaciones inconsistentes.
El efectivo **se remunera al 0 %**: es una cota inferior deliberadamente conservadora, nunca aporta
rentabilidad y solo puede ayudar evitando malas compras y ahorrando costes. Si aun así mejora el
alfa, la mejora no admite discusión.

El tope implica un **suelo de diversificación**: al menos
`ceil((1 − max_cash_weight) · target_size)` plazas deben estar siempre ocupadas —con las mejores por
ranking cuando ninguna supera el umbral—, de modo que replegarse nunca concentra la cartera en unos
pocos nombres (con 12 plazas y tope del 25 %, el suelo son 9 posiciones y ninguna acción puede pasar
de en torno al 15 % del total). El efectivo es además **granular, no continuo**: se mueve en saltos
de una plaza (`1/target_size`).

Una propiedad que debe declararse: con una cartera concentrada (12 posiciones sobre ~250 valores)
todas las posiciones viven en la parte alta del ranking, así que sus alfas esperados —aunque
distintos, porque la recta se evalúa en el rank continuo— quedan muy próximos entre sí. En ese
régimen el efectivo responde sobre todo a la **salud reciente de la señal** (a qué ventana de la
cascada sobrevive y con qué pendiente), no a la dispersión transversal del día, y es casi binario:
la política solo se vuelve gradual con `target_size` 25 o 50.

La decisión de dejar efectivo se deriva **exclusivamente de la sección transversal** —del alfa
esperado de las candidatas— y nunca de una previsión sobre el mercado. Esa restricción no es
cosmética: derivarla de una vista de mercado convertiría el sistema en *market timing* encubierto y
la comparación contra el índice dejaría de ser limpia.

La política de efectivo es una **decisión de cartera, no de modelo**: no altera el Rank-IC, vive en
la etapa diagnóstica y se decide al final ejecutando ambas alternativas con el ganador predictivo ya
congelado. Cuál es mejor es un resultado del trabajo, no un supuesto previo.

#### Orden de decisión en cada snapshot

Toda la lógica vive en una única función, `decide_orders`
([module/evaluation/portfolio.py](../module/evaluation/portfolio.py)); la contabilidad (precios,
costes, turnover) está en [module/evaluation/backtest.py](../module/evaluation/backtest.py). **El
orden importa**: cada paso condiciona los siguientes, y una acción vendida no puede recomprarse en
ese mismo snapshot.

```text
0.     Preparación: ranking, percentiles, alfas, umbrales, protegidas por tenencia
1.     Venta forzada por pérdida de cobertura      (missing_current_score)
1-bis. Venta por suelo de cobertura                (below_coverage_percentile)
2.     Venta a efectivo por umbral                 (expected_alpha_below_exit)
--- a partir de aquí, todo bloqueado si price_only_sell_only y no hay fundamentales ---
3.     Compras nuevas con histéresis               (initial_fill)
4.     Relleno obligatorio hasta el suelo          (fully_invested_fill / cash_floor_fill)
5.     Rotación: outsider desplaza a la peor       (displaced_by_net_edge)
6.     Pesos, tolerancia de deriva y órdenes       (rebalance)
```

Los tres umbrales se calculan una vez por snapshot, **todos en base anual**:

```text
coste_ida_y_vuelta = anualizar(2 × (comisión + slippage), horizonte)
umbral_salida      = exit_expected_alpha_bps / dureza
umbral_entrada     = (exit_expected_alpha_bps + coste_ida_y_vuelta) × dureza
umbral_rotación    = coste_ida_y_vuelta + rotation_edge_bps × dureza
```

donde `dureza` vale 1,0 en snapshots con fundamentales nuevos y
`price_only_strictness_multiplier` en el resto.

El **paso 4** es la razón de que varias reglas «no hagan lo que parece» con tope 0: se completa hasta
el suelo de diversificación con las mejores por ranking **sin aplicar ningún umbral**, así que
vendas lo que vendas, el relleno vuelve a llenar la cartera en el mismo snapshot.

El **paso 5** se detiene —no continúa— en cuanto el mejor par disponible no supera el umbral:
`outsider` es la mejor candidata fuera y `peor` la peor desplazable, de modo que si ese par no lo
supera, ningún otro lo hará; además, continuar sería un bucle infinito porque ninguno de los dos
cambia. Si a alguno de los dos le falta calibración de alfa, la ventaja es indefinida y la rotación
se detiene: una rotación que no puede justificarse económicamente no se hace.

#### Casuísticas

- **Una posición pierde su puntuación** (sale del índice, deja de cotizar) → se vende en el paso 1,
  siempre, incluso bajo mínimo de tenencia, y no se recompra ese snapshot.
- **Todas las posiciones caen bajo el umbral y no hay nada mejor** → con tope 0 no se emite ninguna
  orden, porque vender la cartera entera para recomprarla igual sería una ida y vuelta completa para
  quedar en el mismo sitio; con tope > 0 se venden las peores a efectivo hasta el suelo.
- **Hay menos candidatas que plazas** → si el universo es escaso, el capital se reparte entre las
  plazas ocupadas y el tope decide cuánto se retiene, porque las plazas ausentes no existen. Si en
  cambio la compra está bloqueada por `price_only_sell_only`, el hueco **queda en efectivo y no se
  reparte**: concentrar sería actuar justo cuando se ha decidido no actuar sin información nueva.
- **Una posición protegida por el mínimo de tenencia** no puede venderse por caída de alfa, ni por
  rotación, ni por suelo de cobertura; sí por `missing_current_score`, y sí puede ajustarse su peso
  por rebalanceo, que no es una venta de la posición.
- **El alfa aún no está calibrado** (arranque del backtest) → manda la ordenación por ranking, ningún
  `NaN` dispara una venta ni bloquea una compra, y la rotación se detiene.
- **Los pesos no suman 1** → es legítimo y significa efectivo, por tope activo, por
  `price_only_sell_only` o por un universo menor que `target_size`.

#### El coste entra dos veces, y eso condiciona cómo se mide su efecto

Comisión y slippage no son solo una partida contable. Aparecen en dos sitios:

1. **En la contabilidad** ([module/evaluation/backtest.py](../module/evaluation/backtest.py)): el
   drag de cada snapshot es `Σ(notional × tasa) / valor`. Como `notional = |Δw| × valor` y el
   turnover es `Σ|Δw|`, resulta la identidad exacta **`drag = turnover × tasa`**.
2. **En las decisiones** ([module/evaluation/portfolio.py](../module/evaluation/portfolio.py)): el
   coste de ida y vuelta fija el umbral de entrada y el de rotación, que es el mecanismo por el que
   una operación debe pagarse a sí misma.

La consecuencia importa al interpretar cualquier análisis de costes: **simular con coste cero no
produce «la misma cartera sin comisiones»**, sino una cartera distinta, porque los umbrales se
desploman y se opera mucho más. Y al revés, una cartera que afronta costes altos opera menos y se
protege sola. Por eso el efecto del coste tiene dos lecturas legítimas —sobre la ruta de operaciones
congelada y sobre una cartera que vuelve a decidir— y la distancia entre ambas mide cuánto protege
esta doctrina de umbrales.

#### Qué aportó cada posición

El backtest emite, posición a posición y snapshot a snapshot, el retorno que efectivamente se aplicó
y lo que aportó a la cartera (`contributions.parquet`). Se emite **desde el motor** y no se
reconstruye después porque solo ahí se conocen las dos convenciones que lo determinan —la exclusión
de cotización y la neutralización de retornos imposibles—; recalcularlas fuera crearía una segunda
verdad que se desviaría en silencio.

Como los pesos invertidos y el efectivo suman uno por construcción, **la suma de contribuciones es
exactamente el retorno bruto del periodo**, sin aproximación. Esa identidad es un test de contrato, y
es lo que convierte la atribución por acción en contabilidad en vez de en una estimación.

#### De dónde sale la rotación

La causa raíz del turnover es estructural: se re-decide con la cadencia de snapshot sobre una señal
al horizonte del modelo, de modo que con cadencia mensual y horizonte anual se toman doce decisiones
sobre una etiqueta que solo se cierra una vez. Los motivos de orden persistidos permiten atribuir el
turnover a rotación, rebalanceo de pesos, compras iniciales y ventas a efectivo, y esa atribución es
la que dice qué palanca merece moverse.

`minimum_holding_period` y `rotation_edge_bps` actúan sobre la rotación económica;
`rebalance_drift_tolerance` sobre la cosmética. `snapshot_step_months` es la palanca estructural,
pero es **predictiva**: cambiarla re-ejecuta la selección entera y produce otro ganador, así que no
es un ajuste posterior de cartera. Las cifras concretas de cada palanca viven en
`portfolio_grid.parquet` y en las métricas del ganador, no en este documento.

#### Sizing

```text
ratio = 1 + clip((alfa_esperado − mínimo_cartera) / (máximo_cartera − mínimo_cartera), 0, 1)
```

La posición con menor alfa esperado recibe ratio 1 y la de mayor alfa esperado ratio 2. A diferencia
de escalar por percentil, el peso responde a una magnitud económica estimada y no a la posición
relativa en un ranking. Comisión y slippage se aplican al nocional realmente operado.

### Perfiles

`profiles.py` define ocho perfiles de inversor: `balanced`, `growth`, `value`, `quality`,
`momentum`, `contrarian`, `defensive` y `garp`. Dentro de las acciones «buenas» (`meta_rank` por
encima de `GOOD_THRESHOLD = 0.60`), cada perfil reordena por estilo y produce un `profile_score` que
la cartera consume igual que el `meta_rank`. Son **deterministas y no reentrenan nada**.

Para cada perfil se guardan equity, rentabilidad anual, benchmark, alfa, IR, drawdown, turnover,
posiciones y órdenes. La matriz principal coloca años en filas, perfiles en columnas y alfa contra
SPY en las celdas.

### Estadística y diagnósticos de señal

`stats.py` evalúa la significación del Rank-IC fuera de muestra con **bootstrap por bloques**
temporales (`block_bootstrap_ci()`, `paired_difference_ci()`, `DEFAULT_BLOCK_SIZE = 12` cohortes).
Por bloques y no clásico porque las cohortes se solapan: un bootstrap ingenuo daría intervalos
artificialmente estrechos.

`signal_diagnostics.py` publica diagnósticos PIT de la cola negociada y de la salud de la señal:
`rank_tail_diagnostics()` y `alpha_curve_points()`.

---

## 7. `module/studies`: el catálogo cerrado y la selección

### El catálogo

`studies/catalog.py` (`CATALOG_VERSION = 7`) contiene **todos** los parámetros científicos posibles.
La API rechaza claves desconocidas, valores libres y combinaciones incompatibles. Esta es la defensa
estructural contra el *p-hacking*: no se puede probar una configuración que el catálogo no declare,
así que el espacio de búsqueda está acotado y es auditable.

Constantes de gobierno:

- `SELECTION_ERAS = ((2015,2018), (2019,2021), (2022,2024))`
- `SELECTION_UNTIL_YEAR = 2024` — la única frontera entre la ventana que decide y la que confirma.
- `KNOWN_STRESS_YEARS = (2025, 2026)` — rol `known_stress_not_selection`; jamás entra en una
  decisión.
- `PREDICTIVE_STAGES = ("temporal", "representation", "model", "meta")`
- `STAGE_ORDER = (*PREDICTIVE_STAGES, "portfolio")`

`VARIABLES` son 32 `VariableSpec`, cada una con `id`, `label`, `description`, `stage`, `values`,
`recommended`, `invalidates` (`dataset` | `features` | `fit` | `meta` | `backtest`), `cost`,
`order`, `predictive`, `depends_on` y `simplicity`. La propiedad `modes` devuelve
`("fixed", "optimize")` para las predictivas y solo `("fixed",)` para las de cartera: **una variable
de cartera no puede barrerse en un Model Study porque no cambia el Rank-IC**, y por tanto no puede
elegir modelo.

Reparto por etapa: 6 temporales, 7 de representación, 5 de modelo, 3 de meta y 11 de cartera
(`predictive=False`).

### El baseline `recommended`

El `recommended` de cada variable predictiva es el punto de partida de la optimización secuencial:
la primera evaluación (`predictive:baseline`) y el incumbent sobre el que se prueba la primera
variable optimizable. Se revisó variable por variable si el valor seguía siendo el más defendible
**por lógica de dominio únicamente**, sin usar ningún resultado empírico. Los 21 baselines vigentes
se mantienen. Todos comparten dos criterios rectores:

1. **Simplicidad/regularización como default**: entre valores equivalentes, se prefiere el más
   simple o más conservador; la complejidad (más árboles, más profundidad, más features, pesos
   más libres) es la hipótesis que debe demostrar que mejora el Rank-IC, no algo que se asuma de
   entrada.
2. **PIT y causalidad estrictos por defecto**: donde hay ambigüedad, se prefiere el valor más
   prudente frente a lookahead o sobreajuste (lag de 60 días, poda por estabilidad OOS, límites
   acotados en el meta-agente).

| Variable | `recommended` | Razón (por qué es el punto de partida, no una elección empírica) |
|---|---|---|
| `snapshot_step_months` | 3 | Coherente con la cadencia real de los fundamentales (informes trimestrales): observar más a menudo que la fuente de datos añade ruido de precio, no información nueva. |
| `target_horizon_months` | 12 | Horizonte largo da señal fundamental más estable y menos ruido de corto plazo, coherente con agentes fundamentales (quality/value/growth). |
| `train_lookback_years` | 8 | Cubre más de un ciclo de mercado sin diluir la relación factor-retorno con datos de un régimen demasiado distinto (pre-2008). |
| `execution_lag_days` | 60 | El más conservador de {30,45,60}; PIT estricto es la prioridad del proyecto. |
| `recency_weighting` | off | Neutral: ponderar recencia es una hipótesis adicional que debe ganarse compitiendo, no asumirse. |
| `objective` | rank_regression | Señal continua (magnitud), más informativa como punto de partida que el ranking puro. |
| `feature_preset` | core | Mínimo suficiente por diseño; `all` es la hipótesis de que más profundidad de información ayuda, no el default. |
| `fundamental_momentum` | True | Cambios PIT de fundamentales son información causal legítima y barata de incluir. |
| `market_regime_feature` | True | Contexto de mercado causal y disponible sin lookahead; mismo criterio que el anterior. |
| `neutralize_by_sector` | False | Punto de partida menos restrictivo: deja toda la señal disponible, incluida la sectorial; neutralizar es una hipótesis de diseño adicional a demostrar. |
| `winsorization` | 0.0 | Sin transformación adicional sobre los datos crudos; el recorte de colas es una hipótesis de robustez que debe ganarse con Rank-IC. |
| `max_features_per_agent` | 8 | El más bajo de {8,12,20}; menos parámetros como punto de partida frente a sobreajuste. |
| `feature_weighting_mode` | oos_stability_prune | Más conservador frente a overfitting que la selección nativa del modelo; alineado con generalizar fuera de muestra. |
| `model_family` | lightgbm | Captura no linealidades e interacciones sin especificarlas a mano; Elastic Net es la alternativa lineal más simple a probar. |
| `lgbm_max_depth` | 3 | El más bajo de {3,4,6}; árboles poco profundos regularizan como punto de partida. |
| `lgbm_n_estimators` | 100 | El más bajo de {100,200,400}; mismo criterio de regularización. |
| `lgbm_learning_rate` | 0.05 | Valor medio de {0.03,0.05,0.10}: ni underfitting por lentitud con pocos estimadores, ni inestabilidad por agresividad. |
| `lgbm_min_child_samples` | 50 | Valor medio de {20,50,100}: evita hojas con pocas observaciones sin perder señal por exceso de restricción. |
| `meta_method` | stacked_rolling_bounded | Ya incorpora una salvaguarda de diseño (límites 10–50 %) que evita que el meta colapse en un solo agente; usa solo cohortes ya cerradas (causal). `equal` ignora por completo si un agente es sistemáticamente mejor, que es también una asunción fuerte, no un default neutral. |
| `meta_history_quarters` | 16 | Ventana más larga (4 años) da un ajuste del stacker más estable que 8 trimestres. |
| `meta_recency_weighting` | off | Neutral, mismo criterio que `recency_weighting`. |

### Optimización secuencial

El usuario no manipula un selector de modo separado. Marca directamente valores del catálogo:

- un único valor implica `fixed`;
- dos o más valores de una variable predictiva implican `optimize`;
- dos o más valores de cartera implican una comparación `diagnostic`.

La configuración recomendada ya llega marcada. Si se añade un segundo valor, el presupuesto se
recalcula inmediatamente; si se vuelve a dejar uno, la variable vuelve a ser fija. La API persiste
el modo derivado para que la ejecución sea explícita y auditable.

El proceso es **greedy secuencial, no cartesiano**:

1. Ejecutar baseline.
2. Tomar la primera variable optimizable.
3. Evaluar todos sus valores sobre el incumbent acumulado.
4. Elegir mediante Rank-IC robusto.
5. Fijar el ganador.
6. Continuar con la siguiente variable.

Por tanto:

```text
evaluaciones predictivas = 1 baseline + suma de alternativas por variable optimizada
```

El orden es temporal, representación, modelo y meta. No es reordenable porque cambiarlo alteraría la
trayectoria científica.

Las fases que ejecuta `runner.py::execute_model_study()`:

1. **Baseline** — un run `predictive:baseline`, evidencia en `evidence_baseline/`, incumbent inicial.
2. **Optimización secuencial** — escribe `evaluation_ledger.parquet` y `decisions.json`.
3. **Ganador** — run `winner:evidence` en `evidence/`, más `winner.json` con
   `selection_metric = "rank_ic_only"`.
4. **Diagnósticos post-ganador** (desactivables) — carteras diagnósticas, ocho perfiles, robustez
   (bootstrap, eras, estabilidad de pesos, 2 semillas extra, 5 placebos de etiqueta) y atribución.
5. **Cierre** — `report.md`, `storage_manifest.json`, estudio `succeeded`.

### La regla de selección

Las comparaciones son pareadas por cohorte y solo usan datos hasta 2024.

Las cohortes se emparejan por **periodo mensual**, no por cadena de fecha: los snapshots se
sitúan en `fin_de_mes + execution_lag_days`, de modo que barrer el lag desplaza toda la rejilla y el
emparejamiento literal daría cero fechas comunes.

Elegibilidad:

- observaciones suficientes;
- ninguna era disponible con Rank-IC inferior a −0,02;
- y **una de estas dos** puertas pareadas:
  - **dominancia**: diferencia media de Rank-IC positiva **y** mejor en más de la mitad de las
    cohortes emparejadas;
  - **no inferioridad**: límite inferior del bootstrap pareado al 90 % superior a −0,01.

La puerta doble corrige un defecto de diseño de la versión anterior, que solo exigía no
inferioridad. Aplicada a un candidato *superior*, esa prueba lo penalizaba: cuanto más se
diferenciaba del incumbent, más ancho era su intervalo y más fácil era que el límite inferior cayera
por debajo del margen. La regla premiaba estructuralmente al incumbent y llegó a descartar al mejor
candidato disponible por 0,00023 en un bootstrap de 1 000 extracciones (ahora 2 000).

Cuando las dos series no comparten al menos un bloque completo de fechas, el resultado pareado se
marca **no aplicable** y el candidato no es elegible. Antes ese caso devolvía `ci_low = 0,0` en
silencio, lo que satisfacía cualquier prueba de no inferioridad y dejaba pasar automáticamente a
todos los candidatos de las variables que desplazan la rejilla.

Orden entre elegibles:

1. Mayor **ventaja pareada** de Rank-IC contra el incumbent. La diferencia se mide cohorte a cohorte,
   lo que elimina el factor común de mercado de cada fecha.
2. Diferencias inferiores a 0,002 son empate y decide la simplicidad.
3. Mayor Rank-IC medio.
4. Mayor fracción positiva.
5. Menor variabilidad.
6. Menor complejidad.
7. Conservar incumbent.

El spread de cola, el alfa y el IR se muestran como diagnósticos y **no alteran esta decisión**. La
corrección por multiplicidad de las cifras finales se hace con el Deflated Sharpe Ratio en
`attribution.json`; el proyecto no usa Holm.

### Estudios encadenados

La optimización es greedy secuencial, y eso tiene una consecuencia conocida: **el resultado depende
del punto de partida**, porque cada variable se evalúa sobre el incumbent acumulado hasta ese momento
y no sobre todas las combinaciones posibles. Una variable que se decidió pronto, cuando el resto de
la configuración todavía era la recomendada por defecto, nunca vuelve a revisarse con lo demás ya
optimizado.

Encadenar studies ataca exactamente esa limitación: **el ganador del study *n* es el baseline del
study *n+1***, manteniendo en `values` el abanico que se quiera reexplorar. Cada pasada completa es
una iteración de **ascenso por coordenadas** (*coordinate ascent*), y la mejora entre pasadas es la
evidencia de que el procedimiento converge.

No hay que programar nada para encadenar. `normalized_definition`
([module/studies/config.py](../module/studies/config.py)) admite por variable
`{"mode", "values", "baseline"}` y exige que el `baseline` sea uno de los `values` seleccionados;
`initial_values` siembra con él la primera evaluación. Cada study crea siempre un `study_id` nuevo:
no existe «reanudar una cadena», y son directorios independientes en `results/studies/`, que es justo
lo que permite compararlos.

**Cómo se demuestra que la cadena mejora.**

1. **La métrica de comparación es el Rank-IC robusto de la ventana de selección**, que es el criterio
   con el que se eligió cada ganador. Es lo único comparable entre pasadas.
2. **2025–2026 no participa en ninguna comparación**, en ninguna pasada.
3. Las métricas económicas se reportan pero **no** son el criterio: la cadena optimiza capacidad
   predictiva, no rentabilidad.

Una cadena puede **converger sin mejorar**, y eso también es un resultado publicable: si una pasada
devuelve el mismo ganador que la anterior, significa que el óptimo greedy es estable frente al punto
de partida, y debe contarse como tal en vez de disimularse.

**El riesgo que hay que declarar.** Encadenar **multiplica el número de configuraciones probadas
sobre los mismos datos**, y eso agrava la selección múltiple: el Deflated Sharpe penaliza vía
`n_trials`, que crece con cada pasada. La ganancia de Rank-IC entre pasadas y el riesgo de
sobreajuste por multiplicidad **crecen a la vez**, y la era reservada es la única defensa real,
precisamente porque no participa en ninguna pasada.

**Empates técnicos y versión de catálogo.** Cuando la ventaja pareada de un retador no supera
`TIE_TOLERANCE`, la decisión la resuelve la tabla de simplicidad del catálogo, no la evidencia. Esas
decisiones **no deben presentarse como hallazgos empíricos**: son convenciones de desempate
declaradas de antemano. Cambiar la tabla de simplicidad de una variable —como ocurrió al pasar de
catálogo v6 a v7, invirtiendo el orden de `execution_lag_days` a `(60, 45, 30)`— **no altera ninguna
medición**, pero **sí puede invertir una decisión** tomada por empate y, con ella, el baseline de
todas las pasadas siguientes. La regla operativa: **todas las pasadas de una cadena corren bajo la
misma versión de catálogo**, y la versión se declara junto al `study_id`.

**Los diagnósticos posteriores al ganador son opcionales por pasada.** En una cadena, las pasadas
intermedias existen **solo para elegir configuración**: su ganador es el punto de partida de la
siguiente y su cartera no se publica. Por eso el lanzamiento admite un interruptor,
**`post_winner_diagnostics`, activado por defecto**, que al desactivarse termina el Study en cuanto
escribe `winner.json`. No toca la selección: el recorrido predictivo, los candidatos y la regla de
decisión son idénticos con y sin él.

La razón de fondo no es el coste, sino la era reservada. `attribution.json` contiene la confirmación
2025–2026, que por doctrina se evalúa **exactamente una vez** y se publica sea cual sea el resultado.
Ejecutarla en una pasada intermedia la gastaría sobre una configuración destinada al descarte, y
repetirla en cada pasada convertiría la única defensa contra la multiplicidad en otra variable más
sobre la que se ha mirado muchas veces. La regla operativa: **la confirmación fuera de muestra se
ejecuta en la última pasada de la cadena**, no en las intermedias.

### El Portfolio Study

El Model Study optimiza Rank-IC, que mide la calidad de la *ordenación*. Ordenar bien y ganar dinero
no son lo mismo: se ha observado un ganador con el mejor Rank-IC de su cadena y a la vez el peor
coeficiente de transferencia. Optimizar la ordenación no optimiza la cartera, así que **la cartera
necesita su propio criterio**, y ese criterio es el **Information Ratio**: exceso medio sobre el
índice dividido por la volatilidad de ese exceso. A diferencia del alfa bruto, premia la
consistencia.

Se ejecuta **sobre el ganador ya congelado, sin reentrenar nada**: cada combinación reutiliza los
scores del ganador y solo rehace el backtest, lo que la abarata en dos órdenes de magnitud frente a
un run predictivo con ajuste.

**Cartesiano, no greedy.** Las seis variables de cartera —`target_size`, `max_cash_weight`,
`sizing_mode`, `minimum_holding_period`, `coverage_percentile_floor`,
`rebalance_drift_tolerance`— **interactúan**: el suelo de diversificación se deriva de `target_size`
y `max_cash_weight` a la vez, y lo que hace `coverage_percentile_floor` depende de si la plaza que
libera se recompra (tope 0) o queda en efectivo (tope > 0) — la misma variable hace cosas opuestas
según el tope. Un greedy fijaría la primera antes de mirar la segunda y no vería nada de eso.

**Qué no se optimiza.** `commission_bps` y `slippage_bps` se fijan a un único valor: son *supuestos
de coste*, no decisiones de gestión, y optimizarlos equivaldría a elegir el mundo en el que la
estrategia luce mejor. La validación lo impone; no es una convención de la interfaz.

**Los perfiles quedan fuera de la rejilla.** Un perfil *reordena la señal*, mientras las seis
variables solo gestionan la cartera ya elegida: son planos distintos. Incluirlos multiplicaría el
coste por ocho y, sobre todo, elegiría el estilo de inversor por su rentabilidad conocida. Al
terminar la rejilla, la cartera ganadora se aplica a todos los perfiles para responder «cómo le
habría ido a cada estilo con la mejor gestión»; ninguno se elige por su IR.

**Cómo se aísla la era reservada.** Durante la rejilla el backtest se corta en 2024 recortando los
scores antes de simular (`selection_evidence`), de modo que 2025–2026 **no llega a calcularse** para
ninguna combinación. No basta con filtrar el resumen al elegir: la cartera es secuencial, y si la
simulación entrase en la era reservada su resultado existiría y bastaría con mirarlo. Solo la
combinación ya ganadora se reevalúa sobre la serie completa, y esa evidencia se guarda aparte. La
cartera de partida contra la que se mide la mejora usa la **misma** serie recortada: compararla
sobre la completa mediría ventanas distintas y la mejora sería ficticia.

**Qué se guarda.** Una fila de resumen por combinación en `portfolio_grid.parquet`; la evidencia
completa es solo la del mejor vigente, que se sustituye en cuanto otra la supera. La rejilla vuelca
cada 25 combinaciones y al arrancar salta las ya evaluadas.

**El riesgo que hay que declarar**: probar una rejilla entera sobre los mismos datos añade
multiplicidad, igual que encadenar pasadas. La defensa es que la elección solo ve la ventana de
selección y que el resultado de la era reservada se reporta **junto** al de selección, nunca en su
lugar. Y una consecuencia que no se puede suavizar: la cartera adoptada es la mejor de la rejilla, de
modo que sus cifras **dentro** de la ventana de selección son una cota superior optimista, no una
estimación insesgada.

**La evidencia del ganador es completa, y no se recalcula.** El Portfolio Study no reentrena nada,
así que los artefactos de modelo de su ganador —scores, diagnósticos de Rank-IC, pesos del
meta-agente, atribución de features— **son los mismos ficheros** del Model Study de origen. Se
enlazan a `evidence_best_full/` en lugar de recalcularse: reejecutar el ganador costaría un
entrenamiento completo para producir una copia que, ante cualquier no-determinismo, podría no
coincidir con la evidencia que sí alimentó la decisión.

Las vistas que un Portfolio Study no puede producir por sí mismo —robustez y atribución— se sirven
desde el Model Study de origen declarando la procedencia. Es honesto por el mismo motivo: esa
robustez y esa atribución son, exactamente, las del modelo cuyos scores reutiliza.

---

## 8. `module/research`: diagnósticos posteriores

Todo lo de este paquete es **posterior a la congelación del ganador y no puede moverlo**. Se calcula
solo, de modo que cuando la cadena termina las cifras del capítulo económico ya están completas.
Ninguno escribe en `winner.json`, `decisions.json` ni `portfolio_winner.json`, y si uno falla deja
constancia en su artefacto y **no tumba el estudio**: la rejilla ya ha costado horas.

### `robustness.py`

`bootstrap_and_eras()` produce la batería posterior: semillas 7 y 2026, bootstrap móvil de 12
snapshots, intervalos al 90 % y 95 %, exclusión de eras, permutación transversal add-one, etiquetas
barajadas, Rank-IC por agente, estabilidad de pesos meta y carteras aleatorias PIT generales y
emparejadas por riesgo.

**Nulo de carteras aleatorias.** Las carteras aleatorias juegan con las mismas reglas que el modelo:
exigen cobertura del año completo antes de computar un retorno anual, aplican la misma guarda contra
artefactos de datos y **pagan las mismas comisiones y slippage**. Sin esas tres condiciones el nulo
producía un percentil 95 de CAGR del 107 % anual —imposible para una cartera del S&P 500— y el único
contraste que el modelo aparentemente suspendía era en realidad un fallo del contraste.

**Estabilidad ante la semilla.** Cada agente entrena `SEED_ENSEMBLE = 5` réplicas que solo difieren
en la semilla y promedia sus scores. No es una variable científica a optimizar sino reducción de
varianza del estimador: el Rank-IC apenas dependía de la semilla (±0,001) pero el alfa de una cartera
concentrada llegaba a cambiar de signo, porque doce posiciones amplifican el ruido de inicialización
de LightGBM. El barrido de semillas se conserva como diagnóstico y `robustness.json` publica el rango
mínimo-mediana-máximo de alfa; si ese rango cruza cero, la conclusión económica no es estable y hay
que decirlo.

No produce una etiqueta automática de «aprende/no aprende». El informe debe discutir evidencia a
favor, en contra, contradicciones y limitaciones.

### `attribution.py`

Separa «el sistema aprende» de «el sistema redescubrió un factor conocido»:

1. **Regresión de factores.** El exceso de la cartera se regresa sobre carteras réplica de valor,
   momentum, baja volatilidad, calidad y crecimiento, construidas del propio universo como
   diferencial tercil superior menos tercil inferior y rebalanceadas en cada snapshot. El alfa es el
   intercepto y su significación se evalúa con errores **Newey-West** (12 retardos), porque los
   retornos solapados inflarían la significación con errores clásicos. No hay acceso a las series de
   Fama-French; tamaño e inversión se declaran **no replicables** con este panel en lugar de
   sustituirse por un sucedáneo no auditable.
2. **Rank-IC neutralizado** por las mismas características de estilo.
3. **Deflated Sharpe Ratio** (Bailey y López de Prado) sobre las series de IC candidatas: con
   decenas de evaluaciones, el mejor resultado es alto aunque ninguna configuración tenga capacidad
   real, y un p-valor sin corregir por multiplicidad no es defendible.
4. **Baselines deterministas** (GARP, momentum puro, calidad, valor): si el sistema no los supera, el
   aparato de aprendizaje no está justificado.
5. **Coeficiente de transferencia** y curva de alfa neto frente a rotación.

**Confirmación 2025–2026.** La era reservada no participó en ninguna decisión, así que su Rank-IC es
la única medida predictiva del trabajo libre de sesgo de selección. Se calcula **una sola vez, sobre
el ganador ya congelado**, y se publica salga lo que salga. Con horizonte de 12 meses y cadencia
mensual las cohortes contiguas comparten casi toda la etiqueta, de modo que el número de cohortes
**no** es el número de pruebas independientes: se reporta también el número efectivo de observaciones
independientes y se etiqueta como evidencia direccional del signo, no como contraste con potencia.

El sesgo que esto corrige es de **selección, no de lookahead**: el entrenamiento es walk-forward con
purga y ninguna predicción individual usa información futura, pero la *configuración* que las produce
se eligió por haber quedado mejor sobre esa misma serie 2015–2024.

### `cost_sensitivity.py`

Responde hasta dónde aguanta el supuesto de costes constantes. Publica el escenario bruto (coste
cero, cota que nadie realiza), el estándar y el **equilibrio** `c*`: el coste por operación al que el
exceso geométrico **contra el índice** se anula. Se define contra el índice y no contra rentabilidad
absoluta porque la alternativa real de un inversor es comprar el índice, no quedarse en efectivo.

Se calculan **dos familias**, por el motivo que explica «El coste entra dos veces»:

- *Ruta congelada*: mismas decisiones, distinto coste. Como `drag = turnover × tasa` exactamente, la
  curva entera sale en forma cerrada desde `equity.parquet`, con cero cómputo. Su `c*` es
  **conservador**, porque un gestor que pagase más operaría menos.
- *Resimulada*: la cartera vuelve a decidir con cada coste, ya que el coste alimenta
  `entry_threshold` y `rotation_threshold`. Su `c**` es mayor o igual, y **la diferencia entre ambas
  es en sí misma un resultado**: mide cuánto protege la doctrina de umbrales económicos.

La escalera de costes es una constante de diagnóstico, no una variable del catálogo: no se optimiza.

### `capacity.py`

A partir de qué patrimonio la cartera deja de ser ejecutable. Se mide **participación** —qué fracción
del volumen negociado habitual representaría cada orden a un patrimonio dado— y no impacto de
mercado, que exigiría supuestos que este panel no puede sostener: la participación es observable, el
impacto sería una hipótesis. Se publican dos umbrales, 5 % y 10 % del volumen diario, porque no hay
uno canónico. Una orden sobre un ticker sin volumen medido se cuenta como cobertura incompleta,
**nunca** como ejecutable: un hueco del panel no puede subir la capacidad estimada.

### `portfolio_narrative.py`

Qué tuvo la cartera, cuánto tiempo, cuánto aportó cada nombre y en qué se equivocó: los más
presentes, la contribución bruta y neta por acción, las mejores y peores operaciones **cerradas** (un
recorte de rebalanceo no cuenta: la posición siguió abierta), las ventas que luego subieron —el coste
de oportunidad de salir—, la permanencia, la concentración y la exposición sectorial.

Todo el bloque es descriptivo. Mirar las peores decisiones para cambiar la estrategia sería ajustar
sobre el resultado conocido, y por eso el propio artefacto lo advierte.

---

## 9. `module/storage`: datasets, caché y evidencia

- **`datasets.py`** — una materialización de datos PIT y features por identidad científica.
  `dataset_identity()`, `ensure_prepared()`, `validate_dataset_reference()`. Direccionado por hash y
  compartido entre estudios.
- **`cache.py`** — caché content-addressable de fits y resúmenes. `canonical_json()`,
  `enforce_cache_limit()`, bloqueos con detección de bloqueo obsoleto vía `pid_alive`. Las entradas
  referidas por la evidencia de un estudio quedan protegidas de la limpieza (`pinned_keys()`).
- **`studies.py`** — persistencia auditable. `RESULTS_ROOT`, `STUDIES_ROOT`, `SCHEMA_VERSION = 2`,
  más `create_study`, `create_run`, `update_run`, `list_studies`, `append_event`,
  `safe_study_path()` (guarda contra path traversal) y `write_storage_manifest()`.
- **`evidence.py`** — escribe la evidencia final: `write_report()`, `write_winner()`,
  `write_portfolio_report()`.

**Los datasets preparados nunca se copian dentro de `results/`.** La evidencia guarda un
`dataset_reference.json` que apunta a `data/prepared/<hash>/` y se valida al leerlo.

---

## 10. `module/web` y `app`: API y dashboard

### Ejecución y recuperación

La API crea `study.json` y el run baseline **antes** de lanzar el worker, de modo que un fallo
temprano deja rastro. El proceso hijo actualiza un heartbeat y persiste cada run. Los eventos se
escriben simultáneamente en terminal, `events.jsonl` y la Consola del dashboard.

Estados: `queued`, `running`, `succeeded`, `failed`, `cancelled` e `interrupted`. Al arrancar, el
servidor marca como `interrupted` cualquier Study activo cuyo PID haya desaparecido (`reconcile()`).
Reanudar conserva runs finalizados y reinicia solo los incompletos. **Un artefacto parcial nunca es
caché válida**: la caché solo se publica de forma completa.

`jobs.py::StudyWorkers` lanza `python -m module.studies.worker <id> <ppid>` con
`CREATE_NEW_PROCESS_GROUP` en Windows, y `stop_all()` queda registrado con `atexit` para que cerrar
el servidor no deje workers huérfanos.

### La API

`api.py` usa `ThreadingHTTPServer` de la biblioteca estándar: **sin Flask ni FastAPI**. Sirve además
`app/` como estático, con guarda de traversal y fallback SPA. Convierte `NaN`/`Inf` a `null`
(`_json_safe`) porque `JSON.parse` los rechaza.

- `GET /api/catalog`
- `POST /api/studies/preflight`
- `POST /api/studies`
- `GET /api/studies`
- `GET /api/studies/{id}`
- `POST /api/studies/{id}/cancel` | `/pause` | `/resume`
- `GET /api/studies/{id}/runs/{run_id}`
- `GET /api/studies/{id}/events`
- `GET /api/studies/{id}/analysis/{view}`
- `POST /api/portfolio-studies/preflight` y `POST /api/portfolio-studies`

`queries.py` concentra todas las lecturas del dashboard y declara `ANALYSIS_VIEWS`. `INHERITED_VIEWS`
permite que un Portfolio Study sirva la robustez y la atribución de su Model Study de origen,
declarando la procedencia.

### El dashboard

`app/` es HTML, CSS y un único fichero JavaScript, **sin build ni dependencias**. `node` solo se usa
para `node --check`.

**El frontend nunca calcula métricas ni lee Parquets.** Las métricas que representan tasas, retornos,
pesos, drawdowns, turnover, alfa o Rank-IC se presentan en porcentaje, aunque los artefactos
mantengan su representación decimal para el cálculo. Las vistas analíticas priorizan gráficos SVG con
ejes, escalas y leyendas; las tablas permanecen como respaldo auditable.

El baseline (`predictive:baseline`) retiene su evidencia completa en `evidence_baseline/`, con el
mismo contenido que la del ganador, para poder compararlos directamente. No genera perfiles ni
participa en robustez: esos diagnósticos siguen siendo exclusivos del ganador.

Junto al botón de lanzamiento existe un conmutador, «Ruta científica completa», que hace que **todos**
los runs retengan su evidencia entera en `runs_evidence/<clave_lógica>/`. La restricción que levanta
es **de almacenamiento, no metodológica**: el plan experimental es idéntico con el conmutador activo
o inactivo, no cambia el número de evaluaciones ni el criterio de selección, y 2025-2026 permanece
fuera de toda decisión. Lo que cambia es el coste, que el preflight declara por adelantado. Es la
opción adecuada para auditar por qué un candidato descartado quedó por detrás del ganador, y
desaconsejable como modo por defecto en ejecuciones largas.

`dev.py` sirve fixtures estáticos en `/dev` y **no ejecuta ciencia**.

---

## 11. Tests

`tests/` contiene tests de **contrato**, no unitarios de detalles internos: verifican las identidades
que el sistema promete (que la suma de contribuciones es el retorno bruto, que el catálogo es
coherente, que la cartera no rompe sus propios suelos, que la caché no confunde identidades). No
requieren datos reales ni credenciales.

```powershell
python -m pytest -q
```
