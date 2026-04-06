# DOCUMENTACION TECNICA EXHAUSTIVA

## 1) Objetivo del proyecto

Este repositorio implementa un sistema de stock picking long-only basado en un ensamblado multi-agente de ML para universo US equities (principalmente large/mid caps). El objetivo academico (TFM) es demostrar un pipeline:

- Reproducible.
- Defendible metodologicamente.
- Estricto contra leakage temporal.
- Trazable en cada decision (datos, features, scores, seleccion, performance).
- Comparable frente a benchmark y baselines bajo reglas operativas homogeneas.

El sistema no se limita a reportar accuracy de clasificacion. Tambien mide impacto economico con una simulacion monetaria en USD por folds walk-forward.

---

## 2) Mapa de arquitectura

Pipeline principal:

1. `step_01_data`: descarga y consolidacion de datos brutos.
2. `step_02_dataset`: construccion del panel maestro trimestral (PIT-aware).
3. `step_03_training`: entrenamiento agentes base + meta-learner.
4. `step_04_evaluation`: walk-forward, seleccion de cartera, backtesting y reportes.
5. `step_05_live`: fold live out-of-sample (opcional).

Archivo orquestador:

- `analyzer.py`.

Configuracion global:

- `environment.py`.

### 2.1. Idea de flujo

El pipeline no se ejecuta como una unica caja negra. En realidad hace una cadena de decisiones donde cada paso depende del anterior:

1. Primero se descargan y normalizan los datos brutos.
2. Despues se transforma la informacion en snapshots por quarter.
3. Luego se entrena un conjunto de agentes especializados.
4. A continuacion se decide una cartera por fold y se simula economicamente.
5. Finalmente se consolidan metricas, benchmark, baselines, auditorias y plots.

Esta secuencia importa porque cada etapa define el contexto temporal de la siguiente. Si se modificara un paso intermedio, el resto del pipeline puede cambiar de forma importante. Por eso esta documentacion insiste tanto en el orden real de ejecucion.

---

## 3) Objetivos de diseno y principios

El codigo fue estructurado con estos principios:

- Point-in-time first: cada feature debe ser construible con informacion disponible en la fecha de decision del fold.
- Separacion de responsabilidades: descarga, feature engineering, training, evaluacion y reporting estan desacoplados.
- Evaluacion fuera de muestra: cada fold entrena en historia previa y decide en quarter/frecuencia objetivo.
- Comparabilidad justa: estrategia, benchmark y baselines se simulan con mismo motor USD (fees/slippage/calendario).
- Auditoria exportable: artefactos por fold y consolidados para soporte de defensa academica.

### 3.1. Que significa esto en la practica

Estos principios no son solo declarativos. Se traducen en decisiones concretas:

- Si un dato no estaba publicado en la fecha de decision, no debe usarse.
- Si dos estrategias se comparan, ambas deben sufrir la misma logica de entrada, salida y costes.
- Si un resultado parece bueno, debe poder reconstruirse desde los artefactos.
- Si un fold no tiene datos suficientes, se omite en lugar de forzarlo artificialmente.

### 3.2. Implicaciones operativas

Estos principios delimitan el funcionamiento del sistema. En la practica significan que:

- no se usa informacion posterior a la fecha de decision,
- la comparacion entre estrategias se hace con el mismo protocolo,
- los resultados deben poder reconstruirse con los artefactos exportados,
- y los folds insuficientes se omiten para no falsear la evaluacion.

---

## 4) Configuracion central (`environment.py`)

`environment.py` es la fuente unica de verdad para parametros operativos, financieros y de modelado.

Lo importante de este archivo es que no solo guarda constantes: define el comportamiento del experimento completo. Cambiar un valor aqui puede alterar el universo, el horizonte temporal, la frecuencia de folds, el coste de ejecucion o la comparacion contra baselines.

Bloques relevantes:

- Flags de ejecucion:
  - `SKIP_BACKTEST`
  - `UPDATE_PRICES_ONLY`
  - `FORCE_DOWNLOAD`
  - `RETRY_MISSING_TICKERS`
- Ventana temporal de analisis:
  - `ANALYSIS_START_YEAR`, `ANALYSIS_START_QUARTER`
  - `ANALYSIS_END_YEAR`, `ANALYSIS_END_QUARTER`
  - `ANALYSIS_FREQUENCY` (`quarterly` o `annual`)
  - `ANALYSIS_ANNUAL_START_DATE`
- Punto temporal de decision:
  - `SNAPSHOT_LAG_DAYS`
  - `HOLDING_PERIOD_MONTHS`
- Robustez de cobertura:
  - `ENABLE_FALLBACK_EXTRAPOLATION`
  - `FALLBACK_LOOK_BACK_QUARTERS`
  - `MIN_TEST_TICKERS_PERCENT`
- Backtest monetario:
  - `INITIAL_CAPITAL_USD`
  - `TRANSACTION_FEE_USD`
  - `SLIPPAGE_PCT`
  - `USE_DOLLAR_BACKTEST`
  - `ALLOW_FRACTIONAL_SHARES`
- Comparativas:
  - `RUN_BASELINES`
  - `N_RANDOM_BASELINE_SIMS`
  - `BASELINE_MOMENTUM_LOOKBACK_DAYS`
- Trazabilidad:
  - `EXPORT_RUN_ARTIFACTS`

Nota operativa actual: en el estado mas reciente del repo, `ANALYSIS_FREQUENCY` esta en modo anual.

### 4.1. Como leer estos parametros

Conviene pensar en cinco capas:

1. Flags de ejecucion: deciden que partes del pipeline se activan.
2. Tiempo: deciden que rango y que frecuencia se analizan.
3. Calidad de datos: deciden cuantos datos son necesarios para aceptar un fold.
4. Economia: deciden como se simula la cartera.
5. Reproducibilidad: deciden como documentar el run.

### 4.2. Que cambia si modificas los parametros mas sensibles

- `SNAPSHOT_LAG_DAYS`: cambia la fecha efectiva de decision; si lo subes, eres mas conservador y usas mas retraso temporal.
- `HOLDING_PERIOD_MONTHS`: cambia el horizonte del label y del retorno economico.
- `TOP_N_STOCKS`: cambia el tamano de cartera y por tanto la diversificacion.
- `INITIAL_CAPITAL_USD`: cambia la escala monetaria, no la logica de seleccion.
- `TRANSACTION_FEE_USD` y `SLIPPAGE_PCT`: cambian la penalizacion por operar.

### 4.3. Por que este archivo importa tanto

`environment.py` concentra la definicion operativa del experimento. Eso significa que el comportamiento del sistema puede cambiar de forma relevante si cambian sus parametros, pero la logica general del pipeline permanece estable.

---

## 5) Flujo end-to-end (`analyzer.py`)

`analyzer.py` coordina el run de extremo a extremo:

### 5.0. Vista general del paso

- Entrada: configuracion global, universo de tickers, fechas de analisis y flags de ejecucion.
- Proceso: secuencia de descarga, consolidacion, construccion del dataset, exportacion de trazabilidad y evaluacion.
- Salida: artefactos del run, logs, dataset maestro y, si `SKIP_BACKTEST=False`, resultados de backtest y comparativas.

1. Inicializa logging y semillas globales (`random`, `numpy`).
2. Resuelve rango de analisis segun frecuencia trimestral o anual.
3. Descarga/actualiza datos (`download_data`).
4. Consolida y limpia (`prepare_data`).
5. Determina tickers utilizables (`get_available_tickers`).
6. Construye dataset maestro (`build_master_dataset`).
7. Exporta trazabilidad:
   - `results/run_config.json`
   - `results/data_quality_report.csv`
8. Ejecuta evaluacion walk-forward (`run_walkforward_pipeline`) salvo `SKIP_BACKTEST=True`.

Si `UPDATE_PRICES_ONLY=True`, el flujo termina antes de dataset/training/backtest y conserva compatibilidad con modo mantenimiento de datos.

### 5.1. Que hace exactamente al empezar

El arranque del orquestador es importante porque fija dos cosas antes de tocar los datos:

- el contexto temporal del experimento,
- y la semilla global de reproducibilidad.

Esto significa que, aunque el pipeline tenga componentes estocasticos, el resultado deberia ser replicable si se ejecuta bajo la misma configuracion.

### 5.2. Paso 1: descarga

En esta fase se recuperan datos brutos desde Finnhub y fuentes auxiliares. La idea es tener el repositorio localmente materializado antes de cualquier transformacion.

Que conviene entender aqui:

- no todo ticker descarga igual de bien,
- la descarga puede ser parcial,
- y el pipeline no debe asumir que todas las fuentes existen para todos los activos.

### 5.3. Paso 2: consolidacion

La consolidacion convierte archivos sueltos en estructuras uniformes por ticker. Es el puente entre datos externos y features modelables.

En esta etapa suele hacerse el trabajo sucio:

- homogeneizar nombres,
- alinear fechas,
- unir tablas de distintas fuentes,
- y filtrar objetos inutilizables.

### 5.4. Paso 3: filtrado de tickers

`get_available_tickers` separa universo solicitado de universo realmente utilizable. Esto es critico porque evita entrenar o evaluar con tickers cuya informacion esta incompleta.

Si un ticker queda fuera, no significa que sea irrelevante; significa que no cumple las condiciones de integridad del pipeline.

### 5.5. Paso 4: dataset maestro

Esta es una de las fases mas importantes porque convierte datos historicos en observaciones entrenables. Cada fila ya no representa un archivo o una tabla, sino una decision temporal concreta.

### 5.6. Paso 5: evaluacion

Finalmente se calcula si la estrategia aporta valor. El punto importante es que la evaluacion no solo mide clasificacion, sino tambien curva monetaria, benchmark, baselines y auditoria.

### 5.7. Que controla este archivo

`analyzer.py` es el lugar correcto para:

- activar o desactivar bloques del pipeline,
- definir el rango temporal,
- fijar la politica de semillas,
- o coordinar exportaciones globales.

No es el lugar donde viven las reglas de negocio de features, entrenamiento o simulacion; esos detalles estan repartidos en los modulos especializados.

---

## 6) Capa de datos y fuentes

Las fuentes se almacenan en `data_finnhub/` por ticker y macro.

`DataRouter` abstrae el acceso para evitar acoplamiento a estructura fisica de archivos.

### 6.0. Vista general del paso

- Entrada: carpetas locales con datos descargados y consolidados por ticker.
- Proceso: el router interpreta cada fuente segun su formato y devuelve tablas ya preparadas para el resto del pipeline.
- Salida: precios, fundamentales, sentimiento, insider, valoracion y benchmark accesibles con una interfaz comun.

Se usan, entre otras, estas familias de datos:

- Precios OHLCV.
- Fundamentales consolidados.
- Insider transactions.
- Insider sentiment (MSPR).
- Recommendation trends.
- EPS surprises.
- SPY para benchmark de mercado.

### 6.1. Por que existe `DataRouter`

`DataRouter` evita que el resto del sistema tenga que saber donde vive cada archivo. En vez de acceder directamente a rutas y formatos, el pipeline pide datos a traves de una interfaz comun.

Eso aporta dos ventajas:

- reduce acoplamiento,
- y facilita auditar que cada fuente tenga su propia logica de carga.

### 6.2. Logica temporal de las fuentes

No todas las fuentes tienen la misma naturaleza temporal:

- los precios son series diarias,
- los fundamentales son eventos discretos publicados en fechas concretas,
- el sentimiento y el insider pueden acumularse en ventanas,
- el benchmark sirve como referencia de mercado para comparacion y simulacion.

Por eso el pipeline no puede tratarlas igual. Cada una necesita su propia regla de corte y su propia ventana de uso.

### 6.3. Que aporta esta separacion

Esta separacion permite tratar cada fuente segun su naturaleza temporal. Los precios se consumen como series continuas; los fundamentales se consumen como eventos publicados; el sentimiento y el insider como ventanas temporales; y el benchmark como referencia de mercado.

---

## 7) Construccion del dataset maestro (`step_02_dataset`)

### 7.0. Vista general del paso

- Entrada: datos crudos por ticker, historial de precios, consolidado fundamental y fuentes auxiliares.
- Proceso: se generan snapshots por quarter, se recorta cada fuente a la fecha de corte y se ensamblan features y labels.
- Salida: `master_dataset.csv` con una fila por `(ticker, date)` y todas las variables ya listas para training.

### 7.1 Unidad de observacion

La unidad base es `(ticker, date)` con `date` asociado al cierre de cada quarter de snapshot.

Columnas clave de control temporal:

- `year_quarter`: quarter del snapshot (ej. `2025Q2`).
- `snapshot_date`: primer dia del quarter + `SNAPSHOT_LAG_DAYS`.
- `report_end_date_used` y `report_filed_date_used`: trazan que reporte fundamental se uso realmente.
- `is_fundamental_carry_forward`: marca si se arrastro ultimo reporte publicado por falta de nuevo filing.

### 7.1.1. Que significa realmente un snapshot

Un snapshot es una fotografia del ticker en una fecha de decision. No es simplemente una fila historica. Es una construccion sintetica que intenta representar qué sabia el sistema en ese momento.

Por eso cada snapshot mezcla fuentes distintas pero alineadas a la misma fecha efectiva:

- precio disponible hasta ese momento,
- fundamentales ya publicados,
- senales de sentimiento disponibles,
- historial tecnico acumulado hasta la ventana definida.

### 7.1.2. Por que el quarter importa

El quarter es la unidad natural de decision porque:

- los fundamentales suelen publicarse con periodicidad trimestral,
- el estudio busca comparabilidad por periodos estables,
- y el target se define sobre un horizonte de holding compatible con esa granularidad.

### 7.1.3. Que salida produce esta etapa

El resultado no es una tabla final de predicciones, sino un panel estructurado que despues puede dividirse en train/test por fold.

### 7.2 Regla PIT para fundamentales

La seleccion del snapshot fundamental no usa "ultimo quarter por calendario" de forma ciega.

Regla:

- Se elige el reporte con `filedDate <= snapshot_date` mas reciente.
- Si no hay metadata de filing utilizable, fallback a `report_end_date <= snapshot_date`.

Esto evita look-ahead de fundamentales no publicados aun.

#### Logica operativa paso a paso

1. Se localizan todos los reportes disponibles del ticker.
2. Se calcula para cada reporte la fecha real de publicacion.
3. Se descartan los reportes posteriores a la fecha de snapshot.
4. Se selecciona el reporte mas reciente entre los ya publicados.
5. Si faltan metadatos, se usa la alternativa mas conservadora posible.

Esto es importante porque, sin esta regla, el modelo podria aprender de un balance o resultado que en realidad aun no era publicamente conocido.

### 7.3 Features por familia

- Fundamentales: ratios y tendencias historicas hasta `snapshot_date`.
- Tecnicos: calculados sobre ventana de precios `lookback_days` con corte as-of.
- Valoracion: multiples y derivados con estado as-of.
- Insider/sentiment: ventanas recortadas por tiempo (90d, 6m, etc.) con filtros as-of.

#### Fundamental

Aqui el objetivo es capturar estructura economica y contable:

- rentabilidad,
- crecimiento,
- calidad del balance,
- margenes,
- y tendencias de medio plazo.

Tambien se incorporan transformaciones historicas para saber no solo el nivel actual de una variable, sino su trayectoria.

#### Technical

Las tecnicas intentan describir el comportamiento reciente del precio:

- momentum,
- volatilidad,
- medias moviles,
- distancia a medias,
- y otras medidas de tendencia o reversión.

La ventana de 300 dias evita exigir historia excesiva pero mantiene contexto suficiente para senales de mercado.

#### Valuation

La capa de valoracion intenta capturar si un activo esta barato o caro frente a sus metricas de negocio.

Se combina el estado de mercado con las variables fundamentales para que el score no dependa solo del precio, sino tambien de la situacion financiera.

#### Insider y sentiment

Estas fuentes son mas ruidosas y mas sensibles al tiempo. Por eso se recortan con ventanas temporales especificas.

La idea es evitar que una señal antigua siga influyendo como si fuera actual.

### 7.4 Label del dataset maestro

`forward_return` se calcula desde `snapshot_date` hasta `snapshot_date + HOLDING_PERIOD_MONTHS`.

Luego, en training/evaluation, ese retorno se transforma en label relativo (outperformance sectorial por snapshot).

#### Por que no usar directamente el retorno bruto como label

Usar retorno bruto sin contexto sectorial puede introducir ruido por regimen de mercado. Una compañia puede subir menos que el mercado y aun asi ser mejor que su sector, o al reves.

La version relativa obliga al modelo a aprender seleccion cross-sectional, que es mas coherente con una cartera long-only de acciones individuales.

#### Que permite y que no permite esta definicion

Esta definicion permite entrenar un clasificador relativo al contexto sectorial. No permite, por si sola, afirmar que un activo sea bueno en terminos absolutos; solo indica si supera o no la referencia del sector en ese snapshot.

---

## 8) Anti-leakage: diseno y auditoria

### 8.0. Vista general del paso

- Entrada: datasets y fuentes que ya han pasado por la construccion del snapshot.
- Proceso: se aplican filtros as-of y auditorias de futuras filas para comprobar que no hay informacion posterior a la fecha de decision.
- Salida: un control explicito de leakage por fold y por fuente, exportado a `results/leakage_audit.csv`.

### 8.1 Utilidades comunes

`module/common/asof.py` centraliza:

- `filter_asof`
- `detect_future_rows`
- `assert_no_future_data`

Estas funciones son la primera barrera del sistema contra errores temporales. La idea es muy simple: antes de entrenar o auditar, recortar cualquier fila que quede por delante de la fecha de corte.

### 8.2 En split de folds

El fold usa train con quarters estrictamente anteriores al quarter analizado y test en el quarter analizado.

En terminos de condicion temporal:

- Train: `quarter < analysis_quarter`
- Test: `quarter == analysis_quarter`

#### Por que esto es crucial

Si train y test no estan temporalmente separados, el backtest deja de ser una evaluacion real y se convierte en una mezcla de historia pasada y futura. Ese es el error mas comun y mas peligroso en este tipo de proyectos.

#### Lectura intuitiva

El sistema aprende mirando atras y decide mirando un quarter concreto. Nunca deberia aprender con observaciones que ya pertenecen al quarter que esta intentando predecir.

### 8.3 En score de meta-learner

Las features OOF de agentes base se generan con `TimeSeriesSplit`, preservando orden temporal y evitando mezcla futura en scores de entrenamiento del meta.

#### Que resuelve el OOF

El meta-learner no debe entrenarse sobre predicciones que ya fueron generadas por un modelo que vio ese mismo dato durante el entrenamiento. El OOF evita precisamente eso.

#### Como pensarlo mentalmente

1. Se divide la historia en bloques temporales.
2. Para cada bloque, se entrena con pasado y se predice futuro inmediato.
3. Las predicciones se guardan solo cuando el dato no estuvo en el entrenamiento de ese submodelo.
4. El meta aprende sobre esas predicciones OOF.

### 8.4 Leakage audit exportable

Por fold se auditan fuentes (sentiment, insider, technical, fundamental, valuation input).

Se exporta:

- `results/leakage_audit.csv`

Campos principales:

- `fold_id`
- `ticker`
- `feature_group`
- `n_rows_future_detected`
- `max_future_date_detected`
- `context`

Interpretacion:

- `n_rows_future_detected > 0` indica incidencia de leakage potencial en la fuente auditada.

#### Como usar este reporte para interpretar el sistema

Si detectas incidencias, significa que la fuente auditada entrego filas que exceden la fecha de corte o que la estructura temporal de esa fuente no encaja con la regla as-of esperada. El reporte sirve para localizar exactamente donde ocurre eso.

---

## 9) Entrenamiento (`step_03_training`)

### 9.0. Vista general del paso

- Entrada: `df_train_norm`, `y_train`, `df_test_norm`, `y_test` y configuracion de agentes.
- Proceso: cada agente base aprende su vista del problema, se generan scores OOF, el meta-learner los combina y se obtiene un score final.
- Salida: scores por ticker, diagnosticos de agentes y, en evaluacion, una cartera candidata para el fold.

### 9.1 Agentes base

Configuracion declarativa en `agent_config.py`:

- `fundamental`
- `valuation`
- `momentum`
- `bear` (con inversion de label para riesgo)
- `sentiment`
- `sector_rotation` (entrenado por ruta separada)

#### Que hace cada agente a nivel logico

- Fundamental: intenta detectar calidad y crecimiento sostenible.
- Valuation: intenta detectar activos infravalorados o caros.
- Momentum: intenta capturar persistencia de precio.
- Bear: intenta medir riesgo o comportamiento defensivo.
- Sentiment: intenta capturar revision de analistas e informacion blanda.
- Sector rotation: intenta identificar que sectores estan relativamente mejor posicionados.

La razon de separarlos es que no todas las senales se comportan igual. El ensamblado les permite aportar evidencia complementaria.

### 9.2 Meta-learner

El meta consume scores de agentes base, incorpora ajustes de robustez y produce `final_score`.

Ajustes destacados:

- Shrink por baja dispersion de scores.
- Ajustes sectoriales y priors de confianza.
- Umbrales de seleccion para cartera.

#### Logica del meta en lenguaje simple

El meta-learner no sustituye a los agentes base: los combina.

Si varios agentes apuntan en la misma direccion, el meta puede reforzar esa señal. Si una familia es poco fiable o tiene poca dispersion, su influencia se reduce.

Eso hace que el score final no sea simplemente un promedio ingenuo, sino una agregacion con reglas de robustez.

#### Por que esto es mejor que un solo modelo grande

Un unico modelo puede aprender muchas correlaciones espurias. En cambio, el esquema multi-agente permite:

- especializacion,
- interpretabilidad,
- y mejor analisis de errores por familia de features.

#### Que limita este esquema

El esquema depende de que los scores base sean razonablemente informativos. Si todos los agentes entregan senales pobres o muy correlacionadas entre si, el meta-learner no puede crear informacion nueva de la nada.

### 9.3 Label de entrenamiento

No se optimiza sobre retorno absoluto puro del ticker.

Se utiliza label binaria relativa:

- `1` si `forward_return` del ticker supera mediana de su sector en ese snapshot quarter.
- `0` en caso contrario.

Esto reduce sesgo de regimen de mercado y enfatiza seleccion cross-sectional.

#### Consecuencia practica

El modelo no intenta ganar al mercado en valor absoluto, sino distinguir mejor que sus pares dentro del mismo entorno. Eso es muy util en stock picking porque el exito suele depender de seleccionar los mejores nombres relativos, no de predecir la direccion global del indice.

---

## 10) Walk-forward evaluation (`step_04_evaluation/evaluator.py`)

### 10.0. Vista general del paso

- Entrada: dataset maestro, mapa sectorial, series de precios, benchmark y configuracion temporal.
- Proceso: se generan folds, se construyen train/test por fold, se entrenan modelos, se puntuan tickers, se selecciona cartera y se simula el resultado.
- Salida: metricas por fold, curvas de equity, auditoria de selection/leakage y resumen consolidado de estrategia, benchmark y baselines.

### 10.1 Generacion de folds

El backtester genera ventanas train/test segun configuracion.

En modo anual:

- Se ejecuta un analisis por anio (ancla temporal anual).
- Manteniendo estructura trimestral interna para features y labels.

#### Paso a paso real del fold

1. Se elige una ventana de entrenamiento historica.
2. Se define el snapshot quarter objetivo.
3. Se construyen train y test respetando corte temporal.
4. Se recalculan labels con la informacion disponible de ese momento.
5. Se entrena el sistema completo sobre train.
6. Se puntua test.
7. Se selecciona cartera.
8. Se simula el rendimiento en USD.
9. Se guarda auditoria y resumen.

#### Por que el modo anual sigue usando logica trimestral

Porque las features y los reports siguen naciendo en snapshots trimestrales, pero el analisis puede agruparse por anio para simplificar lectura o centrar la memoria en una ventana mas amplia.

### 10.2 Reglas de elegibilidad de fold

Un fold se omite si falla alguno de estos criterios:

- Universo test por debajo de `MIN_TEST_TICKERS_PERCENT`.
- Benchmark sin suficientes precios en ventana de evaluacion.
- Sin cobertura de precios util para cartera.
- Train insuficiente o test vacio tras preparar labels.

#### Interpretacion de cada filtro

- Universo test insuficiente: el fold no representa bien el mercado, asi que no conviene evaluarlo.
- Benchmark sin precios: no existe una referencia justa.
- Sin cobertura de precios: no puede simularse una cartera real.
- Train insuficiente: el modelo no tiene historia suficiente para aprender con minima estabilidad.

Estos filtros no son caprichosos; protegen la calidad de la comparacion.

### 10.3 Seleccion de cartera

Se rankea por `final_score`.

Regla operativa:

- Filtrar tickers con `score >= PORTFOLIO_MIN_SCORE`.
- Si no hay suficientes, fallback a top-N por ranking.

Pesos:

- Equal-weight o score-weighted segun configuracion.

#### Lectura de la seleccion

La seleccion no es una simple lista de top-N. En realidad es una decision con dos capas:

1. umbral de calidad del score,
2. y, si hace falta, relleno por ranking para no dejar la cartera vacia o demasiado pequeña.

Esto evita que un fold comprimido termine sin operacion o con una cartera irrealmente minima.

---

## 11) Motor de backtest USD (`portfolio_simulator.py`)

### 11.0. Vista general del paso

- Entrada: tickers seleccionados, pesos, precios historicos y capital inicial.
- Proceso: se compran posiciones con fee y slippage, se valoran durante el horizonte de holding y se liquida la cartera al final.
- Salida: trades ejecutados, curva de equity diaria y resumen monetario del fold.

El modo monetario agrega realismo operativo:

- Capital inicial explicitamente modelado.
- Compras/ventas con fee fijo por ticker y por operacion.
- Slippage porcentual configurable.
- Soporte de acciones fraccionales.
- Curva de equity diaria (`cash + mark_to_market`).
- Encadenamiento de capital entre folds (`ending -> starting`).

### 11.1. Logica economica de la simulacion

La simulacion intenta representar una cartera operativa realista, aunque simplificada. La secuencia tipica es:

1. Se parte de un capital inicial.
2. Se eligen tickers y pesos.
3. Se calcula el coste de compra por ticker.
4. Se descuentan fees y slippage.
5. Se mantiene la cartera durante el periodo de holding.
6. Se valoran posiciones con precios diarios.
7. Se liquida al final del fold.
8. El capital final se usa como capital de entrada del siguiente fold.

#### Por que esto importa frente a un backtest por retornos

Un backtest por retornos puede dar una intuicion buena, pero no refleja bien el efecto de fees acumulados, cambios de capital o composicion real de cartera. La version USD obliga a pensar en dinero y no solo en porcentajes.

### 11.1 Resolucion de fechas y precios

Para entry/exit se aplica regla robusta:

- Si no hay precio en fecha solicitada, usar primer precio disponible `>= fecha_solicitada`.

Esto queda trazado en `trades.csv` con campos `entry_date_requested`/`entry_date_used` y `exit_date_requested`/`exit_date_used`.

#### Que resuelve esta regla

Los mercados no siempre tienen precio exactamente en la fecha teorica deseada. Esta regla evita que el pipeline falle o que invente un precio. En vez de eso, mueve la ejecucion al siguiente dato util disponible y lo deja documentado.

#### Que limita esta regla

La regla asume que desplazar la ejecucion al siguiente precio disponible es una aproximacion aceptable. No modela microestructura intradia ni diferencia entre precio de apertura y cierre.

### 11.2 Formato detallado de trades: trades_detailed.csv

Para facilitar la lectura de cada trade individual, se genera `trades_detailed.csv` que agrupa cada compra/venta por ticker en una sola fila, mostrando claramente los USD invertidos y recibidos.

#### Entrada (Input)

- Archivo bruto `trades.csv` con columnas: action (BUY/SELL), ticker, exec_price, shares, notional_usd, fee_usd.

#### Proceso

1. Se lee `trades.csv` del fold.
2. Se separan BUYs y SELLs por ticker.
3. Para cada ticker se emparejan compra → venta en orden cronologico.
4. Se calcula USD total gastado en compra = notional_usd + fee_usd.
5. Se calcula USD total recibido en venta = notional_usd - fee_usd.
6. Se calcula PnL = USD recibido - USD gastado.
7. Se calcula PnL % = (PnL / USD gastado) * 100.

#### Salida (Output)

Cada fila en `trades_detailed.csv` representa un par compra/venta para un ticker en un fold. Columnas principales:

- `ticker`: símbolo del activo.
- `buy_date`, `buy_price`, `buy_shares`: fecha, precio y cantidad de compra.
- `buy_notional_usd`: valor sin fees (shares * price).
- `buy_fees_usd`: comisión de transacción.
- `buy_total_cost_usd`: notional + fees (cantidad total de USD gastado).
- `sell_date`, `sell_price`, `sell_shares`: fecha, precio y cantidad de venta.
- `sell_notional_usd`: valor sin fees.
- `sell_fees_usd`: comisión de transacción.
- `sell_total_received_usd`: notional - fees (cantidad total de USD obtenido).
- `pnl_usd`: ganancia en USD = sell_total_received - buy_total_cost.
- `pnl_pct`: ganancia porcentual = (pnl_usd / buy_total_cost) * 100.
- `hold_days`: días de holding desde compra a venta.

#### Ejemplo interpretacion

Si un trade muestra:
- buy_total_cost_usd = 100.0 (gasté 100 USD)
- sell_total_received_usd = 110.5 (recibí 110.5 USD)
- pnl_usd = 10.5
- pnl_pct = 10.5

Se entiende claramente: "Puse 100 USD, recuperé 110.5 USD, gané 10.5 USD (10.5%)".

### 11.3 Metricas economicas derivadas

Desde curvas de equity se obtienen:

- Retorno total
- Max drawdown
- Sharpe sobre retornos diarios
- Fees acumuladas

#### Como interpretarlas

- Retorno total: que gano o perdio la cartera.
- Max drawdown: cuanto sufrio en la peor racha.
- Sharpe: rendimiento ajustado por volatilidad.
- Fees: cuanto costo operar.

No conviene mirar una sola de estas metricas aislada. Un sistema puede tener buen retorno y mal drawdown, o buen Sharpe y fees demasiado altos.

---

## 12) Benchmark y baselines

### 12.0. Vista general del paso

- Entrada: la misma ventana temporal y el mismo universo de tickers elegibles para la estrategia principal.
- Proceso: se aplican reglas equivalentes de compra, salida, fees y slippage a SPY y a las estrategias baseline.
- Salida: comparativas homogéneas frente a benchmark y baselines, junto con sus curvas de equity y summaries.

### 12.1 Benchmark principal

Se usa SPY buy-and-hold en USD con mismo motor de simulacion.

Si la fecha final solicitada excede el ultimo dato disponible de SPY, se trunca salida a ultimo dia con datos y se marca disponibilidad real.

Artefactos:

- `results/backtest/benchmark_equity_curve.csv`
- `results/backtest/benchmark_summary.json`

#### Por que SPY es el benchmark natural aqui

SPY representa una referencia amplia del mercado US. Para un TFM de stock picking long-only, es una comparacion razonable porque responde a la pregunta central: si selecciono acciones activamente, ¿hago algo mejor que un exposure pasivo al mercado?

### 12.2 Baselines implementados

- `ew_universe`: equal-weight sobre universo elegible por fold.
- `momentum_12m`: top-N por retorno 12m previo al entry.
- `random_topn_mean`: media de N simulaciones aleatorias reproducibles.
- `value_combined`: ranking combinado fijo por P/E y EV/EBITDA.

Todos los baselines usan la misma mecanica de entrada/salida/fees/slippage.

#### Para que sirve cada baseline en la memoria

- Equal-weight universe: mide si la seleccion agrega valor frente a una distribucion simple.
- Momentum 12m: representa una heuristica clasica de mercado.
- Random top-N: define una referencia de azar reproducible.
- Value combined: mide si una regla simple de valor ya explica gran parte del resultado.

#### Lo que deberias mirar al comparar

No solo el retorno. Tambien:

- estabilidad entre folds,
- drawdown,
- costes de operacion,
- disponibilidad real,
- y sensibilidad al universo elegible.

### 12.3 Transparencia de seleccion baseline

Se exportan archivos de detalle de seleccion:

- `results/backtest/baselines/ew_universe_selection_by_fold.csv`
- `results/backtest/baselines/momentum_12m_selection_by_fold.csv`
- `results/backtest/baselines/random_topn_selection_by_sim.csv`
- `results/backtest/baselines/value_combined_selection_by_fold.csv`

#### Por que esto es valioso

Los baselines suelen quedarse en una cifra agregada. Aqui, en cambio, puedes explicar exactamente que activos selecciono cada baseline y por que. Eso te permite defender o criticar su comportamiento con detalle.

---

## 13) Catalogo de artefactos de salida

### 13.1 Raiz `results/`

- `pipeline.log`: log integral del run.
- `master_dataset.csv`: panel maestro construido.
- `run_config.json`: snapshot completo de configuracion/versiones/flags.
- `data_quality_report.csv`: cobertura y missing por ticker/familia de features.
- `leakage_audit.csv`: auditoria PIT por fold/ticker/fuente.
- `baselines_summary.csv`: comparativa consolidada estrategia/benchmark/baselines.
- `final_portfolio_value.json`: resumen final monetario.
- `final_summary.json` y `final_summary.csv`: resumen ejecutivo final.

### 13.2 Carpeta `results/backtest/`

- `missing_prices_report.csv`
- `strategy_equity_curve.csv`
- `benchmark_equity_curve.csv` (si disponible)
- `benchmark_summary.json`
- `fold_comparison_summary.csv` (comparativa por fold entre estrategia, benchmark y baselines)
- `annual_return_comparison.csv` (retorno anual por serie para comparativa global)
- `fold_{k}/` con:
  - `trades.csv`
  - `trades_detailed.csv` (formato legible: compra/venta lado-a-lado con USD gastado y recibido)
  - `equity_curve.csv`
  - `selection.csv`
  - `portfolio_summary.json`
  - `metrics.json`

### 13.3 Carpeta `results/backtest/baselines/`

- Curvas equity baseline.
- Summaries JSON baseline.
- Reporte de disponibilidad value baseline.
- Selecciones por fold/simulacion.
- `random_topn_fold_summary.csv` (media y bandas p05/p95 por fold para random top-N).

### 13.4 Carpeta `results/plots/`

Plots obligatorios generados:

- `equity_curve_usd.png`
- `equity_curve_usd_with_baselines.png`
- `drawdown_usd.png`
- `capital_by_fold.png`
- `pnl_pct_by_fold.png`
- `fold_pnl_comparison_with_baselines.png`
- `annual_return_comparison_with_baselines.png`

#### Como usar los plots

- `equity_curve_usd.png`: ver tendencia y comparacion con benchmark.
- `equity_curve_usd_with_baselines.png`: comparar la estrategia principal con alternativas.
- `drawdown_usd.png`: entender riesgo acumulado.
- `capital_by_fold.png`: detectar fold especialmente fuerte o debil.
- `pnl_pct_by_fold.png`: ver dispersion entre folds.
- `fold_pnl_comparison_with_baselines.png`: comparar por fold la estrategia principal contra benchmark y cada baseline en el mismo eje.
- `annual_return_comparison_with_baselines.png`: comparar por ano calendario la estrategia completa frente a benchmark y baselines en todo el periodo analizado.

---

## 14) Reproducibilidad y trazabilidad

La reproducibilidad se sostiene por:

- `RANDOM_SEED` global aplicado al inicio.
- Seeds derivadas para simulaciones random baseline.
- Export de versiones de librerias y hash de commit en `run_config.json`.
- Export detallado de decisiones y operaciones por fold.

### 14.1. Que significa reproducibilidad aqui

No significa solo que el script "corra otra vez". Significa que puedes reconstruir:

- que datos se usaron,
- que configuracion estaba activa,
- que version de codigo se ejecuto,
- y como llegaron los resultados finales.

### 14.2. Por que esto importa en el analisis del sistema

La trazabilidad permite comparar resultados entre ejecuciones distintas bajo el mismo contexto de datos y configuracion. Eso es importante para saber si una diferencia en resultados procede de los datos, de la configuracion o del propio codigo.

Recomendacion de defensa TFM:

- Congelar entorno con `requirements.txt`.
- Adjuntar `run_config.json` y hash de commit del run mostrado.
- Presentar siempre resultados junto con `leakage_audit.csv` y `data_quality_report.csv`.

### 14.3. Que conviene guardar cuando repitas experimentos

Ademas de los archivos ya generados, conviene conservar:

- la fecha de ejecucion,
- la version de dependencias,
- el hash de commit,
- y cualquier cambio manual aplicado antes del run.

---

## 15) Lectura metodologica de resultados

Al interpretar resultados, separar 3 planos:

1. Plano de modelado (clasificacion):
   - metrica de acierto de label relativa sectorial.
2. Plano economico (USD):
   - valor final, retorno, drawdown, sharpe, fees.
3. Plano de robustez comparativa:
   - mejora sobre benchmark y sobre baselines alternativos.

### 15.1. Como hacer una lectura sana de resultados

La forma correcta de evaluar el proyecto es responder estas preguntas en orden:

1. ¿Los datos son validos y completos?
2. ¿Hay evidencia de leakage?
3. ¿La estrategia supera al benchmark?
4. ¿La estrategia supera a baselines razonables?
5. ¿Lo hace con riesgo y costes aceptables?

Si alguna de esas respuestas es negativa, el resultado no debe venderse como concluyente.

### 15.2. Error tipico al interpretar una estrategia

Un error frecuente es quedarse solo con el retorno final. Eso no basta. Un sistema puede tener:

- buen retorno pero drawdown excesivo,
- buen Sharpe pero fees altos,
- o buena clasificacion pero mala traduccion economica.

Por eso este proyecto separa claramente metricas de modelo y metricas de cartera.

Una estrategia defendible no es solo la que maximiza retorno, sino la que mantiene consistencia bajo:

- costos operativos,
- cobertura de datos real,
- control temporal estricto,
- comparacion contra alternativas no triviales.

---

## 16) Supuestos y limitaciones actuales

Supuestos operativos:

- Long-only.
- Rebalance por fold (no intrafold dinamico).
- Fee fijo por transaccion por ticker.
- Slippage constante (no dependiente de liquidez).
- Sin impuestos ni borrowing costs.

Limitaciones conocidas:

- No hay modelado explicito de market impact por volumen.
- No hay ejecucion intradia ni colas de ordenes.
- Universe puede tener sesgo de supervivencia si no se versiona composicion historica.
- El benchmark depende de cobertura de SPY en el rango del fold.

### 16.1. Por que declarar limitaciones mejora la defensa

No debilita el trabajo. Al contrario, muestra que sabes exactamente donde el modelo es fuerte y donde no lo es. En una defensa academica eso vale mucho mas que prometer realismo total.

### 16.2. Que no hace este sistema

El sistema no ejecuta ordenes reales, no modela coste de mercado variable por liquidez, no hace intradia, no calcula impuestos y no incorpora borrowing costs. Tampoco pretende predecir de forma directa el precio de cada accion en un horizonte continuo; su objetivo es seleccionar carteras por snapshot y evaluar su comportamiento posterior.

---

## 17) Guia rapida de ejecucion

Ejecutar pipeline completo:

```bash
python analyzer.py
```

Modo solo actualizacion de precios/macro:

- Configurar `UPDATE_PRICES_ONLY=True` en `environment.py`.
- Ejecutar `python analyzer.py`.

Modo sin backtest historico:

- Configurar `SKIP_BACKTEST=True`.
- Ejecutar `python analyzer.py`.

---

## 18) Alcance y limites del sistema

### 18.1. Que puede hacer

El sistema puede:

- descargar y consolidar datos financieros y de mercado,
- construir un panel maestro PIT por ticker y quarter,
- entrenar agentes especializados y un meta-learner,
- generar una cartera long-only por fold,
- simular esa cartera en USD con fees y slippage,
- comparar la estrategia contra SPY y baselines,
- y exportar resultados, auditorias y plots para documentar el experimento.

### 18.2. Que no puede hacer

El sistema no puede:

- garantizar que todos los tickers tengan datos completos,
- reconstruir informacion que no exista en las fuentes,
- modelar ejecucion intradia real,
- sustituir una estrategia de trading live completamente automatizada,
- ni demostrar causalidad economica; solo compara desempeno relativo bajo el protocolo definido.

### 18.3. Donde es fuerte

Es fuerte cuando se quiere estudiar seleccion cross-sectional con disciplina temporal, comparacion justa y trazabilidad. Tambien es fuerte para defender por que una cartera se formo de una manera concreta en un quarter concreto.

### 18.4. Donde es mas debil

Es mas debil en realismo de microestructura, en sensibilidad a cambios de liquidez, y en escenarios donde la disponibilidad historica de datos no sea homogénea. En esos casos, el resultado sigue siendo util como experimento academico, pero no como prueba de ejecutabilidad directa en mercado.

---

## 19) Conclusiones tecnicas

El estado actual del proyecto ya incorpora los bloques que normalmente faltan en prototipos academicos:

- control PIT y auditoria,
- simulacion monetaria realista por folds,
- benchmark y baselines homogeneos,
- trazabilidad reproducible,
- artefactos suficientes para defensa tecnica.

La calidad final para memoria/tribunal dependera de ejecutar runs limpios sobre un entorno congelado, documentar supuestos de forma explicita y acompanar cada claim de performance con sus auditorias y comparativas. Ese es el criterio correcto para convertir el proyecto en una memoria defendible y no solo en un experimento interesante.
