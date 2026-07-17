# Plan de reconstrucción del TFM por fases

> Documento vivo. Complementa a `docs/doc.md` (el plan maestro, que define el *qué* y el
> *porqué*) con el **detalle ejecutable de cada fase**: qué ficheros, qué decisiones y qué
> tests. Se actualiza al cerrar cada fase.

## Estado de las fases

| Fase | Estado |
|---|---|
| **0 — Universo dinámico, EDGAR y descarga** | Implementada y validada en modo `dev`; descarga completa pendiente |
| **1 — Dataset point-in-time** | Implementada y validada en modo `dev` |
| **2 — Features y baselines** | Implementada y probada con fixtures point-in-time |
| **3 — Agentes ML + meta-agente** | Implementada y probada con walk-forward sintético |
| **4 — Cartera y backtest** | Implementada y probada con 7 tests (reglas + invariantes PIT + contabilidad) |
| **5 — Informe HTML** | Implementada (`report.html` por run con 6 hojas + `comparison.html` para el barrido con 5 hojas) |
| **6 — Rejilla y selección final** | Implementada (barrido con reutilización por huella + selección automática por consistencia) |
| 7 — Redacción del TFM en LaTeX | Pendiente (documento de rumbo en `latex/plan_tfm.md`) |

Cada fase se diseña y se aprueba antes de implementarla (ver `CLAUDE.md`). Lo descrito en las
fases pendientes es **rumbo acordado**, no código existente.

## Contexto

El repositorio está en reinicio en limpio: solo existe la descarga de datos. `docs/doc.md`
define el destino (§15, roadmap de 6 fases) pero no el detalle ejecutable. Este plan cubre lo
que falta, fase a fase.

**Resultado esperado**: un sistema que simula fielmente haber ejecutado el pipeline en una
fecha pasada — en cada fecha ancla ve, de cada empresa, su último informe *realmente
publicado* — con el sesgo de supervivencia medido por año en lugar de solo declarado.

## La regla central (define todo el diseño)

> Doy un **año, un trimestre y un `lag_days`**. Se suman: `2000-Q1 + 45 d` = **15-feb-2000**.
> Ese es el día en que simulo ejecutar el pipeline. Uso **solo lo publicado hasta ahí**. De
> cada ticker cojo su último informe ya publicado; si aún no ha publicado el trimestre, cojo
> el anterior suyo.

Dos conceptos que **no** deben confundirse (esta distinción corrige un error de diseño previo):

- **`lag_days` = margen de ejecución.** Parámetro tuyo. Define **cuándo miras**, no qué es
  observable. Un solo valor por run.
- **Fecha de publicación = hecho del dato.** No es un parámetro. Cada empresa publica cuando
  publica, y se lee de la fuente.

Aplicar `lag_days` a cada fundamental (el diseño anterior) es incorrecto: retrasa
artificialmente a quien ya había publicado y adelanta a quien tardó más. Verificado que
cualquier retardo fijo falla: **AT&T llegó a tardar 133 días** y un 10-K de AAPL **88 días**.
La única regla fiel es la fecha real.

## Hallazgos verificados que condicionan el plan

1. **Las fechas reales de publicación existen desde 1993 en SEC EDGAR** (`submissions`),
   gratis, oficial y sin clave. Verificado: AAPL cerró trimestre el `2000-01-01` y publicó el
   `2000-02-01` — con ancla 15-feb-2000, **ya era público**. La regla funciona en 2000.
2. **Finnhub solo da `filedDate` desde 2010** (límite del plan free). Medido sobre 20
   veteranas (IBM, GE, KO…): *ninguna* tiene fechas anteriores a 2010. Por eso EDGAR es
   necesario, no un lujo.
3. **`report_dates.parquet` no existe** y hay 0 ficheros `financials_reported.json`. La causa
   **no** es que el endpoint sea premium (probado en vivo: HTTP 200). El bloque que lo descarga
   ([pipeline.py:64-74](module/ingest/pipeline.py#L64-L74)) se añadió en el commit `609a97b2`,
   del mismo día que la última descarga, que corrió con código anterior.
4. **Los deslistados no son reconstruibles.** Enron, Lehman, WaMu, Kodak, Nortel: 0 días de
   precio en Yahoo, 0 métricas en Finnhub, sin CIK en EDGAR. Peor: `CPQ` (Compaq, absorbida en
   2002) devuelve precios de **2004-2025** y `MOB` (Mobil, fusionada en 1999) de **2022** —
   **reciclaje de ticker**: datos de otra empresa bajo el mismo símbolo.
5. **Los precios ya cubren desde 1990** (238 tickers), pese a `DATA_START_DATE = "2000-01-01"`.
   El entrenamiento hacia atrás desde 2000 no requiere descargar precios.
6. **EDGAR `companyfacts` (fundamentales crudos con `filed`) solo llega a 2009**: la SEC no
   exigió XBRL hasta entonces. **No sirve para el ancla de 2000** — ver "Qué se descarta".

## Decisiones acordadas

| Tema | Decisión |
|---|---|
| **Fechas de publicación** | SEC EDGAR `submissions` (1993-2026). Aprobado. |
| **Ancla** | `2000-Q1` por defecto, **configurable** (año, trimestre, `lag_days`) para barrerla en la rejilla de Fase 6. |
| **Universo** | Dinámico por fecha desde el CSV histórico + cuantificar el sesgo restante por año. |
| **Tests** | pytest. Aprobado. |

### Ejecución por etapas

`RUN_MODE` selecciona una etapa concreta (`download`, `dataset`, `features`,
`agents`, `backtest`, `report` o `experiments`) y `RUN_SCOPE` selecciona el
alcance (`dev` o `full`). `RUN_MODE=full` encadena las etapas ya implementadas
en el orden download → dataset → features → agents. Las etapas futuras no se
ejecutan hasta que tengan handler; una etapa aún no implementada falla de forma
explícita. Los agregados de desarrollo se escriben en `data/raw/dev/`.

---

## Fase 0 — Universo dinámico, EDGAR y ampliación de la descarga

> **Esta es la única fase que se implementa ahora.** Al terminar, lanzas `python main.py` y
> obtienes los datos. Nada de Fase 1+ se toca.

**Ficheros**
- `module/universe.py` (nuevo): carga `data/S&P 500 Historical Components & Changes.csv`
  (2718 snapshots diarios, 1996-2026), normaliza tickers (`BRK.B` → `BRK-B`; 11 casos con
  punto) y expone `members_at(date)` (último snapshot ≤ *date*) y `historical_universe()`
  (unión: **1206 tickers únicos**).
- `module/ingest/edgar.py` (nuevo): cliente EDGAR. `User-Agent` identificatorio obligatorio,
  ~10 req/s permitidos. Descarga `company_tickers.json` (mapeo ticker→CIK) y, por empresa,
  `submissions/CIK##########.json` (+ el fichero `-submissions-001.json` con los filings
  antiguos, donde están los de los 90). Extrae `(form, reportDate, filingDate)` de 10-Q/10-K.
  Si el listado actual de CIK es ambiguo por ticker reciclado, valida el emisor con la búsqueda
  de EDGAR antes de registrar que faltan informes.
- `environment.py`: sustituir la lista estática `TICKERS` por carga dinámica desde el CSV;
  bajar `DATA_START_DATE` a `"1990-01-01"` (habilita entrenar hacia atrás y reutiliza la caché
  OHLCV, que ya es de 1990).
- `module/ingest/pipeline.py`: iterar sobre el universo histórico; registrar en
  `download_failures.csv` los que no devuelvan datos (evidencia del sesgo, no un error).

**Guarda contra reciclaje de ticker** (crítico): descartar los datos de un ticker cuyo primer
día de precio sea **posterior** a su última pertenencia al índice. Elimina `CPQ` y `MOB`
automáticamente. Se registra cada descarte.

**Salidas nuevas**
- `data/raw/report_dates.parquet`: `ticker`, `cik`, `form`, `period` (`reportDate`),
  `filed_date` (`filingDate`). **La pieza que hace posible la regla central.**
- `data/raw/universe_coverage.json`: por año, miembros reales del índice vs. tickers con datos
  utilizables — la **medición** del sesgo de supervivencia.

**Cobertura medida** (antes de ampliar): 1996 → 154/487 (32 %), 2000 → 194/491 (40 %), 2008 →
252/497 (51 %), 2016 → 329/504 (65 %), 2024 → 463/503 (92 %). Con CIK en EDGAR: 715/1206 (los
491 sin CIK son los deslistados, coherente con el hallazgo 4).

**Descarga a ejecutar** (requiere tu OK): ~705 tickers nuevos en Finnhub/Yahoo (~40-70 min por
el rate limit de 1,05 s) + EDGAR (~715 empresas, mucho más rápido). Reutiliza la caché
existente de los 504 ya descargados.

**Tests de esta fase** (`tests/download/`, con pytest ya aprobado):
- `members_at(2000-01-03)` no contiene tickers que entraron después (NVDA entró en 2001) —
  **elimina el sesgo de inclusión anticipada**.
- La normalización mapea `BRK.B` → `BRK-B` (11 casos con punto).
- La guarda de reciclaje descarta `CPQ` y `MOB`.
- El parseo de EDGAR extrae `(period, filed_date)` de un `submissions` de ejemplo, incluyendo
  el fichero histórico `-submissions-001.json` (donde están los filings de los 90).

**Verificación end-to-end de la fase**: `python -m pytest tests/download/ -v` verde;
ejecutar `RUN_MODE=download` con `RUN_SCOPE=dev`; comprobar
que se generan `report_dates.parquet` (con `filed_date` real) y `universe_coverage.json`, y que
la prueba manual cuadra: AAPL debe tener el informe `period=2000-01-01` con
`filed_date=2000-02-01`.

### Qué se descarta y por qué

**EDGAR `companyfacts`** (fundamentales crudos con `filed` por dato, `accn` para trazabilidad,
y restatements con ambas versiones). Es técnicamente superior a las series de Finnhub, pero
**empieza en 2009** (obligación XBRL). Con ancla en 2000 cubriría menos de la mitad del panel
y obligaría a mezclar dos fuentes con distinta definición de cada ratio — un cambio de régimen
artificial peor que el problema que resuelve. Se deja documentado como vía de mejora si el TFM
se reorientase a 2009+.

## Fase 1 — Dataset point-in-time

**Objetivo**: panel `(ticker, snapshot_date)` que reconstruye lo observable en cada fecha.

**Configuración** (`environment.py`):
```python
EXECUTION_YEAR = 2000
EXECUTION_QUARTER = 1        # ancla = inicio de trimestre + EXECUTION_LAG_DAYS
EXECUTION_LAG_DAYS = 45      # margen de ejecucion: 2000-01-01 + 45d = 15-feb-2000
TRAIN_LOOKBACK_YEARS = 8     # entrena hacia atras: 2000 -> desde 1992
SNAPSHOT_STEP_MONTHS = 1     # revisar cartera: mensual
FUNDAMENTAL_STEP_MONTHS = 3  # reentrenar: trimestral
```
Los tres primeros son los parámetros que barre la rejilla en Fase 6.

**Algoritmo de observabilidad** (la regla central, en `module/dataset.py`):

Para cada `snapshot_date` y cada ticker:
1. Buscar en `report_dates.parquet` los informes con `filed_date <= snapshot_date`.
2. Quedarse con el de `period` más reciente → ese es el trimestre **conocido** ese día.
3. Leer el valor de las series de Finnhub para ese `period`.
4. Si no hay ninguno publicado aún, se usa el anterior (sale solo del paso 2). Si no hay
   ninguno en absoluto, NA (sin relleno hacia atrás).

Así, si Apple publicó el 20-ene y otra empresa aún no, cada una aporta lo que de verdad se
sabía el 15-feb. **Sin retardo fijo.**

**Fuente de valores** — solo `payload.series.{annual,quarterly}` de `finnhub_metrics.parquet`
(39 y 41 métricas, `list<struct<period, v>>`), fechadas por `period` y **observables según
EDGAR**.

> **Prohibido**: `payload.metric` (133 escalares) y las columnas de `profiles.parquet`
> (`marketCapitalization`, `finnhubIndustry`) son **snapshots de hoy**. Usarlos por fecha
> pasada es lookahead directo. `sector` se excluye hasta disponer de una fuente histórica.

**Esquema** → `data/processed/panel_point_in_time.parquet`, granularidad `(ticker, snapshot_date)`:

`ticker`, `snapshot_date`, `review_type` (`fundamental_quarterly` / `price_monthly`), `price`,
`price_return_1m/3m/6m/12m`, `roe`, `roic`, `net_margin`, `operating_margin`, `gross_margin`,
`fcf_margin`, `pe`, `pb`, `ps`, `ev_ebitda`, `debt_equity`, `current_ratio`, `eps_growth_yoy`,
`sales_per_share_growth_yoy` (t vs. t-4 trimestres), `fundamental_period`,
`fundamental_filed_date` (**fecha real**), `fundamental_age_days`, `in_sp500`.

**Descartado**: el `pe_price_adjusted = pe_base * (1 + price_change)` del código previo
(`git show aee5b9e9^:module/dataset.py`) mezcla un ratio de cierre fiscal con el precio actual,
creando una magnitud híbrida injustificable.

**Ficheros**: `module/dataset.py` construye el panel; `module/utils.py` aporta
`read_parquet`; `main.py` registra la etapa `dataset`. El output es
`data/processed/panel_point_in_time.parquet` o
`data/processed/dev/panel_point_in_time.parquet` según `RUN_SCOPE`.

**Tests** (`tests/dataset/test_leakage.py` — lo más crítico; escribir **antes** que el módulo, según
`docs/doc.md:286`):
1. `fundamental_filed_date <= snapshot_date` en toda fila — **la invariante central**.
2. **Mutar el futuro no cambia el pasado**: construir el panel, borrar todo dato posterior a
   *t*, reconstruir; las filas ≤ *t* idénticas. Caza fugas sin saber por dónde entran.
3. Fundamentales congelados entre publicaciones aunque el precio se mueva (detecta que se coló
   `payload.metric`).
4. Empresas con distinto calendario aportan distinto `fundamental_period` en la misma
   `snapshot_date` — **verifica que la regla es por empresa, no global**.
5. Antes del primer `filed_date`, NA sin relleno hacia atrás.
6. **Contrato**: el panel no contiene `marketCapitalization` ni claves de `payload.metric`.

**Verificación**: `python -m pytest tests/dataset/ -v` verde; `RUN_MODE=dataset`; filas por snapshot **crecientes**
(~194 en 2000 → ~463 en 2024, coherente con la cobertura medida); `fundamental_age_days` nunca
negativo; prueba manual AAPL: el 15-feb-2000 debe verse el informe del `2000-01-01` (publicado
el `2000-02-01`) y **no** el siguiente.

**Mejora futura — sector histórico**: Finnhub solo aporta `finnhubIndustry` actual. Cuando se
incorpore una fuente con vigencia histórica, `sector` podrá añadirse para diagnóstico,
neutralización y diversificación; nunca como señal predictiva.

## Fase 2 — Features y baselines

`module/features.py` genera factores GARP (calidad, crecimiento, valoración) y momentum
relativo a SPY (3/6/12 m), solo desde artefactos point-in-time. SPY se descarga como serie
OHLCV obligatoria y se materializa en `benchmark_point_in_time.parquet`; no se le exigen CIK,
fundamentales ni perfil. `targets_forward_3m.parquet` mantiene separada la etiqueta de retorno
excesivo futuro a tres meses. `module/baselines.py` produce GARP equilibrado y momentum puro;
comprar el índice se representa mediante la serie de benchmark. Los precios con más de siete días
de antigüedad no reciben factores ni score.

## Fase 3 — Agentes ML + meta-agente

`module/agents.py` entrena agentes Ridge de calidad, momentum y valor; cada uno produce una
ordenación. `module/meta.py` los pondera por rank-IC positivo realizado de hasta los últimos
12 trimestres. El reentreno es trimestral y el scoring mensual. El sistema empieza en la fecha
ancla de 2000 usando toda la historia disponible; tras completar ocho años, aplica una ventana
móvil de `TRAIN_LOOKBACK_YEARS`. Las imputaciones y escalados se ajustan únicamente en cada
ventana histórica y conservan indicadores de ausencia. Los runs se guardan con huella de inputs,
coeficientes, scores, pesos y diagnósticos OOS.

### Revisión de las Fases 1-3 (correcciones aplicadas tras auditoría)

Tras implementar las Fases 1-3, una revisión del código encontró y corrigió tres problemas.
Ninguno era una fuga temporal: la separación train/eval, el `fit` solo-sobre-train y la
inmutabilidad del pasado ante cambios futuros se verificaron correctas y no se tocaron.

1. **`_yoy_growth` comparaba por índice posicional, no por fecha.** Contaba cuatro trimestres
   hacia atrás en la serie disponible; si faltaba alguno (frecuente: la cobertura de Finnhub es
   desigual), el "interanual" comparaba contra un trimestre de hace más o menos de un año sin
   avisar. **Arreglo**: se empareja por fecha (el trimestre más cercano a `period - 12 meses`,
   tolerancia ±45 días); sin pareja dentro de tolerancia, NA.
2. **Los targets se perdían en silencio si `SNAPSHOT_DAY` no coincidía con la aritmética de
   calendario.** `future_snapshot_date` se calculaba como `snapshot + 3 meses`, pero la rejilla
   de snapshots clampa los fines de mes con `min(snapshot_day, days_in_month)`, una regla
   distinta a la de `DateOffset`. Con `SNAPSHOT_DAY = 31` (no el valor por defecto, pero sí uno
   de los parámetros que la Fase 6 barre) el 40 % de las etiquetas no encontraba pareja y
   quedaba NaN sin ningún error. **Arreglo**: la fecha de etiqueta se toma de la propia rejilla
   (la posición N pasos adelante), no de una suma de calendario — existe por construcción.
3. **El fallback anual/trimestral podía mezclar magnitudes distintas.** Para las métricas de
   flujo (`net_margin`, `operating_margin`, `gross_margin`, `fcf_margin`) el código buscaba
   primero en `quarterly` y, si faltaba, en `annual` — pero un cierre anual comparte fecha con
   su Q4, así que dos tickers del mismo corte transversal podían aportar un margen de doce
   meses y uno de tres bajo la misma columna. **Decisión (aprobada)**: esas métricas se leen
   **solo** de `quarterly`; sin valor ahí, NA. Los ratios TTM y los de balance (`pb`,
   `debt_equity`, `current_ratio`) sí admiten `annual`, porque no cambian de significado según
   la frecuencia de origen.

Los tres casos están cubiertos por tests de regresión (`tests/dataset/test_fundamental_values.py`,
`tests/features/test_target_alignment.py`) que fallan contra el código anterior al arreglo.

## Fase 4 — Cartera y backtest

Implementada como dos módulos separados: `module/portfolio.py` con la lógica de decisión
(sin I/O) y `module/backtest.py` con el simulador puro (aplica órdenes a precios PIT).
Ambos disparados por `RUN_MODE=backtest`, que localiza el último `run_dir` de agentes y
añade en su interior los cinco parquets del backtest + `backtest_summary.json`.

**Reglas de cartera acordadas** (todas configurables desde `environment.py`):

1. **Expulsión**: un tenente cuyo percentil cae por debajo de `MIN_HOLD_PERCENTILE` (por
   defecto 50) sale, aunque nadie tenga la ventaja para desplazarle. Motivo
   `dropped_below_min`.
2. **Umbral de ventaja**: un candidato fuera solo desplaza a un tenente si le supera por
   `ROTATION_EDGE_PERCENTILES` (por defecto 5). Evita rotar por ruido.
3. **Sin tenencia mínima**: cada revisión decide desde cero. Un ticker puede entrar en
   `t`, salir en `t+1` porque ha subido y su valuación ya no compensa, y volver a entrar
   más tarde. La revisión mensual también decide, no solo la trimestral, porque los
   scores incluyen precio (P/E, P/B, momentum) y cambian aunque los fundamentales no.
4. **Tamaño flexible** `TARGET_MIN` = 5, `TARGET_MAX` = 10, `ENTRY_MIN_PERCENTILE` = 80.
   La cartera llega a 10 solo si hay 10 candidatos por encima del 80; si no, se queda
   con los que haya, sin bajar de 5 mientras haya candidatos que cumplan.
5. **Sizing con tope**: peso proporcional al ranking dentro de la cartera con tope
   `MAX_WEIGHT_PER_POSITION` (por defecto 20 %). El excedente se reparte proporcionalmente
   entre las posiciones que no tocan el tope. Con los defaults, 5 × 20 % = 100 %.
6. **Costes**: `COMMISSION_BPS` = 5, `SLIPPAGE_BPS` = 10 sobre el nocional de cada
   compra/venta. La alfa se reporta siempre neta.

**Salidas dentro del `run_dir` de agentes**:

- `positions.parquet`: `snapshot_date × ticker` con `weight`, `entry_date`, `months_held`,
  `current_percentile`.
- `orders.parquet`: cada compra/venta con `side`, `weight_before`, `weight_after`,
  `price`, `commission`, `slippage`, `reason` (`initial_fill`, `hole_filled_after_drop`,
  `edge_over_worst`, `displaced_by_edge`, `dropped_below_min`, `rebalance`).
- `equity.parquet`: `portfolio_value`, `benchmark_value`, `portfolio_return`,
  `benchmark_return`, `excess_return`, `turnover_pct` por snapshot.
- `annual_metrics.parquet`: `alpha`, `beats_benchmark`, `max_drawdown_year`,
  `information_ratio_year` por año.
- `backtest_summary.json`: señales de aprendizaje (`mean_rank_ic`,
  `rank_ic_positive_fraction`, `rank_ic_std`), consistencia y riesgo (`beat_rate`,
  `max_drawdown`) y alfa informativa (`annualized_alpha`, `median_alpha`, `worst_year_alpha`).
  Las cuatro primeras alimentan la selección de Fase 6; el alfa solo se reporta.
- `manifest.json` actualizado con los parámetros de cartera y costes usados.

**Tests** (`tests/backtest/`):
- Reglas: ventaja insuficiente no rota; expulsión sin sustituto deja hueco; ida y vuelta
  (entrar-salir-volver) sin regla de tenencia mínima; sizing respeta tope y suma 1;
  tamaño flexible con pocos candidatos.
- Invariantes: mutar scores futuros no cambia posiciones ni equity pasados; contabilidad
  cuadra (`equity[t] − equity[t−1]` = retorno de posiciones − costes del día).

## Fase 5 — Informe HTML

Implementada como dos productos en `module/report.py`:

**A) HTML por run** (`build_run_report(run_dir)` → `run_dir/report.html`). Seis hojas
navegables por pestañas:

1. **Resumen** — alfa total, beat rate, IR, drawdown; curva de equity y barras de alfa anual.
2. **Rendimiento** — drawdown continuo y tabla anual completa (retorno, alfa, drawdown, IR).
3. **Aprendizaje** — rank-IC por agente y por revisión, evolución de pesos del meta-agente.
4. **Cartera** — composición actual y turnover reciente; los históricos completos se sirven
   como CSVs sueltos al lado (`positions_history.csv`, `orders_history.csv`) que la
   página carga por `fetch`.
5. **Cobertura** — sesgo de supervivencia leído de `universe_coverage.json` +
   prohibiciones metodológicas activas (sector excluido, `payload.metric` prohibido, etc.).
6. **Posiciones** — cuánto rentó cada ticker mientras estuvo en cartera. Respuesta directa
   a *"¿el sistema mantiene ganadores o corta demasiado pronto?"*.

**B) HTML del barrido** (`build_comparison_report(scenarios_root)` →
`scenarios_root/comparison.html`). Cinco hojas: ranking, heatmap `escenario × año` de
alfa, sensibilidad por parámetro, selección justificada con validación en la era
reservada, tabla completa con enlaces a cada run.

**Gráficos**: `matplotlib` → PNG embebido base64. Se añade `matplotlib>=3.7` a
`requirements.txt`.

**Tests** (`tests/report/`): el HTML contiene los IDs de las seis hojas; las cifras del
Resumen cuadran con `backtest_summary.json`; el ganador del barrido es determinista según
el rango medio (test con escenarios sintéticos donde el estable gana al que tiene un solo
año excepcional).

## Fase 6 — Rejilla de escenarios y selección automática

Implementada en `module/experiments.py`. Los escenarios se definen en `escenarios/*.py` (no
YAML/JSON) como listas de `ScenarioSpec(name, overrides)`, para poder incluir listas y
cálculos derivados. La rejilla base vive en `escenarios/rejilla_base.py` con ~17 escenarios
distribuidos entre ancla temporal, entrenamiento, cadencia, horizonte de etiqueta,
políticas de cartera y observabilidad.

**Reutilización por huella SHA-256**. Cada etapa (`dataset`, `features`, `agents`,
`backtest`) declara qué campos de `Settings` la afectan; se hashean y comparan entre
escenarios. Si coinciden, el escenario reutiliza el artefacto compartido vía symlink en
lugar de regenerarlo. Los escenarios que solo cambian política de cartera (`portfolio_3_7`,
`rotation_strict`, etc.) reusan `dataset/features/agents` del baseline y solo simulan otro
backtest.

**Estructura de resultados** en `results/escenarios/`:

- `<nombre>/processed/` con los parquets de dataset/features (compartidos por symlink).
- `<nombre>/agents/<run_id>/` con scores, pesos, backtest, `report.html` y CSVs.
- `<nombre>/scenario_config.json` documenta overrides, huellas y qué reutilizó.
- `scenarios_summary.parquet` y `.csv` con el ranking.
- `selection.json` con el ganador, sus 4 rangos, señales de aprendizaje y alfa (informativa).
- `comparison.html` con las 5 hojas del barrido.

**Selección por aprendizaje y estabilidad, NUNCA por alfa** (ver `docs/doc.md` §8). Cuatro
dimensiones, calculadas sobre **todos los años disponibles** (un único ranking global, sin
separación en eras):

1. `mean_rank_ic` — rank-IC medio OOS (evidencia de aprendizaje). Mayor es mejor.
2. `rank_ic_positive_fraction` — fracción de cohortes con rank-IC positivo (estabilidad del
   aprendizaje entre eras). Mayor es mejor.
3. `beat_rate` — fracción de años que baten SPY (frecuencia, no magnitud). Mayor es mejor.
4. `max_drawdown` — máximo drawdown (riesgo). Menor es mejor.

Se rankea cada escenario en las cuatro y el ganador es el que **minimiza el rango medio**. El
alfa (anualizada, mediana, peor año) se **reporta** en el HTML y el `selection.json` como
consecuencia, pero **no participa en el ranking**: elegir por rentabilidad cuando el rank-IC
es débil sería seleccionar ruido. Si el rank-IC del ganador es ≈0, la conclusión honesta es
que el sistema no aprende a ordenar de forma estable, y se reporta como tal.

**Tests** (`tests/experiments/`): huellas deterministas y aisladas por etapa (cambiar
`target_max` NO cambia la huella de dataset/features/agents); el ganador es el que aprende
(mayor rank-IC + estabilidad), y el alfa **no** altera el ranking (test con un escenario de
alfa alto pero rank-IC nulo que debe perder ante otro que aprende).

## Fase 7 — Redacción del TFM en LaTeX

**Objetivo**: producir el documento entregable a partir de lo ya construido e implementado en
las fases anteriores, no en paralelo a ellas. Es la última fase porque necesita resultados
reales (Fases 4-6) para los capítulos de resultados y conclusiones; los capítulos que solo
dependen de metodología (introducción, datos, diseño) pueden escribirse antes si conviene.

**Estructura del trabajo**: un plan general y vivo en `latex/plan_tfm.md` (markdown, no LaTeX)
que fija el índice completo del TFM, capítulo por capítulo, antes de escribir ningún `.tex`.
Ese plan es la referencia para mantener coherencia entre capítulos escritos en sesiones
distintas. Un fichero `.tex` por capítulo en `latex/`, pensado para pegar directamente en
Overleaf. El detalle de la estructura, el índice de capítulos y las convenciones de estilo se
acuerdan aparte, en `latex/plan_tfm.md`.

**Mecánica de trabajo**: el autor pide un capítulo o sección concreta; se escribe apoyándose en
`latex/plan_tfm.md`, los capítulos ya redactados (para mantener terminología, notación y tono
consistentes) y el estado real del proyecto en ese momento (código, tests, resultados de las
fases ya cerradas) — nunca inventando resultados que la fase correspondiente aún no ha
producido.

---

## Limitaciones a documentar (medidas, no estimadas)

1. **Sesgo de supervivencia parcialmente irreducible.** El universo dinámico elimina el sesgo
   de *inclusión anticipada*, pero los quebrados no tienen datos en fuentes gratuitas: el panel
   de 2000 cubrirá ~40 % del índice real. **El backtest pre-2010 está sesgado al alza** y el
   sesgo crece hacia atrás. Se reporta por año.
2. **Restatements invisibles**: las series de Finnhub son valores actuales; si una empresa
   reexpresó cuentas de 2008, vemos la cifra corregida. EDGAR fecha *cuándo se publicó*, pero
   el *valor* viene de Finnhub. Lookahead residual **no eliminable** sin `companyfacts` (que
   empieza en 2009).
3. **Cobertura desigual**: `gross_margin` ~430/503, `eps` 486 (JPM no tiene `eps` quarterly).
   Los NA no son aleatorios (sesgados por sector, p. ej. bancos) → **nunca imputar con la media**.
4. **Muestra pequeña**: pocas eras independientes; intervalos de confianza anchos.

## Orden de ejecución

Las fases se abordan **una a una**, cada una diseñada y aprobada antes de implementarse
(`CLAUDE.md`). Al cerrar cada fase se actualizan `docs/doc.md`, `README.md` y la tabla de
estado de este documento.

La descarga completa (Fase 0) la lanza el autor con `python main.py` (~40-70 min por el rate
limit de las APIs): `CLAUDE.md` desaconseja ejecutar descargas largas sin autorización
explícita.
