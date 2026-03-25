# Documentación completa y detallada del proyecto TFM

Propósito: describir de forma exhaustiva el flujo, la lógica temporal, los módulos clave y el contenido exacto de los outputs generados por el pipeline. Esta versión está pensada para auditoría técnica, reproducibilidad y para que cualquier ingeniero entienda y reproduzca el pipeline sin ambigüedades.

Contenido de este documento:
- Resumen ejecutivo
- Variables y configuración clave (archivo `environment.py`)
- Flujo técnico paso a paso (descarga → consolidación → dataset → modelado → evaluación → backtest → outputs)
- Lógica temporal detallada (cómo se forman folds, qué significa "analizar Qx", modos de lag)
- Inventario exhaustivo de outputs: por archivo/patrón, campo por campo, tipos y ejemplo de fila
- Donde mirar en el código para cada output (mapeo archivo → módulo)
- Glosario técnico

--

**Resumen ejecutivo**

El pipeline toma datos crudos (Finnhub y precios), los transforma en snapshots trimestrales (features + metadatos), entrena agentes por fold temporal y un meta-learner, y ejecuta un backtest walk-forward. Cada fold de test está definido por el `filedDate` de los reportes (i.e., "analizar 2026Q1" significa usar reportes con `filedDate` dentro de 2026Q1). 

**Cálculo de entrada (entry_date)**: El día de entrada de cada fold se calcula determinísticamente como `primer_día_del_quarter + MAX_SNAPSHOT_LAG_DAYS` (p. ej., 1 enero + 60 días = ~2 marzo para Q1). Esto garantiza una ventana de precios uniforme y evita discontinuidades causadas por filing dates variables. El sistema loguea claramente la fecha mínima requerida de precios para cada fold.

**Dónde empezar**: `analyzer.py` (punto de entrada), `environment.py` (configuración), `module/steps/step_01_data/*` (descarga y consolidación), `module/steps/step_02_dataset/*` (dataset), `module/steps/step_03_training/*`, `module/steps/step_04_evaluation/*`.

--

**Variables y configuración clave (archivo `environment.py`)**

- `SNAPSHOT_LAG_DAYS: Optional[int]` — **Parámetro heredado, ya no se utiliza en el cálculo actual de entry_date**. Mantiene compatibilidad backward pero no afecta la lógica temporal.
- `AUTO_SNAPSHOT_LAG_MARGIN_DAYS: int` — Parámetro heredado. Se utiliza ahora solo en `_recompute_forward_returns` para ajustes de post-filing. En el cálculo de entry_date de fold, ya no efectivo.
- `MAX_SNAPSHOT_LAG_DAYS: int` — **Parámetro activo y crítico**. Define el número máximo de días que se suma al primer día del quarter para calcular `entry_date`. Defecto: 60 días. Determinante directo de la ventana de precios disponible.
- `HOLDING_PERIOD_MONTHS: int` — duración teórica de la posición desde la entrada.
- `MIN_TEST_TICKERS_PERCENT: int` — porcentaje mínimo del universo total de tickers requerido en el test de un fold. Si el quarter analizado tiene menos de este porcentaje, el fold se descarta. Ejemplo: si hay 500 tickers totales, 50% = 250 mínimo requerido.
- `ENABLE_FALLBACK_EXTRAPOLATION: bool` — si `True`, cuando una empresa no tenga reporte del quarter exacto analizado, se estiman sus features promediando los últimos `FALLBACK_LOOK_BACK_QUARTERS` snapshots históricos.
- `FALLBACK_LOOK_BACK_QUARTERS: int` — número de quarters previos a usar para extrapolación de features cuando falte el reporte exacto.
- `FORCE_DOWNLOAD: bool` — si `True`, borra `data_finnhub/_registry.json` para forzar re-descarga; NO borra los archivos JSON crudos.

Notas operativas:
- La descarga siempre se ejecuta hasta la fecha "hoy" para asegurar cobertura de precios en folds recientes.
- El cálculo de entry_date es determinista: `quarter_start + MAX_SNAPSHOT_LAG_DAYS`, sin dependencias de filing dates específicas por ticker.

Mejoras recientes orientadas a Outperform (selección de features y stacking):
- `FEATURE_SELECTOR_RELEVANCE_WEIGHT: float` — peso de relevancia directa con `y` en el selector (default 0.65).
- `FEATURE_SELECTOR_RF_N_ESTIMATORS: int` — árboles del RF auxiliar usado para importancia en el selector.
- `FEATURE_SELECTOR_RF_MAX_DEPTH: int` — profundidad del RF auxiliar del selector.
- `META_ENABLE_CONSENSUS_FEATURES: bool` — activa features de consenso/confianza entre agentes en el meta learner.
- `META_BULLISH_SCORE_THRESHOLD: float` — umbral para contar agentes claramente alcistas en `bullish_agent_score_count`.

Resumen funcional:
- El selector de cada agente ahora rankea por score combinado:
	`combined = w * norm(|pb_y|) + (1 - w) * norm(importance_rf)`
	para priorizar señales alineadas con Outperform sin depender solo de importancias relativas bajo colinealidad.
- El meta learner añade señales de consenso entre agentes (media, dispersión, rango,
	recuento de agentes alcistas, fuerza de consenso y media ponderada por convicción).

--

**Lógica temporal (explicación técnica y ejemplos)**

1) ¿Qué significa "analizar 2026Q1"?
- Significa: seleccionar para el test de ese fold todos los snapshots cuyos reportes tengan `filedDate` (o `acceptedDate` cuando `filedDate` falte) dentro del rango 2026-01-01 a 2026-03-31.

2) Determinación de la `entry_date` del fold:
- El sistema calcula siempre: `entry_date = quarter_start + MAX_SNAPSHOT_LAG_DAYS`
- Esto significa: para un quarter analizado (ej. 2026Q1 = ene-mar), se comienza a contar desde el primer día del quarter (2026-01-01), y se suma `MAX_SNAPSHOT_LAG_DAYS` (60 días por defecto).
- **Nota**: `SNAPSHOT_LAG_DAYS` es ahora un parámetro heredado y no afecta el cálculo actual. El lag es determinista y consistente entre todos los folds.
- Este enfoque garantiza una ventana de precios suficientemente amplia sin depender del calendario específico de filing de cada empresa.
- En los logs verá: `[Fold X] Snapshot lag empieza en: 1 de enero de 2026 (primer día del Q1) + 60 días máximo = fecha mínima de precios requerida: 2 de marzo de 2026`

**Nota sobre lags uniformes**: Todas las empresas del mismo quarter comparten la misma `entry_date`, determinada por el primer día del quarter + MAX_SNAPSHOT_LAG_DAYS. Esto simplifica la lógica temporal y evita que folds recientes con filing dates irregulares generen ventanas de backtest vacías.

Ejemplo rápido:
- Quarter a analizar: 2026Q1 (enero-marzo)
- Primer día del quarter: 2026-01-01
- MAX_SNAPSHOT_LAG_DAYS: 60
- Fecha mínima requerida de precios: 2026-01-01 + 60 = 2026-03-01 (aprox.)
- Todos los tickers analizados en Fold 2026Q1 usarán esta `entry_date`
- exit_date_theoretical = entry_date + HOLDING_PERIOD_MONTHS

3) **Extrapolación de features para snapshots faltantes** (si `ENABLE_FALLBACK_EXTRAPOLATION=True`):
- Para tickers **sin reporte exacto** del quarter analizado pero con al menos `FALLBACK_LOOK_BACK_QUARTERS` (defecto 4) reports históricos anteriores:
- Se promedian los features numéricos de esos últimos quarters.
- Se crea un snapshot "estimado" para el quarter del análisis, usando ese promedio.
- Esto aumenta el universo de test sin esperar nuevos reportes.
- **Beneficio**: incluye empresas con retraso en reportes o ciclos de filing irregulares.
- **Nota técnica**: las features se estiman por promedio histórico, pero el `forward_return` se calcula normalmente con precios reales en la ventana entry→exit.

--

Sección siguiente: INVENTARIO DETALLADO DE OUTPUTS (campo a campo). Este bloque describe el nombre exacto de archivos/patrones, su lugar en el repositorio, el módulo que los genera, los campos que contienen, el tipo esperado y un ejemplo sintético de una fila.

**IMPORTANTE**: si encuentra un output adicional en su ejecución, indíquemelo y lo añado aquí con sus campos reales (puedo extraer ejemplos de su `results/` si lo desea).

--

**Inventario de outputs — formato por entrada:**

- Nombre/Patrón: ruta relativa (generador)
- Campos: `campo: tipo — descripción` + ejemplo de fila (valores ficticios coherentes)

1) Pipeline log maestro
- `results/pipeline.log` (generado por `analyzer.py` y logger central)
- Contenido (texto libre línea a línea). Ejemplo de líneas relevantes:
	- [INFO] Download start: 2026-03-21 07:00
	- [INFO] Folds generated: 2018Q1..2026Q1 (N folds)
	- [WARN] Fold 2020Q2 omitted: MIN_TEST_TICKERS_PER_FOLD

2) Master dataset
- `results/master_dataset.csv` (generado por `module/steps/step_02_dataset/dataset.py`)
- Campos (ejemplo mínimo común):
	- `ticker: string — símbolo` — "AAPL"
	- `snapshot_end_date: date — fecha de corte del informe` — "2025-12-31"
	- `filed_date: date — fecha de presentación del reporte` — "2026-02-15"
	- `feature_XYZ: float — ejemplo, net_income_margin` — 0.12
	- `sector: string — sector del ticker` — "Technology"
	- `price_entry: float — precio en fecha de entrada` — 145.32
	- `price_exit: float — precio en fecha de salida (si existe)` — 158.21
	- `forward_return: float — (price_exit/price_entry - 1)` — 0.088
	- `fold: string — id del fold de test asignado` — "2026Q1"

3) Registro de descargas
- `data_finnhub/_registry.json` (generado/actualizado por `module/steps/step_01_data/registry.py`)
- Estructura:
	- keys: ticker
	- value: dict de endpoints con `last_downloaded` timestamp y `etag`/`checksum` opcionales.
- Ejemplo:
	{
		"AAPL": {
			"prices": {"last_downloaded": "2026-03-21T07:15:00Z"},
			"financials_reported_quarterly": {"last_downloaded": "2026-03-21T07:20:00Z"}
		}
	}

4) JSON crudos por ticker
- `data_finnhub/<TICKER>/<endpoint>.json` (descargados por `module/steps/step_01_data/downloaders.py`)
- Contenido: payload del endpoint Finnhub/OAuth con metadatos. Campos clave que usa el pipeline:
	- `endDate` (fecha del periodo contable)
	- `filedDate` o `acceptedDate` (fecha de presentación)
	- `reportedCurrency`, `symbol`, `values` (donde vienen las medidas)

Ejemplo (simplificado) `financials_reported_quarterly.json`:
	{
		"symbol": "AAPL",
		"report": [
			{"endDate": "2025-12-31", "filedDate": "2026-02-15", "grossProfit": 120000000},
			...
		]
	}

5) CSV consolidado por ticker
- `data_finnhub/consolidated/<TICKER>_consolidated.csv` (generado por `consolidation.py`)
- Campos típicos:
	- `endDate: date` — periodo del snapshot (e.g., 2025-12-31)
	- `filedDate: date` — fecha de filing
	- `totalRevenue: float`
	- `netIncome: float`
	- `eps: float`
	- `derived_ratios.*` (ROA, ROE, margin, etc.)

6) Scores por fold (detallado)
- `results/agents/fold_{FOLD_ID}/fold_{FOLD_ID}_scores.csv` (generado por `evaluator.py` y agentes)
- Campos:
	- `ticker: string` — "AAPL"
	- `snapshot_end_date: date` — "2025-12-31"
	- `filed_date: date` — "2026-02-15"
	- `agent_X_score: float` — score del agente X (puede ser z-score o probabilidad)
	- `meta_score: float` — score final combinado (0..1)
	- `rank: int` — ranking descendente por `meta_score`
	- `selected: bool` — si entra en cartera top-N o por threshold
	- `price_entry: float` — precio usado para calcular forward
	- `price_exit: float | null` — precio de salida real (si hay)
	- `forward_return: float | null` — etiqueta/retorno calculada

Ejemplo fila:
	AAPL,2025-12-31,2026-02-15,0.12,0.78,1,True,145.32,158.21,0.088

7) Selection audit por fold
- `results/agents/fold_{FOLD_ID}/fold_{FOLD_ID}_selection_audit.csv` y `.json`
- Contenido: por ticker, motivos de inclusión/exclusión, checks fallidos, flags de calidad.
- Campos resumen:
	- `ticker, reason_included, reason_excluded, passed_checks` (lista)

8) Explicaciones por ticker (SHAP / razones)
- `results/agents/fold_{FOLD_ID}/fold_{FOLD_ID}_ticker_explanations.csv` y `.json`
- Campos principales:
	- `ticker, feature, shap_value, contribution_pct` — distribución de contribución de features.

9) Diagnostics agentes
- `results/agents/<agent>/diagnostics_fold_{FOLD_ID}.json`
- Contiene: parámetros de entrenamiento, curva OOF, overfitting metrics, feature importance agregada.

10) Backtest por fold (resultado detallado)
- `results/backtest/fold_{FOLD_ID}_metrics.json` (generado por `backtester.py`)
- Campos:
	- `fold_id: string`
	- `entry_date: date`, `exit_date_theoretical: date`, `exit_date_real: date` (último precio usado)
	- `n_positions: int` — número medio de posiciones (o top-N)
	- `strategy_cumulative_return: float`
	- `benchmark_cumulative_return: float`
	- `alpha: float`, `sharpe: float`, `max_drawdown: float`, `volatility: float`

Ejemplo (JSON):
	{
		"fold_id": "2026Q1",
		"entry_date": "2026-03-27",
		"exit_date_theoretical": "2026-06-27",
		"exit_date_real": "2026-06-25",
		"strategy_cumulative_return": 0.12,
		"benchmark_cumulative_return": 0.05,
		"alpha": 0.07,
		"sharpe": 1.2
	}

11) Backtest summary global
- `results/backtest/backtest_summary.json` / `results/backtest/folds_results.csv`
- Campos agregados:
	- `fold_id, strategy_return, benchmark_return, alpha, sharpe, n_positions, pct_test_universe_used`

12) Plots y figuras
- Carpeta: `results/plots/` (generados por `visualization.py`)
- Patrones: `score_dist_{fold}.png`, `feat_imp_{agent}_{fold}.png`, `fold_{fold}_performance.png`

13) Live predictions
- `results/predictions_LIVE.csv` y `.json` (si se ejecuta modo live)
- Campos:
	- `timestamp, ticker, meta_score, rank, suggested_size, notes`

--

**Mapeo rápido: qué módulo produce cada artifacto**

- Descarga JSON por ticker: `module/steps/step_01_data/downloaders.py`
- Registry: `module/steps/step_01_data/registry.py`
- Consolidated CSV: `module/steps/step_01_data/consolidation.py`
- Master dataset CSV: `module/steps/step_02_dataset/dataset.py`
- Training / OOF / Agents: `module/steps/step_03_training/*` y `module/agents/*`
- Scores / selection audit / explanations: `module/steps/step_04_evaluation/evaluator.py` + `selection_reports.py` + `explainability.py`
- Backtest / metrics: `module/steps/step_04_evaluation/backtester.py` + `metrics.py`
- Visuals: `module/steps/step_04_evaluation/visualization.py`

--

**Checklist rápido para auditar un output**

1. Identificar archivo en `results/`.
2. Localizar módulo generador usando el mapeo anterior.
3. Abrir el log `results/pipeline.log` para ver la iteración/fold que generó ese archivo (timestamp y fold id).
4. Revisar `fold_{FOLD_ID}_selection_audit.csv` para ver por ticker cómo se construyó la selección.
5. Revisar `fold_{FOLD_ID}_ticker_explanations.*` para entender drivers de score.

--

**Glosario operativo (resumen)**

- Snapshot: observación por ticker asociada a un `endDate` contable y `filedDate`.
- Filed quarter: quarter en que cae el `filedDate`.
- Fold: unidad de walk-forward; test = snapshots con `filedDate` en el quarter.
- Entry date: fecha de inicio de exposición calculada según modo lag.
- Exit date theoretical / real: fecha objetivo de salida y la última fecha con precio disponible.

--

Si quieres que haga lo siguiente ahora, dime cuál prefieres (elige una):
1) Extraer ejemplos reales de archivos en `results/` y añadirlos aquí (necesitaré permiso para leer `results/`).
2) Añadir una sección "strict filing mode" que implemente la política de excluir snapshots sin `filedDate` y documentar efectos (lo implemento en el código si confirmas).
3) Generar un README reducido con los pasos mínimos para ejecutar el pipeline y reproducir un fold.

Fin del documento ampliado.

--

**Sección técnica: detalle por módulo, funciones clave y outputs (campo por campo)**

Nota: abajo se listan funciones, inputs, outputs y ejemplos sintéticos por módulo. Sirve para rastrear la generación de cada archivo y entender exactamente qué contiene.

1) `analyzer.py` — Orquestador
- Responsabilidad: parsea argumentos/config, prepara directorios, lanza las etapas en orden: descarga → consolidación → dataset → training → evaluación → backtest → export.
- Flags CLI/Entradas relevantes:
	- `--force-download` (bool) — fuerza limpieza del registry.
	- `--start-quarter`, `--end-quarter` — rango de folds a ejecutar.
	- `--snapshot-lag-days` — override local para lag fijo (None = auto).
- Flujo interno (funciones clave):
	- `main()` — valida config y llama `run_walkforward_pipeline(...)`.
	- `run_walkforward_pipeline(...)` — coordina las llamadas a `download_step`, `consolidate_step`, `build_dataset`, `train_step`, `evaluate_step`, `backtest_step`.
- Outputs que referencia/crea: inicializa `results/pipeline.log` y pasa control a los módulos que generan el resto.

2) `environment.py` — Config global
- Contiene variables con tipos y explicación. Ejemplos:
	- `SNAPSHOT_LAG_DAYS: Optional[int]` — int|None
	- `AUTO_SNAPSHOT_LAG_MARGIN_DAYS: int` — margen días para modo auto
	- `HOLDING_PERIOD_MONTHS: int` — meses de holding
	- `MIN_TEST_TICKERS_PER_FOLD: int` — umbral mínimo
	- `DATA_DIR: str` — `data_finnhub/`
	- `RESULTS_DIR: str` — `results/`

3) `module/steps/step_01_data/downloaders.py` — Descarga
- Endpoints descargados por ticker (nombres de archivos y ubicación):
	- `<DATA_DIR>/<TICKER>/prices.json` — precios (OHLCV) usados por backtester
	- `<DATA_DIR>/<TICKER>/financials_reported_quarterly.json` — reporte trimestral (contiene endDate, filedDate/acceptedDate)
	- `<DATA_DIR>/_macro/sp500.json` — benchmark
- Comportamiento importante:
	- Si `FORCE_DOWNLOAD`, llama `registry.clear(delete_file=True)` para eliminar `_registry.json`.
	- Los JSON descargados se escriben con esquema original del proveedor.

4) `module/steps/step_01_data/registry.py` — Registry
- Archivo: `data_finnhub/_registry.json`
- Estructura JSON: `{ticker: {endpoint: {last_downloaded: str, etag?: str}}}`
- Funciones:
	- `registry.set(ticker, endpoint, meta)` — actualiza timestamp
	- `registry.clear(delete_file=False)` — limpia entradas; si `delete_file=True` borra el fichero físicamente

5) `module/steps/step_01_data/consolidation.py` — Consolidación
- Objetivo: convertir payloads crudos en tablas tabulares por ticker con filas por `endDate` (snapshots).
- Funciones clave:
	- `parse_financials_quarterly(raw_json)` → lista/dict con `endDate`, `filedDate`, métricas contables.
	- `standardize_accounting_fields(df)` → renombra columnas a estandar.
	- `derive_derived_ratios(df)` → calcula ROE, ROA, margins, y otros features contables.
- Salida: `data_finnhub/consolidated/<TICKER>_consolidated.csv` con columnas (ejemplo):
	- `ticker, endDate, filedDate, totalRevenue, netIncome, eps, roa, roe, grossMargin, currentRatio`

6) `module/steps/step_02_dataset/dataset.py` — Construcción del dataset maestro
- Resumen: une consolidated CSVs con signals técnicos/insider/sentiment y genera `results/master_dataset.csv`.
- Funciones clave:
	- `build_master_dataset(consolidated_dir, prices_dir)` — itera tickers, construye filas.
	- `align_feature_date(snapshot_row, lag_mode)` — calcula `feature_date` y `price_entry_date` según lag.
	- `compute_forward_return(prices, entry_date, exit_date)` — devuelve return y coverage flag.
- Campos de `master_dataset.csv` (lista ampliada):
	- `ticker: str`
	- `snapshot_end_date: date` (endDate)
	- `filed_date: date`
	- `feature_*` — todos los features numéricos (netIncome, eps, roa, momentum_30, vol_90, insider_buy_pct, etc.)
	- `sector: str`
	- `price_entry_date: date`, `price_entry: float`
	- `price_exit_date: date | null`, `price_exit: float | null`
	- `forward_return: float | null`
	- `fold: str` (ej. 2026Q1)
	- `label_available: bool` — indica si forward_return fue calculado con cobertura suficiente

7) Builders de features (detalle por archivo)
- `builders/fundamental.py`: crea features contables (margen, crecimiento YoY, leverage ratios). Ejemplo salida: `net_income_margin`, `rev_yoy`.
- `builders/technical.py`: indicadores de precios (momentum_30, rsi_14, vol_90).
- `builders/valuation.py`: ratios PER, P/B, EV/EBITDA (normalizados por sector).
- `builders/insider.py`: agregados de transacciones insiders (`insider_buy_volume_90d`).
- `builders/sentiment.py`: `eps_surprise`, `analyst_reco_change_count`.

8) `module/steps/step_03_training/training.py` — Entrenamiento de agentes
- Output por agente (por fold):
	- `results/agents/<agent>/train_history_fold_{FOLD}.json` — parámetros + curva de loss
	- `results/agents/<agent>/feature_importances_fold_{FOLD}.csv` — columnas `feature, importance`
	- OOF preds: `results/agents/<agent>/oof_predictions_{FOLD}.csv` (si aplica)
- Funciones: `train_agent(agent_config, train_df)` retorna modelo y diagnóstico.

9) `module/steps/step_04_evaluation/evaluator.py` — Fold generation y scoring
- Funciones críticas:
	- `_build_filing_date_map(data_finnhub_dir)` — itera `financials_reported_quarterly.json` y construye map `ticker -> {endDate: filedDate}`.
	- `_prepare_folds_by_filed_quarter(master_dataset)` — agrupa snapshots por `filedDate` quarter y genera train/test splits por fold.
	- `_resolve_entry_date_for_fold(test_filed_dates, SNAPSHOT_LAG_DAYS, AUTO_SNAPSHOT_LAG_MARGIN_DAYS)` — si `SNAPSHOT_LAG_DAYS is None` hace `entry = max(filedDate) + margin`.
	- `_recompute_forward_returns_for_fold(...)` — recalcula etiquetas usando los precios entre `entry` y `exit`.
- Outputs:
	- `results/agents/fold_{FOLD}/fold_{FOLD}_scores.csv`
	- `results/agents/fold_{FOLD}/fold_{FOLD}_selection_audit.csv|.json`
	- `results/agents/fold_{FOLD}/fold_{FOLD}_ticker_explanations.*`

10) `module/steps/step_04_evaluation/backtester.py` — Simulación
- Funciones:
	- `simulate_portfolio(signals_df, prices_df, entry_date, exit_date)` — aplica reglas de sizing y turnover
	- `compute_metrics(returns_series, benchmark_series)` — devuelve sharpe, alpha, mdd
- Métricas definidas y fórmulas (resumen):
	- `cumulative_return = (1 + r1)*(1 + r2)*... - 1`
	- `excess_return = strategy_return - benchmark_return`
	- `sharpe = mean(returns)/std(returns) * sqrt(annualization)` (nota: usar period correct)
	- `max_drawdown = max_peak - subsequent_trough` (porcentaje)

11) `module/steps/step_04_evaluation/explainability.py` — SHAP y explicaciones
- Salidas:
	- `shap_global_fold_{FOLD}.csv` — `feature, mean_abs_shap, direction`
	- `shap_per_ticker_fold_{FOLD}.csv` — `ticker, feature, shap_value`

12) Visualización y reports
- `visualization.py` genera PNGs en `results/plots/` con títulos estandarizados que incluyen `fold_id` y fecha de ejecución.

13) Estructura `results/` y ejemplos concretos de filas
- `results/master_dataset.csv` (fila ejemplo):
	AAPL,2025-12-31,2026-02-15,0.12,Technology,145.32,158.21,0.088,2026Q1,True
- `results/agents/fold_2026Q1/fold_2026Q1_scores.csv` (fila ejemplo):
	AAPL,2025-12-31,2026-02-15,0.12,0.05,0.02,0.78,1,True,145.32,158.21,0.088
	(columns: ticker, snapshot_end_date, filed_date, agent1_score, agent2_score, agent3_score, meta_score, rank, selected, price_entry, price_exit, forward_return)
- `results/backtest/fold_2026Q1_metrics.json` (ejemplo ya mostrado arriba).

--

Si quieres que lo haga ahora, puedo:
- A) Añadir ejemplos reales leyendo `results/` y rellenar los ejemplos con datos reales (solo lectura). — esto haría el documento 100% exacto.
- B) Mantener el documento con ejemplos sintéticos y seguir ampliando más módulos concretos si me indicas prioridades.

Dime si quieres la opción A (extraer ejemplos reales), que actualizaré el documento con filas reales tomadas de `results/`.

Fin de la sección técnica añadida.
