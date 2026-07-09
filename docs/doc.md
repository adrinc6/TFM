# Documentación exhaustiva — GARP AI Portfolio System

## 1. Objetivo del proyecto

Este proyecto (TFM) construye y evalúa empíricamente un sistema de inteligencia artificial
explicable para investigar una pregunta concreta:

> ¿Puede una estrategia GARP (Growth At a Reasonable Price / Value-Growth), operada por varios
> agentes ML especializados más un meta-agente que aprende cómo combinarlos, generar alpha frente
> a SPY, usando datos point-in-time (sin lookahead) y una cartera concentrada gestionada
> mensualmente?

El sistema no intenta predecir precios a corto plazo ni hacer trading de alta frecuencia. Busca
identificar empresas de **calidad** (negocio sólido, ROIC/márgenes altos y sostenibles), con
**crecimiento** razonable, cuya **valoración** actual no refleje todavía esa mejora futura —
"infravaloradas de calidad" — y gestionarlas como una cartera viva con tesis de inversión
explícitas que se revisan periódicamente.

## 2. Filosofía de inversión y arquitectura de decisión

El sistema separa deliberadamente **dos capas de puntuación** que nunca se colapsan en una sola:

1. **Capa estadística** (`module/ml.py`): `final_score` / `opportunity_type`, producidos por 4
   agentes ML especializados + 1 meta-agente que aprende sus pesos de combinación.
2. **Capa de gestión** (`module/strategy/portfolio.py`): `manager_score`, una combinación
   ponderada a mano de `final_score` más overlays de timing/valoración/riesgo, que es la que
   realmente decide entradas y salidas de la cartera.

Esta separación es intencional: permite comparar "qué diría el modelo puro" frente a "qué hace el
gestor" (el análisis de coste de oportunidad en `module/backtest/reviews.py` depende de esta
divergencia).

## 3. Arquitectura del pipeline

```text
download → dataset → features → ml → watchlist → research_ai → backtest → viewer → report
```

Orquestado por `main.py`, gateado por `settings.run_mode` (o `"full"` para ejecutar todo). Cada
etapa lee/escribe bajo `Settings.run_dir` = `results/<dev|full>_<inicio>_<fin>_<frecuencia>_cutoff<fecha_corte>/`
(`environment.py`), de modo que cambiar fechas, `DEV_MODE` o el esquema de entrenamiento apunta a
una carpeta de resultados distinta en vez de sobrescribir un run anterior.

| Etapa | Módulo | Entrada | Salida |
|---|---|---|---|
| download | `module.ingest.pipeline` | APIs Finnhub/Yahoo + caché JSON | `data/raw/{profiles,finnhub_metrics,prices,news}.parquet` |
| dataset | `module.dataset` | `data/raw/*.parquet` | `data/master/master_point_in_time.parquet` |
| features | `module.features.pipeline` | dataset maestro | `data/processed/features.parquet` |
| ml | `module.ml` | features + precios | `data/processed/scored_universe.parquet`, diagnósticos walk-forward |
| watchlist | `module.strategy.selection` | universo puntuado | `data/processed/watchlist.parquet`, `results/<run>/watchlist.csv` |
| research_ai | `module.research.ai` | universo puntuado + noticias | `results/<run>/research_ai.csv` |
| backtest | `module.backtest` | universo puntuado + precios | ~20 CSV en `results/<run>/` y `.../audit/` |
| viewer | `module.viewer` | CSVs del run | `results/<run>/viewer/index.html` |
| report | `module.report` | viewer ya construido | apunta al mismo `index.html` |

## 4. Configuración (`environment.py`)

No hay CLI/argparse: todo se edita directamente como constantes en `environment.py`. `.env` solo
guarda `FINNHUB_API_KEY` y `OPENAI_API_KEY` (parseado a mano, sin `python-dotenv`).

Parámetros clave:

- `RUN_MODE`, `DEV_MODE`, `DEV_TICKERS`, `TICKERS` (universo completo estático), `BENCHMARK_TICKER`.
- `DATA_START_DATE`, `PORTFOLIO_START_DATE`, `PORTFOLIO_END_DATE`, `PORTFOLIO_REVIEW_FREQUENCY`.
- `WALK_FORWARD_SCORING`, `WALK_FORWARD_LABEL_HORIZON_MONTHS` (12 por defecto).
- `MIN_WALK_FORWARD_TRAINING_ROWS`, `MIN_WALK_FORWARD_TRAINING_YEARS`, `MAX_WALK_FORWARD_TRAINING_YEARS`
  (ventana móvil máxima de historia usada en cada reentrenamiento individual).
- `TRAIN_CUTOFF_DATE` (por defecto = `PORTFOLIO_START_DATE`), `WALK_FORWARD_TRAIN_YEARS` (4),
  `WALK_FORWARD_TRAIN_FREQUENCY` ("Q") — controlan el esquema **entrenar-hasta-un-corte-luego-congelar**
  (sección 7).
- `TRANSACTION_COST_BPS`, `SLIPPAGE_BPS`, `FUNDAMENTAL_PUBLICATION_LAG_WEEKS` (7).
- `MIN_PORTFOLIO_SIZE`/`MAX_PORTFOLIO_SIZE` (5–10 posiciones), `MIN_ROTATION_ADVANTAGE`,
  `MIN_SCORE_ADVANTAGE_TO_REPLACE`, `MIN_CONVICTION_ADVANTAGE`, `MIN_OPPORTUNITY_COST_THRESHOLD`.

## 5. Ingesta de datos (`module/ingest/`)

`clients.py` implementa `FinnhubClient` y `YahooClient` (sin dependencia de `yfinance`; llama
directamente al endpoint `v8/finance/chart` de Yahoo). Ambos limitan reintentos por 429 a un
**máximo de 5** antes de registrar el error y devolver `None`, evitando bucles infinitos ante
rate-limiting.

`pipeline.py::download_raw_data` cachea cada respuesta cruda en
`data/raw/json/<fuente>/<ticker>/<dataset>.json`, permitiendo re-ejecuciones sin red mientras
`FORCE_RAW_DOWNLOAD=False`. El manejo de errores es **por ticker**: un fallo individual no aborta
la descarga completa; se registran `data/raw/download_coverage.json` y `download_failures.csv`
para auditar qué tickers/datasets fallaron.

## 6. Dataset maestro point-in-time (`module/dataset.py`)

`build_master_dataset(settings)` construye una fila por `(ticker, snapshot_date)` en
`data/master/master_point_in_time.parquet`. Es el mecanismo central que garantiza **ausencia de
lookahead bias** en todo el proyecto:

- Cada serie (precios, fundamentales) se cachea como una lista ordenada de `(fecha, valor)` por
  ticker.
- La búsqueda point-in-time usa `bisect_right(fechas, fecha_snapshot) - 1`: siempre devuelve el
  último valor disponible **estrictamente en o antes** de la fecha del snapshot
  (`_latest_price`, `_latest_row`, `_historical_growth`).
- **Lag de publicación de fundamentales**: `_prepared_rows` desplaza la fecha de fin de periodo de
  cada fundamental hacia adelante `FUNDAMENTAL_PUBLICATION_LAG_WEEKS` semanas antes de considerarlo
  "conocido" — porque un fundamental con fecha de periodo `Q1` no se publica realmente hasta
  semanas después. Sin este desplazamiento habría un lookahead sutil de varias semanas.
- Columnas producidas: `price`, `price_return_{1,3,6,12}m`, `sector`, múltiplos de valoración
  (`pe`, `forward_pe`, `peg`) tanto en su forma base como **ajustada por precio**
  (`{multiplo}_price_adjusted = valor_base * (1 + variación_de_precio)`, para reflejar que un
  múltiplo calculado con un precio de hace meses no es el múltiplo de hoy), márgenes, ROE/ROIC,
  deuda/equity, crecimiento de ingresos y BPA, y metadatos de frescura (`stale_fundamental_months`,
  `fundamental_asof_date`).
- Frecuencia de snapshots: mensual por defecto (`PRICE_UPDATE_FREQUENCY`), con revisión trimestral
  de fundamentales (`FUNDAMENTAL_REVIEW_FREQUENCY`) marcada en `review_type`.
- Optimización: si el walk-forward está activo, los snapshots empiezan en
  `PORTFOLIO_START_DATE − MAX_WALK_FORWARD_TRAINING_YEARS` (nunca antes de `DATA_START_DATE`), para
  no calcular historia que ningún reentrenamiento va a usar.

## 7. Feature engineering (`module/features/`)

`pipeline.py::build_features()` calcula, **por snapshot_date** (cross-sectional, percentil dentro
del universo de ese mes), los siguientes scores en `[0, 1]`:

```text
quality_score    = media percentil de [roe, roic, gross_margin, operating_margin, net_margin, fcf_margin]
growth_score     = media percentil de [revenue_growth, eps_growth]
valuation_score  = 1 - media percentil de [pe, forward_pe, peg]
price_adjusted_valuation_score = 1 - media percentil de los múltiplos *_price_adjusted (con fallback a valuation_score)
momentum_score   = media percentil de [price_return_3m, price_return_6m, price_return_12m]
moat_score       = media percentil de [roic, fcf_margin]
catalyst_score   = (percentil(eps_growth - revenue_growth) + media percentil([price_return_3m, price_return_6m])) / 2
risk_score       = 1 - percentil(debt_equity)
```

`garp_score` (baseline determinista, **no aprendido** — se conserva deliberadamente fijo como
punto de comparación frente al sistema aprendido):

```text
garp_score = 0.30·quality_score + 0.20·moat_score + 0.20·growth_score + 0.15·valuation_score
           + 0.08·catalyst_score + 0.02·momentum_score + 0.05·risk_score
```

`quality_value_gap` — literalmente el concepto "infravalorada de calidad":

```text
quality_growth  = media(percentil(quality_score), percentil(growth_score))
expensiveness   = percentil(price_adjusted_valuation_score)
quality_value_gap = ((quality_growth - (1 - expensiveness)) + 1) / 2
```

Alto = buena empresa (calidad + crecimiento) que cotiza barata frente al universo.

### 7.1 Expectativas del mercado (`transforms.py::add_expectation_features`)

```text
implied_growth = expected_growth = clip(1 - valuation_score, 0, 1)
realized_growth = clip(media percentil([revenue_growth, eps_growth]), 0, 1)
expectation_gap = clip(realized_growth - implied_growth, -1, 1)
positive_expectation_gap = clip((expectation_gap + 1) / 2, 0, 1)
```

`realized_growth` es el crecimiento fundamental **observado**, no una reproyección circular de
otros scores (ver sección 9.1 sobre por qué esto importa para el ML).

### 7.2 Métricas relativas (`transforms.py::add_relative_features`)

Para `quality_score`, `growth_score`, `valuation_score`: rank percentil dentro de `[snapshot_date,
sector]` (`_vs_sector`) y dentro de `[snapshot_date]` (`_vs_universe`), vía `groupby(...).rank(pct=True)`.

### 7.3 Tendencias temporales (`transforms.py::_historical_delta`, vía `merge_asof`)

Implementado con `pandas.merge_asof` (O(n log n) por ticker, no un bucle O(n²)):

```text
quality_trend_1y / quality_trend_2y  — delta de quality_score a 12/24 meses
roic_trend, fcf_trend                — delta a 12 meses
margin_trend                         — delta de la media de [gross, operating, net, fcf margin] a 12 meses
growth_acceleration = max(growth_trend, 0)
growth_deceleration = max(-growth_trend, 0)
moat_trend                           — delta a 12 meses
```

Todas clip a `[-1, 1]`, NaN rellenado con 0. El objetivo es distinguir una empresa buena de una
empresa que además está mejorando.

## 8. Modelo de IA (`module/ml.py`)

### 8.1 Cuatro agentes especializados

Cada agente es un `LightGBMRegressor` (fallback a `RandomForestRegressor` de sklearn si LightGBM
no está disponible), entrenado sobre su **propio subconjunto de features** para que realmente se
especialicen (importancias distintas, no los mismos ~30 features cuatro veces):

| Agente | Target | Features (`AGENT_FEATURES`) |
|---|---|---|
| `quality_probability` | `target_quality`: cambio forward del ROIC realmente reportado | quality_score, moat_score, risk_score, quality_score_vs_{sector,universo}, quality_trend_1y/2y, roic_trend, margin_trend, fcf_trend, moat_trend |
| `improvement_probability` | `target_improvement`: crecimiento fundamental observado vs. expectativa de hoy | growth_score, expected_growth, growth_acceleration, growth_deceleration |
| `mispricing_probability` | `target_mispricing`: si el descuento de valoración de hoy se resolvió en alpha forward (horizonte propio, 6 meses por defecto — `AGENT_HORIZON_MONTHS_OVERRIDE`) | quality_value_gap, valuation_score_vs_{sector,universo}, positive_expectation_gap |
| `alpha_probability` | `target_future_alpha`: retorno forward menos benchmark, 12 meses — **la señal maestra** | `MODEL_FEATURES` completo (generalista) |

Los 4 targets se construyen desde información **observable** en el futuro (no reproyecciones
same-day de las features de entrada — el defecto de diseño original que se corrigió: la fórmula
circular `realized_growth = 0.6·growth + 0.25·quality + 0.15·moat` se reemplazó por crecimiento
fundamental realmente observado). Toda fila cuyo horizonte forward aún no es observable en la
fecha de entrenamiento se enmascara y cae al fallback GARP determinista — el mismo mecanismo para
los 4 agentes (`_walk_forward_component_scores`).

### 8.2 Meta-agente: combinación aprendida por rank-IC (cambio 2026-07)

El meta-agente combina las 4 probabilidades en `final_score`:

```text
final_score = Σ peso_agente · probabilidad_agente,  pesos ≥ 0, Σ pesos = 1
```

**Cómo se aprenden los pesos** (`_fit_meta_weights`, en cada snapshot de entrenamiento): para cada
agente se calcula el **rank-IC (correlación de Spearman)** de su score frente al
`target_future_alpha` realmente realizado en esa ventana de entrenamiento; los rank-IC se **clipan
a 0** (un agente sin capacidad de ordenar bien las acciones no puede recibir peso negativo, pero
tampoco aporta) y se normalizan para sumar 1:

```text
ic_agente = max(0, spearman(score_agente, target_future_alpha_realizado))
peso_agente = ic_agente / Σ ic_agentes
```

Si ningún agente tiene rank-IC positivo, o hay menos de 50 observaciones con alfa observable, se
usa el prior fijo `AGENT_PRIOR_WEIGHTS = {quality: 0.30, improvement: 0.25, mispricing: 0.25,
alpha: 0.20}`.

**Por qué se cambió (diagnóstico de 2026-07, verificado con datos reales)**: la versión anterior
ajustaba los pesos con `scipy.optimize.nnls` (mínimos cuadrados no-negativos) sobre el alpha
**en bruto**, no sobre el ranking. Esto minimiza error cuadrático, algo que un agente de baja
varianza puede lograr prediciendo cerca de la media sin ninguna capacidad de discriminación. En el
run real se confirmó el patrón: el peso de `quality_probability` subía hasta ~55-60% precisamente
en los trimestres de 2020 donde su rank-IC OOS era **negativo** (-0.10 a -0.12), mientras el peso
de `alpha_probability` bajaba en esos mismos trimestres pese a tener rank-IC fuertemente positivo
(+0.30 a +0.35). El meta-agente estaba premiando al agente que peor ordenaba las acciones. Con el
cambio a rank-IC, ese mismo periodo converge a peso ≈100% en `alpha_probability` y ≈0% en
`quality_probability` — coherente con qué agente realmente aporta poder de selección.

### 8.3 Diagnóstico de calidad del modelo

`_oos_metrics` calcula, por snapshot, el rank-IC (Spearman) y RMSE de cada agente contra el
`target_future_alpha` realizado, out-of-sample (el snapshot puntuado nunca formó parte de su
propio entrenamiento). `_add_rolling_ic_trend` añade una media móvil de 12 snapshots y un t-stat
aproximado por año calendario, para ver si la capacidad predictiva mejora, se mantiene plana u
oscila con el tiempo — reportado tal cual sale, sin forzar una narrativa de mejora dado el tamaño
de muestra (~90-140 snapshots walk-forward).

Todo se escribe a `results/<run>/model_walk_forward_diagnostics.csv` y
`results/<run>/meta_weights_by_snapshot.csv`.

## 9. Esquema de entrenamiento: entrenar-hasta-un-corte, luego congelar

En vez de reentrenar en cada snapshot mensual durante toda la ventana simulada (walk-forward
"puro"), el sistema separa explícitamente dos fases mediante `train_cutoff_date`:

- **Fase de entrenamiento** (`train_dates`, trimestral, desde `train_cutoff_date −
  walk_forward_train_years` hasta `train_cutoff_date` inclusive): en cada fecha trimestral se
  reentrenan los 4 agentes y se reajustan los pesos del meta-agente, con una ventana móvil de hasta
  `max_walk_forward_training_years` de historia — esto **es** aprendizaje real, walk-forward y sin
  lookahead (la máscara de observabilidad se aplica igual que siempre).
- **Fase congelada** (`apply_dates > train_cutoff_date`, mensual): cada mes se puntúa con el
  **último modelo/pesos entrenados** en la fase anterior, sin ningún reentrenamiento nuevo — es
  simulación de despliegue con un modelo fijo, no aprendizaje continuo.

Función clave: `_train_and_apply_dates(all_dates, settings)` separa `train_dates` (subconjunto ≤
corte, en cadencia trimestral, garantizando que el propio corte sea siempre un punto de
entrenamiento) de `apply_dates` (todas las fechas, todas reciben una predicción). El diagnóstico
por snapshot incluye una columna `phase` (`"training"` / `"frozen"`) y `training_snapshot_date`
(qué modelo se usó realmente) para que sea auditable.

Este diseño responde directamente a la propuesta del usuario: elegir un año de arranque de
simulación, definir cuántos años de entrenamiento previos usar, y a partir de ahí ejecutar la
cartera con un modelo fijo tal y como habría funcionado en tiempo real.

**Horizonte de predicción vs. cadencia de reentrenamiento**: son parámetros independientes. La
cadencia (trimestral) es cada cuánto se reentrena; el horizonte
(`walk_forward_label_horizon_months`, 12 meses por defecto) es a qué plazo se mide si la
predicción acertó. Se mantiene 12 meses por defecto porque un diagnóstico comparativo
(`_label_horizon_comparison`, visible en el informe) mide el rank-IC del agente Alpha a 3/6/12
meses reutilizando el modelo ya entrenado, y en las ejecuciones realizadas 12 meses no ha sido
sistemáticamente peor que plazos más cortos — el ruido de precio a corto plazo domina sobre la
señal fundamental, que necesita tiempo para manifestarse.

## 10. Investigación y tesis (`module/research/`)

- **`synthesis.py`**: generación de texto **puramente determinista** (reglas sobre los scores, sin
  ML ni LLM) — descripción de empresa, análisis de moat, catalizador, tesis alcista/bajista/base,
  resumen de riesgos y oportunidades.
- **`thesis.py`** (`enrich_with_thesis_scores`):

```text
thesis_score           = 0.28·calidad + 0.25·moat + 0.20·crecimiento + 0.15·valoración_ajustada
                        + 0.07·catalyst + 0.05·momentum
position_health_score  = 0.35·calidad + 0.25·moat + 0.20·riesgo + 0.20·catalyst
conviction_score       = 0.45·thesis_score + 0.30·position_health_score + 0.15·final_score + 0.10·valoración
thesis_rank_score       = 0.45·thesis_score + 0.30·calidad + 0.15·conviction_score + 0.10·valoración
exit_score              = 1 - (0.45·thesis_score + 0.35·position_health_score + 0.20·valoración)
```

  Estados de tesis: `Broken` (momentum<0.18 y valoración<0.45, o caída del resto de condiciones),
  `Improving` (thesis_score≥0.78 y health≥0.70), `Intact` (thesis_score≥0.62), `Maturing`
  (valoración<0.25 y calidad≥0.65), `Weakening` (thesis_score≥0.45).

- **`ai.py`**: única integración con OpenAI, gateada por `ENABLE_OPENAI_RESEARCH` +
  `OPENAI_API_KEY` (desactivada por defecto). Llama a `https://api.openai.com/v1/responses` con
  `requests` puro (sin SDK `openai`). **Siempre** cae al fallback determinista de `synthesis.py`
  ante cualquier fallo o si está desactivada — ninguna etapa depende de que la llamada externa
  tenga éxito.

## 11. Selección y watchlist (`module/strategy/selection.py`)

`add_buy_today_decision(df)` calcula, por snapshot, el mejor alternativo de cada ticker
(`_best_alternatives`, excluyéndose a sí mismo) y el coste de oportunidad:

```text
opportunity_cost_score = max(0, score_mejor_alternativa - thesis_rank_score)
buy_today_score = 0.28·thesis_score + 0.24·calidad + 0.17·positive_expectation_gap
                + 0.15·valoración_ajustada + 0.08·momentum + 0.08·(1 - opportunity_cost_score)
would_buy_today = buy_today_score≥0.60
                  AND (momentum≥0.35 OR (valoración≥0.72 AND calidad≥0.58))
                  AND thesis_state ∈ {Improving, Intact}
                  AND opportunity_type ∉ {Avoid, Value Trap}
```

`build_watchlist(settings)` filtra por `opportunity_type` no bloqueado, `conviction_score≥0.35`,
`business_quality_score≥0.40`; conserva el top-200 por snapshot en el histórico de auditoría
(`audit/watchlist_history.csv`) y solo el snapshot más reciente en `watchlist.csv`/`.parquet`.

## 12. Construcción y gestión de cartera (`module/strategy/portfolio.py`)

Cartera concentrada de 5-10 posiciones (`MIN_PORTFOLIO_SIZE`/`MAX_PORTFOLIO_SIZE`).

```text
manager_score = 0.45·final_score + 0.13·thesis_rank_score + 0.11·buy_today_score
              + 0.09·momentum_score + 0.08·valoración_ajustada + 0.05·positive_expectation_gap
              + 0.05·moat_score + 0.04·risk_score
```

El peso de `final_score` se subió de 0.40 a 0.45 (ronda de diagnóstico 2026-07) tras verificar que
el rank-IC OOS del agente alpha era positivo en 6 de 8 años calendario.

**Condiciones de entrada** (`_entry_candidates`): estado de tesis invertible, no bloqueado por
`opportunity_type`, `business_quality_score≥0.48`, `buy_today_score≥0.54`, `manager_score≥0.56`, y
(`momentum≥0.35` o (`valoración_ajustada≥0.72` y `calidad≥0.58`)).

**Motivos de salida** (`_exit_reason`), evaluados en orden:

1. `Thesis Broken` — estado de tesis roto.
2. `Exit Score Trigger` — `exit_score≥0.66`.
3. `Price Adjusted Valuation No Longer Attractive` — valoración ajustada<0.20 y no `would_buy_today`.
4. `Momentum And Thesis Deterioration` — momentum<0.35 (umbral subido desde 0.20 en 2026-07, ver
   más abajo) y estado `Weakening`.
5. `Manager Score Below Hold Hurdle` — `manager_score<0.46` y no `would_buy_today`.
6. `Persistent Better Use Of Capital` — 4+ meses seguidos sin `would_buy_today` y coste de
   oportunidad≥0.08.
7. `Repeated Thesis Deterioration` — estado `Weakening` con 3+ deterioros acumulados.

**Cambio 2026-07 — umbral de `Momentum And Thesis Deterioration` subido de 0.20 a 0.35**: en el
análisis del run anterior, este era el motivo de venta con el peor retorno en exceso medio por
operación (-26.4%), muy por debajo del resto — indicio de que disparaba después de que la mayor
parte del daño relativo ya se hubiera producido. Subir el umbral de momentum hace que la salida
reaccione ante el primer signo de tesis debilitándose junto con momentum ya cayendo, en vez de
esperar a que el momentum colapse por debajo de 0.20.

**Rotación** (`_replacement_target`): solo posiciones con ≥4 meses de antigüedad o en estado
`Broken`/`Weakening` son elegibles para ser reemplazadas; se sustituye la más débil por
`manager_score` si el candidato tiene una ventaja de score ≥0.09, o una combinación menor de
ventaja de score+convicción+momentum con `would_buy_today=True`.

## 13. Dimensionamiento de posiciones (`module/strategy/sizing.py`)

Cuatro métodos calculados y comparados, con límites `MIN_POSITION_WEIGHT=0.04`,
`MAX_POSITION_WEIGHT=0.18`:

```text
equal_weight        = 1 / n_posiciones
sizing_score = clip(
    (0.30·conviction_score + 0.22·manager_score + 0.18·calidad + 0.12·buy_today_score
     + 0.08·momentum + 0.06·risk_score + 0.04·positive_expectation_gap)
    · (1 - 0.30·clip(opportunity_cost_score, 0, 1)),
    0, 1)
conviction_weight    = sizing_score / Σ sizing_score  (fallback equal_weight)
risk_adjusted_weight = conviction_weight acotado a [min(0.04, 1/n), min(0.18, 1.65/n)], renormalizado
hybrid_weight        = clip(0.35·equal_weight + 0.65·risk_adjusted_weight, 0, 1)
```

**`hybrid_weight` es el que realmente alimenta la simulación de P&L del backtest** — no es solo
informativo (invariante crítico, ver sección 14).

## 14. Backtest (`module/backtest/`)

`engine.py::run_backtest` es el bucle mensual que llama a
`initial_portfolio`/`review_portfolio` (portfolio.py) por snapshot.

`performance.py`:
- **Lotes FIFO**: `open_lots` mantiene una lista de lotes abiertos por ticker; cada venta cierra el
  lote **más antiguo** primero, calculando `total_return`, `annualized_return = (1+r)^(365.25/días)-1`,
  y `excess_total_return` frente al benchmark durante ese mismo periodo de tenencia.
- **`weighted_basket_return`**: usa `hybrid_weight` normalizado por fecha
  (`Σ w_i · retorno_i / Σ w_i`) — no equal-weight, salvo fallback cuando no hay pesos disponibles.
- **Coste de transacción, ponderado por notional** (`period_transaction_cost`):
  `cost_rate = (transaction_cost_bps + slippage_bps) / 10000`; `notional_traded = Σ peso_actual` de
  las compras + `Σ peso_previo` de las ventas; `coste = notional_traded · cost_rate` — no por
  conteo de operaciones.

`reviews.py` separa dos vistas independientes (soportan el análisis de coste de oportunidad):
- **Revisión de universo completo** (`universe_action`): BUY/SELL/HOLD/AVOID para cualquier
  ticker, esté o no en cartera.
- **Decisión de posiciones en cartera** (`manager_decision`): ADD/REDUCE/WATCH/HOLD, independiente
  de la anterior.

`artifacts.py::excess_return_statistics`:

```text
excess = retorno_cartera - retorno_benchmark (por periodo)
tracking_error      = std(excess, ddof=1) · sqrt(12)
information_ratio   = (media(excess) · 12) / tracking_error
t_stat              = media(excess) / (std(excess) / sqrt(n))
```

`SMALL_SAMPLE_CAVEAT` (texto mostrado siempre junto al t-stat): con solo unas pocas decenas de
observaciones mensuales, el t-stat tiene poder estadístico limitado; se trata como señal
direccional, no como prueba de un edge significativo. Deliberadamente **no** se aplican bootstrap,
Sharpe deflactado ni correcciones por comparaciones múltiples.

`AUDIT_OUTPUTS` (en `engine.py`) enruta las tablas pesadas por ticker/snapshot a
`results/<run>/audit/` (`portfolio_allocation`, `portfolio_decision_log`, `portfolio_evolution`,
`portfolio_monthly_holdings`, `portfolio_review_diagnostics`, `portfolio_transactions`,
`portfolio_turnover`, `rebalance_report`, `universe_monthly_scores`,
`universe_monthly_price_update`, `universe_quarterly_fundamental_review`,
`universe_top_candidates`); todo lo demás (resúmenes ejecutivos compactos) queda en la raíz de
`results/<run>/`.

## 15. Informe HTML (`module/viewer/`)

Página única (`results/<run>/viewer/index.html`), autocontenida, en español, con 7 secciones (ver
`results.md` para el detalle de lectura de cada sección): Resumen, Rendimiento, Cartera,
Aprendizaje, Posiciones, Metodología, Debug/TFM. Cada sección existe porque responde a una
pregunta del TFM o ayuda a depurar — criterio aplicado en `module/viewer/shared.py`. Las tablas
pesadas de auditoría nunca se embeben, solo se enlazan.

`charts.py` usa matplotlib (`Agg`, sin dependencias nuevas) con la paleta validada de la skill
`dataviz`; marca visualmente la fecha de corte del esquema entrenar-luego-congelar con una línea
vertical punteada en los gráficos de pesos aprendidos y rank-IC.

`module/report.py` reutiliza las mismas funciones de métricas (`_metrics`, `drawdown_episodes`)
que consume el viewer — la etapa `report` apunta al mismo `index.html` que genera `viewer`, no
produce un segundo informe distinto.

## 16. Limitaciones metodológicas explícitas

**Sesgo de supervivencia**: `TICKERS` (`environment.py`) es un listado estático de grandes
compañías **actuales** aplicado retroactivamente hasta `DATA_START_DATE`. Excluye nombres
delisted/adquiridos/caídos del índice durante ese periodo e incluye IPOs/spin-offs recientes que
no existían históricamente. Es una decisión de alcance documentada, no un defecto oculto: la
cartera viva solo opera entre `PORTFOLIO_START_DATE`/`PORTFOLIO_END_DATE`, lo que acota pero no
elimina el sesgo, y la ventana de entrenamiento ML (hasta `MAX_WALK_FORWARD_TRAINING_YEARS` de
historia) lo hereda igualmente.

**Tamaño de muestra pequeño**: la ventana de backtest son unas pocas decenas de observaciones
mensuales. `excess_return_statistics` reporta IR/TE/t-stat pero deliberadamente no aplica
correcciones que implicarían más precisión estadística de la que la muestra soporta. El t-stat es
directional.

## 17. Pruebas (`tests/`)

```bash
pytest tests/
```

- `tests/test_train_cutoff_freeze.py`: verifica que ninguna fecha de entrenamiento supera
  `train_cutoff_date`, que las fechas de entrenamiento son trimestrales y arrancan
  `walk_forward_train_years` antes del corte, y que la fase congelada reutiliza un único
  modelo/conjunto de pesos (test end-to-end con datos sintéticos vía fixture `synthetic_run`).
- Otras pruebas cubren invariantes de leakage y rangos de features.

## 18. Estado del sistema (2026-07)

Cambios verificados en la ronda de diagnóstico más reciente, cada uno confirmado con datos reales
del run completo (no solo teoría):

1. **Meta-agente**: pesos aprendidos por rank-IC en vez de mínimos cuadrados (NNLS) sobre alfa en
   bruto — corrige un desajuste de objetivo confirmado empíricamente (el peso de `quality_probability`
   subía precisamente cuando su rank-IC OOS era negativo).
2. **Trigger de salida `Momentum And Thesis Deterioration`**: umbral de momentum subido de 0.20 a
   0.35 — este motivo tenía por lejos el peor retorno en exceso medio por venta (-26.4%) de todas
   las categorías.

Tras aplicar ambos cambios y re-ejecutar el pipeline completo (universo completo, corte
2023-02-15, 4 años de entrenamiento previo), el run resultante mostró alpha acumulado positivo
sustancial y una distribución de pesos del meta-agente coherente con el rank-IC observado de cada
agente por primera vez. Sigue aplicando el mismo criterio de honestidad metodológica del resto del
proyecto: con ~40 observaciones mensuales de cartera, cualquier mejora puntual debe leerse junto al
aviso de muestra pequeña, no como prueba definitiva de un edge persistente.

## 19. Propósito académico

El proyecto está diseñado para ser defendible como Trabajo Fin de Máster porque combina:

- **Ingeniería de datos**: pipeline modular, point-in-time sin lookahead, lag de publicación de
  fundamentales.
- **ML reproducible**: walk-forward con enmascarado auditable, esquema entrenar-hasta-un-corte-
  luego-congelar configurable, diagnóstico de rank-IC visible y comparación de horizontes.
- **IA explicable**: 4 agentes especializados interpretables + meta-agente cuya lógica de
  combinación (rank-IC) es auditable paso a paso, no una caja negra.
- **Gestión dinámica**: cartera concentrada con tesis vivas, distinción compra-hoy vs. mantener,
  coste de oportunidad explícito.
- **Backtesting riguroso**: P&L ponderado por sizing real (`hybrid_weight`), coste de transacción
  por notional rotado, IR/TE/t-stat con aviso de muestra pequeña.
- **Transparencia**: todas las limitaciones documentadas explícitamente, ningún resultado
  presentado como más significativo de lo que la muestra permite.
