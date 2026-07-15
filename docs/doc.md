# Documentación exhaustiva — GARP AI Portfolio System

## 0. Resumen general del proyecto

Este es, ante todo, **un proyecto de Inteligencia Artificial y Machine Learning**. La bolsa no es
el fin sino el **banco de pruebas**: un dominio real, ruidoso y adversarial donde comprobar si un
sistema de IA es capaz de **aprender** —a partir únicamente de su propio histórico y sin ver el
futuro— una estrategia que bata a un índice de referencia (SPY) de forma **consistente**. El
criterio de éxito del TFM no es una rentabilidad puntual espectacular, sino la **estabilidad del
aprendizaje**: que la ventaja sea positiva en el mayor número posible de sub-periodos y no dependa
del disparo afortunado de una sola acción.

**Qué hace el sistema, en una frase.** Cada mes ("snapshot"), a partir de un dataset *point-in-time*
(cada dato es el último conocido estrictamente a esa fecha, sin lookahead), la IA puntúa todo el
universo de acciones, construye y revisa una cartera concentrada con tesis de inversión explícitas,
y se re-entrena continuamente para adaptarse al régimen de mercado vigente.

**El motor de IA (lo central).** Tres agentes especialistas de *gradient boosting* (LightGBM), cada
uno entrenado contra su propio objetivo expresado como **rango transversal** (per-snapshot):

- **Calidad** — rango de la mejora futura del ROIC reportado (calidad fundamental del negocio).
- **Temporización** — rango del exceso de retorno a 3 meses (momentum / punto de entrada).
- **Alpha** — rango del exceso de retorno a 12 meses *en sí mismo*: el agente que aprende a
  **rankear directamente** la magnitud que el sistema quiere predecir.

Un **meta-agente** aprende, en cada snapshot, cuánto pesa cada agente según su **contribución
marginal** (partial rank-IC frente al alpha que los demás no explican), premiando la *consistencia*
entre sub-folds y no el pico puntual, con regularización anti-solución-de-esquina. El resultado,
`final_score`, es la señal aprendida que domina (peso 0.70) la decisión de cartera.

**Cómo se mide el éxito.** Métrica principal: **consistencia entre sub-periodos** (¿en cuántos
tramos de la ventana la alpha fue positiva?). Métrica de calidad del modelo: **rank-IC
out-of-sample** de `final_score` frente al alpha realizado. Como evidencia de apoyo (no como relato
principal): bootstrap de la alpha acumulada, sensibilidad a costes, test placebo/permutación y
comparación contra baselines triviales (notablemente `momentum_only`).

**Honestidad metodológica.** El proyecto reporta también lo que *no* funciona: el rank-IC sigue
siendo débil, y un baseline de solo-momentum ha batido históricamente al sistema completo — un
hallazgo negativo legítimo que forma parte del valor científico del TFM. La solidez metodológica
(walk-forward puro, sin fuga, ponderación por contribución marginal regularizada) se defiende con
independencia de que la señal aprendida sea fuerte o no.

**Limitaciones conocidas y asumidas.** Sesgo de supervivencia (universo estático actual aplicado
hacia atrás) y potencia estadística limitada (pocas decenas de observaciones mensuales) — ambas
documentadas y surfaceadas en el informe, no ocultadas.

Las secciones siguientes profundizan etapa por etapa.

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

1. **Capa estadística** (`module/ml.py`): `final_score` / `opportunity_type`, producidos por 3
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

Existe un modo transversal aparte, `run_mode="experiments"`, que **no** ejecuta este pipeline sino
que barre múltiples configuraciones reutilizando las etapas `ml` y `backtest` — se documenta en §16.

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

- `RUN_MODE` (una etapa, `"full"`, o `"experiments"`), `EXPERIMENTS_FILE` (fichero de escenarios o
  `"todos"`, solo relevante en modo experiments; ver §16), `DEV_MODE`, `DEV_TICKERS`, `TICKERS`
  (universo completo estático), `BENCHMARK_TICKER`.
- `DATA_START_DATE`, `PORTFOLIO_START_DATE`, `PORTFOLIO_END_DATE`, `PORTFOLIO_REVIEW_FREQUENCY`.
- `WALK_FORWARD_SCORING`, `WALK_FORWARD_LABEL_HORIZON_MONTHS` (12 por defecto).
- `MIN_WALK_FORWARD_TRAINING_ROWS`, `MIN_WALK_FORWARD_TRAINING_YEARS`, `MAX_WALK_FORWARD_TRAINING_YEARS`
  (ventana móvil máxima de historia usada en cada reentrenamiento individual).
- `TRAIN_CUTOFF_DATE` (por defecto = `PORTFOLIO_START_DATE`), `WALK_FORWARD_TRAIN_YEARS` (4),
  `WALK_FORWARD_TRAIN_FREQUENCY` ("Q") — controlan el esquema **walk-forward rodante** (§9):
  `TRAIN_CUTOFF_DATE` es el punto más temprano de reentrenamiento, no un punto de congelación.
- `TRANSACTION_COST_BPS`, `SLIPPAGE_BPS`.
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
- **Fecha de observabilidad de fundamentales**: `_prepared_rows` usa la fecha de fin de periodo de
  cada fundamental directamente como su fecha de observabilidad, sin margen adicional automático —
  el margen entre la fecha real de publicación y la fecha en que la estrategia lo trata como
  "conocido" se espera que venga de cómo se elijan `PORTFOLIO_START_DATE` y la cadencia de snapshots,
  no de un lag interno en la capa de datos.
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
implied_growth = clip(1 - valuation_score, 0, 1)
realized_growth = clip(media percentil([revenue_growth, eps_growth]), 0, 1)
expectation_gap = clip(realized_growth - implied_growth, -1, 1)
positive_expectation_gap = clip((expectation_gap + 1) / 2, 0, 1)
```

`positive_expectation_gap` (el gap reescalado a [0,1]) es una *feature* del agente alpha — alto
cuando el crecimiento observado bate lo que la valoración del mercado implica.

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

### 8.1 Tres agentes especializados (rediseño 2026-07)

Cada agente es un `LightGBMRegressor` (fallback a `RandomForestRegressor` de sklearn si LightGBM
no está disponible), entrenado sobre su **propio subconjunto de features**. Todos los targets son
**rangos transversales per-snapshot** en [0,1] (no squashes `(x+1)/2` que saturan las colas): el
rango es robusto a outliers y está alineado con la métrica rank-IC con la que se evalúa el sistema.

| Agente | Target (rango per-snapshot de…) | Horizonte | Features (`AGENT_FEATURES`) |
|---|---|---|---|
| `quality_probability` | cambio forward del ROIC realmente reportado | 12m | quality_score, moat_score, risk_score, quality_score_vs_{sector,universo}, quality_trend_1y/2y, roic_trend, margin_trend, fcf_trend, moat_trend |
| `timing_probability` | exceso de retorno forward corto (entrada/momentum) | 3m | momentum_score, catalyst_score, price_return_1m/3m/**6m/12m**, price_return_since_fundamental, stale_fundamental_months |
| `alpha_probability` | `target_future_alpha` (exceso de retorno) **en sí mismo** — el aprendiz de ranking directo | 12m | price_return_6m/12m, momentum_score, price_adjusted_valuation_score, quality_value_gap, growth_score_vs_{sector,universo}, positive_expectation_gap, catalyst_score |

**Por qué tres y no cuatro (diagnóstico verificado con datos reales).** El diseño anterior tenía
cuatro agentes; dos de ellos (`improvement`, `mispricing`) tenían rank-IC OOS **negativo** —
restaban a la combinación mientras el suelo de peso los obligaba a mantener peso— y, más de fondo,
**ningún** agente entrenaba contra el alpha 12m que la combinación puntúa. El nuevo agente `alpha`
cierra esa brecha: aprende a **rankear directamente** el alpha forward. La información de los
agentes eliminados **no se pierde**: valoración / crecimiento relativo / gap de expectativas
sobreviven como *features* del agente alpha (donde el modelo aprende su interacción no lineal con el
momentum), en vez de como targets construidos a mano que rankeaban el alpha negativamente.

El agente alpha **no** resucita el viejo generalista dominante (que veía *todas* las features): usa
un subconjunto restringido, y la ponderación por contribución marginal del meta-agente (§8.2)
impide que domine si no aporta ranking marginal. Los subconjuntos **ya no son disjuntos** — timing
y alpha comparten deliberadamente los retornos a 6m/12m (el edge literal del baseline
`momentum_only`, antes invisible para todos los agentes).

**Qué agente resulta ser fiable (medido OOS, 2018+).** Contra lo que sugiere el diseño, el agente
`alpha` (pensado como "el que rankea") es el **menos fiable** (mayor rank-IC potencial en años
buenos pero también la mayor varianza y años muy negativos, p. ej. −0.14 en 2021), mientras que
`quality` es el **más consistente** (positivo en 8 de 9 años, la menor varianza) pese a su rank-IC
absoluto más bajo. Este hallazgo es el que motiva el prior inclinado a Calidad del meta-agente
(§8.2.1). Detalle completo por agente y año en `docs/diagnostico_aprendizaje.md`.

Cada target se construye desde información **observable** en su propio horizonte
(`AGENT_HORIZON_MONTHS_OVERRIDE`); toda fila cuyo horizonte forward aún no es observable en la fecha
de entrenamiento se enmascara y cae al fallback GARP determinista
(`_walk_forward_component_scores`). También se enmascara `target_future_alpha` para las filas no
observables antes de que `_select_hyperparameters` lo lea (sin fuga en la selección de
hiperparámetros).

**Selección de hiperparámetros por rank-IC de alpha (no RMSE).** `_select_hyperparameters` elige los
hiperparámetros LightGBM de cada agente (de `LGBM_PARAM_GRID`) por el **rank-IC de Spearman de la
predicción de validación frente a `target_future_alpha`** —la métrica con la que se juzga todo el
sistema— sobre el mismo split cronológico 70/30 sin fuga. Un RMSE bajo no implica buen ranking;
alinear la selección de capacidad con la métrica objetivo es parte del rediseño.

### 8.2 Meta-agente: combinación aprendida por contribución marginal consistente

El meta-agente combina las 3 probabilidades en `final_score`:

```text
final_score = Σ peso_agente · probabilidad_agente,  pesos ≥ 0, Σ pesos = 1
```

**Cómo se aprenden los pesos** (`_fit_meta_weights`, en cada snapshot de entrenamiento): cada agente
se pondera por su **contribución marginal al ranking**, no por su rank-IC bruto. Para cada agente se
regresa (OLS) el `target_future_alpha` sobre los *otros* agentes en la parte de ajuste, se toma el
residuo en la parte de validación (el alpha que los demás **no** explican), y se puntúa al agente
por el **rank-IC parcial de Spearman** de su score contra ese residuo. Ese rank-IC parcial se
calcula sobre `N_CONSISTENCY_FOLDS` sub-folds cronológicos de la validación y se resume como
`mean(fold_ic) − CONSISTENCY_LAMBDA · std(fold_ic)`: un agente que rankea bien pero de forma
errática vale menos que uno que rankea moderada pero fiablemente — **la consistencia se premia, no
el pico**.

Los rank-IC parciales se clipan a 0, se normalizan, y se **mezclan (shrinkage) hacia el prior
informado `AGENT_PRIOR_WEIGHTS` a nivel `META_WEIGHT_FLOOR`** antes de renormalizar (regularización
anti-solución-de-esquina: impide que la ventaja marginal de un agente en un hold-out de ~30 filas
colapse la combinación a 100%/0%/0%). Si no hay historial suficiente o ningún agente aporta ranking
marginal, se usa directamente ese prior.

```python
AGENT_PRIOR_WEIGHTS = {quality: 0.45, timing: 0.30, alpha: 0.25}
```

Este es el bucle "aprende de la simulación": el meta-agente premia a los agentes que aportan
capacidad de selección *complementaria* y *fiable*, no redundante ni afortunada.

#### 8.2.1 Decisiones de diseño y por qué (no) otras alternativas

Cada elección del meta-agente responde a un problema medido, no a una preferencia estética. Las
justificaciones vienen del diagnóstico en `docs/diagnostico_aprendizaje.md` (reproducible con
`scripts/diagnostico_rank_ic.py` y `scripts/buscar_pesos_meta.py`).

- **Por qué ponderar por contribución marginal (rank-IC parcial) y no por el rank-IC bruto de cada
  agente.** El rank-IC bruto pagaría dos veces por una misma señal: si Alpha y Timing comparten el
  momentum, ambos cobrarían por ella. El rank-IC parcial (contra el alpha que los *otros* dejan sin
  explicar) sólo premia información nueva; un agente redundante cae a peso ~0 (garantizado por
  `test_duplicating_the_signal_collapses_its_marginal_ic`). Esto es lo que permite que los
  subconjuntos de features de los agentes **no** tengan que ser disjuntos.

- **Por qué el prior está inclinado a Calidad (0.45) y no a timing/alpha, como antes (0.30/0.35/0.35).**
  El diagnóstico por agente (ventana 2018+, 100 snapshots OOS) mostró que **Calidad es la señal más
  estable** —rank-IC positivo en 8 de 9 años, la menor varianza— mientras que **Alpha, el agente
  diseñado para rankear el alpha directamente, es el más inestable** (rank-IC de +0.14 en 2019 a
  −0.14 en 2021). El `final_score` heredaba esa inestabilidad. El prior original estaba calibrado a
  *dónde vivía la alfa cruda* (momentum), no a *dónde vive la señal de ranking fiable* (calidad
  fundamental). Una búsqueda offline de pesos confirmó que inclinar hacia Calidad sube el rank-IC
  medio y reduce su dispersión.

- **Por qué exactamente 0.45/0.30/0.25 y no un tilt mayor (p. ej. 0.60/0.25/0.15).** La búsqueda
  offline traza un trade-off: más peso a Calidad = más estabilidad, pero un tilt total también
  devuelve alfa/IR (la exposición a los ganadores de cola de momentum de la que procede buena parte
  de la alfa por asimetría, no por ranking). El punto 0.45/0.30/0.25 es el único que **mejora las
  dos cosas a la vez** en el walk-forward real: rank-IC medio +0.021→+0.024 y más estable, **y**
  alfa 1.10→1.14, IR 0.88→0.96, t-stat 2.53→2.77, reforzando además el subperiodo central que era
  el más débil. Un tilt de 0.60 subía el rank-IC pero bajaba alfa/IR/t-stat; por eso se descartó.

- **Por qué el suelo `META_WEIGHT_FLOOR` ancla hacia el prior informado y no hacia equal-weight.**
  Con equal-weight, en los trimestres de partial-IC ruidoso la combinación caía hacia un reparto
  igualitario que **reinflaba al agente Alpha** (el inestable). Anclando el shrinkage hacia el prior
  inclinado a Calidad, un trimestre ruidoso retrocede hacia la señal estable, no hacia el ruido.

- **Alternativa descartada — penalizar la varianza del rank-IC *entre snapshots* en el criterio de
  aprendizaje** (probada y revertida, "experimento 2A"). La idea era medir el partial-IC por
  snapshot y penalizar su dispersión temporal. Se implementó, se re-ejecutó el pipeline y **no
  mejoró** (rank-IC +0.021→+0.017). Motivo: el meta-agente pondera por contribución marginal contra
  el mismo `target_future_alpha` que el agente Alpha optimiza, así que Alpha conserva el mayor
  partial-IC medio **por más que se penalice su varianza** — el problema es estructural, no del
  criterio de consistencia. La conclusión de ese resultado negativo (bien medido) es la que llevó a
  la solución correcta: cambiar el *prior/ancla*, no el criterio de aprendizaje.

### 8.3 Diagnóstico de calidad del modelo

`_oos_metrics` calcula, por snapshot, el rank-IC (Spearman) y RMSE de cada agente contra el
`target_future_alpha` realizado, out-of-sample (el snapshot puntuado nunca formó parte de su
propio entrenamiento). `_add_rolling_ic_trend` añade una media móvil de 12 snapshots y un t-stat
aproximado por año calendario, para ver si la capacidad predictiva mejora, se mantiene plana u
oscila con el tiempo — reportado tal cual sale, sin forzar una narrativa de mejora dado el tamaño
de muestra (~90-140 snapshots walk-forward).

Todo se escribe a `results/<run>/model_walk_forward_diagnostics.csv` y
`results/<run>/meta_weights_by_snapshot.csv`.

## 9. Esquema de entrenamiento: walk-forward rodante, nunca congelado

El sistema es **walk-forward puro y rodante**: reentrena de forma continua a lo largo de **toda** la
ventana simulada, sin congelar nunca. `TRAIN_CUTOFF_DATE` es solo el punto **más temprano** en el
que se permite empezar a entrenar, no un punto tras el cual el modelo deja de aprender.

- Desde `train_cutoff_date` y en cadencia trimestral (`WALK_FORWARD_TRAIN_FREQUENCY`) hasta
  `PORTFOLIO_END_DATE`, en cada fecha de entrenamiento se **reentrenan los 3 agentes y se reajustan
  los pesos del meta-agente**, usando solo la ventana móvil de los últimos
  `MAX_WALK_FORWARD_TRAINING_YEARS` (4 por defecto) disponibles a esa fecha — nunca el futuro (la
  máscara de observabilidad se aplica igual que siempre).
- Los snapshots entre fechas trimestrales de reentrenamiento se puntúan con el último modelo/pesos
  entrenados hasta ese momento (la cadencia trimestral acota el coste), pero el modelo **sigue
  aprendiendo** en cada trimestre a lo largo de todo el periodo de cartera.

Función clave: `_train_and_apply_dates(all_dates, settings)` separa `train_dates` (subconjunto en
cadencia trimestral desde el corte en adelante, garantizando que el propio corte sea siempre el
primer punto de entrenamiento) de `apply_dates` (todas las fechas reciben una predicción). El
diagnóstico por snapshot registra `is_train_date`, `mode` (`walk_forward_model` / `fallback_garp`) y
`training_snapshot_date` (qué modelo se usó realmente) para que sea auditable.

**Por qué rodante y no congelado.** Un esquema anterior entrenaba una vez hasta un corte y
desplegaba ese modelo fijo el resto de la ventana; producía un rank-IC OOS cercano a cero o negativo
en los años finales porque el modelo congelado nunca se adaptaba al régimen de mercado contra el que
luego se le puntuaba. El reentrenamiento rodante mantiene el modelo adaptado al régimen vigente en
cada snapshot.

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
`dataviz`; el gráfico de pesos aprendidos del meta-agente muestra la evolución de los pesos de los
tres agentes (calidad / temporización / alpha) a lo largo de todo el walk-forward rodante.

`module/report.py` reutiliza las mismas funciones de métricas (`_metrics`, `drawdown_episodes`)
que consume el viewer — la etapa `report` apunta al mismo `index.html` que genera `viewer`, no
produce un segundo informe distinto.

## 16. Sistema de experimentos (`module/experiments/`)

### 16.1 Por qué existe

El pipeline corre **una** configuración. Pero la pregunta central del TFM no es "¿cuánto rinde esta
configuración?", sino **"¿la IA aprende de los datos?, ¿ese aprendizaje es estable?, ¿es útil?"**.
Responderla exige **comparar** configuraciones: apagar el aprendizaje y ver si el rank-IC cae,
re-sembrar el modelo y ver si la conclusión aguanta, subir los costes y ver si la utilidad
sobrevive. El sistema de experimentos convierte eso en algo reproducible: **un fichero declara los
escenarios, un comando los corre todos, y el resultado es un informe HTML que los compara entre sí**
(no un informe por escenario).

Hay evidencia de que la necesidad era real: el comentario en `module/ml.py` alrededor de
`AGENT_PRIOR_WEIGHTS` cita un barrido de pesos (`scripts/buscar_pesos_meta.py`) que se hizo *offline*
y no quedó versionado. `experiments/escenarios_pesos_meta.py` lo reconstruye de forma trazable.

### 16.2 Cómo se lanza

- `python -m module.experiments run experiments/escenarios_aprendizaje.py` (o cualquier fichero).
- `python -m module.experiments run todos` — junta los cuatro bloques.
- Desde `main.py`: `RUN_MODE="experiments"` + `EXPERIMENTS_FILE` en `environment.py`.

Un fichero de escenarios declara `SCENARIOS: list[Scenario]`. Cada `Scenario(name, why, overrides)`
lleva su **hipótesis** (`why`) junto al cambio, para que la memoria del TFM explique qué se probó y
por qué.

### 16.3 Qué ejecuta por escenario (y qué NO)

Un cambio de configuración solo puede alterar dos etapas: `ml` (puntuar el universo) y `backtest`.
Por eso el runner ejecuta **solo esas dos** por escenario, y **omite** el resto:

- `download`/`dataset`/`features` producen datos que **no dependen** de los overrides → se preparan
  **una vez** antes del barrido (`_ensure_experiment_inputs`) y se reutilizan. Esa función reproduce
  el pipeline normal corriendo solo las etapas cuyos artefactos falten, en cadena: `download` (que
  reconstruye `prices.parquet` y los demás crudos desde el JSON cacheado en `data/raw/json/`, **sin
  red** mientras `FORCE_RAW_DOWNLOAD=False`), `dataset` y `features`. Una vez en disco, corridas
  posteriores no repiten nada de esto.
- `research_ai` se omite (coste de LLM, y está desactivado por defecto).
- El `viewer`/`report` **individuales** se omiten: su salida es el informe por-run, redundante con el
  informe comparativo que sí genera el runner.

### 16.4 La pieza clave: overrides aislados y con tres mecanismos

Los parámetros configurables del proyecto **no están todos en un sitio**, y se ligan de tres formas
distintas — cada una necesita un override diferente (`module/experiments/overrides.py`):

1. **Campos de `Settings`** (dataclass `frozen`): fechas, costes, ventanas walk-forward. Se aplican
   con `dataclasses.replace`.
2. **Globales del propio módulo**, leídas en tiempo de llamada (`ml.AGENT_PRIOR_WEIGHTS`,
   `sizing.MAX_POSITION_WEIGHT`): `setattr` sobre ese módulo.
3. **Constantes traídas con `from environment import X`** (`portfolio.MIN_ROTATION_ADVANTAGE`,
   `portfolio.MAX_PORTFOLIO_SIZE`): al importarse quedan ligadas **por nombre en el namespace del
   consumidor**, así que reescribir `environment.X` NO tiene efecto — hay que reescribir el nombre en
   `portfolio`/`baselines`. El mapa `_STRATEGY_TARGETS` resuelve a qué módulo(s) va cada nombre.

El context manager `apply_overrides` **guarda el valor previo, aplica y restaura al salir**, de modo
que cada escenario corre aislado en el mismo proceso sin fugas al siguiente, y los valores por
defecto del proyecto nunca se tocan. (Para hacer la semilla del modelo overridable se promovió el
literal `random_state=42` a la constante `ml.RANDOM_STATE`, mismo valor por defecto.)

### 16.5 Caché del scoring caro

`ml` (walk-forward, re-entrena LightGBM cada trimestre) es la etapa cara; `backtest` es barata. Los
escenarios que **solo cambian estrategia** (umbrales, sizing, costes) no alteran el scoring, así que
sería absurdo re-entrenar. El runner calcula un `scoring_cache_key` = hash de **solo** lo que afecta
al scoring (overrides `ml.*`, campos ML de `Settings`, universo de tickers), entrena **una vez por
clave**, y restaura el scoring cacheado para los demás escenarios de esa clave (`re_scored=False` en
la tabla). Un matiz de corrección: `train_and_score` escribe a `data/processed/` (carpeta
**compartida**), así que el runner hace *backup/restore* de esos artefactos alrededor del barrido —
de lo contrario dejaría el scoring del último escenario en su sitio y contaminaría un `main.py`
posterior.

### 16.6 Métricas y salida

Por escenario, `collect_metrics` extrae una fila **priorizando el aprendizaje** sobre la economía —
porque esa es la pregunta del TFM:

- **Aprendizaje**: rank-IC OOS medio de `final_score` y su t-stat anual (promediado por año, no por
  snapshot, para no ponderar de más los años con más observaciones), nº de años con rank-IC positivo,
  percentil del test placebo/permutación, y mejora sobre la mejor baseline simple.
- **Economía**: alfa acumulada, information ratio, t-stat del exceso, multiplicador de coste de
  breakeven, turnover anual.

La salida queda en `results/escenarios/<fecha>/`: `comparison.csv` (una fila por escenario) e
`index.html` — un informe que **compara** los escenarios (tabla rankeada por rank-IC, barras de
rank-IC por escenario, dispersión rank-IC vs. alfa que visualiza el trade-off estabilidad/
rentabilidad, y la hipótesis de cada uno). Reutiliza el sistema de diseño de `module/viewer/shared.py`.

### 16.7 Los cuatro bloques de escenarios

Uno por faceta de la pregunta del TFM (`experiments/escenarios_*.py`):

- **`aprendizaje`** — *ablaciones*: apagan una pieza del aprendizaje (meta-agente aprendido, prior
  tilteado, penalización de consistencia, shrinkage) y se leen contra el baseline. La lógica: si
  apagarlo **no** empeora el rank-IC OOS ni el placebo, el sistema no estaba aprendiendo nada útil
  (resultado negativo válido); si empeora, es evidencia de aprendizaje real, no ruido.
- **`estabilidad`** — semillas (dispersión del resultado al re-sembrar), costes al doble, ventana de
  entrenamiento más corta, horizonte de etiqueta alternativo.
- **`utilidad`** — variaciones de agresividad de cartera (concentración, rotación, nº de posiciones)
  sin tocar el ML, leídas contra las baselines y la alfa neta de costes.
- **`pesos_meta`** — barrido del prior `AGENT_PRIOR_WEIGHTS` (§8.2.1), trazando el trade-off
  estabilidad vs. alfa.

Un resultado **negativo bien medido en cualquiera de los tres ejes es un entregable válido del TFM**,
no un fallo — el sistema está construido para poder mostrarlo con honestidad (coherente con §17).

## 17. Limitaciones metodológicas explícitas

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

**La alfa no proviene (principalmente) del ranking del modelo** — la limitación más importante y la
más fácil de malinterpretar. El rank-IC OOS de `final_score` es débil (~+0.02 de media en la ventana
operativa, negativo en 2020-2021) y la correlación entre el score de entrada `manager_score` y el
exceso realizado es prácticamente **nula** (`edge_attribution.csv`). La alfa acumulada (+114%)
procede de la **asimetría ganador/perdedor** —unas pocas posiciones ganadoras muy grandes, con el
sesgo de supervivencia detrás—, no de que el modelo ordene bien las acciones. Son **dos afirmaciones
distintas**: la robustez de la alfa (bootstrap, subperiodos, placebo/permutación) es real y se
sostiene, pero **no** es prueba de que el ML rankee. El informe y este documento las mantienen
separadas a propósito; confundirlas sería sobrevender el sistema. Mejorar el rank-IC de forma
*estable* (no la rentabilidad) es por eso el objetivo declarado del trabajo de modelo — ver §8.2.1 y
`docs/diagnostico_aprendizaje.md`.

**Baselines triviales que hay que batir (y uno que no se bate)**: `baseline_comparison.csv` compara
el sistema completo contra reglas de una sola señal sobre el mismo universo y fechas. El sistema
supera con holgura a `equal_weight_universe` y `valuation_only`, pero un baseline de **solo momentum**
(`momentum_only`) ha producido históricamente más alfa acumulada que el sistema completo — un
resultado negativo legítimo que se reporta, no se esconde: parte del edge del universo vive en el
momentum simple, y el aparato multi-agente no lo mejora en rentabilidad bruta (aunque sí aporta
gestión de riesgo, tesis explícitas y control de rotación que el baseline no tiene).

## 18. Pruebas (`tests/`)

```bash
pytest tests/
```

- `tests/test_train_cutoff_freeze.py`: verifica que `train_cutoff_date` es solo el punto **más
  temprano** de reentrenamiento, que las fechas de entrenamiento son trimestrales y **continúan
  más allá del corte** hasta el final de la historia, y que cada fecha ajusta su propio modelo
  sobre su ventana móvil — el modelo nunca se congela (test end-to-end con datos sintéticos vía
  fixture `synthetic_run`).
- `tests/test_meta_agent.py`: que `_fit_meta_weights` devuelve un simplex válido; que un agente
  informativo domina a los de ruido; que **duplicar una señal colapsa su rank-IC marginal** (no se
  paga dos veces); y que con historial escaso o sin ranking marginal se cae al prior.
- `tests/test_leakage.py`: que ninguna etiqueta futura sobrevive al enmascarado, el contrato
  point-in-time end-to-end, e invarianza (mutar una fila futura no cambia la ventana enmascarada).
- `tests/test_baselines.py` y `tests/test_robustness.py`: que el placebo/permutación distingue una
  señal real de ruido, que `top_n_monthly_returns` es point-in-time, y que el bootstrap por
  bloques, la sensibilidad a costes y los sub-periodos se comportan como se espera.
- `tests/test_features.py`: invariantes y rangos de las features.
- `tests/test_experiments.py`: que los overrides se **aplican y restauran** por escenario (aislamiento),
  que un override de una constante importada toca el módulo consumidor correcto, que el
  `scoring_cache_key` agrupa bien (estrategia comparte clave, ml no), que `collect_metrics` tolera
  tablas faltantes y promedia el t-stat por año, y que `_ensure_experiment_inputs` corre en cadena
  solo las etapas de datos (`download`→`dataset`→`features`) cuyos artefactos falten.

## 19. Estado del sistema (2026-07)

El run vigente cubre la ventana `2018-02-15`→`2026-06-15` (100 observaciones mensuales de cartera,
universo ~500 large-caps, benchmark SPY), en `results/full_2018-02-15_2026-06-15_M_cutoff2018-02-15/`.
Una copia congelada de las métricas del run *previo* al último cambio de modelo vive en
`results/_baseline_frozen/` para comparación.

Cambios de la última ronda de diagnóstico (2026-07), cada uno **medido** re-ejecutando el pipeline
completo y comparado contra el baseline congelado — no solo teoría:

1. **Diagnóstico de aprendizaje** (`scripts/diagnostico_rank_ic.py`, `docs/diagnostico_aprendizaje.md`):
   se midió el rank-IC OOS por agente y año. Hallazgo: `quality` es la señal estable (positiva 8/9
   años), `alpha` la más inestable; el alfa agregado no proviene del ranking sino de la asimetría
   ganador/perdedor (ver §17).
2. **Reponderación del prior del meta-agente** hacia la señal estable: `AGENT_PRIOR_WEIGHTS` pasó de
   `0.30/0.35/0.35` (Q/T/A) a **`0.45/0.30/0.25`**, y el `META_WEIGHT_FLOOR` ahora ancla el shrinkage
   hacia ese prior informado en vez de hacia equal-weight (§8.2.1). **Resultado medido**: rank-IC OOS
   medio +0.021→+0.024 y más estable (std 0.099→0.092, 2021 −0.109→−0.083), **y** a la vez alfa
   1.104→1.138, IR 0.875→0.958, t-stat 2.53→2.77 — mejora simultánea de aprendizaje y economía.
3. **Alternativa descartada** (documentada, no en el código): penalizar la varianza del rank-IC entre
   snapshots en el *criterio* de aprendizaje del meta-agente no mejoró nada (el problema era el prior,
   no el criterio); ver §8.2.1, "experimento 2A".

Sigue aplicando el criterio de honestidad del resto del proyecto: con ~100 observaciones mensuales,
cualquier mejora debe leerse junto al aviso de muestra pequeña y junto al hecho de que la alfa no
proviene del ranking (§17), no como prueba de un edge persistente del modelo.

Nota metodológica: las rondas de diagnóstico como esta —comparar variantes del modelo midiendo el
rank-IC OOS y la economía frente a un baseline— son justo lo que el **sistema de experimentos** (§16)
sistematiza y hace reproducible de aquí en adelante, en vez de mediante scripts sueltos.

## 20. Propósito académico

El proyecto está diseñado para ser defendible como Trabajo Fin de Máster porque combina:

- **Ingeniería de datos**: pipeline modular, point-in-time sin lookahead, lag de publicación de
  fundamentales.
- **ML reproducible**: walk-forward rodante con enmascarado auditable (reentrenamiento trimestral
  a lo largo de toda la ventana, sin congelar), diagnóstico de rank-IC visible y comparación de
  horizontes.
- **IA explicable**: 3 agentes especializados interpretables + meta-agente cuya lógica de
  combinación (rank-IC) es auditable paso a paso, no una caja negra.
- **Gestión dinámica**: cartera concentrada con tesis vivas, distinción compra-hoy vs. mantener,
  coste de oportunidad explícito.
- **Backtesting riguroso**: P&L ponderado por sizing real (`hybrid_weight`), coste de transacción
  por notional rotado, IR/TE/t-stat con aviso de muestra pequeña.
- **Transparencia**: todas las limitaciones documentadas explícitamente, ningún resultado
  presentado como más significativo de lo que la muestra permite.
