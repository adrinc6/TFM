# GARP AI Portfolio System

Pipeline de investigación (TFM) que responde a una pregunta concreta: **¿puede una estrategia
GARP (Growth At a Reasonable Price / Value-Growth), gestionada por un sistema de IA explicable
con varios agentes especializados, generar alpha frente a SPY?** El sistema usa un dataset
point-in-time (sin lookahead), un modelo ML walk-forward rodante que se reentrena cada trimestre
a lo largo de toda la ventana (nunca se congela), y un backtest de cartera concentrada mensual
con un informe HTML de una sola página como salida.

## Ejecución

```bash
python main.py
```

No hay CLI/argparse: toda la configuración de ejecución vive en `environment.py` (constantes
editables directamente, no se leen de variables de entorno salvo las API keys).

- `RUN_MODE`: `download`, `dataset`, `features`, `ml`, `watchlist`, `backtest`,
  `viewer`, `report`, `full`, o `experiments`. Cada etapa es re-ejecutable de forma independiente
  mientras sus entradas parquet/CSV ya existan en disco; `experiments` barre escenarios en vez de
  correr el pipeline (ver más abajo).
- `DEV_MODE`: `True` restringe el universo a `DEV_TICKERS + SPY` — úsalo para iterar/depurar en
  vez del universo completo (~500 tickers).
- `FORCE_RAW_DOWNLOAD`: `False` reutiliza el JSON crudo cacheado en `data/raw/json/`; `True`
  fuerza redescarga desde Finnhub/Yahoo.
- `.env` solo contiene `FINNHUB_API_KEY` (parseado a mano en `environment.py`,
  sin dependencia de `python-dotenv`).

Instala dependencias con:

```bash
pip install -r requirements.txt
```

## Arquitectura del pipeline

`main.py` ejecuta las etapas en este orden fijo, controladas por `settings.run_mode`:

```text
download → dataset → features → ml → watchlist → backtest → viewer → report
```

| Etapa | Entrada principal | Lee | Escribe |
|---|---|---|---|
| download | `module.ingest.pipeline.download_raw_data` | APIs Finnhub/Yahoo + caché `data/raw/json/` | `data/raw/*.parquet` |
| dataset | `module.dataset.build_master_dataset` | `data/raw/*.parquet` | `data/master/master_point_in_time.parquet` |
| features | `module.features.pipeline.build_features` | `data/master/*.parquet` | `data/processed/features.parquet` |
| ml | `module.ml.train_and_score` | `data/processed/features.parquet`, `data/raw/prices.parquet` | `data/processed/scored_universe.parquet`, diagnósticos walk-forward |
| watchlist | `module.strategy.selection.build_watchlist` | `data/processed/scored_universe.parquet` | `data/processed/watchlist.parquet`, `results/<run>/watchlist.csv` |
| backtest | `module.backtest.run_backtest` | universo puntuado + precios | ~20 CSV en `results/<run>/` y `.../audit/` |
| viewer | `module.viewer.build_viewer` | CSVs de `results/<run>/` | `results/<run>/viewer/index.html` |
| report | `module.report.build_final_report` | viewer ya construido | apunta al mismo `viewer/index.html` |

`Settings.run_dir` (`environment.py`) = `results/<dev|full>_<inicio>_<fin>_<frecuencia>_cutoff<fecha_corte>/`
— cambiar fechas, `DEV_MODE` o el esquema de entrenamiento apunta a una carpeta de resultados
distinta en vez de sobrescribir la anterior.

## Experimentos (barrer escenarios)

El pipeline normal corre **una** configuración. Para responder las preguntas centrales del TFM
—**¿la IA aprende?, ¿es estable?, ¿es útil?**— hace falta comparar muchas: distintos pesos de
agentes, semillas, ventanas de entrenamiento, umbrales de cartera... El **runner de experimentos**
(`module/experiments/`) ejecuta una lista de escenarios de una sola vez y produce **un informe HTML
que los compara entre sí**, no uno por escenario.

```bash
python -m module.experiments run experiments/escenarios_aprendizaje.py
python -m module.experiments run todos      # junta los cuatro bloques en un solo barrido
```

También desde `main.py`: pon `RUN_MODE = "experiments"` y `EXPERIMENTS_FILE` (la ruta de un fichero
de escenarios o `"todos"`) en `environment.py`, y ejecuta `python main.py`.

**Qué hace por escenario (importante):** cada escenario corre el **pipeline completo** como una
ejecución normal —`ml` (puntuar el universo) → `watchlist` → `backtest` → `viewer` → `report`— y deja
su **propio informe navegable** en `results/escenarios/<fecha>/<escenario>/viewer/index.html`. Lo
único que NO repite son las etapas previas comunes a todos los escenarios
(`download`→`dataset`→`features`), que se preparan **una sola vez** antes del barrido: si sus
artefactos ya están en disco se reutilizan, y si faltan se construyen igual que en el pipeline normal
(`download` reconstruye `prices.parquet` desde el JSON cacheado en `data/raw/json/`, sin red mientras
`FORCE_RAW_DOWNLOAD=False`). Además, el **scoring caro** (`ml`) se ejecuta **una sola vez por
combinación de parámetros de ML**: los escenarios que solo cambian estrategia reutilizan ese scoring
cacheado (se marcan `re_scored=False`). Así el barrido "corre el pipeline" completo para cada
escenario pero sin recalcular nada de lo que es común — mucho más rápido que "correr `main.py` N
veces".

- Un **fichero de escenarios** (`experiments/escenarios_*.py`) declara `SCENARIOS: list[Scenario]`.
  Cada `Scenario(name, why, overrides)` cambia parámetros con prefijo de namespace: `settings.*`
  (campos de `Settings`), `ml.*` (constantes de `module/ml.py` como `AGENT_PRIOR_WEIGHTS`,
  `RANDOM_STATE`), `strategy.*` (umbrales de cartera y sizing). Los overrides se **aplican y
  restauran** por escenario, sin tocar los valores por defecto del proyecto.
- Cada escenario se aísla en `results/escenarios/<fecha>/<escenario>/` (con su propio
  `viewer/index.html`); la comparación queda en `results/escenarios/<fecha>/comparison.csv` e
  `index.html`. El informe comparativo es profundo: vista **global** (todos los escenarios + deltas
  frente al baseline) y una subsección **por bloque** (aprendizaje / estabilidad / utilidad /
  pesos_meta), donde cada escenario se compara por pares contra el baseline y enlaza a su informe
  propio. Prioriza las métricas de **aprendizaje** (rank-IC out-of-sample, placebo, mejora sobre
  baselines) sobre las económicas (alpha, IR, breakeven de costes).
- Ficheros incluidos, uno por pregunta del TFM: `escenarios_aprendizaje.py` (ablaciones que apagan
  el aprendizaje para ver si el rank-IC cae), `escenarios_estabilidad.py` (semillas, costes,
  ventana), `escenarios_utilidad.py` (agresividad de cartera), `escenarios_pesos_meta.py` (barrido
  del prior de pesos) y `escenarios_todos.py` (los cuatro juntos). Ver `docs/doc.md §16` y
  `docs/diagnostico_aprendizaje.md`.

## Módulos

- `module/ingest/` — descarga y cachea datos crudos por ticker (Finnhub + Yahoo, sin dependencia
  de `yfinance`), con reintentos limitados y reporte de cobertura.
- `module/dataset.py` — construye el dataset maestro point-in-time (`ticker × snapshot_date`), sin
  lookahead: cada valor es el último disponible estrictamente antes o en la fecha del snapshot.
- `module/features/` — calcula los scores GARP transversales (calidad, crecimiento, valoración,
  momentum, moat, catalyst, riesgo) y features de tendencia/expectativa.
- `module/ml.py` — 3 agentes ML especializados (calidad / temporización / alpha, cada uno
  LightGBM) + 1 meta-agente que aprende cómo combinarlos por **contribución marginal** (rank-IC
  parcial), no por rank-IC bruto, para no pagar dos veces por una señal compartida. El prior/ancla
  del meta-agente está inclinado a Calidad (`0.45/0.30/0.25`) porque es la señal de ranking estable
  (ver `doc.md` §8.2.1 y `docs/diagnostico_aprendizaje.md`). Entrenamiento walk-forward rodante que
  se reentrena cada trimestre a lo largo de toda la ventana, sin congelar.
- `module/research/` — investigación determinista sobre tesis, moat, catalizadores y riesgos por
  empresa (usada por `watchlist` y `backtest`).
- `module/strategy/` — selección de watchlist, lógica de cartera concentrada (entradas/salidas) y
  cálculo de tamaño de posición.
- `module/backtest/` — simulación mensual de cartera viva, métricas de rendimiento (IR/TE/t-stat) y
  tablas de revisión BUY/SELL/HOLD.
- `module/viewer/` — informe HTML de una sola página, en español, con criterio estricto de utilidad
  (cada sección debe responder una pregunta del TFM o ayudar a depurar).
- `module/report.py` — métricas compartidas (CAGR/Sharpe/Sortino/drawdown/alpha) reutilizadas por
  el viewer; la etapa `report` apunta al mismo informe que `viewer` genera.
- `module/experiments/` — runner de experimentos: por escenario aplica overrides aislados, corre
  solo `ml`+`backtest` (cacheando el scoring caro entre escenarios que comparten config de ML),
  recoge métricas de aprendizaje/economía y genera la tabla e informe comparativos.

Ver `doc.md` para la explicación exhaustiva de cada fase y decisión de diseño, y `results.md` para
cómo leer el informe HTML generado.

## Artefactos principales

```text
data/raw/*.parquet
data/raw/json/<fuente>/<ticker>/*.json
data/master/master_point_in_time.parquet
data/processed/features.parquet
data/processed/scored_universe.parquet
data/processed/model_explainability.json
data/processed/meta_weights_by_snapshot.parquet
results/<run>/model_walk_forward_diagnostics.csv
results/<run>/meta_weights_by_snapshot.csv
results/<run>/label_horizon_comparison.csv
results/<run>/portfolio_vs_benchmark.csv
results/<run>/current_portfolio.csv
results/<run>/action_journal.csv
results/<run>/sell_reasons_summary.csv
results/<run>/portfolio_monthly_summary.json
results/<run>/viewer/index.html   ← informe final
results/<run>/audit/*.csv         ← tablas pesadas de auditoría, enlazadas no embebidas
results/escenarios/<fecha>/comparison.csv   ← una fila por escenario (modo experiments)
results/escenarios/<fecha>/index.html       ← informe comparativo de escenarios
```

## Limitaciones metodológicas (explícitas, no corregidas)

**Sesgo de supervivencia**: `TICKERS` en `environment.py` es un listado estático de grandes
compañías actuales aplicado retroactivamente hasta `DATA_START_DATE`. No incluye nombres
delisted/adquiridos/caídos del índice en ese periodo, e incluye IPOs/spin-offs recientes que no
existían históricamente. Es una decisión de alcance documentada, no un bug: la cartera viva solo
opera entre `PORTFOLIO_START_DATE`/`PORTFOLIO_END_DATE`, lo que acota pero no elimina el sesgo, y
la ventana de entrenamiento ML lo hereda igualmente.

**Muestra pequeña**: la ventana de backtest son unas pocas decenas de observaciones mensuales.
`module/backtest/artifacts.py::excess_return_statistics` reporta information ratio, tracking error
y t-stat del retorno en exceso, pero deliberadamente no aplica bootstrap ni correcciones de
comparaciones múltiples — eso implicaría más precisión estadística de la que este tamaño de
muestra soporta. El t-stat es directional, no prueba de significancia.

**La alfa no proviene (principalmente) del ranking del modelo**: el rank-IC out-of-sample de
`final_score` es débil (~+0.02 de media, negativo en 2020-2021) y la correlación entre el score de
entrada y el exceso realizado es casi nula. La alfa acumulada procede de la asimetría
ganador/perdedor (pocas ganadoras grandes, con el sesgo de supervivencia detrás), no de que el
modelo ordene bien. La robustez de la alfa (bootstrap, subperiodos, placebo) es real pero es una
afirmación **distinta** de "el ML rankea"; el proyecto las mantiene separadas a propósito. Por eso
el trabajo de modelo persigue un rank-IC **estable**, no más rentabilidad. Detalle en
`docs/diagnostico_aprendizaje.md`.

**Un baseline de solo momentum bate al sistema en rentabilidad bruta**: `baseline_comparison.csv`
muestra que `momentum_only` ha producido más alfa acumulada que el sistema completo — resultado
negativo legítimo que se reporta, no se esconde.

## Tests

```bash
pytest tests/
```

Cubren: ausencia de lookahead en el walk-forward, invariantes de features, y la correctitud del
esquema walk-forward rodante (el corte es solo el punto más temprano de reentrenamiento; las fechas
de entrenamiento continúan trimestralmente más allá del corte y cada una ajusta su propio modelo
sobre su ventana móvil de historia — el modelo nunca se congela).
