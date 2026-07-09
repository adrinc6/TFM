# GARP AI Portfolio System

Pipeline de investigación (TFM) que responde a una pregunta concreta: **¿puede una estrategia
GARP (Growth At a Reasonable Price / Value-Growth), gestionada por un sistema de IA explicable
con varios agentes especializados, generar alpha frente a SPY?** El sistema usa un dataset
point-in-time (sin lookahead), un modelo ML walk-forward que aprende hasta una fecha de corte y
luego se congela, y un backtest de cartera concentrada mensual con un informe HTML de una sola
página como salida.

## Ejecución

```bash
python main.py
```

No hay CLI/argparse: toda la configuración de ejecución vive en `environment.py` (constantes
editables directamente, no se leen de variables de entorno salvo las API keys).

- `RUN_MODE`: `download`, `dataset`, `features`, `ml`, `watchlist`, `research_ai`, `backtest`,
  `viewer`, `report`, o `full`. Cada etapa es re-ejecutable de forma independiente mientras sus
  entradas parquet/CSV ya existan en disco.
- `DEV_MODE`: `True` restringe el universo a `DEV_TICKERS + SPY` — úsalo para iterar/depurar en
  vez del universo completo (~500 tickers).
- `FORCE_RAW_DOWNLOAD`: `False` reutiliza el JSON crudo cacheado en `data/raw/json/`; `True`
  fuerza redescarga desde Finnhub/Yahoo.
- `.env` solo contiene `FINNHUB_API_KEY` y `OPENAI_API_KEY` (parseado a mano en `environment.py`,
  sin dependencia de `python-dotenv`).

Instala dependencias con:

```bash
pip install -r requirements.txt
```

## Arquitectura del pipeline

`main.py` ejecuta las etapas en este orden fijo, controladas por `settings.run_mode`:

```text
download → dataset → features → ml → watchlist → research_ai → backtest → viewer → report
```

| Etapa | Entrada principal | Lee | Escribe |
|---|---|---|---|
| download | `module.ingest.pipeline.download_raw_data` | APIs Finnhub/Yahoo + caché `data/raw/json/` | `data/raw/*.parquet` |
| dataset | `module.dataset.build_master_dataset` | `data/raw/*.parquet` | `data/master/master_point_in_time.parquet` |
| features | `module.features.pipeline.build_features` | `data/master/*.parquet` | `data/processed/features.parquet` |
| ml | `module.ml.train_and_score` | `data/processed/features.parquet`, `data/raw/prices.parquet` | `data/processed/scored_universe.parquet`, diagnósticos walk-forward |
| watchlist | `module.strategy.selection.build_watchlist` | `data/processed/scored_universe.parquet` | `data/processed/watchlist.parquet`, `results/<run>/watchlist.csv` |
| research_ai | `module.research.ai.build_openai_research` | universo puntuado + noticias | `results/<run>/research_ai.csv` |
| backtest | `module.backtest.run_backtest` | universo puntuado + precios | ~20 CSV en `results/<run>/` y `.../audit/` |
| viewer | `module.viewer.build_viewer` | CSVs de `results/<run>/` | `results/<run>/viewer/index.html` |
| report | `module.report.build_final_report` | viewer ya construido | apunta al mismo `viewer/index.html` |

`Settings.run_dir` (`environment.py`) = `results/<dev|full>_<inicio>_<fin>_<frecuencia>_cutoff<fecha_corte>/`
— cambiar fechas, `DEV_MODE` o el esquema de entrenamiento apunta a una carpeta de resultados
distinta en vez de sobrescribir la anterior.

## Módulos

- `module/ingest/` — descarga y cachea datos crudos por ticker (Finnhub + Yahoo, sin dependencia
  de `yfinance`), con reintentos limitados y reporte de cobertura.
- `module/dataset.py` — construye el dataset maestro point-in-time (`ticker × snapshot_date`), sin
  lookahead: cada valor es el último disponible estrictamente antes o en la fecha del snapshot.
- `module/features/` — calcula los scores GARP transversales (calidad, crecimiento, valoración,
  momentum, moat, catalyst, riesgo) y features de tendencia/expectativa.
- `module/ml.py` — 4 agentes ML especializados + 1 meta-agente que aprende cómo combinarlos, con
  entrenamiento walk-forward hasta una fecha de corte y modelo congelado después (ver `doc.md`).
- `module/research/` — investigación determinista (y opcionalmente LLM) sobre tesis, moat,
  catalizadores y riesgos por empresa.
- `module/strategy/` — selección de watchlist, lógica de cartera concentrada (entradas/salidas) y
  cálculo de tamaño de posición.
- `module/backtest/` — simulación mensual de cartera viva, métricas de rendimiento (IR/TE/t-stat) y
  tablas de revisión BUY/SELL/HOLD.
- `module/viewer/` — informe HTML de una sola página, en español, con criterio estricto de utilidad
  (cada sección debe responder una pregunta del TFM o ayudar a depurar).
- `module/report.py` — métricas compartidas (CAGR/Sharpe/Sortino/drawdown/alpha) reutilizadas por
  el viewer; la etapa `report` apunta al mismo informe que `viewer` genera.

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

## Tests

```bash
pytest tests/
```

Cubren: ausencia de lookahead en el walk-forward, invariantes de features, y la correctitud del
esquema entrenar-hasta-corte-luego-congelar (ninguna fecha de entrenamiento supera el corte; toda
fecha posterior al corte reutiliza el mismo modelo/pesos).
