# GARP AI Portfolio System

Sistema de IA explicable para investigar si una cartera GARP / Value-Growth puede generar alpha frente a SPY mediante una cartera viva, revisiones periódicas y trazabilidad de tesis.

## Ejecución

```bash
python main.py
```

La configuracion operativa vive en `environment.py`:

- `RUN_MODE`: `download`, `dataset`, `features`, `ml`, `backtest`, `viewer`, `full`
- `DATA_START_DATE`
- `PORTFOLIO_START_DATE`
- `PORTFOLIO_END_DATE`
- `PORTFOLIO_REVIEW_FREQUENCY`: `M`, `2M`, `Q`
- `DEV_MODE`: `true` o `false`
- `FORCE_RAW_DOWNLOAD`: `False` reutiliza JSON raw ya descargados; `True` fuerza redescarga.
- `OPENAI_MODEL`
- `ENABLE_OPENAI_RESEARCH`

Configuracion actual relevante:

- Datos desde `2000-01-01`.
- Simulacion desde `PORTFOLIO_START_DATE`.
- Walk-forward con minimo `MIN_WALK_FORWARD_TRAINING_YEARS` anos efectivos y maximo `MAX_WALK_FORWARD_TRAINING_YEARS` anos de ventana historica.
- El dataset maestro no materializa snapshots desde `DATA_START_DATE` si no hacen falta: empieza en `PORTFOLIO_START_DATE - MAX_WALK_FORWARD_TRAINING_YEARS`, respetando siempre el minimo `DATA_START_DATE`.
- En cada fecha, el entrenamiento usa filas desde `fecha_actual - MAX_WALK_FORWARD_TRAINING_YEARS` hasta la propia fecha actual. No se descarta el ultimo ano.
- Las filas recientes entran en el entrenamiento, pero si su alpha futuro aun no seria observable en esa fecha, ese componente usa fallback GARP para evitar fuga de informacion futura.
- Horizonte de etiqueta: 12 meses.

El archivo `.env` se usa solo para API keys:

- `FINNHUB_API_KEY`
- `OPENAI_API_KEY`

En `DEV_MODE=true` el universo se limita a `DEV_TICKERS` más `SPY`.

## Arquitectura

```text
data_download -> dataset_builder -> features -> ml -> research -> thesis -> watchlist -> portfolio -> backtest -> viewer
```

- `module/data_download`: descarga y almacena datos raw. No calcula features ni scores.
- `module/dataset_builder`: construye `ticker x snapshot_date` sin usar información futura.
- `module/business_temporal`: calcula trends historicos de calidad, ROIC, margenes, FCF, crecimiento y moat.
- `module/features`: crea factores de quality, growth, valuation, moat, catalyst y risk.
- `module/ml`: entrena LightGBM, puntúa el universo y exporta explicabilidad.
- `module/expectations`: calcula expected growth, implied growth, realized growth y expectation gap.
- `module/research`: genera investigación automatizada de negocio, moat, catalysts, riesgos y tesis.
- `module/thesis`: calcula thesis, health, conviction, exit y estado de tesis.
- `module/watchlist`: mantiene oportunidades interesantes fuera o antes de cartera.
- `module/portfolio`: construye y revisa una cartera concentrada, distingue comprar hoy de mantener y calcula sizing.
- `module/backtest`: simula cartera viva, decisiones, rotación y benchmark.
- `module/viewer`: genera el visor HTML en `results/<run>/viewer/`.
- `module/report`: genera `results/<run>/final_report.html` como base del analisis experimental.

## Artefactos Principales

- `data/raw/*.parquet`
- `data/raw/json/<source>/<ticker>/*.json`
- `data/master/master_point_in_time.parquet`
- `data/processed/features.parquet`
- `data/processed/scored_universe.parquet`
- `data/processed/model_explainability.json`
- `expl_results.md`: guia de lectura de resultados.
- `results/<run>/result_manifest.json`
- `results/<run>/executive_summary.csv`
- `results/<run>/current_portfolio.csv`
- `results/<run>/tracking_dashboard.csv`
- `results/<run>/action_journal.csv`
- `results/<run>/position_performance.csv`
- `results/<run>/buy_rationale.csv`
- `results/<run>/sell_reasons_summary.csv`
- `results/<run>/top_opportunities_latest.csv`
- `results/<run>/strategy_learning_log.csv`
- `results/<run>/improvement_backlog.csv`
- `results/<run>/portfolio_vs_benchmark.csv`
- `results/<run>/portfolio_transactions.csv`
- `results/<run>/watchlist.csv`
- `results/<run>/portfolio_monthly_summary.json`
- `results/<run>/final_report.html`
- `results/<run>/viewer/index.html`
- `results/<run>/audit/*.csv`: tablas grandes de auditoria.

## Nota de Validación

La prueba local incluida en este estado del workspace se genero con datos sinteticos minimos para validar integracion sin depender de red. Para una ejecucion real, configura `FINNHUB_API_KEY`, ajusta `RUN_MODE = "full"` en `environment.py` y ejecuta `python main.py`.
