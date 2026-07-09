# GARP AI Portfolio System

Sistema de IA explicable para investigar si una cartera GARP / Value-Growth puede generar alpha frente a SPY mediante una cartera viva, revisiones periódicas y trazabilidad de tesis.

## Ejecución

```bash
python main.py
```

La configuración operativa vive en `environment.py`:

- `RUN_MODE`: `download`, `dataset`, `features`, `ml`, `backtest`, `viewer`, `full`
- `DATA_START_DATE`
- `PORTFOLIO_START_DATE`
- `PORTFOLIO_END_DATE`
- `PORTFOLIO_REVIEW_FREQUENCY`: `M`, `2M`, `Q`
- `DEV_MODE`: `true` o `false` (límite a `DEV_TICKERS` + SPY para desarrollo rápido)
- `FORCE_RAW_DOWNLOAD`: `False` reutiliza JSON raw cacheados; `True` fuerza redescarga
- `OPENAI_MODEL`
- `ENABLE_OPENAI_RESEARCH`

Configuración actual:

- Datos históricos desde `2000-01-01` (universo estático, ver "Limitaciones metodológicas" abajo)
- Simulación desde `PORTFOLIO_START_DATE`
- Walk-forward con `MIN_WALK_FORWARD_TRAINING_YEARS` (mín. efectivo) y `MAX_WALK_FORWARD_TRAINING_YEARS` (ventana máxima)
- Targets ML genuinamente forward-looking (quality, improvement, mispricing, alpha) con enmascarado de fuga de 12 meses
- Sizing hybrid (`0.35 * equal_weight + 0.65 * risk_adjusted_weight`) integrado en el P&L del backtest
- Coste de transacción ponderado por notional rotado, no por conteo de operaciones
- Métricas de rigor estadístico: information ratio, tracking error, t-stat con advertencia explícita de muestra pequeña

El archivo `.env` se usa solo para API keys:

- `FINNHUB_API_KEY`
- `OPENAI_API_KEY`

En `DEV_MODE=true` el universo se limita a `DEV_TICKERS` más `SPY`, permitiendo ciclos de desarrollo rápidos.

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

## Limitaciones metodológicas

**Sesgo de supervivencia**: `TICKERS` es un listado estático de empresas del S&P 500 actual aplicado retroactivamente desde 2000. No incluye nombres delisted/adquiridos ni excluye IPOs recientes. Esto constituye sesgo de supervivencia que hereda el entrenamiento ML. La cartera viva solo opera entre `PORTFOLIO_START_DATE` y `PORTFOLIO_END_DATE`, lo que acota pero no elimina el problema. Se documenta explícitamente en todas las conclusiones.

**Muestra pequeña**: ~40 observaciones mensuales típicas. No se aplican correcciones bootstrap, Sharpe deflactado ni ajustes por comparaciones múltiples. El t-stat del retorno en exceso es directional, no prueba de significancia estadística. Ver aviso explícito en `final_report.html`.

## Validación

El sistema ha sido auditado en tres dimensiones:
1. **Metodológica**: sin lookahead bias, targets ML genuinamente predictivos, walk-forward con enmascarado de fuga
2. **Robustez**: error handling por ticker, límite de reintentos en rate-limiting, informe de cobertura de descarga
3. **Presentación**: formato numérico unificado, toda la UI en español, diagnóstico walk-forward visible, tabla de drawdowns

Se incluye un smoke test con datos sintéticos para validar integración. Para ejecución real: configura `FINNHUB_API_KEY`, ajusta `RUN_MODE = "full"` en `environment.py` y ejecuta `python main.py`.
