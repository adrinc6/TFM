# Multi-Agent ML Stock Picker — GARP / Value-Growth

Sistema cuantitativo automatizable para gestionar una **cartera viva GARP / Value-Growth** de 5 a 10 acciones usando datos punto-en-tiempo, agentes ML por dominio, Thesis Engine, Portfolio Intelligence y revisión periódica. La filosofía principal es detectar empresas de calidad, con crecimiento probable, mejora fundamental e infravaloración relativa, mantenerlas mientras la tesis siga viva y vender/rotar solo cuando exista una razón clara.

> Proyecto académico/TFM. No constituye asesoramiento financiero.

## Filosofía actual

El sistema ya no usa TP/SL ni un holding period fijo como objetivo de estrategia principal. La unidad operativa es:

`fecha_inicio → cartera inicial → revisiones periódicas → mantener tesis intactas → vender tesis rotas/sobrevaloradas → añadir oportunidades claramente superiores → fecha_final`.

Los snapshots siguen siendo la base anti-leakage para entrenar y reconstruir historia, pero la estrategia principal es una cartera viva con memoria de posición, persistencia de tesis, control de rotación y comparación continua contra benchmark.

## Pipeline

1. Descarga y consolidación de datos financieros, precios, insiders, sentimiento y macro.
2. Construcción del dataset maestro punto-en-tiempo por ticker/snapshot.
3. Enriquecimiento cross-sectional sectorial/universal sin mirar el futuro.
4. Entrenamiento walk-forward de agentes GARP:
   - `quality`
   - `growth`
   - `valuation`
   - `fundamental_trend`
   - `catalyst`
   - `risk_bear`
   - `technical_guardrail`
   - `sector_rotation` como prior top-down
5. Meta-learner para ranking, alpha esperado y riesgo.
6. Simulación principal de cartera viva con revisiones `M`, `2M` o `Q`.
7. Evaluación continua contra benchmark, turnover, decisiones, tesis y holdings mes a mes.
8. Walk-forward GARP disponible como validación de investigación, no como la unidad principal de gestión.
9. Exportación de auditorías anti-leakage, anti-momentum, explicabilidad, viewer HTML y reportes.

## Target GARP compuesto

`garp_composite_target` combina:

- 30% alpha forward 12M vs SPY.
- 15% alpha sector-neutral.
- 20% mejora fundamental futura (márgenes, ROIC, FCF/EPS/calidad de beneficios cuando existen snapshots posteriores en train).
- 15% expectation gap / mispricing entre calidad-crecimiento observado y valoración actual.
- 10% valoración inicial razonable.
- -5% penalización de expectativas excesivas.
- -5% penalización de downside/fragilidad.

El umbral positivo se calcula únicamente en el fold de entrenamiento (`GARP_OUTPERFORM_QUANTILE`) para evitar leakage.

## Mispricing, moat y expectativas excesivas

- `moat_proxy_score`: proxy de durabilidad dentro de Quality basado en márgenes, ROIC, calidad/FCF y estabilidad histórica de margen cuando hay historial.
- `expectation_gap_score`: mide si la calidad + crecimiento observados parecen infravalorados por la valoración relativa actual.
- `overexpectation_penalty`: penaliza PEG, EV/Sales, P/S o múltiplos relativos extremos cuando el precio parece descontar un futuro perfecto.
- `technical_guardrail_score`: solo controla riesgo/timing; no es tesis compradora.

La cartera principal debe mantenerse entre 5 y 10 posiciones y no tiene vencimiento fijo. TP/SL no participa en features, target, scoring, ranking, selección ni decisiones de cartera; queda aislado como diagnóstico opcional de salidas.

## Configuración principal

`environment.py` es la fuente de verdad. Parámetros clave:

| Parámetro | Descripción |
|---|---|
| `PRIMARY_STRATEGY_PROFILE` | Debe ser `garp_value_growth`. |
| `RUN_MODE` | Modo ejecutado por `python analyzer.py`: `portfolio_evolution`, `portfolio_review`, `full_pipeline` o `update_prices`. |
| `PORTFOLIO_START_DATE` / `PORTFOLIO_END_DATE` | Periodo de cartera viva. Si `PORTFOLIO_END_DATE=None`, se usa el último snapshot disponible. |
| `PORTFOLIO_REVIEW_FREQUENCY` | Frecuencia de revisión: `M`, `2M` o `Q`. |
| `REQUIRED_GARP_AGENTS` | Agentes obligatorios; el sistema falla si falta alguno. |
| `GARP_SCORE_WEIGHTS` | Pesos transparentes de scoring/reporting. |
| `GARP_MIN_STOCKS` / `GARP_MAX_STOCKS` | Rango obligatorio del portafolio: 5-10 acciones. |
| `GARP_TARGET_HORIZON_MONTHS` | Horizonte de etiqueta ML para investigación walk-forward; no es una regla de permanencia. |

## Ejecución

El proyecto no usa `argparse`: se configura en `environment.py` y se ejecuta siempre igual:

```bash
python analyzer.py
```

Tipos de ejecución:

- `RUN_MODE = "portfolio_evolution"`: modo principal. Reutiliza `data_finnhub/master_dataset.parquet`, simula la cartera viva desde `PORTFOLIO_START_DATE` hasta `PORTFOLIO_END_DATE` o último snapshot y genera CSV/JSON/viewer.
- `RUN_MODE = "portfolio_review"`: revisión puntual de `PORTFOLIO_REVIEW_TICKERS` o `PORTFOLIO_POSITIONS_CSV` contra `PORTFOLIO_REVIEW_DATE`.
- `RUN_MODE = "full_pipeline"`: ejecuta descarga/consolidación/dataset/entrenamiento walk-forward de investigación y genera artefactos.
- `RUN_MODE = "update_prices"`: actualiza precios/macro sin entrenar ni simular.

`PORTFOLIO_POSITIONS_CSV` debe apuntar a un CSV que incluya `ticker` y puede añadir `weight`,
`purchase_date`, `avg_cost` y `snapshot_date`. Si existe `snapshot_date`, la
capa compara la tesis original de compra contra la tesis actual.

Además de la revisión puntual, el modo genera historial de tesis y eventos
materiales para revisiones mensuales/trimestrales sin ruido diario:
`portfolio_thesis_history.csv`, `portfolio_thesis_events.csv` y
`portfolio_review_report.md`. La simulación de cartera viva exporta `portfolio_evolution.csv`, `portfolio_transactions.csv`, `portfolio_monthly_holdings.csv`, `portfolio_decision_log.csv`, `portfolio_turnover.csv` y `portfolio_monthly_summary.json`.

Cada ejecución relevante genera también un viewer HTML estático en `viewer/`
con `index.html`, dashboards de resumen, cartera, salud, historial de tesis,
eventos, coste de oportunidad, lifecycle, turnover, decision log, thesis persistence, hold winners, watchlist y páginas por posición. No requiere
servidor ni framework web; solo HTML, CSS, Plotly y DataTables vía CDN.

Tests:

```bash
pytest -q
```

Tests focalizados del rediseño:

```bash
pytest -q tests/test_garp_value_growth_redesign.py
```

## Artefactos de validación

Por fold se exportan, entre otros:

- `garp_feature_leakage_audit.csv`: features usadas, origen temporal y columnas prohibidas.
- `garp_anti_momentum_audit.json`: correlación score final vs momentum 6M/12M y ejemplos anti-momentum.
- `garp_agent_score_contribution.csv`: contribución/correlación de agentes con el score final.
- `survivorship_bias_audit.json`: confirma si el fold usa membresía histórica del S&P 500 en la fecha de entrada y cuantifica miembros activos sin snapshot/precios disponibles.
- `sector_concentration_audit.csv`: pesos por sector, HHI, peso máximo sectorial y número de sectores seleccionados para medir concentración sin imponer límites adicionales.
- `portfolio_review_positions.csv`: salud de posiciones existentes, estado de tesis, rating Buy/Hold/Review/Reduce/Sell, exit score y coste de oportunidad.
- `portfolio_thesis_history.csv` / `portfolio_thesis_events.csv`: evolución histórica de tesis, conviction score, valoración y eventos relevantes por posición.
- `portfolio_evolution.csv` / `portfolio_transactions.csv` / `portfolio_turnover.csv`: película de cartera viva, movimientos ADD/HOLD/REDUCE/SELL, persistencia, holding period, turnover y comparación continua frente a benchmark cuando hay precios.
- `viewer/index.html`: visor HTML estático para revisar visualmente resultados, cartera, tesis, eventos, evolución, asignación, alertas, watchlist y páginas por posición.
- Scores por ticker con `opportunity_type`, `moat_proxy_score`, `expectation_gap_score`, flags value trap / expensive growth, drivers, riesgos y razones de selección/descarte.

## Estructura

```text
analyzer.py                          # Entrypoint
analyzer_II.py                       # Escenarios en paralelo
environment.py                       # Configuración GARP
module/agents/                       # Agentes ML y meta-learner
module/common/                       # As-of, validación GARP, métricas, régimen, optimización
module/steps/step_01_data/           # Descarga/consolidación
module/steps/step_02_dataset/        # Dataset punto-en-tiempo
module/steps/step_03_training/       # Entrenamiento walk-forward
module/steps/step_04_evaluation/     # Evaluación, reporting y backtesting
```
