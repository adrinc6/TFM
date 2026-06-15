# Multi-Agent ML Stock Picker — GARP / Value-Growth

Sistema cuantitativo automatizable para seleccionar acciones del S&P 500 mediante una arquitectura multi-agente de machine learning, snapshots punto-en-tiempo y backtesting walk-forward. La filosofía principal es **GARP / Value-Growth**: detectar empresas de calidad, con crecimiento futuro probable, infravaloradas o razonablemente valoradas frente a sus fundamentales, tendencia y riesgo.

> Proyecto académico/TFM. No constituye asesoramiento financiero.

## Filosofía actual

El sistema ya no usa TP/SL como objetivo de aprendizaje principal. El objetivo primario es seleccionar un portafolio fundamental Buy & Hold 12M y validar alpha frente a SPY y frente al sector. Los indicadores técnicos se mantienen únicamente como guardrails de riesgo/timing.

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
6. Selección de portafolio por fold.
7. Evaluación 12M vs SPY, sector-neutral y Buy & Hold.
8. Exportación de auditorías anti-leakage, anti-momentum, explicabilidad por ticker/fold y reportes.

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

La cartera principal es Buy & Hold 12M y debe mantenerse entre 5 y 10 posiciones. TP/SL no participa en features, target, scoring, ranking ni selección; queda aislado como diagnóstico opcional de salidas.

## Configuración principal

`environment.py` es la fuente de verdad. Parámetros clave:

| Parámetro | Descripción |
|---|---|
| `PRIMARY_STRATEGY_PROFILE` | Debe ser `garp_value_growth`. |
| `REQUIRED_GARP_AGENTS` | Agentes obligatorios; el sistema falla si falta alguno. |
| `GARP_SCORE_WEIGHTS` | Pesos transparentes de scoring/reporting. |
| `GARP_MIN_STOCKS` / `GARP_MAX_STOCKS` | Rango obligatorio del portafolio: 5-10 acciones. |
| `PORTFOLIO_OPTIMIZER` | Optimizador de pesos (`hrp`, `risk_parity`, `markowitz`). |
| `HOLDING_PERIOD_MONTHS` | Horizonte objetivo, normalmente 12 meses. |

## Ejecución

```bash
python analyzer.py
```

Revisión ligera de una posición, varios tickers o una cartera existente sin
ejecutar el backtest completo:

```bash
python analyzer.py portfolio_review --tickers AAPL,MSFT,NVDA --review-date 2026-03-31
python analyzer.py portfolio_review --positions portfolio.csv --review-date 2026-03-31
```

`portfolio.csv` debe incluir `ticker` y puede añadir `weight`,
`purchase_date`, `avg_cost` y `snapshot_date`. Si existe `snapshot_date`, la
capa compara la tesis original de compra contra la tesis actual.

Además de la revisión puntual, el modo genera historial de tesis y eventos
materiales para revisiones mensuales/trimestrales sin ruido diario:
`portfolio_thesis_history.csv`, `portfolio_thesis_events.csv` y
`portfolio_review_report.md`.

Cada ejecución relevante genera también un viewer HTML estático en `viewer/`
con `index.html`, dashboards de resumen, cartera, salud, historial de tesis,
eventos, coste de oportunidad, watchlist y páginas por posición. No requiere
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
- `viewer/index.html`: visor HTML estático para revisar visualmente resultados, cartera, tesis, eventos, watchlist y páginas por posición.
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
