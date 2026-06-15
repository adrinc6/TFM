# Documentación técnica — Arquitectura GARP / Value-Growth

## Objetivo

El proyecto implementa un sistema multi-agent ML para gestionar una **cartera viva GARP / Value-Growth**: detecta empresas infravaloradas o razonablemente valoradas con crecimiento futuro probable, calidad empresarial, mejora fundamental y riesgo controlado; después las mantiene, revisa, reduce, vende o rota mediante Thesis Engine y Portfolio Intelligence. Los snapshots punto-en-tiempo siguen siendo la base anti-leakage, pero la unidad principal de gestión es cartera + evolución temporal + decisiones.

## Arquitectura soportada

La única arquitectura soportada es `garp_value_growth`.

### Agentes

1. **Quality Agent**: márgenes, ROIC/ROE/ROA, FCF, balance, eficiencia y estabilidad.
2. **Growth Agent**: revenue/EPS/FCF growth, aceleración, revisiones y consistencia.
3. **Valuation Agent**: P/E, PEG, EV/EBITDA, P/FCF, FCF yield, earnings yield y percentiles relativos.
4. **Fundamental Trend Agent**: delta de márgenes, ROIC, FCF, leverage y revisiones.
5. **Catalyst Agent**: surprises, revisiones, insiders, buybacks, sentimiento opcional y tailwinds.
6. **Risk/Bear Agent**: value traps, leverage, FCF débil, deterioro estructural y drawdowns.
7. **Technical Guardrail Agent**: momentum roto, volatilidad extrema, drawdowns y timing básico.
8. **Sector Rotation Agent**: prior top-down, no motor principal.
9. **Alpha Meta-Learner**: ranking, alpha esperado y riesgo.

## Target

`garp_composite_target` se calcula dentro de cada fold con:

```text
0.30 * alpha_vs_spy
+ 0.15 * alpha_sector_neutral
+ 0.20 * (future_fundamental_improvement - 0.5)
+ 0.15 * (expectation_gap - 0.5)
+ 0.10 * (initial_valuation_reasonableness - 0.5)
- 0.05 * (overexpectation_penalty - 0.5)
- 0.05 * (downside_penalty - 0.5)
```

El target puede usar retornos forward porque es etiqueta, pero estos campos están prohibidos como features mediante validación fail-fast.

## Anti-leakage

Protecciones principales:

- snapshots as-of;
- purged/embargo CV;
- validación de columnas prohibidas (`forward_`, `future_`, `target`, `tp_sl`, etc.);
- auditoría por fold de features y origen temporal;
- rankings cross-sectionales calculados solo dentro del snapshot/fold.

## Anti-momentum

Momentum no es agente comprador. Se audita por fold:

- correlación final_score vs momentum 6M/12M;
- contribución de `technical_guardrail_score`;
- ejemplos seleccionados con momentum mediocre;
- ejemplos descartados pese a momentum fuerte.

## Clasificación de oportunidades

Cada ticker se clasifica en:

- Growth infravalorado
- Quality Growth razonable
- Value con catalizador
- Compounder a precio razonable
- Turnaround
- Cíclica barata
- Value trap
- Growth caro
- Descartar

La clasificación usa reglas objetivas sobre scores de calidad, crecimiento, valoración, tendencia, catalizadores, riesgo y guardrail técnico.

## Métricas de validación de investigación

- Alpha del horizonte de etiqueta GARP vs SPY.
- Alpha sector-neutral.
- Hit rate de outperform.
- Drawdown y alpha/drawdown.
- Estabilidad por fold, sector y ticker.
- Value traps compradas/evitadas.
- Expensive growth comprado/descartado.
- Ablation por agente.

## Limpieza de arquitectura

La configuración antigua basada en agentes `fundamental`, `momentum` y `bear` como stack principal fue eliminada de los feature sets oficiales. Cualquier intento de crear esos agentes como configuración activa falla explícitamente. Las utilidades de salida secundaria solo pueden usarse como diagnóstico de riesgo/backtest; no son etiqueta ni perfil de entrenamiento soportado.


## Mispricing y moat proxy

`expectation_gap_score` cruza calidad/crecimiento punto-en-tiempo con valoración relativa para detectar crecimiento o calidad infravalorados. `overexpectation_penalty` castiga múltiplos extremos que sugieren expectativas demasiado exigentes. `moat_proxy_score` vive dentro de Quality y aproxima durabilidad mediante márgenes, ROIC, FCF/calidad y estabilidad histórica de margen.

## Configuración y tipos de ejecución

`python analyzer.py` no acepta argumentos. El comportamiento se decide en `environment.py`:

- `RUN_MODE = "portfolio_evolution"`: modo principal. Reutiliza `data_finnhub/master_dataset.parquet`, empieza en `PORTFOLIO_START_DATE`, termina en `PORTFOLIO_END_DATE` o en el último snapshot disponible y revisa según `PORTFOLIO_REVIEW_FREQUENCY`.
- `RUN_MODE = "portfolio_review"`: revisión puntual de `PORTFOLIO_REVIEW_TICKERS` o de `PORTFOLIO_POSITIONS_CSV` en `PORTFOLIO_REVIEW_DATE`.
- `RUN_MODE = "full_pipeline"`: descarga/consolida datos, construye dataset, entrena/evalúa folds walk-forward de investigación y genera artefactos.
- `RUN_MODE = "update_prices"`: actualiza precios/macro sin entrenar.

Configuración principal de cartera viva:

- `PORTFOLIO_START_DATE`: fecha de inicio.
- `PORTFOLIO_END_DATE`: fecha final; `None` usa el último snapshot disponible.
- `PORTFOLIO_REVIEW_FREQUENCY`: `M`, `2M` o `Q`.
- `GARP_TARGET_HORIZON_MONTHS`: horizonte solo para la etiqueta ML de investigación; no es una regla de permanencia.

## Cartera principal

La referencia de construcción de cartera es 5-10 posiciones (`GARP_MIN_STOCKS=5`, `GARP_MAX_STOCKS=10`) sin holding period fijo. La cartera empieza en una fecha inicial, se revisa periódicamente, conserva tesis intactas, vende tesis rotas o excesivamente valoradas y rota solo ante oportunidades claramente superiores. TP/SL no forma parte del núcleo de aprendizaje, ranking ni gestión.

## Portfolio Intelligence / Thesis Engine

La capa `portfolio_review` es independiente del núcleo de selección y no entrena agentes nuevos. Reutiliza snapshots punto-en-tiempo existentes para comparar `snapshot_compra` contra `snapshot_actual` y responder si una posición debe comprarse hoy, mantenerse, revisarse, reducirse o venderse.

Estados de tesis soportados:

- `Improving`: la calidad, crecimiento, tendencia, moat o catalyst mejoran frente a la compra.
- `Intact`: la tesis sigue vigente sin deterioro relevante.
- `Maturing`: la empresa sigue siendo buena, pero la infravaloración/catalyst se ha cerrado.
- `Weakening`: existen señales claras de deterioro.
- `Broken`: la tesis original ya no existe o falta snapshot actual.

Outputs principales:

- `position_health_score` 0-100: salud de la posición, no ranking de compra.
- `conviction_score` 0-100: confianza actual para seguir manteniendo la posición.
- `buy_hold_sell_rating`: `Strong Buy`, `Buy`, `Hold`, `Review`, `Reduce` o `Sell`.
- `exit_score` 0-100 y `exit_reason`: salida basada en tesis, no TP/SL.
- `valuation_status`: `Undervalued`, `Fairly Valued`, `Fully Valued`, `Overvalued` o `Extremely Overvalued`.
- `best_alternative_ticker` y `opportunity_cost_flag`: comparación contra nuevas oportunidades del snapshot actual.

El Thesis History Engine exporta:

- `portfolio_thesis_history.csv`: evolución de thesis score, position health, conviction, valoración, quality, growth, moat, expectation gap y catalyst.
- `portfolio_thesis_events.csv`: eventos automáticos como `Quality Upgrade`, `Growth Slowdown`, `Catalyst Exhausted`, `Overvaluation Risk`, `Thesis Improvement` o `Thesis Deterioration`.
- `portfolio_review_report.md`: resumen mensual/trimestral con posiciones críticas, eventos recientes, posibles ventas y mejores/peores posiciones.

La prioridad de revisión se clasifica como `Critical`, `High`, `Medium` o `Low`, priorizando deterioro reciente, tesis rota/debilitada, riesgo, sobrevaloración y coste de oportunidad.

## Portfolio Evolution Simulator

Con `RUN_MODE = "portfolio_evolution"`, `python analyzer.py` simula una cartera viva revisada periódicamente. No reconstruye fotografías independientes: mantiene posiciones cuya tesis sigue viva, vende o reduce tesis deterioradas, añade nuevas oportunidades para respetar 5-10 posiciones y registra todas las decisiones.

Outputs: `portfolio_evolution.csv`, `portfolio_transactions.csv`, `portfolio_monthly_holdings.csv`, `portfolio_decision_log.csv`, `portfolio_turnover.csv` y `portfolio_monthly_summary.json`. La frecuencia se controla con `PORTFOLIO_REVIEW_FREQUENCY` (`M`, `2M`, `Q`). La rotación exige ventajas materiales (`MIN_ROTATION_ADVANTAGE`, `MIN_SCORE_ADVANTAGE_TO_REPLACE`, `MIN_CONVICTION_ADVANTAGE`) y premia persistencia (`HOLD_WINNER_BONUS`, `THESIS_INTACT_HOLD_PREFERENCE`).

## Static Results Viewer

Cada ejecución relevante genera `viewer/index.html` como entrada a un visor HTML estático. El viewer es solo presentación: reutiliza CSV/JSON/Markdown existentes y no recalcula métricas, modelos, rankings ni backtests.

Páginas generadas:

- `run_summary.html`: configuración/resumen, top posiciones, opportunity types y distribución de scores.
- `portfolio_review.html`: conviction, thesis score, health, valoración, recomendación y prioridad.
- `portfolio_health.html`: posiciones fuertes, débiles, críticas, a revisar, vender o aumentar.
- `thesis_history.html`: evolución temporal de thesis score, conviction, health, moat, catalyst y expectation gap.
- `thesis_events.html`: timeline de eventos como upgrades, deterioros, re-rating y overvaluation risk.
- `opportunity_cost.html`: mejores alternativas y posiciones reemplazables.
- `watchlist.html`: mejores oportunidades disponibles a partir de artefactos ya exportados.
- `position_<ticker>.html`: detalle por posición con tesis original, tesis actual, cambios y recomendación.
- `snapshot_compare_<ticker>.html` y `thesis_radar_<ticker>.html`: comparación original vs actual y radar de tesis.
- `portfolio_evolution.html`, `portfolio_vs_benchmark.html`, `allocation_dashboard.html`, `thesis_change_report.html` y `alerts.html`: evolución de cartera, benchmark, asignación, cambios recientes y alertas.
- `portfolio_lifecycle.html`, `portfolio_turnover.html`, `decision_log.html`, `thesis_persistence.html`, `hold_winners.html` y `portfolio_timeline.html`: memoria de posiciones, churn, explicación profesional de decisiones, persistencia de tesis y ganadores mantenidos.

Uso:

1. Ajustar `RUN_MODE`, `PORTFOLIO_START_DATE`, `PORTFOLIO_END_DATE`, `PORTFOLIO_REVIEW_FREQUENCY` y, si aplica, `PORTFOLIO_REVIEW_TICKERS`/`PORTFOLIO_POSITIONS_CSV` en `environment.py`.
2. Ejecutar `python analyzer.py`.

## Auditoría final de sesgos

La validación empírica debe revisar dos artefactos por fold antes de interpretar resultados:

- `survivorship_bias_audit.json`: documenta si el universo se ancla a la membresía histórica del S&P 500 en la fecha de entrada, cuántos miembros activos existían, cuántos tenían snapshot disponible y cuántos fueron descartados por historial insuficiente.
- `sector_concentration_audit.csv`: reporta peso por sector, HHI sectorial, peso máximo sectorial y número de sectores seleccionados. La arquitectura mide la concentración de forma explícita; los límites sectoriales solo deben activarse si la evidencia local muestra una concentración estructural no deseada.

El target GARP no se ha parametrizado mediante grid search. Sus pesos siguen siendo heurísticos y transparentes para evitar curve fitting: alpha forward, alpha sectorial, mejora fundamental futura y expectation gap dominan la etiqueta; valoración razonable, penalización de expectativas excesivas y downside actúan como controles.
