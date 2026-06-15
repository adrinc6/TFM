# GARP / Value-Growth Redesign Plan

## Arquitectura nueva

El proyecto mantiene el pipeline automatizable existente: descarga/consolidación, dataset maestro punto-en-tiempo, snapshots as-of, walk-forward, auditorías anti-leakage, benchmarks y exportación de artefactos. El cambio central es que el aprendizaje deja de estar orientado principalmente a etiquetas TP/SL y pasa a una etiqueta compuesta GARP/value-growth.

### Agentes

1. `quality`: calidad de negocio, márgenes, ROIC/ROE/ROA, FCF, balance y consistencia.
2. `growth`: crecimiento histórico y probable, aceleración, EPS/FCF/revenue y revisiones.
3. `valuation`: agente central de margen de seguridad, múltiplos, yields y valoración relativa.
4. `fundamental_trend`: dirección de fundamentales, márgenes, ROIC, leverage, FCF y revisiones.
5. `catalyst`: potencial de re-rating por revisiones, surprises, insiders, sector/regime y sentimiento opcional.
6. `risk_bear`: filtro negativo de value traps, leverage, deterioro, FCF débil y drawdowns.
7. `technical_guardrail`: timing/riesgo técnico; no es motor comprador principal.
8. `alpha_meta_learner`: meta-modelo que combina scores para ranking, alpha esperado y riesgo.

`sector_rotation` se conserva como prior top-down, no como motor principal de compra.

## Target nuevo

`PRIMARY_LABEL_MODE = "garp_composite"` genera `garp_composite_target` con:

- 45% alpha forward 12M vs SPY.
- 20% alpha sector-neutral.
- 15% proxy de mejora fundamental/calidad-crecimiento-tendencia.
- 10% valoración inicial razonable.
- -10% penalización de downside/fragilidad.

La clase positiva se define por el percentil `GARP_OUTPERFORM_QUANTILE` dentro del fold de entrenamiento para evitar mirar el test.

## Features

Se añaden universos explícitos por agente: quality, growth, GARP valuation, fundamental trend, catalyst, risk/bear y technical guardrail. También se enriquecen features cross-sectionales con percentiles de valoración sectoriales y universales.

## Scoring

La filosofía por defecto es:

- Quality 20%.
- Growth 20%.
- Valuation / margin of safety 25%.
- Fundamental trend 15%.
- Catalysts 10%.
- Technical guardrail 5%.
- Risk/bear como penalización/safety score equivalente a -20%.

El meta-learner aprende la mezcla final walk-forward con esos agentes como inputs.

## Evaluación

La métrica principal pasa a ser cartera viva GARP / Value-Growth:

- evolución acumulada vs SPY/benchmark;
- alpha continuo de cartera viva;
- turnover, holding period realizado y persistencia de tesis;
- hit rate de outperform;
- alpha/drawdown;
- estabilidad por fold, sector y ticker;
- evitación de value traps y growth caro.

TP/SL se conserva como evaluación secundaria o variante de gestión de riesgo, no como corazón del objetivo de entrenamiento.

## Riesgos metodológicos

- Evitar que proxies de valoración inicial dominen la etiqueta y reproduzcan features de entrada sin generalización.
- Auditar cobertura por feature para no penalizar empresas con datos faltantes de forma no intencionada.
- Medir robustez sectorial: GARP puede concentrarse en sectores con márgenes naturalmente altos.
- Mantener purged/embargo CV y snapshots as-of para evitar leakage.
- No optimizar hiperparámetros contra folds test; usar solo train/OOF.

## Orden de implementación

1. Añadir configuración GARP y universos de features.
2. Sustituir stack de agentes base por agentes GARP/value-growth.
3. Añadir target compuesto GARP en el fold walk-forward.
4. Enriquecer features cross-sectionales de valoración.
5. Mantener TP/SL como salida secundaria.
6. Actualizar documentación y tests de importación/configuración.
7. Ejecutar `python analyzer.py` con overrides pequeños cuando haya datos suficientes.
