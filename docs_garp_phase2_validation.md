# Fase 2 — Validación metodológica y limpieza GARP / Value-Growth

## 1. Estado implementado correctamente

- La configuración principal es `garp_value_growth` y los agentes obligatorios se validan con fail-fast.
- El stack activo está formado por `quality`, `growth`, `valuation`, `fundamental_trend`, `catalyst`, `risk_bear` y `technical_guardrail`.
- El target primario es `garp_composite_target`, no TP/SL.
- Las features forward (`forward_return`, targets, alpha futuro, TP/SL/outcome) están prohibidas como inputs de agentes.
- Se exportan auditorías por fold para leakage, anti-momentum y contribución de agentes.
- Cada ticker recibe clase de oportunidad, drivers, riesgos y razones de selección/descarte.

## 2. Target compuesto GARP

| Componente | Fórmula | Peso | Racional | Riesgos |
|---|---:|---:|---|---|
| Alpha vs SPY | `forward_return - spy_return` | 30% | Premia outperform frente al benchmark invertible. | Puede favorecer beta alta si no se compensa con downside. |
| Alpha sector-neutral | `alpha - median(alpha sector)` | 15% | Evita que el modelo sea solo una apuesta sectorial. | Sectores pequeños pueden tener medianas ruidosas. |
| Mejora fundamental futura | delta/rank futuro de margen, ROIC, FCF/EPS y earnings quality en train | 20% | Premia empresas que después demostraron mejora real. | Disponible para label, nunca como feature. |
| Expectation gap | calidad+crecimiento observado frente a valoración actual | 15% | Captura mispricing: negocio mejor de lo que descuenta el precio. | Puede ser ruidoso si múltiplos sectoriales son extremos. |
| Valoración razonable | rank de yields altos y múltiplos bajos | 10% | Evita growth caro y busca margen de seguridad. | Puede favorecer value traps si calidad/riesgo son débiles. |
| Overexpectation penalty | PEG/EV-Sales/P-S/múltiplos relativos extremos | -5% | Evita empresas donde el precio descuenta futuro perfecto. | Puede penalizar compounders premium. |
| Downside penalty | ranks de leverage, volatilidad y distancia a máximos | -5% | Penaliza fragilidad y drawdown probable. | Puede penalizar turnarounds legítimos. |

## 3. Validación anti-leakage

Controles implementados:

- `validate_no_forward_features()` falla si un agente intenta usar columnas forward, target, outcome o TP/SL.
- `validate_critical_garp_features()` falla si la cobertura del backbone GARP es insuficiente.
- `garp_feature_leakage_audit.csv` lista features usadas, origen temporal y flags de columnas prohibidas.
- Los rankings cross-sectionales se calculan por snapshot (`date`) y sector dentro del dataframe actual, no con datos posteriores.

Variables permitidas solo como etiqueta:

- `forward_return`
- `spy_alpha`
- `sector_alpha`
- `garp_composite_target`
- proxies derivados dentro de `_prepare_fold_labels()`

## 4. Validación anti-momentum

El sistema exporta por fold:

- correlación `final_score` vs `momentum_6m`;
- correlación `final_score` vs `momentum_12m`;
- contribución/correlación de `technical_guardrail_score` frente a agentes fundamentales;
- ejemplos seleccionados con momentum mediocre;
- ejemplos descartados pese a momentum fuerte.

Criterio de aceptación recomendado:

- `abs(corr(final_score, momentum_6m)) < corr(final_score, valuation_score)`;
- `abs(corr(final_score, momentum_6m)) < corr(final_score, quality_score)`;
- `technical_guardrail_score` debe aportar control de riesgo, no dominar ranking.

## 5. Validación de agentes y ablation

El ablation debe ejecutarse quitando un agente por vez y comparando:

- alpha vs SPY;
- alpha sector-neutral;
- Sharpe;
- Sortino;
- max drawdown;
- hit rate;
- estabilidad por fold.

Agentes críticos que no pueden faltar: quality, growth, valuation, risk_bear.

## 6. Backtest y comparaciones

La comparación principal debe ser:

- GARP Buy & Hold 12M vs SPY;
- GARP Buy & Hold 12M vs sector;
- GARP vs misma selección con salida diagnóstica si se habilita;
- GARP vs baselines no entrenados (EW universe, value, momentum baseline, random top-N).

La comparación con la estrategia anterior debe hacerse en una rama o tag histórico; no se mantiene como ruta productiva.

## 7. Value traps

Definición operativa propuesta:

Una empresa es value trap si cumple simultáneamente:

- valoración atractiva (`valuation_score >= 0.62` o múltiplos baratos);
- baja calidad o tendencia (`quality_score < 0.42` o `fundamental_trend_score < 0.40`);
- riesgo alto (`risk_bear_score < 0.42`);
- forward alpha negativo o drawdown severo en etiqueta.

Métricas:

- traps compradas;
- traps descartadas;
- precision de descarte;
- pérdida media de traps compradas;
- contribución de `risk_bear` a evitar traps.

## 8. Expensive growth

Definición operativa propuesta:

- `growth_score >= 0.65`;
- `valuation_score < 0.38`;
- deterioro posterior de alpha o compresión de múltiplos si está disponible.

Métricas:

- expensive growth comprado;
- expensive growth descartado;
- retorno/alpha posterior;
- drawdown posterior;
- peso promedio en cartera.

## 9. Código eliminado / migrado

Eliminado/migrado en esta fase:

- Feature sets oficiales antiguos `FUNDAMENTAL_FEATURE_COLUMNS`, `MOMENTUM_FEATURE_COLUMNS`, `BEAR_FEATURE_COLUMNS`.
- Clase/archivo `UniversalTpSlAgent` renombrado a `GarpDomainAgent`.
- `PRIMARY_LABEL_MODE` y ruta alternativa de label TP/SL eliminados.
- Imports y documentación principal actualizados para `GARP_MAX_STOCKS` / `GARP_MIN_STOCKS`.
- Reportes y visualizaciones migrados a scores GARP.

Pendiente de eliminación física completa:

- Código de salidas TP/SL diagnósticas y tests asociados si se decide que ni siquiera deben existir como comparación secundaria.
- Algunos baselines de momentum no entrenados pueden conservarse solo como benchmark adversarial, no como lógica de selección.

## 10. Simplificación lograda

- Agentes productivos: de 4 antiguos + opcionales a 7 dominios GARP obligatorios.
- Label primario: de ruta TP/SL configurable a un único target GARP compuesto.
- Configuración: se eliminaron feature sets antiguos y se añadieron validadores fail-fast.
- Outputs: se añadieron auditorías de leakage, anti-momentum y explicación por ticker.

## 11. Plan final de limpieza

1. Ejecutar backtest completo GARP y guardar artefactos por fold.
2. Ejecutar ablations por agente con mismos folds.
3. Comparar contra tag/commit anterior sin mantener ruta antigua en código actual.
4. Si TP/SL no se requiere como diagnóstico secundario, eliminar `target_engineering.py`, rutas híbridas y tests TP/SL.
5. Mantener momentum solo como baseline externo y technical guardrail, nunca como agente comprador.

## 12. Auditoría final antes de congelar arquitectura

La fase final añade controles ligeros, sin backtests pesados:

- Survivorship bias: cada fold exporta `survivorship_bias_audit.json` con estado de membresía histórica, conteos de universo activo, snapshots faltantes y descartes por historial. Si la membresía histórica falta o se desactiva, el artefacto marca riesgo explícito.
- Concentración sectorial: cada fold exporta `sector_concentration_audit.csv` con pesos sectoriales, HHI, sector dominante y número de sectores. Esto permite decidir con evidencia local si hace falta endurecer límites sectoriales.
- Robustez del target: los pesos se mantienen heurísticos y no optimizados; no se añade grid search para evitar curve fitting. La validación recomendada es sensibilidad conceptual y ablation por componentes, no búsqueda de parámetros.

## 13. Capa posterior: Portfolio Intelligence

La arquitectura de selección queda congelada. La evolución posterior se implementa como una capa separada de gestión de tesis:

- `portfolio_review` analiza tickers o carteras existentes sin ejecutar walk-forward.
- La comparación principal es `snapshot_compra` vs `snapshot_actual`, no solo estado actual.
- Los estados de tesis son `Improving`, `Intact`, `Maturing`, `Weakening` y `Broken`.
- Las decisiones `Strong Buy`, `Buy`, `Hold`, `Review`, `Reduce` y `Sell` se basan en salud de tesis, valoración, riesgo y coste de oportunidad; no en TP/SL.
- El Thesis History Engine persiste `portfolio_thesis_history.csv`, `portfolio_thesis_events.csv` y `portfolio_review_report.md` para revisión mensual/trimestral de cambios relevantes sin ruido diario.
