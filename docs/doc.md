# Especificación maestra del repositorio: técnica, metodología y decisiones

> Este documento es la explicación completa del repositorio y la fuente académica extensa del proyecto. Describe qué existe, cómo se conecta cada subsistema, qué datos y métricas se usan, cómo se evalúan, por qué se eligió cada decisión relevante, qué se descartó y qué limitaciones siguen abiertas. El README es deliberadamente breve; `bitacora.md` conserva la cronología de decisiones y `informe_final.md` es la plantilla para una ejecución concreta. Esta especificación está pensada para alimentar la memoria y el LaTeX.

## 1. Propósito del proyecto

El proyecto estudia si una combinación configurable de variables fundamentales y de mercado puede **ordenar** acciones del S&P 500 por su exceso de retorno futuro. No se plantea como una promesa de inversión ni como una competición de CAGR histórico.

La pregunta operativa es: dado un snapshot temporal (t), ¿el score aprendido guarda correlación de rangos positiva con el exceso de retorno realizado después de (t)? La métrica central es por ello el **Rank-IC** (correlación de Spearman transversal):

\[
IC_t = \rho_S(\operatorname{rank}(score_{i,t}),\; r^{excess}_{i,t\rightarrow t+h})
\]

La rentabilidad de cartera, el Information Ratio y el drawdown se reportan como consecuencias económicas de una señal ya seleccionada por Rank-IC, no como sustitutos de evidencia predictiva.

Un resultado negativo es válido: puede mostrar que un bloque, un agente o una familia de modelo no generaliza fuera de muestra. El sistema conserva esos resultados en lugar de eliminar el código o los escenarios perdedores.

## 2. Principios no negociables

1. **Point-in-time.** Ninguna feature de (t) puede usar información publicada después de (t).
2. **Etiquetas cerradas.** Un retorno futuro solo puede utilizarse en entrenamiento o ponderación si `label_end_date <= retrain_date`.
3. **Separación temporal.** La selección de configuración termina en 2024. Los años 2025–2026 son una reserva de evaluación final.
4. **Selección por señal.** Bloques, agentes y modelos se seleccionan por Rank-IC OOS; la cartera no puede determinar qué modelo es ganador.
5. **Costes honestos.** Comisión y slippage son hipótesis de ejecución sometidas a estrés, no variables que el sistema pueda reducir para mejorar artificialmente el resultado.
6. **Reproducibilidad.** Cada run conserva configuración, huellas, artefactos y manifiestos; cada study conserva sus fases, decisión y escenarios omitidos.
7. **Explicabilidad.** Importancia, cobertura, estabilidad, atribuciones locales y ablaciones deben acompañar a cualquier resultado agregado.

## 3. Datos y construcción point-in-time

### 3.1 Fuentes activas

| Fuente | Uso | Protección temporal |
|---|---|---|
| Yahoo OHLCV histórico | Retornos, tendencia, volatilidad, volumen y liquidez. | Solo sesiones hasta el snapshot. |
| Finnhub histórico | Ratios, márgenes, crecimiento y series fundamentales. | Se alinean a la última publicación disponible. |
| Componentes históricos S&P 500 | Universo dinámico. | La pertenencia se consulta en cada fecha. |
| SPY | Benchmark y exceso de retorno. | Mismo corte temporal que los activos. |

Se eliminaron flags de SEC adicional, dividendos/recompras, estimaciones, short interest y acciones en circulación. No existía una serie histórica PIT validada para esas fuentes; conservar flags sin datos reales habría creado configuración ficticia y código muerto.

### 3.2 Panel y etiquetas

El panel `panel_point_in_time.parquet` contiene una observación por `(ticker, snapshot_date)`. Los fundamentales se mantienen constantes entre publicaciones, pero nunca se rellenan hacia atrás. Los precios se marcan frescos solo dentro del umbral metodológico fijo de siete días.

Las etiquetas se guardan por separado en `targets_forward_3m.parquet`, aunque el nombre histórico de la columna se conserva por compatibilidad. El horizonte real usado por una configuración está en `target_horizon_months` y en su manifiesto. Separar panel y target evita que una etapa de features consuma accidentalmente información futura.

### 3.3 Controles de calidad

- Universo histórico para reducir sesgo de supervivencia.
- `available_at_date` implícita en la alineación de fundamentales.
- Antigüedad de precio máxima fija: 7 días.
- Retorno mensual de posición superior a +200 % tratado como posible artefacto de datos y neutralizado con registro.
- Cohortes con score o retorno constantes no calculan Spearman: se marcan no evaluables, evitando correlaciones indefinidas y el `ConstantInputWarning`.

## 4. Catálogo de métricas y dirección económica

El catálogo central vive en `module/modeling/catalog.py`. Cada entrada tiene identificador estable, bloque, agentes candidatos, dirección económica y fuente. El modelo recibe percentiles transversales, no valores absolutos heterogéneos.

| Bloque | Métricas y transformación | Intuición |
|---|---|---|
| `quality_core` | ROE, ROIC, ROA, ROTC y márgenes. | Negocios rentables y eficientes. |
| `quality_efficiency` | Rotación de activos, inventario, cobros y liquidez. | Operación eficiente y capacidad de pago. |
| `financial_strength` | Deuda/equity, deuda/capital, deuda neta y SG&A/ventas. | Menor apalancamiento y coste estructural son preferibles. |
| `value_core` | P/E, P/B, P/S, P/FCF, tangible book, EV/EBITDA y EV/revenue. | Múltiplos menores son más atractivos; ratios no interpretables quedan como `NaN`. |
| `value_cashflow` | Earnings yield y FCF yield. | Expresan valoración positiva como rendimiento. |
| `growth_acceleration` | Crecimiento interanual y aceleración de EPS, ventas, EBITDA y FCF. | No solo crecer, sino mejorar el ritmo de crecimiento. |
| `fundamental_stability` | Tendencia/estabilidad de ROE y márgenes; conversión de caja. | Calidad repetible frente a un dato aislado. |
| `momentum_core` | Retorno relativo 1/3/6/12m, 12-1, aceleración y reversión. | Tendencia con separación de reversión reciente. |
| `momentum_trend` | Precio frente a medias y distancia a máximos. | Confirmación técnica de tendencia. |
| `price_risk` | Volatilidad, beta, drawdown y rango intradía. | El agente puede aprender efectos no lineales; el factor direccional premia menor riesgo. |
| `market_liquidity` | Volumen relativo, Amihud, gaps y volatilidad de volumen. | Penaliza costes implícitos y fragilidad de ejecución. |

Antes del ranking transversal, `metric_winsorization_percentile` puede recortar extremos por fecha. El baseline usa 1 %; 0 % y 2,5 % se prueban para medir si la señal depende de outliers.

## 5. Agentes especializados

| Agente | Función | Bloques naturales |
|---|---|---|
| `quality` | Calidad de negocio y fortaleza financiera. | Calidad, eficiencia, estabilidad y deuda. |
| `value` | Precio relativo al negocio y caja. | Valoración y yields. |
| `growth` | Crecimiento, aceleración y estabilidad. | Crecimiento, calidad y estabilidad. |
| `momentum` | Fuerza relativa, reversión y tendencia. | Momentum, tendencia y parte del riesgo. |
| `risk` | Riesgo de precio y liquidez. | Volatilidad, drawdown, volumen e iliquidez. |

La razón para separar agentes no es afirmar que los estilos sean independientes, sino convertir hipótesis económicas en componentes medibles. Una métrica puede alimentar más de un agente si la relación es defendible; la ablación posterior revela si esa redundancia aporta o no.

El sistema prueba: catálogo completo y catálogo completo menos cada agente (sin subconjuntos privilegiados). Al retirar un agente se preservan los demás, por lo que la diferencia frente al baseline estima contribución incremental.

## 6. Familias de modelos, objetivos y ensembles

### 6.1 Familias

- **LightGBM.** Modelo principal de árboles; maneja ausencias e interacciones no lineales. Sus hiperparámetros barribles son número de árboles, profundidad, learning rate y tamaño mínimo de hoja.
- **Elastic Net.** Referencia lineal con imputación mediana y estandarización. Aporta regularización y coeficientes interpretables.
- **CatBoost.** Alternativa de árboles. Usa regresión para `rank_regression`, clasificación para `quartile` y `YetiRank` para `ranking`, de modo que no se compare un objetivo distinto bajo la misma etiqueta.

### 6.2 Objetivos

| Objetivo | Etiqueta | Uso |
|---|---|---|
| `rank_regression` | Percentil transversal de retorno futuro. | Baseline alineado con Rank-IC. |
| `ranking` | Deciles de relevancia agrupados por snapshot. | Ranking directo. |
| `quartile` | Cuartil superior frente a inferior; centro excluido al entrenar. | Ablación de extremos. |

### 6.3 Ensemble intra-agente

`single` usa una familia. `equal_rank` promedia los percentiles de las familias. `rank_ic_weighted` pondera familias con su Rank-IC histórico OOS ya cerrado; si no existe evidencia positiva suficiente, usa pesos iguales. Esto evita calcular pesos con la cohorte que se está puntuando.

## 7. Meta-agente

El meta-agente transforma scores de agentes a rangos por snapshot y los combina.

- `equal`: peso idéntico.
- `rank_ic`: pesos proporcionales al Rank-IC positivo reciente de cada agente.
- `regime`: parte de `rank_ic` y aplica una inclinación moderada de momentum/calidad según retorno previo de SPY.
- `stacked_oos`: Ridge no negativa sobre scores OOS cerrados; requiere historia y recae en pesos iguales si no la hay.

En `rank_ic` y `regime` se aplica al final un límite por agente de **10 %–50 %** con renormalización: mínimo universal (ningún agente baja del 10 %, aunque su Rank-IC reciente sea negativo) y tope 50 %. Así el meta siempre hace algo de caso a todos los agentes y no colapsa sobre uno solo. `equal` (ya 1/N) y `stacked_oos` (modelo aprendido) no se recortan.

La decisión de introducir fallback explícito evita que un modo sofisticado invente pesos cuando no existe una muestra temporal suficiente para estimarlos.

## 8. Selección de métricas y explicabilidad

| Modo | Decisión metodológica |
|---|---|
| `model_native` | Todas las features; el modelo decide relaciones internas. |
| `diagnostic_only` | Todas las features y diagnóstico sin exclusión. Es baseline porque evita poda prematura. |
| `oos_stability_prune` | Elimina métricas sin cobertura, estabilidad de Rank-IC o evidencia positiva suficiente. |
| `regularized_linear_ensemble` | Fuerza una referencia Elastic Net junto a árboles. |
| `block_gated` | Conserva bloques cuya evidencia temporal agregada es positiva. |

La importancia por permutación se calcula temporalmente solo cuando su umbral es positivo: para cada cohorte de validación se ajusta un Ridge con snapshots anteriores, se permuta un factor dentro de la cohorte y se mide la degradación de Rank-IC. Es más costosa que la importancia nativa, por eso no se calcula innecesariamente en `diagnostic_only`.

Los artefactos de diagnóstico incluyen cobertura, Rank-IC univariante, importancia de modelo y atribución local LightGBM. SHAP exhaustivo no se declara como evidencia operativa; las atribuciones guardadas son las contribuciones aditivas disponibles del modelo de árboles.

## 9. Diseño del full study

### 9.1 Ejes barribles

`module/scenarios/variables.py` es la única fuente de opciones permitidas. Agrupa calendario, entrenamiento, objetivo, LightGBM, meta-agente, recencia, artefactos, bloques, agentes, familias, ensembles, selección, ventanas técnicas y cartera.

El contrato se presenta igual en `study` y `full_study`: mismas variables, valores y fases. Un
study manual selecciona subconjuntos; el full study activa todos. Los controles adicionales de
Experimental no se convierten por ello en ejes científicos. Costes son estrés y perfiles son
salidas de Fase 5.

En el study manual, los ejes de conjunto (`enabled_agents`, `enabled_feature_blocks`,
`enabled_model_families`) se configuran eligiendo los elementos base, y el runner aplica sobre esa
selección la misma ablación *full-minus-one* que usa el full study sobre su conjunto fijo (ver
`scenarios.variables.build_ablations`). La fase de robustez y estrés es opcional y granular: el
usuario marca uno a uno los componentes que quiere ejecutar —estrés de costes, estrés de reglas de
cartera (incluye `price_only_strictness_multiplier`), bootstrap por bloques, leave-one-year-out,
placebo por permutación y carteras aleatorias—; los no marcados no se ejecutan. El full study
oficial, en cambio, conserva el conjunto fijo de agentes/bloques/familias y ejecuta siempre todos
los componentes de robustez: es el estudio que sostiene la credibilidad del TFM.

No se hace producto cartesiano. Un full study ejecuta aproximadamente un centenar de escenarios de modelo aislados, combinaciones greedy, afinado, construcción de cartera, nueve stresses de coste y ocho perfiles. El número exacto depende de reutilización de caché y escenarios no viables.

### 9.2 Condicionalidad

Los umbrales de cobertura, historia, fracción positiva, permutación y máximo de features no se prueban solos, porque con `diagnostic_only` no cambiarían el modelo. Se prueban ligados a `oos_stability_prune` y `block_gated`. Esta decisión reduce falsos descubrimientos aparentes y elimina ejecuciones inertes.

### 9.3 Reserva temporal

La función que resume un run para las fases 1–3 reemplaza el Rank-IC agregado por el calculado hasta 2024. Por tanto, ni `_stable_best`, ni la combinación greedy ni el afinado consultan 2025–2026. Esos años se guardan en `reserved_validation` de `decision.json`.

### 9.4 Regla de aceptación inicial

Un eje entra en la combinación greedy únicamente si su mejor candidato supera el Rank-IC del baseline y no reduce su fracción de cohortes positivas. Esta no sustituye bootstrap ni leave-one-year-out, que se calculan sobre el finalista, pero evita combinar automáticamente cambios claramente peores.

### 9.5 Cartera, perfiles y costes

Tras fijar modelo y meta-score, la fase de cartera ajusta el tamaño fijo, el percentil de mantenimiento y la ventaja de rotación por Information Ratio. Los pesos siguen una escala lineal min-max **dentro de la cartera** sobre el `meta_score` crudo: el mejor de los seleccionados pesa el doble que el peor, lineal en medio. Se usa el `meta_score` y no el percentil global (`meta_rank`), porque los seleccionados son siempre el top-N y su percentil se apiña en 0,97–1,00, lo que aplanaría los pesos a ~1/N; el `meta_score` sí ordena a los seleccionados entre sí. No altera las predicciones, por lo que no tiene sentido seleccionarla por Rank-IC.

Comisión y slippage ya no se incluyen como ejes optimizables. Se generan las nueve combinaciones de 0/5/10 bps de comisión y 5/10/20 bps de slippage como `cost_stress`. La conclusión debe ser robusta a costes plausibles, no depender del caso más barato.

Los ocho perfiles reordenan la misma señal final; se comparan después de seleccionar modelo y cartera. Un perfil rentable no demuestra por sí mismo mejor aprendizaje.

## 10. Parámetros fijos y razones

| Parámetro | Valor | Razón para no optimizarlo |
|---|---:|---|
| Inicio OOS | 2015-Q1 | Mantiene una historia comparable entre estudios. |
| Benchmark | SPY | Referencia estable de exceso de retorno. |
| Día de snapshot | 15 | Evita buscar el día más favorable. |
| Antigüedad máxima de precio | 7 días | Regla de calidad de datos, no hipótesis alpha. |
| Mínimo de filas de entrenamiento | 30 | Salvaguarda de viabilidad, no fuente de señal. |
| Vida media de recencia | 3 años | Convención fija para no multiplicar grados de libertad. |
| Mínimo sectorial | 5 empresas | Evita percentiles degenerados en neutralización. |
| Semilla base | 42 | Las semillas se evalúan como robustez, no como optimización. |

## 11. Robustez, resultados y limitaciones

El finalista produce bootstrap por bloques, leave-one-year-out, permutación de etiquetas y comparación con carteras aleatorias. El placebo debe colapsar hacia cero; si no lo hace, hay riesgo de fuga o artefacto.

Limitaciones explícitas:

- universo restringido al S&P 500 y cobertura histórica imperfecta;
- múltiples pruebas: una búsqueda amplia aumenta probabilidad de falsos positivos;
- dependencia serial entre snapshots;
- muestra limitada de cohortes reservadas 2025–2026;
- costes simulados, no ejecución real;
- CatBoost y Elastic Net comparten algunas decisiones de preprocesamiento con el pipeline, por lo que no sustituyen una validación independiente de implementación.

Un resultado no se considera validado solo por media positiva. Debe combinar mejora frente al baseline, estabilidad, bootstrap, LOYO, reserva temporal y estrés de costes.

## 12. Artefactos, interfaz y operación

Cada run se guarda en `results/runs/<run_id>/`; cada study, en `results/studies/<study_id>/`. Los ficheros centrales son manifiesto, configuración, diagnósticos de Rank-IC, pesos meta, atribuciones, `backtest_summary.json`, `comparison_data.parquet` y `decision.json`.

La pantalla Full study permite indicar **nombre** e **hipótesis**. Ambos se guardan explícitamente en `study_manifest.json`; la hipótesis también se copia a `decision.json` y a la descripción visible de los escenarios.

```powershell
# Interfaz
python main.py

# Estudio completo
$env:RUN_MODE = "full_study"
$env:RUN_SCOPE = "full"
python -u main.py

# Verificación
python -m pytest tests/ -q
python -m ruff check .
```

## 13. Estado de evidencia

Los studies anteriores se conservan para auditoría histórica, pero no deben usarse como resultado final si fueron creados antes de separar estrictamente la reserva y los costes.

La primera evidencia cuantitativa completa bajo el protocolo vigente es el study oficial `optimization-official` (`20260724--optimization-official--4dfea2986540`, 104 escenarios, estado `succeeded`). Su lectura detallada está en `docs/informe_resultados_study.md`. Resumen del veredicto:

- **Rank-IC OOS del finalista (selección ≤2024): 0.0944**, con 72.6 % de cohortes con IC>0.
- La señal es **estadísticamente real** dentro del periodo de selección: placebo con `p = 0.0` (colapsa al barajar etiquetas), bootstrap por bloques con IC 95 % `[0.050, 0.131]` (no cruza cero) y leave-one-year-out estable (Rank-IC ∈ `[0.073, 0.104]`).
- Pero **no se confirma en la reserva ciega 2025–2026**: Rank-IC medio `−0.0095`, con alfa anual negativo en 2024 (−7.8 %), 2025 (−11.9 %) y 2026 parcial (−10.2 %).
- La rentabilidad **no supera de forma convincente a carteras aleatorias** del mismo tamaño (percentil 18).

Interpretación: existe señal de ordenación transversal real hasta 2024 que **se degrada a partir de 2024**; su traducción a alfa económico es marginal (IR ≈ 0.02). El resultado principal es, legítimamente, **parcialmente negativo**: el mérito está en el pipeline reproducible y honesto que detecta su propia limitación, no en una ventaja económica demostrada. La configuración ganadora (LightGBM único poco profundo, meta `stacked_oos`, poda estricta de features, horizonte de etiqueta 12 m, `execution_lag_days = 60`, `target_size = 10`) es robusta a semillas e hiperparámetros.

## 14. Anexo de configuración: contrato exacto del full study

Esta sección evita que la memoria confunda una opción disponible en una ejecución manual con una opción que se busca automáticamente. El contrato ejecutable está en `module/scenarios/variables.py`: la tabla siguiente lo reproduce de forma legible a fecha de esta versión. La columna *baseline* es el valor de `Settings()`; no implica que sea el ganador del estudio.

### 14.1 Ejes de datos, calendario y etiqueta

| Variable | Qué modifica | Baseline | Valores que explora el full study |
|---|---|---:|---|
| `execution_lag_days` | **Define el día de observación de cada snapshot**: la rejilla cae en `fin_de_periodo + execution_lag_days` (retardo de publicación tras el cierre del periodo). Barrerlo produce rejillas distintas. | 45 | 15, 30, 45, 60 |
| `train_lookback_years` | Años de historia inmediatamente anterior que ve cada reentreno. | 8 | 4, 8, 12 |
| `snapshot_step_months` | Frecuencia con que se crean snapshots y se reentrena (mes vs. trimestre). | 1 | 1, 3 |
| `fundamental_step_months` | Cadencia con que se actualiza la capa fundamental del panel. | 3 | 3, 6, 12 |
| `target_horizon_months` | Meses de retorno futuro que forman la etiqueta. | 6 | 3, 6, 12 |
| `objective` | Forma de aprender la etiqueta: regresión de percentil, ranking por grupo o extremos. | `rank_regression` | `rank_regression`, `ranking`, `quartile` |

**Calendario de snapshots.** La rejilla ya no usa un día del mes fijo (`snapshot_day` fue eliminado). Cada snapshot cae en el **fin de su periodo más `execution_lag_days`**: cerrado un mes o trimestre, los fundamentales tardan unos días en estar disponibles y la rejilla observa justo entonces. Así el retardo de publicación gobierna cuándo se miran los datos, y `execution_lag_days` pasa a ser un eje con efecto real (antes solo desplazaba el ancla y no cambiaba ningún resultado). El point-in-time sigue garantizado: en cada snapshot solo se leen fundamentales con `filed_date` anterior.

No se barre `execution_year` ni `execution_quarter`: desplazarlos sería buscar una fecha de inicio favorable. El ancla permanece en 2015-Q1 para que todo escenario comparta el mismo periodo OOS (2015→hoy, con 2025-26 reservados).

### 14.2 Ejes de LightGBM, combinación y recencia

| Variable | Qué modifica | Baseline | Valores que explora el full study |
|---|---|---:|---|
| `lgbm_n_estimators` | Número máximo de árboles; afinado greedy en Fase 3. | 200 | 100, 200, 400 |
| `lgbm_max_depth` | Complejidad máxima; afinado greedy en Fase 3. | 4 | 3, 4, 5, 6, 8 |
| `lgbm_learning_rate` | Tamaño de actualización; afinado greedy en Fase 3. | 0,05 | 0,02; 0,03; 0,05; 0,10 |
| `lgbm_min_child_samples` | Regularización por hoja; afinado greedy en Fase 3. | 50 | 20, 50, 100 |
| `meta_type` | Regla que mezcla los scores de los agentes. | `rank_ic` | `equal`, `rank_ic`, `regime`, `stacked_oos` |
| `meta_ic_lookback_quarters` | Trimestres OOS cerrados usados para estimar pesos de Rank-IC. | 12 | 8, 12, 16 |
| `min_rank_ic_cross_section` | Mínimo de empresas válidas para computar un Rank-IC de cohorte. | 10 | 8, 10, 12 |
| `recency_weighting` | Peso temporal de observaciones de entrenamiento: sin sesgo, lineal o exponencial. | `off` | `off`, `linear`, `exponential` |

`stacked_oos` no puede aprender de la cohorte que evalúa: ajusta el Ridge no negativo con scores producidos fuera de muestra y cuya etiqueta ya cerró. Cuando esa historia no es suficiente, deja registrada la caída a combinación equiponderada. No se presenta ese fallback como evidencia de stacking.

### 14.3 Catálogo, agentes, modelos y selección

| Variable | Qué modifica | Baseline | Valores que explora el full study |
|---|---|---|---|
| `enabled_feature_blocks` | Conjunto de bloques que llega a los agentes. | Los 11 bloques | Catálogo completo; y catálogo completo menos uno de los 11 bloques en cada ablación (sin subconjuntos privilegiados). |
| `enabled_agents` | Agentes que entregan score al meta-agente. | Los 5 agentes | Los 5 agentes; y los 5 menos un agente en cada ablación (sin subconjuntos privilegiados). |
| `enabled_model_families` | Familias disponibles dentro de cada agente. | LightGBM + Elastic Net + CatBoost | Solo LightGBM; LightGBM + Elastic Net; las tres familias. |
| `intra_agent_ensemble_mode` | Cómo se combinan familias dentro de un agente. | `rank_ic_weighted` | `single`, `equal_rank`, `rank_ic_weighted` |
| `feature_weighting_mode` | Política de uso o poda de variables. | `diagnostic_only` | `model_native`, `diagnostic_only`, `oos_stability_prune`, `regularized_linear_ensemble`, `block_gated` |
| `feature_selection_min_coverage` | Cobertura mínima de una métrica para que una política de poda pueda conservarla. | 0,55 | 0,40; 0,55; 0,70 |
| `feature_selection_lookback_quarters` | Historia cerrada usada por esa política. | 12 | 8, 12, 16 |
| `feature_selection_min_permutation_importance` | Degradación mínima de Rank-IC al permutar una métrica para conservarla. | 0,0 | 0,0; 0,001; 0,005 |
| `feature_selection_min_positive_fraction` | Fracción mínima de cohortes con señal positiva exigida a una métrica/bloque. | 0,50 | 0,40; 0,50; 0,60 |
| `feature_selection_max_features_per_agent` | Límite de features conservadas por agente; 0 significa sin límite. | 0 | 0, 8, 12, 20 |
| `metric_winsorization_percentile` | Recorte simétrico de extremos antes de rankear por fecha. | 0,01 | 0,0; 0,01; 0,025 |

Los parámetros cuyo nombre empieza por `feature_selection_` son **condicionales**: no se ejecutan como cambios aislados con `diagnostic_only`, pues entonces no tendrían efecto. El orquestador los liga a `oos_stability_prune` y a `block_gated`. De este modo cada run representa una hipótesis efectiva y no una variación nominal sin efecto en las predicciones.

Los once bloques completos son: `quality_core`, `quality_efficiency`, `financial_strength`, `value_core`, `value_cashflow`, `growth_acceleration`, `fundamental_stability`, `momentum_core`, `momentum_trend`, `price_risk` y `market_liquidity`. Los cinco agentes son `quality`, `value`, `growth`, `momentum` y `risk`, y cada bloque pertenece a un solo agente (partición disjunta). La ablación de un bloque o agente significa retirarlo manteniendo todo el resto, no probarlo en solitario; esa diferencia es la que permite interpretar contribución incremental.

### 14.4 Artefactos y ventanas técnicas

| Variable | Qué modifica | Baseline | Valores que explora el full study |
|---|---|---|---|
| `neutralize_by_sector` | Convierte factores a percentiles relativos a su sector cuando hay muestra suficiente. | `false` | `false`, `true` |
| `fundamental_momentum` | Incluye cambios temporales de fundamentales. | `false` | `false`, `true` |
| `market_regime_feature` | Añade contexto del régimen de mercado. | `false` | `false`, `true` |
| `price_momentum_multi` | Añade retornos de varios horizontes. | `false` | `false`, `true` |
| `moving_averages` | Añade relaciones con medias móviles. | `false` | `false`, `true` |
| `regime_extended` | Amplía la descripción del régimen de mercado. | `false` | `false`, `true` |
| `quality_growth_derived` | Añade derivados de calidad y crecimiento. | `false` | `false`, `true` |
| `risk_feature_windows` | Ventanas de sesiones usadas por volatilidad, beta y drawdown. | `(63, 126, 252)` | `(63, 126, 252)`, `(21, 63, 126)`, `(63, 252)` |
| `technical_feature_windows` | Ventanas de sesiones para tendencias y técnicas. | `(21, 63, 252)` | `(21, 63, 252)`, `(10, 21, 63)`, `(21, 126, 252)` |

La neutralización no convierte al sector en una señal: elimina parcialmente diferencias estructurales de nivel dentro de la sección transversal. Se exige un mínimo fijo de cinco empresas por grupo; con grupos menores, neutralizar crearía rankings artificiales.

### 14.5 Ejes de construcción de cartera

| Variable | Qué modifica | Baseline | Valores que explora el full study |
|---|---|---:|---|
| `target_size` | Número fijo de posiciones top-N. | 8 | 5, 8, 10, 12, 15 |
| `min_hold_percentile` | Percentil que una posición debe superar para conservarse. | 80 | 60, 70, 80, 85 |
| `rotation_edge_percentiles` | Ventaja de percentil exigida a una candidata para sustituir la peor tenencia. | 10 | 5, 10, 15 |
| `price_only_strictness_multiplier` | Endurece expulsión, rotación y rebalanceo en las revisiones que solo traen precio nuevo (sin fundamentales nuevos). Regla mecánica: se estresa, no se optimiza. | 1,0 | 1,0, 1,5, 2,0, 3,0 |
| `profile` | Salidas paralelas de estilo; no se optimiza un ganador. | `balanced` (meta puro) | `balanced`, `growth`, `value`, `quality`, `momentum`, `contrarian`, `defensive`, `garp` |

**Tamaño fijo y pesos.** El full study barre `target_size` como eje simple. En cada snapshot se seleccionan las N acciones mejor situadas en `meta_rank`; los pesos siguen una escala lineal min-max dentro de la cartera sobre el `meta_score` crudo (el mejor de los seleccionados pesa el doble que el peor) y se normalizan al 100 %. Se usa el `meta_score` porque el percentil global se apiña en el top y aplanaría los pesos. El rebalanceo es real: una posición solo se reajusta si su peso objetivo difiere del actual en al menos `rebalance_drift_tolerance` (25 % relativo); por debajo se congela y el presupuesto se reparte entre las que sí se mueven conservando las relaciones del target global.

Estos ejes se ejecutan una vez elegida la especificación predictiva. Se comparan por Information Ratio calculado solo hasta 2024, con desempates de riesgo y estabilidad, y no se retropropagan al modelo. Los perfiles se ejecutan después y no participan en la elección.

### 14.6 Constantes y pruebas de robustez que no optimiza el estudio

| Elemento | Valor vigente | Por qué queda fuera de la búsqueda |
|---|---|---|
| Rango de datos | 1990-01-01 a 2026-07-15 | Define el dataset disponible, no una hipótesis de modelo. |
| Benchmark | `SPY` | Cambiarlo alteraría la definición de etiqueta y la comparabilidad. |
| Inicio y trimestre OOS | 2015-Q1 | Ancla común de todos los escenarios. |
| `max_price_age_days` | 7 | Control de frescura de datos. |
| `min_training_rows` | 30 | Salvaguarda para no entrenar modelos inviables. |
| `recency_halflife_years` | 3 | Convención estable para reducir grados de libertad. |
| `neutralize_min_group` | 5 | Previene neutralizaciones degeneradas. |
| `random_seed` | 42 | Fija reproducibilidad; no se busca la semilla más favorable. |
| `rebalance_drift_tolerance` | 0,25 | Fracción mínima de cambio relativo a la posición para rebalancear (25 %); regla mecánica de rebalanceo, no señal predictiva. Por debajo, la posición se congela y no opera. |
| `price_only_strictness_multiplier` | 1,0 | Con `snapshot_step_months=1` hay revisiones que solo traen precio nuevo (`review_type=price_monthly`, sin fundamentales nuevos). Este factor divide el umbral de expulsión y multiplica los de rotación y rebalanceo en esas revisiones, para no rotar la cartera por ruido de precio. Regla mecánica: se estresa, nunca se elige el valor más rentable. Con 1,0 no hay efecto. |
| `max_monthly_position_return` | 2,0 (+200 %) | Filtro de integridad frente a anomalías extremas de datos. |
| `commission_bps` y `slippage_bps` | Caso base 5 y 10 bps | No son ejes de optimización. Se someten a los nueve pares 0/5/10 bps de comisión × 5/10/20 bps de slippage. |

`run_mode` y `run_scope` tampoco son variables científicas: controlan si se descargan datos, se ejecuta una estrategia o se lanza un estudio, y si se opera en muestra de desarrollo o en alcance completo. Se registran por reproducibilidad, pero no forman parte de ninguna comparación de hipótesis.

## 15. Lectura de artefactos y trazabilidad de una afirmación

Una afirmación de la memoria debe poder recorrerse hacia atrás sin interpretación manual. La cadena de evidencia es:

```text
nombre e hipótesis del usuario
        ↓
study_manifest.json (configuración y alcance declarados)
        ↓
run_ids.json (runs realmente ejecutados y reutilizados)
        ↓
artefactos de cada run (predicciones, Rank-IC, pesos, cobertura y cartera)
        ↓
decision.json (selección hasta 2024 y evaluación reservada 2025–2026)
        ↓
tablas y figuras de informe_final.md / LaTeX
```

El campo `hypothesis` no es decorativo: se persiste explícitamente en el manifiesto y en la decisión, además de integrarse en la descripción visible. Debe expresar una proposición falsable —por ejemplo, que añadir riesgo y liquidez mejora Rank-IC OOS frente al catálogo histórico—, no una expectativa de rentabilidad.

Para convertir resultados a LaTeX se debe copiar la cifra junto con: identificador de study y run, rango de fechas, configuración exacta, número de cohortes, métrica de selección hasta 2024, resultado reservado y resultado de costes. Si alguno de esos elementos falta, la cifra puede servir para exploración, pero no para la conclusión del TFM.

## 16. Mapa completo del repositorio y responsabilidad de cada parte

El repositorio no es un único script. Las fronteras entre carpetas son deliberadas: separan adquisición de datos, transformación temporal, aprendizaje, evaluación, orquestación y presentación. Esta tabla es el índice técnico que debe usarse al ampliar o auditar el proyecto.

| Ubicación | Responsabilidad | Entradas principales | Salidas o efecto |
|---|---|---|---|
| `main.py` | Punto de entrada. Sin `RUN_MODE` inicia la consola; con él ejecuta una etapa o el ciclo solicitado. | Variables de entorno y `Settings`. | Servidor local o etapa CLI. |
| `environment.py` | Contrato central de configuración, rutas, defaults, validación y separación `dev`/`full`. | `.env`, entorno y constantes. | Instancia inmutable `Settings`. |
| `module/data/ingest/` | Clientes de Yahoo, Finnhub y SEC/EDGAR, caché JSON y consolidación cruda. | Red, caché y universo histórico. | Parquets raw, cobertura y errores de descarga. |
| `module/data/universe.py` | Membresía histórica del S&P 500, normalización de ticker y exclusión de tickers reciclados. | CSV histórico de componentes. | Universo válido en cada fecha. |
| `module/data/dataset.py` | Construcción del panel point-in-time, benchmark y precios de activos. | Raw de precios, Finnhub y fechas de filing. | `panel_point_in_time`, precios PIT y benchmark PIT. |
| `module/data/baselines.py` | Scores deterministas de referencia por estilo. | Features rankeadas. | `baseline_scores.parquet`. |
| `module/modeling/features.py` | Ingeniería de factores, ranking transversal, neutralización y targets futuros. | Panel, OHLCV PIT y benchmark. | Features, targets, cobertura. |
| `module/modeling/artifacts.py` | Cálculo de artefactos opcionales de precio, régimen, calidad, riesgo y liquidez. | Panel y series de precios. | Columnas adicionales de factor. |
| `module/modeling/catalog.py` | Catálogo declarativo: bloque, agentes candidatos y dirección de cada factor. | Definición estática versionada. | Lista efectiva de features por agente. |
| `module/modeling/agents.py` | Entrenamiento walk-forward por agente, familias de modelo, diagnóstico y atribución. | Features y targets cerrados. | Scores, diagnósticos e importancia. |
| `module/modeling/meta.py` | Combinación temporal de agentes y diagnóstico del score final. | Scores por agente y retornos cerrados. | `meta_rank`, pesos y Rank-IC. |
| `module/evaluation/portfolio.py` | Reglas de entrada, salida, rotación y pesos de cartera. | Rankings PIT y estado anterior. | Órdenes y estado objetivo. |
| `module/evaluation/backtest.py` | Simulación de precios, costes, equity y métricas económicas. | Scores, órdenes, precios PIT y SPY. | Equity, posiciones, órdenes, anual y resumen. |
| `module/evaluation/profiles.py` | Reordenación determinista de acciones buenas por perfil de inversor. | Ranks de agentes y meta-rank. | `meta_rank` alternativo sin reentrenar. |
| `module/evaluation/stats.py` y `robustness.py` | Intervalos, ICIR, bootstrap, LOYO, placebo y cartera aleatoria. | Diagnósticos y retornos. | Evidencia de estabilidad y azar. |
| `module/runs/execution.py` | Ejecución controlada de runs y full study oficial. | `Settings`, store, caché y opciones. | Runs publicados, decisión y exportaciones. |
| `module/runs/experiments.py` | Escenarios manuales, fingerprints y ciclo histórico de experimentos. | Rejillas de escenarios. | Escenarios, elección y conclusiones de experimentos. |
| `module/runs/recycle.py` | Caché de etapas intermedias por huella, incluida la sub-etapa `agents_fit` por familia de modelo. | Inputs y settings de una etapa (más la familia en `agents_fit`). | Restauración/publicación en `data/recycle`. |
| `module/runs/results_store.py` | Registro inmutable de runs y studies. | Artefactos ya calculados. | Manifiestos, hashes y `registry.jsonl`. |
| `module/ui/dashboard.py` | Servidor HTTP local, API JSON, jobs y lectura segura de resultados. | App estática y ResultsStore. | Consola en `127.0.0.1:8765`. |
| `module/ui/reports.py` | Informes HTML de un run y comparativos de escenarios. | Artefactos publicados. | HTML auto-contenido con gráficos. |
| `app/` | Interfaz estática: navegación, formularios, tablas, gráficos y vistas. | API local JSON. | Research Console en navegador. |
| `module/scenarios/` | Catálogo de valores permitidos y rejillas de experimentos dirigidos. | Código Python declarativo. | Opciones visibles y definición de escenarios. |
| `tests/` | Pruebas de contratos de datos, PIT, modelos, cartera, runs, informes y UI de datos. | Fixtures sintéticos. | Protección contra regresiones. |
| `docs/` | Especificación técnica, bitácora e informe de resultados. | Código y resultados auditables. | Fuente narrativa del TFM. |
| `latex/` | Documento final de la memoria. | Evidencia validada y documentación. | PDF/entregable académico. |

Los ficheros `servir_html.py` y `verify_rob_tmp.py` son utilidades locales de apoyo; no intervienen en el pipeline científico ni en el full study. `CLAUDE.md` es una guía compacta para colaboradores y `README.md` es la puerta de entrada. Ninguno sustituye este documento.

## 17. Arranque, configuración y modos de ejecución

### 17.1 Punto de entrada y variables de entorno

`main.py` aplica la siguiente regla:

1. Si la variable `RUN_MODE` no está presente, arranca la Research Console local.
2. Si está presente, construye `Settings`, crea rutas necesarias, activa logging y ejecuta la etapa pedida.
3. `RUN_MODE=full` encadena descarga, dataset, features, agentes, backtest e informe. `RUN_MODE=full_study` delega en la optimización oficial; esta reutiliza datos ya descargados y no provoca descargas por escenario.

Los modos admitidos son `download`, `dataset`, `features`, `agents`, `backtest`, `report`, `experiments`, `full` y `full_study`. `RUN_SCOPE=dev` emplea una muestra pequeña de tickers y directorios aislados; `RUN_SCOPE=full` usa el universo histórico completo. Un resultado `dev` nunca debe aparecer en la evidencia final.

El fichero `.env` se carga sin depender de paquetes adicionales. `FINNHUB_API_KEY` solo es necesaria si falta la caché local de Finnhub. `EDGAR_USER_AGENT` identifica las consultas de filing SEC. Ninguna de estas credenciales se serializa en los manifiestos.

### 17.2 Objeto `Settings` y validaciones

`Settings` es un `dataclass(frozen=True)`: las etapas reciben una configuración explícita y no deben mutarla. Valida modo, alcance, objetivo, meta-model, listas no vacías de agentes, límites de percentiles, ventanas positivas y que el tamaño fijo de cartera sea positivo.

El estudio, la consola y el orquestador comparten `module/scenarios/variables.py` como fuente de valores admitidos. La API transforma listas JSON recibidas para bloques, agentes, familias y ventanas en tuplas antes de validar, para que una selección compuesta de la consola tenga exactamente la misma semántica que una configuración Python.

## 18. Adquisición de datos, universo y controles previos

### 18.1 Universo histórico

El universo no es la lista actual de compañías. `module/data/universe.py` lee `data/S&P 500 Historical Components & Changes.csv`, crea snapshots de miembros y responde `members_at(fecha)`. Así, una acción solo puede estar disponible si pertenecía al índice en ese snapshot. También normaliza símbolos y detecta un ticker reutilizado cuando la primera fecha de precio es incompatible con su período de pertenencia; esa observación se excluye y queda registrada.

### 18.2 Clientes y fuentes

| Cliente | Datos obtenidos | Papel actual |
|---|---|---|
| `YahooClient` | OHLCV histórico ajustado. | Fuente de precios, retornos, volatilidad, volumen, órdenes y benchmark SPY. |
| `FinnhubClient` | Perfil, `basic_financials` histórico y noticias de compañía. | Los fundamentales alimentan el panel; perfil aporta sector; las noticias se conservan como raw exploratorio, pero no se usan como feature ni en la selección actual. |
| `EdgarClient` | Mapa ticker–CIK y filings 10-Q/10-K con período y fecha de filing. | Proporciona la fecha de disponibilidad temporal de los fundamentales. No se usan estados SEC adicionales como variables. |

Cada respuesta de red se guarda con `downloaded_at` en `data/raw/json/<fuente>/<ticker>/`. Si existe esa caché y `FORCE_RAW_DOWNLOAD` es falso, se reutiliza. Tras consolidación, el raw completo incluye `profiles.parquet`, `finnhub_metrics.parquet`, `prices.parquet`, `report_dates.parquet`, `download_coverage.json`, `universe_coverage.json` y `download_failures.csv`; `news.parquet` solo aparece si hubo noticias.

La presencia de `news.parquet` no significa que el modelo use noticias. La documentación lo deja explícito para que no se atribuya al sistema una capacidad NLP o de sentimiento que no existe. De forma coherente, no hay flags de estimaciones, short interest, dividendos/recompras, acciones históricas ni estados financieros SEC como fuentes modelables: fueron retirados al no disponer de series PIT verificadas.

### 18.3 Elegibilidad y cobertura

Para que una empresa sea elegible se exige precio observable y un período de fundamentales que pueda emparejarse con un filing publicado. `universe_coverage.json` mide anualmente miembros del S&P 500, elegibles, porcentaje y causas de exclusión: ticker reciclado, precio ausente o falta de fundamental/filing. La cobertura `dev` se declara expresamente no representativa.

La descarga puede registrar fallos por ticker/dataset y continuar con el resto. Sin embargo, falla de forma explícita si no hay filas de perfiles, métricas o precios, o si falta el benchmark. Este comportamiento evita construir en silencio un panel parcial vacío.

## 19. Contratos de datos point-in-time

### 19.1 Panel fundamental y de precios

`build_point_in_time_dataset` genera una fila por `(ticker, snapshot_date)` para los miembros históricos. Para cada fila busca el último precio disponible hasta el snapshot y registra `price_as_of_date` y `price_age_days`; no rellena con cotizaciones posteriores. Los fundamentales se toman del último filing con `filed_date <= snapshot_date`, usando el período económico más reciente de entre los filings conocidos. Se conservan `fundamental_period`, `fundamental_filed_date` y `fundamental_age_days` para auditarlo.

Las métricas de flujo que podrían confundir un trimestre con un cierre anual solo aceptan fuente trimestral. Los ratios TTM y saldos de balance admiten trimestral o anual porque su significado es compatible. El crecimiento interanual empareja la observación con una fecha a doce meses, con tolerancia de 45 días; si no existe pareja válida o el denominador es cero, devuelve ausencia en lugar de inventar crecimiento.

Los tres contratos producidos son:

| Artefacto procesado | Grano | Contenido |
|---|---|---|
| `panel_point_in_time.parquet` | ticker × snapshot | Precio, retornos 1/3/6/12m, ratios fundamentales, valores auxiliares y fechas PIT. |
| `benchmark_point_in_time.parquet` | snapshot | Precio y retornos del SPY con la misma regla de corte. |
| `asset_price_point_in_time.parquet` | ticker × snapshot | Precio PIT para simular cartera y visualizar trayectorias. |

### 19.2 Cadencia y tipo de revisión

`snapshot_step_months` define las fechas de observación. `fundamental_step_months` define cada cuántos snapshots existe una revisión fundamental/reentrenamiento. El ancla se calcula desde `execution_year`, `execution_quarter` y `execution_lag_days`; la rejilla se alinea contando posiciones de la propia rejilla, no meses absolutos. Esto evita el fallo de no reentrenar nunca cuando una rejilla trimestral no pasa por el mes teórico de anclaje.

Las observaciones entre filings conservan el último dato publicado; no se retroproyecta el filing hacia fechas anteriores. En cambio, precios y factores técnicos sí pueden cambiar en cada snapshot porque hay información de mercado diaria disponible hasta ese instante.

## 20. Ingeniería de factores: fórmulas, tratamiento y catálogo exhaustivo

`features.py` combina el panel con benchmark y precios PIT. Primero construye variables numéricas, luego aplica winsorización transversal opcional por fecha y finalmente las convierte a percentiles transversales. Un factor con dirección `-1` se invierte al rankear para que un valor económico preferible se traduzca en score mayor. Los múltiplos negativos o nulos que no admiten interpretación económica se convierten en `NaN`, no en empresas artificialmente baratas.

### 20.1 Factores fundamentales, exactamente como están en el catálogo

| Bloque | Factores | Construcción / dirección |
|---|---|---|
| Calidad nuclear | ROE, ROIC, ROA, ROTC, margen neto, operativo, bruto, pre-impuestos y FCF. | Valor PIT correspondiente; mayor es mejor. |
| Eficiencia y liquidez | Rotación de activos, inventario y cuentas a cobrar; current, quick y cash ratio. | Valor PIT; mayor es mejor. |
| Fortaleza financiera | Deuda/equity, deuda/activos, deuda/capital, deuda larga/equity, deuda larga/activos, deuda larga/capital, deuda neta/equity, deuda neta/capital y SG&A/ventas. | Valor PIT; menor es mejor, por lo que se invierte su rank. |
| Valoración | P/E, P/B, P/tangible book, P/S, P/FCF, EV/EBITDA y EV/revenue. | Solo ratios interpretables; menor es mejor. |
| Valor por caja | Earnings yield `1 / P/E` y FCF yield `1 / P/FCF`. | Solo denominador positivo y distinto de cero; mayor es mejor. |
| Crecimiento y aceleración | Crecimiento YoY de EPS, ventas/acción, EBITDA y FCF/acción; aceleración de cada uno. | `g_t = x_t/x_{t-12m}-1`; aceleración `g_t-g_{t-1}`; mayor es mejor. |
| Estabilidad fundamental | Tendencia ROE, tendencia ROIC, estabilidad de margen, estabilidad ROE y conversión de caja. | Tendencias contra historia previa, estabilidad como menor dispersión y conversión con EPS positivo; mayor es mejor tras la transformación. |

### 20.2 Factores de mercado y liquidez, exactamente como están en el catálogo

| Bloque | Factores | Construcción / dirección |
|---|---|---|
| Momentum | Retorno relativo a SPY de 3, 6 y 12 meses; momentum 12-1; aceleración; reversión 1m. | Retorno de activo menos benchmark; 12-1 excluye el último mes; aceleración es corto menos largo; reversión es el negativo de 1m. Mayor es mejor. |
| Tendencia | Precio frente a SMA6 y SMA12; distancia al máximo de 12m. | Todos usan solo sesiones previas al snapshot; mayor es mejor. |
| Riesgo de precio | Volatilidad realizada 63/126d, volatilidad bajista 63d, beta 252d, drawdown actual y máximo 252d, rango intradía medio 21/63d. | Se conservan como información no lineal y se ofrece una versión direccional de menor riesgo = mejor. |
| Liquidez de mercado | Volumen relativo, volatilidad de volumen, iliquidez Amihud y gap medio 21d. | `volumen_21d/volumen_252d-1`, desviación de `log(volumen)`, media de `|r|/(precio×volumen)` y media de `|open/close_{t-1}-1|`; menor fragilidad/ilíquidez es mejor. |

Las ventanas listadas en el catálogo son las defaults de las fórmulas. `risk_feature_windows` y `technical_feature_windows` permiten probar conjuntos de ventanas alternativos; el manifiesto conserva la tupla efectiva. El artefacto de riesgo incluye alpha histórico frente a SPY cuando es calculable como señal auxiliar, aunque el catálogo direccional usa los factores enumerados arriba.

### 20.3 Artefactos opcionales y baselines

Los interruptores `fundamental_momentum`, `market_regime_feature`, `price_momentum_multi`, `moving_averages`, `regime_extended` y `quality_growth_derived` añaden columnas derivadas, no fuentes externas. Incluyen, respectivamente, tendencia de fundamentales y descomposición de P/E; régimen bull/bear e interacciones; aceleración/reversión/volatilidad de precio; medias y distancia a máximos; volatilidad/drawdown de SPY; y tendencia/estabilidad/sorpresa de calidad-crecimiento. Cada interruptor se conserva en la configuración y puede ser ablado.

`module/data/baselines.py` produce los scores simples de calidad, crecimiento, valor y momentum mediante medias de factores elegibles. Son comparadores deterministas para cobertura y contexto, no parte obligatoria del meta-model. El artefacto `features_coverage.json` registra observaciones, snapshots, tickers, cobertura por feature y baselines.

### 20.4 Labels

`targets_forward_3m.parquet` es un nombre histórico: contiene el retorno futuro y exceso futuro calculados usando `target_horizon_months` efectivo. Para una observación en `t`, el target utiliza el precio posterior a `t+h`, y guarda `label_end_date`. Las filas sin cierre de etiqueta quedan fuera de entrenamiento/ponderación. El nombre de archivo no debe llevar a afirmar que todo estudio usa tres meses; el manifiesto de cada run es la fuente de horizonte efectivo.

## 21. Aprendizaje walk-forward, agentes y explicabilidad

### 21.1 Secuencia temporal de entrenamiento

Para cada snapshot de predicción, `agents.py` entrena con la ventana anterior de `train_lookback_years`. La condición decisiva no es solo la fecha de feature: un ejemplo entra si su `label_end_date` ya ocurrió antes de la fecha de reentreno. Los agentes se reentrenan en revisiones fundamentales y entregan scores a snapshots intermedios sin mirar retornos aún abiertos.

Antes de ajustar, los factores se convierten a numéricos, se imputan cuando la familia lo requiere y se aplican pesos de recencia `off`, lineales o exponenciales. La vida media usada por el modo exponencial es una constante metodológica de tres años. Si no hay filas suficientes, el run falla explícitamente; no fabrica predicciones con una muestra minúscula.

### 21.2 Agentes y familias de modelo

El catálogo construye las columnas efectivas de cada agente, y **cada feature pertenece a un solo agente** (agentes ortogonales, sin doble contabilidad). `quality` recibe rentabilidad, eficiencia y fortaleza financiera; `value`, valoración y yields; `growth`, aceleración de crecimiento y tendencia/estabilidad de mejora; `momentum`, momentum y tendencia de precio; `risk`, volatilidad/drawdown y liquidez. Antes varias features se asignaban a dos agentes (p. ej. `quality_core` a quality y growth), lo que correlacionaba sus señales (quality~growth 0,78; momentum~risk 0,57) y hacía que el meta colapsara sobre un bloque; separarlas elimina esa redundancia en origen.

| Familia | Preproceso y ajuste | Salida |
|---|---|---|
| LightGBM | Árboles de gradiente con hiperparámetros barribles; acepta ausencias. | Score continuo/ranking según objetivo e importancia nativa. |
| Elastic Net | Imputación por mediana, escalado y regularización lineal. | Score continuo y coeficientes estandarizados. |
| CatBoost | Árboles alternativos tolerantes a ausencias. | Regresor para `rank_regression`, clasificador para `quartile` y `CatBoostRanker` YetiRank por snapshot para `ranking`. |

Los scores de cada familia se transforman a rango dentro de snapshot. El ensemble intra-agente puede usar una sola familia, media de rangos o pesos por Rank-IC OOS cerrado. `regularized_linear_ensemble` garantiza que Elastic Net esté disponible como contraste lineal cuando se selecciona ese modo.

### 21.3 Selección de factores y archivos de explicación

`model_native` deja toda la selección al modelo. `diagnostic_only` también retiene todas las variables, pero publica diagnósticos. `oos_stability_prune` usa sólo historia cerrada para filtrar por cobertura, fracción positiva, ventana y, si se solicita, importancia temporal por permutación. `block_gated` aplica la misma idea agregada por bloque. `feature_selection_max_features_per_agent=0` significa que no existe tope artificial.

La importancia por permutación no usa el mismo bloque entrenado/evaluado: ajusta un Ridge en snapshots anteriores, permuta una variable dentro de una cohorte posterior cerrada y mide la degradación de Rank-IC. No es equivalente a la importancia de split de un árbol. Las contribuciones locales LightGBM se escriben para explicar por qué un ticker recibió un score; SHAP no se usa para afirmar validez predictiva.

Los principales artefactos de esta etapa son `agent_scores.parquet`, `rank_ic_diagnostics.parquet`, `meta_weights.parquet`, `feature_diagnostics.parquet`, `feature_catalog.json`, `model_feature_attribution.parquet`, `agent_local_attribution.parquet` y `manifest.json` del run de agentes.

## 22. Meta-agente, diagnóstico de señal y métricas estadísticas

El meta-agente rankea la salida de cada agente dentro de cada snapshot y crea `meta_rank`. `equal` asigna pesos iguales; `rank_ic` los estima de IC positivos previos; `regime` inclina moderadamente los pesos de calidad/momentum según el retorno anterior de SPY; `stacked_oos` ajusta Ridge no negativa sobre scores OOS con etiqueta cerrada. En `rank_ic` y `regime`, tras el reparto se limita cada peso a 10 %–50 % con renormalización (mínimo universal por agente y tope 50 %), de modo que el meta no ignore del todo a ningún agente ni deje que uno domine; `equal` y `stacked_oos` no se recortan. Todos registran pesos por fecha y, cuando falta evidencia suficiente, usan un fallback explícito.

`rank_ic_diagnostics.parquet` contiene la correlación de Spearman entre score y retorno futuro por agente y cohorte. Si score o target es constante, no se calcula correlación: se registra ausencia, con lo que se evita tanto un IC indefinido como el warning de pandas/Scipy. A partir de la serie de IC se calculan media, fracción positiva, desviación, ICIR, resúmenes por año, bootstrap por bloques y diferencias pareadas cuando corresponde.

La regla de selección del study usa la media de Rank-IC de `meta_final` hasta 2024; una variante no entra en combinación dirigida si no mejora baseline y no mantiene la proporción positiva. La evidencia de 2025–2026 se calcula después y nunca debe usarse para elegir ejes.

## 23. Cartera, backtest y medidas económicas

### 23.1 Reglas de cartera

`portfolio.py` modela una cartera con estado. En cada snapshot:

1. Expulsa posiciones que ya no cumplen la regla de mantenimiento.
2. Completa siempre `target_size` con las mejores acciones por `meta_rank`.
3. Vende obligatoriamente si una tenencia queda en o por debajo de `min_hold_percentile`; fuera de ese caso rota tantas posiciones como permita una ventaja de al menos `rotation_edge_percentiles`.
4. Calcula pesos con una escala lineal min-max dentro de la cartera sobre el `meta_score` crudo (el mejor de los seleccionados pesa el doble que el peor) y los normaliza al 100 %.
5. Rebalancea solo las posiciones cuyo peso objetivo cambia al menos `rebalance_drift_tolerance` (25 % relativo a la posición); las demás se congelan y el presupuesto liberado se reparte entre las que se mueven conservando las relaciones del target global.

En las revisiones que no traen fundamentales nuevos (`review_type=price_monthly`, típicas cuando `snapshot_step_months` es más fino que `fundamental_step_months`), los umbrales de las reglas 1, 3 y 5 se endurecen con `price_only_strictness_multiplier`: sin información fundamental nueva, mover la cartera equivale a rotar por ruido de precio. Con el multiplicador en 1,0 el comportamiento es el mismo en todas las revisiones.

Esto separa el ranking predictivo de la mecánica de trading. El perfil se aplica antes de estas reglas y sólo reordena acciones ya buenas según el meta-score. Los perfiles disponibles, sus pesos y el umbral común de bondad del 60 % están definidos explícitamente en `profiles.py`; no reentrenan ni consultan información futura.

Cada perfil pondera los rangos de los cinco agentes (quality, value, growth, momentum, risk) —y opcionalmente el `meta_rank`— con pesos que pueden ser negativos cuando el estilo apuesta en contra de un agente. En `risk_rank` los factores van en dirección inversa (rango alto = menos riesgo), así que un peso positivo prioriza baja volatilidad y uno negativo la tolera. El conjunto representa arquetipos de inversor reconocibles con solapamiento mínimo entre carteras (momentum y contrarian quedan como polos opuestos por diseño):

| Perfil | Filosofía | Ejes dominantes |
| --- | --- | --- |
| `balanced` | Meta puro, sin sesgo de estilo. Referencia del sistema. | `meta_rank` |
| `growth` | Crecimiento como motor, pero crecimiento de calidad; el momentum confirma la tendencia. | growth, quality, momentum |
| `value` | Empresas infravaloradas; calidad y riesgo filtran las value traps. | value, quality, risk |
| `quality` | Compounder tipo Buffett: negocios excelentes y duraderos; la calidad manda. | quality, growth, value |
| `momentum` | Pura fuerza de precio; acepta la volatilidad que la acompaña. | momentum, −risk |
| `contrarian` | Compra lo castigado: apuesta contra el momentum reciente en nombres baratos, vigilando el riesgo. | −momentum, value, risk |
| `defensive` | Preservación de capital: baja volatilidad y calidad por encima de la rentabilidad máxima. | risk, quality, value |
| `garp` | Growth At a Reasonable Price: crece con disciplina de precio (value junto a growth). | growth, value, quality |

### 23.2 Simulación y costes

`backtest.py` valora posiciones con los precios PIT, traduce cambios de estado en compras/ventas, aplica comisión y slippage por orden, marca equity y calcula rendimiento de cartera, benchmark y exceso. Guarda posiciones, órdenes, equity por snapshot, métricas anuales y un resumen con CAGR, volatilidad, máximo drawdown, Information Ratio, turnover y métricas de señal.

El filtro `max_monthly_position_return=2.0` neutraliza retornos mensuales superiores a +200 % como probable artefacto. No es una regla de inversión. Comisión y slippage del baseline son 5 y 10 bps; el full study no puede elegir la pareja barata, pues evalúa los nueve escenarios de estrés definidos en la sección 14.

Precio y benchmark deben compartir la **misma rejilla** que los scores. Como la rejilla depende de `execution_lag_days` (`fin_de_periodo + execution_lag_days`, §17), un escenario con un retardo distinto produce fechas de snapshot distintas. Por eso el backtest lee `asset_price_point_in_time.parquet` y `benchmark_point_in_time.parquet` **materializados junto al agente del run** (su workspace privado) y solo cae al `data/processed` global en el backtest CLI aislado. Mezclar scores de un retardo con precios de otro dejaría snapshots sin precio de benchmark y abortaría el run. La fase de cartera, que reejecuta el backtest sobre el modelo finalista, hereda de ese modelo tanto el precio como el benchmark para conservar la rejilla.

## 24. Ejecución de escenarios, caché, resultados inmutables y estudios

### 24.1 Runs y reutilización

`execute_run` calcula la huella de inputs raw y una `execution_hash` que incluye settings, etapas, huella de código, Python e inputs. Si ya existe un run `succeeded` con la misma identidad de cálculo, puede reutilizarse y se registra como tal en el study; no se confunde con el hash completo, que también incorpora etiqueta, intención y descripción. La búsqueda de un run reutilizable (`find_completed_execution`) resuelve por el `execution_hash` ya indexado en cada línea de `results/registry.jsonl`, sin reabrir cada manifiesto de disco (solo cae al manifiesto para entradas antiguas que no lo traigan indexado).

Las etapas dataset, features, agents y backtest se pueden restaurar desde `data/recycle`. La clave de caché combina la etapa, su configuración relevante, las huellas de input y una **huella de código acotada a esa etapa** (ver §24.4): un cambio que solo toca el meta-agente invalida la caché de `agents` pero no la de `dataset` ni `features`. El contrato declara qué archivos entran y salen. La restauración enlaza los artefactos por *hardlink* (con copia de reserva si el sistema de ficheros no lo permite): mismos bytes, sin pagar la copia completa de los parquet en cada acierto. La caché acelera el estudio, pero la publicación definitiva siempre se realiza en un run inmutable.

### 24.2 Store y estructura de resultados

`ResultsStore` crea `results/runs/<run_id>/` con `run_manifest.json`, `config.json`, `status.json`, log y `artifacts/`. Al completar, añade una línea JSON canónica a `results/registry.jsonl`; no sobrescribe ejecuciones existentes. La identidad incorpora SHA-256 de configuración, huellas de input, revisión Git disponible, versión de Python y plataforma.

Los artefactos publicados incluyen, cuando existen, resumen de backtest, scores, pesos meta, diagnóstico IC, anual, posiciones, órdenes, equity, atribuciones, catálogo, perfiles, robustez y panel/precios necesarios para consultar un resultado histórico. También se materializan CSV de ranking, posiciones, órdenes y métricas anuales, `position_lifecycle.parquet` y `learning_summary.json` para consumo rápido de la consola.

Un study vive en `results/studies/<study_id>/` y contiene `study_manifest.json` y `run_ids.json`. El manifiesto conserva nombre, hipótesis, configuración, fases y estado. `decision.json` contiene la selección, reserva, robustez, costes y perfiles cuando el study finaliza. No se modifica retroactivamente un study histórico para hacer que parezca generado con el protocolo actual.

### 24.3 Ciclo del full study oficial

El full study usa ejes de `STUDY_OPTIONS`, no un producto cartesiano. Ejecuta baseline, variaciones aisladas, ablaciones de bloque/agente/familia, combinaciones dirigidas de candidatos aceptados, afinado de hiperparámetros, fase de cartera, perfiles, nueve costes, robustez y reserva. Los parámetros de selección de factores se combinan con una política que los consuma, no como barridos inertes.

Además del orquestador oficial, `experiments.py`, `ablaciones.py` y `ejes.py` mantienen el mecanismo de escenarios explícitos para experimentos dirigidos/manuales. Sus fingerprints separan variables que afectan datos, features, modelo o cartera, de modo que sólo se recalcula lo necesario. Los resultados de este mecanismo histórico son útiles para explorar, pero la decisión oficial vigente la produce `execution.py` con corte de selección en 2024.

### 24.4 Rendimiento, configurabilidad y buenas prácticas de optimización

El coste dominante de un study es el walk-forward: cada run reentrena `fechas_de_reentreno × agentes × familias` modelos, y un study encadena decenas de runs. Las optimizaciones reducen tiempo de pared sin alterar universo PIT, etiquetas, fronteras temporales ni reglas de selección. Se admiten diferencias numéricas pequeñas y reproducibles derivadas del paralelismo, siempre dentro de tolerancias explícitas y sin cambiar conclusiones. Se verifican con un oráculo numérico, la suite y `ruff`.

- **Versión de código manual por etapa (`environment.Settings.{dataset,features,agents,backtest}_code_version`).** La clave de caché y el `execution_hash` no dependen de ningún hash automático del código: dependen solo de parámetros, huellas de entrada y de estos enteros manuales. Editar logging o comentarios en cualquier módulo nunca invalida la caché por sí solo; tras un cambio real de lógica en una etapa, el usuario sube a mano el `*_code_version` correspondiente para forzar el recálculo. Es una decisión deliberada de control manual, no automática por contenido de fichero.
- **Restauración de caché por hardlink.** `recycle.restore` enlaza los artefactos inmutables en lugar de copiarlos, con copia de reserva si el FS no lo soporta.
- **`lgbm_n_jobs` configurable (por defecto `-1`, todos los núcleos; `3` dentro de un worker paralelo de study).** LightGBM es determinista con hilos, así que solo cambia la velocidad, no el resultado. No entra en la clave de caché de la etapa (cambiar los hilos no invalida artefactos).
- **Cola dinámica de escenarios en Fase 1/2/3 del full study.** `_run_specs_with_queue` en `module/runs/execution.py` mantiene hasta `PARALLEL_SCENARIO_WORKERS` (4) escenarios en vuelo a la vez; en cuanto uno termina, entra el siguiente pendiente sin esperar a los demás (sustituye al patrón de "tandas" que esperaban a los 4 antes de continuar). Prioriza lanzar primero escenarios cuya firma de materialización (dataset/features/agents) no coincida con la de otro ya en vuelo; si ya no quedan, el siguiente escenario espera en `cache_lock` a que la clave compartida se publique y la recicla, sin recalcularla ni chocar al escribir. Fase 2 paraleliza los trials "segunda mejor opción por eje" (todos parten del mismo `selected` fijo); Fase 3 paraleliza el barrido de valores dentro de cada eje. Las fases y, dentro de fase 3, los ejes, siguen siendo estrictamente secuenciales entre sí.
- **Vectorización de puntos calientes en pandas.** `combine_agent_scores` calcula la media ponderada de rangos por fecha con álgebra vectorizada en vez de iterar fila a fila; el backtest agrupa los scores por snapshot una sola vez en lugar de filtrar con máscara booleana dentro del bucle. Ambos reproducen exactamente la salida previa (mismo manejo de NaN y orden de operaciones).
- **Menos relecturas de disco.** `_summary_for_run` memoiza el resumen por run (los artefactos de un run son inmutables una vez escritos), evitando releer `backtest_summary.json` y `rank_ic_diagnostics.parquet` varias veces por run.

Principio general: una palanca de rendimiento solo se acepta si conserva contratos PIT/OOS y produce resultados reproducibles dentro de tolerancia. El paralelismo se limita para evitar sobresuscripción y cada escenario trabaja en un workspace privado. No se comparten modelos ajustados entre configuraciones distintas.

### 24.5 Caché v2, aislamiento y reanudación

La clave de cada etapa incluye su configuración efectiva, hashes de entradas, huella de código,
versión de esquema y versión de Python. El manifiesto conserva tamaño y SHA-256 de cada fichero;
si falta uno o no coincide, la entrada completa se rechaza. Un bloqueo single-flight evita que
dos procesos publiquen simultáneamente la misma transformación.

Los backtests de Fases 4–5 copian el agente padre exacto a un workspace del run. No consultan el
directorio de agentes «más reciente». El placebo recibe un target permutado privado y no modifica
`targets_forward_3m.parquet`. Los manifiestos guardan fase, escenario y tiempos por etapa; un
study incompleto se puede reanudar y deduplica únicamente ejecuciones con estado `succeeded`.

La Fase 3 afina greedy profundidad, árboles, learning rate y tamaño mínimo de hoja. Después se
ejecutan semillas 7, 42 y 2026 como validación de estabilidad, sin seleccionar la más favorable.
La Fase 4 recalcula sus métricas solo hasta 2024; 2025–2026 permanece reservado hasta congelar la
cartera. En Fase 5 `balanced` conserva `meta_rank` y los ocho perfiles se reportan sin ganador.

### 24.6 Planificación paralela y observabilidad

La Fase 1 analiza antes de ejecutar las firmas de materialización de dataset, features y agentes.
Programa tandas de hasta cuatro procesos y solo junta escenarios cuyas etapas aún no materializadas
sean independientes. Cada worker tiene workspace y log propios, tres hilos de modelo y publicación
atómica; tras materializar una firma, los escenarios posteriores restauran caché en vez de repetirla.
La consola agrega los logs de todos los runs activos, etiquetados por run, y muestra una ventana
compacta desplazable. Las fases greedy conservan orden secuencial; cartera y perfiles usan el
artefacto exacto del modelo finalista y no vuelven a entrenar agentes.

### 24.7 Reciclaje por familia de modelo dentro de la etapa de agentes

La etapa de agentes concentra ~90 % del tiempo de un run: es el reentreno walk-forward
`fechas × agentes × familias`, del orden de varios cientos de `fit` de LightGBM, CatBoost y
ElasticNet. Ese coste se descompone internamente en dos claves de reutilización distintas:

- **`agents_fit`**, el ajuste walk-forward de **una sola familia** sobre todos los pares
  snapshot × agente. Su clave incluye la familia concreta y los campos que afectan al ajuste
  (ancla de ejecución, ventana de entrenamiento, cadencia de reentreno, hiperparámetros del árbol,
  semilla, recencia, objetivo, selección de features), pero **no** `meta_type`,
  `meta_ic_lookback_quarters`, `intra_agent_ensemble_mode` ni las demás familias. Cada familia
  publica sus scores, importancias y contribuciones locales como una entrada content-addressed.
- **`agents`**, la combinación posterior (ensemble intra-agente por rangos y meta-agente). Es
  barata y su identidad la determinan los hashes de las salidas `agents_fit` reutilizadas más los
  campos de meta.

Consecuencia práctica en un barrido: cambiar el meta (`meta_type`, `meta_ic_lookback_quarters`,
`min_rank_ic_cross_section`) reutiliza intactos todos los ajustes y solo rehace la combinación;
una ablación de familia (por ejemplo quitar CatBoost) reutiliza las familias restantes. La
recombinación reproduce exactamente el ensemble intra-agente previo —mismo orden de familias, mismo
promedio de rangos— de modo que los resultados numéricos son idénticos; el reciclaje solo evita
recalcular ajustes ya realizados. La clave `agents_fit` reutiliza la huella de código del módulo
de agentes.

## 25. Research Console, API local e informes HTML

La consola se sirve únicamente en `127.0.0.1:8765`; no hay autenticación remota ni publicación a Internet. `JobManager` ejecuta trabajos en segundo plano y permite consultar estado. `dashboard.py` valida IDs de run para que las rutas API no escapen de `results/runs`.

| Ruta API | Finalidad |
|---|---|
| `GET /api/defaults` | Defaults, grupos de settings, opciones de study, presets, constantes fijas y estrés. |
| `GET /api/runs`, `/api/studies`, `/api/run/<id>`, `/api/study/<id>` | Índice y detalle de resultados inmutables. |
| `GET /api/learning`, `/api/meta_weights`, `/api/performance` | Rank-IC, pesos, equity, anual y resumen. |
| `GET /api/portfolio`, `/api/trades` | Composición y órdenes históricas. |
| `GET /api/stocks`, `/api/stock/summary`, `/api/stock/history`, `/api/stock/agents`, `/api/ticker`, `/api/ranking` | Explorador por acción, ratios PIT, scores, atribución, precio y ranking. |
| `POST /api/experimental` | Lanza un run manual con settings seleccionados. |
| `POST /api/study` | Lanza un study dirigido definido desde consola. |
| `POST /api/optimization` | Lanza el full study con nombre e hipótesis. |

`app/index.html` y los módulos de `app/js/views/` ofrecen vistas de consola, estudios, resultados, aprendizaje, rendimiento, cartera y acciones. `api.js` centraliza llamadas, formateo, escaping HTML, tablas ordenables y ayudas de métricas; `charts.js` centraliza Chart.js. La pantalla Full study marca todos los ejes barribles y los bloquea para que no puedan desmarcarse; muestra por separado constantes fijas y estrés de costes. Las vistas no recalculan resultados: consumen exclusivamente artefactos publicados.

`module/ui/reports.py` construye HTML auto-contenido para un run o una comparación de escenarios, con resumen, rendimiento, aprendizaje, agentes, cartera, cobertura, posiciones, tablas y gráficos Matplotlib embebidos. Es una vía de inspección reproducible adicional a la consola, no un motor de selección.

## 26. Pruebas, reproducibilidad y mantenimiento

La suite está dividida por contrato, no por comodidad de implementación:

| Área de pruebas | Qué protege |
|---|---|
| `tests/download/` | Clientes, pipeline de descarga, EDGAR, universo y cobertura. |
| `tests/dataset/` | Fechas de filing, miembros históricos, valores fundamentales y ausencia de lookahead. |
| `tests/features/` | Alineación de target, fórmulas, cobertura, neutralización y artefactos. |
| `tests/agents/` | Objetivos, modelos, ensembles, meta y correlaciones seguras. |
| `tests/backtest/` | Invariantes de cartera, órdenes, costes y métricas económicas. |
| `tests/experiments/` | Fingerprints, escenarios, selección, decisiones y optimización unificada. |
| `tests/report/` | Informe de run y comparador de escenarios. |
| Tests de raíz | Explorador de acciones, store, perfiles, estadísticas, robustez y huella de código por etapa (aislamiento de caché). |

La verificación mínima es `python -m pytest tests/ -q` y `python -m ruff check .`. Para una modificación metodológica deben añadirse pruebas de fórmula, disponibilidad temporal, serialización de settings y efecto real de cualquier nueva opción de study. Un control visible sin impacto en la predicción no debe considerarse configurable: debe implementarse, eliminarse o documentarse como no operativo.

Las dependencias están fijadas por mínimos en `requirements.txt`: pandas/numpy/pyarrow para datos, LightGBM/scikit-learn/CatBoost/SHAP para aprendizaje y explicación, requests para ingesta, matplotlib para informes y pytest para pruebas. La combinación exacta de versiones de un run se aproxima mediante entorno y versión de Python en el manifiesto; para una reproducción académica estricta se debe congelar adicionalmente el entorno (`pip freeze`) junto al study final.

## 27. Límites de alcance y afirmaciones que el repositorio no permite hacer

Este documento refleja el código actual y, por tanto, también sus límites. El proyecto no es un sistema de ejecución real, asesoramiento financiero, predictor causal ni motor NLP. No usa noticias, estimaciones de analistas, short interest, dividendos/recompras ni estados SEC detallados como señal. No dispone de costes de mercado observados por acción, de datos intradía ni de una validación externa a S&P 500.

El uso de modelos ML no elimina riesgo de sobreajuste: hay muchos ejes, dependencia temporal, cobertura imperfecta y una reserva de sólo dos años. El full study reduce grados de libertad con escenarios dirigidos, corte 2024, placebo, LOYO, bootstrap y estrés, pero no convierte una mejora puntual en certeza. La formulación defendible es siempre condicional: *bajo este universo, fuentes, período, configuración y protocolo, la señal mostró —o no mostró— evidencia OOS*.

## 28. Cómo mantener este documento como fuente maestra

Toda modificación del repositorio debe actualizar esta especificación en la misma entrega si cambia cualquiera de estos contratos: fuente de datos, columna PIT, fórmula de feature, lista de catálogo, comportamiento de agente/meta, regla de cartera, opciones del study, selección, artefacto persistido, endpoint de consola o prueba requerida. La bitácora explica la decisión temporal; este documento describe el estado actual; el informe final reporta una ejecución concreta; y LaTeX debe extraer sólo hechos que puedan recorrerse hasta un manifiesto y sus artefactos.
