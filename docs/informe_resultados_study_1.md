# Informe de resultados del study oficial

> **Nota de vigencia metodológica.** Este documento es la evidencia histórica del study 1 y no se
> ha reescrito con resultados del protocolo v2. Sus 2025–2026 ya fueron observados y ahora se
> clasifican como `known_stress_not_selection`, no como holdout. El placebo antiguo de tres
> permutaciones que reportó `p = 0` no constituye un p-valor válido; el protocolo v2 exige 9.999
> permutaciones con corrección add-one. La configuración ganadora se conserva como incumbent.

**Study:** `optimization-official` · `20260724--optimization-official--4dfea2986540`
**Fecha de ejecución:** 2026-07-24 · **Estado:** `succeeded` · **Escenarios completados:** 104 (167 runs)
**Estrategia histórica:** `unified_full_cycle` · **Métrica de selección:** Rank-IC OOS (hasta 2024) ·
**Tratamiento actual de 2025–2026:** `known_stress_not_selection`

Este informe analiza el barrido completo ejecutado bajo el protocolo entonces vigente: selección de modelo
(Fases 1–3), run final, cartera (Fase 4), perfiles (Fase 5) y robustez. Sigue el principio del proyecto:
**el Rank-IC fuera de muestra es el criterio; la rentabilidad
y los perfiles son comprobaciones económicas posteriores, no criterios de selección**. Los resultados negativos
se conservan y se reportan tal cual.

---

## 1. Resumen ejecutivo

El study seleccionó una configuración parsimoniosa (un solo modelo LightGBM poco profundo, meta-agente apilado,
poda estricta de features, horizonte de etiqueta a 12 meses). Su tramo 2025–2026 sirve hoy como estrés
histórico conocido, no como validación ciega.

**Veredicto:** la señal es **estadísticamente real dentro del periodo de selección (2015–2024)** pero
**se degrada en el estrés conocido 2025–2026**, y su traducción a rentabilidad es **marginal**.

- Rank-IC OOS del finalista (selección ≤2024, 117 cohortes): **0.0944**, con 72.6 % de cohortes con IC>0.
- Test de placebo histórico: el real superó cinco valores placebo, pero con solo tres
  permutaciones efectivas el **`p = 0.0` reportado no es inferencialmente válido**.
- Bootstrap por bloques: IC 95 % **[0.050, 0.131]** → el Rank-IC no cruza cero, es significativo.
- **Reserva 2025–2026: Rank-IC medio −0.0095** (negativo). El alfa anual se vuelve negativo en 2024 (−7.8 %),
  2025 (−11.9 %) y 2026 parcial (−10.2 %).
- La cartera **no supera de forma convincente a carteras aleatorias** del mismo tamaño (percentil 18).

Conclusión operativa: el sistema **no está listo para desplegar como estrategia**. El valor del trabajo es
**metodológico y reproducible** — un pipeline honesto que detecta su propia limitación — y el hallazgo de que
una señal transversal real hasta 2024 **se degrada a partir de 2024** es en sí mismo un resultado publicable.

### 1.1 Métricas del finalista (periodo completo 2015–2026)

| Métrica | Valor |
|---|---:|
| Rank-IC medio OOS (selección ≤2024) | **0.0944** |
| Fracción de cohortes con IC>0 | 72.6 % |
| Desviación típica del Rank-IC | 0.123 |
| Cohortes de selección | 117 |
| CAGR cartera | 14.04 % |
| CAGR benchmark (SPY) | 13.81 % |
| Diferencia de CAGR | +0.23 pp |
| Information Ratio | 0.023 |
| Alfa anual medio / mediano | +1.01 % / +0.69 % |
| Peor año (alfa) | −11.9 % |
| Beat rate (años que baten al SPY) | 58.3 % (7 de 12) |
| Max drawdown cartera vs benchmark | 29.8 % vs 22.9 % |

### 1.2 Configuración ganadora

| Grupo | Ajuste final |
|---|---|
| Entrenamiento | `train_lookback_years = 12` · `target_horizon_months = 12` |
| Meta-agente | `meta_type = stacked_oos` · `meta_ic_lookback_quarters = 16` |
| Selección de features | `oos_stability_prune` · cobertura mín. 0.3 · fracción positiva mín. 0.45 · máx. 8 features/agente |
| Familia de modelo | **única: LightGBM** (`max_depth 3`, `n_estimators 100`, `learning_rate 0.05`, `min_child_samples 20`) |
| Ensemble intra-agente | `single` |
| Calendario / ejecución | `execution_lag_days = 60` |
| Cartera | `target_size = 10` |
| Winsorización de métricas | 0.0 (desactivada) |

Los 5 agentes (`quality`, `value`, `growth`, `momentum`, `risk`) se conservan siempre: no se optimiza
`enabled_agents` para que los perfiles de la Fase 5 sean comparables.

Run del modelo final: `20260724--1e2bb7be33e6` · Run de cartera/perfil final (balanced): `20260724--35794c6ff34f`.

---

## 2. Metodología del study

El ciclo `unified_full_cycle` encadena estas fases (detalle del contrato en `docs/doc.md §9` y §14;
implementación en `module/runs/execution.py`):

1. **Fase 1 — ejes de modelo aislados.** Baseline + una modificación cada vez (no cartesiano). Elige el mejor
   nivel de cada eje por Rank-IC OOS.
2. **Fase 2 — greedy top-2.** Parte del mejor nivel de cada eje y prueba el segundo mejor sobre la combinación,
   aceptándolo solo si mejora de forma estable.
3. **Fase 3 — hiperparámetros LightGBM** (profundidad, nº de árboles, learning rate, min_child) + **Fase 3b —
   estabilidad de semillas** (7, 42, 2026).
4. **Run final.** Entrena el modelo ganador completo (dataset → features → agentes → backtest).
5. **Fase 4 — cartera.** Optimiza `target_size` por **Information Ratio** re-backtesteando sin reentrenar;
   luego **4b estrés de costes** y **4c estrés de reglas mecánicas**.
6. **Fase 5 — perfiles de inversor.** Reporta 8 perfiles sobre el modelo+cartera óptimos (no se optimizan).
7. **Cierre — robustez y estrés histórico 2025–2026.**

**Criterio de selección.** El modelo se elige **siempre por Rank-IC OOS y estabilidad, nunca por rentabilidad**
(desempate: `mean_rank_ic` ↓ → `rank_ic_positive_fraction` ↓ → `rank_ic_std` ↑ → nº cohortes ↓). El Information
Ratio solo interviene para elegir la **cartera** (Fase 4), porque esos ejes no reentrenan el modelo.

**Separación temporal.** La selección solo mira cohortes con fecha de predicción **hasta 2024**. Los años
**2025–2026 no intervienen en ninguna decisión**, pero ya fueron observados y hoy son un estrés histórico
conocido. Sirven para describir el **sesgo de selección** (al comparar 104 escenarios, parte del máximo puede ser
suerte). Separarlos no significa "no entrenar" esos años: el walk-forward avanza con normalidad; solo se apartan
a la hora de **decidir** qué configuración es la buena.

---

## 3. Fase 1 — barrido de ejes de modelo aislados

Se evaluaron 104 escenarios (baseline + variaciones). El Rank-IC de un eje aislado es más bajo que el del
finalista completo porque cada uno mide una sola modificación sobre el baseline.

**Mejores ejes por Rank-IC:**

| Escenario ganador | Rank-IC | Cohortes IC>0 |
|---|---:|---:|
| `target_horizon_months = 12` | 0.0789 | 70.9 % |
| `execution_lag_days = 30` | 0.0622 | 70.1 % |
| `feature_selection_max_features_per_agent = 8` | 0.0607 | 63.2 % |
| `execution_lag_days = 60` | 0.0602 | 67.5 % |
| `feature_selection_min_positive_fraction = 0.45` | 0.0594 | 64.1 % |

**Peores ejes por Rank-IC (evidencia negativa conservada):**

| Escenario | Rank-IC | Cohortes IC>0 |
|---|---:|---:|
| `train_lookback_years = 2` | −0.0706 | 35.0 % |
| `feature_selection_min_positive_fraction = 0.7` | −0.0081 | 50.4 % |
| `train_lookback_years = 6` | −0.0077 | 50.4 % |
| `recency_weighting = exponential` | −0.0071 | 49.6 % |

**Ejes rechazados por "no mejora estable frente al baseline":** cadencia de snapshot y de fundamentales,
`objective`, `min_rank_ic_cross_section`, `recency_weighting`, `neutralize_by_sector`, `regime_extended`,
`quality_growth_derived`, `feature_selection_lookback_quarters`, `feature_selection_min_permutation_importance`,
ventanas de riesgo y técnicas.

**Lectura.** El barrido premia **parsimonia y horizonte largo**: histórico de entrenamiento amplio (los
horizontes de 2 y 6 años degradan), etiqueta a 12 meses, poda de features agresiva (8 por agente). Las
ampliaciones de complejidad (ponderación por recencia, régimen extendido, neutralización sectorial, features
derivadas) **no aportan señal estable** y se descartan. Que `min_positive_fraction = 0.7` colapse indica que
una poda *demasiado* estricta también destruye señal: el óptimo está en 0.45.

---

## 4. Fase 2 — greedy top-2

Partiendo del mejor nivel de cada eje (`combined_best` = Rank-IC 0.0846), se probó el segundo mejor de cada uno:

| Ajuste probado | Rank-IC | ¿Mejora sobre combined_best? |
|---|---:|---|
| `execution_lag_days` (segundo) | **0.0904** | **Sí → adoptado (lag pasa de 30 a 60 días)** |
| `enabled_feature_blocks` (segundo) | 0.0900 | Marginal, no adoptado |
| `meta_type` (segundo) | 0.0863 | No |
| `metric_winsorization` (segundo) | 0.0857 | No |
| `train_lookback_years` (segundo) | 0.0809 | No |
| `target_horizon_months` (segundo) | 0.0226 | No (degrada mucho) |
| Resto de "second" | 0.0846 | Neutro |

**Lectura.** Casi ningún segundo candidato añade valor, lo que **confirma la parsimonia**: la señal proviene de
pocas decisiones. La única corrección relevante es el retardo de publicación, que sube de 30 a **60 días**
(observar los fundamentales con más margen tras el cierre mejora la estabilidad del Rank-IC). Por eso el ajuste
final es `execution_lag_days = 60` pese a que la Fase 1 aislada prefería 30.

---

## 5. Fase 3 — hiperparámetros y estabilidad de semillas

**Afinado de LightGBM.** El Rank-IC es **plano y robusto** en toda la rejilla: el ganador `min_child_samples=20`
da 0.0944, pero prácticamente todos los niveles quedan en el rango **[0.088, 0.0944]**. Solo la profundidad alta
(`max_depth` 6–8) y los learning rates extremos (0.01, 0.2) degradan de forma apreciable.

| Extremo de la rejilla | Rank-IC |
|---|---:|
| Mejor (`min_child_samples = 20`, ganador) | 0.0944 |
| `max_depth = 8` | 0.0792 |
| `learning_rate = 0.2` | 0.0802 |

**Lectura.** Poco margen de sobreajuste: el resultado **no depende de un hiperparámetro afortunado**. Se elige el
modelo más simple y regularizado (`max_depth 3`, 100 árboles), coherente con la parsimonia de las fases previas.

**Fase 3b — semillas.** Reentrenando el finalista con tres semillas:

| Semilla | Rank-IC | Cohortes IC>0 | Rank-IC std |
|---:|---:|---:|---:|
| 7 | 0.0918 | 71.8 % | 0.121 |
| 42 | 0.0944 | 72.6 % | 0.123 |
| 2026 | 0.0921 | 71.8 % | 0.120 |

**Lectura.** El Rank-IC varía menos de 0.003 entre semillas → el resultado es **estable a la aleatoriedad del
entrenamiento**, no un artefacto de una semilla concreta.

---

## 6. Fase 4 — cartera, costes y reglas mecánicas

### 6.1 Tamaño de cartera (optimizado por Information Ratio)

`target_size` es el único eje de cartera que se optimiza (mandato real de diversificación), medido por IR en la
ventana de selección (≤2024):

| target_size | Information Ratio | Δ CAGR | Beat rate | Max DD |
|---:|---:|---:|---:|---:|
| 8 | 0.094 | +2.80 pp | 70 % | 32.0 % |
| **10 (elegido)** | **0.105** | +2.94 pp | 70 % | 29.8 % |
| 12 | 0.095 | +2.52 pp | 60 % | 29.7 % |
| 15 | 0.089 | +2.24 pp | 60 % | 28.1 % |
| 20 | 0.086 | +1.90 pp | 70 % | 26.0 % |

Gana `target_size = 10`: mejor IR y buen equilibrio entre concentración y drawdown. (Nota: el Δ CAGR de esta
tabla se mide en la ventana de selección ≤2024, por eso es más alto que el +0.23 pp del periodo completo, que
incluye los años negativos 2024–2026.)

### 6.2 Estrés de costes (se estresa, no se optimiza)

Se reporta cada par comisión×slippage sobre el finalista, **sin poder elegir el más barato**:

| Comisión (bps) | Slippage (bps) | CAGR cartera | IR | Beat rate |
|---:|---:|---:|---:|---:|
| 0 | 5 | 14.91 % | 0.047 | 66.7 % |
| 5 | 10 (**base**) | 14.04 % | 0.023 | 58.3 % |
| 5 | 20 | 13.18 % | −0.000 | 50.0 % |
| 10 | 20 | 12.75 % | −0.012 | 50.0 % |

**Lectura.** El alfa es **frágil ante costes altos**: con slippage de 20 bps el Information Ratio cruza a
negativo. La estrategia solo tiene sentido con costes de transacción moderados.

### 6.3 Estrés de reglas mecánicas (se reporta, no se elige)

Las reglas de rotación/retención/deriva se estresan pero **nunca se seleccionan** (elegir la más favorable sería
un grado de libertad ilícito):

| Eje | Rango de IR observado | Comentario |
|---|---|---|
| `min_hold_percentile` (60–85) | 0.023 constante | Sin efecto |
| `price_only_strictness` (1.0–2.0) | 0.023 constante | Sin efecto |
| `rebalance_drift_tolerance` (0.15–0.35) | 0.023–0.038 | Efecto leve |
| `rotation_edge_percentiles` (5–15) | 0.017–0.083 | El más sensible |

**Lectura.** El sistema es **estable frente a la mecánica de cartera**. Algunos ajustes (rotación=15,
drift=0.35) incluso mejorarían el resultado, pero **se reportan, no se adoptan**, por disciplina metodológica.

---

## 7. Fase 5 — perfiles de inversor

Ocho perfiles reportados (no optimizados; `balanced` = meta puro es la referencia). Solo tres baten al SPY:

| Perfil | CAGR | Δ vs SPY | Information Ratio | Alfa anual medio | Beat rate | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| **garp** | 15.50 % | **+1.69 pp** | **0.050** | +2.39 % | 58.3 % | 21.5 % |
| **value** | 15.09 % | **+1.28 pp** | 0.042 | +2.72 % | 41.7 % | 20.1 % |
| **balanced** (ref.) | 14.04 % | +0.23 pp | 0.023 | +1.01 % | 58.3 % | 29.8 % |
| contrarian | 13.02 % | −0.80 pp | 0.003 | +1.08 % | 33.3 % | 19.7 % |
| defensive | 12.58 % | −1.23 pp | −0.026 | −0.60 % | 50.0 % | 25.2 % |
| quality | 12.20 % | −1.62 pp | −0.036 | −0.00 % | 50.0 % | 29.7 % |
| momentum | 10.87 % | −2.94 pp | −0.047 | −2.83 % | 33.3 % | 37.0 % |
| growth | 8.63 % | −5.19 pp | −0.136 | −3.59 % | 33.3 % | 28.8 % |

**Lectura.** Hay un **sesgo estructural hacia value/quality/garp**: los perfiles que inclinan la cartera hacia
valor y calidad rentable (garp = value + calidad) baten al índice, mientras que **growth y momentum destruyen
alfa**. Es consistente con la composición del universo y el periodo. Los perfiles value y garp también reducen
notablemente el drawdown (20–21 % vs 30 % de balanced).

---

## 8. Resultados económicos año a año

Rentabilidad del finalista (cartera `balanced`, run `20260724--35794c6ff34f`):

| Año | Retorno cartera | Retorno SPY | Alfa | ¿Bate SPY? | Max DD año | IR año |
|---:|---:|---:|---:|:--:|---:|---:|
| 2015 | −5.84 % | −0.63 % | −5.21 % | No | 11.0 % | −0.25 |
| 2016 | 16.96 % | 17.87 % | −0.91 % | No | 7.1 % | 0.15 |
| 2017 | 24.27 % | 18.81 % | +5.46 % | Sí | 2.2 % | 0.30 |
| 2018 | −10.97 % | −11.32 % | +0.35 % | Sí | 15.1 % | −0.01 |
| 2019 | 28.92 % | 24.20 % | +4.71 % | Sí | 4.2 % | 0.17 |
| 2020 | 17.05 % | 16.02 % | +1.03 % | Sí | 29.8 % | 0.11 |
| 2021 | 51.48 % | 30.38 % | **+21.10 %** | Sí | 1.4 % | 0.56 |
| 2022 | −6.03 % | −12.06 % | +6.03 % | Sí | 12.4 % | 0.07 |
| 2023 | 28.35 % | 18.94 % | +9.41 % | Sí | 7.6 % | 0.20 |
| 2024 | 13.45 % | 21.27 % | −7.82 % | No | 5.4 % | −0.37 |
| **2025** *(estrés conocido)* | 3.62 % | 15.50 % | **−11.88 %** | No | 8.1 % | −0.23 |
| **2026** *(estrés conocido, parcial)* | −2.88 % | 7.33 % | **−10.21 %** | No | 7.9 % | −0.74 |

**La historia clave está en la última columna de años.** El sistema es fuerte en 2017–2023 (con un 2021
excepcional, +21 pp de alfa) y luego **gira negativo justo en 2024 y en todo 2025–2026**. La
degradación no es una anomalía puntual: son tres años consecutivos de alfa negativo, y coinciden con el tramo
que no se usó para seleccionar.

### 8.1 Gráficos disponibles

Los gráficos no se guardan en disco: se generan bajo demanda con `module/ui/reports.py`
(`build_run_report` → `report.html` con figuras matplotlib) y en la Research Console (`app/`). Para el LaTeX,
exportar desde ahí las figuras clave:

- **Curva de equity / rentabilidad acumulada** — `app/js/views/performance.js` (`TFMCharts.equityChart`).
- **Barras de alfa anual** — `app/js/views/performance.js` (`TFMCharts.alphaBars`); refleja la tabla §8.
- **Rank-IC por cohorte en el tiempo** — `app/js/views/learning.js` (`TFMCharts.rankIcOverTime`); permite ver
  visualmente el deterioro post-2024.
- **Evolución de pesos del meta-agente** — `app/js/views/learning.js` (`TFMCharts.metaWeights`).
- **Composición de cartera por snapshot** — `app/js/views/portfolio.js` (`TFMCharts.portfolioComposition`).

---

## 9. Robustez

Cuatro pruebas independientes (artefacto `robustness.json` del run `20260724--1e2bb7be33e6`):

| Prueba | Resultado | Interpretación |
|---|---|---|
| **Placebo** (permutación de etiquetas) | Real 0.0894 vs placebo 0.0018 ± 0.0050; el antiguo **p = 0.0 no es válido** | Resultado descriptivo compatible con señal, pero tres permutaciones no permiten inferencia ni descartar fuga por sí solas. |
| **Bootstrap por bloques** | Media 0.0894; **IC 95 % [0.050, 0.131]** (123 cohortes, bloque 4) | El Rank-IC **no cruza cero** → es estadísticamente significativo. |
| **Leave-one-year-out** | Rank-IC ∈ **[0.073, 0.104]** quitando cualquier año | **Ningún año explica el resultado**; la señal es difusa en el tiempo, no dependiente de un ejercicio concreto. |
| **Random-portfolio** | Modelo CAGR 11.85 % → **percentil 18** (aleatorias: media 18.7 %, p95 47.1 %); `beats_random_convincingly = False` | La **rentabilidad no supera de forma convincente al azar**. *(Matiz: las carteras aleatorias comparten el universo del modelo, que en este periodo fue muy alcista; discutir sesgo de universo.)* |

**Lectura.** Hay una tensión honesta y muy informativa: el Rank-IC es una **señal de ordenación real y
significativa** (placebo + bootstrap + LOYO coinciden), pero esa capacidad de ordenar **no se traduce en
rentabilidad superior al azar** una vez formada la cartera. El modelo "sabe" ordenar activos mejor que el ruido,
pero el margen es demasiado pequeño para batir económicamente a una selección aleatoria en un mercado alcista.

---

## 10. Estrés histórico conocido 2025–2026

Es el punto crítico del study histórico. El finalista se había congelado tras seleccionar solo con datos ≤2024;
el tramo ya fue observado y no puede presentarse hoy como holdout:

| Métrica | Selección (≤2024) | Estrés conocido (2025–2026) |
|---|---:|---:|
| Rank-IC medio | +0.0944 | **−0.0095** |
| Cohortes | 117 | 6 |
| Alfa anual | positivo (media +2.6 % en 2017–2023) | negativo (−11.9 % / −10.2 %) |

**El Rank-IC OOS histórico no se sostiene en 2025–2026.** Pasa de claramente positivo a ligeramente
negativo, y el alfa acompaña. Esto puede deberse a:

1. **Cambio de régimen de mercado** — que la relación entre los factores value/quality y el retorno futuro se
   haya debilitado o invertido a partir de 2024.
2. **Decaimiento de factores** — los factores que sostienen la señal (value, calidad rentable) pasaron por un
   periodo desfavorable, coherente con que growth/momentum ya destruían alfa en la Fase 5.
3. **Sesgo de selección residual** — pese a la robustez interna, parte del Rank-IC de 2015–2024 podría estar
   sobreajustado al periodo de búsqueda; el deterioro posterior es compatible con ese riesgo.

No es posible distinguir estas causas con la evidencia actual (solo seis cohortes en ese tramo). Pero el mensaje
es inequívoco: **con esta configuración, hoy no habría base para confiar la cartera al modelo.**

---

## 11. Conclusiones

1. **Metodológicamente, el study es reproducible pero ya no es confirmatorio.** Conserva configuración,
   artefactos y evidencia negativa, pero sus 104 escenarios, su placebo insuficiente y el hecho de haber
   observado 2025–2026 impiden usarlo como validación final. Su función vigente es incumbent y diagnóstico.

2. **La señal de ordenación (Rank-IC) es positiva dentro de 2015–2024.** Bootstrap (IC95 % sin cruzar cero) y
   LOYO estable aportan evidencia; el antiguo `p=0` no es válido y no permite descartar fuga por sí solo.

3. **Pero el valor económico es marginal y no generaliza.** El alfa medio es de +1 pp anual con un IR ≈ 0.02, la
   rentabilidad no supera al azar de forma convincente, y **la señal se degrada a partir de 2024**, cayendo a
   Rank-IC negativo en el estrés conocido 2025–2026. La configuración parsimoniosa gana precisamente porque la señal es
   débil: no hay margen para complejidad.

4. **¿Estable? No como estrategia; sí como pipeline.** El sistema **no es estable en el sentido de mantener su
   ventaja fuera de muestra** — se rompe justo en el tramo posterior. Lo que sí es estable y reproducible es
   el **proceso**: config robusta a semillas e hiperparámetros, resultados consistentes entre runs, y un veredicto
   que no depende de elecciones favorables. El resultado principal del TFM es, legítimamente, **parcialmente
   negativo**: se construyó un sistema honesto capaz de detectar su propia limitación.

---

## 12. Próximas mejoras y líneas futuras

Ordenadas por prioridad. Cualquier cambio de hipótesis, datos, etiquetas, modelos o cartera requiere instrucción
explícita del usuario (regla del repositorio).

1. **Diagnosticar la caída 2024–2026.** Es la pregunta central. Analizar con los artefactos ya generados
   (`rank_ic_diagnostics.parquet`, `meta_weights.parquet`, `agent_local_attribution.parquet`) si el deterioro es
   (a) un cambio de régimen general, (b) decaimiento específico de value/quality, o (c) fallo de un agente
   concreto. Esto no requiere reejecutar el study, solo leer los diagnósticos por cohorte.

2. **Reforzar la potencia de las pruebas de robustez.** El placebo usó solo 3 permutaciones y el random-portfolio
   comparte universo con el modelo. Subir el nº de permutaciones y usar un universo comparable (controlando el
   sesgo de supervivencia/alcista del periodo) daría un veredicto más firme sobre si el margen económico existe.

3. **Mejorar la traducción Rank-IC → alfa.** El modelo ordena mejor que el azar pero no lo capitaliza. Explorar
   sizing sensible a la convicción, y estrategias conscientes del coste (el frente de costes es frágil: el alfa
   desaparece con slippage de 20 bps).

4. **Reconsiderar features/regímenes que hoy se rechazan.** El barrido descartó régimen extendido,
   neutralización sectorial y ponderación por recencia por "no mejorar de forma estable" en 2015–2024. Es posible
   que alguna ayude específicamente en el tramo 2024+; convendría estudiarlas condicionadas al régimen, dejando
   constancia de que en el periodo histórico completo no mejoran.

5. **Ampliar universo y profundidad temporal de datos** si es viable, para disponer de más cohortes y un
   veredicto fuera de muestra menos ruidoso.

6. **Distinguir explícitamente selección, estrés conocido y operativa en la memoria.** La puesta en producción
   (entrenar hasta hoy y reformar la cartera) es un flujo distinto que solo debe activarse cuando el protocolo
   confirmatorio v2 aporte evidencia robusta; con esta configuración histórica todavía no ocurre.

---

*Fuentes: `results/studies/20260724--optimization-official--4dfea2986540/` (`decision.json`,
`comparison_data.parquet`, `study_manifest.json`) y `results/runs/20260724--35794c6ff34f/artifacts/`
(`annual_metrics.parquet`, `robustness.json`). Métrica y protocolo en `docs/doc.md §9, §11, §13, §14`.*
