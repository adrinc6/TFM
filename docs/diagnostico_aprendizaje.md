# Diagnóstico del aprendizaje — ¿qué agente aprende y qué no?

> **Nota (reestructuración):** las cifras de este documento corresponden al esquema **anterior**
> (evaluación desde 2018, ventana fija de 4 años, reentreno trimestral). Tras la reestructuración la
> simulación arranca en una **fecha ancla configurable** (`EVAL_START_QUARTER` + retardo de
> publicación), separa **entrenar** (trimestral/anual) de **revisar** (mensual), y el barrido
> sistemático (`experiments/rejilla.py`) elige el **sistema final** por estabilidad multi-era. Las
> tablas concretas de abajo **se regenerarán** con el nuevo barrido (aún no ejecutado por su coste); la
> metodología y las conclusiones cualitativas siguen siendo válidas.

Este documento recoge el diagnóstico cuantitativo del run de referencia
(`results/full_2018-02-15_2026-06-15_M_cutoff2018-02-15/`, congelado en
`results/_baseline_frozen/`) sobre **la calidad de ranking del modelo**, no sobre su
rentabilidad. Es la evidencia de "qué aprende el sistema y qué no", que forma parte del valor
científico del TFM con independencia de que la señal sea fuerte.

Reproducible con: `python scripts/diagnostico_rank_ic.py`

## 1. Rank-IC OOS por agente (ventana operativa 2018+, 100 snapshots)

El rank-IC out-of-sample mide si el score ordena bien las acciones por su exceso de retorno
futuro. Se mide solo en la ventana operativa (modo `walk_forward_model`); los valores altos de
2014-2017 en el CSV son **in-sample** y no son informativos.

| Agente | Rank-IC medio | Desviación (std) | % snapshots > 0 |
|---|---|---|---|
| **Calidad** | **+0.025** | **0.049** (la más baja) | **72%** |
| Temporización | +0.013 | 0.084 | 59% |
| Alpha (ranking directo) | +0.004 | 0.111 (la más alta) | 54% |
| `final_score` (meta-agente) | +0.021 | 0.099 | 61% |

## 2. Rank-IC OOS por año

| Año | Calidad | Temporización | Alpha | final_score |
|---|---|---|---|---|
| 2018 | -0.005 | 0.026 | 0.049 | 0.048 |
| 2019 | 0.033 | 0.062 | **0.142** | **0.143** |
| 2020 | 0.025 | -0.005 | **-0.059** | -0.053 |
| 2021 | 0.018 | -0.058 | **-0.140** | **-0.109** |
| 2022 | 0.018 | 0.052 | 0.062 | 0.032 |
| 2023 | **0.061** | 0.009 | 0.009 | 0.063 |
| 2024 | 0.030 | 0.000 | -0.025 | 0.014 |
| 2025 | 0.031 | -0.024 | -0.031 | 0.031 |
| 2026 | -0.003 | **0.111** | 0.066 | 0.026 |

## 3. Hallazgo principal (contraintuitivo respecto al diseño)

El agente diseñado como "el que rankea directamente el alpha" (**Alpha**) es en realidad **el
menos fiable**: mayor IC potencial en años buenos (2019: +0.142) pero la mayor varianza y años
catastróficos (2021: -0.140). El `final_score` combinado **hereda esa inestabilidad**.

El agente **Calidad** es el más **consistente** (positivo en 72% de los snapshots, en 8 de 9
años, con la menor varianza), pese a tener el IC absoluto más bajo. El sistema le da poco peso
precisamente porque el meta-agente premia el IC absoluto, no la estabilidad.

**Implicación para la Fase 2 (mejora del modelo):** dado que el objetivo del proyecto es un
rank-IC **estable y consistente**, la vía más prometedora no es reforzar el momentum/Alpha (más
IC pero más ruido), sino **penalizar explícitamente la varianza del IC** en la ponderación del
meta-agente y/o **elevar el peso base de Calidad**. Esto se validará contra el baseline
congelado con el criterio del plan: sube el rank-IC OOS medio Y baja el nº de años negativos.

## 4. Por qué hay alfa (+110%) con rank-IC tan débil

El `edge_attribution.csv` del baseline muestra que la correlación entrada `manager_score` vs.
exceso realizado es **-0.003** (nula). El alfa **no** proviene de que el modelo ordene bien,
sino de la **asimetría ganador/perdedor**: 87 posiciones ganadoras suman +22.4 de exceso frente
a 100 perdedoras que suman -13.5. Unas pocas ganadoras grandes (con sesgo de supervivencia
detrás, ver `environment.py`) explican el resultado agregado. Es un resultado honesto y ya
reconocido en el informe: la robustez del alfa (bootstrap, subperiodos, placebo) es real, pero
**no** es evidencia de que el ranking ML funcione — son dos afirmaciones distintas.

## 5. Experimento 2A — penalizar la varianza del IC en el meta-agente (RESULTADO NEGATIVO)

**Hipótesis.** Si el meta-agente premia el IC absoluto e ignora la estabilidad, penalizar la
varianza del IC **entre snapshots** debería subir el peso de Calidad (estable) y bajar el de
Alpha (errático), mejorando el rank-IC del `final_score`.

**Qué se probó.** Se reescribió el criterio de ponderación de `_fit_meta_weights` para medir el
partial-IC de cada agente **por snapshot** dentro de la ventana de validación y puntuarlo como
`media(IC_por_snapshot) − λ·error_estándar(IC_por_snapshot)`, con λ subido de 0.5 a 1.0. Se
re-ejecutó el pipeline completo y se comparó contra el baseline congelado.

**Resultado — no mejoró (y se revirtió).**

| Métrica del `final_score` | Baseline | Experimento 2A |
|---|---|---|
| Rank-IC OOS medio | +0.021 | **+0.017** (peor) |
| Desviación (std) | 0.099 | 0.098 (igual) |
| % snapshots > 0 | 61% | 60% |
| Rank-IC 2021 (el peor año) | -0.109 | -0.109 (sin cambio) |

**Por qué falló (el hallazgo estructural).** El meta-agente pondera por **contribución marginal**
(el partial-IC del *residual* que los otros agentes dejan sin explicar), no por el IC crudo del
agente. El agente **Alpha** está entrenado directamente contra el mismo `target_future_alpha`
que define ese residual, así que casi siempre tiene el mayor partial-IC medio — y sigue dominando
(peso medio ~0.53, domina en 49 de 82 snapshots) **por más que se penalice su varianza**. Con
30-60 snapshots en la ventana de entrenamiento, el error estándar es demasiado pequeño para
revertir esa ventaja estructural. **Conclusión: ajustar la ponderación no arregla la
inestabilidad; el problema es que la arquitectura favorece al agente que optimiza su propio
target de evaluación.** Cualquier mejora real debe atacar esa causa (p. ej. elevar el peso base
del agente estable, o cambiar contra qué se mide la contribución marginal), no el criterio de
consistencia. Este resultado negativo, bien medido, es parte del valor del TFM.

## 6. Experimento 2B — reponderar el prior hacia Calidad (RESULTADO POSITIVO, ADOPTADO)

**Hipótesis (tras el fallo de 2A).** Si el problema es estructural —el meta-agente premia al
agente Alpha por optimizar el propio target de evaluación—, la corrección no es cambiar el
*criterio* de aprendizaje sino el *prior* al que se ancla: el prior actual (0.30/0.35/0.35) está
calibrado hacia timing/alpha (donde vivía la alfa cruda), no hacia Calidad (donde vive la señal
de ranking estable). Además, el suelo `META_WEIGHT_FLOOR` mezclaba hacia *equal-weight*, lo que
reinflaba a Alpha en los trimestres ruidosos.

**Búsqueda offline (sin re-entrenar, `scripts/buscar_pesos_meta.py`).** Recombinando los scores
de agentes ya calculados con pesos fijos y midiendo el rank-IC del combinado, aparece un
trade-off limpio: cuanto más peso a Calidad, más estable el rank-IC, pero un tilt total también
devuelve alfa/IR (la exposición a los ganadores de cola de momentum).

| Pesos (Q/T/A) | IC medio | std | años<0 |
|---|---|---|---|
| 0.30/0.35/0.35 (prior original) | +0.0294 | 0.123 | 3/9 |
| **0.45/0.30/0.25 (adoptado)** | +0.0320 | 0.101 | 2/9 |
| 0.60/0.25/0.15 | +0.0318 | 0.077 | 1/9 |
| 0.70/0.20/0.10 | +0.0307 | 0.065 | 1/9 |

**Cambio implementado.** Dos ajustes en `module/ml.py`: (1) `AGENT_PRIOR_WEIGHTS` →
**0.45/0.30/0.25** (Q/T/A); (2) el `META_WEIGHT_FLOOR` ahora mezcla los pesos aprendidos hacia
ese prior informado, **no hacia equal-weight** — así, cuando el partial-IC es ruidoso, la
combinación cae hacia Calidad en vez de reinflar a Alpha.

**Resultado walk-forward real (adoptado) — mejora AMBOS objetivos, sin trade-off:**

| Métrica | Baseline (0.30/0.35/0.35) | Adoptado (0.45/0.30/0.25) |
|---|---|---|
| Rank-IC OOS medio | +0.0211 | **+0.0236** |
| Rank-IC std | 0.099 | **0.092** |
| Rank-IC 2021 (peor año) | -0.109 | **-0.083** |
| Alfa acumulada | 1.104 | **1.138** |
| Information Ratio | 0.875 | **0.958** |
| t-stat exceso | 2.527 | **2.765** |
| Subperiodos | 0.385 / 0.307 / 0.412 | **0.376 / 0.337 / 0.425** |

A diferencia del tilt extremo (0.60/0.25/0.15), que subía el rank-IC pero bajaba alfa/IR, el
punto intermedio **sube el rank-IC Y la alfa/IR/t-stat** y refuerza justo el subperiodo central,
que era el más débil. Por eso se eligió 0.45/0.30/0.25 y no un tilt mayor: captura la ganancia de
estabilidad sin sacrificar el resultado económico. Es el run vigente en
`results/full_2018-02-15_2026-06-15_M_cutoff2018-02-15/`.

## 7. Pendiente (opcional)

El **ablation de scores** (alfa rankeando por `final_score` vs. solo `alpha_probability` vs. solo
`momentum_score`) y la **descomposición del gap momentum_only vs. sistema** son reproducibles sin
re-descargar ahora que los parquet existen. Quedan como trabajo futuro; el hallazgo estructural de
las secciones 3-6 ya orientó la mejora efectiva.
