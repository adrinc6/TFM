# Plan: estudios encadenados, optimización de cartera y migración del TFM

> Documento de trabajo. Fija (1) la estrategia de encadenar Model Studies usando el ganador de cada
> uno como baseline del siguiente, (2) el **Portfolio Study** que optimiza la cartera por
> Information Ratio sobre ese ganador, (3) cómo se mide y se cuenta que el proceso mejora, y (4) la
> migración completa del manuscrito LaTeX al resultado final.
>
> **Estado a 2026-08-14**: cadena de tres Model Studies **terminada**; Portfolio Study
> **implementado y pendiente de ejecutar la rejilla completa**; fases 1 y 2 del manuscrito
> **planificadas y sin ejecutar**.

## Resumen de decisiones tomadas

| Decisión | Resolución |
|---|---|
| ¿Cómo se mejora sobre el greedy secuencial? | Encadenar studies: el ganador de cada pasada es el baseline de la siguiente (ascenso por coordenadas) |
| ¿Qué optimiza la cartera? | Un **Portfolio Study** aparte, por **Information Ratio**, sobre el ganador ya congelado |
| ¿Cartesiano o greedy en la cartera? | **Cartesiano** de 6 variables (1.728 combos): interactúan entre sí y un greedy no lo vería |
| ¿Qué variables de cartera se optimizan? | `target_size`, `max_cash_weight`, `sizing_mode`, `minimum_holding_period`, `coverage_percentile_floor`, `rebalance_drift_tolerance` |
| ¿Y comisión y slippage? | **Nunca**: son supuestos de coste, no decisiones. Optimizarlos sería elegir el mundo que más conviene |
| ¿Entran los 8 perfiles en el cartesiano? | **No**: reordenan la señal, no gestionan la cartera. Se evalúan al final con la cartera ganadora, y ninguno se elige por su IR |
| ¿Puede la rejilla ver 2025-2026? | **No**: el backtest se corta en 2024 recortando los scores. Solo el ganador se reevalúa sobre la serie completa |
| ¿Qué se guarda de las 1.728 carteras? | Una fila de resumen de cada una; **evidencia completa solo del mejor vigente** |

---

## 1. La estrategia: por qué encadenar

La optimización de un Model Study es **greedy secuencial**, no cartesiana
(`docs/metodologia.md` §3): se parte de un baseline, se recorre cada variable predictiva en orden
fijo (temporal → representación → modelo → meta), se elige el mejor valor por Rank-IC robusto y se
congela antes de pasar a la siguiente. Eso tiene una consecuencia conocida: **el resultado depende
del punto de partida**, porque cada variable se evalúa sobre el incumbent acumulado hasta ese
momento, no sobre todas las combinaciones posibles.

Encadenar studies ataca exactamente esa limitación. Si el ganador del study *n* se usa como
baseline del study *n+1*, la segunda pasada explora el catálogo **desde un punto de partida mejor**
y puede descubrir combinaciones que la primera no podía ver: una variable que se decidió pronto,
cuando el resto de la configuración todavía era la recomendada por defecto, se vuelve a evaluar
ahora con todo lo demás ya optimizado.

Es una aproximación por **ascenso de coordenadas** (*coordinate ascent*): cada pasada completa es
una iteración, y la mejora entre pasadas es la evidencia de que el procedimiento converge.

### Por qué este study concreto lo pedía

El ganador del study 1 se aparta del baseline del catálogo en **8 de 21 variables predictivas**:

| Variable | Recomendado del catálogo | Ganador del study 1 |
|---|---|---|
| `snapshot_step_months` | 3 | **1** |
| `execution_lag_days` | 60 | **30** |
| `feature_preset` | core | **all** |
| `market_regime_feature` | True | **False** |
| `max_features_per_agent` | 8 | **20** |
| `lgbm_learning_rate` | 0,05 | **0,03** |
| `lgbm_min_child_samples` | 50 | **20** |
| `coverage_percentile_floor` | 0 | **60** (no predictiva, diagnóstica) |

Y **cuatro decisiones se resolvieron por `tie_simplicity`** —empate estadístico, gana la opción más
simple—: `execution_lag_days`, `market_regime_feature`, `lgbm_learning_rate` y
`lgbm_min_child_samples`. Un empate significa que la ventaja pareada no se distinguía del ruido
**con el incumbent de ese momento**. Reevaluadas desde un baseline mejor, esas cuatro son las
candidatas más probables a resolverse de otra forma. Es el argumento más fuerte para la segunda
pasada.

---

## 2. Cómo se lanza cada eslabón

El mecanismo ya existe, no hay que programar nada. En `module/studies/config.py`:

- `normalized_definition` (línea 41) admite por variable `{"mode", "values", "baseline"}`, y exige
  que el `baseline` **sea uno de los `values` seleccionados** (línea 70-72).
- `initial_values` (línea 173) siembra la primera evaluación (`predictive:baseline`) con esos
  baselines.
- `launch_study` en [module/web/api.py:56-79](../module/web/api.py#L56-L79) crea el study y el run
  baseline y arranca el worker.

Por tanto, **encadenar = lanzar el study siguiente marcando como `baseline` de cada variable el
valor ganador del study anterior**, manteniendo en `values` el abanico que se quiera reexplorar.

> Cada study crea siempre un `study_id` nuevo: no existe «reanudar». La cadena son tres directorios
> independientes en `results/studies/`, y eso es lo que permite compararlos.

### Configuración ganadora del study 1 (baseline del study 2)

`study-20260812-163136-1b104667`, run ganador `run-6eaa47a0597b`, catálogo v6:

```
snapshot_step_months: 1        target_horizon_months: 12     train_lookback_years: 8
execution_lag_days: 30         recency_weighting: off        objective: rank_regression
feature_preset: all            fundamental_momentum: True    market_regime_feature: False
neutralize_by_sector: False    winsorization: 0.0            max_features_per_agent: 20
feature_weighting_mode: oos_stability_prune                  model_family: lightgbm
lgbm_max_depth: 3              lgbm_n_estimators: 100        lgbm_learning_rate: 0.03
lgbm_min_child_samples: 20     meta_method: stacked_rolling_bounded
meta_history_quarters: 16      meta_recency_weighting: off
```

### La cadena ejecutada (trazabilidad)

| # | `study_id` | Run ganador | Cambio respecto a la pasada anterior |
|---|---|---|---|
| 1 | `study-20260812-163136-1b104667` | `run-6eaa47a0597b` | 8 variables frente al baseline del catálogo |
| 2 | `study-20260813-103456-aa733655` | `run-2dc586be8653` | `meta_method`: `stacked_rolling_bounded` → `stacked_rolling_free` |
| 3 | **`study-20260814-095144-5ec17b78`** | **`run-f134d7eb9e06`** | `execution_lag_days`: 30 → **60**; `target_size`: 12 → **8** |

**El study 3 es el de referencia del TFM.** Es el más optimizado de la cadena y del que salen todos
los resultados, conclusiones, figuras y tablas del manuscrito. Los studies 1 y 2 solo aparecen en la
tabla de progresión (§3) como evidencia de que el procedimiento converge.

> Nota: una versión anterior del study 3 (`…232458-05b4d236`) se descartó y se relanzó. Sus cifras
> no valen; las de este documento son del study vigente.

Configuración ganadora final (study 3), idéntica a la del study 1 salvo `meta_method`,
`execution_lag_days` y `target_size`:

```
snapshot_step_months: 1        target_horizon_months: 12     train_lookback_years: 8
execution_lag_days: 60         recency_weighting: off        objective: rank_regression
feature_preset: all            fundamental_momentum: True    market_regime_feature: False
neutralize_by_sector: False    winsorization: 0.0            max_features_per_agent: 20
feature_weighting_mode: oos_stability_prune                  model_family: lightgbm
lgbm_max_depth: 3              lgbm_n_estimators: 100        lgbm_learning_rate: 0.03
lgbm_min_child_samples: 20     meta_method: stacked_rolling_free
meta_history_quarters: 16      meta_recency_weighting: off   target_size: 8
```

**Es esta configuración la que alimenta el Portfolio Study** (§4): sus variables de cartera se
sustituyen por la combinación ganadora de la rejilla, y el resto se mantiene intacto.

> Detalle con valor metodológico para el TFM: la tercera pasada **revirtió** `execution_lag_days` de
> 30 a 60. Pero conviene no sobrevender el hallazgo: **ambas decisiones se tomaron por
> `tie_simplicity`**, no por evidencia. En el study 1 el retador 30 empató con el incumbent 60 y
> ganó por la tabla de simplicidad; en el study 3 ocurrió lo simétrico con los papeles invertidos
> (ventaja pareada de 30 sobre 60 de solo +0,00103, bajo la tolerancia de 0,002). La lectura
> correcta es que **la evidencia nunca distinguió entre 30 y 60 días**, y que el valor final lo fija
> una convención de desempate. Es un buen ejemplo de decisión frágil —el motivo de encadenar— pero
> el TFM no debe presentar el lag 60 como un resultado medido.

---

## 3. Cómo se demuestra que el proceso mejora

Esta es la parte que el TFM debe contar, y exige **comparar los tres studies con la misma métrica y
la misma ventana**. Reglas:

1. **La métrica de comparación es el Rank-IC robusto de la ventana de selección (2015-2024)**, que
   es el criterio con el que se eligió cada ganador. Es lo único comparable entre pasadas.
2. **2025-2026 no participa en ninguna comparación de selección.** Sigue siendo confirmación fuera
   de muestra en los tres studies (regla 3 de `CLAUDE.md`).
3. Las métricas económicas (IR, alfa, turnover) se reportan, pero **no** son el criterio: la
   cadena optimiza capacidad predictiva, no rentabilidad.

### Tabla de progresión (cadena completada el 2026-08-14)

Los tres studies corrieron bajo **catálogo v6**, así que son comparables entre sí.

| Métrica | Study 1 | Study 2 | Study 3 (final) |
|---|---|---|---|
| `study_id` | `…163136-1b104667` | `…103456-aa733655` | **`…095144-5ec17b78`** |
| Run ganador | `run-6eaa47a0597b` | `run-2dc586be8653` | **`run-f134d7eb9e06`** |
| **Rank-IC medio (selección)** | 0,1000 | 0,1074 | **0,1090** |
| **IC-IR** | 0,735 | 0,835 | **0,851** |
| $t$ de Newey-West | 2,95 | 3,36 | **3,46** |
| Cohortes positivas | 70,94 % | 74,36 % | 74,36 % |
| Rank-IC del meta (`meta_final`) | 0,0945 | 0,1047 | 0,1058 |
| Rank-IC `meta_equal_weight` | 0,0618 | 0,0618 | 0,0606 |
| Rank-IC del agente `risk` solo | 0,1172 | 0,1172 | **0,1197** |
| $p$ de permutación | 0,0001 | 0,0001 | 0,0001 |
| Decisiones por `tie_simplicity` | 4 de 19 | 1 de 17 | 2 de 17 |
| Variables que cambian vs. pasada anterior | — (8 vs. catálogo) | `meta_method` | `execution_lag_days`, `target_size` |
| — *Métricas económicas (no criterio de selección)* | | | |
| Coeficiente de transferencia | 0,178 | 0,234 | **0,328** |
| Information Ratio | 0,189 | 0,294 | **0,339** |
| Exceso geométrico | 1,07 % | 1,89 % | **2,61 %** |
| Beat rate | 50 % | 50 % | **70 %** |
| Turnover anualizado | 3,99 | 4,03 | **3,58** |
| Deflated Sharpe | 0,844 | 0,867 | **0,682** |
| — *Era reservada 2025-2026 (nunca decide)* | | | |
| Rank-IC | −0,0139 | +0,0529 | +0,0441 |
| Exceso geométrico | +8,49 % | +4,36 % | **−11,29 %** |
| Information Ratio | 0,898 | 0,476 | **−1,167** |
| Años que baten al S&P | 2/2 | 1/2 | **0/2** |

**Lectura.** La cadena funciona en la métrica que optimiza, y de forma monótona: Rank-IC
0,1000 → 0,1074 → 0,1090 (+9,0 % acumulado), IC-IR 0,735 → 0,851 y $t$ 2,95 → 3,46. Y esta vez
**también mejora la economía**: transferencia 0,178 → 0,328, IR 0,189 → 0,339, exceso 1,07 % →
2,61 %, con **menos** turnover (3,99 → 3,58). No hay divorcio entre señal y cartera.

La cadena además **converge**: una sola variable cambia en la 2.ª pasada y dos en la 3.ª, mientras
las demás se mantienen — señal de que el óptimo greedy es estable frente al punto de partida.

**Pero hay un resultado que domina a todos los demás y no se puede suavizar**: en la era reservada
el study 3 **pierde los dos años** (exceso −11,29 %, IR −1,167), con el mejor Rank-IC de la cadena y
Rank-IC positivo (+0,0441) en esa misma era. Ordena bien y aun así pierde dinero fuera de muestra.
Es exactamente lo que la ventana reservada existe para detectar, y pasa a ser el **hallazgo central
del TFM**. El Deflated Sharpe cae a 0,682, coherente con la multiplicidad que añade encadenar.

**Nota sobre el perfil ganador**: cambia entre pasadas (`defensive` en 1 y 2, `value` en la 3, con
`balanced` a 0,0016 de distancia — ruido). La quinta afirmación vertebradora del manuscrito («el
perfil `balanced` es el mejor») **no se sostiene** con esta cadena y hay que reescribirla.

**Nota sobre el agente `risk`**: por sí solo alcanza Rank-IC 0,1197, **por encima del meta-agente**
(0,1058). El TFM debe matizar la tesis multi-agente en vez de darla por demostrada.

> Una versión anterior del study 3 (descartada) mostró el patrón contrario: mejor Rank-IC con
> transferencia hundida (0,049) e IR 0,121. Se documenta aquí porque ilustra que **el divorcio
> entre ordenación y economía es posible** y que conviene reportar siempre ambas familias de
> métricas, no solo la de selección.

**Lectura honesta que hay que hacer, no dar por supuesta**: la cadena puede converger sin mejorar.
Si el study 2 devuelve el mismo ganador que el study 1, **eso también es un resultado publicable**
—significa que el óptimo greedy es estable frente al punto de partida— y debe contarse como tal, no
disimularse. Lo que no se puede hacer es afirmar que «mejora» sin que la tabla lo respalde.

### Riesgo metodológico que hay que declarar en el TFM

Encadenar pasadas **multiplica el número de configuraciones probadas** sobre los mismos datos, y
eso agrava el problema de selección múltiple. El Deflated Sharpe ya lo penaliza vía `n_trials` (74
en el study 1). Con tres pasadas, `n_trials` efectivo crece y el contraste se vuelve más exigente.
El capítulo de limitaciones debe decirlo explícitamente: **la ganancia de Rank-IC entre pasadas y
el riesgo de sobreajuste por multiplicidad crecen a la vez**, y la ventana reservada 2025-2026 es
la única defensa real, precisamente porque no participa en ninguna de las tres pasadas.

---

## 4. El Portfolio Study: optimizar la cartera por Information Ratio

### Por qué existe

El Model Study optimiza **Rank-IC**, que mide la calidad de la *ordenación*. Pero ordenar bien y
ganar dinero no son lo mismo: se observó un ganador con el mejor Rank-IC de su cadena y a la vez el
peor coeficiente de transferencia. Optimizar la ordenación no optimiza la cartera, así que la
cartera necesita su propio criterio.

Ese criterio es el **Information Ratio**: exceso medio sobre el índice ÷ volatilidad de ese exceso
(*tracking error*). A diferencia del alfa bruto, **premia la consistencia**: una cartera que bate al
índice un 2 % todos los años tiene un IR altísimo; otra que lo bate un 15 % y pierde un 11 % al
siguiente puede tener el mismo alfa medio y un IR pésimo.

### Las seis variables y cómo mueven el IR

| Variable | Efecto |
|---|---|
| `target_size` | El más directo. Pocas posiciones → cada acierto pesa mucho → exceso volátil → denominador grande. Muchas → te pareces al índice → numerador pequeño |
| `max_cash_weight` | El efectivo **amortigua**: baja la volatilidad del exceso y también el exceso (se remunera al 0 %). Sube el IR solo si evita malas compras |
| `sizing_mode` | `equal` reparte riesgo; `alpha_proportional` concentra en las de más alfa (tope 2:1): sube el numerador si acierta y el denominador siempre |
| `minimum_holding_period` | Retener reduce rotación y **costes**, que se restan del numerador. Pero retener de más impide corregir: la relación no es monótona |
| `coverage_percentile_floor` | Corta la cola: saca lo que se hunde en el ranking. Menos desastres, pero puede disparar ventas y su coste |
| `rebalance_drift_tolerance` | Pura fricción: más tolerancia, menos operaciones cosméticas; demasiada, y los pesos derivan |

### Por qué cartesiano y no greedy

**Estas variables interactúan.** El suelo de diversificación sale de `target_size` **y**
`max_cash_weight` a la vez; y lo que hace `coverage_percentile_floor` depende de si la plaza que
libera se recompra (tope 0) o queda en efectivo (tope > 0) — la misma variable hace cosas opuestas
según el tope. Un greedy fijaría la primera antes de mirar la segunda y no vería nada de eso.

Es asequible porque cada combinación **reutiliza los scores congelados** del ganador y solo rehace
el backtest: **~5-6 s**, frente a los ~146 s de un run predictivo con ajuste. Las 1.728
combinaciones son ~2,5 h en vez de 70.

### Qué NO se optimiza, y por qué

`commission_bps` y `slippage_bps` se fijan a **un único valor**: son *supuestos de coste*, no
decisiones de gestión, y optimizarlos equivaldría a elegir el mundo en el que la estrategia luce
mejor. Los umbrales en puntos básicos y las variables `price_only_*` gobiernan cuándo se opera bajo
información incompleta y se estresan aparte. La validación lo impone, no es solo una convención de
la interfaz.

### Los 8 perfiles quedan fuera de la rejilla

Un perfil **reordena la señal** (`apply_profile` sustituye el `meta_rank` y recalibra el alfa),
mientras las seis variables solo gestionan la cartera ya elegida. Son planos distintos. Incluirlos
multiplicaría por 8 (13.824 combinaciones, ~23 h) y, sobre todo, **elegiría el estilo de inversor
por su rentabilidad conocida**: `value` y `balanced` se separaron por 0,0016 en el study 3, que es
ruido.

En su lugar, al terminar la rejilla **la cartera ganadora se aplica a los perfiles seleccionados**
(todos por defecto), para responder «cómo le habría ido a cada estilo con la mejor gestión». Siguen
siendo diagnóstico informativo: ninguno se elige por su IR.

### Cómo se aísla la era reservada

Durante la rejilla el backtest **se corta en 2024**: los scores se recortan antes de simular
(`selection_evidence`), así que 2025-2026 **no llega a calcularse** para ninguna combinación. No
basta con filtrar el resumen al elegir —la cartera es secuencial, y si la simulación entrase en la
era reservada su resultado existiría y bastaría con mirarlo—. Solo la combinación **ya ganadora** se
reevalúa sobre la serie completa, y esa evidencia se guarda aparte (`evidence_best_full/`). La
cartera de partida contra la que se mide la mejora usa la **misma** serie recortada: compararla
sobre la completa mediría ventanas distintas y la mejora sería ficticia.

### Qué se guarda

Cada combinación deja una fila de resumen en `portfolio_grid.parquet`; la evidencia completa es
**solo la del mejor vigente** (`evidence_best/`), que se sustituye en cuanto otra la supera. Al
terminar queda exactamente una carpeta. Es la regla 5 del repositorio aplicada al IR.

La rejilla vuelca cada 25 combinaciones y al arrancar salta las ya evaluadas, de modo que una
interrupción cuesta minutos y no horas.

### Riesgo que hay que declarar

Probar 1.728 carteras sobre los mismos datos **añade multiplicidad**, igual que encadenar pasadas.
La defensa es que la elección solo ve 2015-2024 y que el resultado de la era reservada se reporta
**junto** al de selección, nunca en su lugar. El TFM debe dar las dos cifras siempre.

### Resultado (2026-08-14) · `study-20260814-135754-fdbdf2c5`

Cartera ganadora: `target_size=8`, `max_cash_weight=0.0`, `sizing_mode=alpha_proportional`,
`minimum_holding_period=half_horizon`, `coverage_percentile_floor=60`,
`rebalance_drift_tolerance=0.1`.

| Métrica | Cartera del modelo | Cartera ganadora | Era reservada |
|---|---|---|---|
| Information Ratio | 0,339 | **0,844** | **+0,304** |
| Exceso geométrico | 2,61 % | **6,97 %** | **+2,56 %** |
| Rotación anualizada | 3,58 | 3,24 | 3,91 |
| Años que baten | 70 % | 80 % | 50 % |

**El resultado invierte lo que se esperaba.** Con la cartera del catálogo, la era reservada daba
−11,29 % de exceso e IR −1,167 (0/2 años); con la optimizada, +2,56 % e IR +0,304 (1/2 años). El
Rank-IC de esa era es +0,0441 en ambos casos, porque no depende de la cartera. La señal siempre
había generalizado: lo que fallaba era la construcción de cartera.

Lecturas de la rejilla que merecieron entrar en el manuscrito:

- El IR **no** es función del turnover: hay carteras buenas y malas en todo el rango de rotación, y
  la ganadora **baja** la rotación (3,58 → 3,24) mientras sube el exceso.
- Las variables que mueven el IR son `target_size` y `max_cash_weight`; `rebalance_drift_tolerance`
  es prácticamente **inerte** (sus cuatro cajas son indistinguibles). El barrido de una variable
  cada vez no podía distinguir «este valor es mejor» de «esta variable da igual».
- Los perfiles con la cartera ganadora: `balanced` domina en selección (IR 0,844 vs 0,570 del
  segundo), pero en la era reservada el orden **se invierte** —`momentum`, el peor en selección, es
  el mejor allí con IR 1,889 sobre 6 cohortes—. Es régimen, no hallazgo, y así se reporta.

---

## 5. Fase 2: migración del manuscrito al study final

Se ejecuta **una sola vez**, sobre el **último** study de la cadena. Los studies 1 y 2 no se
documentan como referencia: solo aparecen en la sección de progresión (§3) como evidencia de que el
proceso converge.

### 5.1 Levantar la regla dura del plan del TFM

`latex/plan_tfm.md:24-35` congela el study viejo y exige que un cambio se registre explícitamente
allí y en la bitácora. Hay que **escribir esa decisión**, no saltársela: actualizar tabla de estado
(líneas 17-22), la regla dura y el aviso de reproducibilidad de catálogo v5 (los studies nuevos
corren bajo **v6**, así que ese aviso cambia de sentido: dejan de ser «no reproducibles con el
código actual»).

### 5.2 Regenerar activos y sustituir identificadores

```powershell
python latex/scripts/export_study_assets.py --study-id <STUDY_FINAL>
```

Regenera ~26 figuras y ~25 tablas en `latex/assets/` y reescribe `latex/asset_manifest.json`. **No
borra activos obsoletos**: los huérfanos del study anterior hay que eliminarlos a mano
(`verify_latex_assets.py` los detecta).

Identificadores a sustituir: `latex/main.tex:73` (macro `\studyid`, propaga a toda la prosa),
`latex/assets/a_reproducibilidad.tex:11-18,38` (run ganador, `dataset_hash`, `evaluation_key`,
`catalog_hash`, versión de catálogo, nº de configuraciones y cohortes, comando literal),
`latex/plan_tfm.md:17-35`, `latex/assets/08_desarrollo_cartera.tex:119`,
`latex/assets/t09_limitaciones.tex:38` y el docstring de
`latex/scripts/export_study_assets.py:4`.

### 5.3 Material nuevo (acordado)

El study persiste 10 artefactos que el manuscrito **no usa**. Cada bloque exige añadir funciones
`draw_*`/`table` al script de exportación —nunca dibujar a mano, es la convención de
`plan_tfm.md`— con prefijos `fNN_`/`tNN_`, e insertarlos en algún capítulo o el verificador los
marca como huérfanos.

| Contenido nuevo | Artefacto | Encaje |
|---|---|---|
| **Explicabilidad por acción**: por qué el sistema eligió cada valor, con casos concretos | `evidence/agent_local_attribution.parquet` (1,3 M filas: `feature`, `local_contribution`, `direction`, `importance_rank`) | **Sección o capítulo nuevo**. Hoy el TFM no menciona explicabilidad ni una vez |
| **Análisis real de la cartera**: qué acciones, permanencia, concentración | `evidence/positions.parquet` + `evidence/profiles/*/positions.parquet` | Amplía el capítulo de cartera |
| **Calibración de la señal**: alfa esperado vs. realizado | `evidence/signal_calibration.parquet` | Puente entre el capítulo predictivo y el económico |
| **Evolución temporal de coeficientes** | `evidence/model_feature_attribution.parquet` | Refuerza el capítulo de agentes |
| **Histogramas de distribución nula** | `robustness.json` (ya los persiste desde el 2026-08-12) | Sustituye 3 figuras de resumen; la limitación ya se retiró del anexo |
| **Progresión de la cadena** (§3) | `evidence/summary.json` de los tres studies | **Sección nueva** en el capítulo de diseño experimental o de resultados |

### 5.4 Reescritura de la prosa

Las tablas generadas se actualizan solas; **la prosa no**: ~175 cifras decimales escritas a mano.
Por densidad: `09_resultados.tex` (549 líneas, 82 cifras, reescritura completa),
`06_agentes_y_meta_agente.tex` (43), `07_diseno_experimental.tex` (20 — narra la escalera de
decisiones, que cambia con cada study), `00_resumen.tex` (8, se escribe al final), y cifras sueltas
en `10_limitaciones`, `11_conclusiones`, `01_introduccion`, `03_datos_y_universo`,
`05_desarrollo_metodo`, `08_desarrollo_cartera`.

Tablas escritas a mano que el script **no** regenera (excepción declarada en
`a_reproducibilidad.tex:47-49`): `t01_afirmaciones.tex`, `t08_defectos_validez.tex`,
`t08_versiones_catalogo.tex`, `t09_limitaciones.tex`.

Study-agnósticos, no tocar: `02_estado_del_arte.tex`, `04_diseno_metodologico.tex`,
`12_bibliografia.tex`, `b_catalogo_protocolo.tex`.

### 5.5 Revalidar las cinco afirmaciones vertebradoras

`plan_tfm.md` y `t01_afirmaciones.tex` sostienen el manuscrito sobre cinco afirmaciones. **Se
revalidan una a una contra el study final**; las que no se sostengan se reescriben o se sustituyen,
explicando el cambio en limitaciones y bitácora. Criterio acordado: **honestidad sobre continuidad
narrativa** (regla 9 de `CLAUDE.md`).

> Aviso, medido sobre el study 1: tres de las cinco ya no se sostenían tal cual. `defensive` batía a
> `balanced` (exceso 3,34 % vs. 1,07 %; IR 0,46 vs. 0,19), el agente `risk` solo (Rank-IC 0,117)
> superaba al meta-agente (0,094), y el contraste de carteras aleatorias no se superaba en el
> escenario general (percentil 61,4; sí en el emparejado por riesgo, 96,8). **Estos números son del
> study 1 y no deben copiarse al TFM**: hay que recalcularlos sobre el study final. Se anotan aquí
> solo como aviso de que la revalidación es real y probablemente obligue a reescribir afirmaciones.

### 5.6 Verificación

```powershell
python latex/scripts/verify_latex_assets.py   # rutas, UTF-8, refs cruzadas, activos huérfanos
python -m pytest -q
python -m ruff check .
```

Después, `grep` en `latex/` del `study_id`, run ganador y `dataset_hash` de **todos** los studies
anteriores de la cadena, para confirmar que solo queda el final.

### 5.7 Documentación

`docs/informe_resultados.md:20` (tabla de cabecera y cifras), `docs/gestion_cartera.md` §6
(turnover atribuido al ganador), y entrada en `docs/bitacora.md` registrando el cambio de study de
referencia, la estrategia de encadenado y qué afirmaciones cambiaron. Las menciones históricas en
entradas antiguas de la bitácora **se conservan**: son registro histórico.

---

## 6. Orden de ejecución

1. ~~Lanzar study 1~~ — hecho (`…1b104667`).
2. ~~Lanzar study 2 con el ganador del 1 como baseline~~ — hecho (`…aa733655`).
3. ~~Lanzar study 3 con el ganador del 2 como baseline~~ — hecho (`…095144-5ec17b78`).
4. ~~Rellenar la tabla de progresión (§3)~~ — hecha, cadena completada el 2026-08-14.
5. ~~Implementar el Portfolio Study (§4)~~ — hecho: motor, validación, API, worker, interfaz,
   reanudación incremental, aislamiento de la era reservada y evaluación por perfiles.
6. **Ejecutar la rejilla completa del Portfolio Study** (1.728 combinaciones, ~2,5 h) ← siguiente,
   lo lanza el usuario.
7. Ejecutar la **fase 1** del manuscrito: documentar la cadena y el Portfolio Study, con sus tablas
   y figuras nuevas.
8. Ejecutar la **fase 2** (§5): todos los resultados sobre el ganador del study 3 con la cartera
   ganadora de la rejilla.

No se lanza ningún study desde el asistente: los lanza el usuario (regla del repo — no ejecutar un
estudio real completo sin autorización explícita).

## 7. Aviso: catálogo v7 y comparabilidad

Los tres studies de la cadena corrieron bajo **catálogo v6**. Después de completarla se invirtió la
tabla de `simplicity` de `execution_lag_days` a `(60, 45, 30)` y se subió a **`CATALOG_VERSION = 7`**
(ver bitácora del 2026-08-14): en un empate técnico ahora gana el lag mayor, porque un lag menor no
es una hipótesis más simple sino más fuerte sobre la disponibilidad del dato.

**El study 3 eligió lag 60 por `tie_simplicity`, no por evidencia pareada.** Sus candidatos fueron
60 (incumbent, Rank-IC 0,1005) y 30 (ventaja pareada +0,00103, por debajo de `TIE_TOLERANCE` =
0,002): empate técnico. Bajo la tabla v6 —donde 30 era «el más simple»— el desempate habría elegido
**30**; el ganador es 60 porque el incumbent se mantiene cuando el retador no supera la tolerancia.

Es decir: **el lag ganador del study 3 depende de una regla de desempate, no de la evidencia**, y
justo esa regla se cambió después. Con el catálogo v7 el resultado sería el mismo (60 gana en ambos
casos, por vías distintas), pero la conclusión honesta es que **la evidencia no distingue entre 30 y
60 días**, y el manuscrito debe decirlo así en vez de presentar el lag 60 como un hallazgo. El
capítulo de diseño experimental debe declarar la versión de catálogo junto al `study_id` y explicar
que la cadena es v6 mientras el código vigente es v7.
