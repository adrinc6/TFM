# Plan: estudios encadenados y migración del TFM al study final

> Documento de trabajo. Fija (1) la estrategia de encadenar Model Studies usando el ganador de cada
> uno como baseline del siguiente, (2) cómo se mide y se cuenta que el proceso mejora, y (3) la
> migración completa del manuscrito LaTeX al último study de la cadena.
>
> Estado a 2026-08-13: **study 1 terminado**, studies 2 y 3 **pendientes de lanzar por el usuario**.

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
| 3 | **`study-20260813-232458-05b4d236`** | **`run-d304f6074665`** | `execution_lag_days`: 30 → **60** |

**El study 3 es el de referencia del TFM.** Es el más optimizado de la cadena y del que salen todos
los resultados, conclusiones, figuras y tablas del manuscrito. Los studies 1 y 2 solo aparecen en la
tabla de progresión (§3) como evidencia de que el procedimiento converge.

Configuración ganadora final (study 3), idéntica a la del study 1 salvo `meta_method` y
`execution_lag_days`:

```
snapshot_step_months: 1        target_horizon_months: 12     train_lookback_years: 8
execution_lag_days: 60         recency_weighting: off        objective: rank_regression
feature_preset: all            fundamental_momentum: True    market_regime_feature: False
neutralize_by_sector: False    winsorization: 0.0            max_features_per_agent: 20
feature_weighting_mode: oos_stability_prune                  model_family: lightgbm
lgbm_max_depth: 3              lgbm_n_estimators: 100        lgbm_learning_rate: 0.03
lgbm_min_child_samples: 20     meta_method: stacked_rolling_free
meta_history_quarters: 16      meta_recency_weighting: off
```

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
| `study_id` | `…163136-1b104667` | `…103456-aa733655` | **`…232458-05b4d236`** |
| Run ganador | `run-6eaa47a0597b` | `run-2dc586be8653` | **`run-d304f6074665`** |
| **Rank-IC medio (selección)** | 0,1000 | 0,1074 | **0,1090** |
| **IC-IR** | 0,735 | 0,835 | **0,851** |
| Cohortes positivas | 70,94 % | 74,36 % | 74,36 % |
| Rank-IC del meta (`meta_final`) | 0,0945 | 0,1047 | 0,1058 |
| Rank-IC `meta_equal_weight` | 0,0618 | 0,0618 | 0,0606 |
| Rank-IC del agente `risk` solo | 0,1172 | 0,1172 | 0,1197 |
| Diferencial de cola | 0,0348 | 0,0374 | 0,0360 |
| $p$ de permutación | 0,0001 | 0,0001 | 0,0001 |
| Decisiones por `tie_simplicity` | 4 de 19 | 1 | 2 |
| Variables que cambian vs. pasada anterior | — (8 vs. catálogo) | `meta_method` | `execution_lag_days` |
| — *Métricas económicas (no criterio)* | | | |
| Coeficiente de transferencia | 0,178 | 0,234 | **0,049** |
| Information Ratio | 0,189 | 0,294 | **0,121** |
| Exceso geométrico | 1,07 % | 1,89 % | **0,39 %** |
| Deflated Sharpe | 0,844 | 0,867 | **0,584** |
| Rank-IC confirmación 2025-26 | −0,0139 | +0,0529 | +0,0441 |

**Lectura.** En la métrica de selección la cadena funciona y de forma monótona: Rank-IC
0,1000 → 0,1074 → 0,1090 (+9,0 % acumulado) e IC-IR 0,735 → 0,851. Además la cadena **converge**:
solo cambia una variable por pasada (`meta_method` de `stacked_rolling_bounded` a
`stacked_rolling_free` en la 2.ª; `execution_lag_days` de 30 a 60 en la 3.ª) y las 19 restantes se
mantienen, que es la señal de que el óptimo greedy es estable frente al punto de partida.

**Pero hay que decirlo entero, y es lo más importante de esta tabla**: la mejora predictiva **no se
traduce en economía**. El coeficiente de transferencia se hunde de 0,234 a 0,049 en la última
pasada y el IR cae de 0,294 a 0,121, con el mejor Rank-IC de las tres. Es decir, la tercera pasada
ordena mejor el universo **y a la vez** convierte peor esa ordenación en rentabilidad. También el
Deflated Sharpe empeora (0,584), lo que era esperable: al encadenar pasadas crece el número
efectivo de configuraciones probadas y el contraste por multiplicidad se vuelve más exigente.

Este divorcio entre Rank-IC y alfa es un **resultado del trabajo, no un defecto que ocultar**: es
evidencia directa de que optimizar la ordenación transversal no garantiza una cartera mejor, y de
que el cuello de botella está en la traducción señal → cartera. El TFM debe contarlo así.

**Nota sobre el perfil ganador**: cambia entre pasadas (`defensive` en 1 y 2, `value` en la 3),
y `balanced` no gana en ninguna. La quinta afirmación vertebradora del manuscrito («el perfil
`balanced` es el mejor») **es falsa** con esta cadena y hay que reescribirla.

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

## 4. Fase 2: migración del manuscrito al study final

Se ejecuta **una sola vez**, sobre el **último** study de la cadena. Los studies 1 y 2 no se
documentan como referencia: solo aparecen en la sección de progresión (§3) como evidencia de que el
proceso converge.

### 4.1 Levantar la regla dura del plan del TFM

`latex/plan_tfm.md:24-35` congela el study viejo y exige que un cambio se registre explícitamente
allí y en la bitácora. Hay que **escribir esa decisión**, no saltársela: actualizar tabla de estado
(líneas 17-22), la regla dura y el aviso de reproducibilidad de catálogo v5 (los studies nuevos
corren bajo **v6**, así que ese aviso cambia de sentido: dejan de ser «no reproducibles con el
código actual»).

### 4.2 Regenerar activos y sustituir identificadores

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

### 4.3 Material nuevo (acordado)

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

### 4.4 Reescritura de la prosa

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

### 4.5 Revalidar las cinco afirmaciones vertebradoras

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

### 4.6 Verificación

```powershell
python latex/scripts/verify_latex_assets.py   # rutas, UTF-8, refs cruzadas, activos huérfanos
python -m pytest -q
python -m ruff check .
```

Después, `grep` en `latex/` del `study_id`, run ganador y `dataset_hash` de **todos** los studies
anteriores de la cadena, para confirmar que solo queda el final.

### 4.7 Documentación

`docs/informe_resultados.md:20` (tabla de cabecera y cifras), `docs/gestion_cartera.md` §6
(turnover atribuido al ganador), y entrada en `docs/bitacora.md` registrando el cambio de study de
referencia, la estrategia de encadenado y qué afirmaciones cambiaron. Las menciones históricas en
entradas antiguas de la bitácora **se conservan**: son registro histórico.

---

## 5. Orden de ejecución

1. ~~Lanzar study 1~~ — hecho (`…1b104667`).
2. ~~Lanzar study 2 con el ganador del 1 como baseline~~ — hecho (`…aa733655`).
3. ~~Lanzar study 3 con el ganador del 2 como baseline~~ — hecho (`…05b4d236`).
4. ~~Rellenar la tabla de progresión (§3)~~ — hecha, cadena completada el 2026-08-14.
5. **Ejecutar la fase 2 completa (§4) sobre el study 3** ← siguiente.

No se lanza ningún study desde el asistente: los lanza el usuario (regla del repo — no ejecutar un
estudio real completo sin autorización explícita).

## 6. Aviso: catálogo v7 y comparabilidad

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
