# Plan: estructura y redacción de la memoria del TFM en LaTeX

## Context

El proyecto está terminado a nivel de código y experimentos (19 escenarios ejecutados en
`results/escenarios/20260716_105304/`), pero la memoria académica no existe. Hay que escribirla en
LaTeX, un `.tex` por capítulo, para pegar en Overleaf.

El trabajo se hará en dos tiempos: **primero** se crea la carpeta `latex/` con el plan maestro en
markdown (este documento se copia allí como `latex/PLAN.md`) y el esqueleto compilable; **después**,
el usuario irá pidiendo capítulos concretos, que se escribirán apoyándose en el plan, en los
capítulos ya escritos y en el estado real del código y los resultados.

Decisiones tomadas con el usuario:

1. **Narrativa**: "IA evaluada con rigor". La pregunta es *¿aprende el sistema?*; la bolsa es el
   banco de pruebas. El núcleo es la **disociación** entre alfa robusta y aprendizaje débil, y los
   resultados negativos son entregable, no vergüenza. Alineado con `CLAUDE.md`.
2. **Discrepancias doc↔código**: el LaTeX documenta **lo que el código ejecuta**. Además se
   corregirán `docs/doc.md` y `README.md` (aprobado explícitamente por el usuario).
3. **Rank-IC**: investigado durante la planificación. **Es un bug confirmado** (ver abajo).
4. **Formato**: `main.tex` + `chapters/` + `bibliography.bib`.

---

## Hallazgo bloqueante confirmado durante la planificación

`module/experiments/metrics.py:60` calcula `rank_ic_final_mean` como `ic.mean()` sobre **todas** las
filas de `model_walk_forward_diagnostics.csv`, **sin filtrar por fecha**.

El CSV tiene 148 filas desde 2014-02-28. Las **48 filas pre-2018** tienen `mode = fallback_garp`: no
hay modelo entrenado todavía (el walk-forward arranca en `TRAIN_CUTOFF_DATE = 2018-02-15`), así que
`final_score` cae al `garp_score` determinista y se correlaciona **consigo mismo** → rank-IC **0.615**.
Es el artefacto degenerado que `environment.py` y `ml.py:171-176` ya advertían en otro contexto.

| Métrica (baseline) | Valor |
|---|---|
| `rank_ic_final` media global (lo que se publica) | **0.2155** |
| media pre-2018 (48 filas, `fallback_garp`) | **0.6152** |
| media 2018+ (100 filas, `walk_forward_model`) | **0.0236** |

El **0.0236 OOS coincide exactamente** con el `+0.0236` que `docs/diagnostico_aprendizaje.md` reporta
para el prior adoptado. **El diagnóstico siempre estuvo bien; `collect_metrics` está mal.** La
"contradicción entre documentos" no existía.

**Impacto en las conclusiones (lo grave):** correlación de Spearman entre el ranking publicado y el
ranking OOS real = **-0.092**. No es un ranking ruidoso: es un ranking *distinto*. Se invierten los
dos casos que sostienen el capítulo de ablaciones:

| Escenario | IC publicado | IC OOS real (2018+) | Δ vs baseline |
|---|---|---|---|
| `solo_alpha` | 0.3321 (**nº1**) | **0.0115** (**nº19, el peor**) | -0.0121 |
| `solo_calidad` | 0.0447 (**el peor**) | **0.0255** (**nº3**) | +0.0019 |
| `horizonte_6m` | 0.2706 | **0.0406** (**nº1**) | +0.0170 |
| `sin_meta_aprendido` | 0.2170 | 0.0259 (nº2) | +0.0023 |
| `baseline` | 0.2155 | 0.0236 | 0 |

Con las cifras limpias, el orden **Calidad > Alpha** reproduce `diagnostico_aprendizaje.md`. Y
**`sin_meta_aprendido` sigue batiendo al baseline** (IC +0.0023, alfa 1.438 vs 1.138, IR 1.004 vs
0.958): ese resultado negativo es real y **sobrevive** a la corrección.

**Consecuencia para el plan**: el capítulo de experimentos **no se puede escribir** hasta arreglar
esto. Es la tarea 0. → **✅ Resuelto, ver Fase 0. Los capítulos 6 y 7 están desbloqueados.**

---

## Estado del repo: trabajo concurrente de otro agente (verificado)

Hay **11 ficheros modificados sin commitear** por otro agente. Verificado: **no arregló el bug del
rank-IC**. Su trabajo es bueno y **complementario**; se construye encima, no se revierte.

Lo que hizo:
- **`ml.py` + `metrics.py`**: añade métricas de **breadth** `top_n_alpha` / `top_n_alpha_lift`. El
  rank-IC ordena las ~71k filas del universo, pero la cartera solo compra el top-10: el breadth mide
  el tramo que realmente se ejecuta. **Es el puente honesto entre "la IA rankea" y "la cartera gana"**
  → material de primera para el Cap. 6, que lo debe incorporar.
- **`escenarios_estabilidad.py`**: elimina los escenarios de semilla. **Resuelve el riesgo nº3 de este
  plan**: `LGBM_PARAM_GRID` no fija `subsample` ni `colsample_bytree`, así que LightGBM corre sin
  componente estocástico y `RANDOM_STATE` no cambia nada — de ahí los deltas 0.000. Lo documenta como
  limitación en vez de presentarlo como robustez. Decisión correcta; el Cap. 7 la adopta.
- Añade `ventana_larga`, `horizonte_24m`, `costes_extremos`; nuevo `tests/test_breadth_diagnostics.py`.

**Ojo**: su docstring en `metrics.py` afirma que la correlación Spearman IC↔alfa entre escenarios es
~+0.28. Está calculada **sobre las cifras contaminadas**. Tras la Fase 0 hay que recalcularla y
corregir ese comentario.

**Los escenarios nuevos NO están en el barrido ya ejecutado** (`20260716_105304`, que aún tiene las
semillas). Decisión tomada: el Cap. 7 se escribe con los 19 escenarios en disco recalculados post-hoc,
y las tablas se actualizan cuando se lance un barrido nuevo.

---

## Fase 0 — Arreglar el rank-IC ✅ COMPLETADA (2026-07-16)

Resultó ser **dos** bugs, no uno. El segundo solo apareció al arreglar el primero.

### Bug 1 — `metrics.py` agregaba los snapshots sin modelo
`_learning_metrics` promediaba las 148 filas del diagnóstico. **Arreglado** filtrando por
`mode == "walk_forward_model"` (no por fecha: no fija el cutoff en el código y excluye también el
modo `full_sample` de `ml.py:196`). Si tras filtrar no queda nada, devuelve `nan` en vez de un
aprendizaje falso.

### Bug 2 — `ml.py` contaminaba las medias anuales en origen
`_master_signal_diagnostics` (`ml.py:552`) agrupaba por año **sin filtrar por `mode`**, así que las
columnas `rank_ic_final_year_*` ya venían mal escritas y filtrar en `metrics.py` no bastaba. **2018 es
un año mixto**: 1 snapshot de fallback + 11 con modelo, y ese único snapshot **duplicaba** la media
del año (0.0981 publicado vs. **0.0482** real). **Arreglado** enmascarando el IC de los snapshots sin
modelo antes del rolling y del `groupby(year)`. El `rank_ic_final` por snapshot se conserva sin
enmascarar: es auditable y `mode` dice de dónde sale cada fila.

Los dos modos no se solapan siquiera en rango — `fallback_garp` ∈ [0.4987, 0.7116] frente a
`walk_forward_model` ∈ [-0.1695, 0.2576] — lo que confirma que el fallback es un artefacto puro y no
señal degradada.

### Tests añadidos (5, todos pasan; suite completa: 65 ✅)
- `tests/test_breadth_diagnostics.py`: `test_media_anual_ignora_los_snapshots_en_fallback`,
  `test_sin_columna_mode_agrega_todo` (degradación elegante para diagnósticos antiguos).
- `tests/test_experiments.py`: `test_rank_ic_solo_agrega_snapshots_con_modelo_entrenado`,
  `test_rank_ic_sin_snapshots_con_modelo_devuelve_nan`.

### Recálculo post-hoc — hecho
`results/escenarios/20260716_105304/comparison_oos.csv` (el `comparison.csv` original **se conserva
intacto**: documenta lo que se publicó con el bug). Confirmado todo lo previsto:

| Escenario | IC publicado | **IC OOS limpio** | Δ vs baseline | Alfa |
|---|---|---|---|---|
| `horizonte_6m` | 0.2706 | **0.0406** (nº1) | +0.0171 | 1.237 |
| `sin_meta_aprendido` | 0.2170 | **0.0259** (nº2) | +0.0023 | **1.438** |
| `solo_calidad` | 0.0447 (último) | **0.0255** (nº3) | +0.0019 | 0.833 |
| `baseline` | 0.2155 | **0.0236** | 0 | 1.138 |
| `solo_alpha` | **0.3321** (nº1) | **0.0115** (**último**) | -0.0121 | 0.796 |

- El baseline da **0.0236**, idéntico al `+0.0236` de `diagnostico_aprendizaje.md`. **El diagnóstico
  siempre estuvo bien.**
- `rank_ic_positive_years`: de 11-12 años (contando 2014-2017) a **7 de 9** — la ventana operativa real.
- Orden **Calidad > Alpha** restaurado, reproduciendo el diagnóstico.
- Correlación Spearman entre ranking publicado y limpio: **-0.0922**.
- **Corregido de paso**: el otro agente documentó la correlación IC↔alfa como ~+0.28, calculada sobre
  cifras contaminadas. Con IC limpio es **+0.36**. Actualizado en `ml.py` y `metrics.py`.

**Nota honesta para la memoria**: estos bugs son material del capítulo de metodología. Un artefacto de
evaluación que invierte el ranking de escenarios, detectado porque dos documentos internos no
cuadraban, es un ejemplo real de por qué el TFM insiste en separar "medir" de "medir bien". Y el bug 2
es aún mejor ejemplo: **un solo snapshot de 148 bastaba para duplicar la métrica de un año**. Merece
un párrafo, no un silencio.

---

## Fase 1 — Crear `latex/` ✅ COMPLETADA

**No se crean los `.tex` de capítulo vacíos** — el guion de cada capítulo ya vive en este plan.
Los `.tex` se crean uno a uno según se pidan.

```
latex/
├── PLAN.md              ← este plan (fuente de verdad de la estructura)
├── main.tex             ← preámbulo + \include comentados; compila ya en Overleaf
├── bibliography.bib     ← vacío (ver Riesgos)
├── chapters/            ← se puebla a demanda
└── figuras/             ← PNG copiados de results/<escenario>/viewer/charts/
```

`main.tex`: clase `report`, `babel` español, `booktabs`, `graphicx`, `amsmath`, `siunitx`,
`hyperref`, `listings` (con `literate` para acentos en los listados de código), `natbib`. Portada
provisional, a sustituir por la oficial de la universidad. Los `\include{}` de capítulos aún no
escritos están **comentados**: se descomentan según se redactan.

---

## Fase 2 — Estructura de la memoria (8 capítulos)

La numeración es el orden de lectura, **no** el orden de escritura (ver Fase 3).

### Cap. 1 — Introducción
Motivación: qué se pregunta y por qué la bolsa es el entorno de validación y no el objetivo.
Pregunta de investigación (*¿aprende un sistema multi-agente a ordenar activos fuera de muestra, y es
ese aprendizaje útil económicamente?*). Objetivos. **Contribuciones**: (a) arquitectura multi-agente
con meta-agente por contribución marginal; (b) infraestructura de evaluación honesta (leakage,
walk-forward, baselines, placebo, bootstrap, ablaciones); (c) el hallazgo de la disociación
alfa↔ranking; (d) resultados negativos documentados. Estructura del documento.

### Cap. 2 — Marco teórico y estado del arte
GARP / value-growth. ML aplicado a cross-section de retornos. Rank-IC (Spearman) como métrica y por
qué no RMSE. Walk-forward vs. train/test estático. Lookahead y sesgo de supervivencia. Ensembles y
meta-aprendizaje. IA explicable en finanzas. **Requiere bibliografía real** — ver Riesgos.

### Cap. 3 — Metodología
El contrato metodológico antes de cualquier resultado. Dataset point-in-time (`bisect_right`, sin
lookahead por construcción, no por filtrado posterior). Walk-forward rodante: el cutoff es el punto
**más temprano de reentrenamiento**, nunca congelación — con la justificación medida del rechazo del
esquema anterior (`environment.py:74-86`). Targets como rangos transversales, alineados con la
métrica de evaluación. Selección de hiperparámetros por rank-IC. Protocolo de evaluación: rank-IC OOS,
placebo, baselines, bootstrap por bloques, subperiodos, ablaciones. **La decisión de NO hacer el
barrido multi-cutoff, con su razonamiento** (`robustness.py:16-20`) — es un punto fuerte. Limitaciones
asumidas por adelantado: supervivencia, muestra pequeña, ausencia de lag de publicación.

### Cap. 4 — Datos y variables
Universo (~500 large-caps, estático → sesgo de supervivencia explícito). Fuentes (Finnhub/Yahoo).
Ventanas (`DATA_START_DATE` 2000, cartera 2018-02-15 → 2026-06-15). Dataset maestro
`ticker × snapshot_date`. Scores GARP transversales y `garp_score` como baseline determinista.
Features de expectativa (`realized_growth` observado, no circular) y temporales. Múltiplos ajustados
por precio. **Discutir la ausencia de lag de publicación** (`dataset.py:250-252`): el sistema asume
conocimiento del fundamental el mismo día del cierre de periodo; en la realidad un 10-K tarda 30-90
días. Es la limitación más atacable en la defensa: hay que anticiparla, no esconderla.

### Cap. 5 — Arquitectura del sistema
Pipeline de 8 etapas. Los 3 agentes (calidad / temporización 3m / alpha) y sus targets. El
meta-agente: partial rank-IC sobre residuo, consistencia `mean − λ·std` entre folds, shrinkage hacia
prior inclinado a Calidad (0.45/0.30/0.25) y **por qué no equal-weight**. Capa de gestión:
`manager_score` con **0.70·final_score** (cifra real del código, no el 0.45 de `doc.md`), salidas
HARD/SOFT, stop-loss -25% asimétrico sin take-profit. Sizing: `hybrid_weight = 0.20·equal + 0.80·risk_adjusted`,
convexidad 1.35. Explicabilidad determinista (`synthesis.py`, `manager_score_breakdown`).
**Declarar explícitamente que no hay LLM en el pipeline**: la "IA explicable" son LightGBM + meta-agente
+ narrativa por reglas. Es una fortaleza (reproducible, sin coste, determinista) si se declara; una
sospecha si el tribunal lo descubre solo.

### Cap. 6 — Evaluación del aprendizaje ← **núcleo del TFM**
Rank-IC OOS por agente y por año. **Calidad (+0.025, 72% snapshots positivos) > Temporización
(+0.013) > Alpha (+0.004)**: el agente diseñado para rankear alfa es el menos fiable. `final_score`
hereda esa inestabilidad (+0.0236 con el prior adoptado, verificado sobre los 100 snapshots OOS).
Comparación de horizontes (3m 0.234 > 6m > 12m: **el default no es el óptimo**, decirlo). Placebo
(percentil 1.00).

**Breadth (`top_n_alpha` / `top_n_alpha_lift`)**: el rank-IC ordena las ~71k filas del universo, pero
la cartera solo compra el top-10. El breadth mide el tramo que de verdad se ejecuta y es el puente
entre "la IA rankea" y "la cartera gana" — pieza clave para explicar la disociación del Cap. 9, no un
adorno.

Aquí van los **dos bugs de medición** (Fase 0) como lección metodológica.

### Cap. 7 — Experimentos y ablaciones
Diseño del runner (escenarios declarativos con `why`, overrides aislados, caché de scoring). Los 4
bloques. **Cifras: `comparison_oos.csv`, NO `comparison.csv`** (este último tiene el rank-IC
contaminado; se conserva solo como registro de lo que se publicó con el bug). Resultados:
- **Aprendizaje**: `sin_meta_aprendido` bate al baseline (alfa 1.438 vs 1.138, IR 1.004 vs 0.958,
  IC OOS 0.0259 vs 0.0236) → **por el criterio declarado por el propio proyecto, el meta-agente
  aprendido no aporta**. Es el resultado negativo más importante y sobrevive a la corrección del bug:
  hay que sostenerlo.
- **`solo_alpha` es el peor en IC OOS (0.0115) y en alfa (0.796)**: el agente diseñado para rankear
  alfa es el que peor rankea. Coherente con el diagnóstico por agente del Cap. 6 y con el fracaso
  estructural del experimento 2A.
- **Pesos**: experimento 2A (penalizar varianza) fracasó y se revirtió — con el diagnóstico
  estructural de *por qué* (el meta-agente premia al agente que optimiza el target de evaluación).
  2B (prior a Calidad) mejoró ambos objetivos sin trade-off.
- **Estabilidad**: **resuelto durante la planificación.** Las 4 semillas daban deltas exactamente
  0.000 porque el modelo es **determinista por construcción**: `LGBM_PARAM_GRID` no fija `subsample`
  ni `colsample_bytree`, así que LightGBM corre sin componente estocástico. No era estabilidad
  demostrada, era ausencia de aleatoriedad. Los escenarios de semilla se han eliminado y se documenta
  como **limitación declarada** (no se ha probado robustez ante re-siembra, porque no hay nada que
  re-sembrar). La estabilidad se lee con subperiodos, ventana (3/4/6 años), horizonte y costes.
- **Utilidad**: concentración, rotación, tamaño de cartera.

### Cap. 8 — Resultados económicos y robustez
Alfa acumulada 1.138 (+113.8%), IR 0.958, t-stat 2.765 (n=100) **siempre con `SMALL_SAMPLE_CAVEAT`**.
Bootstrap por bloques: IC 30.3%-197.5%, 99% positiva. Subperiodos: 0.376/0.337/0.425 — repartido, no
un solo tramo. Costes: breakeven 22.5×. **Baselines: `momentum_only` (2.997) bate al sistema (1.138);
`alpha_vs_best_baseline` negativa en los 19 escenarios.** No se entierra: es una sección propia.
**`edge_attribution`**: correlación score de entrada ↔ exceso = **-0.101** (negativa); win rate 49.4%;
ganadoras +21.22 vs perdedoras -11.35. **La alfa viene de asimetría, no de ranking.**

### Cap. 9 — Discusión y conclusiones
La disociación como hallazgo central: se puede tener alfa robusta (bootstrap, placebo, subperiodos)
**y** un modelo que apenas ordena. Son dos afirmaciones distintas y el trabajo las mantiene separadas
a propósito. Por qué la arquitectura favorece estructuralmente al agente que optimiza su propio
target de evaluación. Respuesta explícita a la pregunta de investigación: **el sistema aprende poco
pero de forma medible, y lo poco que aprende no es lo que genera el resultado económico**. Qué
significa que apagar el meta-agente mejore los resultados. Limitaciones. Trabajo futuro (ablation de
scores; descomponer el gap vs momentum_only; lag de publicación; universo con delisted; semillas).
Reflexión sobre resultados negativos como contribución legítima.

**Apéndices**: reproducibilidad (`environment.py`, `requirements.txt`, `pytest tests/`), inventario
de artefactos, tabla completa de los 19 escenarios, glosario.

---

## Fase 3 — Orden de escritura recomendado

No es el orden de lectura. Se escribe de lo más anclado en hechos a lo más interpretativo:

1. **Cap. 3 (Metodología)** — el contrato; todo lo demás se apoya aquí.
2. **Cap. 4 (Datos)** y **Cap. 5 (Arquitectura)** — descriptivos, verificables contra el código.
3. **Fase 0** (arreglar rank-IC) → **Cap. 6 (Aprendizaje)** y **Cap. 7 (Experimentos)**.
4. **Cap. 8 (Resultados económicos)**.
5. **Cap. 9 (Discusión)** — solo cuando los resultados están fijados.
6. **Cap. 2 (Estado del arte)** — necesita bibliografía real.
7. **Cap. 1 (Introducción)** — el último: se escribe mejor sabiendo qué se concluyó.

El usuario pedirá capítulos sueltos; este orden es una recomendación, no una restricción.

## Convenciones de redacción

- Español académico, UTF-8 impecable (acentos, eñes, símbolos). Impersonal ("se entrena"), no
  primera persona.
- **Toda cifra en el LaTeX se verifica contra `results/` o el código antes de escribirla.** Nunca se
  copia de `docs/doc.md` sin comprobar (tiene 12 discrepancias conocidas).
- Etiquetas: `\label{cap:metodologia}`, `\label{tab:ablaciones}`, `\label{fig:rank_ic}`.
- Tablas con `booktabs`. Figuras desde `latex/figuras/` (copiadas de `viewer/charts/`).
- Fórmulas en `amsmath`, con los coeficientes **reales** del código.
- Cada limitación se declara donde corresponde, no solo en un capítulo-vertedero al final.

---

## Fase 4 — Corregir `docs/` (aprobado por el usuario)

Tarea separada de la redacción, a hacer cuando el usuario lo pida. `docs/doc.md` §12/§13 y `README.md`:

| # | Doc dice | Código hace |
|---|---|---|
| 1 | `manager_score = 0.45·final_score` | **0.70** (`strategy/portfolio.py:83-92`) |
| 2 | `hybrid_weight = 0.35·equal + 0.65·risk` | **0.20 / 0.80** (`strategy/sizing.py:26-27`) |
| 3 | `sizing_score` lineal | `conviction_core^1.35` (`sizing.py:61-77`) |
| 4 | 7 motivos de salida | **8**: falta `Stop-Loss` (-25%) |
| 5 | ventaja de score ≥0.09 | `MIN_ROTATION_ADVANTAGE = 0.10` |
| 7 | `meta_agent.policy`: "equal-weight prior" | prior inclinado a Calidad (`ml.py:259`) |
| 9 | README: experimentos corren "solo ml+backtest" | corren `ml→watchlist→backtest→viewer→report` |
| 10 | `scripts/diagnostico_rank_ic.py` reproducible | **`scripts/` no existe** |
| 11 | `module/research/ai.py` | **eliminado** (commit `ee5fa1c7`) |

La #1 no es cosmética: con `final_score` al **0.70**, la afirmación "la IA es el motor, no un
overlay" es mucho más fuerte — y **agudiza la paradoja central**: el ML pesa el 70% de la decisión y
aun así la alfa no viene de su ranking.

---

## Verificación

- **Fase 0** ✅: `pytest tests/` → **65 pasan**, sin regresiones. `comparison_oos.csv` reproduce
  `diagnostico_aprendizaje.md` (baseline 0.0236; Calidad 0.0255 > Alpha 0.0115).
- **LaTeX**: compilar `main.tex` en Overleaf tras cada capítulo. Localmente solo se valida sintaxis
  por inspección (no hay LaTeX instalado; no se instalará sin permiso).
- **Cifras**: cada tabla del LaTeX se contrasta contra el CSV de `results/` que la origina.
- **Docs (Fase 4)**: `pytest tests/` completo, ya que se tocan afirmaciones sobre fórmulas.

## Riesgos

1. **Bibliografía (Cap. 2)**: no hay `.bib` ni referencias en el repo. **No se inventarán citas.**
   Requiere que el usuario aporte las referencias o que se acuerde buscarlas explícitamente. Es el
   único capítulo que no se puede escribir solo con lo que hay en el repo.
2. **Tablas del Cap. 7 provisionales**: se escriben desde `comparison_oos.csv` (19 escenarios de
   `20260716_105304`). Los escenarios nuevos (`ventana_larga`, `horizonte_24m`, `costes_extremos`) y
   la eliminación de las semillas **no** están en ese barrido. Habrá que actualizarlas tras el
   próximo barrido, que requiere autorización por su coste.
3. **`comparison.csv` sigue en disco con las cifras contaminadas.** Se conserva a propósito, pero
   **nunca debe citarse en el LaTeX**: usar siempre `comparison_oos.csv`. Un barrido nuevo lo
   regenerará ya limpio y esta ambigüedad desaparecerá.
4. **Trabajo del otro agente sin commitear** (11 ficheros). Mis cambios se apoyan encima. Conviene
   commitear pronto para no perderlos.
5. ~~Colisión en `metrics.py`~~ — no se materializó.
6. ~~Semillas~~ — **resuelto**: el modelo es determinista; escenarios eliminados y documentado como
   limitación.
7. **Normativa de la universidad** (extensión, portada, estilo de cita) desconocida: `main.tex` es
   genérico y adaptable; la portada es provisional.
