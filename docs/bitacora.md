# Bitácora de decisiones del TFM

> Diario cronológico del desarrollo. Registra **el porqué** de cada decisión, no solo el qué:
> qué problema apareció, qué hipótesis se manejó, qué se probó, qué resultado se **midió** y qué
> se decidió. Es la materia prima del capítulo de metodología del TFM: un trabajo sobre
> aprendizaje se juzga tanto por lo que descubre como por cómo razona sus decisiones, incluidos
> los caminos que se descartaron y por qué.
>
> Formato de cada entrada: **problema/observación · hipótesis · qué se probó · resultado
> medido · decisión**. Las entradas se añaden *cuando ocurren*, no al final.

---

## 1 — La fecha de publicación no puede aproximarse con un retardo fijo

**Observación.** Un fundamental fechado por su cierre fiscal no es observable ese día: la
empresa lo publica semanas después. Aplicar un retardo fijo (p. ej. 45 días) a todos los
fundamentales para aproximar cuándo son públicos es incorrecto.

**Qué se probó y midió.** Se contrastó el retardo fijo contra las fechas reales de presentación
en SEC EDGAR. El retardo real es muy variable: AT&T llegó a tardar **133 días** y un 10-K de
Apple **88 días**. Cualquier retardo fijo introduce lookahead en las empresas que tardan más.

**Decisión.** Se usa la **fecha real de presentación** (`filingDate` de SEC EDGAR, gratuita,
desde 1993). `lag_days` deja de ser una regla de observabilidad y pasa a significar solo el
**margen de ejecución** (simula el día en que se habría lanzado el pipeline).

---

## 2 — Tres fugas silenciosas del pipeline point-in-time

**Observación.** Al auditar el pipeline aparecieron tres defectos que no rompían tests pero
producían números sutilmente falsos.

**Qué se probó y midió.**
- **Crecimiento interanual por índice posicional**: contaba "cuatro trimestres atrás" por
  posición; si faltaba un trimestre (frecuente), comparaba contra una fecha arbitraria. Un caso
  real daba un "+71 %" falso comparando contra 15 meses atrás.
- **Targets perdidos con `SNAPSHOT_DAY ≠ 15`**: la fecha de etiqueta se derivaba por aritmética
  de calendario, que clampa los fines de mes distinto a la rejilla. Con día 31 se perdía el 40 %.
- **Mezcla anual/trimestral en márgenes**: un margen anual y su Q4 comparten cierre; el fallback
  confundía magnitudes de 12 y 3 meses en el mismo corte.

**Decisión.** Los tres corregidos con **tests de regresión** que fallan contra el código
anterior. Lección: en un proyecto cuyo eje es la ausencia de lookahead, los tests de fuga van
**antes** que el código.

---

## 3 — Del enfoque lineal a LightGBM (síntesis del camino recorrido)

**Punto de partida.** El sistema inicial usaba agentes lineales (regresión sobre factores
GARP+momentum) que predecían el exceso de retorno. Se midió con rigor (walk-forward
point-in-time, rank-IC fuera de muestra) y se probaron múltiples mejoras sobre esa base:
neutralización por sector, tratamiento de la etiqueta, momentum y descomposición de
fundamentales, régimen de mercado.

**Resultado medido.** El rank-IC OOS del enfoque lineal se mantuvo **cercano a cero** en todas
las variantes (el mejor apenas +0.001-0.002). Diagnóstico: un modelo lineal **promedia a cero**
las interacciones entre factores, y añadir features no crea señal donde no la hay —solo añade
sitios donde sobreajustar. Se concluyó que el techo del enfoque lineal estaba en rank-IC ≈ 0.

**Decisiones metodológicas que se conservan de esa etapa** (son del sistema, no del modelo
lineal):
- **La selección NO se hace por rentabilidad, sino por aprendizaje (rank-IC) y estabilidad.** Con
  rank-IC ≈ 0, cualquier rentabilidad es ruido afortunado; elegir por ella sería seleccionar
  ruido. La rentabilidad se reporta como consecuencia.
- **La métrica de rentabilidad es CAGR real / anualizada**, nunca el acumulado compuesto (que
  daba cifras absurdas y lo dominaba un solo año).
- **La métrica de aprendizaje se mide sobre el `meta_final`** (el score que opera la cartera),
  no sobre el promedio de los agentes individuales.

**Decisión.** Se adopta **LightGBM** (árboles con gradient boosting), que captura las
interacciones no lineales que el modelo lineal no representa, con objetivo alineado al ranking
(`rank_regression`: regresión sobre el percentil transversal del retorno). Es el punto de
partida del sistema actual.

---

## 4 — LightGBM mejora al lineal, pero la señal sigue siendo débil

**Qué se probó.** Comparación LightGBM vs lineal con el mismo objetivo, barrido de
hiperparámetros, sensibilidad a la semilla, y comparación del meta ponderado vs equiponderado.

**Resultado medido (full, ancla 2000).**
- LightGBM+rank_regression: rank-IC del meta_final **+0.0117** (56.8 % de cohortes positivas),
  frente a +0.0065 del lineal — **casi el doble**, y robusto a la semilla (+0.0088 a +0.0117 en
  4 semillas).
- El meta ponderado por rank-IC (+0.0117) **bate al equiponderado** (+0.0054): la combinación de
  agentes aporta.
- **Diagnóstico temporal**: el rank-IC mejora en años recientes — desde 2014, +0.0178 (63.5 %
  de cohortes positivas), coherente con la mayor cobertura de datos (246 empresas/cohorte en
  2000 → 492 en 2025).

**Pero**: +0.0117 sigue siendo, a efectos prácticos, **cero** (la industria considera útil un
factor a partir de ~0.03-0.05), y la mejora **no es estadísticamente distinguible de cero** (el
bootstrap por bloques cruza cero).

**Y el rendimiento aparente era un espejismo.** Un backtest de 2000-2026 daba CAGR ~18 %/año,
pero se debía **casi por completo a un artefacto de datos**: julio de 2010 registró un retorno
mensual de **+953 %** en una posición (precio corrupto — split mal ajustado o ticker reciclado).
Sin ese año, la alfa mediana anual era **−0.2 %** y la cartera batía al SPY solo 13 de 27 años.
El rank-IC ≈ 0 **no se dejó engañar por el artefacto; el CAGR sí** — la demostración de por qué
el aprendizaje, y no la rentabilidad, es la métrica honesta.

**Decisión.** Consolidar el sistema en LightGBM y convertirlo en una **base modular** con
"artefactos" activables (bloques de features/contexto), centrada en los años recientes (2016+,
más cobertura), con una **guarda anti-artefactos** en el backtest para que la rentabilidad sea
honesta, y un barrido de **ablations** que decide automáticamente —por significancia
estadística— qué artefactos ayudan al aprendizaje. El objetivo sigue siendo el rank-IC.

*(Las entradas de los artefactos y el estudio de ablations se añaden a continuación conforme se
prueban, cada una con su número de rank-IC antes/después.)*

---

## 5 — Estudio final automatizado de principio a fin: decisión de método

**Que se decidio.** Con el sistema consolidado (LightGBM + meta rank_ic, ancla 2016, artefactos
activables), se decide cerrar el proyecto con un **estudio completo de principio a fin lanzable con
un comando** (`RUN_MODE=full_study`) y **sin intervencion humana**: barrido de ablations → decision
automatica por significancia → run final → 8 perfiles de inversor → tests de robustez/placebo →
informes. El objetivo es que la configuracion final no la elija una persona (que sesga), sino un
criterio reproducible y auditable.

**Por que automatico.** Un TFM sobre aprendizaje debe evitar que el investigador escoja "el mejor
resultado" tras mirar muchos. Al delegar la decision en un criterio fijo (estabilidad del rank-IC
del meta_final, no rentabilidad) y registrarla en un JSON, la eleccion es trazable y no depende de
la mano humana. La rentabilidad se mide **como consecuencia** y con la guarda anti-artefactos
activa (§4.3), nunca como selector.

**Que se mide y como se reporta.** El resultado se separa en dos planos que **no se mezclan**: el
**aprendizaje** (rank-IC OOS del meta_final, con bootstrap por bloques, placebo por permutacion de
etiquetas, leave-one-year-out y validacion en la era reservada) y la **rentabilidad** (CAGR real vs
SPY, beat rate, drawdown y turnover por perfil). El diseño no presupone el signo: un rank-IC no
distinguible del azar es un entregable valido.

**Estado.** Esta primera version del orquestador tenia un fallo de metodo —solo optimizaba los
artefactos on/off e ignoraba ventana, profundidad y cadencia aunque el barrido ya las probaba— que
se corrige en la entrada 6 (barrido en dos fases + era reservada). Por eso el **estudio se esta
reejecutando** con el orquestador corregido, y las cifras finales se anadiran aqui y en
docs/informe_final.md al terminar esa ejecucion (procedentes de
results/studies/<study_id>/study_manifest.json y sus runs asociados).

---

## 6 — El orquestador solo optimizaba artefactos: barrido en dos fases y era reservada

**Observacion.** Al revisar el estudio 5 aparecio un fallo de metodo, no de datos: la decision
automatica (`decide_accepted_artifacts`) solo evaluaba los artefactos on/off y componia la config
final con ellos, pero **ignoraba la ventana, la profundidad y la cadencia** aunque el barrido ya
las probaba. Consecuencia real: se eligio `baseline+sector` (rank-IC del meta_final +0.0036) cuando
el propio barrido tenia `train_8y` en **+0.0129** y `depth_6` en +0.0076. Se estaba tirando señal
que ya estaba medida.

**Hipotesis.** Si el sistema debe elegir "lo mejor" sin intervencion humana, la decision tiene que
cubrir **todos los ejes** (ventana, horizonte de etiqueta, ancla, profundidad, cadencia, artefactos),
no solo un subconjunto. Y al ampliar la exploracion (mas ejes y mas niveles) sube el riesgo opuesto:
**overfitting por seleccion** —con muchos escenarios, el maximo puede ser suerte—.

**Que se hizo.** (1) `decide_best_config` generaliza la decision: para cada eje con niveles elige el
**mas estable** (mayor rank-IC medio del meta_final; desempate por fraccion positiva y menor
varianza) y sigue aceptando artefactos por diferencia pareada. (2) El barrido pasa a **dos fases**:
Fase 1 aisla cada eje (ventanas 5-12 años, horizontes 1/3/6/12 meses, anclas 2016/2018/2020,
profundidad 3-6, cadencia) y Fase 2 combina solo los ganadores (no producto cartesiano), con un
afinado final de hiperparametros. (3) Contra el overfitting por seleccion, la eleccion usa **solo
cohortes hasta 2024** y **reserva 2025-2026** para validar al finalista donde nunca se optimizo
(`reserved_era_validation`). Es un filtro point-in-time sobre cohortes ya calculadas: no reentrena
ni mira al futuro.

**Por que importa para la honestidad.** Ampliar el barrido sube el rank-IC que se puede *encontrar*,
pero un maximo mayor no es necesariamente señal: podria ser el mejor de muchos intentos. La era
reservada y el placebo son el contrapeso —solo se declara "señal" si el finalista es significativo
**y** aguanta fuera del periodo de busqueda—. Si no lo hace (lo mas probable segun el historial), el
hallazgo honesto sigue siendo que el ML no aprende de forma fiable, ahora respaldado por una
exploracion mucho mas amplia, lo que **refuerza** la conclusion en vez de debilitarla.

*(Los numeros del estudio con el orquestador nuevo se anaden aqui al re-ejecutar el full con las
dos fases; esta entrada registra la decision de metodo, que es previa a los resultados.)*

---

## 6 — Interfaz de analisis unificada: de HTML incrustado a una app real

**Sintoma.** La interfaz vivia como cadenas de Python incrustadas: un SPA completo dentro de
`_APP_HTML` (en el antiguo `dashboard.py`) y los informes estaticos generados por `report.py`, con
dos paletas distintas (consola azul oscura, informes claros) y los graficos dibujados a mano con
SVG/canvas. Anadir funcionalidad de analisis (mas graficos, cargar CSV/Parquet) era caro y fragil.

**Que se hizo.** Se movio todo el frontend a archivos reales bajo `app/`, en la raiz del
proyecto (junto a `module/`, `results/` y `docs/`), con `index.html`, `css/` y `js/` por vista,
servidos como estaticos por el `http.server` de `module/ui/dashboard.py` (que resuelve la carpeta
`app/` desde `PROJECT_ROOT`). El backend queda puro: su API JSON se conserva intacta y se
anaden `/api/study/<id>` y `/api/meta_weights`. Estetica oscura unica (negros y grises, sin azul),
centralizada en tokens CSS. Los graficos pasan a **Chart.js embebido localmente** (sin CDN, sin
dependencia Python nueva). La vista de Resultados separa **estudios** y **runs**: al elegir un
estudio se analiza el estudio; al elegir un run, su resumen, rendimiento, aprendizaje, cartera,
trades, explorador de stocks (con crecimientos y explicabilidad de agentes) y ficha por ticker.
Los informes estaticos se reescriben en `module/ui/reports.py` con la misma paleta oscura (CSS
compartido inyectado inline, figuras matplotlib oscuras), conservando su API publica y el
comportamiento observable (`report.html` por run, `comparison.html` por fase). Como optimizacion
aditiva y sin tocar el modelo, `results_store.publish_artifacts` precalcula `learning_summary.json`
(mismo patron que `position_lifecycle`), con reserva al calculo al vuelo para runs antiguos.

**Por que importa.** La metodologia (point-in-time, seleccion por rank-IC, era reservada) no
cambia: es solo la capa de presentacion y analisis. Pero separar el frontend en archivos reales
hace el sistema mas facil de revisar y extender, y unifica consola e informes bajo una sola
estetica. Los 92 tests siguen en verde tras el cambio.

---

## 7 — study y full_study convergen en un ciclo completo con TODAS las variables

**Sintoma.** Habia dos orquestadores desiguales. `full_study` hacia el ciclo completo (Fase 1 →
Fase 2 → afinado → run final → perfiles → robustez → era reservada) pero solo barria un catalogo
fijo y reducido (5 ejes de modelo + 7 artefactos): dejaba fuera `execution_lag_days`,
`execution_quarter`, `objective`, `meta_type` y los hiperparametros finos en Fase 1. `execute_study`
barria las variables marcadas por el usuario pero se paraba en Fase 2 —y si el usuario marcaba una
variable de cartera, la barria reentrenando el modelo inutilmente.

**Que se hizo.** Un unico nucleo `run_optimization` que ambos flujos invocan; solo cambia que
variables barren. Las variables se separan por CUANDO actuan (frontera autoritativa ya existente en
`FINGERPRINT_FIELDS`): las de MODELO (mueven el rank-IC) se barren en Fase 1/2 y se eligen por
rank-IC OOS; las de CARTERA (no mueven el rank-IC) se optimizan al final re-backtesteando el
finalista sin reentrenar, por criterio economico (`information_ratio`). `full_study` pasa a barrer
TODO (derivado de `escenarios/variables.py`, antes solo en la UI). Fase 2 evoluciona a greedy
incremental con top-2 por eje (~2·N runs, nunca 2^N). `execute_study` deja de pararse en Fase 2 y
hace el ciclo entero.

**Dos matices de metodo del usuario, incorporados.** (1) `execution_lag_days` es de MODELO, no de
cartera: un lag de 15 dias cambia que fundamentales son observables en cada snapshot (aprovecha
antes la publicacion de resultados) y por tanto cambia el dataset de entrenamiento. (2) Interaccion
`execution_quarter` × `fundamental_step_months`: el trimestre de arranque solo importa cuando el
reentreno es semestral o anual (con reentreno trimestral/mensual se diluye); por eso, si la cadencia
ganadora no es trimestral, la Fase 2 re-explora los trimestres sobre esa combinacion.

**Por que preserva la honestidad.** La seleccion del MODELO sigue siendo por rank-IC OOS y la era
reservada 2025-2026 nunca interviene. El criterio economico solo decide parametros de CARTERA, que
por construccion no alteran el aprendizaje (no estan en el fingerprint de `agents`), asi que el
rank-IC no puede discriminarlos y el criterio economico es el unico que tiene sentido ahi.
`decision.json` cierra con `best_config`: mejor modelo + mejor gestion de cartera + perfil que mas
renta. 100 tests en verde (92 + 8 nuevos del ciclo unificado).

---

## N — Repetición del `full_study` oficial (2026-07-19): señal más débil, más perfiles ganan

**Observación.** Se relanzó `full_study` con `run_optimization` (el núcleo unificado descrito
arriba) para cerrar la pieza pendiente del estudio anterior (`20260718--...--bef48ddfc41f--r02`):
el placebo por permutación de etiquetas. El nuevo estudio es
`20260719--optimization-official--acb6c310dfb8`.

**Qué se midió.** El rank-IC OOS del modelo ganador bajó de +0.0158 a +0.0118, y el intervalo de
bootstrap por bloques pasó de **no cruzar cero** ([0.0053, 0.0265]) a **cruzar cero**
([-0.0113, 0.0340], sobre solo 45 cohortes frente a 147 antes). El leave-one-year-out y la era
reservada 2025-2026 (+0.0210) siguen siendo consistentes con una señal real y estable, pero el
contraste más exigente (bootstrap) ya no es concluyente. El placebo por permutación **sigue sin
ejecutarse** (`n_permutations=0`) — la pieza que se quería cerrar con esta repetición sigue
pendiente.

En el plano de rentabilidad, 5 de 8 perfiles baten ahora al SPY (antes solo `quality`), con
`aggressive` como recomendado (IR 0.153). Esta mejora viene de la fase de cartera (menos
posiciones, mayor peso máximo por posición, menor comisión y slippage), no de una señal de
aprendizaje más fuerte.

**Decisión.** Se documenta como resultado **mixto**, no como mejora: el eje que importa
(aprendizaje) no mejoró, y la pieza de robustez pendiente sigue sin resolverse tras dos estudios
oficiales. `docs/doc.md` §8 y `docs/informe_final.md` se actualizan con las cifras de este estudio.
Queda abierta la decisión de si ejecutar el placebo por permutación de forma aislada antes de dar
por definitivamente cerrado el proyecto, o si aceptar el resultado actual (con la limitación
explícita) como cierre del TFM.
