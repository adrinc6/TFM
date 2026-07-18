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

## 5 — Estudio final automatizado (2016-2026): ML sin señal, factores con rentabilidad

**Que se hizo.** Con el sistema consolidado (LightGBM + meta rank_ic, ancla 2016, artefactos
activables), se lanzo el estudio completo de principio a fin con un comando (`RUN_MODE=full_study`):
barrido de ablations → decision automatica de artefactos por significancia → run final → 8 perfiles
de inversor → tests de robustez/placebo. Sin intervencion humana.

**Decision automatica de artefactos.** De los 7 artefactos, el barrido acepto **solo la
neutralizacion por sector** (rank-IC del meta_final 0.0036 → 0.0094, mejor en el 59 % de las
fechas). Los otros seis (momentum de fundamentales, regimen bull/bear, regimen ampliado, momentum
de precio, medias moviles, calidad/crecimiento derivados) **empeoran o no aportan**. Curiosamente,
la neutralizacion por sector —que con el modelo lineal y ancla 2000 EMPEORABA (ver historia
previa)— ahora ayuda: con LightGBM y mas cobertura (2016+, sectores mejor poblados) la
reorganizacion dentro de sector si limpia ruido. Patron consistente: anadir features no crea
señal; solo reorganizar el ranking aporta algo marginal.

**Aprendizaje: no significativo.** El sistema final alcanza rank-IC **+0.0036**, con IC bootstrap
**[−0.019, +0.024]** (cruza cero). El **placebo** (permutacion de etiquetas) da p-valor **0.20**:
el modelo real no supera al azar. Leave-one-year-out: oscila entre +0.0008 y +0.0085, ninguno lo
sostiene solo, todos ≈0. **El ML no aprende a ordenar de forma estadisticamente significativa.**

**Rentabilidad: los perfiles de estilo baten al SPY (limpio).** Con la guarda anti-artefactos
activa (sin el +953 % corrupto de 2010), en 2016-2026: quality +4.5 %/año, value +4.4 %,
conservative +4.4 %, garp +3.7 % (bate al SPY el 64 % de los años, drawdown 41 %). El perfil
**balanced —el que sigue el meta-score del ML puro— es el PEOR** (−1.5 % vs SPY, drawdown 48 %).

**Hallazgo central del TFM.** Los dos planos son opuestos y coherentes: como el ML no ordena bien
(rank-IC ≈ 0), seguir su ranking puro no bate al mercado; pero inclinar la cartera hacia calidad y
valor captura **primas de factor clasicas** que si existen. **El valor del sistema no esta en su
aprendizaje automatico —que no lo hay— sino en explotar de forma disciplinada primas de factor
conocidas, y en haberlo demostrado con honestidad** (placebo, bootstrap, estabilidad, guarda
anti-artefactos). Un resultado matizado, medido y defendible: ni un exito de IA que no existe, ni
un fracaso, sino una separacion limpia entre lo que el sistema aprende (poco) y lo que rinde (los
factores). Ver docs/informe_final.md y results/escenarios/study_summary.json.

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
