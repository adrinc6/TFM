# Bitácora de decisiones del TFM

> Diario cronológico del desarrollo. Registra **el porqué** de cada decisión, no solo el qué:
> qué problema apareció, qué hipótesis se manejó, qué se probó, qué resultado se **midió** y qué
> se decidió. Es la materia prima del capítulo de metodología y la defensa del trabajo: un TFM
> sobre aprendizaje se juzga tanto por lo que descubre como por cómo razona sus decisiones,
> incluidos los caminos que se descartaron y por qué.
>
> Formato de cada entrada: **problema/observación · hipótesis · qué se probó · resultado
> medido · decisión**. Las entradas se añaden *cuando ocurren*, no al final.

---

## 1 — La fecha de publicación no puede aproximarse con un retardo fijo

**Observación.** Un fundamental fechado por su cierre fiscal no es observable ese día: la
empresa lo publica semanas después. El primer diseño aplicaba un retardo fijo (p. ej. 45 días)
a todos los fundamentales para aproximar cuándo eran públicos.

**Hipótesis.** Un retardo fijo daría una aproximación "suficientemente buena" y homogénea.

**Qué se probó.** Se contrastó el retardo fijo contra las fechas reales de presentación en
SEC EDGAR. Se buscaron casos extremos.

**Resultado medido.** El retardo real es muy variable: AT&T llegó a tardar **133 días** en
presentar, y un 10-K de Apple **88 días**. Cualquier retardo fijo introduce lookahead en las
empresas que tardan más (se les asigna como "conocido" algo que aún no lo era) y retrasa
artificialmente a las que publican rápido.

**Decisión.** Se descarta el retardo fijo. Se usa la **fecha real de presentación** (`filingDate`
de SEC EDGAR, disponible desde 1993, gratuita y oficial). El parámetro `lag_days` deja de ser
una regla de observabilidad y pasa a significar solo el **margen de ejecución**: simula el día
en que se habría lanzado el pipeline, no cuándo un dato concreto se hizo público.

---

## 2 — Tres fugas silenciosas encontradas en revisión del pipeline point-in-time

**Observación.** Al auditar las Fases 1-3 (ya implementadas) aparecieron tres defectos que no
rompían ningún test pero producían números sutilmente falsos.

**Qué se probó y midió.**
- **Crecimiento interanual por índice posicional.** El cálculo tomaba "cuatro trimestres atrás"
  contando posiciones en la serie; si faltaba un trimestre (frecuente, la cobertura de Finnhub
  es irregular), comparaba contra una fecha arbitraria y lo etiquetaba igual como interanual. Un
  caso real comparaba contra 15 meses atrás dando un "+71 %" falso.
- **Targets perdidos con `SNAPSHOT_DAY ≠ 15`.** La fecha de la etiqueta se derivaba por
  aritmética de calendario, que clampa los fines de mes distinto a la rejilla de snapshots. Con
  `SNAPSHOT_DAY = 31` se evaporaba el **40 %** de las etiquetas sin ningún error.
- **Mezcla anual/trimestral en márgenes.** Un margen anual y su Q4 comparten fecha de cierre;
  el fallback confundía magnitudes de 12 meses con las de 3 dentro del mismo corte transversal.

**Decisión.** Los tres se corrigieron con **tests de regresión** que fallan contra el código
anterior: emparejar el interanual por fecha (±45 días), tomar la fecha de etiqueta de la propia
rejilla, y leer los márgenes no-TTM solo del bloque trimestral. Lección metodológica: en un
proyecto cuyo eje es la ausencia de lookahead, los tests de fuga van **antes** que el código.

---

## 3 — El alfa acumulado a 25 años no es una métrica útil

**Observación.** El primer barrido reportaba `total_alpha` como `equity_final/equity_inicial −
1`. Sobre 25 años esto daba cifras absurdas (cientos de miles de %, hasta 1.4·10⁹ en un
escenario).

**Hipótesis.** El acumulado compuesto premia desproporcionadamente un único año excepcional y
crece sin control con el horizonte, así que no permite comparar configuraciones de forma justa.

**Decisión.** Se sustituye por la **alfa anualizada geométrica** (media compuesta del exceso
anual), interpretable como "X % al año", más la alfa mediana anual y el peor año. El acumulado
bruto desaparece de la vista principal por engañoso.

---

## 4 — La selección del sistema no debe hacerse por alfa

**Observación.** La métrica de selección de escenarios incluía dos dimensiones de alfa
(mediana y peor año). Para un TFM cuya pregunta central es *si el sistema aprende a ordenar
activos fuera de muestra*, seleccionar por rentabilidad es poco defendible.

**Hipótesis.** Si el aprendizaje real (rank-IC) es débil, cualquier alfa observado es en buena
parte suerte de composición de cartera y de qué años tocaron; elegir por alfa sería
**seleccionar ruido**.

**Decisión.** La selección pasa a basarse **solo en aprendizaje y estabilidad**: rank-IC medio
OOS, fracción de cohortes con rank-IC positivo, beat rate (frecuencia, no magnitud) y drawdown
(riesgo). **El alfa se reporta como consecuencia, nunca decide.** Además se elimina la
separación en eras: un único ranking global sobre todos los años (decisión del autor, que
prefería no partir la muestra ni por eras ni por años).

---

## 5 — Hallazgo central: el rank-IC fuera de muestra es ~0

**Observación.** Con la métrica ya centrada en aprendizaje, se midió el rank-IC OOS del
escenario base.

**Resultado medido.** Media global del rank-IC **−0.006**; solo el **48.6 %** de las cohortes
tienen rank-IC positivo (lo esperable del azar es 50 %). Por agente: momentum −0.007, quality
−0.021, value +0.011. Por década, el signo oscila alrededor de cero sin ninguna época de
aprendizaje sostenido (`value` tuvo +0.077 en 2000-2004 y se desvaneció después).

**Interpretación.** Con datos gratuitos, universo reducido y factores GARP+momentum lineales,
el sistema **no ordena activos mejor que el azar de forma estable**. Esto no es un fallo del
código —el pipeline es correcto y está probado— sino un **resultado del propio TFM**.

**Decisión.** Antes de dar el resultado por definitivo, se intenta subir el rank-IC con
palancas baratas y bien medidas (neutralización por sector/tamaño, etiqueta menos ruidosa,
momentum y descomposición de fundamentales, régimen de mercado). Cada palanca se acepta **solo
si sube el rank-IC OOS** en el walk-forward completo. Si tras agotarlas el rank-IC sigue ≈0,
ese es el resultado, y se reporta con honestidad en lugar de maquillarlo con alfa.

*(Las entradas de las palancas B1-B5 se añaden a continuación conforme se prueban, cada una con
su número de rank-IC antes/después.)*
