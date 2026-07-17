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

**Baseline de referencia (full, 680 tickers, 942 cohortes):** rank-IC medio **−0.0058**,
fracción de cohortes positivas **0.486**. Por agente: momentum −0.007, quality −0.021, value
+0.011. Este es el número contra el que se compara cada palanca.

---

## B1 — Neutralización por sector

**Hipótesis.** El ranking transversal de cada factor es global, así que un factor de calidad
puede estar midiendo "sector con ROE estructuralmente alto" en vez de "empresa mejor que sus
comparables". Rankear **dentro de sector** debería quitar ese ruido sistemático y dejar solo la
señal relativa, que es la que predice retorno relativo.

**Qué se probó.** Se añade `neutralize_by_sector` (parámetro): cada factor se rankea dentro de
`(fecha, sector)` en vez de `(fecha)`. El sector viene de `profiles.parquet` (snapshot actual
de Finnhub) y se usa **solo para agrupar, nunca como señal** — lookahead residual menor y
documentado. Guarda de tamaño: con menos de `neutralize_min_group` (5) miembros útiles, el
grupo cae a ranking global, porque en los años 2000 muchos sectores tienen 1-2 tickers (27 de
44 sectores tienen <5 miembros en 2000) y un grupo diminuto da un percentil degenerado.

**Resultado medido (full).** **Empeora**, no mejora:

| | baseline | B1 (sector) |
|---|---|---|
| rank-IC medio | −0.0058 | **−0.0083** |
| frac cohortes IC>0 | 0.486 | **0.467** |
| value (agente menos malo) | +0.0107 | +0.0034 |

Todos los agentes bajan.

**Interpretación.** La hipótesis era razonable pero los datos la rechazan. La causa más
probable es el tamaño de los grupos: solo 242 de 492 tickers del índice en 2000 tienen perfil,
y 27 de 44 sectores tienen menos de 5 miembros. Neutralizar en grupos diminutos añade ruido —
el poco orden transversal que había se reparte en subgrupos donde el ranking es casi aleatorio.
La neutralización por sector es una técnica de carteras grandes y bien pobladas; con este
universo reducido y sesgado por supervivencia, hace daño.

**Decisión.** **Se descarta B1.** El código queda en el repositorio desactivado por defecto
(`neutralize_by_sector=False`) porque es una opción legítima y el propio experimento —que no
funcione— es un resultado documentable del TFM, pero no se usa. Diagnóstico complementario
(B4): la correlación de Spearman entre agentes es quality↔value 0.51, momentum casi ortogonal
(0.15). Momentum aporta la señal más independiente; quality y value se solapan. Sugiere que la
mejora, si llega, vendrá de **mejorar la señal de cada agente** (etiqueta, features), no de
neutralizar ni de añadir agentes redundantes.

---

## B2 — Etiqueta menos ruidosa

**Hipótesis.** El exceso de retorno a 3 meses es muy ruidoso; unos pocos outliers dominan la
regresión Ridge y degradan el orden aprendido. Tratar la etiqueta (recortar colas, alargar
horizonte, o entrenar contra el rango en vez del valor) debería subir el rank-IC.

**Qué se probó.** Parámetro `label_transform` aplicado **solo a la etiqueta de entrenamiento**
(nunca al scoring): `winsor` (recorta el 2 % de cada cola), `rank` (percentil transversal del
retorno futuro dentro de cada snapshot), y horizonte 6m con etiqueta cruda y con rank.

**Resultado medido (full, baseline 3m/none = −0.0058, frac 0.486):**

| horizonte | etiqueta | rank-IC medio | frac cohortes IC>0 |
|---|---|---|---|
| 3m | none | −0.0058 | 0.486 |
| 3m | winsor | −0.0021 | 0.511 |
| **3m** | **rank** | **+0.0011** | **0.534** |
| 6m | none | −0.0162 | 0.459 |
| 6m | rank | −0.0137 | 0.481 |

**Interpretación.** Entrenar contra el **rango** del retorno (no su magnitud) es lo que más
ayuda: cruza el rank-IC a positivo y sube la fracción de cohortes ganadoras a 53.4 % (deja de
ser azar puro). Es coherente: el objetivo real es ordenar, y entrenar contra el orden alinea
la pérdida con la métrica. El **horizonte 6m empeora** de forma clara y contraintuitiva —
menos cohortes independientes, más solapamiento temporal y la relación factor→retorno se
diluye más de lo que se limpia. Winsor ayuda algo, menos que rank.

**Decisión.** Se adopta **`label_transform="rank"` a 3 meses** como nueva base. Es una mejora
real pero pequeña: +0.0011 sigue siendo un rank-IC ≈ 0, **no una señal explotable**. Se
mantiene el horizonte de 3 meses. Aviso honesto: tratar mejor la etiqueta no crea señal donde
no la hay; solo deja de destruir la poca que existe. La palanca con potencial de aportar
información nueva es B3 (tendencia de fundamentales) y B5 (régimen).

**Bug encontrado por el camino.** El `_run_id` de `agents.py` no incluía `label_transform` en
su huella, así que winsor y rank colapsaban al mismo `run_dir` y se sobrescribían. Arreglado
(la huella ahora incluye `label_transform`, `label_winsor_pct`, `min_training_rows`,
`min_rank_ic_cross_section`). Lección: la huella de un run debe cubrir **todo** parámetro que
altere el resultado, o dos experimentos distintos se pisan en silencio.

---

## B3 — Momentum y descomposición de fundamentales

**Hipótesis (la más fundamentada económicamente).** El nivel de un ratio dice poco; su
**tendencia** dice más. Y un P/E que cae por beneficios crecientes (empresa mejorando) no es lo
mismo que uno que cae por precio hundido (posible trampa de valor). Añadir la tendencia de los
fundamentales (ΔROE, Δmárgenes… respecto a la publicación anterior de la misma empresa) y la
descomposición del cambio de valoración en su componente-precio y su componente-fundamental
debería aportar señal nueva y subir el rank-IC.

**Qué se probó.** Parámetro `fundamental_momentum`. Point-in-time por construcción: el delta se
calcula solo cuando cambia el `fundamental_period` (nueva publicación) y se mantiene constante
entre publicaciones (no cae a cero: entre trimestres no hay información nueva). Descomposición
del P/E vía Δlog(precio) − Δlog(EPS), con el EPS implícito de `precio / (P/E)`. Se añaden 7
factores (5 tendencias a quality, 2 componentes a value), sobre la base B2 (`label_transform=
rank`). Verificado sin lookahead con test dedicado (mutar el futuro no cambia el pasado).

**Resultado medido (full, vs base rank +0.0011 / 0.534):**

| config | rank-IC medio | frac cohortes IC>0 |
|---|---|---|
| rank (base B2) | +0.0011 | 0.534 |
| rank + B3 | **−0.0046** | **0.513** |

**Empeora.** Confirmado que las 14 features B3 estaban en el modelo (vía coeficientes).

**Interpretación.** La hipótesis económica es sólida pero los datos la rechazan. Dos causas
probables: (1) cobertura limitada — la descomposición del P/E necesita P/E>0 en dos
publicaciones consecutivas y el P/E solo cubre 78 % del panel, así que el factor llega muy
poblado de NA e imputado a la mediana, lo que mete ruido; (2) con Ridge lineal y una etiqueta
casi aleatoria, **añadir más features no crea señal, añade dimensiones donde sobreajustar** —
cada feature ruidosa resta grados de libertad efectivos y degrada el orden OOS.

**Decisión.** **Se descarta B3.** El código queda desactivado por defecto
(`fundamental_momentum=False`); es una idea correcta y su fracaso es un resultado del TFM. El
patrón se repite (B1 y B3, las dos palancas que añaden features, empeoran; B2, que limpia la
etiqueta, ayuda un poco): **el cuello de botella no es la falta de features, es que la relación
factor→retorno es débil y el modelo lineal ya la exprime.**

---

## B5 — Régimen de mercado (bull/bear)

**Hipótesis.** Lo que predice el retorno en un mercado alcista no es lo mismo que en uno
bajista. Si el sistema sabe en qué régimen está (detectado solo con datos pasados del SP500) y
se le dan interacciones factor×régimen, el modelo puede aprender a ponderar distinto en cada
uno — más momentum en bull, más calidad/defensa en bear.

**Qué se probó.** Parámetro `market_regime_feature`. Régimen = SP500 con retorno a 12 meses
positivo (bull) o no (bear), leído del benchmark hasta la fecha del snapshot (sin lookahead).
Se añaden tres features al agente momentum: el régimen, `momentum × bull` y `quality × bear`.
Sobre la base B2 (`label_transform=rank`).

**Resultado medido (full, vs base rank +0.0011 / 0.534):**

| config | rank-IC medio | frac cohortes IC>0 | IR del IC |
|---|---|---|---|
| rank (base B2) | +0.0011 | 0.534 | +0.012 |
| rank + B5 | **+0.0015** | **0.536** | +0.012 |

**Es la mejor configuración**, por muy poco. Detalle por agente relevante: con el régimen,
**quality cruza a positivo** (+0.0035, antes −0.021) y value sube a +0.0125; a momentum le va
algo peor. El régimen ayuda a los agentes fundamentales.

**Decisión.** Mejora marginal (+0.0011 → +0.0015), dentro del ruido. Se **conserva** porque no
hace daño, tiene fundamento y ayuda a los fundamentales, pero **no resuelve el problema**:
+0.0015 sigue siendo un rank-IC ≈ 0.

---

## Cierre de la Parte B — veredicto

| palanca | rank-IC | frac IC>0 | decisión |
|---|---|---|---|
| baseline (none) | −0.0058 | 0.486 | referencia |
| B1 neutralización sector | −0.0083 | 0.467 | descartada (empeora) |
| B2 etiqueta rank | +0.0011 | 0.534 | **adoptada** |
| B3 momentum fundamentales | −0.0046 | 0.513 | descartada (empeora) |
| B5 régimen (sobre B2) | +0.0015 | 0.536 | conservada (marginal) |

**Lo mejor que se consigue tras todas las palancas: rank-IC +0.0015, fracción de cohortes
ganadoras 53.6 %.** Se ha pasado de "peor que el azar" a "empatado con el azar más un pelo".
**No es una señal explotable.** Las dos palancas que añaden información (sector, momentum de
fundamentales) empeoran; solo las que reordenan/limpian lo que ya hay (etiqueta rank, régimen)
ayudan, y muy poco.

**Conclusión metodológica.** Con datos gratuitos, universo del S&P 500 sesgado por
supervivencia, factores GARP+momentum lineales y un modelo Ridge, **el sistema no aprende a
ordenar activos fuera de muestra de forma útil.** El techo de este enfoque está en rank-IC ≈ 0.
Para obtener señal real haría falta un cambio de planteamiento (ver informe de conclusiones):
otro objetivo, otro universo, o capturar no-linealidades — no seguir puliendo esta vía.
