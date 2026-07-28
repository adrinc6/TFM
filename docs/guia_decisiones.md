# Guía de decisiones (para dummies)

Este documento explica **qué hace el sistema y por qué**, sin dar por hecho que sabes de
estadística, inversión o programación. Cada palabra técnica se explica la primera vez que aparece,
con un ejemplo cotidiano, y luego se usa sola. Todos los ejemplos numéricos son **inventados** para
ilustrar la mecánica; no son resultados reales del proyecto (esos están en `informe_resultados.md`).

Si quieres entender **qué pasaría si cambias tal o cual valor**, esta es la guía. Si quieres saber
**qué ha dado el proyecto realmente**, mira `docs/informe_resultados.md`. Si quieres el detalle
técnico exacto de cada regla, mira `docs/metodologia.md`.

---

## 1. Qué es todo esto, en una página

Imagina que tienes 250 acciones (las empresas del índice S&P 500) y quieres decidir, cada cierto
tiempo, cuáles comprar. En vez de decidirlo tú a ojo, montas un pequeño equipo de cinco analistas
especializados (les llamamos **agentes**) y un jefe de equipo que escucha a los cinco y decide
(el **meta-agente**). Cada analista mira una cosa distinta de cada empresa:

- **Quality (calidad)**: ¿la empresa gana dinero de forma sólida y eficiente?
- **Value (valor)**: ¿está barata comparada con lo que gana o con lo que vale su negocio?
- **Growth (crecimiento)**: ¿sus ventas y beneficios están acelerando?
- **Momentum**: ¿su precio ha subido con fuerza últimamente?
- **Risk (riesgo)**: ¿es una acción tranquila o que da bandazos?

Cada analista da una **nota de 0 a 100 relativa a las demás** (no una nota absoluta, sino "mejor o
peor que el resto del grupo ese día" — a esto lo llamamos **percentil** o **ranking**: si una acción
tiene percentil 90, significa que el 90 % de las demás acciones de ese momento quedaron por debajo
de ella según ese analista). El jefe de equipo (meta-agente) combina las cinco notas en una nota
final, y esa nota final decide qué acciones entran en la cartera.

Todo el proceso se entrena y se prueba con **datos históricos reales**, con una regla de oro: **el
sistema nunca puede ver el futuro**. Cuando se evalúa qué habría hecho el sistema en enero de 2018,
solo puede usar información que existía en enero de 2018 (esto se llama disciplina **point-in-time**,
o PIT: "en el momento", nunca con conocimiento posterior). Por ejemplo, los resultados trimestrales
de una empresa no se consideran disponibles el mismo día del cierre del trimestre, porque en la
vida real tardan semanas en publicarse — el sistema respeta ese retraso real.

Hay muchísimas formas de configurar este equipo (qué datos ve cada analista, cuánta historia
estudian, cómo decide el jefe, cómo se traduce todo eso en compras y ventas reales...). Cada una de
esas decisiones es una **variable del catálogo**: una lista cerrada de opciones válidas, para no caer
en la tentación de "probar cualquier cosa hasta que salga bien" (eso se llama *fuga de datos por
selección* y es una de las trampas más comunes de estos proyectos).

### Las cinco etapas

El catálogo agrupa las variables en 5 bloques, cada uno respondiendo a una pregunta:

| Etapa | Pregunta que responde |
|---|---|
| **Temporal** | ¿Cuándo se mira la información, cuánto futuro se intenta predecir y cuánta historia se estudia? |
| **Representación** | ¿Qué datos concretos ve cada analista y cómo se preparan antes de aprender? |
| **Modelo** | ¿Con qué "cerebro" matemático aprende cada analista? |
| **Meta-agente** | ¿Cómo combina el jefe de equipo las cinco notas? |
| **Cartera** | Una vez decidido el ganador, ¿cómo se traduce esa nota en comprar, vender o mantener efectivo? |

**Regla de diseño clave**: la etapa de Cartera **nunca puede cambiar quién gana** en las cuatro
etapas anteriores. Se decide primero qué combinación de "temporal + representación + modelo + meta"
ordena mejor las acciones (usando una métrica llamada **Rank-IC**, explicada en la sección 7), y solo
después, con ese ganador ya congelado, se prueban distintas formas de convertirlo en una cartera real
con dinero, comisiones y efectivo. Así, la elección del "cerebro" nunca se contamina por lo bien o
mal que le fue a la cartera con unos costes de operación concretos.

### La ventana de selección y la "era reservada"

Todo el proceso de elegir configuración se hace mirando datos de **2015 a 2024**. El periodo
**2025-2026** se guarda aparte y no participa en ninguna decisión — es lo que se llama la
**confirmación fuera de muestra**: si el sistema funciona igual de bien ahí, es una señal de que no
fue casualidad ("suerte" en el periodo de selección), porque el sistema nunca tuvo ocasión de
ajustarse a esos datos.

---

## 2. Variables temporales: cuándo se mira y qué se intenta predecir

### `snapshot_step_months` — cadencia de revisión

Cada cuánto tiempo el sistema vuelve a mirar y puntuar las 250 acciones desde cero. Piensa en ello
como "cada cuánto paso revista a mi cartera".

| Valor | Qué significa |
|---|---|
| **1 (mensual)** | Revisa todo cada mes. Reacciona rápido a cambios, pero genera muchas más decisiones y más coste de cálculo. |
| **3 (trimestral, recomendado)** | Revisa cada 3 meses, coincidiendo con cuándo las empresas suelen publicar resultados. Buen equilibrio. |
| **6 (semestral)** | Revisa dos veces al año. Más lento en detectar cambios. |
| **12 (anual)** | Revisa una vez al año. El más conservador en número de decisiones. |

**Ejemplo inventado**: imagina que "TechCorp" publica un mal trimestre en marzo. Con cadencia
mensual, el sistema lo detecta y puede reaccionar en el snapshot de abril. Con cadencia anual, si el
último snapshot fue en enero, puede que no vuelva a mirar TechCorp hasta el enero siguiente — se
pierde casi un año de reacción.

### `target_horizon_months` — cuánto futuro intenta predecir

El sistema no predice "qué va a pasar mañana", predice "qué acciones lo harán mejor que el mercado
en los próximos X meses".

| Valor | Qué significa |
|---|---|
| **3 meses** | Predicción a corto plazo. Más sensible al ruido de días o semanas concretas. |
| **6 meses** | Plazo intermedio. |
| **12 meses (recomendado)** | Predicción a un año. Suaviza el ruido de corto plazo, pero hay que esperar más para saber si acertó. |

**Ejemplo**: si el horizonte es de 3 meses, una acción que sube mucho en el mes 1 pero se desploma en
el mes 4 puede parecer un acierto (solo se mide hasta el mes 3). Con horizonte de 12 meses, ese mismo
patrón se vería como lo que realmente es: una subida seguida de una caída.

### `train_lookback_years` — cuánta historia estudia el modelo

Cada vez que el sistema se reentrena, ¿cuántos años de historia pasada usa como "libro de estudio"?

| Valor | Qué significa |
|---|---|
| **4 años** | Historia corta: se adapta rápido a lo reciente, pero con menos ejemplos para aprender de forma sólida. |
| **8 años (recomendado)** | Compromiso entre tener suficientes ejemplos y no arrastrar un pasado demasiado lejano. |
| **12 años** | Historia larga: más ejemplos, pero incluye épocas de mercado que quizá ya no se parecen al presente. |

**Ejemplo**: con 4 años de historia entrenando en 2021, el modelo nunca "vio" una subida de tipos de
interés como la de 2022-2023 salvo que ya estuviera dentro de esa ventana. Con 12 años, sí habría
visto la crisis de 2015-2016 en materias primas, aunque ese episodio ya esté bastante lejano.

### `execution_lag_days` — cuánto se tarda en "creer" un dato

Cuando una empresa cierra su trimestre fiscal, sus resultados no están disponibles para operar el
mismo día — hay que esperar a que se publiquen. Esta variable fija cuántos días de margen de
seguridad se asumen antes de considerar "disponible" ese dato.

| Valor | Qué significa |
|---|---|
| **30 días** | El supuesto más optimista: el sistema asume que ya puede operar con el dato solo 30 días después del cierre fiscal. Mayor riesgo de usar información que en la realidad todavía no existía (esto se llama **lookahead**: literalmente "mirar hacia adelante", usar datos del futuro sin darse cuenta). |
| **45 días** | Compromiso prudente. |
| **60 días (recomendado)** | El supuesto más conservador: se tarda más en "creer" el dato, pero es el que menos riesgo tiene de hacer trampa sin querer. |

**Ejemplo**: TechCorp cierra su trimestre fiscal el 30 de junio pero no publica sus resultados
oficiales (su *filing*, el documento legal que registra ante el regulador) hasta el 25 de agosto (56
días después). Con un lag de 30 días, el sistema habría asumido —incorrectamente— que esos resultados
ya estaban disponibles el 30 de julio: casi un mes antes de que existieran realmente. Con un lag de
60 días, el sistema no los usa hasta el 29 de agosto, después de que se publicaran de verdad.

### `recency_weighting` — cuánto pesa lo reciente

Al entrenar, ¿todas las observaciones históricas cuentan igual, o las más recientes cuentan más?

| Valor | Qué significa |
|---|---|
| **off (recomendado)** | Todo pesa igual, sea de hace un mes o de hace 8 años. |
| **linear** | Lo reciente pesa progresivamente más, de forma proporcional (como una recta que sube). |
| **exponential** | Lo reciente pesa mucho más y ese peso cae muy rápido cuanto más atrás se mira (como una curva empinada, no una recta). |

**Ejemplo**: con ponderación exponencial, un dato de hace 1 mes puede pesar 10 veces más que uno de
hace 2 años en la misma ventana de entrenamiento. El modelo "olvida" el pasado lejano mucho más
deprisa.

### `objective` — qué intenta aprender el modelo exactamente

| Valor | Qué significa |
|---|---|
| **rank_regression (recomendado)** | El modelo aprende a predecir un número (cuánto se espera que suba o baje una acción) y luego ese número se usa para ordenar. Es indirecto: intenta acertar una magnitud, no un orden. |
| **ranking** | El modelo aprende directamente a poner unas acciones por delante de otras, sin intentar acertar la magnitud exacta. Suele ajustar mejor a lo que realmente hace falta: un orden, no una cifra exacta. |

**Ejemplo**: si TechCorp sube un 8 % y PetroCorp sube un 3 %, con `ranking` el modelo solo necesita
aprender "TechCorp por delante de PetroCorp"; con `rank_regression` intenta además acertar que fue
"8 %" y "3 %" exactamente, una tarea más difícil y menos necesaria para el objetivo real (ordenar).

---

## 3. Variables de representación: qué datos ve cada analista

### `feature_preset` — cuánta información recibe cada analista

Este es uno de los más importantes: define el conjunto de "columnas de datos" (llamadas **features**,
literalmente "características") que recibe cada uno de los cinco agentes. Solo hay dos opciones en el
catálogo, deliberadamente: **ambas deben alimentar a los cinco agentes**, para no dejar a ninguno sin
información (una versión anterior tenía presets que dejaban a algunos agentes completamente vacíos, y
eso invalidaba la comparación: no medía "cuánta información hace falta", medía "qué pasa si le
amputo un brazo al sistema").

| Valor | Qué significa |
|---|---|
| **core (recomendado)** | Cada agente recibe solo lo más esencial y directo de su especialidad. |
| **all** | Cada agente recibe todo lo disponible en su especialidad, con más matices pero también más ruido y más coste de cálculo. |

**Ejemplo con dos acciones inventadas**: "TechCorp" (tecnológica cara, de crecimiento) y "PetroCorp"
(petrolera barata, estable).

| Bloque de datos | TechCorp bajo `core` | TechCorp bajo `all` | PetroCorp bajo `core` | PetroCorp bajo `all` |
|---|---|---|---|---|
| Rentabilidad básica (ROE) | ✅ | ✅ | ✅ | ✅ |
| Eficiencia operativa y solidez de balance | ❌ | ✅ | ❌ | ✅ |
| Múltiplos de precio básicos (PER) | ✅ | ✅ | ✅ | ✅ |
| Múltiplos de flujo de caja | ❌ | ✅ | ❌ | ✅ |
| Aceleración de crecimiento | ✅ | ✅ | ✅ | ✅ |
| Estabilidad de fundamentales | ❌ | ✅ | ❌ | ✅ |
| Retorno relativo reciente (momentum) | ✅ | ✅ | ✅ | ✅ |
| Tendencia frente a medias móviles | ❌ | ✅ | ❌ | ✅ |
| Volatilidad y caídas de precio | ✅ | ✅ | ✅ | ✅ |
| Liquidez de mercado | ❌ | ✅ | ❌ | ✅ |

Bajo `core`, TechCorp y PetroCorp se juzgan solo por lo esencial de cada especialidad. Bajo `all`,
además se mira si TechCorp genera caja de verdad (no solo beneficio contable) y si PetroCorp tiene
un balance sólido que aguante un ciclo de precios bajos del petróleo — matices que `core` no ve.

### `fundamental_momentum` — ¿ver la tendencia o solo la foto?

| Valor | Qué significa |
|---|---|
| **True (recomendado)** | Se añade si un dato fundamental (por ejemplo el ROE) está mejorando o empeorando trimestre a trimestre, no solo su valor actual. |
| **False** | El modelo solo ve el valor de hoy, sin saber si viene mejorando o empeorando. |

**Ejemplo**: PetroCorp tiene un ROE del 12 % hoy. Con esta variable activada, el sistema también sabe
que hace un año era del 8 % (mejorando) — una información muy distinta a si viniera de un 18 % (en
declive), aunque el valor de "hoy" sea idéntico.

### `market_regime_feature` — ¿saber si el mercado en general está alcista o bajista?

| Valor | Qué significa |
|---|---|
| **True (recomendado)** | El modelo recibe una pista sobre si el mercado en conjunto está en un momento alcista o bajista, usando solo información ya disponible en ese momento. |
| **False** | El modelo no tiene ningún contexto general, solo ve datos propios de cada empresa. |

### `neutralize_by_sector` — ¿comparar dentro del mismo sector?

| Valor | Qué significa |
|---|---|
| **True** | Cada empresa se compara solo contra otras de su mismo sector (tecnológicas contra tecnológicas). Evita confundir "esta empresa es buena" con "todo su sector está de moda". |
| **False (recomendado)** | Todas las empresas se comparan entre sí sin ajustar por sector. |

**Ejemplo**: en un año en que todas las petroleras suben porque el petróleo está caro, sin
neutralización PetroCorp podría parecer "buena" solo por estar en el sector correcto en el momento
correcto, no por ser mejor que sus competidoras.

### `winsorization` — recortar los valores extremos

Cuando un dato es disparatado (por ejemplo, un PER —precio dividido entre beneficio— de 5.000 porque
el beneficio es casi cero), puede distorsionar el aprendizaje. Esta variable "aplana" los extremos.

| Valor | Qué significa |
|---|---|
| **0,0 (sin recorte, recomendado)** | No se toca ningún valor extremo. |
| **0,01 (1 %)** | El 1 % más alto y el 1 % más bajo de cada variable se recortan al límite de ese grupo (llamado **percentil**). |
| **0,025 (2,5 %)** | Recorte más agresivo. |

**Ejemplo**: si el PER de "EmpresaRara" es 5.000 (por un beneficio casi nulo) y el percentil 99 real
del resto de empresas es 45, con winsorización al 1 % el PER de EmpresaRara se "aplana" a 45 antes de
entrenar, en vez de dejar que ese número disparatado confunda al modelo.

### `max_features_per_agent` — cuántas variables usa cada analista como máximo

| Valor | Qué significa |
|---|---|
| **8 (recomendado)** | Cada agente se queda solo con las 8 variables más relevantes de su bloque. Más simple, menos riesgo de aprender ruido. |
| **12** | Punto intermedio. |
| **20** | Casi sin recortar. Más rico en información, pero más lento y con más riesgo de sobreajuste (aprender de memoria detalles del pasado que no se repetirán, en vez de un patrón real — como un estudiante que memoriza las respuestas de un examen concreto en vez de entender la materia). |

### `feature_weighting_mode` — quién decide qué variables importan más

| Valor | Qué significa |
|---|---|
| **model_native** | Se deja que el propio modelo decida internamente qué variables usar más, sin filtro previo. |
| **oos_stability_prune (recomendado)** | Antes de entrenar el modelo final, se descartan las variables cuya utilidad no se mantiene estable cuando se prueban en datos que el modelo no vio al aprender ese filtro (esto se llama **fuera de muestra**, u *out-of-sample*: la parte de datos reservada para comprobar, no para aprender). Solo sobreviven las variables realmente consistentes. |

---

## 4. Variables de modelo: el "cerebro" de cada analista

### `model_family` — qué tipo de modelo matemático se usa

| Valor | Qué significa |
|---|---|
| **lightgbm (recomendado)** | Un modelo basado en muchos árboles de decisión combinados (imagina cientos de diagramas de flujo tipo "si el ROE es mayor que X, mira el PER; si no, mira otra cosa", combinados entre sí). Puede capturar relaciones complicadas y no lineales, a cambio de ser menos fácil de interpretar. |
| **elastic_net** | Un modelo lineal (imagina una fórmula simple tipo "puntuación = 0,3 × ROE + 0,5 × crecimiento − 0,2 × volatilidad"). Más simple e interpretable, asume que las relaciones son aproximadamente rectas, y es más resistente a aprender ruido cuando hay pocos datos. |

**Ejemplo**: imagina que el efecto del crecimiento en la puntuación **depende** de si la empresa
también es rentable (crecer sin ganar dinero no vale lo mismo que crecer ganando dinero). Un modelo
lineal como Elastic Net tiene más dificultad para capturar esa "interacción" entre dos variables;
LightGBM, al construir árboles con preguntas encadenadas, puede aprenderla de forma más natural.

### `lgbm_max_depth`, `lgbm_n_estimators`, `lgbm_learning_rate`, `lgbm_min_child_samples`

Estos cuatro solo aplican si `model_family = lightgbm` (por eso se dice que **dependen** de esa
variable: no tiene sentido ajustar la profundidad de un árbol si no hay árboles).

| Variable | Qué controla | Valor bajo | Valor alto |
|---|---|---|---|
| `lgbm_max_depth` | Cuántas "preguntas encadenadas" puede hacer cada árbol antes de decidir | 3: árboles simples, menos riesgo de memorizar ruido | 6: capturan relaciones más complejas, más riesgo de sobreajuste |
| `lgbm_n_estimators` | Cuántos árboles se combinan | 100: entrena rápido, puede quedarse corto de capacidad | 400: más capacidad, más lento, más riesgo si no se controla bien lo demás |
| `lgbm_learning_rate` | Qué tan grande es el paso de corrección de cada árbol nuevo | 0,03: aprendizaje lento pero estable | 0,10: aprendizaje rápido, más riesgo de "pasarse" |
| `lgbm_min_child_samples` | Cuántos ejemplos como mínimo hacen falta para que un árbol cree una regla final | 20: reglas muy específicas, más riesgo de que sean ruido de pocos casos | 100: solo reglas respaldadas por muchos casos, más robustas pero más generales |

**Ejemplo con TechCorp y PetroCorp**: con `lgbm_max_depth = 3`, el modelo solo puede hacer preguntas
del tipo "¿el crecimiento es alto? → ¿la calidad es alta?" (dos niveles). Con `lgbm_max_depth = 6`,
puede además preguntar "¿y además el sector está barato en su conjunto? ¿y el momentum reciente es
positivo?" — más matices, pero también más riesgo de construir una regla que solo funcionó por
casualidad para las empresas concretas del pasado.

---

## 5. Variables de meta-agente: cómo decide el jefe de equipo

### `meta_method` — cómo se combinan las cinco notas

| Valor | Qué significa |
|---|---|
| **equal** | Los cinco agentes pesan siempre lo mismo: 20 % cada uno, sin importar quién lo esté haciendo mejor. Lo más simple y difícil de sobreajustar, pero no aprovecha que unos agentes puedan ser más fiables en cada momento. |
| **stacked_rolling_free** | El meta-agente aprende automáticamente (con una técnica llamada **regresión Ridge**, que es básicamente "encontrar los pesos que mejor explican los resultados pasados, con un freno para no exagerar"), usando solo información ya cerrada y pasada, qué peso dar a cada agente. Sin límite: en teoría un agente podría acaparar el 100 % del peso. |
| **stacked_rolling_bounded (recomendado)** | Igual que la anterior, pero obligado a que cada agente pese entre el 10 % y el 50 %. Evita los dos extremos: ni ignora a nadie del todo, ni depende de uno solo. |

**Ejemplo inventado con "TechCorp"**: los cinco agentes dan estas notas (percentil, de 0 a 100) a
TechCorp en un snapshot concreto:

| Agente | Nota a TechCorp |
|---|---|
| Quality | 55 |
| Value | 20 (está cara) |
| Growth | 90 |
| Momentum | 85 |
| Risk | 95 (muy poco volátil) |

- Bajo **equal**: nota final = (55+20+90+85+95)/5 = **69**.
- Bajo **stacked_rolling_bounded**, imaginando que el histórico dice que Risk y Momentum han sido los
  agentes más fiables últimamente (pesos aprendidos p. ej. Risk 45 %, Momentum 30 %, Quality 15 %,
  Growth 5 %, Value 5 %, dentro del límite 10-50 %): nota final ≈ 0,45×95 + 0,30×85 + 0,15×55 +
  0,05×90 + 0,05×20 = **85,25**. Mucho más alta, porque pesa fuerte a los agentes que más han
  acertado.
- Bajo **stacked_rolling_free** (sin límites), en un caso extremo el sistema podría concluir que solo
  Risk importa (peso 95 %, el resto casi 0 %): nota final ≈ **95**, casi ignorando a los otros cuatro
  analistas por completo. Es el riesgo de la versión libre: puede llegar a depender casi en
  exclusiva de un único agente.

### `meta_history_quarters` — cuánta historia usa el jefe para decidir los pesos

| Valor | Qué significa |
|---|---|
| **8 (2 años)** | Ventana corta: se adapta más rápido a qué agente está funcionando mejor recientemente, con menos datos para decidirlo con solidez. |
| **16 (4 años, recomendado)** | Ventana más larga: decisión más estable, pero tarda más en reaccionar si un agente empieza a fallar. |

---

## 6. Selección secuencial: cómo se elige el ganador entre configuraciones

### El vocabulario mínimo

- **Rank-IC** (coeficiente de información del ranking): mide qué tan bien el orden que da el sistema
  coincide con el orden real de quién subió más. Va de −1 (orden completamente al revés) a +1 (orden
  perfecto), pasando por 0 (sin relación, como tirar un dado). Un Rank-IC de 0,07 suena pequeño, pero
  en mercados financieros —donde nada se predice con precisión— ya es una señal real y valiosa si se
  mantiene de forma consistente mes tras mes.
- **Bootstrap** (de "tirar de las botas", en el sentido de "arreglárselas con lo que hay"): una
  técnica para saber si un resultado es fiable o pura casualidad, repitiendo miles de veces un
  sorteo con reemplazo sobre los mismos datos históricos para ver cuánto varía el resultado. Si al
  repetir el sorteo 2.000 veces casi siempre sale un Rank-IC positivo, es una señal fuerte de que no
  es casualidad.
- **Intervalo de confianza**: el rango donde probablemente cae el valor real, según ese sorteo
  repetido. Si el intervalo es "entre 0,03 y 0,11", es poco probable que el verdadero Rank-IC sea 0
  o negativo.
- **Incumbente y retador**: el incumbente es la configuración actual, "la que manda"; el retador es
  una alternativa candidata a sustituirla.

### La regla para cambiar de incumbente

Cada variable del catálogo se prueba comparando el retador contra el incumbente sobre las mismas
fechas históricas (emparejadas mes a mes). Un retador gana la plaza si:

1. **Domina claramente**: en promedio saca mejor Rank-IC que el incumbente, Y además gana en más de
   la mitad de los meses comparados (no solo por un par de meses excepcionales). O bien:
2. **No es peor** (técnicamente, **no inferior**): aunque no domine claramente, el límite inferior de
   su intervalo de confianza queda por encima de un margen de tolerancia — es decir, ni en el peor
   escenario razonable resulta claramente peor que el incumbente.

Si ninguna de las dos se cumple, gana el incumbente (se queda como estaba). Si dos configuraciones
"empatan" (su diferencia es minúscula, dentro de un margen de tolerancia), gana la más simple —
porque entre dos opciones igual de buenas, no hay motivo para preferir la más complicada.

### Los cuatro desenlaces posibles, con números inventados

Imagina que se compara `feature_preset = core` (incumbente) contra `feature_preset = all` (retador),
sobre 117 meses históricos.

**(a) El retador domina claramente.** Rank-IC medio de `all` = 0,096 frente a 0,073 de `core`;
`all` gana en el 59 % de los meses comparados. → **Gana `all`.**

**(b) El retador no domina, pero tampoco es claramente peor (no inferioridad).** Rank-IC medio de
`all` = 0,075 frente a 0,073 de `core`, prácticamente empatados; `all` gana solo en el 51 % de los
meses (casi monedas al aire) pero el intervalo de confianza de la diferencia es
`[−0,004; +0,008]` — el límite inferior queda dentro del margen de tolerancia. → **Gana `all`**, por
no inferioridad, aunque no haya "arrasado".

**(c) El retador es realmente peor.** Rank-IC medio de `all` = 0,058 frente a 0,073 de `core`; el
intervalo de confianza de la diferencia es `[−0,031; −0,002]`, claramente por debajo de cero y fuera
del margen de tolerancia. → **Gana `core`** (el incumbente se queda).

**(d) Las rejillas no se pueden comparar.** Si se está probando, por ejemplo, cambiar el retardo de
publicación de datos (`execution_lag_days`) de 30 a 60 días, las fechas exactas en las que el sistema
revisa la cartera cambian tanto que apenas quedan meses en común para comparar de forma fiable. En
ese caso no hay evidencia suficiente para decidir nada: el sistema lo marca como **"no aplicable"** y,
por precaución, se queda con el incumbente (nunca se deja pasar un candidato automáticamente solo
porque falte la comparación).

---

## 7. Cartera: cómo se convierte el ranking en compras, ventas y efectivo

Esta es la parte que más detalle pediste, así que vamos despacio.

### Primero, una idea clave: el "alfa esperado"

Cada acción tiene un **percentil** (su posición en el ranking del meta-agente, de 0 a 100). Pero un
percentil no dice **cuánto dinero se espera ganar** — solo dice "mejor o peor que las demás". Para
poder comparar contra el coste real de comprar y vender, el sistema traduce cada percentil en una
cifra concreta: el **alfa esperado**, medido en **puntos básicos** (pb). Un punto básico es una
centésima de un uno por ciento: 100 pb = 1 %, 250 pb = 2,5 %. Esta traducción se hace mirando, de
forma honesta y solo con datos ya cerrados del pasado, "históricamente, ¿cuánto de mejor lo hicieron
las acciones que tenían este percentil, frente al mercado?" — así el percentil 90 de hoy se traduce
en algo como "se espera que esta acción bata al mercado en unos 180 pb (1,8 %) en el horizonte del
modelo".

Mientras el sistema no tiene suficiente historia cerrada para hacer esa traducción con solidez (por
ejemplo, nada más arrancar), el alfa esperado queda "sin dato" (técnicamente **NaN**, "no es un
número"). Es importante: un "sin dato" **nunca** dispara una venta ni bloquea una compra —solo manda
si hay evidencia real—; durante el arranque, manda simplemente el orden del ranking.

### El principio único que gobierna todo

**Una venta solo se hace si el destino del dinero es mejor que quedarse donde está, una vez descontado
lo que cuesta operar.** Vender y comprar no es gratis: cada operación paga una comisión (un
porcentaje fijo que se lleva el bróker) y un **slippage** (la diferencia entre el precio que
esperabas y el precio real al que se ejecuta la operación, porque el mercado se mueve mientras
operas). Al coste de vender y luego volver a comprar algo se le llama **ida y vuelta** (*round
trip*), y siempre son dos costes, no uno.

Solo hay dos destinos posibles para el dinero de una venta, y cada uno tiene su propia regla:

1. **Otra acción** (esto se llama **rotación**): solo se sustituye la peor posición por una acción
   de fuera si la ventaja de alfa esperado de la nueva supera el coste de la ida y vuelta más un
   margen extra de seguridad.
2. **Efectivo** (solo posible si se activa la política `opportunity_cash`, explicada más abajo): solo
   se deja una plaza vacía si el alfa esperado de la posición cae por debajo de un umbral mínimo y,
   además, hacerlo no viola un mínimo de diversificación (explicado más abajo).

Y las compras nuevas tienen una zona de seguridad extra (llamada **histéresis**, palabra que
significa literalmente "quedarse atrás", como cuando un termostato no enciende la calefacción en el
instante exacto en que baja de 20 grados, sino un poco después, para no encender y apagar todo el
rato): **entrar** a una posición exige más alfa esperado que **mantenerse** en ella. Sin esa zona de
seguridad, una acción con un alfa esperado justo en la frontera se compraría y se vendería en meses
consecutivos, pagando comisiones sin ganar nada a cambio.

### Las 9 variables de cartera

| Variable | Qué decide |
|---|---|
| `target_size` | Cuántas acciones distintas mantiene la cartera a la vez. |
| `exit_expected_alpha_bps` | Por debajo de qué alfa esperado (en pb) se considera que una posición "ya no vale la pena". |
| `rotation_edge_bps` | Cuánta ventaja extra (por encima del coste de ida y vuelta) exige una rotación para autorizarse. |
| `cash_policy` | Si la cartera está siempre invertida al 100 % (`fully_invested`) o puede dejar huecos en efectivo (`opportunity_cash`). |
| `max_cash_weight` | Cuánto efectivo como máximo se permite bajo `opportunity_cash`. |
| `rebalance_drift_tolerance` | Cuánto puede desviarse el peso real de una posición de su peso objetivo antes de generar una orden de ajuste. |
| `price_only_strictness_multiplier` | Cuánto más exigentes se vuelven los umbrales en meses donde solo hay precio nuevo, sin resultados financieros nuevos. |
| `sizing_mode` | Si todas las posiciones pesan igual (`equal`) o si pesan más las de mayor alfa esperado (`alpha_proportional`). |
| `commission_bps` / `slippage_bps` | Los costes reales que se descuentan por cada operación. |

### `target_size` — cuántas acciones a la vez

| Valor | Qué significa |
|---|---|
| **8** | Cartera concentrada: acertar o fallar en una sola acción pesa mucho sobre el resultado total. |
| **12 (recomendado)** | Diversificación intermedia. |
| **16** | Más diversificada: cada acierto o fallo individual pesa menos. |
| **25** | Más amplitud (*breadth*, "anchura"): al repartir entre más apuestas independientes, se recupera señal que una cartera muy concentrada desperdicia (existe un principio llamado la "ley fundamental de la gestión activa" que dice, resumido, que cuantas más apuestas independientes hagas con la misma habilidad de selección, mejor resultado ajustado a riesgo obtienes). |
| **50** | Máxima amplitud: cada acción pesa poco, el resultado depende de la calidad media de todo el ranking. Se parece más al índice general, con menos posibilidad de un resultado muy distinto (para bien o para mal). |

### `exit_expected_alpha_bps` — el umbral de salida

| Valor | Qué significa |
|---|---|
| **0 pb** | Solo se vende cuando el alfa esperado se vuelve negativo (se espera que la acción lo haga peor que el mercado). Es el criterio que menos opera. |
| **100 pb (1 %, recomendado)** | Se exige una expectativa positiva clara (al menos un 1 % mejor que el mercado) para seguir ocupando una plaza. |
| **250 pb (2,5 %)** | Solo se conservan convicciones fuertes. Más rotación y más coste. |

### `rotation_edge_bps` — el margen extra para rotar

| Valor | Qué significa |
|---|---|
| **25 pb** | Rota con relativa facilidad, siempre que cubra al menos el coste de operar más 25 pb. |
| **50 pb (recomendado)** | Exigencia intermedia. |
| **100 pb** | Solo rota ante una mejora económica clara. Menos rotación y menos coste. |

### `cash_policy` y `max_cash_weight` — la política de efectivo

| Valor de `cash_policy` | Qué significa |
|---|---|
| **fully_invested (recomendado)** | La cartera está siempre invertida al 100 %. Si una plaza queda libre, se rellena con la mejor candidata disponible aunque su alfa esperado sea bajo. Es la política de referencia, la que se compara "a pelo" contra el índice. |
| **opportunity_cash** | Cuando ninguna candidata supera el umbral de salida, la plaza se deja en efectivo en lugar de comprar "la menos mala". |

| Valor de `max_cash_weight` (solo aplica bajo `opportunity_cash`) | Qué significa |
|---|---|
| **0,0** | Sin efectivo permitido: equivale de hecho a estar siempre invertido. |
| **0,10 (10 %)** | Como mucho el 10 % de la cartera puede quedar en efectivo. |
| **0,25 (25 %, recomendado)** | Como mucho el 25 % puede quedar en efectivo. |

**Dos salvaguardas importantes bajo `opportunity_cash`**:

1. **El efectivo nunca supera el tope.** Con `max_cash_weight = 0,25`, jamás más de un cuarto de la
   cartera queda sin invertir.
2. **Nunca se baja de un mínimo de posiciones** (el **suelo de diversificación**), calculado como
   "cuántas plazas hacen falta para que el 75 % restante no se concentre en muy pocos nombres". Por
   ejemplo, con `target_size = 4` y `max_cash_weight = 0,25`, el suelo son **3 posiciones**: como
   mínimo 3 de las 4 plazas deben estar ocupadas, incluso si la cuarta candidata no supera el umbral,
   porque de lo contrario una única posición admisible podría acabar concentrando el 75 % de la
   cartera ella sola.

**El efectivo se remunera al 0 %**: es una suposición deliberadamente conservadora — el efectivo
nunca aporta rentabilidad por sí mismo, solo puede ayudar evitando comprar algo malo y ahorrando
comisiones. Si aun con esa desventaja el efectivo mejora el resultado final, esa mejora es
incontestable. Y la decisión de ir a efectivo sale **exclusivamente** de comparar las acciones entre
sí (nunca de una previsión sobre si el mercado en general va a subir o bajar) — eso es importante
porque, si dependiera de una previsión de mercado, dejaría de ser "seguir la señal de las acciones" y
pasaría a ser una apuesta sobre el rumbo general del mercado (lo que se llama **market timing**,
"adivinar el momento del mercado"), algo que este proyecto no pretende hacer ni medir.

### `sizing_mode` — cómo se reparte el peso entre posiciones

| Valor | Qué significa |
|---|---|
| **equal** | Todas las posiciones pesan exactamente igual, sin importar si una tiene mucho mejor alfa esperado que otra. |
| **alpha_proportional (recomendado)** | El peso escala con el alfa esperado: la posición con menor alfa esperado **dentro de la cartera actual** recibe un peso base, y la de mayor alfa esperado recibe el doble (un tope de 2 a 1, para que ninguna posición se dispare de forma desproporcionada). |

### `rebalance_drift_tolerance` — cuánta desviación se tolera antes de operar

| Valor | Qué significa |
|---|---|
| **0,0** | Cualquier desviación, por mínima que sea, entre el peso real y el objetivo genera una orden de ajuste. Máxima precisión, muchas más operaciones y comisiones. |
| **0,10** | Solo se ajusta si la desviación supera el 10 % (en términos relativos al peso objetivo). |
| **0,25 (recomendado)** | Tolerancia amplia: pocas operaciones de puro ajuste, pero los pesos reales pueden alejarse bastante del objetivo teórico entre ajustes. |

### `price_only_strictness_multiplier` — prudencia cuando solo hay precio nuevo

En algunos meses solo se actualiza el precio de la acción, sin resultados financieros nuevos que lo
respalden (los resultados trimestrales no salen todos los meses). Esta variable multiplica los
umbrales de venta y compra en esos meses, para no mover la cartera solo por ruido de precio sin
confirmación de fondo.

| Valor | Qué significa |
|---|---|
| **1,0** | Sin prudencia extra: mismos umbrales siempre. |
| **1,5 (recomendado)** | Umbrales un 50 % más exigentes en meses sin resultados nuevos. |
| **2,0** | Umbrales el doble de exigentes: casi no se opera solo por movimiento de precio. |

### `commission_bps` y `slippage_bps` — los costes reales

| Comisión | Qué significa |
|---|---|
| 0 pb | Sin comisión (escenario idealizado, sobreestima el resultado real). |
| 5 pb (recomendado) | Coste de bróker moderado. |
| 10 pb | Coste de bróker más alto. |

| Slippage | Qué significa |
|---|---|
| 5 pb | Supuesto optimista de liquidez. |
| 10 pb (recomendado) | Supuesto intermedio. |
| 20 pb | Supuesto conservador: penaliza más a estrategias que rotan mucho. |

### Ejemplo completo, paso a paso: un snapshot inventado

Imagina una cartera con `target_size = 4`, comisión 5 pb, slippage 10 pb (coste de ida y vuelta:
2 × (5+10) = **30 pb**), umbral de salida `exit_expected_alpha_bps = 100 pb` y ventaja de rotación
`rotation_edge_bps = 50 pb`.

Cartera actual (4 posiciones) con su alfa esperado de este mes, y dos candidatas de fuera:

| Acción | En cartera? | Alfa esperado (pb) |
|---|---|---|
| A | Sí | 320 |
| B | Sí | 180 |
| C | Sí | 90 |
| D | Sí | 60 |
| E (fuera) | No | 250 |
| F (fuera) | No | 70 |

**Caso 1 — política `fully_invested`.**

- **Rotación**: la peor posición en cartera es D (60 pb). La mejor candidata de fuera es E (250 pb).
  Ventaja de E sobre D = 250 − 60 = **190 pb**. El umbral para rotar es
  coste de ida y vuelta (30 pb) + margen (50 pb) = **80 pb**. Como 190 > 80, **se vende D y se compra
  E**.
- Con `fully_invested`, no existe la venta a efectivo: si alguna posición quedara por debajo del
  umbral de salida (100 pb) sin un reemplazo mejor disponible, simplemente se mantiene — vender sin
  destino mejor sería pagar un coste para no ganar nada.
- Resultado: cartera = {A, B, C, E}, sin efectivo.

**Caso 2 — el mismo snapshot con `opportunity_cash` y `max_cash_weight = 0,25`.**

Con 4 plazas y tope del 25 %, el suelo de diversificación es `techo(0,75 × 4) = 3` posiciones: como
mínimo 3 de las 4 plazas deben quedar ocupadas.

- Primero ocurre la misma rotación de arriba (D sale, E entra): {A, B, C, E}.
- Ahora, con el nuevo alfa esperado de C (90 pb), que está por debajo del umbral de salida (100 pb):
  ¿se puede vender a efectivo? Antes de esta venta hay 4 posiciones; venderla dejaría 3, que es
  exactamente el suelo permitido. **Sí se puede vender.**
- Resultado: cartera = {A, B, E} invertidas, **C se vende y su plaza queda en efectivo** (25 % de la
  cartera en efectivo, justo en el tope). Si además D también hubiera estado por debajo del umbral,
  no se le habría permitido salir también a efectivo, porque dejaría la cartera en solo 2 posiciones,
  por debajo del suelo de 3 — el sistema la mantendría invertida a la fuerza para no concentrar de
  más ni pasarse del tope de efectivo.

**Caso 3 — la banda de histéresis, con una acción nueva llamada "G".**

G no está en cartera y tiene un alfa esperado de **105 pb**. El umbral de salida es 100 pb, pero el
umbral de **entrada** para una compra nueva es el umbral de salida más el coste de ida y vuelta:
100 + 30 = **130 pb**. Como 105 < 130, **G no se compra** aunque su alfa esperado (105 pb) esté por
encima del umbral de salida.

Pero si G **ya estuviera** en la cartera con esos mismos 105 pb, no se vendería, porque 105 sí supera
el umbral de mantenerse (100 pb). Esta asimetría a propósito (más exigente para entrar que para
quedarse) es lo que evita comprar y vender la misma acción mes tras mes por estar justo en la
frontera.

**Caso 4 — repartiendo pesos con `sizing_mode = alpha_proportional`.**

Con la cartera final {A: 320 pb, B: 180 pb, E: 250 pb} (caso 1, sin C), el reparto de pesos escala
entre el mínimo (B, 180 pb → peso base ×1) y el máximo (A, 320 pb → peso doble ×2), con E a mitad de
camino (250 pb → aproximadamente ×1,5). Sumando esos "puntos" (1 + 1,5 + 2 = 4,5) y repartiendo el
100 % proporcionalmente: A recibe ≈ 44 %, E ≈ 33 %, B ≈ 22 %. La que tenía el doble de alfa esperado
que la peor de la cartera recibe el doble de peso — nunca más, por el tope de 2 a 1.

### Nota importante: por qué el efectivo puede parecer "todo o nada"

La traducción de percentil a alfa esperado se hace en 20 tramos (llamados **ventiles**, como
percentiles pero agrupados de 5 en 5 puntos). Con una cartera concentrada de 12 posiciones sobre 250
acciones, todas las posiciones de la cartera viven casi siempre en el tramo más alto (el ventil 20),
así que sus alfas esperados son casi idénticos entre sí. Esto significa que, en la práctica, con
carteras pequeñas el efectivo tiende a comportarse de forma casi binaria: o todas las candidatas
superan el umbral (nada de efectivo) o ninguna lo hace (efectivo al máximo permitido), respondiendo
sobre todo a si la señal en general ha estado funcionando bien últimamente, más que a diferencias
finas entre acciones concretas ese mes. Esta granularidad se vuelve más suave con carteras de 25 o 50
posiciones, donde sí se cruzan varios tramos distintos dentro de la misma cartera.

---

## 8. Perfiles de inversor: mirar el mismo ranking con otros ojos

Una vez que el meta-agente ha decidido qué acciones son "buenas" (las que superan el percentil 60,
llamado el **umbral de calidad**), los **perfiles** no cambian esa lista de buenas — solo la
**reordenan** dentro de ella, según distintos estilos de inversión clásicos. Es una forma de
preguntar "¿cómo le habría ido a un inversor con estilo X, usando exactamente la misma información
que ya calcularon los cinco agentes?". No reentrenan nada ni miran al futuro: son una combinación
distinta de las mismas cinco notas.

| Perfil | Cómo pondera las notas de los agentes |
|---|---|
| **balanced** | Usa directamente la nota final del meta-agente, sin sesgo de estilo. Es la referencia. |
| **growth** | 60 % crecimiento, 25 % calidad, 15 % momentum. Crecimiento como motor, pero exige que sea un crecimiento "de calidad", confirmado por la tendencia de precio. |
| **value** | 70 % valor (barata), 15 % calidad, 15 % riesgo (baja volatilidad). La calidad y el riesgo actúan de filtro para evitar las "trampas de valor" (empresas baratas que lo están porque tienen un problema real). |
| **quality** | 70 % calidad, 20 % crecimiento, 10 % valor. El estilo "comprar negocios excelentes y duraderos". |
| **momentum** | 75 % momentum, −25 % riesgo (peso **negativo**: acepta más volatilidad a cambio de perseguir la fuerza del precio). |
| **contrarian** | −55 % momentum (peso negativo: apuesta **en contra** de la tendencia reciente), 30 % valor, 15 % riesgo. Compra lo que ha caído, en nombres baratos, vigilando que no sea una caída justificada. |
| **defensive** | 60 % riesgo (baja volatilidad), 35 % calidad, 5 % valor. Preservar el capital ante todo. |
| **garp** ("crecimiento a precio razonable") | 40 % crecimiento, 35 % valor, 25 % calidad. Crece, pero con disciplina de precio. |

**Ejemplo con "MiningCorp"**, una minera que ha caído mucho de precio en los últimos meses:

- Sus notas: Quality 40, Value 85 (muy barata tras la caída), Growth 30, Momentum 15 (cayendo con
  fuerza), Risk 45 (bastante volátil).
- Perfil **momentum** (75 % momentum, −25 % risk): con momentum tan bajo (15) y un peso negativo en
  riesgo que penaliza su alta volatilidad, MiningCorp queda **muy abajo** en este perfil — un inversor
  momentum huiría de ella.
- Perfil **contrarian** (−55 % momentum, 30 % value, 15 % risk): el mismo momentum bajo (15) se
  convierte en un peso **negativo**, así que aporta positivamente a la puntuación; sumado a su Value
  alto (85), MiningCorp queda **muy arriba** en este perfil — justo lo que busca un inversor
  contrarian: comprar lo que otros están vendiendo.

La misma acción, con las mismas cinco notas objetivas, puede ser el favorito de un perfil y el
descarte del perfil opuesto. Eso es exactamente lo que estos perfiles están diseñados para mostrar.

---

## 9. Lo que nunca puede pasar (garantías de diseño)

Una lista corta de reglas que el sistema cumple siempre, por construcción, y que puedes usar para
comprobar tú mismo que el sistema se comporta como se espera:

- **El índice de referencia (SPY) nunca es una posición de la cartera.** Solo se usa como vara de
  medir ("¿lo hice mejor o peor que simplemente comprar todo el mercado?").
- **La cartera nunca queda vacía si hay datos ese mes.** Si el sistema tiene puntuaciones para ese
  snapshot, siempre habrá al menos una posición o el efectivo estará dentro de su tope, nunca un
  "no sé qué hacer".
- **El efectivo nunca supera el tope configurado** (`max_cash_weight`), ni por error de redondeo ni
  por un mes especialmente malo de candidatas.
- **Nunca se vende una acción solo para volver a comprarla en el mismo mes.** Toda venta tiene que
  tener un destino mejor de verdad (otra acción o efectivo justificado), nunca es un "vender por
  vender".
- **El periodo 2025-2026 nunca participa en ninguna decisión de qué configuración gana.** Solo se usa
  después, para comprobar (una sola vez, sin repetir el experimento) si el ganador ya elegido se
  sostiene fuera de la ventana en la que fue elegido.
- **El efectivo nunca rinde más del 0 %.** Es una suposición deliberadamente pesimista: si el
  efectivo ayuda al resultado final, es solo por evitar malas compras y ahorrar comisiones, nunca
  porque "el efectivo generó dinero por sí mismo".
- **Ninguna decisión de efectivo se basa en una previsión sobre el mercado en general.** Sale
  siempre de comparar las acciones candidatas entre sí, nunca de "creo que el mercado va a bajar".
