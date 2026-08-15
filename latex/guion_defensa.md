# Guion de defensa — 15 minutos

> Acompaña a `latex/presentacion.tex`. El mismo texto vive en los `\note{}` del `.tex`; este
> documento existe para poder ensayar sin abrir el PDF y para ver los tiempos acumulados de un
> vistazo.
>
> Para proyectar con las notas en la segunda pantalla, descomentar en `presentacion.tex`:
> `\setbeameroption{show notes on second screen=right}`.
>
> **No hay material de reserva.** La presentación son 21 diapositivas seguidas: portada, 19 de
> contenido y despedida. Lo que antes eran diapositivas de anexo y de verdad aportaba —la
> divergencia de la cadena, qué mira el agente dominante, la tabla completa de robustez, el detalle
> de la cartera ganadora, el Rank-IC por eras y la procedencia de los datos— está ahora en el
> cuerpo, explicado y defendido. Lo que sólo repetía una cifra ya dicha se ha eliminado.

## Estructura

La charla son **dos objetivos**, y conviene tenerlos siempre presentes porque todo cuelga de ahí:

| | Objetivo | Lo demuestra | Métrica | Respuesta |
|---|---|---|---|---|
| **1** | Un ML **aprende a ordenar acciones** fuera de muestra | Los tres Model Studies | Rank-IC | Sí, con matices |
| **2** | **Las variables de cartera importan**, y optimizando por IR se construye una buena | El Portfolio Study | Information Ratio | Sí |

El punto que hay que dejar claro en la diapositiva 2 y volver a tocar en la 13 y la 14: **en los
tres primeros estudios la cartera es secundaria porque todavía no se había optimizado**. Por eso el
−11,29 % no es un fracaso del trabajo — es el punto de partida del segundo objetivo.

La bisagra de toda la charla es la **diapositiva 13**. Ahí se enseña que la cadena mejora dentro y
se degrada fuera, y —esto es lo importante— que lo que se degradó fue el **IR**, no el **Rank-IC**.
De ahí sale el objetivo 2. Si esa diapositiva se cuenta bien, la segunda mitad se defiende sola.

## Reparto de tiempo

| # | Diapositiva | s | Acumulado |
|---|---|---|---|
| 1 | Portada | 20 | 0:20 |
| 2 | Dos objetivos, deliberadamente separados | 50 | 1:10 |
| 3 | El sistema: cinco especialistas y un árbitro | 65 | 2:15 |
| 4 | De dónde salen los datos | 60 | 3:15 |
| 5 | Nunca mirar el futuro | 60 | 4:15 |
| 6 | Cómo se decide sin hacer trampa | 60 | 5:15 |
| | **Acto 1 — ¿aprende a ordenar?** | | |
| 7 | La métrica del objetivo 1: Rank-IC | 50 | 6:05 |
| 8 | Se equivoca y se corrige solo | 70 | 7:15 |
| 9 | Qué mira el agente que domina | 60 | 8:15 |
| 10 | La ordenación es buena, y no sólo dentro | 60 | 9:15 |
| 11 | Rank-IC por era | 60 | 10:15 |
| 12 | ¿Y si fuera suerte? Ocho contrastes | 65 | 11:20 |
| 13 | **La cadena: mejora dentro, se degrada fuera** | 65 | 12:25 |
| | **Acto 2 — ¿y la cartera?** | | |
| 14 | La cartera nunca se había optimizado | 40 | 13:05 |
| 15 | Sí importa, y no por igual | 65 | 14:10 |
| 16 | 1.728 carteras, una ganadora | 60 | 15:10 |
| 17 | La cartera ganadora, en detalle | 55 | 16:05 |
| 18 | El contraste que cierra el trabajo | 70 | 17:15 |
| | **Cierre** | | |
| 19 | Qué no puedo afirmar | 55 | 18:10 |
| 20 | Conclusiones | 55 | 19:05 |
| 21 | Gracias | — | — |

**Total hablado: 19:05.** Está por encima de los 15 minutos **a propósito**: el texto de los
`\note{}` es la versión completa y desarrollada, y en un ensayo real siempre se recorta al hablar.
Diciéndolo a ritmo normal y sin leer palabra por palabra, cae a 15–16 minutos. Si hace falta
apretar, este es el orden de recorte:

1. **Diapositiva 4** (datos): de 60 a 35 s. Decir sólo las fuentes y el matiz de supervivencia.
2. **Diapositiva 11** (Rank-IC por era): de 60 a 35 s. Quedarse con «`risk` es el único positivo en
   las cuatro» y «los fundamentales se hunden todos a la vez en la reservada».
3. **Diapositiva 9** (qué mira `risk`): de 60 a 40 s. Sólo el contraste beta *vs* microestructura.
4. **Diapositiva 3** (el sistema): de 65 a 45 s, sin enumerar los cinco agentes uno a uno.
5. **Diapositiva 17** (cartera en detalle): de 55 a 35 s, dejando caída máxima y rotación.

**Nunca se aceleran las diapositivas 13, 15, 16 y 18**: son la bisagra y el objetivo 2 entero.

---

## Guion

### 1 · Portada — 20 s

Buenos días. Voy a contar un trabajo que se hace dos preguntas encadenadas: primero, si un sistema
de aprendizaje automático puede aprender a ordenar acciones de mejor a peor y que esa ordenación
siga valiendo en datos que nunca vio; y segundo, qué pasa cuando esa ordenación se convierte en una
cartera de verdad.

Adelanto ya la tesis, para que se entienda hacia dónde va todo: las dos cosas son preguntas
distintas, y confundirlas es el error metodológico que este trabajo intenta señalar.

### 2 · Dos objetivos, deliberadamente separados — 50 s

El trabajo tiene dos objetivos, y esta diapositiva es el índice de toda la charla.

El primero es puramente predictivo: ¿puede un sistema de aprendizaje automático aprender a ordenar
acciones de mejor a peor, y que esa ordenación siga valiendo en datos que nunca vio? Eso lo
responden tres estudios encadenados, y se decide por capacidad de ordenar, nunca por rentabilidad.

El segundo llega después: una vez tengo la señal, ¿da igual cómo la convierta en cartera? Eso lo
responde un cuarto estudio dedicado sólo a la cartera, y ahí sí la métrica es económica.

La caja de abajo es la clave de todo el trabajo: **ordenar bien y ganar dinero son dos capacidades
distintas**. Un sistema puede ordenar razonablemente y perder dinero si la cartera está mal
construida — y voy a enseñar exactamente ese caso. Por eso los dos objetivos van separados, y por
eso van en este orden.

Un detalle que conviene retener: en los tres primeros estudios la cartera es secundaria porque
todavía no se había optimizado.

### 3 · El sistema: cinco especialistas y un árbitro — 65 s

El sistema son cinco agentes especializados, cada uno con su propia forma de mirar una empresa.

«Value» busca lo barato. «Growth», lo que crece. «Quality», lo que está sano. «Momentum», lo que
lleva inercia. Y «risk», lo que se comporta de forma estable.

Fíjense en la nota de la izquierda, porque es una decisión de diseño que un tribunal debe poder
cuestionar: **los bloques de variables son disjuntos**. Ninguna variable la comparten dos agentes.
Eso es deliberado — si les diera a todos las mismas treinta y tres variables acabaría teniendo cinco
copias del mismo predictor, y el meta-agente no tendría nada que arbitrar. El precio que pago es que
cada agente ve menos información de la que podría; la ventaja es que sus errores son razonablemente
independientes, que es lo que hace que combinarlos tenga sentido.

Encima hay un meta-agente que hace de árbitro: aprende cuánto pesa cada especialista. Y a la derecha
está la restricción que lo hace honesto: **sólo aprende de cohortes cuya respuesta ya se conoce**.
Nunca de las que están abiertas. Si aprendiera de cohortes abiertas estaría usando el futuro para
decidir el presente.

Y la salida, importante, no es una predicción de precio ni una probabilidad de subida. Es un
ranking: de la mejor a la peor de unas quinientas acciones. Todo lo que voy a medir después mide la
calidad de ese orden.

### 4 · De dónde salen los datos — 60 s

Antes de ningún resultado, de dónde salen los datos, porque la credibilidad de todo lo demás depende
de esto.

El universo son las empresas del S&P 500. Los precios vienen de Yahoo Finance, los fundamentales de
Finnhub, y las fechas de publicación —que luego verán por qué son críticas— de los propios registros
de la SEC.

El panel va de 2003 a 2026, pero todas las decisiones se toman entre 2015 y 2024. La cobertura
utilizable no baja del 99,4 % ningún año, es decir, casi no hay huecos que rellenar. Y el catálogo
de variables es **cerrado**: treinta y tres, fijadas de antemano. No se puede añadir una variable a
mitad del experimento porque mejore el resultado.

Y ahora el matiz, que es la trampa clásica de este tipo de trabajos: uso la **composición
histórica** del índice, no la actual. Si cogiera las quinientas empresas que están hoy en el S&P 500
y las hiciera cotizar desde 2003, estaría seleccionando supervivientes: empresas que sabemos que no
quebraron. Cualquier modelo parece brillante sobre supervivientes. Aquí una empresa sólo es elegible
en las fechas en que de verdad estaba en el índice.

### 5 · Nunca mirar el futuro — 60 s

Y esta es la regla que lo condiciona absolutamente todo, porque es lo que separa un trabajo creíble
de uno que no lo es.

Los datos financieros tienen una trampa. El balance de una empresa lleva fecha de cierre de
trimestre —el punto rojo de la izquierda—, pero no se publica hasta semanas o incluso meses después,
que es el segundo punto. Si yo uso ese dato el día del cierre fiscal, estoy usando información que
en ese momento literalmente nadie tenía. Y con eso cualquier modelo parece brillante: no está
prediciendo, está leyendo la respuesta.

Aquí un dato entra en la decisión sólo si su fecha real de publicación, más un retardo de ejecución
que da margen para operar, es anterior al día en que decido. Ese es el tramo verde.

Y por el otro lado, el tramo dorado: la etiqueta —el retorno futuro que estoy intentando predecir—
no se usa para nada hasta que ese periodo ha cerrado de verdad. Es la misma restricción que mencioné
en el meta-agente.

Esto no es una intención declarada: está **demostrado formalmente** en la memoria como una
proposición, y cubierto por tests automáticos que fallan si alguien rompe la regla.

### 6 · Cómo se decide sin hacer trampa — 60 s

Todo el trabajo ocurre en la barra azul: de 2015 a 2024, ciento diecisiete cohortes mensuales. Ahí
se toman absolutamente todas las decisiones — qué variables, qué modelo, qué horizonte, qué cartera.

La banda dorada, 2025 y 2026, se apartó el primer día y se congeló. No participa en ninguna
elección, ni siquiera indirectamente. Es el examen final, y sólo se mira al terminar. Esto es
importante decirlo bien: **no es un conjunto de test que se mira varias veces hasta que sale bien**.
Se mira una vez, al final, y lo que salga se reporta.

Debajo están los cuatro estudios. Tres estudios de modelo encadenados —el ganador de cada uno es el
punto de partida del siguiente— responden al objetivo uno. Y un cuarto estudio, ya sobre la señal
congelada y sin reentrenar nada, responde al objetivo dos.

Y la frase de abajo, que a un tribunal le importa: en los tres primeros estudios la selección se
hace por Rank-IC, por capacidad de ordenar, nunca por rentabilidad. **El dinero no elige el modelo.**
Si eligiera por rentabilidad, estaría seleccionando el modelo que mejor se ajusta al ruido de este
periodo concreto de mercado.

---

## ACTO 1 — ¿Aprende a ordenar?

### 7 · La métrica del objetivo 1: Rank-IC — 50 s

Dedico una diapositiva entera a la métrica del objetivo uno porque la voy a usar todo el rato y
porque su escala es contraintuitiva.

El Rank-IC responde a una sola pregunta: ¿se parece el orden que yo propuse al orden que realmente
ocurrió? Es una correlación de rangos. Cero es azar puro: mi ordenación no aporta nada. Uno sería el
orden perfecto, y en finanzas eso no existe.

Y aquí está lo contraintuitivo, que conviene decir antes de enseñar cifras: en la literatura de
selección de activos, **valores entre 0,02 y 0,05 ya se consideran explotables**. Suena
ridículamente bajo si uno viene de otros dominios del aprendizaje automático, donde una correlación
de 0,03 sería una tomadura de pelo. Pero aquí se aplica sobre quinientas acciones, todos los meses,
durante años: un sesgo pequeño y persistente en el orden es exactamente lo que se busca.

Dos cosas más. Se calcula mes a mes y se promedia sobre las 117 cohortes, así que un mes bueno
aislado no lo salva. Y al ser una correlación de rangos, no me exige acertar cuánto sube una acción:
sólo que la ponga por delante de las que suben menos.

### 8 · Se equivoca y se corrige solo — 70 s

Esta es la diapositiva donde el sistema se retrata solo.

Cada barra es un año. Los colores son cuánto peso le da el árbitro a cada especialista.

En 2015 empieza sin saber nada: reparte por igual, un 20 % a cada uno, porque no tiene todavía
cohortes cerradas de las que aprender. En 2016 y 2017 se lanza a lo rojo, que es «momentum». Y
momentum acabará siendo el peor especialista de todo el periodo. O sea: **se equivoca**, y se
equivoca de verdad.

Y entonces, sin que nadie intervenga y sin que yo le diga nada, **se corrige**. A partir de 2018 el
azul oscuro, que es «risk», se lo va comiendo todo hasta llegar prácticamente al 100 % en 2023.

Insisto en por qué esto es evidencia y no anécdota: si yo hubiera fijado de antemano que «risk» era
el bueno, la trayectoria sería plana desde 2015. El hecho de que se equivoque primero, con datos
reales y en tiempo real, y se corrija después, es la evidencia más directa de que hay aprendizaje y
de que no está puesto a mano.

Los números de abajo. Repartir por igual da 0,0675. Lo que aprende da 0,1090: un 61 % mejor. El
árbitro aporta.

Y ahora el matiz, que lo digo yo antes de que me lo pregunten: «risk» por su cuenta da 0,1227. Bate
a la combinación. Es decir, el meta-agente aprende —eso está demostrado— pero acaba funcionando más
como un **selector** que como un **combinador**, y con estos datos no puedo presentar la
arquitectura multi-agente como demostrada. Volveré a esto en las limitaciones.

### 9 · Qué mira el agente que domina — 60 s

Si «risk» domina, la pregunta obligada es qué está mirando «risk». Y la respuesta no es la que uno
esperaría.

Esta tabla sale de más de 1,3 millones de filas de atribución local: para cada acción y cada mes,
qué variable pesó más en la decisión de ese agente.

Lo esperable sería que un agente llamado «risk» estuviera leyendo la **prima de baja volatilidad**,
que es un factor clásico y muy conocido. Pues no. La beta a un año ni siquiera está entre las tres
primeras variables. Lo que encabeza son `gap_21d` y `range_63d` —el hueco de apertura a tres semanas
y el rango de precios a tres meses— que juntas dominan el 78 % de las observaciones.

Eso significa que el agente dominante está leyendo **microestructura de precio a escala de
semanas**, no una prima de riesgo clásica.

Y esto tiene una consecuencia medible: cuando neutralizo la señal contra catorce controles de estilo
conocidos —valor, tamaño, momentum, volatilidad y compañía— conserva el 86,62 %. Es decir, lo que
hace no es una reexpresión de un factor que ya estaba en la literatura. Si lo fuera, la
neutralización se lo habría comido casi entero.

### 10 · La ordenación es buena, y no sólo dentro de muestra — 60 s

Estos son los números del objetivo uno.

El sistema saca un Rank-IC de 0,1090. Con la escala que expliqué hace dos diapositivas, eso está muy
por encima de lo que se considera explotable.

Para que sirva de referencia y no de cifra suelta: el mejor criterio clásico de los que probé como
baseline —una combinación de crecimiento a precio razonable— se queda en 0,0130. El sistema es unas
ocho veces mejor que la mejor regla sencilla que se me ocurrió.

La t de Newey-West es 3,46. Uso Newey-West precisamente porque las cohortes mensuales se solapan y
los errores están autocorrelacionados; con la corrección puesta, sigue siendo claramente
distinguible de cero. Y acierta el sentido de la ordenación en el 74 % de los meses, o sea, no vive
de dos o tres meses excepcionales.

Pero lo que de verdad importa es la caja dorada. En la era reservada, esa que apartamos el primer
día y que no participó en ninguna decisión, el Rank-IC sigue siendo positivo: **+0,0441**.

Y lo digo con la honestidad que toca: baja bastante respecto al 0,1090. Eso es lo normal y lo
esperable —una parte de ese 0,1090 es ajuste al periodo de selección. Lo relevante es que **no se
hunde ni cambia de signo**. Ordena fuera de muestra.

### 11 · Rank-IC por era — 60 s

Esta tabla desagrega el resultado anterior por eras, y me parece de las más informativas del trabajo
porque enseña dónde el sistema funciona y dónde no.

Las tres primeras columnas son subperiodos de la ventana de selección; la última, separada por la
línea, es la era reservada.

Primero, lo que se ve leyendo en horizontal: **el Rank-IC no es estable en el tiempo**. El periodo
2019-2021 es claramente el peor para casi todo el mundo, y 2022-2024 el mejor. Eso ya dice algo
importante: no hay una capacidad predictiva constante, hay regímenes.

Segundo, `risk` es el único agente positivo en las cuatro eras. El meta final va justo por debajo,
que es coherente con lo que vimos: acaba pareciéndose mucho a `risk`.

Y tercero, lo que más me interesa: en la era reservada, **todos los agentes fundamentales —value,
growth, quality— se vuelven negativos a la vez**. No uno: todos, simultáneamente. Eso es una firma
bastante clara de rotación de factores del mercado en ese periodo, y no de un error de signo en un
modelo concreto. Si sólo se hubiera hundido uno, sospecharía de mi implementación; que se hundan los
tres a la vez apunta al mercado.

También explica por qué el meta equiponderado, que reparte por igual, se va a negativo en esa era:
arrastra a los tres que fallan.

### 12 · ¿Y si fuera suerte? Ocho contrastes — 65 s

La pregunta obvia después de enseñar un resultado bueno: ¿y si es suerte? Lo ataqué de ocho maneras
distintas, y aquí están las ocho.

La primera es la más contundente: barajé las etiquetas al azar casi diez mil veces y volví a medir
todo. Ninguna de las 9.999 permutaciones llegó a lo que da el modelo real. p = 0,0001, que es el
mínimo posible con ese número de réplicas.

Los placebos de etiqueta: predigo cosas que no debería poder predecir, y en efecto no las predice —
se queda pegado a cero. Es el control negativo.

El bootstrap por bloques respeta la estructura temporal, y el intervalo al 95 % no toca el cero.
Quitar eras enteras: el resultado se mueve entre 0,078 y 0,135, o sea, no depende de un periodo
concreto. Cambiar la semilla del aleatorio apenas mueve nada. Contra carteras aleatorias con el
mismo riesgo, queda en el percentil 97. Y la neutralización de estilo, que ya comenté, conserva el
86 %.

**Siete de ocho superados.**

Y el octavo no, y lo digo yo antes de que me lo pregunten. El Deflated Sharpe penaliza el ratio por
cuántas configuraciones distintas he probado antes de quedarme con una. Yo he encadenado tres
estudios completos, así que he probado muchísimas. Se queda en 0,682 frente a un umbral de 0,95. Es
el precio declarado de haber buscado, y lo reporto como falla, no lo escondo.

### 13 · La cadena: mejora dentro, se degrada fuera — 65 s

**[Bisagra de la charla. No acelerar.]**

Esta diapositiva es la que más honestidad exige, así que la enseño entera.

Las columnas son los tres estudios encadenados. Leyendo hacia abajo: dentro de la ventana de
selección todo mejora de forma monótona. El Rank-IC sube de 0,1000 a 0,1090, la t sube, el IR de
selección sube de 0,189 a 0,339. Visto así, la cadena funcionó.

Pero miren la última fila, que es el Information Ratio en la era reservada: +0,898, +0,476, −1,167.
**Se degrada monótonamente.** Cada estudio que mejoró las métricas de dentro empeoró el resultado
económico de fuera.

Esa divergencia tiene nombre: **sobreajuste por búsqueda**. Al encadenar estudios, cada iteración
exprime un poco más el periodo de selección. Es la misma causa por la que el Deflated Sharpe no pasa
el umbral. No lo descubro yo en el turno de preguntas: está en la memoria y está aquí.

Y ahora la observación que da sentido a toda la segunda mitad del trabajo, que es el texto en verde.
Lo que se degradó fue el **Information Ratio**. El **Rank-IC**, no: ese se mantuvo positivo dentro y
fuera. Y como el Rank-IC mide la señal y el IR mide la señal *más* la cartera, la conclusión es que
**el problema no estaba en la señal sino en la traducción a posiciones**.

Eso es literalmente lo que motivó el segundo objetivo. No es una idea que se me ocurriera después:
es una respuesta a este resultado.

Un apunte de diseño: la cadena converge. En el primer estudio cambiaron ocho variables respecto al
punto de partida; en el segundo, una; en el tercero, dos. No está oscilando.

---

## ACTO 2 — ¿Y la cartera?

### 14 · La cartera nunca se había optimizado — 40 s

Y aquí empieza la segunda mitad.

La situación es exactamente esta: tengo una señal que ordena bien, incluso fuera de muestra, y al
mismo tiempo tengo una cartera que en la era reservada pierde un 11,3 % contra el índice. Las dos
cosas son verdad a la vez, y esa aparente contradicción es el trabajo.

¿Por qué puede pasar? Porque **una ordenación no se puede comprar entera**. Yo puedo ordenar
quinientas acciones perfectamente, pero una cartera real compra unas pocas —ocho, diez, veinte—,
paga comisiones cada vez que rota, y tiene que decidir cuánto pone en cada una y cuándo vende. Todas
esas decisiones están entre la señal y el resultado, y ninguna la había tocado.

En los tres estudios anteriores la cartera era simplemente la que venía por defecto en el catálogo.
No estaba mal elegida a propósito: es que no era el objeto de estudio.

Así que la pregunta del objetivo dos es: ¿da igual cómo la construya?

### 15 · Sí importa, y no por igual — 65 s

La respuesta es que importa muchísimo, y además de forma muy desigual.

Antes, la métrica: el **Information Ratio** es cuánto le saco al índice por cada unidad de riesgo
que asumo. No es la rentabilidad a secas — es rentabilidad relativa ajustada por cuánto me desvío
del índice. Es la métrica que optimiza este segundo estudio, y es distinta de la del objetivo uno a
propósito, porque la pregunta es distinta.

En la tabla está, para cada variable de construcción de cartera, cuánto mueve el Information Ratio
entre su mejor y su peor valor, medido con la mediana sobre todas las demás combinaciones.

**El número de posiciones lo mueve tres décimas.** Eso es enorme: es del mismo orden que el IR total
de la cartera por defecto. Cuántas acciones compras es, con diferencia, la decisión más cara del
sistema. El tope de efectivo, dos décimas.

Y abajo, dos variables prácticamente **inertes**: la tolerancia de deriva —cuánto dejo que un peso
se desvíe antes de reequilibrar— y cómo reparto los pesos entre las posiciones. Nueve y dieciséis
milésimas. Hay un factor treinta entre la primera variable y la última.

Y esto es lo que quiero subrayar: el resultado útil no es la frase abstracta «la cartera importa».
Es saber **qué dos decisiones concretas** se llevan casi todo, y cuáles no merece la pena discutir.
Eso es accionable.

### 16 · 1.728 carteras, una ganadora — 60 s

Así que monté una rejilla completa: 1.728 carteras, todas sobre exactamente la misma señal ya
congelada. **Sin reentrenar absolutamente nada del modelo.** Lo único que cambia entre un punto y
otro del gráfico son las reglas de construcción de la cartera.

Cada punto es una cartera. El eje horizontal es cuánto rota al año, el vertical su Information
Ratio. La ganadora es la que está marcada arriba.

Y lo que quiero que miren no es la ganadora: es **la dispersión vertical**. Con la misma señal
exacta, hay carteras que dan Information Ratio negativo y carteras que rozan el 0,85. Ese eje
vertical es, gráficamente, el objetivo dos entero. La señal es una constante en este gráfico; toda
esa variación la produce la cartera.

La ganadora pasa el Information Ratio de 0,339 a 0,844, y el exceso sobre el índice del 2,6 al 7 %.

Y un detalle metodológico que está en la esquina y que defiendo: la rejilla es **cartesiana, no
secuencial**. Evalúo todas las combinaciones, no una variable cada vez. Es mucho más caro, pero es
necesario porque las variables interactúan: si hubiera optimizado variable a variable habría elegido
cinco posiciones, que es lo mejor en solitario, y la cartera ganadora usa ocho. Optimizando por
separado no la habría encontrado nunca.

### 17 · La cartera ganadora, en detalle — 55 s

Aquí está la cartera ganadora abierta en canal, porque un Information Ratio suelto no dice qué tipo
de cartera es.

Leyendo la columna del medio: ocho posiciones, sin efectivo, pesos proporcionales al alfa estimado.
Es una cartera **concentrada por diseño**, y eso tiene consecuencias que se ven en la tabla.

La máxima caída es del 28 %, un punto y medio **peor** que la cartera por defecto. O sea, la
optimización no ha reducido el riesgo de caída: ha mejorado el ratio subiendo el numerador, no
bajando el denominador. Con ocho posiciones eso es esperable y hay que decirlo.

La rotación es de 3,2 veces al año, es decir, la cartera se renueva entera unas tres veces por año.
Sobre eso aplico 5 puntos básicos de comisión y 10 de *slippage*, constantes. Y aquí hago una
limitación explícita: con esa rotación y ocho posiciones, **no modelar el impacto de mercado ni la
capacidad** es una simplificación que importa. Con volúmenes pequeños es razonable; con volúmenes
grandes, no.

Bate al índice en el 80 % de los años de la ventana de selección, frente al 70 % de la cartera por
defecto.

Y la última columna, que es la que cuenta: la era reservada. Ahí baja al 50 % de años, con un IR de
0,304. Baja, como todo lo de fuera de muestra, pero se mantiene positivo.

### 18 · El contraste que cierra el trabajo — 70 s

Y este es el contraste que cierra el trabajo, en la era reservada, que no intervino en ninguna
decisión.

Con la cartera que venía por defecto, la señal perdía un 11,29 % frente al índice, con un
Information Ratio de −1,167. Es un mal resultado y no lo escondo: es el que aparecía al final de la
cadena de estudios.

Con la cartera optimizada —mismo modelo, misma señal, mismo panel, sin reentrenar absolutamente
nada— gana un 2,56 %, con Information Ratio +0,304.

Y ahora la caja gris de abajo, que es para mí **el dato más importante de toda la defensa**: el
Rank-IC de esa era es **exactamente el mismo en los dos casos, +0,0441**. Idéntico. Porque el
Rank-IC mide el orden, y el orden no cambia: es la misma señal.

O sea: la capacidad predictiva era idéntica en los dos escenarios. Uno pierde un 11 % y el otro gana
dos y medio. **Todo el diferencial —casi catorce puntos— lo produce la construcción de la cartera,
no el modelo.**

A la izquierda tienen la curva de la cartera ganadora contra el S&P 500, con la banda dorada
marcando la era reservada. Se ve que no gana por un salto puntual.

Con esto el objetivo dos queda respondido: sí, las variables de cartera importan, importan
muchísimo, y optimizando por Information Ratio se construye una que además confirma fuera de la
ventana de decisión.

---

## Cierre

### 19 · Qué no puedo afirmar — 55 s

Antes de terminar, lo que este trabajo no demuestra. Prefiero decirlo yo, y prefiero que ocupe una
diapositiva entera.

Uno: la era reservada son seis cohortes, poco más de un año de cartera. Es poca potencia
estadística. Yo la llamo **confirmación, no validación**, y la diferencia importa: seis cohortes no
permiten descartar que el resultado sea afortunado.

Dos: la cartera ganadora es la mejor de 1.728. Su 0,844 es una **cota superior optimista**, no una
estimación insesgada de lo que haría mañana. Lo que sostiene el objetivo dos no es ese número, es la
dispersión que enseñé en el gráfico de la rejilla.

Tres: el Deflated Sharpe se queda en 0,682. No hago ninguna afirmación de rentabilidad ajustada por
selección, y esa es la consecuencia directa de haber encadenado estudios.

Cuatro: como el agente de riesgo domina al meta, **no puedo decir que la arquitectura de cinco
agentes esté demostrada**. El meta acaba funcionando como un selector más que como un combinador.
Eso es un resultado, no un fallo de diseño —porque el ganador no estaba fijado de antemano— pero se
declara como es.

Y cinco: no modelo impacto de mercado ni capacidad, y con esta rotación sobre ocho posiciones eso no
es un detalle menor.

Nada de esto es una recomendación de inversión. La contribución es **metodológica**.

### 20 · Conclusiones — 55 s

Resumo, y con esto termino.

**Objetivo uno**: sí, un sistema de aprendizaje automático aprende a ordenar acciones, y esa
ordenación aguanta fuera de muestra. Rank-IC de 0,1090 dentro, positivo fuera, siete de ocho
contrastes de robustez superados. Con el matiz de que el agente de riesgo por su cuenta ordena algo
mejor que la combinación, así que la arquitectura multi-agente no queda demostrada.

**Objetivo dos**: sí, las variables de construcción de cartera afectan al resultado, y mucho. Hay
dos que se llevan casi todo, y optimizando por Information Ratio se construye una cartera que
multiplica el ratio por dos y medio y que convierte un −11 % fuera de muestra en un +2,5 %. Con el
matiz de que es la mejor de 1.728 y se confirma sobre seis cohortes.

Y si me tengo que quedar con una sola frase, es la de la caja: **ninguna afirmación sobre la
utilidad de un sistema predictivo es interpretable sin declarar con qué cartera se midió.** Un
trabajo que sólo reporte capacidad predictiva puede estar escondiendo tanto un éxito como un fracaso
— y yo tengo las dos cosas medidas sobre exactamente la misma señal.

Todo lo que he enseñado sale de cuatro estudios con identificador trazable, y cada cifra de esta
presentación tiene su artefacto detrás.

Muchas gracias.

---

## Turno de preguntas

**No hay diapositivas de reserva.** Todo lo que antes estaba en el anexo está ahora en el cuerpo, así
que las respuestas se dan **volviendo a una diapositiva ya proyectada**. La columna «Volver a» indica
cuál.

| Si preguntan… | Volver a | Respuesta corta |
|---|---|---|
| ¿La cadena mejoró o sólo se sobreajustó? | **13** | Ambas cosas, y está declarado: mejora monótona **dentro** (0,1000 → 0,1074 → 0,1090) y degradación monótona **fuera** (+0,898 → +0,476 → −1,167). Esa divergencia es la firma del sobreajuste por búsqueda y es literalmente lo que motivó el Portfolio Study. |
| ¿Por qué domina `risk`? ¿Qué mira? | **9** | No es la prima clásica de baja volatilidad: `gap_21d` y `range_63d` encabezan el 78 % de las observaciones, y `beta_252d` ni está entre las tres primeras. Lee microestructura de precio a semanas. Por eso la neutralización por estilos conserva el 86,62 %. |
| ¿Y la robustez completa? | **12** | Siete de ocho superados. El que falla es el Deflated Sharpe (0,682 frente a un umbral de 0,95) y se reporta como falla, no se esconde. |
| ¿No es sobreajuste elegir 1 de 1.728? | **16** y **19** | Sí, y por eso el 0,844 se presenta como **cota superior optimista**. Lo que sostiene el objetivo 2 no es ese número sino la **dispersión**: que la misma señal produzca IR entre negativo y 0,85 según la cartera. Y la confirmación en la era reservada (+2,56 %) es fuera de la ventana de decisión. |
| ¿Por qué rejilla cartesiana y no secuencial? | **16** | Porque las variables interactúan: marginalmente el mejor número de posiciones es 5, pero la ganadora usa 8. Optimizando variable a variable no se habría encontrado. |
| ¿Aguanta por eras? | **11** | `risk` es el único positivo en las cuatro. En la era reservada todos los agentes fundamentales se vuelven negativos **a la vez**, lo que apunta a rotación de factores del mercado, no a un error de signo. |
| ¿Cuánto cae la cartera? | **17** | 28,40 % de máxima caída en selección, 12,09 % en la era reservada. Es una cartera de 8 posiciones: concentrada por diseño. Y la optimización **no** mejoró la caída: mejoró el numerador del ratio. |
| ¿De dónde salen los datos? ¿Y el sesgo de supervivencia? | **4** | Composición **histórica** del S&P 500, no la actual: una empresa sólo es elegible en las fechas en que pertenecía al índice. Fechas de publicación reales de SEC EDGAR. Cobertura ≥ 99,4 % cada año. |
| ¿Costes realistas? | **17** | 5 pb de comisión más 10 pb de *slippage*, constantes. Es una limitación declarada: no modelo impacto de mercado ni capacidad, y con 3,2 rotaciones al año sobre 8 posiciones eso importa. |
| ¿Y los perfiles de inversión? | — | Quedaron **fuera de la rejilla** a propósito: se construyen sobre la cartera ya elegida y no influyeron en su selección. El perfil que **no** reordena la señal (`balanced`, IR 0,844) domina la ventana de selección, y el orden de los otros siete se predice desde el Rank-IC de los agentes que cada uno pondera: `defensive` (0,570) carga 0,60 en `risk`, el mejor agente; `momentum` (0,017) carga 0,75 en el peor y encima penaliza a `risk`. **Un perfil no añade información, redistribuye la que ya hay** — por eso no está en la presentación. |
| ¿Por qué bloques de variables disjuntos? | **3** | Si todos los agentes vieran las 33 variables serían cinco copias del mismo predictor y el meta no tendría nada que arbitrar. El coste es que cada agente ve menos; la ventaja es que sus errores son razonablemente independientes. |
| ¿Esto se puede usar para invertir? | **19** | No. Seis cohortes de confirmación, DSR por debajo del umbral y una cartera elegida entre 1.728. La contribución es metodológica. |

### Tres preguntas incómodas, con respuesta preparada

**«Si `risk` solo es mejor que el meta, ¿para qué los cinco agentes?»**
Es la objeción correcta y está en la memoria antes que en el tribunal. La respuesta honesta: con
estos datos la arquitectura multi-agente **no queda demostrada**. Lo que sí queda demostrado es que
el meta **aprende** —parte de pesos iguales, se equivoca en 2016 y se corrige— y que no se le fijó de
antemano el ganador. Que acabe concentrando en uno es un resultado, no un fallo de diseño; pero
convierte al meta en un selector más que en un combinador, y así se declara. (Diapositivas 8 y 19.)

**«El −11,29 % de la cartera por defecto, ¿no invalida el objetivo 1?»**
No, y por una razón medible: el Rank-IC de la era reservada es **+0,0441 con las dos carteras**,
porque no depende de la cartera. La capacidad de ordenar es la misma en los dos casos. Lo que cambia
es la traducción a posiciones. De hecho ese contraste es justamente la evidencia del objetivo 2.
(Diapositiva 18.)

**«Si el Deflated Sharpe no pasa, ¿qué queda en pie?»**
Queda en pie el objetivo 1, que **no se decide por Sharpe sino por Rank-IC**, y que supera siete
contrastes incluyendo permutación con p = 0,0001 y neutralización de estilos. El DSR penaliza la
búsqueda, y yo he buscado mucho: eso afecta a cualquier afirmación de *rentabilidad ajustada por
selección*, y por eso no hago ninguna. La contribución que reclamo es metodológica —que ordenar y
rentar son cosas distintas y hay que medirlas por separado— y esa no depende del DSR.
(Diapositivas 12 y 19.)
