# Guion de defensa — 10 minutos

> Acompaña a `latex/presentacion.tex`. El mismo texto vive en los `\note{}` del `.tex`; este
> documento existe para poder ensayar sin abrir el PDF y para ver los tiempos acumulados de un
> vistazo.
>
> Para proyectar con las notas en la segunda pantalla, descomentar en `presentacion.tex`:
> `\setbeameroption{show notes on second screen=right}`.

## Estructura

La charla son **dos objetivos**, y conviene tenerlos siempre presentes porque todo cuelga de ahí:

| | Objetivo | Lo demuestra | Respuesta |
|---|---|---|---|
| **1** | Un ML **aprende a ordenar acciones** fuera de muestra | Los tres Model Studies | Sí, con matices |
| **2** | **Las variables de cartera importan**, y optimizando por IR se construye una buena | El Portfolio Study | Sí |

El punto que hay que dejar claro en la diapositiva 2 y volver a tocar en la 9: **en los tres
primeros estudios la cartera es secundaria porque todavía no se había optimizado**. Por eso el
−11,29 % no es un fracaso del trabajo — es el punto de partida del segundo objetivo.

## Reparto de tiempo

| # | Diapositiva | s | Acumulado |
|---|---|---|---|
| 1 | Portada | 15 | 0:15 |
| 2 | Dos objetivos | 40 | 0:55 |
| 3 | El sistema | 55 | 1:50 |
| 4 | Nunca mirar el futuro | 45 | 2:35 |
| 5 | Cómo se decide sin hacer trampa | 55 | 3:30 |
| | **Acto 1 — ¿aprende a ordenar?** | | |
| 6 | Se equivoca y se corrige solo | 60 | 4:30 |
| 7 | Y la ordenación es buena | 50 | 5:20 |
| 8 | ¿Y si fuera suerte? | 50 | 6:10 |
| | **Acto 2 — ¿y la cartera?** | | |
| 9 | La cartera nunca se optimizó | 30 | 6:40 |
| 10 | Sí importa, y no por igual | 60 | 7:40 |
| 11 | 1.728 carteras, una ganadora | 55 | 8:35 |
| 12 | Y aguanta en la era reservada | 65 | 9:40 |
| | **Cierre** | | |
| 13 | Qué no puedo afirmar | 45 | 10:25 |
| 14 | Conclusiones | 40 | 11:05 |
| 15 | Gracias | — | — |

**Total hablado: 10:25.** Va por encima de los 10 minutos a propósito: en un ensayo real siempre
se recorta. Si hay que ajustar, el orden de recorte es:

1. Diapositiva 4 (point-in-time): bajar de 45 a 30 s diciendo sólo el ejemplo del balance.
2. Diapositiva 3 (el sistema): bajar de 55 a 40 s sin enumerar los cinco agentes uno a uno.
3. Diapositiva 13 (limitaciones): bajar de 45 a 35 s dejando dos de las cuatro.

**Nunca se aceleran las diapositivas 10, 11 y 12**: son el objetivo 2 entero.

---

## Guion

### 1 · Portada — 15 s

Buenos días. Voy a contar en diez minutos un trabajo sobre si un sistema de aprendizaje automático
puede aprender a ordenar acciones, y qué pasa cuando esa ordenación se convierte en una cartera de
verdad.

### 2 · Dos objetivos — 40 s

El trabajo tiene dos objetivos, y esta diapositiva es el índice de toda la charla.

El primero es puramente predictivo: ¿puede un sistema de aprendizaje automático aprender a ordenar
acciones de mejor a peor, y que esa ordenación siga valiendo en datos que nunca vio? Eso lo
responden tres estudios encadenados.

El segundo llega después: una vez tengo la señal, ¿da igual cómo la convierta en cartera? Spoiler:
no da igual en absoluto. Eso lo responde un cuarto estudio dedicado sólo a la cartera.

Fíjense en el orden, porque importa: en los tres primeros estudios la cartera es secundaria,
todavía no se había optimizado.

### 3 · El sistema — 55 s

El sistema son cinco agentes especializados, cada uno con su propia forma de mirar una empresa.

«Value» busca lo barato. «Growth», lo que crece. «Quality», lo que está sano. «Momentum», lo que
lleva inercia. Y «risk», lo que se comporta de forma estable.

Cada uno ve sólo su vocabulario: eso es deliberado, porque si les diera a todos las mismas
variables acabaría teniendo cinco copias del mismo predictor.

Encima hay un meta-agente que hace de árbitro: aprende cuánto pesa cada especialista. Y aquí está
lo importante: sólo aprende de cohortes cuya respuesta ya se conoce. Nunca de las que están
abiertas.

La salida no es una predicción de precio. Es un ranking: de la mejor a la peor de unas quinientas
acciones.

### 4 · Nunca mirar el futuro — 45 s

Antes de enseñar un solo resultado tengo que explicar la regla que lo condiciona todo, porque es lo
que separa un trabajo creíble de uno que no lo es.

Los datos financieros tienen una trampa: el balance de una empresa lleva fecha de cierre de
trimestre, pero no se publica hasta semanas o meses después. Si yo uso ese dato el día del cierre,
estoy usando información que en ese momento nadie tenía. Y con eso cualquier modelo parece
brillante.

Aquí un dato entra en la decisión sólo si su fecha real de publicación, más un retardo de
ejecución, es anterior al día en que decido. Y la respuesta —el retorno futuro— no se usa para nada
hasta que ese periodo ha cerrado de verdad.

Está demostrado formalmente en la memoria y cubierto por tests.

### 5 · Cómo se decide sin hacer trampa — 55 s

Todo el trabajo ocurre en la barra azul: de 2015 a 2024, ciento diecisiete cohortes mensuales. Ahí
se toman absolutamente todas las decisiones.

La banda dorada, 2025 y 2026, se apartó desde el primer día. No participa en ninguna elección. Es
el examen final, y sólo se mira al terminar.

Debajo están los cuatro estudios. Tres estudios de modelo encadenados —el ganador de cada uno es el
punto de partida del siguiente— que responden al objetivo uno. Y un cuarto estudio, ya sobre la
señal congelada, que responde al objetivo dos.

Un detalle que a un tribunal le importa: la selección se hace por capacidad de ordenar, nunca por
rentabilidad. El dinero no elige el modelo.

---

## ACTO 1 — ¿Aprende a ordenar?

### 6 · Se equivoca y se corrige solo — 60 s

Esta es mi diapositiva favorita, porque el sistema se retrata solo.

Cada barra es un año. Los colores son cuánto peso le da el árbitro a cada especialista.

En 2015 empieza sin saber nada: reparte por igual, un veinte por ciento a cada uno. En 2016 y 2017
se lanza a lo rojo, que es «momentum» —y momentum acabará siendo el peor especialista de todo el
periodo. O sea: se equivoca.

Y entonces, sin que nadie intervenga, se corrige. A partir de 2018 el azul oscuro, que es «risk»,
se lo va comiendo todo hasta llegar prácticamente al cien por cien en 2023.

Que un sistema se equivoque primero y se corrija después es la evidencia más directa de que el
aprendizaje es real.

Los números: repartir por igual da 0,0675; lo aprendido da 0,1090. Un sesenta y uno por ciento
mejor.

Y ahora el matiz, que lo digo yo antes de que me lo pregunten: «risk» por su cuenta da 0,1227. Bate
a la combinación. Es decir, el árbitro aprende, pero no puedo presentar la arquitectura
multi-agente como demostrada.

### 7 · Y la ordenación es buena — 50 s

Primero, qué es el Rank-IC, porque lo voy a usar todo el rato: mide si mi ordenación se parece a la
que de verdad ocurrió. Cero es azar puro.

El sistema saca 0,1090. Para que se hagan una idea, el mejor criterio clásico de los que probé —una
combinación de crecimiento a precio razonable— se queda en 0,0130. Es unas ocho veces más.

La t de Newey-West es 3,46, es decir, es distinguible de cero incluso teniendo en cuenta que los
meses se solapan. Y acierta el sentido en el setenta y cuatro por ciento de los meses.

Pero lo que de verdad importa es la caja dorada: en la era reservada, esa que apartamos desde el
principio y que no participó en ninguna decisión, el Rank-IC sigue siendo positivo. Más cuarenta y
cuatro diezmilésimas. Ordena fuera de muestra.

### 8 · ¿Y si fuera suerte? — 50 s

La pregunta obvia: ¿y si todo esto es suerte?

Lo ataqué de ocho maneras distintas. La del gráfico es la más contundente: barajé las respuestas al
azar diez mil veces y volví a medir. Ninguna de las nueve mil novecientas noventa y nueve
permutaciones llegó a lo que da el modelo. p igual a cero coma cero cero cero uno.

Y además: placebos de etiqueta, bootstrap por bloques, quitar eras enteras, cambiar la semilla, y
neutralizar los estilos de factor conocidos —después de eso conserva el ochenta y seis por ciento
de la señal.

Siete de ocho contrastes superados.

El octavo no lo supera, y lo digo yo: el Deflated Sharpe, que penaliza por cuántas configuraciones
he probado, se queda en 0,682 y no llega a su umbral. Es el precio declarado de haber encadenado
tres estudios.

Con eso el objetivo uno queda respondido: sí, el sistema aprende a ordenar.

---

## ACTO 2 — ¿Y la cartera?

### 9 · La cartera nunca se optimizó — 30 s

Y aquí empieza la segunda mitad.

Hasta ahora tengo una señal que ordena bien. Pero una ordenación no se puede comprar entera: una
cartera real compra unas pocas acciones, paga comisiones y rota.

En los tres estudios anteriores la cartera era simplemente la que venía por defecto. Nadie la había
optimizado, porque no era el objeto de estudio.

Así que la pregunta es: ¿da igual cómo la construya?

### 10 · Sí importa, y no por igual — 60 s

La respuesta es que importa muchísimo, y además de forma muy desigual.

Antes, qué es el Information Ratio: cuánto le saco al índice por cada unidad de riesgo que asumo.
Es la métrica que optimiza este segundo estudio.

En la tabla está, para cada variable de construcción de cartera, cuánto mueve el Information Ratio
entre su mejor y su peor valor.

El número de posiciones lo mueve tres décimas. Eso es enorme. El tope de efectivo, dos décimas.

Y abajo, dos variables que son prácticamente inertes: la tolerancia de deriva y cómo se reparten
los pesos apenas mueven una centésima.

O sea: no es que «la cartera importe» en abstracto. Es que hay dos decisiones concretas que se
llevan casi todo, y otras que dan igual. Eso es un resultado útil, y es el objetivo dos.

### 11 · 1.728 carteras, una ganadora — 55 s

Así que monté una rejilla completa: mil setecientas veintiocho carteras, todas sobre la misma señal
ya congelada. Sin reentrenar nada.

Cada punto del gráfico es una cartera. El eje horizontal es cuánto rota al año, el vertical su
Information Ratio. La ganadora es la que está rodeada arriba.

Y fíjense en la dispersión vertical: con la misma señal exacta hay carteras que dan menos de cero y
carteras que rozan el 0,85. Eso es, gráficamente, el objetivo dos.

La ganadora pasa el Information Ratio de 0,339 a 0,844, y el exceso sobre el índice del 2,6 al 7
por ciento.

Un detalle técnico por si me lo preguntan: la rejilla es cartesiana, no secuencial. Si hubiera
optimizado variable a variable habría elegido cinco posiciones, que es lo mejor en solitario, y la
ganadora usa ocho. Optimizando por separado no la habría encontrado.

### 12 · Y aguanta en la era reservada — 65 s

Y ahora el examen final: la era reservada, que no intervino en ninguna decisión.

A la derecha está la comparación que cierra el trabajo. Con la cartera que venía por defecto, la
señal perdía un once coma veintinueve por ciento frente al índice, con Information Ratio menos uno
con dieciséis.

Con la cartera optimizada —mismo modelo, misma señal, mismo panel, sin reentrenar absolutamente
nada— gana un dos coma cincuenta y seis por ciento, con Information Ratio más cero coma tres.

Y un dato que me parece clave: el Rank-IC de esa era es exactamente el mismo en los dos casos, más
0,0441, porque el Rank-IC no depende de la cartera. La capacidad de ordenar era idéntica. Lo que
cambió fue la gestión.

A la izquierda, la curva de la cartera ganadora contra el S&P 500, con la banda dorada marcando la
era reservada.

Objetivo dos respondido: sí, las variables de cartera importan, y optimizando por Information Ratio
se construye una que además confirma fuera de muestra.

---

## Cierre

### 13 · Qué no puedo afirmar — 45 s

Antes de terminar, lo que este trabajo no demuestra. Prefiero decirlo yo.

Uno: la era reservada son seis cohortes, poco más de un año de cartera. Es poca potencia. No puedo
tratarla como una validación definitiva.

Dos: la cartera ganadora es la mejor de mil setecientas veintiocho. Su 0,844 es una cota superior
optimista, no una estimación insesgada de lo que haría mañana.

Tres: el Deflated Sharpe se queda en 0,682. No hago ninguna afirmación de rentabilidad ajustada por
selección.

Y cuatro: como el agente de riesgo domina al meta, no puedo decir que la arquitectura de cinco
agentes esté demostrada.

Nada de esto es una recomendación de inversión.

### 14 · Conclusiones — 40 s

Resumo.

Objetivo uno: sí, un sistema de aprendizaje automático aprende a ordenar acciones, y esa ordenación
aguanta fuera de muestra. Con el matiz de que el agente de riesgo por su cuenta ordena algo mejor
que la combinación.

Objetivo dos: sí, las variables de construcción de cartera afectan al resultado, hay dos que se
llevan casi todo, y optimizando por Information Ratio se construye una cartera que casi triplica el
ratio y que confirma fuera de la ventana de decisión. Con el matiz de que es la mejor de mil
setecientas veintiocho y se confirma sobre seis cohortes.

Y si me tengo que quedar con una sola frase, es la de abajo: ninguna afirmación sobre la utilidad
de un sistema predictivo es interpretable sin declarar con qué cartera se midió. Un trabajo que
sólo reporte capacidad predictiva puede estar escondiendo tanto un éxito como un fracaso.

Muchas gracias.

---

## Turno de preguntas

Las diapositivas de reserva van después de «Gracias» y no cuentan en la numeración. El orden en el
PDF es el de esta tabla.

| Si preguntan… | Saltar a | Respuesta corta |
|---|---|---|
| ¿La cadena mejoró o sólo se sobreajustó? | R1 y R2 | Ambas cosas, y está declarado: mejora monótona **dentro** (0,1000 → 0,1074 → 0,1090) y degradación monótona **fuera** (+0,898 → +0,476 → −1,167). Esa divergencia es la firma del sobreajuste por búsqueda y es literalmente lo que motivó el Portfolio Study. |
| ¿Por qué domina `risk`? ¿Qué mira? | R3 | No es la prima clásica de baja volatilidad: `gap_21d` y `range_63d` encabezan el 78 % de las observaciones, y `beta_252d` ni está entre las tres primeras. Lee microestructura de precio a semanas. Por eso la neutralización por estilos conserva el 86,62 %. |
| ¿Y la robustez completa? | R4 | Siete de ocho superados. El que falla es el Deflated Sharpe (0,682 frente a un umbral de 0,95) y se reporta como falla, no se esconde. |
| ¿No es sobreajuste elegir 1 de 1.728? | R4 y R5 | Sí, y por eso el 0,844 se presenta como **cota superior optimista**. Lo que sostiene el objetivo 2 no es ese número sino la **dispersión**: que la misma señal produzca IR entre negativo y 0,85 según la cartera. Y la confirmación en la era reservada (+2,56 %) es fuera de la ventana de decisión. |
| ¿Por qué rejilla cartesiana y no secuencial? | R5 y R6 | Porque las variables interactúan: marginalmente el mejor `target_size` es 5, pero la ganadora usa 8. Optimizando variable a variable no se habría encontrado. |
| ¿Aguanta por eras? | R7 | `risk` es el único positivo en las cuatro. En la era reservada todos los agentes fundamentales se vuelven negativos **a la vez**, lo que apunta a rotación de factores del mercado, no a un error de signo. |
| ¿Y los perfiles de inversión? | R8 | Quedaron fuera de la rejilla a propósito: se construyen sobre la cartera ya elegida y no influyeron en su selección. El perfil que **no** reordena la señal (`balanced`, IR 0,844) domina la ventana de selección, y **el orden de los otros siete se predice desde el Rank-IC de los agentes que cada uno pondera**: `defensive` (0,570) carga 0,60 en `risk`, el mejor agente; `momentum` (0,017) carga 0,75 en el peor y encima penaliza a `risk`. Un perfil no añade información, redistribuye la que ya hay. |
| ¿Cuánto cae la cartera? | R9 | 28,40 % de máxima caída en selección, 12,09 % en la era reservada. Es una cartera de 8 posiciones: concentrada por diseño. |
| ¿De dónde salen los datos? ¿Y el sesgo de supervivencia? | R10 | Composición **histórica** del S&P 500, no la actual: una empresa sólo es elegible en las fechas en que pertenecía al índice. Fechas de publicación reales de SEC EDGAR. Cobertura ≥ 99,4 % cada año. |
| ¿Costes realistas? | R9 | 5 pb de comisión más 10 pb de *slippage*, constantes. Es una limitación declarada: no modelo impacto de mercado ni capacidad, y con 324 % de rotación anual sobre 8 posiciones eso importa. |
| ¿Esto se puede usar para invertir? | 13 | No. Seis cohortes de confirmación, DSR por debajo del umbral y una cartera elegida entre 1.728. La contribución es metodológica. |

### Dos preguntas incómodas, con respuesta preparada

**«Si `risk` solo es mejor que el meta, ¿para qué los cinco agentes?»**
Es la objeción correcta y está en la memoria antes que en el tribunal. La respuesta honesta: con
estos datos la arquitectura multi-agente no queda demostrada. Lo que sí queda demostrado es que el
meta **aprende** —parte de pesos iguales, se equivoca en 2016 y se corrige— y que no se le fijó de
antemano el ganador. Que acabe concentrando en uno es un resultado, no un fallo de diseño; pero
convierte al meta en un selector más que en un combinador, y así se declara.

**«El −11,29 % de la cartera por defecto, ¿no invalida el objetivo 1?»**
No, y por una razón medible: el Rank-IC de la era reservada es **+0,0441 con las dos carteras**,
porque no depende de la cartera. La capacidad de ordenar es la misma en los dos casos. Lo que
cambia es la traducción a posiciones. De hecho ese contraste es justamente la evidencia del
objetivo 2.
