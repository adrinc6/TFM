# Guion de defensa

> **El guion hablado no vive aquí.** Vive en los `\note{}` de `presentacion.tex`, junto a la
> diapositiva que acompaña, y ése es el único sitio donde se edita. Este documento recoge lo que no
> cabe en una nota: el reparto de tiempo, el orden de recorte si hay que apretar y las preguntas
> previsibles con su respuesta corta.
>
> Antes tenía además una transcripción completa del guion. Se retiró porque se desincronizó: llegó a
> citar un Rank-IC, un Deflated Sharpe y un número de carteras que ya no eran los del estudio
> adoptado. Es el mismo motivo por el que las cifras del proyecto no se copian en ningún `.md` y se
> leen de los artefactos.
>
> Para proyectar con las notas en la segunda pantalla, descomentar en `presentacion.tex`:
> `\setbeameroption{show notes on second screen=right}`.
>
> **No hay material de reserva.** Son 22 diapositivas seguidas: portada, 20 de contenido y
> despedida. Todo lo que antes era anexo y aportaba —la divergencia de la cadena, qué mira el agente
> dominante, la tabla completa de robustez, el detalle de la cartera ganadora, el Rank-IC por eras y
> la procedencia de los datos— está en el cuerpo, explicado y defendido.

## Estructura

La charla son **dos objetivos**, y conviene tenerlos siempre presentes porque todo cuelga de ahí:

| | Objetivo | Lo demuestra | Métrica | Respuesta |
|---|---|---|---|---|
| **1** | Un ML **aprende a ordenar acciones** fuera de muestra | Los tres Model Studies | Rank-IC | Sí, con matices |
| **2** | **Las variables de cartera importan**, y optimizando por IR se construye una buena | El Portfolio Study | Information Ratio | Sí, con matices |

El acto 1 responde al primero, el acto 2 al segundo, y el cierre acota los dos. La bisagra entre
ambos es la diapositiva 13: la cadena converge dentro de la ventana de selección y no se cobra
fuera. Ése es el problema que el objetivo 2 viene a resolver, y por eso el acto 2 no es un apéndice.

## Reparto de tiempo

| # | Diapositiva | s | Acumulado |
|---|---|---|---|
| 1 | Portada | 20 | 0:20 |
| 2 | Dos objetivos, deliberadamente separados | 50 | 1:10 |
| 3 | El sistema: cinco especialistas y un árbitro | 65 | 2:15 |
| 4 | De dónde salen los datos | 60 | 3:15 |
| 5 | La regla que lo condiciona todo: nunca mirar el futuro | 60 | 4:15 |
| 6 | Cómo se decide sin hacer trampa | 60 | 5:15 |
| | **Acto 1 — ¿aprende a ordenar?** | | |
| 7 | La métrica del objetivo 1: Rank-IC | 50 | 6:05 |
| 8 | El sistema se equivoca… y se corrige solo | 70 | 7:15 |
| 9 | Qué mira realmente el agente que domina | 60 | 8:15 |
| 10 | La ordenación es buena, y no sólo dentro de muestra | 60 | 9:15 |
| 11 | Rank-IC por era: dónde funciona y dónde no | 60 | 10:15 |
| 12 | ¿Y si fuera suerte? Ocho contrastes | 65 | 11:20 |
| 13 | **La cadena: converge dentro, no se cobra fuera** | 65 | 12:25 |
| | **Acto 2 — ¿y la cartera?** | | |
| 14 | Contra qué se compite: un juego de suma cero | 45 | 13:10 |
| 15 | Pero la cartera nunca se había optimizado | 40 | 13:50 |
| 16 | Sí importa, y no todas las variables por igual | 65 | 14:55 |
| 17 | 1.440 carteras sobre la misma señal, una ganadora | 60 | 15:55 |
| 18 | La cartera ganadora, en detalle | 55 | 16:50 |
| 19 | El contraste que cierra el trabajo | 70 | 18:00 |
| | **Cierre** | | |
| 20 | Qué no puedo afirmar | 55 | 18:55 |
| 21 | Conclusiones | 55 | 19:50 |
| 22 | Gracias | — | — |

**Total hablado: 19:50.** Está por encima del objetivo **a propósito**: el texto de los `\note{}` es
la versión desarrollada, y en un ensayo real siempre se recorta al hablar. A ritmo normal y sin leer
palabra por palabra cae a 15–16 minutos. Si hace falta apretar, este es el orden de recorte:

1. **Diapositiva 4** (datos): de 60 a 35 s. Sólo las fuentes y el matiz de cobertura.
2. **Diapositiva 11** (Rank-IC por era): de 60 a 35 s. Quedarse con «`risk` es el único positivo en
   las cuatro» y «los demás se hunden todos a la vez en la reservada».
3. **Diapositiva 9** (qué mira `risk`): de 60 a 40 s. Sólo el contraste beta *vs* microestructura.
4. **Diapositiva 3** (el sistema): de 65 a 45 s, sin enumerar los cinco agentes uno a uno.
5. **Diapositiva 18** (cartera en detalle): de 55 a 35 s, dejando caída máxima y rotación.

**Nunca se aceleran las diapositivas 13, 16, 17 y 19**: son la bisagra y el objetivo 2 entero.

---

## Turno de preguntas

**No hay diapositivas de reserva.** Las respuestas se dan **volviendo a una diapositiva ya
proyectada**; la columna «Volver a» indica cuál.

| Si preguntan… | Volver a | Respuesta corta |
|---|---|---|
| ¿La cadena mejoró o sólo se sobreajustó? | **13** | Las dos cosas, y está declarado: mejora monótona **dentro** de la ventana de selección, por milésimas, y Information Ratio negativo en la era reservada en las tres pasadas. Esa divergencia es la firma del sobreajuste por búsqueda, y es literalmente lo que motivó el Portfolio Study. |
| ¿Por qué domina `risk`? ¿Qué mira? | **9** | No es la prima clásica de baja volatilidad: `gap_21d` y `range_63d` encabezan cuatro de cada cinco observaciones y `beta_252d` ni está entre las tres primeras. Lee microestructura de precio a semanas. Por eso la neutralización por catorce controles de estilo conserva la mayor parte de la señal. |
| ¿Y la robustez completa? | **12** | Siete de ocho superados. El que falla es el Deflated Sharpe, por debajo de su umbral de 0,95, y se reporta como falla en la misma tabla que los otros siete. |
| ¿No es sobreajuste elegir 1 de 1.440? | **17** y **20** | Sí, y por eso el Information Ratio de la ganadora se presenta como **cota superior optimista**. Lo que sostiene el objetivo 2 no es ese número sino la **dispersión**: que la misma señal congelada produzca desde IR negativo hasta 0,84 según cómo se construya la cartera. Y la confirmación de la era reservada queda fuera de la ventana de decisión. |
| ¿Por qué rejilla cartesiana y no secuencial? | **17** | Porque las variables interactúan. Marginalmente el mejor suelo de cobertura es 60 y el mejor reparto de pesos es el equiponderado; la ganadora usa suelo 0 y reparto proporcional al alfa. Optimizando variable a variable no se habría encontrado. |
| ¿Aguanta por eras? | **11** | `risk` es el único positivo en las cuatro. En la era reservada los cuatro agentes distintos de `risk` se vuelven negativos **a la vez**, lo que apunta a rotación de factores del mercado y no a un error de signo en un modelo. |
| ¿Cuánto cae la cartera? | **18** | La máxima caída de la ganadora es **menor** que la de la cartera de partida, y eso es contraintuitivo porque además renuncia al colchón de efectivo. La explicación es la tenencia mínima: al no deshacer posiciones a mitad de horizonte no cristaliza caídas transitorias. Aun así, que una cartera de ocho nombres caiga como un índice de quinientos no es control de riesgo: es consecuencia de la señal. |
| ¿De dónde salen los datos? ¿Y el sesgo de supervivencia? | **4** | Composición **histórica** del S&P 500, no la actual: una empresa sólo es elegible en las fechas en que perteneció al índice. Fechas de publicación reales de SEC EDGAR. El sesgo que queda está medido y es de **cobertura del proveedor**, no de mortalidad: la cobertura del índice sube del 49 % en 2003 al 99 % en 2026, y sólo el 5,5 % de los ausentes llevan marcador de quiebra. Ojo con la otra columna de esa tabla: el 99,4 % es la fracción utilizable de las filas construidas, que mide calidad y no alcance. |
| ¿Costes realistas? | **18** | 10 pb de comisión más 20 pb de *slippage*, constantes. Y está medido hasta dónde aguanta: el exceso no desaparece hasta casi diez veces ese coste. Lo que no se modela es el impacto de mercado, y por eso la capacidad se reporta como participación sobre el volumen habitual y acota el trabajo a un patrimonio pequeño. |
| ¿Y los perfiles de inversión? | — | Quedaron **fuera de la rejilla** a propósito: se construyen sobre la cartera ya elegida y no influyeron en su selección. El perfil que **no** reordena la señal domina la ventana de selección y los siete que sí la reordenan quedan por debajo, en un orden que se predice desde el Rank-IC de los agentes que cada uno pondera. **Un perfil no añade información, redistribuye la que ya hay** — por eso no está en la presentación. |
| ¿Por qué bloques de variables disjuntos? | **3** | Si los cinco agentes vieran las mismas 68 variables serían cinco copias del mismo predictor y el meta no tendría nada que arbitrar. El coste es que cada agente ve menos; la ventaja es que sus errores son razonablemente independientes. |
| ¿Qué compró realmente la cartera? | **18** | Cuarenta y dos acciones en diez años, siempre ocho posiciones y sin efectivo, con una permanencia mediana de quince meses. El resultado está concentrado: la primera posición aporta casi el doble que la segunda, y eso figura como limitación en la memoria. |
| ¿Esto se puede usar para invertir? | **20** | No. Seis cohortes de confirmación, Deflated Sharpe por debajo del umbral y una cartera elegida entre 1.440. La contribución es metodológica. |

### Tres preguntas incómodas, con respuesta preparada

**«Si `risk` solo es mejor que el meta, ¿para qué los cinco agentes?»**
Es la objeción correcta y está en la memoria antes que en el tribunal. La respuesta honesta: con
estos datos la arquitectura multi-agente **no queda demostrada**. Lo que sí queda demostrado es que
el meta **aprende** —parte de pesos iguales, se equivoca en 2016 concentrando en el que resultará ser
el peor agente, y se corrige solo— y que no se le fijó de antemano el ganador. Elegir `risk` de
antemano no era una opción disponible en 2015. Que acabe concentrando en uno es un resultado, no un
fallo de diseño; pero convierte al meta en un selector más que en un combinador, y así se declara.
(Diapositivas 8 y 20.)

**«Que la cartera por defecto pierda en la era reservada, ¿no invalida el objetivo 1?»**
No, y por una razón medible: el Rank-IC de la era reservada es **el mismo con las dos carteras**,
porque no depende de la cartera. La capacidad de ordenar es idéntica en los dos casos; lo que cambia
es la traducción a posiciones, hasta el punto de cambiar el signo del resultado. Ese contraste no es
un problema para el objetivo 1: es justamente la evidencia del objetivo 2. (Diapositiva 19.)

**«Si el Deflated Sharpe no pasa, ¿qué queda en pie?»**
Queda en pie el objetivo 1, que **no se decide por Sharpe sino por Rank-IC**, y que supera siete
contrastes incluyendo permutación con p = 0,0001 y neutralización por catorce controles de estilo.
El Deflated Sharpe penaliza la búsqueda, y este trabajo ha buscado mucho: eso afecta a cualquier
afirmación de *rentabilidad ajustada por selección*, y por eso no se hace ninguna. La contribución
que se reclama es metodológica —que ordenar y rentar son cosas distintas y hay que medirlas por
separado— y ésa no depende del Deflated Sharpe. (Diapositivas 12 y 20.)
