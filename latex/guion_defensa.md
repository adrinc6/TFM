# Guion de defensa

> El guion hablado vive en los `\note{}` de `TFM_ppt.tex`, junto a la diapositiva que acompaña.
> Este documento conserva el reparto de tiempo, el orden de recorte y el mapa de preguntas. Las
> cifras se leen de los artefactos adoptados; no se mantienen aquí como una segunda fuente.
>
> Para proyectar con las notas en la segunda pantalla: `python latex/build.py --solo-defensa --notas`,
> que genera `TFM_ppt_notes.pdf` sin tocar el fichero original.

## Estructura

La defensa tiene **32 diapositivas narradas**, **doce de reserva** sin numerar, y se organiza en
**seis bloques** más un cierre. El pie muestra el bloque en curso y una barra de progreso. Sólo los
tres bloques centrales abren con divisor a pantalla completa: gastar seis páginas en divisores dentro
de veinte minutos no sale a cuenta.

| Bloque | Diapositivas | Qué establece |
|---|---|---|
| **1 · El problema** | 1–4 | Por qué casi nadie bate al índice, los dos objetivos y las tres preguntas |
| **2 · Cómo aprende** | 5–9 | Qué es ordenar, por qué rangos, sobreajuste, walk-forward, catálogo cerrado |
| **3 · Los datos** | 10–11 | De dónde salen y qué se hace para que la evaluación sea honesta |
| **4 · El sistema** | 12–15 | Cinco agentes disjuntos y un meta que aprende a ponderarlos |
| **5 · Los estudios** | 16–19 | La cadena, la optimización secuencial y el muro de la era reservada |
| **6 · Resultados** | 20–30 | Las tres preguntas resueltas, perfiles y composición de la cartera |
| **Cierre** | 31–32 | Límites y conclusión juntos, y agradecimiento |

El hilo es una **incógnita que se resuelve por partes**. La diapositiva 4 plantea tres preguntas con
un interrogante al lado; cada interrogante se sustituye por una cifra en la diapositiva donde queda
demostrado —la 21, la 25 y la 30—, y la 31 recoge las tres ya resueltas. No hay `\pause` en todo el
fichero: donde haría falta una revelación, hay una diapositiva propia.

**Por qué este orden.** La versión anterior entraba directa al resultado. Ésta cuenta primero cómo
funciona el sistema y con qué datos, de modo que cuando llegan las cifras el tribunal ya sabe
interpretarlas. Los bloques 2 y 3 son el precio de esa decisión: nueve minutos antes del primer
resultado. A cambio, ningún número necesita explicarse dos veces.

## Reparto de tiempo

Las duraciones se calculan a **150 palabras por minuto sobre el texto real de cada nota**, no a ojo.
Si se reescribe una nota, hay que recalcularlas. Total: **21:10** sobre un objetivo de veinte
minutos, que a ritmo de defensa cae dentro.

| Bloque | Diapositivas | Tiempo | Acumulado |
|---|---:|---:|---:|
| 1 · El problema | 1–4 | 2:21 | 2:21 |
| 2 · Cómo aprende | 5–9 | 3:18 | 5:40 |
| 3 · Los datos | 10–11 | 1:46 | 7:26 |
| 4 · El sistema | 12–15 | 2:22 | 9:48 |
| 5 · Los estudios | 16–19 | 2:06 | 11:54 |
| 6 · Resultados | 20–30 | 8:02 | 19:57 |
| Cierre | 31–32 | 1:13 | **21:10** |

Si hace falta recortar, en este orden:

1. Diapositiva 6 (por qué rangos): enunciar la conclusión sin desarrollar el argumento de las colas.
2. Diapositiva 18 (dieciocho fases): decir que dieciséis no cambiaron nada, sin la lectura de robustez.
3. Diapositiva 29 (qué compró): dar sólo la concentración, sin recorrer sectores ni nombres.
4. Diapositiva 24 (atribución factorial): enunciar el 87,2 % sin desarrollar la objeción.

**No acelerar las diapositivas 3, 4, 7, 19, 23, 25, 30 y 31**: fijan los dos objetivos, abren la
incógnita, explican el sobreajuste, declaran el muro de la era reservada, exponen el contraste que
falla, resuelven la bisagra entre objetivos, cierran fuera de muestra y delimitan el alcance.

## Las tres preguntas y dónde se resuelven

| # | Pregunta | Se abre | Se resuelve | Con qué cifra |
|---:|---|---:|---:|---|
| 1 | ¿Aprende a ordenar fuera de muestra? | 4 | **21** | Rank-IC nueve veces la mejor fórmula sin aprendizaje |
| 2 | ¿Cuánto pesan las reglas de cartera? | 4 | **25** | Information Ratio casi ×3 con la señal congelada |
| 3 | ¿Sobrevive en la ventana nunca consultada? | 4 | **30** | De destruir valor a empatar, sin reentrenar nada |

## Turno de preguntas

| Si preguntan… | Volver a | Respuesta corta |
|---|---:|---|
| ¿Por qué es tan difícil batir al índice? | 2 | La aritmética de Sharpe: antes de costes la gestión activa agregada iguala al mercado, y después queda por debajo. SPIVA aporta el correlato empírico. |
| ¿Qué es exactamente el rank-IC? | 5 | La correlación entre el orden predicho y el que ocurre después. No mide rentabilidad: mide si acerté quién iba delante de quién. |
| ¿Cómo evitas el sobreajuste? | 7, 8 y 9 | Walk-forward con reentreno mensual, catálogo cerrado antes de mirar, y una era reservada que no participa en ninguna decisión. |
| Si `risk` supera al meta, ¿para qué cinco agentes? | **R1** | El trabajo lo reporta: no queda demostrado que hagan falta cinco. Sí que el meta aprende sin que se le fije el ganador, y que en 2015 elegir `risk` no era una opción disponible. |
| ¿No será un factor clásico con otro nombre? | 24 | Al neutralizar los estilos conocidos, la capacidad de ordenar retiene el 87,2 %. El alfa factorial no es significativa, y eso se declara. |
| ¿Qué robustez falla? | 23 y **R8** | El Deflated Sharpe, que penaliza la multiplicidad. Por eso no se reclama rentabilidad ajustada por búsqueda aunque sí haya evidencia de ordenación. |
| ¿No es sobreajuste elegir una cartera de una rejilla grande? | 25 y 27 | La cifra de la ganadora es una cota superior optimista. La evidencia estable del objetivo 2 es la dispersión con la misma señal congelada. |
| ¿Por qué una rejilla cartesiana? | **R10** | Porque las reglas interactúan: el suelo de cobertura significa cosas opuestas según el tope de efectivo. Optimizar cada eje por separado no habría encontrado esta cartera. |
| ¿Y si se impone un estilo de inversión? | 28 | Se midió con ocho perfiles. El consenso aprendido obtiene 0,841 y el mejor estilo fijo 0,342; en la era reservada seis de los siete se vuelven negativos. |
| ¿Qué compró realmente? | 29 | Ocho posiciones simultáneas, 42 acciones distintas en diez años, permanencia mediana de 15 meses, sesgo hacia tecnología pero repartida en veinte sectores. |
| ¿El resultado depende de pocas acciones? | 29 y **R6** | Sí, y se reporta como limitación: una sola posición aporta más de un cuarto de la contribución neta. |
| ¿De dónde salen los datos y qué sesgo queda? | 10, 11 y **R5** | Cuatro fuentes; la composición del índice y las fechas de publicación son históricas. El sesgo residual es la cobertura incompleta del proveedor en los años antiguos: medido en causa y dirección, decreciente, no cuantificado en puntos. |
| ¿La señal aguanta igual en todas las épocas? | 22 y **R4** | No es homogénea: es más fuerte en la era reciente. Eso contradice la explicación simple por sesgo de cobertura, que sería máximo en la era antigua. |
| ¿Los costes son realistas? | **R7** | El margen se mide, no se afirma: el exceso aguanta hasta unas diez veces el coste adoptado en selección. En la era reservada el margen es mucho menor. |
| ¿A qué patrimonio aplica esto? | **R2** | El ámbito de un patrimonio individual o familiar, no el de un fondo. |
| ¿Esto se puede usar para invertir? | 31 | No como recomendación. La confirmación económica es corta y la búsqueda fue amplia. La contribución defendible es metodológica. |

## Diapositivas de reserva

| # | Título | Para qué pregunta |
|---:|---|---|
| R1 | El agente de riesgo solo bate al meta | Si hacen falta cinco agentes |
| R2 | Capacidad: hasta qué tamaño aguanta | A qué patrimonio aplica el resultado |
| R3 | Por qué tres pasadas encadenadas | Si encadenar estudios es sobreajuste |
| R4 | La señal por eras | Si la capacidad predictiva es homogénea |
| R5 | El sesgo de cobertura, medido | Calidad de los datos y supervivencia |
| R6 | El resultado se concentra en pocos nombres | Cuántas apuestas independientes lo sostienen |
| R7 | Margen antes de que los costes se lo coman | Si los costes son realistas |
| R8 | Deflated Sharpe: el contraste que falla | Qué robustez no se supera y qué implica |
| R9 | Los ocho perfiles informativos | Si imponer un estilo habría sido mejor |
| R10 | Las tres reglas que mueven la rentabilidad | Por qué una rejilla cartesiana |
| R11 | Ordenar bien no es acertar las que compras | Por qué el orden no se cobra solo |
| R12 | La muestra sobre la que se mide todo | Sobre qué datos se calcula exactamente |

## Tres respuestas que conviene ensayar

**«Si el agente de riesgo supera al meta, ¿para qué cinco agentes?»**

La arquitectura multiagente no queda demostrada como necesaria, y el trabajo lo dice. Sí queda
demostrado que el meta aprende sin que se le fije de antemano el especialista ganador: parte de pesos
iguales, se equivoca en 2016 concentrándose en el peor agente del periodo, y se corrige solo. Que
termine concentrándose es un resultado del protocolo, no permiso para reescribir el diseño después de
observarlo.

**«Que la cartera original falle en la era reservada, ¿invalida la señal?»**

No. El rank-IC es idéntico con cualquier cartera, porque se calcula antes de construirla —es la cifra
que aparece abajo a la derecha en la diapositiva 30—. El cambio económico al modificar sólo las reglas
es precisamente la evidencia del objetivo 2: señal e implementación son capas distintas, y
confundirlas habría llevado a concluir que la señal no se traslada fuera de muestra.

**«Entonces, ¿bate al mercado o no?»**

No está demostrado. Lo que está demostrado es que con la cartera por defecto perdía y con la
optimizada empata, sin reentrenar nada. En un juego de suma cero eso es un resultado sobre dónde
estaba el cuello de botella —la implementación, no el modelo—, no una ventaja competitiva. Con seis
cohortes cerradas y año y medio de cartera, afirmar más sería exactamente el tipo de lectura que este
trabajo se ha propuesto evitar.
