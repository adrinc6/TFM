# Guion de defensa

> El guion hablado vive en los `\note{}` de `TFM_ppt.tex`, junto a la diapositiva que acompaña.
> Este documento conserva el reparto de tiempo, el orden de recorte y el mapa de preguntas. Las
> cifras se leen de los artefactos adoptados; no se mantienen aquí como una segunda fuente.
>
> Para proyectar con las notas en la segunda pantalla, descomentar en `TFM_ppt.tex`:
> `\setbeameroption{show notes on second screen=right}`.

## Estructura

La defensa tiene **26 diapositivas narradas**, **siete de reserva** sin numerar y se organiza en
tres actos con divisor propio. El pie muestra el acto en curso y una barra de progreso, de modo que
el tribunal sabe en todo momento en qué mitad del argumento está.

| Acto | Diapositivas | Qué establece |
|---|---|---|
| **El problema** | 1–4 | Los dos objetivos, el listón que mide cada uno y las tres preguntas |
| **Acto I · ¿Aprende?** | 5–17 | La señal existe, aporta sobre no aprender y sobre no imponer estilo, resiste siete de ocho contrastes y no se cobra sola |
| **Acto II · ¿Se cobra?** | 18–24 | La cartera decide el signo del resultado; empatar no es batir |
| **Cierre** | 25–26 | Las tres preguntas resueltas y el alcance de lo demostrado |

El hilo es una **incógnita que se resuelve por partes**. La diapositiva 4 plantea tres preguntas con
un interrogante al lado; cada interrogante se sustituye por una cifra en la diapositiva donde queda
demostrado —la 12, la 19 y la 24—, y la 25 las reúne ya resueltas. No hay `\pause` en todo el
fichero: donde haría falta una revelación, hay una diapositiva propia.

**El arranque es deliberado.** La diapositiva 2 son los dos objetivos, no el listón de mercado:
abrir con «batir al índice es un juego de suma cero» haría parecer que ése es el objetivo del
trabajo, cuando la contribución es metodológica. La 3 introduce el listón después, ya subordinado, y
sirve a los dos objetivos: es la aritmética contra la que se mide el segundo, y la razón por la que
el primero exige tanta disciplina de medición.

## Reparto de tiempo

Las duraciones se calculan a **150 palabras por minuto** sobre el texto real de cada nota, no a
ojo. Si se reescribe una nota, hay que recalcularlas. Con veintiséis diapositivas en quince
minutos el ritmo es vivo: **la carga la llevan las figuras**, y la voz acompaña.

| # | Acto | Diapositiva | s | Acumulado |
|---:|---|---|---:|---:|
| 1 | Problema | Portada | 25 | 0:25 |
| 2 | Problema | **Dos objetivos, dos listones, dos métricas** | 35 | 1:00 |
| 3 | Problema | Contra qué se mide cada objetivo | 45 | 1:45 |
| 4 | Problema | **Tres preguntas que esta defensa deja resueltas** | 30 | 2:15 |
| 5 | I | *Divisor* · ¿Aprende el sistema a ordenar acciones? | 10 | 2:25 |
| 6 | I | Cinco especialistas y un meta-agente | 30 | 2:55 |
| 7 | I | Sobre qué datos se mide todo esto | 25 | 3:20 |
| 8 | I | Cuatro condiciones que hacen honesta la evaluación | 40 | 4:00 |
| 9 | I | Las decisiones terminan antes de abrir la era reservada | 30 | 4:30 |
| 10 | I | El meta se equivoca, se corrige y acaba concentrándose | 35 | 5:05 |
| 11 | I | Aprender los pesos aporta sobre repartirlos a partes iguales | 30 | 5:35 |
| 12 | I | **Pregunta 1, resuelta**: aprende lo que las fórmulas no ven | 40 | 6:15 |
| 13 | I | La señal aparece en tres de cada cuatro fechas | 30 | 6:45 |
| 14 | I | ¿Y si esto ya lo hacían los factores conocidos? | 35 | 7:20 |
| 15 | I | Ocho contrastes: siete a favor, uno en contra | 55 | 8:15 |
| 16 | I | **Aprender el reparto gana a decidirlo por convicción** | 70 | 9:25 |
| 17 | I | Ordenar bien el universo no garantiza acertar las que compras | 45 | 10:10 |
| 18 | II | *Divisor* · ¿Puede cobrarse ese orden frente al mercado? | 10 | 10:20 |
| 19 | II | **Pregunta 2, resuelta**: 1.440 carteras sobre la misma señal | 35 | 10:55 |
| 20 | II | Tres reglas concentran casi toda la sensibilidad | 30 | 11:25 |
| 21 | II | La ganadora opera menos, y por eso gana | 45 | 12:10 |
| 22 | II | Qué habría ganado quien la hubiera seguido | 40 | 12:50 |
| 23 | II | Qué compró: pocas acciones y mucho tiempo | 30 | 13:20 |
| 24 | II | **Pregunta 3, resuelta**: la cartera cambia el signo | 60 | 14:20 |
| 25 | Cierre | Las tres preguntas, resueltas | 35 | 14:55 |
| 26 | — | Gracias | 10 | **15:05** |

Si hace falta recortar, hacerlo en este orden:

1. Diapositiva 20: decir solo cuáles mueven y cuáles no, sin las cifras.
2. Diapositiva 23: resumir la concentración en una frase, sin comentar los tres paneles.
3. Diapositiva 7: dar la escala del panel sin detallar el recambio del universo.
4. Diapositiva 14: enunciar el 87,2 % sin desarrollar la objeción factorial.

**No acelerar las diapositivas 3, 4, 12, 15, 16, 17, 24 y 25**: fijan el listón, abren la incógnita,
resuelven la primera pregunta, declaran el contraste que falla, cierran el argumento del
aprendizaje, sostienen la bisagra entre actos y delimitan el alcance final.

## Las tres preguntas y dónde se resuelven

| # | Pregunta | Se abre | Se resuelve | Con qué cifra |
|---:|---|---:|---:|---|
| 1 | ¿Aprende a ordenar fuera de muestra? | 4 | **12** | Rank-IC 0,1067: nueve veces la mejor fórmula sin aprendizaje |
| 2 | ¿Cuánto pesan las reglas de cartera? | 4 | **19** | Information Ratio ×2,7 con la señal congelada |
| 3 | ¿Sobrevive en la ventana nunca consultada? | 4 | **24** | Exceso +0,24 %, de destruir valor a empatar |

## Turno de preguntas

| Si preguntan… | Volver a | Respuesta corta |
|---|---:|---|
| ¿Por qué es tan difícil batir al índice? | 3 | La aritmética de Sharpe: antes de costes la gestión activa agregada iguala al mercado, y después queda por debajo. SPIVA aporta el correlato empírico. |
| ¿Aprender aporta algo de verdad? | 11 y 12 | Dos comparaciones: los pesos aprendidos baten al reparto uniforme (0,0690 → 0,1067), y el sistema ordena nueve veces mejor que la mejor fórmula determinista sobre el mismo panel. |
| Si `risk` supera al meta, ¿para qué cinco agentes? | **R1** | El trabajo lo reporta: no queda demostrado que hagan falta cinco. Sí queda demostrado que el meta aprende sin que se le fije el ganador de antemano, y que en 2015 elegir `risk` no era una opción disponible. |
| ¿No será un factor clásico con otro nombre? | 14 | Al neutralizar las exposiciones a los estilos conocidos, la capacidad de ordenar retiene el 87,2 %. El alfa factorial, en cambio, no es significativo, y eso se declara. |
| ¿Qué robustez falla? | 15 | El Deflated Sharpe, que penaliza la multiplicidad. Por eso no se reclama rentabilidad ajustada por búsqueda aunque sí haya evidencia favorable de ordenación. |
| ¿La cadena mejoró o se sobreajustó? | 17 y **R3** | Mejoró el orden en selección y no logró cobrarlo fuera con la cartera de partida. Esa divergencia motiva estudiar la implementación sin reabrir el modelo. |
| ¿No es sobreajuste elegir una cartera de una rejilla grande? | 19 y 22 | La cifra de la ganadora es una cota superior optimista. La evidencia estable del objetivo 2 es la dispersión con la misma señal congelada. |
| ¿Por qué una rejilla cartesiana? | 20 | Porque las seis reglas interactúan: el suelo de cobertura significa cosas opuestas según el tope de efectivo. Optimizar cada eje por separado no habría encontrado esta cartera. |
| ¿Cuánto habría ganado de verdad? | 22 | CAGR 19,86 % frente al 13,17 % del índice, costes descontados, batiendo al índice los diez años. Y con la salvedad: es la ventana sobre la que se optimizó. |
| ¿El resultado depende de pocas acciones? | 23 | Sí, y se reporta como limitación: 42 acciones en total y una sola aporta 27,8 puntos de contribución neta. |
| ¿De dónde salen los datos y qué sesgo queda? | 7, 8 y **R5** | La composición del índice y las fechas de publicación son históricas. El sesgo residual es la cobertura incompleta del proveedor en los años antiguos: medido en causa y dirección, decreciente, y no cuantificado en puntos de rentabilidad. |
| ¿La señal aguanta igual en todas las épocas? | 13 y **R4** | No es homogénea: es más fuerte en la era reciente. Eso contradice la explicación simple por sesgo de cobertura, que sería máximo en la era antigua. |
| ¿Los costes son realistas? | **R7** | El margen se mide, no se afirma: el exceso aguanta hasta unas diez veces el coste adoptado en selección. En la era reservada casi no hay margen. |
| ¿A qué patrimonio aplica esto? | **R2** | Unos 20 millones con el 5 % de participación, 40 con el 10 %. Es el ámbito de un patrimonio individual o familiar, no el de un fondo. |
| ¿Qué son los perfiles, y por qué no gana ninguno? | 16 | Un perfil usa los mismos cinco agentes entrenados pero fija los pesos de antemano. El consenso aprendido obtiene IR 0,841 y el mejor estilo fijo 0,342; en la reserva, seis de los siete se vuelven negativos. No compiten por ser elegidos: se reevalúan como diagnóstico. |
| ¿Esto se puede usar para invertir? | 25 | No como recomendación. La confirmación económica es corta y la búsqueda fue amplia. La contribución defendible es metodológica. |

## Diapositivas de reserva

| # | Título | Para qué pregunta |
|---:|---|---|
| R1 | El matiz incómodo: `risk` solo bate al meta | Si hacen falta cinco agentes |
| R2 | Capacidad: hasta qué tamaño aguanta | A qué patrimonio aplica el resultado |
| R3 | Por qué tres pasadas encadenadas | Si encadenar estudios es sobreajuste |
| R4 | La señal por eras | Si la capacidad predictiva es homogénea en el tiempo |
| R5 | El sesgo de cobertura, medido | Calidad de los datos y sesgo de supervivencia |
| R6 | El resultado se concentra en pocos nombres | Cuántas apuestas independientes lo sostienen |
| R7 | Margen antes de que los costes se lo coman | Si los costes son realistas |

## Tres respuestas que conviene ensayar

**«Si `risk` supera al meta, ¿para qué cinco agentes?»**

La arquitectura multiagente no queda demostrada como necesaria, y el trabajo lo dice. Sí queda
demostrado que el meta aprende sin que se le fije de antemano el especialista ganador: parte de
pesos iguales, se equivoca en 2016 concentrándose en el peor agente del periodo, y se corrige solo.
Que termine concentrándose es un resultado del protocolo, no permiso para reescribir el diseño
después de observarlo.

**«Que la cartera original falle en reserva, ¿invalida la señal?»**

No. El Rank-IC es idéntico con cualquier cartera, porque se calcula antes de construirla —es el
0,0364 que aparece abajo a la derecha en la diapositiva 24—. El cambio económico al modificar solo
las reglas es precisamente la evidencia del objetivo 2: señal e implementación son capas distintas,
y confundirlas habría llevado a concluir que la señal no se traslada fuera de muestra.

**«Entonces, ¿bate al mercado o no?»**

No está demostrado. Lo que está demostrado es que con la cartera por defecto perdía y con la
optimizada empata, sin reentrenar nada. En un juego de suma cero eso es un resultado sobre dónde
estaba el cuello de botella —la implementación, no el modelo—, no una ventaja competitiva. Con seis
cohortes cerradas y año y medio de cartera, afirmar más sería exactamente el tipo de lectura que
este trabajo se ha propuesto evitar.
