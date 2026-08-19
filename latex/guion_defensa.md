# Guion de defensa

> El guion hablado vive en los `\note{}` de `presentacion.tex`, junto a la diapositiva que
> acompaña. Este documento conserva el reparto de tiempo, el orden de recorte y el mapa de
> preguntas. Las cifras económicas se leen de los activos y artefactos adoptados; no se mantienen
> aquí como una segunda fuente.
>
> Para proyectar con las notas en la segunda pantalla, descomentar en `presentacion.tex`:
> `\setbeameroption{show notes on second screen=right}`.

## Estructura

La defensa tiene **20 diapositivas narradas** y una diapositiva de reserva sin numerar, situada
después de «Gracias». El hilo es:

1. El mercado fija un listón competitivo alto.
2. El objetivo 1 pregunta si el sistema aprende a ordenar y se decide con Rank-IC.
3. El objetivo 2 pregunta cuánto importa materializar ese orden y se decide con Information Ratio.
4. La era 2025--2026 no decide nada y solo aporta evidencia direccional.

La aritmética completa del juego de suma cero permanece en el capítulo 7 de la memoria. En la
defensa se introduce al comienzo mediante SPIVA, antes de los objetivos, para que el tribunal sepa
desde el inicio por qué ordenar y batir al índice no son la misma promesa.

## Reparto de tiempo

| # | Diapositiva | s | Acumulado |
|---:|---|---:|---:|
| 1 | Portada | 20 | 0:20 |
| 2 | Batir al mercado es un listón excepcionalmente alto | 45 | 1:05 |
| 3 | Dos objetivos, dos métricas | 50 | 1:55 |
| 4 | Cinco especialistas alimentan un meta-agente | 55 | 2:50 |
| 5 | Cada decisión usa solo información disponible entonces | 55 | 3:45 |
| 6 | Las decisiones terminan antes de abrir la era reservada | 60 | 4:45 |
| 7 | Rank-IC mide el orden, no la rentabilidad | 50 | 5:35 |
| 8 | El meta aprende a concentrarse, pero no supera al mejor especialista | 65 | 6:40 |
| 9 | Dos variables explican el liderazgo de `risk` | 50 | 7:30 |
| 10 | La señal es positiva, pero cambia de intensidad entre eras | 60 | 8:30 |
| 11 | Siete contrastes respaldan la señal; uno limita la rentabilidad | 55 | 9:25 |
| 12 | La cadena mejora el orden dentro y no cobra fuera | 65 | 10:30 |
| 13 | Tres reglas concentran casi toda la sensibilidad de la cartera | 55 | 11:25 |
| 14 | La rejilla revela cuánto pesa la implementación | 55 | 12:20 |
| 15 | La ganadora opera menos y mejora dentro de selección | 70 | 13:30 |
| 16 | El resultado se concentra en pocos nombres | 65 | 14:35 |
| 17 | La optimización cambia el signo, pero aún no demuestra generalización | 70 | 15:45 |
| 18 | Los perfiles muestran sensibilidad, no una clasificación de estilos | 55 | 16:40 |
| 19 | Qué queda demostrado y qué no | 75 | 17:55 |
| 20 | Gracias | 20 | **18:15** |

La diapositiva de reserva «Cómo se construyen los perfiles» no se narra ni entra en el contador.

Si hace falta recortar, hacerlo en este orden:

1. Diapositiva 16: resumir la concentración en una frase.
2. Diapositiva 9: limitarse a las dos variables dominantes.
3. Diapositiva 4: presentar la arquitectura sin enumerar los cinco especialistas.
4. Diapositiva 18: explicar solo que `balanced` es meta puro y que seis cohortes no ordenan estilos.

No acelerar las diapositivas 2, 12, 14, 17 y 19: fijan el listón, la tensión central, la evidencia
del objetivo 2 y el alcance final.

## Turno de preguntas

| Si preguntan… | Volver a | Respuesta corta |
|---|---:|---|
| ¿Por qué es tan difícil batir al índice? | 2 | SPIVA aporta el correlato empírico; el capítulo 7 añade la aritmética de Sharpe: antes de costes la gestión activa agregada iguala al mercado y después de costes queda por debajo. Los porcentajes por horizonte no son monótonos, pero sí persistentemente altos. |
| ¿La cadena mejoró o se sobreajustó? | 12 | Mejoró el orden en selección y no logró cobrarlo fuera. Esa divergencia se reporta como señal de búsqueda y motiva estudiar la implementación sin reabrir el modelo. |
| ¿Por qué domina `risk`? | 9 | El liderazgo procede sobre todo de variables de microestructura y rango, no de imponer manualmente una prima clásica de baja volatilidad. |
| ¿Qué robustez falla? | 11 | Falla el contraste que penaliza la multiplicidad de configuraciones. Por eso no se reclama rentabilidad ajustada por búsqueda aunque sí haya evidencia favorable de ordenación. |
| ¿No es sobreajuste elegir una cartera de una rejilla grande? | 14 y 19 | La cifra de la ganadora es optimista. La evidencia más estable del objetivo 2 es la dispersión de resultados con la misma señal congelada; la reserva queda como comprobación preliminar. |
| ¿Por qué una rejilla cartesiana? | 14 | Porque las reglas de cartera interactúan. Optimizar cada eje por separado puede perder combinaciones cuyo valor solo aparece conjuntamente. |
| ¿De dónde salen los datos y qué sesgo queda? | 5 | La composición del S\&P 500 y las fechas de publicación son históricas. El sesgo residual es cobertura incompleta del proveedor en los años antiguos; está medido y disminuye hacia el final. |
| ¿Los costes son realistas? | 15 | La memoria muestra una escalera de costes y dos puntos de equilibrio. El margen existe en selección, pero la reserva parte prácticamente sin ventaja; además, la capacidad acota el tamaño ejecutable. |
| ¿Qué son los perfiles? | 18 o reserva | Son reordenaciones deterministas dentro de las acciones que el meta ya considera buenas. No reentrenan, no participaron en la selección y no deben leerse como un ranking de estilos. |
| ¿Por qué `balanced` no tiene pesos de agentes? | Reserva | Porque es exactamente el `meta_rank` aprendido. Asignarle pesos ficticios duplicaría o alteraría el meta; por eso aparece como 100 % meta. |
| ¿Esto se puede usar para invertir? | 19 | No como recomendación. La confirmación económica es corta, la búsqueda fue amplia y los perfiles son diagnósticos. La contribución defendible es metodológica. |

## Tres respuestas que conviene ensayar

**«Si `risk` supera al meta, ¿para qué cinco agentes?»**

La arquitectura multiagente no queda demostrada como necesaria. Sí queda demostrado que el meta
aprende sin que se le fije de antemano el especialista ganador. Que termine concentrándose es un
resultado del protocolo, no permiso para reescribir el diseño después de observarlo.

**«Que la cartera original falle en reserva, ¿invalida la señal?»**

No. El Rank-IC es idéntico cualquiera que sea la cartera porque se calcula antes de construirla. El
cambio económico al modificar las reglas es precisamente la evidencia del objetivo 2: señal e
implementación son capas distintas.

**«Si el Deflated Sharpe no pasa, ¿qué queda en pie?»**

Queda la evidencia predictiva evaluada con Rank-IC y los contrastes que no dependen de rentabilidad
de cartera. Lo que se descarta es una afirmación fuerte de rentabilidad ajustada por búsqueda. El
cierre de la diapositiva 19 separa explícitamente ambas conclusiones.
