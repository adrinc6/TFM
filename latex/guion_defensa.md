# Guion de defensa

> El guion hablado vive en los `\note{}` de `TFM_ppt.tex`, junto a la diapositiva que
> acompaña. Este documento conserva el reparto de tiempo, el orden de recorte y el mapa de
> preguntas. Las cifras económicas se leen de los activos y artefactos adoptados; no se mantienen
> aquí como una segunda fuente.
>
> Para proyectar con las notas en la segunda pantalla, descomentar en `TFM_ppt.tex`:
> `\setbeameroption{show notes on second screen=right}`.

## Estructura

La defensa tiene **20 diapositivas narradas**, **cinco de reserva** sin numerar y se organiza en
tres actos con divisor propio. El pie muestra el acto en curso y una barra de progreso, de modo que
el tribunal sabe en todo momento en qué mitad del argumento está.

| Acto | Diapositivas | Qué establece |
|---|---|---|
| **El problema** | 1–4 | El listón es de suma cero, los objetivos son dos y la respuesta se adelanta |
| **Acto I · ¿Aprende?** | 5–11 | La señal existe, se mide con Rank-IC y resiste siete de ocho contrastes |
| **Acto II · ¿Se cobra?** | 12–19 | La cartera decide el signo del resultado; empatar no es batir |

El hilo es:

1. Batir al mercado es aritméticamente un juego de suma cero, y SPIVA mide cuánta gente pierde.
2. El objetivo 1 pregunta si el sistema aprende a ordenar y se decide con Rank-IC, contra un listón
   absoluto.
3. El objetivo 2 pregunta si ese orden se cobra frente al índice y se decide con Information Ratio,
   contra un listón relativo.
4. La era 2025–2026 no decide nada y solo aporta evidencia direccional.

**La diapositiva 4 adelanta el resultado.** Es deliberado: el tribunal escucha el resto sabiendo
adónde va, y los tres límites que acompañan a las tres cifras se repiten después una por una.

## Reparto de tiempo

| # | Acto | Diapositiva | s | Acumulado |
|---:|---|---|---:|---:|
| 1 | Problema | Portada | 20 | 0:20 |
| 2 | Problema | Batir al mercado es un juego de suma cero | 55 | 1:15 |
| 3 | Problema | Dos objetivos, dos listones, dos métricas | 45 | 2:00 |
| 4 | Problema | **La respuesta, en tres cifras** | 70 | 3:10 |
| 5 | I | *Divisor* · ¿Aprende el sistema a ordenar acciones? | 12 | 3:22 |
| 6 | I | Cinco especialistas, un meta-agente y dos relojes | 65 | 4:27 |
| 7 | I | Las decisiones terminan antes de abrir la era reservada | 60 | 5:27 |
| 8 | I | El meta aprende a concentrarse, y aun así no gana al mejor especialista | 80 | 6:47 |
| 9 | I | Dos variables explican el liderazgo de `risk` | 50 | 7:37 |
| 10 | I | La señal es positiva, pero cambia de intensidad entre eras | 60 | 8:37 |
| 11 | I | Siete contrastes respaldan la señal; uno limita la rentabilidad | 60 | 9:37 |
| 12 | II | *Divisor* · ¿Puede cobrarse ese orden frente al mercado? | 12 | 9:49 |
| 13 | II | **El orden mejora mientras el pago empeora** | 75 | 11:04 |
| 14 | II | 1.440 carteras sobre la misma señal congelada | 60 | 12:04 |
| 15 | II | Tres reglas concentran casi toda la sensibilidad | 50 | 12:54 |
| 16 | II | La ganadora opera menos, y por eso gana | 70 | 14:04 |
| 17 | II | **La era reservada: la cartera cambia el signo del resultado** | 85 | 15:29 |
| 18 | II | El resultado se concentra en muy pocos nombres | 60 | 16:29 |
| 19 | Cierre | Qué queda demostrado y qué no | 86 | 17:55 |
| 20 | — | Gracias | 20 | **18:15** |

Las cinco diapositivas de reserva no se narran ni entran en el contador.

Si hace falta recortar, hacerlo en este orden:

1. Diapositiva 18: resumir la concentración en una frase y no comentar los tres paneles.
2. Diapositiva 9: limitarse a las dos variables dominantes, sin el matiz de la reasignación anual.
3. Diapositiva 6: presentar la arquitectura sin enumerar los cinco especialistas.
4. Diapositiva 15: decir solo cuáles mueven y cuáles no, sin las cifras.

**No acelerar las diapositivas 2, 4, 13, 17 y 19**: fijan el listón, adelantan el resultado,
sostienen la tensión central y delimitan el alcance final.

## Construcción progresiva

Cinco diapositivas usan `\pause` y hay que contarlas al ensayar, porque cambian el ritmo:

- **2** — el 92,89 % aparece después del argumento aritmético, no a la vez.
- **6** — las cuatro garantías temporales se descubren una a una.
- **8** — el Rank-IC del promedio ingenuo aparece al final, cuando ya se ha comparado meta con `risk`.
- **11** — el Deflated Sharpe aparece separado de los siete contrastes que sí se superan.
- **17** — primero la cartera original en rojo, después la optimizada en verde.

## Turno de preguntas

| Si preguntan… | Volver a | Respuesta corta |
|---|---:|---|
| ¿Por qué es tan difícil batir al índice? | 2 | La aritmética de Sharpe: antes de costes la gestión activa agregada iguala al mercado, y después queda por debajo. SPIVA aporta el correlato empírico; los porcentajes por horizonte no son monótonos, pero sí persistentemente altos. |
| ¿La cadena mejoró o se sobreajustó? | 13 y reserva 5 | Mejoró el orden en selección y no logró cobrarlo fuera. Esa divergencia se reporta como señal de búsqueda y motiva estudiar la implementación sin reabrir el modelo. |
| ¿Por qué domina `risk`? | 9 | El liderazgo procede de variables de microestructura y rango, no de imponer una prima clásica de baja volatilidad. Es descriptivo: el mecanismo económico queda sin explicar y así se declara. |
| ¿Qué robustez falla? | 11 | El contraste que penaliza la multiplicidad de configuraciones. Por eso no se reclama rentabilidad ajustada por búsqueda aunque sí haya evidencia favorable de ordenación. |
| ¿No es sobreajuste elegir una cartera de una rejilla grande? | 14 y 19 | La cifra de la ganadora es una cota superior optimista. La evidencia estable del objetivo 2 es la dispersión con la misma señal congelada; la reserva es una comprobación preliminar. |
| ¿Por qué una rejilla cartesiana? | 14 | Porque las seis reglas interactúan: el suelo de cobertura significa cosas opuestas según el tope de efectivo. Optimizar cada eje por separado no habría encontrado la combinación final. |
| ¿De dónde salen los datos y qué sesgo queda? | 7 y reserva 3 | La composición del índice y las fechas de publicación son históricas. El sesgo residual es la cobertura incompleta del proveedor en los años antiguos: medido en causa y dirección, decreciente, y no cuantificado en puntos de rentabilidad. |
| ¿Los costes son realistas? | reserva 4 | El margen se mide, no se afirma: el exceso aguanta hasta unos diez veces el coste adoptado en selección. En la era reservada casi no hay margen, y la capacidad acota el patrimonio ejecutable. |
| ¿Qué son los perfiles? | reserva 1 y 2 | Reordenaciones deterministas dentro de las acciones que el meta ya considera buenas. No reentrenan, no participaron en la selección y no son un ranking de estilos. |
| ¿Por qué `balanced` no tiene pesos de agentes? | reserva 1 | Porque es exactamente el `meta_rank` aprendido. Asignarle pesos ficticios duplicaría o alteraría el meta. |
| ¿Esto se puede usar para invertir? | 19 | No como recomendación. La confirmación económica es corta, la búsqueda fue amplia y los perfiles son diagnósticos. La contribución defendible es metodológica. |

## Diapositivas de reserva

| # | Título | Para qué pregunta |
|---:|---|---|
| R1 | Cómo se construyen los perfiles | Qué es un perfil y por qué `balanced` no tiene pesos |
| R2 | Ningún estilo mejora a no imponer estilo | Si algún estilo habría funcionado mejor |
| R3 | El sesgo de cobertura, medido | Calidad de los datos y sesgo de supervivencia |
| R4 | Costes y capacidad | Si los costes son realistas y a qué tamaño aplica |
| R5 | Por qué tres pasadas encadenadas | Si encadenar estudios es sobreajuste |

## Tres respuestas que conviene ensayar

**«Si `risk` supera al meta, ¿para qué cinco agentes?»**

La arquitectura multiagente no queda demostrada como necesaria, y el trabajo lo dice. Sí queda
demostrado que el meta aprende sin que se le fije de antemano el especialista ganador: parte de
pesos iguales, se equivoca en 2016 concentrándose en el peor agente del periodo, y se corrige solo.
Que termine concentrándose es un resultado del protocolo, no permiso para reescribir el diseño
después de observarlo.

**«Que la cartera original falle en reserva, ¿invalida la señal?»**

No. El Rank-IC es idéntico con cualquier cartera, porque se calcula antes de construirla. El cambio
económico al modificar solo las reglas es precisamente la evidencia del objetivo 2: señal e
implementación son capas distintas, y confundirlas habría llevado a concluir que la señal no se
traslada fuera de muestra.

**«Entonces, ¿bate al mercado o no?»**

No está demostrado. Lo que está demostrado es que con la cartera por defecto perdía y con la
optimizada empata, sin reentrenar nada. En un juego de suma cero eso es un resultado sobre dónde
estaba el cuello de botella —la implementación, no el modelo—, no una ventaja competitiva. Con seis
cohortes cerradas y año y medio de cartera, afirmar más sería exactamente el tipo de lectura que
este trabajo se ha propuesto evitar.
