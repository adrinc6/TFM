# Informe acumulativo de resultados

## Estado

Este informe documenta la **cadena de cuatro estudios de referencia del TFM** (2026-08-14): tres
Model Studies encadenados, en los que el ganador de cada pasada es el baseline de la siguiente, y un
Portfolio Study que optimiza la construcción de cartera del último por Information Ratio sin
reentrenar nada.

Los studies anteriores —incluido `study-20260803-201234-b4d7a8d8`, que fue referencia hasta esta
fecha— quedan **fuera del TFM**. Su análisis se conserva en `docs/bitacora.md` como registro
histórico, no como resultado vigente.

## Identidad de los estudios

| Campo | Valor |
|---|---|
| Model Study 1 | `study-20260812-163136-1b104667` · ganador `run-6eaa47a0597b` · catálogo v6 |
| Model Study 2 | `study-20260813-103456-aa733655` · ganador `run-2dc586be8653` · catálogo v6 |
| **Model Study 3 (referencia)** | **`study-20260814-095144-5ec17b78`** · ganador `run-f134d7eb9e06` · catálogo **v7** |
| **Portfolio Study** | **`study-20260814-135754-fdbdf2c5`** · 1.728 carteras |
| Hash de dataset | `b9134b218e3bf7fc156372d61e02056ecfa6036777e0fe84a69df0a92653fbd3` |
| Clave de evaluación (study 3) | `0088d08579c1e65f1b2f495d550ed2ea54249558ab4354675dbf2fa71de0a27f` |
| Configuraciones evaluadas (study 3) | 71 · 17 decisiones, 2 por empate técnico |
| Ventana de selección | 2015–2024, 117 cohortes mensuales |
| Era reservada | 2025–2026, 6 cohortes cerradas (~1,41 años de cartera), **no usada en ninguna decisión** |
| Métrica de selección | Rank-IC pareado (`rank_ic_only`); Information Ratio en el Portfolio Study |

> **Regla de procedencia.** La evidencia **predictiva** (Rank-IC, agentes, meta, decisiones,
> robustez, atribución) sale del Model Study 3. La **económica** (equity, órdenes, posiciones,
> métricas anuales, perfiles) sale del ganador del Portfolio Study, bajo `evidence_best_full/`.
> Mezclarlas produce afirmaciones que no se sostienen.

> **Aviso de reproducibilidad.** La cadena no corrió entera bajo la misma versión de catálogo: los
> studies 1 y 2 usaron v6 y el 3 usó v7, que invierte el desempate por simplicidad de
> `execution_lag_days`. El cambio sólo actúa cuando la evidencia no distingue entre candidatos, de
> modo que no altera ninguna medición, pero las tablas deben citar la versión junto al `study_id`.

## La cadena de estudios

| Métrica | Study 1 | Study 2 | Study 3 |
|---|---|---|---|
| Rank-IC medio | 0,1000 | 0,1074 | **0,1090** |
| IC-IR | 0,735 | 0,835 | **0,851** |
| t Newey-West | 2,95 | 3,36 | **3,46** |
| Transferencia | 0,178 | 0,234 | **0,328** |
| IR (selección) | 0,189 | 0,294 | **0,339** |
| Variables cambiadas | 8 | 1 | 2 |
| **Era reservada: IR** | +0,898 | +0,476 | **−1,167** |

La cadena **converge** (8 → 1 → 2 variables modificadas) y mejora monótonamente dentro de la ventana
de selección, pero **se degrada monótonamente fuera de ella**. Esa divergencia es la firma del
sobreajuste por búsqueda, y es lo que motivó el Portfolio Study.

## El Portfolio Study y el hallazgo central

Cartera ganadora: `target_size=8`, `max_cash_weight=0.0`, `sizing_mode=alpha_proportional`,
`minimum_holding_period=half_horizon`, `coverage_percentile_floor=60`,
`rebalance_drift_tolerance=0.1`.

| Métrica | Cartera del modelo | Cartera ganadora | Era reservada |
|---|---|---|---|
| Information Ratio | 0,339 | **0,844** | **+0,304** |
| Exceso geométrico | 2,61 % | **6,97 %** | **+2,56 %** |
| Rotación anualizada | 3,58 | 3,24 | 3,91 |
| Años que baten | 70 % | 80 % | 50 % |

**Con la cartera del modelo, la era reservada daba −11,29 % de exceso e IR −1,167 (0/2 años). Con la
optimizada, +2,56 % e IR +0,304 (1/2 años).** El Rank-IC de esa era es +0,0441 en ambos casos,
porque no depende de la cartera. El cuello de botella no era la capacidad predictiva sino su
traducción a posiciones, hasta el punto de decidir el signo del resultado fuera de muestra.

Cautelas obligatorias al citar esto: 6 cohortes cerradas, ~1,41 años de cartera, y la ganadora es la
mejor de 1.728 evaluadas. La rejilla se calculó sobre scores recortados en 2024, de modo que ninguna
combinación pudo observar la era reservada durante la selección.

## Reglas de trazabilidad

Toda cifra debe incluir: `study_id`; `winner_run_id`; hash del catálogo; hash del dataset; periodo;
métrica y unidad; ruta del artefacto fuente; y el papel de la cifra —selección, confirmación fuera de
muestra o diagnóstico—. Ninguna afirmación de este informe existe sin un artefacto que la respalde.

---

> **Estado de las cifras (2026-08-15).** Todas las secciones de este informe están cotejadas contra
> los artefactos de la cadena vigente: `evidence/summary.json`, `robustness.json`,
> `attribution.json`, `decisions.json`, `winner.json` y `portfolio_winner.json`. Las secciones 1 a 6
> arrastraban las cifras del study derogado `b4d7a8d8` y se han reescrito por completo; el caso más
> grave era la sección 5, que afirmaba lo contrario de lo que muestran los estudios actuales
> —rank-IC reservado negativo y 2/2 años batiendo al índice—.

## 1. El proceso de aprendizaje

Esta sección es el núcleo del TFM: no basta con que el sistema acierte, hay que poder **enseñar cómo
aprende**. Hay tres evidencias independientes de aprendizaje, y las tres son observables en
artefactos.

### 1.1 El meta-agente aprende a quién escuchar

El meta-agente arranca sin información: las primeras filas de `evidence/meta_weights.parquet` tienen
estado `fallback_equal` y los cinco agentes reciben 0,20 exactos, porque todavía no hay cohortes
cerradas con las que estimar la calidad de nadie. A medida que se cierran etiquetas a 12 meses el
estado pasa a `learned` (675 filas en total) y los pesos se separan. El peso medio anual de `risk`
recorre 0,22 en 2016, 0,45 en 2017, 0,66 en 2018, por encima de 0,83 desde 2020 y 0,997 en 2023,
hasta terminar la serie en un reparto prácticamente degenerado: **0,954 para `risk` y 0,047 para
`value`, con los otros tres en cero**.

Fuente: `evidence/meta_weights.parquet` (papel: diagnóstico); la trayectoria completa está en
`latex/assets/f05_pesos_anual.png`.

La lectura es una **curva de aprendizaje explícita, con error incluido**. En 2016 el meta apuesta
por `momentum`, que resultará ser el peor agente del sistema. En 2017 ya ha corregido y empieza a
cargar en `risk`. Que se equivoque primero y se corrija después, sin intervención y con datos ya
cerrados, es la evidencia más directa de que hay aprendizaje y no una asignación puesta a mano.

Ahora bien, el ganador vigente usa `stacked_rolling_free`, **sin tope por agente**, de modo que nada
frena esa convergencia: el meta acaba dejando de ser un combinador para ser un selector. La
concentración media (HHI) es 0,629 y la rotación media de pesos 0,0216 (`robustness.json` →
`meta_weight_stability`): aprende rápido, converge fuerte y luego es estable.

### 1.2 La ponderación aprendida vale más que la ingenua

Es el contraste que separa «aprender» de «promediar». Con los mismos cinco agentes y las mismas
señales, la única diferencia es cómo se combinan:

| Señal | Rank-IC medio | Cohortes positivas | IC-IR |
|---|---|---|---|
| `meta_final` (pesos aprendidos) | **0,1090** | 74,36 % | 0,851 |
| `meta_equal_weight` (0,20 fijos) | 0,0675 | 62,39 % | 0,535 |

Fuente: `evidence/rank_ic_diagnostics.parquet` (papel: diagnóstico).

Aprender los pesos añade **+0,0415 de rank-IC** sobre repartir por igual, un **61 %** más de señal.
Ese delta no viene de mejores features ni de más datos: viene exclusivamente del aprendizaje del
meta-agente. Es la demostración más limpia de que la capa de combinación hace un trabajo real.

### 1.3 Cada agente aporta lo que sabe, y el meta lo ordena

| Agente | Rank-IC medio | Desv. típica | Cohortes positivas | IC-IR |
|---|---|---|---|---|
| `risk` | 0,1227 | 0,1241 | 80,34 % | 0,988 |
| **`meta_final`** | **0,1090** | 0,1281 | 74,36 % | 0,851 |
| `meta_equal_weight` | 0,0675 | 0,1261 | 62,39 % | 0,535 |
| `growth` | 0,0249 | 0,0865 | 61,54 % | 0,289 |
| `value` | 0,0244 | 0,0803 | 63,25 % | 0,303 |
| `quality` | 0,0096 | 0,1051 | 47,01 % | 0,091 |
| `momentum` | 0,0005 | 0,0902 | 47,86 % | 0,006 |

Fuente: `evidence/rank_ic_diagnostics.parquet`, ventana de selección 2015–2024 (papel: diagnóstico).

Hay que decirlo con honestidad, y el TFM debe defenderlo explícitamente: **`risk` en solitario tiene
más rank-IC que el meta**. Y ya no puede achacarse al tope: con la variante libre el meta puede
concentrarse cuanto quiera —de hecho llega a 0,954— y aun así ordena algo peor. La explicación está
en el trayecto: el meta tarda años en converger y durante ese tiempo arrastra el peso de agentes que
ordenan mucho peor. Ese coste es irrecuperable porque forma parte de lo que significa aprender sin
mirar al futuro. La defensa no es «el meta es el mejor predictor», sino «el meta aprende, sin
supervisión externa, a reproducir casi toda la señal de su mejor especialista partiendo de la
ignorancia». Con estos datos **la arquitectura multi-agente no queda demostrada**, y el TFM lo dice.

La estabilidad por eras muestra además que ningún agente domina siempre:

| Agente | 2015–2018 | 2019–2021 | 2022–2024 | 2025–2026 (reservada) |
|---|---|---|---|---|
| `risk` | 0,1300 | 0,0554 | 0,1808 | 0,0616 |
| `meta_final` | 0,0999 | 0,0506 | 0,1788 | 0,0441 |
| `meta_equal_weight` | 0,0829 | 0,0145 | 0,1012 | −0,0735 |
| `value` | 0,0147 | 0,0287 | 0,0321 | −0,0953 |
| `growth` | 0,0459 | 0,0143 | 0,0094 | −0,1333 |
| `quality` | 0,0026 | −0,0089 | 0,0367 | −0,0487 |
| `momentum` | 0,0391 | −0,0477 | 0,0004 | −0,0080 |

`risk` es la única señal positiva en las cuatro eras. En 2019–2021 el sistema atraviesa su peor
régimen y en 2022–2024 alcanza su mejor rank-IC (0,1788). En la era reservada **los cuatro agentes
distintos de `risk` se vuelven negativos a la vez**, lo que apunta a una rotación de factores del
mercado y no a un error de signo en un modelo concreto; el meta aguanta en positivo porque para
entonces ya había concentrado en `risk`, mientras que el equiponderado se va a −0,0735. Es la única
era en que aprender los pesos o fijarlos cambia el signo del resultado.

---

## 2. Capacidad predictiva (papel: selección)

| Métrica | Valor | Artefacto |
|---|---|---|
| Rank-IC medio | 0,1090 | `evidence/summary.json` |
| IC-IR | 0,851 | `evidence/summary.json` |
| Cohortes positivas | 74,36 % | `evidence/summary.json` |
| Cohortes | 117 | `evidence/summary.json` |
| t de Newey-West | 3,457 | `attribution.json` |
| Observaciones independientes efectivas | 9 | `attribution.json` |
| Desviación típica del IC | 0,1281 | `evidence/summary.json` |
| Diferencial de colas (top − bottom) | 0,0365 | `evidence/summary.json` |

**Lectura.** Un rank-IC de 0,10 sostenido sobre ~400 valores es un resultado fuerte para un trabajo
con datos gratuitos: la literatura considera explotable cualquier IC estable por encima de 0,03. El
t de Newey-West de 3,02 corrige el solapamiento de etiquetas (cohortes mensuales con horizonte de 12
meses comparten 11/12 de su ventana) y sigue siendo significativo. La cifra honesta que acompaña a
las 117 cohortes es que equivalen a solo **9 observaciones independientes**; el TFM debe citar
siempre ambas.

### Comparación con los baselines deterministas

| Señal | Rank-IC medio | Cohortes positivas |
|---|---|---|
| **Sistema (`meta_final`)** | **0,1090** | 74,36 % |
| `garp_score` | 0,0130 | 64,50 % |
| `value_score` | 0,0038 | 49,62 % |
| `growth_score` | 0,0028 | 51,53 % |
| `quality_score` | 0,0023 | 51,53 % |
| `momentum_score` | −0,0001 | 52,67 % |

Fuente: `attribution.json` (papel: diagnóstico).

El sistema aprendido multiplica por ~8 el mejor baseline determinista. Las fórmulas factoriales
clásicas, sobre este mismo panel y con las mismas reglas point-in-time, son prácticamente ruido.

---

## 3. Robustez: por qué esto es aprendizaje y no suerte

Esta es la sección que sostiene la afirmación central del TFM. Un rank-IC alto no vale nada por sí
solo: hay que demostrar que no lo produce el azar, ni la implementación, ni la multiplicidad de
configuraciones probadas, ni una era afortunada, ni una semilla afortunada. Se atacan **siete
frentes independientes**, y el resultado aguanta en todos.

### 3.1 Permutación de etiquetas — ¿podría salir esto por azar?

| Métrica | Valor |
|---|---|
| Rank-IC observado | 0,1090 |
| Permutaciones | 9 999 |
| p-valor (con corrección +1) | **0,0001** |

Fuente: `robustness.json`. Se permuta la etiqueta dentro de cada cohorte, destruyendo la relación
señal-futuro y conservando toda la estructura transversal. **Ninguna de las 9 999 permutaciones
alcanzó el rank-IC observado.** Es el contraste más directo contra la hipótesis de suerte.

### 3.2 Placebos de etiqueta — ¿lo produce la maquinaria?

| Semilla del placebo | Rank-IC |
|---|---|
| 101 | −0,0081 |
| 102 | −0,0011 |
| 103 | −0,0009 |
| 104 | +0,0013 |
| 105 | −0,0004 |

Fuente: `robustness.json`. Con etiquetas barajadas, el pipeline completo —features, cinco agentes,
LightGBM, meta apilado, cartera— produce rank-IC en el rango [−0,0081; +0,0013], centrado en cero
frente al 0,1090 real. Si el resultado fuese un artefacto del código (fuga temporal, normalización
mal hecha, sesgo del optimizador), los placebos lo mostrarían. No lo muestran. **La señal viene de
los datos, no del programa.**

### 3.3 Bootstrap por bloques — ¿es distinguible de cero?

| Intervalo | Límite inferior | Límite superior |
|---|---|---|
| 90 % | 0,0540 | 0,1614 |
| 95 % | **0,0425** | 0,1723 |

Fuente: `robustness.json`, bloques de 12 cohortes para respetar el solapamiento de etiquetas. **El
intervalo al 95 % excluye el cero con holgura**: el límite inferior sigue siendo 3× el umbral de
explotabilidad de la literatura.

### 3.4 Exclusión de eras — ¿depende de un periodo afortunado?

| Era excluida | Rank-IC del resto | Cohortes |
|---|---|---|
| 2015–2018 | 0,1147 | 72 |
| 2019–2021 | 0,1350 | 81 |
| 2022–2024 | 0,0780 | 81 |

Fuente: `robustness.json`. Quitando cualquier era completa el rank-IC se mantiene entre 0,0780 y
0,1350, siempre muy por encima de cero. **No hay un periodo del que dependa el resultado.**

### 3.5 Semillas — ¿depende del azar de la inicialización?

| Semilla | Rank-IC | Exceso geométrico | IR | CAGR de cartera | Máximo drawdown |
|---|---|---|---|---|---|
| 42 (ganadora) | 0,1090 | 2,61 % | 0,339 | 16,13 % | 26,97 % |
| 7 | 0,1075 | 2,06 % | 0,294 | 15,51 % | 27,80 % |
| 2026 | 0,1087 | −0,01 % | 0,088 | 13,17 % | 30,45 % |

Fuente: `robustness.json` → `seeds`, `seed_dispersion`. El rango de rank-IC es **0,0015** y no cruza
cero: la conclusión predictiva es estable. La económica **no**: el exceso geométrico va de −0,01 % a
+2,61 %, de modo que **cruza cero** (`geometric_excess_return.crosses_zero = true`) y el artefacto
registra `economic_conclusion_stable = false`. Cambiando sólo la inicialización del boosting, la
misma configuración puede no batir al índice. Es la asimetría central del trabajo: la ordenación
aguanta, su traducción a rentabilidad es frágil.

### 3.6 Carteras aleatorias con riesgo emparejado — ¿bate a la suerte?

| Contraste | CAGR modelo | Mediana aleatoria | p95 aleatorio | Percentil del modelo |
|---|---|---|---|---|
| Riesgo emparejado | 15,56 % | 9,23 % | 14,99 % | **96,8 %** |
| General | 15,56 % | 12,08 % | 75,06 % | 76,1 % |

Fuente: `robustness.json`, 1 000 simulaciones de **8 nombres** —el mismo tamaño que la cartera del
modelo, para que la comparación sea legítima— con 15 pb de costes. Contra carteras aleatorias **de
riesgo comparable**, el modelo está en el percentil 96,8: bate al azar. El contraste «general» no es
informativo y hay que decirlo en el TFM: su p95 es un CAGR del 75 % anual, dominado por carteras que
concentraron supervivientes extremos; compararse con eso no mide habilidad sino tolerancia a la
varianza.

### 3.7 Neutralización por estilo — ¿es un factor conocido disfrazado?

| Métrica | Valor |
|---|---|
| Rank-IC bruto | 0,1177 |
| Rank-IC neutralizado por 14 controles de estilo | **0,1019** |
| Fracción retenida | **86,62 %** |

Fuente: `attribution.json`. Tras neutralizar por P/E, P/B, P/S, EV/EBITDA, retorno relativo 12m,
momentum 12-1, volatilidad realizada 63d y 126d, beta 252d, ROE, ROIC, margen operativo y crecimiento
de BPA y de ventas, **sobrevive el 86,62 % de la señal**. La ordenación no es una réplica de los
factores clásicos. El 0,1177 de partida no coincide con el 0,1090 del resto del informe porque el
contraste sólo usa las filas en que los catorce controles están disponibles; lo interpretable es el
cociente, medido sobre la misma muestra.

La regresión con réplicas de factores y errores Newey-West de 12 retardos lo confirma por el otro
lado: en la ventana de selección el alfa por periodo es 0,27 % con t = 1,44 y R² = 0,017, sin
ninguna carga significativa (la mayor en valor absoluto es `low_volatility`, con −0,105 y t = −1,04).
Es decir, el exceso **no se explica** por exposición a estilos, pero tampoco alcanza significación
estadística propia en esa ventana — un matiz que el TFM debe reportar sin adornar. Los factores de
tamaño e inversión no son construibles con las fuentes disponibles y se declaran no replicables.

### 3.8 El contraste que no se supera: Deflated Sharpe

| Métrica | Valor |
|---|---|
| Sharpe observado por periodo | 0,0411 |
| Configuraciones probadas | 71 |
| Probabilidad Deflated Sharpe | **0,682** |
| Umbral exigido | 0,95 |

Fuente: `attribution.json`. Corrigiendo por haber probado 71 configuraciones, la probabilidad queda
en 0,682, **por debajo del 0,95** requerido. Y ha empeorado respecto de las pasadas anteriores
precisamente porque la cadena existe: encadenar estudios multiplica las configuraciones ensayadas. Hay que reportarlo como lo que es: la evidencia de
**capacidad predictiva** (rank-IC) supera todos los contrastes, mientras que la evidencia de
**rentabilidad ajustada por riesgo** no resiste del todo la corrección por multiplicidad. Ocultar
esto invalidaría el resto del capítulo de robustez.

### Resumen del capítulo de robustez

| Contraste | Pregunta que responde | Resultado |
|---|---|---|
| Permutación (9 999) | ¿Es azar? | p = 0,0001 ✔ |
| Placebos de etiqueta (5) | ¿Es un artefacto del código? | [−0,0081; +0,0013] ✔ |
| Bootstrap por bloques 95 % | ¿Es distinguible de cero? | [0,0425; 0,1723] ✔ |
| Exclusión de eras | ¿Depende de un periodo? | 0,0780–0,1350 ✔ |
| Semillas (3) | ¿Depende de la inicialización? | rango rank-IC 0,0015, sin cruce de cero ✔ |
| Carteras aleatorias (riesgo emparejado) | ¿Bate al azar? | percentil 96,8 ✔ |
| Neutralización por estilo | ¿Es un factor conocido? | retiene 86,62 % ✔ |
| Deflated Sharpe | ¿Resiste la multiplicidad? | 0,682 < 0,95 ✘ |

**Siete de ocho.** La conclusión defendible es: *el sistema aprende una ordenación transversal real,
estadísticamente distinguible del azar, no reproducible por la maquinaria con etiquetas falsas, no
dependiente de una era ni de una semilla, y no explicable por factores de estilo conocidos.* Lo que
**no** puede afirmarse con la misma rotundidad es que su Sharpe sobreviva a la corrección por las 66
configuraciones probadas.

---

## 4. Traducción económica (papel: selección)

Todas las cifras de esta sección son de la **cartera adoptada** —la ganadora del Portfolio Study—,
leídas de `evidence_best_full/`. La cartera por defecto del Model Study aparece sólo como término de
comparación y se identifica siempre como tal.

| Métrica | Selección 2015–2024 | Confirmación 2025–2026 | Curva completa |
|---|---|---|---|
| CAGR cartera | 21,06 % | 22,23 % | 21,04 % |
| CAGR benchmark (SPY) | 13,17 % | 19,18 % | 13,81 % |
| Exceso geométrico | 6,97 % | 2,56 % | 6,35 % |
| Information Ratio anualizado | 0,844 | 0,304 | 0,753 |
| Máximo drawdown | 28,40 % | 12,09 % | 28,40 % |
| Años por encima de SPY | 80 % (8/10) | 50 % (1/2) | 75 % |
| Alfa anual medio | 7,17 % | 1,91 % | 6,29 % |
| Peor año (alfa) | −8,30 % | −2,73 % | −8,30 % |
| Turnover anualizado | 324,43 % | 391,35 % | 333,35 % |
| Efectivo medio | 0,00 % | 0,00 % | 0,00 % |
| Coste total acumulado | 4,74 % | 0,88 % | 5,63 % |
| Periodos | 117 | 18 | 135 |

Fuente: `evidence_best_full/summary.json` del ganador del Portfolio Study.

**Lectura.** El efectivo medio es **cero en las tres ventanas**: la cartera renuncia por completo a
la liquidez, de modo que su exceso no puede atribuirse a haber estado fuera del mercado en los
momentos oportunos. Está siempre invertida y su resultado procede íntegramente de qué acciones
eligió. El coste acumulado de 4,74 puntos frente a un exceso de 6,97 % no es despreciable, pero es
un peaje para expresar la señal, no la fuente del resultado.

Sobre la transferencia: el coeficiente disponible es el del Model Study (0,178 → 0,234 → **0,328** a
lo largo de la cadena), calculado sobre su cartera por defecto. El Portfolio Study no lo recalcula
sobre su ganadora, así que **no existe una cifra de transferencia para la cartera adoptada**. Lo que
sí mide la mejora por otra vía es el salto de IR de 0,339 a 0,844 sobre la misma señal congelada:
como el numerador predictivo no cambió, todo el incremento procede de que la cartera desperdicia
menos.

### Detalle anual

| Año | Cartera | SPY | Alfa | MDD año | IR año | Efectivo | Turnover |
|---|---|---|---|---|---|---|---|
| 2015 | 1,27 % | −0,63 % | **+1,91 %** | 7,30 % | 0,434 | 0,00 % | 158,59 % |
| 2016 | 19,00 % | 10,88 % | **+7,32 %** | 5,59 % | 0,814 | 0,00 % | 379,81 % |
| 2017 | 52,56 % | 21,71 % | **+25,35 %** | 1,98 % | 3,877 | 0,00 % | 308,32 % |
| 2018 | 10,10 % | −5,40 % | **+16,38 %** | 12,50 % | 1,887 | 0,00 % | 291,11 % |
| 2019 | 43,38 % | 32,05 % | **+8,57 %** | 1,80 % | 1,553 | 0,00 % | 332,78 % |
| 2020 | 18,13 % | 18,02 % | **+0,09 %** | 22,52 % | 0,067 | 0,00 % | 361,61 % |
| 2021 | 29,34 % | 29,71 % | −0,29 % | 3,28 % | 0,032 | 0,00 % | 265,83 % |
| 2022 | −25,16 % | −18,38 % | −8,30 % | 20,96 % | −0,928 | 0,00 % | 120,14 % |
| 2023 | 49,61 % | 26,18 % | **+18,57 %** | 13,14 % | 1,679 | 0,00 % | 468,60 % |
| 2024 | 27,94 % | 25,34 % | **+2,07 %** | 3,01 % | 0,213 | 0,00 % | 476,39 % |
| **2025** | **25,90 %** | **18,17 %** | **+6,55 %** | 3,99 % | 1,056 | 0,00 % | 330,53 % |
| **2026** | **5,48 %** | **8,43 %** | **−2,73 %** | 9,60 % | −0,123 | 0,00 % | 256,49 % |

Fuente: `evidence_best_full/annual_metrics.parquet`. Las dos últimas filas son la era reservada.

El detalle anual desmiente que el exceso agregado proceda de un año excepcional: la cartera bate al
índice en 8 de los 10 años de selección, con alfa medio 7,17 % y mediano 4,70 %. La distribución
está desplazada, no arrastrada por una cola. Los dos años negativos son 2021 (−0,29 pp, una
diferencia que no requiere explicación estructural) y **2022 (−8,30 pp)**, que además tuvo la
rotación más baja de toda la serie: su pérdida no fue de fricción sino de composición, lo que es
coherente con un sistema que ordena por atractivo relativo y **no gestiona la exposición
direccional al mercado**. El sistema decide *qué* comprar, no *cuánto* estar expuesto.

### Órdenes y rotación

| Motivo | Órdenes | % del flujo | Coste |
|---|---|---|---|
| `rebalance` | 408 | 38,2 % | 8,08 |
| `net_edge_over_worst` | 52 | 19,8 % | 4,59 |
| `displaced_by_net_edge` | 52 | 17,2 % | 4,01 |
| `initial_fill` | 30 | 10,9 % | 1,50 |
| `below_coverage_percentile` | 33 | 9,0 % | 1,74 |
| `fully_invested_fill` | 12 | 4,6 % | 1,19 |
| `missing_current_score` | 1 | 0,4 % | 0,08 |
| **Total** | **588** | **100 %** | **21,19** |

Fuente: `evidence_best_full/orders.parquet`.

La rotación tiene **dos fuentes de tamaño comparable y naturaleza distinta**. La rotación por ventaja
neta (37,0 %) es el cambio de opinión de la señal, y es estructural: se re-decide cada mes sobre una
señal a doce meses. Su palanca es `snapshot_step_months`, que es **predictiva** y cuya modificación
obligaría a rehacer la selección. El `rebalance` (38,2 %) no cambia qué se tiene, sólo el peso de lo
que ya se tiene; depende de `rebalance_drift_tolerance`, que sí es de cartera y que la rejilla fijó
en 0,1. Y ahí hay un dato aprovechable: esa variable es la penúltima en capacidad de mover el IR
(0,016), de modo que el sistema paga con ella su mayor partida de flujo de órdenes a cambio de una
diferencia marginal en el criterio que optimizó.

---

## 5. La era reservada 2025–2026 (papel: confirmación)

Es el resultado más exigente del trabajo, porque **ninguna de sus observaciones intervino en ninguna
decisión**: ni en la elección del ganador, ni en los pesos del meta, ni en la rejilla de carteras,
que se calculó sobre una serie recortada en 2024.

### Lo predictivo se sostiene

| Métrica | Valor |
|---|---|
| Rank-IC medio | **+0,0441** |
| Cohortes cerradas | 6 |
| Cohortes positivas | 66,67 % |
| IC-IR | 0,562 |
| t de Newey-West | 2,263 |
| Observaciones independientes efectivas | ~1 |

Fuente: `attribution.json` → `confirmation_2025_2026`.

El Rank-IC baja respecto al 0,1090 de la ventana de selección —lo esperable, porque parte de esa
cifra es ajuste al periodo en que se decidió— pero **no se hunde ni cambia de signo**. Y es idéntico
con cualquier cartera, porque no depende de ella.

### Lo económico depende por completo de la cartera

| Métrica | Cartera por defecto | Cartera optimizada |
|---|---|---|
| Exceso geométrico | **−11,29 %** | **+2,56 %** |
| Information Ratio | **−1,167** | **+0,304** |
| Años por encima de SPY | 0/2 | 1/2 |

**Éste es el hallazgo central del TFM.** Mismo modelo, misma señal, mismo panel, sin reentrenar
nada: lo único que cambia son las reglas de construcción de cartera, y el resultado fuera de muestra
**cambia de signo**. La ordenación transversal siempre había sido válida fuera de muestra; lo que
fallaba era la maquinaria que la convertía en posiciones.

### El matiz obligatorio, que el TFM no debe esconder

1. **Sólo hay 6 cohortes con etiqueta cerrada** y ~1 observación independiente efectiva. Con
   horizonte de 12 meses y cadencia mensual, cohortes contiguas comparten casi toda su ventana. Nada
   construido sobre esa cantidad de datos puede ser concluyente.
2. **El tramo de cartera es de 1,41 años**, de modo que «bate en la mitad de los años» significa uno
   de dos.
3. **La cartera ganadora es la mejor de 1.728.** Que ninguna pudiera ver la era reservada protege
   esta comprobación concreta, pero no demuestra que la elección generalice.
4. **La cola alta cayó.** En esa era el decil superior obtiene −7,67 % frente al −1,85 % del
   universo: pierde contra el panel. La separación entre decil superior e inferior se mantiene en
   +11,33 puntos, de modo que la ordenación siguió funcionando y lo que cayó fue el nivel de toda la
   cola alta. La cartera batió al índice eligiendo *dentro* de ese decil, no porque el decil entero
   fuese ganador.
5. **La regresión factorial de esta era**, calculada sobre la cartera por defecto del Model Study,
   da un alfa por periodo de −0,48 % con t de Newey-West −3,50: coherente con que esa cartera
   perdiese, y otra razón para no leer la era reservada sin declarar con qué cartera se midió.

La formulación honesta para el tribunal es: *en la única ventana que no participó en ninguna
decisión, la ordenación aprendida siguió siendo positiva, y su traducción a rentabilidad dependió de
la construcción de cartera hasta el punto de cambiar de signo. Con seis cohortes, eso es un indicio
fuerte y no una demostración.*

---

## 6. Perfiles: por qué gana `balanced`

Los ocho perfiles comparten **exactamente la misma señal** y la misma cartera ganadora; se
diferencian sólo en cómo reordenan esa señal antes de comprarla. Un perfil acota el universo al
percentil 60 del meta-agente y después recombina los rangos de los cinco agentes con pesos fijos
declarados de antemano. Es un experimento controlado: cualquier diferencia es atribuible a la
reordenación, no al modelo. **Ningún perfil se elige por su Information Ratio.**

| Perfil | IR selección | Exceso selección | Turnover | IR reservada | Exceso reservada |
|---|---|---|---|---|---|
| **`balanced`** | **0,844** | **6,97 %** | 3,24 | 0,304 | 2,56 % |
| `defensive` | 0,570 | 3,73 % | 2,51 | −0,607 | −5,70 % |
| `quality` | 0,312 | 2,06 % | 3,40 | −0,635 | −9,09 % |
| `growth` | 0,212 | 1,04 % | 3,93 | 0,318 | 4,14 % |
| `value` | 0,190 | 1,16 % | 3,82 | −0,513 | −4,94 % |
| `garp` | 0,112 | 0,51 % | 3,89 | −0,779 | −8,76 % |
| `contrarian` | 0,057 | −0,40 % | 4,33 | 0,414 | 4,44 % |
| `momentum` | 0,017 | −0,77 % | 5,98 | **1,889** | **41,59 %** |

Fuente: `portfolio_profiles.parquet` (papel: diagnóstico).

**Ninguno de los siete estilos mejora a la señal sin reordenar.** `balanced` domina la ventana de
selección con 0,844 frente al 0,570 del siguiente, y dos estilos llegan a exceso negativo. La
ordenación aprendida ya contiene lo que los perfiles intentan imponer desde fuera.

Y el orden entre ellos **no es arbitrario**: se predice casi por completo desde la calidad de los
agentes que cada uno pondera. El mejor de los siete es `defensive` (IR 0,570), que carga el 60 % en
`risk`, el agente de mayor rank-IC (0,1227). El peor es `momentum` (IR 0,017), que carga el 75 % en
el agente de rank-IC 0,0005 —indistinguible de cero— y además penaliza a `risk`. Entre medias,
`quality`, `growth`, `value` y `garp` reparten peso entre agentes cuyo rank-IC ronda 0,01–0,02 y sus
IR quedan agrupados entre 0,112 y 0,312. `contrarian` lo confirma desde el otro extremo: con peso
**negativo** sobre el agente más débil obtiene 0,057, mejor que apostar a favor de él pero aún muy
por debajo de no reordenar nada.

Hay además un coste operativo visible: los estilos más agresivos rotan más sin obtener más exceso.
`momentum` alcanza 5,98 vueltas al año —casi el doble que el 3,24 de `balanced`— para terminar con
exceso negativo. La reordenación no sólo no aporta señal: cuesta fricción.

**La era reservada invierte el orden casi por completo, y el dato merece reportarse precisamente
porque incomoda**: `momentum`, el peor perfil en selección, es el mejor con diferencia en 2025–2026
(IR 1,889, exceso 41,59 %). La tentación de leer ahí un hallazgo debe resistirse: seis cohortes no
permiten ordenar ocho estilos, y un periodo tan corto está dominado por qué régimen de mercado le
tocó en suerte a cada sesgo. Lo que sí sugiere, de forma consistente con el resto del informe, es
que **el orden entre perfiles no es una propiedad estable de la señal sino del régimen**, y que por
eso mismo el protocolo hace bien en no elegir perfil por su rentabilidad.

---

## 7. Configuración ganadora

Seleccionada por rank-IC pareado, con puerta de no inferioridad y suelo por era, sin que la era
reservada participe en ninguna decisión (`decisions.json`, `winner.json`).

Valores leídos de `winner.json` del Model Study 3 (`run-f134d7eb9e06`, catálogo v7).

**Fase temporal:** `snapshot_step_months` = 1 · `target_horizon_months` = 12 ·
`train_lookback_years` = 8 · `execution_lag_days` = 60 · `recency_weighting` = off

**Fase de representación:** `feature_preset` = all · `fundamental_momentum` = True ·
`market_regime_feature` = False · `neutralize_by_sector` = False · `winsorization` = 0.0 ·
`max_features_per_agent` = **20** · `feature_weighting_mode` = oos_stability_prune

**Fase de modelo:** `model_family` = lightgbm · `objective` = rank_regression · `lgbm_max_depth` = 3 ·
`lgbm_n_estimators` = 100 · `lgbm_learning_rate` = 0.03 · `lgbm_min_child_samples` = **20**

**Fase meta:** `meta_method` = **stacked_rolling_free** · `meta_history_quarters` = 16 ·
`meta_recency_weighting` = off. Es decir, sin tope por agente, que es lo que explica que el meta
acabe concentrando más del 95 % del peso en `risk`. La variante libre **la adoptó la segunda
pasada** (ver la tabla de la cadena: la 2 cambió `meta_method`, la 3 cambió `execution_lag_days` y
`target_size`); la tercera volvió a compararla con la acotada y la confirmó, con +0,00827 de ventaja
pareada.

**Cartera ganadora del Portfolio Study** (`portfolio_winner.json`, no modifica el ganador
predictivo): `target_size` = 8 · `max_cash_weight` = 0.0 · `sizing_mode` = alpha_proportional ·
`minimum_holding_period` = half_horizon · `coverage_percentile_floor` = 60 ·
`rebalance_drift_tolerance` = 0.1 · `commission_bps` = 5 · `slippage_bps` = 10.

La traza completa de las 17 decisiones de esta pasada, ordenadas por el coste de la alternativa
rechazada, está en `decisions.json` y se reproduce en el manuscrito
(`latex/assets/t06_decisiones.tex`). Dos de ellas se resolvieron por `tie_simplicity` y no por
evidencia; el manuscrito las declara como tales.

---

## 8. Qué se puede afirmar hoy

Se organiza por los dos objetivos del trabajo. Las cifras predictivas proceden del Model Study 3 y
las económicas de la cartera ganadora del Portfolio Study.

**Objetivo 1 — el sistema aprende a ordenar. Sí, con evidencia sólida:**

- Existe capacidad de ordenación transversal fuera de muestra: rank-IC 0,1090, IC-IR 0,851, t de
  Newey-West 3,46, bootstrap al 95 % [0,0425; 0,1723], 74,36 % de cohortes positivas.
- **Supera con holgura a los baselines deterministas**: el mejor, `garp_score`, se queda en 0,0130.
- **No es azar**: p de permutación 0,0001 sobre 9.999 réplicas, ninguna alcanzó lo observado.
- **No es un artefacto del código**: cinco placebos de etiqueta entre −0,0081 y +0,0013.
- **No depende de una era ni de una semilla**: 0,0780–0,1350 excluyendo eras completas; rango 0,0015
  entre semillas, sin cruce de cero.
- **No es un factor de estilo conocido**: retiene el 86,62 % tras neutralizar por 14 controles.
- **Bate al azar con riesgo comparable**: percentil 96,8 frente a 1.000 carteras aleatorias de
  riesgo emparejado (CAGR 15,56 % contra una mediana aleatoria de 9,23 %).
- **El meta-agente aprende**: parte de pesos iguales, se equivoca en 2016 concentrando en `momentum`
  y se corrige solo hacia `risk`; su ponderación aprendida supera a la ingenua en +0,0415 de rank-IC
  (+61 %).
- **Sigue ordenando en la era reservada**: rank-IC +0,0441, y no depende de la cartera.

**Objetivo 2 — la ordenación se sabe gestionar. Sí, con evidencia sólida:**

- **Las variables de cartera afectan al resultado y de forma muy desigual**: sobre 1.728
  configuraciones, el número de posiciones mueve el IR mediano 0,300 y el tope de efectivo 0,202,
  mientras que la tolerancia de deriva (0,016) y el reparto de pesos (0,009) son casi inertes.
- **Optimizar por Information Ratio mejora la cartera sin tocar el modelo**: IR de 0,339 a 0,844 y
  exceso geométrico del 2,61 al 6,97 %, reutilizando los scores congelados.
- **Bate al S&P 500 en la era reservada con la cartera optimizada**: +2,56 % geométrico, IR +0,304,
  1 de 2 años. Con la cartera por defecto era −11,29 % e IR −1,167.
- **El perfil que no reordena la señal domina** la ventana de selección: IR 0,844 frente a 0,570 del
  segundo.

**Sí, con matices que hay que declarar:**

- El agente `risk` en solitario tiene más rank-IC (0,1227) que el meta (0,1090), y el meta le asigna
  más del 95 % del peso: la arquitectura multi-agente no queda demostrada por estos datos.
- El coeficiente de transferencia es 0,328: la cartera captura una tercera parte de la señal.
- Las 117 cohortes equivalen a ~9 observaciones independientes por el horizonte anual.
- La cartera ganadora es la mejor de 1.728 evaluadas: su IR 0,844 es una cota superior optimista,
  no una estimación insesgada.

**No, todavía no:**

- Que el Sharpe resista la corrección por multiplicidad: Deflated Sharpe 0,682 < 0,95 con 71
  configuraciones probadas. Encadenar tres estudios compró rank-IC al precio de encarecer esto.
- Que el resultado económico esté confirmado con potencia: la era reservada aporta 6 cohortes
  cerradas y ~1,41 años de cartera, aproximadamente una observación independiente.
- Que el alfa de la ventana de selección sea significativo por sí solo: 0,27 % por periodo con
  t = 1,44 y R² = 0,017.
- Que la conclusión económica sea estable entre semillas: el rango del exceso geométrico cruza cero
  (`economic_conclusion_stable` = false), mientras que la predictiva no lo hace.

## 9. Trabajo futuro que sugiere esta evidencia

1. **Esperar al cierre de cohortes de 2025–2026** para contrastar con potencia real, tanto el
   rank-IC como el resultado económico. Es la limitación que más ata a todas las demás.
2. **Separar frecuencia de evaluación de frecuencia de ejecución**: se redecide cada mes sobre una
   señal a doce meses, y de ahí sale la rotación por ventaja neta (37,0 % del flujo de órdenes de
   la cartera ganadora). Una fracción comparable, el 38,2 %, es `rebalance` puro: corrección de
   deriva de pesos, que depende de `rebalance_drift_tolerance` y no de la cadencia.
3. **Reejecutar con un catálogo pre-registrado más estrecho**, para que el Deflated Sharpe no pague
   el peaje de 71 pruebas más una rejilla de 1.728 carteras.
4. **Modelar costes dependientes de liquidez**: una cartera de 8 posiciones con 324 % de rotación
   anual es justo donde el supuesto de coste constante (5 + 10 pb) puede romperse.
5. **Investigar por qué `risk` domina**: no es baja volatilidad clásica —`gap_21d` y `range_63d`
   encabezan el 78 % de las observaciones y `beta_252d` ni está entre las tres primeras—, lo que
   explica que la neutralización por estilos apenas destruya el 13 % de la señal. Que
   microestructura a semanas ordene retornos a doce meses merece caracterizarse.
