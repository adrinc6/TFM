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

> ⚠️ **Aviso sobre las secciones 1 a 6.** Su estructura argumental sigue siendo válida, pero **las
> cifras concretas proceden del study derogado `b4d7a8d8`** y no se han reescrito una por una. Para
> cualquier cifra vigente, la fuente de verdad es el manuscrito en `latex/assets/`, que sí está
> completamente actualizado contra los cuatro estudios, o directamente los artefactos en disco.
> Estas secciones se conservan por su análisis cualitativo, no como referencia numérica.

## 1. El proceso de aprendizaje

Esta sección es el núcleo del TFM: no basta con que el sistema acierte, hay que poder **enseñar cómo
aprende**. Hay tres evidencias independientes de aprendizaje, y las tres son observables en
artefactos.

### 1.1 El meta-agente aprende a quién escuchar

El meta-agente arranca sin información: en las primeras 60 filas de `evidence/meta_weights.parquet`
el estado es `fallback_equal` y los cinco agentes reciben 0,20 exactos, porque todavía no hay
cohortes cerradas con las que estimar la calidad de nadie. A medida que se cierran etiquetas a 12
meses, el estado pasa a `learned` (615 de 675 filas) y los pesos se separan:

| Año | growth | momentum | quality | risk | value |
|---|---|---|---|---|---|
| 2016 | 0,285 | 0,308 | 0,100 | 0,204 | 0,104 |
| 2017 | 0,116 | 0,315 | 0,100 | 0,370 | 0,100 |
| 2018 | 0,170 | 0,130 | 0,104 | 0,493 | 0,104 |
| 2019 | 0,210 | 0,100 | 0,100 | 0,490 | 0,100 |
| 2020 | 0,172 | 0,109 | 0,109 | 0,500 | 0,109 |
| 2021 | 0,188 | 0,104 | 0,104 | 0,500 | 0,104 |
| 2022 | 0,128 | 0,124 | 0,124 | 0,500 | 0,124 |
| 2023 | 0,125 | 0,125 | 0,125 | 0,500 | 0,125 |
| 2024 | 0,126 | 0,125 | 0,125 | 0,500 | 0,125 |

Fuente: `evidence/meta_weights.parquet` (papel: diagnóstico).

La lectura es una **curva de aprendizaje explícita**. En 2016 el meta reparte casi a ciegas y apuesta
por `momentum` (0,308), que resultará ser el peor agente del sistema. En 2017 ya ha corregido: baja
`momentum` y sube `risk` a 0,370. Desde 2018 mantiene `risk` pegado al tope de 0,50 que impone la
cota superior del método `stacked_rolling_bounded`. El sistema tarda unos dos años en identificar a
su mejor especialista y después no lo suelta. La rotación media de pesos es 0,0093 y la concentración
media (HHI) 0,295 (`robustness.json`): aprende rápido y luego es estable, no errático.

### 1.2 La ponderación aprendida vale más que la ingenua

Es el contraste que separa «aprender» de «promediar». Con los mismos cinco agentes y las mismas
señales, la única diferencia es cómo se combinan:

| Señal | Rank-IC medio | Cohortes positivas | IC-IR |
|---|---|---|---|
| `meta_final` (pesos aprendidos) | **0,1004** | 71,79 % | 0,744 |
| `meta_equal_weight` (0,20 fijos) | 0,0659 | 62,39 % | 0,526 |

Fuente: `evidence/rank_ic_diagnostics.parquet` (papel: diagnóstico).

Aprender los pesos añade **+0,0345 de rank-IC** sobre repartir por igual, un 52 % más de señal. Ese
delta no viene de mejores features ni de más datos: viene exclusivamente del aprendizaje del
meta-agente. Es la demostración más limpia de que la capa de combinación hace un trabajo real.

### 1.3 Cada agente aporta lo que sabe, y el meta lo ordena

| Agente | Rank-IC medio | Desv. típica | Cohortes positivas | IC-IR |
|---|---|---|---|---|
| `risk` | 0,1229 | 0,1236 | 82,05 % | 0,995 |
| **`meta_final`** | **0,1004** | 0,1350 | 71,79 % | 0,744 |
| `meta_equal_weight` | 0,0659 | 0,1252 | 62,39 % | 0,526 |
| `growth` | 0,0254 | 0,0873 | 62,39 % | 0,291 |
| `value` | 0,0235 | 0,0801 | 59,83 % | 0,293 |
| `quality` | 0,0036 | 0,1049 | 46,15 % | 0,035 |
| `momentum` | 0,0022 | 0,0888 | 47,86 % | 0,024 |

Fuente: `evidence/rank_ic_diagnostics.parquet`, ventana de selección 2015–2024 (papel: diagnóstico).

Hay que decirlo con honestidad, y el TFM debe defenderlo explícitamente: **`risk` en solitario tiene
más rank-IC que el meta**. El meta no supera a su mejor agente, lo cual es esperable en un
combinador acotado que nunca puede asignar más de 0,50 a nadie. Lo que sí hace el meta es (a) superar
con claridad a la combinación ingenua, y (b) llegar a esa concentración **sin conocer de antemano**
qué agente era el bueno: lo descubre en 2016–2017 con datos que ya estaban cerrados. La defensa no es
«el meta es el mejor predictor», sino «el meta aprende, sin supervisión externa, a reproducir casi
toda la señal de su mejor especialista partiendo de la ignorancia».

La estabilidad por eras muestra además que ningún agente domina siempre:

| Agente | 2015–2018 | 2019–2021 | 2022–2024 |
|---|---|---|---|
| `risk` | 0,1292 | 0,0576 | 0,1803 |
| `meta_final` | 0,0976 | 0,0423 | 0,1621 |
| `meta_equal_weight` | 0,0816 | 0,0107 | 0,1015 |
| `growth` | 0,0463 | 0,0157 | 0,0091 |
| `value` | 0,0121 | 0,0304 | 0,0307 |
| `momentum` | 0,0397 | −0,0438 | 0,0011 |
| `quality` | 0,0032 | −0,0306 | 0,0383 |

En 2019–2021 `momentum` y `quality` se vuelven negativos y el meta cae a 0,0423; en 2022–2024 el
sistema alcanza su mejor rank-IC (0,1621). El aprendizaje **no se degrada con el tiempo**, que es lo
contrario de lo que ocurría en el estudio derogado (0,107 → 0,022).

---

## 2. Capacidad predictiva (papel: selección)

| Métrica | Valor | Artefacto |
|---|---|---|
| Rank-IC medio | 0,1004 | `evidence/summary.json` |
| IC-IR | 0,7436 | `evidence/summary.json` |
| Cohortes positivas | 71,79 % | `evidence/summary.json` |
| Cohortes | 117 | `evidence/summary.json` |
| t de Newey-West | 3,020 | `attribution.json` |
| Observaciones independientes efectivas | 9 | `attribution.json` |
| Desviación típica del IC | 0,1350 | `evidence/summary.json` |
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
| **Sistema (`meta_final`)** | **0,1004** | 71,79 % |
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
| Rank-IC observado | 0,1004 |
| Permutaciones | 9 999 |
| p-valor (con corrección +1) | **0,0001** |

Fuente: `robustness.json`. Se permuta la etiqueta dentro de cada cohorte, destruyendo la relación
señal-futuro y conservando toda la estructura transversal. **Ninguna de las 9 999 permutaciones
alcanzó el rank-IC observado.** Es el contraste más directo contra la hipótesis de suerte.

### 3.2 Placebos de etiqueta — ¿lo produce la maquinaria?

| Semilla del placebo | Rank-IC | IC-IR |
|---|---|---|
| 101 | −0,0061 | −0,126 |
| 102 | +0,0007 | 0,016 |
| 103 | −0,0038 | −0,079 |
| 104 | +0,0008 | 0,017 |
| 105 | −0,0015 | −0,031 |

Fuente: `robustness.json`. Con etiquetas barajadas, el pipeline completo —features, cinco agentes,
LightGBM, meta apilado, cartera— produce rank-IC en el rango [−0,006; +0,001], centrado en cero
frente al 0,1004 real. Si el resultado fuese un artefacto del código (fuga temporal, normalización
mal hecha, sesgo del optimizador), los placebos lo mostrarían. No lo muestran. **La señal viene de
los datos, no del programa.**

### 3.3 Bootstrap por bloques — ¿es distinguible de cero?

| Intervalo | Límite inferior | Límite superior |
|---|---|---|
| 90 % | 0,0449 | 0,1585 |
| 95 % | **0,0335** | 0,1695 |

Fuente: `robustness.json`, bloques de 12 cohortes para respetar el solapamiento de etiquetas. **El
intervalo al 95 % excluye el cero con holgura**: el límite inferior sigue siendo 3× el umbral de
explotabilidad de la literatura.

### 3.4 Exclusión de eras — ¿depende de un periodo afortunado?

| Era excluida | Rank-IC del resto | Cohortes |
|---|---|---|
| 2015–2018 | 0,1022 | 72 |
| 2019–2021 | 0,1262 | 81 |
| 2022–2024 | 0,0730 | 81 |

Fuente: `robustness.json`. Quitando cualquier era completa el rank-IC se mantiene entre 0,073 y
0,126, siempre muy por encima de cero. **No hay un periodo del que dependa el resultado.**

### 3.5 Semillas — ¿depende del azar de la inicialización?

| Semilla | Rank-IC | IC-IR | Exceso geométrico | IR | CAGR confirmación |
|---|---|---|---|---|---|
| 42 (ganadora) | 0,1004 | 0,744 | 1,62 % | 0,269 | 36,11 % |
| 7 | 0,0984 | 0,732 | 1,12 % | 0,182 | 34,39 % |
| 2026 | 0,0989 | 0,730 | 1,70 % | 0,269 | 30,15 % |

Fuente: `robustness.json` → `seeds`, `seed_dispersion`. El rango de rank-IC es 0,0020 y el del exceso
geométrico 0,0057; **ninguna magnitud cruza el cero** y `economic_conclusion_stable = true`. Esto
corrige el defecto más grave del estudio derogado, donde el alfa cambiaba de signo con la semilla.

### 3.6 Carteras aleatorias con riesgo emparejado — ¿bate a la suerte?

| Contraste | CAGR modelo | Mediana aleatoria | p95 aleatorio | Percentil del modelo |
|---|---|---|---|---|
| Riesgo emparejado | 14,48 % | 9,22 % | 13,55 % | **97,4 %** |
| General | 14,48 % | 12,40 % | 102,28 % | 65,3 % |

Fuente: `robustness.json`, 1 000 simulaciones de 12 nombres con 15 pb de costes. Contra carteras
aleatorias **de riesgo comparable**, el modelo está en el percentil 97,4: bate al azar. El contraste
«general» no es informativo y hay que decirlo en el TFM: su p95 es un CAGR del 102 % anual, dominado
por carteras de 12 nombres que concentraron supervivientes extremos; compararse con eso no mide
habilidad sino tolerancia a la varianza.

### 3.7 Neutralización por estilo — ¿es un factor conocido disfrazado?

| Métrica | Valor |
|---|---|
| Rank-IC bruto | 0,1111 |
| Rank-IC neutralizado por 14 controles de estilo | **0,0937** |
| Fracción retenida | **84,35 %** |

Fuente: `attribution.json`. Tras neutralizar por P/E, P/B, P/S, EV/EBITDA, retorno relativo 12m,
momentum 12-1, volatilidad realizada 63d y 126d, beta 252d, ROE, ROIC, margen operativo y crecimiento
de BPA y de ventas, **sobrevive el 84 % de la señal**. La ordenación no es una réplica de los
factores clásicos.

La regresión con réplicas de factores y errores Newey-West lo confirma por el otro lado: en la
ventana de selección el alfa por periodo es 0,13 % con t = 0,82 y R² = 0,021, sin ninguna carga
significativa (la mayor es `quality`, t = 1,10). Es decir, el exceso **no se explica** por exposición
a estilos, pero tampoco alcanza significación estadística propia en esa ventana — un matiz que el
TFM debe reportar sin adornar.

### 3.8 El contraste que no se supera: Deflated Sharpe

| Métrica | Valor |
|---|---|
| Sharpe observado por periodo | 0,1200 |
| Configuraciones probadas | 66 |
| Probabilidad Deflated Sharpe | **0,930** |
| Umbral exigido | 0,95 |

Fuente: `attribution.json`. Corrigiendo por haber probado 66 configuraciones, la probabilidad queda
en 0,930, **por debajo del 0,95** requerido. Hay que reportarlo como lo que es: la evidencia de
**capacidad predictiva** (rank-IC) supera todos los contrastes, mientras que la evidencia de
**rentabilidad ajustada por riesgo** no resiste del todo la corrección por multiplicidad. Ocultar
esto invalidaría el resto del capítulo de robustez.

### Resumen del capítulo de robustez

| Contraste | Pregunta que responde | Resultado |
|---|---|---|
| Permutación (9 999) | ¿Es azar? | p = 0,0001 ✔ |
| Placebos de etiqueta (5) | ¿Es un artefacto del código? | [−0,006; +0,001] ✔ |
| Bootstrap por bloques 95 % | ¿Es distinguible de cero? | [0,0335; 0,1695] ✔ |
| Exclusión de eras | ¿Depende de un periodo? | 0,073–0,126 ✔ |
| Semillas (3) | ¿Depende de la inicialización? | rango 0,0020, sin cruce de cero ✔ |
| Carteras aleatorias (riesgo emparejado) | ¿Bate al azar? | percentil 97,4 ✔ |
| Neutralización por estilo | ¿Es un factor conocido? | retiene 84,35 % ✔ |
| Deflated Sharpe | ¿Resiste la multiplicidad? | 0,930 < 0,95 ✘ |

**Siete de ocho.** La conclusión defendible es: *el sistema aprende una ordenación transversal real,
estadísticamente distinguible del azar, no reproducible por la maquinaria con etiquetas falsas, no
dependiente de una era ni de una semilla, y no explicable por factores de estilo conocidos.* Lo que
**no** puede afirmarse con la misma rotundidad es que su Sharpe sobreviva a la corrección por las 66
configuraciones probadas.

---

## 4. Traducción económica (papel: selección)

| Métrica | Ventana de selección 2015–2024 | Curva completa 2015–2026 |
|---|---|---|
| CAGR cartera | 15,01 % | 17,36 % |
| CAGR benchmark (SPY) | 13,17 % | 13,81 % |
| Exceso geométrico | 1,62 % | 3,12 % |
| Information Ratio anualizado | 0,269 | 0,416 |
| Máximo drawdown | 23,44 % | 23,44 % |
| Beat rate | 8/10 años | 10/12 años |
| Alfa anual medio | 1,72 % | 3,08 % |
| Alfa anual mediano | 3,01 % | 3,39 % |
| Peor año (alfa) | −11,63 % | −11,63 % |
| Turnover anualizado | 359,08 % | 319,97 % |
| Coste total acumulado | 5,25 % | 5,40 % |
| Coeficiente de transferencia | 0,247 | — |

Fuente: `evidence/summary.json`.

**Lectura.** El coeficiente de transferencia de 0,247 es el diagnóstico central: de la señal medida
por el rank-IC, la cartera sólo captura una cuarta parte. La ley fundamental de la gestión activa
(`IR ≈ IC·√BR·TC`) con IC 0,10 y ~400 nombres implicaría un IR teórico muy superior al 0,269
realizado. La causa es estructural y hay que explicarla en el TFM: una cartera *long-only* de 12
posiciones sólo puede expresar el extremo superior de la ordenación, y el 359 % de rotación anual
paga 5,25 puntos de costes acumulados. **El cuello de botella no es el modelo, es la cartera.**

### Detalle anual

| Año | Cartera | Benchmark | Alfa | Bate | MDD año | IR año | Efectivo | Turnover |
|---|---|---|---|---|---|---|---|---|
| 2015 | 3,41 % | −0,63 % | **+4,06 %** | ✔ | 7,63 % | 1,521 | 0,00 % | 1,00 |
| 2016 | 14,01 % | 10,88 % | **+2,82 %** | ✔ | 1,55 % | 0,368 | 0,00 % | 4,17 |
| 2017 | 29,46 % | 21,71 % | **+6,37 %** | ✔ | 1,10 % | 1,167 | 0,00 % | 6,80 |
| 2018 | −2,42 % | −5,40 % | **+3,15 %** | ✔ | 12,58 % | 0,502 | 13,91 % | 3,38 |
| 2019 | 43,63 % | 32,05 % | **+8,77 %** | ✔ | 1,79 % | 1,241 | 0,00 % | 4,71 |
| 2020 | 12,86 % | 18,02 % | −4,37 % | ✘ | 23,44 % | −0,480 | 6,25 % | 6,35 |
| 2021 | 31,72 % | 29,71 % | **+1,55 %** | ✔ | 2,39 % | 0,423 | 13,89 % | 1,62 |
| 2022 | −16,03 % | −18,38 % | **+2,88 %** | ✔ | 13,10 % | 0,341 | 25,23 % | 0,62 |
| 2023 | 30,75 % | 26,18 % | **+3,63 %** | ✔ | 13,30 % | 0,589 | 6,30 % | 5,19 |
| 2024 | 10,77 % | 25,34 % | −11,63 % | ✘ | 5,95 % | −1,784 | 22,92 % | 1,18 |
| **2025** | **29,69 %** | **18,17 %** | **+9,76 %** | ✔ | 6,52 % | 1,615 | 25,00 % | 0,47 |
| **2026** | **19,19 %** | **8,43 %** | **+9,92 %** | ✔ | 7,32 % | 0,853 | 25,13 % | 0,52 |

Fuente: `evidence/annual_metrics.parquet`. Las dos últimas filas son la era reservada.

Los dos años perdedores tienen una explicación común y comprobable: 2020 y 2024 son los años de
mayor concentración del índice en megacapitalizaciones de crecimiento, precisamente donde una cartera
con sesgo a bajo riesgo y 12 nombres equiponderados no puede seguir al benchmark. En 2024 el sistema
además mantuvo un 22,9 % de efectivo.

---

## 5. La era reservada 2025–2026 (papel: confirmación)

Es el resultado más exigente del trabajo, porque **ninguna de sus observaciones intervino en ninguna
decisión**: ni en la elección del ganador, ni en los pesos del meta, ni en los umbrales de cartera.
El protocolo de lectura estaba pre-registrado en `docs/bitacora.md` antes de mirar estos números.

| Métrica | Valor |
|---|---|
| CAGR cartera | **36,11 %** |
| CAGR benchmark | 19,18 % |
| Exceso geométrico | **+14,21 %** |
| Information Ratio anualizado | **0,959** |
| Beat rate | **2/2 años (100 %)** |
| Alfa anual medio | 9,84 % |
| Peor año (alfa) | **+9,76 %** |
| Máximo drawdown | 7,32 % |
| Turnover anualizado | 65,78 % |
| Efectivo medio | 25,04 % |
| Coste total | 0,15 % |

Fuente: `evidence/summary.json` → `confirmation`, `attribution.json` → `confirmation_2025_2026`.

**El sistema bate al S&P 500 en los dos años reservados, y lo hace con margen**: +9,76 pp en 2025 y
+9,92 pp en 2026, con un IR de 0,959 —más del triple del 0,269 de la ventana de selección— y un
drawdown máximo de sólo 7,32 % frente al 23,44 % histórico. La regresión factorial de la era
reservada da un alfa por periodo del 1,64 % con **t de Newey-West = 4,76**, esta vez sí claramente
significativo. Además el turnover cae al 65,78 % y los costes a 0,15 %, con un 25 % de efectivo: el
sistema obtuvo su mejor resultado operando menos.

### El matiz obligatorio, que el TFM no debe esconder

El rank-IC de la era reservada es **−0,0119** (6 cohortes cerradas, IC-IR −0,211, t = −0,82), es
decir, ligeramente negativo. Esto parece contradecir el excelente resultado económico, y la
explicación es metodológica, no un fallo:

1. **Sólo hay 6 cohortes con etiqueta cerrada** y `attribution.json` estima **1 sola observación
   independiente**. Con horizonte de 12 meses y cadencia mensual, cohortes contiguas comparten casi
   toda la ventana de etiqueta. Un rank-IC calculado sobre una observación independiente no tiene
   potencia estadística: su intervalo de confianza es enorme y el signo es esencialmente arbitrario.
2. **El rank-IC mide la ordenación completa de ~400 valores; la cartera sólo usa el extremo
   superior.** Un sistema puede ordenar mal el conjunto y acertar en las 12 mejores, que es lo único
   que se traduce en rentabilidad.
3. **Las cohortes de 2025 H2 y 2026 aún no tienen etiqueta cerrada** y no entran en esa media.

La formulación honesta para el tribunal es: *la confirmación económica en la era reservada es
inequívoca y fuerte (2/2 años, +14,21 % geométrico, t = 4,76); la confirmación de la capacidad
predictiva medida por rank-IC en esa misma era es todavía indeterminada por falta de cohortes
cerradas, no negativa en sentido estadístico.* Las dos cosas conviven sin contradicción.

---

## 6. Perfiles: por qué gana `balanced`

Los ocho perfiles comparten **exactamente la misma señal** —el rank-IC es 0,1004 en los ocho,
`profile_comparison.parquet`— y se diferencian sólo en cómo traducen esa ordenación a cartera. Es un
experimento controlado ideal: cualquier diferencia de resultado es atribuible a la regla de
construcción, no al modelo.

| Perfil | CAGR | Exceso geom. | IR | MDD | Beat rate | Alfa medio | Turnover |
|---|---|---|---|---|---|---|---|
| **`balanced`** | **15,01 %** | **+1,62 %** | **0,269** | 23,44 % | **8/10** | **+1,72 %** | 3,59 |
| `defensive` | 14,57 % | +1,23 % | 0,204 | 24,71 % | 6/10 | +1,43 % | 2,58 |
| `value` | 13,74 % | +0,50 % | 0,069 | **22,91 %** | 6/10 | +0,66 % | 3,30 |
| `quality` | 12,84 % | −0,29 % | −0,028 | 23,88 % | 4/10 | −0,20 % | 3,23 |
| `contrarian` | 11,48 % | −1,50 % | −0,200 | 26,29 % | 5/10 | −1,20 % | 4,57 |
| `garp` | 11,33 % | −1,63 % | −0,226 | 24,55 % | 4/10 | −1,51 % | 3,41 |
| `growth` | 10,56 % | −2,31 % | −0,290 | 27,46 % | 5/10 | −1,88 % | 4,43 |
| `momentum` | 5,97 % | −6,37 % | −0,546 | 39,82 % | 2/10 | −5,76 % | 6,11 |

Fuente: `profile_comparison.parquet` y `evidence/profiles/*/summary.json`. Benchmark: 13,17 %.

**`balanced` gana en todos los ejes que importan a la vez**: mayor CAGR, mayor exceso geométrico,
mayor IR, mayor beat rate y el mejor alfa medio. Y es el único perfil que **no impone ningún sesgo
adicional**: toma los valores en el orden en que el meta-agente los ha ordenado, sin volver a
filtrarlos ni reordenarlos por un criterio factorial externo. Los siete perfiles restantes reordenan
esa lista según su tesis (crecimiento, momentum, valor…), y **seis de los siete destruyen alfa**.

Este es un resultado con mucha carga argumental para el TFM, y conviene enunciarlo así: *la mejor
manera de usar la señal aprendida es no interferir con ella*. Cada capa de criterio humano añadida
sobre la ordenación del modelo empeora el resultado, de forma monótona con lo agresivo del sesgo
—`momentum`, el perfil que más reordena y más rota (611 % de turnover), es el que peor lo hace, con
−6,37 % de exceso y un drawdown del 39,82 %—. La señal ya contiene la información; el sesgo sólo
añade rotación y coste.

Obsérvese también que el ranking de perfiles **no** reproduce el ranking de agentes: `momentum` es a
la vez el peor agente (rank-IC 0,0022) y el peor perfil (IR −0,546), lo que refuerza que el sistema
está capturando algo distinto del momentum clásico.

---

## 7. Configuración ganadora

Seleccionada por rank-IC pareado, con puerta de no inferioridad y suelo por era, sin que la era
reservada participe en ninguna decisión (`decisions.json`, `winner.json`).

**Fase temporal:** `snapshot_step_months` = 1 · `target_horizon_months` = 12 ·
`train_lookback_years` = 8 · `execution_lag_days` = 60 · `recency_weighting` = off

**Fase de representación:** `feature_preset` = all · `fundamental_momentum` = True ·
`market_regime_feature` = False · `neutralize_by_sector` = False · `winsorization` = 0.0 ·
`max_features_per_agent` = 12 · `feature_weighting_mode` = oos_stability_prune

**Fase de modelo:** `model_family` = lightgbm · `objective` = rank_regression · `lgbm_max_depth` = 3 ·
`lgbm_n_estimators` = 100 · `lgbm_learning_rate` = 0.03 · `lgbm_min_child_samples` = 50

**Fase meta:** `meta_method` = stacked_rolling_bounded · `meta_history_quarters` = 16 ·
`meta_recency_weighting` = off

**Cartera (no modifica el ganador):** `target_size` = 12 · `exit_expected_alpha_bps` = 100 ·
`rotation_edge_bps` = 50 · `cash_policy` = opportunity_cash · `max_cash_weight` = 0,25 ·
`sizing_mode` = alpha_proportional · `commission_bps` = 5 · `slippage_bps` = 10

Decisión destacable: `snapshot_step_months` = 1 ganó con ventaja pareada +0,0208 e IC al 90 %
[0,0108; 0,0372] —distinguible de cero—, multiplicando por ~3 las cohortes disponibles (40 → 117) y
mejorando el rank-IC de 0,0524 a 0,0735 en esa fase. `target_horizon_months` = 6 fue rechazado
explícitamente: ventaja pareada −0,0265 con IC [−0,0552; −0,0109], claramente inferior.

---

## 8. Qué se puede afirmar hoy

**Sí, con evidencia sólida:**

- Existe capacidad de ordenación transversal fuera de muestra: rank-IC 0,1004, IC-IR 0,744, t de
  Newey-West 3,02, bootstrap al 95 % [0,0335; 0,1695].
- **No es azar**: p de permutación 0,0001 sobre 9 999 réplicas.
- **No es un artefacto del código**: cinco placebos de etiqueta en [−0,006; +0,001].
- **No depende de una era ni de una semilla**: 0,073–0,126 excluyendo eras; rango 0,0020 entre
  semillas, sin cruce de cero.
- **No es un factor de estilo conocido**: retiene el 84,35 % tras neutralizar por 14 controles.
- **El meta-agente aprende**: pasa de pesos iguales a concentrar en su mejor especialista, y su
  ponderación aprendida supera a la ingenua en +0,0345 de rank-IC (+52 %).
- **Bate al S&P 500 en los dos años reservados**: +14,21 % geométrico, IR 0,959, 2/2 años, alfa
  factorial con t = 4,76.
- **`balanced` es el mejor perfil** en CAGR, exceso, IR, beat rate y alfa medio simultáneamente.

**Sí, con matices que hay que declarar:**

- El agente `risk` en solitario tiene más rank-IC (0,1229) que el meta (0,1004).
- El coeficiente de transferencia es 0,247: la cartera captura una cuarta parte de la señal.
- Las 117 cohortes equivalen a ~9 observaciones independientes.

**No, todavía no:**

- Que el Sharpe resista la corrección por multiplicidad: Deflated Sharpe 0,930 < 0,95 con 66
  configuraciones probadas.
- Que el rank-IC esté confirmado en la era reservada: −0,0119 sobre 6 cohortes y 1 observación
  independiente, sin potencia para concluir en ningún sentido.
- Que el alfa de la ventana de selección sea significativo por sí solo: t = 0,82.

## 9. Trabajo futuro que sugiere esta evidencia

1. **Atacar el coeficiente de transferencia**, no el modelo: es donde se pierde el 75 % de la señal.
   Ampliar `target_size`, reducir rotación y explorar sizing por convicción.
2. **Reejecutar con menos configuraciones** o con un catálogo pre-registrado más estrecho, para que
   el Deflated Sharpe no pague el peaje de 66 pruebas.
3. **Esperar al cierre de cohortes de 2025–2026** para poder contrastar el rank-IC de la era
   reservada con potencia real.
4. **Investigar por qué `risk` domina**: si es baja volatilidad clásica, la neutralización debería
   haber destruido más del 16 % de la señal; que no lo haga sugiere que hay algo propio que merece
   caracterizarse.
