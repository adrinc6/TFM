# Metodología completa del Model Study

## 1. Pregunta de investigación

El objetivo principal no es encontrar retrospectivamente la cartera con mayor rentabilidad. La
pregunta es si un sistema de agentes puede aprender una ordenación transversal de acciones que
mantenga capacidad predictiva fuera de muestra y a través de regímenes diferentes.

La variable central es Rank-IC: correlación de Spearman entre la puntuación producida en una fecha
y el retorno futuro observado cuando la etiqueta queda cerrada. Rentabilidad y alfa son resultados
económicos posteriores. Separar ambas capas evita seleccionar reglas de cartera porque casualmente
funcionaron mejor en la historia conocida.

## 2. Universo y datos point-in-time

El universo se reconstruye con la composición histórica del S&P 500. Una empresa solo puede
participar en fechas en las que pertenecía al universo. Los fundamentales se incorporan según su
fecha de publicación y un lag de ejecución de 30, 45 o 60 días. Los precios y SPY se alinean con
cada snapshot.

Los datos crudos se regeneran con `python main.py ingest`, que descarga y consolida precios,
fundamentales de Finnhub y fechas reales de publicación de EDGAR en `data/raw/`. Es el único punto
de entrada a la ingesta y existe para que el trabajo pueda reproducir sus propios datos desde cero.

La identidad del dataset incluye fuentes, fechas, universo, cadencia, horizonte, lag y versiones
de transformación. Una identidad igual reutiliza `data/prepared/<dataset_hash>/`; una identidad
distinta crea otra materialización. Los Studies guardan referencias, no copias.

**La identidad de una evaluación incluye el hash del dataset.** Sin él, dos configuraciones
idénticas evaluadas sobre datos distintos compartían clave de caché y la segunda leía el resultado
de la primera: en los artefactos anteriores se observa la misma `evaluation_key` asociada a dos CAGR
distintos. Es un fallo de corrección, no de reporte, y por eso la clave se calcula ahora a partir de
la identidad del dataset resuelta antes de materializarlo.

Controles de ausencia de lookahead:

1. La fecha efectiva del fundamental no supera el snapshot.
2. El target se calcula hacia delante, pero solo se usa para evaluar o entrenar meta cuando su
   `label_end_date` ya ha pasado.
3. El meta solo consume cohortes OOS trimestrales cerradas.
4. 2025–2026 se separa antes de cualquier selector.

## 3. Optimización secuencial

El usuario no manipula un selector de modo separado. Marca directamente valores del catálogo:

- un único valor implica `fixed`;
- dos o más valores de una variable predictiva implican `optimize`;
- dos o más valores de cartera implican una comparación `diagnostic`.

La configuración recomendada ya llega marcada. Si se añade un segundo valor, el presupuesto se
recalcula inmediatamente; si se vuelve a dejar uno, la variable vuelve a ser fija. La API persiste
el modo derivado para que la ejecución sea explícita y auditable.

El proceso es greedy secuencial, no cartesiano:

1. Ejecutar baseline.
2. Tomar la primera variable optimizable.
3. Evaluar todos sus valores sobre el incumbent acumulado.
4. Elegir mediante Rank-IC robusto.
5. Fijar el ganador.
6. Continuar con la siguiente variable.

Por tanto:

```text
evaluaciones predictivas = 1 baseline + suma de alternativas por variable optimizada
```

El orden es temporal, representación, modelo y meta. No es reordenable porque cambiarlo alteraría
la trayectoria científica.

## 4. Variables predictivas

### 4.1 Temporal

- **Cadencia 1/3/6/12 meses:** frecuencia de observación y puntuación.
- **Horizonte 3/6/12 meses:** retorno futuro que aprende el modelo.
- **Lookback 4/8/12 años:** historia disponible en cada fit walk-forward.
- **Lag 30/45/60 días:** prudencia sobre disponibilidad real de fundamentales.
- **Recencia off/lineal/exponencial:** peso temporal de observaciones de entrenamiento.
- **Objetivo:** regresión de ranking o ranking directo.

Un lag menor puede mejorar actualidad, pero exige una hipótesis más fuerte sobre publicación. Por
eso el informe debe mostrar su sensibilidad explícitamente.

### 4.2 Representación

Los presets core y all seleccionan bloques cerrados. **Ambos alimentan a los cinco agentes**: un
preset que deja a un agente sin ningún bloque activo lo elimina de hecho del sistema, y entonces la
comparación deja de medir qué información necesita cada agente para medir qué pasa al amputar parte
de la arquitectura. `core` da a cada agente su bloque esencial; `all` le da toda la profundidad
disponible de su especialidad. También pueden
compararse momentum fundamental, régimen de mercado, neutralización sectorial, winsorización,
máximo de features y poda por estabilidad OOS. No se admiten listas manuales de features.

### 4.3 Modelo

Cada uno de los agentes quality, value, growth, momentum y risk ajusta LightGBM o Elastic Net.
LightGBM permite comparar profundidad, estimadores, learning rate y mínimo por hoja. Los
parámetros incompatibles quedan ocultos e inactivos.

### 4.4 Meta-agente

- **Equal:** 20 % exacto por agente.
- **Rolling free:** Ridge positivo, pesos 0–100 %.
- **Rolling bounded:** Ridge positivo, pesos 10–50 %.

El stacker convierte scores y retornos en rangos transversales. Se ajusta en cada fecha con las
últimas 8 o 16 cohortes trimestrales cerradas. Si no existe evidencia suficiente, usa equal.

## 5. Regla de selección

Las comparaciones son pareadas por cohorte y solo usan datos hasta 2024.

Las cohortes se emparejan por **periodo mensual**, no por cadena de fecha: los snapshots se
sitúan en `fin_de_mes + execution_lag_days`, de modo que barrer el lag desplaza toda la rejilla y el
emparejamiento literal daría cero fechas comunes.

Elegibilidad:

- observaciones suficientes;
- ninguna era disponible con Rank-IC inferior a −0,02;
- y **una de estas dos** puertas pareadas:
  - **dominancia**: diferencia media de Rank-IC positiva **y** mejor en más de la mitad de las
    cohortes emparejadas;
  - **no inferioridad**: límite inferior del bootstrap pareado al 90 % superior a −0,01.

La puerta doble corrige un defecto de diseño de la versión anterior, que solo exigía no
inferioridad. Aplicada a un candidato *superior*, esa prueba lo penalizaba: cuanto más se
diferenciaba del incumbent, más ancho era su intervalo y más fácil era que el límite inferior cayera
por debajo del margen. La regla premiaba estructuralmente al incumbent y llegó a descartar al mejor
candidato disponible por 0,00023 en un bootstrap de 1 000 extracciones (ahora 2 000).

Cuando las dos series no comparten al menos un bloque completo de fechas, el resultado pareado se
marca **no aplicable** y el candidato no es elegible. Antes ese caso devolvía `ci_low = 0,0` en
silencio, lo que satisfacía cualquier prueba de no inferioridad y dejaba pasar automáticamente a
todos los candidatos de las variables que desplazan la rejilla.

Orden entre elegibles:

1. Mayor **ventaja pareada** de Rank-IC contra el incumbent. La diferencia se mide cohorte a cohorte,
   lo que elimina el factor común de mercado de cada fecha; la mediana entre tres eras que se usaba
   antes es la mediana de tres números y no distingue diferencias del orden del ruido.
2. Diferencias inferiores a 0,002 son empate y decide la simplicidad.
3. Mayor Rank-IC medio.
4. Mayor fracción positiva.
5. Menor variabilidad.
6. Menor complejidad.
7. Conservar incumbent.

El spread de cola, el alfa y el IR se muestran como diagnósticos y no alteran esta decisión. La
corrección por multiplicidad de las cifras finales se hace con el Deflated Sharpe Ratio en
`attribution.json`; el proyecto no usa Holm.

## 6. Cartera dinámica

SPY es únicamente benchmark y nunca una posición. Los umbrales de la cartera son **económicos, en
puntos básicos de alfa esperado**, no percentiles del ranking: un percentil no dice cuánto se espera
ganar y por tanto no puede compararse contra lo que cuesta operar.

El alfa esperado procede de la calibración isotónica causal de `meta_rank` a retorno excedente
(`signal_calibration.parquet`), que solo usa cohortes ya cerradas. Mientras no hay suficientes
cohortes cerradas el valor es `NaN`, no cero: son cosas distintas y la cartera las trata distinto.
Un `NaN` nunca dispara una venta ni bloquea una compra —la regla es actuar solo ante evidencia
económica—, de modo que durante el arranque manda la ordenación y, en cuanto hay calibración, mandan
los umbrales.

En cada snapshot se marcan posiciones a mercado y:

- una posición sale si su alfa esperado cae por debajo de `exit_expected_alpha_bps`;
- un outsider desplaza a la peor posición solo si

  ```text
  alfa_esperado(outsider) − alfa_esperado(peor) > 2·(comisión + slippage) + rotation_edge_bps
  ```

  es decir, **la rotación paga su propio coste de ida y vuelta** antes de autorizarse. Este es el
  mecanismo que faltaba: con 877 % de rotación anual a 15 pb por operación, el coste drenaba en torno
  a 1,3 puntos porcentuales al año contra una ventaja bruta de unos 3,1;
- las posiciones dentro de la tolerancia mantienen sus unidades y el presupuesto restante se reparte
  respetando las relaciones objetivo.

`price_only_strictness_multiplier` es el **único** mecanismo de prudencia: en los snapshots que solo
traen precio nuevo (sin fundamentales publicados) baja el umbral de salida y sube la ventaja exigida
para rotar, de modo que la cartera no se mueve por ruido de precio sin confirmación fundamental.

### Política de efectivo

`cash_policy` decide si la cartera está siempre invertida al 100 % (`fully_invested`, referencia) o
si deja una plaza en efectivo cuando ninguna candidata supera el umbral (`opportunity_cash`, con
tope `max_cash_weight`). El efectivo **se remunera al 0 %**: es una cota inferior deliberadamente
conservadora, nunca aporta rentabilidad y solo puede ayudar evitando malas compras y ahorrando
costes. Si aun así mejora el alfa, la mejora no admite discusión.

La decisión de dejar efectivo se deriva **exclusivamente de la sección transversal** —del alfa
esperado de las candidatas— y nunca de una previsión sobre el mercado. Esa restricción no es
cosmética: derivarla de una vista de mercado convertiría el sistema en *market timing* encubierto y
la comparación contra el índice dejaría de ser limpia.

La política de efectivo es una **decisión de cartera, no de modelo**: no altera el Rank-IC, vive en
la etapa diagnóstica y se decide al final ejecutando ambas alternativas con el ganador predictivo ya
congelado. Cuál es mejor es un resultado del trabajo, no un supuesto previo.

### Sizing

```text
ratio = 1 + clip((alfa_esperado − mínimo_cartera) / (máximo_cartera − mínimo_cartera), 0, 1)
```

La posición con menor alfa esperado recibe ratio 1 y la de mayor alfa esperado ratio 2. A diferencia
de escalar por percentil, el peso responde a una magnitud económica estimada y no a la posición
relativa en un ranking. Comisión y slippage se aplican al nocional realmente operado.

Las alternativas de cartera cambian un solo eje contra la base. No hay cartesiano ni ganador
económico y sus resultados no cambian el modelo.

## 7. Perfiles

Balanced, growth, value, quality, momentum, contrarian, defensive y GARP transforman
determinísticamente los rankings del ganador. No reentrenan. Para cada perfil se guardan equity,
rentabilidad anual, benchmark, alfa, IR, drawdown, turnover, posiciones y órdenes.

La matriz principal coloca años en filas, perfiles en columnas y alfa contra SPY en las celdas.

## 8. Robustez

La batería posterior contiene semillas 7 y 2026, bootstrap móvil de 12 snapshots, intervalos al
90 % y 95 %, exclusión de eras, permutación transversal add-one, etiquetas barajadas, Rank-IC por
agente, estabilidad de pesos meta y carteras aleatorias PIT generales y emparejadas por riesgo.

**Nulo de carteras aleatorias.** Las carteras aleatorias juegan con las mismas reglas que el modelo:
exigen cobertura del año completo antes de computar un retorno anual, aplican la misma guarda contra
artefactos de datos y **pagan las mismas comisiones y slippage**. Sin esas tres condiciones el nulo
producía un percentil 95 de CAGR del 107 % anual —imposible para una cartera del S&P 500— y el único
contraste que el modelo aparentemente suspendía era en realidad un fallo del contraste.

**Estabilidad ante la semilla.** Cada agente entrena `SEED_ENSEMBLE = 5` réplicas que solo difieren
en la semilla y promedia sus scores. No es una variable científica a optimizar sino reducción de
varianza del estimador: el Rank-IC apenas dependía de la semilla (±0,001) pero el alfa de una cartera
concentrada llegaba a cambiar de signo, porque doce posiciones amplifican el ruido de inicialización
de LightGBM. El barrido de semillas se conserva como diagnóstico y `robustness.json` publica el rango
mínimo-mediana-máximo de alfa; si ese rango cruza cero, la conclusión económica no es estable y hay
que decirlo.

No produce una etiqueta automática de “aprende/no aprende”. El informe debe discutir evidencia a
favor, en contra, contradicciones y limitaciones.

## 8 bis. Atribución y confirmación fuera de muestra

`attribution.json` contiene la evidencia que separa «el sistema aprende» de «el sistema redescubrió
un factor conocido»:

1. **Regresión de factores.** El exceso de la cartera se regresa sobre carteras réplica de valor,
   momentum, baja volatilidad, calidad y crecimiento, construidas del propio universo como
   diferencial tercil superior menos tercil inferior y rebalanceadas en cada snapshot. El alfa es el
   intercepto y su significación se evalúa con errores **Newey-West** (12 retardos), porque los
   retornos solapados inflarían la significación con errores clásicos. No hay acceso a las series de
   Fama-French; tamaño e inversión se declaran **no replicables** con este panel en lugar de
   sustituirse por un sucedáneo no auditable.
2. **Rank-IC neutralizado** por las mismas características de estilo.
3. **Deflated Sharpe Ratio** (Bailey y López de Prado) sobre las series de IC candidatas: con
   decenas de evaluaciones, el mejor resultado es alto aunque ninguna configuración tenga capacidad
   real, y un p-valor sin corregir por multiplicidad no es defendible.
4. **Baselines deterministas** (GARP, momentum puro, calidad, valor): si el sistema no los supera, el
   aparato de aprendizaje no está justificado.
5. **Coeficiente de transferencia** y curva de alfa neto frente a rotación.

**Confirmación 2025–2026.** La era reservada no participó en ninguna decisión, así que su Rank-IC es
la única medida predictiva del trabajo libre de sesgo de selección. Se calcula **una sola vez, sobre
el ganador ya congelado**, y se publica salga lo que salga; el protocolo queda pre-registrado en
`docs/bitacora.md` antes de ejecutar el Study. Con horizonte de 12 meses y cadencia mensual las
cohortes contiguas comparten casi toda la etiqueta, de modo que el número de cohortes **no** es el
número de pruebas independientes: se reporta también el número efectivo de observaciones
independientes y se etiqueta como evidencia direccional del signo, no como contraste con potencia.

El sesgo que esto corrige es de **selección, no de lookahead**: el entrenamiento es walk-forward con
purga y ninguna predicción individual usa información futura, pero la *configuración* que las produce
se eligió por haber quedado mejor sobre esa misma serie 2015–2024.

## 9. Ejecución y recuperación

La API crea `study.json` y el run baseline antes de lanzar el worker. El proceso hijo actualiza
heartbeat y persiste cada run. Los eventos se escriben simultáneamente en terminal,
`events.jsonl` y Consola.

Estados: queued, running, succeeded, failed, cancelled e interrupted. Al arrancar, el servidor
marca como interrupted cualquier Study activo cuyo PID haya desaparecido. Reanudar conserva runs
finalizados y reinicia solo los incompletos. La caché solo se publica de forma completa.

## 10. Dashboard y API

El dashboard tiene Inicio y Resultados. Inicio configura el catálogo y muestra ocho tarjetas de
presupuesto en dos filas. Resultados presenta primero una tabla compacta de Studies sin scroll
interno. Al abrir un Study se muestra una cabecera estable con sus métricas globales y los botones
Runs, Consola, Robustez, Perfiles y la acción Pausar o Reanudar que corresponda a su estado. La
zona inferior se sustituye al cambiar de vista; no se refresca automáticamente para no desplazar
el scroll del usuario. La consola es la única excepción con scroll propio y muestra un viewport de
veinte líneas.

Al abrir un run aparece una segunda página con cabecera, explicación, tarjetas de resumen y vistas
de resumen, rendimiento, aprendizaje, cartera y acciones. Cartera permite elegir el snapshot,
mostrar posiciones y las órdenes ejecutadas en esa fecha. Acciones conserva esa misma fecha y permite
consultar situación en cartera, agentes, puntuaciones de parámetros, valores PIT y evolución de un
parámetro seleccionado. Un candidato descartado solo muestra su evidencia compacta. El run de
evidencia del ganador habilita los artefactos pesados.

Las métricas que representan tasas, retornos, pesos, drawdowns, turnover, alfa o Rank-IC se
presentan en porcentaje, aunque los artefactos mantengan su representación decimal para el cálculo.
Las vistas analíticas priorizan gráficos SVG con ejes, escalas y leyendas: curva de cartera frente a
SPY, Rank-IC temporal por agente, evolución de los pesos meta, alfa anual por perfil y comparación
de robustez entre modelo, semillas y placebos. Las tablas permanecen disponibles como respaldo
auditable.

API:

- `GET /api/catalog`
- `POST /api/studies/preflight`
- `POST /api/studies`
- `GET /api/studies`
- `GET /api/studies/{id}`
- `POST /api/studies/{id}/cancel`
- `POST /api/studies/{id}/pause`
- `POST /api/studies/{id}/resume`
- `GET /api/studies/{id}/runs/{run_id}`
- `GET /api/studies/{id}/events`
- `GET /api/studies/{id}/analysis/{view}` con `view` ∈ {winner, learning, robustness,
  attribution, profiles, portfolio-comparisons, portfolio, stocks, report}

No existen interfaces alternativas ni compatibilidad con protocolos anteriores.

## 11. Interpretación para el TFM

Un Study dev demuestra que el software funciona, no que exista señal económica. La evidencia del
TFM debe proceder de Studies completos, identificar configuración y hashes, separar selección de
estrés conocido y discutir multiplicidad, solapamiento temporal, costes, universo y sesgos
residuales.
