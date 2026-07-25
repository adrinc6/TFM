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

La identidad del dataset incluye fuentes, fechas, universo, cadencia, horizonte, lag y versiones
de transformación. Una identidad igual reutiliza `data/prepared/<dataset_hash>/`; una identidad
distinta crea otra materialización. Los Studies guardan referencias, no copias.

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

Los presets core, fundamental, technical y all seleccionan bloques cerrados. También pueden
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

Elegibilidad:

- observaciones suficientes;
- ninguna era disponible con Rank-IC inferior a −0,02;
- límite inferior del bootstrap pareado al 90 % de ΔRank-IC superior a −0,01.

Orden:

1. Mayor mediana del Rank-IC entre 2015–2018, 2019–2021 y 2022–2024.
2. Diferencias inferiores a 0,002 son empate.
3. Mayor Rank-IC medio.
4. Mayor fracción positiva.
5. Menor variabilidad.
6. Menor complejidad.
7. Conservar incumbent.

Holm, spread de cola, alfa e IR se muestran como diagnósticos y no alteran esta decisión.

## 6. Cartera dinámica

La cartera está siempre invertida en acciones. SPY es únicamente benchmark. No hay vintages,
holding mínimo, núcleo pasivo ni efectivo estructural.

En cada snapshot se marcan posiciones a mercado. Una acción sale al caer bajo el percentil mínimo.
Un outsider desplaza a la peor posición solo si supera la ventaja mínima. Las posiciones dentro de
la tolerancia mantienen sus unidades. El presupuesto restante se reparte respetando las relaciones
objetivo originales.

Sizing lineal:

```text
ratio = 1 + clip((meta_rank - mínimo_efectivo) / (1 - mínimo_efectivo), 0, 1)
```

El umbral recibe ratio 1 y un rank 1 recibe ratio 2. Dos ranks próximos reciben pesos próximos.
Comisión y slippage se aplican al nocional realmente operado.

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
agente, estabilidad de pesos meta, estrés 2025–2026 y carteras aleatorias PIT generales y
emparejadas por riesgo.

No produce una etiqueta automática de “aprende/no aprende”. El informe debe discutir evidencia a
favor, en contra, contradicciones y limitaciones.

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
- `GET /api/studies/{id}/runs`
- `GET /api/studies/{id}/runs/{run_id}`
- `GET /api/studies/{id}/events`
- `GET /api/studies/{id}/analysis/{view}`

No existen interfaces alternativas ni compatibilidad con protocolos anteriores.

## 11. Interpretación para el TFM

Un Study dev demuestra que el software funciona, no que exista señal económica. La evidencia del
TFM debe proceder de Studies completos, identificar configuración y hashes, separar selección de
estrés conocido y discutir multiplicidad, solapamiento temporal, costes, universo y sesgos
residuales.
