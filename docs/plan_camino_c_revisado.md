# Plan revisado: motor LightGBM orientado a ranking, diagnóstico temporal y optimización por etapas

## Contexto

El estudio de palancas de la Parte B concluyó que el enfoque **Ridge lineal + regresión del
exceso de retorno** alcanza un techo práctico de rank-IC cercano a cero: la mejor configuración
obtuvo aproximadamente **+0,0015**. El pipeline point-in-time, el walk-forward y los tests de
fuga constituyen una base sólida; el problema observado es la falta de señal predictiva estable
con el modelo y el objetivo actuales.

Esta tanda probará el **Camino C** —modelo no lineal y objetivo mejor alineado con el ranking—,
aprovechando que la cobertura y el número de empresas por corte transversal son mucho mayores en
los años recientes. El objetivo es determinar si el cambio de motor aporta una mejora consistente
frente a Ridge y frente a baselines deterministas, y comprobar en qué período existe suficiente
información para que el sistema aprenda de forma útil.

## Objetivo de la tanda

1. Corregir primero las métricas y huellas experimentales para que midan el sistema realmente
   utilizado por la cartera.
2. Comparar Ridge y LightGBM con objetivos alineados con la ordenación transversal.
3. Medir el aprendizaje por año, cobertura y régimen para decidir si el período completo o una
   ventana reciente de 10-15 años representa mejor el sistema actual.
4. Avanzar a meta-agente contextual y optimización de cartera únicamente si la señal base mejora
   de forma clara, estable y estadísticamente defendible.
5. Ejecutar el sistema final sobre todo el período elegido y reportar tanto ese resultado como la
   sensibilidad a usar el histórico completo.

## Principios metodológicos obligatorios

### 1. Walk-forward continuo hasta los últimos datos

- El sistema entrena y evalúa en walk-forward desde 2000 hasta la última cohorte con etiqueta
  completamente observada.
- **2026** se incluye solo para cohortes cuyo `label_end_date` ya esté disponible; las cohortes
  incompletas no entran en rank-IC ni en comparaciones predictivas.
- Cada reentrenamiento utiliza exclusivamente targets ya realizados:
  `label_end_date <= fecha_de_reentrenamiento`.
- Se conserva la ventana móvil de entrenamiento, de modo que una predicción reciente aprende de
  los años inmediatamente anteriores y no de información futura.
- Se reportan por separado el histórico completo y ventanas recientes de 15 y 10 años.
- El período principal puede ser reciente si presenta mucha más cobertura, más empresas por
  cohorte y aprendizaje más estable. Esta decisión debe justificarse conjuntamente por cobertura
  y comportamiento predictivo, no solo por el mayor número puntual de rank-IC.

### 2. Diagnóstico y selección del período útil

`execution_year` determina principalmente desde cuándo se evalúa el sistema; con ventana móvil,
no crea por sí solo un modelo estructuralmente distinto en los años posteriores. Aun así, es
legítimo centrar el trabajo en los años con datos suficientes para responder a la utilidad actual
del modelo. Por tanto:

- Primero se genera una tabla anual con rank-IC, fracción positiva, cobertura media, número de
  empresas por cohorte y número de cohortes evaluables.
- Después se comparan períodos con inicio 2000, 2010, 2014 y 2016, o fechas equivalentes sugeridas
  por saltos objetivos de cobertura.
- Se admite elegir una ventana reciente de 10-15 años si ofrece mejor cobertura y una mejora
  persistente del aprendizaje.
- No se elegirá un año aislado porque maximice el resultado. Se buscará una meseta temporal:
  varios años consecutivos, suficiente número de cohortes y resultados que no dependan de uno o
  dos ejercicios excepcionales.
- El informe mostrará siempre también el resultado desde 2000 para que el lector vea cuánto cambia
  la conclusión al concentrarse en el período reciente.
- Si la fecha de inicio se decide después de observar los resultados, se declarará expresamente
  como **selección temporal basada en cobertura y diagnóstico**, con el riesgo de sesgo que ello
  implica. Esto no invalida el análisis, pero limita la fuerza causal de la conclusión.

### 3. Separar selección predictiva y diseño de cartera

- La primera puerta se decide solo por la calidad del ranking OOS del **score final negociable**.
- No se optimizan simultáneamente motor, meta-agente y reglas de cartera.
- `beat_rate`, alfa y drawdown se reportan, pero no rescatan un modelo cuyo rank-IC sea
  indistinguible de cero.
- La cartera se optimiza únicamente después de congelar una señal predictiva que haya superado
  la primera puerta.

## Decisiones revisadas

| Tema | Decisión revisada |
|---|---|
| Motor | Comparar Ridge y LightGBM. LightGBM será pequeño y regularizado. |
| Objetivo principal | **Ranking transversal**. Preferencia: `LGBMRanker` agrupado por snapshot; alternativa operativa: `LGBMRegressor` sobre el percentil transversal del retorno. |
| Cuartiles | No serán el objetivo por defecto. Se admite top-vs-bottom como ablación secundaria porque descarta el 50 % de las filas y no optimiza directamente el rank-IC de todo el universo. |
| Meta inicial | Promedio equiponderado y meta por rank-IC histórico. No implementar aún un LightGBM adicional para predecir pesos. |
| Meta contextual | Solo se prueba si los agentes LightGBM ya muestran señal robusta. Preferencia por un stacker lineal regularizado y walk-forward antes que otro árbol. |
| Período | Comparar histórico completo y ventanas recientes. Puede ganar una ventana de 10-15 años si mejora cobertura y estabilidad, no solo el IC puntual. |
| Selección | Rank-IC del **meta-score final**, fracción de fechas positivas, ICIR e intervalo de confianza/diferencia pareada frente a Ridge. |
| Evaluación | Walk-forward continuo hasta la última etiqueta disponible, sin holdout separado. |
| Cartera | Se estudia después de congelar modelo y meta; rejilla pequeña. |
| Resultados | No borrar evidencia sin conservar resúmenes, manifiestos, configuraciones y hashes. |

## Cambios de código

### 1. `module/meta.py`: diagnosticar lo que realmente compra la cartera

El diagnóstico actual calcula rank-IC para `quality`, `momentum` y `value`, pero la selección y
la cartera consumen `meta_score`/`meta_rank`. Debe corregirse antes de medir modelos nuevos.

- Añadir al `rank_ic_diagnostics.parquet` filas para:
  - cada agente individual;
  - promedio equiponderado;
  - meta por rank-IC histórico;
  - cualquier meta alternativo probado;
  - **meta seleccionado/final**.
- Incluir campos como `score_type` o `agent="meta_final"` para distinguirlos sin ambigüedad.
- Calcular el rank-IC del meta usando el score disponible en cada snapshot frente al retorno
  futuro de esa misma cohorte.
- La métrica principal del escenario debe proceder exclusivamente de `meta_final`, no de la
  media de los tres agentes.
- Mantener diagnósticos individuales como explicación y ablación, no como sustituto de la
  métrica final.
- Verificar que cambiar `META_TYPE` cambia los diagnósticos y, cuando corresponda, la huella del
  escenario.

### 2. `module/agents.py`: motores y objetivos

Implementar una API explícita, por ejemplo:

- `MODEL_TYPE = "ridge" | "lightgbm"`
- `OBJECTIVE = "rank_regression" | "ranking" | "regression" | "quartile"`

Orden de prioridad:

1. **Ridge + etiqueta rank**: control principal ya conocido.
2. **LightGBMRegressor + etiqueta rank**: comparación más limpia; cambia el motor manteniendo
   el objetivo.
3. **LGBMRanker** agrupado por `snapshot_date`: opción preferida si se integra de forma clara y
   comprobable.
4. **LightGBM + cuartiles**: ablación secundaria, no base automática.

Requisitos:

- Ridge conserva imputer y scaler.
- LightGBM no necesita scaler y puede usar su tratamiento nativo de NA.
- Los grupos del ranker deben corresponder exactamente a cada cohorte/snapshot y conservar un
  orden determinista.
- La etiqueta de ranking se genera solo con las filas de entrenamiento.
- Cuartiles, si se conservan, se calculan dentro de cada snapshot; se entrena solo con extremos,
  pero se puntúa a todo el universo.
- Guardar `feature_importances_` para LightGBM y coeficientes para Ridge con un nombre de artefacto
  neutral, por ejemplo `model_feature_attribution.parquet`.
- Mantener semilla fija durante la comparación principal y añadir después una prueba de
  sensibilidad con varias semillas para el candidato ganador.
- El nombre del `run_id` debe reflejar el motor (`ridge-...` o `lightgbm-...`).

### 3. `module/experiments.py`: huellas completas y selección correcta

Incorporar a `FINGERPRINT_FIELDS["agents"]` y a las etapas dependientes todos los campos que
alteren resultados:

- `model_type`;
- `objective`;
- hiperparámetros LightGBM;
- `meta_type` y parámetros del meta;
- semilla;
- límites del período evaluado y ventana temporal seleccionada;
- cualquier opción de ranking o clasificación.

Actualizar también manifiestos y configuración del escenario. El manifiesto debe registrar:

- versión de LightGBM y scikit-learn;
- motor, objetivo e hiperparámetros;
- semilla;
- período de entrenamiento/evaluación y última etiqueta observable;
- hashes de inputs;
- número de cohortes y filas de entrenamiento/evaluación.

La selección de la primera etapa debe usar solo métricas predictivas de `meta_final`. No incluir
rentabilidad ni drawdown en esta selección inicial. Debe poder calcular métricas tanto para toda
la historia como para cada ventana reciente candidata sin reentrenar con futuro.

### 4. `module/backtest.py`: corregir métricas económicas

- Sustituir la actual media geométrica de `1 + alfa anual` por métricas estándar:
  - CAGR de la cartera;
  - CAGR del benchmark;
  - diferencia de CAGR;
  - exceso geométrico relativo:
    `geomean((1 + retorno_cartera) / (1 + retorno_benchmark)) - 1`.
- Mantener la alfa aritmética media anual como estadístico separado y nombrado correctamente.
- Revisar el cálculo anual para que incluya el retorno desde el último snapshot del año anterior
  hasta el primero del año actual.
- Reportar máximo drawdown de toda la curva y por año.
- `backtest_summary.json` debe tomar el rank-IC del `meta_final`, no el promedio de agentes.

### 5. Baselines deterministas

Los artefactos de GARP y momentum ya existen, pero deben recibir el mismo tratamiento económico
que el modelo final:

- construir una cartera con score GARP determinista;
- construir una cartera con momentum determinista;
- aplicar las mismas fechas, costes y reglas de cartera;
- comparar su rank-IC y rendimiento con Ridge, LightGBM y SPY.

El resultado final debe poder concluir si LightGBM aporta valor frente a reglas simples, no solo
si mejora a un Ridge débil.

### 6. Meta contextual, solo tras superar la primera puerta

No implementar de entrada un `LGBMRegressor` que prediga pesos. Si los agentes base muestran
señal suficiente:

1. Construir un stacker walk-forward que reciba los tres scores/ranks y contexto observable.
2. Empezar por Ridge/ElasticNet regularizado.
3. Contexto mínimo y predefinido:
   - régimen bull/bear conocido en la fecha;
   - volatilidad reciente del SPY;
   - dispersión transversal de los scores;
   - desacuerdo entre agentes.
4. Entrenar solo con targets cuyo `label_end_date <= fecha_de_reentrenamiento`.
5. Compararlo de forma pareada contra equiponderado y meta rank-IC en los mismos años y cohortes
   del walk-forward.
6. Adoptarlo solo si mejora de forma consistente, no por una diferencia puntual mínima.

La salvaguarda se evalúa walk-forward sobre el mismo conjunto de períodos comparables y se adopta
solo cuando la mejora aparece en varios años y ventanas, no únicamente en el agregado final.

## Rejilla revisada y secuencial

### Etapa A: motor y objetivo

Rejilla pequeña, sobre el walk-forward completo y con desglose por ventanas temporales:

1. `ridge + rank_regression`.
2. `lightgbm + rank_regression`.
3. `lightgbm + ranking`, si el ranker queda correctamente implementado.
4. `lightgbm + quartile`, como ablación.

Todos usan el mismo meta simple, las mismas features, ventana, fechas y cartera neutral de
diagnóstico. Esta etapa debe aislar el efecto del motor/objetivo.

### Etapa B: regularización del ganador LightGBM

Solo si LightGBM supera a Ridge. Rejilla dirigida y pequeña, por ejemplo:

- profundidad efectiva: 2, 3 y 5;
- `n_estimators`: 100 y 300 compensado con `learning_rate`;
- `min_child_samples`: dos valores conservadores;
- `subsample` y `colsample_bytree` fijados inicialmente.

No hacer un producto cartesiano grande. Elegir de 6 a 8 configuraciones justificadas. Evaluar
la estabilidad del ganador con varias semillas antes de congelarlo.

### Etapa C: meta-agente

Solo si la Etapa B confirma señal:

1. equiponderado;
2. rank-IC histórico;
3. régimen con reglas simples;
4. stacker lineal contextual.

No incluir LightGBM contextual salvo evidencia clara de que el stacker lineal queda corto y haya
suficientes eras de entrenamiento.

### Etapa D: cartera

Con señal y meta congelados:

- cartera 5-10;
- cartera 3-7;
- cartera 8-15;
- como máximo dos variantes justificadas de umbral de entrada/rotación.

La configuración de cartera se decide por utilidad económica y robustez dentro del período
seleccionado. Debe reportarse también la sensibilidad para evitar presentar un único top-N
afortunado y se contrastará con el mismo diseño sobre el histórico completo.

## Métricas predictivas y significancia

Para cada score candidato se reportará:

- rank-IC medio OOS del `meta_final`;
- mediana del rank-IC;
- desviación e ICIR;
- fracción de cohortes con IC positivo;
- rank-IC por año y por régimen;
- número de empresas por cohorte;
- intervalo de confianza del IC medio mediante bootstrap por bloques temporales;
- diferencia pareada de rank-IC frente a Ridge por fecha, con intervalo de confianza;
- sensibilidad a semillas para el candidato LightGBM final.

No tratar las cohortes como independientes a nivel de fila. El remuestreo debe hacerse por bloques
temporales suficientemente largos para respetar el solapamiento del horizonte y de las ventanas.

## Puertas de decisión

### Puerta 1: ¿el cambio de motor aporta señal?

LightGBM avanza solo si, en el walk-forward evaluable:

- mejora a Ridge en rank-IC medio del `meta_final`;
- la diferencia pareada no depende de uno o dos años;
- la fracción de cohortes positivas mejora o al menos no se deteriora materialmente;
- el intervalo de confianza y el análisis por bloques no indican que la mejora sea puramente
  accidental;
- supera o complementa de forma creíble los baselines deterministas.

No se fija un umbral arbitrario como 0,03 para aprobar o suspender, pero una mejora de unas pocas
diezmilésimas sin estabilidad no justifica añadir complejidad.

Si no supera esta puerta, se detiene el barrido y se documenta que el Camino C tampoco produjo
señal explotable.

### Puerta 2: ¿el meta contextual añade valor?

Solo avanza si mejora de forma pareada al equiponderado y al meta rank-IC en los mismos períodos
comparables. Si no, se conserva el meta más simple.

### Puerta 3: elección del período y evaluación final

Tras escoger motor, objetivo, hiperparámetros, meta y cartera:

- seleccionar el período principal usando conjuntamente cobertura, tamaño transversal y
  estabilidad del rank-IC;
- exigir un mínimo recomendado de 10 años evaluables;
- reportar el resultado principal en ese período;
- reportar en paralelo el histórico completo desde 2000 y las ventanas de 10 y 15 años;
- mostrar qué años explican la diferencia entre períodos;
- declarar si la mejora reciente es gradual y estable o si depende de pocos episodios.

## Tests requeridos

### Agentes

- La etiqueta rank se calcula dentro de cada snapshot y solo con entrenamiento.
- El ranker respeta los grupos y no mezcla cohortes.
- Cuartiles, si se mantienen, etiquetan extremos correctamente, excluyen el centro solo de train
  y puntúan todo el universo.
- LightGBM produce scores finitos y atribuciones de features.
- Ridge reproduce el comportamiento anterior.
- Mutar features o targets futuros no cambia predicciones pasadas.
- Cambiar motor, objetivo, semilla o hiperparámetros cambia huellas y `run_id`.

### Meta

- Se genera diagnóstico para `meta_final`.
- El resumen usa `meta_final`, no la media de agentes.
- Cambiar el meta cambia el score y el diagnóstico correspondiente.
- Mutar el futuro no cambia pesos ni predicciones pasadas.
- El stacker, si se implementa, solo usa etiquetas ya realizadas.

### Experimentos y métricas

- No se reutilizan artefactos entre configuraciones incompatibles.
- El filtrado por período no altera predicciones ya calculadas ni introduce targets futuros.
- La selección temporal exige el mínimo de años, cohortes y cobertura configurado.
- Las métricas CAGR y exceso geométrico cuadran con casos manuales pequeños.
- Las métricas anuales incluyen correctamente el salto entre años.
- El resumen de rank-IC coincide con las filas `meta_final` de diagnósticos.

Ejecutar `pytest tests/ -q` tras cada bloque lógico.

## Ejecución en orden

1. Determinar la última cohorte con etiqueta completamente observable y excluir las posteriores de
   las métricas predictivas.
2. Corregir diagnóstico del meta-score final, métricas económicas, manifiestos y huellas.
3. Añadir tests de regresión para esas correcciones.
4. Implementar LightGBM con etiqueta rank; integrar ranker si resulta claro y mantenible.
5. Mantener cuartiles solo como ablación secundaria.
6. Ejecutar humo `RUN_SCOPE=dev` de principio a fin.
7. Ejecutar Etapa A sobre todo el walk-forward disponible.
8. Aplicar Puerta 1 y anotar resultados por año, bloque y modelo en la bitácora.
9. Solo si procede, ejecutar regularización, sensibilidad a semillas y Etapa C.
10. Elegir justificadamente el período principal —histórico completo o ventana reciente de al
    menos 10 años— y congelar la señal antes de la rejilla de cartera.
11. Preservar de los resultados anteriores al menos resúmenes, manifiestos, selección, bitácora y
    hashes; después se pueden eliminar artefactos pesados regenerables.
12. Ejecutar la configuración final sobre el período principal y calcular en paralelo la
    sensibilidad desde 2000 y en ventanas recientes.
13. Generar el informe final separando claramente resultado principal, histórico completo y
    sensibilidad temporal.

## Informe final

Debe incluir, como mínimo:

- Ridge frente a cada variante LightGBM;
- baselines deterministas GARP y momentum;
- rank-IC del score final, no promedio de agentes;
- intervalo de confianza y diferencia pareada frente a Ridge;
- estabilidad anual y por régimen;
- resultado del período principal frente al histórico completo y ventanas de 10/15 años;
- CAGR cartera y SPY, diferencia de CAGR y exceso geométrico relativo;
- alfa aritmética media anual, porcentaje de años que baten, costes, turnover y drawdown máximo;
- limitaciones de cobertura, supervivencia, dependencia temporal y búsqueda de modelos;
- relación entre complejidad añadida y mejora obtenida.

## Punto de cierre honesto

LightGBM puede capturar interacciones que Ridge no representa, pero no puede crear información que
no existe en los datos. Si el cambio no mejora de forma estable el rank-IC del score final, el
resultado correcto es cerrar el Camino C y presentar el hallazgo negativo con toda la evidencia.

Si solo mejora en años recientes con mayor cobertura, se podrá centrar la conclusión en ese período,
explicando que el sistema necesita una densidad mínima de datos y que el inicio temporal fue elegido
tras un diagnóstico empírico. Si la mejora desaparece al quitar uno o dos años, no se considerará
estable.

El éxito de esta tanda no consiste en obtener obligatoriamente alfa positivo, sino en producir una
respuesta que siga siendo creíble después de mostrar de forma transparente cómo influyen la
cobertura y la selección del período temporal.
