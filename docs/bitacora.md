# Bitácora de decisiones y cambios metodológicos

Esta bitácora complementa `doc.md`: conserva el motivo de decisiones que no debe perderse al redactar el LaTeX. No reemplaza los manifiestos ni resultados; explica por qué la arquitectura actual es distinta de versiones previas.

## Principios mantenidos durante todo el proyecto

1. La bolsa es el entorno de validación, no el fin del proyecto.
2. La señal se mide por Rank-IC fuera de muestra y no por rentabilidad aislada.
3. Un dato fundamental solo existe cuando se publicó; no cuando cerró el trimestre fiscal.
4. Las configuraciones perdedoras son evidencia y se guardan.
5. Los costes y las semillas no deben convertirse en grados de libertad para mejorar retrospectivamente un resultado.

## Primera arquitectura: dataset, features y tres estilos

La primera versión separó descarga, panel point-in-time, factores, agentes, meta-agente y cartera. Sus tres agentes originales eran calidad, valor y momentum. Esta separación ya permitía atribuir un score a una hipótesis económica, pero dejaba fuera crecimiento y riesgo como componentes explícitos.

La lección de esta etapa fue que una cartera rentable no bastaba: el informe debía diferenciar el aprendizaje del modelo de las decisiones de construcción de cartera. De ahí surgió el uso sistemático de Rank-IC y la separación de targets futuros.

## Incorporación de métricas existentes

Se inspeccionaron las series descargadas de Finnhub y OHLCV. Se comprobó que era posible derivar, sin descargar fuentes nuevas, ROA, ROTC, márgenes, rotaciones, liquidez, deuda, P/FCF, EV/revenue, yields, aceleración, estabilidad, riesgo y liquidez de mercado.

La decisión fue no añadir campos arbitrarios ni tratar ratios negativos como “baratos”. Los múltiplos no interpretables se mantienen como ausencias. Los extremos se tratan mediante winsorización configurable previa al ranking transversal.

## Catálogo de bloques y cinco agentes

Se creó `module/modeling/catalog.py` como fuente declarativa de factores. Los bloques sustituyeron listas dispersas de columnas y permiten tres usos: organización económica, explicabilidad y ablaciones.

Se añadieron `growth` y `risk`. La decisión no fue asumir que ambos mejoran la señal, sino poder retirar cada uno y medir su contribución incremental. La misma lógica se aplicó a bloques: baseline completo, baseline menos un bloque y conjunto básico.

Un problema detectado fue que las listas históricas de features seguían entrando aunque el bloque se desactivase. Eso hacía que una ablación fuese solo nominal. Se eliminó esa duplicación: el catálogo controla el conjunto real de features de cada agente.

## Diversidad de modelos y ensembles

LightGBM se mantuvo como modelo de árboles principal. Se añadió Elastic Net para una referencia lineal regularizada y CatBoost para una alternativa de árboles. La intención es medir diversidad de error, no imponer que tres modelos sean necesariamente mejores.

Al principio, los modos `rank_ic_weighted` y `stacked_oos` estaban expuestos antes de tener lógica completa. Se completaron con pesos de familias derivados de predicciones OOS cerradas y stacking Ridge no negativo. CatBoost también se alineó con el objetivo configurado, evitando comparar regresión bajo una etiqueta de ranking o clasificación.

## Selección de factores y diagnóstico

Se decidió que los pesos de factores no debían introducirse manualmente. Los modelos aprenden relaciones internas; el laboratorio mide si estas generalizan.

La poda OOS usa cobertura y Rank-IC de cohortes cerradas. Cuando se solicita un umbral de permutación, la importancia se calcula temporalmente entrenando antes de validar y permutando dentro de la cohorte de validación. La decisión de no calcularla siempre responde a coste computacional: sería desperdicio en `diagnostic_only`, donde no puede cambiar ninguna selección.

`block_gated` agrupa evidencia por bloque. Su propósito es responder si una familia económica completa merece entrar, no afirmar causalidad financiera fuerte.

## Eliminación de fuentes y parámetros ficticios

Se habían dejado flags para SEC detallado, dividendos/recompras, estimaciones, short interest y acciones en circulación. Como no existía descarga PIT histórica y validada, los flags no añadían datos ni experimentos reales. Se eliminaron de configuración, caché, fingerprints y consola.

También se fijaron antigüedad máxima de precio, mínimo de filas, vida media de recencia y tamaño mínimo sectorial. Son salvaguardas de calidad y estabilidad, no hipótesis de alpha; optimizarlas aumentaba grados de libertad sin una justificación económica suficiente.

## Problemas detectados al auditar el full study

La revisión de 2026-07-20 encontró cuatro riesgos importantes:

1. La documentación declaraba 2025–2026 como reserva, pero la selección consultaba un resumen de Rank-IC agregado con toda la muestra.
2. Comisión y slippage se barrían durante cartera; un selector económico tendería a elegir el coste más bajo.
3. La semilla se ofrecía como eje de selección; eso permite elegir una realización afortunada de un algoritmo estocástico.
4. Umbrales de poda se probaban aisladamente aun cuando el baseline `diagnostic_only` no los usaba, creando runs sin efecto.

Estas observaciones invalidaban presentar el full study anterior como cierre definitivo, aunque sus runs sigan siendo reproducibles y útiles como contexto histórico.

## Corrección metodológica vigente

Se implementaron los siguientes cambios:

- el resumen usado para seleccionar fases de modelo recalcula Rank-IC solo hasta 2024;
- 2025–2026 se guarda en `reserved_validation` y se consulta al final;
- una variante solo entra en greedy si supera el baseline y mantiene la fracción de IC positivos;
- las semillas se reservan para sensibilidad/robustez;
- los costes pasan a nueve escenarios de estrés que no cambian el ganador;
- los umbrales de selección se combinan explícitamente con poda o gating;
- la consola de Full study muestra ejes bloqueados, parámetros fijos y costes de estrés de forma transparente.

No se afirma que estas medidas eliminen todos los falsos positivos. Reducen grados de libertad y hacen visible qué parte de la evidencia se usó para seleccionar.

## Hipótesis y trazabilidad narrativa

Se añadió nombre e hipótesis a la pantalla Full study. La hipótesis se persiste explícitamente en `study_manifest.json`, se replica en `decision.json` y sirve de descripción de los runs del study. La razón es que un study sin hipótesis es reproducible técnicamente, pero difícil de defender académicamente: no queda claro qué esperaba validar ni cómo interpretar un resultado negativo.

## Resultados históricos y estado actual

Los studies anteriores a la corrección se conservan en `results/`. No se borran porque permiten reconstruir la evolución del proyecto y demostrar qué cambió. Sin embargo, sus cifras no deben usarse como evidencia final del protocolo vigente, especialmente si el ganador se benefició de costes seleccionados o de años que ahora son reserva.

El siguiente full study completado con esta metodología será el primer candidato a resultado final. Antes de escribir una conclusión cuantitativa deben revisarse: configuración elegida, Rank-IC hasta 2024, validación 2025–2026, bootstrap, LOYO, placebo, cartera aleatoria, estrés de costes y ablaciones.

## Optimización de rendimiento sin cambiar resultados

Los studies eran lentos porque cada run reentrena `fechas × agentes × familias` modelos y un study encadena decenas de runs. Se aplicó un conjunto de optimizaciones de ingeniería con una regla estricta: **resultados numéricos idénticos**. Ninguna toca hipótesis, datos, etiquetas, modelos ni cartera; se validan con un oráculo (dataset sintético → `build_agent_scores` → diff exacto de scores, diagnósticos y pesos) más la suite y `ruff`.

Cambios: (1) la clave de caché y el `execution_hash` dejan de depender de la revisión Git global y pasan a una **huella de código por etapa** (clausura transitiva de imports de primera parte), de modo que editar el meta no invalida la caché de dataset/features; (2) la restauración de caché usa *hardlink* en vez de copiar; (3) `lgbm_n_jobs` es configurable (por defecto usa todos los núcleos), determinista y por tanto sin efecto en el resultado; (4) se vectorizaron `combine_agent_scores` y el filtrado por snapshot del backtest; (5) `find_completed_execution` lee el `execution_hash` indexado en el registro sin reabrir manifiestos y `_summary_for_run` memoiza por run.

La razón de documentarlo es dejar explícito el límite: acelerar es legítimo mientras sea demostrablemente equivalente; cualquier palanca que alterara resultados —tocar el bucle de reentreno, cachear fits entre escenarios o paralelizar escenarios por procesos— queda fuera hasta que se autorice de forma expresa. Detalle técnico en `doc.md` §24.4.

## El día de observación lo define el retardo de publicación, no un día de mes fijo

Al auditar un full study en marcha se detectó que barrer `execution_lag_days` producía escenarios **numéricamente idénticos** al baseline (mismos `agent_scores` bit a bit). La causa: el parámetro solo desplazaba el ancla del walk-forward unos días, pero la rejilla de snapshots caía en un día del mes fijo (`snapshot_day`), así que ninguna fecha se movía. Era un eje muerto con un nombre que sugería un efecto que no tenía.

La decisión (instrucción explícita del usuario, cambia resultados) fue rediseñar la rejilla: cada snapshot cae ahora en **`fin_de_periodo + execution_lag_days`**. Cerrado un mes o trimestre, los fundamentales tardan unos días en publicarse y la rejilla observa justo entonces. Con esto el retardo de publicación gobierna cuándo se miran los datos y `execution_lag_days` pasa a tener efecto real; `snapshot_day` se eliminó por completo (era una elección arbitraria de calendario). El point-in-time no se debilita: `_fundamentals_at` sigue leyendo solo lo que tiene `filed_date` anterior al snapshot; de hecho el criterio queda más fiel a la operativa real.

## El full study es inteligente; el study manual es exploratorio

El `study` manual conserva todos los valores de cada eje (exploración libre en la consola). El `full_study` automático encadena ~un centenar de reentrenos caros, así que su barrido (`FULL_STUDY_OPTIONS`) recorta la densidad de niveles donde la curva es suave y los niveles contiguos rara vez cambian el ganador (p. ej. `train_lookback_years` 6→3, `lgbm_max_depth` 5→3), apoyándose en que la fase de afinado de hiperparámetros ya reafina lr/n_estimators/min_child_samples sobre el ganador. No se elimina ningún **eje** "por si no aporta": eso lo decide el propio estudio midiendo su contribución incremental; solo se baja densidad donde el solapamiento es evidente.

La construcción de cartera se simplificó después a tamaños fijos (`target_size`: 5, 8, 10, 12 y 15). La cartera compra el top-N del `meta_rank`, expulsa una tenencia que no supera el percentil de mantenimiento y rota solo con ventaja suficiente. Los pesos dependen del ranking efectivo, pero ninguna posición puede pesar más del doble que la menor.

## Aceleración del walk-forward por multihilo, no por procesos

El cuello de botella es la etapa de agentes (reentreno `fechas × agentes × familias`). Se aceleró manteniendo **resultados idénticos** (oráculo): (1) `train`/`target`/pesos de recencia se preparan una vez por fecha de reentreno en vez de una vez por agente/familia —eran invariantes al agente—; (2) CatBoost recibe `thread_count` ligado al mismo setting que LightGBM, de modo que ambas familias usan todos los núcleos. Se descartó paralelizar escenarios por procesos: el portátil tiene poca RAM libre y varios procesos causarían swap; además las fechas de reentreno no son independientes en el modo por defecto (`rank_ic_weighted` usa predicciones de fechas previas). CatBoost y LightGBM son deterministas frente al número de hilos (verificado con diferencia exacta 0.0), así que el multihilo acelera sin cambiar resultados.


## 2026-07-21 — Reciclaje por familia dentro de la etapa de agentes

La etapa de agentes (el ~90% del tiempo de un run: reentreno `fechas × agentes × familias`) se
descompone internamente en dos claves de caché. `agents_fit` guarda el ajuste walk-forward de una
sola familia de modelo y no depende del meta ni de las demás familias; `agents` combina esos
ajustes (ensemble intra-agente y meta-agente) y solo su clave incluye `meta_type`,
`meta_ic_lookback_quarters` e `intra_agent_ensemble_mode`. Un barrido que solo cambia el meta
reutiliza intactos todos los ajustes, y una ablación de familia reutiliza las familias restantes,
sin recalcular nada del centenar largo de `fit`.

La recombinación reproduce exactamente el ensemble intra-agente previo (mismo orden de familias,
mismo promedio de rangos), de modo que los resultados numéricos son idénticos: el cambio solo
evita recomputar ajustes ya realizados. Verificado con igualdad bit-a-bit de las predicciones y
con el reciclaje efectivo al cambiar el meta y al ablacionar una familia; la suite completa pasa.


## 2026-07-24 — Study oficial: resultados y veredicto

Se ejecutó el primer study oficial completo bajo el protocolo vigente (`optimization-official`,
104 escenarios, 167 runs, estado `succeeded`). El análisis detallado está en
`docs/informe_resultados_study.md`; aquí se conserva el porqué del desenlace.

El barrido eligió una configuración **parsimoniosa**: un único LightGBM poco profundo (`max_depth 3`,
100 árboles), meta `stacked_oos`, poda estricta de features (8 por agente), etiqueta a 12 meses,
`execution_lag_days = 60` y `target_size = 10`. La Fase 1 rechazó por "no mejora estable" casi toda
ampliación de complejidad (régimen extendido, neutralización sectorial, ponderación por recencia,
features derivadas); la Fase 3 mostró un Rank-IC plano ante los hiperparámetros y estable a la semilla.
La lección es que **la señal es débil y no admite complejidad**: gana lo simple porque no hay margen.

El veredicto es **honesto y parcialmente negativo**, y se documenta como tal por disciplina del proyecto
(las configuraciones y periodos perdedores son evidencia). El Rank-IC OOS del finalista es real dentro
de 2015–2024 —placebo `p = 0`, bootstrap con IC 95 % que no cruza cero, leave-one-year-out estable— pero
**se degrada a partir de 2024 y cae a Rank-IC negativo (−0.0095) en la reserva ciega 2025–2026**, con
alfa anual negativo tres años seguidos. Además, la rentabilidad no supera de forma convincente a carteras
aleatorias (percentil 18). El sistema ordena activos mejor que el ruido, pero ese margen no se traduce en
alfa económico y no generaliza fuera del periodo de selección.

Punto metodológico que motivó el diseño: la reserva 2025–2026 **no es "no entrenar" esos años** —el
walk-forward avanza con normalidad— sino apartarlos de la *decisión* para medir el sesgo de selección al
comparar 104 escenarios. Precisamente por existir, la reserva evitó desplegar con confianza injustificada
un modelo que llevaba fallando desde 2024. La operativa de producción (entrenar hasta hoy y reformar la
cartera periódicamente) es un flujo distinto que solo debe activarse una vez el sistema supere la
validación fuera de muestra, cosa que con esta configuración aún no ocurre. La siguiente línea de trabajo
es diagnosticar la caída 2024–2026 (¿régimen, decaimiento de value/quality, o sesgo residual?) con los
diagnósticos ya generados, sin reejecutar el study.
