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

## Corrección metodológica v1 (superada por el protocolo v2)

Se implementaron los siguientes cambios:

- el resumen usado para seleccionar fases de modelo recalcula Rank-IC solo hasta 2024;
- 2025–2026 se separaba al final; desde v2 se denomina exclusivamente
  `known_stress_not_selection`, porque el periodo ya fue observado;
- una variante solo entra en greedy si supera el baseline y mantiene la fracción de IC positivos;
- las semillas se reservan para sensibilidad/robustez;
- los costes pasan a nueve escenarios de estrés que no cambian el ganador;
- los umbrales de selección se combinan explícitamente con poda o gating;
- la consola de Full study muestra ejes bloqueados, parámetros fijos y costes de estrés de forma transparente.

No se afirma que estas medidas eliminen todos los falsos positivos. Reducen grados de libertad y hacen visible qué parte de la evidencia se usó para seleccionar.

## Hipótesis y trazabilidad narrativa

Se añadió nombre e hipótesis a la pantalla Full study. La hipótesis se persiste explícitamente en `study_manifest.json`, se replica en `decision.json` y sirve de descripción de los runs del study. La razón es que un study sin hipótesis es reproducible técnicamente, pero difícil de defender académicamente: no queda claro qué esperaba validar ni cómo interpretar un resultado negativo.

## Resultados históricos y estado actual en aquel momento

Los studies anteriores a la corrección se conservan en `results/`. No se borran porque permiten reconstruir la evolución del proyecto y demostrar qué cambió. Sin embargo, sus cifras no deben usarse como evidencia final del protocolo vigente, especialmente si el ganador se benefició de costes seleccionados o de años que ahora son reserva.

Este protocolo quedó sustituido por v2 antes de considerarse evidencia final. El siguiente full
study válido debe ejecutar exactamente 48 evaluaciones, seleccionar únicamente con las tres eras
hasta 2024 y presentar 2025–2026 como estrés conocido, junto con bootstrap, placebos,
permutación inferencial, carteras aleatorias PIT y stresses económicos.

## Optimización de rendimiento sin cambiar resultados

Los studies eran lentos porque cada run reentrena `fechas × agentes × familias` modelos y un study encadena decenas de runs. Se aplicó un conjunto de optimizaciones de ingeniería con una regla estricta: **resultados numéricos idénticos**. Ninguna toca hipótesis, datos, etiquetas, modelos ni cartera; se validan con un oráculo (dataset sintético → `build_agent_scores` → diff exacto de scores, diagnósticos y pesos) más la suite y `ruff`.

Cambios: (1) la clave de caché y el `execution_hash` dejan de depender de la revisión Git global y pasan a una **huella de código por etapa** (clausura transitiva de imports de primera parte), de modo que editar el meta no invalida la caché de dataset/features; (2) la restauración de caché usa *hardlink* en vez de copiar; (3) `lgbm_n_jobs` es configurable (por defecto usa todos los núcleos), determinista y por tanto sin efecto en el resultado; (4) se vectorizaron `combine_agent_scores` y el filtrado por snapshot del backtest; (5) `find_completed_execution` lee el `execution_hash` indexado en el registro sin reabrir manifiestos y `_summary_for_run` memoiza por run.

La razón de documentarlo es dejar explícito el límite: acelerar es legítimo mientras sea demostrablemente equivalente; cualquier palanca que alterara resultados —tocar el bucle de reentreno, cachear fits entre escenarios o paralelizar escenarios por procesos— queda fuera hasta que se autorice de forma expresa. Detalle técnico en `doc.md` §24.4.

## El día de observación lo define el retardo de publicación, no un día de mes fijo

Al auditar un full study en marcha se detectó que barrer `execution_lag_days` producía escenarios **numéricamente idénticos** al baseline (mismos `agent_scores` bit a bit). La causa: el parámetro solo desplazaba el ancla del walk-forward unos días, pero la rejilla de snapshots caía en un día del mes fijo (`snapshot_day`), así que ninguna fecha se movía. Era un eje muerto con un nombre que sugería un efecto que no tenía.

La decisión (instrucción explícita del usuario, cambia resultados) fue rediseñar la rejilla: cada snapshot cae ahora en **`fin_de_periodo + execution_lag_days`**. Cerrado un mes o trimestre, los fundamentales tardan unos días en publicarse y la rejilla observa justo entonces. Con esto el retardo de publicación gobierna cuándo se miran los datos y `execution_lag_days` pasa a tener efecto real; `snapshot_day` se eliminó por completo (era una elección arbitraria de calendario). El point-in-time no se debilita: `_fundamentals_at` sigue leyendo solo lo que tiene `filed_date` anterior al snapshot; de hecho el criterio queda más fiel a la operativa real.

## Contratos separados: protocolo oficial cerrado y study manual exploratorio

El `study` manual conserva `MANUAL_STUDY_OPTIONS` para investigación libre. El full study ya no
recorta ni comparte ese barrido: ejecuta `OFFICIAL_STUDY_PROTOCOL`, una lista cerrada de 12
challengers de señal y una secuencia cerrada de 12 políticas de cartera. No hay greedy genérico,
reafinado plano de LightGBM ni expansión dinámica del catálogo. El presupuesto total es 48 y el
preflight aborta antes de crear el study si proyecta más de 10 fits caros o 5 GiB.

La cartera oficial compara estructuras mensuales/trimestrales y vintages únicamente como
challengers pre-registrados. Después decide secuencialmente sizing, overlay SPY y hurdle de costes;
ningún stress ni perfil puede alterar el ganador.

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

Se ejecutó el primer study oficial completo bajo el protocolo entonces vigente (`optimization-official`,
104 escenarios, 167 runs, estado `succeeded`). El análisis detallado está en
`docs/informe_resultados_study_1.md`; aquí se conserva el porqué del desenlace.

El barrido eligió una configuración **parsimoniosa**: un único LightGBM poco profundo (`max_depth 3`,
100 árboles), meta `stacked_oos`, poda estricta de features (8 por agente), etiqueta a 12 meses,
`execution_lag_days = 60` y `target_size = 10`. La Fase 1 rechazó por "no mejora estable" casi toda
ampliación de complejidad (régimen extendido, neutralización sectorial, ponderación por recencia,
features derivadas); la Fase 3 mostró un Rank-IC plano ante los hiperparámetros y estable a la semilla.
La lección es que **la señal es débil y no admite complejidad**: gana lo simple porque no hay margen.

El veredicto es **honesto y parcialmente negativo**, y se documenta como tal por disciplina del proyecto
(las configuraciones y periodos perdedores son evidencia). El Rank-IC OOS fue positivo dentro
de 2015–2024 —el antiguo placebo `p = 0` con solo tres permutaciones no es un p-valor válido; el
bootstrap sí dio un IC 95 % que no cruza cero y leave-one-year-out estable— pero
**se degrada a partir de 2024 y cae a Rank-IC negativo (−0.0095) en 2025–2026**, periodo que hoy
se clasifica como estrés conocido y no como reserva ciega, con
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

## 2026-07-24 — Protocolo confirmatorio v2: de Rank-IC a alfa

La auditoría del study 1 mostró tres problemas de proceso: 167 runs eran demasiado para una
confirmación, el stacking no aplicaba realmente su ventana trimestral y la cartera mensual
desalineaba una etiqueta de 12 meses con ~760 % de turnover anual. Además, el motor podía actualizar
pesos objetivo sin una orden equivalente y cada run copiaba paneles/features/atribuciones pesados.

Se sustituyó el ciclo oficial por un presupuesto determinista de 48 evaluaciones y máximo 10
walk-forwards caros. El study manual mantiene `MANUAL_STUDY_OPTIONS`; el oficial usa
`OFFICIAL_STUDY_PROTOCOL` con 12 challengers de señal y 12 políticas de cartera pre-registradas.
La selección queda limitada a tres eras hasta 2024. Los años 2025–2026 ya observados pasan a estrés
conocido y nunca vuelven a presentarse como holdout ciego.

El meta rolling/exponencial usa exclusivamente cohortes trimestrales cuya etiqueta ha cerrado, con
cap y shrinkage opcionales. La cartera incorpora lotes por vintage mantenidos 12 meses, núcleo SPY,
exposición activa causal y calibración isotónica point-in-time. La contabilidad deriva pesos por
mark-to-market, carga el coste inicial y exige una orden para todo cambio de holdings.

La publicación distingue candidatos, backtests y evidencia final. Los runs compactos referencian
al padre por hash/run, el workspace temporal se elimina solo tras publicar y la compactación
histórica es un comando separado cuyo modo por defecto es `dry-run`. La decisión puede terminar
explícitamente en `no_improvement`; el protocolo no está autorizado a elegir retrospectivamente la
alternativa con mayor alfa histórico.
