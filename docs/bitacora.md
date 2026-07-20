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
