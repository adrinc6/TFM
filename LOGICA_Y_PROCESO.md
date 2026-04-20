# Logica y Proceso

## Como funciona el sistema multi-agente de seleccion de acciones

## 1. Problema que resuelve

El sistema busca seleccionar una cartera reducida de acciones del S&P 500 de forma sistematica, repetible y auditable.

En lugar de depender de una sola senal, integra multiples dimensiones de decision para reducir fragilidad frente a cambios de mercado.

## 2. Idea central: comite de analistas virtuales

La arquitectura modela un comite:

- seis agentes especializados generan opiniones cuantificadas (scores),
- una capa meta combina esas opiniones,
- y se aplican reglas de robustez y riesgo antes de construir la cartera.

El enfoque evita un modelo monolitico unico y reparte la complejidad por especialidad.

## 3. Que analiza cada agente

- Fundamental: calidad de negocio y salud financiera.
- Valoracion: precio relativo frente a historia y comparables.
- Momentum: continuidad o cambio de tendencia de mercado.
- Bear: riesgo de deterioro fuerte (deuda, perdidas, estres operativo, etc.).
- Sentimiento: lectura de analistas, insiders y sorpresas de resultados.
- Rotacion sectorial: contexto top-down por sector frente al benchmark.

El BearAgent produce una medida de riesgo y puede activar una barrera dura para excluir casos extremos.

## 4. Como combina la capa meta

La capa meta recibe los scores base y genera un score final por ticker.

Adicionalmente, utiliza senales de consenso entre agentes y ajustes de robustez:

- contraccion de scores poco informativos,
- ajuste sectorial con confianza segun tamano de muestra,
- exclusiones por riesgo extremo.

En el camino alpha modernizado, la combinacion se alinea mas con ranking de alpha y control de riesgo por regimen de mercado.

## 5. Proceso temporal (walk-forward)

La evaluacion se ejecuta en ciclos temporales consecutivos:

1. entrenar con pasado disponible,
2. predecir sobre el siguiente periodo,
3. simular resultado,
4. avanzar la ventana y repetir.

Esto replica el uso real y evita entrenar con informacion futura.

Controles clave anti-leakage:

- snapshots as-of,
- lag de publicacion (`SNAPSHOT_LAG_DAYS`),
- OOF temporal para entrenar la capa meta,
- auditoria por fold.

## 6. Variantes operativas

El pipeline permite ajustar:

- frecuencia (trimestral o anual),
- universo (dinamico historico o manual),
- nivel de cache,
- modo de ponderacion de cartera,
- nivel de complejidad (camino clasico o alpha/risk-aware).

## 7. Salidas e interpretacion

La ejecucion produce:

- metricas globales y por fold,
- detalle de seleccion de tickers,
- series de retornos y equity,
- explicaciones por ticker,
- comparativa frente a benchmark y baselines.

Interpretacion recomendada:

1. verificar superacion consistente del benchmark,
2. revisar estabilidad (hit rate y drawdown),
3. validar que las explicaciones sean coherentes,
4. confirmar que supera baselines simples y aleatorios,
5. revisar sensibilidad por regimen/periodo.

## 8. Robustez y seguridad operativa

El sistema incluye:

- validacion de entradas y rutas,
- mecanismos de fallback controlados,
- diagnostico por componente,
- trazabilidad de decisiones,
- pruebas automatizadas de integridad temporal y politica de features.

## 9. Resultado practico

La herramienta no pretende reemplazar criterio profesional, sino ofrecer un motor cuantitativo con explicabilidad y control metodologico para apoyar decisiones de inversion con mejor disciplina de proceso.
