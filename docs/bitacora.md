# Bitácora

## 2026-07-25 · Reconstrucción a Model Study único

### Decisión

Se elimina el protocolo Exploratory → hipótesis → Confirmatory. La unidad científica pasa a ser
un único Model Study automático. Solo las fases predictivas seleccionan mediante Rank-IC.

### Motivo

El flujo anterior mezclaba entidades, multiplicaba rutas y podía dejar Studies fantasma. El Study
iniciado el 24 de julio terminó un fit caro pero falló antes del ledger al consultar
`signal_health_lookback_quarters`, campo ya eliminado. El error solo vivía en memoria y el proceso
desapareció sin reconciliación.

### Cambios

- Catálogo v2 con variables predictivas y cartera informativa.
- Tres meta-agentes: equal, rolling free y rolling 10–50 %.
- Persistencia de Study, runs y eventos antes del cálculo.
- Worker hijo por Study, heartbeat, cancelación, interrupción y reanudación.
- API y dashboard reducidos a Inicio y Resultados.
- Cartera 100 % acciones; SPY solo benchmark.
- Robustez y ocho perfiles posteriores al ganador.
- Eliminación de Exploratory, hipótesis, Confirmatory y modelos promovidos.
- Corrección de la referencia al campo eliminado.

### Validación final

- Suite crítica: 15 tests superados.
- Ruff, compilación Python y sintaxis JavaScript superados.
- Auditoría UTF-8 sin secuencias de mojibake en fuentes.
- Smoke real corregido: `study-20260725-132255-c49da9ff`, estado `succeeded`.
- 27 runs físicos finalizados, 53 eventos persistentes y 872.775 bytes de evidencia.
- Reanudación del Study finalizado: cero runs añadidos y mismos identificadores.
- Worker finalizado y `worker_pid = null`.

### Incidencias descubiertas por los smokes

1. La primera ejecución falló al serializar valores de cartera de tipo texto y número en una
   columna Parquet. Se normalizaron ambos valores como JSON.
2. El primer smoke técnicamente exitoso produjo scores constantes: 50 observaciones mínimas por
   hoja impedían dividir árboles con 65 filas. No se aceptó como validación. El modo dev limita
   ahora el mínimo a 5; el smoke repetido produjo 23 cohortes y Rank-IC no degenerado.
3. La lista de Studies fallaba cuando un run aún tenía `result = null`. La consulta trata ahora
   correctamente los runs creados antes de calcular.
4. La concentración meta mezclaba la columna de cohortes realizadas con los pesos. Se sustituyó por
   HHI de pesos por fecha y turnover medio de media norma L1.
5. Se añadió vigilancia del PID padre: si termina abruptamente el dashboard, el worker se marca
   `interrupted` y se detiene por sí mismo.
6. El dashboard pasó de tablas aisladas a visualización analítica: porcentajes en escala humana,
   equity con ejes y leyenda, evolución multicolor de Rank-IC y pesos, perfiles por año y barras de
   robustez para semillas, placebos y agentes.
7. Los ejes de las curvas se calculan ahora por métrica. En particular, los pesos se limitan al
   intervalo válido 0–100 % y se ajustan al rango observado; cada punto ofrece fecha, serie y valor
   exacto al situar el cursor. La configuración de cada run se presenta en tarjetas temáticas en
   lugar de una tabla plana.
8. La navegación contextual vuelve bajo Resultados. Los gráficos de líneas usan cursor vertical y
   una leyenda flotante de todas las series en la fecha más cercana, sin puntos visibles. Performance
   usa años como marcas del eje X, divisores verticales secundarios y ticks enteros en equity.
   Portfolio y Stocks comparten snapshot; Portfolio integra las órdenes del día y Stocks permite
   consultar cartera, agentes, parámetros PIT, puntuaciones de factores y evolución temporal.

Las cifras del smoke sirven para validar el flujo, no como evidencia económica o científica del TFM.
