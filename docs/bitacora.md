# Bitácora

## 2026-07-25 · Limpieza integral: código muerto, legacy y features huérfanas

### Decisión

Auditoría función a función de todo el repositorio para eliminar código sin consumidor,
duplicación de la misma verdad en varios sitios y modos legacy con sesgo hardcodeado, en línea
con las reglas de `AGENTS.md`.

### Hallazgo científico principal

`module/modeling/catalog.py::FEATURE_CATALOG` declaraba los bloques `momentum_core` y
`momentum_trend` (factores `mom_acceleration`, `mom_reversal_1m`, `ma_price_vs_sma6`,
`ma_price_vs_sma12`, `ma_distance_to_high12`), presentes en todos los `feature_preset` reales y
barridos en `recommended_definition()`. Pero esas columnas solo se calculaban si
`settings.price_momentum_multi`/`moving_averages` eran `True`, y esas dos variables nunca
existieron en el catálogo cerrado de `module/studies/catalog.py`: no eran alcanzables desde
ningún Study real. El smoke de 5 tickers documentado en `docs/informe_resultados.md`
(`study-20260725-132255-c49da9ff`) se ejecutó **sin** estas columnas de momentum multi-horizonte.

### Cambios

- `price_momentum_multi`/`moving_averages` dejan de ser flags: sus artefactos
  (`add_price_momentum_multi`, `add_moving_averages`) se calculan siempre, igual que ya ocurría
  con `add_market_risk_liquidity`. `features_code_version` y `agents_fit_code_version` suben
  para invalidar cachés y materializaciones previas sin estas columnas.
- Eliminados `regime_extended`/`quality_growth_derived` y sus artefactos
  (`add_regime_extended`, `add_quality_growth_derived`): a diferencia de momentum, sus factores
  nunca estuvieron declarados en `FEATURE_CATALOG` — código huérfano de punta a punta.
- Eliminados los modos `meta_type="regime"`/`"rank_ic"` (sesgo `REGIME_TILT` hardcodeado a
  mano, nunca aprendido de datos) y `meta_history_mode="expanding"`/`"exponential"` (el runner
  siempre forzaba `"rolling"`). Solo quedan `"equal"` y `"stacked_oos"`, los dos únicos modos
  que el catálogo cerrado puede producir.
- Simplificada la infraestructura de ensemble multi-familia en `module/modeling/agents.py`
  (nunca se ejecutaba con más de una familia; el catálogo obliga a exactamente una).
- Eliminados módulos y funciones sin ningún consumidor: `module/studies/budget.py`,
  `module/evaluation/robustness.py`, `settings_payload`, `discard_summary_cache`,
  `append_ledger`, `cache_usage`, `prune_prepared`/`pinned_dataset_hashes`/`prepared_usage`,
  cuatro funciones huérfanas de `signal_diagnostics.py` (`summarize_tail`, `era_summary`,
  `moving_block_bootstrap_delta`, `holm_adjust`), endpoint `GET /api/studies/{id}/runs`.
- Consolidada duplicación de la misma verdad: `PROFILE_NAMES` (antes definido dos veces),
  `SELECTION_ERAS` (antes triplicado literalmente), `_link_or_copy` (antes duplicada byte a
  byte en `cache.py` y `datasets.py`, ahora en `module/common/utils.py`).
- Añadido botón Cancelar en el dashboard junto a Pausar/Reanudar (el endpoint ya existía sin
  cliente de UI).
- `CLAUDE.md` actualizado: ya no describe la arquitectura Exploratory→Confirmatory eliminada
  el mismo día.

### Validación

- Suite completa: 15 tests superados, ruff y `node --check` sin avisos.
- Smoke dirigido: dataset dev reconstruido con `ensure_prepared`; las seis columnas de
  momentum multi-horizonte/medias móviles aparecen con 100 % de cobertura (antes ausentes).
- `build_agent_scores` verificado de extremo a extremo con `meta_type="equal"` y
  `meta_type="stacked_oos"` sobre el dataset dev tras la simplificación de `meta.py`.

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
