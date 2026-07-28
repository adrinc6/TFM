# Bitácora

## 2026-07-28 · Rediseño de la gestión de cartera: toda venta necesita un destino mejor

### Decisión

Reordenar la lógica de compras, ventas y rotaciones bajo un principio único —**una venta solo se
emite si el destino del dinero es mejor que la posición después de costes**— y subir el catálogo a
la versión 4 con `max_cash_weight` en `(0 %, 10 %, 25 %)`, por defecto 25 %. Rompe la
comparabilidad con los Studies de catálogo 3 (que no llegaron a ejecutarse).

### Defectos que motivan el cambio

Los tres tenían la misma raíz: las ventas se decidían sin mirar a dónde iba el dinero.

1. **`opportunity_cash` con todas las candidatas bajo el umbral** (el escenario para el que existe
   la política): `decide_orders` vendía la cartera entera y devolvía un objetivo vacío; el backtest
   interpretaba «vacío» como «mantener posiciones» y las resucitaba, pero ya había cobrado los
   costes de ventas que nunca ocurrieron. La cartera pagaba por replegarse sin replegarse.
2. **`fully_invested` con la curva calibrada bajo el umbral**: como el alfa esperado es monótono con
   el ranking, se vendían las 12 posiciones por umbral y el relleno obligatorio recompraba
   exactamente las mismas en el mismo snapshot: una ida y vuelta completa de la cartera para quedar
   igual. Contribuía al 877 % de rotación anual del diagnóstico.
3. **Concentración sin tope**: con una sola candidata admisible y tope de efectivo del 20 %, el
   80 % de la cartera acababa en una única acción, porque el suelo de inversión se repartía entre
   las admisibles que hubiera.

### Reglas resultantes

- **Rotación**: única vía de venta bajo `fully_invested`; exige ventaja superior al coste de ida y
  vuelta más `rotation_edge_bps` (sin cambios).
- **Venta a efectivo**: solo bajo `opportunity_cash`, con alfa bajo el umbral **y** respetando el
  suelo de diversificación `ceil((1 − max_cash_weight) · target_size)`, que garantiza a la vez el
  tope de efectivo y un mínimo de posiciones (12 plazas y tope 25 % → suelo de 9 → ninguna acción
  supera ~15 % del total).
- **Compra con histéresis**: entrar exige el umbral de salida más el coste de ida y vuelta de la
  propia operación; mantener exige solo el umbral. Elimina el churn de frontera.
- **Prudencia sin fundamentales**: el multiplicador ensancha ahora la banda entera (baja la salida,
  sube la entrada y la rotación).
- **Invariante nuevo**: con puntuaciones en la fecha, la cartera objetivo nunca queda vacía; el
  backtest lo verifica y falla ruidosamente si se viola.

Se documenta además una propiedad, no un defecto: con calibración en 20 ventiles y cartera
concentrada, el efectivo es casi binario y responde a la salud reciente de la señal; solo se vuelve
gradual con `target_size` 25 o 50.

### Correcciones menores del mismo día

- El nulo de carteras aleatorias aplicaba la guarda **mensual** de artefactos a retornos
  **anuales**, excluyendo del azar a ganadores legítimos de más del 100 % anual que el modelo sí
  puede cobrar: sesgaba el nulo a favor del modelo. Ahora usa la cota compuesta `(1+g)^12 − 1`.
- El Deflated Sharpe declara en su docstring la aproximación que hace (Sharpe de cartera contra
  dispersión de series de Rank-IC de candidatos) y que se lee como orden de magnitud del haircut.

### Validación

Suite completa (45 tests, incluidos cinco nuevos de este rediseño), ruff sin errores nuevos,
sintaxis JS y smoke dev end-to-end.

## 2026-07-28 · Pre-registro del protocolo de confirmación 2025–2026

### Decisión

Este bloque se escribe **antes** de ejecutar el Study con la regla de selección corregida, y fija
qué se medirá en la era reservada y cómo se leerá el resultado. La corrección de la puerta pareada
cambia el ganador con alta probabilidad, de modo que la confirmación solo es creíble si no podemos
elegir configuración sabiendo cómo se comporta en 2025–2026.

### Protocolo cerrado

1. **Qué se mide.** Rank-IC transversal por cohorte del `meta_rank` contra `forward_excess_return`
   en 2025–2026, y el alfa geométrico de la cartera en esa misma ventana.
2. **Cuándo.** Una sola vez, después de congelar `winner.json`. La evaluación vive fuera de todo
   bucle de decisión, en `module/research/attribution.py`, invocada por el runner tras escribir el
   ganador.
3. **Estadísticos.** Rank-IC medio, fracción de cohortes positivas, IC-IR, t de Newey-West con 12
   retardos y número efectivo de observaciones independientes.
4. **Cómo se lee.** Con horizonte de 12 meses y cadencia mensual, las cohortes cerradas disponibles
   son aproximadamente seis y comparten casi toda la etiqueta. El resultado se declara **evidencia
   direccional del signo**, no un contraste con potencia. Un Rank-IC medio positivo apoya que la
   ordenación aprendida se mantiene fuera de la ventana de selección; uno negativo o nulo se publica
   igualmente y obliga a matizar la conclusión principal.
5. **Compromiso.** El resultado se publica sea cual sea, sin repetir la evaluación con otra
   configuración ni ampliar la ventana a posteriori.

### Por qué

El entrenamiento es walk-forward con purga, así que no hay lookahead y ninguna cohorte está
contaminada. El sesgo que esto ataca es distinto: las 17 decisiones secuenciales se tomaron
comparando Rank-IC sobre las 117 cohortes de 2015–2024, y la cifra de portada procede de esa misma
serie. Es un estimador insesgado del Rank-IC de esa configuración en ese periodo, pero optimista
como estimador del Rank-IC futuro. La era reservada y el Deflated Sharpe atacan ese sesgo por dos
vías complementarias: la primera mide fuera de la muestra de selección, el segundo descuenta el
efecto de haber buscado.

## 2026-07-28 · Corrección de validez, alfa neto y limpieza

### Decisión

Corregir los defectos que invalidaban parte de la evidencia, sustituir los umbrales de cartera por
umbrales económicos y eliminar el código sin consumidor.

### Correcciones de validez y su efecto

| Defecto | Efecto medido |
|---|---|
| La puerta de no inferioridad penalizaba a los candidatos **superiores**: cuanto mayor la diferencia, más ancho el intervalo | `feature_preset` pasa de `core` (Rank-IC 0,0730) a `all` (0,0958), con ventaja pareada +0,0216. Ningún retador era elegible pese a dominar |
| `market_regime_feature` se decidió sobre una ventaja de 0,00112, por debajo del ruido | Pasa de `True` a `False` por simplicidad |
| `meta_method` se decidió sobre una ventaja de 0,00033 | Se mantiene, pero ahora registrado como empate técnico, no como victoria |
| Emparejamiento por cadena de fecha con rejillas desplazadas por `execution_lag_days` | Se empareja por periodo mensual; sin bloque completo común el resultado se marca no aplicable en vez de devolver `ci_low = 0,0` |
| `evaluation_key` no incluía el hash del dataset | La misma clave aparecía con CAGR 0,1468 y 0,1692. Ahora la clave separa datasets |
| Seis factores de precio se inyectaban en el agente momentum fuera de todo condicional | La ablación `fundamental` ya no recibe información de precio y mide lo que declara |
| Dos definiciones incompatibles de *information ratio* bajo el mismo nombre | Una sola, anualizada |
| CAGR, drawdown y alfa mezclaban la era reservada con la de selección | Métricas segmentadas: selección, confirmación y curva completa |
| `geometric_excess_return` era una resta de CAGR | Cociente de acumulados |
| Una posición sin precio se marcaba plana | Convención de exclusión tipo CRSP (−30 %) y liquidación contra efectivo |
| `subsample=0.8` sin `subsample_freq`: el bagging nunca se activaba | `subsample_freq=1` |
| Nulo de carteras aleatorias con percentil 95 de CAGR del 107 % | Exige cobertura anual completa, aplica la guarda de datos y paga los mismos costes |

### Catálogo de presets: solo `core` y `all`

Se retiran `fundamental` y `technical`. El motivo no es de resultado sino de diseño: ninguno de los
dos alimenta a los cinco agentes —`fundamental` deja sin features a momentum y risk, y `technical` a
quality, value y growth—, de modo que un Study que los eligiera no estaría respondiendo «qué
información necesita cada agente» sino «qué pasa si amputo parte de la arquitectura». Ambos presets
supervivientes mantienen los cinco agentes activos: `core` con su bloque esencial y `all` con toda la
profundidad disponible de cada especialidad.

Este cambio interactúa con la corrección de la puerta pareada: con el catálogo reducido, el ganador
de `feature_preset` pasa de `core` a `all`.

### Cartera: de percentiles a puntos básicos

`min_hold_percentile` y `rotation_edge_percentiles` desaparecen. Los sustituyen
`exit_expected_alpha_bps` y `rotation_edge_bps`, y una rotación solo se autoriza si la ventaja de
alfa esperado supera `2·(comisión + slippage) + margen`. `sizing_mode` pasa de `score_linear`
—anclado a un percentil arbitrario— a `alpha_proportional`. `CATALOG_VERSION` sube a 3 y se asume la
ruptura de comparabilidad con los Studies anteriores.

Se añaden `cash_policy` y `max_cash_weight` en la etapa **diagnóstica**: el efectivo no altera el
Rank-IC y por tanto no puede elegir modelo. Se ejecutan ambas políticas al final y se comparan en
`portfolio_comparison.parquet`. El efectivo se remunera al 0 %.

`target_size` se amplía a (8, 12, 16, 25, 50) para medir cuánta señal recupera la amplitud: por la
ley fundamental, un IC de 0,074 sobre ~250 valores implica un IR teórico en torno a 1,1 frente al
~0,18 realizado, y una cartera de 12 nombres con 877 % de rotación destruía cerca del 85 % de la
señal.

### Estabilidad

Cada agente entrena cinco réplicas que solo difieren en la semilla y promedia los scores. Motivo: el
Rank-IC variaba ±0,001 entre semillas pero el exceso sobre SPY cambiaba de signo (semilla 7: −0,51
pp; semilla 42: +3,11 pp). `robustness.json` publica ahora el rango de alfa entre semillas.

### Evidencia nueva

`module/research/attribution.py`: regresión sobre réplicas de factores con Newey-West, Rank-IC
neutralizado, Deflated Sharpe, baselines deterministas, cobertura del universo por año y coeficiente
de transferencia. Da consumidor a `signal_calibration.parquet`, `signal_health.parquet`,
`top_minus_bottom` y `module/data/baselines.py`, que se calculaban y no leía nadie.

### Eliminado

`_temporal_permutation_importance` (rama inalcanzable), `meta_equal_shrinkage` (siempre 0,0),
`meta_ic_lookback_quarters` (duplicaba `meta_history_quarters`), `ensure_directories`,
`agents_code_version`, `backtest_code_version`, `PortfolioState.cash`, `price_guard`,
`stock_sleeve_return`, `accounting_error`, las columnas `commission`/`slippage` duplicadas, el
fallback a `top_decile_spread`, el endpoint `GET /api/studies/{id}/runs` y la duplicación de
`AGENT_NAMES`. `module/data/ingest/` se conserva con entrada explícita `python main.py ingest`.


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
