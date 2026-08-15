# Gestión de cartera

Documento de referencia de **todas** las decisiones que toma la cartera: qué variables existen, en
qué orden se aplican, qué casuísticas pueden darse y qué pasa en cada una.

**Cómo usar este documento**: si quieres cambiar un comportamiento, anótalo aquí (en la sección
correspondiente o al final, en «Cambios pedidos») y se traslada al código. Cada regla indica el
fichero y la función donde vive, para que el cambio sea localizable.

Toda la lógica de decisión está en una única función: `decide_orders`, en
[module/evaluation/portfolio.py](../module/evaluation/portfolio.py). La contabilidad (precios,
costes, turnover) está en [module/evaluation/backtest.py](../module/evaluation/backtest.py).

---

## 1. Principio rector

> **Una venta solo se emite si el destino del dinero es mejor que la posición, después de costes.**

Hay exactamente dos destinos posibles, y cada uno tiene su regla:

| Destino | Regla | Requiere |
|---|---|---|
| **Otra acción** (rotación) | La ventaja de alfa esperado debe superar el coste de ida y vuelta más un margen | `rotation_edge_bps` |
| **Efectivo** | El alfa esperado debe caer bajo el umbral de salida y la plaza debe poder quedar vacía | `max_cash_weight > 0` |

**Dos excepciones** no se comparan contra ningún coste, porque no deciden si una operación es
rentable sino si la acción **pertenece al universo invertible**. Son mandatos, no decisiones
económicas: `missing_current_score` y `coverage_percentile_floor` (sección 4).

### Las tres magnitudes y su papel

| Magnitud | Qué es | Para qué se usa |
|---|---|---|
| `meta_rank` / percentil | Posición relativa en el ranking (0-100) | Orden de preferencia y desempates |
| `expected_excess_return` (alfa) | Retorno excedente anual esperado, en pb | Decidir si una operación se paga a sí misma |
| Comisión + slippage | Coste real de operar | El listón que toda operación debe superar |

El alfa sale de la curva causal percentil → retorno real anualizado. **Es `NaN` mientras no haya
cohortes cerradas suficientes**: durante el arranque manda la ordenación, y en cuanto hay evidencia
mandan los umbrales. Un `NaN` de alfa nunca dispara una venta ni bloquea una compra.

---

## 2. Las 12 variables del catálogo

Todas son `predictive=False`: se ejecutan **después** de congelar el ganador y **nunca** influyen en
la selección del modelo (regla 4 de CLAUDE.md). Solo sirven para explicar y estabilizar.

| Variable | Valores | Actual | Qué controla |
|---|---|---|---|
| `target_size` | 5, 8, 12, 16, 25, 50 | 12 | Nº de posiciones simultáneas |
| `exit_expected_alpha_bps` | 0, 100, 250 | 100 | Alfa mínimo (pb/año) para conservar |
| `rotation_edge_bps` | 25, 50, 100 | 50 | Ventaja exigida **sobre el coste** para sustituir |
| `max_cash_weight` | 0, 0.10, 0.25 | 0.25 | Tope de efectivo. **0 = siempre invertido** |
| `minimum_holding_period` | none, quarter, half, full | none | Meses mínimos antes de poder vender |
| `coverage_percentile_floor` | 0, 60, 80 | 0 | Percentil bajo el cual se vende entera |
| `rebalance_drift_tolerance` | 0, 0.10, 0.25, 0.40 | 0.25 | Desviación relativa mínima para reajustar peso |
| `price_only_strictness_multiplier` | 1.0, 1.5, 2.0 | 1.5 | Endurece umbrales sin fundamentales nuevos |
| `price_only_sell_only` | False, True | False | Sin fundamentales: vender sí, comprar no |
| `sizing_mode` | equal, alpha_proportional | alpha_proportional | Reparto de pesos |
| `commission_bps` | 0, 5, 10 | 5 | Comisión por operación |
| `slippage_bps` | 5, 10, 20 | 10 | Impacto de mercado por operación |

Definidas en [module/studies/catalog.py](../module/studies/catalog.py); llegan a `Settings` vía
`settings_from_values` en [module/studies/config.py](../module/studies/config.py).

### Los tres umbrales derivados

Se calculan una vez por snapshot, **todos en base anual**:

```
coste_ida_y_vuelta = anualizar(2 × (comisión + slippage), horizonte)
umbral_salida      = exit_expected_alpha_bps / dureza
umbral_entrada     = (exit_expected_alpha_bps + coste_ida_y_vuelta) × dureza
umbral_rotación    = coste_ida_y_vuelta + rotation_edge_bps × dureza
```

`dureza` = 1,0 en snapshots con fundamentales nuevos; `price_only_strictness_multiplier` en el resto.

**Por qué entrada > salida (histéresis)**: sin esa banda, una acción oscilando alrededor del umbral
se compraría y vendería en snapshots consecutivos, pagando costes con ventaja esperada nula.

**Por qué se anualiza el coste y no el alfa**: comisión y slippage se pagan **una vez por operación**,
no cada año. Con horizonte de 12 meses la conversión es la identidad; con horizontes más cortos el
coste anualizado es mayor, porque la misma comisión se repite más veces al año.

> Ejemplo con la configuración actual (comisión 5 + slippage 10, horizonte 12 m):
> `coste = 2 × 15 = 30 pb`. `umbral_salida = 100 pb`. `umbral_entrada = 130 pb`.
> `umbral_rotación = 30 + 50 = 80 pb`.

---

## 3. Orden de decisión en cada snapshot

El orden **importa**: cada paso condiciona los siguientes. Una acción vendida entra en `removed` y
**no puede recomprarse en ese mismo snapshot**.

```
0.  Preparación: ranking, percentiles, alfas, umbrales, protegidas por tenencia
1.  Venta forzada por pérdida de cobertura      (missing_current_score)
1-bis. Venta por suelo de cobertura             (below_coverage_percentile)
2.  Venta a efectivo por umbral                 (expected_alpha_below_exit)
--- a partir de aquí, todo bloqueado si price_only_sell_only y no hay fundamentales ---
3.  Compras nuevas con histéresis               (initial_fill)
4.  Relleno obligatorio hasta el suelo          (fully_invested_fill / cash_floor_fill)
5.  Rotación: outsider desplaza a la peor       (displaced_by_net_edge / net_edge_over_worst)
6.  Pesos, tolerancia de deriva y órdenes       (rebalance)
```

### Paso 0 — Preparación

Se descartan las acciones sin `meta_rank`; el resto se ordena por percentil descendente. Se marcan
como **protegidas** las posiciones que aún no cumplen `minimum_holding_period`.

### Paso 1 — Pérdida de cobertura (`missing_current_score`)

Una posición **sin `meta_rank` en el snapshot actual** ya no pertenece al universo evaluable: se
vende siempre. **Ignora el mínimo de tenencia** y no puede recomprarse ese snapshot.

> **Ejemplo real** (bitácora 2026-07-30): MAC salió del S&P 500 el 2019-12-10 pero seguía en cartera
> sin fila puntuable. El bucle de rotación la elegía como «peor posición», no podía calcular su
> ventaja y **detenía toda rotación**, incluida la de ACN, que estaba en p4,96 con −9,01 % anual.
> Por eso esta regla existe y por eso ignora el mínimo de tenencia.

### Paso 1-bis — Suelo de cobertura (`below_coverage_percentile`)

Una posición que cae por debajo de `coverage_percentile_floor` se vende entera **siempre**, **aunque
su alfa no active ninguna otra regla**. La venta no se frena por el suelo de diversificación: qué
pasa con la plaza liberada (recompra o efectivo) lo decide después el relleno obligatorio del paso 4,
no esta regla. Ver sección 4: es la variable más delicada del bloque.

### Paso 2 — Venta a efectivo (`expected_alpha_below_exit`)

Solo existe si **`max_cash_weight > 0`** o si `price_only_sell_only` está bloqueando compras.
Requisitos: alfa por debajo de `umbral_salida`, posición no protegida, y que la plaza pueda quedar
vacía sin bajar del suelo de diversificación.

> **Por qué no existe con `max_cash_weight = 0`**: vender por umbral con la obligación de recomprar
> en el mismo snapshot pagaría una ida y vuelta para quedar exactamente igual.

### Paso 3 — Compras nuevas (`initial_fill`)

Se recorre el ranking de mejor a peor hasta llenar `target_size`, exigiendo `umbral_entrada`.

### Paso 4 — Relleno obligatorio (`fully_invested_fill` / `cash_floor_fill`)

Se completa hasta el **suelo de diversificación** con las mejores por ranking, **sin aplicar ningún
umbral**. Con `max_cash_weight = 0` ese suelo es `target_size`, así que la cartera queda siempre
llena. El motivo distingue ambos casos para que la auditoría diga por qué se compró.

> Este paso es la razón de que varias reglas «no hagan lo que parece» con tope 0: vendas lo que
> vendas, el relleno vuelve a llenar la cartera en el mismo snapshot.

### Paso 5 — Rotación

```
mientras la cartera esté llena:
    outsider = la mejor candidata NO en cartera
    peor     = la posición con menor alfa entre las NO protegidas
    si alfa(outsider) − alfa(peor) >= umbral_rotación:  intercambiar
    si no: parar
```

**Por qué `parar` y no `seguir`**: `outsider` es la mejor disponible y `peor` la peor desplazable, así
que si ese par no supera el umbral, ningún otro par lo hará. Además, `seguir` sería un bucle infinito
porque ninguno de los dos cambia. *(No es un bug: está verificado.)*

**Caso límite**: si a alguno de los dos le falta calibración de alfa, la ventaja es indefinida y la
rotación se detiene. Es conservador por diseño: una rotación que no puede justificarse
económicamente no se hace.

### Paso 6 — Pesos y rebalanceo

1. **Fracción invertida**: cuánto del capital se invierte (ver casuística C en la sección 5).
2. **Reparto**: `equal` da el mismo peso a todas; `alpha_proportional` escala con el alfa, con tope
   **2:1** entre la mejor y la peor posición.
3. **Tolerancia de deriva**: si un peso se desvía menos de `rebalance_drift_tolerance` (relativo) de
   su objetivo, la posición se **congela** y el presupuesto se reparte entre las demás.

> Ejemplo: peso actual 10 %, objetivo 12 %. Desviación relativa = 20 %. Con tolerancia 0.25 (25 %)
> **no se opera**; con 0.10 sí. Evita pagar comisiones por ajustes cosméticos.

---

## 4. El suelo de cobertura, en detalle

Es la variable añadida por petición y la que más conviene entender antes de tocarla.

### Qué hace

Vende entera toda posición cuyo percentil actual caiga por debajo del suelo, **una vez cumplido el
mínimo de tenencia**, sin mirar el alfa ni el coste.

### Por qué es legítima siendo un percentil

El proyecto tiene una doctrina explícita: *los umbrales que compiten contra el coste de operar deben
ser económicos, porque un percentil no dice cuánto se espera ganar*. Esta regla **no la viola**
porque no compite contra ningún coste: no decide si una operación es rentable, sino si la acción
**sigue perteneciendo al universo invertible**. Es la generalización de `missing_current_score`
(percentil *ausente* → percentil *demasiado bajo*). Es un mandato, igual que el mínimo de tenencia.

### Diferencia clave con `missing_current_score`

| | Mínimo de tenencia |
|---|---|
| `missing_current_score` | **Lo ignora** — la posición ya no es evaluable |
| `below_coverage_percentile` | **Lo respeta** — la posición sigue siendo evaluable |

### La consecuencia que hay que tener presente

La venta por suelo de cobertura es **incondicional**: se vende toda posición bajo el suelo, sin
excepción, y esta regla no decide qué pasa después con la plaza liberada — eso es siempre cosa del
paso 4 (relleno obligatorio).

Con **`max_cash_weight = 0`**, el paso 4 recompra la mejor disponible en el mismo snapshot sin
aplicar umbrales. Es decir: la venta por cobertura **fuerza una rotación que el bucle económico
habría rechazado** por no cubrir su coste, y por tanto **aumenta la rotación**.

Con **`max_cash_weight > 0`**, el paso 4 solo rellena hasta el suelo de diversificación, así que la
plaza puede quedarse en efectivo y la regla sí de-arriesga de verdad. La configuración actual usa
0.25, así que estás en este caso.

### Calibración de los valores

Con `target_size = 8` —el valor de la cartera adoptada— sobre un universo de ~500 acciones, las
posiciones se compran en torno a **p98,4 o mejor**. Por tanto:

| Valor | Equivale a | Efecto esperado |
|---|---|---|
| **0** | Desactivado | Comportamiento de referencia |
| **60** | Fuera del top 200 | Caída de ~190 puestos desde la compra |
| **80** | Fuera del top 100 | Caída de ~90 puestos; **el más agresivo** |

> **Nota honesta**: las patologías documentadas en la bitácora ocurrieron en **p4-p5**, no en p60-p80.
> Un suelo en p80 puede disparar ventas con frecuencia y **subir** el turnover. El barrido
> diagnóstico lo medirá contra el ganador congelado; si el efecto no compensa, bajar a valores más
> cercanos a p10-p25 es un cambio de una línea en el catálogo.

**Efecto medido** (previo al cambio del 2026-08-12: la venta pasó a ser incondicional, ver §7) sobre
un panel de prueba pequeño (4 tickers, 12 meses, `target_size=2`), suficiente para ver la dirección
aunque no para extrapolar magnitudes:

| Suelo | Tope efectivo | Ventas por cobertura | Turnover |
|---|---|---|---|
| 0 | cualquiera | 0 | 1,00 |
| 60 | cualquiera | 0 | 1,00 |
| **80** | **0** | **11** | **11,94** |
| 80 | 0,25 | 0 | 1,00 |

Confirmaba entonces dos propiedades: con **tope 0** el suelo dispara rotación forzada (el relleno
recompra en el mismo snapshot), y con **tope > 0** el suelo de diversificación **bloqueaba** las
ventas por completo. Esta segunda propiedad ya no es cierta: desde el 2026-08-12 la venta por suelo
de cobertura es incondicional en ambos casos, y solo cambia lo que hace el paso 4 con la plaza
liberada (recompra con tope 0, posible efectivo con tope > 0). El panel necesita remedirse contra
esta nueva lógica.

---

## 5. Casuísticas: qué pasa cuando…

### A. Una posición pierde su puntuación (sale del índice, deja de cotizar)

→ Se vende en el paso 1, siempre, incluso bajo mínimo de tenencia. No se recompra ese snapshot.
Si no hay reemplazo admisible, queda efectivo.

### B. Todas las posiciones caen bajo el umbral y no hay nada mejor

→ Con **tope 0**: no se emite ninguna orden. Vender la cartera entera para recomprarla igual sería
una ida y vuelta completa para quedar en el mismo sitio.
→ Con **tope > 0**: se venden las peores a efectivo hasta el suelo de diversificación.

### C. Hay menos candidatas que plazas

Este caso distingue **dos situaciones que se confundían** (corregido el 2026-08-04):

| Situación | Qué pasa | Por qué |
|---|---|---|
| **Universo escaso** (no hay con qué llenar) | El capital se reparte entre las plazas ocupadas; el tope decide cuánto se retiene | Las plazas ausentes no existen |
| **Compra bloqueada** (`price_only_sell_only`) | El hueco **queda en efectivo**, no se reparte | Concentrar sería actuar sin información nueva |

> El bug corregido: con 9 de 12 plazas, la antigua política «siempre invertida» invertía el 75 % y
> dejaba un 25 % en efectivo **que ninguna variable declaraba**, mientras que el tope 0 invertía el
> 100 %. La política llamada «siempre invertida» retenía más efectivo que la política de efectivo.

### D. Un snapshot solo trae precio nuevo, sin fundamentales

→ `price_only_strictness_multiplier` endurece los tres umbrales (se opera menos en ambos sentidos).
→ Si además `price_only_sell_only = True`: se puede **vender** pero **no comprar** nada (ni compra
nueva, ni relleno, ni rotación). El efectivo resultante es transitorio, sin tope, y **no se reparte
entre las supervivientes**.

### E. Una posición está protegida por el mínimo de tenencia

→ No puede venderse por caída de alfa, ni por rotación, ni por suelo de cobertura.
→ **Sí** puede venderse por `missing_current_score`.
→ **Sí** puede ajustarse su peso por rebalanceo (no es una venta de la posición).

### F. El alfa aún no está calibrado (arranque del backtest)

→ Manda la ordenación por ranking. Ningún `NaN` dispara una venta ni bloquea una compra.
→ La rotación **se detiene** si a algún lado le falta calibración.

### G. Los pesos no suman 1

Es legítimo y significa efectivo. Puede ocurrir por: tope de efectivo activo, `price_only_sell_only`
bloqueando compras, o universo más pequeño que `target_size`. El efectivo **se remunera al 0 %**:
nunca aporta rentabilidad, solo puede ayudar evitando malas compras.

---

## 6. Rotación (turnover): diagnóstico actual

`annualized_turnover` es **bruto/dos vías**: suma compras y ventas. Un valor de 3,59 significa que se
opera 3,6 veces el patrimonio al año ≈ **1,8 rotaciones completas**.

### De dónde viene (ganador `study-20260803-201234-b4d7a8d8`)

| Motivo | % del turnover |
|---|---|
| **Rotación** (`net_edge_over_worst` + `displaced_by_net_edge`) | **56 %** |
| Rebalanceo de pesos | 20 % |
| Compras iniciales | 13 % |
| Ventas a efectivo | 10 % |

**Causa raíz**: se re-decide **mensualmente** (`snapshot_step_months = 1`) sobre una señal a **12
meses** (`target_horizon_months = 12`), con `minimum_holding_period = "none"`.

### Palancas medidas sobre el ganador congelado

| Cambio | Turnover | IR | Alfa geom. |
|---|---|---|---|
| *(actual)* | 3,59 | 0,269 | 1,62 % |
| `minimum_holding_period` → `half_horizon` | **2,28** | **0,398** | **2,58 %** |
| `rotation_edge_bps` → 100 | 2,78 | 0,315 | 2,06 % |
| `price_only_sell_only` → true | 2,53 | 0,362 | 2,24 % |
| `sizing_mode` → equal | 2,99 | 0,308 | 1,93 % |
| `minimum_holding_period` → `full_horizon` | 1,67 | −0,050 | −0,67 % |

**`half_horizon` domina**: baja el turnover un 36 % y **a la vez** sube IR y alfa. `full_horizon`
muestra que el compromiso no es monótono — retener demasiado destruye el alfa.

> El valor actual `"none"` viene del `recommended` del catálogo, **no de una elección medida**.
> Cambiarlo a `half_horizon` es una línea en `catalog.py` y exige re-ejecutar el study.

### Nueva palanca sobre el 20 % de rebalanceo de pesos

Se añadió **`rebalance_drift_tolerance` = 0.40** al catálogo (2026-08-12), como opción disponible
para el barrido de escenarios. `rebalance_drift_tolerance` no es predictiva, así que probar 0.40 no
exige re-ejecutar el study: solo repetir el barrido de cartera sobre el ganador congelado. El valor
**actual** sigue en 0.25 hasta que el barrido mida si 0.40 mejora la relación turnover/alfa; no se
asume aquí que lo haga. Congela más posiciones con desviaciones moderadas del peso objetivo, a costa
de tolerar carteras algo menos ajustadas al objetivo teórico entre rebalanceos.

### Palanca estructural, fuera del bloque de cartera

`snapshot_step_months` 1 → 3 reduciría el turnover ~3× por construcción, pero es una variable
**predictiva**: cambiarla re-ejecuta la selección entera y produce otro ganador. No es un ajuste
posterior de cartera.

---

## 7. Cambios pedidos

> Anota aquí lo que quieras modificar. Formato sugerido: qué regla, qué comportamiento nuevo, y por
> qué. Se traslada al código, con sus tests y su entrada de bitácora.

### Resueltos (2026-08-12)

1. **Suelo de cobertura siempre vende.** El paso 1-bis (`below_coverage_percentile`) dejó de
   frenarse por el suelo de diversificación: toda posición bajo `coverage_percentile_floor` se
   vende, sin excepción (respetando solo `minimum_holding_period`). El destino de la plaza liberada
   —recompra o efectivo— lo sigue decidiendo el paso 4, sin cambios. Implementado en
   [module/evaluation/portfolio.py](../module/evaluation/portfolio.py), función `decide_orders`.

2. **Reducir rotación por rebalanceo de pesos.** Se añadió `rebalance_drift_tolerance = 0.40` al
   catálogo ([module/studies/catalog.py](../module/studies/catalog.py)) como nueva opción, sin
   tocar la lógica de `_apply_rebalance_tolerance` (sigue siendo tolerancia relativa pura). El
   valor "actual" no cambia (sigue en 0.25); 0.40 queda disponible para medir con el barrido de
   escenarios si compensa el 20 % de turnover atribuido a rebalanceo de pesos.

*(sin más pendientes)*
