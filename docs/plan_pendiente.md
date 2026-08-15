# Plan pendiente

> Trabajo planificado y **no hecho**, con su justificación. Lo que se va completando sale de aquí y
> entra en `docs/bitacora.md`. Aquí tampoco se copian cifras: se dice qué medir y con qué artefacto.

**Estado a 2026-08-15.** Hecho todo lo que debía preceder al relanzamiento: reorganización
documental, auditoría de resolución de tickers, denominador de cobertura del universo y corrección
de los dos filtros de EDGAR que excluían a los emisores extranjeros. **El siguiente paso lo ejecuta
el usuario**: relanzar la cadena. Todo lo de abajo va después.

---

## Paso 0 — Relanzar (lo lanza el usuario)

Regla del repositorio: no se ejecuta un estudio real sin autorización explícita.

```powershell
python main.py ingest      # regenera data/raw con el panel corregido
python main.py serve       # y lanzar la cadena desde el dashboard
```

Tres condiciones que el relanzamiento debe cumplir:

1. **Las tres pasadas, bajo la misma versión de catálogo.** Es lo que elimina de raíz la limitación
   «versiones heterogéneas» en vez de obligar a redactarla con cuidado. Ver `docs/metodologia.md`,
   «Empates técnicos y versión de catálogo».
2. **Encadenadas**: el ganador de cada pasada es el `baseline` de la siguiente, manteniendo en
   `values` el abanico a reexplorar.
3. **Portfolio Study al final**, sobre el ganador ya congelado.

Antes de nada, comprobar la ingesta: `data/raw/universe_coverage.json` debe traer ahora el bloque
`ticker_resolution`, y sus recuentos deben sumar el universo histórico. Si no suman, hay tickers
perdiéndose sin dejar registro de fallo, y eso es un defecto que hay que resolver antes de gastar
horas de cómputo.

---

## Paso 1 — Leer la auditoría de tickers y decidir si hace falta el backfill XBRL

**Esto es una decisión, no una tarea**, y el paso 0 la habilita.

`universe_coverage.json` → `ticker_resolution` reparte el universo histórico por motivo de
exclusión. La pregunta que responde: **¿cuánto del agujero de cobertura es mortalidad real y cuánto
es fallo de resolución?**

- Si domina `missing_cik`, hay que mirar una muestra a mano: un símbolo que no resuelve puede ser un
  cambio de ticker o una absorción, y en ambos casos **los filings siguen en EDGAR**. Entonces la
  mejora que toca es resolver mejor el CIK histórico, no descargar más fundamentales.
- Si domina `no_metric_period_match`, el problema es cobertura de Finnhub sobre empresas que sí
  resuelven, y entonces **sí** tiene sentido el backfill XBRL de abajo.
- Si domina `missing_price`, el problema está en Yahoo y ninguna de las dos vías lo arregla.

**No emprender el backfill sin haber leído esta tabla.** El comentario que decía que los ~491
ausentes eran «en su mayoría quebrados o absorbidos» era una interpretación sin medir, y toda la
justificación del backfill descansaba sobre ella.

### Backfill de fundamentales desde EDGAR XBRL, si la auditoría lo justifica

Añadir `company_facts(cik)` (`/api/xbrl/companyfacts/CIK##########.json`) a `EdgarClient`, que ya
tiene rate limiting, caché y resolución de CIK, y mapear las series XBRL a las métricas que hoy
vienen de Finnhub casando por `period` como ya hace `module/data/ingest/pipeline.py`.

Lo que hay que saber antes de decidirlo:

- **XBRL es obligatorio desde ~2009-2011: no recupera 2003-2008.** El resultado honesto sería un
  panel que empieza hacia 2010 con cobertura casi completa, en lugar de uno que empieza en 2003 con
  cobertura parcial. Es un intercambio, no una mejora pura.
- Con un lookback de 8 años y la primera cohorte de selección en 2015, el entrenamiento necesita
  datos desde 2007: habría que recortar la ventana de selección o aceptar menos lookback.
- Cambia el `dataset_hash` otra vez, así que obliga a repetir el paso 0.

---

## Paso 2 — Diagnósticos sobre los scores congelados del nuevo ganador

Los dos reutilizan el patrón que ya existe: `run_profile_evaluation`
(`module/studies/runner.py`) y `selection_evidence` (`module/studies/portfolio_study.py`) leen
`agent_scores.parquet` congelado y solo rehacen el backtest, sin reentrenar.

### 2.1 Sensibilidad a costes ← **el mejor valor por esfuerzo de esta lista**

Hoy los costes son constantes (comisión y slippage fijos) sobre una cartera de rotación alta, y esa
es la cifra más atacable del capítulo económico. La limitación figura con severidad Media y **sin
ninguna cifra que la acote**. Lo que falta no es un supuesto de coste mejor, sino decir **hasta
dónde aguanta el que hay**.

Entregable: tres escenarios sobre la cartera que el TFM adopta —la **ganadora del Portfolio
Study**, no la del catálogo por defecto—:

| Escenario | Qué responde |
|---|---|
| **Bruto** (coste 0) | Cuánto vale la señal antes de fricciones. Cota superior que nadie realiza |
| **Estándar** (5 + 10 pb) | El resultado que ya se reporta |
| **Equilibrio** (`c*`) | El coste que anula el exceso geométrico contra el S&P 500 |

`c*` se publica en puntos básicos, en porcentaje y en ida y vuelta, para que la frase final sea:
*«el sistema bate al índice mientras operar cueste menos de X pb (Y %) por operación»*. El
equilibrio se define **contra el índice**, no contra rentabilidad absoluta: la alternativa real de un
inversor es comprar el índice, no quedarse en efectivo.

#### Los dos hechos que determinan el diseño

**1. El coste es exactamente `turnover × tasa`.** En `_price_orders`
([module/evaluation/backtest.py](../module/evaluation/backtest.py)) el drag total es
`Σ(notional × tasa) / value`, y como `notional = |Δw| × value` y `turnover = Σ|Δw|`, sale
`drag = turnover × tasa` sin aproximación. `equity.parquet` ya persiste `turnover_pct` y `cost_drag`
por snapshot, así que **sobre la ruta de operaciones ya ejecutada la curva de costes entera se
obtiene en forma cerrada, con cero cómputo**.

**2. Pero el coste entra dos veces.** Además de la contabilidad, alimenta los umbrales de decisión:
`round_trip` en [module/evaluation/portfolio.py](../module/evaluation/portfolio.py) fija
`entry_threshold` y `rotation_threshold`. Poner el coste a cero **no es la misma cartera sin
comisiones**: los umbrales se desploman y operaría mucho más. Y al revés, con costes altos la
cartera opera menos y se protege sola.

Por eso un solo número sería engañoso y hacen falta **dos familias**:

- **Ruta congelada** (forma cerrada): mismas decisiones, distinto coste. Es la comparación pura, y su
  `c*` es **conservador**, porque un gestor que pagase más operaría menos.
- **Resimulada**: la cartera vuelve a decidir con cada coste, de modo que su `c**` ≥ `c*`. Cuesta
  ~5-6 s por peldaño reutilizando los scores congelados.

**La diferencia entre ambas es en sí misma un resultado**: mide cuánto protege la doctrina de
umbrales económicos, que existe precisamente para que cada operación pague su coste.

#### Orden de magnitud esperado (estimación, no resultado)

Con las cifras de la cadena **derogada** —exceso ≈ 6,97 %/año, rotación ≈ 3,24/año— el equilibrio de
ruta congelada saldría en `0,0697 / 3,24 ≈ 215 pb` por operación, frente a los 15 pb asumidos: un
margen de ~14×. Si se confirma tras el relanzamiento, el titular honesto es que **el resultado es
robusto al supuesto de coste por un margen amplio**, y la limitación baja de severidad en vez de
quedar abierta. Es una estimación de servilleta sobre cifras derogadas: sirve para dimensionar el
barrido, no para citarla.

De ahí una consecuencia práctica: **el catálogo no puede expresar el barrido**, porque sus valores
dan entre 5 y 30 pb por operación —ni el cero ni nada cercano a 215—. La escalera va como constante
de diagnóstico, con el precedente de `SEED_ENSEMBLE` y de los `iterations` del bootstrap, que
tampoco son variables del catálogo porque no se optimizan. No hace falta tocar el catálogo cerrado:
`settings_from_values` ([module/studies/config.py](../module/studies/config.py)) no valida contra él
—la validación ocurre antes, al definir el study— así que basta pasar los costes en el `values` del
backtest.

#### Implementación

Módulo nuevo `module/research/cost_sensitivity.py`:

- `COST_LADDER_BPS`: peldaños de coste **por operación** (comisión + slippage), de 0 hasta pasado el
  equilibrio esperado.
- `frozen_path_curve(equity)`: forma cerrada. Para cada peldaño sustituye `cost_drag` por
  `turnover_pct × tasa`, capitaliza y devuelve exceso geométrico e IR.
- `break_even_bps(...)`: interpola el coste donde el exceso geométrico cruza cero.
- Todo **por ventana**: selección y era reservada, siempre por separado.

Enganche en [module/studies/portfolio_study.py](../module/studies/portfolio_study.py), justo después
de la llamada a `run_profile_evaluation` que reevalúa la combinación ganadora sobre la serie completa
y deja `evidence_best_full/`:

- **Familia congelada**: leer `evidence_best_full/equity.parquet`. Coste cero.
- **Familia resimulada**: `run_profile_evaluation({**config, "commission_bps": c,
  "slippage_bps": s}, "balanced", evidence_dir)` por peldaño, sin `retain_dir`.
- Escribir `cost_sensitivity.json` junto a `portfolio_winner.json`.

#### Salvedades que deben viajar dentro del propio artefacto

1. El escenario bruto es una cota que **ningún inversor realiza**.
2. El equilibrio de ruta congelada es **conservador** por construcción.
3. El exceso de la ventana de selección ya es una **cota superior optimista** (la cartera es la mejor
   de la rejilla), así que `c*` hereda ese optimismo.
4. Los costes **nunca seleccionan**: elegir el coste sería elegir el mundo que más conviene. La
   validación del Portfolio Study ya lo impide y debe seguir impidiéndolo.

#### Verificación

Tests en `tests/test_cost_sensitivity_contract.py`:

- **La identidad `drag = turnover × tasa`**, que es el supuesto que sostiene toda la forma cerrada.
  Si falla, la curva entera es ficción.
- **Autoconsistencia**: evaluada en el coste adoptado, la forma cerrada debe reproducir
  **exactamente** el exceso que ya reporta el ganador. Es lo que prueba que la familia congelada no
  se ha desviado del motor, y el motivo de que este diagnóstico no se implementara antes del
  relanzamiento: sin un Portfolio Study vigente no hay contra qué comprobarlo.
- **`c**` ≥ `c*`**: la resimulada aguanta al menos tanto como la congelada, porque opera menos al
  encarecerse. Si sale al revés, hay un error de signo en los umbrales.
- Exceso bruto > exceso estándar, y monotonía decreciente del exceso con el coste.
- Selección y era reservada se calculan por separado y no se mezclan.
- `cost_sensitivity.json` no escribe en `winner.json`, `decisions.json` ni `portfolio_winner.json`.

### 2.2 Capacidad y liquidez

Los OHLCV ya traen volumen. Calcular por snapshot el nocional operado como fracción del volumen
medio de 21 días, y reportar el patrimonio a partir del cual la cartera deja de ser ejecutable.
Responde a la crítica de capacidad sin tener que modelar impacto de mercado, que sería un trabajo
aparte.

---

## Paso 3 — Cerrar la deuda con el manuscrito

Todo lo acumulado en `docs/cambios_latex.md`, siguiendo el procedimiento de `latex/plan_tfm.md`. Va
al final a propósito: corregir hoy la redacción de unas cifras que el relanzamiento va a sustituir
sería trabajo perdido.

---

## Descartado, y por qué

- **Ablación de agentes.** Con un meta-agente sin pesos mínimos, el propio meta ya puede anular
  agentes por su cuenta; los pesos aprendidos muestran esa información sin necesidad de una
  ablación aparte.
- **Ampliar el catálogo o la rejilla de carteras.** Cada configuración adicional empeora el Deflated
  Sharpe, que ya es la limitación de severidad más alta.
- **Extender el universo fuera del S&P 500.** Es otro trabajo, no una corrección de este.
- **Optimizar comisión o slippage.** Sería elegir el mundo en el que la estrategia luce mejor.
