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
es la cifra más atacable del capítulo económico. `commission_bps` y `slippage_bps` ya son variables
del catálogo con tres valores cada una, así que la rejilla 3×3 sobre la cartera adoptada cuesta
minutos.

Entregable: el **coste de equilibrio**, es decir a partir de cuántos puntos básicos se anulan el
exceso y el Information Ratio. Convierte «asumimos 15 pb» en «el resultado sobrevive hasta X pb».

**Diagnóstico, nunca selector**: elegir el coste sería elegir el mundo que más conviene. La
validación del Portfolio Study ya lo impide y debe seguir impidiéndolo.

Verificación: el barrido con los costes del ganador debe reproducir **exactamente** su resultado
congelado. Si no coincide, el adaptador está mal.

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
