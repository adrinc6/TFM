# Plan pendiente

> **En qué fijarse** cuando termine el relanzamiento, y **qué llevar al manuscrito**. Aquí ya no hay
> trabajo de implementación: todo lo que se podía automatizar se calcula solo al lanzar la cadena.
> Aquí tampoco se copian cifras: se dice qué mirar, en qué artefacto y qué decide.

**Estado a 2026-08-16.** El código está listo y no queda nada que programar antes de relanzar. Los
tres diagnósticos del capítulo económico —costes, capacidad y narrativa de cartera— se calculan
automáticamente cuando el Portfolio Study elige ganador, así que al terminar la cadena las cifras
están completas. **El siguiente paso lo ejecuta el usuario**: relanzar.

---

## Paso 0 — Relanzar (lo lanza el usuario)

Regla del repositorio: no se ejecuta un estudio real sin autorización explícita.

```powershell
python main.py ingest      # regenera data/raw y el panel corregido
python main.py serve       # y lanzar la cadena desde el dashboard
```

> **La ingesta es obligatoria, no opcional.** El panel incorpora ahora `median_dollar_volume_21d`
> y `dataset_code_version` ha subido a 2, de modo que el `dataset_hash` cambia. Sin reingesta no hay
> volumen y el diagnóstico de capacidad se declarará no disponible.

Tres condiciones que el relanzamiento debe cumplir:

1. **Las tres pasadas, bajo la misma versión de catálogo.** Es lo que elimina de raíz la limitación
   «versiones heterogéneas» en vez de obligar a redactarla con cuidado. Ver `docs/metodologia.md`,
   «Empates técnicos y versión de catálogo».
2. **Encadenadas**: el ganador de cada pasada es el `baseline` de la siguiente, manteniendo en
   `values` el abanico a reexplorar.
3. **Portfolio Study al final**, sobre el ganador ya congelado. Es quien dispara los diagnósticos.

Antes de nada, comprobar la ingesta: `data/raw/universe_coverage.json` debe traer el bloque
`ticker_resolution`, y sus recuentos deben sumar el universo histórico. Si no suman, hay tickers
perdiéndose sin dejar registro de fallo, y eso es un defecto que hay que resolver antes de gastar
horas de cómputo.

---

## Paso 1 — En qué fijarse cuando termine

Un bloque por artefacto: la pregunta que responde y qué decide. Todos cuelgan de
`results/studies/<study_id>/`.

### 1.1 `universe_coverage.json` → `ticker_resolution` · **la única decisión que sigue abierta**

**La pregunta**: ¿cuánto del agujero de cobertura es mortalidad real y cuánto es fallo de
resolución?

- Si domina `missing_cik`, mirar una muestra a mano: un símbolo que no resuelve puede ser un cambio
  de ticker o una absorción, y en ambos casos **los filings siguen en EDGAR**. Entonces lo que toca
  es resolver mejor el CIK histórico, no descargar más fundamentales.
- Si domina `no_metric_period_match`, el problema es cobertura de Finnhub sobre empresas que sí
  resuelven, y entonces **sí** tiene sentido el backfill XBRL.
- Si domina `missing_price`, el problema está en Yahoo y ninguna de las dos vías lo arregla.

**No emprender el backfill sin haber leído esta tabla.** La afirmación de que los ausentes eran «en
su mayoría quebrados o absorbidos» era una interpretación sin medir, y toda la justificación del
backfill descansaba sobre ella.

Si la auditoría lo justifica, lo que habría que añadir es `company_facts(cik)`
(`/api/xbrl/companyfacts/CIK##########.json`) a `EdgarClient` —que ya tiene rate limiting, caché y
resolución de CIK— casando las series XBRL con las métricas por `period`. Antes de decidirlo hay que
saber que **XBRL es obligatorio desde ~2009-2011: no recupera 2003-2008**. El resultado honesto sería
un panel que empieza hacia 2010 con cobertura casi completa en lugar de uno que empieza en 2003 con
cobertura parcial. Es un intercambio, no una mejora pura, y obliga a repetir el paso 0.

### 1.2 `cost_sensitivity.json` · ¿aguanta el supuesto de coste?

**La pregunta**: ¿hasta qué coste por operación sigue batiendo al índice?

Mirar `break_even.frozen_path.selection` y `break_even.resimulated.selection`, y el bloque
`margin_over_adopted`. La frase que habilita es del tipo *«el sistema bate al índice mientras operar
cueste menos de X pb (Y %) por operación»*.

Tres lecturas obligadas:

- **El resimulado debe salir mayor o igual que el congelado.** Si sale al revés hay un error de
  signo en los umbrales, y el test de contrato debería haberlo cazado antes.
- **Si aparece `beyond_ladder`**, el equilibrio cae fuera de la escalera medida y hay que ampliar
  `COST_LADDER_BPS` antes de citar nada. No se extrapola.
- **Si el margen es amplio**, la limitación de costes constantes baja de severidad en el capítulo de
  limitaciones en vez de quedar abierta. Es el objetivo de todo el diagnóstico.

### 1.3 `capacity.json` · ¿hasta qué patrimonio es ejecutable?

**La pregunta**: ¿a partir de cuánto dinero la cartera deja de poder operarse como se simuló?

Mirar `windows.selection.maximum_aum_usd` bajo los dos umbrales y, antes que nada,
`volume_coverage`: **si la cobertura es baja, el límite está calculado sobre pocas órdenes y no se
cita**. `binding_names` dice qué acciones lo atan, que es lo que hace la cifra explicable.

### 1.4 `portfolio_narrative.json` · ¿qué hizo la cartera?

**La pregunta**: más allá de cuánto ganó, ¿qué tuvo, cuánto tiempo y en qué se equivocó?

Es el material de la sección de cartera del paso 2. Al leerlo, comprobar que la historia es
coherente con el resto: si la permanencia mediana es de un solo snapshot, la cartera rota mucho más
de lo que sugiere el turnover reportado y hay que explicarlo; si un puñado de nombres concentra casi
toda la contribución, el resultado depende de pocas apuestas y eso debe decirse junto al alfa.

`sold_and_recovered` es el bloque incómodo a propósito: las ventas que luego subieron. Se lee con el
resultado ya conocido, así que señala dónde falló la doctrina de umbrales, **no** una regla que se
pueda añadir sin volver a ajustar sobre el resultado.

### 1.5 `robustness.json` y `attribution.json` · ¿sigue en pie la lectura?

**La pregunta**: ¿el Deflated Sharpe y las carteras aleatorias aguantan tras encadenar tres pasadas?

Es el punto que ya se debilitó en la cadena derogada. Si el DSR vuelve a bajar y el percentil frente
a carteras aleatorias no supera el umbral, la afirmación «no es suerte» se enuncia con esa cautela,
no se omite.

### 1.6 `report.md` del Portfolio Study · el resumen con procedencia

Reúne lo anterior con la ruta de cada cifra. Es el punto de partida para redactar y el que conviene
leer primero.

---

## Paso 2 — Qué añadir al manuscrito

> **Encargo cerrado.** Esto se le pasa a un agente cuando la cadena haya terminado. El manuscrito
> está congelado y el exportador (`latex/scripts/export_study_assets.py`) **no** se ha tocado, así
> que el encargo incluye generar las figuras y tablas nuevas. Procedimiento general en
> `latex/plan_tfm.md`. Ninguna cifra entra en el `.tex` sin poder señalar el artefacto exacto.

### 2.0 Antes que nada: la migración completa

Todas las cifras del manuscrito proceden de la cadena derogada y **ninguna sobrevive**: cambian el
`dataset_hash`, los `study_id`, los ganadores y las ~175 cifras escritas a mano en la prosa. Hay que
regenerar activos (`--study-id <NUEVO>`), sustituir identificadores (macro `\studyid` en
`latex/main.tex` y `latex/assets/a_reproducibilidad.tex`), reescribir la prosa capítulo a capítulo y
revalidar las cinco afirmaciones vertebradoras de `t01_afirmaciones.tex`. Los activos huérfanos del
study anterior se borran a mano; `latex/scripts/verify_latex_assets.py` los detecta.

Sigue vigente lo ya anotado en `docs/cambios_latex.md` sobre la tabla de cobertura anual y la
corrección del error factual sobre el tamaño del universo.

### 2.1 Sección de cartera · capítulo 7 · **material nuevo**

Es lo que hoy falta: el TFM reporta la cartera como una curva y unas métricas agregadas, y no dice
**qué hizo**. Respaldo: `portfolio_narrative.json`, `portfolio_narrative_holdings.parquet`,
`evidence_best_full/positions.parquet`, `orders.parquet` y `contributions.parquet`.

**Figuras a generar**

| Figura | Qué muestra | De dónde sale |
|---|---|---|
| Mapa de posiciones | Qué acciones tuvo la cartera en cada snapshot y con qué peso | `positions.parquet` (ya es la tabla larga snapshot × ticker × peso) |
| Exposición sectorial | Peso por sector a lo largo del tiempo | `portfolio_narrative.json` → `sector_exposure` |
| Distribución de permanencia | Cuánto dura una posición en esta cartera | `holding_duration` |
| Exceso frente al coste | La curva de las dos familias con `c*` marcado | `cost_sensitivity.json` |

**Tablas a generar**

| Tabla | Contenido |
|---|---|
| Las más presentes | Meses en cartera, episodios, peso medio, contribución neta |
| Mayores y menores contribuciones | Las que hicieron el resultado y las que lo restaron |
| Mejores y peores operaciones cerradas | Con fecha de entrada, salida, permanencia y motivo |
| Ventas que luego subieron | Coste de oportunidad de salir, medido contra el índice |
| Capacidad por umbral | Patrimonio máximo al 5 % y al 10 % del volumen diario |

**Prosa: qué afirma cada figura y qué no.** Tres salvedades son obligatorias y no se pueden omitir
por brevedad:

1. **El sector no es point-in-time**: procede de una foto actual de Finnhub y solo agrupa.
2. **El nocional diario es aproximado**: precio ajustado por splits y dividendos, volumen solo por
   splits.
3. **La cartera es la mejor de 1.728 evaluadas**, así que sus cifras dentro de la ventana de
   selección son una cota superior optimista.

Y una cuarta específica de las peores decisiones: se leen con el resultado ya conocido.

### 2.2 Sensibilidad a costes → baja la severidad de una limitación

`t09_limitaciones.tex` y el capítulo económico. La fila de costes constantes figura hoy con severidad
Media y **sin ninguna cifra que la acote**. Pasa a tener `c*` y `c**` y el margen sobre el coste
adoptado. Enunciar el equilibrio siempre **contra el índice** y con las dos familias, nunca con un
solo número: el congelado es conservador y el resimulado es el realista.

### 2.3 Capacidad y liquidez → responde a la crítica de capacidad

Limitaciones y conclusiones. Con el patrimonio máximo medido, la crítica deja de ser una objeción
abierta y pasa a ser un límite declarado. Decir explícitamente que se mide participación y no impacto
de mercado.

### 2.4 Columna de volumen en el panel

`03_datos_y_universo.tex` describe el panel point-in-time y debe incluir `median_dollar_volume_21d`:
qué es, para qué está (capacidad, nunca señal) y su salvedad de ajuste.

### 2.5 Artefactos nuevos en el anexo de reproducibilidad

`a_reproducibilidad.tex` enumera los artefactos citables. Añadir `contributions.parquet`,
`cost_sensitivity.json`, `capacity.json`, `portfolio_narrative.json`,
`portfolio_narrative_holdings.parquet` y el `report.md` del Portfolio Study.

---

## Descartado, y por qué

- **Ablación de agentes.** Con un meta-agente sin pesos mínimos, el propio meta ya puede anular
  agentes por su cuenta; los pesos aprendidos muestran esa información sin necesidad de una
  ablación aparte.
- **Ampliar el catálogo o la rejilla de carteras.** Cada configuración adicional empeora el Deflated
  Sharpe, que ya es la limitación de severidad más alta.
- **Extender el universo fuera del S&P 500.** Es otro trabajo, no una corrección de este.
- **Optimizar comisión o slippage.** Sería elegir el mundo en el que la estrategia luce mejor. Por
  eso la sensibilidad a costes es un diagnóstico y nunca un criterio de selección.
- **Reejecutar el ganador al cerrar el Portfolio Study.** Se evaluó y se descartó: ese estudio no
  reentrena, así que sus artefactos de modelo son los mismos del Model Study de origen y se enlazan.
  Reejecutarlos gastaría un entrenamiento completo para obtener una copia que podría no coincidir.
- **Preparar ya el exportador de LaTeX para la sección de cartera.** Generar activos para un capítulo
  que todavía no existe sería código muerto si el capítulo cambia de forma. El encargo del paso 2 lo
  incluye.
