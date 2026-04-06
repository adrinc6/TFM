# Interpretacion de Resultados

## 1. Estado del run actual

El ultimo run registrado en `results/pipeline.log` confirma que el pipeline se lanzo en modo anual con los siguientes rasgos operativos:

- Universo solicitado: 400 tickers.
- Periodo de descarga: 2015-01-01 a 2026-04-05.
- Modo de analisis: anual.
- Anchor anual: 2023-03-02.
- Quarter inicial de analisis: 2023Q1.
- Holding configurado en la evaluacion anual: 12 meses.

Tambien se observa que la descarga no fue completamente uniforme:

- 399 tickers quedaron completos.
- 1 ticker quedo parcial.
- 0 tickers quedaron sin datos.

Durante la consolidacion se reporto:

- 399 tickers listos para pipeline.
- 1 ticker sin datos completos.

## 2. Que significa esto metodologicamente

El resultado anterior es importante porque demuestra dos cosas:

1. El pipeline no depende de un universo perfecto e idealizado.
2. El sistema ya incorpora filtros de cobertura para no forzar folds con datos insuficientes.

En otras palabras, la ejecucion no asume que todos los tickers siempre estaran disponibles. Eso es sano desde el punto de vista metodologico.

## 3. Lectura del pipeline actual

Aunque en el momento de esta documentacion no estan presentes en `results/` todos los CSV consolidados del backtest final, el codigo del pipeline deja clara la estructura de salida esperada:

- `results/run_config.json`
- `results/data_quality_report.csv`
- `results/leakage_audit.csv`
- `results/baselines_summary.csv`
- `results/final_summary.csv`
- `results/final_summary.json`
- `results/final_portfolio_value.json`
- `results/backtest/strategy_equity_curve.csv`
- `results/backtest/benchmark_equity_curve.csv`
- `results/backtest/missing_prices_report.csv`
- artefactos fold-level en `results/backtest/fold_*`

Si esos archivos no aparecen todavia, significa que el run que quedo registrado en el log fue interrumpido o que los artefactos consolidados no se regeneraron en esta sesion concreta. El diseno del pipeline, sin embargo, si contempla su produccion.

## 4. Como interpretar cada bloque de salida

### 4.1 `run_config.json`

Este archivo documenta el contexto exacto del experimento:

- commit hash,
- versiones de librerias,
- flags activos,
- universo de tickers,
- rango de fechas,
- frecuencia de analisis,
- y parametros economicos del backtest.

Sirve como firma del experimento.

### 4.2 `data_quality_report.csv`

Resume cobertura y calidad por ticker y familia de features. Es el primer archivo que conviene mirar si un fold tiene pocas observaciones o si ciertas familias presentan muchos nulos.

### 4.3 `leakage_audit.csv`

Es el control mas importante para defender la metodologia. Si contiene filas con `n_rows_future_detected > 0`, hay que inspeccionar la fuente y el contexto. No significa automaticamente que la estrategia este invalidada, pero si exige explicar el motivo y revisar el filtro as-of correspondiente.

### 4.4 `baselines_summary.csv`

Permite comparar la estrategia principal contra:

- benchmark SPY,
- equal-weight universe,
- momentum 12m,
- random top-N,
- value combined.

La lectura correcta no es solo mirar retorno final. Hay que revisar tambien:

- `total_return_pct`,
- `max_drawdown`,
- `sharpe`,
- `total_fees_usd`,
- y `availability_flag`.

### 4.5 `final_portfolio_value.json`

Recoge el valor final monetario de la estrategia y del benchmark cuando este ultimo esta disponible. Es el archivo mas directo para una comparacion economica global.

### 4.6 `final_summary.csv` y `final_summary.json`

Son el resumen ejecutivo del experimento. Deben leerse junto con los folds y con el audit de leakage, no de forma aislada.

## 5. Lectura del benchmark

El codigo ya corrige un problema importante: si SPY no cubre la fecha final solicitada, la salida se trunca al ultimo dato disponible. Eso evita un benchmark vacio o con NaN silenciosos.

Por tanto, si un resumen anterior mostro benchmark vacio, esa lectura pertenece a una ejecucion previa al fix o a una cobertura temporal insuficiente. La interpretacion correcta es:

- el benchmark solo debe compararse en el rango realmente cubierto,
- y su disponibilidad debe declararse de forma explicita.

## 6. Lectura de los baselines

Los baselines estan pensados para responder a una pregunta simple: la estrategia principal supera alternativas razonables o solo parece buena frente a un benchmark facil?

### 6.1 Equal-weight universe

Sirve para medir si la fase de seleccion aporta valor respecto a repartir capital entre todo el universo elegible.

### 6.2 Momentum 12m

Es una referencia clasica y dificil de batir cuando el mercado tiene persistencia de tendencia.

### 6.3 Random top-N

No pretende ser competitivo. Su utilidad es mostrar dispersion y establecer una cota de azar reproducible.

### 6.4 Value combined

Es una baseline fundamentalista simple pero razonable. Si la estrategia principal no supera esta referencia, el modelo necesita una justificacion mas fuerte.

## 7. Como leer el resultado de un fold

Cada fold puede interpretarse asi:

- el modelo se entrena con historia anterior,
- se construyen features del quarter analizado,
- se selecciona cartera long-only,
- se ejecuta simulacion USD,
- y se calcula alpha frente a benchmark del mismo periodo.

Los campos mas utiles por fold son:

- `selected_tickers`
- `ticker_weights`
- `strategy_cumulative_return`
- `benchmark_cumulative_return`
- `alpha`
- `strategy_sharpe`
- `strategy_max_drawdown`

## 8. Que falta si no ves artefactos finales

Si ahora mismo solo ves `pipeline.log`, la lectura mas probable es una de estas tres:

1. El run no llego a la fase final de exportacion.
2. La carpeta `results/` fue limpiada despues.
3. Estabas viendo una sesion anterior y no el estado regenerado del backtest.

En cualquiera de los tres casos, el pipeline ya tiene implementada la generacion de los archivos finales; solo falta reejecutarlo para poblarlos.

## 9. Conclusion practica

La evidencia disponible hoy permite afirmar lo siguiente:

- el pipeline arranco correctamente,
- el universo real fue casi completo,
- el modo anual quedo activado,
- y la tuberia esta preparada para producir comparativas monetarias y auditoria temporal.

Lo que no se debe afirmar todavia, si no se regeneran los artefactos finales, es un ranking cuantitativo definitivo de estrategia vs benchmark vs baselines.

## 10. Siguiente accion recomendada

La siguiente accion util es reejecutar `python analyzer.py` y revisar despues:

- `results/final_summary.csv`,
- `results/baselines_summary.csv`,
- `results/final_portfolio_value.json`,
- `results/leakage_audit.csv`,
- y `results/backtest/benchmark_equity_curve.csv`.
