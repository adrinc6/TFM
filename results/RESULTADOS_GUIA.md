# Guía de Resultados — Pipeline ML Multi-Agente Stock Picker

Todos los artefactos se generan bajo `results/` con la siguiente estructura:

```
results/
├── pipeline.log
├── master_dataset.csv
├── plots/
├── backtest/
└── agents/
    ├── fundamental/
    ├── valuation/
    ├── momentum/
    ├── bear/
    └── meta_learner/
```

---

## `results/pipeline.log`

Log completo de ejecución con timestamps. Contiene el progreso de cada paso: descarga de datos, construcción del dataset, entrenamiento fold a fold, y métricas finales. Es el primer sitio donde mirar si algo falla.

---

## `results/master_dataset.csv`

Dataset maestro con una fila por (ticker, fecha trimestral). Columnas:
- `ticker`, `date` — índice
- `sector`, `industry` — clasificación sectorial de companies.csv
- `forward_return` — retorno real a 252 días (el label)
- Todas las features fundamentales, técnicas, de valoración, insider y macro

Sirve para inspeccionar qué datos entran al modelo, detectar outliers, o analizar la cobertura de tickers por trimestre.

---

## `results/plots/`

### `sector_performance.png`
Barras con el retorno medio del forward return a 1 año por sector (todo el universo, no solo los seleccionados). Verde = retorno medio positivo, rojo = negativo. Sirve para ver qué sectores son estructuralmente más rentables en el período analizado y si el modelo tiene sesgo sectorial.

### `score_dist_fold{N}.png`
Un panel por agente (fundamental, valuation, momentum, bear, final). Cada panel muestra la distribución de scores del agente separada por label real (azul = Outperform, rojo = Underperform). Un agente bueno muestra distribuciones bien separadas. Si se solapan completamente, el agente no discrimina.

### `feat_imp_{agente}_fold{N}.png`
Top 20 features más importantes del agente indicado en el fold indicado. Barras horizontales ordenadas por importancia (Gain de XGBoost/LGBM). Sirve para interpretar qué variables está usando cada agente y detectar si hay features dominantes que podrían indicar leakage.

### `full_report.png`
Dashboard completo de 7 paneles con el rendimiento global de toda la serie walk-forward concatenada:

| Panel | Contenido |
|---|---|
| Curva de Riqueza | Evolución del valor de la cartera vs S&P 500 (base 1). Anota el retorno total final. |
| Drawdown | Caída máxima pico-valle en % para la estrategia (área) y benchmark (línea). |
| Alpha Acumulado | Diferencia diaria acumulada de retornos (estrategia − S&P 500). Verde = outperformance, rojo = underperformance. |
| Retorno por Fold | Barras de retorno acumulado del período de test por fold: estrategia vs benchmark. |
| Distribución Mensual | Histograma de retornos mensuales (estrategia vs S&P 500). Muestra sesgo y amplitud de la distribución. |
| Sharpe por Fold | Línea del Sharpe ratio en cada fold para estrategia y benchmark. Referencia en Sharpe=1. |
| AUC por Agente | Barras del AUC-ROC de cross-validation de cada agente en el último fold. Referencia en 0.5 (aleatorio). |

### `folds_results.png`
Dashboard de 4 paneles generado por `backtester.save_folds_summary()`. Complementa al full_report con el desglose por longitud de ventana de entrenamiento:

| Panel | Contenido |
|---|---|
| Alpha por Fold | Barras de alpha del período (estrategia − benchmark). Coloreadas por años de train. Media en línea discontinua. |
| Retorno Acumulado por Fold | Barras de retorno acumulado real del período de test: estrategia vs benchmark. |
| Sharpe por Fold | Líneas de Sharpe para estrategia y benchmark con área verde/roja según quién supera. |
| Boxplot Alpha por Train Years | Distribución de alphas agrupada por ventana de entrenamiento (3Y, 4Y, 5Y...). Permite ver si más datos de train ayudan. |

---

## `results/backtest/`

### `fold_{NNN}_{N}Y_metrics.json`
Métricas de cartera de un fold concreto. El nombre indica el número de fold y los años de entrenamiento usados.

| Campo | Descripción |
|---|---|
| `fold` | Número de fold |
| `train_years` | Años de ventana de entrenamiento |
| `train_start` | Inicio del período de entrenamiento |
| `test_start` / `test_end` | Período de test (1 trimestre) |
| `selected_tickers` | Lista de tickers seleccionados para la cartera |
| `n_stocks` | Número de posiciones (top N por score) |
| `strategy_cumulative_return` | Retorno real de la cartera durante el trimestre de test |
| `benchmark_cumulative_return` | Retorno real del S&P 500 durante el mismo período |
| `strategy_sharpe` | Sharpe anualizado de la cartera en el período |
| `benchmark_sharpe` | Sharpe anualizado del S&P 500 en el período |
| `strategy_sortino` | Sortino ratio (solo penaliza volatilidad negativa) |
| `strategy_max_drawdown` | Máxima caída pico-valle de la cartera en el trimestre |
| `strategy_calmar` | Retorno / |max drawdown| (mayor = mejor gestión de riesgo) |
| `strategy_volatility` | Volatilidad anualizada de retornos diarios |
| `alpha` | Retorno cartera − Retorno benchmark en el período |
| `excess_sharpe` | Sharpe cartera − Sharpe benchmark |

### `backtest_summary.json`
Resumen global agregando todos los folds. Campos principales:

| Campo | Descripción |
|---|---|
| `n_folds` | Total de folds completados |
| `mean_alpha` | Alpha medio por fold (métrica principal del sistema) |
| `pct_folds_positive_alpha` | % de folds en que la estrategia superó al benchmark |
| `by_train_years` | Desglose por ventana de train: alpha medio, % positivo, Sharpe medio |
| `global_strategy_*` | Métricas calculadas sobre la serie completa concatenada de todos los folds |
| `global_benchmark_*` | Idem para el S&P 500 |

### `returns_series.csv`
Serie temporal diaria con dos columnas: `strategy` y `benchmark`. Son los retornos diarios de todos los folds concatenados en orden cronológico. Útil para calcular métricas adicionales externamente o para graficar con otras herramientas.

### `folds_results.csv`
Tabla con una fila por fold y todas las métricas del JSON de cada fold. Versión tabular conveniente para explorar en Excel o pandas. Incluye columna `test_period` con el rango de fechas en formato legible.

---

## `results/agents/`

Cada agente tiene su propia subcarpeta (`fundamental/`, `valuation/`, `momentum/`, `bear/`, `meta_learner/`).

### `{agente}/diagnostics_fold{N}.json`
Estado interno del agente tras entrenar y predecir en un fold. Contiene:
- `last_train_metrics`: AUC, accuracy, F1 y std del cross-validation de entrenamiento
- `feature_cols`: columnas de entrada que usó el agente
- `n_train_samples`, `n_test_samples`: observaciones en train y test
- Cualquier diagnóstico específico del agente (p.ej. threshold del BearAgent)

### `{agente}/feature_importances_fold{N}.csv`
Importancia de cada feature para el modelo del agente en ese fold. Columnas: nombre de feature, `importance` (Gain normalizado). Útil para ver la evolución de importancias fold a fold.

### `{agente}/predictions_fold{N}.csv`
Predicciones del agente sobre el conjunto de test: score de probabilidad y label predicho para cada (ticker, fecha). Sirve para analizar en detalle qué stocks clasifica bien o mal cada agente.

### `{agente}/train_history.json`
Historial acumulado de entrenamiento del agente a lo largo de todos los folds. Lista de entradas con métricas de cada fold. Permite ver si el modelo mejora o empeora a medida que avanzan los períodos.

### `bear/flag_report_fold{N}.json`
Exclusivo del BearAgent. Estadísticas de las señales de riesgo activadas: cuántos tickers dispararon cada flag (alta deuda, pérdidas consecutivas, accruals elevados, etc.) y con qué frecuencia.

### `meta_learner/evaluation_fold{N}.json`
Evaluación completa del MetaLearner sobre el test de ese fold:
- AUC-ROC, accuracy, precision, recall, F1
- Matriz de confusión (TP, FP, TN, FN)
- Classification report completo por clase
- Threshold de decisión utilizado

### `meta_learner/lr_coefficients_fold{N}.json`
Pesos asignados por el MetaLearner a cada score de agente. Muestra cuánto "confía" el meta-modelo en cada agente para ese fold. Un coeficiente alto positivo = ese agente tiene poder predictivo; negativo = el meta-learner lo invierte.

### `all_folds_diagnostics.json`
Histórico consolidado de diagnósticos de todos los agentes a lo largo de todos los folds. Estructura: `{agente: [diag_fold1, diag_fold2, ...]}`. Permite analizar la evolución del AUC y otros indicadores sin abrir uno a uno los JSONs de cada fold.

### `fold_{N}_ticker_explanations.json`
Explicaciones SHAP de las predicciones del fold N para los 10 tickers con mayor score (candidatos Outperform) y los 10 con menor score (candidatos Underperform). Por cada ticker incluye:
- `score`: puntuación final del meta-learner
- `label`: "Outperform" o "Underperform"
- `agents`: para cada agente, texto explicativo y lista de los 6 drivers principales con su contribución SHAP

Es el artefacto más interpretable: explica *por qué* el sistema recomienda o descarta cada acción.
