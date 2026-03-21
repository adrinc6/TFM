# Documentación — TFM: Multi-Agente ML Stock Picker

## Índice

1. [Visión general](#1-visión-general)
2. [Cómo se ejecuta el proyecto](#2-cómo-se-ejecuta-el-proyecto)
3. [Estructura del repositorio](#3-estructura-del-repositorio)
4. [Pipeline completo paso a paso](#4-pipeline-completo-paso-a-paso)
5. [Fuentes de datos y almacenamiento local](#5-fuentes-de-datos-y-almacenamiento-local)
6. [Construcción del dataset](#6-construcción-del-dataset)
7. [Agentes del sistema](#7-agentes-del-sistema)
8. [Entrenamiento y validación](#8-entrenamiento-y-validación)
9. [Meta-learner y stacking](#9-meta-learner-y-stacking)
10. [Backtest walk-forward](#10-backtest-walk-forward)
11. [Outputs y artefactos](#11-outputs-y-artefactos)
12. [Explicabilidad y ablation](#12-explicabilidad-y-ablation)
13. [Configuración global](#13-configuración-global)
14. [Comportamientos y decisiones de diseño relevantes](#14-comportamientos-y-decisiones-de-diseño-relevantes)
15. [Problemas habituales y cómo interpretarlos](#15-problemas-habituales-y-cómo-interpretarlos)
16. [Lectura rápida por archivo](#16-lectura-rápida-por-archivo)
17. [Outputs por fold: archivos y columnas](#17-outputs-por-fold-archivos-y-columnas)

---

## 1. Visión general

Este repositorio implementa un pipeline de stock picking multi-agente sobre el universo S&P 500. La idea es combinar varios modelos especializados —cada uno con una "lente" distinta sobre la empresa— para producir una predicción de tipo `Outperform / Underperform` por ticker y evaluarla históricamente con walk-forward.

### Tres capas bien separadas

- **Capa de datos**: descarga desde Finnhub + precios, parseo, consolidación en CSV por ticker, y construcción del dataset maestro con features y labels.
- **Capa de modelado**: seis agentes especializados por dominio (fundamental, valoración, momentum, riesgo, sentimiento, rotación sectorial) más un meta-learner que combina sus scores.
- **Capa de evaluación**: backtest walk-forward, explicabilidad SHAP, auditoría de selección, CSV de scores por fold y gráficos.

### Principios de diseño

- **Sin look-ahead**: los snapshots de features se construyen con datos estrictamente anteriores a la fecha de observación. Los labels usan precios futuros, pero nunca cruzan el corte temporal del train.
- **Horizonte de predicción**: ~1 quarter. La pregunta que responde el modelo es "¿este ticker outperformará a la **mediana de su sector** en el próximo trimestre?".
- **Análisis top-down + bottom-up**: el `SectorRotationAgent` predice qué sectores van a superar al S&P 500 (top-down). Los agentes base evalúan la empresa dentro de su sector (bottom-up). El MetaLearner combina ambas señales.
- **Scores de agentes como features del meta-learner**: los agentes base no predicen directamente la cartera; su salida (probabilidad 0-1) alimenta el nivel 2 (stacking).
- **Robustez ante datos faltantes**: si un agente no puede entrenar (p. ej. por falta de muestras en el SentimentAgent), el pipeline no se rompe y usa score neutro 0.5 como fallback.

---

## 2. Cómo se ejecuta el proyecto

El punto de entrada es [analyzer.py](analyzer.py).

```bash
python analyzer.py
```

### Flujo de alto nivel

```text
main()
  ├── download_data()              # Step 1: descarga JSONs de Finnhub por ticker
  ├── prepare_data()               # Step 1: consolida JSONs → CSV por ticker
  ├── get_available_tickers()      # Filtra tickers sin datos suficientes
  ├── [retry_missing_tickers()]    # Opcional: reintenta tickers incompletos
  ├── build_master_dataset()       # Step 2: construye DataFrame (ticker, date) con features + label
   └── [run_walkforward_pipeline()] # Step 4: backtest histórico fold a fold
```

### Flags de control principales en `environment.py`

| Flag | Efecto |
|---|---|
| `SKIP_BACKTEST` | Salta el walk-forward histórico |
| `UPDATE_PRICES_ONLY` | Solo actualiza precios y macro; no consolida ni entrena |
| `FORCE_DOWNLOAD` | Re-descarga todos los JSONs aunque ya existan en disco |
| `RETRY_MISSING_TICKERS` | Reintenta descargar tickers con datos incompletos |
| `RUN_ABLATION_STUDY` | Activa el estudio de ablación (lento, desactivado por defecto) |
| `UPDATE_PRICES_ONLY` | Solo actualiza precios y macro; no consolida ni entrena |
| `DAYS_BEFORE_QUARTER_START` | Días de adelanto respecto al inicio del quarter de test para comprar la cartera |

### Cálculo de fechas de test

```python
# test_start_date = fin del quarter ANTERIOR al primero que se quiere predecir
# Ejemplo: TEST_START_YEAR=2025, TEST_START_QUARTER=3
#   → predice Q3 2025 (julio-septiembre)
#   → train_end = 30 junio 2025

test_start_date = quarter_end(TEST_START_YEAR, TEST_START_QUARTER) - QuarterEnd(1)
end_date        = quarter_end(END_YEAR, END_QUARTER)
```

---

## 3. Estructura del repositorio

```text
TFM/
├── analyzer.py                        # Orquestador principal
├── environment.py                     # Configuración global (fuente única de verdad)
├── DOCUMENTATION.md                   # Este documento
├── requirements.txt
│
├── data_finnhub/
│   ├── _macro/
│   │   ├── sp500.json                 # Precios históricos S&P 500
│   │   ├── vix.json                   # VIX histórico
│   │   ├── us10y.json                 # Bono 10 años
│   │   └── us2y.json                  # Bono 2 años
│   ├── {TICKER}/
│   │   ├── profile.json               # Sector, industria, país
│   │   ├── quote.json                 # Último precio (referencia)
│   │   ├── basic_financials.json      # Ratios financieros y series básicas
│   │   ├── financials_reported_quarterly.json  # 10-Q trimestrales
│   │   ├── financials_reported_annual.json     # 10-K anuales
│   │   ├── eps_surprises.json         # Sorpresas de BPA (beat/miss)
│   │   ├── recommendation_trends.json # Consenso de analistas
│   │   ├── insider_transactions.json  # Compras/ventas de insiders
│   │   ├── insider_sentiment.json     # MSPR de insiders
│   │   └── peers.json                 # Peers del ticker
│   └── consolidated/
│       └── {TICKER}.csv               # Serie unificada lista para el dataset
│
├── results/
│   ├── pipeline.log                   # Log completo de la ejecución
│   ├── master_dataset.csv             # Dataset maestro con todas las features
│   ├── agents/
│   │   ├── fold_{N}_scores.csv        # Scores + explicaciones por ticker por fold
│   │   ├── fold_{N}_selection_audit.csv / .json
│   │   ├── fold_{N}_ticker_explanations.csv / .json
│   │   ├── {agent}/
│   │   │   ├── feature_importances_fold{N}.csv
│   │   │   ├── diagnostics_fold{N}.json
│   │   │   ├── train_history.json
│   │   │   ├── shap_global_fold{N}.csv / .json
│   │   │   ├── shap_bar_fold{N}.png
│   │   │   └── flag_report_fold{N}.json   # Solo BearAgent
│   │   └── meta_learner/
│   │       ├── evaluation_fold{N}.json
│   │       ├── predictions_fold{N}.csv
│   │       ├── lr_coefficients_fold{N}.json
│   │       ├── shap_global_fold{N}.csv / .json
│   │       └── shap_bar_fold{N}.png
│   ├── backtest/
│   │   ├── fold_{NNN}_{X}Y_metrics.json   # Métricas de cada fold
│   │   └── (resumen global)
│   └── plots/
│       ├── feat_imp_{agent}_fold{N}.png
│       ├── score_dist_fold{N}.png
│       └── fold_{NNN}_{QUARTER}_performance.png
│
└── module/
    ├── common/
    │   └── data_router.py             # Acceso unificado a datos por ticker
    ├── agents/
    │   ├── base.py                    # BaseAgent: fit/predict/save/load/diagnostics
    │   ├── fundamental.py             # FundamentalAgent (XGBoost)
    │   ├── valuation.py               # ValuationAgent (GBM)
    │   ├── momentum.py                # MomentumAgent (Random Forest)
    │   ├── bear.py                    # BearAgent (RF híbrido + reglas)
    │   ├── sentiment.py               # SentimentAgent (Random Forest)
    │   ├── sector_rotation.py         # SectorRotationAgent (GBM, nivel sector)
    │   └── meta_learner.py            # MetaLearner (LR + GBM stacking)
    └── steps/
        ├── step_01_data/
        │   ├── clients.py             # Cliente HTTP Finnhub con rate limiting
        │   ├── downloaders.py         # Descarga paralela por ticker
        │   ├── pipeline.py            # download_data(), prepare_data(), get_available_tickers()
        │   └── consolidation.py       # Parseo y unificación de JSONs → CSV
        ├── step_02_dataset/
        │   ├── builders/
        │   │   ├── fundamental.py     # FundamentalFeatureBuilder
        │   │   ├── technical.py       # TechnicalFeatureBuilder
        │   │   ├── valuation.py       # ValuationFeatureBuilder
        │   │   ├── insider.py         # InsiderFeatureBuilder
        │   │   ├── sentiment.py       # SentimentFeatureBuilder
        │   │   └── sector.py          # SectorNormalizer (z-score sectorial)
        │   ├── dataset.py             # build_master_dataset()
        │   └── normalization.py       # apply_sector_normalization()
        ├── step_03_training/
        │   ├── agent_config.py        # build_agents_config(), build_sector_rotation_agent()
        │   ├── oof.py                 # generate_oof_scores(): OOF anti-leakage
        │   └── training.py            # train_fold(), train_full_history()
        ├── step_04_evaluation/
        │   ├── evaluator.py           # run_walkforward_pipeline(): orquestador del backtest
        │   ├── backtester.py          # WalkForwardBacktester: folds, simulación de cartera
        │   ├── metrics.py             # compute_all_metrics(): Sharpe, Sortino, alpha, etc.
        │   ├── visualization.py       # Visualizer: curvas de riqueza, drawdown, etc.
        │   ├── explainability.py      # AgentExplainer: SHAP por agente
        │   ├── fold_report.py         # build_fold_scores_df(), export_fold_scores()
        │   ├── selection_reports.py   # Auditoría de selección y explicaciones por ticker
        │   ├── reports.py             # generate_text_report(): resumen narrativo
        │   └── ablation.py            # run_ablation_study(), summarize_ablation()
```

---

## 4. Pipeline completo paso a paso

### 4.1 Descarga de datos (`step_01_data`)

`download_data()` descarga en paralelo (hasta `DOWNLOAD_MAX_WORKERS=8` workers) los siguientes endpoints de Finnhub para cada ticker:

- `profile` → sector, industria, capitalización
- `basic_financials` → P/E, P/B, ROE, márgenes, y series históricas
- `financials_reported_quarterly` → 10-Q con métricas operativas
- `financials_reported_annual` → 10-K para derivar Q4 y tendencias anuales
- `eps_surprises` → historial de sorpresas de BPA
- `recommendation_trends` → consenso de analistas (strong buy/buy/hold/sell)
- `insider_transactions` → compras y ventas de directivos
- `insider_sentiment` → MSPR (Monthly Share Purchase Ratio)
- `peers` → tickers comparables del mismo sector

Los datos macro (VIX, S&P 500, bono 10Y, bono 2Y) se descargan a `_macro/`.

Si `FORCE_DOWNLOAD=False`, los JSONs existentes se reutilizan **siempre que el archivo también exista en disco** — si el registro está marcado como descargado pero el archivo no existe, se vuelve a descargar. El rate limit de Finnhub se respeta con `FINNHUB_MIN_INTERVAL=1` segundo entre requests por ticker.

### 4.2 Consolidación (`consolidation.py`)

`prepare_data()` convierte los JSONs dispersos de cada ticker en un único CSV consolidado (`data_finnhub/consolidated/{TICKER}.csv`):

- Unifica 10-Q y 10-K en una serie trimestral ordenada.
- **Deriva Q4** desde el anual cuando el trimestre no está reportado explícitamente: `Q4 = Anual - (Q1 + Q2 + Q3)`.
- Calcula ratios derivados: net_margin, roe, debt_to_ebitda, current_ratio, etc.
- Asigna una columna `date` alineada al cierre del trimestre (último día del mes).
- Incluye TTM (trailing twelve months) para revenue, EPS, EBITDA y FCF cuando hay cuatro trimestres disponibles.

### 4.3 Construcción del dataset maestro (`dataset.py`)

`build_master_dataset()` recorre cada ticker disponible y genera una observación por cada fecha de corte trimestral:

```python
for ticker in tickers:
    for as_of in quarter_dates:  # fechas de cierre de trimestre
        snapshot = fundamental_data_up_to(as_of)
        features = {
            **fundamental_builder.build(snapshot),
            **technical_builder.build(prices_up_to(as_of), as_of),
            **valuation_builder.build(prices, snapshot, hist, as_of),
            **insider_builder.build(transactions_up_to(as_of)),
            **sentiment_builder.build(eps_surprises, recommendations, as_of),
        }
        label = forward_return_63d(prices, as_of)   # label futuro
        records.append((ticker, as_of, features, label))
```

El resultado es un DataFrame indexado por `(ticker, date)` guardado en `results/master_dataset.csv`.

**Anti-leakage de construcción**: cada builder recibe únicamente datos con `index <= as_of`. El label (`forward_return`) usa precios futuros, pero estos nunca se filtran al train porque el split train/test se hace por fecha después de construir el dataset.

### 4.4 Label `forward_return` y binarización

El `forward_return` mide el retorno del **holding period real**, alineado con `DAYS_BEFORE_QUARTER_START`:

```
as_of = Mar 31 (fin de Q1), DAYS_BEFORE_QUARTER_START = 30
  → entry = Apr 1 − 30 días ≈ Mar 2   (precio de compra real)
  → exit  = Jul 1 − 30 días ≈ Jun 1   (precio de venta real)
  forward_return = (precio_Jun1 − precio_Mar2) / precio_Mar2
```

Con `DAYS_BEFORE_QUARTER_START=0` equivale al comportamiento clásico: retorno entre cierres de quarter.

El label binario:
- `y = 1` si el `forward_return` del ticker **supera la mediana de su sector en el quarter forward** (el quarter que se está midiendo, no el del snapshot).
- `y = 0` en caso contrario.

La agrupación sectorial usa el **quarter forward** para que la comparativa sea siempre entre tickers con el mismo período de medición. Con offset de 30 días, el snapshot de Q1 (Mar 31) tiene forward_return en Q2; la mediana se calcula entre todos los tickers con forward_return en Q2.

Este enfoque separa el stock picking de la rotación de sectores: el modelo aprende "esta empresa es la mejor dentro de su sector en ese período", no "este sector tuvo un buen trimestre". La señal de rotación sectorial la captura el `SectorRotationAgent`.

Fallback: si `sector_map` no está disponible, compara contra la mediana del universo completo en ese forward quarter.

### 4.5 Normalización sectorial (`normalization.py`, `builders/sector.py`)

Antes de entrenar, las features se normalizan por sector con z-score:

```python
for sector in sectors:
    peers = df[df["sector"] == sector]
    if len(peers) >= SECTOR_ZSCORE_MIN_PEERS:  # mínimo 3 empresas
        df.loc[peers.index, feature] = zscore(peers[feature])
    # Si hay menos peers, se mantiene el valor original (comportamiento conservador)
```

Esto reduce el sesgo sectorial: no comparas el P/E de una utility con el de un semiconductor.

### 4.6 Walk-forward backtest (`evaluator.py`, `backtester.py`)

`run_walkforward_pipeline()` itera sobre los folds generados por `WalkForwardBacktester.generate_folds()`:

```text
Fold 1:  train [2015-2023]  → test [Q3 2023]
Fold 2:  train [2015-2023]  → test [Q4 2023]
Fold 3:  train [2015-2024]  → test [Q1 2024]
...
```

Cada fold:
1. Separa `df` en train y test por fecha (`train_end` y `test_end` siempre caen en fin de quarter).
2. Aplica normalización sectorial sobre el train (fit) y luego sobre el test (transform only).
3. Entrena todos los agentes y el meta-learner (`train_fold()`).
4. Predice scores sobre el test (`predict_score()`).
5. Simula la cartera (`simulate_portfolio()`).
6. Calcula métricas del fold.
7. Exporta artefactos (scores CSV, auditoría, SHAP, gráficos).

Si no hay precios suficientes despues de la fecha de entrada del fold, ese fold se omite sin entrenar. Si hay precios parciales, la simulacion se calcula solo hasta el ultimo dia disponible.

Si `DAYS_BEFORE_QUARTER_START > 0`, tanto la fecha de entrada como la de salida se adelantan ese número de días (manteniendo el período de tenencia de ~1 quarter).

## 5. Fuentes de datos y almacenamiento local

### 5.1 Finnhub (datos fundamentales y de sentimiento)

| Endpoint | Contenido | Uso principal |
|---|---|---|
| `profile` | Sector, industria, país, capitalización | Sector map, filtrado |
| `basic_financials` | P/E, P/B, ROE, márgenes, series históricas de ratios | Features de valoración e historial |
| `financials_reported_quarterly` | 10-Q: revenue, net_income, EPS, EBITDA, FCF, deuda | Features fundamentales |
| `financials_reported_annual` | 10-K: mismas métricas en base anual | Derivar Q4, tendencias anuales |
| `eps_surprises` | Historial de sorpresa de EPS (expected vs actual) | Beat rate, eps_surprise_pct |
| `recommendation_trends` | Strong buy / buy / hold / sell por mes | Consenso analistas |
| `insider_transactions` | Compras y ventas de directivos (shares, price, date) | insider_net_shares_90d |
| `insider_sentiment` | MSPR (Monthly Share Purchase Ratio) mensual | mspr_3m, mspr_trend |
| `peers` | Lista de tickers comparables | Sector context |
| `quote` | Último precio de cierre | Precio aproximado |

### 5.2 Precios de mercado

Los precios OHLCV se obtienen de Finnhub y se almacenan localmente. Se usan para:
- Construir features técnicas (RSI, MACD, momentum, SMA, Bollinger).
- Calcular el label `forward_return`.
- Simular retornos de cartera en backtest.

### 5.3 Macro

`data_finnhub/_macro/` contiene series temporales descargadas una sola vez:

| Archivo | Variable | Uso |
|---|---|---|
| `sp500.json` | Precio del S&P 500 | Benchmark de retorno para el SectorRotationAgent y backtester |
| `vix.json` | VIX | Disponible como dato de contexto |
| `us10y.json` | Rendimiento bono 10Y | Disponible como dato de contexto |
| `us2y.json` | Rendimiento bono 2Y | `yield_curve = us10y - us2y` (disponible como dato de contexto) |

> **Nota**: las features macro (VIX, yield_curve, sp500_momentum) se eliminaron de los agentes base y del MetaLearner porque son iguales para todos los tickers de un mismo quarter, lo que hacía que el modelo aprendiera timing de mercado en lugar de stock picking.

### 5.4 Consolidated CSV

`data_finnhub/consolidated/{TICKER}.csv` es la fuente de verdad que consume el dataset builder. Contiene la serie trimestral ya unificada, con Q4 derivado, TTM calculado, y ratios preprocesados.

---

## 6. Construcción del dataset

### 6.1 `FundamentalFeatureBuilder`

**Archivo**: [module/steps/step_02_dataset/builders/fundamental.py](module/steps/step_02_dataset/builders/fundamental.py)

Genera features a partir del snapshot trimestral de fundamentales:

- **Crecimiento YoY** (`diff(4)`, diferencia absoluta vs mismo trimestre del año anterior): `revenue_yoy_growth`, `net_income_yoy_growth`, `operating_income_yoy_growth`, `fcf_yoy_growth`, `eps_yoy_growth`, `total_debt_yoy_growth`.
- **Cambios YoY de ratios** (tendencia de calidad): `roa_change_yoy`, `gross_margin_change_yoy`, `current_ratio_change_yoy`.
- **Calidad**: `accruals_ratio` (diferencia entre net_income y CFO como % de activos, mide si los beneficios son "reales"), `earnings_quality` (FCF/net_income), `interest_coverage`.
- **Ratios de cobertura**: deuda/EBITDA, current_ratio, deuda/fondos propios.
- **Flags de riesgo**: `consecutive_losses` (trimestres consecutivos con pérdidas), `revenue_decline`.
- **Piotroski F-score** (Piotroski 2000): composite de 8 señales binarias normalizadas a [0,1]. Mide calidad financiera global en tres dimensiones:
  - *Rentabilidad*: ROA>0, CFO>0, ΔROA>0, calidad de accruals (CFO > Net Income).
  - *Apalancamiento/liquidez*: deuda no crece (< +5% YoY), current ratio mejora.
  - *Eficiencia operativa*: margen bruto mejora, ingresos crecen.
  - Score 1.0 = empresa sana en todas las dimensiones. Uno de los indicadores con mayor respaldo académico para separar winners de losers.
- **Tendencias de slope**: `roe_trend_3y`, `gross_margin_trend_3y`, `net_margin_trend_3y`, `roe_trend_2y`, `net_margin_trend_2y`. Se calculan como la pendiente normalizada de una regresión lineal sobre los últimos 8 trimestres (solo con datos hasta `as_of`).

### 6.2 `TechnicalFeatureBuilder`

**Archivo**: [module/steps/step_02_dataset/builders/technical.py](module/steps/step_02_dataset/builders/technical.py)

Genera features técnicas a partir de la serie de precios hasta `as_of`:

- **Momentum**: retorno a 1, 3, 6, 12 meses.
- **RSI** de 14 y 28 días.
- **MACD**, señal e histograma.
- **Medias móviles**: SMA 20, SMA 50, SMA 200, posición del precio relativa a ellas.
- **Bollinger Bands**: posición del precio.
- **Posición 52 semanas**: distancia al máximo y mínimo anual.
- **Volatilidad realizada**: 20 y 60 días.
- **ATR 14 días**.
- **Volumen relativo**: ratio volumen 20d vs 50d.

### 6.3 `ValuationFeatureBuilder`

**Archivo**: [module/steps/step_02_dataset/builders/valuation.py](module/steps/step_02_dataset/builders/valuation.py)

Cruza el precio actual con los fundamentales:

- **Múltiplos actuales**: `pe_ratio`, `pb_ratio`, `ps_ratio`, `ev_to_ebitda`, `fcf_yield`, `earnings_yield`.
- **Múltiplos vs histórico propio**: `pe_vs_5y_median`, `pb_vs_5y_median`, `ev_ebitda_vs_5y_median`. Se calculan comparando el múltiplo actual con la mediana de los últimos ~5 años del mismo ticker (usando únicamente precios hasta `as_of`).
- **Múltiplos de Bloomberg** (si disponibles en `basic_financials`): `bf_ev_ebitda`, `bf_fcf_yield`, `bf_pe`, `bf_pb_annual`, `bf_ps_ttm`.

La feature `pe_vs_5y_median = PE_actual / PE_mediana_historica - 1` indica si la acción está cara o barata respecto a su propia historia. Un valor de 0.20 significa que el P/E actual es un 20% más caro que su mediana histórica.

### 6.4 `InsiderFeatureBuilder`

**Archivo**: [module/steps/step_02_dataset/builders/insider.py](module/steps/step_02_dataset/builders/insider.py)

Procesa las transacciones de insiders hasta `as_of`:

- `insider_net_shares_90d`: acciones compradas netas en los últimos 90 días (compras - ventas).
- `insider_sell_ratio`: proporción de ventas sobre total de transacciones (>0.7 es red flag).
- `mspr_3m`: MSPR medio de los últimos 3 meses.
- `mspr_trend`: tendencia reciente del MSPR.

### 6.5 `SentimentFeatureBuilder`

**Archivo**: [module/steps/step_02_dataset/builders/sentiment.py](module/steps/step_02_dataset/builders/sentiment.py)

Agrega señales de sentimiento externo:

- **EPS surprises**: `eps_surprise_pct` (último), `beat_rate_4q` (proporción de trimestres en que el EPS real superó la estimación en los últimos 4 quarters), `eps_surprise_avg_4q`.
- **Consenso de analistas**: `analyst_buy_ratio`, `analyst_bearish_score`, `analyst_consensus`, `analyst_dispersion`, `analyst_strong_buy_pct`, `analyst_consensus_change`.
- **MSPR**: `mspr_3m`, `mspr_trend`.
- **Insiders**: `insider_net_shares_90d`, `insider_sell_ratio`.

> **Nota sobre distribución de señales EPS**: `beat_rate_4q`, `eps_surprise_avg_4q` y `eps_revision` son señales de **earnings momentum**, no de sentimiento. Por eso también aparecen como features del `MomentumAgent`.

### 6.6 Normalización sectorial (`SectorNormalizer`)

`apply_sector_normalization()` aplica z-score por sector sobre todas las features numéricas. Los sectores con menos de `SECTOR_ZSCORE_MIN_PEERS=3` empresas no se normalizan (se mantiene el valor original) para evitar estadísticas inestables.


## 7. Agentes del sistema

### 7.1 `BaseAgent`

**Archivo**: [module/agents/base.py](module/agents/base.py)

Clase base que todos los agentes heredan. Provee:

- `fit()` / `predict_score()` con firma estándar.
- `clean_features()`: elimina filas con demasiados NaN, imputa con mediana, elimina features con correlación > `FEATURE_CORR_THRESHOLD=0.85`.
- `clean_features_predict()`: versión para inference (usa estadísticos fijados en train).
- `save_feature_importances()`, `save_diagnostics()`, `record_train_metrics()`: guardan artefactos en `results/agents/{agent_name}/`. Con `save_artifacts=False` (agentes OOF temporales) todas las escrituras a disco se omiten.
- `class_balance()`: calcula proporción Outperform/Underperform para logging.
- `FeatureSelector`: retiene hasta el `TOP_N` configurado por agente según importancia del modelo.

### 7.2 `FundamentalAgent` (XGBoost)

**Archivo**: [module/agents/fundamental.py](module/agents/fundamental.py)

Predice la salud financiera de la empresa. Sus features principales son ratios de rentabilidad (ROE, margen neto, margen operativo), crecimiento (revenue YoY, EPS YoY, FCF YoY), calidad contable (accruals ratio), cambios YoY de ratios clave, el **Piotroski F-score** como indicador composite de calidad financiera global, y tendencias de largo plazo (slope de ROE y márgenes). Las features se comparan contra el sector usando z-scores (`_zsector`); no usa sector dummies one-hot, que aprenden el nivel medio del sector en lugar de la posición relativa del ticker.

Hiperparámetros:
- `n_estimators=400`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample=0.7`, `min_child_weight=5`.
- `FUNDAMENTAL_FEATURE_TOP_N=12` features seleccionadas.

### 7.3 `ValuationAgent` (Gradient Boosting)

**Archivo**: [module/agents/valuation.py](module/agents/valuation.py)

Predice si la empresa está bien valorada en relación a su historia y al sector. Features: P/E, P/B, EV/EBITDA, FCF yield y sus comparativas vs mediana histórica propia. El agente calcula internamente el percentil sectorial de cada múltiplo (con estadísticas fijadas en train, sin leakage).

Hiperparámetros:
- `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`.
- `VALUATION_FEATURE_TOP_N=8` features seleccionadas.

### 7.4 `MomentumAgent` (Random Forest)

**Archivo**: [module/agents/momentum.py](module/agents/momentum.py)

Captura señales de momentum de precio **y de beneficios**. Las features macro se eliminaron para que el agente aprenda stock picking y no timing de mercado.

- **Momentum técnico**: RSI 14/28, MACD, posición relativa a SMA 20/50/200, Bollinger, precio vs 52w-high/low, retorno 1m/3m/6m/12m, volatilidad, ATR, volumen.
- **Earnings momentum** (señales más potentes académicamente): `beat_rate_4q`, `eps_surprise_avg_4q`, `eps_revision`. Derivados: `consistent_beater` (beat_rate ≥ 75%), `earnings_momentum` (sorpresa positiva + revisión al alza).

Hiperparámetros:
- `n_estimators=300`, `max_depth=8`, `min_samples_leaf=5`.
- `min_samples_leaf=5` (vs 10 anterior): hojas más pequeñas → probabilidades más extremas y mejor dispersión de scores. Con 300 árboles el riesgo de overfitting es bajo.
- `MOMENTUM_FEATURE_TOP_N=12` features seleccionadas.

### 7.5 `BearAgent` (Random Forest híbrido + reglas)

**Archivo**: [module/agents/bear.py](module/agents/bear.py)

Detecta riesgo de deterioro financiero. Es un híbrido: combina un modelo Random Forest con una capa de reglas explícitas.

**Capa de reglas**: puntúa flags directamente observables:
- Deuda/EBITDA > 6x.
- Current ratio < 1.
- FCF negativo.
- Deuda creciendo >20% YoY.
- Trimestres consecutivos con pérdidas.
- Insider selling masivo (>70% de transacciones son ventas).
- EPS miss >5%.

**Score final**: `bear_score = BEAR_RULE_WEIGHT * rule_score + BEAR_ML_WEIGHT * ml_score` (por defecto 50/50).

Un `bear_score` alto es **malo** — significa que hay señales de riesgo. El meta-learner lo recibe **invertido** (`1 - bear_score`) para que semánticamente contribuya igual que los demás scores.

Hiperparámetros:
- `n_estimators=200`, `max_depth=6`.
- `BEAR_RULE_WEIGHT=0.5`, `BEAR_ML_WEIGHT=0.5`.
- `BEAR_HARD_THRESHOLD=0.90`: umbral a partir del cual el meta-learner fuerza score 0.05 (ver sección 9).
- `BEAR_FEATURE_TOP_N=8` features seleccionadas.

El BearAgent genera adicionalmente un `flag_report_fold{N}.json` con el detalle de qué flags se activaron por ticker.

### 7.6 `SentimentAgent` (Random Forest)

**Archivo**: [module/agents/sentiment.py](module/agents/sentiment.py)

Captura señales de sentimiento externo: consenso de analistas, MSPR de insiders, sorpresas de EPS y beat rate. Construye derivados como `analyst_net_bullish`, `insider_net_zscore`, `mspr_positive/negative`, `consistent_beater`.

**Comportamiento crítico**: si tras limpiar NaN e imputar quedan menos de 20 filas válidas, el agente **no entrena** y se marca como `is_trained=False`. En predicción, el pipeline usa `sentiment_score=0.5` (neutro) como fallback.

Hiperparámetros:
- `n_estimators=200`, `max_depth=6`, `min_samples_leaf=5`.
- `SENTIMENT_FEATURE_TOP_N=8` features seleccionadas.

### 7.7 `SectorRotationAgent` (Gradient Boosting)

**Archivo**: [module/agents/sector_rotation.py](module/agents/sector_rotation.py)

Agente **top-down** que opera a nivel **sector**, no ticker. Predice si un sector va a superar al S&P 500 el próximo quarter.

**Proceso**:
1. Agrega las features de todos los tickers de cada sector (mediana robusta) por quarter.
2. Construye el label sectorial: 1 si el retorno mediano del sector superó al SPY en ese quarter.
3. Entrena un GBM sobre esas observaciones `(sector × quarter)`.
4. En predicción, agrega las features del universo actual por sector y devuelve `{sector: score}`.
5. El score sectorial se mapea a los tickers de ese sector como `sector_score`.

**Features sectoriales** (medianas de los tickers del sector):
- Fundamentales: ROE, márgenes, crecimiento revenue/EPS, deuda.
- Valoración: P/E, P/B, EV/EBITDA, FCF yield, comparativas vs historial.
- Momentum: retornos 1m/3m/6m/12m, RSI, volatilidad.
- Sentimiento: analyst_buy_ratio, beat_rate_4q, eps_surprise_pct, mspr_3m.

> El `sector_score` alto indica que ese sector tiene momentum y fundamentos favorables para superar al índice. Un ticker excelente en un sector fuerte recibe la combinación más potente.

### 7.8 `MetaLearner` (LR + GBM stacking)

Ver sección 9 para el detalle completo.

---

## 8. Entrenamiento y validación

### 8.1 `train_fold()`

**Archivo**: [module/steps/step_03_training/training.py](module/steps/step_03_training/training.py)

Ejecutado en cada fold del walk-forward:

1. **Instancia** los agentes base desde `build_agents_config()`.
2. **Entrena** los agentes base sobre `df_train_norm` (ya normalizado).
   - El `BearAgent` recibe `y` invertido (`invert_y=True`): su target es detectar Underperform (riesgo).
3. **Entrena el `SectorRotationAgent`** sobre el mismo `df_train_norm` (agrega a nivel sector internamente).
4. **Genera OOF scores** con `generate_oof_scores()`.
5. **Añade `sector_score`** al DataFrame OOF del train.
6. **Entrena el meta-learner** sobre los OOF scores del train (con `sector_score` incluido).
7. **Predice scores** de los agentes base + `sector_score` sobre el test.
8. **Predice el score final** del meta-learner sobre el test.

Devuelve: `(agents_dict, df_test_scored, df_train_with_oof)`.

### 8.2 OOF anti-leakage (`oof.py`)

`generate_oof_scores()` usa KFold temporal (`TimeSeriesSplit`) con `OOF_N_SPLITS=3` para generar scores fuera de muestra sobre el train. Para cada split:

- Entrena una copia del agente con `save_artifacts=False` (no escribe nada a disco).
- Predice en el sub-validation.
- Acumula predicciones.

Los OOF scores resultantes alimentan el meta-learner sin que este vea las predicciones "in-sample" de los agentes base, evitando el leakage de nivel 1 → nivel 2.

### 8.3 `train_full_history()`

Versión del entrenamiento sin split de test: usa todo el DataFrame histórico. También entrena el `SectorRotationAgent`.

### 8.4 Selección de features por agente

`FeatureSelector` (en `base.py`) aplica dos pasos de filtrado — **sin escalar** las features seleccionadas:

1. **Eliminación de redundancia por correlación**: si dos features tienen correlación de Pearson > `FEATURE_CORR_THRESHOLD=0.85`, se elimina la que tenga **menor correlación punto-biserial con el target `y`** (la menos informativa). Si el resultado tiene menos de `min_features`, se revierte el filtro.
2. **Selección top-N por importancia RF**: entrena un RandomForest rápido (100 árboles, `max_depth=5`) sobre las features que superaron el paso 1 y conserva hasta `TOP_N` features con mayor importancia Gini.

**Nota**: no se aplican pesos (ni escalado) a las features seleccionadas. Los modelos de árbol (XGBoost, RF, GBM) son invariantes al escalado monotónico de features, y para la LR del MetaLearner el `StandardScaler` normalizaría los pesos de todas formas. La selección es suficiente.

| Agente | `FEATURE_TOP_N` |
|---|---|
| FundamentalAgent | 12 |
| MomentumAgent | 12 |
| ValuationAgent | 8 |
| BearAgent | 8 |
| SentimentAgent | 8 |
| SectorRotationAgent | 10 |

---

## 9. Meta-learner y stacking

**Archivo**: [module/agents/meta_learner.py](module/agents/meta_learner.py)

### 9.1 Inputs

El meta-learner recibe como features:

| Grupo | Columnas |
|---|---|
| Scores de agentes | `fundamental_score`, `valuation_score`, `momentum_score`, `1 - bear_score`, `sentiment_score`, `sector_score` |
| Rankings sectoriales | `fundamental_score_sector_rank`, `valuation_score_sector_rank`, `momentum_score_sector_rank`, `sentiment_score_sector_rank` (percentil dentro del sector) |
| Sector (one-hot) | `sector_Technology`, `sector_Healthcare`, ... |
| Interacciones | `fund_x_val`, `mom_x_safety`, `fund_x_sentiment`, `mom_x_sentiment`, `sector_x_fundamental`, `sector_x_momentum` |

Las interacciones se calculan internamente en `_prepare()` **después de invertir** `bear_score`:
- `fund_x_val = fundamental_score × valuation_score`
- `mom_x_safety = momentum_score × bear_score_safety` (ambos ya en dirección positiva: alto = bueno)
- `fund_x_sentiment = fundamental_score × sentiment_score`
- `mom_x_sentiment = momentum_score × sentiment_score`
- `sector_x_fundamental = sector_score × fundamental_score` ← ticker fuerte en sector fuerte
- `sector_x_momentum = sector_score × momentum_score`

El `bear_score` se invierte al inicio de `_prepare()` (`1 - bear_score`) → se convierte en "safety score" (alto = empresa segura). La interacción `mom_x_safety` es alta solo cuando el momentum es alto **y** el riesgo es bajo, que es la señal más potente de compra.

### 9.2 Dos modelos en paralelo

**Logistic Regression** (interpretable):
- `Pipeline([StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced")])`.
- Produce coeficientes directamente interpretables guardados en `lr_coefficients_fold{N}.json`.

**Gradient Boosting** (captura no-linealidades):
- `Pipeline([StandardScaler(), GradientBoostingClassifier(...)])`.
- Con `n_estimators=150`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`.
- Sin calibración externa: las probabilidades nativas del GBM se usan directamente para maximizar la dispersión de scores.

### 9.3 Pesos dinámicos LR vs GBM

Los pesos se calibran en función del AUC medio en CV temporal (`TimeSeriesSplit`, 5 splits):

```python
lr_weight  = auc_lr  / (auc_lr + auc_gbm)
gbm_weight = auc_gbm / (auc_lr + auc_gbm)
final_score = lr_weight * lr_proba + gbm_weight * gbm_proba
```

En la práctica el GBM suele tener mayor AUC y recibe más peso.

### 9.4 Hard rule del BearAgent

Si `bear_score >= BEAR_HARD_THRESHOLD=0.90`, el score final se fuerza a `0.05` (Underperform extremo) independientemente de lo que digan los demás agentes. Esto solo aplica en situaciones de riesgo muy alto detectadas por el BearAgent. El soft penalty está desactivado (`BEAR_SOFT_PENALTY=0.0`) para evitar doble penalización.

### 9.5 Score final y cartera

`final_score ∈ [0, 1]`, donde:
- `>= 0.5` → predicción Outperform
- `< 0.5` → predicción Underperform
- `>= PORTFOLIO_MIN_SCORE (0.55)` → candidato a la cartera long

---

## 10. Backtest walk-forward

### 10.1 Generación de folds

`WalkForwardBacktester.generate_folds()` crea ventanas con paso trimestral:

- Cada fold tiene `train_years=8` años de datos y `test_quarters=1` quarter de test.
- `train_end` y `test_end` siempre caen en el último día de su quarter natural.
- No hay solapamiento en el test (cada quarter se evalúa exactamente una vez).

Ejemplo con `TEST_START_YEAR=2025, TEST_START_QUARTER=3`:

```
Fold 1:  train [2017Q2 → 2025Q2]  test [2025Q3]
```

### 10.2 Simulación de cartera (`simulate_portfolio()`)

Para cada fold:

1. Ordena tickers por `final_score` descendente.
2. Filtra por `PORTFOLIO_MIN_SCORE=0.55` (solo candidatos con score ≥ 0.55).
3. Toma como máximo `TOP_N_STOCKS=10`, pero **siempre al menos `TOP_N_STOCKS // 2`** (mínimo 5 con la config por defecto). Si hay candidatos cualificados pero son menos del mínimo, se completa con los siguientes por ranking.
4. Si `SCORE_WEIGHTED_PORTFOLIO=True`, asigna pesos linealmente proporcionales al ranking: el primer ticker pesa el doble que el último.
5. Si `SCORE_WEIGHTED_PORTFOLIO=False`, equiponderado.
6. Simula retorno diario de la cartera usando precios reales del período de test.

### 10.3 Métricas calculadas por fold

`compute_all_metrics()` calcula:

- Retorno acumulado de la estrategia y del benchmark.
- Alpha = retorno estrategia - retorno benchmark.
- Sharpe ratio (anualizado, con `RISK_FREE_RATE=4%`).
- Sortino ratio.
- Maximum drawdown.
- Calmar ratio.

### 10.4 Outputs del backtest por fold

- `results/backtest/fold_{NNN}_{X}Y_metrics.json`: métricas del fold.
- `results/agents/fold_{N}_scores.csv`: scores + explicaciones por ticker.
- `results/agents/fold_{N}_selection_audit.csv / .json`: auditoría de selección.
- `results/agents/fold_{N}_ticker_explanations.csv / .json`: explicaciones SHAP por ticker seleccionado.
- `results/plots/fold_{NNN}_{QUARTER}_performance.png`: curva de riqueza del fold.
- `results/plots/score_dist_fold{N}.png`: distribución de scores finales.
- `results/plots/feat_imp_{agent}_fold{N}.png`: importancias por agente.

### 10.5 Visualizaciones globales al final del backtest

`Visualizer` genera:
- Curva de riqueza acumulada (estrategia vs benchmark, todos los folds encadenados).
- Drawdown acumulado.
- Alpha por fold (barplot).
- Sharpe por fold.

---

## 11. Outputs y artefactos

### 11.1 `results/master_dataset.csv`

Dataset completo: una fila por `(ticker, date)`, todas las features y el label. Útil para inspección, debugging y análisis ad-hoc fuera del pipeline.

### 11.2 `results/agents/fold_{N}_scores.csv`

El archivo más rico de outputs. Una fila por ticker por fold, con:

| Columna | Contenido |
|---|---|
| `year_quarter` | Quarter del período de test (ej. `2025Q3`) |
| `fold` | Número de fold |
| `ticker`, `sector`, `industry` | Identificación |
| `final_score` | Score del meta-learner [0-1] |
| `final_score_raw` | Score antes del prior sectorial y penalizacion por peers |
| `prediccion` | `Outperform` / `Underperform` |
| `confianza` | Alta (>0.7 o <0.3), Moderada, Baja |
| `selected` | `True` si entró en la cartera |
| `rank` | Posición por score descendente |
| `selection_reason` | `selected_above_threshold`, `below_threshold`, etc. |
| `portfolio_weight` | Peso en la cartera (si fue seleccionado) |
| `{agent}_score` | Score de cada agente base (incluido `sector_score`) |
| `{agent}_interpretacion` | Texto en lenguaje natural ("Salud financiera sólida") |
| `{agent}_explicacion` | Factores a favor y en contra con valores reales de las features |
| `sector_peer_count` | Numero de tickers unicos en el sector durante el fold |
| `sector_confidence` | Factor [0,1] aplicado al score final segun peers |
| `retorno_real` | Retorno real del ticker en el período de test |
| `alpha_real` | `retorno_real - benchmark_return` |
| `beat_benchmark` | `True` si el ticker superó al benchmark |
| `label_real` | `1` si fue Outperform real, `0` si no |

### 11.3 `results/agents/fold_{N}_selection_audit.csv / .json`

Auditoría de por qué cada ticker fue seleccionado, cerca del umbral, o descartado. Incluye el score, el umbral, la razón y flags adicionales de la selección.

### 11.4 `results/agents/fold_{N}_ticker_explanations.csv / .json`

Explicaciones SHAP detalladas por ticker: qué features contribuyeron más a su score, con valor absoluto de la feature y valor SHAP. Se exportan para los tickers seleccionados, los cercanos al umbral y los de mayor interés.

### 11.5 `results/agents/{agent}/`

Por cada agente base:
- `feature_importances_fold{N}.csv`: importancias normalizadas.
- `diagnostics_fold{N}.json`: métricas de CV, balance de clases, AUC.
- `train_history.json`: evolución de métricas a lo largo de los folds.
- `shap_global_fold{N}.csv / .json`: importancias SHAP globales.
- `shap_bar_fold{N}.png`: gráfico de barras SHAP.
- `flag_report_fold{N}.json`: (solo BearAgent) detalle de flags activadas por ticker.

### 11.6 `results/agents/meta_learner/`

- `evaluation_fold{N}.json`: accuracy, precision, recall, F1, AUC, confusion matrix.
- `predictions_fold{N}.csv`: score y predicción por ticker.
- `lr_coefficients_fold{N}.json`: coeficientes del LR separados en positive/negative drivers.
- `shap_global_fold{N}.csv / .json`: importancias SHAP sobre el GBM del meta-learner.
- `shap_bar_fold{N}.png`: gráfico SHAP.

### 11.7 `results/pipeline.log`

Log UTF-8 completo de la ejecución. Contiene información de cada agente, fold, selección de cartera, métricas y warnings.

---

## 12. Explicabilidad y ablation

### 12.1 `AgentExplainer` (SHAP)

**Archivo**: [module/steps/step_04_evaluation/explainability.py](module/steps/step_04_evaluation/explainability.py)

Se construye un explainer SHAP sobre el modelo de cada agente:
- Para modelos de árboles (XGBoost, GBM, RF): `shap.TreeExplainer`.
- Para LR: `shap.LinearExplainer`.

Produce:
- **SHAP global**: importancias medias por feature sobre todo el conjunto de test.
- **SHAP local**: contribución de cada feature para un ticker concreto.

`FEATURE_DESCRIPTIONS` es un diccionario con descripciones en lenguaje natural de cada feature, usado en los CSV de explicaciones para que sean legibles por un humano no técnico.

### 12.2 `fold_report.py` — Explicaciones legibles por agente

Cada agente tiene umbrales de texto configurados en `AGENT_LABELS`:
- Score > 0.65 → etiqueta "alta" (p. ej. "Salud financiera sólida").
- Score 0.35-0.65 → etiqueta "media" (p. ej. "Salud financiera aceptable").
- Score < 0.35 → etiqueta "baja" (p. ej. "Debilidades financieras detectadas").

La columna `{agent}_explicacion` en el CSV lista los factores SHAP a favor y en contra del score, con los valores reales de las métricas formateados en lenguaje natural.

### 12.3 `selection_reports.py`

`build_selection_audit_df()` clasifica cada ticker según su posición respecto al umbral:
- `selected_above_threshold`: seleccionado (score ≥ umbral).
- `near_threshold_above/below`: cerca del umbral (±5%).
- `below_threshold`: descartado.

`build_explanation_candidate_tickers()` selecciona qué tickers reciben explicaciones SHAP detalladas: los seleccionados + los más cercanos al umbral + los de mayor interés analítico (hasta 60 tickers).

### 12.4 Ablation study

**Archivo**: [module/steps/step_04_evaluation/ablation.py](module/steps/step_04_evaluation/ablation.py)

Solo se ejecuta si `RUN_ABLATION_STUDY=True`. Para cada fold, reentrena el meta-learner eliminando un agente cada vez y mide la caída de AUC:

```
AUC_baseline - AUC_sin_agente_X = contribución marginal del agente X
```

Exporta `ablation_fold{N}.json` con el AUC de cada configuración. Permite identificar qué agentes aportan más al sistema.

---

## 13. Configuración global

**Archivo**: [environment.py](environment.py) — fuente única de verdad para todos los parámetros.

### 13.1 Flags de ejecución

| Variable | Tipo | Default | Efecto |
|---|---|---|---|
| `SKIP_BACKTEST` | bool | `False` | Salta el walk-forward histórico |
| `FORCE_DOWNLOAD` | bool | `False` | Re-descarga aunque los JSONs existan |
| `RETRY_MISSING_TICKERS` | bool | `False` | Reintenta tickers incompletos |
| `UPDATE_PRICES_ONLY` | bool | `False` | Solo actualiza precios y macro; no consolida ni entrena |
| `RUN_ABLATION_STUDY` | bool | `False` | Activa el estudio de ablación |
| `DOWNLOAD_MAX_WORKERS` | int | `8` | Workers de descarga paralela |
| `FINNHUB_MIN_INTERVAL` | float | `1` | Segundos mínimos entre requests |

### 13.2 Período de análisis

| Variable | Descripción |
|---|---|
| `DOWNLOAD_START_DATE` | Fecha inicial de descarga de datos (`"2015-01-01"`) |
| `TEST_START_YEAR` | Año del primer quarter a predecir |
| `TEST_START_QUARTER` | Trimestre (1-4) del primer quarter a predecir |
| `END_YEAR` | Año fin del histórico |
| `END_QUARTER` | Trimestre fin |
| `DAYS_BEFORE_QUARTER_START` | Días de adelanto respecto al inicio del quarter siguiente para compra y venta. Afecta también al `forward_return` del dataset para alinear el label con el holding period real. |

El `end_date` del backtest es `quarter_end(END_YEAR, END_QUARTER)`.

### 13.3 Parámetros del pipeline ML

| Variable | Valor | Descripción |
|---|---|---|
| `MIN_HISTORY_QUARTERS` | `4` | Mínimo de trimestres por ticker para incluirlo en train |
| `SECTOR_ZSCORE_MIN_PEERS` | `3` | Mínimo de empresas del mismo sector para normalizar |
| `OOF_N_SPLITS` | `3` | Folds KFold internos para OOF del meta-learner |
| `PORTFOLIO_MIN_SCORE` | `0.55` | Umbral mínimo de score para la cartera long |
| `TOP_N_STOCKS` | `10` | Máximo de posiciones en la cartera long |
| `FEATURE_CORR_THRESHOLD` | `0.85` | Umbral de correlación para eliminar features redundantes |
| `SCORE_WEIGHTED_PORTFOLIO` | `True` | Pondera por score (True) o equipondera (False) |
| `WALKFORWARD_TRAIN_LOOKBACK_YEARS` | `8` | Ventana de train del walk-forward en años |
| `RISK_FREE_RATE` | `0.04` | Tasa libre de riesgo para Sharpe/Sortino |
| `SECTOR_CONFIDENCE_PEERS` | `10` | Peers necesarios para confianza sectorial plena |
| `SECTOR_SCORE_PRIOR_BASE` | `0.5` | Base del prior sectorial aplicado al score final |
| `SECTOR_SCORE_PRIOR_WEIGHT` | `0.5` | Peso del `sector_score` en el prior sectorial |
| `SCORE_DISPERSION_MIN_STD` | `0.03` | Dispersión mínima antes de contraer scores hacia 0.5 |

### 13.4 FEATURE_TOP_N por agente

| Variable | Valor | Agente |
|---|---|---|
| `FUNDAMENTAL_FEATURE_TOP_N` | `12` | FundamentalAgent |
| `MOMENTUM_FEATURE_TOP_N` | `12` | MomentumAgent |
| `VALUATION_FEATURE_TOP_N` | `8` | ValuationAgent |
| `BEAR_FEATURE_TOP_N` | `8` | BearAgent |
| `SENTIMENT_FEATURE_TOP_N` | `8` | SentimentAgent |
| `FEATURE_TOP_N` | `8` | Default (otros agentes) |

### 13.5 Hiperparámetros de agentes

**FundamentalAgent (XGBoost)**:
`n_estimators=400`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample=0.7`, `min_child_weight=5`

**ValuationAgent (GBM)**:
`n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`

**MomentumAgent (Random Forest)**:
`n_estimators=300`, `max_depth=8`, `min_samples_leaf=5`

**BearAgent (Random Forest)**:
`n_estimators=200`, `max_depth=6`, `BEAR_RULE_WEIGHT=0.5`, `BEAR_ML_WEIGHT=0.5`, `BEAR_HARD_THRESHOLD=0.90`

**SentimentAgent (Random Forest)**:
`n_estimators=200`, `max_depth=6`, `min_samples_leaf=5`

**SectorRotationAgent (GBM)**:
`n_estimators=200`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`

**MetaLearner (LR + GBM)**:
LR: `C=0.5` | GBM: `n_estimators=150`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`

---

## 14. Comportamientos y decisiones de diseño relevantes

### 14.1 Label sectorial, no vs SPY

El label de entrenamiento compara cada ticker contra la **mediana de su sector** en ese quarter, no contra el SPY. Esto aisla el problema de stock picking del de rotación sectorial. La señal de "¿qué sectores van a superar al índice?" la captura el `SectorRotationAgent` como una capa top-down independiente.

### 14.2 Sin calibración de probabilidades en los agentes base

Los agentes base (XGBoost, GBM, RF) producen probabilidades nativas sin `CalibratedClassifierCV`. La calibración isotónica/Platt comprime las probabilidades hacia el prior de clase (~0.5 en problemas balanceados), reduciendo artificialmente la dispersión de scores. Para que los scores del MetaLearner sean informativos, se necesita la máxima separación posible.

### 14.3 Macro eliminada del pipeline de agentes

Las features macro (VIX, yield_curve, sp500_momentum) son **idénticas para todos los tickers** en un mismo quarter. Incluirlas en los agentes base o en el MetaLearner hace que el modelo aprenda timing de mercado en lugar de stock picking, y domina las importancias SHAP (~49% el VIX). Se eliminaron de todos los agentes.

### 14.4 Bear score invertido en el MetaLearner

El `bear_score` se invierte antes de entrar al MetaLearner: `1 - bear_score`. Así todos los inputs del MetaLearner siguen la misma semántica: score alto = señal positiva.

### 14.5 `TOP_N_STOCKS` es un techo, no una cuota (pero hay un piso)

La cartera tiene como máximo `TOP_N_STOCKS=10` acciones. Si solo pocos tickers superan el umbral, se garantiza un mínimo de `TOP_N_STOCKS // 2 = 5` seleccionando los siguientes por ranking.

### 14.6 Rankings sectoriales en el MetaLearner

El MetaLearner calcula el percentil de cada agente-score dentro del sector (`fundamental_score_sector_rank`, etc.). Esto captura la posición relativa del ticker respecto a sus peers, que es más informativa que el score absoluto.

### 14.7 Penalización por sectores pequeños

Para sectores con pocos peers en el fold, se calcula `sector_confidence` y se reduce el `final_score`.
El factor es:

$$\text{sector\_confidence} = \min\left(1, \sqrt{n\_peers / k}\right)$$

con $k = \text{SECTOR\_CONFIDENCE\_PEERS}$. Esto evita señales inestables cuando el sector tiene poca muestra.

### 14.8 Prior sectorial en el score final

Se aplica un prior suave usando `sector_score`:

$$\text{final\_score} \leftarrow \text{final\_score} \times (b + w \cdot \text{sector\_score})$$

con $b = \text{SECTOR\_SCORE\_PRIOR\_BASE}$ y $w = \text{SECTOR\_SCORE\_PRIOR\_WEIGHT}$. Asi, un sector debil reduce el score, pero no lo anula.

### 14.9 Contraccion de scores con baja dispersion

Si un agente produce scores muy concentrados, se contraen hacia 0.5 con:

$$s' = 0.5 + (s - 0.5) \cdot \min\left(1, \frac{\sigma}{\sigma_{min}}\right)$$

donde $\sigma_{min} = \text{SCORE\_DISPERSION\_MIN\_STD}$. Esto evita que un agente plano meta ruido en el meta-learner.

### 14.10 OOF agents no escriben a disco

Los agentes temporales creados durante la generación de OOF scores se instancian con `save_artifacts=False`. Esto evita que los archivos `diagnostics_fold0`, `feature_importances_fold0`, etc., contaminen los directorios de resultados con artefactos de entrenamiento interno.

### 14.11 Sector dummies eliminados del FundamentalAgent

Los dummies one-hot de sector aprenden el **nivel medio** de cada sector (Technology tiene mayor ROE que Utilities). Esto ya está capturado mejor por los z-scores sectoriales (`_zsector`), que miden la posición relativa del ticker dentro de su sector. Los dummies añadían ruido sin información adicional.

### 14.12 Earnings momentum en el MomentumAgent

`beat_rate_4q`, `eps_surprise_avg_4q` y `eps_revision` son señales de momentum de beneficios, no de sentimiento de analistas. Por eso se añadieron al MomentumAgent: las sorpresas de EPS tienen uno de los efectos de momentum más documentados en la literatura académica (PEAD, post-earnings announcement drift).

### 14.13 `forward_return` alineado con el holding period real

El dataset usa `DAYS_BEFORE_QUARTER_START` para calcular el `forward_return` con los precios de entrada y salida reales, no los cierres de quarter:

```
entry_date = q_end_current + 1 día − DAYS_BEFORE_QUARTER_START
exit_date  = q_end_next    + 1 día − DAYS_BEFORE_QUARTER_START
forward_return = precio(exit_date) / precio(entry_date) − 1
```

Esto garantiza que el label que aprende el modelo mida exactamente el retorno que se obtendría en producción, eliminando el desajuste que existiría si el modelo entrenara con cierres de quarter pero comprara/vendiera con el offset aplicado.

### 14.14 Label agrupa por quarter forward, no por quarter del snapshot

La mediana sectorial para el label se calcula agrupando por el **quarter forward** (el período cuyo retorno se mide), no el quarter del snapshot. Con `DAYS_BEFORE_QUARTER_START=30`:
- Snapshot: `date = Mar 31` → `date.to_period("Q") = Q1`
- Forward period: `Apr 1 − 30d → Jul 1 − 30d` → cae en Q2
- La mediana se calcula agrupando por Q2, comparando retornos del mismo holding period.

Esto previene que tickers con snapshots en diferentes quarters calendario pero con el mismo forward period queden en grupos distintos.

### 14.15 Piotroski F-score como composite de calidad financiera

El Piotroski F-score (Piotroski, *Journal of Accounting Research*, 2000) resume 8 señales binarias en un único valor [0,1]. Su ventaja sobre usar las señales individuales es que actúa como "voto de calidad" robusto: un ticker puede tener ROA alto pero liquidez deteriorada; el composite captura el balance global. El `FeatureSelector` puede seleccionarlo como una de las top-12 features del `FundamentalAgent` incluso cuando las señales individuales están disponibles, porque aporta información de nivel superior.

### 14.16 `FeatureSelector` sin ponderación de features

La versión anterior de `FeatureSelector` multiplicaba las features seleccionadas por su importancia RF normalizada antes de pasarlas al modelo. Esto fue eliminado porque:
- Los modelos de árbol (XGBoost, RF, GBM) son **invariantes al escalado monotónico** de features: multiplicar por una constante positiva no cambia ningún split.
- Para la Logistic Regression del MetaLearner, el `StandardScaler` que precede al modelo normaliza todas las features, deshaciendo los pesos.
- El resultado era complejidad extra sin beneficio funcional, y una fuente potencial de bugs sutiles en reproducibilidad.

---

## 15. Problemas habituales y cómo interpretarlos

### 15.1 `SentimentAgent` informa insuficientes muestras

No significa que el dataset sea pequeño. Significa que, después de limpiar NaN e imputar, quedan menos de 20 filas válidas. Causas habituales:
- Features de analistas no disponibles para muchas fechas históricas.
- Datos de MSPR intermitentes.
- `clean_features()` elimina filas con >50% NaN antes de imputar.

Consecuencia: el agente no entrena y el pipeline usa `sentiment_score=0.5`. El fold continúa.

### 15.2 `SectorRotationAgent` con pocas observaciones sectoriales

Si hay menos de 10 observaciones `(sector × quarter)` disponibles, el agente no entrena y usa `sector_score=0.5`. Esto puede ocurrir al principio del histórico cuando hay pocos quarters. Con el universo S&P 500 completo y datos desde 2015 no debería ocurrir.

### 15.3 Fold con pocas selecciones

Normal si pocos tickers superan `PORTFOLIO_MIN_SCORE=0.55`. El sistema garantiza al menos `TOP_N_STOCKS // 2` selecciones completando con los mejores por ranking. Si sistemáticamente ningún ticker supera el umbral, considerar bajar `PORTFOLIO_MIN_SCORE` a 0.52 o revisar la dispersión de scores en `score_dist_fold{N}.png`.

### 15.4 Mismatch de features en predicción

Los agentes alinean el DataFrame de prediccion con las features vistas en train. Las columnas faltantes se rellenan con 0.0. Un exceso de mismatches indica que el dataset de prediccion tiene features que no estaban en el train — revisar builders.

### 15.5 BearAgent genera scores extremos

Si muchos tickers reciben `bear_score > 0.90`, el meta-learner forzará `final_score=0.05` para todos. Revisar los `flag_report_fold{N}.json` para entender qué reglas se están disparando.

### 15.6 OOF scores ruidosos con pocos datos

Con `OOF_N_SPLITS=3` y un train pequeño, los OOF pueden ser inestables. Aumentar a 5 mejora la estabilidad pero alarga el entrenamiento.

### 15.7 Un ticker tiene precios pero no features

Puede ocurrir si el consolidated CSV existe pero tiene menos de `MIN_HISTORY_QUARTERS=4` trimestres válidos. El ticker se filtra en `get_available_tickers()` y no entra en el dataset.

---

## 16. Lectura rápida por archivo

| Archivo | Responsabilidad |
|---|---|
| [analyzer.py](analyzer.py) | Orquestador principal. Punto de entrada. |
| [environment.py](environment.py) | Todos los parámetros configurables. |
| [module/common/data_router.py](module/common/data_router.py) | Acceso unificado a JSONs, precios, snapshots y macro. |
| [module/steps/step_01_data/pipeline.py](module/steps/step_01_data/pipeline.py) | ETL: descarga, consolidación, filtrado de tickers. |
| [module/steps/step_01_data/clients.py](module/steps/step_01_data/clients.py) | Cliente Finnhub con rate limiting. |
| [module/steps/step_01_data/consolidation.py](module/steps/step_01_data/consolidation.py) | Parseo y unificación JSONs → CSV por ticker. |
| [module/steps/step_02_dataset/dataset.py](module/steps/step_02_dataset/dataset.py) | `build_master_dataset()`. |
| [module/steps/step_02_dataset/builders/fundamental.py](module/steps/step_02_dataset/builders/fundamental.py) | Features de ratios fundamentales y tendencias. |
| [module/steps/step_02_dataset/builders/technical.py](module/steps/step_02_dataset/builders/technical.py) | Features técnicas y momentum de precio. |
| [module/steps/step_02_dataset/builders/valuation.py](module/steps/step_02_dataset/builders/valuation.py) | Múltiplos actuales e históricos. |
| [module/steps/step_02_dataset/builders/insider.py](module/steps/step_02_dataset/builders/insider.py) | Transacciones de insiders y MSPR. |
| [module/steps/step_02_dataset/builders/sentiment.py](module/steps/step_02_dataset/builders/sentiment.py) | EPS surprises, consenso analistas, MSPR. |
| [module/steps/step_02_dataset/builders/sector.py](module/steps/step_02_dataset/builders/sector.py) | Z-score sectorial (`SectorNormalizer`). |
| [module/steps/step_03_training/training.py](module/steps/step_03_training/training.py) | `train_fold()` y `train_full_history()`. |
| [module/steps/step_03_training/oof.py](module/steps/step_03_training/oof.py) | OOF anti-leakage para el meta-learner. |
| [module/steps/step_03_training/agent_config.py](module/steps/step_03_training/agent_config.py) | Configuración e instanciación de agentes. |
| [module/steps/step_04_evaluation/evaluator.py](module/steps/step_04_evaluation/evaluator.py) | `run_walkforward_pipeline()`: loop principal del backtest. |
| [module/steps/step_04_evaluation/backtester.py](module/steps/step_04_evaluation/backtester.py) | `WalkForwardBacktester`: folds y simulación de cartera. |
| [module/steps/step_04_evaluation/metrics.py](module/steps/step_04_evaluation/metrics.py) | Sharpe, Sortino, alpha, drawdown, etc. |
| [module/steps/step_04_evaluation/visualization.py](module/steps/step_04_evaluation/visualization.py) | Gráficos del backtest. |
| [module/steps/step_04_evaluation/explainability.py](module/steps/step_04_evaluation/explainability.py) | SHAP por agente. `FEATURE_DESCRIPTIONS`. |
| [module/steps/step_04_evaluation/fold_report.py](module/steps/step_04_evaluation/fold_report.py) | CSV de scores con explicaciones legibles por fold. |
| [module/steps/step_04_evaluation/selection_reports.py](module/steps/step_04_evaluation/selection_reports.py) | Auditoría de selección y explicaciones por ticker. |
| [module/steps/step_04_evaluation/ablation.py](module/steps/step_04_evaluation/ablation.py) | Estudio de ablación por agente. |
| [module/agents/base.py](module/agents/base.py) | `BaseAgent`: lógica común de entrenamiento, limpieza y exportación. |
| [module/agents/fundamental.py](module/agents/fundamental.py) | `FundamentalAgent` (XGBoost, ratios + z-scores sectoriales). |
| [module/agents/valuation.py](module/agents/valuation.py) | `ValuationAgent` (GBM, múltiplos vs historial y sector). |
| [module/agents/momentum.py](module/agents/momentum.py) | `MomentumAgent` (RF, técnico + earnings momentum). |
| [module/agents/bear.py](module/agents/bear.py) | `BearAgent` (RF + reglas, detecta riesgo). |
| [module/agents/sentiment.py](module/agents/sentiment.py) | `SentimentAgent` (RF, con fallback si datos insuficientes). |
| [module/agents/sector_rotation.py](module/agents/sector_rotation.py) | `SectorRotationAgent` (GBM, top-down, nivel sector vs S&P500). |
| [module/agents/meta_learner.py](module/agents/meta_learner.py) | `MetaLearner` (LR + GBM stacking, pesos dinámicos, hard rule). |

---

## 17. Outputs por fold: archivos y columnas

Este capítulo describe los archivos que se generan por cada fold del backtest y las columnas más importantes de cada uno. El objetivo es que puedas auditar el porqué de una selección y cómo se compone cada score.

### 17.1 `results/agents/fold_{N}_scores.csv`

Archivo principal de resultados por ticker (una fila por ticker en el fold). Incluye scores por agente, explicación legible y resultados reales (si existen).

**Identificación**
- `year_quarter`: quarter evaluado (ej. `2025Q3`).
- `fold`: id del fold.
- `ticker`: símbolo.
- `sector`: sector GICS.
- `industry`: sub-industria GICS.

**Predicción y selección**
- `final_score`: score final del meta-learner (0-1).
- `prediccion`: `Outperform` si `final_score >= 0.5`.
- `confianza`: `Alta` si `abs(final_score - 0.5) > 0.25`.
- `selected`: si el ticker entró en la cartera.
- `rank`: ranking dentro del universo por `final_score`.
- `selection_reason`: motivo de selección (`selected_above_threshold`, `qualified_but_not_selected`, `below_threshold`, `selected_by_fallback`).
- `portfolio_weight`: peso asignado en cartera (si aplica).

**Scores por agente**
- `fundamental_score`, `valuation_score`, `momentum_score`, `bear_score`, `sentiment_score`: score [0,1] por agente.
- `sector_score`: score sectorial del `SectorRotationAgent` (mismo valor para todos los tickers del mismo sector).

**Interpretación textual por agente**
- `fundamental_interpretacion`, `valuation_interpretacion`, `momentum_interpretacion`, `bear_interpretacion`, `sentiment_interpretacion`:
   frase corta basada en umbrales internos (alto/medio/bajo) para el score del agente.

**Explicaciones legibles por agente**
- `fundamental_explicacion`, `valuation_explicacion`, `momentum_explicacion`, `bear_explicacion`, `sentiment_explicacion`:
   explicación en lenguaje natural. Usa SHAP si hay explainer, y si no, usa reglas de fallback.
- `sector_rotation_explicacion`:
   explicación top-down del sector; no evalúa la empresa individual sino el sector completo. Todos los tickers del mismo sector comparten esta explicación.

**Resultados reales (si ya ocurrió el periodo de test)**
- `retorno_real`: retorno observado del ticker durante el periodo del fold.
- `alpha_real`: `retorno_real - retorno_benchmark` (si se dispone de benchmark).
- `beat_benchmark`: `True` si el retorno real supera el benchmark.
- `label_real`: label binario real (1 si el ticker supera la mediana de su sector en el forward quarter).

**Campos adicionales de robustez**
- `final_score_raw`: score antes del prior sectorial y penalizacion por peers.
- `sector_peer_count`: numero de tickers unicos en el sector durante el fold.
- `sector_confidence`: factor [0,1] derivado de `sector_peer_count`.

### 17.2 `results/agents/fold_{N}_selection_audit.csv`

Auditoría detallada de selección de cartera (una fila por ticker).

- `ticker`: símbolo.
- `final_score`: score final del meta-learner.
- `label`: `Outperform` o `Underperform` según el score.
- `selected`: si entró en cartera.
- `rank`: posición por score.
- `selection_reason`: motivo de selección.
- `sector_score`: score de rotación sectorial para el sector del ticker.
- `fundamental_score`, `valuation_score`, `momentum_score`, `bear_score`, `sentiment_score`: scores por agente (cuando estén disponibles).

### 17.3 `results/agents/fold_{N}_ticker_explanations.csv`

Una fila por `ticker × agente` con explicación compacta y drivers.

- `fold`, `ticker`, `rank`, `selected`, `selection_reason`: contexto de selección.
- `agent`: nombre del agente (incluye `sector_rotation` y `meta_learner`).
- `agent_score`: score del agente (para `sector_rotation` es el `sector_score`).
- `agent_label`: `Outperform` / `Underperform` según el score del agente.
- `has_explainer`: `True` si hay SHAP disponible para ese agente.
- `explanation_text`: texto explicativo (SHAP o reglas).
- `favor_factors`: lista de factores a favor (texto plano).
- `contra_factors`: lista de factores en contra (texto plano).
- `top_drivers_json`: JSON con drivers y valores (útil para UI o análisis estructurado).

### 17.4 `results/agents/{agent}/feature_importances_fold{N}.csv`

Importancia de variables del modelo del agente (cuando aplica, p. ej. GBM/RF).

- `(columna sin nombre)`: nombre de la feature.
- `importance`: importancia relativa en el modelo.

**Notas específicas**
- En `agents/sector_rotation/feature_importances_fold{N}.csv`, las features son agregados sectoriales (medianas por sector). La feature `_sector_return` es usada en entrenamiento interno, no como predictor.
- En `agents/fundamental/shap_global_fold{N}.csv` y otros agentes, las columnas con sufijo `_zsector` son z-scores relativos al sector (posición del ticker dentro de su sector).

### 17.5 `results/agents/{agent}/shap_global_fold{N}.csv`

Importancia global de SHAP por feature para el agente.

- `feature`: nombre de la variable.
- `importance`: impacto medio absoluto de SHAP.

### 17.6 `results/agents/meta_learner/predictions_fold{N}.csv`

Predicciones del meta-learner en el fold.

- `ticker`: símbolo.
- `score`: score final.
- `label`: label real (si está disponible en el set de test).

### 17.7 `results/backtest/fold_{NNN}_{X}Y_metrics.json`

Métricas del fold: CAGR, Sharpe, Sortino, max drawdown, alpha, etc.

### 17.8 `results/plots/*_fold{N}.png`

Gráficos de soporte:
- `score_dist_fold{N}.png`: distribución de scores en el fold.
- `feat_imp_{agent}_fold{N}.png`: importancias de features por agente.
- `fold_{NNN}_{QUARTER}_performance.png`: rendimiento de cartera vs benchmark.
