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
11. [Fold live out-of-sample](#11-fold-live-out-of-sample)
12. [Outputs y artefactos](#12-outputs-y-artefactos)
13. [Explicabilidad y ablation](#13-explicabilidad-y-ablation)
14. [Configuración global](#14-configuración-global)
15. [Comportamientos y decisiones de diseño relevantes](#15-comportamientos-y-decisiones-de-diseño-relevantes)
16. [Problemas habituales y cómo interpretarlos](#16-problemas-habituales-y-cómo-interpretarlos)
17. [Lectura rápida por archivo](#17-lectura-rápida-por-archivo)

---

## 1. Visión general

Este repositorio implementa un pipeline de stock picking multi-agente sobre el universo S&P 500. La idea es combinar varios modelos especializados —cada uno con una "lente" distinta sobre la empresa— para producir una predicción de tipo `Outperform / Underperform` por ticker, evaluarla históricamente con walk-forward, y ejecutar un fold live fuera de muestra con datos reales de mercado.

### Tres capas bien separadas

- **Capa de datos**: descarga desde Finnhub + precios, parseo, consolidación en CSV por ticker, y construcción del dataset maestro con features y labels.
- **Capa de modelado**: cinco agentes especializados por dominio (fundamental, valoración, momentum, riesgo, sentimiento) más un meta-learner que combina sus scores.
- **Capa de evaluación**: backtest walk-forward, explicabilidad SHAP, auditoría de selección, CSV de scores por fold, gráficos, y fold live.

### Principios de diseño

- **Sin look-ahead**: los snapshots de features se construyen con datos estrictamente anteriores a la fecha de observación. Los labels usan precios futuros, pero nunca cruzan el corte temporal del train.
- **Horizonte de predicción**: 63 días de trading (~1 quarter). La pregunta que responde el modelo es "¿este ticker outperformará al S&P 500 (`^GSPC`) en el próximo trimestre?".
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
  ├── [run_walkforward_pipeline()] # Step 4: backtest histórico fold a fold
  └── [run_live_fold()]            # Step 5: predicción real sobre el universo actual
```

### Flags de control principales en `environment.py`

| Flag | Efecto |
|---|---|
| `SKIP_BACKTEST` | Salta el walk-forward histórico, solo ejecuta el fold live |
| `RUN_LIVE_FOLD` | Activa o desactiva el fold live al final del pipeline |
| `FORCE_DOWNLOAD` | Re-descarga todos los JSONs aunque ya existan en disco |
| `RETRY_MISSING_TICKERS` | Reintenta descargar tickers con datos incompletos |
| `RUN_ABLATION_STUDY` | Activa el estudio de ablación (lento, desactivado por defecto) |

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
│   │   ├── quote.json                 # Último precio (para live)
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
        │   ├── dataset.py             # build_master_dataset(), build_live_features()
        │   └── normalization.py       # apply_sector_normalization()
        ├── step_03_training/
        │   ├── agent_config.py        # build_agents_config(): instancia y configura agentes
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
        └── step_05_live/
            ├── live_fold.py           # run_live_fold(): predicción real
            ├── live_prices.py         # download_live_prices(): descarga vía yfinance
            └── returns.py             # qtd_return(): retorno acumulado desde as_of

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

Si `FORCE_DOWNLOAD=False`, los JSONs existentes se reutilizan. El rate limit de Finnhub se respeta con `FINNHUB_MIN_INTERVAL=1` segundo entre requests por ticker.

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
            **macro_features(as_of),
        }
        label = forward_return_63d(prices, as_of)   # label futuro
        records.append((ticker, as_of, features, label))
```

El resultado es un DataFrame indexado por `(ticker, date)` guardado en `results/master_dataset.csv`.

**Anti-leakage de construcción**: cada builder recibe únicamente datos con `index <= as_of`. El label (`forward_return`) usa precios futuros, pero estos nunca se filtran al train porque el split train/test se hace por fecha después de construir el dataset.

### 4.4 Label `forward_return` y binarización

- `forward_return = (precio_{t+63d} - precio_t) / precio_t`
- El label binario `y = 1` si el `forward_return` del ticker **supera el retorno del S&P 500 (`^GSPC`)** en ese mismo período, `y = 0` en caso contrario.
- El retorno del S&P 500 se calcula precio-a-precio entre el último día del quarter anterior y el último día del quarter de la observación, usando `data_finnhub/_macro/sp500.json`.
- Fallback: si el S&P 500 no tiene datos para un período concreto, se compara contra la media del universo de tickers en ese quarter.
- Horizonte: `FORWARD_RETURN_DAYS = 63` días de trading (~1 quarter).

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

### 4.7 Fold live (`live_fold.py`)

Tras el backtest, si `RUN_LIVE_FOLD=True`:

1. Construye features live a `as_of_date` (= fin de `END_QUARTER`) con `build_live_features()`.
2. Entrena agentes sobre **todo** el histórico disponible con `train_full_history()`.
3. Genera scores para todo el universo.
4. Selecciona top/bottom tickers.
5. Descarga precios live vía yfinance (en memoria, sin guardar en disco).
6. Calcula retornos reales y alpha vs SPY.
7. Exporta CSV, JSON, auditoría y explicaciones con la nomenclatura `LIVE` en lugar de un quarter concreto.

---

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
| `peers` | Lista de tickers comparables | Sector context (futuro) |
| `quote` | Último precio de cierre | Precio live aproximado |

### 5.2 Precios de mercado

Los precios OHLCV se obtienen de Finnhub y se almacenan localmente. Se usan para:
- Construir features técnicas (RSI, MACD, momentum, SMA, Bollinger).
- Calcular el label `forward_return`.
- Simular retornos de cartera en backtest y fold live.

### 5.3 Macro

`data_finnhub/_macro/` contiene series temporales descargadas una sola vez:

| Archivo | Variable | Uso |
|---|---|---|
| `sp500.json` | Precio del S&P 500 | Benchmark de retorno, `sp500_momentum_3m`, `sp500_momentum_12m` |
| `vix.json` | VIX | Feature macro de volatilidad implícita |
| `us10y.json` | Rendimiento bono 10Y | Feature macro |
| `us2y.json` | Rendimiento bono 2Y | `yield_curve = us10y - us2y` |

### 5.4 Consolidated CSV

`data_finnhub/consolidated/{TICKER}.csv` es la fuente de verdad que consume el dataset builder. Contiene la serie trimestral ya unificada, con Q4 derivado, TTM calculado, y ratios preprocesados.

---

## 6. Construcción del dataset

### 6.1 `FundamentalFeatureBuilder`

**Archivo**: [module/steps/step_02_dataset/builders/fundamental.py](module/steps/step_02_dataset/builders/fundamental.py)

Genera features a partir del snapshot trimestral de fundamentales:

- **Crecimiento YoY**: `revenue_yoy_growth`, `net_income_yoy_growth`, `operating_income_yoy_growth`, `fcf_yoy_growth`, `eps_yoy_growth`, `total_debt_yoy_growth` (comparando trimestre actual vs mismo trimestre del año anterior, `pct_change(periods=4)`).
- **Calidad**: `accruals_ratio` (diferencia entre net_income y CFO como % de activos), `interest_coverage`.
- **Ratios de cobertura**: deuda/EBITDA, current_ratio, deuda/fondos propios.
- **Flags de riesgo**: trimestres consecutivos con pérdidas, crecimiento de deuda anormal.
- **Tendencias de slope**: `roe_trend_3y`, `gross_margin_trend_3y`, `net_margin_trend_3y`, `roe_trend_2y`, `net_margin_trend_2y`. Se calculan como la pendiente normalizada de una regresión lineal sobre los últimos 8 trimestres (solo con datos hasta `as_of`).

### 6.2 `TechnicalFeatureBuilder`

**Archivo**: [module/steps/step_02_dataset/builders/technical.py](module/steps/step_02_dataset/builders/technical.py)

Genera features técnicas a partir de la serie de precios hasta `as_of`:

- **Momentum**: retorno a 3, 6, 12 meses.
- **RSI** de 14 días.
- **MACD** y señal.
- **Medias móviles**: SMA 50, SMA 200, posición del precio relativa a ellas.
- **Bollinger Bands**: anchura y posición del precio.
- **Volatilidad realizada**: 21 días.
- **Contexto macro**: VIX, yield_curve, sp500_momentum_3m, sp500_momentum_12m (como features del ticker).

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

### 6.5 `SentimentFeatureBuilder`

**Archivo**: [module/steps/step_02_dataset/builders/sentiment.py](module/steps/step_02_dataset/builders/sentiment.py)

Agrega señales de sentimiento externo:

- **EPS surprises**: `eps_surprise_pct` (último), `beat_rate_4q` (proporción de trimestres en que el EPS real superó la estimación en los últimos 4 quarters).
- **Consenso de analistas**: `analyst_strong_buy_pct`, `analyst_buy_pct`, tendencia de recomendación.
- **MSPR**: `mspr_3m` (valor reciente), `mspr_trend` (pendiente en los últimos meses), `mspr_negative` (flag si negativo).

### 6.6 Normalización sectorial (`SectorNormalizer`)

`apply_sector_normalization()` aplica z-score por sector sobre todas las features numéricas. Los sectores con menos de `SECTOR_ZSCORE_MIN_PEERS=3` empresas no se normalizan (se mantiene el valor original) para evitar estadísticas inestables.

### 6.7 `build_live_features()`

Construye una única fila por ticker a fecha `as_of`, sin label. Es lo que alimenta el fold live. Internamente usa los mismos builders que el dataset histórico, pero no calcula `forward_return`.

---

## 7. Agentes del sistema

### 7.1 `BaseAgent`

**Archivo**: [module/agents/base.py](module/agents/base.py)

Clase base que todos los agentes heredan. Provee:

- `fit()` / `predict_score()` con firma estándar.
- `clean_features()`: elimina filas con demasiados NaN, imputa con mediana, elimina features con correlación > `FEATURE_CORR_THRESHOLD=0.85`.
- `clean_features_predict()`: versión para inference (usa estadísticos fijados en train).
- `save_feature_importances()`, `save_diagnostics()`, `record_train_metrics()`: guardan artefactos en `results/agents/{agent_name}/`.
- `class_balance()`: calcula proporción Outperform/Underperform para logging.
- Selector de features: retiene hasta `FEATURE_TOP_N=10` features más relevantes según importancia del modelo.

### 7.2 `FundamentalAgent` (XGBoost)

**Archivo**: [module/agents/fundamental.py](module/agents/fundamental.py)

Predice la salud financiera de la empresa. Sus features principales son ratios de rentabilidad (ROE, margen neto, margen operativo), crecimiento (revenue YoY, EPS YoY, FCF YoY), calidad contable (accruals ratio) y tendencias de largo plazo (slope de ROE y márgenes). Usa XGBoost calibrado con Platt scaling.

Hiperparámetros:
- `n_estimators=400`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample=0.7`, `min_child_weight=5`.

### 7.3 `ValuationAgent` (Gradient Boosting)

**Archivo**: [module/agents/valuation.py](module/agents/valuation.py)

Predice si la empresa está bien valorada en relación a su historia y al mercado. Features: P/E, P/B, EV/EBITDA, FCF yield y sus comparativas vs mediana histórica propia. Un score alto indica que la valoración es atractiva (barata respecto a su historia); un score bajo indica cara o sobrevalorada.

Hiperparámetros:
- `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`.

### 7.4 `MomentumAgent` (Random Forest)

**Archivo**: [module/agents/momentum.py](module/agents/momentum.py)

Captura la dirección técnica de la acción y el contexto macro:

- RSI, MACD, posición relativa a SMA 50/200.
- Momentum a 3, 6, 12 meses.
- Volatilidad realizada.
- VIX, yield_curve, momentum del S&P 500.

Hiperparámetros:
- `n_estimators=300`, `max_depth=8`, `min_samples_leaf=10`.

### 7.5 `BearAgent` (Random Forest híbrido + reglas)

**Archivo**: [module/agents/bear.py](module/agents/bear.py)

Detecta riesgo de deterioro financiero. Es un híbrido: combina un modelo Random Forest con una capa de reglas explícitas.

**Capa de reglas**: puntúa flags directamente observables:
- Deuda/EBITDA > 6x.
- Current ratio < 1.
- FCF negativo.
- Deuda creciendo >50% YoY.
- Trimestres consecutivos con pérdidas.
- Insider selling masivo (>70% de transacciones son ventas).

**Score final**: `bear_score = BEAR_RULE_WEIGHT * rule_score + BEAR_ML_WEIGHT * ml_score` (por defecto 50/50).

Un `bear_score` alto es **malo** — significa que hay señales de riesgo. El meta-learner lo interpreta como factor negativo (`1 - bear_score`).

Hiperparámetros:
- `n_estimators=200`, `max_depth=6`.
- `BEAR_RULE_WEIGHT=0.5`, `BEAR_ML_WEIGHT=0.5`.
- `BEAR_HARD_THRESHOLD=0.90`: umbral a partir del cual el meta-learner fuerza score 0.05 (ver sección 9).

El BearAgent genera adicionalmente un `flag_report_fold{N}.json` con el detalle de qué flags se activaron por ticker.

### 7.6 `SentimentAgent` (Random Forest)

**Archivo**: [module/agents/sentiment.py](module/agents/sentiment.py)

Captura señales de sentimiento externo: consenso de analistas, MSPR de insiders, sorpresas de EPS y beat rate. A diferencia de los demás agentes, el sentimiento tiene disponibilidad irregular: muchos tickers tienen datos históricos escasos de recomendaciones o MSPR.

**Comportamiento crítico**: si tras limpiar NaN e imputar quedan menos de 20 filas válidas, el agente **no entrena** y se marca como `is_trained=False`. En predicción, el pipeline usa `sentiment_score=0.5` (neutro) como fallback. Esto ocurre especialmente en tickers con baja cobertura de analistas.

Hiperparámetros:
- `n_estimators=200`, `max_depth=6`, `min_samples_leaf=8`.

### 7.7 `MetaLearner` (LR + GBM stacking)

Ver sección 9 para el detalle completo.

---

## 8. Entrenamiento y validación

### 8.1 `train_fold()`

**Archivo**: [module/steps/step_03_training/training.py](module/steps/step_03_training/training.py)

Ejecutado en cada fold del walk-forward:

1. **Instancia** los agentes base desde `build_agents_config()`.
2. **Entrena** los agentes base sobre `df_train_norm` (ya normalizado).
   - El `BearAgent` recibe `y` invertido (`invert_y=True`): su target es detectar Underperform (riesgo).
3. **Genera OOF scores** con `generate_oof_scores()`.
4. **Predice scores** de los agentes base sobre el test.
5. **Entrena el meta-learner** sobre los OOF scores del train.
6. **Predice el score final** del meta-learner sobre el test.

Devuelve: `(agents_dict, df_test_scored, df_train_with_oof)`.

### 8.2 OOF anti-leakage (`oof.py`)

`generate_oof_scores()` usa KFold temporal (`TimeSeriesSplit`) con `OOF_N_SPLITS=3` para generar scores fuera de muestra sobre el train. Para cada split:

- Entrena una copia del agente en el sub-train.
- Predice en el sub-validation.
- Acumula predicciones.

Los OOF scores resultantes alimentan el meta-learner sin que este vea las predicciones "in-sample" de los agentes base, evitando el leakage de nivel 1 → nivel 2.

### 8.3 `train_full_history()`

Versión del entrenamiento sin split de test: usa todo el DataFrame histórico. Se llama en el fold live para entrenar los agentes finales antes de predecir sobre el universo actual.

### 8.4 Selección de features por agente

`BaseAgent.clean_features()` aplica tres filtros:

1. **Limpieza de NaN**: elimina filas con >50% de NaN; imputa las restantes con la mediana de columna.
2. **Eliminación de correlaciones altas**: si dos features tienen correlación de Pearson > `FEATURE_CORR_THRESHOLD=0.85`, elimina la de menor varianza.
3. **Selección top-N**: retiene hasta `FEATURE_TOP_N=10` features según importancia del modelo (para agentes basados en árboles) o coeficiente absoluto (para LR).

### 8.5 Calibración de probabilidades

Todos los clasificadores producen probabilidades calibradas, no scores crudos de árbol. La calibración se hace con `CalibratedClassifierCV(method="sigmoid")` en el meta-learner (GBM) y `CalibratedClassifierCV` en los agentes base. Esto es importante para que los scores sean comparables entre agentes y sirvan como features de stacking.

---

## 9. Meta-learner y stacking

**Archivo**: [module/agents/meta_learner.py](module/agents/meta_learner.py)

### 9.1 Inputs

El meta-learner recibe como features:

| Grupo | Columnas |
|---|---|
| Scores de agentes | `fundamental_score`, `valuation_score`, `momentum_score`, `bear_score`, `sentiment_score` |
| Macro | `vix`, `yield_curve`, `sp500_momentum_3m`, `sp500_momentum_12m` |
| Sector (one-hot) | `sector_Technology`, `sector_Healthcare`, ... |
| Interacciones | `fund_x_val`, `mom_minus_bear`, `fund_x_sentiment`, `mom_x_sentiment` |

Las interacciones se calculan internamente en `_prepare()`:
- `fund_x_val = fundamental_score × valuation_score`
- `mom_minus_bear = momentum_score - bear_score`
- `fund_x_sentiment = fundamental_score × sentiment_score`
- `mom_x_sentiment = momentum_score × sentiment_score`

### 9.2 Dos modelos en paralelo

**Logistic Regression** (interpretable):
- `Pipeline([StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced")])`.
- Produce coeficientes directamente interpretables guardados en `lr_coefficients_fold{N}.json`.

**Gradient Boosting** (captura no-linealidades):
- `Pipeline([StandardScaler(), CalibratedClassifierCV(GradientBoostingClassifier(...))])`.
- Con `n_estimators=150`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`.
- Calibrado con sigmoid CV=5.

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
- `>= PORTFOLIO_MIN_SCORE (0.5)` → candidato a la cartera long

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
2. Filtra por `PORTFOLIO_MIN_SCORE=0.5` (solo candidatos con score ≥ 0.5).
3. Toma como máximo `TOP_N_STOCKS=10`.
4. Si `SCORE_WEIGHTED_PORTFOLIO=True`, asigna pesos linealmente proporcionales al ranking: el primer ticker pesa el doble que el último (distribución lineal normalizada a suma 1).
5. Si `SCORE_WEIGHTED_PORTFOLIO=False`, equiponderado.
6. Simula retorno diario de la cartera usando precios reales del período de test.

**Importante**: `TOP_N_STOCKS` es un **techo**, no una obligación. Si solo 3 tickers superan el umbral, la cartera tendrá 3 tickers.

### 10.3 Métricas calculadas por fold

`compute_all_metrics()` calcula:

- Retorno acumulado de la estrategia y del benchmark.
- Alpha = retorno estrategia - retorno benchmark.
- Sharpe ratio (anualizado, con `RISK_FREE_RATE=4%`).
- Sortino ratio.
- Maximum drawdown.
- Calmar ratio.
- Hit rate (% de tickers seleccionados que superaron al benchmark).

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
- Distribución de retornos mensuales.
- Sharpe por fold.

---

## 11. Fold live out-of-sample

**Archivo**: [module/steps/step_05_live/live_fold.py](module/steps/step_05_live/live_fold.py)

### 11.1 Propósito

Ejecuta el modelo como si fuera producción real: entrena con todo el histórico disponible y puntúa el universo actual a fecha `as_of_date`. Produce la selección de cartera para el trimestre en curso.

### 11.2 Proceso detallado

```
1. as_of = END_QUARTER (p. ej. 31 marzo 2026)

2. build_live_features(tickers_ok, as_of)
   → Una fila por ticker con features a fecha as_of, sin label

3. train_full_history(df_historico)
   → Entrena todos los agentes sobre el histórico completo (sin split)

4. Predice scores por agente + meta-learner para cada ticker

5. Selección:
   - Filtra tickers con final_score >= PORTFOLIO_MIN_SCORE
   - Toma los top_n mejores (top_bulls) y los peores (top_bears)

6. download_live_prices(top_bulls + ["SPY"], start=as_of, end=hoy)
   → Precios en memoria desde as_of hasta hoy (para calcular retorno si ya pasó tiempo)

7. Calcula retornos reales (si el período ya ocurrió)
   → bull_returns: retorno acumulado de cada top_bull desde as_of
   → benchmark_return: retorno de SPY en el mismo período
   → alpha = portfolio_return - benchmark_return

8. Exporta:
   → results/agents/LIVE_scores.csv  (una fila por ticker, todos los scores y explicaciones)
   → results/agents/LIVE_selection_audit.csv / .json
   → results/agents/LIVE_ticker_explanations.csv / .json
```

### 11.3 Nomenclatura LIVE

Todos los artefactos del fold live usan el sufijo o prefijo `LIVE` en lugar de un quarter concreto (p. ej. `2025Q3`). Esto permite distinguirlos de los folds históricos y hace que sean autoexplicativos.

### 11.4 Selección de cartera live

El mismo mecanismo que en el backtest: `PORTFOLIO_MIN_SCORE` como umbral mínimo, `TOP_N_STOCKS` como techo, ponderación por score si `SCORE_WEIGHTED_PORTFOLIO=True`.

---

## 12. Outputs y artefactos

### 12.1 `results/master_dataset.csv`

Dataset completo: una fila por `(ticker, date)`, todas las features y el label. Útil para inspección, debugging y análisis ad-hoc fuera del pipeline.

### 12.2 `results/agents/fold_{N}_scores.csv`

El archivo más rico de outputs. Una fila por ticker por fold, con:

| Columna | Contenido |
|---|---|
| `year_quarter` | Quarter del período de test (ej. `2025Q3`) |
| `fold` | Número de fold |
| `ticker`, `sector`, `industry` | Identificación |
| `final_score` | Score del meta-learner [0-1] |
| `prediccion` | `Outperform` / `Underperform` |
| `confianza` | Alta (>0.7 o <0.3), Moderada, Baja |
| `selected` | `True` si entró en la cartera |
| `rank` | Posición por score descendente |
| `selection_reason` | `selected_above_threshold`, `below_threshold`, etc. |
| `portfolio_weight` | Peso en la cartera (si fue seleccionado) |
| `{agent}_score` | Score de cada agente base |
| `{agent}_interpretacion` | Texto en lenguaje natural ("Salud financiera sólida") |
| `{agent}_explicacion` | Factores a favor y en contra con valores reales de las features |
| `retorno_real` | Retorno real del ticker en el período de test |
| `alpha_real` | `retorno_real - benchmark_return` |
| `beat_benchmark` | `True` si el ticker superó al benchmark |
| `label_real` | `1` si fue Outperform real, `0` si no |

### 12.3 `results/agents/fold_{N}_selection_audit.csv / .json`

Auditoría de por qué cada ticker fue seleccionado, cerca del umbral, o descartado. Incluye el score, el umbral, la razón y flags adicionales de la selección.

### 12.4 `results/agents/fold_{N}_ticker_explanations.csv / .json`

Explicaciones SHAP detalladas por ticker: qué features contribuyeron más a su score, con valor absoluto de la feature y valor SHAP. Se exportan para los tickers seleccionados, los cercanos al umbral y los de mayor interés.

### 12.5 `results/agents/{agent}/`

Por cada agente base:
- `feature_importances_fold{N}.csv`: importancias normalizadas.
- `diagnostics_fold{N}.json`: métricas de CV, balance de clases, AUC.
- `train_history.json`: evolución de métricas a lo largo de los folds.
- `shap_global_fold{N}.csv / .json`: importancias SHAP globales.
- `shap_bar_fold{N}.png`: gráfico de barras SHAP.
- `flag_report_fold{N}.json`: (solo BearAgent) detalle de flags activadas por ticker.

### 12.6 `results/agents/meta_learner/`

- `evaluation_fold{N}.json`: accuracy, precision, recall, F1, AUC, confusion matrix.
- `predictions_fold{N}.csv`: score y predicción por ticker.
- `lr_coefficients_fold{N}.json`: coeficientes del LR separados en positive/negative drivers.
- `shap_global_fold{N}.csv / .json`: importancias SHAP sobre el GBM del meta-learner.
- `shap_bar_fold{N}.png`: gráfico SHAP.

### 12.7 `results/pipeline.log`

Log UTF-8 completo de la ejecución. Contiene información de cada agente, fold, selección de cartera, métricas y warnings.

---

## 13. Explicabilidad y ablation

### 13.1 `AgentExplainer` (SHAP)

**Archivo**: [module/steps/step_04_evaluation/explainability.py](module/steps/step_04_evaluation/explainability.py)

Se construye un explainer SHAP sobre el modelo de cada agente:
- Para modelos de árboles (XGBoost, GBM, RF): `shap.TreeExplainer`.
- Para LR: `shap.LinearExplainer`.

Produce:
- **SHAP global**: importancias medias por feature sobre todo el conjunto de test.
- **SHAP local**: contribución de cada feature para un ticker concreto.

`FEATURE_DESCRIPTIONS` es un diccionario con descripciones en lenguaje natural de cada feature, usado en los CSV de explicaciones para que sean legibles por un humano no técnico.

### 13.2 `fold_report.py` — Explicaciones legibles por agente

Cada agente tiene umbrales de texto configurados en `AGENT_LABELS`:
- Score > 0.65 → etiqueta "alta" (p. ej. "Salud financiera sólida").
- Score 0.35-0.65 → etiqueta "media" (p. ej. "Salud financiera aceptable").
- Score < 0.35 → etiqueta "baja" (p. ej. "Debilidades financieras detectadas").

La columna `{agent}_explicacion` en el CSV listan los factores SHAP a favor y en contra del score, con los valores reales de las métricas formateados en lenguaje natural (usando `FEATURE_DESCRIPTIONS`).

### 13.3 `selection_reports.py`

`build_selection_audit_df()` clasifica cada ticker según su posición respecto al umbral:
- `selected_above_threshold`: seleccionado (score ≥ umbral).
- `near_threshold_above/below`: cerca del umbral (±5%).
- `below_threshold`: descartado.

`build_explanation_candidate_tickers()` selecciona qué tickers reciben explicaciones SHAP detalladas: los seleccionados + los más cercanos al umbral + los de mayor interés analítico (hasta 60 tickers).

### 13.4 Ablation study

**Archivo**: [module/steps/step_04_evaluation/ablation.py](module/steps/step_04_evaluation/ablation.py)

Solo se ejecuta si `RUN_ABLATION_STUDY=True`. Para cada fold, reentrena el meta-learner eliminando un agente cada vez y mide la caída de AUC:

```
AUC_baseline - AUC_sin_agente_X = contribución marginal del agente X
```

Exporta `ablation_fold{N}.json` con el AUC de cada configuración. Permite identificar qué agentes aportan más al sistema.

---

## 14. Configuración global

**Archivo**: [environment.py](environment.py) — fuente única de verdad para todos los parámetros.

### 14.1 Flags de ejecución

| Variable | Tipo | Default | Efecto |
|---|---|---|---|
| `SKIP_BACKTEST` | bool | `False` | Salta el walk-forward histórico |
| `FORCE_DOWNLOAD` | bool | `False` | Re-descarga aunque los JSONs existan |
| `RETRY_MISSING_TICKERS` | bool | `False` | Reintenta tickers incompletos |
| `RUN_ABLATION_STUDY` | bool | `False` | Activa el estudio de ablación |
| `RUN_LIVE_FOLD` | bool | `True` | Ejecuta el fold live |
| `DOWNLOAD_MAX_WORKERS` | int | `8` | Workers de descarga paralela |
| `FINNHUB_MIN_INTERVAL` | float | `1` | Segundos mínimos entre requests |

### 14.2 Período de análisis

| Variable | Descripción |
|---|---|
| `DOWNLOAD_START_DATE` | Fecha inicial de descarga de datos (`"2015-01-01"`) |
| `TEST_START_YEAR` | Año del primer quarter a predecir |
| `TEST_START_QUARTER` | Trimestre (1-4) del primer quarter a predecir |
| `END_YEAR` | Año fin del histórico / referencia fold live |
| `END_QUARTER` | Trimestre fin |

El `as_of_date` del fold live es `quarter_end(END_YEAR, END_QUARTER)`.

### 14.3 Parámetros del pipeline ML

| Variable | Valor | Descripción |
|---|---|---|
| `MIN_HISTORY_QUARTERS` | `4` | Mínimo de trimestres por ticker para incluirlo en train |
| `SECTOR_ZSCORE_MIN_PEERS` | `3` | Mínimo de empresas del mismo sector para normalizar |
| `OOF_N_SPLITS` | `3` | Folds KFold internos para OOF del meta-learner |
| `PORTFOLIO_MIN_SCORE` | `0.5` | Umbral mínimo de score para la cartera long |
| `TOP_N_STOCKS` | `10` | Máximo de posiciones en la cartera long |
| `FEATURE_TOP_N` | `10` | Máximo de features retenidas por agente |
| `FEATURE_CORR_THRESHOLD` | `0.85` | Umbral de correlación para eliminar features redundantes |
| `SCORE_WEIGHTED_PORTFOLIO` | `True` | Pondera por score (True) o equipondera (False) |
| `FORWARD_RETURN_DAYS` | `63` | Días de trading para el label forward return |
| `WALKFORWARD_TRAIN_LOOKBACK_YEARS` | `8` | Ventana de train del walk-forward en años |
| `RISK_FREE_RATE` | `0.04` | Tasa libre de riesgo para Sharpe/Sortino |

### 14.4 Hiperparámetros de agentes

**FundamentalAgent (XGBoost)**:
`n_estimators=400`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample=0.7`, `min_child_weight=5`

**ValuationAgent (GBM)**:
`n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`

**MomentumAgent (Random Forest)**:
`n_estimators=300`, `max_depth=8`, `min_samples_leaf=10`

**BearAgent (Random Forest)**:
`n_estimators=200`, `max_depth=6`, `BEAR_RULE_WEIGHT=0.5`, `BEAR_ML_WEIGHT=0.5`, `BEAR_HARD_THRESHOLD=0.90`

**SentimentAgent (Random Forest)**:
`n_estimators=200`, `max_depth=6`, `min_samples_leaf=8`

**MetaLearner (LR + GBM)**:
LR: `C=0.5` | GBM: `n_estimators=150`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`

---

## 15. Comportamientos y decisiones de diseño relevantes

### 15.1 `TOP_N_STOCKS` es un techo, no una cuota

Si solo 3 tickers superan `PORTFOLIO_MIN_SCORE`, la cartera tendrá 3 tickers. El sistema no fuerza seleccionar hasta `TOP_N_STOCKS` si no hay candidatos suficientes con score válido.

### 15.2 BearAgent con `invert_y`

El BearAgent recibe el target invertido (`1 - y`): es entrenado para predecir Underperform (riesgo), no Outperform. Su score se interpreta al revés: un `bear_score=0.8` indica alto riesgo. El meta-learner lo usa directamente (un bear alto penaliza el score final).

### 15.3 Cartera ponderada por score

Con `SCORE_WEIGHTED_PORTFOLIO=True`, el peso del ticker #1 es el doble que el del último. La distribución es lineal y normalizada a suma 1. Esto da más exposición a las predicciones de mayor confianza.

### 15.4 Derivación de Q4 en la consolidación

El consolidador derive Q4 = Anual - (Q1+Q2+Q3) cuando el trimestre no está reportado explícitamente. Esto es crítico para empresas que solo reportan el 10-K sin desglose trimestral completo.

### 15.5 Soft penalty del Bear desactivado

`BEAR_SOFT_PENALTY=0.0` está desactivado para evitar doble penalización: el Bear ya contribuye como feature al meta-learner, que aprende su peso. Forzar una penalización adicional proporcional solapaba ambos mecanismos.

### 15.6 SHAP sobre el GBM del meta-learner, no sobre el LR

La explicabilidad SHAP se calcula sobre el GBM (que captura no-linealidades y tiene mayor AUC típicamente). Los coeficientes del LR se guardan por separado en `lr_coefficients_fold{N}.json` como fuente de interpretabilidad directa.

### 15.7 Nomenclatura de artefactos

Los folds históricos usan sufijos `_fold{N}` y el campo `year_quarter` indica el quarter predicho (ej. `2025Q3`). El fold live usa el tag `LIVE` en todos sus artefactos.

### 15.8 Precios live solo en memoria

`download_live_prices()` descarga precios vía yfinance exclusivamente para calcular retornos reales del período live. No se persisten en disco, solo en RAM durante la ejecución.

---

## 16. Problemas habituales y cómo interpretarlos

### 16.1 `SentimentAgent` informa insuficientes muestras

No significa que el dataset sea pequeño. Significa que, después de limpiar NaN e imputar, quedan menos de 20 filas válidas. Causas habituales:
- Features de analistas no disponibles para muchas fechas históricas.
- Datos de MSPR intermitentes.
- `clean_features()` elimina filas con >50% NaN antes de imputar.

Consecuencia: el agente no entrena y el pipeline usa `sentiment_score=0.5`. El fold continúa.

### 16.2 Fold con pocas selecciones

Normal si pocos tickers superan `PORTFOLIO_MIN_SCORE=0.5`. No es un error. Puede ocurrir si el mercado tiene momentum bajista y los scores del meta-learner son globalmente bajos.

### 16.3 Mismatch de features en predicción

Los agentes alinean el DataFrame de predicción con las features vistas en train. Las columnas faltantes se rellenan con 0.0 (valor neutro). Un exceso de mismatches indica que el live dataset tiene features que no estaban en el train — revisar builders.

### 16.4 BearAgent genera scores extremos

Si muchos tickers reciben `bear_score > 0.90`, el meta-learner forzará `final_score=0.05` para todos. Revisar los `flag_report_fold{N}.json` para entender qué reglas se están disparando.

### 16.5 OOF scores ruidosos con pocos datos

Con `OOF_N_SPLITS=3` y un train pequeño, los OOF pueden ser inestables. Aumentar a 5 mejora la estabilidad pero alarga el entrenamiento.

### 16.6 Un ticker tiene precios pero no features

Puede ocurrir si el consolidated CSV existe pero tiene menos de `MIN_HISTORY_QUARTERS=4` trimestres válidos. El ticker se filtra en `get_available_tickers()` y no entra en el dataset.

---

## 17. Lectura rápida por archivo

| Archivo | Responsabilidad |
|---|---|
| [analyzer.py](analyzer.py) | Orquestador principal. Punto de entrada. |
| [environment.py](environment.py) | Todos los parámetros configurables. |
| [module/common/data_router.py](module/common/data_router.py) | Acceso unificado a JSONs, precios, snapshots y macro. |
| [module/steps/step_01_data/pipeline.py](module/steps/step_01_data/pipeline.py) | ETL: descarga, consolidación, filtrado de tickers. |
| [module/steps/step_01_data/clients.py](module/steps/step_01_data/clients.py) | Cliente Finnhub con rate limiting. |
| [module/steps/step_01_data/consolidation.py](module/steps/step_01_data/consolidation.py) | Parseo y unificación JSONs → CSV por ticker. |
| [module/steps/step_02_dataset/dataset.py](module/steps/step_02_dataset/dataset.py) | `build_master_dataset()` y `build_live_features()`. |
| [module/steps/step_02_dataset/builders/fundamental.py](module/steps/step_02_dataset/builders/fundamental.py) | Features de ratios fundamentales y tendencias. |
| [module/steps/step_02_dataset/builders/technical.py](module/steps/step_02_dataset/builders/technical.py) | Features técnicas y momentum. |
| [module/steps/step_02_dataset/builders/valuation.py](module/steps/step_02_dataset/builders/valuation.py) | Múltiplos actuales e históricos. |
| [module/steps/step_02_dataset/builders/insider.py](module/steps/step_02_dataset/builders/insider.py) | Transacciones de insiders. |
| [module/steps/step_02_dataset/builders/sentiment.py](module/steps/step_02_dataset/builders/sentiment.py) | EPS surprises, consenso analistas, MSPR. |
| [module/steps/step_02_dataset/builders/sector.py](module/steps/step_02_dataset/builders/sector.py) | Z-score sectorial. |
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
| [module/steps/step_05_live/live_fold.py](module/steps/step_05_live/live_fold.py) | Fold live: predicción real + retornos + exportación. |
| [module/steps/step_05_live/live_prices.py](module/steps/step_05_live/live_prices.py) | Descarga de precios live vía yfinance. |
| [module/steps/step_05_live/returns.py](module/steps/step_05_live/returns.py) | `qtd_return()`: retorno acumulado desde as_of hasta hoy. |
| [module/agents/base.py](module/agents/base.py) | `BaseAgent`: lógica común de entrenamiento, limpieza y exportación. |
| [module/agents/fundamental.py](module/agents/fundamental.py) | `FundamentalAgent` (XGBoost). |
| [module/agents/valuation.py](module/agents/valuation.py) | `ValuationAgent` (GBM). |
| [module/agents/momentum.py](module/agents/momentum.py) | `MomentumAgent` (Random Forest). |
| [module/agents/bear.py](module/agents/bear.py) | `BearAgent` (RF + reglas, detecta riesgo). |
| [module/agents/sentiment.py](module/agents/sentiment.py) | `SentimentAgent` (RF, con fallback si datos insuficientes). |
| [module/agents/meta_learner.py](module/agents/meta_learner.py) | `MetaLearner` (LR + GBM stacking, pesos dinámicos, hard rule). |
