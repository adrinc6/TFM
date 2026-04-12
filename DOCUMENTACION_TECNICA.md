# DOCUMENTACIÓN TÉCNICA

## Sistema Multi-Agente ML Stock Picker — Referencia Completa

---

## 1. Estructura del proyecto

```
TFM/
├── analyzer.py                    # Punto de entrada principal del pipeline
├── analyzer_ticker.py             # Análisis interactivo post-backtest por ticker
├── environment.py                 # Configuración global (~450 líneas de constantes)
├── requirements.txt               # Dependencias del proyecto
├── pyproject.toml                 # Configuración de packaging y pytest
├── tests/
│   └── test_antileakage_and_policy.py  # Tests anti-leakage y política de features
├── data_finnhub/                  # Datos descargados (generado en ejecución)
│   ├── _registry.json             # Estado de descargas por ticker/endpoint
│   ├── sp500_historic.csv         # Miembros históricos del S&P 500
│   ├── _macro/                    # Datos macro (precios S&P 500)
│   ├── consolidated/              # CSVs consolidados por ticker
│   └── <TICKER>/                  # Datos crudos por ticker (JSON)
├── module/
│   ├── __init__.py
│   ├── agents/                    # Agentes ML especializados
│   │   ├── base.py                # Clase base abstracta + FeatureSelector
│   │   ├── fundamental.py         # Agente de calidad fundamental (XGBoost)
│   │   ├── valuation.py           # Agente de valoración (GBM)
│   │   ├── momentum.py            # Agente de momentum técnico (Random Forest)
│   │   ├── bear.py                # Agente de riesgo (reglas + RF)
│   │   ├── sentiment.py           # Agente de sentimiento (Random Forest)
│   │   ├── sector_rotation.py     # Agente de rotación sectorial (GBM)
│   │   └── meta_learner.py        # Meta-learner (LR + GBM stacking)
│   ├── common/                    # Utilidades compartidas
│   │   ├── asof.py                # Filtros point-in-time anti-leakage
│   │   ├── cache.py               # CacheManager basado en archivos
│   │   ├── data_router.py         # DataRouter: acceso centralizado a datos
│   │   ├── feature_controls.py    # Resolución de columnas include/exclude
│   │   └── feature_policy.py      # Política de features ratio/normalizados
│   └── steps/
│       ├── step_01_data/          # ETL: descarga y consolidación
│       │   ├── pipeline.py        # Funciones de entrada (download, prepare)
│       │   ├── clients.py         # FinnhubClient, YahooClient
│       │   ├── downloaders.py     # Descarga paralela + registry
│       │   ├── parsers.py         # Parsing de filings SEC y ratios
│       │   ├── consolidation.py   # Fusión 10-Q + 10-K + basic_financials
│       │   └── registry.py        # Persistencia de estado de descargas
│       ├── step_02_dataset/       # Construcción de features
│       │   ├── dataset.py         # build_master_dataset, build_live_features
│       │   ├── normalization.py   # Z-score sectorial
│       │   └── builders/
│       │       ├── fundamental.py # FundamentalFeatureBuilder
│       │       ├── technical.py   # TechnicalFeatureBuilder
│       │       ├── valuation.py   # ValuationFeatureBuilder
│       │       ├── insider.py     # InsiderFeatureBuilder
│       │       ├── sentiment.py   # SentimentFeatureBuilder
│       │       └── sector.py      # SectorNormalizer
│       ├── step_03_training/      # Entrenamiento de agentes
│       │   ├── training.py        # train_fold: pipeline de un fold
│       │   ├── oof.py             # Generación de scores OOF
│       │   └── agent_config.py    # Factory de configuración de agentes
│       └── step_04_evaluation/    # Evaluación y backtesting
│           ├── evaluator.py       # run_walkforward_pipeline: orquestador
│           ├── backtester.py      # WalkForwardBacktester
│           ├── metrics.py         # Funciones de métricas financieras
│           ├── portfolio_simulator.py # Simulación monetaria USD
│           ├── explainability.py  # AgentExplainer (SHAP)
│           ├── visualization.py   # Visualizer (matplotlib plots)
│           ├── reports.py         # Reportes de texto ASCII
│           ├── fold_report.py     # Reportes CSV por fold
│           ├── selection_reports.py # Auditoría de selección
│           └── ablation.py        # Estudio de ablación de agentes
```

---

## 2. Clases principales y métodos públicos

### 2.1 `BaseAgent` (module/agents/base.py)

Clase base abstracta para todos los agentes del sistema.

| Método | Parámetros | Retorno | Descripción |
|--------|-----------|---------|-------------|
| `fit(X, y, **kwargs)` | `X: pd.DataFrame`, `y: pd.Series` | `BaseAgent` | Entrena el agente (abstracto) |
| `predict_score(X)` | `X: pd.DataFrame` | `pd.Series` | Devuelve scores [0,1] por observación |
| `save_diagnostics(fold, extra)` | `fold: Optional[int\|str]`, `extra: Optional[Dict]` | `None` | Guarda JSON de diagnósticos |
| `save_feature_importances(importances, fold)` | `importances: pd.Series`, `fold: Optional[int\|str]` | `None` | Exporta CSV de importancias |
| `save_predictions(preds_df, fold)` | `preds_df: pd.DataFrame`, `fold: Optional[int\|str]` | `None` | Exporta CSV de predicciones |
| `clean_features(X, y)` | `X: pd.DataFrame`, `y: pd.Series` | `(pd.DataFrame, pd.Series)` | Limpieza para train: drop rows >50% NaN |
| `clean_features_predict(X)` | `X: pd.DataFrame` | `pd.DataFrame` | Limpieza para predict: imputa sin borrar |

**FeatureSelector** (clase interna):

| Método | Parámetros | Retorno | Descripción |
|--------|-----------|---------|-------------|
| `fit(X, y)` | `X: pd.DataFrame`, `y: pd.Series` | `FeatureSelector` | Selecciona features por correlación + importancia |
| `transform(X)` | `X: pd.DataFrame` | `pd.DataFrame` | Aplica selección a nuevos datos |
| `fit_transform(X, y)` | `X: pd.DataFrame`, `y: pd.Series` | `pd.DataFrame` | fit + transform |

### 2.2 `FundamentalAgent` (module/agents/fundamental.py)

| Método | Parámetros | Retorno | Descripción |
|--------|-----------|---------|-------------|
| `fit(X, y, fold, sector_col)` | `X: pd.DataFrame`, `y: pd.Series`, `fold: int`, `sector_col: str` | `FundamentalAgent` | Entrena XGBoost con features fundamentales |
| `predict_score(X, sector_col)` | `X: pd.DataFrame`, `sector_col: str` | `pd.Series` | Scores de calidad fundamental |

**Hiperparámetros** (desde `environment.py`):
- `n_estimators=400`, `max_depth=5`, `learning_rate=0.05`
- `subsample=0.8`, `colsample_bytree=0.7`, `min_child_weight=5`

### 2.3 `ValuationAgent` (module/agents/valuation.py)

| Método | Parámetros | Retorno | Descripción |
|--------|-----------|---------|-------------|
| `fit(X, y, fold)` | `X: pd.DataFrame`, `y: pd.Series`, `fold: int` | `ValuationAgent` | Entrena GBM con features de valoración |
| `predict_score(X)` | `X: pd.DataFrame` | `pd.Series` | Scores de infravaloración |

**Hiperparámetros**: `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`

### 2.4 `MomentumAgent` (module/agents/momentum.py)

| Método | Parámetros | Retorno | Descripción |
|--------|-----------|---------|-------------|
| `fit(X, y, fold)` | `X: pd.DataFrame`, `y: pd.Series`, `fold: int` | `MomentumAgent` | Entrena Random Forest con indicadores técnicos |
| `predict_score(X)` | `X: pd.DataFrame` | `pd.Series` | Scores de momentum |

**Features internas generadas**: `rsi_overbought`, `rsi_oversold`, `above_sma200`, `macd_bullish`, `cross_sma_20_50`, `momentum_quality`, `vol_expansion`, `consistent_beater`, `earnings_momentum`.

**Hiperparámetros**: `n_estimators=300`, `max_depth=8`, `min_samples_leaf=5`

### 2.5 `BearAgent` (module/agents/bear.py)

| Método | Parámetros | Retorno | Descripción |
|--------|-----------|---------|-------------|
| `fit(X, y, fold)` | `X: pd.DataFrame`, `y: pd.Series`, `fold: int` | `BearAgent` | Entrena modelo híbrido reglas+RF |
| `predict_score(X)` | `X: pd.DataFrame` | `pd.Series` | Score de RIESGO [0,1] (alto = más riesgo) |

**Flags de riesgo** (10 señales):

| Flag | Columna | Condición | Peso |
|------|---------|-----------|------|
| debt_growth | total_debt_yoy_growth | > 0.20 | 1.0 |
| debt_equity | debt_equity | > 2.0 | 1.0 |
| debt_ebitda | debt_to_ebitda | > 5.0 | 0.8 |
| fcf_negative | fcf_margin | < 0.0 | 1.0 |
| consecutive_losses | consecutive_losses | >= 2 | 1.2 |
| revenue_decline | revenue_decline | == 1 | 0.8 |
| interest_coverage | interest_coverage | < 2.0 | 0.8 |
| current_ratio_low | current_ratio | < 1.0 | 0.6 |
| insider_selling | insider_sell_ratio | > 0.7 | 0.6 |
| eps_miss | eps_surprise_pct | < -0.10 | 0.5 |

**Pesos de composición**: `BEAR_RULE_WEIGHT=0.5`, `BEAR_ML_WEIGHT=0.5`. Financieros: 60%, Mercado: 40%.

### 2.6 `SentimentAgent` (module/agents/sentiment.py)

| Método | Parámetros | Retorno | Descripción |
|--------|-----------|---------|-------------|
| `fit(X, y, fold)` | `X: pd.DataFrame`, `y: pd.Series`, `fold: int` | `SentimentAgent` | Entrena RF sobre señales de analistas/insiders |
| `predict_score(X)` | `X: pd.DataFrame` | `pd.Series` | Scores de sentimiento |

**Hiperparámetros**: `n_estimators=200`, `max_depth=6`, `min_samples_leaf=5`

### 2.7 `SectorRotationAgent` (module/agents/sector_rotation.py)

| Método | Parámetros | Retorno | Descripción |
|--------|-----------|---------|-------------|
| `fit(X, y, sector_map, spy_prices)` | Ver firma | `SectorRotationAgent` | Entrena GBM a nivel sectorial |
| `predict_sector_scores(X, sector_map)` | `X: pd.DataFrame`, `sector_map: Dict` | `Dict[str, float]` | Scores por sector |
| `map_to_tickers(sector_scores, tickers, sector_map)` | Ver firma | `pd.Series` | Expande scores sector → ticker |

### 2.8 `MetaLearner` (module/agents/meta_learner.py)

| Método | Parámetros | Retorno | Descripción |
|--------|-----------|---------|-------------|
| `fit(X, y, fold, sector_col)` | `X: pd.DataFrame`, `y: pd.Series`, `fold: int`, `sector_col: str` | `MetaLearner` | Entrena stacking LR+GBM sobre scores base |
| `predict_score(X, sector_col)` | `X: pd.DataFrame`, `sector_col: str` | `pd.Series` | Score final meta [0,1] |

**Features de consenso generadas**:
- `agent_score_mean`, `agent_score_std`, `agent_score_max`, `agent_score_min`
- `bullish_agent_count` (agentes con score > `META_BULLISH_SCORE_THRESHOLD`)
- `confidence_weighted_score_mean`
- Interacciones cross: `fund_x_val`, `mom_x_safety`, `fund_x_sentiment`, `sector_x_fundamental`
- `sector_rank` (percentil intra-sectorial)

**Hiperparámetros**: `META_LR_C=0.5`, `META_GBM_N_ESTIMATORS=150`, `META_GBM_MAX_DEPTH=3`, `META_GBM_LEARNING_RATE=0.05`

### 2.9 `DataRouter` (module/common/data_router.py)

| Método | Parámetros | Retorno | Descripción |
|--------|-----------|---------|-------------|
| `load_prices(ticker)` | `ticker: str` | `Optional[pd.DataFrame]` | Carga OHLCV desde CSV |
| `load_consolidated(ticker)` | `ticker: str` | `Optional[pd.DataFrame]` | Carga consolidado con índice report_date |
| `get_sector_map()` | — | `Dict[str, str]` | Mapa {ticker: sector} |
| `get_ticker_info(ticker)` | `ticker: str` | `Dict` | sector, industry, market_cap |
| `get_fundamental_snapshot(ticker, as_of)` | `ticker: str`, `as_of: pd.Timestamp` | `Optional[pd.Series]` | Último reporte hasta as_of |
| `get_price_window(ticker, end, lookback_days)` | Ver firma | `Optional[pd.DataFrame]` | Precios [end-lookback, end] |
| `compute_forward_return_from_snapshot(...)` | Ver firma | `Optional[float]` | Return desde entry hasta holding end |

### 2.10 `CacheManager` (module/common/cache.py)

| Método | Parámetros | Retorno | Descripción |
|--------|-----------|---------|-------------|
| `__init__(cache_dir, context, namespace)` | `cache_dir: str`, `context: dict`, `namespace: str` | — | Crea directorio con hash SHA-256 del contexto |
| `exists(name, ext)` | `name: str`, `ext: str` | `bool` | ¿Existe el artefacto? |
| `save_json(name, payload)` | `name: str`, `payload: dict` | `Path` | Guarda JSON |
| `load_json(name)` | `name: str` | `Optional[dict]` | Carga JSON |
| `save_pickle(name, obj)` | `name: str`, `obj: Any` | `Path` | Guarda pickle (DataFrames) |
| `load_pickle(name)` | `name: str` | `Optional[Any]` | Carga pickle |

### 2.11 `WalkForwardBacktester` (module/steps/step_04_evaluation/backtester.py)

| Método | Parámetros | Retorno | Descripción |
|--------|-----------|---------|-------------|
| `generate_folds(start, end, train_years, test_quarters)` | Ver firma | `List[Tuple]` | Genera pares (train_start, test_end) |
| `simulate_portfolio(scores, prices, top_n)` | Ver firma | `Dict` | Simula portfolio por fold |
| `summarize(fold_results)` | `fold_results: List[Dict]` | `Dict` | Métricas globales |

### 2.12 `AgentExplainer` (module/steps/step_04_evaluation/explainability.py)

| Método | Parámetros | Retorno | Descripción |
|--------|-----------|---------|-------------|
| `fit_explainer(model, X_train)` | `model`, `X_train: pd.DataFrame` | `None` | Inicializa TreeExplainer o KernelExplainer |
| `global_importance()` | — | `pd.Series` | Mean \|SHAP\| por feature |
| `explain_prediction(row, ticker, score, top_n, fold)` | Ver firma | `Dict` | Texto narrativo + top_drivers JSON |
| `save_global_explanation(fold)` | `fold: int` | `None` | Exporta JSON + CSV + plot |

---

## 3. Variables globales y constantes en `environment.py`

### 3.1 Flags de ejecución

| Variable | Tipo | Default | Descripción |
|----------|------|---------|-------------|
| `SKIP_BACKTEST` | bool | `False` | Salta walk-forward, solo ejecuta fold live |
| `FORCE_DOWNLOAD` | bool | `False` | Re-descarga todos los datos |
| `UPDATE_PRICES_ONLY` | bool | `False` | Solo actualiza precios y macro |
| `RETRY_MISSING_TICKERS` | bool | `False` | Reintenta tickers fallidos |
| `RUN_ABLATION_STUDY` | bool | `False` | Ejecuta estudio de ablación |
| `DEBUG_EXPORT_AGENT_INPUTS` | bool | `False` | Exporta CSVs de debug por ticker en agentes |
| `ENABLE_CACHE` | bool | `True` | Habilita caché de artefactos |
| `CACHE_DIR` | str | `"cache"` | Carpeta raíz de caché |
| `CACHE_SCHEMA_VERSION` | int | `2` | Versión de esquema de caché |
| `CACHE_USE_MASTER_DATASET` | bool | `True` | Reutiliza dataset maestro |
| `CACHE_USE_ROUTER_DERIVED` | bool | `True` | Reutiliza datos de DataRouter |
| `CACHE_USE_WALKFORWARD_SUMMARY` | bool | `False` | Reutiliza resumen walk-forward |
| `DOWNLOAD_MAX_WORKERS` | int | `8` | Workers para descarga paralela |
| `FINNHUB_MIN_INTERVAL` | int | `1` | Segundos entre requests Finnhub |

### 3.2 API Keys

| Variable | Tipo | Default | Descripción |
|----------|------|---------|-------------|
| `FINNHUB_API_KEY` | str | `""` | Clave API Finnhub (requiere env var `FINNHUB_API_KEY`) |

### 3.3 Rutas

| Variable | Tipo | Default | Descripción |
|----------|------|---------|-------------|
| `FINNHUB_DATA_DIR` | str | `"data_finnhub"` | Directorio de datos descargados |
| `RESULTS_DIR` | str | `"results"` | Directorio raíz de resultados |
| `AGENTS_RESULTS_DIR` | str | `"results/agents"` | Diagnósticos de agentes |
| `BACKTEST_RESULTS_DIR` | str | `"results/backtest"` | Resultados del backtest |
| `PLOTS_DIR` | str | `"results/plots"` | Gráficos generados |

### 3.4 Universo de tickers

| Variable | Tipo | Default | Descripción |
|----------|------|---------|-------------|
| `TICKERS` | list[str] | ~500 tickers | Lista manual de fallback |
| `USE_DYNAMIC_SP500_UNIVERSE` | bool | `True` | Usa universo dinámico desde CSV |
| `SP500_HISTORIC_CSV_PATH` | str | `"data_finnhub/sp500_historic.csv"` | Miembros históricos |
| `SP500_DYNAMIC_TOP_N` | bool\|int | `False` | TopN por market cap (False = sin recorte) |

### 3.5 Periodo de análisis

| Variable | Tipo | Default | Descripción |
|----------|------|---------|-------------|
| `DOWNLOAD_START_DATE` | str | `"2000-01-01"` | Inicio de descarga de datos |
| `ANALYSIS_START_YEAR` | int | `2015` | Año de inicio del análisis |
| `ANALYSIS_START_QUARTER` | int | `3` | Quarter de inicio |
| `ANALYSIS_END_YEAR` | int | `2026` | Año de fin del análisis |
| `ANALYSIS_END_QUARTER` | int | `2` | Quarter de fin |
| `ANALYSIS_FREQUENCY` | str | `"annual"` | `"quarterly"` o `"annual"` |
| `ANALYSIS_ANNUAL_START_DATE` | str\|None | `None` | Fecha ancla para modo anual |
| `SNAPSHOT_LAG_DAYS` | int | `60` | Días de retraso desde cierre de quarter |
| `ENABLE_FALLBACK_EXTRAPOLATION` | bool | `True` | Extrapola features faltantes |
| `FALLBACK_LOOK_BACK_QUARTERS` | int | `4` | Quarters para extrapolación |
| `HOLDING_PERIOD_MONTHS` | int | `3` | Duración del holding period |

### 3.6 Parámetros ML

| Variable | Tipo | Default | Descripción |
|----------|------|---------|-------------|
| `MIN_HISTORY_QUARTERS` | int | `4` | Mínimo de quarters por ticker |
| `SECTOR_ZSCORE_MIN_PEERS` | int | `3` | Mínimo de peers para Z-score |
| `BASE_AGENTS_LABEL_MODE` | str | `"vs_sector"` | Modo de label (vs_sector o vs_universe) |
| `BASE_LABEL_SECTOR_MIN_PEERS` | int | `3` | Mínimo peers para benchmark sectorial |
| `OOF_N_SPLITS` | int | `3` | Folds internos de OOF |
| `PORTFOLIO_MIN_SCORE` | float | `0.55` | Score mínimo para inclusión |
| `TECHNICAL_LOOKBACK_DAYS` | int | `300` | Ventana de precios para técnicos |

### 3.7 Robustez de scoring

| Variable | Tipo | Default | Descripción |
|----------|------|---------|-------------|
| `SECTOR_CONFIDENCE_PEERS` | int | `10` | Divisor para confianza sectorial |
| `SECTOR_SCORE_PRIOR_BASE` | float | `0.5` | Base del prior sectorial |
| `SECTOR_SCORE_PRIOR_WEIGHT` | float | `0.5` | Peso del prior sectorial |
| `SCORE_DISPERSION_MIN_STD` | float | `0.03` | Umbral de dispersión mínima |
| `SCORE_DISPERSION_MIN_SCALE` | float | `0.35` | Suelo de escala en shrink |

### 3.8 Walk-forward backtesting

| Variable | Tipo | Default | Descripción |
|----------|------|---------|-------------|
| `WALKFORWARD_TRAIN_LOOKBACK_YEARS` | int | `10` | Ventana máxima de train |
| `WALKFORWARD_TRAIN_MIN_YEARS` | int | `5` | Ventana mínima de train |
| `WALKFORWARD_TEST_QUARTERS` | int | `1` | Quarters de test por fold |
| `MIN_TEST_TICKERS_PERCENT` | int | `80` | % mínimo de tickers en test |
| `RISK_FREE_RATE` | float | `0.04` | Tasa libre de riesgo anualizada |
| `TOP_N_STOCKS` | int | `10` | Máximo de stocks en cartera |
| `INITIAL_CAPITAL_USD` | float | `1000.0` | Capital inicial USD |
| `TRANSACTION_FEE_USD` | float | `1.0` | Coste por transacción |
| `SLIPPAGE_PCT` | float | `0.0` | Slippage porcentual |
| `USE_DOLLAR_BACKTEST` | bool | `True` | Ejecutar simulación USD |
| `ALLOW_FRACTIONAL_SHARES` | bool | `True` | Permitir fracciones |
| `RUN_BASELINES` | bool | `True` | Ejecutar baselines |
| `N_RANDOM_BASELINE_SIMS` | int | `100` | Simulaciones random |
| `SCORE_WEIGHTED_PORTFOLIO` | bool | `True` | Ponderar por score |

### 3.9 FeatureSelector

| Variable | Tipo | Default | Descripción |
|----------|------|---------|-------------|
| `FEATURE_CORR_THRESHOLD` | float | `0.85` | Umbral de correlación |
| `FEATURE_SELECTOR_RELEVANCE_WEIGHT` | float | `0.65` | Peso de relevancia con y |
| `FEATURE_TOP_N` | int | `14` | Pre-filtro global |
| `FEATURE_IMPORTANCE_CUTOFF_FRACTION` | float | `0.50` | Fracción de corte |
| `FEATURE_IMPORTANCE_MIN_KEEP` | int | `4` | Mínimo features finales |
| `FEATURE_IMPORTANCE_MAX_KEEP` | int | `10` | Máximo features finales |

### 3.10 Reproducibilidad

| Variable | Tipo | Default | Descripción |
|----------|------|---------|-------------|
| `RANDOM_SEED` | int | `42` | Semilla global |

---

## 4. Flujo de datos: qué recibe y produce cada paso

### 4.1 Step 01 — ETL

```
ENTRADA:
  - Lista de tickers (dinámico desde CSV o manual)
  - DOWNLOAD_START_DATE → fecha actual
  - FINNHUB_API_KEY

PROCESO:
  download_data() → para cada ticker descarga 11 endpoints:
    - company_profile2.json
    - basic_financials.json
    - financials_reported_quarterly.json
    - financials_reported_annual.json
    - eps_surprises.json
    - recommendation_trends.json
    - insider_transactions.json
    - insider_sentiment.json
    - prices.csv (Yahoo Finance OHLCV)
    + _macro/sp500_prices.csv (Yahoo Finance ^GSPC)

  prepare_data() → FinnhubConsolidator:
    - Parsea 10-Q, 10-K y basic_financials
    - Genera serie trimestral continua
    - Rellena Q4 desde 10-K anual
    - Produce un CSV consolidado por ticker

SALIDA:
  - data_finnhub/<TICKER>/*.json (crudos)
  - data_finnhub/<TICKER>/prices.csv
  - data_finnhub/consolidated/<TICKER>.csv
  - data_finnhub/_registry.json (estado)
```

### 4.2 Step 02 — Dataset

```
ENTRADA:
  - Lista de tickers disponibles (con precios + consolidado)
  - DataRouter (acceso a ficheros)
  - Feature builders (Fundamental, Technical, Valuation, Insider, Sentiment)

PROCESO:
  build_master_dataset():
    Para cada ticker:
      1. Carga consolidado + precios
      2. Enriquece fundamentales (YoY growth, Piotroski, tendencias)
      3. Para cada quarter evaluable:
         a. Calcula snapshot_date = quarter_start + SNAPSHOT_LAG_DAYS
         b. Filtra datos hasta snapshot_date (anti-leakage)
         c. Construye features técnicos sobre price_window
         d. Construye features de valoración
         e. Construye features de insider (90d)
         f. Construye features de sentimiento
         g. Calcula forward_return como label
      4. Ensambla registro con todas las features

  _enforce_master_feature_schema():
    - Mantiene solo columnas explícitamente configuradas
    - Añade columnas faltantes como NaN
    - Descarta columnas no reconocidas

SALIDA:
  - pd.DataFrame con MultiIndex (ticker, date)
  - Columnas: ~80 features + metadata (year_quarter, sector, forward_return, etc.)
```

### 4.3 Step 03 — Training

```
ENTRADA:
  - df_train: DataFrame de entrenamiento (observaciones históricas)
  - df_test: DataFrame de test (un quarter)
  - sector_map: {ticker: sector}
  - spy_prices: precios del S&P 500

PROCESO:
  train_fold():
    1. Instancia 6 agentes base + MetaLearner
    2. Aplica normalización sectorial (SectorNormalizer.fit en train, .transform en test)
    3. Entrena cada agente base en train
    4. Genera scores OOF con TimeSeriesSplit (3 folds):
       - Para cada split: train → predict → acumular scores
       - Resultado: DataFrame con columnas *_score por fila de train
    5. Entrena MetaLearner sobre scores OOF + sector dummies
    6. Predice en test:
       a. Cada agente base genera scores
       b. Dispersion shrink: si std < SCORE_DISPERSION_MIN_STD → contrae hacia 0.5
       c. Sector adjustments: prior sectorial × confianza por nº de peers
       d. MetaLearner produce score final
       e. Hard risk gate: si bear_risk ≥ BEAR_HARD_THRESHOLD → score = 0.05
    7. Exporta reportes de features usadas por agente

SALIDA:
  - Dict[str, object]: agentes entrenados
  - pd.DataFrame df_test con columnas *_score añadidas
  - pd.Series final_scores
```

### 4.4 Step 04 — Evaluation

```
ENTRADA:
  - master_df: dataset completo
  - DataRouter, sector_map, spy_prices
  - Configuración walk-forward (folds, train_years, etc.)

PROCESO:
  run_walkforward_pipeline():
    1. Genera folds: [(train_start, train_end, test_end, train_years), ...]
    2. Para cada fold:
       a. Resuelve universo de test (membresía SP500 en esa fecha)
       b. Divide train/test por filing date
       c. Recomputa forward returns con precios reales
       d. Audita leakage
       e. Llama a train_fold()
       f. Selecciona Top-N portfolio (score-weighted)
       g. Si USE_DOLLAR_BACKTEST: simulate_fold_usd() con capital encadenado
       h. Genera explicaciones SHAP
       i. Exporta reportes CSV por fold
    3. Agrega métricas across folds
    4. Ejecuta baselines (EW, Momentum, Value, Random)
    5. Genera visualizaciones + reporte de texto
    6. Si RUN_ABLATION_STUDY: mide impacto de cada agente

SALIDA:
  - results/backtest/walkforward_summary.json
  - results/backtest/folds_detail.csv
  - results/backtest/portfolio_returns.csv
  - results/backtest/equity_curve_usd.csv
  - results/agents/<agent>/diagnostics_<fold>.json
  - results/agents/<agent>/feature_importances_<fold>.csv
  - results/plots/*.png
  - results/pipeline.log
  - results/run_config.json
  - results/data_quality_report.csv
```

---

## 5. Variables internas críticas en funciones clave

### 5.1 `_build_feature_record()` (dataset.py)

```python
feature_date    # = quarter_start + SNAPSHOT_LAG_DAYS — fecha de "análisis"
fund_hist_asof  # = consolidated[consolidated.index <= feature_date] — histórico filtrado
price_window    # = prices[feature_date - lookback : feature_date] — ventana de precios
forward_return  # = (price_exit - price_entry) / price_entry — label
```

### 5.2 `train_fold()` (training.py)

```python
dispersion_scales  # Dict[str, float] — escala por agente según std en train
sector_scores      # Dict[str, float] — score del SectorRotationAgent por sector
sector_confidence  # float = min(1, sqrt(peer_count / SECTOR_CONFIDENCE_PEERS))
sector_prior       # float = SECTOR_SCORE_PRIOR_BASE + SECTOR_SCORE_PRIOR_WEIGHT * sector_score
```

### 5.3 `run_walkforward_pipeline()` (evaluator.py)

```python
folds              # List[Tuple] — (train_start, train_end, test_end, train_years)
entry_date         # pd.Timestamp — fecha de entrada al mercado
exit_date          # pd.Timestamp — fecha de salida (entry + holding_period)
usd_carry_capital  # float — capital encadenado entre folds
usd_fold_contexts  # List[Dict] — resultados de simulación USD por fold
fold_leak_rows     # List[Dict] — auditoría de leakage por fold
```

### 5.4 `simulate_fold_usd()` (portfolio_simulator.py)

```python
exec_buy_date     # Fecha real de compra (primera con precio disponible ≥ entry_date)
exec_sell_date    # Fecha real de venta
buy_price         # Precio ajustado × (1 + slippage)
sell_price        # Precio ajustado × (1 - slippage)
shares            # Cantidad comprada (fraccional o entera según config)
pnl               # sell_price * shares - buy_price * shares - 2 * fee
```

---

## 6. Flags de configuración y sus efectos

### 6.1 Modos de ejecución

| Flag | Efecto |
|------|--------|
| `SKIP_BACKTEST=True` | Solo ejecuta el último fold (live) sin walk-forward histórico |
| `UPDATE_PRICES_ONLY=True` | Descarga solo precios, sin consolidación ni entrenamiento |
| `FORCE_DOWNLOAD=True` | Ignora registry y re-descarga todo |
| `RETRY_MISSING_TICKERS=True` | Re-intenta endpoints fallidos para tickers incompletos |
| `RUN_ABLATION_STUDY=True` | Ejecuta estudio removiendo un agente a la vez |

### 6.2 Caché

| Flag | Efecto |
|------|--------|
| `ENABLE_CACHE=True` | Activa persistencia de artefactos intermedios |
| `CACHE_USE_MASTER_DATASET=True` | Reutiliza dataset maestro si el contexto coincide |
| `CACHE_USE_ROUTER_DERIVED=True` | Reutiliza lista de tickers disponibles y sector map |
| `CACHE_USE_WALKFORWARD_SUMMARY=False` | Si True, salta walk-forward completo si hay resumen previo |

### 6.3 Universo

| Flag | Efecto |
|------|--------|
| `USE_DYNAMIC_SP500_UNIVERSE=True` | Construye universo desde sp500_historic.csv |
| `SP500_DYNAMIC_TOP_N=False` | Sin recorte por market cap; con un entero aplica Top-N |

### 6.4 Frecuencia

| Flag | Efecto |
|------|--------|
| `ANALYSIS_FREQUENCY="quarterly"` | Un fold por trimestre |
| `ANALYSIS_FREQUENCY="annual"` | Un fold por año (holding_period se ajusta a 12 meses) |

### 6.5 Scoring

| Flag | Efecto |
|------|--------|
| `SCORE_WEIGHTED_PORTFOLIO=True` | Ponderación lineal por ranking (top pesa ~2× del bottom) |
| `META_ENABLE_CONSENSUS_FEATURES=True` | Añade features de consenso inter-agente al meta |
| `META_ENABLE_SCORE_RECALIBRATION=False` | Si True, aplica transformación sigmoide al score meta |
| `META_BASE_SCORE_BLEND_WEIGHT=0.55` | Peso del blend con media de scores base |

### 6.6 Backtesting monetario

| Flag | Efecto |
|------|--------|
| `USE_DOLLAR_BACKTEST=True` | Ejecuta simulación completa en USD |
| `ALLOW_FRACTIONAL_SHARES=True` | Permite comprar fracciones de acciones |
| `RUN_BASELINES=True` | Genera baselines de comparación |

---

## 7. Seguridad y validación de datos

### 7.1 Validación de tickers (path-traversal prevention)

El `DataRouter` valida todos los parámetros `ticker` recibidos antes de construir rutas al sistema de archivos. Se aplica:

1. **Regex de formato**: solo se aceptan tickers de 1-10 caracteres alfanuméricos, guión o punto (e.g. `AAPL`, `BRK-B`).
2. **Resolución de ruta**: se verifica que la ruta resultante (`data_dir / ticker`) no escape del directorio de datos mediante `Path.resolve()`.

Cualquier ticker que no pase la validación lanza `ValueError` con un mensaje descriptivo.

### 7.2 Debug condicional de agentes

El flag `DEBUG_EXPORT_AGENT_INPUTS` controla la exportación de CSVs de auditoría por ticker (e.g. datos de AAPL en el SentimentAgent). Por defecto está desactivado para evitar I/O innecesario en producción.
