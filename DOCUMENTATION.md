# Documentación — TFM: Multi-Agente ML Stock Picker

> Descripción completa del repositorio: arquitectura, flujo de ejecución y decisiones de diseño.

---

## Índice

1. [Visión general](#1-visión-general)
2. [Estructura de archivos](#2-estructura-de-archivos)
3. [Flujo de ejecución](#3-flujo-de-ejecución)
4. [Configuración global — `environment.py`](#4-configuración-global--environmentpy)
5. [Descarga de datos](#5-descarga-de-datos)
6. [Consolidación de fundamentales](#6-consolidación-de-fundamentales)
7. [DataRouter](#7-datarouter)
8. [Feature Engineering](#8-feature-engineering)
9. [Dataset maestro](#9-dataset-maestro)
10. [Walk-Forward Backtest](#10-walk-forward-backtest)
    - [10.1 Generación de folds](#101-generación-de-folds)
    - [10.2 Selección de features](#102-selección-de-features)
    - [10.3 Agente Fundamental](#103-agente-fundamental)
    - [10.4 Agente de Valoración](#104-agente-de-valoración)
    - [10.5 Agente de Momentum](#105-agente-de-momentum)
    - [10.6 Agente de Riesgo (Bear)](#106-agente-de-riesgo-bear)
    - [10.7 Meta-Learner](#107-meta-learner)
    - [10.8 OOF anti-leakage](#108-oof-anti-leakage)
    - [10.9 Backtester y métricas](#109-backtester-y-métricas)
    - [10.10 Explicabilidad SHAP](#1010-explicabilidad-shap)
11. [Fold live out-of-sample](#11-fold-live-out-of-sample)
12. [Visualizaciones](#12-visualizaciones)
13. [Parámetros clave de `environment.py`](#13-parámetros-clave-de-environmentpy)

---

## 1. Visión general

El sistema es un **pipeline multi-agente de selección de acciones** basado en Machine Learning. Combina cuatro agentes especializados mediante un meta-learner de stacking para generar señales de inversión en acciones del S&P 500.

### Agentes

| Agente | Modelo | Señal |
|---|---|---|
| FundamentalAgent | XGBoost calibrado | Salud financiera (rentabilidad, solvencia, crecimiento) |
| ValuationAgent | GBM calibrado | Infravaloración relativa (múltiplos vs. historial y sector) |
| MomentumAgent | Random Forest calibrado | Momentum técnico + contexto macro |
| BearAgent | RF híbrido (reglas + ML) | Riesgo de deterioro (señal inversa) |
| MetaLearner | LR + GBM stacking | Combina los cuatro scores en una decisión final |

### Filosofía de diseño

- **Walk-forward temporal**: toda la evaluación respeta el orden cronológico; no hay leakage de datos futuros en train.
- **OOF (Out-Of-Fold)**: el meta-learner se entrena sobre predicciones fuera de muestra de cada agente.
- **Selección de features por fold**: cada agente selecciona sus features usando solo datos de train del fold correspondiente.
- **Modo headless**: todos los gráficos se guardan en disco (backend Agg); no se abren ventanas.

---

## 2. Estructura de archivos

```
TFM/
├── analyzer.py              # Punto de entrada principal
├── run_pipeline.py          # Alternativa con argumentos CLI
├── environment.py           # Configuración central (única fuente de verdad)
├── requirements.txt
│
├── data/                    # Datos de entrada (no versionados)
│   ├── prices/              # OHLCV diario por ticker (CSV)
│   ├── consolidated/        # Fundamentales trimestrales por ticker (CSV)
│   ├── insider/             # Transacciones insider por ticker (CSV)
│   ├── analyst/             # Estimaciones de analistas por ticker (CSV)
│   ├── macro/               # macro_consolidated.csv (VIX, yield curve, SP500)
│   └── companies.csv        # Universo: ticker, sector, industria, market cap
│
├── results/                 # Salidas del pipeline
│   ├── pipeline.log
│   ├── master_dataset.csv
│   ├── agents/              # Diagnósticos, importancias, predicciones por agente
│   ├── backtest/            # Métricas y portfolio por fold
│   └── plots/               # Gráficos (PNG)
│
└── module/
    ├── data_router.py       # Carga y enruta datos
    ├── feature_engineering.py  # Builders de features y SectorNormalizer
    ├── backtester.py        # WalkForwardBacktester
    ├── visualizer.py        # Generación de gráficos
    ├── explainer.py         # SHAP wrappers
    ├── fetcher.py           # Descarga de datos (yfinance)
    │
    ├── agents/
    │   ├── base_agent.py    # BaseAgent + FeatureSelector
    │   ├── fundamental_agent.py
    │   ├── valuation_agent.py
    │   ├── momentum_agent.py
    │   ├── bear_agent.py
    │   └── meta_learner.py
    │
    └── pipeline/
        ├── data_ops.py      # ETL: descarga, consolidación, filtrado de tickers
        ├── dataset_builder.py  # Construcción del dataset maestro
        ├── trainer.py       # Entrenamiento por fold
        ├── evaluator.py     # Walk-forward loop + SHAP + backtest
        ├── live_fold.py     # Predicción out-of-sample con precios reales
        └── ablation.py      # Estudio de ablación por agente
```

---

## 3. Flujo de ejecución

```
analyzer.py / run_pipeline.py
  │
  ├─ 1. fetch_tickers()                 → lista de tickers del universo
  ├─ 2. download_data()                 → descarga OHLCV, fundamentales, insider, analyst
  ├─ 3. DataRouter()                    → enrutador de datos
  ├─ 4. Builders de features            → FundamentalFeatureBuilder, TechnicalFeatureBuilder,
  │                                        ValuationFeatureBuilder, InsiderFeatureBuilder
  ├─ 5. build_master_dataset()          → DataFrame con todas las features y labels
  ├─ 6. [diagnóstico sectorial]         → distribución por sector
  ├─ 7. run_walkforward_pipeline()      → bucle walk-forward:
  │     para cada fold:
  │       ├─ apply_sector_normalization()  → Z-scores sectoriales (solo en train)
  │       ├─ train_fold()                 → entrena los 4 agentes + meta-learner
  │       │   ├─ FundamentalAgent.fit()
  │       │   ├─ ValuationAgent.fit()
  │       │   ├─ MomentumAgent.fit()
  │       │   ├─ BearAgent.fit()
  │       │   └─ MetaLearner.fit() (con OOF scores)
  │       ├─ predict_score() por agente  → scores en test
  │       ├─ backtester.run()            → métricas portfolio (Sharpe, Sortino, etc.)
  │       └─ explain_top_tickers()       → SHAP por ticker
  └─ 8. run_live_fold()                 → predicción en datos recientes
```

---

## 4. Configuración global — `environment.py`

Fuente única de verdad. Todos los módulos importan sus parámetros desde aquí.

### Flags de ejecución

| Variable | Valor defecto | Descripción |
|---|---|---|
| `SKIP_BACKTEST` | `False` | Si True, omite el walk-forward y solo ejecuta el fold live |
| `FORCE_DOWNLOAD` | `False` | Si True, re-descarga todos los datos aunque ya existan |

### Período de análisis

| Variable | Descripción |
|---|---|
| `START_DATE` | Inicio del histórico de precios y fundamentales |
| `END_DATE` | Fin del histórico / inicio del fold live |
| `DAYS_UPDATE` | Días tras los que un ticker se considera desactualizado |

### Parámetros del pipeline ML

| Variable | Descripción |
|---|---|
| `FORWARD_RETURN_DAYS` | Horizonte del label (63 días ≈ 1 quarter) |
| `MIN_HISTORY_QUARTERS` | Mínimo de trimestres por ticker para incluirlo en train |
| `SECTOR_ZSCORE_MIN_PEERS` | Mínimo de empresas del mismo sector para Z-score sectorial |
| `OOF_N_SPLITS` | Folds KFold internos para OOF del meta-learner |

### Walk-forward

| Variable | Descripción |
|---|---|
| `WALKFORWARD_TRAIN_YEARS` | Años mínimos de train (prueba 1Y, 2Y, 3Y…) |
| `WALKFORWARD_TEST_QUARTERS` | Trimestres de test por fold (siempre 1) |
| `RISK_FREE_RATE` | Tasa libre de riesgo anualizada para Sharpe/Sortino |
| `TOP_N_STOCKS` | Acciones seleccionadas en el portfolio long por fold |

---

## 5. Descarga de datos

**Módulo**: `module/fetcher.py`, `module/pipeline/data_ops.py`

- `fetch_tickers()` devuelve la lista de tickers del universo (definida en `fetcher.py`).
- `download_data()` descarga, para cada ticker:
  - Precios OHLCV diarios → `data/prices/TICKER.csv`
  - Fundamentales trimestrales → `data/consolidated/TICKER.csv` (vía `prepare_data`)
  - Transacciones insider → `data/insider/TICKER.csv`
  - Estimaciones de analistas → `data/analyst/TICKER.csv`
- Los datos macro (VIX, yield curve, SP500) se descargan a `data/macro/macro_consolidated.csv`.
- Con `FORCE_DOWNLOAD=False`, se reutilizan los datos si tienen menos de `DAYS_UPDATE` días.

---

## 6. Consolidación de fundamentales

**Módulo**: `module/pipeline/data_ops.py` → `prepare_data()`

Transforma los datos financieros crudos de cada ticker en un CSV trimestral con ratios calculados:
- Rentabilidad: ROE, ROA, ROI, ROIC, márgenes neto/bruto/EBITDA/operativo/FCF
- Solvencia: debt/equity, debt/EBITDA, interest coverage, current ratio, quick ratio
- Crecimiento YoY: revenue, net income, EPS, FCF, operating income, deuda total
- Calidad contable: accruals ratio, capex/revenue, trimestres consecutivos con pérdidas
- EPS base para valoración

---

## 7. DataRouter

**Módulo**: `module/data_router.py`

Punto central de carga de datos. Sirve datos de disco a los builders y al dataset.

```python
router = DataRouter(data_dir=DATA_DIR)
```

Métodos principales:
- `load_prices(ticker)` — OHLCV diario
- `load_consolidated(ticker)` — fundamentales trimestrales
- `load_insider(ticker)` — transacciones insider
- `load_analyst(ticker)` — estimaciones de analistas
- `load_macro()` — contexto macro (cacheado)
- `get_sector_map()` — `{ticker: sector}` desde `companies.csv`
- `enrich_with_sector(df)` — añade columnas `sector` e `industry`

---

## 8. Feature Engineering

**Módulo**: `module/feature_engineering.py`

Cuatro builders independientes, cada uno genera features para su dominio:

### FundamentalFeatureBuilder
Ratios financieros desde el CSV consolidado. Calcula crecimientos YoY, banderas de pérdidas consecutivas, caída de revenue.

### TechnicalFeatureBuilder
Indicadores técnicos desde precios OHLCV:
- RSI (14d, 28d), MACD, Bollinger Bands
- SMAs (20, 50, 200) como distancia % al precio
- Momentum (1m, 3m, 6m, 12m)
- Volatilidad realizada (20d, 60d), ATR 14d
- Volumen relativo (ratio 20/50 días)
- Posición relativa a máximo/mínimo de 52 semanas

### ValuationFeatureBuilder
Múltiplos (P/E, P/B, P/S, EV/EBITDA, FCF yield, earnings yield), comparativa vs. mediana propia de 5 años y señales de analistas (EPS surprise, revisiones).

### InsiderFeatureBuilder
Compras/ventas netas de insiders en ventana de 90 días, ratio de ventas.

### SectorNormalizer
Calcula Z-scores sectoriales para los ratios fundamentales. **Se fittea solo en datos de train** para evitar leakage. Las columnas resultantes tienen sufijo `_zsector`.

---

## 9. Dataset maestro

**Módulo**: `module/pipeline/dataset_builder.py` → `build_master_dataset()`

Para cada ticker genera observaciones trimestrales (una por fecha de cierre de trimestre) que combinan:
- Features fundamentales del trimestre anterior
- Features técnicos al cierre de ese trimestre
- Múltiplos de valoración (precio × fundamentales)
- Señales de insiders (ventana 90 días previa)
- Snapshot macro más reciente disponible
- **Label**: retorno forward a `FORWARD_RETURN_DAYS` días de trading; binarizado como 1 (Outperform) si supera la mediana del universo ese trimestre, 0 (Underperform) en caso contrario.

El resultado es un DataFrame multi-índice `(ticker, date)` guardado en `results/master_dataset.csv`.

---

## 10. Walk-Forward Backtest

**Módulo**: `module/pipeline/evaluator.py` → `run_walkforward_pipeline()`

### 10.1 Generación de folds

El backtester genera folds temporales expansivos. Para cada año de ventana de train (1Y, 2Y, 3Y…) y cada trimestre disponible como test:
- **Train**: datos hasta el inicio del trimestre de test
- **Test**: ese trimestre exacto
- El mínimo de train es `WALKFORWARD_TRAIN_YEARS` años

### 10.2 Selección de features

**Módulo**: `module/agents/base_agent.py` → `FeatureSelector`

Antes del entrenamiento de cada agente (excepto BearAgent), se aplica una selección de features en dos pasos, usando **solo datos de train** del fold:

1. **Eliminación por correlación**: se elimina una feature de cada par con |corr| > 0.90, reduciendo redundancia.
2. **Selección por importancia**: se entrena un Random Forest rápido (100 árboles) y se conserva el top 70% de features por importancia Gini. Se garantiza un mínimo de features (5–8 según el agente).

La selección se aplica también en predicción usando `selector.transform()`, que respeta exactamente las features seleccionadas en train.

Los detalles de la selección (features eliminadas y seleccionadas) se guardan en `diagnostics_fold{N}.json` de cada agente.

### 10.3 Agente Fundamental

**Modelo**: XGBoost calibrado (Isotonic Regression, 5-fold CV)

**Features principales**:
- Rentabilidad: ROE, ROA, ROI, ROIC, márgenes
- Solvencia: debt/equity, debt/EBITDA, interest coverage, current ratio
- Crecimiento YoY: revenue, net income, EPS, FCF, operating income, deuda
- Calidad: accruals ratio, capex/revenue, pérdidas consecutivas
- Sector: Z-scores sectoriales (`*_zsector`) + dummies one-hot (`sector_*`)

**Hiperparámetros** (`environment.py`): 400 estimadores, profundidad 5, lr 0.05, subsample 0.8, colsample 0.7, min_child_weight 5. `scale_pos_weight` ajustado por desequilibrio de clases.

### 10.4 Agente de Valoración

**Modelo**: Gradient Boosting calibrado (Sigmoid, 5-fold CV)

**Features principales**:
- Múltiplos actuales: P/E, P/B, P/S, EV/EBITDA, FCF yield, earnings yield
- Comparativa vs. historial propio 5 años: `pe_vs_5y_median`, `pb_vs_5y_median`, `ev_ebitda_vs_5y_median`
- Percentiles sectoriales (calculados en train): `pe_sector_pct`, `pb_sector_pct`, `evebitda_sector_pct`, `fcfyield_sector_pct`
- Señales de analistas: EPS surprise, revisiones, EPS estimado vs. reportado

**Hiperparámetros**: 200 estimadores, profundidad 4, lr 0.05, subsample 0.8.

### 10.5 Agente de Momentum

**Modelo**: Random Forest calibrado (Sigmoid, 5-fold CV)

**Features base**:
- Osciladores: RSI 14d, RSI 28d
- Tendencia MACD: macd, macd_signal, macd_hist
- SMAs (distancia % al precio): 20d, 50d, 200d
- Bollinger Bands: bb_pct
- Posición 52 semanas: price_vs_52w_high, price_vs_52w_low
- Momentum puro: 1m, 3m, 6m, 12m
- Volatilidad: 20d, 60d, ATR 14d
- Volumen: ratio 20/50 días
- Macro: VIX, yield curve, SP500 momentum 3m y 12m

**Features derivados** (calculados en `_prepare`): rsi_overbought, rsi_oversold, above_sma200, macd_bullish, cross_sma_20_50, momentum_quality, vol_expansion, high_vix_regime, inverted_yield_curve.

**Hiperparámetros**: 300 estimadores, profundidad 8, min_samples_leaf 10.

### 10.6 Agente de Riesgo (Bear)

**Módulo**: `module/agents/bear_agent.py`

Detecta riesgo de deterioro mediante un score estructurado en dos capas.

#### Capa de reglas ponderada

Los flags se agrupan en dos sub-scores con ponderación diferenciada por severidad:

**Sub-score financiero** (peso 60% en el score de reglas):

| Flag | Condición | Peso relativo |
|---|---|---|
| `fcf_negative` | FCF < 0 | 2.0 |
| `consecutive_losses` | ≥2 trimestres con pérdidas | 2.0 |
| `debt_equity_high` | Debt/Equity > 3 | 1.5 |
| `debt_ebitda_high` | Debt/EBITDA > 6 | 1.5 |
| `low_coverage` | Interest Coverage < 1.5 | 1.5 |
| `debt_growth_high` | Deuda creciendo >20% YoY | 1.0 |
| `revenue_decline` | Caída de revenue YoY | 1.0 |

**Sub-score de mercado** (peso 40% en el score de reglas):

| Flag | Condición | Peso relativo |
|---|---|---|
| `liquidity_risk` | Current Ratio < 1 | 1.5 |
| `insider_selling` | Insider sell ratio > 0.70 | 1.0 |
| `eps_miss` | EPS surprise < -5% | 1.0 |

Score de reglas = 0.6 × sub_financiero + 0.4 × sub_mercado

#### Capa ML

Random Forest calibrado (Sigmoid, 5-fold CV). Aprende patrones de riesgo más sutiles a partir de todas las features base más los flags calculados. Hiperparámetros: 200 estimadores, profundidad 6.

#### Combinación

```
bear_score = BEAR_RULE_WEIGHT × rule_score + BEAR_ML_WEIGHT × ml_score
```

Por defecto: 50% reglas + 50% ML (`environment.py`). El meta-learner usa `(1 - bear_score)` como señal, de modo que un riesgo alto penaliza el score final.

Si `bear_score > BEAR_HARD_THRESHOLD` (0.90), el meta-learner puede forzar la etiqueta "Underperform" independientemente de los demás agentes.

### 10.7 Meta-Learner

**Módulo**: `module/agents/meta_learner.py`

Stacking de dos niveles:
1. **Nivel 1**: recibe los scores OOF de los 4 agentes como features.
2. **Nivel 2**: Logistic Regression (regularización L2) + GBM, cuyo promedio da el score final.

El score del BearAgent entra como `(1 - bear_score)` para que sea alcista cuando el riesgo es bajo.

**Guardrail**: si `bear_score > BEAR_HARD_THRESHOLD`, la predicción final se fuerza a 0 (Underperform).

### 10.8 OOF anti-leakage

Los scores que recibe el meta-learner en train son **out-of-fold**: cada observación es puntuada por un agente entrenado sin verla. Esto evita que el meta-learner aprenda sobre predicciones in-sample infladas.

Implementado con `OOF_N_SPLITS`-fold CV interno (por defecto 3 folds).

### 10.9 Backtester y métricas

**Módulo**: `module/backtester.py` → `WalkForwardBacktester`

Para cada fold, construye un portfolio long con los `TOP_N_STOCKS` tickers de mayor score meta y calcula:
- Retorno acumulado vs. benchmark (SP500 equiweighted)
- Sharpe ratio (anualizado, con `RISK_FREE_RATE`)
- Sortino ratio
- Max Drawdown
- Hit rate (% de predicciones correctas)
- Alpha y Beta vs. benchmark

Los resultados se guardan en `results/backtest/`.

### 10.10 Explicabilidad SHAP

**Módulo**: `module/explainer.py`

Para los `TOP_N_STOCKS` con mayor y menor score de cada fold, genera explicaciones SHAP que desglosan la contribución de cada feature a la predicción. Se guardan en JSON en `results/agents/<agente>/`.

---

## 11. Fold live out-of-sample

**Módulo**: `module/pipeline/live_fold.py` → `run_live_fold()`

Ejecuta la predicción sobre los datos más recientes (desde `END_DATE` en adelante) usando los agentes entrenados en el último fold del walk-forward. Si hay precios reales disponibles en ese período, calcula también el retorno real y las métricas de evaluación.

El resultado incluye el ranking de tickers con sus scores desagregados por agente y la cartera long sugerida.

---

## 12. Visualizaciones

**Módulo**: `module/visualizer.py` (backend Agg — modo headless, sin ventanas)

Todos los gráficos se guardan automáticamente en `results/plots/`. Incluyen:
- Equity curve del portfolio vs. benchmark por fold
- Distribución de scores por agente
- Mapa de calor de correlaciones entre features
- Feature importances top-N por agente
- Resumen de métricas walk-forward
- Distribución de retornos del portfolio

---

## 13. Parámetros clave de `environment.py`

### Hiperparámetros de agentes

| Agente | Parámetro | Valor |
|---|---|---|
| FundamentalAgent | `FUNDAMENTAL_N_ESTIMATORS` | 400 |
| FundamentalAgent | `FUNDAMENTAL_MAX_DEPTH` | 5 |
| FundamentalAgent | `FUNDAMENTAL_LEARNING_RATE` | 0.05 |
| ValuationAgent | `VALUATION_N_ESTIMATORS` | 200 |
| ValuationAgent | `VALUATION_MAX_DEPTH` | 4 |
| MomentumAgent | `MOMENTUM_N_ESTIMATORS` | 300 |
| MomentumAgent | `MOMENTUM_MAX_DEPTH` | 8 |
| BearAgent | `BEAR_N_ESTIMATORS` | 200 |
| BearAgent | `BEAR_RULE_WEIGHT` | 0.5 |
| BearAgent | `BEAR_ML_WEIGHT` | 0.5 |
| BearAgent | `BEAR_HARD_THRESHOLD` | 0.90 |
| MetaLearner | `META_LR_C` | 0.5 |
| MetaLearner | `META_GBM_N_ESTIMATORS` | 150 |

### Reproducibilidad

`RANDOM_SEED = 42` — usado en todos los modelos y splits para reproducibilidad completa.
