# Análisis Técnico del Repositorio: Multi-Agent ML Stock Picker

---

## Tabla de Contenidos

1. [Visión General del Proyecto](#1-visión-general-del-proyecto)
2. [Estructura del Repositorio](#2-estructura-del-repositorio)
3. [Arquitectura del Sistema](#3-arquitectura-del-sistema)
4. [Flujo de Datos y Ejecución](#4-flujo-de-datos-y-ejecución)
5. [Configuración Global (`environment.py`)](#5-configuración-global-environmentpy)
6. [Paso 1 – Adquisición y Preparación de Datos](#6-paso-1--adquisición-y-preparación-de-datos)
7. [Paso 2 – Construcción del Dataset Maestro](#7-paso-2--construcción-del-dataset-maestro)
8. [Paso 3 – Entrenamiento de Agentes (Walk-Forward OOF)](#8-paso-3--entrenamiento-de-agentes-walk-forward-oof)
9. [Agentes Especializados](#9-agentes-especializados)
10. [Paso 4 – Evaluación y Backtesting](#10-paso-4--evaluación-y-backtesting)
11. [Métricas Financieras](#11-métricas-financieras)
12. [Análisis Crítico y Áreas de Mejora](#12-análisis-crítico-y-áreas-de-mejora)

---

## 1. Visión General del Proyecto

Este repositorio implementa un **sistema de selección de acciones del S&P 500 basado en múltiples agentes de Machine Learning (ML)**. Su propósito es elegir automáticamente, en cada período temporal, un pequeño portafolio de acciones con alta probabilidad de superar al mercado.

El sistema no es un simple modelo único; es una **arquitectura de ensamblado jerárquico (stacking)** en la que varios modelos especializados —llamados *agentes*— analizan distintas dimensiones de cada empresa (solidez financiera, valoración, tendencia técnica, riesgo, sentimiento de analistas y contexto sectorial). Un meta-modelo integra esas señales para emitir una puntuación final de *Outperform* (superar al mercado) por acción.

El proyecto sigue un enfoque **walk-forward** (validación hacia adelante en el tiempo) para evitar que el modelo vea datos del futuro durante el entrenamiento, lo que garantiza que las métricas de rendimiento reflejen condiciones reales de inversión.

---

## 2. Estructura del Repositorio

```
TFM/
├── analyzer.py                     # Orquestador principal del pipeline
├── analyzer_ticker.py              # Análisis individual de un ticker
├── environment.py                  # Parámetros globales de configuración
├── requirements.txt                # Dependencias Python
├── data_finnhub/                   # Datos descargados por ticker
│   ├── <TICKER>/                   # Carpeta por acción (p.ej. AAPL/)
│   │   ├── prices.json
│   │   ├── profile.json
│   │   ├── basic_financials.json
│   │   ├── financials_reported_quarterly.json
│   │   ├── eps_surprises.json
│   │   ├── recommendation_trends.json
│   │   ├── insider_transactions.json
│   │   └── insider_sentiment.json
│   ├── consolidated/               # CSVs unificados por ticker
│   ├── _macro/                     # Datos macroeconómicos (SPY)
│   ├── sp500_historic.csv          # Historial de membresía del S&P 500
│   └── master_dataset.parquet      # Dataset maestro en caché
├── module/
│   ├── agents/                     # Agentes ML especializados
│   │   ├── base.py                 # Clase abstracta BaseAgent + FeatureSelector
│   │   ├── fundamental.py          # Agente XGBoost: salud financiera
│   │   ├── valuation.py            # Agente GBM: valoración relativa
│   │   ├── momentum.py             # Agente RF: impulso técnico
│   │   ├── bear.py                 # Agente híbrido: riesgo bajista
│   │   ├── sentiment.py            # Agente RF: sentimiento institucional
│   │   ├── sector_rotation.py      # Agente GBM: rotación sectorial
│   │   ├── sector_specialized.py   # Wrapper: un modelo por sector
│   │   └── meta_learner.py         # Meta-modelo (LR + GBM stacking)
│   ├── common/                     # Utilidades compartidas
│   │   ├── data_router.py          # Carga y enrutamiento de datos
│   │   ├── feature_controls.py     # Resolución de columnas de features
│   │   ├── feature_policy.py       # Política de variables (solo ratios)
│   │   ├── cache.py                # Caché de datos
│   │   └── asof.py                 # Control anti-leakage temporal
│   └── steps/
│       ├── step_01_data/           # ETL: descarga y consolidación
│       ├── step_02_dataset/        # Construcción del dataset maestro
│       ├── step_03_training/       # Entrenamiento OOF de agentes
│       └── step_04_evaluation/     # Backtest, métricas y visualización
└── tests/                          # Pruebas unitarias
```

---

## 3. Arquitectura del Sistema

### 3.1 Diseño Multi-Agente

El sistema sigue el patrón de **ensamblado jerárquico en dos niveles**:

**Nivel 1 — Agentes base** (cada uno especializado en una dimensión de análisis):

| Agente | Algoritmo | Dimensión analizada |
|--------|-----------|---------------------|
| `FundamentalAgent` | XGBoost | Salud financiera (rentabilidad, liquidez, solvencia) |
| `ValuationAgent` | Gradient Boosting | Valoración relativa (ratios precio/valor) |
| `MomentumAgent` | Random Forest | Impulso técnico de precio |
| `BearAgent` | Híbrido (reglas + RF) | Riesgo bajista y deterioro financiero |
| `SentimentAgent` | Random Forest | Sentimiento de analistas e insiders |
| `SectorRotationAgent` | Gradient Boosting | Ciclo sectorial (top-down) |

**Nivel 2 — Meta-Learner** (`MetaLearner`):
- Recibe las puntuaciones de los 6 agentes base como inputs.
- También recibe contexto macroeconómico (VIX, curva de tipos, momentum del S&P 500) y dummies de sector.
- Combina todo con un **ensamble de Regresión Logística + GBM**, ponderando los modelos por su AUC en validación cruzada interna.

### 3.2 Clase Base Abstracta (`BaseAgent`)

Todos los agentes heredan de `BaseAgent` y deben implementar:
- `fit(X, y, **kwargs)` → entrena el agente en un fold.
- `predict_score(X)` → devuelve un score en `[0, 1]` donde `1 = Outperform`.

La clase base también provee:
- `clean_features()` → limpieza de NaN e infinitos.
- `class_balance()` → estadísticas de balance de clases.
- `save_diagnostics()` → serialización de métricas a JSON.
- `save_feature_importances()` → guardado de importancias de features.
- `FeatureSelector` → selector interno que filtra features por correlación e importancia.

### 3.3 `SectorSpecializedAgent` (wrapper de especialización sectorial)

Este wrapper entrena **un modelo independiente por sector** en lugar de un modelo global. Es especialmente útil para el agente fundamental, donde las métricas óptimas varían mucho según el sector (ej: el D/E de una empresa de utilities es distinto al de una tecnológica).

La estrategia es:
- Si un sector tiene suficientes muestras (`SECTOR_SPECIALIST_MIN_SAMPLES = 40`), se entrena un modelo propio.
- Si el sector es muy pequeño, se usa un score de fallback conservador (`SECTOR_SPECIALIST_LONG_FALLBACK_SCORE = 0.25`).

---

## 4. Flujo de Datos y Ejecución

El pipeline se ejecuta desde `analyzer.py` → función `main()` y sigue estos pasos:

```
[1] Resolución de universo de tickers
         ↓
[2] Descarga de datos (Finnhub + Yahoo Finance)
         ↓
[3] Consolidación de datos crudos → CSVs por ticker
         ↓
[4] Construcción del Dataset Maestro (observaciones punto-en-tiempo)
         ↓
[5] Walk-Forward Loop:
      Para cada fold temporal:
        [5a] Split train/test por fecha
        [5b] Generación de scores OOF (anti-leakage)
        [5c] Entrenamiento de agentes base en datos de entrenamiento
        [5d] Entrenamiento del MetaLearner sobre scores OOF
        [5e] Predicción sobre datos de test
        [5f] Selección de portafolio (top-N con caps sectoriales)
        [5g] Simulación de retornos y métricas
         ↓
[6] Reportes, visualizaciones y exportación de artefactos
```

### 4.1 Identificador de Ejecución (Slug)

Cada ejecución genera un directorio con nombre único:
```
results/annual_2024Q1_2026Q4/
```
Esto evita que diferentes configuraciones se sobreescriban entre sí.

---

## 5. Configuración Global (`environment.py`)

Este archivo es la **única fuente de verdad** para todos los parámetros del sistema. Está organizado en secciones:

### 5.1 Universo de Tickers

```python
USE_DYNAMIC_SP500_UNIVERSE = True   # Usa sp500_historic.csv en lugar de lista manual
SP500_DYNAMIC_TOP_N = False          # False = todos los activos; 200/300 = top-N por capitalización
```

Cuando `USE_DYNAMIC_SP500_UNIVERSE = True`, el sistema lee `data_finnhub/sp500_historic.csv` (que contiene el historial de membresía del índice: ticker, fecha de entrada, fecha de salida) y filtra los tickers que estuvieron activos en el período de análisis. Esto **elimina el survivor bias** (sesgo de supervivencia), ya que incluye acciones que existieron históricamente aunque ya no estén en el índice.

**Survivor bias explicado:** Si un análisis histórico solo usa las acciones que actualmente forman parte del S&P 500, ignora las que quebraron o fueron excluidas, sobreestimando los retornos pasados.

### 5.2 Período de Análisis

```python
ANALYSIS_START_YEAR = 2024
ANALYSIS_START_QUARTER = 1
ANALYSIS_END_YEAR = 2026
ANALYSIS_END_QUARTER = 4
ANALYSIS_FREQUENCY = "annual"   # "quarterly" o "annual"
SNAPSHOT_LAG_DAYS = 45          # Días desde el cierre del trimestre hasta la entrada al mercado
```

El `SNAPSHOT_LAG_DAYS` es crucial: simula el tiempo real que un inversor necesita para obtener los datos financieros publicados y actuar. Los resultados de Q1 (cierre 31 de marzo) no están disponibles inmediatamente; hay que esperar ~45 días hasta que la empresa los publique.

### 5.3 Parámetros del Portafolio

```python
TOP_N_STOCKS = 10               # Máximo de acciones en el portafolio
PORTFOLIO_MIN_SCORE = 0.55      # Score mínimo para entrar al portafolio
PORTFOLIO_MAX_STOCKS_PER_SECTOR = 3  # Máximo por sector (diversificación)
PORTFOLIO_MAX_STOCK_WEIGHT = 0.15    # Peso máximo por acción (15%)
SCORE_WEIGHTED_PORTFOLIO = True      # Ponderación por score (no igual)
```

### 5.4 Parámetros del Backtest

```python
INITIAL_CAPITAL_USD = 1000.0    # Capital inicial en USD
TRANSACTION_FEE_USD = 1.0       # Comisión fija por operación
SLIPPAGE_PCT = 0.001            # Deslizamiento de precio (0.1%)
RISK_FREE_RATE = 0.04           # Tasa libre de riesgo anual (4%)
WALKFORWARD_TRAIN_LOOKBACK_YEARS = 5  # Años de ventana de entrenamiento
```

### 5.5 Hiperparámetros de los Agentes

Cada agente tiene sus propios parámetros. Por ejemplo, para `FundamentalAgent`:

```python
FUNDAMENTAL_N_ESTIMATORS    = 400
FUNDAMENTAL_MAX_DEPTH       = 5
FUNDAMENTAL_LEARNING_RATE   = 0.05
FUNDAMENTAL_SUBSAMPLE       = 0.8
FUNDAMENTAL_COLSAMPLE       = 0.7
FUNDAMENTAL_MIN_CHILD_WEIGHT = 5
```

---

## 6. Paso 1 – Adquisición y Preparación de Datos

### 6.1 Fuentes de Datos

El sistema obtiene datos de dos fuentes principales:

**Finnhub API** (datos fundamentales y de sentimiento):
- `profile.json` → Sector, industria, capitalización de mercado.
- `basic_financials.json` → Ratios financieros precomputados (PE, PB, ROE...).
- `financials_reported_quarterly.json` → Estados financieros trimestrales.
- `financials_reported_annual.json` → Estados financieros anuales.
- `eps_surprises.json` → Sorpresas de BPA respecto a estimaciones de analistas.
- `recommendation_trends.json` → Consenso de analistas (compra/venta/neutral).
- `insider_transactions.json` → Transacciones de directivos e insiders.
- `insider_sentiment.json` → Índice MSPR (Monthly Share Purchase Ratio).

**Yahoo Finance** (precios OHLCV):
- `prices.json` → Precios históricos diarios (Open, High, Low, Close, Volume).

### 6.2 Control de Descarga

La descarga es **paralela** (`DOWNLOAD_MAX_WORKERS = 8`) con control de rate-limit para Finnhub (`FINNHUB_MIN_INTERVAL = 1` segundo). Un **registro** (`_registry.json`) trackea qué endpoints ya han sido descargados para evitar repeticiones innecesarias.

### 6.3 Consolidación

El módulo `consolidation.py` normaliza y une los datos crudos en un único CSV por ticker en `data_finnhub/consolidated/<TICKER>.csv`. Este CSV contiene todas las variables fundamentales alineadas por `report_date` (fecha de cierre del trimestre).

---

## 7. Paso 2 – Construcción del Dataset Maestro

El `build_master_dataset()` en `dataset.py` produce el **dataset maestro**: una tabla con índice `(ticker, date)` donde cada fila representa una **observación punto-en-tiempo** para un ticker en una fecha de snapshot.

### 7.1 Concepto de Snapshot Punto-en-Tiempo

Para cada trimestre analizado y cada ticker:
1. Se toma la fecha de cierre del trimestre + `SNAPSHOT_LAG_DAYS`.
2. Se busca el último informe financiero disponible en o antes de esa fecha.
3. Se calculan los indicadores técnicos usando solo precios hasta esa fecha.
4. Se calcula el **retorno futuro** del precio desde la fecha de entrada hasta `HOLDING_PERIOD_MONTHS` meses después.

Esto garantiza que **ningún dato futuro contamina el entrenamiento** (principio conocido como no look-ahead bias).

### 7.2 Feature Builders

Los constructores de features (`builders/`) calculan distintas familias de variables:

#### `FundamentalFeatureBuilder`

Variables derivadas de los estados financieros trimestrales:

- **Crecimiento interanual (YoY Growth):**

  ```
  revenue_yoy_growth = (revenue_t - revenue_{t-4}) / |revenue_{t-4}|
  ```

  Capeado entre -5 y +5 para evitar outliers por efecto base.

- **Piotroski F-Score** (Piotroski, 2000): Suma de 8 señales binarias que miden la salud financiera:
  - F1: ROA > 0 (rentabilidad positiva sobre activos)
  - F2: Cash Flow Operativo > 0
  - F3: ROA mejorando vs. año anterior
  - F4: Calidad de beneficios (CFO > Net Income → bajos devengos)
  - F5: Deuda no creciendo más del 5%
  - F6: Current Ratio mejorando
  - F7: Sin dilución de acciones
  - F8: Margen bruto mejorando

  El F-Score va de 0 a 8 y se normaliza por el número de señales disponibles. Un valor alto indica una empresa financieramente saludable.

- **Tendencias de largo plazo** (calculadas *solo con datos hasta la fecha de snapshot*):

  ```
  roe_trend_3y = pendiente del ROE en los últimos 8 trimestres
  ```

#### `TechnicalFeatureBuilder`

Variables de análisis técnico sobre precios OHLCV:

- **RSI (Relative Strength Index, 14 y 28 períodos):**

  ```
  RSI = 100 - 100 / (1 + RS)
  RS = Promedio ganancias / Promedio pérdidas (media exponencial)
  ```

  *Explicación intuitiva:* El RSI mide si una acción está sobrecomprada (>70) o sobrevendida (<30) en comparación con su propia historia reciente.

- **MACD (Moving Average Convergence Divergence):**

  ```
  MACD = EMA(12) - EMA(26)
  Signal = EMA(9) del MACD
  Histograma = MACD - Signal
  ```

  *Explicación intuitiva:* Compara la velocidad de la tendencia a corto y largo plazo. Un MACD positivo indica momentum alcista.

- **Bandas de Bollinger:**

  ```
  bb_pct = (precio - (SMA20 - 2·σ20)) / (4·σ20)
  ```

  Mide dónde está el precio dentro de su rango de volatilidad reciente (0 = borde inferior, 1 = borde superior).

- **Momentum multi-horizonte:**

  ```
  momentum_3m  = precio_hoy / precio_hace_63_dias  - 1
  momentum_12m = precio_hoy / precio_hace_252_dias - 1
  ```

- **Volatilidad anualizada:**

  ```
  volatility_60d = std(retornos_diarios_60d) × √252
  ```

- **ATR (Average True Range):**

  ```
  TR  = max(High-Low, |High-Close_{t-1}|, |Low-Close_{t-1}|)
  ATR = media exponencial del TR en 14 períodos
  ```

#### `ValuationFeatureBuilder`

Variables de valoración relativa:
- **PE Ratio vs. mediana histórica 5 años:** `pe_vs_5y_median = pe_actual / pe_mediana_5y - 1`. Un valor negativo indica que la acción está más barata que su propia historia.
- **FCF Yield:** `FCF / Market Cap` (rentabilidad del flujo de caja libre).
- **EV/EBITDA:** Valor de empresa dividido por EBITDA (métrica de valoración neutral a la estructura de capital).

#### `InsiderFeatureBuilder`

Métricas de transacciones de personas con información privilegiada:
- **MSPR (Monthly Share Purchase Ratio):** Ratio de compras netas mensuales de insiders. Un MSPR alto señala que los directivos están comprando acciones de su propia empresa.
- **Insider sell ratio:** Proporción de transacciones que son ventas.

#### `SentimentFeatureBuilder`

Variables del consenso de analistas:
- **Analyst buy ratio:** % de analistas con recomendación de compra.
- **Beat rate 4q:** % de trimestres en los últimos 4 en que la empresa superó estimaciones de BPA.
- **EPS surprise avg:** Sorpresa promedio de BPA en los últimos 4 trimestres.

### 7.3 Variable Objetivo (`forward_return`)

Para cada observación `(ticker, snapshot_date)`:

```
forward_return = precio_cierre(entry_date + holding_period) / precio_cierre(entry_date) - 1
```

Este retorno se binariza para el entrenamiento:
- `label = 1` si el ticker supera la **mediana sectorial** de retornos en ese snapshot (modo `"vs_sector"`).
- `label = 0` en caso contrario.

El modo `"vs_sector"` evita que las etiquetas estén sesgadas por el ciclo de mercado: en un trimestre alcista todos tenderían a ser 1, y en uno bajista todos a 0.

---

## 8. Paso 3 – Entrenamiento de Agentes (Walk-Forward OOF)

### 8.1 Walk-Forward Cross-Validation

El sistema usa una **validación cruzada temporal hacia adelante** (walk-forward). En lugar de un único split train/test, se generan múltiples *folds* sucesivos:

```
Fold 1:  Train [2019 - 2023] → Test [2024 Q1]
Fold 2:  Train [2020 - 2024] → Test [2024 Q2]  (modo quarterly)
...
```

En modo `"annual"`, los folds avanzan año a año en lugar de trimestre a trimestre.

**¿Por qué es importante?** Los modelos financieros sufren de *non-stationarity*: las relaciones entre variables cambian con el tiempo (cambio de régimen). Un modelo entrenado en 2015 puede no ser válido en 2022. El walk-forward entrena y evalúa en condiciones similares a las de inversión real.

### 8.2 Scores Out-of-Fold (OOF) para el Meta-Learner

Para evitar *data leakage* (filtración de datos) al entrenar el MetaLearner, los scores de los agentes base se generan con la técnica **Out-of-Fold (OOF)**:

1. Los datos de entrenamiento se dividen en `OOF_N_SPLITS = 3` subfolds temporales.
2. Para cada subfold, los agentes se entrenan en los subfolds anteriores y predicen sobre el subfold actual.
3. Se obtienen así scores para **cada observación de entrenamiento sin haberla visto durante el entrenamiento**.

Sin OOF, el MetaLearner aprendería a confiar en agentes que memorizaron los datos de entrenamiento, produciendo un modelo ilusoriamente bueno pero ineficaz en producción.

### 8.3 Ajuste de Dispersión de Scores

Cuando un agente produce scores con poca variabilidad (todos cercanos a 0.5), las predicciones son poco informativas. El sistema aplica *shrinkage* (contracción) hacia 0.5:

```
scale = min(1, σ_train / SCORE_DISPERSION_MIN_STD)
score_ajustado = 0.5 + (score - 0.5) × max(scale, SCORE_DISPERSION_MIN_SCALE)
```

Esto evita que un agente "seguro pero vacío" domine el meta-modelo.

### 8.4 Tilt Sectorial

El sistema aplica un ajuste suave basado en la puntuación del sector:

```
final_score += (sector_score - 0.5) × SECTOR_SCORE_PRIOR_WEIGHT × sector_confidence
sector_confidence = min(1, √(n_peers / SECTOR_CONFIDENCE_PEERS))
```

*Explicación intuitiva:* Si el sector tecnológico está en un buen momento (`sector_score = 0.7`), todas las acciones tecnológicas reciben un pequeño impulso adicional. El ajuste es proporcional a cuántos pares hay en ese sector (más datos = más confianza en el score sectorial).

---

## 9. Agentes Especializados

### 9.1 `FundamentalAgent` (XGBoost)

**Objetivo:** Detectar empresas financieramente sólidas con alta probabilidad de superar al sector.

**Features principales:**
- Rentabilidad: ROA, ROE, ROI, ROIC, márgenes (neto, bruto, EBITDA, FCF, operativo).
- Liquidez: Current Ratio, Quick Ratio.
- Apalancamiento: D/E, Deuda/EBITDA, Cobertura de intereses.
- Crecimiento: YoY Revenue, YoY FCF.
- Calidad: Piotroski F-Score, CAPEX/Revenue.
- Tendencias: Slopes de ROE, márgenes en 3 años.

**Detalles técnicos:** XGBoost con `scale_pos_weight = n_neg/n_pos` para manejar el desequilibrio de clases. Features derivadas internas:
- `profitability_quality`: Media del rango percentil de margen neto y calidad de beneficios.
- `fundamental_momentum`: Promedio de tendencias de ratios clave.

### 9.2 `ValuationAgent` (Gradient Boosting)

**Objetivo:** Identificar empresas subvaloradas relativas a su historia y a su sector.

**Features principales:**
- Múltiplos absolutos: PE, PB, PS, EV/EBITDA.
- Yields: FCF Yield, Earnings Yield.
- Comparación histórica: `pe_vs_5y_median`, `ev_ebitda_vs_5y_median`.

**Lógica:** La valoración relativa es más predictiva que la absoluta: un PE de 20 puede ser barato para tecnología pero caro para utilities.

### 9.3 `MomentumAgent` (Random Forest)

**Objetivo:** Capturar el factor de momentum (las acciones que han subido tienden a seguir subiendo en horizontes intermedios).

**Features principales:**
- Posición de precio: `price_vs_52w_high`.
- Momentum multi-horizonte: 3m, 6m, 12m.
- Tendencia: SMA50, SMA200 (como % sobre precio actual).
- Régimen de volatilidad: `volatility_60d`.
- Volumen: `vol_ratio_20_50` (volumen reciente vs. promedio).
- RSI: 14 y 28 períodos.

**Excluye intencionalmente:** MACD (altamente correlacionado con SMAs), SMA20 (redundante), momentum 1m (ruido de reversal de corto plazo).

### 9.4 `BearAgent` (Híbrido Reglas + Random Forest)

**Objetivo:** Filtrar empresas con alto riesgo de deterioro o caída significativa.

**Arquitectura en dos capas:**

**Capa de reglas (35% del score final):**

Evalúa 10 flags binarios ponderados:

| Flag | Condición | Peso | Categoría |
|------|-----------|------|-----------|
| `debt_growth_high` | Deuda YoY > 20% | 1.0 | Financiero |
| `debt_equity_high` | D/E > 3 | 1.5 | Financiero |
| `debt_ebitda_high` | Deuda/EBITDA > 6 | 1.5 | Financiero |
| `fcf_negative` | FCF Margin < 0 | 2.0 | Financiero |
| `consecutive_losses` | ≥2 trimestres consecutivos con pérdidas | 2.0 | Financiero |
| `revenue_decline` | Ingresos decrecientes YoY | 1.0 | Financiero |
| `low_coverage` | Cobertura de intereses < 1.5 | 1.5 | Financiero |
| `liquidity_risk` | Current Ratio < 1 | 1.5 | Mercado |
| `insider_selling` | Insiders vendiendo >70% | 1.0 | Mercado |
| `eps_miss` | EPS miss >5% | 1.0 | Mercado |

Sub-scores internos:
```
rule_score = 0.6 × financial_subscore + 0.4 × market_subscore
```

**Capa ML (65% del score final):** Random Forest que aprende patrones de riesgo más sutiles que las reglas simples no capturan (ej. que una deuda alta es aceptable en utilities pero peligrosa en retail).

```
bear_score = 0.35 × rule_score + 0.65 × ml_score
```

**Integración en el MetaLearner:** Si el score supera `BEAR_HARD_THRESHOLD = 0.90`, el MetaLearner fuerza directamente una predicción de *Underperform*, independientemente de lo que digan los demás agentes.

### 9.5 `SentimentAgent` (Random Forest)

**Objetivo:** Capturar la "sabiduría de la multitud" institucional (analistas) y la señal de personas con información privilegiada (insiders).

**Features:**
- Analistas: `analyst_buy_ratio`, `analyst_consensus`, `analyst_dispersion`, `analyst_consensus_change`.
- Insiders: `mspr_3m`, `mspr_trend`, `insider_net_ratio_90d`, `insider_sell_ratio`.
- BPA: `beat_rate_4q`, `eps_surprise_avg_4q`, `eps_surprise_pct`.

### 9.6 `SectorRotationAgent` (Gradient Boosting, top-down)

**Objetivo:** Identificar qué sectores del mercado están en un ciclo favorable.

**Funcionamiento:**
1. Agrega las features de todos los tickers de un sector (mediana).
2. Construye una variable objetivo sectorial: `1` si el retorno promedio del sector supera al benchmark (SPY) en ese trimestre.
3. Entrena un GBM sobre estas observaciones sectoriales.
4. En predicción, asigna el mismo score sectorial a todos los tickers del sector.

**Integración:** El score sectorial entra al MetaLearner como contexto top-down (también con el ajuste de prior sectorial descrito en §8.4).

### 9.7 `MetaLearner` (LR + GBM Stacking)

**Objetivo:** Combinar óptimamente las señales de todos los agentes base.

**Inputs:**
- Scores de los 6 agentes base: `fundamental_score`, `valuation_score`, `momentum_score`, `bear_score`, `sentiment_score`, `sector_score`.
- Contexto macro: `vix`, `yield_curve`, `sp500_momentum_3m`, `sp500_momentum_12m`.
- Dummies de sector (one-hot): permiten al meta-modelo aprender que ciertos agentes son más fiables en ciertos sectores.
- Features de consenso (opcionales): conteo de agentes bullish, dispersión de scores.

**Modelos:**
- **Regresión Logística** (interpretable, evita sobreajuste).
- **Gradient Boosting Machine** (captura interacciones no lineales).

**Ponderación dinámica:** Los pesos entre LR y GBM se determinan por validación cruzada interna (TimeSeriesSplit):
```
weight_lr  = AUC_lr  / (AUC_lr + AUC_gbm)
weight_gbm = 1 - weight_lr
```

**Recalibración de scores** (opcional):
```
score_recalibrado = temperatura × (score_raw - media_train) / std_train + 0.5
```

---

## 10. Paso 4 – Evaluación y Backtesting

### 10.1 Walk-Forward Backtest (`WalkForwardBacktester`)

Para cada fold temporal:

1. **Selección del portafolio:**
   - Filtra tickers con `final_score >= PORTFOLIO_MIN_SCORE`.
   - Aplica cap sectorial (`PORTFOLIO_MAX_STOCKS_PER_SECTOR = 3`).
   - Toma el top-N (`TOP_N_STOCKS = 10`).

2. **Ponderación:**
   - Si `SCORE_WEIGHTED_PORTFOLIO = True`: el ticker con mayor score tiene peso ~2× el de menor score (distribución lineal normalizada).
   - Si `False`: pesos iguales.

3. **Simulación de retornos:**
   - Modo **retorno puro**: retorno del portafolio = promedio ponderado de `forward_return`.
   - Modo **USD** (`USE_DOLLAR_BACKTEST = True`): simula compra/venta real con comisiones y slippage.

### 10.2 Simulación en USD (`portfolio_simulator.py`)

Para cada fold:
1. **Entrada:** El día `entry_date`, compra cada acción seleccionada ajustando por slippage:
   ```
   precio_compra = precio_cierre × (1 + SLIPPAGE_PCT)
   ```
2. **Salida:** Al cabo de `HOLDING_PERIOD_MONTHS`, vende ajustando:
   ```
   precio_venta = precio_cierre × (1 - SLIPPAGE_PCT)
   ```
3. **Comisiones:** `TRANSACTION_FEE_USD` por cada operación de compra y de venta.
4. El capital resultante se reinvierte en el siguiente fold.

### 10.3 Baselines de Comparación

El sistema genera automáticamente estrategias de referencia:

- **SPY buy & hold:** Rentabilidad del índice S&P 500.
- **Equal-weight universe:** Portafolio con todas las acciones del universo a igual peso.
- **Top-N random baseline:** Monte Carlo con `N_RANDOM_BASELINE_SIMS = 100` portafolios aleatorios de N acciones.
- **Momentum baseline:** Selección de las N acciones con mayor momentum a `BASELINE_MOMENTUM_LOOKBACK_DAYS = 252` días.

### 10.4 Explicabilidad (SHAP)

Para cada fold y los tickers seleccionados, el sistema calcula valores SHAP (*SHapley Additive exPlanations*) que explican cuánto contribuye cada feature a la predicción de cada ticker. Esto permite responder preguntas como: "¿Por qué el sistema seleccionó AAPL en Q1 2024?" con respuestas detalladas por feature.

---

## 11. Métricas Financieras

El módulo `metrics.py` calcula las siguientes métricas estándar de la industria de gestión de activos:

### 11.1 Retorno Acumulado

```
cumulative_return = ∏(1 + r_t) - 1
```

*Ejemplo:* Retornos mensuales del 5%, 3%, -2% → retorno acumulado = 1.05 × 1.03 × 0.98 - 1 = 5.99%.

### 11.2 Retorno Anualizado (CAGR)

```
CAGR = (1 + cumulative_return)^(252/n) - 1
```

Donde n es el número de períodos (días). Permite comparar estrategias con distintos horizontes temporales.

### 11.3 Ratio de Sharpe

```
Sharpe = (r̄_excess / σ_excess) × √252
r̄_excess = r̄_portfolio - r_f/252
```

*Explicación intuitiva:* Mide el retorno por unidad de riesgo asumida. Un Sharpe de 1.0 significa que se obtiene un 1% de retorno adicional por cada 1% de volatilidad. En fondos de gestión activa, valores > 0.5 se consideran buenos; > 1.0 son excelentes.

La tasa libre de riesgo anual es `RISK_FREE_RATE = 4%`.

### 11.4 Ratio de Sortino

```
Sortino = (r̄_excess / σ_downside) × √252
σ_downside = std(r_excess | r_excess < 0)
```

Similar al Sharpe, pero solo penaliza la **volatilidad bajista** (pérdidas), no la volatilidad alcista.

### 11.5 Maximum Drawdown (MDD)

```
wealth_t = ∏(1 + r_s  para s ≤ t)
MDD = min_t((wealth_t - max_{s≤t}(wealth_s)) / max_{s≤t}(wealth_s))
```

*Explicación intuitiva:* La mayor pérdida desde un máximo histórico hasta el valle siguiente. Si el portafolio alcanzó 1.000 € y luego cayó a 700 €, el MDD es -30%.

### 11.6 Ratio de Calmar

```
Calmar = CAGR / |MDD|
```

Relaciona el retorno anualizado con la peor caída observada.

---

## 12. Análisis Crítico y Áreas de Mejora

### 12.1 Fortalezas del Sistema

- **Anti-leakage robusto:** El uso de OOF para el MetaLearner, los splits temporales por quarter y el control de fechas en `asof.py` y `feature_policy.py` son mecanismos sólidos para evitar data leakage.
- **Modularidad:** Cada agente es completamente independiente; añadir uno nuevo solo requiere heredar de `BaseAgent`.
- **Survivor bias mitigado:** El uso de `sp500_historic.csv` con historial de membresía es un enfoque correcto.
- **Explicabilidad:** La integración de SHAP permite auditar cada decisión de inversión.
- **Configuración centralizada:** Todo en `environment.py` facilita la reproducibilidad y experimentación.

### 12.2 Debilidades y Limitaciones

#### a) Dependencia de un único proveedor de datos
Finnhub tiene limitaciones de API en el plan gratuito. Si la API cambia o el proveedor cierra, todo el sistema queda inutilizado. No hay proveedores de datos alternativos implementados.

#### b) Lookback finito para features de tendencia
Las tendencias (`roe_trend_3y`) se calculan con solo 8 trimestres de historia. Para empresas con menos de 4 años de datos cotizados, estas features serán NaN y el sistema las imputará mediante fallback extrapolation, lo que introduce ruido.

#### c) Reentrenamiento completo en cada fold
En modo walk-forward, todos los agentes se entrenan desde cero en cada fold. Esto es computacionalmente costoso y no aprovecha el conocimiento de folds anteriores. Técnicas como *continual learning* o *warm start* podrían reducir el tiempo de ejecución.

#### d) El BearAgent usa una regla dura arbitraria
El umbral `BEAR_HARD_THRESHOLD = 0.90` fuerza la exclusión de una acción si el BearAgent da un score ≥ 0.90. Este umbral es fijo y no se adapta al régimen de mercado.

#### e) Hiperparámetros fijos sin optimización automática
Los hiperparámetros de todos los agentes están fijados en `environment.py`. No hay búsqueda automática (grid search, Bayesian optimization) que los optimice para cada fold o período.

#### f) Ausencia de factores de riesgo sistemáticos
El sistema no modela explícitamente factores como los del modelo de Fama-French (tamaño, valor, momentum, calidad). Sus agentes capturan algunas de estas dimensiones de forma implícita, pero no hay una descomposición formal del retorno en factores.

#### g) Limitación a posiciones largas
El sistema es *long-only*: solo puede comprar acciones que espera que suban. No implementa estrategias de cobertura (hedging) ni posiciones cortas.

#### h) Datos de insider sentiment con retraso regulatorio
Las transacciones de insiders se reportan con retrasos variables. El `SNAPSHOT_LAG_DAYS = 45` mitiga esto parcialmente, pero no hay verificación explícita de la fecha de publicación de cada transacción.

### 12.3 Mejoras Sugeridas

1. **Añadir fuentes de datos alternativos:** Datos de texto (news NLP), datos alternativos (tráfico web, búsquedas de Google) para enriquecer el `SentimentAgent`.

2. **Optimización bayesiana de hiperparámetros:** Usar `Optuna` o `hyperopt` para encontrar automáticamente los mejores hiperparámetros en cada fold.

3. **Factorización explícita del retorno:** Implementar un modelo de factores (Fama-French 5) para separar el alpha generado de la exposición a factores sistemáticos.

4. **Posiciones cortas (long/short):** Invertir el uso del BearAgent para generar señales cortas, permitiendo una estrategia market-neutral.

5. **Gestión de riesgo dinámica:** Ajustar el tamaño del portafolio según el régimen de volatilidad (ej. reducir exposición cuando el VIX supera cierto umbral).

6. **Validación estadística más rigurosa:** Añadir tests de significancia estadística (p-valor del alpha, test de Diebold-Mariano entre estrategias).

7. **Ensemble de estrategias:** Combinar los resultados de múltiples ejecuciones con distintas semillas aleatorias para reducir la varianza de las predicciones.

8. **Pipeline de monitoreo en producción:** Implementar alertas para detectar *model drift* cuando la distribución de los datos de entrada cambia respecto al período de entrenamiento.

### 12.4 Áreas con Documentación Escasa

- **`module/common/cache.py`:** No está claro qué datos se cachean y bajo qué condiciones se invalida la caché.
- **`module/steps/step_04_evaluation/ablation.py`:** El estudio de ablación (que mide la contribución de cada agente) está implementado pero no se ejecuta por defecto (`RUN_ABLATION_STUDY = False`) y carece de documentación sobre cómo interpretar sus resultados.
- **`analyzer_ticker.py`:** El script de análisis individual de un ticker no tiene documentación sobre su uso o parámetros.
- **Integración del contexto macroeconómico:** Se mencionan variables como `vix` y `yield_curve` en el MetaLearner, pero no está completamente documentado de dónde provienen ni cómo se calculan en el dataset maestro.

---

*Documento generado mediante análisis estático del código fuente del repositorio `adrian-nunhez-costa/TFM`, abril de 2026.*
