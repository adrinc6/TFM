# Multi-Agent ML Stock Picker — Documentación del Proyecto

## 1. Objetivo del sistema

Construir y evaluar una estrategia de selección de acciones del S&P 500 basada en aprendizaje automático con arquitectura multi-agente. El sistema aprende, por cada snapshot trimestral, qué combinaciones de métricas fundamentales, de valoración, técnicas, de momentum, de riesgo y de sentimiento predicen que el precio de la acción alcanzará un Take-Profit antes que un Stop-Loss dentro de un período de holding de 12 meses.

Los pilares del diseño son:

- **Reglas interpretables** como mecanismo primario de señal: el sistema minea activamente combinaciones de métricas (crecimientos, ratios, variaciones) que históricamente elevan la probabilidad de TP-before-SL. Estas reglas se inyectan directamente como features del modelo de cada agente.
- **Agentes especializados por dominio**: cada agente trabaja sobre su universo de features propio y genera una probabilidad de éxito independiente.
- **Meta-learner de stacking**: consolida las señales de todos los agentes y añade contexto cross-sectional, de régimen macro y de coherencia entre agentes.
- **TP/SL adaptativos**: los niveles de Take-Profit y Stop-Loss se calibran por ticker y snapshot usando el historial de precios propio de cada acción, no valores fijos globales.
- **Walk-forward riguroso** con purga y embargo temporal para garantizar cero leakage.
- **Penalización de neutralidad**: valores neutrales (0.5) se penalizan a 0.25 para que la ausencia de evidencia no eleve artificialmente el ranking.

---

## 2. Estructura de ficheros

```
TFM/
├── analyzer.py                          # Entrypoint principal del pipeline completo
├── analyzer_II.py                       # Lanzador de grid de escenarios en paralelo (9 configs)
├── environment.py                       # Única fuente de verdad para todos los parámetros
├── requirements.txt
├── pyproject.toml
│
├── data_finnhub/                        # Datos descargados de Finnhub
│   ├── sp500_historic.csv               # Universo dinámico S&P 500 histórico
│   ├── _registry.json                   # Registro de descargas por ticker
│   ├── _macro/                          # Datos macro: VIX, yield curve, SP500
│   └── <TICKER>/                        # Un directorio por ticker con JSONs de Finnhub
│
├── results/
│   └── anchor_<fecha>_h<holding>_n<tests>/   # Un directorio por run
│       ├── general/                     # run_config.json, feature usage, audits
│       ├── strategy/                    # Métricas globales, fold results, report.txt
│       ├── config/                      # Snapshot de environment.py del run
│       └── T<N>_<fecha>_<quarter>/      # Artefactos por fold (scores, portafolio, SHAP)
│
└── module/
    ├── agents/
    │   ├── universe_tp_sl.py            # Agente base unificado (fundamental, valuation, momentum, bear, sentiment)
    │   ├── sector_rotation.py           # Agente de rotación sectorial
    │   ├── alpha_meta_learner.py        # Meta-learner XGBoost (regressor + ranker + risk classifier)
    │   └── base.py                      # Clase abstracta Agent + FeatureConfig + AgentScoreResult
    │
    ├── common/
    │   ├── asof.py                      # Validaciones punto-en-tiempo (as-of)
    │   ├── cross_sectional_features.py  # Percentiles sectoriales y universales
    │   ├── data_router.py               # Enrutamiento de datos por ticker/endpoint
    │   ├── feature_controls.py          # Control de disponibilidad de features por fecha
    │   ├── feature_policy.py            # Políticas de recencia por feature
    │   ├── finbert_features.py          # Sentimiento NLP vía FinBERT (polarity, risk)
    │   ├── performance_metrics.py       # Sharpe, Sortino, Calmar, drawdown, etc.
    │   ├── portfolio_optimization.py    # HRP, Risk Parity, Markowitz
    │   ├── purged_cv.py                 # PurgedEmbargoKFold para OOF temporal
    │   ├── recency_weights.py           # Pesos exponenciales por recencia de observación
    │   ├── regime.py                    # Detección y encoding de régimen macro
    │   └── target_engineering.py       # Construcción de targets forward TP/SL
    │
    └── steps/
        ├── step_01_data/
        │   ├── pipeline.py              # Orquestador ETL: descarga, consolidación, filtrado
        │   ├── downloaders.py           # Descarga por endpoint de Finnhub
        │   ├── clients.py               # Cliente HTTP Finnhub con rate limiting
        │   ├── parsers.py               # Parseo y normalización de JSONs
        │   ├── consolidation.py         # Consolidación de datos crudos por ticker
        │   └── registry.py              # Registro de estado de descarga
        │
        ├── step_02_dataset/
        │   ├── dataset.py               # Construcción del master dataset punto-en-tiempo
        │   └── builders/
        │       ├── fundamental.py       # Features fundamentales por snapshot
        │       ├── valuation.py         # Features de valoración
        │       ├── technical.py         # Features técnicos (RSI, momentum, volatilidad)
        │       ├── sentiment.py         # Sentimiento analista + insider + FinBERT
        │       └── insider.py           # Actividad insider (compras/ventas netas)
        │
        ├── step_03_training/
        │   ├── training.py              # Entrenamiento por fold con OOF anti-leakage
        │   ├── oof.py                   # OOF por quarter con PurgedEmbargoKFold
        │   ├── agent_diagnostics.py     # Diagnóstico de redundancia y calidad de reglas OOF
        │   └── agent_config.py          # Configuración por agente (features, modelo, params)
        │
        └── step_04_evaluation/
            ├── evaluator.py             # Loop walk-forward principal
            ├── backtesting.py           # Motor de simulación TP/SL por ticker
            ├── strategy.py              # Definición y selección de estrategia adaptativa
            ├── reporting.py             # Generación de reportes y CSVs de resultados
            ├── analysis.py              # Análisis de alpha, métricas por fold y globales
            ├── explainability.py        # SHAP por agente y fold
            └── visualization.py         # Gráficas: equity curve, fold PnL, drawdown, heatmaps
```

---

## 3. Configuración global — `environment.py`

Única fuente de verdad para todos los parámetros del sistema. Soporta sobreescritura en tiempo de ejecución mediante la variable de entorno `ENV_OVERRIDES_JSON` (útil para el grid de escenarios de `analyzer_II.py`).

### Flags de ejecución

| Flag | Por defecto | Descripción |
|------|-------------|-------------|
| `SKIP_BACKTEST` | `False` | Omite el walk-forward, solo ejecuta el fold live |
| `FORCE_DOWNLOAD` | `False` | Re-descarga todos los datos aunque ya existan |
| `UPDATE_PRICES_ONLY` | `False` | Solo actualiza precios y macro, sin reconsolidar |
| `REBUILD_MASTER_DATASET` | `False` | Fuerza reconstrucción del dataset maestro |
| `DEBUG_OUTPUT_PROFILE` | `"focused"` | `"focused"`: solo artefactos clave; `"full"`: todos |
| `EXPORT_RUN_ARTIFACTS` | `True` | Exporta config, calidad y resúmenes del run |

### Universo y período

- `USE_DYNAMIC_SP500_UNIVERSE = True`: el universo se construye desde `sp500_historic.csv` en función del período analizado. El fallback manual es la lista `TICKERS`.
- `SP500_DYNAMIC_TOP_N = False`: usa todos los miembros activos del período (sin recorte por market cap).
- `DOWNLOAD_START_DATE = "2000-01-01"`: inicio de datos históricos.
- `ANALYSIS_REFERENCE_DATE = "2026-05-03"`: ancla del fold más reciente.
- `WALKFORWARD_NUM_TESTS = 8`: número de folds generados hacia atrás.
- `HOLDING_PERIOD_MONTHS = 12`: duración de cada posición.
- `ENABLE_FALLBACK_EXTRAPOLATION = True`: si no hay reporte del trimestre exacto, extrapola con los últimos `FALLBACK_LOOK_BACK_QUARTERS = 4` trimestres.

### Pipeline ML

- `MIN_HISTORY_QUARTERS = 4`: mínimo de trimestres por ticker para entrar en entrenamiento.
- `OOF_N_SPLITS = 3`: splits internos para generación de OOF del meta-learner.
- `PURGED_CV_GAP_DAYS = 90`, `EMBARGO_DAYS = 30`: purga y embargo para evitar leakage intra-fold.
- `ENABLE_RECENCY_WEIGHTING = True`, `TRAINING_RECENCY_HALFLIFE_YEARS = 2.0`: los trimestres recientes reciben más peso exponencialmente.
- `NEUTRAL_SCORE_PENALTY_ENABLED = True`: scores exactamente neutrales (0.5) se penalizan a `NEUTRAL_SCORE_PENALIZED_VALUE = 0.25`.
- `DEGENERATE_MODEL_FALLBACK_SCORE = 0.25`: fallback conservador cuando un modelo sectorial no puede entrenarse.
- `SECTOR_SPECIALIST_LONG_FALLBACK_SCORE = 0.25`: alineado con el anterior, para penalizar folds vacíos.

### Walk-forward

- `WALKFORWARD_TRAIN_LOOKBACK_YEARS = 12` (fallback mínimo `WALKFORWARD_TRAIN_MIN_YEARS = 8`): ventana máxima de entrenamiento. El pipeline reduce dinámicamente si la cobertura de tickers es insuficiente.
- `RISK_FREE_RATE = 0.04`: tasa libre de riesgo para Sharpe/Sortino.

### Portfolio

- `PORTFOLIO_MIN_SCORE = 0.57`: umbral de score para entrar en el portafolio.
- `TP_SL_MAX_STOCKS = 7`, `TP_SL_MIN_STOCKS = 3`: tamaño del portafolio final.
- `TP_SL_SECTOR_CAP = 3`: máximo de acciones del mismo sector GICS.
- `PORTFOLIO_MAX_STOCK_WEIGHT = 0.20`: peso máximo por ticker.
- `PORTFOLIO_MAX_STOCKS_PER_SECTOR = 3`.
- `SCORE_WEIGHTED_PORTFOLIO = True`: ponderación lineal por ranking (el #1 pesa el doble que el #N).
- `PORTFOLIO_OPTIMIZER = "hrp"`: optimizador post-ranking (`hrp`, `risk_parity`, `markowitz`).

### Sector prior

- `SECTOR_SCORE_PRIOR_WEIGHT = 0.15`: ajuste aditivo del score sectorial sobre el score final.
- `SECTOR_CONFIDENCE_PEERS = 10`: penaliza sectores con pocos peers: `confidence = min(1, sqrt(n / 10))`.
- `SCORE_DISPERSION_MIN_STD = 0.05`, `SCORE_DISPERSION_MIN_SCALE = 0.35`: agentes con baja dispersión se encogen hacia 0.5.

### Simulación monetaria

- `INITIAL_CAPITAL_USD = 1000.0`, `TRANSACTION_FEE_USD = 1.0`, `SLIPPAGE_PCT = 0.001`.
- `USE_DOLLAR_BACKTEST = True`: simula el portafolio en USD además de métricas de retorno puro.
- `ALLOW_FRACTIONAL_SHARES = True`.

### Baselines

- `RUN_BASELINES = True`: ejecuta benchmark (S&P 500), EW universe, momentum 12m, value combinado y random-TopN.
- `N_RANDOM_BASELINE_SIMS = 100`.

---

## 4. Universo de datos — `step_01_data`

El ETL descarga de Finnhub los siguientes endpoints por ticker:

- `financials_reported`: estados financieros trimestrales y anuales.
- `basic_financials`: ratios derivados (P/E, P/B, ROE, etc.).
- `stock_candles`: precios diarios OHLCV.
- `insider_transactions`: transacciones de insiders.
- `recommendation_trends`: consenso analista (buy/hold/sell).
- `earnings_surprises`: sorpresas de EPS trimestrales.
- `news` (opcional, controlado por `DOWNLOAD_OPTIONAL_ENDPOINTS`): noticias de empresa para FinBERT.
- Datos macro en `_macro/`: VIX, yield curve (10Y-2Y), momentum S&P 500.

El pipeline usa `sp500_historic.csv` para construir el universo dinámico: solo se incluyen tickers que eran miembros del S&P 500 en el período analizado, evitando survivorship bias.

---

## 5. Dataset maestro — `step_02_dataset`

El dataset maestro es una tabla punto-en-tiempo donde cada fila es `(ticker, snapshot_date)`. La fecha de snapshot coincide con la fecha de análisis (modo as-of), usando exclusivamente información publicada hasta ese momento.

### Builders de features

#### `FundamentalFeatureBuilder`
Métricas de calidad de negocio:
- Rentabilidad: `roa`, `roe`, `roic`, `capex_to_revenue`
- Márgenes: `net_margin`, `gross_margin`, `ebitda_margin`, `operating_margin`, `fcf_margin`
- Liquidez: `current_ratio`
- Apalancamiento: `debt_equity`, `debt_to_ebitda`, `interest_coverage`
- Crecimiento: `revenue_yoy_growth`, `fcf_yoy_growth`
- Calidad: `piotroski_fscore`
- Tendencias: `roic_trend_2y`, `eps_growth_trend_3y`

#### `ValuationFeatureBuilder`
Múltiplos de valoración:
- `pe_ratio`, `ps_ratio`, `ev_to_ebitda`, `pb_ratio`
- Yields: `fcf_yield`, `earnings_yield`
- Comparación histórica: `pe_vs_5y_median`, `ev_ebitda_vs_5y_median`

#### `TechnicalFeatureBuilder`
Señales de precio y volumen (ventana `TECHNICAL_LOOKBACK_DAYS = 300`):
- Posición: `price_vs_52w_high`
- Momentum multi-horizonte: `momentum_3m`, `momentum_6m`
- Volatilidad: `volatility_60d`
- Volumen: `vol_ratio_20_50`
- RSI: `rsi_14`, `rsi_28`

#### `SentimentFeatureBuilder`
- Consenso analista: `analyst_consensus`, `analyst_dispersion`, `analyst_consensus_change`
- MSPR (short interest): `mspr_3m`, `mspr_trend`
- Insider: `insider_net_ratio_90d`
- Sorpresas EPS: `eps_surprise_avg_4q`
- NLP: `finbert_sentiment_polarity`, `finbert_risk_intensity`

#### `InsiderFeatureBuilder`
- Actividad insider: ratios de compra/venta neta en ventanas de 30/60/90 días.
- `insider_sell_ratio`, `consecutive_losses`

### Features cross-seccionales (`cross_sectional_features.py`)

Calculadas sobre la sección transversal del snapshot (todos los tickers en la misma fecha):

- Percentiles sectoriales: `fcf_yield_rank_sector`, `roic_rank_sector`, `ev_ebitda_rank_sector`
- Percentiles universales: `quality_rank_universe`, `value_rank_universe`, `piotroski_rank_universe`, `eps_revision_rank_universe`, `beat_rate_rank_universe`
- Interacciones: `value_x_momentum`, `quality_x_lowvol`, `quality_x_value_universe`
- Consistencia de momentum: `momentum_consistency` (fracción de horizontes 1m/3m/6m positivos)

### Features de régimen macro (`regime.py`)

Disponibles para todos los snapshots: `vix`, `yield_curve`, `sp500_momentum_3m`, `sp500_momentum_12m`.

---

## 6. Agentes base — `module/agents/`

### Agente unificado `UniverseTpSlAgent` (`universe_tp_sl.py`)

Clase base de la que heredan los cinco agentes fundamentales/cuantitativos. Flujo interno:

1. **Feature selection**: selecciona las mejores features de su universo usando un RandomForest interno + correlación con el target. `FEATURE_TOP_N = 14` pre-filtro, luego `[FEATURE_IMPORTANCE_MIN_KEEP=6, FEATURE_IMPORTANCE_MAX_KEEP=10]` features finales.
2. **Entrenamiento del modelo**: modelo XGBoost/GBM/RF según el agente.
3. **Rule engine** (si `ENABLE_AGENT_RULE_ENGINE = True`):
   - Minería de reglas single-feature (rangos de percentiles).
   - Minería de reglas de pares de features.
   - Filtrado por `AGENT_RULE_MIN_EDGE = 0.035`, `AGENT_RULE_MIN_SAMPLES = 45`, `AGENT_RULE_MIN_STABILITY = 0.52`.
   - Las reglas que pasan el filtro generan `rule_signal`, `rule_hits`, `rule_confidence` que se añaden como features adicionales al modelo.
   - Blend final: `score_final = clip(model_proba + AGENT_RULE_BLEND * rule_signal, 0, 1)` con `AGENT_RULE_BLEND = 0.22`.
4. **Score**: probabilidad de TP-before-SL en [0, 1]. Valores exactamente 0.5 se penalizan a 0.25.

#### FundamentalAgent (XGBoost)
Detecta calidad de negocio: ROE, ROIC, márgenes, apalancamiento, crecimiento de FCF, Piotroski score, tendencias de ROIC y EPS.
- `FUNDAMENTAL_N_ESTIMATORS = 400`, `MAX_DEPTH = 5`, `LEARNING_RATE = 0.05`

#### ValuationAgent (GBM)
Detecta acciones baratas/caras: P/E, P/B, EV/EBITDA, FCF yield, earnings yield, comparación con mediana histórica a 5 años.
- `VALUATION_N_ESTIMATORS = 200`, `MAX_DEPTH = 4`

#### MomentumAgent (Random Forest)
Detecta tendencias de precio: posición vs máximo 52 semanas, momentum 3m/6m, volatilidad 60d, volumen, RSI.
- `MOMENTUM_N_ESTIMATORS = 300`, `MAX_DEPTH = 8`, `MIN_SAMPLES_LEAF = 5`
- Blends con una componente TFT-lite: `MOMENTUM_DEEP_BLEND_WEIGHT = 0.35`

#### BearAgent (Random Forest híbrido)
Detecta señales de riesgo/deterioro: deuda, cobertura, FCF negativo, quiebras de ingresos, actividad insider bajista.
- `BEAR_N_ESTIMATORS = 300`, `MAX_DEPTH = 5`
- Combina capa de reglas explícitas (45%) con capa ML (55%): `BEAR_RULE_WEIGHT = 0.45`, `BEAR_ML_WEIGHT = 0.55`
- `BEAR_HARD_THRESHOLD = 0.90`: si el score de riesgo supera 0.90, el meta-learner fuerza Underperform.

#### SentimentAgent (Random Forest)
Detecta sentimiento de mercado: consenso analista, MSPR, insider activity, sorpresas EPS, FinBERT.
- `ENABLE_SENTIMENT_AGENT = False` por defecto: análisis empírico mostró que degrada el alpha medio (+10.44% sin vs +9.94% con). Sus features sí se usan en el BearAgent y en el MetaLearner.
- `SENTIMENT_N_ESTIMATORS = 200`, `MAX_DEPTH = 6`, `MIN_SAMPLES_LEAF = 5`

### SectorRotationAgent (`sector_rotation.py`)

Agente top-down que evalúa la fortaleza relativa de cada sector. Features: momentum 3m/6m, posición vs máximo 52s, crecimiento de ingresos, valoración (FCF yield, EV/EBITDA), calidad (ROIC, FCF margin, net margin), sentimiento de analistas, volatilidad. El score sectorial se usa como prior aditivo sobre el score final del ticker.

---

## 7. Meta-learner — `alpha_meta_learner.py`

Triple meta-learner XGBoost con tres cabezas:

### 7.1 Regressor (alpha prediction)
Predice el alpha esperado por ticker. `ALPHA_META_REG_N_ESTIMATORS = 450`, `MAX_DEPTH = 4`, `LEARNING_RATE = 0.03`, L1=0.5, L2=2.0.

### 7.2 Ranker (pairwise ranking)
Ordena cross-sectionalmente los tickers del snapshot. `ALPHA_META_RANK_N_ESTIMATORS = 300`, `MAX_DEPTH = 3`. Contribuye con peso `ALPHA_META_RANK_BLEND = 0.70` al score de régimen.

### 7.3 Risk classifier
Estima probabilidad de pérdida severa. `ALPHA_META_RISK_N_ESTIMATORS = 250`. Penaliza el score final con `ALPHA_META_RISK_BLEND = 0.15`: `final = (1-0.15)*rank_blend + 0.15*(1-risk_score)`.

### Features del meta-learner

**Scores de agentes base:**
`fundamental_score`, `valuation_score`, `momentum_score`, `bear_score`, `sentiment_score`, `sector_score`, `regime_adjusted_score`, `rules_consensus_signal`, `rules_consensus_confidence`

**Coherencia entre agentes** (calculadas en `agent_diagnostics.py`):
`agent_score_mean`, `agent_score_std`, `agent_disagreement`, `bullish_agents`, `bearish_agents`, `agent_contradiction_flag`

**Percentiles cross-seccionales:**
`fcf_yield_rank_sector`, `roic_rank_sector`, `ev_ebitda_rank_sector`, `quality_rank_universe`, `value_rank_universe`, `piotroski_rank_universe`, `eps_revision_rank_universe`, `beat_rate_rank_universe`

**Interacciones y composites:**
`momentum_consistency`, `value_x_momentum`, `quality_x_lowvol`, `quality_x_value_universe`

**Contexto macro:**
`vix`, `yield_curve`, `sp500_momentum_3m`, `sp500_momentum_12m`

### Blend final del score

```
regime_adjusted_score = 0.70 * ranking_score + 0.30 * regime_score
final_meta_score = 0.85 * regime_adjusted_score + 0.15 * (1 - risk_score)
output_score = (1 - 0.40) * meta_score + 0.40 * agent_base_mean
```

El blend con la media de agentes base (`META_BASE_SCORE_BLEND_WEIGHT = 0.40`) actúa como regularizador contra el colapso del meta-learner en folds con pocos datos.

---

## 8. Entrenamiento — `step_03_training`

### `training.py` — Entrenamiento por fold

Flujo por fold de walk-forward:

1. **Preparación**: filtra el dataset al ventana de train, aplica recency weights exponenciales.
2. **Validación temporal de agentes** (`ENABLE_TEMPORAL_AGENT_VALIDATION = True`): evalúa en-fold la calidad predictiva de cada agente por quarter para detectar deterioro. El multiplicador de fiabilidad resultante (`[0.75, 1.25]`) ajusta la dispersión de scores del agente.
3. **Entrenamiento de agentes base**: cada agente entrena su modelo + rule engine.
4. **Diagnósticos de redundancia** (`agent_diagnostics.py`):
   - Correlación entre scores de agentes; pares con correlación > `FEATURE_CORR_THRESHOLD = 0.85` reciben penalización de diversidad (shrink a 0.5).
   - Calidad OOF de reglas: IC, spread top-bottom, estabilidad quarter-a-quarter.
   - Export: `agent_redundancy_summary_<fold>.csv`, `rule_quality_<fold>.csv`, `iterative_research_actions_<fold>.json`.
5. **OOF para meta-learner** (`oof.py`): genera scores OOF con `PurgedEmbargoKFold` (gap=90d, embargo=30d).
6. **Entrenamiento del meta-learner**: sobre los scores OOF de los agentes base + features cross-seccionales.
7. **Predicción del fold de test**: todos los agentes base + meta-learner aplicados al snapshot de test.
8. **Ajuste de dispersión**: agentes con baja dispersión se encogen hacia 0.5.
9. **Regime weighting**: ajuste final por contexto macro.

### `oof.py` — OOF con purga

`PurgedEmbargoKFold` garantiza que las observaciones de validación no contaminen el entrenamiento por proximidad temporal. Las observaciones dentro de 90 días del límite de fold se purgan; las observaciones dentro de 30 días post-límite se embargan.

### `agent_diagnostics.py` — Diagnóstico y bucle iterativo

- Detecta redundancia: si dos agentes tienen scores muy correlados, reduce la diversidad del stacking.
- Mide calidad OOF de señales de reglas por agente: si una regla tiene IC negativo o inestable, su contribución se atenúa automáticamente para ese fold.
- Exporta acciones recomendadas (`iterative_research_actions_<fold>.json`) para el siguiente ciclo de ajuste.

---

## 9. TP/SL adaptativos — `strategy.py` y `backtesting.py`

### Calibración de TP/SL por ticker

Los niveles no son fijos: se calibran por historial propio del ticker usando:

- `TP_SL_BASE_TP = 0.15` (15%), `TP_SL_BASE_SL = 0.10` (10%) como base para holding de 12 meses.
- Sensibilidad al score: `TP_SL_TP_SENSITIVITY = 0.10`, `TP_SL_SL_SENSITIVITY = 0.04`.
- Límites duros: TP en [8%, 55%], SL en [8%, 35%].
- **TP Edge overlay** (`TP_EDGE_ENABLE = True`): ajusta la confianza usando el historial de TP-before-SL del ticker en la ventana de train del fold actual. Prior bayesiano con `TP_EDGE_PRIOR_STRENGTH = 10`, `TP_EDGE_RELIABILITY_K = 12`. Blend: `TP_EDGE_CONFIDENCE_BLEND = 0.10`.
- **Penalización de TP stretch**: si el TP propuesto es más ambicioso que lo que el ticker suele alcanzar históricamente, se aplica una penalización de factibilidad: `TP_EDGE_TP_STRETCH_PENALTY = 0.60`, `TP_EDGE_MIN_FEASIBILITY = 0.35`.

### Estrategia de trailing stop

Una vez que el precio cruza el TP, la posición no se cierra inmediatamente. En cambio, el TP se convierte en un floor para un trailing stop que se revisa cada `TP_SL_TRAILING_REVIEW_DAYS = 30` días. El trailing stop se fija en el percentil `TP_SL_TRAILING_DRAWDOWN_QUANTILE = 0.65` de las correcciones históricas desde máximos del ticker.

### Grace period

Los primeros `TP_SL_GRACE_PERIOD_FRACTION = 0.5` del holding (primeros 6 meses para un holding de 12) son un período de gracia donde ni TP ni SL pueden ejecutarse, evitando salidas prematuras por volatilidad inicial.

### Fine-tuning de estrategia

Si la estrategia base tiene hit rate < `TP_SL_FINE_TUNE_MIN_HIT_RATE = 0.48` o utilidad < 0, el evaluador prueba automáticamente hasta `TP_SL_FINE_TUNE_MAX_RELAX_STEPS = 2` variantes relajadas.

### Selección final de tickers

La puntuación de ranking de cada ticker combina:
- Score meta-learner (principal).
- Reglas consensus: `risk_benefit_score *= (1 + 0.25 * rules_consensus_signal)`.
- `TP_SL_MIN_ACCEPTABLE_TP = 0.12`: mínimo TP aceptable para entrar en el portafolio.
- Peso de certeza (`TP_SL_SELECTION_CERTAINTY_WEIGHT = 0.35`) vs calidad de TP (`TP_SL_SELECTION_TP_QUALITY_WEIGHT = 0.25`).

---

## 10. Evaluación y backtesting — `step_04_evaluation`

### `evaluator.py` — Loop walk-forward

Por cada fold de test:

1. Obtiene el snapshot de test (features punto-en-tiempo).
2. Ejecuta todos los agentes base + meta-learner para generar scores.
3. Aplica el selector de estrategia TP/SL adaptativo.
4. Construye el portafolio con restricciones de sector.
5. Simula el holding en precios reales: el backtester lee precios diarios y registra si se ejecutó TP, SL, trailing stop, o se llegó al vencimiento del holding ("NONE").
6. Calcula métricas por fold: retorno, alpha vs benchmark, Sharpe, AUC.
7. Exporta artefactos: portfolio trail CSV, SHAP, scores completos.

### `backtesting.py` — Motor de simulación

- Simula tick a tick (día a día) el holding de cada posición.
- `TP_SL_MAX_HOLDING_DAYS = 365`.
- Registra: fecha de salida, razón de salida (TP/SL/NONE), retorno efectivo.
- El retorno del portafolio usa la salida TP/SL-first (no el cierre del trimestre), evitando inflar retornos con ganancias post-SL.
- Exporta `ticker_exit_dates` y `ticker_exit_reasons` para debugging.

### `analysis.py` — Métricas globales

Consolida todos los folds y calcula:
- Alpha medio por fold, % folds con alpha positivo.
- Sharpe, Sortino, Calmar, max drawdown globales (estrategia vs benchmark).
- Desglose por longitud de ventana de train.
- Comparación con baselines: benchmark S&P 500, EW universe, momentum 12m, value combinado, random TopN.

### `explainability.py` — SHAP

Genera valores SHAP por agente y fold para explicar qué features impulsan las decisiones en cada período.

---

## 11. Grid de escenarios — `analyzer_II.py`

Ejecuta 9 experimentos en paralelo (3×3 grid) sobre los parámetros de selección final de tickers:

| Dimensión | Valores |
|-----------|---------|
| `TP_SL_MIN_ACCEPTABLE_TP` | 0.06, 0.07, 0.09 |
| Perfil de selección | certainty-heavy (0.45/0.18), balanced (0.35/0.25), TP-quality-heavy (0.28/0.34) |

El experimento de control es `exp05_floor007_cert035_q025` (balanced, floor 7%). El resto de parámetros del pipeline se mantienen fijos para que la comparación sea interpretable.

Al finalizar todos los runs, el script consolida los resultados de folds comunes y genera una gráfica comparativa de alpha por fold.

---

## 12. Resultados del último run

**Run:** `anchor_20260503_h12_n8` — 8 folds generados desde 2026-05-03 hacia atrás en pasos de 12 meses. De los 8 folds generados, 6 completan con datos suficientes.

### Métricas globales (6 folds: 2020–2026)

| Métrica | Estrategia | Benchmark (S&P 500) |
|---------|-----------|---------------------|
| Retorno acumulado | +271.0% | +133.0% |
| Retorno anualizado | +24.9% | +15.4% |
| Sharpe | 1.079 | 0.691 |
| Sortino | 1.456 | 0.934 |
| Max Drawdown | -23.3% | -25.4% |
| Calmar | 1.071 | 0.607 |
| Win rate (folds) | 55.6% | 54.4% |

**Alpha medio por fold: +9.23% — 67% de folds con alpha positivo.**

### Detalle por fold

| Fold | Train | Período | Ret Estrategia | Ret Benchmark | Alpha | Sharpe |
|------|-------|---------|---------------|--------------|-------|--------|
| T02 | 8Y | 2020-05-03 → 2021-05-03 | +42.3% | +47.5% | -5.2% | 1.570 |
| T03 | 9Y | 2021-05-03 → 2022-05-03 | +17.4% | -0.4% | +17.8% | 0.804 |
| T04 | 11Y | 2022-05-03 → 2023-05-03 | +26.6% | -2.0% | +28.6% | 0.939 |
| T05 | 12Y | 2023-05-03 → 2024-05-03 | +51.7% | +25.4% | +26.3% | 2.354 |
| T06 | 12Y | 2024-05-03 → 2025-05-02 | +11.2% | +10.9% | +0.3% | 0.496 |
| T07 | 12Y | 2025-05-03 → 2026-04-02 | +4.0% | +16.5% | -12.5% | 0.090 |

### Simulación USD (capital inicial $1.000)

| Run | Capital final | Retorno total | Max DD | Sharpe |
|-----|--------------|--------------|--------|--------|
| Estrategia | $2.512 | +153.2% | -24.0% | 0.869 |
| Benchmark S&P 500 | $2.308 | +131.2% | -25.4% | 0.683 |
| Momentum 12m | $1.915 | +93.0% | -34.6% | 0.387 |
| Random Top-N (media 100 sims) | $2.010 | +103% | -22.9% | 0.553 |
| Value combinado | $2.535 | +155.5% | -29.3% | 0.643 |

Fees totales (6 folds × 14 USD/fold = 84 USD) ya descontados.

---

## 13. Integridad temporal y anti-leakage

El sistema aplica múltiples capas de protección:

- **`asof.py`**: todas las features se validan como disponibles en la `snapshot_date` de análisis. No se puede usar ningún dato publicado después de esa fecha.
- **`feature_policy.py`**: cada feature tiene una política de recencia que define cuántos días pueden pasar desde su última actualización antes de considerarla stale.
- **`purged_cv.py`**: `PurgedEmbargoKFold` con gap=90d y embargo=30d para la generación de OOF del meta-learner.
- **Targets forward**: `forward_return` y outcomes TP/SL se calculan usando solo precios posteriores a la fecha de entrada.
- **Walk-forward estricto**: el modelo del fold T+1 nunca ve datos del período de test del fold T.
- **Audit de leakage**: `leakage_audit.csv` exportado por run.

---

## 14. Módulos de utilidad — `module/common`

| Módulo | Función |
|--------|---------|
| `asof.py` | Validaciones punto-en-tiempo |
| `cross_sectional_features.py` | Percentiles sectoriales y universales en cada snapshot |
| `data_router.py` | Enrutamiento de endpoints Finnhub por ticker |
| `feature_controls.py` | Disponibilidad de features por fecha y política |
| `feature_policy.py` | Reglas de recencia por tipo de feature |
| `finbert_features.py` | Sentimiento NLP con FinBERT (polarity, risk intensity) |
| `performance_metrics.py` | Sharpe, Sortino, Calmar, max drawdown, win rate |
| `portfolio_optimization.py` | HRP (Hierarchical Risk Parity), Risk Parity, Markowitz |
| `purged_cv.py` | `PurgedEmbargoKFold` con gap y embargo configurables |
| `recency_weights.py` | Pesos exponenciales: `weight = 2^(-age_years / halflife)` |
| `regime.py` | Detección de régimen (bull/bear/neutral) y encoding para features |
| `target_engineering.py` | Construcción de targets TP/SL forward sin leakage |

---

## 15. Tests

`tests/test_antileakage_and_policy.py`: suite de tests que valida:
- Que ninguna feature se use antes de su fecha de disponibilidad.
- Que el `PurgedEmbargoKFold` no genere solape entre train y validation.
- Que los targets forward sean estrictamente posteriores a la snapshot date.

---

## 16. Flujo operativo completo

```
1. DESCARGA (step_01_data)
   └─ Finnhub API → data_finnhub/<TICKER>/
   └─ Macro: VIX, yield curve, SP500 candles

2. PREPARACIÓN (step_01_data/consolidation.py)
   └─ Parseo, normalización, consolidación por ticker
   └─ Filtrado de tickers con historia insuficiente

3. DATASET (step_02_dataset/dataset.py)
   └─ Tabla (ticker, snapshot_date) punto-en-tiempo
   └─ Builders: fundamental, valuation, technical, sentiment, insider
   └─ Features cross-seccionales (percentiles sector/universo)
   └─ Contexto macro por snapshot

4. WALK-FORWARD (step_04_evaluation/evaluator.py)
   Por cada fold T_k:
   │
   ├─ TRAINING (step_03_training/training.py)
   │   ├─ Recency weights
   │   ├─ Validación temporal de agentes
   │   ├─ Entrenamiento agentes base + rule engine
   │   ├─ Diagnósticos de redundancia y calidad OOF de reglas
   │   ├─ OOF con PurgedEmbargoKFold
   │   └─ Entrenamiento meta-learner (regressor + ranker + risk)
   │
   ├─ PREDICCIÓN (fold de test)
   │   ├─ Scores de agentes base
   │   ├─ Features cross-seccionales del snapshot de test
   │   └─ Score final del meta-learner
   │
   ├─ ESTRATEGIA TP/SL
   │   ├─ Calibración adaptativa por ticker (TP edge + trailing stop)
   │   ├─ Fine-tuning automático si hit rate < umbral
   │   └─ Selección de portafolio (7-3, sector cap, score min)
   │
   ├─ BACKTESTING
   │   ├─ Simulación día a día hasta TP/SL/vencimiento
   │   └─ Retorno efectivo con salida TP/SL-first
   │
   └─ MÉTRICAS Y EXPORT
       ├─ Alpha, Sharpe, drawdown vs benchmark
       ├─ SHAP por agente
       ├─ Portfolio trail CSV
       └─ Artefactos de auditoría

5. ANÁLISIS GLOBAL
   ├─ Consolidación de todos los folds
   ├─ Comparación con baselines (S&P500, momentum, value, random)
   ├─ Gráficas: equity curve, PnL por fold, drawdown, heatmaps
   └─ final_summary.json + report.txt
```
