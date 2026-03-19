# Documentacion - TFM: Multi-Agente ML Stock Picker

## Indice

1. Vision general
2. Arquitectura y estructura del proyecto
3. Flujo de ejecucion
4. Modulos principales
5. Agentes
6. Configuracion global (environment.py)
7. Salidas y resultados

---

## 1. Vision general

Pipeline multi-agente para seleccion de acciones con evaluacion walk-forward y fold live out-of-sample. Usa Finnhub para fundamentales y Yahoo Finance HTTP directo para precios. Todo el flujo respeta orden temporal y evita leakage.

---

## 2. Arquitectura y estructura del proyecto

```
TFM/
├── analyzer.py
├── environment.py
├── requirements.txt
├── DOCUMENTATION.md
├── data_finnhub/
│   ├── {TICKER}/
│   │   ├── profile.json
│   │   ├── prices.json
│   │   ├── financials_reported_quarterly.json
│   │   ├── financials_reported_annual.json
│   │   ├── basic_financials.json
│   │   ├── eps_surprises.json
│   │   ├── recommendation_trends.json
│   │   ├── insider_transactions.json
│   │   └── insider_sentiment.json
│   ├── _macro/
│   │   ├── vix.json
│   │   ├── sp500.json
│   │   ├── us10y.json
│   │   └── us2y.json
│   └── consolidated/
│       └── {TICKER}.csv
├── results/
│   ├── pipeline.log
│   ├── master_dataset.csv
│   ├── agents/
│   ├── backtest/
│   └── plots/
└── module/
    ├── __init__.py
    ├── agents/
    │   ├── base.py
    │   ├── fundamental.py
    │   ├── valuation.py
    │   ├── momentum.py
    │   ├── bear.py
    │   ├── sentiment.py
    │   └── meta_learner.py
    ├── common/
    │   ├── __init__.py
    │   └── data_router.py
    └── steps/
        ├── step_01_data/
        │   ├── clients.py
        │   ├── downloaders.py
        │   ├── parsers.py
        │   ├── consolidation.py
        │   ├── registry.py
        │   └── pipeline.py
        ├── step_02_dataset/
        │   ├── builders/
        │   │   ├── fundamental.py
        │   │   ├── technical.py
        │   │   ├── valuation.py
        │   │   ├── insider.py
        │   │   ├── sentiment.py
        │   │   └── sector.py
        │   ├── dataset.py
        │   └── normalization.py
        ├── step_03_training/
        │   ├── agent_config.py
        │   ├── oof.py
        │   └── training.py
        ├── step_04_evaluation/
        │   ├── evaluator.py
        │   ├── backtester.py
        │   ├── metrics.py
        │   ├── visualization.py
        │   ├── reports.py
        │   ├── explainability.py
        │   └── ablation.py
        └── step_05_live/
            ├── live_fold.py
            ├── live_prices.py
            └── returns.py
```

---

## 3. Flujo de ejecucion

```
analyzer.py::main()
  1) step_01_data.pipeline.download_data
  2) step_01_data.pipeline.prepare_data
  3) DataRouter + get_available_tickers
  4) step_02_dataset.dataset.build_master_dataset
  5) step_04_evaluation.evaluator.run_walkforward_pipeline
  6) step_05_live.live_fold.run_live_fold
```

---

## 4. Modulos principales

### 4.1 Step 01 - Datos

- clients.py: clientes Finnhub y Yahoo
- downloaders.py: descarga por ticker y series macro
- parsers.py: parsers de payloads Finnhub (SEC, EPS, recomendaciones, insiders)
- consolidation.py: consolidacion trimestral y ratios
- registry.py: tracking de estado de descarga
- pipeline.py: entrypoints de descarga y consolidacion

### 4.2 Step 02 - Dataset

- builders/*: builders por dominio (fundamental, tecnico, valoracion, insiders, sentiment)
- dataset.py: construccion de dataset maestro y features live
- normalization.py: normalizacion sectorial

### 4.3 Step 03 - Entrenamiento

- agent_config.py: configuracion de agentes base
- oof.py: generacion de scores OOF
- training.py: entrenamiento por fold y entrenamiento final

### 4.4 Step 04 - Evaluacion

- evaluator.py: loop walk-forward
- backtester.py: simulacion de cartera y pipeline de metrics
- metrics.py: metricas de performance
- visualization.py: graficos headless
- reports.py: reporte textual
- explainability.py: SHAP local y global
- ablation.py: estudio de ablacion

### 4.5 Step 05 - Live

- live_fold.py: orquestacion del fold live
- live_prices.py: descarga de precios live en memoria
- returns.py: helpers de retornos

---

## 5. Agentes

- FundamentalAgent: salud financiera, rentabilidad y crecimiento
- ValuationAgent: multiples relativos y senales de analistas
- MomentumAgent: indicadores tecnicos y contexto macro
- BearAgent: riesgo de deterioro y señales defensivas
- SentimentAgent: consenso de analistas, insiders y EPS
- MetaLearner: stacking de scores de agentes base

---

## 6. Configuracion global (environment.py)

Variables principales:

- START_DATE, END_DATE: ventana historica
- FORWARD_RETURN_DAYS: horizonte del label
- MIN_HISTORY_QUARTERS: minimo de trimestres por ticker
- SECTOR_ZSCORE_MIN_PEERS: minimo de peers por sector
- WALKFORWARD_TRAIN_YEARS, WALKFORWARD_TEST_QUARTERS
- OOF_N_SPLITS
- RISK_FREE_RATE
- SKIP_BACKTEST, FORCE_DOWNLOAD, RETRY_MISSING_TICKERS

---

## 7. Salidas y resultados

- results/master_dataset.csv: dataset maestro
- results/agents/: diagnosticos, importancias y predicciones por agente
- results/backtest/: metricas por fold y portfolio
- results/plots/: graficos headless
- results/pipeline.log: log completo de ejecucion
