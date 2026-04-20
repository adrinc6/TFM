# Documentacion Tecnica

## Sistema Multi-Agente de Seleccion de Acciones

Este documento resume la referencia tecnica del proyecto: componentes, contratos principales, variables de configuracion y flujo de datos por etapa.

## 1. Componentes del proyecto

Estructura funcional:

- Orquestacion: `analyzer.py`, `analyzer_ticker.py`.
- Configuracion: `environment.py`.
- Agentes: `module/agents/`.
- Utilidades compartidas: `module/common/`.
- Pipeline por etapas: `module/steps/step_01_data` a `step_04_evaluation`.
- Datos y artefactos: `data_finnhub/`, `results/`.
- Calidad: `tests/`.

## 2. Contratos de clases y metodos clave

### 2.1 Base de agentes

`BaseAgent` define el contrato comun:

- `fit(X, y, **kwargs)`: entrena el agente.
- `predict_score(X)`: devuelve scores en `[0, 1]`.
- utilidades de limpieza, diagnostico y exportacion de artefactos.

`FeatureSelector` aplica seleccion por correlacion/importancia para reducir ruido y redundancia.

### 2.2 Agentes especializados

- `FundamentalAgent`: calidad financiera y fortaleza estructural.
- `ValuationAgent`: valoracion relativa e historica.
- `MomentumAgent`: tendencia tecnica y persistencia de retornos.
- `BearAgent`: riesgo bajista con capa de reglas + capa ML.
- `SentimentAgent`: consenso de analistas, insiders y sorpresas.
- `SectorRotationAgent`: score top-down por sector.
- `SectorSpecializedAgent`: entrenamiento por sector con fallback si falta muestra.

### 2.3 Capa meta

La capa meta combina outputs base y contexto para score final de seleccion.

- camino clasico: `meta_learner.py` (stacking LR + GBM),
- camino evolucionado alpha: `alpha_meta_learner.py` (ranking y ajuste por riesgo/regimen).

### 2.4 Utilidades de datos y seguridad

- `DataRouter`: acceso centralizado a precios y consolidado, snapshots as-of, ventanas temporales y retornos forward.
- `filter_asof`: filtro point-in-time anti-leakage.
- `feature_policy`: controla que las features sean comparables (ratios/normalizaciones).
- validacion de tickers/rutas para evitar path-traversal.

### 2.5 Entrenamiento y evaluacion

- `train_fold` (`step_03_training/training.py`): entrenamiento de agentes y capa meta por fold.
- generacion OOF (`step_03_training/oof.py`): predicciones fuera de fold para capa meta sin leakage.
- `run_walkforward_pipeline` (`step_04_evaluation/evaluator.py`): orquestacion de evaluacion temporal.
- `WalkForwardBacktester` y `portfolio_simulator.py`: simulacion de cartera y capital.
- `explainability.py`: explicaciones SHAP y drivers.

## 3. Variables de configuracion (resumen)

Bloques principales en `environment.py`:

- flags de ejecucion (`SKIP_BACKTEST`, `FORCE_DOWNLOAD`, `RUN_ABLATION_STUDY`, etc.),
- rutas de entrada/salida,
- universo de tickers y uso de universo historico dinamico,
- rango temporal y frecuencia,
- parametros ML y OOF,
- parametros de cartera/backtest (capital, fees, slippage, limites),
- parametros de robustez de scoring,
- semillas de reproducibilidad.

Recomendacion operativa:

- tratar `environment.py` como fuente unica de configuracion,
- versionar cambios relevantes en reportes de ejecucion,
- no mezclar experimentos incompatibles en una misma carpeta de resultados.

## 4. Flujo de datos por etapa

### Step 01 - ETL

Entrada:

- universo de tickers,
- fechas de descarga,
- claves API.

Proceso:

- descarga de endpoints financieros y precios,
- consolidacion por ticker,
- persistencia de estado para reintentos.

Salida:

- datos crudos por ticker,
- consolidado por ticker,
- datos macro auxiliares.

### Step 02 - Dataset maestro

Entrada:

- consolidado + precios por ticker.

Proceso:

- construccion de snapshots punto-en-tiempo,
- generacion de families de features,
- calculo de retorno futuro y etiquetas.

Salida:

- dataset maestro con indice temporal por ticker y columnas de features/target.

### Step 03 - Training

Entrada:

- subconjuntos train/test del fold,
- mapa sectorial y contexto de mercado.

Proceso:

- entrenamiento de agentes,
- OOF temporal,
- entrenamiento de capa meta,
- generacion de scores finales.

Salida:

- modelos por fold,
- scores por ticker,
- artefactos de diagnostico.

### Step 04 - Evaluation

Entrada:

- scores de fold,
- precios para simulacion,
- reglas de construccion de cartera.

Proceso:

- seleccion de posiciones,
- simulacion de retornos,
- metricas financieras,
- comparativa con baselines,
- explicabilidad y auditoria.

Salida:

- resumen global,
- detalle por fold,
- series de retornos/equity,
- reportes y visualizaciones.

## 5. Variables internas criticas (conceptual)

Durante dataset/training/evaluation son especialmente relevantes:

- fecha de snapshot y ventana de datos as-of,
- retorno forward usado para target,
- escalado por dispersion de scores,
- score/prior sectorial y confianza por numero de peers,
- capital encadenado en simulacion monetaria,
- auditorias de leakage por fold.

## 6. Seguridad y robustez

Medidas implementadas:

- validacion estricta de identificadores de ticker,
- control de rutas para evitar accesos fuera del directorio de datos,
- tolerancia a fallos en etapas parciales,
- fallbacks neutros en componentes no disponibles,
- pruebas automatizadas para integridad temporal y politica de features.

## 7. Resultado tecnico esperado

Una ejecucion completa debe producir:

- artefactos reproducibles de entrenamiento y evaluacion,
- reportes por fold y globales,
- trazabilidad de decisiones por ticker,
- comparacion cuantitativa frente a benchmark y estrategias baseline.
