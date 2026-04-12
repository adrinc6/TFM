# MEMORIA DE TRABAJO FIN DE MÁSTER

## Sistema Multi-Agente de Machine Learning para Selección de Acciones en el S&P 500

---

## 1. Introducción y motivación

La selección activa de acciones (*stock picking*) constituye uno de los desafíos más relevantes y persistentes de las finanzas cuantitativas. A pesar de la hipótesis de los mercados eficientes formulada por Fama (1970), la evidencia empírica muestra que determinadas anomalías de mercado —como el efecto momentum (Jegadeesh y Titman, 1993), el value premium (Fama y French, 1992) y la calidad de los fundamentales (Piotroski, 2000)— pueden ser explotadas sistemáticamente mediante modelos cuantitativos.

Los avances recientes en *machine learning* (ML) han permitido integrar múltiples fuentes de información (fundamentales, técnicas, de sentimiento) en modelos que capturan relaciones no lineales entre variables financieras y retornos futuros. Sin embargo, la aplicación naive de ML a datos financieros presenta riesgos severos: *overfitting*, *look-ahead bias* (sesgo de anticipación) y la ilusión de rendimientos pasados que no se materializan fuera de muestra.

Este Trabajo Fin de Máster aborda el problema de la selección de acciones dentro del universo del S&P 500 mediante el diseño, implementación y evaluación de un sistema multi-agente de ML. El sistema integra seis agentes especializados —cada uno enfocado en un dominio específico del análisis financiero— y un meta-learner que combina sus opiniones para producir una señal final de inversión. La contribución principal reside en la arquitectura completa del pipeline, que incorpora garantías anti-leakage rigurosas, validación *walk-forward* realista y un backtesting monetario que simula costes de transacción y slippage.

La motivación práctica es clara: proporcionar un sistema end-to-end que un analista cuantitativo o gestor de carteras pueda utilizar para generar recomendaciones fundamentadas, explicables y reproducibles, con un proceso de validación que refleje las condiciones reales de inversión.

---

## 2. Objetivos del proyecto

Los objetivos de este TFM se articulan en tres niveles:

### 2.1 Objetivos generales
1. Diseñar e implementar un pipeline completo de ML para la selección de acciones en el S&P 500.
2. Evaluar la viabilidad de un enfoque multi-agente frente a modelos monolíticos.
3. Garantizar la ausencia de *look-ahead bias* en todo el proceso.

### 2.2 Objetivos específicos
1. Implementar seis agentes especializados que analicen dimensiones complementarias: fundamentales, valoración, momentum, riesgo (bear), sentimiento y rotación sectorial.
2. Diseñar un meta-learner basado en *stacking* (Logistic Regression + Gradient Boosting) que combine las señales de los agentes base.
3. Construir un backtester *walk-forward* con simulación monetaria en USD que incluya costes de transacción, *slippage* y capital encadenado entre folds.
4. Implementar explicabilidad a nivel de ticker mediante SHAP y reglas heurísticas.
5. Desarrollar un sistema de ETL robusto que descargue, consolide y normalice datos de múltiples fuentes (Finnhub API, Yahoo Finance).
6. Proveer baselines de comparación: benchmark S&P 500, estrategia de momentum, estrategia value, y selección aleatoria (Monte Carlo).

### 2.3 Objetivos de calidad
1. Implementar una política estricta de features que solo permita ratios y magnitudes normalizadas, excluyendo valores absolutos que podrían introducir sesgo por tamaño de empresa.
2. Generar reportes de auditoría por fold que documenten la traza de datos utilizada.
3. Incorporar tests automatizados para las garantías anti-leakage y la política de features.
4. Validar las entradas del sistema (tickers, rutas) para prevenir vulnerabilidades de seguridad como path-traversal.

---

## 3. Estado del arte

### 3.1 Machine learning aplicado a stock picking

La aplicación de ML en la predicción de retornos bursátiles ha evolucionado significativamente desde los primeros trabajos con redes neuronales artificiales en los años 90. Gu, Kelly y Xiu (2020) demostraron que modelos no lineales —en particular árboles de decisión potenciados (*gradient boosting*)— superan sistemáticamente a los modelos lineales tradicionales en la predicción de retornos accionarios cross-seccionales.

XGBoost (Chen y Guestrin, 2016) se ha convertido en uno de los algoritmos más utilizados en finanzas cuantitativas debido a su capacidad para manejar datos faltantes, su regularización implícita y su eficiencia computacional. Random Forest (Breiman, 2001) complementa esta aproximación con su capacidad de estimar incertidumbre y reducir varianza.

En este TFM, cada agente utiliza un algoritmo adaptado a su dominio: XGBoost para fundamentales (donde las interacciones entre ratios son cruciales), Random Forest para momentum y sentimiento (donde la estabilidad de las estimaciones es prioritaria), y Gradient Boosting para valoración y rotación sectorial.

### 3.2 Walk-forward validation

La validación cruzada estándar (K-Fold) es inadecuada para datos financieros porque viola la estructura temporal. Bailey, Borwein, López de Prado y Zhu (2014) demostraron que incluso K-Fold con embargo (*purging*) puede producir estimaciones infladas del rendimiento.

La validación *walk-forward* es el estándar de facto en finanzas cuantitativas: el modelo se entrena con una ventana temporal creciente y se evalúa exclusivamente en el periodo inmediatamente posterior. Este TFM implementa un walk-forward con las siguientes características:

- **Ventana de entrenamiento dinámica**: intenta un máximo configurable (por defecto 10 años) y se reduce automáticamente hasta un mínimo (5 años) si la cobertura del universo de test es insuficiente.
- **Frecuencia configurable**: trimestral o anual, con fecha ancla personalizable.
- **Out-of-fold (OOF) para el meta-learner**: los scores que alimentan al meta-learner se generan con TimeSeriesSplit interno, nunca con datos de test.
- **Auditoría de leakage**: cada fold genera un informe estructurado que verifica la ausencia de datos futuros en las features de entrenamiento.

### 3.3 Sistemas multi-agente en finanzas

La idea de combinar múltiples modelos especializados tiene raíces en la teoría de *ensemble learning* (Dietterich, 2000) y en la arquitectura de agentes inteligentes (Wooldridge, 2009). En finanzas, los sistemas multi-agente se han utilizado para simular mercados (LeBaron, 2006) y para combinar señales de trading (Dempster y Leemans, 2006).

La arquitectura de este TFM se inspira en la práctica de los fondos cuantitativos multi-estrategia, donde equipos especializados generan señales independientes que un comité central pondera. Cada agente funciona como un "analista virtual" con su propia perspectiva:

- **FundamentalAgent**: analiza la calidad financiera (ROE, márgenes, Piotroski F-Score).
- **ValuationAgent**: evalúa si la acción está infravalorada respecto a múltiplos históricos.
- **MomentumAgent**: detecta tendencias de precio y momentum de beneficios.
- **BearAgent**: identifica riesgos financieros críticos (deuda excesiva, pérdidas consecutivas).
- **SentimentAgent**: procesa señales de analistas e insiders.
- **SectorRotationAgent**: opera a nivel macro-sectorial determinando qué sectores superarán al índice.

El meta-learner actúa como un *portfolio manager* algorítmico que integra estas señales, detecta consensos y aplica ajustes de robustez.

### 3.4 Explicabilidad con SHAP

SHAP (*SHapley Additive exPlanations*, Lundberg y Lee, 2017) proporciona una descomposición aditiva de las predicciones basada en la teoría de juegos cooperativos. En un contexto de inversión, la explicabilidad no es solo un requisito regulatorio sino una herramienta práctica: permite al gestor entender *por qué* el modelo recomienda una acción y evaluar si las razones son consistentes con su visión del mercado.

Este TFM implementa explicaciones SHAP a nivel de agente y ticker, complementadas con reglas heurísticas cuando SHAP no está disponible (por ejemplo, para el BearAgent que combina reglas explícitas con ML).

---

## 4. Descripción del sistema desarrollado

### 4.1 Arquitectura general

El sistema se organiza como un pipeline secuencial de cuatro pasos:

1. **ETL (step_01_data)**: Descarga de datos desde Finnhub API y Yahoo Finance, parsing de filings SEC (10-Q, 10-K), y consolidación en un CSV por ticker.
2. **Construcción del dataset (step_02_dataset)**: Generación de features para cada combinación (ticker, quarter) respetando estrictamente el punto temporal.
3. **Entrenamiento (step_03_training)**: Entrenamiento de los seis agentes base y el meta-learner con generación de scores OOF anti-leakage.
4. **Evaluación (step_04_evaluation)**: Walk-forward backtesting con simulación monetaria, baselines y explicabilidad SHAP.

La configuración global reside en `environment.py`, que centraliza más de 100 parámetros organizados en nueve secciones: flags de ejecución, claves API, rutas, universo de tickers, periodo de análisis, parámetros ML, walk-forward, hiperparámetros de agentes y reproducibilidad.

### 4.2 Pipeline de datos (ETL)

El módulo de datos implementa:

- **FinnhubClient**: cliente HTTP con rate limiting (1 request/segundo para el tier gratuito) y manejo de errores HTTP 429.
- **YahooClient**: descarga de precios OHLCV sin rate limiting.
- **FinnhubSECParser**: extracción de datos financieros desde filings SEC con mapeo de tags XBRL.
- **FinnhubConsolidator**: fusión de datos trimestrales (10-Q), anuales (10-K) y ratios precalculados en un CSV consolidado por ticker.
- **Registry**: sistema de persistencia JSON que rastrea el estado de descarga por ticker/endpoint para evitar re-descargas innecesarias.

### 4.3 Construcción de features

Los feature builders son cinco módulos especializados:

- **FundamentalFeatureBuilder**: crecimientos YoY (revenue, EPS, FCF), ratios de calidad (accruals, capex/revenue), Piotroski F-Score (8 señales binarias), tendencias de 2-3 años en ROE y márgenes.
- **TechnicalFeatureBuilder**: RSI (14, 28 días), MACD con señal e histograma, medias móviles (SMA 20/50/200), Bandas de Bollinger (%B), ATR, momentum (1m/3m/6m/12m), volatilidad (20d/60d).
- **ValuationFeatureBuilder**: múltiplos (P/E, P/B, P/S, EV/EBITDA), yields (FCF, earnings), comparación vs mediana de 5 años.
- **InsiderFeatureBuilder**: ratio neto de compras/ventas de insiders (90 días), MSPR y su tendencia.
- **SentimentFeatureBuilder**: ratio de recomendaciones buy, score bearish, consenso, dispersión, sorpresa de EPS, revisiones.

Todos los builders utilizan `filter_asof()` para garantizar que solo se accede a datos disponibles hasta la fecha del snapshot.

### 4.4 Agentes y meta-learner

Cada agente sigue el contrato de `BaseAgent`: recibe un DataFrame de features y un vector de labels, entrena un modelo de clasificación binaria, y devuelve scores [0, 1] donde 1 indica señal alcista (Outperform).

El **BearAgent** es único por su arquitectura dual: una capa de reglas con 10 flags de riesgo ponderados (60% peso) y una capa ML (40% peso). Los flags incluyen crecimiento excesivo de deuda, pérdidas consecutivas, insuficiente cobertura de intereses y ventas de insiders.

El **MetaLearner** implementa *stacking* de dos niveles: Logistic Regression (interpretable) + Gradient Boosting (captura no-linealidades). Los pesos de ambos se calibran por AUC en validación cruzada temporal. Además, genera features de consenso entre agentes: media, desviación estándar, cuenta de agentes alcistas, y score ponderado por confianza.

### 4.5 Normalización sectorial

El `SectorNormalizer` calcula Z-scores intra-sectoriales para features fundamentales, evitando que las diferencias inherentes entre sectores (ej. márgenes más altos en tecnología vs. retail) dominen las señales. Se requiere un mínimo de 3 peers por sector para calcular estadísticas fiables. La normalización se ajusta en train y se aplica a test, evitando leakage.

---

## 5. Metodología

### 5.1 Walk-forward backtesting

El backtester genera folds chronológicos donde cada fold:
1. Define una ventana de entrenamiento [train_start, train_end].
2. Entrena los seis agentes base con datos hasta train_end.
3. Genera scores OOF con TimeSeriesSplit interno (3 folds) para alimentar al meta-learner.
4. Entrena el meta-learner con los scores OOF.
5. Aplica ajustes de robustez: *dispersion shrink* (contrae scores hacia 0.5 cuando la dispersión es baja), *sector prior* (ajusta el score por la confianza del SectorRotationAgent), y *hard risk gate* (fuerza score a 0.05 si el BearAgent detecta riesgo extremo).
6. Selecciona los Top-N stocks por score final, con ponderación lineal por ranking.
7. Simula la inversión en USD con capital encadenado entre folds.

### 5.2 Garantías anti-leakage

El sistema implementa múltiples capas de protección:

- **filter_asof**: toda consulta de datos se filtra por fecha <= as_of.
- **Filing date map**: las features fundamentales se alinean con la fecha de filing (no la fecha del reporte), garantizando que solo se usan datos que estaban públicamente disponibles.
- **Snapshot lag**: un retraso configurable (por defecto 60 días) entre el cierre del trimestre y la fecha de entrada simula el tiempo real de publicación.
- **Auditoría por fold**: cada fold genera un JSON con el resultado de `assert_no_future_data()` para cada componente.
- **OOF temporal**: los scores base para el meta-learner se generan con splits que respetan el orden temporal.
- **Política de features**: solo ratios y magnitudes normalizadas pasan el filtro; valores absolutos (revenue, total_assets, market_cap) son bloqueados.
- **Tests automatizados**: 55 tests pytest verifican las garantías anti-leakage, la política de features y la validación de entradas.
- **Validación de tickers**: el `DataRouter` verifica formato y previene ataques de path-traversal en todos los métodos de carga de datos.

### 5.3 Construcción del label

El label binario se calcula como: ¿superó el ticker la mediana de su sector en el forward return? El forward return se mide desde la fecha de entrada (snapshot + lag) hasta el final del holding period (por defecto 3 meses). Se requiere un mínimo de peers sectoriales; si no se alcanza, se usa la mediana del universo como fallback.

### 5.4 Selección de features

Cada agente ejecuta un `FeatureSelector` que:
1. Elimina pares exclusivos (base vs zsector): selecciona el de mayor correlación con el target.
2. Aplica eliminación greedy de correlaciones altas (> 0.85).
3. Calcula un score combinado = w × relevancia_con_y + (1-w) × importancia_RF.
4. Aplica corte por importancia: mantiene entre 4 y 10 features finales.

---

## 6. Resultados esperados y métricas de evaluación

### 6.1 Métricas del portfolio

El sistema calcula las siguientes métricas para el portfolio y cada baseline:

| Métrica | Descripción |
|---------|-------------|
| Retorno acumulado | Producto geométrico de retornos por fold |
| Retorno anualizado | Normalizado a 252 días de trading |
| Sharpe Ratio | Exceso de retorno / volatilidad (rf = 4%) |
| Sortino Ratio | Como Sharpe, pero solo penaliza desviación negativa |
| Maximum Drawdown | Mayor caída desde máximo histórico |
| Calmar Ratio | Retorno anualizado / |Max Drawdown| |
| Hit Rate | Porcentaje de folds con retorno positivo |

### 6.2 Baselines de comparación

1. **S&P 500 (benchmark)**: retorno del índice en el mismo periodo.
2. **Equal-Weight Universe**: todos los tickers elegibles con peso igual.
3. **Momentum 12M**: Top-N tickers por momentum de 12 meses.
4. **Value Combined**: Top-N por ranking combinado P/E + EV/EBITDA.
5. **Random Top-N**: N simulaciones Monte Carlo seleccionando N tickers al azar; se reporta media, mediana y percentiles.

### 6.3 Resultados esperados

Se espera que el sistema multi-agente:
- Genere un Sharpe Ratio superior al benchmark S&P 500 en periodos de 5+ años.
- Produzca un alpha positivo frente a las estrategias de factor puro (momentum, value).
- Supere consistentemente al baseline aleatorio en percentil 95.
- Muestre un Maximum Drawdown inferior al del índice gracias al filtro del BearAgent.

---

## 7. Conclusiones y trabajo futuro

### 7.1 Conclusiones

Este TFM presenta un sistema completo de stock picking basado en ML multi-agente que aborda los principales desafíos del dominio:

1. **Rigor metodológico**: la validación walk-forward con múltiples capas anti-leakage garantiza que los resultados son representativos del rendimiento fuera de muestra.
2. **Modularidad**: la arquitectura multi-agente permite añadir, modificar o desactivar agentes sin afectar al resto del sistema.
3. **Explicabilidad**: la integración de SHAP proporciona transparencia sobre las decisiones del modelo.
4. **Realismo**: la simulación monetaria con costes de transacción y slippage refleja las condiciones operativas reales.
5. **Reproducibilidad**: la semilla aleatoria fija, la configuración centralizada y los artefactos de auditoría aseguran la reprodución exacta de resultados.

### 7.2 Trabajo futuro

1. **Datos alternativos**: incorporar datos de NLP sobre noticias financieras y calls de earnings como features adicionales para el SentimentAgent.
2. **Gestión de riesgo dinámica**: implementar sizing adaptativo basado en la volatilidad del portfolio y la confianza del meta-learner.
3. **Modelos deep learning**: explorar transformers temporales (Temporal Fusion Transformer) como alternativa a los gradient boosting trees.
4. **Optimización de hiperparámetros**: integrar Optuna para búsqueda bayesiana de hiperparámetros dentro del walk-forward.
5. **Ejecución en tiempo real**: adaptar el pipeline para operación diaria con integración a brokers via API.
6. **Universo expandido**: extender el análisis a mercados internacionales (MSCI World, emergentes).
7. **Análisis de atribución**: implementar análisis de Brinson para descomponer el alpha en efecto selección vs efecto sector.

---

## 8. Referencias

- Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2014). *Pseudo-mathematics and financial charlatanism*. Notices of the AMS, 61(5), 458–471.
- Breiman, L. (2001). *Random forests*. Machine Learning, 45(1), 5–32.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. Proceedings of the 22nd ACM SIGKDD.
- Dempster, M. A. H., & Leemans, V. (2006). *An automated FX trading system using adaptive reinforcement learning*. Expert Systems with Applications, 30(3), 543–552.
- Dietterich, T. G. (2000). *Ensemble methods in machine learning*. Multiple Classifier Systems, Springer, 1–15.
- Fama, E. F. (1970). *Efficient capital markets: A review of theory and empirical work*. Journal of Finance, 25(2), 383–417.
- Fama, E. F., & French, K. R. (1992). *The cross-section of expected stock returns*. Journal of Finance, 47(2), 427–465.
- Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical asset pricing via machine learning*. The Review of Financial Studies, 33(5), 2223–2273.
- Jegadeesh, N., & Titman, S. (1993). *Returns to buying winners and selling losers*. Journal of Finance, 48(1), 65–91.
- LeBaron, B. (2006). *Agent-based computational finance*. Handbook of Computational Economics, 2, 1187–1233.
- Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. Advances in Neural Information Processing Systems, 30.
- Piotroski, J. D. (2000). *Value investing: The use of historical financial statement information to separate winners from losers*. Journal of Accounting Research, 38, 1–41.
- Wooldridge, M. (2009). *An introduction to multiagent systems*. John Wiley & Sons.

### 8.1 Software y APIs

- Finnhub Stock API: https://finnhub.io/ — datos fundamentales, insiders, analistas y SEC filings.
- Yahoo Finance (yfinance): precios OHLCV y datos de mercado.
- scikit-learn (Pedregosa et al., 2011): modelos base, pipelines y validación cruzada.
- XGBoost (Chen y Guestrin, 2016): gradient boosting para agentes fundamentales.
- SHAP (Lundberg y Lee, 2017): explicabilidad de predicciones.
- pandas, NumPy, SciPy: infraestructura de datos y cálculo numérico.
- matplotlib: visualización de resultados.
