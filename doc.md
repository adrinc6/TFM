# Documentacion del Proyecto

## 1. Objetivo

Este proyecto implementa un sistema de Inteligencia Artificial aplicada a inversion cuantitativa con enfoque GARP, es decir, empresas de calidad, con crecimiento sostenible y valoracion razonable.

La pregunta central del proyecto es:

> Puede una arquitectura de IA explicable identificar empresas GARP capaces de generar alpha frente al benchmark y gestionar una cartera dinamica de largo plazo mejor que una estrategia pasiva?

El sistema no intenta predecir precios ni hacer trading de corto plazo. Su objetivo es construir y gestionar una cartera viva, revisar periodicamente las tesis de inversion, mantener posiciones cuando la tesis sigue intacta y vender cuando la tesis se deteriora.

## 2. Filosofia de Inversion

La filosofia del sistema se basa en combinar:

- Calidad empresarial.
- Crecimiento sostenible.
- Valoracion razonable.
- Mejora fundamental futura.
- Catalizadores positivos.
- Infravaloracion por parte del mercado.

El sistema busca empresas que puedan ser mejores negocios en el futuro y cuya valoracion actual todavia no refleje completamente esa mejora.

## 3. Arquitectura General

La arquitectura esta dividida en modulos independientes:

```text
data_download
-> dataset_builder
-> features
-> ml
-> research
-> thesis
-> watchlist
-> portfolio
-> backtest
-> viewer
```

Cada modulo tiene una responsabilidad clara:

- `module/data_download`: descarga datos desde Finnhub y Yahoo Finance.
- `module/dataset_builder`: construye el dataset maestro point-in-time.
- `module/features`: genera variables financieras y scores GARP.
- `module/ml`: entrena el modelo de IA, puntua empresas y genera explicabilidad.
- `module/research`: genera investigacion automatizada sobre negocio, moat, catalyst, riesgos y oportunidades.
- `module/thesis`: calcula scores de tesis, salud, conviccion y salida.
- `module/watchlist`: mantiene empresas interesantes que todavia no estan necesariamente en cartera.
- `module/portfolio`: construye y revisa la cartera.
- `module/backtest`: simula la evolucion temporal de la cartera.
- `module/viewer`: genera el visor HTML de resultados.

La entrada principal del sistema es:

```bash
python main.py
```

## 4. Configuracion

Toda la configuracion principal vive en `environment.py`.

Variables principales:

- `RUN_MODE`: controla que fase ejecutar.
- `DATA_START_DATE`: fecha inicial de descarga raw. El dataset maestro empieza mas tarde si el walk-forward no necesita todo ese historico.
- `PORTFOLIO_START_DATE`: fecha inicial de la simulacion.
- `PORTFOLIO_END_DATE`: fecha final de la simulacion.
- `PORTFOLIO_REVIEW_FREQUENCY`: frecuencia de revision, con valores `M`, `2M` o `Q`.
- `DEV_MODE`: limita el universo a pocos tickers para desarrollo.
- `DEV_TICKERS`: universo reducido de prueba.
- `TICKERS`: universo completo.
- `BENCHMARK_TICKER`: benchmark, actualmente `SPY`.
- `FORCE_RAW_DOWNLOAD`: controla si la descarga reutiliza JSON raw existentes o fuerza redescarga.
- `OPENAI_MODEL`: modelo usado por la capa Research AI.
- `ENABLE_OPENAI_RESEARCH`: activa o desactiva llamadas reales a OpenAI.

El archivo `.env` se reserva solo para secretos:

```text
FINNHUB_API_KEY
OPENAI_API_KEY
```

Modos disponibles:

```text
download
dataset
features
ml
backtest
viewer
full
watchlist
```

En desarrollo se recomienda usar:

```text
DEV_MODE=true
```

Esto permite validar el pipeline con pocos tickers antes de escalar al universo completo.

## 5. Datos

El proyecto usa un data lake simple basado en Parquet:

```text
data/
  raw/
    json/
  processed/
  master/
```

La descarga solo se encarga de:

- Obtener datos.
- Validarlos.
- Guardarlos.
- Reutilizar JSON raw existentes si `FORCE_RAW_DOWNLOAD = False`.

No calcula features, no entrena modelos y no genera scores.

Cada respuesta raw se guarda como JSON por fuente, ticker y tipo de dato:

```text
data/raw/json/finnhub/<ticker>/profile.json
data/raw/json/finnhub/<ticker>/basic_financials.json
data/raw/json/finnhub/<ticker>/company_news_<start>_<end>.json
data/raw/json/yahoo/<ticker>/ohlcv_<start>_<end>.json
```

Los Parquet agregados de `data/raw/` se reconstruyen a partir de esos JSON cacheados.

Fuentes previstas:

- Finnhub: fundamentales, metricas financieras, perfiles empresariales, earnings y datos de compania.
- Yahoo Finance: precios, OHLCV, dividendos y splits.

## 6. Dataset Maestro Point-in-Time

El dataset maestro se guarda en:

```text
data/master/master_point_in_time.parquet
```

Cada fila representa:

```text
ticker + snapshot_date
```

La regla principal es evitar leakage:

- No usar informacion futura.
- Cada snapshot debe contener solo informacion disponible en esa fecha.
- El dataset debe ser reproducible y auditable.
- Para no hacer trabajo inutil, si el walk-forward esta activo los snapshots empiezan en `PORTFOLIO_START_DATE - MAX_WALK_FORWARD_TRAINING_YEARS`, sin bajar nunca de `DATA_START_DATE`.

## 7. Feature Engineering

El modulo de features transforma el dataset maestro en variables agrupadas por dimensiones financieras:

- `quality_score`: calidad del negocio.
- `growth_score`: crecimiento.
- `valuation_score`: valoracion razonable.
- `moat_score`: persistencia de ventajas competitivas.
- `catalyst_score`: mejora o aceleracion.
- `risk_score`: riesgo financiero.
- `garp_score`: combinacion ponderada de los factores anteriores.

El resultado se guarda en:

```text
data/processed/features.parquet
```

## 8. Inteligencia Artificial

La IA es una parte central del proyecto. El modelo principal es LightGBM.

El objetivo del modelo no es predecir que accion subira mas. El modelo aprende componentes fundamentales separados **genuinamente forward-looking** (no restatements de inputs):

- `quality_probability`: ¿mejorará la calidad del negocio en los próximos 12 meses?
- `improvement_probability`: ¿superará el crecimiento realizado las expectativas actuales?
- `mispricing_probability`: ¿se resolverá el gap de valoración mediante un retorno futuro favorable?
- `alpha_probability`: ¿generará retorno en exceso vs. benchmark en 12 meses?

Cada target se construye desde información observable ~12 meses en el futuro, con enmascarado de fuga consistente: las filas recientes cuyo futuro aún no es observable caen al fallback GARP determinista, no se entrenan con datos que aún no existían en esa fecha.

El score GARP final se calcula después como combinación ponderada de esos componentes. Esto evita que el sistema sea una caja negra de ranking y fuerza una lectura explicable de negocio.

El universo puntuado se guarda en:

```text
data/processed/scored_universe.parquet
```

La explicabilidad se guarda en:

```text
data/processed/model_explainability.json
```

Este archivo incluye:

- Importancia de variables.
- Resumen SHAP si la libreria esta disponible.
- Formula de composicion del score GARP.
- Importancia por componente.

## 8.1 Expectation Engine

La capa `module/expectations` hace explicita la idea central de la estrategia:

```text
expectativas del mercado
vs
realidad fundamental futura
```

Variables principales:

- `expected_growth`: crecimiento esperado por calidad, crecimiento actual y catalyst.
- `implied_growth`: crecimiento implicito aproximado por la valoracion.
- `realized_growth`: proxy de crecimiento realizado o persistente.
- `expectation_gap`: diferencia entre crecimiento esperado e implicito.

Un gap positivo indica una posible situacion en la que el mercado infravalora la mejora futura del negocio.

## 8.2 Metricas Relativas

El sistema refuerza comparaciones relativas:

- vs universo.
- vs sector.
- vs industria.

Estas variables ayudan a evitar conclusiones absolutas fuera de contexto. Una empresa puede parecer cara en terminos absolutos y seguir siendo razonable frente a su sector si su calidad y crecimiento son superiores.

## 8.3 Temporal Business Engine

La capa `module/business_temporal` refuerza el aprendizaje de evolucion empresarial. Todas sus variables se derivan exclusivamente de snapshots historicos disponibles para cada ticker.

Features principales:

- `quality_trend_1y`
- `quality_trend_2y`
- `roic_trend`
- `margin_trend`
- `fcf_trend`
- `growth_acceleration`
- `growth_deceleration`
- `moat_trend`

El objetivo es que el sistema aprenda a distinguir entre una empresa buena y una empresa que ademas esta mejorando.

## 9. Tesis de Inversion

Cada empresa recibe scores interpretables:

- `thesis_score`: fortaleza de la tesis.
- `position_health_score`: salud actual de la posicion.
- `conviction_score`: conviccion para comprar o mantener.
- `exit_score`: probabilidad de venta.
- `exit_thesis`: tesis de salida explicable.

Estados posibles:

- `Improving`
- `Intact`
- `Maturing`
- `Weakening`
- `Broken`

La idea principal es que la cartera no sea solo un ranking, sino un conjunto de tesis vivas que evolucionan con el tiempo.

La arquitectura actual trata la tesis como entidad central. El flujo conceptual es:

```text
datos financieros
-> IA explicable
-> investigacion automatizada
-> tesis de inversion
-> watchlist
-> cartera
```

La cartera es una consecuencia de las tesis, no al reves.

## 9.1 Research AI Layer

La capa `module/research` convierte datos estructurados en lenguaje de analista.

Componentes:

- `company_research.py`: descripcion de empresa, resumen de negocio, riesgos y oportunidades.
- `thesis_generator.py`: tesis de inversion, tesis alcista, tesis bajista y tesis base.
- `moat_analyzer.py`: ventajas competitivas, durabilidad y calidad de negocio.
- `catalyst_detector.py`: catalizadores, rerating y triggers de entrada.
- `news_analyzer.py`: resumen de noticias cuando exista informacion estructurada disponible.

La IA deja de producir solamente un score y pasa a producir una interpretacion del negocio.

La capa OpenAI Research AI, cuando `ENABLE_OPENAI_RESEARCH=True`, genera JSON por empresa con:

- resumen del negocio;
- moat;
- catalysts;
- risks;
- 3 noticias recientes si existen en el data lake;
- thesis;
- exit thesis;
- classification.

Si no hay `OPENAI_API_KEY` o la capa esta desactivada, el sistema genera un fallback determinista para mantener el pipeline reproducible.

## 9.2 Clasificacion de Oportunidades

Las empresas se clasifican en categorias interpretables:

- `Growth Undervalued`
- `Quality Growth Reasonable`
- `Compounder`
- `Value with Catalyst`
- `Turnaround`
- `Cyclical Opportunity`
- `Deep Value`
- `Fully Valued Compounder`
- `Expensive Growth`
- `Value Trap`
- `Avoid`

## 10. Construccion y Gestion de Cartera

La cartera es concentrada:

- Minimo 5 posiciones.
- Maximo 10 posiciones.

La simulacion sigue este flujo:

```text
fecha inicial
-> cartera inicial
-> revision periodica
-> mantener tesis intactas
-> vender tesis deterioradas
-> incorporar nuevas oportunidades
-> comparar contra benchmark
```

No se usa un holding period fijo. Las posiciones se mantienen mientras la tesis siga siendo defendible.

El ranking de entrada no es un simple ranking de acciones. La cartera prioriza:

- Calidad de negocio.
- Moat.
- Crecimiento sostenible.
- Salud de la tesis.
- Valoracion y catalizadores como filtros complementarios.

Cada posicion almacena memoria:

- Fecha de entrada.
- Snapshot original.
- Tesis original.
- Scores originales.
- Motivo de compra.
- Meses en cartera.
- Persistencia de tesis.
- Deterioros y mejoras acumuladas.
- Tesis de salida.
- Pesos de asignacion.

## 10.1 Position Sizing Intelligence

La asignación de capital se calcula de forma transparente:

- `equal_weight`: mismo peso para cada posición (1 / número de posiciones)
- `conviction_weight`: peso proporcional a la convicción (normalizado)
- `risk_adjusted_weight`: peso inversamente proporcional al riesgo estimado
- `hybrid_weight`: `0.35 * equal_weight + 0.65 * risk_adjusted_weight` → esto **realmente se usa** en el cálculo de retornos simulados del backtest

El `hybrid_weight` se integra en `module/backtest/performance.py::portfolio_vs_benchmark()` para que la P&L simulada refleje el sizing real, no un simple equal-weight. El coste de transacción también se pondera por notional rotado (weight traded × cost_rate), no por conteo de operaciones.

El objetivo es que el viewer pueda explicar no solo qué se compra y por qué, sino también cuánto capital recibe cada tesis y cómo eso afecta el retorno final.

## 10.2 Buy Today Engine

El sistema distingue entre comprar, mantener y vender.

Mantener una posicion no implica automaticamente que hoy sea una compra nueva. Para eso existe el Buy Today Engine, con:

- `would_buy_today`
- `buy_today_score`
- `best_alternative_ticker`
- `opportunity_cost_score`

Esto permite separar capital nuevo de tesis ya existentes y medir explicitamente el coste de oportunidad.

## 11. Backtesting

El backtest principal simula una cartera viva desde la fecha inicial hasta la fecha final.

Outputs principales:

```text
results/<run>/portfolio_evolution.csv
results/<run>/portfolio_transactions.csv
results/<run>/portfolio_monthly_holdings.csv
results/<run>/portfolio_turnover.csv
results/<run>/portfolio_decision_log.csv
results/<run>/portfolio_vs_benchmark.csv
results/<run>/rebalance_report.csv
results/<run>/watchlist.csv
results/<run>/portfolio_allocation.csv
results/<run>/research_ai.csv
results/<run>/portfolio_monthly_summary.json
results/<run>/final_report.html
```

El sistema compara la cartera contra `SPY` y genera informacion sobre:

- Evolucion de posiciones.
- Compras.
- Ventas.
- Razones de decision.
- Turnover.
- Alpha frente al benchmark.
- Informes de rebalance con ADD, SELL y HOLD.
- Tesis de salida.
- Asignacion de capital.

## 11.2 Final Report Engine

La capa `module/report` genera automáticamente:

```text
results/<run>/final_report.html
```

Incluye:
- Resumen ejecutivo en español
- CAGR, Sharpe, Sortino, Max Drawdown, Alpha neto/bruto
- **Information Ratio, Tracking Error, t-stat de retorno en exceso** (nuevos, con aviso de muestra pequeña)
- Tabla de episodios de drawdown (profundidad, duración, recuperación)
- Comparación contra benchmark
- Mejores y peores decisiones, mejores y peores tesis
- Clasificación de oportunidades
- Importancia de variables e importancia de componentes del modelo
- **Diagnóstico walk-forward** (ventana de entrenamiento, tasa de fallback)
- Conclusiones automáticas con limitaciones metodológicas explícitas (sesgo de supervivencia, muestra pequeña)

Este informe está pensado como base directa para la defensa del TFM, incluyendo toda la rigor metodológico y las limitaciones documentadas.

## 11.1 Watchlist Engine

La watchlist separa empresas interesantes de empresas efectivamente compradas.

Una empresa puede evolucionar asi:

```text
Watchlist
-> Cartera
-> Salida
-> Watchlist
```

La watchlist almacena:

- Tesis.
- Catalyst.
- Moat.
- Valoracion.
- Conviccion.
- Trigger de entrada.

Outputs:

```text
data/processed/watchlist.parquet
results/<run>/watchlist.csv
results/<run>/viewer/watchlist.html
```

## 12. Viewer HTML

El visor se genera en:

```text
results/<run>/viewer/
```

Estructura en dos grupos de navegación (Principal / Secundaria/Auditoría):

**Principal**:
- `dashboard.html` — KPIs, gráficos, tablas filtradas interactivas, cartera actual, oportunidades, operaciones
- `current_portfolio.html` — cartera actual con pesos, scores, tesis
- `portfolio_vs_benchmark.html` — comparativa completa (P&L, gráficos, **tabla de episodios de drawdown**)
- `position_performance.html` — rendimiento por acción vs. benchmark durante cada holding
- `top_opportunities.html` — mejores candidatos del universo

**Secundaria/Auditoría**:
- `index.html` — vista estática clásica
- `tracking_dashboard.html` — seguimiento mensual compacto
- `action_journal.html` — diario unificado de compras/ventas con resultado económico
- `buy_rationale.html` — justificación de cada compra
- `sell_reasons.html` — resumen de motivos de salida
- `sector_exposure.html` — exposición sectorial en el tiempo
- `allocation_dashboard.html` — dimensionamiento de posiciones e historia de pesos
- `watchlist.html` — mapa de oportunidades (valoración vs. convicción)
- `strategy_learning.html` — pistas automatizadas de mejora + registro de evidencia
- `model_explainability.html` — importancia de variables, SHAP, **diagnóstico walk-forward**
- `audit.html` — índice de archivos pesados de auditoría
- `position_<ticker>.html` — página individual por cada posición (gráfico de convicción/peso + tabla histórica)

**Características**:
- Todo en **español** (títulos, etiquetas, prosa)
- Formato numérico unificado (porcentajes, múltiplos, floats)
- Aislamiento de errores: una página rota no tumba todo el visor
- Sin crashes por NaN/Inf (saneador recursivo aplicado antes de JSON)
- Leyendas de gráficos agrupadas en "Otros" para evitar desbordamiento

El objetivo del viewer es que el usuario pueda entender:

- Qué empresas hay en cartera y por qué (tesis, scores, catalyst)
- Cuánto capital recibe cada una (hybrid_weight)
- Por qué entraron y por qué siguen dentro (persistencia de tesis)
- Por qué salieron (tesis_exit, motivo de venta)
- Cómo evoluciona la cartera y se comporta vs. benchmark
- Qué episodios de drawdown ocurrieron y cuánto duraron
- Qué aprendemos de entradas/salidas exitosas vs. fallidas
- Cómo se entrenó el modelo walk-forward y qué tasa de fallback tuvo

## 13. Buenas Practicas Implementadas

El proyecto sigue estas reglas:

- Configuracion centralizada.
- Pipeline modular.
- Separacion estricta entre descarga, features, modelo, cartera y viewer.
- Datos en Parquet.
- Modo desarrollo con pocos tickers.
- Fail fast cuando faltan datos criticos.
- Logs claros.
- Explicabilidad del modelo.
- Outputs auditables.

## 14. Estado Actual

El sistema ha completado una auditoría integral en tres frentes:

### Metodología ML/Backtest
✅ Bug `industry` eliminado (features `_vs_industry` no existían)
✅ Targets ML rediseñados: `quality`, `improvement`, `mispricing` son ahora predicciones genuinas de información futura observable (no restatements de inputs)
✅ Enmascarado de fuga consistente: todos los 4 targets comparten el mismo mecanismo de observabilidad a 12 meses
✅ Position sizing `hybrid_weight` integrado en el P&L real del backtest
✅ Coste de transacción ponderado por notional rotado, no por conteo de operaciones
✅ Métricas estadísticas: information ratio, tracking error, t-stat con aviso de muestra pequeña

### Robustez de Ingesta
✅ Reintentos 429 limitados a 5 intentos (antes: ilimitados)
✅ Error handling por ticker: una descarga fallida no aborta todo el run
✅ Informe de cobertura: `download_coverage.json` + `download_failures.csv` con breakdown por dataset
✅ Tiempo de ejecución registrado

### Presentación (Visor/Informe)
✅ Bugs de robustez corregidos: sanitizador de NaN/Inf, aislamiento de errores por página
✅ Formato numérico unificado: porcentajes, múltiplos, floats con precisión consistente
✅ **Todo en español**: títulos, etiquetas, prosa del informe
✅ Jerarquía de navegación: Principal (5 páginas) / Secundaria (12 páginas)
✅ Análisis curado: diagnóstico walk-forward + tabla de episodios de drawdown visibles
✅ Limitaciones documentadas: sesgo de supervivencia + muestra pequeña explícitos en conclusiones

### Importante

Los datos actualmente presentes en `data/` y `results/` proceden de un smoke test de integración con datos sintéticos. Sirven para validar que el sistema funciona técnicamente, pero **no representan resultados financieros reales**.

Para ejecutar con datos reales:
1. Configura `FINNHUB_API_KEY` en `.env`
2. Ajusta fechas en `environment.py` si lo deseas
3. Ejecuta `python main.py` con `RUN_MODE = "full"`

En desarrollo rápido, usa `DEV_MODE = True` para limitar a pocos tickers.

## 15. Limitaciones Metodológicas (Explícitas)

### Sesgo de supervivencia
`TICKERS` es un listado estático de empresas del S&P 500 actual. No incluye:
- Nombres delisted, adquiridos o caídos del índice desde 2000
- Tampoco excluye IPOs/spin-offs recientes que no existían históricamente

Esto constituye sesgo de supervivencia que el entrenamiento ML hereda automáticamente. La cartera viva solo opera entre `PORTFOLIO_START_DATE` y `PORTFOLIO_END_DATE`, lo que acota pero no elimina el problema.

**Tratamiento**: Documentado explícitamente en:
- `environment.py` (comentario junto a `TICKERS`)
- `CLAUDE.md` (subsección "Methodological limitations")
- `module/report.py::_conclusions()` (texto de limitaciones)

### Tamaño de muestra
~40 observaciones mensuales típicas. **No se aplican**:
- Bootstrap o intervalos de confianza
- Corrección Sharpe deflactado
- Ajustes por comparaciones múltiples

El t-stat del retorno en exceso es **directional**, no prueba de significancia estadística. Ver aviso explícito en `final_report.html` y en `SMALL_SAMPLE_CAVEAT`.

## 16. Propósito Académico

El proyecto está diseñado para ser defendible como Trabajo Fin de Máster porque combina:

- **Ingeniería de datos**: pipeline modular, point-in-time sin leakage
- **ML reproducible**: walk-forward con enmascarado auditable, fallback determinista, diagnóstico visible
- **IA explicable**: componentes fundamentales interpretables, no caja negra
- **Gestión dinámica**: cartera concentrada con tesis vivas, distinción buy-today vs. hold
- **Backtesting riguroso**: P&L ponderado por sizing real, coste de transacción por notional, IR/TE/t-stat
- **Transparencia**: todas las limitaciones documentadas, errores aislados, visor en español navegable

El objetivo final es responder de forma empírica y defendible:

> Si se hubiera iniciado una cartera GARP/Value-Growth con IA explicable en una fecha determinada, ¿habría superado al benchmark SPY durante el período de prueba? ¿Bajo qué supuestos metodológicos y limitaciones?
