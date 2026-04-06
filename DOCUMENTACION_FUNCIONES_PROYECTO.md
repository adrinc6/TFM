# GUÍA OPERACIONAL DEL PROYECTO

## analyzer.py

**Para qué sirve:** Punto de entrada único que ejecuta el análisis completo de inversión desde la descarga de datos hasta la generación de reportes de desempeño del modelo.

**Sin estas funciones no:** No se ejecutaría nada.

### Funciones principales:

**`_quarter_end_date(year, quarter)`**
- Para qué: Convierte año y trimestre (1-4) a fecha exacta de fin de trimestre.
- Entradas: Año (número) y trimestre (1 a 4).
- Salida: Fecha del último día del trimestre.
- Pasos: 1. Calcula la fecha / 2. Ajusta al último día del mes del trimestre.

**`_set_global_seeds(seed)`**
- Para qué: Asegurar que cada ejecución sea idéntica (para probar cambios de manera reproducible).
- Entradas: Número de semilla (cualquier número).
- Salida: Configuración interna (sin valor visible).
- Pasos: 1. Aplica semilla a generador de números / 2. Bloquea valores aleatorios para que sean siempre iguales.

**`_safe_git_commit_hash()`**
- Para qué: Registra qué versión del código generó cada análisis.
- Entradas: Ninguna.
- Salida: ID único del código (40 caracteres) o nada si no hay registro.
- Pasos: 1. Lee el registro del código / 2. Extrae el ID más reciente / 3. Lo devuelve o notifica si no hay.

**`_export_run_config(...)`**
- Para qué: Guarda un archivo con TODOS los parámetros usados en el análisis (es el "acta de decisión").
- Entradas: Directorio de resultados, lista de tickers, fechas, parámetros del modelo.
- Salida: Archivo `run_config.json` en la carpeta de resultados.
- Pasos: 1. Reúne todos los parámetros / 2. Añade timestamp y versión del código / 3. Guarda en formato legible.

**`_export_data_quality_report(...)`**
- Para qué: Identifica rápidamente qué tickers tienen problemas o datos incompletos.
- Entradas: Datos maestros, lista de tickers.
- Salida: Archivo CSV mostrando cobertura de datos por ticker.
- Pasos: 1. Revisa cada ticker / 2. Cuenta qué datos se tienen / 3. Marca si faltan precios o fundamentales / 4. Reporta porcentaje de cobertura.

**`main()`**
- Para qué: Ejecutar el flujo completo. Lo que ves cuando ejecutas `python analyzer.py`.
- Entradas: Variables de configuración global (de `environment.py`).
- Salida: Carpeta `results/` con modelos, predicciones y benchmarks.
- Pasos: 
  1. Preparar (semillas, logs)
  2. Descargar datos nuevos si es necesario
  3. Consolidar datos (limpiar, mezclar fuentes)
  4. Armar banco de observaciones (features + retornos reales)
  5. Si no está saltado: entrenar y hacer backtest
  6. Guardar reportes finales

---

## analyzer_ticker.py

**Para qué sirve:** Herramienta de análisis específico después del hecho: permite ver por qué el modelo recomendó (o no) un ticker en un trimestre específico.

**Ejemplo de uso:** "¿Por qué el modelo dijo comprar AAPL en 2023Q1?" → responde con explicación detallada.

### Funciones principales:

**`_qnorm(q)`**
- Para qué: Convertir entrada del usuario a formato estándar (ej: "2023 Q1" → "2023Q1").
- Entradas: Trimestre en cualquier formato (usuario tipea).
- Salida: Trimestre validado en formato "YYYYQ#".
- Pasos: 1. Limpia espacios / 2. Estandariza mayúsculas / 3. Valida que sea válido.

**`_safe_read_csv(path)`**
- Para qué: Leer un archivo si existe; si no, devolver vacío sin error.
- Entradas: Ruta del archivo.
- Salida: Tabla de datos o tabla vacía.
- Pasos: 1. Busca archivo / 2. Lo lee si existe / 3. Retorna vacío si falla.

**`_load_artifacts(results_dir, quarter)`**
- Para qué: Traer todos los datos guardados de un trimestre específico en un diccionario único.
- Entradas: Carpeta de resultados, trimestre.
- Salida: Diccionario con scores, predicciones y explicaciones del período.
- Pasos: 1. Construye rutas / 2. Lee cada archivo / 3. Agrupa en diccionario.

**`analyze_ticker_quarter(ticker, quarter, results_dir)`**
- Para qué: Consulta la base de datos guardada y devuelve respuesta legible: "¿qué se recomendó y por qué?"
- Entradas: Ticker (ej "AAPL"), trimestre (ej "2023Q1"), carpeta de resultados.
- Salida: Diccionario con decisión (comprar/esperar/no), scores de cada módulo de análisis y factores.
- Pasos:
  1. Carga datos del trimestre
  2. Busca el ticker
  3. Extrae decisión y puntuación
  4. Recopila qué "módulo" votó a favor/en contra
  5. Devuelve todo estructurado

**`_print_human(report)`**
- Para qué: Imprimir la respuesta en formato legible para el terminal.
- Entradas: Diccionario de análisis.
- Salida: Texto impreso en pantalla.
- Pasos: 1. Estructura secciones / 2. Formatea tablas / 3. Imprime decisión y breakdown.

---

## environment.py

**Para qué sirve:** Archivo de configuración centralizado. Cambias valores aquí para alterar el comportamiento del modelo SIN tocar código.

**Sin funciones/métodos** — Solo constantes:
- `TICKERS`: Qué acciones analizar (ej: 100 empresas del S&P 500).
- `FINNHUB_API_KEY`: Contraseña para descargar datos de internet.
- `ANALYSIS_FREQUENCY`: Si analizar cada trimestre o cada año.
- `PORTFOLIO_MIN_SCORE`: Puntuación mínima para recomendar compra (ej: 0.55 de 1.0).
- `WALKFORWARD_TRAIN_LOOKBACK_YEARS`: Cuántos años de datos históricos usar para entrenar.
- Hiperparámetros de cada módulo de análisis (número de árboles, profundidad, etc.).

**Instrucción:** Si necesitas cambiar cómo se comporta el análisis, edita este archivo primero.

---

## module/__init__.py

**Para qué sirve:** Marca la carpeta `module/` como un paquete reutilizable.

**Sin funciones/métodos** — Solo documentación del paquete.

---

## module/common/asof.py

**Para qué sirve:** Auditoría temporal: verifica que no uses datos que "no estarían disponibles" en la fecha de análisis (evita trampa de predicción).

**Analogía:** Asegurarse de que no "ves el futuro" antes de predecir.

### Funciones principales:

**`_resolve_dates(df, date_col)`**
- Para qué: Encontrar la columna de fechas (puede estar en el índice o en columna específica).
- Entradas: Tabla de datos, nombre de columna o nada.
- Salida: Serie de fechas extraída.
- Pasos: 1. Busca columna si se indicó / 2. Sino, revisa el índice / 3. Extrae fechas.

**`filter_asof(df, as_of, date_col)`**
- Para qué: Filtrar tabla para que solo tenga datos hasta cierta fecha (incluida).
- Entradas: Tabla, fecha límite, columna de fechas.
- Salida: Tabla filtrada (solo fechas ≤ límite).
- Pasos: 1. Compara cada fila con la fecha límite / 2. Mantiene solo las válidas / 3. Devuelve tabla.

**`detect_future_rows(df, as_of, date_col)`**
- Para qué: Encontrar (y contar) datos que violaron la regla: qué tiene fechas MÁS RECIENTES de lo permitido.
- Entradas: Tabla, fecha límite, columna de fechas.
- Salida: Número de violaciones encontradas y fecha máxima violadora.
- Pasos: 1. Revisa cada fila / 2. Marca las que tienen fecha > límite / 3. Cuenta y reporta máximo.

**`assert_no_future_data(df, as_of, context, date_col)`**
- Para qué: Verificación formal con registro: ¿hay datos futuros? Guardará el resultado en un archivo.
- Entradas: Tabla, fecha límite, descripción del contexto, columna de fechas.
- Salida: Diccionario con resultado (ok/no ok) y detalles.
- Pasos: 1. Ejecuta detección / 2. Arma resultado estructurado / 3. Lo retorna para guardar.

---

## module/common/data_router.py

**Para qué sirve:** Central de distribución de datos: una consulta única para traer precios, fundamentales y análisis de sentimiento de un ticker sin duplicaciones.

**Analogía:** Como un router Wi-Fi que centraliza conexiones.

### Clase `DataRouter`:

**`__init__(data_dir)`**
- Para qué: Inicializar el router en una carpeta específica.
- Entradas: Ruta a la carpeta `data_finnhub/`.
- Salida: Objeto router listo para usar.
- Pasos: 1. Guarda ruta / 2. Prepara caché de empresas.

**`load_companies(tickers)`**
- Para qué: Traer información de sector/industria para una lista de tickers.
- Entradas: Lista de tickers o nada (auto-descubre).
- Salida: Tabla con ticker, sector, industria, capitalización.
- Pasos: 1. Si ya lo cargó una vez, usa datos guardados / 2. Si no, lee perfiles / 3. Arma tabla / 4. Cachea para próxima vez.

**`get_sector_map(tickers)`**
- Para qué: Mapeo rápido {ticker → sector}; útil para normalizaciones sectoriales.
- Entradas: Lista de tickers.
- Salida: Diccionario ticker-sector.
- Pasos: 1. Carga companies / 2. Extrae columna sector / 3. Convierte a diccionario.

**`load_prices(ticker)`**
- Para qué: Traer serie histórica de precios (abierto, máximo, mínimo, cierre, volumen) diarios.
- Entradas: Ticker (ej "AAPL").
- Salida: Tabla de precios históricos o nada si no existe.
- Pasos: 1. Busca archivo / 2. Lo lee / 3. Indexa por fecha / 4. Limpia duplicados.

**`load_consolidated(ticker)`**
- Para qué: Traer fundamentales consolidados (ratios financieros) por trimestre/año.
- Entradas: Ticker.
- Salida: Tabla de fundamentales o nada.
- Pasos: 1. Busca el consolidado del ticker / 2. Lo lee / 3. Limpia duplicados.

**`load_eps_surprises(ticker)`, `load_recommendation_trends(ticker)`, `load_insider_transactions(ticker)`, `load_insider_sentiment(ticker)`**
- Para qué: Traer datos especializados (sorpresas de ganancias, recomendaciones de analistas, compra-venta de insiders).
- Entradas: Ticker.
- Salida: Tabla del dato específico o nada.
- Pasos: 1. Busca archivo JSON / 2. Interpreta contenido / 3. Devuelve tabla.

**`load_sp500_prices()`**
- Para qué: Traer precios del índice S&P 500 (para comparación).
- Entradas: Ninguna.
- Salida: Serie histórica del S&P 500.
- Pasos: 1. Busca el archivo macro / 2. Lo lee / 3. Devuelve serie.

**`get_fundamental_snapshot(consolidated, as_of)`**
- Para qué: Traer la fundamental MÁS RECIENTE publicada ANTES de cierta fecha (sin "ver el futuro").
- Entradas: Tabla de fundamentales, fecha límite.
- Salida: Fila única más reciente o nada.
- Pasos: 1. Filtra hasta la fecha / 2. Toma último registro / 3. Lo devuelve.

**`get_price_window(prices, as_of, lookback_days)`**
- Para qué: Traer precios de un rango de días anteriores (ej: últimos 400 días).
- Entradas: Tabla de precios, fecha final, días hacia atrás.
- Salida: Sub-tabla del rango solicitado.
- Pasos: 1. Calcula fecha inicial (final - días) / 2. Filtra en ese rango / 3. Devuelve.

**`compute_quarterly_forward_return(prices, as_of, lag_days, holding_period_months)`**
- Para qué: Calcular cuánto "realmente subió el precio" después de nuestra decisión (esto es el "resultado real" que el modelo intenta predecir).
- Entradas: Tabla de precios, fecha de decisión, días de espera antes de entrar, meses de tenencia.
- Salida: Retorno porcentual (ej: 0.08 = +8%).
- Pasos: 1. Calcula fecha de entrada (decisión + espera) / 2. Calcula fecha de salida (entrada + meses) / 3. Lee precio en entrada y salida / 4. Calcula % cambio.

---

## module/steps/step_01_data/__init__.py

**Para qué sirve:** Marca la carpeta de Step 1 como paquete.

**Sin funciones/métodos** — Solo documentación.

---

## module/steps/step_01_data/registry.py

**Para qué sirve:** Evitar descargas repetidas: registro de "qué ya bajamos".

**Analogía:** Como una lista de compras tachada: "ya compré AAPL, no lo descargo nuevamente".

### Clase `Registry`:

**`__init__(base_dir)`**
- Para qué: Inicializar registro en carpeta específica.
- Entradas: Ruta base.
- Salida: Objeto registro.

**`is_done(group, endpoint)`**
- Para qué: Preguntar "¿ya descargué esto?"
- Entradas: Grupo (ej "prices"), endpoint (ej "AAPL").
- Salida: Sí/No.
- Pasos: 1. Busca entrada / 2. Responde.

**`mark_done(group, endpoint)`**
- Para qué: Marcar item como ya descargado.
- Entradas: Grupo, endpoint.
- Salida: Registro actualizado guardado en disco.
- Pasos: 1. Añade línea de marca / 2. Guarda.

**`clear(group, delete_file)`**
- Para qué: Limpiar registro (para re-descargar si hay datos corrompidos).
- Entradas: Grupo a limpiar, si también borrar el archivo.
- Salida: Registro limpio.
- Pasos: 1. Borra entradas / 2. Si requested, borra archivo disco.

---

## module/steps/step_01_data/pipeline.py

**Para qué sirve:** Orquestador del Paso 1: controla descarga, consolidación y filtrado de datos.

**Flujo:** Entrada = Lista de tickers | Salida = Carpeta `data_finnhub/` con datos consolidados.

### Funciones principales:

**`download_data(tickers, start_date, end_date, data_dir, ...)`**
- Para qué: Descargar datos nuevos de internet (precios, fundamentales, análisis).
- Entradas: Tickers, fechas de rango, carpeta destino.
- Salida: Archivos descargados en `data_finnhub/`.
- Pasos: 1. Crea conexiones / 2. Por cada ticker: descarga precios y datos fundamentales / 3. Guarda en JSON / 4. Marca en registro.

**`prepare_data(tickers, data_dir)`**
- Para qué: Consolidar: mezclar quarterly + annual, llenar huecos, calcular ratios.
- Entradas: Lista de tickers, carpeta.
- Salida: Archivos `consolidated/{ticker}.csv` con fundamentales limpios.
- Pasos: 1. Lee filings SEC / 2. Combines quarterly y annual / 3. Calcula ratios derivados / 4. Guarda consolidado.

**`get_available_tickers(tickers, data_dir)`**
- Para qué: Filtrar tickers que tienen datos suficientes (precios + fundamentales completos).
- Entradas: Lista inicial, carpeta.
- Salida: Lista de tickers OK, reporte de qué faltó en los descartados.
- Pasos: 1. Revisa cada ticker / 2. Verifica precio y consolidado / 3. Retorna OS y problemas.

**`retry_missing_tickers(missing_detail, start_date, end_date, data_dir, api_key)`**
- Para qué: Re-intentar descarga para tickers que fallaron la primera vez.
- Entradas: Reporte de problemas, fechas, carpeta, credencial.
- Salida: Lista de tickers recuperados.
- Pasos: 1. Identifica tickers con fallas / 2. Los descarga nuevamente / 3. Reporta cuáles se arreglaron.

---

## module/steps/step_01_data/parsers.py

**Para qué sirve:** Convertir payloads JSON crudos de la API a tablas estructuradas.

**Analogía:** Traducir/limpiar documentos desordenados en bases de datos organizadas.

### Clases de parsers:

**`FinnhubSECParser`**
- Para qué: Interpretar reportes SEC (10-K, 10-Q) → ratios financieros.
- Métodos clave:
  - `parse_filing()`: Lee un reporte → extrae números financieros (revenue, ganancia, deuda, etc.).
  - `parse_financials_json()`: Lee archivo con múltiples reportes → retorna tabla limpia.

**`BasicFinancialsParser`**
- Para qué: Interpretar ratios pre-calculados (P/E, P/B, ROE, etc.).

**`EPSSurprisesParser`**
- Para qué: Interpretar sorpresas de ganancias (¿ganancia real vs esperada? ¿cuánto varió?).

**`RecommendationParser`**
- Para qué: Interpretar consenso de analistas (% que dice comprar, vender, esperar).

**`InsiderTransactionsParser`**, **`InsiderSentimentParser`**
- Para qué: Interpretar compra-venta de insiders y sentimiento de compra de insiders.

---

## module/steps/step_01_data/downloaders.py

**Para qué sirve:** Descarga eficiente y en paralelo: múltiples tickers simultáneamente, respetando límites de API.

**Velocidad:** Descarga ~100 tickers en ~10-15 minutos (8 descargas paralelas).

### Funciones principales:

**`download_prices(ticker, ticker_dir, start, end, registry, force, yahoo, ...)`**
- Para qué: Descargar precios diarios de un ticker.
- Proceso: Consulta Yahoo Finance → guarda JSON.

**`download_macro(base_dir, registry, start, end, force, yahoo)`**
- Para qué: Descargar benchmark (S&P 500).

**`download_ticker(ticker, client, yahoo, base_dir, registry, ...)`**
- Para qué: Descargar un ticker completo (perfil, fundamentales, recomendaciones, insider, noticias).

**`run_download(api_key, tickers, start, end, base_dir, force, prices_only, max_workers, min_interval)`**
- Para qué: Orquestar descarga de todos los tickers en paralelo.
- Pasos: 1. Crea workers / 2. Distribuye tickers / 3. Espera completación / 4. Reporta resumen.

---

## module/steps/step_01_data/consolidation.py

**Para qué sirve:** Unificar datos de múltiples fuentes: SEC filings (quarterly + annual) + ratios calculados + llenar huecos.

**Resultado final:** Cada ticker tiene su archivo `consolidated/{ticker}.csv` con fundamentales limpios por trimestre.

### Funciones principales:

**`FinnhubConsolidator.consolidate_ticker(ticker)`**
- Para qué: Consolidar UN ticker: mezclar quarter + annual, llenar Q4 faltante, calcular ratios.
- Pasos:
  1. Lee SEC filings (quarterly y annual)
  2. Si falta Q4: lo calcula restando Q1+Q2+Q3 del annual
  3. Mezcla con ratios pre-calculados
  4. Calcula ~50 ratios derivados (ROE, margen, solvencia, etc.)
  5. Guarda en `consolidated/{ticker}.csv`

**`build_companies_df(finnhub_data_dir, tickers)`**
- Para qué: Construir tabla de sector/industria/capitalización de mercado.
- Entrada: Carpeta de datos, lista de tickers.
- Salida: Tabla con sector/industria en cada fila.

---

## module/steps/step_01_data/clients.py

**Para qué sirve:** Conexión a APIs externas (Finnhub, Yahoo) con manejo de errores y límites de velocidad.

**Protección:** Respeta límites de API (Finnhub: ~1 request/segundo para no ser bloqueado).

### Clases:

**`FinnhubClient`**
- Para qué: Hablar con API Finnhub (descarga fundamentales, reportes, analistas, insiders).
- Métodos: company_profile2(), basic_financials(), financials_as_reported(), eps_surprises(), recommendation_trends(), insider_transactions(), insider_sentiment(), company_news(), peers(), quote().

**`YahooClient`**
- Para qué: Hablar con Yahoo Finance (descarga precios diarios).
- Método: ohlcv(ticker, start, end) → retorna precios.

---

## module/steps/step_02_dataset/__init__.py

**Para qué sirve:** Marca la carpeta de Step 2 como paquete.

**Sin funciones/métodos** — Solo documentación.

---

## module/steps/step_02_dataset/dataset.py

**Para qué sirve:** Construir el "banco de observaciones" de entrenamiento: cada fila = 1 ticker en 1 trimestre con features e "resultado real".

**Entrada:** Datos consolidados, precios, análisis de sentimiento.
**Salida:** Tabla maestro con ~100 columnas (features) + etiqueta (retorno real).

### Funciones principales:

**`_build_filing_date_map_for_ticker(data_dir, ticker)`**
- Para qué: Mapeo de fechas: cuándo se PUBLICÓ cada reporte (para evitar "ver el futuro").
- Salida: Diccionario {fecha_reporte → fecha_publicación}.

**`_fund_snapshot_as_of_filing(fund_enriched, filing_date_map, snapshot_date)`**
- Para qué: Traer fundamental más reciente PUBLICADO antes de la fecha de análisis.
- Protección: Evita usar datos aún no disponibles.

**`_load_ticker_sources(router, ticker)`**
- Para qué: Traer datos de UN ticker agrupados (precios, fundamentales, análisis, insiders).
- Salida: Diccionario único con todos los datos del ticker.

**`_build_feature_record(ticker, as_of, sources, fund_enriched, router, builders, ...)`**
- Para qué: Construir UNA observación (fila) del dataset para UN ticker en UN trimestre.
- Pasos:
  1. Calcula fecha de análisis (trimestre + 45 días de espera)
  2. Lee fundamental más reciente (sin look-ahead)
  3. Calcula señales de mercado (RSI, MACD, promedios móviles)
  4. Calcula features fundamentales (ratios de rentabilidad, solvencia)
  5. Calcula features de valoración (P/E, P/B vs. histórico)
  6. Calcula features de insiders y sentimiento
  7. Calcula label: "cuánto subió el precio después (retorno real)"
  8. Devuelve observación completa

**`build_master_dataset(tickers, router, builders, ...)`**
- Para qué: Construir dataset MAESTRO: todos los tickers, todos los trimestres disponibles.
- Pasos: 1. Por cada ticker / 2. Por cada trimestre del histórico / 3. _build_feature_record / 4. Apila todas las filas.

**`build_live_features(tickers, as_of, router, builders, ...)`**
- Para qué: Construir features HOY (sin etiqueta de "resultado real" porque aún no pasó el tiempo).
- Uso: Para predicciones en tiempo real.

---

## module/steps/step_02_dataset/normalization.py

**Para qué sirve:** Normalizar features por sector: comparar empresas dentro de su industria, no globalmente.

**Analogía:** En lugar de comparar precio de banco con startup (incomparable), compara cada uno con su sector.

### Funciones:

**`apply_sector_normalization(df, sector_map, normalizer, fit)`**
- Para qué: Añadir columnas normalizadas: posición de cada empresa dentro del sector.
- Proceso: 1. Di a medias y varianzas sectoriales / 2. Por cada empresa: convierte su ratio al "percentil dentro del sector" / 3. Guarda en columnas nuevas.

---

## module/steps/step_02_dataset/builders/fundamental.py

**Para qué sirve:** Construir features de "salud financiera" de la empresa: rentabilidad, solvencia, calidad de ganancias.

**Resultado:** ~20 nuevas variables que capturan "¿qué tan bien está esta empresa?"

### Clase `FundamentalFeatureBuilder`:

**`build(df)`**
- Para qué: Enriquecer tabla de fundamentales con ratios, tendencias y señales de calidad.

**`snapshot_trends(fund_hist_asof)`**
- Para qué: Calcular TENDENCIAS de 2-3 años de rentabilidad y márgenes (sin datos futuros).
- Uso: "¿ROE está subiendo o bajando?"

---

## module/steps/step_02_dataset/builders/technical.py

**Para qué sirve:** Construir features de precios: osciladores, momentum, volatilidad, tendencias.

**Resultado:** ~20 variables del lectura de precio y movimiento (RSI, MACD, promedios móviles, bandas).

### Clase `TechnicalFeatureBuilder`:

**`build(prices_df, as_of, lookback_days)`**
- Para qué: Calcular indicadores de mercado sobre precios hasta la fecha (sin futuro).
- Osciladores: RSI (sobreventa/sobrecompra), MACD (cambios de tendencia).
- Tendencias: Promedios móviles a 20/50/200 días.
- Bandas: Volatilidad envolvedora.
- Momentum: Retorno en ventanas 1/3/6/12 meses.
- Volatilidad: Variabilidad histórica del precio.

---

## module/steps/step_02_dataset/builders/valuation.py

**Para qué sirve:** Construir features de valoración: múltiplos vs. histórico y sector.

**Resultado:** ~12 variables que detectan si un precio está caro o barato.

### Clase `ValuationFeatureBuilder`:

**`build(prices_df, fund_snapshot, hist_fund, as_of)`**
- Para qué: Calcular múltiplos (P/E, P/B, P/S, FCF yield) y compararlos vs. 5 años histórico.
- Uso: "¿El P/E actual está arriba o abajo del promedio de 5 años?" (mean-reversion).

---

## module/steps/step_02_dataset/builders/insider.py

**Para qué sirve:** Construir features de confianza insider: compra-venta neta de ejecutivos.

**Lógica:** Si ejecutivos compran sus propias acciones → bullish; si venden → warning.

### Clase `InsiderFeatureBuilder`:

**`build(insider_df, mspr_df, as_of)`**
- Para qué: Calcular ratios de compra vs. venta de insiders en ventana 90 días.
- Variables: Net buying ratio, insider sell %, tendencia mensual.

---

## module/steps/step_02_dataset/builders/sentiment.py

**Para qué sirve:** Construir features de "multitud": consenso de analistas, consenso de insiders, sorpresas de ganancias.

**Lógica:** Si todos (analistas + insiders + ganancias) están optimistas → bullish.

### Clase `SentimentFeatureBuilder`:

**`build(recommendation_df, mspr_df, insider_df, eps_df, as_of)`**
- Para qué: Calcular % de analistas compradores, tasa de sorpresas de ganancias, cambios de consenso.

---

## module/steps/step_02_dataset/builders/sector.py

**Para qué sirve:** Normalización sectorial: entrenar máquinas que conviertan ratios en "posición dentro del sector".

### Clase `SectorNormalizer`:

**`fit(features_dict, sector_map)`**
- Para qué: Aprender media y desviación estándar de cada variable POR SECTOR.

**`transform(features, sector)`**
- Para qué: Convertir valor absoluto a "posición normalizada dentro del sector".

**`fit_transform(features_dict, sector_map)`**
- Para qué: Aprender + aplicar en una llamada.

---

## module/agents/__init__.py

**Para qué sirve:** Marca el paquete `agents` y expone todas las clases de análisis.

**Sin funciones/métodos** — Solo imports.

---

## module/agents/base.py

**Para qué sirve:** Base común de todos los módulos de análisis: interfaz estándar para entrenar y predecir.

**Lógica:** Cada módulo sigue el patrón: fit(datos históricos) → predict(datos nuevos) → scores [0,1].

### Clase `BaseAgent`:

**`fit(X, y, **kwargs)`** (abstracto)
- Para qué: Entrenar sobre datos históricos.

**`predict_score(X)`** (abstracto)
- Para qué: Predecir scores [0,1] para nuevos datos.

**`clean_features(X, y)`**
- Para qué: Limpiar: eliminar NaN masivos, outliers, asegurar solo ratios comparables (no magnitudes absolutas).
- Protección: Bloquea revenue/net_income (incomparable entre empresas) e impone solo ratios.

**`save_diagnostics(fold, extra)`**
- Para qué: Guardar métricas de entrenamiento en JSON (auditoría).

**`save_feature_importances(importances, fold)`**
- Para qué: Guardar ranking de variables más influyentes en CSV.

---

## module/agents/fundamental.py

**Para qué sirve:** Módulo de análisis 1/5: detecta empresas financieramente SANAS (rentables, solventes, de calidad).

**Entrada:** ~25 ratios de rentabilidad, solvencia, crecimiento, calidad.
**Salida:** Score [0,1] de "salud financiera".

### Clase `FundamentalAgent`:

**`fit(X, y, fold, sector_col)`**
- Para qué: Entrenar modelo sobre datos históricos.
- Proceso: 1. Selecciona top 12 features (elimina ruido) / 2. Entrena máquina estadística / 3. Calibra umbral.

**`predict_score(X, sector_col)`**
- Para qué: Predecir score para nuevas observaciones.

---

## module/agents/momentum.py

**Para qué sirve:** Módulo de análisis 2/5: captura TENDENCIAS (precio al alza, ganancias sorpresas positivas).

**Entrada:** Osciladores (RSI, MACD), momentum, volatilidad, sorpresas de ganancias.
**Salida:** Score [0,1] de "momentum".

### Clase `MomentumAgent`:

**`fit(X, y, fold)`**
- Para qué: Entrenar.
- Proceso: 1. Crea features derivadas (RSI alto = overbought, etc.) / 2. Entrena modelo / 3. Guarda feature ranking.

---

## module/agents/valuation.py

**Para qué sirve:** Módulo de análisis 3/5: detecta INFRAVALORACIÓN (múltiplos bajos vs. histórico y sector).

**Entrada:** P/E, P/B, P/S, EV/EBITDA vs. 5Y y percentil sectorial.
**Salida:** Score [0,1] de "infravaloración".

### Clase `ValuationAgent`:

**`fit(X, y, fold, sector_col)`**
- Para qué: Entrenar.
- Normalización: Calcula de media/desv de cada múltiplo POR SECTOR durante train → para usar en predicción.

---

## module/agents/bear.py

**Para qué sirve:** Módulo de análisis 4/5: DETECCIÓN DE RIESGOS (deuda alta, insiders vendiendo, pérdidas).

**Entrada:** Ratios de deuda, insiders, liquidez, calidad.
**Salida:** Score [0,1] de "riesgo" → inverted en meta-learner para no invertir en alto riesgo.

### Clase `BearAgent`:

**`fit(X, y, fold)`**
- Para qué: Entrenar.
- Estrategia dual: 1. 10 reglas de riesgo (deuda >20%, insiders vendiendo, etc.) / 2. Modelo estadístico / 3. Combina ambos → score final.

**Reglas de riesgo ponderadas:**
1. Deuda creció > 20% en último año
2. Deuda/Equity > 3.0
3. Deuda/EBITDA > 6.0
4. Flujo de caja operativo negativo
5. Pérdidas en ≥2 trimestres consecutivos
6. Revenue bajó vs. año anterior
7. Cobertura de intereses < 1.5x
8. Current ratio < 1 (problemas de liquidez)
9. Insiders vendiendo > 70%
10. Sorpresa negativa de ganancias

---

## module/agents/sentiment.py

**Para qué sirve:** Módulo de análisis 5/5: SENTIMIENTO del "crowd" (analistas + insiders + sorpresas).

**Entrada:** Consenso de analistas, insiders, sorpresas de ganancias.
**Salida:** Score [0,1] de "sentimiento bullish".

### Clase `SentimentAgent`:

**`fit(X, y, fold)`**
- Para qué: Entrenar.
- Feature derivadas: % neto bullish de analistas, insider buying ratio normalizado, tendencia de sorpresas.

---

## module/agents/sector_rotation.py

**Para qué sirve:** Módulo TOP-DOWN (complementario): predice si UN SECTOR entero va a outperformear en próximo trimestre.

**Uso:** Refuerza picks de tickers si su sector es alcista; las debilita si sector es bajista.

### Clase `SectorRotationAgent`:

**`fit()`**
- Para qué: Entrenar.
- Proceso: 1. Agrega features por sector-trimestre (media de ROE, P/E, momentum, etc.) / 2. Etiqueta: ¿sector superó S&P500? / 3. Entrena modelo sectorial.

**`predict_sector_scores()`**
- Para qué: Predecir score de outperformance para cada sector.

---

## module/agents/meta_learner.py

**Para qué sirve:** Combina los 5 módulos base + sector en predicción FINAL única.

**Entrada:** 5 scores de módulos + sector score.
**Salida:** Score final [0,1] de "recomendación de compra".

### Clase `MetaLearner`:

**Protecciones críticas:**
1. **Hard risk gate:** Si riesgo bear ≥ 90% → fuerza score a 0.05 (no compra).
2. **Soft penalty:** Si riesgo bear alto → reduce score de forma proporcional.
3. **Recalibración:** Si train tiene distribución sesgada → re-ajusta umbral para mantener interpretabilidad.

**Forma de combinar señales:** Combina dos modelos en paralelo (regresión + árbol) + consenso de módulos base → pesa ambos modelos por precisión.

**Interacciones derivadas:**
- Fundamental × Valuation (si ambos bullish → señal fuerte)
- Momentum × Riesgo (momentum alto pero alto riesgo → cautela)
- Sector × Fundamental (si sector alcista y empresa sana → refuerza)

---

## module/steps/step_03_training/__init__.py

**Para qué sirve:** Marca la carpeta de Step 3 como paquete.

**Sin funciones/métodos** — Solo documentación.

---

## module/steps/step_03_training/training.py

**Para qué sirve:** Orquesta entrenamiento de CADA PERÍODO (fold): entrena 5 módulos + meta-learner, genera scores.

**Flujo por fold:** Datos históricos → Entrena 5 módulos → Entrena meta → Predice test → Genera scores.

### Funciones principales:

**`train_fold(df_train_norm, df_test_norm, y_train, y_test, fold_id, agents_results_dir, ...)`**
- Para qué: Entrenar 1 período walk-forward completo.
- Pasos:
  1. Entrena 5 módulos sobre datos históricos
  2. Genera scores anti-leakage de los 5 módulos (sin permitir que el módulo vea sus propios datos test)
  3. Usa esos scores como entrada para entrenar meta-learner
  4. Predice en período test
  5. Guardó reportes JSON de cada módulo

**`train_full_history(df_norm, y, agents_results_dir, ...)`**
- Para qué: Entrenar sobre TODO el histórico (sin split train-test, para modelo FINAL de predicción live).
- Uso: Cuando queremos la mejor predicción posible con todos los datos que tenemos.

---

## module/steps/step_03_training/oof.py

**Para qué sirve:** Generación "honesta" de scores de entrenamiento: evitar que el módulo vea sus propios datos de test.

**Analogía:** Como estudiar para examen con problemas del año PASADo, no del examen actual.

### Funciones:

**`generate_oof_scores(X, y, agents_config, n_splits, random_seed)`**
- Para qué: Generar scores para cada observación, pero entrenadoSIN verla.
- Proceso: 1. Divide datos históricos en 3 períodos / 2. Por cada período: entrena módulo en períodos anteriores / 3. Predice en el período actual / 4. Apila predicciones.

---

## module/steps/step_03_training/agent_config.py

**Para qué sirve:** Configuración declarativa de qué módulos entrenar.

**Uso:** Cambiar aquí si quieres activar/desactivar módulos SIN editar código de entrenamiento.

### Funciones:

**`build_agents_config(agents_results_dir, random_seed)`**
- Para qué: Devolver diccionario de configuración de los 5 módulos.

**`build_sector_rotation_agent(agents_results_dir, random_seed)`**
- Para qué: Factory para el módulo de sector.

---

## module/steps/step_04_evaluation/__init__.py

**Para qué sirve:** Marks el paquete de Step 4.

**Sin funciones/métodos** — Solo documentación.

---

## module/steps/step_04_evaluation/backtester.py

**Para qué sirve:** Simulación walk-forward: "Si hubieras seguido mis recomendaciones histórico, ¿cuánto hubiera ganado?"

**Resultado:** Reportes de retorno, Sharpe, máximo drawdown, alfa vs. S&P 500.

### Clase `WalkForwardBacktester`:

**`generate_folds(analysis_start_date, analysis_end_date)`**
- Para qué: Dividir período histórico en folds de entrenamiento y test.
- Proceso: 1. Cada fold: 8 años train, 1 trimestre test / 2. Genera calendario de folds.

**`simulate_portfolio(predictions_df, prices_dict, benchmark, fold_id, test_start, test_end, ...)`**
- Para qué: SIM ULACIÓN USD: "si compré estos tickers con estos pesos, ¿cuánto gané?"
- Pasos:
  1. Selecciona tickers con score ≥ umbral
  2. Asigna pesos (equiponderado o ponderado por score)
  3. Por cada ticker: busca precio de entrada y salida (real)
  4. Calcula rotornos diarios
  5. Compara vs. S&P 500
  6. Calcula Sharpe, drawdown, alfa

---

## module/steps/step_04_evaluation/evaluator.py

**Para qué sirve:** Orquestador del walk-forward completo: crea folds, entrena, simula, genera reportes.

**Entrada:** Dataset maestro, precios, configuración.
**Salida:** Carpeta `results/` con reportes y gráficos.

### Funciones principales:

**`run_walkforward_pipeline(df, sector_map, prices_dict, benchmark, spy_prices, ...)`**
- Para qué: Loop completo: por cada fold (trimestre) → entrena, predice, simula, reporta.
- Proceso: 1. Genera folds / 2. For cada fold: prepara datos / 3. Entrena 5 módulos + meta / 4. Simula cartera / 5. Genera reportes / 6. Stack resultados.

---

## module/steps/step_04_evaluation/metrics.py

**Para qué sirve:** Funciones estándar de cálculo de retorno, volatilidad, Sharpe, máximo drawdown.

**Uso:** Métricas para evaluar desempeño de la estrategia.

### Funciones:

**`cumulative_return(returns)`** → Retorno acumulado %
**`annualized_return(returns)`** → Anualized %
**`sharpe_ratio(returns, risk_free)`** → Excess return / volatilidad (qué tal el riesgo-retorno)
**`max_drawdown(returns)`** → Máxima caída desde pico (worst case)
**`calmar_ratio(returns)`** → Retorno / Max drawdown (qué tan buen timing tiene)
**`sortino_ratio(returns)`** → Sharpe pero solo penaliza downside (volatilidad mala)
**`compute_all_metrics(returns, risk_free, label)`** → Retorna diccionario con todas

---

## module/steps/step_04_evaluation/portfolio_simulator.py

**Para qué sirve:** Simulación realista en dólares: entrada a precio, slippage, comisiones, salida.

**Resultado:** Curva de equity en USD, PnL, comisiones pagadas.

### Funciones:

**`simulate_fold_usd(fold_id, prices_dict, selected_tickers, weights, entry_date_requested, exit_date_requested, starting_cash_usd, ...)`**
- Para qué: Simular 1 fold Long-Only con fricción realista.
- Pasos:
  1. Busca precio de entrada (en o después de fecha solicitada)
  2. Busca precio de salida
  3. Calcula shares por ticker = (cash × weights - comisión) / precio
  4. Rastrea equity diario
  5. Cierra posiciones
  6. Retorna trades, equity curve, PnL

---

## module/steps/step_04_evaluation/explainability.py

**Para qué sirve:** Genera explicaciones LOCALES: "¿por qué el modelo dijo X?" vía contribuciones de features.

**Forma:** Si disponible, contribución numérica de cada variable a la predicción.

### Funciones:

**`build_explainer_for_agent(agent_name, model, feature_cols, X_train, results_dir, fold, model_type)`**
- Para qué: Construir explicador para 1 módulo.
- Proceso: 1. Si disponible, crea máquina de explicación / 2. Calcula contribuciones en training / 3. Guarda gráficos.

---

## module/steps/step_04_evaluation/ablation.py

**Para qué sirve:** Medir cuánto contribuye CADA MÓDULO: ¿sin Fundamental cuánto baja el desempeño?

**Resultado:** Tabla de contribución de cada módulo a AUC.

### Funciones:

**`run_ablation_study(df_test_scored, y_test, df_train_norm, y_train, agents_results_dir, fold_id, ...)`**
- Para qué: Para cada módulo: eliminalo, re-entrena meta, mide drop.
- Salida: Dict con contribución de cada módulo.

---

## module/steps/step_04_evaluation/reports.py

**Para qué sirve:** Generar reportes textuales legibles: resumen ejecutivo del backtest.

**Formato:** Markdown/TXT con tablas ASCII.

### Funciones:

**`generate_text_report(summary, fold_results, agent_diag_history, backtest_results_dir)`**
- Para qué: Crear archivo `backtest_report.txt` con gráficas ASCII, tablas de folds, desglose de módulos.

---

## module/steps/step_04_evaluation/visualization.py

**Para qué sirve:** Generar gráficos: curva de riqueza, drawdown, retornos anuales, Sharpe por fold, importancia de módulos.

**Formato:** 7 gráficos juntos en PDF/PNG.

### Clase `Visualizer`:

**`plot_full_report(strategy_returns, benchmark_returns, fold_results, agent_diagnostics, suffix)`**
- Para qué: Hacer dashboard visual del backtest.
- Gráficos:
  1. Curva de riqueza (estrategia vs. S&P 500)
  2. Máximo drawdown por día
  3. Alpha acumulado (diferencia vs. benchmark)
  4. Retornos anuales por fold
  5. Distribución de retornos (histograma)
  6. Sharpe por fold
  7. Importancia de cada módulo

---

## module/steps/step_05_live/__init__.py

**Para qué sirve:** Marks el paquete de Step 5.

**Sin funciones/métodos** — Solo documentación.

---

## module/steps/step_05_live/live_fold.py

**Para qué sirve:** Monitoreo en tiempo real: qué recomienda HOY y cómo se desempeña después (realmente).

**Entrada:** Modelo entrenado en TODOS los datos históricos.
**Salida:** Predicciones HOY + retornos reales desde HOY adelante.

### Funciones principales:

**`run_live_fold(df, sector_map, tickers_ok, as_of_date, router, builders, results_dir, agents_results_dir, ...)`**
- Para qué: Hacer predicción live y luego esperar precios reales.
- Pasos:
  1. Entrena modelo en TODOS los históricos
  2. Genera features HOY
  3. Predice scores HOY
  4. Deja registro de predicción
  5. Descarga precios reales posteriores
  6. Calcula retorno real vs. S&P 500
  7. Reporta "¿acertó el modelo?"

---

## module/steps/step_05_live/live_prices.py

**Para qué sirve:** Descarga precios recientes desde Yahoo para calcular retornos REALES (sin guardar en disco).

### Funciones:

**`download_live_prices(tickers, start, end)`**
- Para qué: Traer precios en memoria (efímero).
- Salida: Dict {ticker: serie de prices}.

---

## module/steps/step_05_live/returns.py

**Para qué sirve:** Helpers para calcular retorno simple: (cierre_final - cierre_inicial) / cierre_inicial.

### Funciones:

**`qtd_return(close_series)`** → Retorno flotante (ej: 0.08 = +8%).

---

## module/common/__init__.py

**Para qué sirve:** Marks `module.common` como paquete de utilidades compartidas.

**Sin funciones/métodos** — Solo documentación.

---

## module/steps/__init__.py

**Para qué sirve:** Marks `module.steps` como paquete y documenta orden de ejecución sugerido.

**Orden:** Step 01 (Datos) → Step 02 (Features) → Step 03 (Entrenamiento) → Step 04 (Evaluación) → Step 05 (Live).

**Sin funciones/métodos** — Solo documentación.

---

## module/steps/step_02_dataset/builders/__init__.py

**Para qué sirve:** Marks paquete de builders (feature engineering).

**Sin funciones/métodos** — Solo documentación.

---

## module/steps/step_04_evaluation/fold_report.py

**Para qué sirve:** Construir reportes tabulares por período (fold/trimestre): scores, explicaciones, auditoría.

**Resultado:** CSVs con "decisión + razón" para cada ticker cada trimestre.

### Funciones principales:

**`_is_ratio_or_normalized_feature(feature)`**
- Para qué: Filtrar: ¿esta variable es comparable entre empresas o magnitud absoluta?
- Lógica: Acepta ratios, rechaza revenue/shares (incomparables).

**`_agent_text_label(agent, score)`**
- Para qué: Traducir score numérico a etiqueta legible (ej: 0.72 → "Bullish").

**`_describe_feature_value(feature, value)`**
- Para qué: Convertir "ROE=0.15" a frase: "Rentabilidad equity muy buena".

**`_build_agent_explanation(row, agent, agent_score, shap_drivers=None, top_n=4)`**
- Para qué: Generar explicación textual para UN módulo + UN ticker: "¿qué llevó a esta decisión?"
- Pasos: 1. Etiqueta base (bullish/bearish) / 2. Si hay contribuciones, lista positivas y negativas / 3. Si no, usa heurísticas / 4. Construye frase legible.

**`build_fold_scores_df(df_test_scored, y_test, fold_id, year_quarter, agents, audit_df=None, actual_returns=None)`**
- Para qué: Construir tabla maestro de un período: ticker/score/decisión/explicación/resultado real.
- Salida: DataFrame ordenado por score.

**`export_fold_scores(df, agents_results_dir, fold_id)`**
- Para qué: Guardar tabla de fold a CSV.

**`export_quarter_snapshot_audit(df_test_scored, year_quarter, agents_results_dir)`**
- Para qué: Exportar snapshot: "¿cuáles eran TODOS los features de cada ticker en este período?"
- Uso: Auditoría fina.

**`export_quarter_agent_feature_audit(df_test_scored, agents, year_quarter, agents_results_dir)`**
- Para qué: Tabla granular: (ticker, módulo, feature, valor) → permite auditoría por variable.

**`export_all_folds_scores(agents_results_dir)`**
- Para qué: Concatenar todos los períodos → histórico completo en 1 sólo archivo.

---

## module/steps/step_04_evaluation/selection_reports.py

**Para qué sirve:** Auditoría explicativa: "¿por qué compramos X? ¿por qué no compramos Y?"

**Resultado:** CSVs y JSONs estructurados con razones de selección/rechazo.

### Funciones principales:

**`_score_label(score)`**
- Para qué: Convertir score a etiqueta binaria (bullish/bearish) con corte en 0.5.

**`_normalise_ticker_list(tickers)`**
- Para qué: Deduplicar lista de tickers preservando orden.

**`_rule_based_drivers(row, score)`**
- Para qué: Generar drivers (factores) cuando no hay explicador disponible.
- Lógica: Evalúa reglas heurísticas de cada variable.

**`_fallback_explanation(agent_name, row, ticker, score)`**
- Para qué: Explicación de respaldo si falla explicador avanzado.

**`_format_driver_list(drivers)`**
- Para qué: Serializar lista de factores a string compacto.

**`_split_driver_groups(drivers)`**
- Para qué: Separar factores a favor vs. en contra.

**`build_selection_audit_df(df_scored, selected_tickers, score_col="final_score", threshold=0.5)`**
- Para qué: Construir tabla de auditoría: ranking, seleccionado/no, razón.
- Categorías:
  1. Seleccionados (score ≥ umbral)
  2. Descartados por score bajo
  3. Cercanos al umbral (edge cases)

**`build_explanation_candidate_tickers(audit_df, threshold=0.5, top_extra=10, near_margin=0.05, max_candidates=30)`**
- Para qué: Seleccionar sub-conjunto de tickers para explicar en detalle (no todos, para ahorrar tiempo).
- Criterios: Seleccionados + top + bottom + cercanos al umbral.

**`export_selection_audit(audit_df, results_dir, fold_id=None, prefix="fold")`**
- Para qué: Guardar auditoría de selección en CSV + resumen en JSON.

**`export_ticker_explanations(agents, df_test, scores, fold_id, agents_results_dir, candidate_tickers, audit_df=None, explanation_top_n=6, prefix="fold")`**
- Para qué: Generar explicaciones por ticker y módulo: "¿qué contribuyó a la decisión?"
- Formato: CSv plano + JSON jerárquico.
- Pasos: 1. Por cada ticker candidato / 2. Por cada módulo / 3. Extrae score / 4. Si existe explicador, lo usa; si no, fallback / 5. Exporta.

---

**FIN DE DOCUMENTACIÓN**

**Total: 53 archivos .py documentados** en lenguaje operacional, sin jerga técnica, orientado a "qué hace" y "cómo se usa" en el día a día.
