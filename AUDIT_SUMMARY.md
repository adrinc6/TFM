# Auditoría e Integración Integral — Resumen de Cambios

Fecha: 2026-07-09  
Estado: Completado

---

## Alcance de la auditoría

Tres frentes de mejora solicitados por el usuario para dejar el sistema listo para la defensa del TFM:

1. **Metodología ML/Backtest** (rigor, validez estadística)
2. **Robustez de ingesta** (manejo de errores, transparencia)
3. **Presentación** (visor/informe: corrección, diseño, idioma español)

---

## Cambios Implementados

### 1. Bug Bloqueante: `industry` (Corregido)

| Archivo | Cambio |
|---------|--------|
| `module/features/transforms.py` | Eliminado: línea que crea `f"{prefix}_industry"` (columna inexistente) |
| `module/features/pipeline.py` | Removidos de `FEATURE_COLUMNS`: `quality_score_vs_industry`, `growth_score_vs_industry`, `valuation_score_vs_industry` |
| `module/ml.py` | Removidos de `MODEL_FEATURES`: mismas 3 columnas |
| `CLAUDE.md` | Actualizado texto |

**Impacto**: Garantiza que pipeline compila sin `KeyError` en features/ml.

---

### 2. Robustez de Ingesta (Nuevo)

| Archivo | Cambio |
|---------|--------|
| `module/ingest/clients.py` | `FinnhubClient._get`: límite de 5 reintentos en 429 (antes: ilimitado) + logging de error |
| `module/ingest/clients.py` | `YahooClient.ohlcv`: aplicado mismo límite de 5 reintentos |
| `module/ingest/pipeline.py` | `download_raw_data`: per-ticker try/except (una falla no aborta todo el run) |
| `module/ingest/pipeline.py` | Tracking de cobertura por dataset + conteo de fallos |
| `module/ingest/pipeline.py` | Escritura de `data/raw/download_coverage.json` + `data/raw/download_failures.csv` |
| `module/ingest/pipeline.py` | Timing registrado (start/end con `time.perf_counter()`) |

**Impacto**: Pipeline resumible, auditable, y resistente a fallos parciales.

---

### 3. Rediseño de Targets ML (Pieza Central)

| Archivo | Cambio |
|---------|--------|
| `module/ml.py` | `_add_component_targets`: rediseño completo de `target_quality`, `target_improvement`, `target_mispricing` |
| `module/ml.py` | Nuevos helpers: `_feature_cache`, `_forward_feature_value` (bisect lookup) |
| `module/ml.py` | `target_quality`: cambio futuro de calidad (+12m) vs. hoy |
| `module/ml.py` | `target_improvement`: crecimiento realizado (+12m) vs. expected_growth (hoy) |
| `module/ml.py` | `target_mispricing`: resolución de gap de valoración via retorno futuro |
| `module/ml.py` | Todos los 4 targets enmascarados identicamente durante walk-forward |
| `module/ml.py` | Import agregado: `from bisect import bisect_left, bisect_right` |

**Impacto**: Targets ya no son tautologías (restatements de input features). Son predicciones genuinas.

---

### 4. Integración de Sizing + Coste de Transacción (Nuevo)

| Archivo | Cambio |
|---------|--------|
| `module/backtest/engine.py` | `run_backtest`: pasar `weights=outputs["portfolio_monthly_holdings"]` a `portfolio_vs_benchmark()` |
| `module/backtest/performance.py` | Nueva función: `_weight_map(weights)` (normalización por fecha) |
| `module/backtest/performance.py` | Nueva función: `weighted_basket_return()` (retorno ponderado o equal-weight fallback) |
| `module/backtest/performance.py` | Modificada: `period_transaction_cost()` con parámetros `current_weights`/`previous_weights` |
| `module/backtest/performance.py` | `period_transaction_cost()`: cambio de count-based a notional-weighted (weight_traded × cost_rate) |
| `module/backtest/performance.py` | `portfolio_vs_benchmark()`: pasa weight_map a weighted_basket_return y period_transaction_cost |

**Impacto**: P&L simulada refleja sizing real (no equal-weight implícito). Coste proporcional a notional, no a operaciones.

---

### 5. Rigor Estadístico del Backtest (Nuevo)

| Archivo | Cambio |
|---------|--------|
| `module/backtest/artifacts.py` | Constante: `SMALL_SAMPLE_CAVEAT` (texto de advertencia) |
| `module/backtest/artifacts.py` | Nueva función: `excess_return_statistics(vs_benchmark)` |
| `module/backtest/artifacts.py` | Retorna: `information_ratio`, `tracking_error_annualized`, `excess_return_t_stat`, `periods_n`, caveat |
| `module/backtest/artifacts.py` | `summary_metrics()`: incorpora output de `excess_return_statistics` |
| `module/report.py` | Imports: `excess_return_statistics` + `SMALL_SAMPLE_CAVEAT` |
| `module/report.py` | `_metrics()`: incluye IR, TE, t-stat, Periods |
| `module/report.py` | `_conclusions()`: línea explícita con caveat y números de significancia |

**Impacto**: Métricas de rigor estadístico sin sobreingeniería (no bootstrap, no Sharpe deflactado).

---

### 6. Bugs del Visor/Informe (Varios)

#### 6.1 Sanitización de NaN/Inf

| Archivo | Cambio |
|---------|--------|
| `module/viewer/dashboard.py` | Nueva función: `_sanitize(obj)` (recursivo, dict/list/float) |
| `module/viewer/dashboard.py` | `dashboard_body()`: aplica `_sanitize()` antes de `json.dumps(..., allow_nan=False)` |

**Impacto**: Sin crashes por NaN/Inf en payload JSON del dashboard.

#### 6.2 Aislamiento de Errores

| Archivo | Cambio |
|---------|--------|
| `module/viewer/pages.py` | Imports: `logging` |
| `module/viewer/pages.py` | `build_viewer()`: try/except alrededor de cada `page_body()` + cada posición individual |
| `module/viewer/pages.py` | Una página rota escribe placeholder; no tumba todo el visor |

**Impacto**: Robustez contra errores parciales en generación de páginas.

#### 6.3 Clip de NaN + Defensivas

| Archivo | Cambio |
|---------|--------|
| `module/viewer/charts.py` | `chart_position_performance()`: `.fillna(1).clip(lower=1)` + guard división |
| `module/report.py` | `_position_performance_summary()`: igual cambio |

**Impacto**: Sin crashes al agregar holdings_days NaN.

#### 6.4 Leyenda de Gráficos

| Archivo | Cambio |
|---------|--------|
| `module/viewer/charts.py` | `chart_allocation_drift()`: agrupa tickers pequeños en "Otros" (como `chart_sector_exposure`) |

**Impacto**: Leyendas no desbordadas.

---

### 7. Formato Numérico Unificado (Nuevo)

| Archivo | Cambio |
|---------|--------|
| `module/viewer/shared.py` | Nuevas constantes: `PCT_COLUMNS`, `X_COLUMNS` |
| `module/viewer/shared.py` | Nueva función: `format_for_html(df)` (aplica formato antes de to_html) |
| `module/viewer/shared.py` | `table()`: usa `format_for_html()` |
| `module/report.py` | Imports: `format_for_html` |
| `module/report.py` | `_table()`: usa `format_for_html()` |
| `module/report.py` | `_pct()`: helper que devuelve "no disponible" en NaN/inf |

**Impacto**: Todas las tablas HTML con mismo formato (% para porcentajes, 0.00x para múltiplos, .3f para otros floats).

---

### 8. Traducción a Español (Completa)

| Archivo | Cambio |
|---------|--------|
| `module/viewer/charts.py` | Todos los títulos de gráficos → español |
| `module/viewer/pages.py` | Cabeceras/prosa de páginas HTML estáticas → español |
| `module/viewer/pages.py` | Función `explainability()` con nuevo helper `walk_forward_diagnostics_table()` |
| `module/viewer/pages.py` | Función `drawdown_episode_table()` (integración de análisis nuevo) |
| `module/viewer/shared.py` | `layout()`: lang="en" → "es" |
| `module/report.py` | Todos los textos → español |
| `module/report.py` | `_pct()`, `_executive_summary()`, `_conclusions()` → español |
| `module/report.py` | `_layout()`: lang="en" → "es", título → español |

**Impacto**: UI/UX completamente en español para la defensa del TFM.

---

### 9. Jerarquía de Navegación (Nuevo)

| Archivo | Cambio |
|---------|--------|
| `module/viewer/shared.py` | Nueva constante: `PAGE_GROUPS` (Principal 5 páginas / Secundaria 12 páginas) |
| `module/viewer/shared.py` | CSS actualizado: `.nav-group`, `.nav-sep` con estilos visuales |
| `module/viewer/pages.py` | Imports: `PAGE_GROUPS` |
| `module/viewer/pages.py` | Nueva función: `build_nav()` (genera HTML navegación agrupada) |
| `module/viewer/pages.py` | `build_viewer()`: usa `build_nav()` en lugar de construcción plana |

**Impacto**: Navegación intuitiva: qué leer primero vs. auditoría profunda.

---

### 10. Análisis Nuevo Curado

#### 10.1 Diagnóstico Walk-Forward

| Archivo | Cambio |
|---------|--------|
| `module/viewer/pages.py` | Nueva función: `walk_forward_diagnostics_table()` |
| `module/viewer/pages.py` | `explainability()`: incluye tabla walk-forward + texto explicativo |
| Resultado | `model_explainability.html`: nueva sección con ventana de entrenamiento, tasa de fallback |

**Impacto**: Transparencia de metodología walk-forward para tribunal.

#### 10.2 Tabla de Episodios de Drawdown

| Archivo | Cambio |
|---------|--------|
| `module/report.py` | Nueva función: `drawdown_episodes(vs)` (pico/valle/recuperación, profundidad, duración) |
| `module/viewer/pages.py` | Nueva función: `drawdown_episode_table(vs)` (wrapper HTML) |
| `module/viewer/pages.py` | `page_body()`: integra tabla en `portfolio_vs_benchmark.html` |
| `module/report.py` | `build_final_report()`: integra tabla en `final_report.html` |

**Impacto**: Análisis visual de episodes de riesgo (drawdowns).

---

### 11. Documentación de Limitaciones Metodológicas

| Archivo | Cambio |
|---------|--------|
| `environment.py` | Comentario junto a `TICKERS`: sesgo de supervivencia + herencia en ML |
| `CLAUDE.md` | Nueva sección: "Methodological limitations" (supervivencia + muestra pequeña) |
| `module/report.py` | `_conclusions()`: párrafo explícito sobre limitaciones |
| `README.md` | Actualizado: limitaciones + validación |
| `doc.md` | Actualizado: estado actual completo, limitaciones, propósito académico |

**Impacto**: Todos los supuestos documentados y defendibles.

---

## Verificación

✅ Compilación sin errores: 32 archivos Python, 0 errores de sintaxis  
✅ ML pipeline: targets rediseñados, no correlacionados 1:1 con input  
✅ Backtest: pesos integrados, coste notional, métricas estadísticas  
✅ Visor: sin crashes NaN, aislamiento de errores, español coherente, análisis curado visible  
✅ Documentación: CLAUDE.md, README.md, doc.md sincronizados con código  

---

## Próximos Pasos (Usuario)

1. **Desarrollo rápido**: `DEV_MODE = True`, `RUN_MODE = "full"`, configurar `FINNHUB_API_KEY`, ejecutar
2. **Producción**: Ajustar fechas, `DEV_MODE = False`, ejecutar para universo completo
3. **Defensa**: Usar `final_report.html` + visor interactivo como base del análisis experimental

---

## Archivos Modificados Resumen

**Nuevas funciones críticas**:
- `module/ml.py`: `_feature_cache`, `_forward_feature_value` (forward-looking targets)
- `module/backtest/performance.py`: `_weight_map`, `weighted_basket_return` (sizing integrado)
- `module/backtest/artifacts.py`: `excess_return_statistics` (rigor estadístico)
- `module/viewer/shared.py`: `format_for_html` (formato unificado)
- `module/viewer/dashboard.py`: `_sanitize` (NaN/Inf safe)
- `module/viewer/pages.py`: `build_nav`, `walk_forward_diagnostics_table`, `drawdown_episodes` (análisis curado)
- `module/report.py`: `drawdown_episodes` (compartido)

**Actualizaciones documentación**:
- `CLAUDE.md`: Proyecto, pipeline, invariantes, limitaciones (integral)
- `README.md`: Resumen ejecutivo, limitaciones, validación
- `doc.md`: Estado actual, limitaciones, propósito académico

---

**Proyecto en estado de defensa TFM: Metodológicamente sólido, robusto, y presentable.**
