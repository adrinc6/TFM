# Multi-Agent ML Stock Picker

Sistema de investigación cuantitativa para seleccionar acciones del S&P 500 con una arquitectura multi-agente de ML, snapshots punto-en-tiempo y backtesting walk-forward. El objetivo principal es evaluar si un portafolio seleccionado por modelos especializados puede generar alpha frente al benchmark cuando las posiciones se gestionan con TP/SL adaptativos durante un horizonte objetivo de 12 meses.

> **Estado del proyecto:** repositorio de investigación académica/TFM. No constituye asesoramiento financiero ni una estrategia lista para producción sin validación adicional.

## Qué hace el proyecto

El pipeline completo:

1. Descarga y consolida datos financieros, precios y macro.
2. Construye un dataset maestro con features punto-en-tiempo por ticker y snapshot.
3. Entrena agentes especializados por dominio.
4. Combina señales con un meta-learner.
5. Selecciona un portafolio por fold walk-forward.
6. Simula la estrategia principal TP/SL adaptativa.
7. Compara contra:
   - benchmark S&P 500/SPY;
   - Buy & Hold 12M contrafactual sobre el mismo portafolio;
   - variantes TP/SL investigables como `hybrid_learned`.
8. Exporta CSVs, JSONs, gráficos, auditorías anti-leakage y reportes interpretativos.

## Arquitectura de alto nivel

```text
Data / Finnhub / Yahoo / Macro
        |
        v
Point-in-time master dataset
        |
        v
Multi-agent training
  - fundamental
  - valuation
  - momentum
  - bear/risk
  - sector rotation
  - optional sentiment
        |
        v
Meta-learner + ranking + portfolio selection
        |
        v
Walk-forward backtest
  - TP/SL base strategy
  - hybrid_learned TP/SL variant
  - Buy & Hold 12M counterfactual
  - benchmark comparison
        |
        v
results/<run>/strategy artifacts
```

## Estructura principal

```text
.
├── analyzer.py                          # Entrypoint principal del pipeline
├── analyzer_II.py                       # Grid/escenarios en paralelo
├── environment.py                       # Configuración global del proyecto
├── DOCUMENTACION_PROYECTO.md            # Documentación técnica extensa
├── requirements.txt
├── pyproject.toml
├── data_finnhub/                        # Datos locales y dataset maestro
├── module/
│   ├── agents/                          # Agentes ML y meta-learner
│   ├── common/                          # Métricas, as-of, régimen, optimización
│   └── steps/
│       ├── step_01_data/                # Descarga/consolidación
│       ├── step_02_dataset/             # Dataset punto-en-tiempo
│       ├── step_03_training/            # Entrenamiento walk-forward
│       └── step_04_evaluation/          # Backtesting, reporting, análisis
├── tests/                               # Tests unitarios/regresión
└── results/                             # Artefactos generados por corrida
```

## Instalación

Requiere Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .[dev]
```

Si se van a descargar datos desde Finnhub, configura la API key:

```bash
export FINNHUB_API_KEY="<tu_api_key>"
```

También puedes crear un archivo `.env` en la raíz del proyecto con la misma variable.

## Configuración

`environment.py` es la fuente de verdad de parámetros. Algunos flags clave:

| Parámetro | Descripción |
|---|---|
| `ANALYSIS_REFERENCE_DATE` | Fecha ancla del fold más reciente. |
| `WALKFORWARD_NUM_TESTS` | Número de folds walk-forward. |
| `HOLDING_PERIOD_MONTHS` | Horizonte objetivo, normalmente 12 meses. |
| `TP_SL_MAX_STOCKS` / `TP_SL_MIN_STOCKS` | Tamaño del portafolio seleccionado. |
| `PORTFOLIO_OPTIMIZER` | Optimizador de pesos (`hrp`, `risk_parity`, `markowitz`, etc.). |
| `ENABLE_BUY_HOLD_COUNTERFACTUAL` | Activa comparación Buy & Hold 12M sobre el mismo portafolio. |
| `EXPORT_TP_SL_VS_BUY_HOLD` | Exporta CSV/JSON/gráficos de comparación TP/SL vs Buy & Hold. |
| `ENABLE_TP_SL_RESEARCH_VARIANTS` | Habilita variantes investigables de TP/SL. |
| `TP_SL_VARIANT_MODE` | Modo TP/SL activo: `base`, `vol_adjusted`, `momentum_adjusted`, `regime_adjusted`, `hybrid_learned`. |

Puedes sobreescribir parámetros sin editar el archivo usando `ENV_OVERRIDES_JSON`:

```bash
ENV_OVERRIDES_JSON='{"TP_SL_VARIANT_MODE":"hybrid_learned","WALKFORWARD_NUM_TESTS":4}' python analyzer.py
```

## Ejecución

### Pipeline completo

```bash
python analyzer.py
```

El pipeline crea una carpeta de resultados versionada bajo `results/`, con subcarpetas de configuración, logs, folds, agentes, backtest y estrategia.

### Grid de escenarios

```bash
python analyzer_II.py
```

`analyzer_II.py` ejecuta escenarios en paralelo con overrides de `environment.py` y consolida métricas comparables, incluyendo TP/SL vs Buy & Hold e híbrido vs base cuando están habilitados.

### Tests

```bash
pytest -q
```

Tests focalizados:

```bash
pytest -q tests/test_buy_hold_counterfactual.py
pytest -q tests/test_financial_strategy_constraints.py
pytest -q tests/test_tp_sl_strategy.py
```

## Estrategias evaluadas

### TP/SL base

Es la estrategia principal. El sistema entra con el portafolio seleccionado por ML y gestiona cada posición con niveles adaptativos de Take-Profit, Stop-Loss y trailing stop. No se elimina TP/SL: todos los análisis nuevos comparan contra esta ruta base.

### Buy & Hold 12M contrafactual

Simula qué habría pasado si se comprara exactamente el mismo portafolio seleccionado por TP/SL, con la misma fecha de entrada y los mismos pesos iniciales, pero sin ejecutar TP, SL ni trailing stop.

Este contrafactual es solo evaluación: no reentrena modelos, no cambia scores, no modifica selección y no usa información futura para decidir tickers.

### `hybrid_learned`

Variante investigable de TP/SL que aprende niveles más realistas a partir de trayectorias históricas train-only del propio ticker. Combina:

- volatilidad histórica;
- momentum;
- régimen macro;
- score/confianza del modelo;
- distribución de runups/drawdowns de 12 meses;
- probabilidad de recuperación tras drawdowns;
- evidencia histórica de TP-before-drawdown.

Al alcanzar TP, activa una lógica de profit protection con trailing dinámico que recalcula el stop desde máximos recientes, nunca lo baja y ajusta la distancia según volatilidad, momentum, régimen y beneficio acumulado.

## Outputs principales

Los artefactos suelen generarse en `results/<run>/strategy/` y en carpetas por fold:

| Archivo | Descripción |
|---|---|
| `report.txt` | Reporte textual global con métricas, conclusiones y advertencias metodológicas. |
| `backtest_summary.json` | Resumen global de estrategia, benchmark, Buy & Hold e híbrido. |
| `folds_results.csv` | Métricas principales por fold. |
| `tp_sl_vs_buy_hold_by_fold.csv` | Comparación TP/SL vs Buy & Hold por fold. |
| `tp_sl_vs_buy_hold_by_ticker.csv` | Comparación TP/SL vs Buy & Hold por ticker. |
| `tp_sl_vs_buy_hold_summary.json` | Resumen agregado de la comparación contrafactual. |
| `tp_sl_hybrid_vs_base_by_fold.csv` | Comparación TP/SL híbrido vs base por fold. |
| `tp_sl_hybrid_vs_base_by_ticker.csv` | Comparación TP/SL híbrido vs base por ticker. |
| `tp_sl_hybrid_robustness_summary.json` | Folds/tickers/sectores/regímenes donde el híbrido gana o pierde. |
| `learned_tp_sl_levels_by_ticker.csv` | Niveles aprendidos TP/SL/trailing y diagnósticos por ticker. |
| `runup_drawdown_recovery_stats.csv` | Estadísticas históricas train-only de runup/drawdown/recuperación. |
| `trailing_dynamics_by_ticker.csv` | Eventos de trailing stop y recalculaciones. |
| `leakage_audit.csv` | Auditoría anti-leakage. |
| `missing_prices_report.csv` | Tickers/folds omitidos por falta de precios. |

Gráficos típicos:

- equity curve TP/SL vs Buy & Hold vs benchmark;
- alpha por fold;
- diferencia TP/SL - Buy & Hold;
- distribución de exits TP/SL;
- gráficos de performance y drawdown global.

## Validación anti-leakage

El diseño busca evitar leakage mediante:

- snapshots punto-en-tiempo;
- features disponibles as-of;
- purga/embargo temporal en entrenamiento;
- aprendizaje `hybrid_learned` solo con trayectorias históricas que terminan antes del snapshot/fold evaluado;
- Buy & Hold como contrafactual post-selección, sin afectar ranking ni pesos;
- tests de regresión para compatibilidad base y comparación sobre el mismo portafolio.

Checklist mínimo antes de interpretar resultados:

1. `TP_SL_VARIANT_MODE="base"` reproduce la ruta TP/SL base.
2. `latest_train_path_end < snapshot/entry_date` en diagnósticos híbridos.
3. TP/SL base, híbrido y Buy & Hold comparten tickers y pesos.
4. El contrafactual no modifica scores ni selección.
5. No se eligen parámetros mirando el fold de test.
6. La mejora no está concentrada en un único ticker, sector o régimen.

## Interpretación de resultados

Preguntas clave que el proyecto permite responder:

- Dado el mismo portafolio seleccionado por ML, ¿TP/SL mejora frente a Buy & Hold 12M?
- ¿El TP/SL base protege capital o recorta upside?
- ¿`hybrid_learned` mejora al TP/SL base por fold, ticker, sector o régimen?
- ¿El trailing actual protege beneficios o sale demasiado pronto?
- ¿Los SL saltan por ruido o evitan pérdidas persistentes?

Una mejora de `hybrid_learned` frente a base debe interpretarse como evidencia de investigación walk-forward, no como optimización final lista para producción.

## Notas y limitaciones

- La calidad de la simulación depende de la disponibilidad y limpieza de precios/fundamentales.
- Tickers con poca historia usan más fallback hacia niveles base.
- Los patrones históricos de runup/drawdown pueden no repetirse en nuevos regímenes.
- El objetivo es investigación interpretable y auditada, no maximización por grid sobre test.
- No se modelan impuestos ni retornos after-tax.

## Documentación ampliada

Para detalles completos de arquitectura, features, entrenamiento, auditorías, outputs y metodología, consulta:

```text
DOCUMENTACION_PROYECTO.md
```
