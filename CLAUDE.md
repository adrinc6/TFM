# Guía del repositorio

## Propósito

Este TFM estudia si cinco agentes especializados aprenden una ordenación transversal de
acciones que se mantenga fuera de muestra, con datos point-in-time.

```text
catálogo cerrado → Model Study → optimización secuencial por Rank-IC → ganador
                                → robustez, carteras y perfiles informativos → informe
```

Antes de realizar cambios, leer `docs/architecture.md`, `docs/results.md` y `README.md`. Son las
fuentes de verdad del proyecto.

## Una sola ruta

- No crear Exploratory, Hypothesis, Confirmatory, Scenario, Experiment, Full Study ni runs sueltos.
- No crear una segunda forma de ejecutar la misma ciencia.
- No mantener aliases, adaptadores, esquemas antiguos, flags deprecated ni compatibilidad legacy.
- Cuando se sustituya algo, eliminar implementación, imports, API, interfaz, tests y documentación
  anteriores.
- Todo archivo y función debe tener un consumidor directo. Si no lo tiene, eliminarlo.

## Arquitectura

- `module/data`: ingesta, universo y panel PIT.
- `module/modeling`: features, agentes y meta-agente.
- `module/evaluation`: cartera y métricas.
- `module/research`: diagnósticos posteriores al ganador.
- `module/studies`: catálogo, runner y selección secuencial del único Model Study.
- `module/storage`: datasets, caché y evidencia.
- `module/web` y `app`: API y dashboard.

No se deben crear scenarios, experiments, runs sueltos ni rutas alternativas.

## Contrato científico

- Todo parámetro científico procede de `module/studies/catalog.py`.
- Rechazar claves desconocidas, valores libres y combinaciones incompatibles.
- No introducir lookahead.
- Solo las fases temporal, representación, modelo y meta pueden modificar el ganador.
- Seleccionar únicamente mediante Rank-IC robusto y comparaciones pareadas por cohorte.
- Alfa, IR, rentabilidad, turnover, perfiles, costes y robustez posterior son informativos.
- 2025–2026 es `known_stress_not_selection` y no puede entrar en ninguna decisión.
- Toda feature, etiqueta y cohorte del meta debe ser point-in-time y estar cerrada.
- Los cinco agentes quality, value, growth, momentum y risk permanecen activos.
- Los descartados guardan solo resúmenes.
- SPY solo es benchmark y nunca una posición. Los umbrales de la cartera son económicos, en
  puntos básicos ANUALES de alfa esperado (convertidos geométricamente al horizonte del modelo), y
  una venta solo se emite si el destino del dinero (otra acción o efectivo) es mejor que la
  posición después de costes; las entradas tienen histéresis y un mínimo de tenencia
  (`minimum_holding_period`) puede bloquear toda venta por tiempo, no por economía. La política de
  efectivo la gobierna únicamente `max_cash_weight` (remunerado al 0 %, con suelo de diversificación
  derivado del tope; 0 significa siempre invertida al 100 %) y es una decisión de cartera,
  diagnóstica: no altera el Rank-IC y por tanto no puede elegir modelo. Todas las variables de
  cartera comparten el mismo propósito: estabilidad del modelo ya congelado, nunca más alfa.

## Ejecución y persistencia

- La API crea el Study y el run antes de iniciar cálculo.
- Cada Study se ejecuta en un worker hijo.
- Todo progreso relevante se emite a terminal, `events.jsonl` y Consola.
- Un error debe persistir mensaje y traceback.
- Cancelar o cerrar el servidor debe terminar sus workers.
- Reanudar no repite runs finalizados ni duplica el ledger.
- Un artefacto parcial nunca es caché válida.
- No copiar datasets preparados dentro de resultados.

## Código mantenible

- Preferir funciones pequeñas, dataclasses y entradas/salidas explícitas.
- El orquestador no contiene cálculos científicos.
- El frontend nunca calcula métricas ni lee Parquets.
- No introducir abstracciones genéricas sin consumidor actual.
- Mantener los módulos por debajo de unas 500 líneas cuando exista una división natural.
- Actualizar juntos catálogo, Settings, API, dashboard, documentación y tests.

## Documentación

`docs/` contiene documentación duradera del proyecto:

| Fichero | Qué contiene |
|---|---|
| `docs/architecture.md` | Qué hace cada pieza del código y **por qué** está decidida así: universo y datos PIT, agentes y meta, doctrina de cartera, catálogo cerrado, regla de selección, Portfolio Study, diagnósticos, almacenamiento, API. |
| `docs/usage.md` | Cómo instalarlo y usarlo: requisitos, `.env`, ingesta, lanzar estudios, compilar el manuscrito, verificación y problemas frecuentes. |
| `docs/results.md` | Qué produce el sistema, cómo auditarlo y qué hay en cada artefacto de `results/studies/<study_id>/`. |

- Cambios metodológicos o de arquitectura: actualizar `docs/architecture.md`.
- Cambios en la operativa o en los comandos: actualizar `docs/usage.md`.
- Cambios en los artefactos que se producen: actualizar `docs/results.md`.
- **Las cifras viven en los artefactos de `results/studies/<study_id>/`** (`winner.json`,
  `evidence/summary.json`, `robustness.json`, `attribution.json`, `decisions.json`,
  `portfolio_grid.parquet`) y se leen de ahí. **No se duplican en ningún documento.** Toda
  afirmación numérica cita el `study_id` y la ruta del artefacto que la respalda.
- No presentar un test sintético o smoke dev como evidencia económica.

## El manuscrito LaTeX está congelado

`latex/TFM.tex`, `latex/TFM_ppt.tex` y `latex/assets/*.tex` **no se editan** como parte de un
cambio, ni se regeneran sus figuras y tablas.

Sí son editables `latex/scripts/*.py` y `latex/build.py` (es código) y `latex/COMO_COMPILAR.md`
(es documentación: cubre cómo compilar, las convenciones de escritura y las decisiones de formato),
pero el exportador **no se ejecuta** como parte del cambio. Entre actualizaciones el manuscrito está
desactualizado a propósito.

Para compilar, regenerar activos o verificar el proyecto: `python latex/build.py`, que lleva
interruptores `True`/`False` al principio del fichero para elegir qué se hace en cada ejecución.

## UTF-8

Todos los archivos se escriben en UTF-8. Cuidado con acentos y tildes. No introducir texto
recodificado, fragmentos de mojibake ni caracteres de sustitución.

## Verificación

```powershell
python -m pytest -q
python -m ruff check .
node --check app/js/app.js
```

Además:

- comprobar el preflight y el presupuesto;
- buscar referencias legacy;
- buscar mojibake;
- ejecutar el smoke real de cinco tickers si cambia el flujo;
- confirmar que no queda un worker vivo.

No ejecutar un estudio real completo sin autorización explícita.

No afirmar resultados financieros sin vincularlos a artefactos reales, citando `study_id` y ruta
(aunque en informe y presentación no hay que citarlos, pero sí comprobar que los resultados existen
en los artefactos y no se usan valores antiguos).
