# Instrucciones obligatorias para agentes

## Propósito

Este repositorio estudia si cinco agentes especializados aprenden una ordenación transversal de
acciones que se mantenga fuera de muestra. Solo existe este flujo:

```text
catálogo → Model Study → optimización secuencial por Rank-IC → ganador
         → robustez, carteras y perfiles informativos → informe
```

Antes de cambiar ciencia, ejecución, almacenamiento o dashboard, leer `docs/metodologia.md`,
`docs/bitacora.md`, `docs/informe_resultados.md` y `README.md`.

## Una sola ruta

- No crear Exploratory, Hypothesis, Confirmatory, Scenario, Experiment, Full Study ni runs sueltos.
- No crear una segunda forma de ejecutar la misma ciencia.
- No mantener aliases, adaptadores, esquemas antiguos, flags deprecated ni compatibilidad legacy.
- Cuando se sustituya algo, eliminar implementación, imports, API, interfaz, tests y documentación
  anteriores.
- Todo archivo y función debe tener un consumidor directo. Si no lo tiene, eliminarlo.

## Contrato científico

- Todo parámetro científico procede de `module/studies/catalog.py`.
- Rechazar claves desconocidas, valores libres y combinaciones incompatibles.
- Solo las fases temporal, representación, modelo y meta pueden modificar el ganador.
- Seleccionar únicamente mediante Rank-IC robusto y comparaciones pareadas por cohorte.
- Alfa, IR, rentabilidad, turnover, perfiles, costes y robustez posterior son informativos.
- 2025–2026 es `known_stress_not_selection` y no puede entrar en ninguna decisión.
- Toda feature, etiqueta y cohorte del meta debe ser point-in-time y estar cerrada.
- Los cinco agentes quality, value, growth, momentum y risk permanecen activos.
- SPY solo es benchmark y nunca una posición. Los umbrales de la cartera son económicos, en
  puntos básicos de alfa esperado, y una venta solo se emite si el destino del dinero (otra
  acción o efectivo) es mejor que la posición después de costes; las entradas tienen histéresis.
  La política de efectivo (`fully_invested` u `opportunity_cash`, remunerado al 0 %, con tope y
  suelo de diversificación) es una decisión de cartera, diagnóstica: no altera el Rank-IC y por
  tanto no puede elegir modelo.

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

- Cambios metodológicos: actualizar `docs/metodologia.md`.
- Decisiones, fallos, correcciones y ejecuciones: añadir entrada en `docs/bitacora.md`.
- Cifras reales: actualizar `docs/informe_resultados.md` indicando el artefacto de origen.
- No presentar un test sintético o smoke dev como evidencia económica.

## UTF-8

Todos los archivos se escriben en UTF-8. Cuidado con acentos y tildes. No introducir texto
recodificado, fragmentos de mojibake ni caracteres de sustitución.

## Validación

Antes de dar por terminado un cambio:

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
