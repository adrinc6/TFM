# Guía del repositorio

## Propósito

Este TFM estudia si cinco agentes especializados aprenden una ordenación transversal de
acciones que se mantenga fuera de muestra, con datos point-in-time.

```text
catálogo cerrado → Model Study → optimización secuencial por Rank-IC → ganador
                                → robustez, carteras y perfiles informativos → informe
```

Antes de realizar cambios, leer `docs/metodologia.md`, `docs/bitacora.md`,
`docs/informe_resultados.md` y `AGENTS.md`. Son las fuentes de verdad del proyecto y del futuro TFM.

`docs/gestion_cartera.md` es la referencia operativa de la cartera: variables, orden de decisión,
casuísticas y ejemplos. Es donde el usuario anota los cambios que quiere en las reglas de cartera;
si su sección «Cambios pedidos» no está vacía, hay trabajo pendiente que trasladar al código.

## Arquitectura

- `module/data`: ingesta, universo y panel PIT.
- `module/modeling`: features, agentes y meta-agente.
- `module/evaluation`: cartera y métricas.
- `module/studies`: catálogo, runner y selección secuencial del único Model Study.
- `module/storage`: datasets, caché y evidencia.
- `module/web` y `app`: API y dashboard.

No se deben crear scenarios, experiments, runs sueltos ni rutas alternativas.

## Reglas

1. Todo parámetro científico debe pertenecer al catálogo cerrado.
2. No introducir lookahead.
3. 2025–2026 es estrés conocido y no participa en selección.
4. Solo las fases predictivas (temporal, representación, modelo, meta) pueden modificar el ganador.
5. Los descartados guardan solo resúmenes.
6. Mantener UTF-8 y evitar mojibake.
7. Preferir funciones pequeñas y flujos explícitos.
8. Actualizar metodología, bitácora e informe cuando cambien decisiones o evidencia.
9. No afirmar resultados financieros sin vincularlos a artefactos reales.

## Verificación

```powershell
python -m pytest -q
python -m ruff check .
node --check app/js/app.js
```

No ejecutar un estudio real completo sin autorización explícita.
