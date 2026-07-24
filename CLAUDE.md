# Guía del repositorio

## Propósito

Este TFM construye y corrobora hipótesis de inversión con datos point-in-time.

```text
data/raw → Exploratory Study → hipótesis congelada → Confirmatory Study → evidencia final
```

Antes de realizar cambios, leer `docs/metodologia.md`, `docs/bitacora.md`,
`docs/informe_resultados.md` y `AGENTS.md`. Son las fuentes de verdad del proyecto y del futuro TFM.

## Arquitectura

- `module/data`: ingesta, universo y panel PIT.
- `module/modeling`: features, agentes y meta-agente.
- `module/evaluation`: cartera y métricas.
- `module/studies`: catálogo, runner, exploración y confirmación.
- `module/storage`: datasets, caché y evidencia.
- `module/web` y `app`: API y dashboard.

No se deben crear scenarios, experiments, runs sueltos ni rutas alternativas.

## Reglas

1. Todo parámetro científico debe pertenecer al catálogo cerrado.
2. No introducir lookahead.
3. 2025–2026 es estrés conocido y no participa en selección.
4. Confirmatory no modifica una hipótesis congelada.
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
