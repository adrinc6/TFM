# Guía del repositorio

## Propósito

Este TFM estudia si cinco agentes especializados aprenden una ordenación transversal de
acciones que se mantenga fuera de muestra, con datos point-in-time.

```text
catálogo cerrado → Model Study → optimización secuencial por Rank-IC → ganador
                                → robustez, carteras y perfiles informativos → informe
```

Antes de realizar cambios, leer `docs/metodologia.md`, `docs/bitacora.md` y `AGENTS.md`. Son las
fuentes de verdad del proyecto y del futuro TFM.

## Documentación: tres ficheros y ninguno más

| Fichero | Qué contiene |
|---|---|
| `docs/metodologia.md` | El **cómo**, en profundidad: universo y datos PIT, optimización secuencial, estudios encadenados, Portfolio Study, doctrina operativa de cartera, robustez y atribución. |
| `docs/bitacora.md` | La **agenda**: qué se decidió, qué falló, qué se corrigió y qué se ejecutó, en orden cronológico. |
| `docs/plan_latex.md` | El **plan del manuscrito**: qué hay que escribir en el LaTeX, dónde y con qué artefacto detrás, más la deuda pendiente y el trabajo planificado. |

Las cifras **no viven en ningún documento**: viven en los artefactos de
`results/studies/<study_id>/` y se leen de ahí. Duplicarlas en un `.md` crea una segunda verdad que
se desincroniza.

## El manuscrito LaTeX está congelado

`latex/main.tex`, `latex/presentacion.tex` y `latex/assets/*.tex` **no se editan** como parte de un
cambio, ni se regeneran sus figuras y tablas. En su lugar se añade una entrada a la sección «Deuda
nueva» de `docs/plan_latex.md` explicando qué cambió, a qué capítulos, tablas o figuras afecta y qué
artefacto lo respalda. Cuando el usuario ordene la actualización, ese fichero es el contexto de
partida.

Sí son editables `latex/scripts/*.py` (es código) y `latex/plan_tfm.md` (es documentación, cubre
formato y convenciones), pero el exportador **no se ejecuta** como parte del cambio. Entre
actualizaciones el manuscrito está desactualizado a propósito: `docs/plan_latex.md` es el registro de
esa deuda.

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
8. Actualizar metodología y bitácora cuando cambien decisiones o evidencia; anotar en
   `docs/plan_latex.md` lo que el manuscrito tendrá que recoger.
9. No afirmar resultados financieros sin vincularlos a artefactos reales, citando `study_id` y ruta (aunque en informe y presentacion no hay que citarlos, pero si comprobar que los resultados existen en los artefactos y no se usan valores antiguos).

## Verificación

```powershell
python -m pytest -q
python -m ruff check .
node --check app/js/app.js
```

No ejecutar un estudio real completo sin autorización explícita.
