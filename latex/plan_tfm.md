# Plan del TFM en LaTeX

> Documento vivo. Fija el índice completo del TFM y las convenciones de escritura **antes** de
> redactar ningún capítulo. Complementa a `docs/doc.md` (el plan maestro del sistema) y a
> `docs/plan_fases.md` (el detalle ejecutable por fase de implementación): aquí se decide cómo
> se **cuenta** el proyecto, no cómo se construye. Se actualiza cada vez que se cierra un
> capítulo o cambia una decisión de estructura.

## Cómo se trabaja este plan

1. El autor pide un capítulo o una sección concreta.
2. Antes de escribir, se relee este plan, los capítulos `.tex` ya redactados (para mantener
   terminología y notación consistentes) y el estado real del proyecto en ese momento —
   `docs/plan_fases.md`, el código, los tests, y los resultados de las fases ya cerradas.
3. Un capítulo solo se escribe con datos y resultados que existen. Si una fase que el capítulo
   necesita todavía no se ha implementado, el capítulo se deja marcado como pendiente o se
   escribe su esqueleto metodológico sin resultados, nunca con cifras inventadas.
4. Un `.tex` por capítulo, en `latex/`, pensado para pegar directamente en Overleaf. Nombres de
   fichero: `caps/01_introduccion.tex`, `caps/02_estado_del_arte.tex`, etc. (ver estructura de
   carpetas más abajo).
5. Al cerrar un capítulo, se actualiza la tabla de estado de este documento.

## Decisiones de formato (acordadas)

| Tema | Decisión |
|---|---|
| Plantilla | Estructura académica libre, no hay plantilla obligatoria del máster. |
| Idioma | Español. |
| Motor LaTeX | **XeLaTeX** + `biber`. UTF-8 nativo (sin `inputenc`), fuentes con `fontspec`. En Overleaf: Menu > Compiler: XeLaTeX. |
| Granularidad | Capitulado clásico (9 capítulos, ver índice). |
| Referencias | **Autor-año** con `biblatex` (`style=authoryear`, `backend=biber`), un único `referencias.bib`. Se evita el estilo `apa` (paquete `biblatex-apa`) por ser pesado y lento en el plan gratuito de Overleaf. |
| Figuras/tablas | Generadas por el pipeline (Research Console, `module/ui/report.py` y exports de runs), no dibujadas a mano. Se referencian por ruta desde `latex/figuras/`. |
| Compilación | Subir `latex/` a Overleaf y seleccionar **XeLaTeX** como compilador. |

**Pendiente de acordar cuando se llegue ahí** (no bloquea): portada oficial de la universidad
(hay una provisional en `main.tex`) y estructura de anexos.

## Estructura de carpetas dentro de `latex/`

```text
latex/
  plan_tfm.md          # este documento
  main.tex             # documento maestro: preámbulo + \input de cada capítulo
  referencias.bib
  caps/
    01_introduccion.tex
    02_estado_del_arte.tex
    03_datos_y_universo.tex
    04_diseno_metodologico.tex
    05_agentes_y_meta_agente.tex
    06_diseno_experimental.tex
    07_resultados.tex
    08_limitaciones.tex
    09_conclusiones.tex
  figuras/              # imágenes/tablas exportadas, referenciadas desde los capítulos
```

El preámbulo de `main.tex` fija los paquetes, la geometría, la bibliografía APA (`biblatex`) y la
notación compartida (`\tsnap`, `\tfiled`, `\hlabel`, `\rankic`). La notación se define ahí una vez
y los capítulos la reutilizan sin redefinirla.

## Índice de capítulos

| # | Capítulo | Depende de (fase del proyecto) | Estado |
|---|---|---|---|
| 1 | Introducción y motivación | — | **Borrador escrito** |
| 2 | Estado del arte | — | **Borrador escrito** (refs APA sembradas en `referencias.bib`, verificar DOIs) |
| 3 | Datos y universo de inversión | Fase 0 | **Borrador escrito** |
| 4 | Diseño metodológico: point-in-time y ausencia de lookahead | Fases 0-1 | **Borrador escrito** |
| 5 | Agentes especializados y meta-agente | Fases 2-3 | **Borrador escrito** |
| 6 | Diseño experimental: cartera, backtest y rejilla de escenarios | Fases 4-6 | Esqueleto (método sin cifras; a la espera de la reejecución) |
| 7 | Resultados | Fases 4-6 | Esqueleto (bloqueado; a la espera de la reejecución) |
| 8 | Limitaciones y amenazas a la validez | Todas | **Borrador escrito** (se cierra con los resultados) |
| 9 | Conclusiones y trabajo futuro | Todas | Esqueleto (bloqueado; necesita resultados) |

### 1. Introducción y motivación

Por qué IA/ML aplicado a bolsa como banco de pruebas de aprendizaje, no como objetivo de
rentabilidad (ver `CLAUDE.md`, "Propósito del proyecto"). Pregunta de investigación: ¿puede un
sistema de agentes aprender señales con valor predictivo fuera de muestra, medido con rigor
point-in-time, y ese aprendizaje se traduce en utilidad económica neta de costes? Objetivos
del TFM y de qué manera un resultado negativo bien medido también es un resultado válido.

### 2. Estado del arte

Factor investing clásico (GARP, momentum, calidad) como listón de comparación. Aprendizaje
automático aplicado a selección de activos: qué se ha probado, qué problemas metodológicos
son recurrentes en la literatura (lookahead bias, sesgo de supervivencia, overfitting a pocas
eras de mercado). Por qué la separación temporal estricta y el rank-IC OOS son el criterio de
evidencia elegido frente a solo reportar rentabilidad.

### 3. Datos y universo de inversión

Fuentes (Finnhub, Yahoo, SEC EDGAR) y por qué cada una — ver `docs/plan_fases.md` (Fase 0,
"Hallazgos verificados"). Universo dinámico por fecha desde la composición histórica real del
S&P 500 (`module/data/universe.py`): qué sesgo elimina (inclusión anticipada) y cuál quedaría
igualmente si se usara un índice actual. Guarda de reciclaje de ticker, con los casos reales
(`CPQ`, `MOB`) como ilustración. Sesgo de supervivencia **medido** por año, no solo declarado
(`universe_coverage.json`).

### 4. Diseño metodológico: point-in-time y ausencia de lookahead

La regla central del proyecto (año + trimestre + `lag_days` como margen de ejecución, nunca
como retardo aplicado al dato) y por qué el diseño alternativo —un retardo fijo por
fundamental— es metodológicamente incorrecto, con los contraejemplos reales que lo
demostraron (AT&T 133 días, un 10-K de AAPL 88 días). Algoritmo de observabilidad
(`module/data/dataset.py`): fecha de publicación real vía SEC EDGAR frente a fecha de cierre fiscal.
Qué se prohíbe explícitamente (`payload.metric`, columnas de `profiles.parquet`, `sector`) y
por qué. Estrategia de tests de fuga temporal como parte del método, no como añadido.

### 5. Agentes especializados y meta-agente

Factores por agente (calidad, momentum, valor) y su justificación económica. Modelo Ridge por
agente con entrenamiento walk-forward expandible y luego con ventana móvil de
`TRAIN_LOOKBACK_YEARS`. Meta-agente: ponderación por rank-IC OOS reciente, no por alfa. Rank-IC
como evidencia de aprendizaje, reportado separado de cualquier medida de rentabilidad.

### 6. Diseño experimental: cartera, backtest y rejilla de escenarios

*(Bloqueado hasta que existan las Fases 4-6.)* Construcción de cartera (top-N, rotación,
umbral de ventaja), simulación con costes y slippage, métricas de backtest. Rejilla de
escenarios y criterio de selección por estabilidad multi-era.

### 7. Resultados

*(Bloqueado.)* Resultados frente a los baselines (índice, momentum puro, GARP determinista).
Rank-IC por agente y por era. Alfa neta, information ratio, drawdown. Resultados desagregados
por periodo de cobertura (pre/post 2009-2010, coherente con las limitaciones de la Fase 0).

### 8. Limitaciones y amenazas a la validez

Sesgo de supervivencia residual medido por año (no eliminable con fuentes gratuitas).
Restatements invisibles antes de 2009 (Finnhub da el valor actual, EDGAR solo fecha la
publicación). Cobertura desigual de métricas y por qué no se imputan con la media. Tamaño de
muestra pequeño en número de eras independientes. Este capítulo puede empezar a escribirse en
paralelo a las fases restantes —cada limitación ya está identificada y en parte medida— y se
cierra con las que aparezcan en las Fases 4-6.

### 9. Conclusiones y trabajo futuro

*(Bloqueado, necesita resultados.)* Qué aprendió el sistema y qué no. Honestidad sobre
significancia estadística dado el tamaño de muestra. Vías de mejora ya identificadas durante
el proyecto (p. ej., EDGAR `companyfacts` si el TFM se reorientase a un ancla ≥2009; fuente de
sector histórico).

## Convenciones de escritura

- Notación matemática consistente entre capítulos: fecha de snapshot como \(t\), fecha de
  publicación como \(t_{\text{filed}}\), horizonte de la etiqueta como \(h\). Se fija la
  primera vez que aparece (Capítulo 4) y se reutiliza sin redefinir.
- Cuando un capítulo cite un resultado, debe poder trazarse a un artefacto real del repositorio
  (parquet, json, figura) — igual que el código, nada de cifras sin fuente verificable.
- Términos en inglés sin traducción asentada en español (*lookahead bias*, *walk-forward*,
  *rank-IC*) se usan en cursiva la primera vez y se mantienen así el resto del documento.
