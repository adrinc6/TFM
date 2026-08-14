# Plan del TFM en LaTeX

> Documento vivo. Fija el índice completo del TFM y las convenciones de escritura **antes** de
> redactar ningún capítulo. Complementa a `docs/metodologia.md` (cómo se construye el sistema),
> `docs/bitacora.md` (el porqué de cada decisión) y `docs/informe_resultados.md` (las cifras
> trazables). Aquí se decide cómo se **cuenta** el proyecto, no cómo se construye. Se actualiza
> cada vez que se cierra un capítulo o cambia una decisión de estructura.

## Estado del proyecto a 2026-08-14

**El TFM pasa a redactarse sobre una cadena de cuatro estudios, no sobre uno solo.** Todos han
terminado y su evidencia está completa en disco.

| Campo | Valor |
|---|---|
| Model Study 1 | `study-20260812-163136-1b104667` · ganador `run-6eaa47a0597b` · catálogo v6 |
| Model Study 2 | `study-20260813-103456-aa733655` · ganador `run-2dc586be8653` · catálogo v6 |
| **Model Study 3 (referencia)** | **`study-20260814-095144-5ec17b78`** · ganador `run-f134d7eb9e06` · catálogo **v7** |
| **Portfolio Study** | **`study-20260814-135754-fdbdf2c5`** · 1.728 carteras por Information Ratio |
| Hash de dataset | `b9134b218e3bf7fc156372d61e02056ecfa6036777e0fe84a69df0a92653fbd3` |
| Selección | 2015–2024, 117 cohortes mensuales, sólo Rank-IC pareado |
| Era reservada | 2025–2026 (6 cohortes, hasta 2025-06-29), sin participación en ninguna decisión |

**Decisión de 2026-08-14 que levanta la regla dura anterior.** El plan exigía redactar el TFM
«exclusivamente» con `study-20260803-201234-b4d7a8d8` y registrar aquí y en la bitácora cualquier
cambio. Se registra: **ese study queda fuera del documento y lo sustituye la cadena**. El motivo no
es cosmético. El study antiguo corría bajo el catálogo v5, anterior a la corrección de
`decide_orders`, y su sustituto no solo está bajo catálogo vigente sino que responde a una pregunta
más honesta —qué pasa cuando se optimiza de verdad y se mide fuera de la ventana de decisión—. La
regla que la sustituye:

> **Regla dura vigente:** el TFM se redacta con los cuatro estudios de la tabla y con ninguno más.
> La evidencia **predictiva** sale del Model Study 3; la **económica**, del ganador del Portfolio
> Study. Ninguna cifra del study `b4d7a8d8` sobrevive en el documento.

**Aviso de reproducibilidad (hay que declararlo en el TFM).** La cadena no corrió entera bajo la
misma versión de catálogo: los studies 1 y 2 usaron `CATALOG_VERSION` 6 y el study 3 la 7, que
invierte el desempate por simplicidad de `execution_lag_days`. Los tres son evidencia válida y
trazable —los hashes de dataset y de evaluación lo acreditan—, pero las tablas deben **citar la
versión de catálogo junto al `study_id`** y el capítulo de limitaciones debe recogerlo. El Portfolio
Study no reentrena nada: reutiliza los scores congelados del ganador del study 3.

### Los cinco resultados que vertebran el documento

Cifras de la **ventana de selección** (2015–2024) del Model Study 3, salvo indicación contraria. Las
cifras económicas son de la **cartera ganadora** del Portfolio Study. Cuidado al citar: los
artefactos también contienen la ventana completa (123 cohortes), donde el meta da 0,1058 y `risk`
0,1197.

1. **El sistema aprende.** ✅ El meta aprendido alcanza Rank-IC 0,1090 frente a 0,0675 de la
   ponderación ingenua (+61 %). *Matiz obligatorio:* `risk` por separado llega a 0,1227, es decir,
   **bate al meta**, y el propio meta acaba asignándole **más del 95 %** del peso (variante
   `stacked_rolling_free`, sin tope). La tesis multi-agente se matiza, no se da por demostrada.
2. **Lo aprendido es señal real.** ✅ Rank-IC 0,1090, IC-IR 0,851, $t$ de Newey-West 3,46 y 74,36 %
   de cohortes positivas. Neutralización de estilo: retiene 86,62 %.
3. **No es suerte.** ⚠️ **Se debilita.** La permutación sigue en $p=0{,}0001$, pero las carteras
   aleatorias del escenario general dejan al modelo en el percentil 0,761 —no supera el umbral de
   0,95— y el Deflated Sharpe baja a 0,682. Encadenar estudios compra Rank-IC y encarece el DSR.
4. **Bate al S&P 500 en la era reservada.** ✅ **pero sólo con la cartera optimizada.** Con ella:
   exceso **+2,56 %**, IR **+0,304**, 1 de 2 años. Con la cartera del catálogo: **−11,29 %**, IR
   **−1,167**, 0 de 2 años. El Rank-IC de esa era es **+0,0441** en ambos casos (no depende de la
   cartera).
5. **El perfil `balanced` es el mejor.** ✅ en selección, con IR 0,844 frente a 0,570 del segundo
   (`defensive`). ⚠️ En la era reservada el orden se invierte casi por completo —`momentum`, el peor
   en selección, es el mejor allí con IR 1,889— sobre 6 cohortes: es régimen, no hallazgo.

**El hallazgo central del TFM es el contraste de la afirmación 4.** El mismo modelo, la misma señal
y el mismo panel producen fuera de la ventana de decisión un exceso de −11,29 % o de +2,56 % según
cómo se construya la cartera. La capacidad predictiva no era el cuello de botella: lo era su
traducción a posiciones, hasta el punto de decidir el signo del resultado. Se enuncia con la cautela
que imponen 6 cohortes cerradas y ~1,41 años de cartera, y declarando que la ganadora es la mejor de
1.728 evaluadas.

## Cómo se trabaja este plan

1. El autor pide un capítulo o una sección concreta.
2. Antes de escribir, se relee este plan, los capítulos `.tex` ya redactados (para mantener
   terminología y notación consistentes) y el estado real del proyecto: `docs/metodologia.md`,
   `docs/informe_resultados.md`, el código, los tests y los artefactos del study de referencia.
3. Un capítulo solo se escribe con datos y resultados que existen. **Ninguna cifra entra en el `.tex`
   sin que se pueda señalar el artefacto exacto de donde sale.** Nada de cifras inventadas ni
   redondeadas «de memoria».
4. Un `.tex` por capítulo, en `latex/assets/`, pensado para pegar directamente en Overleaf.
5. Al cerrar un capítulo, se actualiza la tabla de estado de este documento.

## Decisiones de formato (acordadas)

| Tema | Decisión |
|---|---|
| Plantilla | Estructura académica libre, no hay plantilla obligatoria del máster. |
| Idioma | Español. |
| Motor LaTeX | **XeLaTeX**, sin biber. UTF-8 nativo (sin `inputenc`), fuentes con `fontspec`. En Overleaf: Menu > Compiler: XeLaTeX. |
| Granularidad | Capitulado clásico (9 capítulos, ver índice). |
| Referencias | **Autor-año escritas a mano** en `12_bibliografia.tex`. No se usa `biblatex` ni `biber`: la cadena impedía compilar. `referencias.bib` se conserva como registro de los datos completos. |
| Figuras/tablas | Generadas por el script de exportación desde los artefactos del study, no dibujadas a mano. Se referencian por nombre suelto desde `latex/assets/`, junto a los capítulos. |
| Compilación | Subir `latex/` a Overleaf y seleccionar **XeLaTeX**. |

**Pendiente de acordar** (no bloquea): portada oficial de la universidad (hay una provisional en
`main.tex`) y estructura definitiva de anexos.

## Estructura de carpetas dentro de `latex/`

```text
latex/
  plan_tfm.md          # este documento
  main.tex             # documento maestro: preámbulo + \input de cada capítulo
  referencias.bib
  assets/               # capítulos, tablas y figuras, todo suelto sin subcarpetas
    00_resumen.tex
    01_introduccion.tex
    02_estado_del_arte.tex
    03_datos_y_universo.tex
    04_diseno_metodologico.tex
    05_desarrollo_metodo.tex       # historia del desarrollo (I): estudio único y contratos
    06_agentes_y_meta_agente.tex
    07_diseno_experimental.tex
    08_desarrollo_cartera.tex      # historia del desarrollo (II): cartera y reproducibilidad
    09_resultados.tex
    10_limitaciones.tex
    11_conclusiones.tex
    12_bibliografia.tex
    a_reproducibilidad.tex
    b_catalogo_protocolo.tex
    c_evidencia_complementaria.tex
    t*.tex               # cuerpos de tabla generados, \input desde los capítulos
    f*.png                # figuras generadas (no versionadas, ver .gitignore)
  # Nota (2026-08): el capítulo de desarrollo se dividió en dos y el resto de ficheros se
  # renumeró en cascada para que el prefijo vuelva a documentar el orden real de lectura;
  # capítulos, tablas y figuras se unificaron en assets/ sin subcarpetas. Ver docs/bitacora.md.
```

El preámbulo de `main.tex` fija paquetes, geometría, bibliografía y la notación compartida
(`\tsnap`, `\tfiled`, `\hlabel`, `\rankic`). La notación se define ahí una vez y los capítulos la
reutilizan sin redefinirla.

## Índice de capítulos

Todos los capítulos están escritos y viven en `latex/assets/`. **El manuscrito es la fuente de
verdad de su propio contenido**: este plan ya no reproduce el guion capítulo a capítulo, porque
mantener dos copias de las mismas cifras garantiza que una de ellas quede obsoleta. Lo que sí fija
este documento es qué estudios alimentan cada parte y qué activos deben existir.

| # | Fichero | Evidencia que lo alimenta |
|---|---|---|
| 0 | `00_resumen.tex` | Todos; se redacta al final |
| 1 | `01_introduccion.tex` | `t01_afirmaciones.tex` (escrita a mano) |
| 2 | `02_estado_del_arte.tex` | `referencias.bib` (bibliografía manual) |
| 3 | `03_datos_y_universo.tex` | `dataset_reference.json`, `universe_coverage`, features |
| 4 | `04_diseno_metodologico.tex` | `docs/metodologia.md`, tests de contrato |
| 5 | `05_desarrollo_metodo.tex` | Bitácora + **cadena de studies** (`t06_cadena`, `f06_cadena_*`) |
| 6 | `06_agentes_y_meta_agente.tex` | `meta_weights.parquet`, `rank_ic_diagnostics.parquet` |
| 7 | `07_diseno_experimental.tex` | `decisions.json`, `config.json` |
| 8 | `08_desarrollo_cartera.tex` | Bitácora + diseño del Portfolio Study |
| 9 | `09_resultados.tex` | `summary.json`, `robustness.json`, `attribution.json`, **rejilla de cartera** |
| 10 | `10_limitaciones.tex` | `attribution.json`, `docs/bitacora.md` |
| 11 | `11_conclusiones.tex` | Todo |
| 12 | `12_bibliografia.tex` | Manual, sin biblatex |
| A-C | Anexos | Reproducibilidad, catálogo y evidencia complementaria |

### Regla de procedencia

Cada activo declara de qué estudio sale, y esa separación es la que permite documentar el modelo del
study 3 con la cartera del Portfolio Study sin mezclar procedencias:

- **Predictivo** (Rank-IC, agentes, meta, decisiones, robustez, atribución) → Model Study 3.
- **Económico** (equity, órdenes, métricas anuales, rejilla, perfiles) → ganador del Portfolio Study,
  leído de `evidence_best_full/`.
- **Cadena** (`t06_cadena*`, `f06_cadena_*`, `f09_seleccion_vs_reservada`) → las tres pasadas.

El comando de regeneración está en `assets/a_reproducibilidad.tex` y el manifiesto resultante,
`latex/asset_manifest.json`, registra ambas familias de fuentes por separado.

## Convenciones de escritura

- Notación consistente entre capítulos: snapshot \(\tsnap\), publicación \(\tfiled\), horizonte
  \(\hlabel\), coeficiente de información por rango \(\rankic\). Se fija en el Capítulo 4 y se
  reutiliza sin redefinir.
- **Toda cifra citada debe poder trazarse a un artefacto real del repositorio** (parquet, json,
  figura). Igual que el código: nada sin fuente verificable.
- Términos en inglés sin traducción asentada (*lookahead bias*, *walk-forward*, *rank-IC*) en cursiva
  la primera vez y así el resto del documento.
- **Cada resultado positivo va acompañado de su matiz** en el mismo párrafo o en el inmediatamente
  siguiente. El valor del trabajo está en la honestidad de la medición, no en el tamaño del alfa: un
  tribunal premia el matiz declarado y penaliza el matiz encontrado.
- Distinguir siempre y explícitamente el papel de cada cifra: **selección**, **confirmación fuera de
  muestra** o **diagnóstico**. Nunca mezclar los tres en una misma tabla sin etiquetarlos.
- Los decimales en español con coma; los identificadores, rutas y nombres de variables en
  `\texttt{}` y sin traducir.
