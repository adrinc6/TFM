# Plan del TFM en LaTeX

> Documento vivo. Fija el índice completo del TFM y las convenciones de escritura **antes** de
> redactar ningún capítulo. Complementa a `docs/metodologia.md` (cómo se construye el sistema),
> `docs/bitacora.md` (el porqué de cada decisión) y los artefactos de `results/studies/<study_id>/`
> (las cifras trazables). Aquí se decide cómo se **cuenta** el proyecto, no cómo se construye. Se
> actualiza cada vez que se cierra un capítulo o cambia una decisión de estructura.
>
> **El manuscrito está congelado entre actualizaciones**: los cambios de código no lo editan, sino
> que se acumulan en `docs/plan_latex.md`. Ese fichero es el punto de partida de la próxima
> actualización: dice **qué** escribir y dónde; este plan cubre el **cómo** (formato, estructura y
> convenciones).

## Estado del proyecto a 2026-08-19

**El TFM pasa a redactarse sobre una cadena de cuatro estudios, no sobre uno solo.** Todos han
terminado y su evidencia está completa en disco.

| Campo | Valor |
|---|---|
| Model Study 1 | `study-20260816-182345-3cc1a5fb` · ganador `run-803c214fd2ac` |
| Model Study 2 | `study-20260817-021135-b5926b62` · ganador `run-95dcffb1640f` |
| **Model Study 3 (referencia)** | **`study-20260817-094411-568bd37e`** · ganador `run-1bb331e23a18` |
| **Portfolio Study** | **`study-20260817-212856-f86ca822`** · 1.440 carteras por Information Ratio |
| Hash de dataset | `b5db6211d03fcd56815afb316514215234e7300adf88f5a07f182071e5887cd9` |
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

**Aviso de procedencia (hay que declararlo en el TFM).** Las tres pasadas corrieron bajo el mismo
catálogo, así que la antigua limitación por versiones heterogéneas desaparece y **no se menciona el
versionado en el manuscrito**. Lo que sí hay que declarar es otra cosa: las pasadas 1 y 2 corrieron
con los diagnósticos posteriores al ganador desactivados, de modo que **no tienen** `robustness.json`
ni `attribution.json`. Ninguna tabla de robustez o atribución puede citarlas: solo el Model Study 3.
El Portfolio Study no reentrena nada, reutiliza los scores congelados del ganador del study 3.

### Los cinco resultados que vertebran el documento

Cifras de la **ventana de selección** (2015–2024) del Model Study 3, salvo indicación contraria. Las
cifras económicas son de la **cartera ganadora** del Portfolio Study. Cuidado al citar: los
artefactos también contienen la ventana completa (123 cohortes), donde el meta da 0,1033 y `risk`
0,1175.

1. **El sistema aprende.** ✅ El meta aprendido alcanza Rank-IC 0,1067 frente a 0,0690 de la
   ponderación ingenua (+55 %). *Matiz obligatorio:* `risk` por separado llega a 0,1201, es decir,
   **bate al meta**, y el propio meta acaba asignándole el grueso del peso (variante
   `stacked_rolling_free`, sin tope). La tesis multi-agente se matiza, no se da por demostrada.
2. **Lo aprendido es señal real.** ✅ Rank-IC 0,1067, IC-IR 0,833, $t$ de Newey-West 3,33 y 76,07 %
   de cohortes positivas. Neutralización de estilo: retiene 87,21 %.
3. **No es suerte.** ⚠️ **Se debilita.** La permutación sigue en $p=0{,}0001$, pero el Deflated
   Sharpe baja a **0,573** sobre 46 configuraciones. Encadenar estudios compra Rank-IC y encarece el
   DSR. *Ojo con la base:* el DSR penaliza por las configuraciones **predictivas**, no por la rejilla
   de cartera, que se aplica después sobre un modelo ya congelado.
4. **Bate al S&P 500 en la era reservada.** ⚠️ **Matizado:** con la cartera optimizada el exceso pasa
   a ser positivo (**+0,24 %**, IR **+0,109**, 1 de 2 años) frente al IR **−1,795** de la cartera del
   catálogo, que no batía ningún año. Es un **cambio de signo, no una superioridad demostrada**: en la
   era reservada la cartera empata con el índice. El Rank-IC de esa era es **+0,0364** en ambos casos
   (no depende de la cartera).
5. **El perfil `balanced` es el mejor.** ✅ en selección, con IR 0,841 frente a 0,342 del segundo
   (`value`). `balanced` **es** la cartera ganadora de la rejilla, no un perfil aparte. En la era
   reservada solo `balanced` y `momentum` conservan exceso positivo, mientras los estilos más
   marcados caen por debajo del −15 %: cuanto más se aparta un perfil de la ordenación aprendida, más
   depende del régimen.

**Hallazgo nuevo que atraviesa todo (2026-08-17).** El Rank-IC por era **sube** (0,0949 → 0,0543 →
0,1739) mientras el Information Ratio **cae** hasta volverse negativo (1,103 → 0,478 → −0,204). Sirve
para descartar que el sesgo de cobertura del panel fabrique la capacidad predictiva —es máxima donde
el panel está más completo— y **no** sirve para extender esa tranquilidad al resultado económico.
Las dos series se enuncian juntas o no se enuncian.

### El marco narrativo vigente: dos objetivos (2026-08-15)

**El TFM se cuenta como dos objetivos sucesivos, no como un hallazgo único.** El marco anterior
elevaba el contraste de carteras a «hallazgo central del TFM» y subordinaba a él la evidencia
predictiva. Se sustituye por éste, que es el que el autor considera correcto y el que responde a
cómo se ejecutó realmente el trabajo:

| | Objetivo | Lo demuestra | Respuesta |
|---|---|---|---|
| **1** | Un sistema de ML **aprende a ordenar acciones** con valor predictivo fuera de muestra | Los tres Model Studies | Sí, con matices |
| **2** | **Las variables de construcción de cartera afectan al resultado**, y optimizando por Information Ratio se construye una buena | El Portfolio Study | Sí |

La clave del marco: durante los tres Model Studies **la cartera es secundaria porque todavía no se
había optimizado** —se mantuvo la configuración por defecto del catálogo precisamente para que
ninguna decisión predictiva pudiera apoyarse en ella—. Por eso el IR de −1,795 de la era reservada con
esa cartera no es el fracaso del trabajo: es el punto de partida del Objetivo 2 y la medida de
cuánto depende el resultado de la gestión.

**Ninguna cifra cambia con este reenfoque.** Cambia qué se presenta como titular y se explicita el
objetivo que faltaba: el Portfolio Study se incorporó el 2026-08-14 y la introducción nunca se
actualizó —sus cinco objetivos operativos eran panel, agentes, meta, selección y auditoría, todos
del Objetivo 1—. El realineamiento afecta a `00_resumen.tex` (ES y EN), `01_introduccion.tex`
(dos objetivos, H1–H4 y sexto objetivo operativo), `07_resultados_economicos.tex`,
`09_conclusiones.tex` y `t01_afirmaciones.tex`. Los matices declarados se conservan sin excepción:
DSR 0,573, seis cohortes reservadas, la ganadora es la mejor de 1.440 y `risk` bate al meta.

---

Contraste que sostiene el Objetivo 2: el mismo modelo, la misma señal
y el mismo panel producen fuera de la ventana de decisión un IR de −1,795 o de +0,109 según
cómo se construya la cartera. La capacidad predictiva no era el cuello de botella: lo era su
traducción a posiciones, hasta el punto de decidir el signo del resultado. Se enuncia con la cautela
que imponen 6 cohortes cerradas y ~1,41 años de cartera, y declarando que la ganadora es la mejor de
1.440 evaluadas.

## Cómo se trabaja este plan

1. El autor pide un capítulo o una sección concreta.
2. Antes de escribir, se relee este plan, los capítulos `.tex` ya redactados (para mantener
   terminología y notación consistentes) y el estado real del proyecto: `docs/metodologia.md`,
   `docs/plan_latex.md`, el código, los tests y los artefactos del study de referencia.
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
| Granularidad | Capitulado clásico (9 capítulos más bibliografía y cuatro anexos, ver índice). |
| Referencias | **Autor-año escritas a mano** en `10_bibliografia.tex`. No se usa `biblatex` ni `biber`: la cadena impedía compilar. El proyecto no contiene ningún fichero `.bib`. |
| Figuras/tablas | Generadas por el script de exportación desde los artefactos del study, no dibujadas a mano. Se referencian por nombre suelto desde `latex/assets/`, junto a los capítulos. |
| Compilación | Subir `latex/` a Overleaf y seleccionar **XeLaTeX**. |
| Presentación de defensa | `latex/presentacion.tex`, Beamer 16:9 con XeLaTeX y tema propio. Vive **junto a `main.tex`**, no en una subcarpeta, para reutilizar las figuras con las mismas rutas `assets/` sin duplicar ningún PNG; en Overleaf sólo se cambia *Main document*. Guion hablado en `latex/guion_defensa.md`. |

**Pendiente de acordar** (no bloquea): portada oficial de la universidad (hay una provisional en
`main.tex`) y estructura definitiva de anexos.

## Estructura de carpetas dentro de `latex/`

```text
latex/
  plan_tfm.md          # este documento
  main.tex             # documento maestro: preámbulo + \input de cada capítulo
  assets/               # capítulos, tablas y figuras, todo suelto sin subcarpetas
    00_resumen.tex
    01_introduccion.tex
    02_estado_del_arte.tex
    03_datos_y_universo.tex          # universo + fuentes + observabilidad PIT + features
    04_agentes_y_meta_agente.tex     # arquitectura del sistema, sin resultados
    05_protocolo_experimental.tex    # selección predictiva + cartera + Portfolio Study
    06_resultados_predictivos.tex    # qué aprendió el sistema y si resiste contrastes
    07_resultados_economicos.tex     # rejilla, cartera ganadora y era reservada
    08_limitaciones.tex
    09_conclusiones.tex
    10_bibliografia.tex
    a_reproducibilidad.tex
    b_catalogo_protocolo.tex
    c_evidencia_complementaria.tex
    d_auditoria_desarrollo.tex       # historia del desarrollo, íntegra, fuera del hilo principal
    t*.tex               # cuerpos de tabla generados, \input desde los capítulos
    f*.png                # figuras generadas (no versionadas, ver .gitignore)
```

**Reestructuración narrativa (2026-08-14).** El documento se reordenó para que se lea como un relato
lineal —problema → datos observables → sistema → protocolo → resultados predictivos → resultados
económicos → límites → conclusión— sin que el lector tenga que avanzar y retroceder. Los cambios:

- Los antiguos capítulos de datos y de diseño PIT se fusionan en `03_datos_y_universo.tex`.
- El capítulo de agentes conserva sólo la arquitectura; su evidencia empírica pasa al capítulo de
  resultados predictivos, donde ya existe un protocolo que explique por qué esa configuración.
- El antiguo capítulo único de resultados se divide en dos, porque respondía a dos preguntas
  distintas: si la señal existe y si sobrevive al convertirse en cartera.
- Las dos historias del desarrollo se fusionan íntegras en el Anexo D. Se conservan enteras: sólo
  cambian de lugar, para no interrumpir la explicación del sistema final con errores ya corregidos.
- La tabla de las cinco afirmaciones se traslada de la introducción a las conclusiones. La
  introducción plantea las cinco *preguntas*; las respuestas llegan cuando hay evidencia detrás.

El preámbulo de `main.tex` fija paquetes, geometría, bibliografía y la notación compartida
(`\tsnap`, `\tfiled`, `\hlabel`, `\rankic`). La notación se define ahí una vez y los capítulos la
reutilizan sin redefinirla.

## Índice de capítulos

Todos los capítulos están escritos y viven en `latex/assets/`. **El manuscrito es la fuente de
verdad de su propio contenido**: este plan ya no reproduce el guion capítulo a capítulo, porque
mantener dos copias de las mismas cifras garantiza que una de ellas quede obsoleta. Lo que sí fija
este documento es qué estudios alimentan cada parte y qué activos deben existir.

| # | Fichero | Pregunta que responde | Evidencia que lo alimenta |
|---|---|---|---|
| 0 | `00_resumen.tex` | — | Todos; se redacta al final |
| 1 | `01_introduccion.tex` | ¿Qué se pregunta y por qué? | SPIVA Report 1a y planteamiento de los dos objetivos |
| 2 | `02_estado_del_arte.tex` | ¿Qué se sabe y qué puede fallar? | Bibliografía primaria ampliada, sin fichero `.bib` |
| 3 | `03_datos_y_universo.tex` | ¿Qué información existía y cuál fue la muestra efectiva? | `dataset_reference.json`, `universe_coverage`, `agent_scores`, features, tests |
| 4 | `04_agentes_y_meta_agente.tex` | ¿Cómo se genera la ordenación? | Arquitectura, objetivo ordinal y relojes temporales; sin resultados |
| 5 | `05_protocolo_experimental.tex` | ¿Cómo se decide y se calibra sin mirar la respuesta? | `decisions.json`, calibración, catálogo y diseño del Portfolio Study |
| 6 | `06_resultados_predictivos.tex` | ¿Existe la señal, qué usa y resiste? | `summary.json`, `robustness.json`, atribución local, estabilidad y factores |
| 7 | `07_resultados_economicos.tex` | ¿Sobrevive al convertirse en cartera? | `portfolio_grid`, `evidence_best_full/`, `portfolio_profiles` |
| 8 | `08_limitaciones.tex` | ¿Hasta dónde llega lo anterior? | `attribution.json`, `docs/bitacora.md` |
| 9 | `09_conclusiones.tex` | ¿Cuál es la respuesta? | `t01_afirmaciones.tex` (escrita a mano) |
| 10 | `10_bibliografia.tex` | — | Manual, sin biblatex |
| A-D | Anexos | Reproducibilidad, catálogo, evidencia complementaria y auditoría del desarrollo |  |

### Reparto de contenido entre los cuatro capítulos centrales

La frontera entre los capítulos 4 a 7 es una regla, no una preferencia de estilo: cada elemento
aparece **una sola vez**, en el capítulo cuya pregunta responde.

| Capítulo | Contiene | No contiene |
|---|---|---|
| 4 · Arquitectura | Agentes, bloques, walk-forward, alternativas del catálogo, ejemplo de combinación | Ningún valor ganador, ningún hiperparámetro elegido, ningún peso observado |
| 5 · Protocolo | Reglas de decisión, traza auditable (`t06_decisiones`), catálogo cerrado, diseño del Portfolio Study, ejemplos pareados | Interpretación de resultados; ninguna configuración de cartera presentada como «la final» |
| 6 · Evidencia predictiva | Cadena y configuración ganadora, meta y agentes, atribución local, Rank-IC, baselines, neutralización, robustez, Deflated Sharpe | Narrativa económica; cifras de cartera salvo las estrictamente diagnósticas y etiquetadas |
| 7 · Evidencia económica | Rejilla, cartera ganadora, tres ventanas, órdenes, transferencia, perfiles, era reservada | Contrastes predictivos, que ya se cerraron en el 6 |

Dos consecuencias operativas que conviene respetar al editar:

- **La era reservada se cuenta una sola vez**, en la Sección `sec:era-reservada` del capítulo 7,
  después de que el lector conozca cartera, selección y métricas. Las menciones anteriores son
  remisiones, no desarrollos.
- **Ningún capítulo se cita a sí mismo.** Para reenviar dentro del mismo capítulo se usa «más
  adelante en este capítulo» o la etiqueta de sección concreta.

### Regla de procedencia

La cadena y el Portfolio Study se **explican como procedimiento**, pero las cifras salen únicamente
de **los ganadores**:

- **Predictivo** (Rank-IC, agentes, meta, decisiones, robustez, atribución) → ganador del Model
  Study 3.
- **Económico** (equity, drawdown, órdenes, métricas anuales, tres ventanas, rejilla, perfiles) →
  ganador del Portfolio Study, leído de `evidence_best_full/`.
- **Cadena** (`t06_cadena*`, `f09_seleccion_vs_reservada`) → las tres pasadas.

**Ninguna cifra de la cartera del Model Study se presenta como resultado del trabajo**, porque esa
cartera se descartó al optimizarla. Por eso se eliminaron la sección de transferencia (el
coeficiente 0,328 describe la cartera por defecto) y la fila de confirmación de la regresión
factorial.

Dos excepciones declaradas, y ninguna más: el **Deflated Sharpe** y las **carteras aleatorias** se
calculan sobre la cartera del Model Study porque el Portfolio Study no reejecuta la batería de
robustez. Se conservan porque miden el procedimiento de búsqueda y no una cartera concreta, y
porque presentarlos con la cartera menos favorable es conservador: no inflan ninguna afirmación.

### Inventario de activos (2026-08-19)

**21 tablas y 22 figuras.** Un activo se conserva si permite consultar valores exactos o reconocer
una relación que la prosa no comunica con igual claridad. No se duplica tabla y gráfico cuando
cuentan lo mismo; el diagrama anual de turnover es la excepción deliberada porque muestra una
relación que la tabla anual no contiene.

La migración visual incorpora siete figuras:

- `f01_spiva_horizontes.png`: dificultad competitiva antes de formular los objetivos.
- `f03_cobertura_anual.png`: cobertura canónica y ventanas del experimento; sustituye a
  `t03_cobertura_anual.tex`.
- `f07_alpha_turnover_anual.png`: relación entre rotación y exceso anual.
- `f07_costes_escalera.png`: rutas congelada y resimulada frente al coste.
- `f07_capacidad.png`: participación P95 y umbrales de patrimonio, solo en selección.
- `f07_perfiles_pesos.png`: pesos firmados, con `balanced` como meta puro.
- `f07_perfiles_resultados.png`: exceso agregado y anual de los perfiles, con reserva separada.

La ampliación explicativa posterior incorpora cuatro figuras y tres tablas generadas:

- `f03_muestra_oos.png` y `t03_muestra_oos.tex`: cohorte realmente puntuada, filas de
  entrenamiento y disponibilidad de variables, sin confundirlas con cobertura del índice.
- `f05_calibracion_alfa.png` y `t05_calibracion_estados.tex`: estados causales de calibración y
  traducción del ranking a alfa esperado.
- `f06_estabilidad_features.png`: persistencia anual de las variables más usadas por agente.
- `f06_atribucion_factorial.png`: cargas con incertidumbre y señal conservada tras neutralización.
- `tB_diccionario_features.tex`: diccionario auditable de las 68 variables declaradas.

Las tablas de catálogo, decisiones, métricas completas, limitaciones y auditoría siguen siendo
tablas porque allí importa localizar una cifra. Los diagramas temporales y de arquitectura se
mantienen porque explican mecanismos, no resultados.

`latex/asset_manifest.json` registra por separado estudios, panel, ocho series anuales de perfiles
y la fuente bibliográfica SPIVA. El comando de regeneración permanece en
`assets/a_reproducibilidad.tex`; `--audit` comprueba activos y procedencia obligatoria.

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
