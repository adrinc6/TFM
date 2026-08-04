# Plan del TFM en LaTeX

> Documento vivo. Fija el índice completo del TFM y las convenciones de escritura **antes** de
> redactar ningún capítulo. Complementa a `docs/metodologia.md` (cómo se construye el sistema),
> `docs/bitacora.md` (el porqué de cada decisión) y `docs/informe_resultados.md` (las cifras
> trazables). Aquí se decide cómo se **cuenta** el proyecto, no cómo se construye. Se actualiza
> cada vez que se cierra un capítulo o cambia una decisión de estructura.

## Estado del proyecto a 2026-08-04

**El estudio de referencia está cerrado y con evidencia completa.** Ya no hay ningún capítulo
bloqueado por falta de resultados: todos los datos necesarios para redactar el TFM entero existen
en disco.

| Campo | Valor |
|---|---|
| Study de referencia | `study-20260803-201234-b4d7a8d8` |
| Run ganador | `run-51e95a09a8f0` |
| Hash de dataset | `b9134b218e3bf7fc156372d61e02056ecfa6036777e0fe84a69df0a92653fbd3` |
| Configuraciones evaluadas | 66 |
| Selección | 2015–2024, 117 cohortes mensuales, sólo Rank-IC pareado |
| Era reservada | 2025–2026, sin participación en ninguna decisión |

**Regla dura de este plan:** el TFM se redacta **exclusivamente** con este study. Cualquier otro
estudio en `results/studies/` —anterior o posterior— queda fuera del documento. Si se ejecuta un
estudio nuevo, no entra en el TFM sin una decisión explícita registrada aquí y en la bitácora.

**Aviso de reproducibilidad (hay que declararlo en el TFM).** Este study se ejecutó con
`CATALOG_VERSION` 5; la vigente es 6 desde la corrección de `decide_orders` del 2026-08-04
(`docs/bitacora.md`). Sus artefactos siguen siendo evidencia válida y trazable —los hashes de
dataset y de evaluación lo acreditan—, pero **no es reproducible bit a bit con el código actual** y
no es comparable con estudios bajo el catálogo v6. El Capítulo 8 (Limitaciones) debe recogerlo, y
las tablas de resultados deben citar la versión de catálogo junto al `study_id`. Reejecutar para
«actualizar» las cifras cambiaría el ganador y obligaría a reescribir los Capítulos 5 a 7 enteros:
la decisión tomada es **congelar este study como referencia del documento**.

### Los cinco resultados que vertebran el documento

Todo el TFM se ordena alrededor de estas cinco afirmaciones, cada una con artefacto que la respalda:

1. **El sistema aprende.** El meta-agente parte de pesos iguales (0,20) y descubre solo, en dos años,
   a su mejor especialista, llegando al tope de 0,50. La ponderación aprendida bate a la ingenua por
   +0,0345 de rank-IC (+52 %).
2. **Lo aprendido es señal real.** Rank-IC 0,1004, IC-IR 0,744, t de Newey-West 3,02, 71,79 % de
   cohortes positivas, frente a un mejor baseline determinista de 0,0130.
3. **No es suerte.** Siete contrastes independientes de robustez superados (permutación p = 0,0001,
   placebos, bootstrap, exclusión de eras, semillas, carteras aleatorias con riesgo emparejado,
   neutralización por estilo) y uno no superado que se reporta igual (Deflated Sharpe 0,930).
4. **Bate al S&P 500 en los años reservados.** 2/2 años, +14,21 % de exceso geométrico, IR 0,959,
   alfa factorial con t = 4,76, sobre datos que no tocaron ninguna decisión.
5. **El perfil `balanced` es el mejor.** Gana en CAGR, exceso, IR, beat rate y alfa medio a la vez;
   es el único que no reordena la señal, y seis de los siete perfiles que sí la reordenan destruyen
   alfa.

## Cómo se trabaja este plan

1. El autor pide un capítulo o una sección concreta.
2. Antes de escribir, se relee este plan, los capítulos `.tex` ya redactados (para mantener
   terminología y notación consistentes) y el estado real del proyecto: `docs/metodologia.md`,
   `docs/informe_resultados.md`, el código, los tests y los artefactos del study de referencia.
3. Un capítulo solo se escribe con datos y resultados que existen. **Ninguna cifra entra en el `.tex`
   sin que se pueda señalar el artefacto exacto de donde sale.** Nada de cifras inventadas ni
   redondeadas «de memoria».
4. Un `.tex` por capítulo, en `latex/caps/`, pensado para pegar directamente en Overleaf.
5. Al cerrar un capítulo, se actualiza la tabla de estado de este documento.

## Decisiones de formato (acordadas)

| Tema | Decisión |
|---|---|
| Plantilla | Estructura académica libre, no hay plantilla obligatoria del máster. |
| Idioma | Español. |
| Motor LaTeX | **XeLaTeX** + `biber`. UTF-8 nativo (sin `inputenc`), fuentes con `fontspec`. En Overleaf: Menu > Compiler: XeLaTeX. |
| Granularidad | Capitulado clásico (9 capítulos, ver índice). |
| Referencias | **Autor-año** con `biblatex` (`style=authoryear`, `backend=biber`), un único `referencias.bib`. Se evita `biblatex-apa` por pesado y lento en el plan gratuito de Overleaf. |
| Figuras/tablas | Generadas por el script de exportación desde los artefactos del study, no dibujadas a mano. Se referencian por nombre desde `latex/figuras/`. |
| Compilación | Subir `latex/` a Overleaf y seleccionar **XeLaTeX**. |

**Pendiente de acordar** (no bloquea): portada oficial de la universidad (hay una provisional en
`main.tex`) y estructura definitiva de anexos.

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
  figuras/              # imágenes y tablas exportadas del study de referencia
  tablas/               # .tex de tablas generadas, para \input desde los capítulos
```

El preámbulo de `main.tex` fija paquetes, geometría, bibliografía y la notación compartida
(`\tsnap`, `\tfiled`, `\hlabel`, `\rankic`). La notación se define ahí una vez y los capítulos la
reutilizan sin redefinirla.

## Índice de capítulos

`caps/` está **vacío**: ningún capítulo se ha escrito todavía. Toda la evidencia necesaria existe,
así que los nueve son redactables ya. El orden recomendado de redacción está en la última columna.

| # | Capítulo | Fuentes principales | Estado | Orden |
|---|---|---|---|---|
| 1 | Introducción y motivación | `CLAUDE.md`, este plan | Pendiente | 8.º |
| 2 | Estado del arte | `referencias.bib` | Pendiente | 7.º |
| 3 | Datos y universo de inversión | `evidence/dataset_reference.json`, `universe_coverage` | Pendiente | 3.º |
| 4 | Diseño metodológico: point-in-time | `docs/metodologia.md`, tests | Pendiente | 4.º |
| 5 | Agentes y meta-agente | `meta_weights.parquet`, `rank_ic_diagnostics.parquet` | Pendiente | 1.º |
| 6 | Diseño experimental | `config.json`, `decisions.json` | Pendiente | 5.º |
| 7 | Resultados | `evidence/summary.json`, `robustness.json`, `attribution.json` | Pendiente | 2.º |
| 8 | Limitaciones | `attribution.json`, `docs/bitacora.md` | Pendiente | 6.º |
| 9 | Conclusiones | Todo | Pendiente | 9.º |

Se empieza por 5 y 7 porque son los que sostienen el trabajo y los que fijan la terminología que
todos los demás reutilizan.

---

### 1. Introducción y motivación

Por qué IA/ML aplicado a bolsa como banco de pruebas de aprendizaje, no como objetivo de
rentabilidad. Pregunta de investigación: *¿puede un sistema de agentes especializados aprender una
ordenación transversal de acciones con valor predictivo fuera de muestra, medida con rigor
point-in-time, y esa ordenación se traduce en utilidad económica neta de costes?* Objetivos, y por
qué un resultado negativo bien medido también sería válido.

Cierra anticipando las cinco afirmaciones vertebradoras y el mapa del documento.

**Elementos a incluir:**
- *Figura 1.1* — Diagrama de bloques del sistema completo: catálogo cerrado → Model Study →
  optimización secuencial por Rank-IC → ganador → robustez/carteras/perfiles → informe. Es el mismo
  esquema de `CLAUDE.md`, dibujado en TikZ.
- *Tabla 1.1* — Las cinco afirmaciones del trabajo con su métrica y su artefacto. Funciona como
  resumen ejecutivo.

### 2. Estado del arte

Factor investing clásico (valor, momentum, calidad, baja volatilidad, GARP) como listón de
comparación. Aprendizaje automático aplicado a selección de activos: qué se ha probado y qué
problemas metodológicos son recurrentes (lookahead bias, sesgo de supervivencia, overfitting a pocas
eras, *p-hacking* por multiplicidad de configuraciones). Por qué el rank-IC OOS es el criterio de
evidencia elegido frente a reportar sólo rentabilidad. Ley fundamental de la gestión activa
(`IR ≈ IC·√BR·TC`) como marco para separar calidad de señal de calidad de implementación — se usará
literalmente en el Capítulo 7 para explicar el coeficiente de transferencia de 0,247.

Referencias que hay que tener sí o sí: Grinold y Kahn (ley fundamental), Bailey y López de Prado
(Deflated Sharpe), Fama y French, Jegadeesh y Titman (momentum), Ang et al. o Baker et al. (anomalía
de baja volatilidad — crítica para interpretar el dominio del agente `risk`), Gu, Kelly y Xiu
(ML empírico en *asset pricing*).

**Elementos a incluir:**
- *Tabla 2.1* — Comparativa de trabajos previos: horizonte, universo, métrica reportada, y si
  controlan o no por multiplicidad. Sirve para situar la contribución metodológica del TFM.

### 3. Datos y universo de inversión

Fuentes (Finnhub, Yahoo, SEC EDGAR) y por qué cada una. Universo dinámico por fecha desde la
composición histórica real del S&P 500 (`module/data/universe.py`): qué sesgo elimina (inclusión
anticipada) y cuál quedaría igualmente. Guarda de reciclaje de ticker con los casos reales (`CPQ`,
`MOB`). Sesgo de supervivencia **medido** por año, no sólo declarado.

**Elementos a incluir:**
- *Figura 3.1* — Cobertura del universo por año: `distinct_tickers` y `usable_fraction`, de
  `attribution.json` → `universe_coverage`. Muestra el crecimiento de 278 tickers (2003) a ~400
  (2015+) y una fracción utilizable ≥ 99,4 % en todos los años.
- *Tabla 3.1* — Catálogo de features por agente, de `evidence/feature_catalog.json`.
- *Figura 3.2* — Línea temporal del dataset: rango total, ventana de entrenamiento, ventana de
  selección 2015–2024 y era reservada 2025–2026 en color distinto. Es la figura que hace visible el
  diseño experimental y conviene repetirla en miniatura en el Capítulo 6.

### 4. Diseño metodológico: point-in-time y ausencia de lookahead

La regla central (año + trimestre + `lag_days` como margen de ejecución, nunca como retardo aplicado
al dato) y por qué el diseño alternativo —un retardo fijo por fundamental— es incorrecto, con los
contraejemplos reales que lo demostraron (AT&T 133 días, un 10-K de AAPL 88 días). Algoritmo de
observabilidad: fecha de publicación real vía SEC EDGAR frente a fecha de cierre fiscal. Qué se
prohíbe explícitamente (`payload.metric`, columnas de `profiles.parquet`, `sector`) y por qué.
Los tests de fuga temporal como parte del método.

Aquí se fija la notación: \(\tsnap\), \(\tfiled\), \(\hlabel\).

**Elementos a incluir:**
- *Figura 4.1* — **Diagrama de observabilidad**, la figura más importante del capítulo: eje temporal
  con cierre fiscal, \(\tfiled\) real, \(\tsnap\) con `execution_lag_days` = 60, y la ventana de
  etiqueta \(\hlabel\) = 12 meses. Con el contraejemplo de AT&T superpuesto para que se vea que un
  retardo fijo de 90 días habría usado un dato aún no publicado.
- *Tabla 4.1* — Contraejemplos reales de retardo de publicación (empresa, cierre fiscal, filingDate,
  días), demostrando la dispersión que invalida el retardo fijo.
- *Demostración formal* — Proposición: bajo la regla `\tsnap \geq \tfiled + \text{lag}`, ningún
  dato usado en la señal de \(\tsnap\) es posterior a \(\tsnap\). Media página, con la prueba
  siguiendo la implementación de `module/data/dataset.py`. Da rigor y es fácil de defender.
- *Tabla 4.2* — Batería de tests de fuga temporal: qué comprueba cada uno y qué fallo detectaría.

### 5. Agentes especializados y meta-agente

**Es el capítulo del aprendizaje y uno de los dos centrales.** Cinco agentes con features disjuntas
(quality, value, growth, momentum, risk) y su justificación económica. Modelo LightGBM por agente
(`max_depth` 3, 100 estimadores, `learning_rate` 0,03, `min_child_samples` 50) con entrenamiento
walk-forward y ventana móvil de 8 años. Meta-agente `stacked_rolling_bounded` con 16 trimestres de
historia y cota superior de 0,50 por agente.

La narración clave: el meta **no sabe** de antemano qué agente es bueno. Arranca en
`fallback_equal` (0,20 cada uno, 60 filas), y conforme se cierran etiquetas a 12 meses pasa a
`learned` (615 filas) y redistribuye. En 2016 apuesta por `momentum` (0,308) —que resultará el peor
agente—, en 2017 lo corrige y sube `risk` a 0,370, y desde 2018 lo mantiene en el tope de 0,50.

**Elementos a incluir:**
- *Figura 5.1* — **Curva de aprendizaje del meta-agente**: gráfico de áreas apiladas de los pesos de
  los cinco agentes por snapshot, 2015→2026, de `evidence/meta_weights.parquet`. Debe verse la banda
  plana de 0,20 al inicio, el error de 2016 con `momentum`, la corrección de 2017 y la meseta de
  `risk` en 0,50. **Es la figura más importante del TFM**: es literalmente la imagen del aprendizaje.
- *Figura 5.2* — Barras de rank-IC medio por agente con barras de error, ordenadas, marcando
  `meta_final` y `meta_equal_weight` en color distinto. Hace visible de un golpe que aprender los
  pesos (0,1004) bate a promediar (0,0659).
- *Tabla 5.1* — Rank-IC por agente: media, desviación, cohortes positivas, IC-IR (los 7 valores del
  informe de resultados).
- *Tabla 5.2* — Rank-IC por agente **y por era** (2015–2018 / 2019–2021 / 2022–2024), que muestra
  que ningún agente domina siempre y que `momentum` y `quality` llegan a ser negativos.
- *Figura 5.3* — Arquitectura del sistema en TikZ: features disjuntas → 5 agentes LightGBM →
  meta apilado acotado → score final → cartera.
- **Discusión obligatoria**: `risk` en solitario (0,1229) supera al meta (0,1004). Hay que abordarlo
  de frente, explicar que un combinador acotado a 0,50 no puede igualar a su mejor componente, y
  reformular la tesis como «el meta aprende sin supervisión a reproducir casi toda la señal de su
  mejor especialista partiendo de la ignorancia». Anticipa la neutralización del Capítulo 7, que
  descarta que `risk` sea sólo baja volatilidad clásica.

### 6. Diseño experimental: cartera, backtest y selección secuencial

Construcción de cartera (12 posiciones, `alpha_proportional`, umbrales en puntos básicos,
`opportunity_cash` con tope del 25 %), simulación con 5 pb de comisión y 10 pb de slippage.

Protocolo de selección secuencial por fases (temporal → representación → modelo → meta), con
**Rank-IC como única métrica de selección**, puerta pareada de no inferioridad, suelo por era de
−0,02 y bootstrap pareado de 2 000 réplicas. Por qué sólo las fases predictivas pueden modificar el
ganador y por qué robustez, perfiles y carteras son evidencia posterior sobre un ganador congelado.
Pre-registro del protocolo de lectura de la era reservada.

**Elementos a incluir:**
- *Figura 6.1* — Diagrama del protocolo secuencial: las cuatro fases predictivas, qué se decide en
  cada una y en qué momento el ganador queda congelado.
- *Tabla 6.1* — Catálogo cerrado completo: cada variable, sus valores candidatos y su baseline
  (de `config.json`). Es la tabla que demuestra que nada se optimizó fuera del catálogo.
- *Tabla 6.2* — **Traza de decisiones** de `decisions.json`: por variable, valor ganador, ventaja
  pareada, IC al 90 %, y si domina o es no inferior. Casos que merecen comentario en el texto:
  `snapshot_step_months` = 1 (ventaja +0,0208, IC [0,0108; 0,0372], distinguible de cero, y triplica
  las cohortes de 40 a 117) y `target_horizon_months` = 6 **rechazado** (−0,0265, IC
  [−0,0552; −0,0109]). Enseñar un rechazo es tan importante como enseñar una elección.
- *Figura 6.2* — Reproducción en miniatura de la línea temporal (Figura 3.2) marcando qué ventana
  alimenta cada fase.

### 7. Resultados

**El capítulo central.** Se estructura en cuatro bloques, en este orden, y conviene numerarlos así
en el propio documento porque la secuencia es el argumento.

#### 7.1 Capacidad predictiva

Rank-IC 0,1004, IC-IR 0,744, 71,79 % de cohortes positivas, 117 cohortes, t de Newey-West 3,02,
diferencial de colas 0,0365. Comparación con los cinco baselines deterministas (mejor: `garp_score`
con 0,0130 — el sistema lo multiplica por ~8).

- *Figura 7.1* — Serie temporal del rank-IC por cohorte 2015–2024 con media móvil y banda del
  bootstrap; la era reservada al final en sombreado distinto.
- *Figura 7.2* — Histograma de rank-IC por cohorte con la media marcada; hace visible el 71,79 %
  positivo.
- *Tabla 7.1* — Sistema frente a baselines deterministas.
- *Tabla 7.2* — Rank-IC por era, mostrando que **no se degrada** (0,0976 / 0,0423 / 0,1621).

#### 7.2 Robustez: aprendizaje, no suerte

**Es la sección que sostiene el trabajo y debe ir antes que los resultados económicos**, porque sin
ella la rentabilidad no significa nada. Ocho contrastes, siete superados y uno no, todos reportados.

- *Tabla 7.3* — **Tabla resumen de robustez**: contraste, pregunta que responde, resultado, veredicto.
  Ocho filas. Es la tabla que el tribunal va a mirar primero.
- *Figura 7.3* — **Distribución nula de la permutación** (9 999 réplicas) con el rank-IC observado
  marcado muy a la derecha, fuera del soporte. Visualmente demoledora contra la hipótesis de azar.
- *Figura 7.4* — Placebos de etiqueta: los cinco valores en [−0,006; +0,001] frente al 0,1004 real,
  en la misma escala. Demuestra que la maquinaria no fabrica señal.
- *Figura 7.5* — Bootstrap por bloques: distribución e intervalos al 90 % y 95 %, con el cero fuera.
- *Tabla 7.4* — Exclusión de eras (0,073–0,126) y estabilidad entre semillas (rango 0,0020, sin
  cruce de cero, `economic_conclusion_stable = true`).
- *Figura 7.6* — Carteras aleatorias con riesgo emparejado: distribución de 1 000 CAGR con el modelo
  en el percentil 97,4. Explicar en el texto por qué el contraste «general» (p95 = 102 % anual) no es
  informativo — es un punto de honestidad metodológica que suma.
- *Tabla 7.5* — Neutralización por estilo: rank-IC bruto 0,1111 → neutralizado 0,0937, retiene
  84,35 % con 14 controles. Y la regresión factorial con Newey-West (alfa 0,13 %, t = 0,82,
  R² = 0,021, sin cargas significativas).
- **Deflated Sharpe 0,930 < 0,95**: se reporta explícitamente como el contraste no superado, con la
  distinción clave —la evidencia de *capacidad predictiva* supera todo, la de *rentabilidad ajustada
  por riesgo* no resiste del todo la corrección por 66 configuraciones—.

#### 7.3 Traducción económica y la era reservada

Ventana de selección: CAGR 15,01 % vs 13,17 %, exceso 1,62 %, IR 0,269, MDD 23,44 %, beat rate 8/10.
Curva completa: CAGR 17,36 %, exceso 3,12 %, IR 0,416, beat rate 10/12.

**Era reservada 2025–2026 (confirmación pura):** CAGR 36,11 % vs 19,18 %, exceso **+14,21 %**,
IR **0,959**, beat rate **2/2**, alfa +9,76 pp (2025) y +9,92 pp (2026), MDD 7,32 %, turnover 65,78 %,
alfa factorial con **t = 4,76**. El sistema bate al S&P 500 sobre datos que no tocaron ninguna
decisión, con mejor IR y menor drawdown que en la ventana de selección, y operando menos.

- *Figura 7.7* — **Curva de equity** cartera vs SPY, 2015–2026, con la era reservada sombreada. La
  figura de portada de los resultados.
- *Figura 7.8* — Barras de alfa anual, 12 años, verde/rojo, con 2025–2026 destacados. Se ve
  inmediatamente que sólo 2020 y 2024 son negativos y que los dos años reservados son de los mejores.
- *Tabla 7.6* — Detalle anual completo (12 filas: retorno, benchmark, alfa, MDD, IR, efectivo,
  turnover) de `evidence/annual_metrics.parquet`.
- *Tabla 7.7* — Comparativa selección / confirmación / curva completa en tres columnas.
- *Figura 7.9* — Drawdown de cartera y benchmark superpuestos.
- **El matiz obligatorio**: el rank-IC de la era reservada es −0,0119, sobre 6 cohortes cerradas y
  **1 observación independiente**. Explicar las tres razones (potencia nula, el rank-IC mide los ~400
  valores mientras la cartera usa sólo el extremo superior, y las cohortes de 2025 H2 en adelante no
  tienen etiqueta cerrada) y formularlo como *indeterminado*, no como negativo. Es exactamente el
  tipo de matiz que un tribunal premia si lo declaras tú y penaliza si lo encuentra él.
- *Análisis de 2020 y 2024*: los dos años perdedores son los de mayor concentración del índice en
  megacaps de crecimiento, donde una cartera de 12 nombres con sesgo a bajo riesgo no puede seguir al
  benchmark; en 2024 además con 22,9 % de efectivo.

#### 7.4 Perfiles y el coeficiente de transferencia

Los ocho perfiles comparten **la misma señal** (rank-IC 0,1004 en los ocho) y difieren sólo en la
regla de construcción de cartera: experimento controlado perfecto. `balanced` gana en CAGR (15,01 %),
exceso (+1,62 %), IR (0,269), beat rate (8/10) y alfa medio (+1,72 %) **simultáneamente**, y es el
único que no reordena la señal. Seis de los siete perfiles que sí la reordenan destruyen alfa, y el
más agresivo (`momentum`, 611 % de turnover) es el peor con −6,37 % y MDD 39,82 %.

Tesis a defender: **la mejor manera de usar la señal aprendida es no interferir con ella**. Nótese
además que `momentum` es a la vez el peor agente y el peor perfil, lo que refuerza que el sistema
captura algo distinto del momentum clásico.

Coeficiente de transferencia 0,247: la cartera captura una cuarta parte de la señal. Aplicar aquí la
ley fundamental del Capítulo 2 para mostrar que el cuello de botella es la cartera (long-only, 12
nombres, 359 % de rotación, 5,25 % de costes acumulados), **no el modelo**.

- *Tabla 7.8* — Los ocho perfiles con las siete métricas, ordenados por IR.
- *Figura 7.10* — Barras horizontales de exceso geométrico por perfil, con `balanced` destacado y la
  línea del benchmark en cero. Se ve el orden monótono: cuanto más reordena el perfil, peor.
- *Figura 7.11* — Dispersión turnover vs exceso geométrico por perfil, que hace visible la relación
  entre rotación y destrucción de alfa.
- *Figura 7.12* — Descomposición del coeficiente de transferencia: señal bruta → tras restricción
  long-only → tras 12 nombres → tras costes → alfa realizado.

### 8. Limitaciones y amenazas a la validez

Se cierra con lo que los resultados han revelado, no sólo con lo previsto:

1. **Deflated Sharpe 0,930 < 0,95** — 66 configuraciones probadas. La limitación más importante y la
   primera que hay que enunciar.
2. **~9 observaciones independientes efectivas** frente a 117 cohortes, por el solapamiento de
   etiquetas a 12 meses. Toda afirmación de significación debe leerse con esto delante.
3. **Rank-IC de la era reservada indeterminado** (6 cohortes, 1 observación independiente).
4. **Alfa de la ventana de selección no significativo por sí solo** (t = 0,82), aunque sí lo sea en
   la era reservada (t = 4,76).
5. **Coeficiente de transferencia 0,247**: la mayor parte de la señal no llega a la cartera.
6. **`risk` domina al meta**; si el TFM se leyera como «el sistema redescubrió baja volatilidad», la
   defensa es la neutralización (retiene 84 %) — pero es una defensa parcial, no una refutación.
7. **Sesgo de supervivencia residual** medido por año, no eliminable con fuentes gratuitas.
8. **Restatements invisibles antes de 2009** (Finnhub da el valor actual; EDGAR sólo fecha la
   publicación).
9. **Cobertura desigual de métricas** y por qué no se imputan con la media.
10. **Universo restringido al S&P 500** y a un único mercado y régimen monetario.
11. **Costes modelados como constante** (5 pb + 10 pb), sin impacto de mercado dependiente del
    tamaño.
12. **Reproducibilidad acotada por la versión de catálogo**: el study se ejecutó con
    `CATALOG_VERSION` 5 y el código actual está en 6, de modo que una reejecución no devolvería el
    mismo ganador. Se declara explícitamente, con los hashes de dataset y evaluación como garantía
    de trazabilidad de las cifras publicadas.

- *Tabla 8.1* — Cada limitación con su severidad, la evidencia que la cuantifica y qué afirmación
  del Capítulo 7 acota. Convierte el capítulo en un instrumento de precisión y no en una disculpa.

### 9. Conclusiones y trabajo futuro

Respuesta directa a la pregunta de investigación, en los términos de las cinco afirmaciones: el
sistema **sí** aprende una ordenación transversal que se sostiene fuera de muestra y que **sí** batió
al S&P 500 en los dos años reservados; lo que no está establecido con la misma solidez es que su
Sharpe resista la corrección por multiplicidad. Honestidad sobre el tamaño de muestra efectivo.

Trabajo futuro que la propia evidencia sugiere:

1. **Atacar el coeficiente de transferencia, no el modelo** — ahí se pierde el 75 % de la señal.
   Ampliar `target_size`, reducir rotación, sizing por convicción.
2. **Reejecutar con un catálogo pre-registrado más estrecho** para que el Deflated Sharpe no pague el
   peaje de 66 pruebas.
3. **Esperar al cierre de cohortes de 2025–2026** para contrastar el rank-IC de la era reservada con
   potencia real.
4. **Caracterizar qué es `risk`**: si fuera baja volatilidad clásica, la neutralización habría
   destruido mucho más del 16 %.
5. Fuente de sector histórico point-in-time; EDGAR `companyfacts` si se reorientase a un ancla ≥2009.

---

## Inventario de figuras y tablas

30 elementos, todos generables desde artefactos existentes del study de referencia. Prioridad **A** =
imprescindible, **B** = recomendable.

| Id | Elemento | Fuente | Prio |
|---|---|---|---|
| F1.1 | Diagrama de bloques del sistema | TikZ | A |
| T1.1 | Las cinco afirmaciones y sus artefactos | `informe_resultados.md` | A |
| T2.1 | Comparativa de trabajos previos | `referencias.bib` | B |
| F3.1 | Cobertura del universo por año | `attribution.json` → `universe_coverage` | A |
| T3.1 | Catálogo de features por agente | `evidence/feature_catalog.json` | A |
| F3.2 | Línea temporal del dataset y las ventanas | TikZ | A |
| F4.1 | Diagrama de observabilidad point-in-time | TikZ | A |
| T4.1 | Contraejemplos de retardo de publicación | `docs/metodologia.md` | A |
| D4.1 | Demostración de ausencia de lookahead | `module/data/dataset.py` | A |
| T4.2 | Batería de tests de fuga temporal | `tests/` | B |
| F5.1 | **Curva de aprendizaje del meta-agente** | `evidence/meta_weights.parquet` | **A** |
| F5.2 | Rank-IC por agente (barras) | `evidence/rank_ic_diagnostics.parquet` | A |
| T5.1 | Rank-IC por agente (tabla) | `evidence/rank_ic_diagnostics.parquet` | A |
| T5.2 | Rank-IC por agente y era | `evidence/rank_ic_diagnostics.parquet` | A |
| F5.3 | Arquitectura del sistema | TikZ | A |
| F6.1 | Protocolo de selección secuencial | TikZ | A |
| T6.1 | Catálogo cerrado completo | `config.json` | A |
| T6.2 | Traza de decisiones con IC pareados | `decisions.json` | A |
| F7.1 | Serie temporal del rank-IC por cohorte | `evidence/summary.json` | A |
| F7.2 | Histograma de rank-IC | `evidence/summary.json` | B |
| T7.1 | Sistema vs baselines deterministas | `attribution.json` → `baselines` | A |
| T7.2 | Rank-IC por era | `evidence/summary.json` → `eras` | A |
| T7.3 | **Resumen de robustez (8 contrastes)** | `robustness.json` | **A** |
| F7.3 | **Distribución nula de la permutación** | `robustness.json` | **A** |
| F7.4 | Placebos de etiqueta | `robustness.json` | A |
| F7.5 | Bootstrap por bloques | `robustness.json` | A |
| T7.4 | Exclusión de eras y semillas | `robustness.json` | A |
| F7.6 | Carteras aleatorias (riesgo emparejado) | `robustness.json` | A |
| T7.5 | Neutralización por estilo y regresión factorial | `attribution.json` | A |
| F7.7 | **Curva de equity vs SPY** | `evidence/equity.parquet` | **A** |
| F7.8 | Alfa anual (barras) | `evidence/annual_metrics.parquet` | A |
| T7.6 | Detalle anual completo | `evidence/annual_metrics.parquet` | A |
| T7.7 | Selección / confirmación / curva completa | `evidence/summary.json` | A |
| F7.9 | Drawdown superpuesto | `evidence/equity.parquet` | B |
| T7.8 | Los ocho perfiles | `profile_comparison.parquet` | A |
| F7.10 | Exceso por perfil (barras) | `profile_comparison.parquet` | A |
| F7.11 | Turnover vs exceso por perfil | `profile_comparison.parquet` | B |
| F7.12 | Descomposición del coef. de transferencia | `attribution.json` → `transfer` | B |
| T8.1 | Limitaciones con severidad y evidencia | `attribution.json`, bitácora | A |

**Pendiente de implementar:** un script de exportación (p. ej. `module/reporting/export_latex.py`)
que lea el study de referencia y escriba `latex/figuras/*.pdf` y `latex/tablas/*.tex`. Debe recibir
el `study_id` como argumento y ser reejecutable, para que ninguna figura se genere a mano y todas
queden trazadas a su artefacto. Figuras en PDF vectorial, tablas en `booktabs`.

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
