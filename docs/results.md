# Qué produce el sistema y cómo leerlo

Este documento es la muestra de lo que el sistema entrega y de cómo comprobarlo por tu cuenta. No
transcribe ni una cifra: **las cifras viven en los artefactos**, y aquí se explica qué artefacto
responde a qué pregunta.

Para la arquitectura del código, ver [architecture.md](architecture.md); para ejecutarlo,
[usage.md](usage.md).

---

## 0. Cómo interpretar estos resultados

Antes de mirar nada, conviene saber qué se puede afirmar y qué no.

**La única métrica que decide es el Rank-IC robusto de la ventana de selección (2015–2024).** Alfa,
Information Ratio, rentabilidad, turnover y perfiles son **informativos**: se calculan, se publican y
se discuten, pero ninguno eligió jamás una configuración. Esa separación es deliberada y es lo que
impide seleccionar reglas de cartera porque casualmente funcionaron mejor en la historia conocida.

**2025–2026 es estrés reservado.** No participó en ninguna decisión, de ninguna pasada. Su Rank-IC
es la única medida del trabajo libre de sesgo de selección, se calcula una sola vez sobre el ganador
ya congelado y se publica salga lo que salga.

**Lo que queda dentro de la ventana de selección es una cota superior optimista.** El ganador es el
mejor de un conjunto de configuraciones probadas sobre los mismos datos; sus cifras de selección no
son una estimación insesgada. Por eso hay un Deflated Sharpe Ratio corrigiendo por multiplicidad, y
por eso la era reservada existe.

**Un Study en modo dev no es evidencia de nada**: demuestra que el software funciona, no que exista
señal económica.

---

## 1. Qué produce una ejecución completa

Al terminar la cadena, el sistema ha producido seis cosas:

1. **Una ordenación transversal de acciones validada fuera de muestra.** No una lista de valores
   «buenos», sino una puntuación por acción y fecha cuya capacidad predictiva se mide con Rank-IC
   sobre cohortes ya cerradas.
2. **El registro auditable de cada decisión** que llevó del baseline al modelo ganador: qué
   candidatos se evaluaron, con qué Rank-IC, qué puerta estadística pasaron y por qué regla se
   decidió cada una.
3. **Una cartera con reglas económicas explícitas**, donde cada venta tiene un motivo persistido y
   ninguna operación se emite si no se paga a sí misma después de costes.
4. **Ocho perfiles de inversor** sobre la misma señal —balanced, growth, value, quality, momentum,
   contrarian, defensive y garp—, deterministas y sin reentrenar, para responder cómo le habría ido a
   cada estilo con la misma gestión.
5. **Cuatro diagnósticos que intentan tumbar el resultado**, no confirmarlo:
   - **robustez** — bootstrap por bloques, exclusión de eras, dispersión entre semillas, permutación
     y **placebos de etiqueta** (si el sistema «acierta» con etiquetas barajadas, no está aprendiendo
     nada);
   - **atribución** — regresión sobre réplicas de factores conocidos con errores Newey-West y
     **Deflated Sharpe Ratio**, para separar «aprende» de «redescubrió el momentum»;
   - **capacidad** — a partir de qué patrimonio la cartera deja de ser ejecutable;
   - **sensibilidad a costes** — hasta qué coste por operación aguanta la ventaja, incluido el coste
     de equilibrio que la anula.
6. **Un informe** (`report.md`) con la procedencia de cada cifra.

---

## 2. Los cuatro estudios que vienen en el repositorio

`results/studies/` contiene la evidencia real del TFM: tres Model Studies encadenados y el Portfolio
Study adoptado.

| Directorio | Papel |
|---|---|
| `study-20260816-182345-3cc1a5fb` | Model Study, primera pasada |
| `study-20260817-021135-b5926b62` | Model Study, segunda pasada |
| `study-20260817-094411-568bd37e` | Model Study, tercera pasada — el modelo de referencia |
| `study-20260817-212856-f86ca822` | Portfolio Study adoptado (1.440 combinaciones) |

### Por qué son tres y no uno

Porque la optimización es **greedy secuencial**: cada variable se evalúa sobre el incumbent
acumulado, no sobre todas las combinaciones posibles. Una variable decidida pronto —cuando el resto
de la configuración todavía era la recomendada por defecto— nunca vuelve a revisarse con lo demás ya
optimizado. El resultado depende del punto de partida.

Encadenar ataca exactamente eso: **el ganador de la pasada *n* es el baseline de la pasada *n+1***.
Cada pasada completa es una iteración de ascenso por coordenadas, y la mejora entre pasadas es la
evidencia de que el procedimiento converge. La comparación entre pasadas se hace **solo** con el
Rank-IC robusto de la ventana de selección, que es el criterio con el que se eligió cada ganador.

Una cadena puede **converger sin mejorar**, y eso también es un resultado: si una pasada devuelve el
mismo ganador que la anterior, el óptimo greedy es estable frente al punto de partida.

**El riesgo que esto obliga a declarar:** encadenar multiplica las configuraciones probadas sobre los
mismos datos y agrava la selección múltiple. El Deflated Sharpe penaliza vía `n_trials`, que crece
con cada pasada. La ganancia y el riesgo de sobreajuste crecen a la vez, y la era reservada es la
única defensa real —precisamente porque no participa en ninguna pasada.

---

## 3. Dos recorridos: de una pregunta a su respuesta

Esta es la parte que demuestra que el sistema es auditable de verdad y no de palabra.

### «¿Por qué el modelo usa este lag de publicación y no otro?»

1. Abre `results/studies/study-20260817-094411-568bd37e/decisions.json`.
2. Busca en `decisions` la entrada con `variable_id: "execution_lag_days"`.
3. Ahí está `selection_rule`. En este estudio concreto vale **`tie_simplicity`**, no
   `robust_rank_ic`: significa que ninguno de los candidatos superó a los demás por encima de la
   tolerancia de empate, y la decisión la resolvió la tabla de simplicidad del catálogo.
4. Dentro de `candidates` tienes, valor a valor: `paired_advantage` (la ventaja pareada contra el
   incumbent), `paired_bootstrap_90`, `mean_rank_ic`, `positive_fraction`, `rank_ic_std`,
   `observations`, si era `eligible`, qué `gates` pasó y el `reason` textual.

**Y esa distinción importa al leer el TFM**: una decisión resuelta por `tie_simplicity` es una
convención de desempate declarada de antemano, **no un hallazgo empírico**, y no debe presentarse
como tal. El mismo fichero permite comprobar cuáles de las dieciocho decisiones se resolvieron por
evidencia (`robust_rank_ic`) y cuáles por empate.

### «¿Qué compró de verdad la cartera, y cuándo?»

1. `portfolio_winner.json` del Portfolio Study da la combinación ganadora y su resumen.
2. `evidence_best_full/` contiene su evidencia sobre la **serie completa** (incluida la era
   reservada); `evidence_best/` es la misma cartera pero recortada a la ventana de selección. No
   mezclarlas: miden ventanas distintas.
3. Dentro, `orders.parquet` tiene las operaciones concretas, **cada una con su motivo persistido**
   (`missing_current_score`, `below_coverage_percentile`, `expected_alpha_below_exit`,
   `initial_fill`, `displaced_by_net_edge`, `rebalance`…). Se puede reconstruir por qué se emitió
   cada orden.
4. `contributions.parquet` dice qué aportó cada posición a cada periodo. Su suma es **exactamente**
   el retorno bruto del periodo —es una identidad, no una aproximación, y hay un test de contrato que
   lo verifica—, lo que convierte la atribución por acción en contabilidad y no en estimación.

---

## 4. Cómo ver todo esto sin ejecutar nada

Hay dos caminos que no requieren ni credenciales ni horas de cómputo:

- **`latex/TFM.pdf`** está versionado. Es el análisis completo, ya escrito, con sus figuras y tablas.
  También `latex/TFM_ppt.pdf`, la presentación de defensa.
- **`python main.py serve`** y abrir la pestaña **Resultados**: los cuatro estudios aparecen ahí con
  sus runs, su consola, su robustez y sus perfiles, navegables. El dashboard lee los artefactos que
  ya están en el repositorio.

---

## 5. Mapa de artefactos

### Model Study

```text
results/studies/<study_id>/
├── study.json                    Estado, presupuesto y progreso
├── config.json                   Definición lanzada
├── catalog_snapshot.json         El catálogo tal como estaba al lanzar
├── decisions.json                Cada decisión, su regla y todos sus candidatos
├── evaluation_ledger.parquet     Una fila por evaluación de candidato
├── winner.json                   Configuración ganadora y su resumen
├── robustness.json               Semillas, bootstrap, eras, permutación, placebos
├── attribution.json              Regresión factorial, Deflated Sharpe, confirmación OOS
├── events.jsonl                  Todos los eventos, en orden
├── report.md                     Informe con la procedencia de cada cifra
├── storage_manifest.json         Huella de almacenamiento
├── runs/run-<hex>.json           Un JSON por run
├── evidence/                     Evidencia completa del ganador (18 artefactos)
│   └── profiles/<perfil>/        Uno por cada uno de los ocho perfiles
└── evidence_baseline/            La misma evidencia, para el baseline
```

Los 18 artefactos de `evidence/`:

| Artefacto | Qué contiene |
|---|---|
| `summary.json` | Rank-IC, IC-IR, cohortes y métricas económicas del ganador |
| `agent_scores.parquet` | Puntuación de cada agente por acción y fecha |
| `meta_weights.parquet` | Peso de cada agente en cada fecha |
| `rank_ic_diagnostics.parquet` | Rank-IC por cohorte y por agente |
| `rank_tail_diagnostics.parquet` | Comportamiento de la cola efectivamente negociada |
| `signal_calibration.parquet` | Curva percentil → alfa esperado, y qué ventana la produjo |
| `signal_health.parquet` | Salud de la señal por snapshot |
| `model_feature_attribution.parquet` | Importancia de features por modelo |
| `agent_local_attribution.parquet` | Atribución local por agente |
| `feature_diagnostics.parquet` | Diagnósticos de las features |
| `feature_catalog.json` | Qué features vio cada agente |
| `equity.parquet` | Curva de capital |
| `annual_metrics.parquet` | Métricas año a año |
| `positions.parquet` | Posiciones por snapshot |
| `orders.parquet` | Órdenes con su motivo |
| `contributions.parquet` | Aportación de cada posición |
| `manifest.json` | Inventario y hashes |
| `dataset_reference.json` | Puntero al dataset preparado |

### Portfolio Study

```text
results/studies/<study_id>/
├── portfolio_grid.parquet              Una fila por combinación evaluada
├── portfolio_winner.json               Combinación ganadora
├── evidence_best/                      Evidencia en la ventana de selección
├── evidence_best_full/                 Evidencia sobre la serie completa
├── profiles/<perfil>/                  Los ocho perfiles con la cartera ganadora
├── portfolio_profiles.parquet          Comparación entre perfiles
├── capacity.json                       Participación sobre volumen y patrimonio máximo
├── cost_sensitivity.json               Curva de exceso frente al coste y equilibrio c*
├── portfolio_narrative.json            Presencia, contribución, aciertos y errores
├── portfolio_narrative_holdings.parquet  Una fila por acción
└── report.md
```

---

## 6. Dos advertencias sobre lo que hay en el clon

**Los `.parquet` no están versionados.** `.gitignore` los excluye, así que un clon recibe los
`.json`, `.jsonl` y `.md` de los cuatro estudios —que son las decisiones, los ganadores, la robustez,
la atribución y los informes— pero **no las series numéricas**. Consecuencia práctica:
`python latex/build.py` con `REGENERAR_ACTIVOS=True` no puede funcionar en un clon. Para leer el
trabajo no hace falta: el PDF está versionado.

**La evidencia apunta al dataset, no lo copia.** `dataset_reference.json` referencia
`data/prepared/<hash>/` y se valida al leerlo. Los datasets preparados **nunca** se copian dentro de
`results/`: son grandes, inmutables y compartidos entre estudios.

---

## 7. La regla que gobierna todo

**Ninguna cifra vive en este documento, ni en ningún otro `.md`.** Viven en los artefactos de
`results/studies/<study_id>/` y se leen de ahí.

Copiar una cifra a un documento crea una segunda verdad que se desincroniza en cuanto se relanza un
estudio — que es exactamente lo que esta regla existe para impedir. Toda afirmación numérica cita el
`study_id` y la ruta del artefacto que la respalda.
