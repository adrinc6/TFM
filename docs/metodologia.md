# Metodología y arquitectura vigente

## 0. Propósito y estado del documento

Este documento es la referencia técnica y metodológica principal del proyecto. Su finalidad es
permitir:

1. comprender qué pregunta científica responde el sistema;
2. reconstruir una ejecución sin depender del conocimiento tácito del autor;
3. distinguir decisiones exploratorias, pruebas confirmatorias y observaciones posteriores;
4. justificar en el TFM cada transformación de datos, modelo, cartera y prueba estadística;
5. identificar con honestidad qué está implementado, qué está validado por tests y qué falta por
   demostrar empíricamente.

La implementación vigente sustituyó la arquitectura anterior. No hay compatibilidad con scenarios,
experiments, study manual, full study, runs sueltos ni resultados anteriores. El único flujo es:

```text
fuentes raw
  → dataset point-in-time compartido
  → Exploratory Study secuencial
  → hipótesis congelada
  → Confirmatory Study cerrado
  → evidencia final
```

Estado a 25 de julio de 2026:

- La nueva arquitectura, API y dashboard están implementados.
- El catálogo científico está cerrado y versionado.
- El presupuesto se valida tanto en frontend como en backend.
- Hay tests de los contratos críticos.
- Los resultados del protocolo nuevo aún no se han generado. Por tanto, no se sostiene todavía
  ninguna conclusión empírica sobre Rank-IC, alfa o confirmación.

## 1. Pregunta de investigación

El proyecto estudia si información financiera y de mercado disponible en cada fecha puede:

1. ordenar transversalmente las acciones del universo mediante una señal con Rank-IC estable; y
2. transformar esa capacidad de ordenación en alfa neto frente a SPY mediante una cartera
   implementable.

Son dos afirmaciones distintas. Un Rank-IC positivo no garantiza alfa. La conversión depende de la
forma de la cola superior, el horizonte de mantenimiento, los costes, el turnover, el sizing, la
exposición activa y la estabilidad temporal. Por eso el sistema evalúa por separado:

- **Calidad predictiva:** Rank-IC, frecuencia de cohortes positivas, dispersión y spread de cola.
- **Traducción económica:** alfa neto, Information Ratio, turnover y comportamiento por eras.
- **Robustez:** semillas, perfiles, costes, calendario, placebos, bootstrap, permutación y carteras
  aleatorias.

La formulación defendible no es «se encontró la configuración que más ganó históricamente», sino:

> Se construyó una hipótesis mediante un proceso exploratorio declarado, se congeló y después se
> sometió a un protocolo confirmatorio que no podía modificarla.

## 2. Separación epistemológica

### 2.1 Exploratory Study

Exploratory sirve para construir una hipótesis. El usuario elige qué variables quedan fijas y qué
valores cerrados se comparan. La optimización es secuencial: se modifica una variable, se decide su
valor y ese ganador pasa a ser el baseline del paso siguiente.

Sus resultados son evidencia exploratoria. Aunque se usen gates, bootstrap y eras, han participado
en la elección de la configuración y no deben presentarse como una prueba independiente.

### 2.2 Hipótesis congelada

La hipótesis es la frontera entre descubrimiento y corroboración. Al congelarla se persisten:

- definición completa del catálogo;
- configuración efectiva;
- orden de optimización;
- decisiones automáticas e intervenciones humanas;
- métricas de selección;
- versión del catálogo;
- hashes del dataset y de la evaluación;
- evidencia necesaria para reproducir las vistas analíticas.

No existe una operación de edición. Cualquier modificación exige crear otro Exploratory.

### 2.3 Confirmatory Study

Confirmatory recibe una hipótesis congelada y no acepta overrides científicos. Ejecuta 23
evaluaciones predefinidas. No busca una alternativa mejor: intenta refutar o corroborar una
afirmación ya fijada.

### 2.4 Estrés conocido 2025–2026

Los años 2025 y 2026 se consideran observados con anterioridad al reinicio metodológico. Se
calculan y muestran únicamente después del veredicto como:

```text
known_stress_not_selection
```

No pueden participar en gates, desempates, calibración del veredicto ni selección humana. La
primera evidencia verdaderamente futura comenzará después de congelar el modelo final.

## 3. Periodización

La selección hasta 2024 se divide en tres eras:

| Era | Años | Función |
|---|---:|---|
| Era 1 | 2015–2018 | Régimen histórico inicial |
| Era 2 | 2019–2021 | Régimen intermedio |
| Era 3 | 2022–2024 | Régimen reciente de selección |
| Estrés conocido | 2025–2026 | Informe posterior, nunca selección |

Las eras evitan que una media agregada oculte una degradación concentrada. El runner filtra
explícitamente diagnósticos y métricas de selección a años menores o iguales que 2024.

## 4. Datos y diseño point-in-time

### 4.1 Fuentes conservadas

`data/raw/` contiene las fuentes originales y no se duplica dentro de cada estudio. El histórico de
componentes del S&P 500 se conserva para construir el universo vigente en cada fecha. La ingesta
incluye clientes para información de mercado y fundamentales, además de un cliente EDGAR.

Las credenciales residen fuera de la evidencia científica, en la configuración local y `.env`.

### 4.2 Universo

El universo debe determinarse en cada snapshot a partir de membresía histórica. La regla
metodológica es:

```text
una acción solo puede seleccionarse si pertenecía al universo en esa fecha
```

Esto reduce sesgo de supervivencia. Los nulos aleatorios también deben usar el universo disponible
en cada fecha, no una lista actual reconstruida retrospectivamente.

### 4.3 Disponibilidad temporal

Los fundamentales se incorporan con un `execution_lag_days` del catálogo. El lag representa el
tiempo entre el cierre del periodo contable y el momento prudente en que el dato puede considerarse
operable. Los valores admitidos son 30, 45 y 60 días.

La construcción del dataset separa:

- panel point-in-time;
- precios de activos;
- benchmark;
- features;
- targets futuros.

El target se utiliza para entrenamiento y evaluación solo cuando su `label_end_date` ya está
cerrada en la fecha que consume la observación. Los nombres vigentes son neutrales:
`forward_return`, `forward_benchmark_return` y `forward_excess_return`.

### 4.4 Dataset compartido

La identidad del dataset se deriva de la configuración relevante y de la huella de los inputs raw.
La materialización vive en:

```text
data/prepared/<dataset_hash>/
```

Los paneles y precios comunes se crean una sola vez. Las variantes pueden compartir ficheros
mediante hard links y referencias. Ningún study copia el dataset completo.

## 5. Representación de la señal

### 5.1 Familias conceptuales

Las features están agrupadas por bloques interpretables:

- calidad;
- valor;
- crecimiento;
- momentum;
- riesgo;
- liquidez;
- estabilidad y fortaleza financiera cuando el preset lo incluye.

El catálogo no permite escribir listas libres. Los presets iniciales son:

- `core`;
- `fundamental`;
- `technical`;
- `all`.

La selección se expresa por bloques deseados, no por presets de exclusión. Cada opción explica en
el dashboard qué información incorpora antes de que el usuario la elija.

### 5.2 Transformaciones opcionales

El catálogo controla:

- momentum fundamental;
- feature de régimen de mercado;
- neutralización por sector;
- winsorización transversal;
- máximo de features por agente;
- selección nativa o poda por estabilidad OOS.

Toda transformación que usa distribución transversal debe calcularse dentro del snapshot. Toda
transformación temporal debe usar únicamente historia disponible.

## 6. Agentes y modelos

### 6.1 Agentes

Los cinco agentes conceptuales son:

- `quality`;
- `value`;
- `growth`;
- `momentum`;
- `risk`.

Cada agente se activa explícitamente mediante una variable positiva: Quality, Value, Growth,
Momentum y Risk. Debe permanecer activo al menos uno. El catálogo de features asigna a cada agente
solo columnas coherentes con su función.

### 6.2 Familias de modelo

Las familias cerradas son:

- LightGBM;
- Elastic Net.

Para LightGBM se pueden fijar u optimizar profundidad, número de estimadores, learning rate y
mínimo de observaciones por hoja. Estos parámetros quedan inactivos cuando la familia elegida no
es LightGBM y no consumen presupuesto.

### 6.3 Walk-forward

Los modelos se ajustan siguiendo un calendario walk-forward. En cada fecha:

1. se identifica la historia permitida por `train_lookback_years`;
2. se excluyen etiquetas aún no cerradas;
3. se seleccionan o podan features con evidencia disponible;
4. se ajusta cada agente;
5. se producen scores para el siguiente snapshot;
6. se registran importancias y diagnósticos.

La recencia puede estar desactivada, ser lineal o exponencial. Esta ponderación afecta a las
observaciones de entrenamiento, no modifica retrospectivamente las etiquetas.

## 7. Meta-agente

El meta-agente combina rankings de agentes. Los métodos admitidos son:

- `equal`: equiponderación;
- `rank_ic`: peso según Rank-IC histórico cerrado;
- `stacked_rolling`: stacker con ventana móvil;
- `stacked_exponential`: stacker con decaimiento temporal.

Los métodos aprendidos admiten:

- historia de 8 o 16 trimestres;
- cap por agente de 0,50, 0,75 o 1,00;
- contracción hacia equal de 0, 0,25 o 0,50;
- semivida de 4 u 8 trimestres cuando corresponde.

Un trimestre es una cohorte trimestral marcada como tal. La evidencia del meta solo puede usar
cohortes con etiqueta cerrada. Si no existe evidencia suficiente, el sistema recae en pesos
equiponderados. Después del cap y la contracción, los pesos se normalizan.

Artefactos principales:

- `agent_scores.parquet`;
- `meta_weights.parquet`;
- `rank_ic_diagnostics.parquet`;
- `rank_tail_diagnostics.parquet`;
- `feature_diagnostics.parquet`;
- atribuciones globales y locales.

## 8. Diagnósticos de señal

La métrica central es la correlación de Spearman entre ranking y retorno futuro dentro de cada
cohorte. No se agregan todas las filas sin control temporal: primero se calcula por cohorte y
después se resumen cohortes.

Se guardan:

- Rank-IC medio;
- fracción de cohortes con Rank-IC positivo;
- desviación temporal del Rank-IC;
- Rank-IC por era;
- spread del decil superior frente al universo;
- métricas de cola operada;
- cobertura y número de observaciones;
- salud de señal;
- calibración entre ranking y retorno excedente esperado.

La cola es imprescindible porque la cartera compra pocas posiciones. Una señal puede ordenar bien
el centro de la distribución y no separar suficientemente el top que se opera.

## 9. Motor de cartera

### 9.1 Estructuras

Las estructuras cerradas son:

- `quarterly`: rebalanceo trimestral;
- `four_vintages`: cuatro vintages escalonados.

En `four_vintages`, el tamaño objetivo debe ser divisible entre cuatro. Cada vintage abre una parte
de la cartera, conserva lotes separados y expira conforme al horizonte de 12 meses. Esto alinea la
decisión económica con la etiqueta anual y reduce rotación innecesaria.

### 9.2 Núcleo y satélite

La cartera combina:

- satélite activo de acciones seleccionadas;
- núcleo pasivo en SPY.

Los overlays disponibles son:

- `full`: 100 % del presupuesto asignable al satélite;
- `fixed_50`: 50 % activo y 50 % SPY;
- `binary`: activo o SPY según salud de señal;
- `continuous`: exposición proporcional a la salud de señal.

La salud solo utiliza cohortes cerradas. El efectivo activo no asignado se destina al benchmark.

### 9.3 Sizing

- `equal`: equiponderación;
- `score_linear`: escala lineal del score como comparador;
- `calibrated_alpha`: sizing basado en retorno excedente esperado calibrado.

El hurdle exige cero, uno o dos costes completos antes de asignar presupuesto a una posición.

### 9.4 Costes

Las comisiones permitidas son 0, 5 o 10 bps en Exploratory. El slippage permitido es 5, 10 o 20
bps. Confirmatory añade el caso severo 15/30 bps como stress fijo.

Los costes se aplican al nocional de cada orden. La curva de equity incorpora costes iniciales y
posteriores.

### 9.5 Invariantes contables

El estado conserva unidades, precios, efectivo y lotes. En cada snapshot:

- los holdings cambian únicamente mediante órdenes;
- el mark-to-market deriva los pesos por variación de precios;
- no se reajustan pesos objetivo gratuitamente;
- equity es efectivo más valor de posiciones;
- no se introduce apalancamiento implícito;
- las neutralizaciones por precios inválidos quedan reflejadas.

## 10. Catálogo científico cerrado

`module/studies/catalog.py` es la única fuente de opciones científicas. La versión actual es 1.
Cada variable declara identificador, descripción, etapa, valores, recomendado, coste, artefacto
invalidado, dependencias, orden y regla de simplicidad.

| Etapa | Variable | Valores permitidos | Recomendado |
|---|---|---|---|
| Temporal | `snapshot_step_months` | 1, 3, 6, 12 | 1 |
| Temporal | `target_horizon_months` | 3, 6, 12 | 12 |
| Temporal | `train_lookback_years` | 4, 8, 12 | 8 |
| Temporal | `execution_lag_days` | 30, 45, 60 | 60 |
| Temporal | `recency_weighting` | off, linear, exponential | off |
| Temporal | `objective` | rank_regression, ranking | rank_regression |
| Representación | `feature_preset` | presets declarados | core |
| Representación | `fundamental_momentum` | false, true | false |
| Representación | `market_regime_feature` | false, true | false |
| Representación | `neutralize_by_sector` | false, true | false |
| Representación | `winsorization` | 0, 0.01, 0.025 | 0 |
| Representación | `max_features_per_agent` | 8, 12, 20 | 8 |
| Modelo | `model_family` | lightgbm, elastic_net | lightgbm |
| Modelo | `use_quality_agent` | false, true | true |
| Modelo | `use_value_agent` | false, true | true |
| Modelo | `use_growth_agent` | false, true | true |
| Modelo | `use_momentum_agent` | false, true | true |
| Modelo | `use_risk_agent` | false, true | true |
| Modelo | `lgbm_max_depth` | 3, 4, 6 | 3 |
| Modelo | `lgbm_n_estimators` | 100, 200, 400 | 100 |
| Modelo | `lgbm_learning_rate` | 0.03, 0.05, 0.10 | 0.05 |
| Modelo | `lgbm_min_child_samples` | 20, 50, 100 | 50 |
| Modelo | `feature_weighting_mode` | model_native, oos_stability_prune | oos_stability_prune |
| Meta | `meta_method` | equal, rank_ic, stacked_rolling, stacked_exponential | equal |
| Meta | `meta_history_quarters` | 8, 16 | 16 |
| Meta | `meta_weight_cap` | 0.50, 0.75, 1.00 | 1.00 |
| Meta | `meta_equal_shrinkage` | 0, 0.25, 0.50 | 0 |
| Meta | `meta_half_life_quarters` | 4, 8 | 8 |
| Cartera | `portfolio_structure` | quarterly, four_vintages | four_vintages |
| Cartera | `investor_profile` | balanced, growth, value, quality, momentum, contrarian, defensive, garp | balanced |
| Cartera | `target_size` | 8, 12, 16 | 12 |
| Cartera | `sizing_mode` | equal, score_linear, calibrated_alpha | equal |
| Cartera | `active_overlay` | full, fixed_50, binary, continuous | full |
| Cartera | `cost_hurdle` | 0, 1, 2 | 0 |
| Cartera | `commission_bps` | 0, 5, 10 | 5 |
| Cartera | `slippage_bps` | 5, 10, 20 | 10 |

No se aceptan claves desconocidas, valores libres, JSON científico arbitrario ni cambios de orden.
Nombre y nota son los únicos textos libres y no afectan al cálculo.

La cadencia de snapshots puede ser mensual, trimestral, semestral o anual. Se puede fijar u
optimizar desde el dashboard igual que el resto de ejes. La única restricción es física y se valida
antes de ejecutar: el horizonte debe contener un número entero y positivo de snapshots; por ejemplo,
un horizonte de 3 meses no puede evaluarse sobre snapshots semestrales.

## 11. Definición y presupuesto

Cada variable activa tiene uno de dos modos vigentes:

- `fixed`: exactamente un valor;
- `optimize`: entre dos valores y el máximo declarado por la variable.

`disabled` no se utiliza en la versión 1: la opcionalidad se expresa con valores booleanos
cerrados. Los parámetros dependientes quedan inactivos si no aplica su controlador. Un parámetro
dependiente solo se puede optimizar si el controlador está fijo, garantizando un presupuesto
determinista.

La fórmula exacta es:

```text
evaluaciones exploratorias =
1 baseline
+ suma de valores seleccionados de cada variable optimize activa
```

No es un producto cartesiano. Si se optimizan horizonte con tres valores y meta con dos:

```text
1 + 3 + 2 = 6 evaluaciones
```

El ganador del horizonte se usa en las dos evaluaciones del meta.

No hay límite global de evaluaciones exploratorias, fits caros ni disco incremental. El preflight
los calcula y el dashboard los muestra antes de lanzar para que el usuario decida el alcance.
Confirmatory sí conserva 23 evaluaciones porque es un protocolo fijo de corroboración. El catálogo
mantiene máximos por variable y dependencias explícitas para que toda comparación siga siendo
finita, reproducible y científicamente interpretable.

La estimación vigente usa 35 minutos por fit, 3 por recombinación meta y 2 por backtest. El disco
se estima con una base de 200 MiB, 400 MiB por fit, 20 MiB por meta y 10 MiB por backtest. Son
estimaciones conservadoras de preflight, no mediciones de runtime.

### 11.1 Configuración recomendada de apertura

El dashboard no se abre con una exploración vacía ni con todos los ejes activados. Carga una
recomendación explícita:

| Variable | Valores |
|---|---|
| Horizonte | 6, 12 |
| Lookback | 8, 12 |
| Recencia | off, linear |
| Preset de features | core, fundamental, all |
| Meta-agente | equal, rank_ic, stacked_rolling |
| Estructura | quarterly, four_vintages |
| Sizing | equal, calibrated_alpha |
| Overlay | full, fixed_50, continuous |

El resto permanece fijo en el valor recomendado del catálogo. El presupuesto resultante es 20
evaluaciones exploratorias, exactamente 10 fits caros y 43 evaluaciones para el ciclo completo;
es una recomendación inicial, no un máximo impuesto.
La lógica es dedicar el presupuesto de fit a horizonte, longitud de historia, adaptación temporal
y representación; después recombinar esos scores y probar la traducción a cartera con evaluaciones
baratas. No se optimizan simultáneamente hiperparámetros finos, agentes, costes ni transformaciones
aisladas porque diluirían la hipótesis y agotarían el presupuesto.

## 12. Ejecución exploratoria secuencial

### 12.1 Máquina de estados

```text
draft → running → awaiting_decision → running
      → ... → awaiting_freeze → succeeded
```

1. Preflight normaliza y valida la definición.
2. Se crea el baseline con todos los valores iniciales.
3. Se toma la siguiente variable optimizada según el orden fijo.
4. Se evalúan todos sus valores contra el baseline acumulado.
5. El sistema recomienda un candidato.
6. El usuario acepta o realiza una intervención catalogada.
7. El elegido se convierte en baseline.
8. Se descartan resúmenes no seleccionados y se podan datasets no referenciados.
9. Al terminar todas las variables, se habilita la congelación.

### 12.2 Orden fijo

1. Objetivo temporal.
2. Representación.
3. Modelo y agentes.
4. Meta-agente.
5. Cartera.

El orden forma parte del procedimiento y afecta al resultado. No se presenta la búsqueda secuencial
como equivalente a un óptimo global; se adopta por interpretabilidad, coste y trazabilidad.

### 12.3 Gate de señal

Respecto al baseline del paso, un candidato de señal exige simultáneamente:

- Rank-IC medio no peor en más de 0,005;
- fracción positiva no peor en más de 3 puntos porcentuales;
- Rank-IC de cada era superior a −0,02;
- límite inferior del bootstrap pareado al 90 % superior a −0,01;
- spread de cola no inferior.

Entre elegibles se priorizan mediana por eras, spread de cola, Rank-IC medio, menor variabilidad y
menor complejidad catalogada.

### 12.4 Gate de cartera

Un candidato de cartera exige:

- alfa positivo en al menos dos eras;
- Information Ratio no inferior al baseline;
- turnover anualizado menor o igual que 200 %;
- peor era no peor que el baseline en más de 2 puntos porcentuales.

El desempate prioriza Information Ratio, menor turnover y menor complejidad.

### 12.5 Intervención humana

El usuario puede apartarse de la recomendación solo dentro de la comparación actual. Debe elegir:

- mayor simplicidad;
- menor coste computacional;
- menor turnover;
- mayor estabilidad;
- restricción metodológica.

El ledger registra candidato automático, elegido, motivo y `human_override`. No existe texto libre
que altere la ejecución.

### 12.6 Limitación que debe declararse

El baseline de cada paso es el ganador anterior. Por ello, los tests de varios pasos no constituyen
comparaciones independientes y existe riesgo de path dependence. La defensa metodológica es la
separación posterior mediante Confirmatory, no afirmar que Exploratory elimina el sesgo de
selección.

## 13. Confirmatory: 23 evaluaciones exactas

| Grupo | Cantidad | Qué comprueba |
|---|---:|---|
| Semillas 7 y 2026 | 2 | Sensibilidad del ajuste |
| Ocho perfiles | 8 | Comportamiento descriptivo por estilo |
| Cuatro niveles de costes | 4 | Viabilidad económica |
| Calendario +1 mes | 1 | Dependencia de fecha de entrada |
| Cinco placebos reentrenados | 5 | Señales de fuga o aprendizaje espurio |
| Bootstrap y exclusión de eras | 1 | Incertidumbre temporal |
| Permutación transversal | 1 | Significancia de la ordenación congelada |
| Carteras aleatorias PIT | 1 | Comparación contra selección aleatoria |
| **Total** | **23** | Protocolo cerrado |

### 13.1 Semillas

Se repite el pipeline con semillas 7 y 2026. No se elige la mejor semilla. Se estudia si la señal
se sostiene frente a variación estocástica del ajuste.

### 13.2 Perfiles

Exploratory puede fijar u optimizar un **perfil base** de cartera como parte de la hipótesis. Ese
perfil reordena la cola superior del meta antes de construir las posiciones. La selección exige que
los agentes que utiliza estén explícitamente activos. Una vez congelada la hipótesis, Confirmatory
no sustituye ese perfil: ejecuta los ocho perfiles siguientes como diagnósticos descriptivos.

Los perfiles son:

- balanced;
- growth;
- value;
- quality;
- momentum;
- contrarian;
- defensive;
- garp.

Todos parten de los mismos scores. Excepto balanced, reordenan solo acciones cuyo `meta_rank` es al
menos 0,60. Son diagnósticos descriptivos y no pueden modificar la hipótesis. Si el modelo carece
de un agente requerido, el perfil se registra como `not_applicable`; no se deforma silenciosamente.

### 13.3 Costes

Se evalúan pares comisión/slippage:

- 0/5 bps;
- 5/10 bps;
- 10/20 bps;
- 15/30 bps.

El caso 10/20 participa en el gate del veredicto. El resto caracteriza sensibilidad.

### 13.4 Calendario

Se desplaza un mes el calendario de vintages. La finalidad es detectar resultados dependientes de
una fecha concreta de entrada.

### 13.5 Placebos

Cinco reentrenamientos usan etiquetas barajadas con semillas 101–105. Se informa media, rango y
diferencia frente al real. No se interpreta cinco como tamaño suficiente para un p-valor preciso.
El real debe superar el máximo del rango placebo para el chequeo descriptivo de fuga.

### 13.6 Bootstrap y exclusión de eras

Se usa bootstrap por bloques sobre el Rank-IC temporal y se informa intervalo al 95 %. También se
recalcula la media excluyendo cada era. La unidad conjunta ocupa una sola fila del presupuesto,
aunque produzca varias estadísticas.

### 13.7 Permutación

Se realizan 9.999 permutaciones dentro de cohorte y se aplica corrección add-one:

```text
p = (excedencias + 1) / (9.999 + 1)
```

Por construcción, el p-valor nunca es cero. El veredicto exige `p ≤ 0,10`.

### 13.8 Carteras aleatorias

La evaluación contiene 1.000 simulaciones para dos nulos:

- general;
- emparejado por riesgo.

El modelo debe alcanzar al menos el percentil 95 en ambos. La implementación vigente resume
retornos por año y construye pools observados por ticker-año. Esta aproximación debe describirse
como un nulo PIT simplificado, no como una réplica orden por orden del motor de vintages. Mejorarlo
para reproducir exactamente calendario, exposición y costes es una línea de refuerzo futuro.

## 14. Veredictos

### 14.1 `confirmed`

La señal supera semillas, bootstrap, permutación y placebos; la cartera tiene alfa positivo en al
menos dos eras, soporta costes altos, turnover ≤200 % y supera ambos nulos aleatorios.

### 14.2 `signal_only`

La evidencia predictiva supera los criterios, pero la traducción a cartera no es robusta.

### 14.3 `non_inferior`

La señal es no inferior al baseline exploratorio, el bootstrap es compatible con no inferioridad,
los costes altos no producen IR negativo y el turnover respeta el límite. Este veredicto no debe
describirse como superioridad.

### 14.4 `rejected`

Falla la señal o una condición esencial de robustez y tampoco se cumple no inferioridad.

Los modelos con `confirmed` o `non_inferior` conservan evidencia final. `signal_only` y `rejected`
mantienen el study y su decisión, pero no se promocionan como modelo final.

## 15. Almacenamiento y trazabilidad

### 15.1 Estructura

```text
data/raw/                         fuentes originales
data/prepared/<dataset_hash>/     dataset PIT compartido
data/cache/                       fits y resúmenes por contenido
results/studies/<study_id>/       estado, ledger y decisión
results/hypotheses/<hyp_id>/      configuración y evidencia congelada
results/models/<model_id>/        evidencia final confirmatoria
```

### 15.2 Candidatos descartados

No tienen directorio propio. Cada evaluación exploratoria guarda una fila con:

- número;
- etapa y variable;
- valor candidato;
- configuración serializada;
- métricas y eras;
- hashes;
- origen computed/cached;
- tiempo;
- selección y motivo.

### 15.3 Caché

La caché es direccionable por contenido. Los hashes incorporan configuración, seed, perfil,
overrides, identidad del target y versión del runner. El límite lógico es 5 GiB. La poda conserva
entradas referenciadas por hipótesis y modelos y elimina solo entradas no protegidas.

### 15.4 Evidencia retenida

Una hipótesis conserva menos de 100 MiB; un modelo final tiene objetivo de menos de 250 MiB. La
evidencia incluye scores, pesos meta, diagnósticos, atribuciones, equity, métricas anuales, órdenes,
posiciones, vintages, exposición y referencia al dataset.

### 15.5 Fallos

El trabajo pesado se realiza en directorios temporales y se elimina al finalizar. Si la congelación
falla, se elimina la hipótesis incompleta. Los studies conservan estados y errores a nivel de API.
Los jobs HTTP son actualmente memoria local: reiniciar el servidor pierde el estado del job, aunque
los estudios ya persistidos permanecen en disco. No hay todavía un scheduler persistente ni una
reanudación automática de una evaluación interrumpida.

## 16. API y dashboard

### 16.1 API

| Método | Ruta | Función |
|---|---|---|
| GET | `/api/catalog` | Catálogo y límites |
| POST | `/api/exploratory/preflight` | Validación y presupuesto |
| POST | `/api/exploratory` | Inicio del estudio |
| POST | `/api/exploratory/{id}/advance` | Decisión y siguiente variable |
| POST | `/api/exploratory/{id}/freeze` | Congelación |
| POST | `/api/confirmatory/preflight` | Validación de hipótesis |
| POST | `/api/confirmatory` | Corroboración |
| GET | `/api/studies` | Lista de estudios |
| GET | `/api/studies/{id}` | Estado, ledger y decisión |
| GET | `/api/hypotheses` | Hipótesis congeladas |
| GET | `/api/models` | Modelos finales |
| GET | `/api/storage` | Uso de results, cache y prepared |
| GET | `/api/entities/{id}/performance` | Rentabilidad |
| GET | `/api/entities/{id}/learning` | Aprendizaje y diagnósticos |
| GET | `/api/entities/{id}/rankings` | Ranking por snapshot |
| GET | `/api/entities/{id}/portfolio` | Posiciones, vintages y exposición |
| GET | `/api/entities/{id}/trades` | Órdenes |
| GET | `/api/entities/{id}/stocks/{ticker}` | Historia de una acción |

Los POST de ejecución devuelven un `job_id`; `/api/jobs/{id}` permite consultar su estado.

### 16.2 Vistas

La app ofrece:

- configuración de nuevo Exploratory;
- estudios;
- hipótesis;
- análisis de rentabilidad;
- aprendizaje;
- rankings;
- cartera;
- trades;
- stocks.

Las vistas completas solo están disponibles para hipótesis y modelos con evidencia. Los
descartados aparecen como resumen del ledger, evitando duplicar datos.

## 17. Arquitectura de código

```text
module/data/          ingesta, universo, dataset y baselines
module/modeling/      catálogo de features, targets, agentes y meta
module/evaluation/    diagnósticos, estadísticas, perfiles, cartera y backtest
module/studies/       catálogo, configuración, runner, exploración y confirmación
module/storage/       datasets, caché y evidencia
module/web/           API y consultas
app/                  interfaz HTML/CSS/JavaScript
```

Existe un solo runner científico: `run_evaluation`. Exploratory y Confirmatory lo componen. El
orquestador no implementa modelos ni contabilidad; la app no lee Parquet directamente.

## 18. Reproducibilidad

Una afirmación de resultados debe poder vincularse con:

- `study_id`;
- `hypothesis_id`;
- `model_id`, si existe;
- `catalog_version`;
- configuración completa;
- `dataset_hash`;
- `evaluation_key`;
- seeds;
- ledger;
- decisión;
- evidencia retenida.

La versión 1 aún no persiste automáticamente un hash Git del código en la hipótesis. Antes de usar
una ejecución como evidencia definitiva del TFM debe añadirse o registrarse manualmente el commit
exacto en la bitácora. Esta carencia no debe ocultarse.

## 19. Validación automatizada actual

La suite mantiene 19 tests distribuidos en tres contratos:

- catálogo, valores y presupuesto;
- workflow, hipótesis, exclusión temporal y UTF-8;
- economía: p-valor add-one, overlay y vintages.

Comandos:

```powershell
python -m pytest -q
python -m ruff check .
node --check app/js/app.js
```

Los tests garantizan contratos de software, no rentabilidad. Un test sintético exitoso no sustituye
un estudio real.

## 20. Amenazas a la validez

1. **Selección secuencial:** el resultado depende del orden y puede perder interacciones.
2. **Múltiples comparaciones:** los gates reducen, pero no eliminan, el sesgo exploratorio.
3. **Cobertura histórica:** calidad y disponibilidad de fuentes pueden variar por época.
4. **Sesgo de universo:** la membresía histórica debe auditarse en cada snapshot.
5. **Modelo de costes:** bps fijos no capturan toda la capacidad o impacto de mercado.
6. **Solapamiento de etiquetas:** horizontes largos inducen dependencia temporal; de ahí los
   bloques.
7. **Nulo aleatorio simplificado:** la simulación actual no reproduce aún todo el libro de órdenes.
8. **Perfiles:** el perfil base participa en Exploratory y, por tanto, comparte su riesgo de
   selección; las repeticiones de perfiles en Confirmatory son diagnósticos de estilo y no cambian
   el veredicto.
9. **Estrés conocido:** 2025–2026 no es un holdout nuevo.
10. **Ausencia de forward live:** la prueba más fuerte será operar o paper-tradear después de
    congelar.

## 21. Uso para la memoria del TFM

Este documento puede alimentar la memoria con la siguiente correspondencia:

| Capítulo del TFM | Secciones fuente |
|---|---|
| Motivación y pregunta | 1–2 |
| Datos y prevención de sesgos | 3–4 |
| Factores y modelos | 5–7 |
| Métricas predictivas | 8 |
| Construcción de cartera | 9 |
| Diseño experimental | 10–13 |
| Criterios de decisión | 14 |
| Reproducibilidad e implementación | 15–19 |
| Limitaciones | 20 |

Las cifras empíricas se tomarán únicamente de `docs/informe_resultados.md` una vez vinculadas a
artefactos reales. La narración cronológica y las decisiones se tomarán de `docs/bitacora.md`.
