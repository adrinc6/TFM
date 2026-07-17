# Plan maestro del TFM — Sistema multi-agente de IA aplicado a la bolsa

> Este documento es la **guía única** del proyecto. Describe qué se quiere
> conseguir, con qué metodología y con qué arquitectura, y sirve de hoja de ruta
> para reconstruir el sistema desde el reinicio en limpio. Ahora mismo el
> repositorio solo contiene la **descarga de datos**; todo lo demás descrito aquí
> está por construir. Es un documento vivo: se actualiza cuando cambie el diseño.

---

## 1. Propósito y tesis

Este es un Trabajo Fin de Máster sobre **Inteligencia Artificial y Machine
Learning**. El objetivo principal **no** es ganar dinero ni batir a un índice: es
**estudiar cómo aprende un sistema de IA**, evaluar ese aprendizaje con rigor y
comprobar si puede tener **utilidad económica**.

La bolsa es el **banco de pruebas**, no el fin. Se elige porque ofrece un entorno
difícil, ruidoso, no estacionario y con una métrica de éxito clara (batir al
S&P 500), pero el mérito académico está en el **método**, no en un número de
rentabilidad aislado.

De ahí una consecuencia importante para todo el proyecto: **un resultado negativo
bien medido es un entregable válido**. Si el sistema no aprende a ordenar activos
fuera de muestra, o si su alfa no viene de donde creíamos, decirlo con claridad y
con evidencia es tan valioso como una cartera rentable. La deshonestidad
metodológica (ocultar limitaciones, elegir la mejor semilla, confundir suerte con
aprendizaje) invalidaría el trabajo.

Principios rectores: **académico, reproducible, explicable y honesto**.

## 2. Pregunta de investigación

> **¿Aprende el sistema a ordenar activos fuera de muestra de forma estable a lo
> largo de muchas eras, y es ese aprendizaje útil?**

Se descompone en dos afirmaciones que el proyecto mantiene **separadas a
propósito**, porque no son lo mismo:

1. **¿El ML rankea?** ¿La ordenación que produce el modelo tiene poder predictivo
   fuera de muestra (p.ej. rank-IC positivo y estable por eras)?
2. **¿Hay alfa útil?** ¿Una cartera construida con esas señales bate al benchmark
   de forma neta (tras costes) y estable?

Puede haber alfa sin que el modelo rankee bien (por asimetrías ganador/perdedor,
por unos pocos aciertos grandes) y puede haber buen ranking sin alfa neto (si el
turnover se come la ventaja). El proyecto mide y reporta ambas cosas por separado.

## 3. Principios metodológicos (rigen todo el diseño)

Estos principios son **innegociables** y condicionan cada etapa:

1. **Sin lookahead ni fuga de información.** Una señal fechada en el día *t* solo
   puede usar datos **observables** en *t*. Los fundamentales no existen para el
   modelo hasta su fecha real de publicación (o hasta un retardo conservador).
2. **Point-in-time.** El dataset reconstruye, para cada fecha, exactamente lo que
   se sabía entonces: precios hasta *t*, último fundamental ya publicado, sin
   revisiones posteriores.
3. **Separación temporal train/eval.** Entrenar y evaluar nunca se solapan en el
   tiempo. El walk-forward reentrena solo con historia anterior a cada punto.
4. **Medición fuera de muestra.** La rentabilidad in-sample no es evidencia. Se
   reportan métricas OOS (rank-IC por era, resultados de la cartera en el periodo
   evaluado).
5. **Baselines y resultados negativos visibles.** Se comparan siempre contra
   baselines simples (comprar el índice, momentum puro, GARP determinista). Si un
   baseline gana, se dice.
6. **Sesgo de supervivencia y muestra pequeña como limitaciones explícitas.** El
   universo es una lista estática de grandes capitalizaciones actuales; el número
   de eras independientes es reducido. Nunca se ocultan; se cuantifican y se
   discuten.

## 4. Estrategia de inversión

El sistema persigue una o varias de estas familias de estrategia, y su
**combinación** es una de las variables a estudiar:

- **GARP (Growth At a Reasonable Price).** Comprar calidad y crecimiento sin
  pagar de más. Señales típicas: crecimiento de ventas/beneficios, ROE/ROIC,
  márgenes, deuda razonable, múltiplos (P/E, P/B, EV/FCF) no excesivos. Combina
  valor y calidad: penaliza lo caro y lo frágil.
- **Momentum.** Comprar lo que sube y evitar lo que cae. Señales: rentabilidad
  relativa a distintos horizontes (p.ej. 3/6/12 meses), fuerza frente al índice.
  En el trabajo previo el momentum puro resultó un baseline difícil de batir: es
  un competidor serio, no un adorno.
- **Combinación GARP + momentum.** Fundamentales para *qué* comprar (calidad a
  precio razonable) y momentum para *cuándo* (timing). La hipótesis es que se
  complementan; comprobarlo es parte del objeto de estudio.

La estrategia concreta (qué señales, qué pesos, qué umbrales) es una **decisión
metodológica** y no se fija a la ligera: cambiarla requiere aprobación explícita
(ver `CLAUDE.md`).

## 5. Diseño multi-agente

El corazón de IA/ML es un conjunto de **agentes especializados** más un
**meta-agente** que aprende a combinarlos:

- **Agentes especializados**, cada uno experto en una dimensión, por ejemplo:
  - *Calidad*: fundamentales de solidez (ROE/ROIC, márgenes, deuda).
  - *Timing / momentum*: señales de precio y fuerza relativa.
  - *Valor / alpha*: baratura ajustada a calidad, potencial de revalorización.
  Cada agente produce una **ordenación** (score) del universo en cada fecha.
- **Meta-agente**: aprende cómo **ponderar** a los agentes según su fiabilidad
  fuera de muestra (p.ej. por su rank-IC parcial reciente), en lugar de fijar los
  pesos a mano. Así el sistema puede fiarse más del agente que está funcionando y
  menos del que no.

El esquema debe ser **editable**: añadir, quitar o reponderar agentes tiene que
ser sencillo, porque los *ablations* (quitar un agente y ver qué pasa) son una de
las variables del barrido de escenarios. La elección de modelos, etiquetas,
objetivos y pesos requiere aprobación explícita.

## 6. Autonomía y walk-forward

El sistema debe poder **entrenarse y evaluarse solo** sobre toda la historia:

- **Datos desde el año 2000.** Se descarga historia larga para poder simular
  desde entonces (aunque la cartera opere en una ventana más corta).
- **Fecha ancla.** La simulación arranca en una fecha derivada de un **trimestre**
  configurable **más un retardo de publicación** de fundamentales. Ejemplo: arrancar
  en 2010Q1 + 45 días ≈ 15-feb-2010 garantiza que ya están publicados los
  resultados del trimestre anterior de todas las empresas, que es cuando cambian
  de verdad los ratios. Este retardo corrige el lookahead sutil de tratar un
  fundamental como conocido el mismo día del cierre del periodo.
- **Entrenar vs. revisar (separado a propósito):**
  - *Entrenar* (reajustar el modelo) solo tiene sentido cuando hay
    **fundamentales nuevos**: cadencia trimestral o anual. Reentrenar mensualmente
    no aporta (mismos fundamentales tres meses seguidos) y se descarta.
  - *Revisar* la cartera es **mensual**: se re-precia y se decide con el modelo ya
    entrenado, sin reentrenar.
- **Walk-forward rodante.** En cada punto de reentrenamiento se usa solo la
  historia disponible **hasta esa fecha** (una ventana móvil de los últimos N
  años), nunca datos futuros. El modelo se mantiene siempre entrenado con datos
  recientes, adaptándose a los cambios de régimen.

## 7. Barrido de escenarios (rejilla)

Para no depender de una sola configuración arbitraria, el sistema genera y evalúa
**muchos escenarios de forma sistemática**, cubriendo las variables relevantes.
Ejes previstos del barrido (a confirmar y ampliar):

- **Ventana de entrenamiento** (cuántos años de historia usa cada reentrenamiento).
- **Cadencia de reentrenamiento** (trimestral vs. anual).
- **Horizonte de la etiqueta** (a cuántos meses se define el objetivo a predecir).
- **Tamaño de cartera** (cuántas posiciones: breadth top-N).
- **Ablations de agentes y esquemas de pesos** (quitar agentes, fijar vs. aprender
  pesos).

Cada escenario ejecuta el pipeline completo reutilizando lo que se pueda (etapas
comunes, scoring cacheado) para que el barrido sea viable en tiempo. El resultado
es una tabla comparable de escenarios con sus métricas.

## 8. Selección del sistema final (lo importante)

El sistema final **no se elige por la mayor alfa**. Elegir el escenario con más
rentabilidad sobre el S&P 500 sería *overfitting por selección*: casi siempre el
ganador aparente es el más afortunado, no el más robusto.

La selección se hace **solo por aprendizaje y estabilidad, nunca por alfa**. La
pregunta del proyecto es si el sistema *aprende a ordenar activos fuera de muestra*;
si el aprendizaje (rank-IC) es débil, cualquier rentabilidad observada es en buena
parte suerte de composición, y elegir por ella sería seleccionar ruido.

La métrica de selección es el **rango medio de cuatro dimensiones**, ninguna de las
cuales es magnitud de rentabilidad:

- **rank-IC medio** fuera de muestra (evidencia de aprendizaje).
- **fracción de cohortes con rank-IC positivo** (estabilidad del aprendizaje entre
  eras: que no sea un pico afortunado).
- **beat rate**: fracción de años que baten al benchmark (frecuencia de acierto, no
  cuánto).
- **máximo drawdown** (riesgo).

El **alfa se reporta como consecuencia**, junto a los resultados, pero **no
interviene** en qué configuración se elige. La comparación es un **único ranking
global** sobre todos los años disponibles, sin separar en eras: se prefiere una
lectura honesta de la consistencia año a año antes que dividir la muestra (ya
pequeña) o quedarse con lo que mejor funcionó en un tramo. Si el rank-IC del ganador
es cercano a cero, la conclusión —válida— es que el sistema no aprende de forma
estable, y se dice con claridad.

## 9. Cartera y backtest

Sobre las señales del sistema se construye y simula una cartera realista:

- **Construcción de cartera**: selección de las mejores posiciones (watchlist →
  cartera), con tamaño de cartera configurable.
- **Sizing**: cómo se reparte el capital entre posiciones.
- **Rotación**: cuándo se sustituye una posición por otra mejor (umbrales de
  ventaja de score/convicción, coste de oportunidad, periodo mínimo de tenencia),
  controlando el turnover.
- **Costes**: comisiones de transacción y slippage, para medir alfa **neto**.
- **Simulación frente a benchmark** (S&P 500 / SPY) y **métricas**: rentabilidad,
  alfa, information ratio, tracking error, drawdown, t-stat de la alfa, además de
  las métricas de aprendizaje (rank-IC por era y por agente).

## 10. Visualización e informes HTML

El proyecto genera **informes HTML navegables** para ver y comparar resultados
sin tener que leer código o CSVs:

- Informe de un run: resumen ejecutivo, rendimiento, cartera, **evidencia de
  aprendizaje**, posiciones, metodología y depuración.
- Informe del barrido de escenarios: comparación de todos los escenarios para
  entender qué variables importan y sostener la selección del sistema final.

Los informes deben ser autocontenidos y honestos: muestran también los baselines
y los resultados negativos.

## 11. Editabilidad

El proyecto debe ser **fácil de modificar y experimentar**:

- Toda la configuración vive en `environment.py` (fechas, universo, tamaños,
  cadencias, umbrales), sin duplicados escondidos en el código.
- Añadir una variable al barrido, un agente nuevo o un baseline debe ser una
  operación acotada y local.
- Estilo de código sencillo, lineal y explicable (ver `CLAUDE.md`): se prima la
  claridad para poder revisar y justificar cada decisión en la memoria del TFM.

## 12. Datos

**Fuentes** (ya implementadas en la descarga, `module/ingest/`):

- **Finnhub** (`finnhub.io/api/v1`, requiere `FINNHUB_API_KEY`):
  - Perfil de empresa (nombre, sector, capitalización).
  - Fundamentales calculados y series (`/stock/metric`: P/E, P/B, ROE, ROIC,
    márgenes, crecimiento…).
  - Noticias por empresa (descargadas, aún sin procesar).
- **SEC EDGAR** (`data.sec.gov`, sin clave): fechas reales de presentación de los
  formularios 10-Q y 10-K. Es la fuente de `report_dates.parquet`; no se usa
  un retardo fijo ni el endpoint equivalente de Finnhub. Los CIK ambiguos por
  reutilización de ticker se validan con el buscador SEC.
- **Yahoo Finance** (API v8 de charts, sin clave): OHLCV diario, cierre ajustado,
  dividendos y splits. No se usa `yfinance`; se llama la API directamente.

**Universo**: composición histórica dinámica del S&P 500 desde el CSV de
componentes. En cada fecha se consideran solo las empresas que pertenecían al
índice entonces; el benchmark es SPY.

**Salidas de la descarga** (en `data/raw/`, no versionadas):

- Cache JSON por ticker: `data/raw/json/<fuente>/<ticker>/<dataset>.json`.
- Parquet agregados: `profiles.parquet`, `finnhub_metrics.parquet`,
  `prices.parquet`, `news.parquet`, `report_dates.parquet`.
- Metadatos: `download_coverage.json`, `download_failures.csv`,
  `universe_coverage.json`.

## 13. Estado actual vs. arquitectura objetivo

### Ejecución por etapas

La entrada única usa dos parámetros independientes: `RUN_MODE` selecciona la
etapa y `RUN_SCOPE` el alcance de los datos. `download` adquiere datos crudos;
las etapas disponibles son `dataset`, `features` y `agents`; `backtest`, `report`
y `experiments` siguen pendientes. `full` ejecuta en orden todas las etapas que ya estén
implementadas. Una etapa solicitada antes de existir falla explícitamente.

`RUN_SCOPE=dev` guarda agregados bajo `data/raw/dev/` y paneles bajo
`data/processed/dev/`; `RUN_SCOPE=full` usa las rutas sin ese subdirectorio.
Por tanto una verificación de desarrollo no sobrescribe una descarga completa.

**Lo que existe hoy** (tras el reinicio en limpio):

- `environment.py`: configuración, `RUN_MODE` y `RUN_SCOPE`.
- `main.py`: entrada única que selecciona una etapa o el flujo completo.
- `module/ingest/`: clientes HTTP de Finnhub, Yahoo y EDGAR, y `download_raw_data`.
- `module/dataset.py`: panel mensual, precio de activos y benchmark point-in-time.
- `module/features.py` y `module/baselines.py`: factores observables, baselines y etiquetas separadas.
- `module/agents.py` y `module/meta.py`: Ridge walk-forward, meta-pesos y rank-IC OOS.
- `module/utils.py`: utilidades de logging y escritura de archivos.

**Arquitectura objetivo por etapas** (a reconstruir sobre la descarga):

```text
download → dataset → features → ml (agentes) → selección → cartera → backtest → informe
                                        │
                              experimentos (rejilla) → agregación → selección del sistema final
```

- `download` (hecho): adquisición de datos crudos.
- `dataset` (hecho): panel point-in-time (ticker × fecha) con observabilidad por fecha de
  publicación; prevención de lookahead.
- `features` (hecho): variables GARP/momentum, baselines deterministas y etiquetas futuras separadas.
- `ml` (hecho): agentes especializados + meta-agente, entrenamiento walk-forward,
  combinación de señales y diagnóstico de aprendizaje.
- `selección`: watchlist a partir de los scores.
- `cartera` + `backtest`: construcción, sizing, rotación, costes, simulación y
  métricas frente al benchmark.
- `informe`: viewer HTML de un run.
- `experimentos`: barrido de escenarios, agregación y selección del sistema final
  por estabilidad multi-era.

## 14. Limitaciones explícitas

Se mantienen visibles y se discuten en la memoria:

- **Sesgo de supervivencia**: universo estático de líderes actuales aplicado hacia
  atrás; excluye deslistados/adquiridos e incluye OPVs recientes.
- **Muestra pequeña**: pocas eras verdaderamente independientes; los intervalos de
  confianza sobre la estabilidad son anchos.
- **Disociación alfa/ranking**: que haya alfa no demuestra que el ML rankee, y
  viceversa; el proyecto las reporta por separado.
- **Datos y cobertura**: dependencia de la cobertura de las APIs gratuitas
  (algunos fundamentales o fechas de publicación pueden faltar).

## 15. Roadmap de reconstrucción por fases

Orden recomendado para reconstruir sobre la descarga existente. Cada fase se
diseña y aprueba antes de implementarla (ver reglas en `CLAUDE.md`), y añade sus
propios tests (empezando por leakage y separación temporal):

1. **Dataset point-in-time**: panel ticker × fecha con observabilidad por fecha de
   publicación. Tests de leakage antes que nada.
2. **Features y baselines**: variables GARP y momentum + baselines deterministas
   (momentum puro, GARP determinista) contra los que comparar siempre.
3. **Agentes ML + meta-agente**: entrenamiento walk-forward, scoring, diagnóstico
   de aprendizaje (rank-IC OOS por era y por agente).
4. **Cartera y backtest**: selección, sizing, rotación, costes y métricas frente
   al benchmark.
5. **Informe HTML**: viewer de un run con la evidencia de aprendizaje.
6. **Experimentos (rejilla) + selección del sistema final**: barrido de
   escenarios, agregación por estabilidad y protocolo dev/confirmación.
7. **Redacción del TFM en LaTeX**: el documento entregable, escrito capítulo a
   capítulo sobre los resultados ya producidos por las fases 1-6. Plan de
   estructura en `latex/plan_tfm.md`.

Cada fase mantiene el flujo de datos limpio de entrada a salida y respeta los
principios metodológicos de la sección 3. El detalle ejecutable de cada fase
—ficheros, decisiones y tests— vive en `docs/plan_fases.md`.
