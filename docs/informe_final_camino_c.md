# Informe final — Camino C (LightGBM) y resultado del sistema optimizado

*Fecha: 2026-07-18. Rama `fresh-start`. Universo `full` (680 tickers con datos, S&P 500
histórico). Este informe reporta el resultado del sistema tras cambiar el motor a LightGBM,
optimizar por etapas con puertas de decisión, y lanzar el run final. Es deliberadamente
crítico, como pidió el autor.*

## 1. Qué se hizo

Tras concluir que el enfoque Ridge lineal alcanzaba un techo de rank-IC ≈ 0, se implementó el
Camino C con rigor metodológico por etapas:

- **Motor no lineal**: LightGBM (árboles con gradient boosting) frente a Ridge.
- **Objetivo alineado con el ranking**: `rank_regression` (regresión sobre el percentil
  transversal del retorno), con LGBMRanker y cuartiles como alternativas.
- **Corrección metodológica clave**: se diagnostica el **meta_final** (el score que opera la
  cartera), no el promedio de los agentes. Antes se medía la métrica equivocada.
- **Métricas económicas correctas**: CAGR real en vez de acumulados inflados.
- **Significancia estadística**: bootstrap por bloques temporales + diferencia pareada vs Ridge.
- **Optimización por etapas con puertas** (A motor/objetivo → B regularización → C meta →
  período → D cartera), sin optimizar todo a la vez.

## 2. Evidencia de aprendizaje (rank-IC) — la métrica que importa

| modelo | rank-IC meta_final | frac cohortes>0 | ¿distinto de 0? |
|---|---|---|---|
| **lightgbm / rank_regression / depth 4** | **+0.0117** | 56.8 % | no (IC cruza 0) |
| ridge / regression (control) | +0.0065 | 53.3 % | no |
| lightgbm / quartile | +0.0059 | 53.3 % | no |
| lightgbm / ranking (lambdarank) | +0.0009 | 51.7 % | no |

**LightGBM+rank_regression casi dobla el rank-IC de Ridge** (+0.0117 vs +0.0065) y es robusto a
la semilla (+0.0088 a +0.0117 en 4 semillas). El meta ponderado por rank-IC bate al equiponderado
(+0.0117 vs +0.0054): **la combinación aporta**. Concentrando en años recientes (2014+) sube a
**+0.0178, frac 63.5 %**, coherente con la mayor cobertura.

**Pero**: un rank-IC de +0.0117 es, a efectos prácticos, **cero**. La industria considera útil un
factor a partir de ~0.03-0.05 sostenido. Estamos un orden de magnitud por debajo, y **la mejora
no es estadísticamente distinguible de cero** (el bootstrap cruza cero; la diferencia pareada vs
Ridge no es significativa). LightGBM mejora a Ridge de forma direccional y consistente, pero
demasiado débil para ser fiable.

## 3. Rendimiento económico del sistema optimizado — y por qué NO hay que creérselo

Sistema congelado: LightGBM/rank_regression/depth 4/meta rank_ic. Backtest 2000-2026:

| cartera | CAGR cartera | CAGR SPY | diferencia | beat_rate | drawdown máx |
|---|---|---|---|---|---|
| 5-10 posiciones | **18.6 %** | 8.4 % | +10.2 %/año | **48 %** | **71.7 %** |
| 3-7 posiciones | 18.5 % | 8.4 % | +10.1 %/año | 44 % | 72.9 % |
| 8-15 posiciones | 18.3 % | 8.4 % | +9.9 %/año | 48 % | 65.7 % |

A primera vista parece un éxito rotundo: **18.6 % anual vs 8.4 % del SPY**. Es falso, y las tres
cifras de la derecha lo delatan:

**a) El CAGR entero descansa sobre UN mes artefactual.** En julio de 2010 la cartera registró
un retorno mensual de **+953 %** (año 2010 completo: **+1277 %**). Un retorno así es físicamente
imposible para una cartera de acciones diversificada: es un **artefacto de datos** — una posición
con precio corrupto (split mal ajustado, ticker reciclado, o precio erróneo de la fuente
gratuita). El mismo tipo de problema de "ticker reciclado" que se detectó y filtró en la Fase 0,
pero que se coló en el backtest por otra vía.

**b) Sin ese año, el sistema es mediocre o peor.** La **alfa mediana anual es −0.2 %**: el año
típico, la cartera **pierde ligeramente** al SPY. Bate al índice solo **13 de 27 años (48 %)** —
menos de la mitad. Es exactamente el patrón que el autor quería evitar: *"prefiero superar todos
los años un 2 % que superarlo gracias a un +100 % en uno solo"*. Aquí es un +1277 % en uno solo.

**c) El drawdown del 72 %** es catastrófico e ininvertible: ningún inversor real mantiene una
estrategia que puede perder tres cuartas partes del capital.

**El rank-IC de +0.0117 es coherente con todo esto**: el sistema **no ordena las acciones mejor
que el azar**, así que su rentabilidad no puede venir de habilidad. Viene de concentración (pocas
posiciones) + un artefacto de datos + suerte. El CAGR alto y el rank-IC nulo no se contradicen:
**el rank-IC dice la verdad, el CAGR engaña.**

## 4. Conclusión honesta

1. **El Camino C mejoró la señal, pero no la resolvió.** LightGBM+rank_regression es
   consistentemente mejor que Ridge (casi el doble de rank-IC, robusto a la semilla, mejor en el
   período reciente), y la corrección de medir el meta_final reveló que la combinación de agentes
   sí aporta. Es un avance metodológico real. Pero el rank-IC sigue en ~0.01, indistinguible de
   cero: **el sistema no aprende a ordenar activos de forma económicamente fiable**.

2. **El rendimiento espectacular es un espejismo.** El 18.6 % anual es un artefacto de un dato
   corrupto en 2010; el sistema real tiene alfa mediana anual negativa y bate al SPY menos de la
   mitad de los años. Esto es, en sí mismo, un resultado valioso del TFM: **demuestra por qué no
   se debe seleccionar ni juzgar un sistema por su rentabilidad acumulada**, y por qué el rank-IC
   —que no se dejó engañar— es la métrica correcta.

3. **Lo que sí es defendible y publicable.** El trabajo produce una respuesta honesta y bien
   medida a la pregunta del TFM: con datos gratuitos, universo del S&P 500 sesgado por
   supervivencia, y factores GARP+momentum, ni el modelo lineal ni el no lineal aprenden a
   ordenar activos de forma estable y significativa. El sistema es riguroso (point-in-time,
   walk-forward, sin lookahead, 63 tests), la metodología de evaluación es sólida, y el hallazgo
   negativo está cuantificado con significancia estadística. Eso es un buen TFM.

## 5. Qué haría falta para tener resultados reales (más allá de este trabajo)

- **Limpiar el artefacto de 2010** y auditar el backtest contra saltos de precio imposibles
  (una guarda de retorno mensual máximo). No dará señal, pero dejará el CAGR honesto (~8-9 %,
  en línea con el SPY, coherente con rank-IC ≈ 0).
- **Datos de pago sin sesgo de supervivencia** y un universo mayor: cortes transversales más
  grandes darían un rank-IC medible con menos ruido.
- **Un objetivo o mercado donde exista señal explotable** con datos abiertos: quizá no sea el
  ranking de acciones del S&P 500 a 3 meses.

El éxito de este proyecto no es un alfa positivo que no existe, sino una respuesta creíble
—sostenida tras mostrar de forma transparente cómo influyen la cobertura, el período y los
artefactos de datos— a si un sistema de IA aprende a batir al mercado con datos abiertos. La
respuesta, medida con rigor, es que no de forma fiable.
