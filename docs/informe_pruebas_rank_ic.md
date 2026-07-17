# Informe 1 — Resultados de las pruebas de mejora del rank-IC

*Fecha: 2026-07-17. Rama `fresh-start`. Universo `full` (680 tickers con datos, S&P 500
histórico 1996-2026). Métrica: rank-IC OOS del walk-forward completo (942 cohortes).*

## Qué se medía y por qué

El barrido inicial reveló que el sistema tenía un **rank-IC fuera de muestra ≈ 0**: no ordenaba
los activos mejor que el azar. Como la pregunta del TFM es *si el sistema aprende a ordenar*, y
no *cuánto gana*, se decidió medir el aprendizaje directamente (rank-IC) y probar una serie de
palancas para subirlo, aceptando cada una **solo si mejoraba el rank-IC OOS** en todo el
walk-forward, no en un tramo. El alfa quedó fuera de la selección: con rank-IC nulo, cualquier
rentabilidad sería ruido afortunado.

**Baseline de referencia:** rank-IC medio **−0.0058**, fracción de cohortes con rank-IC positivo
**0.486** (el azar es 0.500). Por agente: momentum −0.007, quality −0.021, value +0.011.

## Resultados de cada palanca

| Palanca | Qué hace | rank-IC medio | cohortes IC>0 | Veredicto |
|---|---|---:|---:|---|
| **baseline** | factores GARP+momentum, etiqueta cruda | −0.0058 | 48.6 % | referencia |
| **B1** neutralización sector | rankea factores dentro de sector | −0.0083 | 46.7 % | **descartada** (empeora) |
| **B2** etiqueta *rank* | entrena contra el orden del retorno, no su valor | **+0.0011** | 53.4 % | **adoptada** |
| B2 etiqueta *winsor* | recorta colas de la etiqueta | −0.0021 | 51.1 % | descartada (mejora menos) |
| B2 horizonte 6m | etiqueta a 6 meses en vez de 3 | −0.0162 | 45.9 % | descartada (empeora) |
| **B3** momentum fundamentales | tendencia de ratios + descomposición P/E | −0.0046 | 51.3 % | **descartada** (empeora) |
| **B5** régimen de mercado | bull/bear + interacciones factor×régimen | **+0.0015** | 53.6 % | **conservada** (marginal) |

*(B4 fue un diagnóstico, no una palanca: la correlación de Spearman entre agentes es
quality↔value 0.51, momentum casi ortogonal 0.15. Momentum aporta la señal más independiente.)*

## Lectura

**La mejor configuración alcanzada es rank-IC +0.0015 con 53.6 % de cohortes ganadoras.** Se ha
pasado de "peor que el azar" a "azar más un pelo". La mejora es real pero minúscula, y está
dentro de lo que podría ser ruido.

Hay un patrón nítido y revelador:

- Las palancas que **añaden información nueva** (B1 sector, B3 momentum de fundamentales)
  **empeoran** el rank-IC.
- Las que **reordenan o limpian** lo que ya hay (B2 etiqueta rank, B5 régimen) **ayudan**, muy
  poco.

Esto dice que **el cuello de botella no es la falta de features**. Añadir variables a un modelo
lineal (Ridge) cuya señal factor→retorno es genuinamente débil no crea señal: añade dimensiones
donde sobreajustar, y cada feature ruidosa —especialmente las de cobertura parcial, como la
descomposición del P/E que solo cubre el 78 % del panel— degrada el orden fuera de muestra.

## Detalle por agente en la mejor configuración (B2+B5)

| agente | rank-IC medio | cohortes IC>0 |
|---|---:|---:|
| value | +0.0125 | 56.4 % |
| quality | +0.0035 | 55.6 % |
| momentum | −0.0113 | 48.9 % |

El régimen ayuda a los agentes **fundamentales**: quality cruza a positivo (venía de −0.021) y
value se afianza. Momentum, en cambio, empeora — el mercado no premia de forma estable seguir la
tendencia reciente en este universo. Es coherente con la literatura: el momentum puro es difícil
de explotar y muy dependiente del régimen.

## Un bug encontrado por el camino (relevante para la reproducibilidad)

La huella que identifica cada run de agentes (`_run_id`) no incluía el parámetro de tratamiento
de la etiqueta, así que dos configuraciones distintas (winsor y rank) escribían en el mismo
directorio y se sobreescribían en silencio. Se detectó al ver que dos pruebas daban idéntico
resultado. Corregido: la huella ahora cubre todo parámetro que altere el resultado. Es una
lección metodológica que merece figurar en el TFM — un experimento que se pisa a sí mismo
produce conclusiones falsas sin dar ningún error.

## Conclusión del informe

El estudio de palancas **no ha conseguido una señal de aprendizaje explotable**. El techo de
este enfoque —factores GARP+momentum, modelo lineal, universo del S&P 500 con datos gratuitos—
está en rank-IC ≈ 0. Esto es un resultado honesto y medido, no un fallo de implementación (el
pipeline está probado con 51 tests). Las implicaciones y los caminos alternativos para obtener
resultados de verdad se desarrollan en el **Informe 2**.
