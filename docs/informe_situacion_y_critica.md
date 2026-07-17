# Informe 2 — Situación del proyecto, qué aprende el sistema, y valoración crítica

*Fecha: 2026-07-17. Rama `fresh-start`. Este informe es deliberadamente crítico, a petición
del autor: el objetivo no es solo un TFM defendible, sino un sistema que dé resultados reales.*

## 1. Qué se ha construido

El proyecto es un **selector de acciones multiagente** con un pipeline completo y probado
(51 tests, ~3 250 líneas en `module/`), organizado en etapas encadenadas:

```
descarga → dataset point-in-time → features → agentes ML → cartera → backtest → informe
                                                                         + barrido de escenarios
```

- **Datos (Fase 0).** Universo dinámico del S&P 500 por fecha (composición histórica real, no
  la actual), con fechas de publicación **reales** de SEC EDGAR para evitar lookahead, y el
  sesgo de supervivencia **medido por año** (no solo declarado).
- **Dataset point-in-time (Fase 1).** Panel `(ticker, fecha)` que reconstruye lo observable en
  cada fecha: de cada empresa, su último informe *realmente publicado*.
- **Features y baselines (Fase 2).** Factores GARP (calidad, crecimiento, valoración) y
  momentum relativo, todos rankeados en el corte transversal.
- **Agentes + meta-agente (Fase 3).** Tres agentes Ridge (calidad, momentum, valor) entrenados
  walk-forward; un meta-agente los pondera por su rank-IC reciente.
- **Cartera y backtest (Fase 4).** Reglas de rotación con umbral de ventaja y expulsión, costes
  y alfa neta.
- **Informes y barrido (Fases 5-6).** HTML navegables por run y comparación de escenarios con
  selección automática por **aprendizaje** (rank-IC), no por alfa.

La calidad de ingeniería es alta: separación temporal estricta, tests de leakage, huellas de
reproducibilidad, todo point-in-time. **El problema no está en cómo está hecho, sino en lo que
el sistema es capaz de aprender con estos ingredientes.**

## 2. Qué hacen exactamente los agentes y qué "aprende" la IA

Cada agente es una **regresión lineal Ridge** que, en cada trimestre, mira las empresas de los
últimos 8 años y ajusta unos pesos: "dado el ranking de una empresa en estos factores, ¿cuál
fue su exceso de retorno a 3 meses?". Con esos pesos puntúa las empresas del momento. El
meta-agente combina los tres según cuál ha acertado más recientemente (rank-IC).

Lo que "aprende", por tanto, es **una combinación lineal de factores contables y de precio que
intenta ordenar las acciones de mejor a peor retorno futuro**. La evidencia de si aprende o no
es el **rank-IC**: la correlación de rangos entre lo que el modelo predijo y lo que de verdad
pasó, fuera de muestra.

## 3. Qué dicen los números, sin adornos

**El sistema no aprende a ordenar activos de forma útil.** Tras todo el estudio de mejora:

- rank-IC medio fuera de muestra: **+0.0015** (la mejor configuración). El azar es 0.
- cohortes en las que acierta el orden: **53.6 %**. El azar es 50 %.
- por agente: value +0.013, quality +0.004, momentum −0.011.

Un rank-IC de 0.0015 es, a efectos prácticos, **cero**. Para comparar: en la industria se
considera que un factor tiene valor a partir de rank-IC ~0.03-0.05 sostenido. Estamos un orden
de magnitud por debajo. Y en el barrido completo, el "ganador" por estas métricas cambiaba de
comportamiento por completo entre periodos: **parecía elegido al azar**, que es justo lo que el
autor no quiere.

**Esto no es un bug.** El pipeline está probado y es correcto. Es la respuesta empírica honesta:
con estos datos, este universo y este modelo, la señal no está.

## 4. Por qué pasa esto — diagnóstico crítico

Cuatro razones, de más a menos de fondo:

1. **La señal factor→retorno es genuinamente débil a este horizonte.** Que el momentum de
   fundamentales (B3) y la neutralización por sector (B1) —ideas con fundamento— empeoren el
   resultado confirma que el problema no es de features. Añadir variables a un modelo lineal sin
   señal solo añade ruido y sobreajuste.

2. **El universo es pequeño y sesgado.** Solo ~50 % de las empresas del índice en 2000 tienen
   datos (el resto quebraron o fueron absorbidas y no están en fuentes gratuitas). El backtest
   pre-2010 está sesgado al alza, y los cortes transversales tienen pocos nombres, lo que hace
   el rank-IC ruidoso.

3. **El modelo lineal (Ridge) no captura la realidad.** La relación entre "estar barato" y
   "subir" no es lineal ni estable: depende del régimen, del sector, de interacciones. Un modelo
   lineal promedia todo eso a casi nada. Las interacciones que metimos a mano (B5) ayudaron un
   pelo, señal de que ahí hay algo, pero un lineal no puede explotarlo bien.

4. **La etiqueta es muy ruidosa.** El exceso de retorno a 3 meses es casi todo ruido de mercado;
   la parte explicable por fundamentales es una fracción diminuta. Por eso "entrenar contra el
   orden" (B2) fue lo único que ayudó: reduce el peso del ruido, pero no lo elimina.

## 5. Valoración: ¿seguir, simplificar o cambiar de camino?

El autor ha pedido resultados reales, no un rank-IC de cero. Con honestidad: **por la vía actual
no van a llegar.** Se ha exprimido el enfoque lineal+GARP y su techo es este. Hay tres caminos, y
recomiendo una combinación del primero y el tercero.

### Camino A — Reencuadrar el TFM (recomendado como base, coste bajo)

El resultado negativo, **bien medido, ya es un TFM válido y defendible**: "un sistema
multiagente riguroso demuestra que los factores GARP+momentum lineales no predicen el orden de
retornos fuera de muestra en el S&P 500 con datos abiertos, y cuantifica por qué". Esto es
honesto, reproducible y tiene valor académico. La bitácora (`docs/bitacora.md`) es exactamente el
material de este relato. **Pero por sí solo no da "resultados" en el sentido que pide el autor.**

### Camino B — Simplificar y buscar un resultado modesto pero real (coste bajo-medio)

En vez de perseguir un modelo ML que aprenda, **medir si un factor simple y bien conocido
funciona en este universo**: p. ej., una cartera de momentum a 12 meses o de calidad
determinista, sin ML, comparada honestamente contra el índice con costes. Puede que un factor
clásico sí dé un alfa pequeño y estable. Sería un resultado real (aunque no "de IA"), y el
sistema ya tiene toda la maquinaria de cartera y backtest para medirlo. **Riesgo:** puede que
tampoco funcione, pero al menos la pregunta es más contestable.

### Camino C — Cambiar el motor de aprendizaje y/o el objetivo (coste medio-alto, más potencial)

Si se quiere insistir en la parte de IA con opción real de señal:

- **Modelo no lineal**: sustituir Ridge por *gradient boosting* (LightGBM/XGBoost). Captura
  interacciones y no-linealidades que el lineal promedia a cero. Es el cambio con más potencial
  de mover el rank-IC. Coste: una dependencia nueva y cuidado con el sobreajuste (pocas eras).
- **Objetivo distinto y más aprendible**: en vez de predecir el exceso de retorno (ruidosísimo),
  predecir algo con más estructura — p. ej. **clasificar** el cuartil superior vs. inferior
  (problema más fácil que la regresión), o predecir a horizonte más largo la *dirección* relativa.
- **Universo distinto**: un mercado más amplio y mejor cubierto (p. ej. todo el US large+mid cap
  vía otra fuente) reduce el sesgo de supervivencia y da cortes transversales más grandes, donde
  el rank-IC es más estable y medible.

### Qué NO recomiendo

Seguir añadiendo features al Ridge. El estudio B lo ha dejado claro: no es el camino.

## 6. Recomendación concreta

1. **Conservar todo lo hecho**: es infraestructura sólida y el resultado negativo es publicable.
2. **Dar un paso barato con potencial real**: probar **LightGBM** en lugar de Ridge (Camino C,
   primer punto) y, en paralelo, **clasificación de cuartiles** en vez de regresión de retorno.
   Ambos reutilizan casi todo el pipeline. Medir con la misma vara (rank-IC OOS). Si el rank-IC
   sube de forma clara y estable, hay TFM con resultados. Si no, se cierra con el Camino A, que
   sigue siendo un buen trabajo.
3. **Decidir con el autor** si el TFM prioriza "demostrar rigor sobre un resultado negativo"
   (A), "un factor simple que funcione" (B) o "hacer que la IA aprenda de verdad" (C). Son tres
   TFM distintos y honestos; el esfuerzo restante depende de cuál se elija.

**En una frase:** el proyecto está bien construido y ha producido un hallazgo honesto —el
enfoque actual no aprende—; para tener resultados de verdad hay que cambiar el motor de
aprendizaje o el objetivo, no seguir afinando lo que ya toca techo.
