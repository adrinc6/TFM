# Memoria TFM

## 1. Resumen

Este trabajo desarrolla un sistema de seleccion de acciones long-only basado en un ensamblado multi-agente de aprendizaje automatico. El objetivo no es solo obtener una buena metrica de clasificacion, sino demostrar que una estrategia puede evaluarse de forma reproducible, sin leakage temporal, con comparacion justa frente a benchmark y baselines, y con una simulacion economica en USD suficientemente realista para una defensa academica.

## 2. Problema que resuelve

El problema central es seleccionar una cartera de acciones a partir de informacion fundamental, tecnica, de sentimiento, de insider activity y de valoracion, todo ello bajo un esquema walk-forward. El reto no es entrenar un clasificador aislado, sino construir una tuberia completa que:

- use informacion disponible en cada fecha de decision,
- evite incorporar datos futuros,
- produzca trazabilidad completa del run,
- y traduzca las predicciones en una cartera monetaria con costes y slippage.

## 3. Hipotesis de trabajo

La hipotesis principal es que un enfoque multi-agente, donde cada submodelo se especializa en una familia de senales, puede generar una seleccion mas robusta que una unica vista de mercado. La segunda hipotesis es que esa seleccion puede mantenerse valida en evaluacion fuera de muestra si se aplica un control PIT estricto y un backtest monetario consistente.

## 4. Arquitectura general

La arquitectura se organiza en cinco etapas:

1. Ingesta y consolidacion de datos.
2. Construccion del dataset maestro por snapshot.
3. Entrenamiento de agentes base y meta-learner.
4. Evaluacion walk-forward con backtest USD.
5. Fase live opcional para un fold de prediccion fuera de muestra.

El orquestador principal es `analyzer.py`. La configuracion global vive en `environment.py`.

## 5. Datos y fuentes

El sistema trabaja con datos almacenados en `data_finnhub/` y usa principalmente:

- precios historicos,
- fundamentales consolidados,
- recommendations,
- EPS surprises,
- insider transactions,
- insider sentiment,
- y benchmark SPY.

El universo actual del proyecto esta configurado con 400 tickers, aunque el pipeline filtra aquellos que no tienen cobertura suficiente para completar la evaluacion.

## 6. Construccion del dataset

La unidad de analisis es el par `(ticker, date)`, donde `date` representa el cierre del quarter de snapshot. Para cada observacion se construyen variables de diversas familias y se anade un label de retorno futuro.

Puntos clave del diseno:

- los fundamentales se seleccionan segun `filedDate` y no solo por quarter calendario,
- las features tecnicas se calculan con una ventana acotada de historial,
- el sentimiento y el insider se filtran por fecha as-of,
- el label se calcula desde la fecha de snapshot hasta el horizonte de holding configurado.

Este enfoque hace que el panel sea apto para entrenamiento y, al mismo tiempo, defendible frente a preguntas de leakage.

## 7. Etiquetado y aprendizaje

El sistema no aprende una etiqueta absoluta del estilo "sube o baja" sin contexto. Primero calcula `forward_return` y luego deriva una etiqueta relativa.

En el estado actual, los agentes base (fundamental, valuation, momentum, bear, sentiment) entrenan con objetivo de outperformance sectorial por snapshot: clasificar si un ticker supera la mediana de su sector en ese periodo.

Si no hay peers sectoriales suficientes en un snapshot, el etiquetado aplica fallback a mediana del universo para evitar ruido por muestras muy pequeñas.

Esto reduce el sesgo de mercado general y obliga a que el modelo capture senales cross-sectional, que es mas coherente con una estrategia de stock picking.

## 8. Entrenamiento multi-agente

Los agentes base se especializan en diferentes vistas del mercado:

- Fundamental,
- Valuation,
- Momentum,
- Bear,
- Sentiment,
- Sector rotation.

Cada agente produce un score interpretable. Sobre esos scores trabaja un meta-learner que integra la informacion y genera un `final_score`.

`Sector rotation` se entrena en una ruta top-down separada: su objetivo es decidir si cada sector va a outperformar al benchmark SPY. El meta combina esa evidencia sectorial con la evidencia company-vs-sector de los agentes base para estimar company-vs-benchmark.

Ademas, el pipeline genera scores OOF para que el meta-learner no vea ejemplos contaminados por entrenamiento y validacion en el mismo fold.

## 9. Walk-forward y evaluacion

La evaluacion se realiza de forma walk-forward. El periodo de entrenamiento siempre es anterior al quarter analizado. La decision se toma en una fecha de entrada construida con `SNAPSHOT_LAG_DAYS` y el holding se extiende durante el horizonte configurado.

En el estado actual del proyecto, el modo de analisis esta configurado en anual, con anchor temporal al inicio de 2023 mas el lag de snapshot. Esto reduce el numero de folds y simplifica la lectura academica, aunque la logica interna sigue siendo trimestral en la construccion de snapshots.

## 10. Backtest monetario en USD

La principal mejora del proyecto frente a un backtest puramente por retornos es la simulacion monetaria en USD. Esta simulacion modela:

- capital inicial,
- fracciones de accion,
- fee fijo por operacion,
- slippage configurable,
- equity diaria,
- y encadenamiento de capital entre folds.

El resultado no es solo una curva de retorno, sino una evolucion de capital que puede ser defendida como una cartera operativa.

## 11. Benchmark y baselines

El benchmark principal es SPY buy-and-hold. Ademas, el sistema compara contra baselines no triviales:

- equal-weight universe,
- momentum 12 meses,
- random top-N repetido muchas veces,
- y value combined basado en P/E + EV/EBITDA.

Todos los comparadores usan la misma logica de entrada, salida, costes y slippage. Esto es importante porque evita comparaciones artificiosas entre una estrategia realista y un benchmark idealizado.

## 12. Control anti-leakage

El proyecto incorpora varias barreras contra leakage:

- filtro as-of comun,
- uso de filedDate en fundamentales,
- auditoria por fold de fuentes sensibles,
- y generacion OOF para el meta-learner.

La idea no es afirmar que nunca existira una fuga, sino que el sistema la busca y la registra de forma explicita. Eso es metodologicamente mas solido para un TFM.

## 13. Trazabilidad y reproducibilidad

El run exporta configuracion, versionado y resumen operacional. En particular se registran:

- hash de commit,
- versiones de librerias,
- semilla global,
- flags de ejecucion,
- universo de tickers usado,
- y periodo exacto de analisis.

La trazabilidad no es un adorno. Es la base para repetir el experimento y defender por que un resultado se obtuvo exactamente en esas condiciones.

## 14. Artefactos

El proyecto genera artefactos en `results/` y subcarpetas. Los mas relevantes son:

- `pipeline.log`
- `run_config.json`
- `data_quality_report.csv`
- `leakage_audit.csv`
- `baselines_summary.csv`
- `final_summary.csv`
- `final_summary.json`
- `final_portfolio_value.json`
- curvas de equity y plots.

## 15. Interpretacion cientifica del enfoque

La contribucion del trabajo no es inventar un indicador unico, sino componer una tuberia que integre tres capas:

1. Feature engineering temporalmente seguro.
2. Clasificacion multi-agente y meta-aprendizaje.
3. Traduccion a performance economica con costes reales.

Eso convierte el proyecto en una pieza defendible tanto desde ML como desde evaluacion cuantitativa.

## 16. Limitaciones

El sistema sigue teniendo limitaciones que conviene reconocer en la memoria:

- no modela market impact variable,
- no simula intradia,
- el slippage es constante,
- no incorpora impuestos ni lending costs,
- y puede conservar cierto sesgo de supervivencia segun la disponibilidad historica de tickers.

Reconocer estas limitaciones fortalece la defensa porque muestra criterio tecnico y no una pretension de realismo absoluto.

## 17. Conclusiones

El trabajo demuestra que un sistema de stock picking puede estructurarse con disciplina metodologica suficiente para una memoria de TFM seria. El valor principal del proyecto esta en el rigor del pipeline: control temporal, comparacion honesta, backtest monetario y trazabilidad completa.

## 18. Anexo: como leer el resto de documentos

- `DOCUMENTATION.md`: documento tecnico exhaustivo.
- `RESULTS_INTERPRETATION.md`: lectura del run actual y de sus artefactos.
- `EXECUTIVE_SUMMARY.md`: version breve para lectura rapida.
