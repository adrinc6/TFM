# Instrucciones para agentes

## Propósito del proyecto

Este repositorio contiene un Trabajo Fin de Máster (TFM) sobre Inteligencia
Artificial y Machine Learning aplicados a bolsa. La bolsa es el entorno de
validación, no el objetivo principal: lo importante es estudiar cómo aprende
un sistema de IA, evaluar ese aprendizaje con rigor y comprobar si puede tener
utilidad económica.

El proyecto debe ser académico, reproducible, explicable y honesto. Un buen
resultado no es solo una cartera rentable: también puede ser un resultado
negativo bien medido, que explique con claridad qué aprende el modelo y qué no.

## Estado actual y modelo mental del sistema

El proyecto está en un **reinicio en limpio**. Ahora mismo el repositorio solo
contiene la **descarga de datos**; el resto de etapas se reconstruirá siguiendo
el plan maestro en `docs/doc.md`.

Código que existe hoy:

- `environment.py`: configuración editable de la descarga (universo de tickers,
  fechas, modo desarrollo, clave de API).
- `main.py`: entrada única que ejecuta la descarga.
- `module/ingest/`: clientes HTTP (Finnhub, Yahoo) y orquestación de la descarga
  de datos crudos (`download_raw_data`).
- `module/utils.py`: utilidades compartidas de logging y escritura de archivos.

Arquitectura **objetivo** por etapas (a reconstruir sobre la descarga; descrita
en detalle en `docs/doc.md`):

```text
download → dataset → features → ml (agentes) → selección → cartera → backtest → informe
```

con un barrido de escenarios (rejilla) y una selección del sistema final por
estabilidad multi-era, no solo por alfa. La simulación arrancará en una fecha
ancla derivada de trimestre + retardo de publicación de fundamentales, separando
entrenar (fundamentales nuevos) de revisar (mensual, re-precio).

Antes de reconstruir una etapa, léela en `docs/doc.md`, respeta el flujo de datos
de entrada a salida y no resuelvas un problema en una capa ajena.

## Prioridades metodológicas

1. No introducir lookahead bias ni fugas de información. Una señal de una fecha
   solo puede usar datos observables en esa fecha.
2. Mantener el entrenamiento y la evaluación temporalmente separados.
3. Medir el aprendizaje fuera de muestra; no presentar rentabilidad aislada
   como evidencia suficiente.
4. Conservar y mostrar limitaciones, baselines y resultados negativos.
5. Tratar el tamaño reducido de muestra y el sesgo de supervivencia como
   limitaciones explícitas, nunca como detalles que se puedan ocultar.
6. Siempre que te pida un plan debes mostrarmelo entero y hasta que no te diga que lo implementes no cambies nada, solo planea.

## Cambios que requieren aprobación previa

Pide opinión o permiso explícito antes de cambiar cualquiera de estos aspectos:

- La hipótesis de inversión, reglas de cartera o umbrales de decisión.
- Objetivos, etiquetas, modelos ML, pesos o esquema de entrenamiento.
- Fechas, universo de activos, benchmark, costes o configuración que altere los
  resultados de un experimento.
- Métricas, metodología de validación o cómo se interpreta la evidencia.
- Dependencias, APIs, servicios externos, llamadas a LLM o costes asociados.
- Eliminación de datos, resultados o código que pueda seguir siendo necesario
  para reproducibilidad o comparación.

Sí están permitidos sin pedir permiso los arreglos acotados de bugs, claridad,
tipado, documentación, tests y refactorizaciones que no cambien el significado
ni el resultado esperado del sistema. Si hay duda sobre el impacto, pregunta.

## Estilo de código

Escribe código sencillo de leer, revisar y explicar por una persona:

- Prefiere flujo lineal y nombres descriptivos a abstracciones profundas.
- Mantén los archivos pequeños y con una responsabilidad clara. Si un archivo
  crece demasiado, divide por concepto del dominio, no por capas artificiales.
- Usa funciones cortas solo cuando expresen una operación o idea concreta; no
  fragmentes lógica lineal en muchas funciones triviales.
- Evita patrones, clases, configuraciones, compatibilidades legacy y
  generalizaciones que no resuelvan una necesidad actual del TFM.
- Cuando algo deje de utilizarse, propón eliminarlo en lugar de mantenerlo por
  compatibilidad. Solicita permiso si la eliminación afecta a resultados,
  datos o comportamiento.
- No añadas dependencias sin autorización. Prefiere las ya presentes en
  `requirements.txt` y la biblioteca estándar.
- Añade comentarios solo para decisiones no obvias, especialmente las
  metodológicas o temporales.

## Datos, resultados y secretos

- No modifiques `.env`, claves API ni secretos.
- No sobrescribas resultados existentes salvo que la tarea lo pida claramente.
- Conserva la trazabilidad de los artefactos de cada ejecución.
- Distingue con precisión entre datos crudos, procesados y salidas de un run.

## Pruebas y verificación

Todavía no hay carpeta `tests/` tras el reinicio. Al reconstruir cada etapa,
añade sus pruebas (empezando por las de leakage y separación temporal, que son
las críticas del proyecto) y ejecútalas tras cada cambio relacionado. Indica con
claridad qué se ejecutó y qué no se pudo verificar.

No ejecutes un pipeline completo ni descargues datos externos sin que sea
necesario para la tarea o sin autorización cuando pueda tener coste o tardar
mucho. Indica con claridad qué prueba se ejecutó y qué no se pudo verificar.

## Documentación y codificación

- Escribe documentación, comentarios y mensajes orientados al usuario en
  español claro.
- Conserva UTF-8 al modificar archivos. Revisa especialmente acentos, eñes,
  flechas y símbolos matemáticos: nunca introduzcas texto mal codificado.
- Actualiza `README.md` o `docs/` cuando un cambio autorizado altere el uso,
  la arquitectura o la metodología descrita.
