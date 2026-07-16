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

## Modelo mental del sistema

El pipeline principal es:

```text
download → dataset → features → ml → watchlist → backtest → viewer → report
```

- `environment.py`: configuración central de la ejecución.
- `main.py`: orquestación de etapas.
- `module/dataset.py`: dataset point-in-time y prevención de lookahead.
- `module/features/`: variables y baselines deterministas.
- `module/ml.py`: agentes ML, combinación de señales y diagnóstico.
- `module/strategy/`: selección, cartera y tamaño de posición.
- `module/backtest/`: simulación, métricas, baselines y artefactos.
- `module/viewer/` y `module/report.py`: informe final y visualización.
- `tests/`: pruebas de leakage, ML, features, estrategia y robustez.

Antes de cambiar comportamiento, localiza la etapa propietaria y sigue el flujo
de datos de entrada a salida. Evita solucionar un problema en una capa ajena.

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

Tras un cambio, ejecuta los tests relacionados en `tests/`. Por ejemplo,
ejecuta pruebas de leakage y ML ante cambios temporales o de entrenamiento, y
pruebas de estrategia/backtest ante cambios de cartera o métricas.

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
