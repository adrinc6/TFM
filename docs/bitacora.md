# Bitácora de investigación y desarrollo

## Cómo utilizar esta bitácora

Este archivo registra decisiones, ejecuciones, incidencias y cambios interpretativos. No es un
changelog automático. Cada entrada debe permitir responder:

- qué se intentó;
- por qué;
- qué cambió;
- qué evidencia se obtuvo;
- qué se decidió;
- qué limitaciones quedan.

Reglas:

1. Añadir entradas en orden cronológico inverso.
2. No modificar retrospectivamente una decisión sin dejar una nueva entrada.
3. Vincular todo resultado con sus IDs y hashes.
4. Diferenciar evidencia sintética, exploratoria, confirmatoria y futura.
5. No copiar métricas a mano sin identificar el artefacto fuente.
6. Registrar el commit Git antes de una ejecución que vaya a usarse en el TFM.
7. Mantener UTF-8 y no introducir mojibake.

## Plantilla de entrada

```markdown
## AAAA-MM-DD · Título

### Objetivo

### Contexto y decisión previa

### Cambios o ejecución

- Commit:
- Catálogo:
- Study:
- Hipótesis:
- Modelo:
- Dataset:

### Evidencia observada

### Interpretación

### Decisión

### Incidencias y limitaciones

### Próximo paso
```

---

## 2026-07-25 · Auditoría de cierre y preset exploratorio recomendado

### Objetivo

Comprobar que la reconstrucción no dejó rutas legacy y hacer que el dashboard abra con una
recomendación metodológica útil, no con todas las variables fijas.

### Evidencia observada

- No existen `module/runs`, `module/scenarios`, `module/ui`, `data/processed` ni `data/recycle`.
- `results/` contiene solo las raíces necesarias `studies`, `hypotheses` y `models`, actualmente
  vacías.
- Los módulos científicos conservados tienen consumidores directos en el flujo actual.
- La configuración inicial anterior era válida, pero equivalía a una sola evaluación exploratoria.

### Decisión

Cargar por defecto ocho ejes optimizables que consumen 20 evaluaciones exploratorias y exactamente
10 fits caros. Mantener fijos los controles PIT, agentes, hiperparámetros finos y costes. Añadir
botones para restaurar la recomendación o dejar todo fijo.

### Próximo paso

Ejecutar un smoke study reducido desde el dashboard antes del primer ciclo completo.

### Corrección posterior

Se corrigió la normalización JSON de valores numéricos del catálogo. JavaScript representa `0.0`
como `0`; el backend ahora transforma únicamente números equivalentes de variables flotantes a su
objeto canónico del catálogo. La validación sigue rechazando el entero `0` como sustituto de un
booleano `false`.

### Ampliación posterior

La cadencia de snapshots se incorporó al catálogo temporal con 1, 3, 6 y 12 meses. El dashboard
explica ahora, por etapa, la pregunta metodológica, las comparaciones elegidas y sus evaluaciones
incrementales. Las tarjetas métricas de Estudios se muestran en una fila horizontal desplazable.

### Cambio posterior de presupuesto

Se retiraron los límites globales de Exploratory para número de evaluaciones, fits caros y disco
estimado. El preflight sigue mostrando las estimaciones, pero no bloquea el lanzamiento por ellas.
Confirmatory conserva sus 23 evaluaciones fijas y el catálogo sigue cerrado.

### Selección posterior de agentes y perfiles

Se eliminaron los presets con semántica de exclusión `without_*`. Los cinco agentes se activan de
forma positiva e independiente. Se incorporó el perfil de inversor como variable de cartera
seleccionable; Confirmatory continúa ejecutando los ocho perfiles como diagnósticos sin modificar
la hipótesis congelada.

### Gestión posterior del dashboard

El dashboard se ejecuta únicamente en primer plano con `python main.py`. Se eliminó el proceso de
prueba en segundo plano y el servidor cierra explícitamente sus recursos al recibir `Ctrl+C` o al
cerrarse la terminal que lo contiene.

## 2026-07-25 · Reinicio metodológico y documentación de referencia

### Objetivo

Simplificar el proyecto y convertirlo en un proceso defendible que separe descubrimiento y
corroboración, reduzca cómputo y almacenamiento y pueda manejarse completamente desde el
dashboard.

### Contexto y decisión previa

La arquitectura anterior acumulaba scenarios, experiments, estudios manuales y full studies con
más de 150 runs y decenas de GiB. Esa amplitud hacía difícil identificar qué resultado pertenecía
a una hipótesis, qué se había usado para seleccionar y qué constituía robustez.

Se decidió no migrar resultados ni protocolos anteriores. Los años 2025–2026, ya observados, se
clasificaron como estrés conocido y no como holdout.

### Cambios

- Se eliminó la arquitectura legacy y sus resultados.
- Se preservaron `data/raw/`, el histórico de componentes y el material académico `latex/`.
- Se creó el flujo único Exploratory → hipótesis congelada → Confirmatory.
- Se creó un catálogo científico cerrado y versionado.
- Se limitó Exploratory a 24 evaluaciones y 10 fits caros.
- Se fijó Confirmatory en 23 evaluaciones.
- Se implementó un runner científico único.
- Se implementó almacenamiento compartido por hashes y evidencia compacta.
- Se reconstruyeron API y dashboard.
- Se conservaron las vistas de rentabilidad, aprendizaje, rankings, cartera, trades y stocks.
- Se creó `reset_manifest.json` como auditoría de la limpieza.

### Evidencia observada

- 19 tests pasan.
- Ruff no detecta errores.
- El JavaScript principal supera `node --check`.
- Los endpoints HTTP principales responden en una prueba local.
- No se ha ejecutado todavía un ciclo real completo del protocolo nuevo.

### Interpretación

La evidencia disponible valida contratos de software, no la hipótesis financiera. El proyecto
queda preparado para generar resultados trazables, pero no existe aún un veredicto científico.

### Decisión

Usar `docs/metodologia.md` como fuente metodológica del TFM, este archivo como rastro cronológico y
`docs/informe_resultados.md` como único borrador de resultados empíricos.

### Incidencias y limitaciones

- Los jobs del servidor son en memoria y no sobreviven a un reinicio.
- La hipótesis no guarda todavía automáticamente el commit Git.
- El nulo de carteras aleatorias es una aproximación anual simplificada.
- Falta ejecutar un smoke study real reducido antes del primer estudio completo.

### Próximo paso

Ejecutar el pipeline sintético, registrar commit y catálogo, y después lanzar un Exploratory
reducido desde el dashboard.
