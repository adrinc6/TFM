# Instrucciones para agentes

## Propósito del repositorio

Este proyecto construye y corrobora hipótesis de inversión con datos point-in-time:

```text
data/raw → Exploratory Study → hipótesis congelada → Confirmatory Study → evidencia final
```

Antes de modificar ciencia, ejecución, almacenamiento o dashboard, leer completos:

1. `docs/metodologia.md`;
2. `docs/bitacora.md`;
3. `docs/informe_resultados.md`;
4. `CLAUDE.md`.

## Arquitectura obligatoria

- `module/data`: ingesta, universo y dataset PIT.
- `module/modeling`: features, agentes, targets y meta.
- `module/evaluation`: métricas, cartera, backtest y robustez.
- `module/studies`: catálogo, configuración, runner, Exploratory y Confirmatory.
- `module/storage`: datasets, caché y evidencia.
- `module/web` y `app`: API y dashboard.

No crear scenarios, experiments, runs sueltos, study manual, full study ni rutas alternativas.
Exploratory y Confirmatory deben usar el mismo `run_evaluation`.

## Reglas científicas

1. Todo parámetro científico procede de `module/studies/catalog.py`.
2. No aceptar inputs científicos libres, claves desconocidas ni JSON arbitrario.
3. Mantener el orden secuencial: temporal, representación, modelo, meta y cartera.
4. No introducir lookahead. Toda etiqueta consumida debe estar cerrada.
5. Usar el universo histórico point-in-time.
6. 2025–2026 es `known_stress_not_selection` y nunca participa en selección.
7. Confirmatory no acepta overrides ni modifica una hipótesis congelada.
8. El perfil base de cartera procede del catálogo; los ocho perfiles repetidos en Confirmatory son
   diagnósticos y no pueden cambiar la hipótesis ya congelada.
9. No afirmar mejora, alfa o confirmación sin artefactos reales.
10. Un test sintético valida software, no evidencia económica.

## Simplicidad y coherencia

- Preferir funciones pequeñas, dataclasses simples y flujos lineales.
- No crear abstracciones genéricas sin consumidor actual.
- No duplicar cálculos científicos entre backend y frontend.
- El backend es la autoridad del presupuesto y la validación.
- Si cambia una opción, actualizar catálogo, mapping a `Settings`, dashboard, documentación y test.
- Si se elimina una ruta, eliminar también imports, API, interfaz, tests y documentación huérfanos.
- No mantener compatibilidad legacy salvo instrucción explícita del usuario.

## Persistencia

- Preservar `data/raw/`.
- No copiar datasets dentro de studies.
- Los descartados viven solo en el ledger compacto.
- Solo hipótesis y modelos guardan evidencia analítica completa.
- La caché se elimina por referencias, nunca por antigüedad ciega.
- No borrar evidencia, raw o hipótesis sin autorización explícita y auditoría del alcance.
- No introducir `parent_run` ni directorios de runs.

## Documentación

Después de un cambio material:

- actualizar `docs/metodologia.md` si cambia el funcionamiento;
- añadir una entrada a `docs/bitacora.md` si cambia una decisión o se ejecuta un estudio;
- actualizar `docs/informe_resultados.md` solo con cifras trazables;
- mantener README y dashboard coherentes con Python.

El material del TFM saldrá de estos documentos. Distinguir siempre implementación, validación
sintética, evidencia exploratoria, evidencia confirmatoria y observación futura.

## Codificación

Todos los archivos se escriben en UTF-8. Cuidado con acentos y tildes: no introducir secuencias de
mojibake ni caracteres de sustitución. No normalizar texto español a ASCII.

## Validación mínima

Antes de dar por terminado un cambio:

```powershell
python -m pytest -q
python -m ruff check .
node --check app/js/app.js
```

Además:

- comprobar el presupuesto si cambia el catálogo;
- comprobar que Confirmatory sigue sumando exactamente 23;
- comprobar que frontend y backend muestran las mismas estimaciones de presupuesto;
- revisar UTF-8 en código, JSON, Markdown e interfaz.

No ejecutar un estudio real completo ni descargar datos sin autorización explícita. Para el
dashboard, usar `python main.py` en primer plano; no crear procesos Python desvinculados de la
terminal.
