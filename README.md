# TFM · Hipótesis de inversión point-in-time

Proyecto local para construir y corroborar hipótesis de inversión de forma trazable.

El flujo único es:

```text
data/raw → Exploratory Study → hipótesis congelada → Confirmatory Study → evidencia final
```

No existen escenarios sueltos, grids cartesianos ni rutas de ejecución alternativas.

## Arranque

```powershell
python main.py
```

La aplicación queda disponible en `http://127.0.0.1:8765`.

Requisitos previos:

```powershell
python -m pip install -r requirements.txt
```

El terminal debe permanecer abierto mientras se use el dashboard. Para detenerlo, pulsar
`Ctrl+C`. No se inicia como proceso en segundo plano: al cancelar o cerrar esa terminal, Python se
detiene también.

## Exploratory Study

El usuario decide, exclusivamente desde el catálogo:

- Qué variables permanecen fijas.
- Qué variables se optimizan.
- Qué valores cerrados se comparan.

Las variables se procesan secuencialmente en este orden:

1. Objetivo temporal.
2. Representación.
3. Modelo y agentes.
4. Meta-agente.
5. Cartera.

El dashboard muestra el presupuesto exacto antes de lanzar, sin bloquear por número de
evaluaciones, fits caros o disco estimado. Al terminar se congela una hipótesis inmutable.

La pantalla se abre con la recomendación metodológica cargada:

- Cadencia de snapshots fija inicialmente en mensual; puede elegirse mensual, trimestral,
  semestral o anual desde la etapa Temporal.
- Optimizar horizonte: 6 y 12 meses.
- Optimizar historia: 8 y 12 años.
- Optimizar recencia: off y lineal.
- Optimizar representación: core, fundamental y all.
- Optimizar meta: equal, rank-IC y stacked rolling.
- Gestionar una única cartera dinámica: salida por percentil, sustitución con ventaja mínima y rebalanceo con tolerancia.
- Optimizar sizing: equal y calibrated alpha.
- Optimizar overlay: 100 %, 50 % y continuo.
- Mantener fijos controles PIT, agentes activados explícitamente, hiperparámetros finos, perfil
  balanced y costes base.

Esto produce 20 evaluaciones exploratorias, 10 fits caros y 43 evaluaciones para el ciclo completo.
Los botones «Restaurar recomendación» y «Dejar todo fijo» permiten volver a cualquiera de los dos
puntos de partida.

### Uso del dashboard

1. Revisar la barra de presupuesto y ajustar variables si es necesario.
2. Pulsar «Lanzar Exploratory».
3. Abrir «Estudios». En cada variable, aceptar el candidato automático o elegir otro con un motivo
   cerrado y pulsar «Aceptar y continuar».
4. Cuando el estado sea `awaiting_freeze`, pulsar «Congelar hipótesis».
5. Consultar la hipótesis en «Hipótesis» y sus vistas en «Análisis».
6. Volver a «Nuevo estudio», seleccionar la hipótesis y lanzar Confirmatory.
7. Seguir el progreso en «Estudios» y revisar el veredicto y el modelo final en «Análisis».

## Confirmatory Study

Confirmatory necesita una hipótesis congelada y no acepta overrides científicos. Ejecuta 23
evaluaciones fijas: semillas, perfiles, costes, calendario, placebos, bootstrap, permutación y
carteras aleatorias PIT.

Los veredictos son:

- `confirmed`
- `signal_only`
- `non_inferior`
- `rejected`

Los años 2025–2026 se calculan después del veredicto como `known_stress_not_selection`.

## Almacenamiento

- `data/raw`: fuentes originales preservadas.
- `data/prepared/<hash>`: una materialización PIT compartida.
- `data/cache`: fits y resúmenes por contenido, máximo lógico de 5 GiB.
- `results/studies`: ledger y decisiones.
- `results/hypotheses`: hipótesis congeladas.
- `results/models`: evidencia de modelos confirmados.

Los candidatos descartados no tienen directorio de artefactos: viven como filas compactas del
ledger. Solo hipótesis y modelos habilitan las vistas de rentabilidad, aprendizaje, rankings,
cartera, trades y stocks.

## Desarrollo

```powershell
python -m pytest -q
python -m ruff check .
node --check app/js/app.js
```

La configuración científica se amplía únicamente en `module/studies/catalog.py`. La API rechaza
campos desconocidos y valores que no pertenezcan a ese catálogo.

## Documentación

- [Metodología y arquitectura](docs/metodologia.md): referencia profunda del sistema y fuente
  metodológica para el TFM.
- [Bitácora](docs/bitacora.md): decisiones, ejecuciones e incidencias en orden cronológico.
- [Informe de resultados](docs/informe_resultados.md): plantilla auditable que se completará con
  evidencia del protocolo vigente.
- [Instrucciones para agentes](AGENTS.md): reglas obligatorias para mantener ciencia, Python,
  dashboard y documentación coherentes.
