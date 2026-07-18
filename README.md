# TFM — Sistema de IA aplicado a la selección de acciones

Trabajo Fin de Máster sobre **cómo aprende un sistema de IA** en un entorno financiero y su
evaluación rigurosa. El objetivo no es batir a un índice sino **medir el aprendizaje** (rank-IC
fuera de muestra) con honestidad: la rentabilidad se reporta como consecuencia, con
significancia estadística y tests de robustez, y un resultado negativo bien medido es un
entregable válido.

El sistema es un conjunto de **agentes LightGBM** (calidad, momentum, valor) combinados por un
meta-agente, con un catálogo de **artefactos activables** (bloques de features/contexto) que un
barrido de ablations activa automáticamente según cuáles mejoran el aprendizaje. Diseño completo
en [docs/doc.md](docs/doc.md); el porqué de cada decisión en [docs/bitacora.md](docs/bitacora.md).

## Requisitos

```bash
pip install -r requirements.txt
```

Clave de Finnhub en `.env` (`FINNHUB_API_KEY=...`), solo para la descarga. `EDGAR_USER_AGENT` es
opcional (identifica las solicitudes a la SEC).

## Consola local y CLI

Sin `RUN_MODE` definido, la entrada principal abre una consola local:

```powershell
python main.py
# http://127.0.0.1:8765
```

La consola es una aplicación web de una sola página con estética oscura (negros y grises),
servida como archivos reales desde `module/ui/app/` por el servidor `http.server` de
`module/ui/dashboard.py` (sin dependencias externas ni CDN; los gráficos usan una copia local
de Chart.js). Tiene dos vistas:

- **Consola**: lanzar un **Experimental** (configuración libre y trazable), crear un **Study**
  (rejilla de combinaciones dirigida) y revisar la **Optimization** oficial, con todos los
  parámetros, presets y selects guiados por los valores admitidos.
- **Resultados**: dos listas separadas, una de **estudios** y otra de **runs**. Al seleccionar
  un estudio se analiza el estudio (fases, decisión y comparativa de sus runs); al seleccionar
  un run se analiza el run con pestañas de resumen, rendimiento, aprendizaje, cartera, trades,
  explorador de stocks (con crecimientos) y ficha por ticker, con tablas y gráficos.

Cada resultado nuevo queda registrado bajo `results/runs/<YYYYMMDD--hash>/` con manifiesto,
configuración efectiva, artefactos y trazabilidad. Los studies agrupan sus runs en
`results/studies/` y se indexan en `results/registry.jsonl`.

Cuando se define `RUN_MODE`, se conserva el flujo CLI:

Con `data/raw` ya descargado, un único comando ejecuta el estudio completo sin decisiones humanas:

```powershell
$env:RUN_MODE = "full_study"; $env:RUN_SCOPE = "full"; python main.py
```

Hace, en dos fases y sin decisiones humanas: **Fase 1** barre cada eje del sistema aislado (ventana
de entrenamiento, horizonte de etiqueta, ancla, profundidad, cadencia y los 7 artefactos) y
**decide automáticamente** el mejor nivel de cada eje + qué artefactos ayudan (por significancia);
**Fase 2** combina los ganadores; luego afina hiperparámetros → configuración final → run
optimizado → 8 perfiles de inversor → tests de robustez/placebo → **validación en la era reservada
2025-2026** (que no interviene en la selección, para no sobreajustar por explorar mucho) → informes
HTML.

Los resultados se organizan de forma inmutable por `runs/` y `studies/`: cada study conserva su
manifiesto, definición y lista de runs; cada run guarda configuración, estado, artefactos de
agentes, backtest, cartera, CSVs y panel de stocks reproducible.

## Organización del código

`module/` se divide por responsabilidad para que la navegación del proyecto sea directa:

- `data/`: universo, dataset point-in-time, baselines e ingestión de Finnhub/Yahoo/EDGAR.
- `modeling/`: features, artefactos activables, agentes LightGBM y meta-agente.
- `evaluation/`: cartera, backtest, perfiles, robustez y estadística.
- `runs/`: ejecución de runs/studies, caché y almacenamiento inmutable de resultados.
- `ui/`: Research Console (`dashboard.py` + frontend en `app/`) e informes estáticos (`reports.py`).
- `common/`: utilidades transversales.

## Resultados

La consola muestra los manifiestos, métricas, rankings, carteras, órdenes y evolución por ticker
desde los Parquet y CSV del run seleccionado. El rank-IC OOS es el criterio de aprendizaje; el
rendimiento de cartera se muestra como consecuencia y no como selector de configuraciones.

## Etapas sueltas

`RUN_MODE` selecciona una etapa; `RUN_SCOPE` el alcance (`dev` = muestra pequeña aislada,
`full` = universo completo).

| `RUN_MODE` | Qué hace |
|---|---|
| `download` | Descarga y consolida datos crudos (Finnhub, Yahoo, SEC EDGAR). |
| `dataset` | Panel point-in-time, precios de activos y benchmark. |
| `features` | Factores GARP/momentum + artefactos activos + etiquetas futuras separadas. |
| `agents` | Entrena los 3 agentes LightGBM walk-forward y el meta-agente. |
| `backtest` | Simula la cartera (con guarda anti-artefactos y perfil de inversor) y calcula métricas. |
| `report` | Genera el informe HTML estático (autocontenido, estética oscura) del último run. La consola es la vía principal de análisis. |
| `experiments` | Barrido de artefactos (`escenarios/rejilla_base.py`) + decisión automática. |
| `full_study` | **Todo de principio a fin en 2 fases** con era reservada (ver arriba). |

## Arquitectura de datos (point-in-time)

- **Universo dinámico** del S&P 500 por fecha (composición histórica real): sin sesgo de
  inclusión anticipada. El sistema se centra en **2016+** (mayor cobertura, menos sesgo de
  supervivencia).
- **Fechas de publicación reales** de SEC EDGAR: un fundamental solo es observable cuando se
  publicó, no en su cierre fiscal. Sin lookahead.
- **Panel** `(ticker, snapshot_date)` con lo observable en cada fecha; sin relleno hacia atrás.

## Modelo y aprendizaje

- **3 agentes LightGBM** (calidad, momentum, valor), objetivo `rank_regression` (percentil
  transversal del retorno). Walk-forward estricto (solo etiquetas ya realizadas).
- **Meta-agente** por rank-IC reciente; el `meta_score` es lo que opera la cartera y sobre lo que
  se mide el rank-IC.
- **Artefactos activables** (`module/modeling/artifacts.py`): momentum de fundamentales, régimen bull/bear,
  neutralización por sector, momentum de precio multi-horizonte, medias móviles, régimen
  ampliado, calidad/crecimiento derivados. El barrido decide cuáles entran.

## Cartera, perfiles y robustez

- **Cartera** 8-12 posiciones, peso máx 15 %, rotación con umbral de ventaja y expulsión.
  **Guarda anti-artefactos**: neutraliza retornos mensuales imposibles (>200 %) como datos
  corruptos.
- **Perfiles de inversor** (`module/evaluation/profiles.py`): entre las buenas del meta, cada perfil
  (conservador, agresivo, value, calidad, momentum, GARP, contrarian, balanceado) reordena según
  estilo. Explicabilidad como funcionalidad.
- **Robustez** (`module/evaluation/robustness.py`): permutación de etiquetas (placebo), carteras aleatorias
  (Monte Carlo), bootstrap por bloques, leave-one-year-out. Demuestran que el resultado no es
  suerte.

## Tests

```bash
pytest tests/ -q
```

Cubren la ausencia de lookahead (point-in-time, artefactos, walk-forward), las reglas de cartera,
la guarda anti-artefactos, la decisión automática de artefactos, los perfiles y la robustez.
