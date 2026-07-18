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

## El comando de principio a fin

Con `data/raw` ya descargado, un único comando ejecuta el estudio completo sin decisiones humanas:

```powershell
$env:RUN_MODE = "full_study"; $env:RUN_SCOPE = "full"; python main.py
```

Hace: barrido de ablations → **decisión automática** de qué artefactos ayudan (por significancia)
→ configuración final → run optimizado → 8 perfiles de inversor → tests de robustez/placebo →
informes HTML. Produce `results/escenarios/study_summary.json` y `comparison.html`.

## Ver los informes

Algunas pestañas cargan CSVs grandes por `fetch`, que requieren un servidor local:

```bash
python servir_html.py
# abre http://localhost:8000/results/escenarios/comparison.html
```

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
| `report` | Genera el `report.html` del último run. |
| `experiments` | Barrido de ablations + decisión automática de artefactos. |
| `full_study` | **Todo de principio a fin** (ver arriba). |

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
- **Artefactos activables** (`module/artifacts.py`): momentum de fundamentales, régimen bull/bear,
  neutralización por sector, momentum de precio multi-horizonte, medias móviles, régimen
  ampliado, calidad/crecimiento derivados. El barrido decide cuáles entran.

## Cartera, perfiles y robustez

- **Cartera** 8-12 posiciones, peso máx 15 %, rotación con umbral de ventaja y expulsión.
  **Guarda anti-artefactos**: neutraliza retornos mensuales imposibles (>200 %) como datos
  corruptos.
- **Perfiles de inversor** (`module/profiles.py`): entre las buenas del meta, cada perfil
  (conservador, agresivo, value, calidad, momentum, GARP, contrarian, balanceado) reordena según
  estilo. Explicabilidad como funcionalidad.
- **Robustez** (`module/robustness.py`): permutación de etiquetas (placebo), carteras aleatorias
  (Monte Carlo), bootstrap por bloques, leave-one-year-out. Demuestran que el resultado no es
  suerte.

## Tests

```bash
pytest tests/ -q
```

Cubren la ausencia de lookahead (point-in-time, artefactos, walk-forward), las reglas de cartera,
la guarda anti-artefactos, la decisión automática de artefactos, los perfiles y la robustez.
