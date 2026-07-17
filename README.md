# TFM — Sistema multiagente de IA aplicado a la selección de acciones

Trabajo Fin de Máster sobre aprendizaje de un sistema de IA y su evaluación rigurosa en un entorno financiero. La rentabilidad no se interpreta de forma aislada: se mide junto con baselines, evidencia fuera de muestra, limitaciones y sesgos.

El repositorio está en reconstrucción por fases. Implementadas: descarga de datos crudos con universo histórico del S&P 500 y fechas reales de publicación SEC EDGAR (Fase 0), dataset point-in-time (Fase 1), features y baselines (Fase 2) y agentes ML + meta-agente (Fase 3). El diseño completo está en [docs/doc.md](docs/doc.md) y el estado ejecutable por fases en [docs/plan_fases.md](docs/plan_fases.md). El plan de redacción del TFM en LaTeX (Fase 7, al cierre del proyecto) está en [latex/plan_tfm.md](latex/plan_tfm.md).

## Requisitos

```bash
pip install -r requirements.txt
```

Configura la clave de Finnhub en `.env`:

```text
FINNHUB_API_KEY=tu_clave
```

`EDGAR_USER_AGENT` es opcional; identifica las solicitudes a la SEC.

## Ejecución por etapas

`RUN_MODE` selecciona la etapa y `RUN_SCOPE` el alcance de los datos. Se pueden establecer temporalmente desde PowerShell:

```powershell
$env:RUN_MODE = "download"
$env:RUN_SCOPE = "dev"
python main.py
```

| `RUN_MODE` | Comportamiento actual o futuro |
|---|---|
| `download` | Descarga y consolida datos crudos. |
| `dataset` | Construye el panel, precios de activos y benchmark point-in-time desde raw. |
| `features` | Genera factores GARP/momentum, baselines y etiquetas futuras separadas. |
| `agents` | Entrena agentes Ridge walk-forward y el meta-agente de rank-IC. |
| `backtest`, `report`, `experiments` | Fallan explícitamente hasta que se implementen. |
| `full` | Ejecuta `download → dataset → features → agents`. |

| `RUN_SCOPE` | Resultado |
|---|---|
| `dev` | Usa una muestra pequeña y guarda los agregados en `data/raw/dev/`. Nunca sobrescribe datos completos. |
| `full` | Usa todo el universo histórico y guarda los agregados en `data/raw/`. |

Por ejemplo, la descarga completa se solicita explícitamente así:

```powershell
$env:RUN_MODE = "download"
$env:RUN_SCOPE = "full"
python main.py
```

## Datos producidos por la Fase 0

- Caché por fuente en `data/raw/json/`, incluida la caché de respuestas EDGAR.
  Si el mapa SEC actual es ambiguo por reutilización de ticker, se valida con el
  buscador SEC antes de aceptar el CIK.
- Agregados: `profiles.parquet`, `finnhub_metrics.parquet`, `prices.parquet`, `news.parquet` y `report_dates.parquet`.
- SPY se descarga siempre como benchmark de precios; no requiere CIK, fundamentales ni perfil de empresa.
- `report_dates.parquet` usa SEC EDGAR y contiene `ticker`, `cik`, `form`, `period` y `filed_date`.
- Metadatos: `download_coverage.json`, `download_failures.csv` y `universe_coverage.json`.

La cobertura anual mide miembros históricos del índice frente a empresas observables con precio y un fundamental asociado a un informe SEC publicado. Las ejecuciones `dev` marcan su cobertura como no representativa.

## Dataset point-in-time

`RUN_MODE=dataset` requiere que la descarga del mismo `RUN_SCOPE` haya generado
`prices.parquet`, `finnhub_metrics.parquet` y `report_dates.parquet`. Produce:

- `data/processed/panel_point_in_time.parquet` para alcance completo.
- `data/processed/dev/panel_point_in_time.parquet` para desarrollo.
- `benchmark_point_in_time.parquet` y `asset_price_point_in_time.parquet` en el mismo directorio procesado.

El panel usa solo precios ajustados, series históricas de Finnhub y fechas de presentación SEC. No contiene perfiles actuales, `payload.metric`, noticias ni sector.

## Features y agentes ML

`RUN_MODE=features` requiere los tres artefactos PIT del mismo alcance y genera
`features_point_in_time.parquet`, `baseline_scores.parquet` y
`targets_forward_3m.parquet`. Las etiquetas futuras se mantienen separadas de
las variables observables. Se excluyen precios con más de siete días de antigüedad.

`RUN_MODE=agents` usa una etiqueta de retorno excesivo de tres meses frente a SPY.
Arranca en la fecha ancla configurada (por defecto, febrero de 2000), usa toda la
historia disponible hasta completar ocho años y después una ventana móvil de ocho.
Sus resultados quedan versionados bajo `data/processed[/dev]/agents/` con scores,
pesos meta, rank-IC, coeficientes y manifiesto. No construye todavía una cartera.
