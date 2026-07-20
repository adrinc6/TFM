# TFM — laboratorio ML point-in-time para selección de acciones

Este repositorio implementa un laboratorio reproducible para estudiar señales financieras fuera de muestra. El objetivo principal es medir la capacidad de ordenación del modelo mediante Rank-IC; la rentabilidad de cartera es una consecuencia y nunca el criterio para elegir el modelo.

La documentación metodológica completa está en [docs/doc.md](docs/doc.md). El estado y las decisiones históricas se registran en [docs/bitacora.md](docs/bitacora.md).

## Inicio rápido

```powershell
python -m pip install -r requirements.txt
python main.py
```

Sin `RUN_MODE`, el proyecto abre la Research Console en `http://127.0.0.1:8765`.

Para descargar datos se necesita `FINNHUB_API_KEY` en `.env`. Las fuentes activas son precios OHLCV y series históricas de Finnhub; el sistema no declara conectores externos sin datos históricos point-in-time verificados.

## Full study oficial

Con datos ya descargados:

```powershell
$env:RUN_MODE = "full_study"
$env:RUN_SCOPE = "full"
python -u main.py
```

También puede lanzarse desde **Full study → Revisar y lanzar** en la consola. Esa pantalla permite escribir nombre e hipótesis y muestra, sin posibilidad de editar:

- todos los ejes que el estudio barre;
- parámetros metodológicos fijos;
- escenarios de estrés de costes.

El ciclo es dirigido, no un producto cartesiano: ablaciones aisladas de modelo, combinación greedy de candidatos aceptados, afinado, configuración de cartera, perfiles, estrés de costes, robustez y validación reservada.

La selección del modelo usa solo cohortes con fecha de predicción hasta **2024**. Los años **2025–2026** quedan reservados para una única validación final. Las comisiones, el slippage y la semilla no se optimizan: se reportan como estrés y robustez.

## Arquitectura

```text
datos crudos PIT → dataset → factores/bloques → agentes → meta-agente → cartera/backtest → estudio OOS
```

- `module/data/`: descarga, universo histórico y panel point-in-time.
- `module/modeling/`: catálogo de factores, features, agentes y meta-agente.
- `module/evaluation/`: cartera, backtest, perfiles y robustez.
- `module/runs/`: runs, studies, caché y resultados inmutables.
- `module/ui/` y `app/`: Research Console e informes.

## Modelo

El catálogo actual contiene bloques de calidad, eficiencia, fortaleza financiera, valoración, caja, crecimiento, estabilidad, momentum, tendencia, riesgo y liquidez. Cinco agentes configurables —`quality`, `value`, `growth`, `momentum` y `risk`— usan LightGBM, Elastic Net y CatBoost. El meta-agente admite combinación equiponderada, por Rank-IC, por régimen o stacking OOS.

Cada bloque, agente y familia puede medirse mediante ablación. La poda OOS, la importancia de permutación temporal y el gating de bloques solo usan etiquetas que ya han cerrado antes del reentrenamiento.

## Etapas

| `RUN_MODE` | Acción |
|---|---|
| `download` | Descarga y consolida datos crudos. |
| `dataset` | Construye panel point-in-time y precios. |
| `features` | Construye factores y etiquetas futuras separadas. |
| `agents` | Entrena agentes walk-forward y meta-agente. |
| `backtest` | Simula cartera y métricas económicas. |
| `report` | Genera informe HTML del último run. |
| `experiments` | Ejecuta experimentos configurados. |
| `full_study` | Ejecuta el ciclo oficial completo. |

## Resultados y pruebas

Los runs se guardan en `results/runs/`; los studies, en `results/studies/`; y el registro inmutable está en `results/registry.jsonl`. Cada resultado conserva configuración, manifiesto, artefactos, diagnósticos y exportaciones CSV.

```powershell
python -m pytest tests/ -q
python -m ruff check .
```

Los resultados históricos no sustituyen a un nuevo full study cuando cambia la metodología. Consulta `docs/informe_final.md` para el estado de evidencia vigente.
