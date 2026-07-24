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

También puede lanzarse desde **Full study → Revisar y lanzar** en la consola. El protocolo oficial
es confirmatorio y cerrado: ejecuta exactamente **48 evaluaciones**, aborta antes de crear el study
si proyecta más de 10 walk-forwards caros o 5 GiB incrementales, y no expande dinámicamente el
catálogo manual.

El presupuesto se reparte en 12 challengers de señal, 2 semillas confirmatorias, 12 políticas de
cartera, 8 perfiles, 6 stresses económicos y 8 evaluaciones estadísticas. Los diez primeros
challengers del meta reutilizan los mismos fits. En caché fría se presupuestan de forma
conservadora 10 fits caros —incumbent, variantes de lookback/recencia, semillas y placebos—; si
el incumbent ya está reciclado, el consumo real será menor.

La selección histórica usa tres eras —2015–2018, 2019–2021 y 2022–2024—. Los años 2025–2026 son
un estrés histórico conocido y se muestran separados con la marca `known_stress_not_selection`;
nunca deciden el ganador.

`study` conserva el catálogo amplio para exploración manual. `full_study` usa
`OFFICIAL_STUDY_PROTOCOL`, con challengers y políticas pre-registrados; ambos modos ya no tienen
que compartir todos los ejes.

La cartera oficial compara el legacy corregido con rebalanceo trimestral y cuatro vintages
trimestrales mantenidos 12 meses. Puede asignar el presupuesto entre un satélite de acciones y un
núcleo SPY, con sizing equiponderado, legacy o alfa calibrado point-in-time. La selección maximiza
Information Ratio neto por era bajo gates de alfa, turnover y costes.

Los ocho perfiles se publican como diagnósticos paralelos, con la misma trayectoria de exposición
activa y sizing equal. Ningún perfil puede modificar `best_config`. Si ninguna alternativa supera
los criterios predefinidos, se conserva el incumbent y el veredicto es `no_improvement`.

## Rendimiento, aislamiento y operación

Cada run se publica en un workspace privado y conserva manifiesto, hashes de entradas, huella de
código y telemetría. La caché es por etapa: dataset, features, agentes y backtest solo se
restauran cuando su configuración efectiva y sus entradas coinciden; un run interrumpido nunca se
trata como completo.

El preflight informa claves únicas de dataset, features, fits y agentes, tiempo estimado desde la
telemetría y almacenamiento máximo. La traducción de cartera es deliberadamente secuencial por
familias —estructura, sizing, overlay y hurdle— para evitar un cartesiano sin hipótesis económica.
En el dashboard el botón permanece deshabilitado hasta superar ese preflight. Durante la ejecución
se muestran fase, escenario y progreso sobre 48; al terminar, la vista presenta ganadores,
rechazos, resultados por era, stress 2025–2026, perfiles, robustez, almacenamiento y todos los
artefactos del study para descarga.

La consola del study muestra los logs agregados de todos los runs activos, etiquetados por run, y
puede sustituir temporalmente la comparativa. Las líneas visibles se limitan a unas veinte, con
scroll para el historial reciente.

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

La caché de etapas es content-addressed y versionada. Antes de reutilizar un artefacto valida su
tamaño y SHA-256; los backtests usan workspaces privados y los studies interrumpidos pueden
reanudarse desde la consola reutilizando únicamente runs completos.

```powershell
python -m pytest tests/ -q
python -m ruff check .
```

Los resultados históricos no sustituyen a un nuevo full study cuando cambia la metodología. Consulta `docs/informe_final.md` para el estado de evidencia vigente.
