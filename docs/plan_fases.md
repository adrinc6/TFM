# Estado del proyecto por fases

> La arquitectura y la metodología completas están en [docs/doc.md](doc.md); el porqué de cada
> decisión, en [docs/bitacora.md](bitacora.md). Este documento solo resume el estado de cada
> pieza y lo que queda.

## Estado

| Componente | Estado |
|---|---|
| Descarga de datos (Finnhub, Yahoo, SEC EDGAR) | Implementado. Universo dinámico del S&P 500 + fechas reales de publicación. |
| Dataset point-in-time | Implementado. Panel `(ticker, fecha)` sin lookahead. |
| Features + artefactos activables | Implementado. Factores base + 7 artefactos (`module/modeling/artifacts.py`). |
| Agentes LightGBM + meta-agente | Implementado. 3 agentes, objetivo `rank_regression`, walk-forward. |
| Cartera + backtest | Implementado. 8-12 posiciones, guarda anti-artefactos, perfiles de inversor. |
| Barrido en 2 fases + decisión automática de todos los ejes | Implementado. Fase 1 aísla cada eje (ventana, horizonte, ancla, profundidad, cadencia, artefactos); Fase 2 combina ganadores; afina hiperparámetros. Selección con era reservada 2025-2026. |
| Perfiles de inversor | Implementado. 8 estilos (`module/evaluation/profiles.py`). |
| Robustez / placebo | Implementado. Permutación, Monte Carlo, bootstrap, leave-one-year-out. |
| Comando de principio a fin | Implementado. `RUN_MODE=full_study`. |
| Consola local + registro de runs | Implementado. Entrada local desde `python main.py`, manifiestos versionados, hash y registro global. |
| Dashboard de resultados | Implementado. Research Console web (frontend en `app/` a nivel raíz, estética oscura, Chart.js local) servida por `module/ui/dashboard.py`, con listas separadas de estudios y runs y análisis por lectura directa de artefactos; informes estáticos en `module/ui/reports.py` con la misma estética. |
| **Estudio final (resultados)** | **Reejecutándose** con el orquestador de dos fases + era reservada. Cifras pendientes; se documentarán en `docs/informe_final.md` y `docs/doc.md` §8 al terminar. |
| **Redacción del TFM en LaTeX** | **En curso** (fase de cierre; estructura en `latex/plan_tfm.md`). Caps 1-5 y 8 redactados; 6-7-9 a la espera de resultados. |

## Lo que queda

**Fase 7 — redacción del TFM en LaTeX (en curso).** Se redacta el documento del TFM capítulo a
capítulo tomando como materia prima `docs/doc.md` y `docs/bitacora.md`. Los capítulos de método
(introducción, estado del arte, datos, point-in-time, agentes, limitaciones) se escriben ya; los
de **diseño experimental, resultados y conclusiones** quedan a la espera de que termine la
reejecución del estudio (sus cifras vendrán del study inmutable en `results/studies/<study_id>/`). El plan de
estructura está en [latex/plan_tfm.md](../latex/plan_tfm.md).

## Historia del desarrollo

El sistema pasó por un enfoque lineal (Ridge sobre factores GARP) que tocó techo en rank-IC ≈ 0,
lo que motivó el cambio a LightGBM y la consolidación en el sistema modular actual. La narrativa
completa —qué se probó, qué funcionó y qué no— está en la bitácora, que es el registro honesto del
proyecto y la base del capítulo de metodología.
