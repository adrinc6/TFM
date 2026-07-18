# Estado del proyecto por fases

> La arquitectura y la metodología completas están en [docs/doc.md](doc.md); el porqué de cada
> decisión, en [docs/bitacora.md](bitacora.md). Este documento solo resume el estado de cada
> pieza y lo que queda.

## Estado

| Componente | Estado |
|---|---|
| Descarga de datos (Finnhub, Yahoo, SEC EDGAR) | Implementado. Universo dinámico del S&P 500 + fechas reales de publicación. |
| Dataset point-in-time | Implementado. Panel `(ticker, fecha)` sin lookahead. |
| Features + artefactos activables | Implementado. Factores base + 7 artefactos (`module/artifacts.py`). |
| Agentes LightGBM + meta-agente | Implementado. 3 agentes, objetivo `rank_regression`, walk-forward. |
| Cartera + backtest | Implementado. 8-12 posiciones, guarda anti-artefactos, perfiles de inversor. |
| Barrido en 2 fases + decisión automática de todos los ejes | Implementado. Fase 1 aísla cada eje (ventana, horizonte, ancla, profundidad, cadencia, artefactos); Fase 2 combina ganadores; afina hiperparámetros. Selección con era reservada 2025-2026. |
| Perfiles de inversor | Implementado. 8 estilos (`module/profiles.py`). |
| Robustez / placebo | Implementado. Permutación, Monte Carlo, bootstrap, leave-one-year-out. |
| Comando de principio a fin | Implementado. `RUN_MODE=full_study`. |
| Informes HTML | Implementado. Por run y de barrido, con `servir_html.py`. |
| **Redacción del TFM en LaTeX** | **Pendiente** (fase de cierre; rumbo en `latex/plan_tfm.md`). |

## Lo que queda

**Fase 7 — redacción del TFM en LaTeX.** Con el estudio completo ejecutado y documentado
(`docs/doc.md`, `docs/bitacora.md`, el informe de resultados y `results/escenarios/study_summary.json`
como materia prima), se redactará el documento del TFM capítulo a capítulo. El plan de estructura
está en [latex/plan_tfm.md](../latex/plan_tfm.md).

## Historia del desarrollo

El sistema pasó por un enfoque lineal (Ridge sobre factores GARP) que tocó techo en rank-IC ≈ 0,
lo que motivó el cambio a LightGBM y la consolidación en el sistema modular actual. La narrativa
completa —qué se probó, qué funcionó y qué no— está en la bitácora, que es el registro honesto del
proyecto y la base del capítulo de metodología.
