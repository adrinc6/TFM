# Informe final — Sistema de IA para selección de acciones

> **Estudio cerrado (salvo placebo).** Cifras procedentes de
> `results/studies/20260719--optimization-official--acb6c310dfb8/` (orquestador de ciclo único
> `unified_full_cycle`: Fase 1 + Fase 2 greedy + Fase 3 afinado + cartera + era reservada
> 2025-2026), reproducibles con `RUN_MODE=full_study` (ver [docs/doc.md](doc.md)). El hilo de
> decisiones está en [docs/bitacora.md](bitacora.md).

El resultado se presenta en **dos planos separados**, sin mezclarlos: qué **aprende** el sistema
y qué **renta**. La configuración final se eligió por aprendizaje y estabilidad (rank-IC OOS hasta
2024), nunca por rentabilidad.

## Configuración final seleccionada

Elegida automáticamente por el barrido dirigido (Fase 1 aísla cada eje, Fase 2 combina los
ganadores en greedy, Fase 3 afina hiperparámetros LightGBM, luego cartera):

| Parámetro | Valor |
|---|---|
| Ventana de entrenamiento | 12 años |
| Horizonte de etiqueta | 3 meses |
| Cadencia de fundamentales | trimestral (`fundamental_step_months=6`) |
| `objective` | quartile |
| `lgbm_n_estimators` / `max_depth` / `learning_rate` | 400 / 6 / 0.10 |
| `meta_type` | equal |
| `recency_weighting` | linear |
| Artefactos activos | neutralización por sector, momentum fundamental, régimen de mercado,
  momentum de precio multi-horizonte, medias móviles, régimen extendido, `quality_growth_derived` |
| Cartera: `target_min`/`target_max` | 8 / 10 |
| Cartera: `entry_min_percentile` / `min_hold_percentile` | 70 / 40 |
| Cartera: `rotation_edge_percentiles` | 10 |
| Cartera: `max_weight_per_position` | 0.20 |
| Costes: comisión / slippage | 0 bps / 5 bps |

`run_id` final del modelo: `20260719--a18d7b5bcf26`. `run_id` final de cartera: `20260719--ab30bc3fe2e2`.

> **Nota de reproducibilidad:** un escenario de Fase 1 (`lgbm_min_child_samples_20`) se saltó por
> un error de permisos de Windows al mover un archivo temporal de caché
> (`data/recycle/backtest/...`, `WinError 5: Acceso denegado`), no por un problema metodológico.
> No afecta a la configuración ganadora porque `lgbm_min_child_samples_100` ya había ganado sin
> necesitar ese candidato adicional. Ver `skipped_scenarios` en `decision.json`.

## Plano 1 — El aprendizaje (rank-IC OOS del `meta_final`)

| Métrica | Valor |
|---|---|
| rank-IC medio (OOS) | **+0.0118** |
| Fracción de cohortes con IC > 0 | 60 % |
| Desviación típica del rank-IC | 0.0869 |
| **Bootstrap por bloques** (IC 95 %) | **[-0.0113, 0.0340]** — **cruza cero** (45 cohortes, bloque 4) |
| **Estabilidad leave-one-year-out** | rank-IC entre 0.0079 y 0.0183 al excluir cualquier año; ningún año domina |
| **Era reservada 2025-2026** (nunca optimizada) | **rank-IC +0.0210** sobre 6 cohortes |
| **Placebo por permutación de etiquetas** | **sin ejecutar** (`n_permutations=0`, sigue "pendiente de ejecución aislada") |

Respecto al estudio previo (`20260718--...--bef48ddfc41f--r02`, rank-IC +0.0158, IC bootstrap
[0.0053, 0.0265] sin cruzar cero), este nuevo estudio da un rank-IC OOS **más bajo** (+0.0118) y,
sobre todo, un **intervalo de bootstrap que ahora sí cruza cero**. Con solo 45 cohortes de
bootstrap (menos que las 147 del estudio anterior), la evidencia estadística de que el modelo
ordena mejor que el azar es **más débil que antes, no más fuerte**. El leave-one-year-out sigue sin
mostrar un año que domine, y la era reservada 2025-2026 (+0.0210, aunque con solo 6 cohortes) sigue
siendo positiva, pero la pieza que faltaba antes —el **placebo por permutación**— sigue sin
ejecutarse: seguimos sin poder descartar formalmente que parte de la señal sea artefacto de
proceso de búsqueda en vez de señal real.

## Plano 2 — La rentabilidad como consecuencia (por perfil de inversor)

Todos los perfiles parten del **mismo modelo** (rank-IC común +0.0118); solo cambia cómo se
construye la cartera. Frente al **SPY (CAGR 13.86 %)**:

| Perfil | CAGR cartera | Dif. vs SPY | Alfa medio anual | Information Ratio | Beat rate | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| **aggressive** (recomendado) | **16.66 %** | **+2.80 pp** | **+1.88 %** | **0.153** | 66.7 % | 25.21 % |
| contrarian | 16.34 % | +2.48 pp | +1.02 % | 0.137 | 50.0 % | 28.02 % |
| momentum | 15.76 % | +1.90 pp | +1.24 % | 0.115 | 58.3 % | 24.21 % |
| balanced | 15.11 % | +1.25 pp | +0.66 % | 0.102 | 50.0 % | 23.99 % |
| quality | 14.32 % | +0.46 pp | +1.28 % | 0.075 | 50.0 % | 22.83 % |
| conservative | 14.02 % | +0.16 pp | −0.22 % | 0.043 | 33.3 % | 32.45 % |
| value | 13.91 % | +0.05 pp | −1.03 % | 0.042 | 33.3 % | 32.85 % |
| garp | 12.47 % | −1.39 pp | −2.81 % | −0.013 | 41.7 % | 23.57 % |

**5 de los 8 perfiles baten al índice** en este estudio (frente a solo 1 de 8 en el estudio
anterior), y el perfil recomendado por Information Ratio es `aggressive`. Este cambio de resultado
económico frente al estudio previo **no viene de una señal de aprendizaje más fuerte** (el rank-IC
bajó, no subió) sino de un cambio en la fase de cartera: menos posiciones concentradas
(`target_max` 10 en vez de 12), mayor peso máximo por posición (0.20 vs 0.15) y coste de operación
más bajo (comisión 0 bps vs 5 bps, slippage 5 bps vs 10 bps). Conviene leer la mejora de
rentabilidad con cautela: depende de parámetros de ejecución/costes que se optimizaron sobre el
mismo histórico, no solo de una señal predictiva más sólida.

## Lectura conjunta

Los dos planos, comparados con el estudio anterior, cuentan una historia distinta a la de aquel
cierre:

1. **El aprendizaje es, si acaso, más débil e incierto que antes**: el rank-IC OOS baja de +0.0158
   a +0.0118 y el intervalo de bootstrap pasa de no cruzar cero a cruzarlo. La estabilidad
   leave-one-year-out y la era reservada siguen siendo consistentes con una señal real, pero ya no
   se puede decir sin matices que "la señal sobrevive el contraste más exigente" — el contraste más
   exigente (bootstrap) ahora es ambiguo, y el placebo por permutación, que debía zanjar la duda,
   sigue sin ejecutarse.
2. **La rentabilidad mejora**, pero principalmente por cómo se construye la cartera y se pagan los
   costes, no porque el modelo prediga mejor. Presentar "5 de 8 perfiles baten al SPY" sin esta
   matización sería engañoso.

Es un resultado **mixto, no un cierre limpio**: metodológicamente sigue siendo honesto reportarlo
así, pero no es una mejora en el eje que de verdad importa (el aprendizaje). Ver
[docs/doc.md](doc.md) §8 para la recomendación sobre si iterar más o cerrar aquí.

## Trazabilidad

Todas las cifras proceden de
`results/studies/20260719--optimization-official--acb6c310dfb8/decision.json` y de los
`backtest_summary.json`/entradas de `results/registry.jsonl` de cada `run_id` (final y perfiles:
`balanced`=`20260719--c17769a1926e`, `conservative`=`20260719--3f961d4f3bdc`,
`aggressive`=`20260719--5ace65748bff`, `value`=`20260719--bffacc58f134`,
`quality`=`20260719--3e6a0b6a9c76`, `momentum`=`20260719--5bd6c080582b`,
`garp`=`20260719--27a02d208a94`, `contrarian`=`20260719--79b3c0fb66a3`), reproducibles con
`RUN_MODE=full_study` (ver [docs/doc.md](doc.md)). Todo se puede explorar en la Research Console
(`python main.py`).
