# Informe final — Sistema de IA para selección de acciones

> **Estudio cerrado.** Cifras procedentes de
> `results/studies/20260718--optimization-official--bef48ddfc41f--r02/` (orquestador de dos fases
> + era reservada 2025-2026), reproducibles con `RUN_MODE=full_study` (ver [docs/doc.md](doc.md)).
> El hilo de decisiones está en [docs/bitacora.md](bitacora.md).

El resultado se presenta en **dos planos separados**, sin mezclarlos: qué **aprende** el sistema
y qué **renta**. La configuración final se eligió por aprendizaje y estabilidad (rank-IC OOS hasta
2024), nunca por rentabilidad.

## Configuración final seleccionada

Elegida automáticamente por el barrido dirigido (Fase 1 aísla cada eje, Fase 2 combina los
ganadores, luego afinado de hiperparámetros):

| Parámetro | Valor |
|---|---|
| Ventana de entrenamiento | 12 años |
| Ancla de ejecución | 2014 |
| Horizonte de etiqueta | 3 meses (baseline) |
| Cadencia de fundamentales | trimestral (baseline) |
| `lgbm_max_depth` | 6 |
| `lgbm_learning_rate` | 0.10 |
| Artefactos activos | neutralización por sector, `quality_growth_derived` |

`run_id` final: `20260718--bb47e7715141`.

## Plano 1 — El aprendizaje (rank-IC OOS del `meta_final`)

Es la respuesta directa a la pregunta de investigación: **el sistema sí aprende a ordenar acciones
fuera de muestra, con una señal débil pero real y estadísticamente estable.**

| Métrica | Valor |
|---|---|
| rank-IC medio (OOS) | **+0.0158** |
| Fracción de cohortes con IC > 0 | 56.5 % |
| Desviación típica del rank-IC | 0.0566 |
| **Bootstrap por bloques** (IC 95 %) | **[0.0053, 0.0265]** — no cruza cero (147 cohortes, bloque 4) |
| **Estabilidad leave-one-year-out** | rank-IC entre 0.013 y 0.018 al excluir cualquier año; ningún año domina |
| **Era reservada 2025-2026** (nunca optimizada) | **rank-IC +0.0426** sobre 16 cohortes |

La señal **sobrevive fuera del periodo de búsqueda** (era reservada) y es robusta al bootstrap y
al leave-one-year-out. Es el contraste más exigente y lo pasa.

> **Pendiente:** el placebo por **permutación de etiquetas** (`label_permutation`) figura como
> "pendiente de ejecución aislada" (0 permutaciones); el resto de robustez sí se ejecutó. Conviene
> cerrarlo antes de la versión definitiva para reportar el p-valor del placebo.

## Plano 2 — La rentabilidad como consecuencia (por perfil de inversor)

Todos los perfiles parten del **mismo modelo** (de ahí el rank-IC común 0.0158); solo cambia cómo
se construye la cartera. Con la guarda anti-artefactos activa, frente al **SPY (CAGR 13.92 %)**:

| Perfil | CAGR cartera | Dif. vs SPY | Alfa medio anual | Beat rate | Max drawdown |
|---|---:|---:|---:|---:|---:|
| **quality** | **16.94 %** | **+3.02 pp** | **+1.70 %** | 46.2 % | 23.44 % |
| contrarian | 12.44 % | −1.49 pp | −1.07 % | 46.2 % | 32.02 % |
| balanced | 12.12 % | −1.81 pp | −3.04 % | 30.8 % | 24.05 % |
| conservative | 11.99 % | −1.93 pp | −1.57 % | 38.5 % | 32.01 % |
| garp | 9.74 % | −4.19 pp | −3.90 % | 38.5 % | 32.85 % |
| aggressive | 9.49 % | −4.44 pp | −5.33 % | 30.8 % | 29.11 % |
| momentum | 9.47 % | −4.46 pp | −5.25 % | 38.5 % | 29.03 % |
| value | 8.78 % | −5.15 pp | −3.92 % | 46.2 % | 42.15 % |

**El perfil `quality` es el único que bate al índice** (+3.02 pp/año, alfa positivo y el menor
drawdown). El resto queda por debajo del SPY: la señal aprendida no basta, en general, para superar
al índice tras la construcción de cartera y los costes.

## Lectura conjunta

Los dos planos no se contradicen, se complementan y **refuerzan** la conclusión del proyecto:

1. **El sistema aprende**: rank-IC OOS positivo, estable, robusto y que aguanta en la era reservada.
   No es azar.
2. **Aprender no equivale a batir al mercado**: salvo el sesgo hacia calidad, las carteras no
   superan al SPY. La señal es real pero económicamente marginal una vez pasada por la cartera y los
   costes.

Es un resultado negativo bien medido en lo económico y positivo en lo metodológico: separa con
nitidez *qué aprende* el modelo de *cuánto renta*, que era el objetivo del TFM.

## Trazabilidad

Todas las cifras proceden de `results/studies/20260718--optimization-official--bef48ddfc41f--r02/`
(`decision.json`, `comparison_data.parquet`) y de los `backtest_summary.json` de cada `run_id`
(final y perfiles), reproducibles con `RUN_MODE=full_study` (ver [docs/doc.md](doc.md)). Todo se
puede explorar en la Research Console (`python main.py`).
