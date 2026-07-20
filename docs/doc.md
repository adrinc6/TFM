# Documentación del sistema — TFM: IA aplicada a la selección de acciones

> Guía única del proyecto: qué se quiere conseguir, con qué metodología, con qué arquitectura, y
> qué se ha encontrado. Es la materia prima del TFM (la redacción en LaTeX es la fase de cierre).
> Documento vivo: se actualiza cuando cambia el diseño o aparecen resultados.

---

## 1. Propósito y tesis

Trabajo Fin de Máster sobre **Inteligencia Artificial y Machine Learning**. El objetivo principal
**no** es ganar dinero ni batir a un índice: es **estudiar cómo aprende un sistema de IA**,
evaluar ese aprendizaje con rigor y comprobar si puede tener **utilidad económica**.

La bolsa es el **banco de pruebas**, no el fin. Se elige porque ofrece un entorno difícil,
ruidoso, no estacionario y con una métrica de éxito clara (batir al S&P 500), pero el mérito
académico está en el **método**, no en un número de rentabilidad aislado.

Consecuencia central: **un resultado negativo bien medido es un entregable válido**. Si el sistema
no aprende a ordenar activos fuera de muestra, decirlo con evidencia es tan valioso como una
cartera rentable. Elegir la mejor semilla, confundir suerte con aprendizaje u ocultar un artefacto
de datos invalidaría el trabajo. Principios: **académico, reproducible, explicable y honesto**.

**La pregunta de investigación**: *¿aprende el sistema a ordenar acciones por su retorno futuro,
de forma estable y fuera de muestra, y ese aprendizaje se traduce en utilidad económica neta?*

---

## 2. Métrica de éxito: el aprendizaje, no la rentabilidad

La métrica principal es el **rank-IC out-of-sample**: la correlación de Spearman, en cada corte
transversal, entre el orden que predice el modelo y el orden que de verdad ocurrió. Se calcula
sobre el **meta-score final** (el que opera la cartera), no sobre agentes individuales.

Por qué el rank-IC y no la rentabilidad: con una señal débil, la rentabilidad de una cartera
concentrada la puede dominar un puñado de aciertos afortunados o —como se observó— **un único
artefacto de datos**. El rank-IC mide si el modelo *ordena bien*, que es lo que de verdad se
pregunta. La rentabilidad se reporta como **consecuencia**, con CAGR real (nunca acumulados
compuestos que inflan y que domina un solo año), y solo tras limpiar artefactos.

Referencia de escala: en la industria un factor se considera útil a partir de rank-IC ~0.03-0.05
sostenido. Se acompaña de **significancia estadística** (bootstrap por bloques temporales, porque
las cohortes no son independientes) y de **tests de robustez/placebo** para distinguir señal de
azar.

---

## 3. Arquitectura del sistema (flujo de datos)

```
descarga → dataset point-in-time → features (+ artefactos) → agentes LightGBM → meta-agente
        → cartera (+ perfil de inversor) → backtest → run inmutable → consola de resultados
                                                     + studies / optimización oficial → decisión
```

Cada etapa se ejecuta por separado (`RUN_MODE`) o encadenada. Sin `RUN_MODE`, `python main.py`
abre la consola local. Cada ejecución nueva se identifica con fecha y hash, y guarda su manifiesto
en `results/runs/`; los studies agrupan escenarios comparables en `results/studies/`.

### 3.1 Datos y universo (point-in-time)
- **Universo dinámico** del S&P 500 por fecha (composición histórica real, no la actual): una
  señal en 2016 solo ve las empresas que estaban en el índice en 2016. Elimina el sesgo de
  inclusión anticipada.
- **Fechas de publicación reales** de SEC EDGAR (`filingDate`): un fundamental solo es observable
  el día en que se publicó, no el día del cierre fiscal. Evita el lookahead.
- **Sesgo de supervivencia medido por año** (no solo declarado): los quebrados no tienen datos en
  fuentes gratuitas; la cobertura pasa de ~50 % en 2000 a ~92 % en 2024. Por eso el sistema se
  centra en **2016+** (más cobertura, menos sesgo).

### 3.2 Dataset point-in-time
Panel `(ticker, snapshot_date)` que reconstruye lo observable en cada fecha: de cada empresa, su
último informe *realmente publicado*. Sin relleno hacia atrás, sin mezclar magnitudes.

### 3.3 Features y artefactos
Factores base (GARP: calidad, crecimiento, valoración; momentum relativo), todos rankeados en el
corte transversal. Encima, un **sistema de artefactos activables** (§4).

### 3.4 Agentes LightGBM
Tres agentes especializados (**calidad, momentum, valor**), cada uno un modelo **LightGBM**
(árboles con gradient boosting, que captura interacciones no lineales que un modelo lineal
promedia a cero). Objetivo `rank_regression`: regresión sobre el percentil transversal del
retorno futuro, alineado con el rank-IC. Entrenamiento **walk-forward**: en cada reentreno solo se
usa historia anterior con etiqueta ya realizada (`label_end_date <= fecha_de_reentreno`).

### 3.5 Meta-agente
Combina los tres agentes en el `meta_score`. Por defecto pondera por el **rank-IC reciente** de
cada agente (`meta_type=rank_ic`), que se midió superior al equiponderado. El `meta_rank` (el
percentil del meta_score) es lo que consume la cartera.

### 3.6 Cartera y backtest
Selección top-N flexible (**8-12 posiciones** por defecto, peso máx 15 %) con reglas de rotación
(umbral de ventaja, expulsión por caída, sin tenencia mínima). Backtest con costes (5 pb comisión
+ 10 pb slippage) y **guarda anti-artefactos**: un retorno mensual imposible de una posición
(>200 %) se trata como dato corrupto y se neutraliza (§4.3). Métricas: rank-IC del meta_final,
CAGR real cartera vs SPY, beat rate, drawdown, turnover.

---

## 4. Sistema de artefactos activables

Un **artefacto** es un bloque de features/contexto que se activa o desactiva por un flag. Todos
son **point-in-time** (verificados con tests de no-lookahead). El barrido los prueba de uno en
uno como *ablations* para medir cuáles suben el rank-IC. Catálogo en `module/modeling/artifacts.py`:

| Artefacto | Qué añade |
|---|---|
| `neutralize_by_sector` | Rankea los factores dentro de sector en vez de global. |
| `fundamental_momentum` | Tendencia de ROE/márgenes + descomposición del cambio de P/E en su parte de precio y su parte fundamental. |
| `market_regime_feature` | Régimen bull/bear del SP500 (vs su media) + interacciones factor×régimen. |
| `price_momentum_multi` | Aceleración (r3m−r12m), reversión (−r1m), volatilidad reciente del activo. |
| `moving_averages` | Precio vs SMA6/SMA12, distancia al máximo de 12m (tendencia individual). |
| `regime_extended` | Volatilidad y drawdown del SP500 (contexto macro más rico). |
| `quality_growth_derived` | Tendencia de ROE, estabilidad de márgenes, sorpresa de crecimiento. |

### 4.2 Cadencia como escenario
El reentreno puede ser **trimestral / semestral / anual** (`fundamental_step_months`), barrido como
escenario. La revisión de cartera es mensual.

### 4.3 Guarda anti-artefactos (rentabilidad honesta)
El estudio previo reveló que un CAGR aparente del 18 % se debía casi por completo a un **artefacto
de datos**: julio de 2010, +953 % en un mes en una posición (precio corrupto). La guarda neutraliza
cualquier retorno mensual de una posición mayor que `max_monthly_position_return` (200 %) y lo
registra, para que la rentabilidad reportada sea honesta.

---

## 5. Ciclo unificado de optimización (study = full_study)

`study` y `full_study` comparten un único orquestador (`run_optimization` en
`module/runs/execution.py`). La diferencia es solo **qué variables se barren**: `full_study` barre
**todas** las barribles (derivadas de `escenarios/variables.py`); un `study` barre las que el
usuario marque. En ambos casos se ejecuta el **ciclo completo**, y la decisión es **automática, sin
intervención humana**.

Las variables se separan por **cuándo actúan en el pipeline** (frontera autoritativa en
`FINGERPRINT_FIELDS`, `module/runs/experiments.py`):
- **Ancla temporal FIJA**: `execution_year = 2015` y `execution_quarter = 1` **no se barren**. Así
  todos los escenarios comparten el mismo periodo OOS (2015→hoy, con 2025-2026 reservados) y la
  comparación de rank-IC es limpia. El **retardo de publicación** `execution_lag_days` sí es variable.
- **Ejes de MODELO** (cambian el rank-IC): ventana (`train_lookback_years`), horizonte, cadencia,
  `execution_lag_days`, `objective`, `meta_type`, `recency_weighting` (pesos de recencia:
  `off`/`linear`/`exponential`), hiperparámetros LightGBM y los 7 artefactos. Se barren en
  **Fase 1/2**, seleccionando por rank-IC OOS.
- **Ejes de CARTERA** (no cambian el rank-IC): `target_min/max`, percentiles, rotación,
  `max_weight`, comisiones, slippage. Se optimizan en una **fase de cartera** por criterio económico
  (Information Ratio, re-backtest sin reentrenar).

### 5.1 Fase 1 — cada eje de modelo aislado
Sobre un baseline común se mueve **un solo eje por escenario** para que su rank-IC mida el efecto
de esa única cosa. `execution_lag_days` entra aquí porque un lag de 15 días cambia qué fundamentales
son observables en cada snapshot (aprovecha antes la publicación) → cambia el dataset → cambia el
aprendizaje. Se elige, por eje, el nivel más estable (mayor rank-IC medio del meta_final; desempate
por fracción de cohortes positivas y menor varianza).

### 5.2 Fase 2 — combinación greedy con top-2 por eje (`_greedy_phase2`)
No es producto cartesiano (2^N, inviable con muchos ejes). Se parte del mejor nivel de cada eje
combinado, se recorren los ejes por su impacto en Fase 1, y en cada eje se prueba su **1º y 2º
mejor** sobre la combinación en curso, fijando el que sube el rank-IC (~2·N runs).

### 5.3 Afinado de hiperparámetros
Sobre el ganador de la Fase 2 se prueban variantes finas de LightGBM (learning rate, nº de árboles,
mínimo de muestras por hoja). Rejilla deliberadamente pequeña por el número limitado de eras
independientes. El resultado es la **configuración final de modelo**, que se entrena en el run final.

### 5.4 Fase de cartera (`_portfolio_phase`)
Sobre el finalista de modelo **ya entrenado**, se optimizan **todos** los ejes de cartera
**re-backtesteando sin reentrenar** (`mode="backtest"` sobre el mismo `agent_dir`). Como estos ejes
no mueven el rank-IC, el criterio es **económico**: **Information Ratio** (rentabilidad ajustada al
riesgo), greedy por eje, fijando el mejor valor. Las combinaciones que violan restricciones de
`Settings` (p.ej. `max_weight · target_min < 1`) se omiten. Estos runs se etiquetan como fase
`cartera` y entran en `comparison_data.parquet`. Salida: la **cartera base óptima**.

### 5.4.b Fase final — perfiles de inversor (la salida del study)
Sobre el modelo y la cartera ya optimizados se aplican los **8 perfiles de inversor**
(`PROFILE_NAMES`) como backtests sin reentrenar (fase `perfiles`). **El resultado final del study
son estos 8 runs**, uno por perfil, todos sobre la configuración óptima. La decisión
(`decision.json`) incluye `final_profile_run_ids` (la salida), `recommended_profile` (el perfil de
mayor **Information Ratio**) y `best_config` (mejor modelo + mejor gestión de cartera + perfil
recomendado).

### 5.5 Control de overfitting por selección: era reservada
Mirar muchos escenarios sube el riesgo de que el máximo sea **suerte**. Por eso la selección
(Fases 1 y 2 y afinado) usa **solo** cohortes hasta **2024** (`SELECTION_UNTIL_YEAR`); **2025-2026
se reservan** y no intervienen en ninguna elección. Al final se mide el rank-IC del finalista en
esa era reservada (`reserved_era_validation` en `study_summary.json`): si aguanta ahí, la señal no
es solo un artefacto de haber explorado mucho. Es un filtro *point-in-time* sobre cohortes ya
calculadas — no reentrena ni mira al futuro.

La reutilización por huella SHA-256 evita recomputar etapas compartidas entre escenarios.

---

## 6. Perfiles de inversor (explicabilidad como funcionalidad)

El sistema explica *por qué* cada acción está arriba: cada agente aporta su rango (calidad,
momentum, valor). Sobre eso se construyen **perfiles de inversor** que, entre las **buenas**
acciones del meta (percentil alto), reordenan según estilo — no siempre cogen el top-N puro
(`module/evaluation/profiles.py`):

`balanced` (referencia), `conservative` (calidad + estabilidad), `aggressive` (momentum),
`value` (barato y bueno), `quality` (mejor negocio), `momentum` (fuerza relativa), `garp`
(equilibrio), `contrarian` (bueno pero castigado, apuesta a reversión).

Cada perfil se mide como un backtest (misma señal, distinta cartera): permite comparar el
trade-off rentabilidad/riesgo de cada estilo usando el mismo modelo.

---

## 7. Robustez / placebo (credibilidad)

Sobre la configuración final se ejecutan tests que demuestran que el resultado no es suerte
(`module/evaluation/robustness.py`):

- **Permutación de etiquetas** (placebo): se reentrena con los retornos futuros barajados; el
  rank-IC debe **colapsar a ~0**. Si no colapsa, hay fuga de información (también detecta leakage).
- **Carteras aleatorias (Monte Carlo)**: la cartera del modelo se compara con ~1000 aleatorias del
  mismo tamaño; su percentil dice si su rendimiento es distinguible del azar.
- **Bootstrap por bloques** del rank-IC (`module/evaluation/stats.py`): intervalo de confianza que respeta el
  solapamiento temporal.
- **Leave-one-year-out**: se quita cada año para ver si el resultado depende de uno o dos.

---

## 8. Resultados

> **Estado: cerrado, con una pieza de robustez aún pendiente.** Cifras del estudio
> `results/studies/20260719--optimization-official--acb6c310dfb8/` (sustituye al estudio
> `20260718--...--bef48ddfc41f--r02`, citado en versiones anteriores de este documento). El
> detalle completo con tablas está en [informe_final.md](informe_final.md).

La configuración final (elegida por rank-IC OOS hasta 2024, no por rentabilidad): ventana de 12
años, horizonte 3 meses, `objective`=quartile, `max_depth`=6, `learning_rate`=0.10, `meta_type`=equal,
`recency_weighting`=linear, y siete artefactos activos (neutralización por sector, momentum
fundamental, régimen de mercado, momentum de precio multi-horizonte, medias móviles, régimen
extendido, `quality_growth_derived`). Resultado en dos planos separados:

- **Aprendizaje (rank-IC OOS del `meta_final`)**: **+0.0118** de media, 60 % de cohortes positivas.
  Bootstrap por bloques con IC 95 % **[-0.0113, 0.0340]** — **cruza cero** (45 cohortes, bloque 4).
  Leave-one-year-out entre 0.0079 y 0.0183 (ningún año domina) y **+0.0210 en la era reservada
  2025-2026** (6 cohortes, nunca optimizada). El **placebo por permutación de etiquetas sigue sin
  ejecutarse** (`n_permutations=0`), igual que en el estudio anterior. Comparado con el estudio
  previo (rank-IC +0.0158, IC bootstrap sin cruzar cero), esta repetición da una señal **más débil y
  con evidencia estadística más ambigua**, no más sólida. La respuesta a la pregunta de
  investigación (§1) sigue siendo *tentativamente* sí, pero con menos margen que antes: el
  contraste más exigente (bootstrap) ya no es concluyente, y sin el placebo por permutación no se
  puede descartar formalmente que parte de la señal sea artefacto del proceso de búsqueda.
- **Rentabilidad como consecuencia** (por perfil de inversor, §6, guarda anti-artefactos activa,
  §4.3): frente al SPY (CAGR 13.86 %), **5 de 8 perfiles baten al índice** (`aggressive` +2.80 pp,
  `contrarian` +2.48 pp, `momentum` +1.90 pp, `balanced` +1.25 pp, `quality` +0.46 pp); solo
  `conservative`, `value` y `garp` quedan por debajo. El perfil recomendado por Information Ratio es
  **`aggressive`** (IR 0.153, alfa medio +1.88 %, drawdown 25.2 %). Esta mejora frente al estudio
  anterior (donde solo 1 de 8 perfiles ganaba) se explica sobre todo por parámetros de cartera y
  costes más favorables (menos posiciones, mayor peso máximo, comisión y slippage más bajos), **no**
  por una señal de aprendizaje más fuerte. Nunca se usó la rentabilidad como selector de
  configuración de modelo (§2).

**Lectura conjunta**: es un resultado **mixto**, no una mejora limpia respecto al cierre anterior.
El aprendizaje (lo que de verdad responde a la pregunta de investigación) es igual o algo más débil
que antes, con una pieza de robustez —el placebo por permutación— todavía sin ejecutar tras dos
estudios oficiales consecutivos. La rentabilidad mejora, pero por construcción de cartera y costes,
no por mejor predicción. Todas las cifras son trazables al manifiesto y la comparación de
`results/studies/<study_id>/` y reproducibles con un comando (§9, §11). Ver conclusiones y
recomendación completas en el mensaje de cierre de este análisis (resumen crítico entregado al
usuario tras este estudio).

---

## 9. Cómo ejecutar

Requiere `data/raw` ya descargado (la descarga es un paso aparte, `RUN_MODE=download`).

- **Estudio completo de principio a fin** (recomendado):
  `RUN_MODE=full_study RUN_SCOPE=full python main.py`
  Ejecuta Fase 1 (ejes aislados) → decisión automática del mejor nivel de cada eje + artefactos →
  Fase 2 (combinaciones dirigidas) → afinado de hiperparámetros → run final optimizado → perfiles →
  robustez → **validación en la era reservada 2025-2026** → informes HTML. Sin decisiones humanas.
- **Etapas sueltas**: `dataset`, `features`, `agents`, `backtest`, `report`, `experiments`
  (`experiments` corre solo el barrido de artefactos de `escenarios/rejilla_base.py`).
- **Resultados inmutables**: `results/runs/<id>/` contiene los artefactos de cada ejecución y
  `results/studies/<id>/` su manifiesto, comparación y lista de runs. La Research Console los
  consulta directamente, sin depender del directorio mutable `data/processed/`.

### 9.1 Organización del código

La implementación sigue seis dominios: `module/data/` (datos e ingestión), `module/modeling/`
(features y agentes), `module/evaluation/` (cartera y validación), `module/runs/` (orquestación,
caché y resultados), `module/ui/` (Research Console: servidor/API en `dashboard.py` e informes
estáticos en `reports.py`) y `module/common/` (utilidades). El frontend vive en `app/`, en la
raíz del proyecto (junto a `module/`, `results/` y `docs/`).
- **Ver resultados**: ejecutar `python main.py` sin `RUN_MODE` y abrir la Research Console en
  `http://127.0.0.1:8765`. Es una aplicación web oscura (negros y grises) servida como archivos
  reales desde `app/` (Chart.js embebido localmente, sin CDN). Su vista de Resultados
  tiene listas separadas de **estudios** y **runs**: al elegir un estudio se analiza el estudio
  y al elegir un run se analizan resumen, rendimiento, aprendizaje, cartera, trades, explorador
  de stocks y ficha por ticker, con tablas y gráficos.

---

## 10. Limitaciones (medidas, no estimadas)

1. **Sesgo de supervivencia parcialmente irreducible**: los quebrados no tienen datos gratuitos.
   Se mitiga centrándose en 2016+ (cobertura ~65-92 %) y se reporta por año.
2. **Restatements invisibles**: las series de fundamentales son valores actuales; EDGAR fecha
   *cuándo* se publicó, pero el *valor* podría estar reexpresado. Lookahead residual no eliminable
   con datos gratuitos.
3. **Muestra pequeña de eras independientes**: intervalos de confianza anchos; por eso la
   significancia se mide con bootstrap por bloques y se exige estabilidad temporal.
4. **Selección temporal declarada**: centrarse en 2016+ es una decisión basada en cobertura y
   diagnóstico; se reporta también el histórico completo para que el lector vea cuánto cambia.

---

## 11. Reproducibilidad

Todo el estudio es re-ejecutable con un comando. Cada run lleva su `manifest.json` con la
configuración, la semilla, las versiones y los hashes de los inputs. La **bitácora**
(`docs/bitacora.md`) registra el porqué de cada decisión y cada resultado —el hilo narrativo del
desarrollo—. Los tests (`pytest tests/`) cubren la ausencia de lookahead, las reglas de cartera, la
guarda anti-artefactos, la decisión automática, los perfiles y la robustez.
