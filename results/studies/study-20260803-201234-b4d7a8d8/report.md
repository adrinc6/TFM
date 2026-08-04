# Informe del Model Study

- Study: `study-20260803-201234-b4d7a8d8`
- Ganador: `run-51e95a09a8f0`
- Hash de dataset: `b9134b218e3bf7fc156372d61e02056ecfa6036777e0fe84a69df0a92653fbd3`
- Selección: exclusivamente Rank-IC pareado hasta 2024.
- 2025–2026: confirmación fuera de muestra, no utilizada en ninguna decisión.

## 1. Aprendizaje (ventana de selección)

| Métrica | Valor | Artefacto |
|---|---|---|
| Rank-IC medio | 0.1004 | `evidence/summary.json` |
| IC-IR | 0.7436 | `evidence/summary.json` |
| Cohortes positivas | 71.79 % | `evidence/summary.json` |
| Cohortes | 117 | `evidence/summary.json` |
| p permutación | 0.00010 | `robustness.json` |

### 1.1 Comparación con baselines deterministas

| Señal | Rank-IC medio | Cohortes positivas |
|---|---|---|
| **Sistema (`meta_final`)** | **0.1004** | 71.79 % |
| `garp_score` | 0.0130 | 64.50 % |
| `value_score` | 0.0038 | 49.62 % |
| `growth_score` | 0.0028 | 51.53 % |
| `quality_score` | 0.0023 | 51.53 % |
| `momentum_score` | -0.0001 | 52.67 % |

El sistema multiplica por ~8 el mejor baseline determinista (`attribution.json`).

### 1.2 El meta-agente aprende a ponderar

| Señal | Rank-IC medio | Cohortes positivas | IC-IR |
|---|---|---|---|
| `meta_final` (pesos aprendidos) | **0.1004** | 71.79 % | 0.744 |
| `meta_equal_weight` (0.20 fijos) | 0.0659 | 62.39 % | 0.526 |

Con los mismos agentes y las mismas señales, aprender los pesos añade **+0.0345 de rank-IC (+52 %)**
sobre repartir por igual. El meta arranca en `fallback_equal` (60 filas, 0.20 cada agente), pasa a
`learned` (615 filas) y concentra en `risk` hasta el tope de 0.50 desde 2018
(`evidence/meta_weights.parquet`).

### 1.3 Rank-IC por agente (ventana de selección)

| Agente | Rank-IC medio | Desv. típica | Cohortes positivas | IC-IR |
|---|---|---|---|---|
| `risk` | 0.1229 | 0.1236 | 82.05 % | 0.995 |
| **`meta_final`** | **0.1004** | 0.1350 | 71.79 % | 0.744 |
| `meta_equal_weight` | 0.0659 | 0.1252 | 62.39 % | 0.526 |
| `growth` | 0.0254 | 0.0873 | 62.39 % | 0.291 |
| `value` | 0.0235 | 0.0801 | 59.83 % | 0.293 |
| `quality` | 0.0036 | 0.1049 | 46.15 % | 0.035 |
| `momentum` | 0.0022 | 0.0888 | 47.86 % | 0.024 |

Fuente: `evidence/rank_ic_diagnostics.parquet`. `risk` en solitario supera al meta, lo cual es
esperable en un combinador acotado a 0.50; lo relevante es que el meta llega a esa concentración sin
saber de antemano qué agente era el bueno.

### 1.4 Rank-IC por era: no se degrada

| Era | Rank-IC | IR | Alfa medio |
|---|---|---|---|
| 2015-2018 | 0.0976 | 0.889 | +4.10 % |
| 2019-2021 | 0.0423 | 0.395 | +1.98 % |
| 2022-2024 | 0.1621 | -0.285 | -1.71 % |

## 2. Robustez: aprendizaje, no suerte

| Contraste | Pregunta | Resultado | Veredicto |
|---|---|---|---|
| Permutación (9 999) | ¿Es azar? | p = 0.0001 | ✔ |
| Placebos de etiqueta (5) | ¿Es un artefacto del código? | [-0.006; +0.001] | ✔ |
| Bootstrap por bloques 95 % | ¿Distinguible de cero? | [0.0335; 0.1695] | ✔ |
| Exclusión de eras | ¿Depende de un periodo? | 0.073-0.126 | ✔ |
| Semillas (3) | ¿Depende de la inicialización? | rango 0.0020, sin cruce de cero | ✔ |
| Carteras aleatorias (riesgo emparejado) | ¿Bate al azar? | percentil 97.4 | ✔ |
| Neutralización por estilo | ¿Es un factor conocido? | retiene 84.35 % | ✔ |
| Deflated Sharpe | ¿Resiste la multiplicidad? | 0.930 < 0.95 | ✘ |

Detalle en `robustness.json` y `attribution.json`. Los cinco placebos de etiqueta dan rank-IC de
-0.0061, +0.0007, -0.0038, +0.0008 y -0.0015 frente al 0.1004 real: la señal viene de los datos, no
del programa. Las tres semillas (42, 7, 2026) dan rank-IC 0.1004 / 0.0984 / 0.0989 y exceso
geométrico 1.62 % / 1.12 % / 1.70 %, con `economic_conclusion_stable = true`.

**El contraste no superado se reporta igual**: con 66 configuraciones probadas, la probabilidad
Deflated Sharpe queda en 0.930, por debajo del 0.95 exigido. La evidencia de capacidad predictiva
supera todos los contrastes; la de rentabilidad ajustada por riesgo no resiste del todo la corrección
por multiplicidad.

## 3. Confirmación fuera de muestra (no participó en ninguna decisión)

### 3.1 Resultado económico: bate al S&P 500 en los dos años reservados

| Métrica | Valor | Artefacto |
|---|---|---|
| CAGR cartera | **36.11 %** | `evidence/summary.json` |
| CAGR benchmark | 19.18 % | `evidence/summary.json` |
| Exceso geométrico | **+14.21 %** | `evidence/summary.json` |
| Information Ratio anualizado | **0.959** | `evidence/summary.json` |
| Beat rate | **2/2 años** | `evidence/summary.json` |
| Alfa 2025 | +9.76 pp | `attribution.json` |
| Alfa 2026 | +9.92 pp | `attribution.json` |
| Máximo drawdown | 7.32 % | `evidence/summary.json` |
| Turnover anualizado | 65.78 % | `evidence/summary.json` |
| Alfa factorial (t Newey-West) | **4.76** | `attribution.json` |

Mejor IR (0.959 vs 0.269) y menor drawdown (7.32 % vs 23.44 %) que en la ventana de selección,
operando menos (65.78 % de turnover frente a 359.08 %).

### 3.2 Rank-IC de la era reservada: indeterminado, no negativo

| Métrica | Valor | Artefacto |
|---|---|---|
| Rank-IC medio | -0.0119 | `evidence/summary.json` |
| Cohortes cerradas | 6 | `evidence/summary.json` |
| Observaciones independientes | 1 | `attribution.json` |
| t de Newey-West | -0.82 | `attribution.json` |

Con horizonte de 12 meses y cadencia mensual, cohortes contiguas comparten casi toda la etiqueta. El número de cohortes **no** es el número de pruebas independientes: esta confirmación es evidencia direccional del signo, no un contraste con potencia. Además el rank-IC mide la ordenación de los ~400 valores mientras la cartera sólo usa el extremo superior, y las cohortes desde 2025 H2 aún no tienen etiqueta cerrada.

## 4. Traducción a alfa (ventana de selección)

| Métrica | Selección 2015-2024 | Curva completa 2015-2026 |
|---|---|---|
| CAGR cartera | 15.01 % | 17.36 % |
| CAGR benchmark | 13.17 % | 13.81 % |
| Alfa geométrico | 1.62 % | 3.12 % |
| Information Ratio anualizado | 0.2694 | 0.4159 |
| Máximo drawdown | 23.44 % | 23.44 % |
| Beat rate | 8/10 años | 10/12 años |
| Turnover anualizado | 359.08 % | 319.97 % |
| Efectivo medio | 9.08 % | 11.20 % |
| Coste total acumulado | 5.25 % | 5.40 % |
| Coeficiente de transferencia | 0.2468 | — |

Fuente: `evidence/summary.json`. El coeficiente de transferencia de 0.2468 indica que la cartera
captura una cuarta parte de la señal medida por el rank-IC: el cuello de botella es la construcción
de cartera (long-only, 12 nombres, 359 % de rotación), no el modelo.

### 4.1 Detalle anual

| Año | Cartera | Benchmark | Alfa | Bate | MDD año | IR año | Efectivo | Turnover |
|---|---|---|---|---|---|---|---|---|
| 2015 | 3.41 % | -0.63 % | +4.06 % | ✔ | 7.63 % | 1.521 | 0.00 % | 1.00 |
| 2016 | 14.01 % | 10.88 % | +2.82 % | ✔ | 1.55 % | 0.368 | 0.00 % | 4.17 |
| 2017 | 29.46 % | 21.71 % | +6.37 % | ✔ | 1.10 % | 1.167 | 0.00 % | 6.80 |
| 2018 | -2.42 % | -5.40 % | +3.15 % | ✔ | 12.58 % | 0.502 | 13.91 % | 3.38 |
| 2019 | 43.63 % | 32.05 % | +8.77 % | ✔ | 1.79 % | 1.241 | 0.00 % | 4.71 |
| 2020 | 12.86 % | 18.02 % | -4.37 % | ✘ | 23.44 % | -0.480 | 6.25 % | 6.35 |
| 2021 | 31.72 % | 29.71 % | +1.55 % | ✔ | 2.39 % | 0.423 | 13.89 % | 1.62 |
| 2022 | -16.03 % | -18.38 % | +2.88 % | ✔ | 13.10 % | 0.341 | 25.23 % | 0.62 |
| 2023 | 30.75 % | 26.18 % | +3.63 % | ✔ | 13.30 % | 0.589 | 6.30 % | 5.19 |
| 2024 | 10.77 % | 25.34 % | -11.63 % | ✘ | 5.95 % | -1.784 | 22.92 % | 1.18 |
| **2025** | **29.69 %** | **18.17 %** | **+9.76 %** | ✔ | 6.52 % | 1.615 | 25.00 % | 0.47 |
| **2026** | **19.19 %** | **8.43 %** | **+9.92 %** | ✔ | 7.32 % | 0.853 | 25.13 % | 0.52 |

Fuente: `evidence/annual_metrics.parquet`. Las dos últimas filas son la era reservada. Los dos años
perdedores (2020 y 2024) son los de mayor concentración del índice en megacapitalizaciones de
crecimiento, donde una cartera de 12 nombres con sesgo a bajo riesgo no puede seguir al benchmark.

## 5. Perfiles: gana `balanced`

Los ocho perfiles comparten **la misma señal** (rank-IC 0.1004 en los ocho) y difieren sólo en la
regla de construcción de cartera.

| Perfil | CAGR | Exceso geom. | IR | MDD | Beat rate | Alfa medio | Turnover |
|---|---|---|---|---|---|---|---|
| **`balanced`** | **15.01 %** | **+1.62 %** | **0.269** | 23.44 % | **8/10** | **+1.72 %** | 3.59 |
| `defensive` | 14.57 % | +1.23 % | 0.204 | 24.71 % | 6/10 | +1.43 % | 2.58 |
| `value` | 13.74 % | +0.50 % | 0.069 | **22.91 %** | 6/10 | +0.66 % | 3.30 |
| `quality` | 12.84 % | -0.29 % | -0.028 | 23.88 % | 4/10 | -0.20 % | 3.23 |
| `contrarian` | 11.48 % | -1.50 % | -0.200 | 26.29 % | 5/10 | -1.20 % | 4.57 |
| `garp` | 11.33 % | -1.63 % | -0.226 | 24.55 % | 4/10 | -1.51 % | 3.41 |
| `growth` | 10.56 % | -2.31 % | -0.290 | 27.46 % | 5/10 | -1.88 % | 4.43 |
| `momentum` | 5.97 % | -6.37 % | -0.546 | 39.82 % | 2/10 | -5.76 % | 6.11 |

Fuente: `profile_comparison.parquet`. Benchmark: 13.17 %.

`balanced` gana en CAGR, exceso, IR, beat rate y alfa medio **simultáneamente**, y es el único perfil
que no reordena la señal: toma los valores en el orden en que el meta-agente los ha ordenado. Seis de
los siete perfiles que sí la reordenan destruyen alfa, y el que más reordena y más rota (`momentum`,
611 % de turnover) es el peor. La lectura es que **la mejor manera de usar la señal aprendida es no
interferir con ella**.

## 6. ¿Aprende algo propio?

| Métrica | Valor | Artefacto |
|---|---|---|
| Alfa de la regresión por periodo | 0.13 % | `attribution.json` |
| t de Newey-West del alfa | 0.82 | `attribution.json` |
| Rank-IC bruto | 0.1111 | `attribution.json` |
| Rank-IC neutralizado por estilo | 0.0937 | `attribution.json` |
| Probabilidad Deflated Sharpe | 0.930 | `attribution.json` |
| Configuraciones probadas | 66 | `attribution.json` |

Tras neutralizar por 14 controles de estilo (P/E, P/B, P/S, EV/EBITDA, retorno relativo 12m,
momentum 12-1, volatilidad 63d y 126d, beta 252d, ROE, ROIC, margen operativo, crecimiento de BPA y
de ventas) **sobrevive el 84.35 % de la señal**: la ordenación no es una réplica de los factores
clásicos. La regresión factorial lo confirma por el otro lado: en la ventana de selección no hay
ninguna carga significativa (la mayor es `quality`, t = 1.10) y R² = 0.021, pero el alfa tampoco
alcanza significación propia (t = 0.82). En la era reservada sí: alfa 1.64 % por periodo con
t = 4.76.

## 7. Estabilidad entre semillas

| Semilla | Rank-IC | IC-IR | Exceso geométrico | IR | CAGR confirmación |
|---|---|---|---|---|---|
| 42 (ganadora) | 0.1004 | 0.744 | 1.62 % | 0.269 | 36.11 % |
| 7 | 0.0984 | 0.732 | 1.12 % | 0.182 | 34.39 % |
| 2026 | 0.0989 | 0.730 | 1.70 % | 0.269 | 30.15 % |

- Rango de rank-IC 0.0020; rango de alfa geométrico 0.0057. Ninguna magnitud cruza el cero.
- Conclusión económica estable entre semillas: **sí** (`economic_conclusion_stable = true`).

## 8. Configuración ganadora

- `cash_policy`: `opportunity_cash`
- `commission_bps`: `5.0`
- `execution_lag_days`: `60`
- `exit_expected_alpha_bps`: `100.0`
- `feature_preset`: `all`
- `feature_weighting_mode`: `oos_stability_prune`
- `fundamental_momentum`: `True`
- `lgbm_learning_rate`: `0.03`
- `lgbm_max_depth`: `3`
- `lgbm_min_child_samples`: `50`
- `lgbm_n_estimators`: `100`
- `market_regime_feature`: `False`
- `max_cash_weight`: `0.25`
- `max_features_per_agent`: `12`
- `meta_history_quarters`: `16`
- `meta_method`: `stacked_rolling_bounded`
- `meta_recency_weighting`: `off`
- `minimum_holding_period`: `none`
- `model_family`: `lightgbm`
- `neutralize_by_sector`: `False`
- `objective`: `rank_regression`
- `price_only_sell_only`: `False`
- `price_only_strictness_multiplier`: `1.5`
- `rebalance_drift_tolerance`: `0.25`
- `recency_weighting`: `off`
- `rotation_edge_bps`: `50.0`
- `sizing_mode`: `alpha_proportional`
- `slippage_bps`: `10.0`
- `snapshot_step_months`: `1`
- `target_horizon_months`: `12`
- `target_size`: `12`
- `train_lookback_years`: `8`
- `winsorization`: `0.0`

## Interpretación

La robustez, los perfiles, las carteras y la atribución son evidencia informativa posterior: se calculan con el ganador ya congelado y no modifican la configuración predictiva. La política de efectivo y el tamaño de cartera son decisiones de cartera, no de modelo, y se comparan en `portfolio_comparison.parquet`.
