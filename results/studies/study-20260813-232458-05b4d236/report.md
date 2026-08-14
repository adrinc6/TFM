# Informe del Model Study

- Study: `study-20260813-232458-05b4d236`
- Ganador: `run-d304f6074665`
- Hash de dataset: `b9134b218e3bf7fc156372d61e02056ecfa6036777e0fe84a69df0a92653fbd3`
- Selección: exclusivamente Rank-IC pareado hasta 2024.
- 2025–2026: confirmación fuera de muestra, no utilizada en ninguna decisión.

## 1. Aprendizaje (ventana de selección)

| Métrica | Valor | Artefacto |
|---|---|---|
| Rank-IC medio | 0.1090 | `evidence/summary.json` |
| IC-IR | 0.8506 | `evidence/summary.json` |
| Cohortes positivas | 74.36 % | `evidence/summary.json` |
| Cohortes | 117 | `evidence/summary.json` |
| p permutación | 0.00010 | `robustness.json` |

## 2. Confirmación fuera de muestra (no participó en ninguna decisión)

| Métrica | Valor | Artefacto |
|---|---|---|
| Rank-IC medio | 0.0441 | `evidence/summary.json` |
| Cohortes cerradas | 6 | `evidence/summary.json` |
| Observaciones independientes | 1 | `attribution.json` |

Con horizonte de 12 meses y cadencia mensual, cohortes contiguas comparten casi toda la etiqueta. El número de cohortes **no** es el número de pruebas independientes: esta confirmación es evidencia direccional del signo, no un contraste con potencia.

## 3. Traducción a alfa (ventana de selección)

| Métrica | Valor | Artefacto |
|---|---|---|
| CAGR cartera | 13.61 % | `evidence/summary.json` |
| CAGR benchmark | 13.17 % | `evidence/summary.json` |
| Alfa geométrico | 0.39 % | `evidence/summary.json` |
| Information Ratio anualizado | 0.1210 | `evidence/summary.json` |
| Turnover anualizado | 353.63 % | `evidence/summary.json` |
| Efectivo medio | 7.27 % | `evidence/summary.json` |
| Coeficiente de transferencia | 0.0486 | `evidence/summary.json` |

## 4. ¿Aprende algo propio?

| Métrica | Valor | Artefacto |
|---|---|---|
| Alfa de la regresión por periodo | 0.08 % | `attribution.json` |
| t de Newey-West del alfa | 0.46 | `attribution.json` |
| Rank-IC bruto | 0.1177 | `attribution.json` |
| Rank-IC neutralizado por estilo | 0.1019 | `attribution.json` |
| Probabilidad Deflated Sharpe | 0.584 | `attribution.json` |
| Configuraciones probadas | 71 | `attribution.json` |

## 5. Estabilidad entre semillas

- Alfa geométrico: mínimo -0.46 %, mediana -0.36 %, máximo 0.39 %.
- Conclusión económica estable entre semillas: **no**.

## 6. Configuración ganadora

- `commission_bps`: `5.0`
- `coverage_percentile_floor`: `60.0`
- `execution_lag_days`: `60`
- `exit_expected_alpha_bps`: `100.0`
- `feature_preset`: `all`
- `feature_weighting_mode`: `oos_stability_prune`
- `fundamental_momentum`: `True`
- `lgbm_learning_rate`: `0.03`
- `lgbm_max_depth`: `3`
- `lgbm_min_child_samples`: `20`
- `lgbm_n_estimators`: `100`
- `market_regime_feature`: `False`
- `max_cash_weight`: `0.25`
- `max_features_per_agent`: `20`
- `meta_history_quarters`: `16`
- `meta_method`: `stacked_rolling_free`
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
