# Informe del Model Study

- Study: `study-20260804-122546-f140435d`
- Ganador: `run-a7412bcbb17a`
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

## 2. Confirmación fuera de muestra (no participó en ninguna decisión)

| Métrica | Valor | Artefacto |
|---|---|---|
| Rank-IC medio | -0.0119 | `evidence/summary.json` |
| Cohortes cerradas | 6 | `evidence/summary.json` |
| Observaciones independientes | 1 | `attribution.json` |

Con horizonte de 12 meses y cadencia mensual, cohortes contiguas comparten casi toda la etiqueta. El número de cohortes **no** es el número de pruebas independientes: esta confirmación es evidencia direccional del signo, no un contraste con potencia.

## 3. Traducción a alfa (ventana de selección)

| Métrica | Valor | Artefacto |
|---|---|---|
| CAGR cartera | 14.60 % | `evidence/summary.json` |
| CAGR benchmark | 13.17 % | `evidence/summary.json` |
| Alfa geométrico | 1.26 % | `evidence/summary.json` |
| Information Ratio anualizado | 0.2466 | `evidence/summary.json` |
| Turnover anualizado | 624.21 % | `evidence/summary.json` |
| Efectivo medio | 0.00 % | `evidence/summary.json` |
| Coeficiente de transferencia | 0.1919 | `evidence/summary.json` |

## 4. ¿Aprende algo propio?

| Métrica | Valor | Artefacto |
|---|---|---|
| Alfa de la regresión por periodo | 0.13 % | `attribution.json` |
| t de Newey-West del alfa | 0.91 | `attribution.json` |
| Rank-IC bruto | 0.1111 | `attribution.json` |
| Rank-IC neutralizado por estilo | 0.0937 | `attribution.json` |
| Probabilidad Deflated Sharpe | 0.902 | `attribution.json` |
| Configuraciones probadas | 42 | `attribution.json` |

## 5. Estabilidad entre semillas

- Alfa geométrico: mínimo 1.05 %, mediana 1.26 %, máximo 1.61 %.
- Conclusión económica estable entre semillas: **sí**.

## 6. Configuración ganadora

- `commission_bps`: `5.0`
- `coverage_percentile_floor`: `80.0`
- `execution_lag_days`: `60`
- `exit_expected_alpha_bps`: `250.0`
- `feature_preset`: `all`
- `feature_weighting_mode`: `oos_stability_prune`
- `fundamental_momentum`: `True`
- `lgbm_learning_rate`: `0.03`
- `lgbm_max_depth`: `3`
- `lgbm_min_child_samples`: `50`
- `lgbm_n_estimators`: `100`
- `market_regime_feature`: `False`
- `max_cash_weight`: `0.0`
- `max_features_per_agent`: `12`
- `meta_history_quarters`: `16`
- `meta_method`: `stacked_rolling_bounded`
- `meta_recency_weighting`: `off`
- `minimum_holding_period`: `none`
- `model_family`: `lightgbm`
- `neutralize_by_sector`: `False`
- `objective`: `rank_regression`
- `price_only_sell_only`: `False`
- `price_only_strictness_multiplier`: `2.0`
- `rebalance_drift_tolerance`: `0.25`
- `recency_weighting`: `off`
- `rotation_edge_bps`: `25.0`
- `sizing_mode`: `alpha_proportional`
- `slippage_bps`: `10.0`
- `snapshot_step_months`: `1`
- `target_horizon_months`: `12`
- `target_size`: `12`
- `train_lookback_years`: `8`
- `winsorization`: `0.0`

## Interpretación

La robustez, los perfiles, las carteras y la atribución son evidencia informativa posterior: se calculan con el ganador ya congelado y no modifican la configuración predictiva. La política de efectivo y el tamaño de cartera son decisiones de cartera, no de modelo, y se comparan en `portfolio_comparison.parquet`.
