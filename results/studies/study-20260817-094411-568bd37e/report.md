# Informe del Model Study

- Study: `study-20260817-094411-568bd37e`
- Ganador: `run-1bb331e23a18`
- Hash de dataset: `b5db6211d03fcd56815afb316514215234e7300adf88f5a07f182071e5887cd9`
- Selección: exclusivamente Rank-IC pareado hasta 2024.
- 2025–2026: confirmación fuera de muestra, no utilizada en ninguna decisión.

## 1. Aprendizaje (ventana de selección)

| Métrica | Valor | Artefacto |
|---|---|---|
| Rank-IC medio | 0.1067 | `evidence/summary.json` |
| IC-IR | 0.8325 | `evidence/summary.json` |
| Cohortes positivas | 76.07 % | `evidence/summary.json` |
| Cohortes | 117 | `evidence/summary.json` |
| p permutación | 0.00010 | `robustness.json` |

## 2. Confirmación fuera de muestra (no participó en ninguna decisión)

| Métrica | Valor | Artefacto |
|---|---|---|
| Rank-IC medio | 0.0364 | `evidence/summary.json` |
| Cohortes cerradas | 6 | `evidence/summary.json` |
| Observaciones independientes | 1 | `attribution.json` |

Con horizonte de 12 meses y cadencia mensual, cohortes contiguas comparten casi toda la etiqueta. El número de cohortes **no** es el número de pruebas independientes: esta confirmación es evidencia direccional del signo, no un contraste con potencia.

## 3. Traducción a alfa (ventana de selección)

| Métrica | Valor | Artefacto |
|---|---|---|
| CAGR cartera | 15.57 % | `evidence/summary.json` |
| CAGR benchmark | 13.17 % | `evidence/summary.json` |
| Alfa geométrico | 2.12 % | `evidence/summary.json` |
| Information Ratio anualizado | 0.3125 | `evidence/summary.json` |
| Turnover anualizado | 241.96 % | `evidence/summary.json` |
| Efectivo medio | 8.57 % | `evidence/summary.json` |
| Coeficiente de transferencia | 0.2640 | `evidence/summary.json` |

## 4. ¿Aprende algo propio?

| Métrica | Valor | Artefacto |
|---|---|---|
| Alfa de la regresión por periodo | 0.20 % | `attribution.json` |
| t de Newey-West del alfa | 1.02 | `attribution.json` |
| Rank-IC bruto | 0.1135 | `attribution.json` |
| Rank-IC neutralizado por estilo | 0.0990 | `attribution.json` |
| Probabilidad Deflated Sharpe | 0.573 | `attribution.json` |
| Configuraciones probadas | 46 | `attribution.json` |

## 5. Estabilidad entre semillas

- Alfa geométrico: mínimo 0.63 %, mediana 0.84 %, máximo 2.12 %.
- Conclusión económica estable entre semillas: **sí**.

## 6. Configuración ganadora

- `commission_bps`: `10.0`
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
- `max_features_per_agent`: `12`
- `meta_history_quarters`: `16`
- `meta_method`: `stacked_rolling_free`
- `meta_recency_weighting`: `off`
- `minimum_holding_period`: `half_horizon`
- `model_family`: `lightgbm`
- `neutralize_by_sector`: `False`
- `objective`: `rank_regression`
- `price_only_sell_only`: `False`
- `price_only_strictness_multiplier`: `1.5`
- `rebalance_drift_tolerance`: `0.25`
- `recency_weighting`: `off`
- `rotation_edge_bps`: `50.0`
- `sizing_mode`: `alpha_proportional`
- `slippage_bps`: `20.0`
- `snapshot_step_months`: `1`
- `target_horizon_months`: `12`
- `target_size`: `8`
- `train_lookback_years`: `8`
- `winsorization`: `0.0`

## Interpretación

La robustez, los perfiles, las carteras y la atribución son evidencia informativa posterior: se calculan con el ganador ya congelado y no modifican la configuración predictiva. La política de efectivo y el tamaño de cartera son decisiones de cartera, no de modelo, y se comparan en `portfolio_comparison.parquet`.
