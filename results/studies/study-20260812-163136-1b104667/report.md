# Informe del Model Study

- Study: `study-20260812-163136-1b104667`
- Ganador: `run-6eaa47a0597b`
- Hash de dataset: `54727df0fe999d9e1fb0fe83aef154a050956bee4308caba7472a61f6aa5508e`
- Selección: exclusivamente Rank-IC pareado hasta 2024.
- 2025–2026: confirmación fuera de muestra, no utilizada en ninguna decisión.

## 1. Aprendizaje (ventana de selección)

| Métrica | Valor | Artefacto |
|---|---|---|
| Rank-IC medio | 0.1000 | `evidence/summary.json` |
| IC-IR | 0.7348 | `evidence/summary.json` |
| Cohortes positivas | 70.94 % | `evidence/summary.json` |
| Cohortes | 117 | `evidence/summary.json` |
| p permutación | 0.00010 | `robustness.json` |

## 2. Confirmación fuera de muestra (no participó en ninguna decisión)

| Métrica | Valor | Artefacto |
|---|---|---|
| Rank-IC medio | -0.0139 | `evidence/summary.json` |
| Cohortes cerradas | 6 | `evidence/summary.json` |
| Observaciones independientes | 1 | `attribution.json` |

Con horizonte de 12 meses y cadencia mensual, cohortes contiguas comparten casi toda la etiqueta. El número de cohortes **no** es el número de pruebas independientes: esta confirmación es evidencia direccional del signo, no un contraste con potencia.

## 3. Traducción a alfa (ventana de selección)

| Métrica | Valor | Artefacto |
|---|---|---|
| CAGR cartera | 14.51 % | `evidence/summary.json` |
| CAGR benchmark | 13.30 % | `evidence/summary.json` |
| Alfa geométrico | 1.07 % | `evidence/summary.json` |
| Information Ratio anualizado | 0.1886 | `evidence/summary.json` |
| Turnover anualizado | 398.93 % | `evidence/summary.json` |
| Efectivo medio | 8.63 % | `evidence/summary.json` |
| Coeficiente de transferencia | 0.1777 | `evidence/summary.json` |

## 4. ¿Aprende algo propio?

| Métrica | Valor | Artefacto |
|---|---|---|
| Alfa de la regresión por periodo | 0.11 % | `attribution.json` |
| t de Newey-West del alfa | 0.72 | `attribution.json` |
| Rank-IC bruto | 0.1115 | `attribution.json` |
| Rank-IC neutralizado por estilo | 0.0927 | `attribution.json` |
| Probabilidad Deflated Sharpe | 0.844 | `attribution.json` |
| Configuraciones probadas | 74 | `attribution.json` |

## 5. Estabilidad entre semillas

- Alfa geométrico: mínimo 1.07 %, mediana 1.71 %, máximo 1.93 %.
- Conclusión económica estable entre semillas: **sí**.

## 6. Configuración ganadora

- `commission_bps`: `5.0`
- `coverage_percentile_floor`: `60.0`
- `execution_lag_days`: `30`
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
