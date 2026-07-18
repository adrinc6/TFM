"""Catálogo de valores admitidos por variable, fuente única para la consola y el orquestador.

`STUDY_OPTIONS` define, para cada campo de `Settings` barrible, la lista de valores permitidos.
Lo usan tanto la UI (consola: selects guiados de Experimental y Study) como el orquestador de
estudios (`module/runs/execution.py`), que deriva de aquí el barrido completo del `full_study`.
Vive fuera de `module/ui/` para que el orquestador no dependa de la capa de presentación.

La separación entre variables de modelo y de cartera NO se decide aquí: es autoritativa en
`module/runs/experiments.py` (`MODEL_FIELDS` / `PORTFOLIO_FIELDS`, derivadas de `FINGERPRINT_FIELDS`).
"""

from __future__ import annotations


STUDY_OPTIONS: dict[str, list] = {
    "execution_year": [2012, 2014, 2016, 2018, 2020], "execution_quarter": [1, 2, 3, 4],
    "execution_lag_days": [15, 30, 45, 60], "train_lookback_years": [5, 6, 7, 8, 10, 12],
    "snapshot_step_months": [1, 3], "fundamental_step_months": [3, 6, 12],
    "target_horizon_months": [1, 3, 6, 12], "objective": ["rank_regression", "ranking", "quartile"],
    "lgbm_n_estimators": [100, 200, 400], "lgbm_max_depth": [3, 4, 5, 6],
    "lgbm_learning_rate": [0.03, 0.05, 0.10], "lgbm_min_child_samples": [20, 50, 100],
    "meta_type": ["equal", "rank_ic", "regime"],
    "neutralize_by_sector": [False, True], "fundamental_momentum": [False, True],
    "market_regime_feature": [False, True], "price_momentum_multi": [False, True],
    "moving_averages": [False, True], "regime_extended": [False, True],
    "quality_growth_derived": [False, True], "target_min": [6, 8, 10, 12],
    "target_max": [8, 10, 12, 15], "entry_min_percentile": [70, 80, 90],
    "min_hold_percentile": [40, 50, 60], "rotation_edge_percentiles": [3, 5, 10],
    "max_weight_per_position": [0.10, 0.15, 0.20], "commission_bps": [0, 5, 10],
    "slippage_bps": [5, 10, 20], "profile": ["balanced", "conservative", "aggressive",
                                             "value", "quality", "momentum", "garp", "contrarian"],
}
