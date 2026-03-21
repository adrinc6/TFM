"""Out-of-fold score generation for base agents."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def generate_oof_scores(
    X: pd.DataFrame,
    y: pd.Series,
    agents_config: Dict[str, Dict],
    n_splits: int = 3,
    random_seed: int = 42,
) -> Dict[str, pd.Series]:
    # TimeSeriesSplit respeta el orden temporal: el fold k siempre entrena
    # con quarters anteriores al de validación. Así los scores OOF del meta-learner
    # no sufren look-ahead entre trimestres.
    # IMPORTANTE: el índice es (ticker, date). Sin ordenar por date primero,
    # TimeSeriesSplit partiría AAPL contra AAPL en lugar de quarters contra quarters.
    from sklearn.model_selection import TimeSeriesSplit

    date_order = X.index.get_level_values("date")
    sort_idx = date_order.argsort()
    X = X.iloc[sort_idx]
    y = y.reindex(X.index)

    kf = TimeSeriesSplit(n_splits=n_splits)
    oof: Dict[str, pd.Series] = {}

    import logging as _logging
    _log = _logging.getLogger(__name__)

    for ag_name, cfg in agents_config.items():
        score_col = f"{ag_name}_score"
        oof_vals = pd.Series(np.nan, index=X.index, name=score_col)
        _log.info(f"  [OOF] {ag_name}: generando scores anti-leakage ({n_splits} splits temporales)")

        for fold_tr, fold_val in kf.split(X):
            X_tr = X.iloc[fold_tr]
            X_val = X.iloc[fold_val]
            y_tr = y.iloc[fold_tr]

            agent = cfg["cls"](**cfg["kwargs"], save_artifacts=False)
            y_fit = (1 - y_tr) if cfg.get("invert_y") else y_tr

            try:
                if cfg.get("sector_col"):
                    agent.fit(X_tr, y_fit, fold=0, sector_col=cfg["sector_col"])
                    if getattr(agent, "is_trained", False):
                        preds = agent.predict_score(X_val, cfg["sector_col"])
                    else:
                        preds = pd.Series(0.5, index=X_val.index, name=score_col)
                else:
                    agent.fit(X_tr, y_fit, fold=0)
                    if getattr(agent, "is_trained", False):
                        preds = agent.predict_score(X_val)
                    else:
                        preds = pd.Series(0.5, index=X_val.index, name=score_col)
            except Exception:
                preds = pd.Series(0.5, index=X_val.index, name=score_col)

            oof_vals.iloc[fold_val] = preds.values

        oof[score_col] = oof_vals.fillna(0.5)

    return oof
