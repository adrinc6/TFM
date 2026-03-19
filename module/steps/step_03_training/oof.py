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
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    oof: Dict[str, pd.Series] = {}

    for ag_name, cfg in agents_config.items():
        score_col = f"{ag_name}_score"
        oof_vals = pd.Series(np.nan, index=X.index, name=score_col)

        for fold_tr, fold_val in kf.split(X):
            X_tr = X.iloc[fold_tr]
            X_val = X.iloc[fold_val]
            y_tr = y.iloc[fold_tr]

            agent = cfg["cls"](**cfg["kwargs"])
            y_fit = (1 - y_tr) if cfg.get("invert_y") else y_tr

            if cfg.get("sector_col"):
                agent.fit(X_tr, y_fit, fold=0, sector_col=cfg["sector_col"])
                preds = agent.predict_score(X_val, cfg["sector_col"])
            else:
                agent.fit(X_tr, y_fit, fold=0)
                preds = agent.predict_score(X_val)

            oof_vals.iloc[fold_val] = preds.values

        oof[score_col] = oof_vals.fillna(0.5)

    return oof
