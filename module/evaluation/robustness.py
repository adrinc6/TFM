"""Contrato estadístico de los placebos con reentrenamiento."""

from __future__ import annotations

import numpy as np
import pandas as pd


def label_permutation_test(
    diagnostics_real: pd.DataFrame,
    ic_permuted: list[float],
    agent: str = "meta_final",
) -> dict:
    """Resume el real y los placebos; el p-valor empírico usa corrección add-one."""
    real = diagnostics_real.loc[diagnostics_real["agent"] == agent, "rank_ic"].mean()
    permuted = np.asarray(ic_permuted, dtype=float)
    permuted = permuted[np.isfinite(permuted)]
    if len(permuted) == 0:
        return {
            "rank_ic_real": float(real),
            "placebo_mean": 0.0,
            "p_value": 1.0,
            "n_permutations": 0,
            "add_one_correction": True,
        }
    exceedances = int((permuted >= real).sum())
    return {
        "rank_ic_real": float(real),
        "placebo_mean": float(permuted.mean()),
        "placebo_std": float(permuted.std()),
        "placebo_min": float(permuted.min()),
        "placebo_max": float(permuted.max()),
        "p_value": float((exceedances + 1) / (len(permuted) + 1)),
        "n_permutations": int(len(permuted)),
        "add_one_correction": True,
    }
