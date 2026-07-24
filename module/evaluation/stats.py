"""Significancia estadística del rank-IC OOS.

Las cohortes NO son independientes a nivel de fila: el horizonte de la etiqueta (3 meses) y la
ventana de entrenamiento se solapan entre fechas contiguas. Por eso el remuestreo se hace por
**bloques temporales** suficientemente largos, no fila a fila — un bootstrap ingenuo daría
intervalos de confianza artificialmente estrechos.

Dos herramientas:
- `block_bootstrap_ci`: intervalo de confianza del rank-IC medio de una serie de cohortes.
- `paired_difference_ci`: intervalo de confianza de la diferencia pareada de rank-IC por fecha
  entre dos modelos o configuraciones, para decidir si la mejora es real o ruido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def block_bootstrap_ci(
    values_by_date: pd.Series,
    block_size: int = 4,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """IC del rank-IC medio remuestreando bloques contiguos de cohortes.

    `values_by_date`: serie de rank-IC por fecha, ordenada temporalmente. `block_size` en número
    de cohortes (4 ≈ un año si son trimestrales; cubre el solapamiento del horizonte de 3 meses).
    """
    values = values_by_date.dropna().to_numpy()
    n = len(values)
    if n < block_size or n == 0:
        mean = float(values.mean()) if n else 0.0
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n_cohorts": n, "block_size": block_size}

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    starts = np.arange(0, n - block_size + 1)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        sample = np.concatenate([values[s : s + block_size] for s in chosen])[:n]
        boot_means[b] = sample.mean()

    alpha = (1 - confidence) / 2
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(boot_means, alpha)),
        "ci_high": float(np.quantile(boot_means, 1 - alpha)),
        "n_cohorts": n,
        "block_size": block_size,
    }


def paired_difference_ci(
    ic_a: pd.Series,
    ic_b: pd.Series,
    block_size: int = 4,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """IC de la diferencia pareada (modelo A − modelo B) del rank-IC por fecha.

    `ic_a` e `ic_b` son series indexadas por fecha. Se emparejan por fecha (solo fechas comunes)
    y se remuestrea por bloques la serie de diferencias. Si el IC no cruza cero, la diferencia es
    estadísticamente distinguible del ruido al nivel de confianza dado.
    """
    paired = pd.concat([ic_a.rename("a"), ic_b.rename("b")], axis=1).dropna()
    if paired.empty:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_dates": 0,
                "fraction_a_better": 0.0, "distinguishable_from_zero": False}
    diff = (paired["a"] - paired["b"]).to_numpy()
    boot = block_bootstrap_ci(pd.Series(diff), block_size=block_size, n_boot=n_boot,
                              confidence=confidence, seed=seed)
    return {
        "mean_diff": boot["mean"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "n_dates": len(diff),
        "fraction_a_better": float((diff > 0).mean()),
        "distinguishable_from_zero": bool(boot["ci_low"] > 0 or boot["ci_high"] < 0),
    }


def icir(values_by_date: pd.Series) -> float:
    """Information Coefficient IR: media / desviación del rank-IC. Estabilidad del aprendizaje."""
    values = values_by_date.dropna()
    if len(values) < 2:
        return 0.0
    std = float(values.std(ddof=1))
    return float(values.mean() / std) if std > 0 else 0.0

