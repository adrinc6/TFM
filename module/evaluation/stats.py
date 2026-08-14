"""Significancia estadística del rank-IC OOS.

Las cohortes NO son independientes a nivel de fila: el horizonte de la etiqueta (hasta 12 meses) y
la ventana de entrenamiento se solapan entre fechas contiguas. Por eso el remuestreo se hace por
**bloques temporales** suficientemente largos, no fila a fila — un bootstrap ingenuo daría
intervalos de confianza artificialmente estrechos.

Dos herramientas:
- `block_bootstrap_ci`: intervalo de confianza del rank-IC medio de una serie de cohortes.
- `paired_difference_ci`: intervalo de confianza de la diferencia pareada de rank-IC por fecha
  entre dos modelos o configuraciones, para decidir si la mejora es real o ruido.

`DEFAULT_BLOCK_SIZE` es el bloque por defecto en ambas: 12 cohortes cubren un año completo con la
cadencia mensual del ganador y absorben el solapamiento del horizonte de 12 meses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_BLOCK_SIZE = 12


def block_bootstrap_ci(
    values_by_date: pd.Series,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """IC del rank-IC medio remuestreando bloques contiguos de cohortes.

    `values_by_date`: serie de rank-IC por fecha, ordenada temporalmente. `block_size` en número de
    cohortes: debe cubrir al menos el solapamiento inducido por el horizonte de la etiqueta.

    Devuelve además `replicates`: las medias remuestreadas que generaron el intervalo. Guardarlas
    permite representar la distribución completa en vez de solo sus extremos; sin ellas, una figura
    del intervalo tendría que simular la forma de la distribución, que sería inventar datos.
    """
    values = values_by_date.dropna().to_numpy()
    n = len(values)
    if n < block_size or n == 0:
        mean = float(values.mean()) if n else 0.0
        return {
            "mean": mean, "ci_low": mean, "ci_high": mean, "n_cohorts": n,
            "block_size": block_size, "replicates": [],
        }

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
        "replicates": [float(value) for value in boot_means],
    }


def paired_difference_ci(
    ic_a: pd.Series,
    ic_b: pd.Series,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """IC de la diferencia pareada (modelo A − modelo B) del rank-IC por fecha.

    `ic_a` e `ic_b` son series indexadas por fecha. Se emparejan por índice (solo fechas comunes) y
    se remuestrea por bloques la serie de diferencias. Si el IC no cruza cero, la diferencia es
    estadísticamente distinguible del ruido al nivel de confianza dado.

    **`applicable` es la clave que hay que mirar antes que ninguna otra.** Cuando las dos series no
    comparten suficientes fechas —caso real al barrer `execution_lag_days` o `snapshot_step_months`,
    que desplazan la rejilla de snapshots y producen índices disjuntos— el emparejamiento no existe
    y devolver `ci_low = 0.0` haría pasar automáticamente cualquier prueba de no inferioridad
    formulada como `ci_low > margen_negativo`. En ese caso `applicable` vale `False` y los límites
    son `None`: quien decide debe tratarlo como "sin evidencia", nunca como "empate".
    """
    paired = pd.concat([ic_a.rename("a"), ic_b.rename("b")], axis=1).dropna()
    if len(paired) < block_size:
        return {
            "mean_diff": None, "ci_low": None, "ci_high": None, "n_dates": len(paired),
            "fraction_a_better": None, "distinguishable_from_zero": False,
            "applicable": False, "block_size": block_size,
        }
    diff = (paired["a"] - paired["b"]).to_numpy()
    boot = block_bootstrap_ci(pd.Series(diff), block_size=block_size, n_boot=n_boot,
                              confidence=confidence, seed=seed)
    # `boot["replicates"]` se descarta a propósito: esta función alimenta la selección del Study y
    # se persiste una vez por candidato de cada variable, así que arrastrar 2.000 flotantes por
    # candidato multiplicaría el tamaño de `decisions.json` sin que nada lo lea. Las réplicas solo
    # se guardan donde se van a representar: los tres contrastes de robustez del informe.
    return {
        "mean_diff": boot["mean"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "n_dates": len(diff),
        "fraction_a_better": float((diff > 0).mean()),
        "distinguishable_from_zero": bool(boot["ci_low"] > 0 or boot["ci_high"] < 0),
        "applicable": True,
        "block_size": block_size,
    }
