"""Tests de robustez / placebo: demostrar que un resultado no es suerte.

El argumento de credibilidad del TFM. Sobre la configuracion final, comprobar que la señal (si
la hay) es real y no un artefacto del azar o del proceso de busqueda:

- `label_permutation_test`: reentrenar/evaluar con los retornos futuros BARAJADOS. Si el sistema
  aprende de verdad, el rank-IC debe COLAPSAR a ~0 con etiquetas aleatorias. Si no colapsa, hay
  fuga de informacion (el placebo tambien detecta leakage).
- `random_portfolio_test`: comparar la cartera del modelo contra muchas carteras aleatorias del
  mismo tamano. El percentil del modelo dice si su rendimiento es distinguible del azar.
- `leave_one_year_out`: quitar cada año y ver si el rank-IC aguanta. Formaliza la estabilidad
  temporal: un resultado que depende de 1-2 años no es fiable.
- El bootstrap por bloques del rank-IC (en `module.stats`) complementa estos con el IC de confianza.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def label_permutation_test(
    diagnostics_real: pd.DataFrame,
    ic_permuted: list[float],
    agent: str = "meta_final",
) -> dict:
    """Compara el rank-IC real con el de N reentrenamientos sobre etiquetas barajadas.

    `diagnostics_real`: diagnosticos del sistema real (para extraer el rank-IC del meta_final).
    `ic_permuted`: lista de rank-IC medios obtenidos al reentrenar con retornos futuros barajados.

    Devuelve el rank-IC real, la media del placebo (deberia ser ~0) y el p-valor empirico
    (fraccion de permutaciones que igualan o superan el real). Un p-valor bajo = la señal real
    esta por encima de lo que produce el azar.
    """
    real = diagnostics_real.loc[diagnostics_real["agent"] == agent, "rank_ic"].mean()
    permuted = np.asarray(ic_permuted, dtype=float)
    permuted = permuted[np.isfinite(permuted)]
    if len(permuted) == 0:
        return {"rank_ic_real": float(real), "placebo_mean": 0.0, "p_value": 1.0, "n_permutations": 0}
    # Corrección add-one: con un número finito de permutaciones el p-valor nunca puede ser cero.
    # Con tres placebos el mínimo inferencial es 1/4, no 0.
    exceedances = int((permuted >= real).sum())
    p_value = float((exceedances + 1) / (len(permuted) + 1))
    return {
        "rank_ic_real": float(real),
        "placebo_mean": float(permuted.mean()),
        "placebo_std": float(permuted.std()),
        "p_value": p_value,
        "n_permutations": int(len(permuted)),
        "add_one_correction": True,
        "signal_above_chance": bool(p_value < 0.05),
    }


def random_portfolio_test(
    model_annual_returns: pd.Series,
    asset_returns_by_year: dict[int, np.ndarray],
    portfolio_size: int,
    n_simulations: int = 1000,
    seed: int = 42,
) -> dict:
    """Compara el CAGR de la cartera del modelo contra `n_simulations` carteras aleatorias.

    `model_annual_returns`: retorno anual de la cartera del modelo, indexado por año.
    `asset_returns_by_year`: {año: array de retornos anuales de los activos disponibles ese año}.
    Cada cartera aleatoria elige `portfolio_size` activos al azar cada año. Se compara el CAGR.

    Devuelve el CAGR del modelo, la distribucion aleatoria y el percentil del modelo (si esta en
    la cola alta, su rendimiento no es atribuible al azar de escoger acciones).
    """
    rng = np.random.default_rng(seed)
    years = sorted(asset_returns_by_year)

    def cagr(annual: np.ndarray) -> float:
        annual = annual[np.isfinite(annual)]
        if len(annual) == 0:
            return 0.0
        return float(np.prod(1 + annual) ** (1 / len(annual)) - 1)

    model_cagr = cagr(model_annual_returns.reindex(years).to_numpy())
    random_cagrs = np.empty(n_simulations)
    for s in range(n_simulations):
        yearly = []
        for year in years:
            pool = asset_returns_by_year[year]
            pool = pool[np.isfinite(pool)]
            if len(pool) == 0:
                yearly.append(0.0)
                continue
            k = min(portfolio_size, len(pool))
            picks = rng.choice(pool, size=k, replace=False)
            yearly.append(picks.mean())
        random_cagrs[s] = cagr(np.asarray(yearly))

    percentile = float((random_cagrs < model_cagr).mean())
    return {
        "model_cagr": model_cagr,
        "random_cagr_mean": float(random_cagrs.mean()),
        "random_cagr_p95": float(np.quantile(random_cagrs, 0.95)),
        "model_percentile": percentile,
        "beats_random_convincingly": bool(percentile > 0.95),
        "n_simulations": n_simulations,
    }


def leave_one_year_out(diagnostics: pd.DataFrame, agent: str = "meta_final") -> pd.DataFrame:
    """rank-IC medio quitando cada año, para ver si el resultado depende de un año concreto.

    Si al quitar un año el rank-IC se desploma, ese año lo sostenia (resultado fragil). Si
    apenas cambia al quitar cualquiera, el aprendizaje es estable.
    """
    meta = diagnostics.loc[diagnostics["agent"] == agent].copy()
    if meta.empty:
        return pd.DataFrame(columns=["excluded_year", "rank_ic_without_it", "n_cohorts"])
    meta["year"] = pd.to_datetime(meta["prediction_date"]).dt.year
    full_mean = meta["rank_ic"].mean()
    rows = []
    for year in sorted(meta["year"].unique()):
        without = meta.loc[meta["year"] != year, "rank_ic"]
        rows.append({
            "excluded_year": int(year),
            "rank_ic_without_it": float(without.mean()),
            "delta_vs_full": float(without.mean() - full_mean),
            "n_cohorts": int(len(without)),
        })
    return pd.DataFrame(rows)
