"""Diagnósticos posteriores que nunca modifican el ganador predictivo."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from environment import MAX_MONTHLY_POSITION_RETURN
from module.evaluation.stats import DEFAULT_BLOCK_SIZE, block_bootstrap_ci
from module.studies.catalog import SELECTION_ERAS, SELECTION_UNTIL_YEAR


def bootstrap_and_eras(diagnostics: pd.DataFrame, *, iterations: int = 2_000) -> dict[str, Any]:
    frame = diagnostics.loc[diagnostics["agent"].eq("meta_final")].copy()
    frame["year"] = pd.to_datetime(frame["prediction_date"]).dt.year
    frame = frame.loc[frame["year"].le(SELECTION_UNTIL_YEAR)]
    values = frame.set_index("prediction_date")["rank_ic"].dropna()
    results = {
        "interval_90": block_bootstrap_ci(
            values, block_size=DEFAULT_BLOCK_SIZE, n_boot=iterations, confidence=0.90,
        ),
        "interval_95": block_bootstrap_ci(
            values, block_size=DEFAULT_BLOCK_SIZE, n_boot=iterations, confidence=0.95,
        ),
        "era_exclusions": [],
    }
    for start, end in SELECTION_ERAS:
        years = pd.to_datetime(values.index).year
        without = values.loc[~pd.Series(years, index=values.index).between(start, end)]
        results["era_exclusions"].append({
            "excluded_era": f"{start}-{end}",
            "mean_rank_ic": float(without.mean()) if len(without) else None,
            "n_cohorts": int(len(without)),
        })
    return results


def score_permutation(
    scores: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    iterations: int = 9_999,
    minimum_cross_section: int = 8,
) -> dict[str, Any]:
    merged = scores[["ticker", "snapshot_date", "meta_rank"]].merge(
        targets[["ticker", "snapshot_date", "forward_excess_return", "target_available"]],
        on=["ticker", "snapshot_date"],
        how="inner",
    )
    merged = merged.loc[merged["target_available"].fillna(False)].copy()
    merged["year"] = pd.to_datetime(merged["snapshot_date"]).dt.year
    merged = merged.loc[merged["year"].le(SELECTION_UNTIL_YEAR)]
    groups = [
        group[["meta_rank", "forward_excess_return"]].dropna().to_numpy()
        for _, group in merged.groupby("snapshot_date")
        if len(group) >= minimum_cross_section
    ]
    if not groups:
        return {
            "observed_mean_rank_ic": None,
            "p_value": 1.0,
            "n_permutations": iterations,
            "add_one_correction": True,
            "applicable": False,
        }
    observed = float(np.mean([_spearman(group[:, 0], group[:, 1]) for group in groups]))
    rng = np.random.default_rng(42)
    exceedances = 0
    for _ in range(iterations):
        statistic = float(np.mean([
            _spearman(group[:, 0], rng.permutation(group[:, 1]))
            for group in groups
        ]))
        exceedances += statistic >= observed
    return {
        "observed_mean_rank_ic": observed,
        "p_value": float((exceedances + 1) / (iterations + 1)),
        "n_permutations": iterations,
        "add_one_correction": True,
        "applicable": True,
    }


def random_portfolios(
    prepared: Any,
    evidence: Any,
    values: Mapping[str, Any],
    *,
    simulations: int = 1_000,
) -> dict[str, Any]:
    """Nulo de carteras aleatorias del mismo tamaño, jugando con las mismas reglas que el modelo.

    La versión anterior producía un percentil 95 de CAGR del 107 % anual, imposible para una cartera
    de acciones del S&P 500, y por eso el único test que el modelo «suspendía» era en realidad un
    fallo del test. Tres defectos lo causaban y aquí se corrigen los tres:

    1. El retorno anual se calculaba como `último/primero` de los precios observados en el año, sin
       exigir cobertura completa: una acción que solo aparecía en diciembre aportaba el retorno de
       unos días anualizado como si fuera el del año entero.
    2. No se aplicaba la guarda contra artefactos de datos (splits mal ajustados, tickers
       reciclados) que el backtest sí aplica, así que el nulo heredaba precisamente los outliers que
       el modelo tiene prohibido cobrar.
    3. El nulo no pagaba comisiones ni slippage mientras el modelo sí, comparando una cartera bruta
       contra una neta.
    """
    prices = pd.read_parquet(prepared / "asset_price_point_in_time.parquet")
    scores = pd.read_parquet(evidence / "agent_scores.parquet")
    annual = pd.read_parquet(evidence / "annual_metrics.parquet")
    guard = float(values.get("max_monthly_position_return", MAX_MONTHLY_POSITION_RETURN))
    cost = (float(values["commission_bps"]) + float(values["slippage_bps"])) / 10_000
    returns = _annual_returns(prices, guard)
    pools = {int(year): group["return"].to_numpy() for year, group in returns.groupby("year")}
    model = annual.loc[annual["year"].le(SELECTION_UNTIL_YEAR)].set_index("year")["portfolio_return"]
    size = int(values["target_size"])
    general = _simulate(model, pools, size, 42, simulations, cost)
    scores["year"] = pd.to_datetime(scores["snapshot_date"]).dt.year
    risk = scores.groupby(["year", "ticker"])["risk_rank"].mean().reset_index()
    risk["quintile"] = risk.groupby("year")["risk_rank"].transform(
        lambda series: pd.qcut(series.rank(method="first"), 5, labels=False, duplicates="drop")
    )
    matched = returns.merge(risk, on=["year", "ticker"], how="inner")
    central = matched.loc[matched["quintile"].isin([1, 2, 3])]
    matched_pools = {
        int(year): group["return"].dropna().to_numpy()
        for year, group in central.groupby("year")
    }
    risk_matched = _simulate(model, matched_pools, size, 43, simulations, cost)
    return {
        "general": general,
        "risk_matched": risk_matched,
        "costs_applied_bps": cost * 10_000,
        "model_above_p95_both": bool(
            general["model_percentile"] >= 0.95
            and risk_matched["model_percentile"] >= 0.95
        ),
    }


def _annual_returns(prices: pd.DataFrame, guard: float) -> pd.DataFrame:
    """Retorno anual por acción exigiendo cobertura del año completo y filtrando artefactos.

    Se exige presencia en el primer y el último tercio del año para no confundir «la acción subió»
    con «la acción solo cotizó durante el tramo que subió», que es como el nulo anterior llegaba a
    rentabilidades imposibles.
    """
    frame = prices.copy()
    frame["snapshot_ts"] = pd.to_datetime(frame["snapshot_date"])
    frame["year"] = frame["snapshot_ts"].dt.year
    frame = frame.loc[frame["year"].le(SELECTION_UNTIL_YEAR)].sort_values(["ticker", "snapshot_ts"])
    frame["month"] = frame["snapshot_ts"].dt.month
    grouped = frame.groupby(["year", "ticker"])
    summary = grouped.agg(
        first=("price", "first"), last=("price", "last"),
        first_month=("month", "min"), last_month=("month", "max"),
        observations=("price", "size"),
    ).reset_index()
    covered = summary.loc[
        summary["first_month"].le(4) & summary["last_month"].ge(9)
        & summary["first"].gt(0) & summary["observations"].ge(2)
    ].copy()
    covered["return"] = covered["last"] / covered["first"] - 1
    covered = covered.loc[covered["return"].abs().le(guard) & covered["return"].notna()]
    return covered[["year", "ticker", "return"]]


def _simulate(
    model: pd.Series,
    pools: Mapping[int, np.ndarray],
    size: int,
    seed: int,
    simulations: int,
    cost: float,
) -> dict[str, Any]:
    """Simula carteras equiponderadas de `size` nombres, renovadas cada año y con costes.

    Renovar la cartera completa cada año implica una rotación del 100 %, que a coste `cost` por
    operación se paga dos veces (venta y compra). Es el mismo coste que soporta el modelo.
    """
    years = sorted(set(model.index) & set(pools))
    rng = np.random.default_rng(seed)
    samples = np.zeros(simulations)
    annual_cost = 2 * cost
    for index in range(simulations):
        yearly = []
        for year in years:
            pool = np.asarray(pools[year])
            pool = pool[np.isfinite(pool)]
            count = min(size, len(pool))
            gross = float(rng.choice(pool, count, replace=False).mean()) if count else 0.0
            yearly.append(gross - annual_cost)
        samples[index] = np.prod(1 + np.asarray(yearly)) ** (1 / max(len(yearly), 1)) - 1
    model_values = model.reindex(years).dropna().to_numpy()
    model_cagr = (
        float(np.prod(1 + model_values) ** (1 / len(model_values)) - 1)
        if len(model_values) else 0.0
    )
    return {
        "model_cagr": model_cagr,
        "random_mean": float(samples.mean()),
        "random_median": float(np.median(samples)),
        "random_p95": float(np.quantile(samples, 0.95)),
        "model_percentile": float((samples < model_cagr).mean()),
        "n_simulations": simulations,
        "portfolio_size": size,
    }


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(pd.Series(left).rank().corr(pd.Series(right).rank()))
