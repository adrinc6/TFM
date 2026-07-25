"""Diagnósticos posteriores que nunca modifican el ganador predictivo."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from module.evaluation.stats import block_bootstrap_ci
from module.studies.catalog import SELECTION_ERAS


def bootstrap_and_eras(diagnostics: pd.DataFrame, *, iterations: int = 2_000) -> dict[str, Any]:
    frame = diagnostics.loc[diagnostics["agent"].eq("meta_final")].copy()
    frame["year"] = pd.to_datetime(frame["prediction_date"]).dt.year
    frame = frame.loc[frame["year"].le(2024)]
    values = frame.set_index("prediction_date")["rank_ic"].dropna()
    results = {
        "interval_90": block_bootstrap_ci(
            values, block_size=12, n_boot=iterations, confidence=0.90,
        ),
        "interval_95": block_bootstrap_ci(
            values, block_size=12, n_boot=iterations, confidence=0.95,
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
    merged = merged.loc[merged["year"].le(2024)]
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
    prices = pd.read_parquet(prepared / "asset_price_point_in_time.parquet")
    scores = pd.read_parquet(evidence / "agent_scores.parquet")
    annual = pd.read_parquet(evidence / "annual_metrics.parquet")
    prices["year"] = pd.to_datetime(prices["snapshot_date"]).dt.year
    prices = prices.loc[prices["year"].le(2024)].sort_values(["ticker", "snapshot_date"])
    returns = prices.groupby(["year", "ticker"])["price"].agg(["first", "last"]).reset_index()
    returns["return"] = returns["last"] / returns["first"] - 1
    pools = {int(year): group["return"].dropna().to_numpy() for year, group in returns.groupby("year")}
    model = annual.loc[annual["year"].le(2024)].set_index("year")["portfolio_return"]
    general = _simulate(model, pools, int(values["target_size"]), 42, simulations)
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
    risk_matched = _simulate(model, matched_pools, int(values["target_size"]), 43, simulations)
    return {
        "general": general,
        "risk_matched": risk_matched,
        "model_above_p95_both": (
            general["model_percentile"] >= 0.95
            and risk_matched["model_percentile"] >= 0.95
        ),
    }


def _simulate(
    model: pd.Series,
    pools: Mapping[int, np.ndarray],
    size: int,
    seed: int,
    simulations: int,
) -> dict[str, Any]:
    years = sorted(set(model.index) & set(pools))
    rng = np.random.default_rng(seed)
    samples = np.zeros(simulations)
    for index in range(simulations):
        yearly = []
        for year in years:
            pool = np.asarray(pools[year])
            pool = pool[np.isfinite(pool)]
            count = min(size, len(pool))
            yearly.append(float(rng.choice(pool, count, replace=False).mean()) if count else 0.0)
        samples[index] = np.prod(1 + np.asarray(yearly)) ** (1 / max(len(yearly), 1)) - 1
    model_values = model.reindex(years).dropna().to_numpy()
    model_cagr = (
        float(np.prod(1 + model_values) ** (1 / len(model_values)) - 1)
        if len(model_values) else 0.0
    )
    return {
        "model_cagr": model_cagr,
        "random_mean": float(samples.mean()),
        "random_p95": float(np.quantile(samples, 0.95)),
        "model_percentile": float((samples < model_cagr).mean()),
        "n_simulations": simulations,
    }


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(pd.Series(left).rank().corr(pd.Series(right).rank()))
