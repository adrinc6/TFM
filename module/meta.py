"""Combinación temporal de agentes mediante rank-IC fuera de muestra."""

from __future__ import annotations

import numpy as np
import pandas as pd

from environment import Settings

AGENT_NAMES = ("quality", "momentum", "value")


def combine_agent_scores(
    scores: pd.DataFrame, targets: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devuelve scores meta, pesos secuenciales y diagnósticos OOS finales."""
    labelled = scores.merge(
        targets[["ticker", "snapshot_date", "label_end_date", "target_available", "forward_excess_return_3m"]],
        on=["ticker", "snapshot_date"],
        how="left",
        validate="many_to_one",
    )
    labelled["snapshot_ts"] = pd.to_datetime(labelled["snapshot_date"])
    labelled["label_end_ts"] = pd.to_datetime(labelled["label_end_date"])
    diagnostics = _diagnostics(labelled, settings)
    meta_scores: list[dict] = []
    weights: list[dict] = []

    for date in sorted(labelled["snapshot_ts"].dropna().unique()):
        date_frame = labelled.loc[labelled["snapshot_ts"] == date].copy()
        available_agents = set(date_frame["agent"].unique())
        agent_weights, evidence = _weights_as_of(labelled, date, available_agents, settings)
        learned = any(item.get("mean_rank_ic", 0) > 0 for item in evidence.values())
        for agent in AGENT_NAMES:
            weights.append(
                {
                    "snapshot_date": pd.Timestamp(date).date().isoformat(),
                    "agent": agent,
                    "weight": agent_weights.get(agent, 0.0),
                    "mean_rank_ic": evidence.get(agent, {}).get("mean_rank_ic"),
                    "realized_cohorts": evidence.get(agent, {}).get("realized_cohorts", 0),
                    "weight_status": "learned" if learned else "fallback_equal",
                }
            )

        pivot = date_frame.pivot(index="ticker", columns="agent", values="score")
        ranks = pivot.rank(method="average", pct=True)
        usable = [agent for agent in AGENT_NAMES if agent in ranks.columns and agent_weights.get(agent, 0) > 0]
        if not usable:
            usable = [agent for agent in AGENT_NAMES if agent in ranks.columns]
            equal = 1 / len(usable) if usable else 0
            agent_weights = {agent: equal for agent in usable}
        for ticker, row in ranks.iterrows():
            present = [agent for agent in usable if pd.notna(row.get(agent))]
            if not present:
                meta_score = np.nan
            else:
                denominator = sum(agent_weights[agent] for agent in present)
                meta_score = sum(row[agent] * agent_weights[agent] for agent in present) / denominator
            meta_scores.append(
                {
                    "ticker": ticker,
                    "snapshot_date": pd.Timestamp(date).date().isoformat(),
                    "meta_score": meta_score,
                }
            )

    meta = pd.DataFrame(meta_scores)
    if not meta.empty:
        meta["meta_rank"] = meta.groupby("snapshot_date")["meta_score"].rank(method="average", pct=True)
    return meta, pd.DataFrame(weights), diagnostics


def _weights_as_of(
    labelled: pd.DataFrame,
    date: pd.Timestamp,
    available_agents: set[str],
    settings: Settings,
) -> tuple[dict[str, float], dict[str, dict[str, float | int]]]:
    history = labelled.loc[
        (labelled["snapshot_ts"] < date)
        & labelled["is_quarterly"].fillna(False)
        & labelled["target_available"].fillna(False)
        & labelled["label_end_ts"].le(date)
    ]
    evidence: dict[str, dict[str, float | int]] = {}
    positive: dict[str, float] = {}
    for agent in AGENT_NAMES:
        if agent not in available_agents:
            continue
        agent_history = history.loc[history["agent"] == agent]
        cohort_ics: list[float] = []
        for _, cohort in agent_history.groupby("snapshot_date", sort=True):
            value = _rank_ic(cohort, settings.min_rank_ic_cross_section)
            if value is not None:
                cohort_ics.append(value)
        recent = cohort_ics[-settings.meta_ic_lookback_quarters :]
        mean_ic = float(np.mean(recent)) if recent else 0.0
        evidence[agent] = {"mean_rank_ic": mean_ic, "realized_cohorts": len(recent)}
        positive[agent] = max(mean_ic, 0.0)
    total = sum(positive.values())
    if total > 0:
        return {agent: value / total for agent, value in positive.items()}, evidence
    equal = 1 / len(available_agents) if available_agents else 0.0
    return ({agent: equal for agent in available_agents}, evidence)


def _diagnostics(labelled: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    rows: list[dict] = []
    usable = labelled.loc[labelled["target_available"].fillna(False)]
    for (agent, date), cohort in usable.groupby(["agent", "snapshot_date"], sort=True):
        value = _rank_ic(cohort, settings.min_rank_ic_cross_section)
        if value is None:
            continue
        rows.append(
            {
                "agent": agent,
                "prediction_date": date,
                "label_end_date": pd.to_datetime(cohort["label_end_date"]).max().date().isoformat(),
                "observations": int(cohort[["score", "forward_excess_return_3m"]].dropna().shape[0]),
                "rank_ic": value,
                "is_quarterly": bool(cohort["is_quarterly"].iloc[0]),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["agent", "prediction_date", "label_end_date", "observations", "rank_ic", "is_quarterly"],
    )


def _rank_ic(cohort: pd.DataFrame, minimum: int) -> float | None:
    usable = cohort[["score", "forward_excess_return_3m"]].dropna()
    if len(usable) < minimum:
        return None
    value = usable["score"].corr(usable["forward_excess_return_3m"], method="spearman")
    return float(value) if pd.notna(value) else None
