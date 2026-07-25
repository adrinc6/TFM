"""Combinación point-in-time de agentes en un meta-score.

Admite equiponderación y stacking OOS (Ridge no negativo, causal, con cap y contracción hacia
equiponderación) sobre cohortes trimestrales ya cerradas. Ningún peso puede usar una etiqueta cuyo
``label_end_date`` sea posterior a la fecha de decisión.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from environment import Settings
from module.modeling.catalog import AGENT_NAMES as CATALOG_AGENT_NAMES
from module.modeling.targets import normalize_target_columns

def combine_agent_scores(
    scores: pd.DataFrame, targets: pd.DataFrame, settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devuelve scores meta, pesos secuenciales y diagnósticos OOS finales."""
    targets = normalize_target_columns(targets)
    labelled = scores.merge(
        targets[["ticker", "snapshot_date", "label_end_date", "target_available", "forward_excess_return"]],
        on=["ticker", "snapshot_date"],
        how="left",
        validate="many_to_one",
    )
    labelled["snapshot_ts"] = pd.to_datetime(labelled["snapshot_date"])
    labelled["label_end_ts"] = pd.to_datetime(labelled["label_end_date"])
    diagnostics = _diagnostics(labelled, settings)
    meta_scores: list[dict] = []
    weights: list[dict] = []

    all_agents = tuple(dict.fromkeys([*CATALOG_AGENT_NAMES, *scores["agent"].dropna().unique().tolist()]))
    for date in sorted(labelled["snapshot_ts"].dropna().unique()):
        date_frame = labelled.loc[labelled["snapshot_ts"] == date].copy()
        available_agents = set(date_frame["agent"].unique())
        agent_weights, evidence = _weights_as_of(labelled, date, available_agents, settings)
        learned = any(item.get("mean_rank_ic", 0) > 0 for item in evidence.values())
        for agent in all_agents:
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
        usable = [agent for agent in all_agents if agent in ranks.columns and agent_weights.get(agent, 0) > 0]
        if not usable:
            usable = [agent for agent in all_agents if agent in ranks.columns]
            equal = 1 / len(usable) if usable else 0
            agent_weights = {agent: equal for agent in usable}
        # Media ponderada de rangos por ticker, vectorizada: para cada fila el meta-score es
        # sum(w_a * rank_a) / sum(w_a) sobre los agentes usables presentes (no NaN); si no hay
        # ninguno presente, NaN. Equivale al bucle fila-a-fila anterior pero en una operacion.
        date_iso = pd.Timestamp(date).date().isoformat()
        if usable:
            weight_vector = pd.Series({agent: agent_weights[agent] for agent in usable})
            usable_ranks = ranks[usable]
            present_mask = usable_ranks.notna()
            numerator = usable_ranks.mul(weight_vector, axis=1).sum(axis=1, min_count=1)
            denominator = present_mask.mul(weight_vector, axis=1).sum(axis=1)
            meta_series = numerator.div(denominator.where(denominator > 0))
        else:
            meta_series = pd.Series(np.nan, index=ranks.index)
        for ticker, meta_score in meta_series.items():
            meta_scores.append(
                {
                    "ticker": ticker,
                    "snapshot_date": date_iso,
                    "meta_score": meta_score,
                }
            )

    meta = pd.DataFrame(meta_scores)
    if not meta.empty:
        meta["meta_rank"] = meta.groupby("snapshot_date")["meta_score"].rank(method="average", pct=True)

    # Diagnostico del META_FINAL: es el score que de verdad consume la cartera, asi que su rank-IC
    # es la métrica principal de la evaluación.
    # rank-IC de los agentes individuales, que NO es lo que se opera. Tambien se anade el
    # equiponderado como referencia (que aporta la combinacion frente a promediar sin pesos).
    diagnostics = pd.concat(
        [
            diagnostics,
            _meta_diagnostics(meta, labelled, settings, score_col="meta_score", name="meta_final"),
            _equal_weight_diagnostics(labelled, settings),
        ],
        ignore_index=True,
    )
    return meta, pd.DataFrame(weights), diagnostics


def _meta_diagnostics(
    meta: pd.DataFrame, labelled: pd.DataFrame, settings: Settings, score_col: str, name: str
) -> pd.DataFrame:
    """rank-IC por fecha del meta-score frente al retorno futuro realizado."""
    if meta.empty:
        return pd.DataFrame(columns=_DIAG_COLUMNS)
    labels = (
        labelled.loc[labelled["target_available"].fillna(False)]
        .drop_duplicates(["ticker", "snapshot_date"])[
            ["ticker", "snapshot_date", "label_end_date", "forward_excess_return", "is_quarterly"]
        ]
    )
    merged = meta.merge(labels, on=["ticker", "snapshot_date"], how="inner")
    merged = merged.rename(columns={score_col: "score"})
    return _diagnostics_for_score(merged, settings, name)


def _equal_weight_diagnostics(labelled: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """rank-IC del promedio EQUIPONDERADO de los rangos de los agentes (referencia sin pesos)."""
    usable = labelled.loc[labelled["target_available"].fillna(False)].copy()
    if usable.empty:
        return pd.DataFrame(columns=_DIAG_COLUMNS)
    usable["agent_rank"] = usable.groupby(["snapshot_date", "agent"])["score"].rank(
        method="average", pct=True
    )
    equal = (
        usable.groupby(["ticker", "snapshot_date"])
        .agg(
            score=("agent_rank", "mean"),
            label_end_date=("label_end_date", "max"),
            forward_excess_return=("forward_excess_return", "first"),
            is_quarterly=("is_quarterly", "first"),
        )
        .reset_index()
    )
    return _diagnostics_for_score(equal, settings, "meta_equal_weight")


# Clamp de peso por agente tras la contracción del stacker (ver `_constrain_stacked_weights`):
# el meta nunca ignora del todo a un agente ni deja que uno domine. Mínimo UNIVERSAL (todo agente
# disponible recibe al menos meta_weight_min, aunque su rank-IC reciente sea negativo) para
# garantizar diversificación; máximo meta_weight_cap. Con 5 agentes el rango es siempre factible
# (5·0.10 ≤ 1 ≤ 5·0.50). No aplica a `equal` (ya es 1/N).
def _clamp_agent_weights(weights: dict[str, float], minimum: float, maximum: float) -> dict[str, float]:
    """Ajusta los pesos a [AGENT_WEIGHT_MIN, AGENT_WEIGHT_MAX] conservando la suma 1.

    Una sola pasada iterativa: recorta a los límites y redistribuye el exceso/defecto SOLO entre los
    agentes aún libres (los que no están fijados en un límite), en proporción a su peso, hasta que la
    suma vuelve a 1. Si el rango no fuese factible (p. ej. demasiados agentes para el mínimo), cae a
    equiponderación. El resultado es independiente del orden de los agentes.
    """
    agents = list(weights)
    n = len(agents)
    if n == 0:
        return {}
    if n * minimum > 1 + 1e-9 or n * maximum < 1 - 1e-9:
        return {agent: 1 / n for agent in agents}  # rango no factible: equiponderado
    w = {agent: max(float(weights[agent]), 0.0) for agent in agents}
    total = sum(w.values())
    w = {agent: (value / total if total > 0 else 1 / n) for agent, value in w.items()}
    for _ in range(100):
        w = {agent: min(max(value, minimum), maximum) for agent, value in w.items()}
        deficit = 1.0 - sum(w.values())
        if abs(deficit) < 1e-12:
            break
        # Agentes que aún pueden absorber el ajuste: si falta peso (deficit>0) los que no topan en el
        # máximo; si sobra (deficit<0) los que no topan en el mínimo. Se reparte a partes iguales
        # entre ellos (no proporcional: al estar en un límite el proporcional no convergería).
        if deficit > 0:
            movable = [agent for agent, value in w.items() if value < maximum - 1e-12]
        else:
            movable = [agent for agent, value in w.items() if value > minimum + 1e-12]
        if not movable:
            break
        share = deficit / len(movable)
        w = {agent: (value + share if agent in movable else value) for agent, value in w.items()}
    return w


def _weights_as_of(
    labelled: pd.DataFrame,
    date: pd.Timestamp,
    available_agents: set[str],
    settings: Settings,
) -> tuple[dict[str, float], dict[str, dict[str, float | int]]]:
    """Pesos de los agentes a una fecha, según `settings.meta_type`.

    Siempre calcula el rank-IC reciente de cada agente (evidencia para trazabilidad). El modo
    elige cómo se convierte esa evidencia en pesos:
    - "equal": 1/N fijo, ignora el rank-IC reciente.
    - "stacked_oos": Ridge no negativo causal sobre cohortes ya cerradas.
    Sin evidencia suficiente para el stacker, cae a equiponderado.
    """
    closed_history = labelled.loc[
        (labelled["snapshot_ts"] < date)
        & labelled["target_available"].fillna(False)
        & labelled["label_end_ts"].le(date)
    ].copy()
    quarterly_history = closed_history.loc[
        closed_history["is_quarterly"].fillna(False)
    ].copy()
    evidence: dict[str, dict[str, float | int]] = {}
    for agent in tuple(dict.fromkeys([*CATALOG_AGENT_NAMES, *available_agents])):
        if agent not in available_agents:
            continue
        agent_history = quarterly_history.loc[quarterly_history["agent"] == agent]
        cohort_ics: list[float] = []
        for _, cohort in agent_history.groupby("snapshot_date", sort=True):
            value = _rank_ic(cohort, settings.min_rank_ic_cross_section)
            if value is not None:
                cohort_ics.append(value)
        recent = cohort_ics[-settings.meta_ic_lookback_quarters :]
        mean_ic = float(np.mean(recent)) if recent else 0.0
        evidence[agent] = {"mean_rank_ic": mean_ic, "realized_cohorts": len(recent)}

    equal = 1 / len(available_agents) if available_agents else 0.0

    if settings.meta_type == "equal":
        return ({agent: equal for agent in available_agents}, evidence)

    # "stacked_oos": solo usa cohortes trimestrales cerradas y una ventana rolling causal.
    stack_history = _meta_history_window(quarterly_history, settings)
    weights = _stacked_oos_weights(stack_history, available_agents, settings)
    if weights is not None:
        for agent, weight in weights.items():
            evidence.setdefault(agent, {})["stacked_weight"] = weight
        return weights, evidence
    # Until enough *closed* cohorts exist, the fallback is intentionally
    # deterministic.  It never fits on the cohort being scored.
    return ({agent: equal for agent in available_agents}, evidence)


def _meta_history_window(history: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Recorta la historia a la ventana rolling causal del stacker."""
    if history.empty:
        return history
    dates = sorted(pd.to_datetime(history["snapshot_date"]).dropna().unique())
    keep = dates[-settings.meta_history_quarters :]
    return history.loc[pd.to_datetime(history["snapshot_date"]).isin(keep)].copy()


def _stacked_oos_weights(
    history: pd.DataFrame, available_agents: set[str], settings: Settings
) -> dict[str, float] | None:
    """Fit a non-negative Ridge stacker only on already closed OOS agent scores.

    Each row is a historical ticker/snapshot.  Scores are converted to cross-sectional
    ranks and the realised return is also ranked within its snapshot, so the stacker
    learns the same ordering task evaluated by Rank-IC.  It is refit at every date
    from ``history`` only; the current cohort and any unfinished label are excluded
    by the caller.
    """
    agents = sorted(available_agents)
    if len(agents) < 2 or history.empty:
        return None
    wide = history.pivot_table(index=["ticker", "snapshot_date"], columns="agent", values="score", aggfunc="last")
    labels = history.groupby(["ticker", "snapshot_date"], as_index=True)["forward_excess_return"].first()
    wide = wide.reindex(columns=agents).join(labels.rename("target"), how="inner").dropna(subset=["target"])
    if wide.shape[0] < max(40, len(agents) * 12) or wide[agents].notna().sum().min() < 20:
        return None
    dates = wide.index.get_level_values("snapshot_date")
    ranked = wide[agents].groupby(dates).rank(method="average", pct=True)
    target = wide["target"].groupby(dates).rank(method="average", pct=True)
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=8.0, positive=True))
        model.fit(ranked, target)
        coefficients = np.maximum(np.asarray(model[-1].coef_, dtype=float), 0.0)
    except Exception:
        return None
    total = float(coefficients.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return None
    learned = {agent: float(value / total) for agent, value in zip(agents, coefficients, strict=True)}
    return _constrain_stacked_weights(learned, settings)


def _constrain_stacked_weights(weights: dict[str, float], settings: Settings) -> dict[str, float]:
    """Contrae hacia equal y aplica un cap factible conservando suma uno."""
    if not weights:
        return {}
    agents = sorted(weights)
    equal = 1.0 / len(agents)
    shrink = float(settings.meta_equal_shrinkage)
    constrained = {
        agent: (1.0 - shrink) * max(float(weights.get(agent, 0.0)), 0.0) + shrink * equal
        for agent in agents
    }
    return _clamp_agent_weights(
        constrained,
        float(settings.meta_weight_min),
        float(settings.meta_weight_cap),
    )


_DIAG_COLUMNS = ["agent", "prediction_date", "label_end_date", "observations", "rank_ic", "is_quarterly"]


def _diagnostics(labelled: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """rank-IC por fecha de cada agente individual (para explicacion y ablacion)."""
    usable = labelled.loc[labelled["target_available"].fillna(False)]
    frames = [
        _diagnostics_for_score(cohort.assign(agent=agent), settings, agent)
        for agent, cohort in usable.groupby("agent", sort=True)
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_DIAG_COLUMNS)


def _diagnostics_for_score(frame: pd.DataFrame, settings: Settings, name: str) -> pd.DataFrame:
    """rank-IC por snapshot de un `score` cualquiera frente al retorno futuro. `name` etiqueta la
    fila en la columna `agent` (un agente, `meta_final`, `meta_equal_weight`, ...)."""
    rows: list[dict] = []
    for date, cohort in frame.groupby("snapshot_date", sort=True):
        value = _rank_ic(cohort, settings.min_rank_ic_cross_section)
        if value is None:
            continue
        rows.append(
            {
                "agent": name,
                "prediction_date": date,
                "label_end_date": pd.to_datetime(cohort["label_end_date"]).max().date().isoformat(),
                "observations": int(cohort[["score", "forward_excess_return"]].dropna().shape[0]),
                "rank_ic": value,
                "is_quarterly": bool(cohort["is_quarterly"].iloc[0]),
            }
        )
    return pd.DataFrame(rows, columns=_DIAG_COLUMNS)


def _rank_ic(cohort: pd.DataFrame, minimum: int) -> float | None:
    usable = cohort[["score", "forward_excess_return"]].dropna()
    if len(usable) < minimum:
        return None
    if usable["score"].nunique() < 2 or usable["forward_excess_return"].nunique() < 2:
        return None
    value = usable["score"].corr(usable["forward_excess_return"], method="spearman")
    return float(value) if pd.notna(value) else None
