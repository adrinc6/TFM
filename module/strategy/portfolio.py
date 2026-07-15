"""Concentrated portfolio construction and review policy."""

from __future__ import annotations

import pandas as pd

from environment import (
    MAX_PORTFOLIO_SIZE,
    MIN_CONVICTION_ADVANTAGE,
    MIN_OPPORTUNITY_COST_THRESHOLD,
    MIN_PORTFOLIO_SIZE,
    MIN_ROTATION_ADVANTAGE,
    MIN_SCORE_ADVANTAGE_TO_REPLACE,
)

INVESTABLE_STATES = {"Improving", "Intact"}
HOLDABLE_STATES = {"Improving", "Intact", "Maturing"}
BLOCKED_OPPORTUNITIES = {"Avoid", "Value Trap"}
MIN_ENTRY_SCORE = 0.56
MIN_ENTRY_QUALITY = 0.48
MIN_HOLD_SCORE = 0.46
MIN_BUY_TODAY_SCORE = 0.54
MIN_HOLD_MONTHS_BEFORE_ROTATION = 4
REVIEW_TOP_N = max(MAX_PORTFOLIO_SIZE * 3, 20)
# Hard price-risk brake: sell immediately if a position's price return since entry falls at or
# below this, regardless of thesis state or holding period. Without this, nothing bounds a single
# position's downside — the full-universe backtest's worst closed trade lost >50% before this
# existed, and the book's result should not hinge on whether a single name's price collapse is
# caught by the (fundamentals-only) thesis logic in time. Deliberately no symmetric take-profit:
# only the loss tail is capped, winners are left to run (see manager_score-driven exit/rotation
# logic for how gains are eventually realized).
STOP_LOSS_THRESHOLD = -0.25


def initial_portfolio(universe: pd.DataFrame, snapshot_date: str) -> dict[str, dict]:
    ranked = _rank(universe, snapshot_date)
    by_ticker = ranked.set_index("ticker")
    candidates = _entry_candidates(ranked).head(MAX_PORTFOLIO_SIZE)
    if len(candidates) < MIN_PORTFOLIO_SIZE:
        raise RuntimeError(f"Not enough investable candidates at {snapshot_date} to build portfolio.")
    return {row.ticker: _position_from_row(row, snapshot_date, by_ticker) for row in candidates.itertuples(index=False)}


def review_portfolio(current: dict[str, dict], universe: pd.DataFrame, snapshot_date: str) -> tuple[dict[str, dict], list[dict]]:
    today = _rank(universe, snapshot_date)
    by_ticker = today.set_index("ticker")
    transactions: list[dict] = []
    updated = dict(current)

    for ticker, position in list(current.items()):
        if ticker not in by_ticker.index:
            transactions.append(_sell_missing(ticker, snapshot_date, position))
            updated.pop(ticker, None)
            continue
        row = by_ticker.loc[ticker]
        _refresh_position(position, row, snapshot_date)
        exit_reason = _exit_reason(row, position)
        if exit_reason:
            transactions.append(_sell(ticker, snapshot_date, row, exit_reason, position))
            updated.pop(ticker, None)

    for row in _entry_candidates(today).head(REVIEW_TOP_N).itertuples(index=False):
        if row.ticker in updated:
            continue
        if len(updated) >= MAX_PORTFOLIO_SIZE:
            replacement = _replacement_target(updated, row)
            if replacement is None:
                continue
            sell_ticker, sell_position = replacement
            transactions.append(_sell(sell_ticker, snapshot_date, by_ticker.loc[sell_ticker], _replacement_reason(row, sell_position), sell_position))
            updated.pop(sell_ticker, None)
        updated[row.ticker] = _position_from_row(row, snapshot_date, by_ticker)
        transactions.append(_buy(row, snapshot_date, by_ticker))
        if len(updated) >= MAX_PORTFOLIO_SIZE:
            break

    return updated, transactions


# (weight, row attribute, fallback attribute/value, Spanish label) — single source of truth for
# both `manager_score` and `manager_score_breakdown`, so the narrative can never drift from the
# actual formula that drives buy/sell/hold decisions.
MANAGER_SCORE_TERMS = [
    (0.70, "final_score", 0.5, "señal ML aprendida (meta-agente)"),
    (0.07, "thesis_rank_score", 0.5, "ranking de tesis"),
    (0.06, "buy_today_score", 0.5, "convicción de compra hoy"),
    (0.05, "momentum_score", 0.5, "momentum de precio"),
    (0.04, "price_adjusted_valuation_score", "valuation_score", "valoración ajustada por precio"),
    (0.03, "positive_expectation_gap", 0.5, "expectativas de mercado vs. crecimiento real"),
    (0.03, "moat_score", 0.5, "moat / ventaja competitiva"),
    (0.02, "risk_score", 0.5, "riesgo financiero (menor deuda = mejor)"),
]


def manager_score(row) -> float:
    """Live manager score.

    The ML block is `final_score`, the output of the learned meta-agent: the three specialist
    agents (quality / timing / alpha — see module/ml.py::COMPONENT_TARGETS) combined with weights
    learned walk-forward from each agent's marginal contribution to realized forward alpha (see
    module/ml.py::_meta_agent_scores). The agents' feature subsets are no longer required to be
    disjoint; the meta-agent's partial-IC weighting is what prevents double-counting a shared
    signal. `final_score` carries weight 0.70 here — the dominant term, by design: this project's
    core claim is that the AI is the engine of the strategy, not a minor overlay. The remaining
    0.30 are manager risk/timing/valuation overlays the ML score does not fully capture (thesis
    breakdown, near-term buy conviction, momentum, price-adjusted valuation, expectation gap, moat,
    leverage risk) — kept small and explicitly secondary to the learned signal.
    """
    return float(sum(weight * _term_value(row, attr, fallback) for weight, attr, fallback, _ in MANAGER_SCORE_TERMS))


def _term_value(row, attr: str, fallback) -> float:
    if isinstance(fallback, str):
        return float(getattr(row, attr, getattr(row, fallback, 0.5)))
    return float(getattr(row, attr, fallback))


def manager_score_contributions(row) -> list[tuple[str, float, float]]:
    """Per-term (label, contribution, share-of-total) breakdown of `manager_score`, sorted by
    absolute contribution. This is the actual arithmetic behind the ranking, not a proxy metric —
    reusing MANAGER_SCORE_TERMS keeps it impossible to drift from `manager_score` itself."""
    total = manager_score(row)
    contributions = [
        (label, weight * _term_value(row, attr, fallback))
        for weight, attr, fallback, label in MANAGER_SCORE_TERMS
    ]
    return sorted(
        [(label, contribution, contribution / total if total else 0.0) for label, contribution in contributions],
        key=lambda item: item[1],
        reverse=True,
    )


def manager_score_breakdown(row, top_n: int = 3) -> str:
    """Frase en español: qué componentes de manager_score explican la puntuación de esta acción
    en este snapshot, ordenados por impacto — los que más suman primero, luego los que más restan
    si penalizan. Esto explica la decisión real de compra/venta/mantener, no una métrica lateral."""
    contributions = manager_score_contributions(row)
    if not contributions:
        return ""
    total = manager_score(row)
    strongest = contributions[:top_n]
    weakest = [item for item in contributions[-2:] if item not in strongest]
    parts = [f"{label} ({share:.0%} del total)" for label, _, share in strongest]
    lead = f"Puntuación de gestión {total:.2f}, explicada sobre todo por: " + "; ".join(parts) + "."
    if weakest:
        drag = "; ".join(f"{label} ({share:.0%})" for label, _, share in weakest)
        lead += f" Componentes que menos aportan: {drag}."
    return lead


def manager_score_comparison(row, alternative_row) -> str:
    """Frase en español comparando el manager_score del ganador frente a la mejor alternativa
    descartada en ese snapshot, desglosando en qué componente estuvo la mayor diferencia — la
    justificación numérica de "por qué esta y no aquella"."""
    winner_total = manager_score(row)
    alt_total = manager_score(alternative_row)
    advantage = winner_total - alt_total
    winner_terms = dict((label, contribution) for label, contribution, _ in manager_score_contributions(row))
    alt_terms = dict((label, contribution) for label, contribution, _ in manager_score_contributions(alternative_row))
    deltas = sorted(
        ((label, winner_terms[label] - alt_terms.get(label, 0.0)) for label in winner_terms),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    top_deltas = "; ".join(f"{label} ({delta:+.2f})" for label, delta in deltas[:2])
    alt_ticker = getattr(alternative_row, "ticker", "la alternativa")
    return (
        f"Frente a {alt_ticker} (manager_score {alt_total:.2f} vs. {winner_total:.2f}, "
        f"ventaja {advantage:+.2f}): la diferencia viene sobre todo de {top_deltas}."
    )


def _rank(universe: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    today = universe[universe["snapshot_date"] == snapshot_date].copy()
    if today.empty:
        return today
    today["manager_score"] = today.apply(manager_score, axis=1)
    return today.sort_values(["manager_score", "thesis_rank_score", "buy_today_score"], ascending=False)


def _entry_candidates(universe: pd.DataFrame) -> pd.DataFrame:
    if universe.empty:
        return universe
    return universe[
        universe["thesis_state"].isin(INVESTABLE_STATES)
        & ~universe["opportunity_type"].isin(BLOCKED_OPPORTUNITIES)
        & (universe["business_quality_score"] >= MIN_ENTRY_QUALITY)
        & (universe["buy_today_score"] >= MIN_BUY_TODAY_SCORE)
        & (universe["manager_score"] >= MIN_ENTRY_SCORE)
        & (
            (universe.get("momentum_score", pd.Series(0.5, index=universe.index)) >= 0.35)
            | (
                (universe.get("price_adjusted_valuation_score", universe["valuation_score"]) >= 0.72)
                & (universe["business_quality_score"] >= 0.58)
            )
        )
    ]


def _return_since_entry(position: dict, row: pd.Series) -> float | None:
    """Actual price return since the position was opened — the number the stop-loss checks.

    Deliberately independent of `price_return_3m/6m/12m` (fixed trailing windows relative to
    today, not to the entry date) and of `price_return_since_fundamental` (measured from the last
    reported fundamental, not from when the position was actually opened)."""
    entry_price = position.get("entry_price")
    current_price = row.get("price")
    if entry_price is None or current_price is None or pd.isna(current_price) or entry_price <= 0:
        return None
    return float(current_price) / entry_price - 1


def _refresh_position(position: dict, row: pd.Series, snapshot_date: str) -> None:
    position["months_since_entry"] += 1
    position["last_snapshot_date"] = snapshot_date
    position["return_since_entry"] = _return_since_entry(position, row)
    fields = {
        "current_thesis_state": row["thesis_state"],
        "current_conviction_score": float(row["conviction_score"]),
        "current_thesis_rank_score": float(row["thesis_rank_score"]),
        "current_manager_score": float(row["manager_score"]),
        "current_would_buy_today": bool(row["would_buy_today"]),
        "current_buy_today_score": float(row["buy_today_score"]),
        "current_best_alternative_ticker": row["best_alternative_ticker"],
        "current_opportunity_cost_score": float(row["opportunity_cost_score"]),
        "current_business_quality_score": float(row["business_quality_score"]),
        "current_risk_score": float(row["risk_score"]),
        "current_valuation_score": float(row["valuation_score"]),
        "current_price_adjusted_valuation_score": float(row.get("price_adjusted_valuation_score", row["valuation_score"])),
        "current_momentum_score": float(row.get("momentum_score", 0.5)),
        "current_price_return_3m": float(row.get("price_return_3m", 0)),
        "current_price_return_6m": float(row.get("price_return_6m", 0)),
        "current_price_return_12m": float(row.get("price_return_12m", 0)),
        "current_price_return_since_fundamental": float(row.get("price_return_since_fundamental", 0)),
        "current_stale_fundamental_months": float(row.get("stale_fundamental_months", 0)),
        "current_expectation_gap": float(row["expectation_gap"]),
        "current_positive_expectation_gap": float(row["positive_expectation_gap"]),
        "current_thesis": row["investment_thesis"],
        "current_exit_thesis": row["exit_thesis"],
        "current_catalyst": row["catalyst"],
        "sector": row.get("sector", "Unknown"),
    }
    position.update(fields)
    position["entry_vs_current"] = _entry_vs_current_narrative(position)
    if row["thesis_state"] in HOLDABLE_STATES:
        position["months_thesis_intact"] += 1
        position["thesis_improvement_count"] += int(row["thesis_state"] == "Improving")
    else:
        position["thesis_deterioration_count"] += 1
    if not bool(row["would_buy_today"]):
        position["not_buy_today_count"] += 1
    else:
        position["not_buy_today_count"] = 0
    position["thesis_persistence_score"] = position["months_thesis_intact"] / max(position["months_since_entry"], 1)


_ENTRY_VS_CURRENT_METRICS = [
    ("conviction_score", "current_conviction_score", "convicción"),
    ("business_quality_score", "current_business_quality_score", "calidad"),
    ("price_adjusted_valuation_score", "current_price_adjusted_valuation_score", "valoración"),
    ("manager_score", "current_manager_score", "manager_score"),
]


def _entry_vs_current_narrative(position: dict) -> str:
    """Frase en español: valor de cada métrica clave al comprar vs. su valor ahora, con delta
    explícito, para que una venta se pueda justificar con "esto es lo que cambió desde la compra"
    en vez de solo una etiqueta fija."""
    original = position.get("original_scores", {})
    parts = []
    for original_key, current_key, label in _ENTRY_VS_CURRENT_METRICS:
        entry_value = original.get(original_key)
        current_value = position.get(current_key)
        if entry_value is None or current_value is None:
            continue
        delta = float(current_value) - float(entry_value)
        parts.append(f"{label} {float(entry_value):.2f} → {float(current_value):.2f} ({delta:+.2f})")
    if not parts:
        return ""
    months = position.get("months_since_entry", 0)
    return f"Desde la compra ({months} meses): " + "; ".join(parts) + "."


def _exit_reason(row: pd.Series, position: dict) -> str | None:
    """Why (if at all) to sell a held position.

    Two tiers. HARD triggers cut losses/risk immediately regardless of how long the name has been
    held — a broken thesis, an exit-score spike, a name that has become genuinely expensive, or a
    joint momentum+thesis breakdown. SOFT triggers (a mediocre manager score, capital that is better
    used elsewhere, slow repeated deterioration) are about reallocating capital, not damage control,
    so they respect `MIN_HOLD_MONTHS_BEFORE_ROTATION` — the same minimum holding period the
    opportunity-cost rotation path already honors. This stops the most frequent, lowest-conviction
    sell reason from churning the book before a thesis has had time to play out.
    """
    adjusted_valuation = float(row.get("price_adjusted_valuation_score", row["valuation_score"]))
    momentum = float(row.get("momentum_score", 0.5))
    # --- hard triggers: fire immediately (cut losses / risk fast) ---
    return_since_entry = position.get("return_since_entry")
    if return_since_entry is not None and return_since_entry <= STOP_LOSS_THRESHOLD:
        # Pure price risk brake, independent of thesis state: a name can look fundamentally intact
        # (quality/valuation scores unchanged) while the market price has already fallen sharply —
        # without this, nothing bounds the downside of a single position (the backtest's worst
        # closed trade lost >50% before this existed). Upside is deliberately left unbounded (no
        # take-profit) — only the loss tail is capped.
        return "Stop-Loss"
    if row["thesis_state"] == "Broken":
        return "Thesis Broken"
    if row["exit_score"] >= 0.66:
        return "Exit Score Trigger"
    if adjusted_valuation < 0.20 and not bool(row["would_buy_today"]):
        return "Price Adjusted Valuation No Longer Attractive"
    if momentum < 0.35 and row["thesis_state"] == "Weakening":
        # Threshold raised from 0.20 to 0.35 (2026-07 diagnostic round): this trigger had by far
        # the worst mean excess return per sale (-26.4%) of any exit reason on the full-universe
        # run, indicating it fired only after most of the relative damage was already priced in.
        # Reacting at the first sign of weakening thesis + fading momentum (rather than waiting for
        # momentum to collapse below 0.20) should cut losses earlier.
        return "Momentum And Thesis Deterioration"
    # --- soft triggers: only after the minimum holding period (reduce needless rotation) ---
    if position.get("months_since_entry", 0) < MIN_HOLD_MONTHS_BEFORE_ROTATION:
        return None
    if row["manager_score"] < MIN_HOLD_SCORE and not bool(row["would_buy_today"]):
        return "Manager Score Below Hold Hurdle"
    if position["not_buy_today_count"] >= 4 and row["opportunity_cost_score"] >= MIN_OPPORTUNITY_COST_THRESHOLD:
        return "Persistent Better Use Of Capital"
    if row["thesis_state"] == "Weakening" and position["thesis_deterioration_count"] >= 3:
        return "Repeated Thesis Deterioration"
    return None


def _replacement_target(updated: dict[str, dict], candidate) -> tuple[str, dict] | None:
    eligible = {
        ticker: position
        for ticker, position in updated.items()
        if position.get("months_since_entry", 0) >= MIN_HOLD_MONTHS_BEFORE_ROTATION
        or position.get("current_thesis_state") in {"Broken", "Weakening"}
    }
    if not eligible:
        return None
    weakest = min(eligible.items(), key=lambda item: item[1]["current_manager_score"])
    score_advantage = float(candidate.manager_score) - weakest[1]["current_manager_score"]
    conviction_advantage = float(candidate.conviction_score) - weakest[1]["current_conviction_score"]
    momentum_advantage = float(getattr(candidate, "momentum_score", 0.5)) - weakest[1].get("current_momentum_score", 0.5)
    # Primary path: a clear manager_score edge alone justifies the swap. Secondary path: a smaller
    # score edge is enough only when the challenger is also higher-conviction, has better momentum,
    # and is a buy today. All thresholds live in environment.py (single source, no hardcoded copy).
    better_capital_use = score_advantage >= MIN_ROTATION_ADVANTAGE or (
        score_advantage >= MIN_SCORE_ADVANTAGE_TO_REPLACE
        and conviction_advantage >= MIN_CONVICTION_ADVANTAGE
        and momentum_advantage >= 0.05
        and bool(candidate.would_buy_today)
    )
    if better_capital_use:
        return weakest
    return None


def _replacement_reason(candidate, weakest_position: dict) -> str:
    advantage = float(candidate.manager_score) - weakest_position["current_manager_score"]
    breakdown = manager_score_breakdown(candidate)
    reason = (
        f"Opportunity Cost: {candidate.ticker} manager_score advantage "
        f"{advantage:+.2f} sobre {weakest_position.get('ticker', 'la posición más débil')}."
    )
    if breakdown:
        reason = f"{reason} {breakdown}"
    return reason


def _position_from_row(row, snapshot_date: str, by_ticker: pd.DataFrame) -> dict:
    return {
        "ticker": row.ticker,
        "entry_date": snapshot_date,
        "entry_price": float(row.price) if pd.notna(getattr(row, "price", None)) else None,
        "original_snapshot_date": snapshot_date,
        "original_thesis": row.thesis_state,
        "investment_thesis": row.investment_thesis,
        "catalyst": row.catalyst,
        "sector": getattr(row, "sector", "Unknown"),
        "exit_thesis": row.exit_thesis,
        "entry_trigger": row.entry_trigger,
        "manager_score_breakdown_at_entry": manager_score_breakdown(row),
        "vs_best_alternative_at_entry": _vs_best_alternative(row, by_ticker),
        "entry_vs_current": "",
        "would_buy_today": bool(row.would_buy_today),
        "buy_today_score": float(row.buy_today_score),
        "manager_score": float(row.manager_score),
        "best_alternative_ticker": row.best_alternative_ticker,
        "opportunity_cost_score": float(row.opportunity_cost_score),
        "original_scores": {
            "final_score": float(row.final_score),
            "thesis_score": float(row.thesis_score),
            "conviction_score": float(row.conviction_score),
            "business_quality_score": float(row.business_quality_score),
            "manager_score": float(row.manager_score),
            "price_adjusted_valuation_score": float(getattr(row, "price_adjusted_valuation_score", row.valuation_score)),
        },
        "opportunity_type_original": row.opportunity_type,
        "buy_reason": row.buy_reason,
        "months_since_entry": 0,
        "months_thesis_intact": 0,
        "thesis_persistence_score": 1.0,
        "thesis_improvement_count": 0,
        "thesis_deterioration_count": 0,
        "not_buy_today_count": 0,
        "current_thesis_state": row.thesis_state,
        "current_conviction_score": float(row.conviction_score),
        "current_thesis_rank_score": float(row.thesis_rank_score),
        "current_manager_score": float(row.manager_score),
        "current_would_buy_today": bool(row.would_buy_today),
        "current_buy_today_score": float(row.buy_today_score),
        "current_best_alternative_ticker": row.best_alternative_ticker,
        "current_opportunity_cost_score": float(row.opportunity_cost_score),
        "current_business_quality_score": float(row.business_quality_score),
        "current_risk_score": float(row.risk_score),
        "current_valuation_score": float(row.valuation_score),
        "current_price_adjusted_valuation_score": float(getattr(row, "price_adjusted_valuation_score", row.valuation_score)),
        "current_momentum_score": float(getattr(row, "momentum_score", 0.5)),
        "current_price_return_3m": float(getattr(row, "price_return_3m", 0)),
        "current_price_return_6m": float(getattr(row, "price_return_6m", 0)),
        "current_price_return_12m": float(getattr(row, "price_return_12m", 0)),
        "current_price_return_since_fundamental": float(getattr(row, "price_return_since_fundamental", 0)),
        "current_stale_fundamental_months": float(getattr(row, "stale_fundamental_months", 0)),
        "current_expectation_gap": float(row.expectation_gap),
        "current_positive_expectation_gap": float(row.positive_expectation_gap),
        "current_thesis": row.investment_thesis,
        "current_exit_thesis": row.exit_thesis,
        "current_catalyst": row.catalyst,
    }


def _vs_best_alternative(row, by_ticker: pd.DataFrame) -> str:
    """Compara manager_score contra la mejor alternativa descartada (best_alternative_ticker, ya
    calculado en module/strategy/selection.py) para justificar "por qué esta y no aquella"."""
    alt_ticker = getattr(row, "best_alternative_ticker", "")
    if not alt_ticker or by_ticker is None or alt_ticker not in by_ticker.index:
        return ""
    alt_row = by_ticker.loc[alt_ticker]
    return manager_score_comparison(row, alt_row)


def _buy(row, snapshot_date: str, by_ticker: pd.DataFrame) -> dict:
    breakdown = manager_score_breakdown(row)
    vs_alternative = _vs_best_alternative(row, by_ticker)
    reason = row.buy_reason
    if breakdown:
        reason = f"{reason} {breakdown}"
    if vs_alternative:
        reason = f"{reason} {vs_alternative}"
    return {
        "date": snapshot_date,
        "ticker": row.ticker,
        "action": "BUY",
        "reason": reason,
        "thesis": row.investment_thesis,
        "catalyst": row.catalyst,
        "exit_thesis": row.exit_thesis,
        "manager_score_breakdown": breakdown,
        "vs_best_alternative": vs_alternative,
        "would_buy_today": bool(row.would_buy_today),
        "buy_today_score": float(row.buy_today_score),
        "manager_score": float(row.manager_score),
        "best_alternative_ticker": row.best_alternative_ticker,
        "opportunity_cost_score": float(row.opportunity_cost_score),
    }


def _sell(ticker: str, snapshot_date: str, row, reason: str, position: dict) -> dict:
    entry_vs_current = position.get("entry_vs_current", "")
    breakdown = manager_score_breakdown(row)
    return_since_entry = position.get("return_since_entry")
    full_reason = f"{reason}: state={row['thesis_state']}, exit={row['exit_score']:.2f}, manager={row['manager_score']:.2f}"
    if return_since_entry is not None:
        full_reason = f"{full_reason}, price_return_since_entry={return_since_entry:+.1%}"
    if entry_vs_current:
        full_reason = f"{full_reason}. {entry_vs_current}"
    if breakdown:
        full_reason = f"{full_reason} {breakdown}"
    return {
        "date": snapshot_date,
        "ticker": ticker,
        "action": "SELL",
        "reason": full_reason,
        "thesis": row["investment_thesis"],
        "exit_thesis": row["exit_thesis"],
        "catalyst": row["catalyst"],
        "entry_vs_current": entry_vs_current,
        "manager_score_breakdown": breakdown,
    }


def _sell_missing(ticker: str, snapshot_date: str, position: dict) -> dict:
    return {
        "date": snapshot_date,
        "ticker": ticker,
        "action": "SELL",
        "reason": "No current data available for review",
        "thesis": position.get("current_thesis", ""),
        "exit_thesis": position.get("current_exit_thesis", ""),
        "catalyst": position.get("current_catalyst", ""),
        "entry_vs_current": position.get("entry_vs_current", ""),
        "manager_score_breakdown": "",
    }
