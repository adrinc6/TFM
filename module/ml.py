"""Train and score the GARP model: three specialist agents + a learned meta-agent.

Three agents, each fit against its OWN per-snapshot *rank* target (COMPONENT_TARGETS), from its OWN
curated feature subset (AGENT_FEATURES):

    quality  — durable business quality (per-snapshot rank of the forward change in reported ROIC)
    timing   — entry/momentum (per-snapshot rank of short-horizon forward excess return)
    alpha    — the ranking learner (per-snapshot rank of the 12m forward excess return itself)

This replaced an earlier FOUR-agent design (quality/improvement/mispricing/timing) whose
`improvement` and `mispricing` agents had *negative* out-of-sample rank-IC — they dragged the blend
while the meta-agent's weight floor forced them to keep weight — and, more fundamentally, whose four
agents each predicted a proxy target (ROIC change, growth-vs-expectation, a binary cheapness×sign
interaction, squashed short return) while the blend was scored against `target_future_alpha` (12m
excess return). NO agent was trained to rank the very quantity `final_score` is judged on. The new
`alpha` agent closes that gap directly: it learns to rank 12m forward alpha from a curated
momentum+valuation feature set. The information the dropped agents carried is NOT lost — valuation /
relative-growth / expectation-gap survive as *features* of the alpha agent (where the model learns
their non-linear interaction with momentum), instead of as hand-built targets that ranked alpha
negatively.

The alpha agent does NOT reintroduce the old dominating generalist (which saw *every* feature and
the same target): it sees only a restricted forward-signal subset, and the meta-agent weights each
agent by its *marginal* ranking contribution (partial rank-IC against the alpha the other agents
leave unexplained), so redundancy earns no weight — an agent is rewarded only for information the
others don't already carry.

Targets are per-snapshot cross-sectional ranks (not `(x+1)/2` squashes): rank targets are bounded
[0,1] without clipping, are outlier-robust, and match the rank-IC objective the whole system is
measured on. Each target is observable only `its own horizon` ahead (AGENT_HORIZON_MONTHS_OVERRIDE),
leakage-masked identically during walk-forward training (see _walk_forward_component_scores).
Per-snapshot OOS rank-IC / RMSE of each agent and of the combined `final_score` is written to
model_walk_forward_diagnostics.csv so model quality is auditable.
"""

from __future__ import annotations

import logging
import time
from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd

from environment import MAX_PORTFOLIO_SIZE, PROCESSED_DIR, RAW_DIR, Settings
from module.utils import price_cache_by_ticker, read_parquet, write_json, write_parquet

log = logging.getLogger(__name__)


MODEL_FEATURES = [
    "quality_score",
    "growth_score",
    "valuation_score",
    "price_adjusted_valuation_score",
    "momentum_score",
    "moat_score",
    "catalyst_score",
    "risk_score",
    "quality_value_gap",
    "implied_growth",
    "positive_expectation_gap",
    "quality_score_vs_sector",
    "quality_score_vs_universe",
    "growth_score_vs_sector",
    "growth_score_vs_universe",
    "valuation_score_vs_sector",
    "valuation_score_vs_universe",
    "price_return_since_fundamental",
    "price_return_1m",
    "price_return_3m",
    "price_return_6m",
    "price_return_12m",
    "stale_fundamental_months",
    "quality_trend_1y",
    "quality_trend_2y",
    "roic_trend",
    "margin_trend",
    "fcf_trend",
    "growth_acceleration",
    "growth_deceleration",
    "moat_trend",
]

COMPONENT_TARGETS = {
    "quality_probability": "target_quality",
    "timing_probability": "target_timing",
    "alpha_probability": "target_alpha_rank",
}

# Per-agent label horizon override (months). Each dimension resolves on its own clock: an
# entry-timing edge is by nature short-term, so `timing` uses a shorter horizon, while `quality`
# (a fundamental change) and `alpha` (the 12m evaluation label itself) keep the default
# `walk_forward_label_horizon_months`. Each is masked separately in the walk-forward loop under the
# same no-lookahead rule, just a different offset.
AGENT_HORIZON_MONTHS_OVERRIDE = {
    "timing_probability": 3,
}

# Each agent sees a curated feature subset. Subsets are no longer required to be mutually disjoint:
# the disjointness rule existed only to prevent two agents from being redundant views of the same
# signal, but the meta-agent's partial-IC weighting already handles that directly (a redundant agent
# earns no marginal weight — see _fit_meta_weights and test_duplicating_the_signal_collapses_its_
# marginal_ic). So the alpha agent is free to reuse momentum/valuation features that also feed the
# timing agent; it only earns weight for ranking power the others don't already provide.
AGENT_FEATURES = {
    # Durable business quality: profitability, moat, balance-sheet risk and their multi-year trends.
    "quality_probability": [
        "quality_score", "moat_score", "risk_score",
        "quality_score_vs_sector", "quality_score_vs_universe",
        "quality_trend_1y", "quality_trend_2y", "roic_trend", "margin_trend", "fcf_trend", "moat_trend",
    ],
    # Entry/momentum: price momentum across all horizons (incl. the 6m/12m the momentum_only baseline
    # rides — previously dark to every agent), earnings-surprise catalyst, short-window price returns
    # and how stale the last fundamental is (a fresh print re-times the entry).
    "timing_probability": [
        "momentum_score", "catalyst_score",
        "price_return_1m", "price_return_3m", "price_return_6m", "price_return_12m",
        "price_return_since_fundamental", "stale_fundamental_months",
    ],
    # The ranking learner: a curated forward-signal set spanning momentum + valuation + relative
    # growth + expectation gap, trained to rank 12m forward alpha directly. This is where the
    # information the dropped improvement/mispricing agents carried now lives — as features whose
    # non-linear interaction the model learns, not as hand-built targets that ranked alpha negatively.
    "alpha_probability": [
        "price_return_6m", "price_return_12m", "momentum_score",
        "price_adjusted_valuation_score", "quality_value_gap",
        "growth_score_vs_sector", "growth_score_vs_universe",
        "positive_expectation_gap", "catalyst_score",
    ],
}


def _agent_features(probability: str) -> list[str]:
    return [feature for feature in AGENT_FEATURES.get(probability, MODEL_FEATURES) if feature in MODEL_FEATURES]


# Cadencia de ENTRENAMIENTO en meses: trimestral (3) o anual (12). El entrenamiento mensual (1) se
# permite por completitud pero se desaconseja (durante 3 meses los fundamentales no cambian).
_TRAIN_FREQUENCY_MONTHS = {"M": 1, "2M": 2, "Q": 3, "A": 12, "Y": 12}


def _months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def _train_and_apply_dates(all_dates: list[pd.Timestamp], settings: Settings) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """Reparte los snapshots en (train_dates, apply_dates) para walk-forward rodante puro.

    apply_dates = todos los snapshots (cada uno recibe la puntuación del último modelo entrenado).
    train_dates = desde la fecha ancla (`eval_start_date`), los snapshots que distan de ella un
    múltiplo de la cadencia de entrenamiento (`walk_forward_train_frequency`): trimestral por
    defecto, anual como eje del barrido. Como la rejilla de snapshots está fasada al ancla, esos
    puntos coinciden con los refrescos de fundamentales — solo se reentrena cuando hay datos nuevos.
    Cada reentreno usa solo los últimos `max_walk_forward_training_years` de historia disponible en
    esa fecha (sin lookahead); el modelo nunca se congela.
    """
    all_dates = sorted(all_dates)
    apply_dates = list(all_dates)
    cutoff = pd.Timestamp(settings.eval_start_date)
    eligible = [d for d in all_dates if d >= cutoff]
    if not eligible:
        return [], apply_dates
    anchor = eligible[0]
    step = _TRAIN_FREQUENCY_MONTHS.get(settings.walk_forward_train_frequency, 3)
    train_dates = [d for d in eligible if _months_between(anchor, d) % step == 0]
    return (train_dates or [anchor]), apply_dates


def train_and_score(settings: Settings) -> pd.DataFrame:
    # Ventana de entrenamiento coherente: el max acota la historia usada y el min es el mínimo de años
    # para NO caer a fallback. Si min>max, ningún snapshot puede entrenar (siempre training_years<=max<min)
    # y todo cae a fallback GARP, lo que produce un rank-IC degenerado (~1.0) que parece señal y no lo es.
    # Fallar pronto evita presentar ese artefacto como resultado.
    if settings.walk_forward_scoring and settings.min_walk_forward_training_years > settings.max_walk_forward_training_years:
        raise ValueError(
            "Ventana de entrenamiento inválida: min_walk_forward_training_years "
            f"({settings.min_walk_forward_training_years}) > max_walk_forward_training_years "
            f"({settings.max_walk_forward_training_years}). El min debe ser <= max."
        )
    features = read_parquet(PROCESSED_DIR / "features.parquet")
    prices = read_parquet(RAW_DIR / "prices.parquet")
    log.info(
        "Training/scoring universe rows=%s snapshots=%s tickers=%s features=%s",
        len(features),
        features["snapshot_date"].nunique(),
        features["ticker"].nunique(),
        len(MODEL_FEATURES),
    )
    stage_start = time.perf_counter()
    labeled = _add_component_targets(features, prices, settings.benchmark_ticker, settings.walk_forward_label_horizon_months)
    log.info("Forward-looking labels built in %.1fs", time.perf_counter() - stage_start)
    stage_start = time.perf_counter()
    if settings.walk_forward_scoring:
        labeled, importance, diagnostics = _walk_forward_component_scores(labeled, settings)
    else:
        models, importance = _fit_component_models(labeled)
        for probability, model in models.items():
            labeled[probability] = _predict(model, labeled[_agent_features(probability)])
        diagnostics = pd.DataFrame([{"mode": "full_sample", "training_rows": len(labeled)}])
    log.info("Walk-forward agent training/scoring done in %.1fs", time.perf_counter() - stage_start)
    # Meta-agent (the global decision-maker): instead of hard-coded prior weights, it LEARNS,
    # walk-forward, how much to trust each specialist agent from each agent's marginal ranking
    # contribution to realized forward alpha. Falls back to the fixed GARP prior when a snapshot has
    # too little observable history. Weights + partial ICs are persisted for the "learning" view.
    stage_start = time.perf_counter()
    labeled, meta_weights = _meta_agent_scores(labeled, settings)
    log.info("Meta-agent weight learning done in %.1fs", time.perf_counter() - stage_start)
    # The master signal is the meta-agent output `final_score` (the blend of the three agents). Its
    # per-snapshot OOS rank-IC vs. realized forward alpha — plus a rolling and per-year trend — is
    # appended to the diagnostics here, once final_score exists.
    diagnostics = _master_signal_diagnostics(diagnostics, labeled)
    settings.run_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(settings.run_dir / "model_walk_forward_diagnostics.csv", index=False)
    write_parquet(meta_weights, PROCESSED_DIR / "meta_weights_by_snapshot.parquet")
    meta_weights.to_csv(settings.run_dir / "meta_weights_by_snapshot.csv", index=False)
    stage_start = time.perf_counter()
    horizon_comparison = _label_horizon_comparison(labeled, prices, settings)
    horizon_comparison.to_csv(settings.run_dir / "label_horizon_comparison.csv", index=False)
    log.info("Label horizon comparison done in %.1fs", time.perf_counter() - stage_start)
    labeled["business_quality_score"] = (
        0.30 * labeled["quality_probability"]
        + 0.25 * labeled["quality_score_vs_sector"]
        + 0.20 * labeled["moat_score"]
        + 0.15 * labeled["growth_score_vs_sector"]
        + 0.10 * labeled["risk_score"]
    ).clip(0, 1)
    labeled["opportunity_type"] = labeled.apply(_opportunity_type, axis=1)
    write_parquet(labeled, PROCESSED_DIR / "scored_universe.parquet")
    write_json(
        {
            "component_models": list(COMPONENT_TARGETS.keys()),
            "scoring_mode": "walk_forward" if settings.walk_forward_scoring else "full_sample",
            "walk_forward_label_horizon_months": settings.walk_forward_label_horizon_months,
            "min_walk_forward_training_rows": settings.min_walk_forward_training_rows,
            "min_walk_forward_training_years": settings.min_walk_forward_training_years,
            "max_walk_forward_training_years": settings.max_walk_forward_training_years,
            "walk_forward_training_policy": (
                "For each snapshot, train with rows from current_date minus max years through the "
                "current snapshot. All three component targets (quality, timing, alpha) are genuine "
                "forward-looking labels, each a per-snapshot cross-sectional rank observed at "
                "current_date + that agent's own horizon; any label not yet observable at the "
                "training cutoff is masked and replaced by the deterministic GARP fallback for that "
                "component."
            ),
            "component_target_definitions": {
                "quality_probability": "Per-snapshot rank of the forward change in reported ROIC (fundamental quality improving).",
                "timing_probability": "Per-snapshot rank of the short-horizon forward excess return — an entry/momentum timing signal.",
                "alpha_probability": "Per-snapshot rank of the 12m forward excess return itself — the direct alpha-ranking learner.",
            },
            "agent_feature_subsets": {key: _agent_features(key) for key in COMPONENT_TARGETS},
            "agent_horizon_months": {
                key: AGENT_HORIZON_MONTHS_OVERRIDE.get(key, settings.walk_forward_label_horizon_months)
                for key in COMPONENT_TARGETS
            },
            "meta_agent": {
                "policy": (
                    "Per training snapshot, each agent's weight is its MARGINAL ranking contribution: "
                    "the partial Spearman rank-IC of the agent's score against the realized forward "
                    "alpha left unexplained by the other agents, scored for consistency across "
                    "chronological sub-folds (mean - lambda*std) on a 30% chronological hold-out "
                    "inside the training window, clipped at 0 and blended with the equal-weight prior "
                    "at META_WEIGHT_FLOOR before normalizing. Falls back to the prior when history is "
                    "thin or no agent adds marginal ranking power."
                ),
                "prior_weights": AGENT_PRIOR_WEIGHTS,
            },
            "feature_importance": importance,
        },
        PROCESSED_DIR / "model_explainability.json",
    )
    log.info(
        "Scored universe written rows=%s final_score_range=(%.3f, %.3f)",
        len(labeled),
        float(labeled["final_score"].min()),
        float(labeled["final_score"].max()),
    )
    return labeled


def _add_component_targets(df: pd.DataFrame, prices: pd.DataFrame, benchmark_ticker: str, horizon_months: int) -> pd.DataFrame:
    """Build the master evaluation label plus the three agent targets, each from information
    observable ONLY `its own horizon` ahead, and each expressed as a per-snapshot cross-sectional
    rank in [0,1].

    `target_future_alpha` (realized forward excess return over the default horizon) is the master
    evaluation label — the meta-agent learns against it and OOS rank-IC is measured on it. The three
    agent targets:
        target_quality     — rank of the forward change in reported ROIC (12m)
        target_timing      — rank of the short-horizon (3m) forward excess return
        target_alpha_rank  — rank of `target_future_alpha` itself (12m) — the direct ranking learner
    Ranks (not `(x+1)/2` squashes) because rank-IC is the objective: rank targets are bounded without
    clipping, outlier-robust, and preserve the ordering that matters most in the tails. Each is masked
    separately in the walk-forward loop under the same no-lookahead rule, just a different offset.
    """
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    price_cache = price_cache_by_ticker(prices)
    df = df.copy()
    df["snapshot_date_dt"] = pd.to_datetime(df["snapshot_date"])
    # roic is a hard reported fundamental (the only forward feature still needed now that the growth-
    # based improvement agent is gone) — not a re-projection of the input scores.
    forward_columns = [col for col in ("roic",) if col in df.columns]
    feature_cache = _feature_cache(df, forward_columns)

    timing_horizon = AGENT_HORIZON_MONTHS_OVERRIDE.get("timing_probability", horizon_months)

    def forward_alpha(ticker: str, start: pd.Timestamp, months: int) -> float | None:
        stock_ret = _forward_return(price_cache, ticker, start, months=months)
        bench_ret = _forward_return(price_cache, benchmark_ticker, start, months=months)
        return None if stock_ret is None or bench_ret is None else stock_ret - bench_ret

    alphas = []
    quality_deltas = []   # raw forward ROIC change, rank-transformed per snapshot after the loop
    timing_alphas = []    # raw 3m forward excess return, rank-transformed per snapshot after the loop
    log.info(
        "Building forward-looking labels rows=%s horizon_months=%s timing_horizon=%s",
        len(df), horizon_months, timing_horizon,
    )
    for index, row in enumerate(df.itertuples(index=False), start=1):
        alpha = forward_alpha(row.ticker, row.snapshot_date_dt, horizon_months)
        alphas.append(alpha)

        # Timing needs its own (shorter) forward alpha at its overridden horizon.
        timing_alpha = alpha if timing_horizon == horizon_months else forward_alpha(row.ticker, row.snapshot_date_dt, timing_horizon)
        timing_alphas.append(timing_alpha)

        # quality raw signal: forward change in the ACTUALLY REPORTED ROIC (fundamental quality),
        # not the derived quality_score. NA when roic unavailable at either end.
        current_roic = getattr(row, "roic", None) if "roic" in forward_columns else None
        fwd_roic = _forward_feature_value(feature_cache, row.ticker, row.snapshot_date_dt, "roic", horizon_months) if "roic" in forward_columns else None
        if fwd_roic is None or current_roic is None or pd.isna(current_roic):
            quality_deltas.append(None)
        else:
            quality_deltas.append(float(fwd_roic) - float(current_roic))

        if index % 10000 == 0:
            log.info("Forward-looking labels progress rows=%s/%s", index, len(df))

    df["target_future_alpha"] = alphas
    # Per-snapshot cross-sectional rank in [0,1] for each agent target. NaNs (unobservable future)
    # stay NaN and are handled by the walk-forward masking + garp fallback downstream.
    df["_quality_delta_raw"] = quality_deltas
    df["_timing_alpha_raw"] = timing_alphas
    grouped = df.groupby("snapshot_date")
    df["target_quality"] = grouped["_quality_delta_raw"].rank(pct=True)
    df["target_timing"] = grouped["_timing_alpha_raw"].rank(pct=True)
    df["target_alpha_rank"] = grouped["target_future_alpha"].rank(pct=True)
    df = df.drop(columns=["_quality_delta_raw", "_timing_alpha_raw"])
    trainable = df["target_future_alpha"].notna()
    if trainable.sum() < 5:
        log.warning("Few future labels available; model will lean on deterministic GARP score.")
        df["target_future_alpha"] = df["garp_score"]
    return df.drop(columns=["snapshot_date_dt"])


def _walk_forward_component_scores(df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Pure rolling walk-forward: relearn the 3 agents every quarter (train_dates), across the
    ENTIRE evaluated window — never freezing. Each relearning point trains only on the trailing
    `max_walk_forward_training_years` of history available as of that date, so the model is always
    adapted to recent market regime rather than fixed at a single point-in-time snapshot. Dates
    between quarterly relearning points are scored with the latest trained-so-far model rather than
    retrained individually (quarterly cadence keeps the cost bounded); apply_dates receives a score
    from whichever model was most recently fit as of that date.
    """
    scored = df.copy()
    scored["snapshot_date_dt"] = pd.to_datetime(scored["snapshot_date"])
    for probability, target in COMPONENT_TARGETS.items():
        scored[probability] = _fallback_probability(scored, target)

    all_dates = sorted(scored["snapshot_date_dt"].dropna().unique())
    train_dates, apply_dates = _train_and_apply_dates(all_dates, settings)
    train_dates_set = set(train_dates)
    diagnostics = []
    latest_importance: dict = {probability: {feature: 0 for feature in MODEL_FEATURES} for probability in COMPONENT_TARGETS}
    label_offset = pd.DateOffset(months=settings.walk_forward_label_horizon_months)
    # Each agent's label is only observable `its own horizon` ahead — mispricing uses a shorter
    # override (see AGENT_HORIZON_MONTHS_OVERRIDE) so it is unmasked earlier than the 12m agents,
    # while still respecting the same no-lookahead rule (label_offset <= current snapshot date).
    label_offsets = {
        probability: pd.DateOffset(months=AGENT_HORIZON_MONTHS_OVERRIDE.get(probability, settings.walk_forward_label_horizon_months))
        for probability in COMPONENT_TARGETS
    }
    max_history = pd.DateOffset(years=settings.max_walk_forward_training_years)
    log.info(
        "Rolling walk-forward snapshots=%s train_dates=%s (quarterly relearning) components=%s",
        len(apply_dates), len(train_dates), len(COMPONENT_TARGETS),
    )
    models: dict = {}
    last_training_snapshot: pd.Timestamp | None = None
    last_training_stats: dict = {}
    loop_start = time.perf_counter()
    for date_index, date in enumerate(apply_dates, start=1):
        current_mask = scored["snapshot_date_dt"] == date
        is_train_date = date in train_dates_set

        if is_train_date:
            train_start_cutoff = pd.Timestamp(date) - max_history
            train_mask = (
                (scored["snapshot_date_dt"] >= train_start_cutoff)
                & (scored["snapshot_date_dt"] <= pd.Timestamp(date))
            )
            train = scored[train_mask].copy()
            observable_mask = train["snapshot_date_dt"] + label_offset <= pd.Timestamp(date)
            alpha_label_rows = int((observable_mask & train["target_future_alpha"].notna()).sum())
            # Mask the 12m evaluation alpha for rows whose future is not yet observable at this
            # training date — `_select_hyperparameters` reads target_future_alpha to pick capacity by
            # rank-IC, so it must not see any not-yet-realized alpha (no lookahead).
            train.loc[~observable_mask, "target_future_alpha"] = pd.NA
            for probability, target in COMPONENT_TARGETS.items():
                target_observable = train["snapshot_date_dt"] + label_offsets[probability] <= pd.Timestamp(date)
                train.loc[~target_observable, target] = pd.NA
            training_start = train["snapshot_date_dt"].min() if not train.empty else pd.NaT
            training_end = train["snapshot_date_dt"].max() if not train.empty else pd.NaT
            training_years = (
                (training_end - training_start).days / 365.25
                if pd.notna(training_start) and pd.notna(training_end)
                else 0.0
            )
            fallback_reasons = []
            if len(train) < settings.min_walk_forward_training_rows:
                fallback_reasons.append("not_enough_rows")
            if train.empty or train["ticker"].nunique() < 10:
                fallback_reasons.append("not_enough_tickers")
            if training_years < settings.min_walk_forward_training_years:
                fallback_reasons.append("not_enough_years")
            if not fallback_reasons:
                log.info(
                    "Walk-forward TRAIN %s/%s %s fitting training_rows=%s training_years=%.2f alpha_labels=%s",
                    date_index, len(apply_dates), pd.Timestamp(date).date().isoformat(),
                    len(train), training_years, alpha_label_rows,
                )
                fit_start = time.perf_counter()
                models, latest_importance = _fit_component_models(train)
                log.info(
                    "Walk-forward TRAIN %s/%s %s fitted 3 agents in %.1fs",
                    date_index, len(apply_dates), pd.Timestamp(date).date().isoformat(),
                    time.perf_counter() - fit_start,
                )
                last_training_snapshot = pd.Timestamp(date)
                last_training_stats = {
                    "training_start": training_start.date().isoformat(),
                    "training_end": training_end.date().isoformat(),
                    "training_years": float(training_years),
                    "training_rows": int(len(train)),
                    "training_tickers": int(train["ticker"].nunique()),
                    "alpha_label_observable_rows": alpha_label_rows,
                    "alpha_label_fallback_rows": int(len(train) - alpha_label_rows),
                }

        if models:
            for probability, model in models.items():
                features = _agent_features(probability)
                scored.loc[current_mask, probability] = _predict(model, scored.loc[current_mask, features]).values
            oos_metrics = _oos_metrics(scored.loc[current_mask])
            diagnostics.append({
                "snapshot_date": pd.Timestamp(date).date().isoformat(),
                "mode": "walk_forward_model",
                "is_train_date": bool(is_train_date),
                "training_snapshot_date": last_training_snapshot.date().isoformat() if last_training_snapshot is not None else "",
                "fallback_reason": "",
                "uses_rows_through_current_date": True,
                "min_training_years": settings.min_walk_forward_training_years,
                "max_training_years": settings.max_walk_forward_training_years,
                "label_horizon_months": settings.walk_forward_label_horizon_months,
                **last_training_stats,
                **oos_metrics,
            })
            if date_index % 10 == 0 or date_index == len(apply_dates):
                elapsed = time.perf_counter() - loop_start
                log.info(
                    "Walk-forward APPLY %s/%s %s scored elapsed=%.1fs avg=%.2fs/snapshot",
                    date_index, len(apply_dates), pd.Timestamp(date).date().isoformat(),
                    elapsed, elapsed / date_index,
                )
        else:
            # No trained model yet available at all (before the first successful training
            # snapshot) — fall back to the deterministic GARP score, same as before.
            log.info(
                "Walk-forward %s/%s %s no trained model yet, using GARP fallback",
                date_index, len(apply_dates), pd.Timestamp(date).date().isoformat(),
            )
            diagnostics.append({
                "snapshot_date": pd.Timestamp(date).date().isoformat(),
                "mode": "fallback_garp",
                "is_train_date": bool(is_train_date),
                "training_snapshot_date": "",
                "fallback_reason": "not_enough_years" if is_train_date else "before_first_training_snapshot",
                "uses_rows_through_current_date": True,
                "min_training_years": settings.min_walk_forward_training_years,
                "max_training_years": settings.max_walk_forward_training_years,
                "label_horizon_months": settings.walk_forward_label_horizon_months,
            })
    return scored.drop(columns=["snapshot_date_dt"]), latest_importance, pd.DataFrame(diagnostics)


# Curva de BREADTH / tamaño de cartera para el análisis "¿5 o 50 acciones?". Para cada N se mide, por
# snapshot, el alpha medio de los N mejores del ranking y su ventaja (lift) sobre la media del universo
# — el tramo que de verdad se compraría. Se guarda por snapshot en el diagnóstico para poder elegir el
# tamaño óptimo a posteriori sin re-correr el backtest por cada N.
BREADTH_TOP_NS = [5, 10, 20, 50]


def _master_signal_diagnostics(
    diagnostics: pd.DataFrame,
    labeled: pd.DataFrame,
    window: int = 12,
    top_n: int = MAX_PORTFOLIO_SIZE,
) -> pd.DataFrame:
    """Append the OOS quality of the MASTER signal (`final_score`, the meta-agent output) to the
    per-snapshot diagnostics: its Spearman rank-IC vs. realized forward alpha, a rolling mean, a
    per-calendar-year mean + approximate t-stat (mean / (std / sqrt(n))), y la CURVA de BREADTH por
    tamaño de cartera (`top{N}_alpha`, `top{N}_alpha_lift` para N en BREADTH_TOP_NS; `top_n_alpha`/
    `top_n_alpha_lift` son alias del tamaño real de cartera para compatibilidad).

    This replaces the old per-agent "alpha" IC column: the signal whose quality-over-time actually
    matters is the combined meta-agent score (`final_score`), not any single agent. With ~90 walk-forward
    snapshots the rolling/year trend is descriptive, not a formal significance test — reported as-is,
    without forcing a "the system is improving" narrative if the data doesn't show one.

    `top_n` es el tamaño real de la cartera: las métricas top-N miden el tramo del ranking que de
    verdad se compra, que es donde el rank-IC global no llega.
    """
    diagnostics = diagnostics.copy()
    if diagnostics.empty or "snapshot_date" not in diagnostics.columns or "final_score" not in labeled.columns:
        diagnostics["rank_ic_final"] = pd.NA
        diagnostics["rank_ic_final_rolling"] = pd.NA
        diagnostics["top_n_alpha"] = pd.NA
        diagnostics["top_n_alpha_lift"] = pd.NA
        return diagnostics
    scored = labeled[["snapshot_date"]].copy()
    scored["pred"] = pd.to_numeric(labeled["final_score"], errors="coerce")
    scored["real"] = pd.to_numeric(labeled.get("target_future_alpha"), errors="coerce")
    ic_by_snapshot: dict[str, float] = {}
    for snapshot, group in scored.groupby("snapshot_date"):
        pair = group[["pred", "real"]].dropna()
        if len(pair) >= 5 and pair["pred"].nunique() > 1:
            ic = pair["pred"].corr(pair["real"], method="spearman")
            if np.isfinite(ic):
                ic_by_snapshot[str(snapshot)] = float(ic)

    # Métrica de BREADTH: el rank-IC mide el orden de TODO el universo (~71k filas/snapshot), pero la
    # cartera solo compra el top-N. Ordenar bien 71k filas no implica acertar el top-N (correlación
    # empírica IC↔alpha entre escenarios, medida sobre el rank-IC OOS: Spearman ~+0.36). Se calcula la
    # CURVA para varios N (BREADTH_TOP_NS): el alpha medio de los N mejores y su ventaja sobre el
    # universo, para responder "¿5 o 50 acciones?" desde el ranking guardado.
    breadth_ns = sorted(set(BREADTH_TOP_NS) | {top_n})
    alpha_by_n: dict[int, dict[str, float]] = {n: {} for n in breadth_ns}
    lift_by_n: dict[int, dict[str, float]] = {n: {} for n in breadth_ns}
    for snapshot, group in scored.groupby("snapshot_date"):
        pair = group[["pred", "real"]].dropna()
        if len(pair) < 5:
            continue
        universe_mean = float(pair["real"].mean())
        ranked = pair.sort_values("pred", ascending=False)
        for n in breadth_ns:
            if len(pair) < n:
                continue
            top_alpha = float(ranked.head(n)["real"].mean())
            alpha_by_n[n][str(snapshot)] = top_alpha
            lift_by_n[n][str(snapshot)] = top_alpha - universe_mean

    keys = diagnostics["snapshot_date"].astype(str)
    for n in breadth_ns:
        diagnostics[f"top{n}_alpha"] = keys.map(alpha_by_n[n])
        diagnostics[f"top{n}_alpha_lift"] = keys.map(lift_by_n[n])
    # Alias del tamaño real de cartera (top_n) para compatibilidad con metrics/viewer.
    diagnostics["top_n_alpha"] = diagnostics[f"top{top_n}_alpha"]
    diagnostics["top_n_alpha_lift"] = diagnostics[f"top{top_n}_alpha_lift"]

    diagnostics["rank_ic_final"] = diagnostics["snapshot_date"].astype(str).map(ic_by_snapshot)
    ic = pd.to_numeric(diagnostics["rank_ic_final"], errors="coerce")
    # Solo los snapshots con modelo entrenado entran en las agregaciones. Antes del cutoff
    # `final_score` cae al `garp_score` determinista y su IC (~0.62) mide la correlación del baseline
    # consigo mismo, no aprendizaje. Un único snapshot de fallback dentro de un año mixto bastaba para
    # duplicar la media de ese año (2018: 0.098 contaminado vs. 0.048 real), y esa media es la que
    # consume el runner de experimentos. El IC por snapshot se conserva sin enmascarar: es auditable y
    # `mode` dice de dónde sale cada fila.
    if "mode" in diagnostics.columns:
        ic = ic.where(diagnostics["mode"] == "walk_forward_model")
    diagnostics["rank_ic_final_rolling"] = ic.rolling(window=window, min_periods=max(3, window // 2)).mean()
    years = pd.to_datetime(diagnostics["snapshot_date"], errors="coerce").dt.year
    year_stats = ic.groupby(years).agg(["mean", "std", "count"])
    year_stats["t_stat"] = year_stats["mean"] / (year_stats["std"] / year_stats["count"].pow(0.5))
    diagnostics["rank_ic_final_year_mean"] = years.map(year_stats["mean"].to_dict())
    diagnostics["rank_ic_final_year_tstat"] = years.map(year_stats["t_stat"].to_dict())
    return diagnostics


# Prior (the fixed GARP combination) — the meta-agent fallback AND the anchor the learned weights
# are shrunk toward (see META_WEIGHT_FLOOR). Tilted toward `quality` because that is where the
# *stable* cross-sectional signal lives: across the 2018+ operative window the quality agent's OOS
# rank-IC is positive in 8/9 years with the lowest variance, while timing/alpha have higher peak IC
# but swing sign year to year (see docs/diagnostico_aprendizaje.md and scripts/buscar_pesos_meta.py).
# An offline weight search over the already-scored agents (scripts/buscar_pesos_meta.py) traces a
# clean trade-off: tilting further toward quality keeps raising the combined signal's stability
# (0.30/0.35/0.35 -> std 0.123, 3 negative years; 0.60/0.25/0.15 -> std 0.077, 1 negative year) but
# a full tilt also gives back realized alpha/IR (the tail-winner momentum exposure that drove the
# headline alpha by asymmetry, not by ranking). 0.45/0.30/0.25 is the chosen middle ground: it lifts
# mean OOS rank-IC (+0.029 -> +0.032 offline), lowers dispersion (0.123 -> 0.101) and cuts a negative
# year, while keeping most of the momentum weight — a stability gain without over-sacrificing the
# economic result. The earlier 0.30/0.35/0.35 tilt toward timing/alpha was calibrated to where the
# raw alpha lived, not where the *reliable* ranking signal lived, so final_score inherited the alpha
# agent's instability. See docs/diagnostico_aprendizaje.md for the measured walk-forward outcome.
AGENT_PRIOR_WEIGHTS = {
    "quality_probability": 0.45,
    "timing_probability": 0.30,
    "alpha_probability": 0.25,
}
AGENT_KEYS = list(AGENT_PRIOR_WEIGHTS.keys())

# ¿El meta-agente APRENDE los pesos por trimestre, o los fija al prior? Con True (por defecto) los
# pesos se reaprenden walk-forward cada trimestre; con False se congelan en AGENT_PRIOR_WEIGHTS en
# TODOS los snapshots, pero los agentes siguen puntuando walk-forward (percentiles por snapshot). Es
# la ablación limpia "sin meta-aprendido": aísla el aporte del aprendizaje de pesos sin cambiar la
# escala del final_score (a diferencia de walk_forward_scoring=False, que además pasa los agentes a
# un único ajuste full-sample y comprime la señal por debajo del umbral de entrada de la cartera).
LEARN_META_WEIGHTS = True


def _meta_agent_scores(labeled: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Learn, every training snapshot (quarterly, across the whole evaluated window), how to weight
    the three agents against the realized forward alpha — pure rolling walk-forward, same scheme as
    _walk_forward_component_scores: the meta-agent keeps relearning every quarter all the way
    through the live-portfolio window, never freezing.

    Each training snapshot fits ONLY on rows whose forward alpha was already observable at that
    date, weighting each agent by its consistency-scored partial rank-IC — interpretable and hard
    to overfit with 3 agents. Snapshots without enough observable history fall back to the fixed
    prior weights; snapshots between quarterly relearning points reuse the last learned weights
    until the next relearning point.
    """
    labeled = labeled.copy()
    labeled["snapshot_date_dt"] = pd.to_datetime(labeled["snapshot_date"])
    labeled["final_score"] = _prior_combination(labeled)
    labeled["meta_weight_source"] = "prior"

    if not settings.walk_forward_scoring:
        rows = [{
            "snapshot_date": "all", "source": "prior", "n_train": 0,
            **AGENT_PRIOR_WEIGHTS, **{f"partial_ic_{key}": float("nan") for key in AGENT_KEYS},
        }]
        return labeled.drop(columns=["snapshot_date_dt"]), pd.DataFrame(rows)

    all_dates = sorted(labeled["snapshot_date_dt"].dropna().unique())
    train_dates, apply_dates = _train_and_apply_dates(all_dates, settings)
    train_dates_set = set(train_dates)
    label_offset = pd.DateOffset(months=settings.walk_forward_label_horizon_months)
    max_history = pd.DateOffset(years=settings.max_walk_forward_training_years)
    weight_rows = []
    weights = dict(AGENT_PRIOR_WEIGHTS)
    partial_ics = {key: float("nan") for key in AGENT_KEYS}
    source = "prior"
    last_training_snapshot: pd.Timestamp | None = None
    meta_loop_start = time.perf_counter()
    for meta_index, date in enumerate(apply_dates, start=1):
        if date in train_dates_set and LEARN_META_WEIGHTS:
            train_mask = (
                (labeled["snapshot_date_dt"] >= (pd.Timestamp(date) - max_history))
                & (labeled["snapshot_date_dt"] <= pd.Timestamp(date))
                & (labeled["snapshot_date_dt"] + label_offset <= pd.Timestamp(date))
                & labeled["target_future_alpha"].notna()
            )
            train = labeled[train_mask]
            fitted = _fit_meta_weights(train)
            if fitted is not None:
                weights, partial_ics, source = fitted[0], fitted[1], "learned"
            else:
                weights, partial_ics, source = dict(AGENT_PRIOR_WEIGHTS), {key: float("nan") for key in AGENT_KEYS}, "prior"
            last_training_snapshot = pd.Timestamp(date)
        else:
            train = labeled.iloc[0:0]

        current_mask = labeled["snapshot_date_dt"] == date
        combo = sum(weights[key] * labeled.loc[current_mask, key] for key in AGENT_KEYS)
        labeled.loc[current_mask, "final_score"] = combo.clip(0, 1)
        labeled.loc[current_mask, "meta_weight_source"] = source
        weight_rows.append({
            "snapshot_date": pd.Timestamp(date).date().isoformat(),
            "source": source,
            "training_snapshot_date": last_training_snapshot.date().isoformat() if last_training_snapshot is not None else "",
            "n_train": int(len(train)),
            **{key: float(weights[key]) for key in AGENT_KEYS},
            **{f"partial_ic_{key}": float(partial_ics[key]) for key in AGENT_KEYS},
        })
        if meta_index % 20 == 0 or meta_index == len(apply_dates):
            log.info(
                "Meta-agent %s/%s %s elapsed=%.1fs",
                meta_index, len(apply_dates), pd.Timestamp(date).date().isoformat(),
                time.perf_counter() - meta_loop_start,
            )
    return labeled.drop(columns=["snapshot_date_dt"]), pd.DataFrame(weight_rows)


def _prior_combination(df: pd.DataFrame) -> pd.Series:
    return sum(AGENT_PRIOR_WEIGHTS[key] * df[key] for key in AGENT_KEYS).clip(0, 1)


META_FIT_MIN_ROWS = 80
META_FIT_MIN_VALIDATION_ROWS = 30
META_FIT_VALIDATION_FRACTION = 0.30
# Consistency floor: an agent's raw partial-IC weight is blended with AGENT_PRIOR_WEIGHTS (the
# quality-tilted, stability-calibrated prior — NOT equal-weight) before normalizing. Without this,
# a single agent that edges out the others on one quarter's 30% hold-out collapses the whole
# meta-agent to 100%/0%/0% (observed repeatedly in meta_weights_by_snapshot.parquet across the
# evaluated window) — noise in a small validation split, not a genuinely learned preference.
# META_WEIGHT_FLOOR keeps every agent contributing something AND anchors the shrinkage target toward
# the stable-signal prior, so when the learned partial-ICs are noisy the combination falls back
# toward quality rather than toward an even split that re-inflates the erratic alpha agent.
META_WEIGHT_FLOOR = 0.10
# Consistency penalty: within the validation slice, the partial-IC is computed on N_CONSISTENCY_FOLDS
# chronological sub-folds instead of the whole slice at once, and an agent's score is
# mean(fold_ic) - CONSISTENCY_LAMBDA * std(fold_ic) rather than the single-slice IC — so an agent
# that ranks well but erratically (high mean, high variance across folds) is worth less than one
# that ranks moderately but reliably. This is the direct implementation of "prefer consistency over
# a lucky spike" for the meta-agent's own selection criterion, not just for the final portfolio.
N_CONSISTENCY_FOLDS = 3
CONSISTENCY_LAMBDA = 0.5


def _fit_meta_weights(train: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]] | None:
    """Weight each agent by its MARGINAL ranking contribution, not its raw rank-IC.

    Design choices, all to reward complementarity over redundancy and to keep the combination
    from overfitting a single quarter's small validation split:

    1. A chronological 70/30 split INSIDE the training window (fit vs. validation), same walk-forward
       discipline as the outer train/apply split — no lookahead, and an agent must generalize within
       the window to earn weight, not just fit it.

    2. For each agent, we regress the realized forward alpha on the OTHER agents (OLS on the
       fit part), take the residual on the validation part — the alpha the others left
       unexplained — and score the agent by its Spearman rank-IC against THAT residual (partial
       rank-IC), computed on N_CONSISTENCY_FOLDS chronological sub-folds of the validation slice
       rather than the slice as a whole. An agent's score is mean(fold_ic) - CONSISTENCY_LAMBDA *
       std(fold_ic): an agent that ranks well but erratically across folds scores lower than one
       that ranks moderately but reliably — consistency is rewarded, not just peak correlation. An
       agent that merely echoes the others' ranking scores ~0; an agent that ranks the leftover
       alpha consistently earns weight.

    3. Raw partial ICs are clipped at 0, normalized, then shrunk toward AGENT_PRIOR_WEIGHTS (the
       quality-tilted, stability-calibrated prior — not equal-weight) at META_WEIGHT_FLOOR — this is
       the anti-corner-solution regularization: it prevents one agent's marginal edge on a ~30-row
       hold-out from zeroing out the others, and it anchors the shrinkage toward the stable-signal
       agent so noisy quarters fall back to quality rather than re-inflating the erratic alpha agent.

    Returns (weights, partial_ics) or None when the window is too thin or no agent adds marginal
    ranking power (the caller then falls back to the fixed prior).
    """
    t = train.dropna(subset=["target_future_alpha"]).sort_values("snapshot_date_dt")
    if len(t) < META_FIT_MIN_ROWS:
        return None
    y = pd.to_numeric(t["target_future_alpha"], errors="coerce").to_numpy(dtype=float)
    if np.ptp(y) == 0:
        return None
    scores = t[AGENT_KEYS].astype(float).to_numpy()
    split = int(len(t) * (1 - META_FIT_VALIDATION_FRACTION))
    if split < META_FIT_MIN_ROWS - META_FIT_MIN_VALIDATION_ROWS or len(t) - split < META_FIT_MIN_VALIDATION_ROWS:
        return None
    fit_slice, val_slice = slice(0, split), slice(split, len(t))
    n_val = len(t) - split
    fold_bounds = np.linspace(0, n_val, N_CONSISTENCY_FOLDS + 1, dtype=int)

    partial_ics: dict[str, float] = {}
    for position, key in enumerate(AGENT_KEYS):
        others = [i for i in range(len(AGENT_KEYS)) if i != position]
        design_fit = np.column_stack([scores[fit_slice][:, others], np.ones(split)])
        coefficients, *_ = np.linalg.lstsq(design_fit, y[fit_slice], rcond=None)
        design_val = np.column_stack([scores[val_slice][:, others], np.ones(n_val)])
        residual = y[val_slice] - design_val @ coefficients
        agent_val = scores[val_slice][:, position]

        fold_ics = []
        for fold_start, fold_end in zip(fold_bounds[:-1], fold_bounds[1:]):
            pair = pd.DataFrame({
                "agent": agent_val[fold_start:fold_end], "residual": residual[fold_start:fold_end],
            }).dropna()
            if len(pair) < max(5, META_FIT_MIN_VALIDATION_ROWS // N_CONSISTENCY_FOLDS) or pair["agent"].nunique() < 2:
                continue
            ic = pair["agent"].corr(pair["residual"], method="spearman")
            if np.isfinite(ic):
                fold_ics.append(float(ic))
        if not fold_ics:
            partial_ics[key] = 0.0
            continue
        consistency_score = float(np.mean(fold_ics)) - CONSISTENCY_LAMBDA * float(np.std(fold_ics))
        partial_ics[key] = max(0.0, consistency_score)

    total = sum(partial_ics.values())
    if total <= 0:
        return None
    raw_weights = {key: partial_ics[key] / total for key in AGENT_KEYS}
    # Shrink the learned weights toward the quality-tilted prior (not equal-weight): when the
    # partial-ICs are noisy, the floor pulls the combination back toward the stable-signal agent
    # rather than toward an even split that re-inflates the erratic alpha agent.
    blended = {
        key: (1 - META_WEIGHT_FLOOR) * raw_weights[key] + META_WEIGHT_FLOOR * AGENT_PRIOR_WEIGHTS[key]
        for key in AGENT_KEYS
    }
    return blended, partial_ics


LABEL_HORIZON_CANDIDATES_MONTHS = [3, 6, 12]


def _label_horizon_comparison(scored: pd.DataFrame, prices: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Does the master signal predict better at 3, 6, or 12 months out?

    Reuses the ALREADY-COMBINED meta-agent score `final_score` and checks its Spearman rank-IC
    against the realized forward alpha computed at each candidate horizon, across every snapshot —
    with pure rolling walk-forward every snapshot carries its own freshly-retrained-as-of-then
    model, so there is no stale "frozen phase" to exclude. This answers "does the strategy's edge
    live at a shorter or longer horizon?" without retraining 3x — an evidence-based check on the
    default 12m label horizon.
    """
    if "final_score" not in scored.columns:
        return pd.DataFrame()
    scored = scored.copy()
    scored["snapshot_date_dt"] = pd.to_datetime(scored["snapshot_date"])
    training_rows = scored.copy()
    price_cache = price_cache_by_ticker(prices.assign(date=pd.to_datetime(prices["date"])))

    rows = []
    for months in LABEL_HORIZON_CANDIDATES_MONTHS:
        alphas = [
            _forward_return(price_cache, r.ticker, r.snapshot_date_dt, months=months)
            for r in training_rows.itertuples(index=False)
        ]
        bench_alphas = [
            _forward_return(price_cache, settings.benchmark_ticker, r.snapshot_date_dt, months=months)
            for r in training_rows.itertuples(index=False)
        ]
        realized = pd.Series(
            [None if s is None or b is None else s - b for s, b in zip(alphas, bench_alphas)],
            index=training_rows.index,
        )
        realized = pd.to_numeric(realized, errors="coerce")
        pair = pd.DataFrame({"pred": training_rows["final_score"], "real": realized}).dropna()
        if len(pair) < 20 or pair["pred"].nunique() < 2:
            rows.append({"horizon_months": months, "rank_ic_mean": None, "n_snapshots": 0, "n_rows": len(pair)})
            continue
        by_snapshot = pd.DataFrame({
            "snapshot_date_dt": training_rows.loc[pair.index, "snapshot_date_dt"],
            "pred": pair["pred"], "real": pair["real"],
        })
        ic_by_snapshot = by_snapshot.groupby("snapshot_date_dt").apply(
            lambda g: g["pred"].corr(g["real"], method="spearman") if len(g) >= 10 and g["pred"].nunique() > 1 else None,
            include_groups=False,
        ).dropna()
        rows.append({
            "horizon_months": months,
            "rank_ic_mean": float(ic_by_snapshot.mean()) if not ic_by_snapshot.empty else None,
            "n_snapshots": int(len(ic_by_snapshot)),
            "n_rows": int(len(pair)),
        })
    result = pd.DataFrame(rows)
    if not result["rank_ic_mean"].dropna().empty:
        best = result.loc[result["rank_ic_mean"].idxmax(), "horizon_months"]
        result["is_current_default"] = result["horizon_months"] == settings.walk_forward_label_horizon_months
        result["is_best_observed"] = result["horizon_months"] == best
    return result


def _oos_metrics(current: pd.DataFrame) -> dict:
    """Out-of-sample quality of each agent's score vs. the realized forward alpha at this snapshot.

    The snapshot being scored is out-of-sample for the walk-forward model (it is never trained on
    its own labels). We report Spearman rank-IC and RMSE of each agent's prediction against the
    realized `target_future_alpha`. If the 12m future is not yet observable for these rows the
    labels are NaN and metrics are left empty (n_oos = 0).
    """
    metrics: dict = {}
    realized = pd.to_numeric(current.get("target_future_alpha"), errors="coerce")
    valid = realized.notna()
    n_oos = int(valid.sum())
    metrics["n_oos"] = n_oos
    for probability in COMPONENT_TARGETS:
        rank_key = f"rank_ic_{probability}"
        rmse_key = f"rmse_{probability}"
        if n_oos < 5 or probability not in current.columns:
            metrics[rank_key] = ""
            metrics[rmse_key] = ""
            continue
        pred = pd.to_numeric(current[probability], errors="coerce")
        pair = pd.DataFrame({"pred": pred, "real": realized})[valid].dropna()
        if len(pair) < 5 or pair["pred"].nunique() < 2:
            metrics[rank_key] = ""
            metrics[rmse_key] = ""
            continue
        metrics[rank_key] = float(pair["pred"].corr(pair["real"], method="spearman"))
        # RMSE against a min-max normalized realized alpha keeps predictions and labels on [0,1].
        real_norm = pair["real"]
        span = real_norm.max() - real_norm.min()
        real_norm = (real_norm - real_norm.min()) / span if span else pd.Series(0.5, index=pair.index)
        metrics[rmse_key] = float(np.sqrt(((pair["pred"] - real_norm) ** 2).mean()))
    return metrics


def _fallback_probability(df: pd.DataFrame, target: str) -> pd.Series:
    if target in df.columns:
        return df[target].fillna(df["garp_score"]).clip(0, 1)
    return df["garp_score"].fillna(0.5).clip(0, 1)


def _forward_return(price_cache: dict[str, tuple[list[pd.Timestamp], list[float]]], ticker: str, start: pd.Timestamp, months: int) -> float | None:
    series = price_cache.get(ticker)
    if series is None:
        return None
    dates, closes = series
    start_index = bisect_right(dates, start) - 1
    end_index = bisect_right(dates, start + pd.DateOffset(months=months)) - 1
    if start_index < 0 or end_index < 0:
        return None
    start_price = float(closes[start_index])
    end_price = float(closes[end_index])
    if start_price <= 0:
        return None
    return end_price / start_price - 1


def _feature_cache(df: pd.DataFrame, columns: list[str]) -> dict[str, tuple[list[pd.Timestamp], dict[str, list[float]]]]:
    cache = {}
    for ticker, group in df.sort_values(["ticker", "snapshot_date_dt"]).groupby("ticker", sort=False):
        cache[ticker] = (
            group["snapshot_date_dt"].tolist(),
            {column: group[column].astype(float).tolist() for column in columns},
        )
    return cache


def _forward_feature_value(
    cache: dict[str, tuple[list[pd.Timestamp], dict[str, list[float]]]],
    ticker: str,
    start: pd.Timestamp,
    column: str,
    months: int,
) -> float | None:
    series = cache.get(ticker)
    if series is None:
        return None
    dates, values_by_column = series
    values = values_by_column[column]
    target_date = start + pd.DateOffset(months=months)
    index = bisect_left(dates, target_date)
    if index >= len(dates):
        return None
    value = values[index]
    return None if pd.isna(value) else float(value)


def _fit_component_models(df: pd.DataFrame):
    """Fit the three specialist agents, each on its own feature subset."""
    models = {}
    importances = {}
    for probability, target in COMPONENT_TARGETS.items():
        features = _agent_features(probability)
        model, importance = _fit_model(df, target, features)
        models[probability] = model
        importances[probability] = importance
    return models, importances


# Semilla de los modelos (LightGBM y el fallback RandomForest). Se centraliza aquí, en vez de
# repetir el literal 42, para que el runner de experimentos pueda variarla y medir la dispersión
# del resultado entre semillas (estabilidad). El valor por defecto es el de siempre.
RANDOM_STATE = 42


# Small, walk-forward-safe hyperparameter grid: searched only against the SAME chronological
# 70/30 in-window holdout the meta-agent already uses (see _select_hyperparameters), never against
# future data. Kept small (7 combos) because it reruns every quarterly retrain for 3 agents.
LGBM_PARAM_GRID = [
    {"n_estimators": n, "learning_rate": lr, "num_leaves": nl, "min_child_samples": mc}
    for n, lr, nl, mc in [
        (80, 0.05, 31, 20),
        (120, 0.05, 31, 20),
        (80, 0.03, 15, 20),
        (150, 0.03, 15, 30),
        (80, 0.05, 15, 30),
        (120, 0.08, 31, 15),
        (200, 0.03, 31, 30),  # deeper/slower combo, viable now selection rewards ranking not RMSE
    ]
]
_HPARAM_MIN_VALIDATION_ROWS = 25


def _fit_model(df: pd.DataFrame, target: str, features: list[str]):
    x = df[features].fillna(0.5)
    y = df[target].fillna(df["garp_score"])
    try:
        from lightgbm import LGBMRegressor

        params = _select_hyperparameters(df, target, features)
        model = LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1, **params)
        model.fit(x, y)
        importance = dict(zip(features, model.feature_importances_.tolist()))
        return model, importance
    except Exception as exc:
        log.warning("LightGBM unavailable or failed (%s). Falling back to sklearn.", exc)
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=80, random_state=RANDOM_STATE, min_samples_leaf=2, n_jobs=-1)
        model.fit(x, y)
        importance = dict(zip(features, model.feature_importances_.tolist()))
        return model, importance


def _select_hyperparameters(df: pd.DataFrame, target: str, features: list[str]) -> dict:
    """Pick LightGBM hyperparameters for this agent/snapshot by the Spearman rank-IC of the
    validation prediction against `target_future_alpha` — the exact metric the whole system is
    scored on — measured on a chronological 70/30 in-window holdout. The agent still fits on its own
    `target`; only the *capacity selection* is aligned with the alpha-ranking objective (RMSE against
    a rank/garp-filled target does not imply good ranking). Same chronological discipline as the
    meta-agent's split, so this never touches future data. Falls back to a fixed default when the
    window is too thin, or when the validation alpha has no rank variance to correlate against.
    """
    from lightgbm import LGBMRegressor

    ordered = df.sort_values("snapshot_date_dt") if "snapshot_date_dt" in df.columns else df
    n = len(ordered)
    split = int(n * 0.7)
    if split < 30 or n - split < _HPARAM_MIN_VALIDATION_ROWS:
        return LGBM_PARAM_GRID[0]
    x = ordered[features].fillna(0.5)
    y = ordered[target].fillna(ordered["garp_score"])
    x_fit, x_val = x.iloc[:split], x.iloc[split:]
    y_fit = y.iloc[:split]
    # Selection metric: rank the validation predictions against the realized 12m alpha. If the
    # window lacks a usable alpha column (or it is constant on the holdout), fall back to RMSE.
    alpha_val = pd.to_numeric(ordered.get("target_future_alpha"), errors="coerce")
    alpha_val = alpha_val.iloc[split:] if alpha_val is not None else None
    use_ic = alpha_val is not None and alpha_val.notna().sum() >= 10 and alpha_val.dropna().nunique() > 1

    best_params, best_score = LGBM_PARAM_GRID[0], -float("inf")
    y_val_rmse = y.iloc[split:].to_numpy()
    for params in LGBM_PARAM_GRID:
        model = LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1, **params)
        model.fit(x_fit, y_fit)
        pred = pd.Series(model.predict(x_val), index=x_val.index)
        if use_ic:
            pair = pd.DataFrame({"pred": pred.to_numpy(), "alpha": alpha_val.to_numpy()}).dropna()
            ic = pair["pred"].corr(pair["alpha"], method="spearman") if len(pair) >= 10 else float("nan")
            score = ic if np.isfinite(ic) else -float("inf")
        else:
            score = -float(np.sqrt(((pred.to_numpy() - y_val_rmse) ** 2).mean()))  # higher = better
        if score > best_score:
            best_score, best_params = score, params
    return best_params


def _predict(model, x: pd.DataFrame) -> pd.Series:
    raw = pd.Series(model.predict(x.fillna(0.5)), index=x.index)
    if raw.max() == raw.min():
        return pd.Series(0.5, index=x.index)
    return (raw - raw.min()) / (raw.max() - raw.min())


def _opportunity_type(row: pd.Series) -> str:
    quality = row["quality_score"]
    moat = row["moat_score"]
    growth = row["growth_score"]
    valuation = row["valuation_score"]
    adjusted_valuation = row.get("price_adjusted_valuation_score", valuation)
    catalyst = row["catalyst_score"]
    risk = row["risk_score"]

    momentum = row.get("momentum_score", 0.5)

    if quality < 0.35 or (risk < 0.20 and quality < 0.50):
        return "Avoid"
    if momentum < 0.18 and adjusted_valuation < 0.55:
        return "Avoid"
    if valuation >= 0.80 and catalyst < 0.45 and quality < 0.55:
        return "Value Trap"
    if quality >= 0.60 and moat >= 0.55 and growth >= 0.55 and adjusted_valuation >= 0.35:
        return "Quality Growth Reasonable"
    if quality >= 0.80 and moat >= 0.75 and growth >= 0.50:
        return "Compounder" if adjusted_valuation >= 0.35 else "Fully Valued Compounder"
    if growth >= 0.75 and adjusted_valuation >= 0.55:
        return "Growth Undervalued"
    if adjusted_valuation >= 0.70 and catalyst >= 0.55 and quality >= 0.45:
        return "Value with Catalyst"
    if catalyst >= 0.70 and growth >= 0.45 and quality >= 0.40:
        return "Turnaround"
    if adjusted_valuation >= 0.65 and risk >= 0.45:
        return "Cyclical Opportunity"
    if adjusted_valuation >= 0.85:
        return "Deep Value"
    if growth >= 0.70 and adjusted_valuation < 0.35:
        return "Expensive Growth"
    if quality >= 0.50 and growth >= 0.35 and adjusted_valuation >= 0.45:
        return "Fully Valued Compounder" if adjusted_valuation < 0.55 else "Quality Growth Reasonable"
    return "Avoid"
