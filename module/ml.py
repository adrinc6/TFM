"""Train and score the GARP model: four disjoint specialist agents + a learned meta-agent.

The four agents are genuine GARP experts, each fit on its OWN, non-overlapping feature subset
(AGENT_FEATURES) against its OWN forward-looking target (COMPONENT_TARGETS):

    quality      — durable business quality (forward change in reported ROIC)
    improvement  — growth trajectory (forward observed growth vs. today's market expectation)
    mispricing   — valuation (whether today's real discount resolved into forward alpha)
    timing       — entry/momentum (short-horizon forward excess return)

There is deliberately NO generalist "alpha" agent that sees every feature and predicts the same
`target_future_alpha` the meta-agent scores against — that agent structurally dominated the blend
(same target, superset of features) and was removed. `target_future_alpha` (realized 12m forward
excess return) survives only as the *evaluation* label: what the meta-agent learns its weights
against and what OOS rank-IC is measured on — never as an agent that predicts it.

Every target is observable only `its own horizon` ahead (AGENT_HORIZON_MONTHS_OVERRIDE), and is
leakage-masked identically during walk-forward training (see _walk_forward_component_scores). The
meta-agent (`_meta_agent_scores` / `_fit_meta_weights`) then learns how to combine the four agents
by each agent's *marginal* ranking contribution (partial rank-IC against the alpha left unexplained
by the other three), so an agent is rewarded for information the others don't already carry, not for
redundancy. Per-snapshot OOS rank-IC / RMSE of each agent and of the combined `final_score` is
written to model_walk_forward_diagnostics.csv so model quality is auditable.
"""

from __future__ import annotations

import logging
from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd

from environment import PROCESSED_DIR, RAW_DIR, Settings
from module.utils import read_parquet, write_json, write_parquet

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
    "expected_growth",
    "implied_growth",
    "realized_growth",
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
    "improvement_probability": "target_improvement",
    "mispricing_probability": "target_mispricing",
    "timing_probability": "target_timing",
}

# Per-agent label horizon override (months). Each GARP dimension resolves on its own clock: a
# valuation discount plausibly closes faster than a 12-month fundamental-quality change, and an
# entry-timing edge is by nature short-term — so `mispricing` and `timing` get their own shorter
# horizons while `quality`/`improvement` keep the default `walk_forward_label_horizon_months`. Each
# is masked separately in the walk-forward loop under the same no-lookahead rule, just a different
# offset.
AGENT_HORIZON_MONTHS_OVERRIDE = {
    "mispricing_probability": 6,
    "timing_probability": 3,
}

# Each specialist agent sees ONLY the features relevant to its GARP dimension, and the four subsets
# are mutually DISJOINT — no feature is shared across agents. This is what makes them genuine
# complements (distinct feature importances, distinct targets) rather than four views of the same
# signal, and it is what lets the meta-agent's partial-IC weighting reward marginal contribution.
AGENT_FEATURES = {
    # Durable business quality: profitability, moat, balance-sheet risk and their multi-year trends.
    "quality_probability": [
        "quality_score", "moat_score", "risk_score",
        "quality_score_vs_sector", "quality_score_vs_universe",
        "quality_trend_1y", "quality_trend_2y", "roic_trend", "margin_trend", "fcf_trend", "moat_trend",
    ],
    # Growth trajectory: level of growth, market-implied expectation, and growth acceleration.
    "improvement_probability": [
        "growth_score", "expected_growth", "growth_acceleration", "growth_deceleration",
    ],
    # Valuation: the undervalued-quality gap and cross-sectional cheapness vs. sector/universe.
    "mispricing_probability": [
        "quality_value_gap", "valuation_score_vs_sector",
        "positive_expectation_gap", "valuation_score_vs_universe",
    ],
    # Entry/momentum: price momentum, earnings-surprise catalyst, short-window price returns and how
    # stale the last fundamental is (a fresh print re-times the entry). Deliberately excludes every
    # feature already owned by the other three agents (e.g. growth_acceleration → improvement).
    "timing_probability": [
        "momentum_score", "catalyst_score",
        "price_return_1m", "price_return_3m", "price_return_since_fundamental",
        "stale_fundamental_months",
    ],
}


def _agent_features(probability: str) -> list[str]:
    return [feature for feature in AGENT_FEATURES.get(probability, MODEL_FEATURES) if feature in MODEL_FEATURES]


_TRAIN_FREQUENCY_ALIASES = {"M": "ME", "Q": "QE"}


def _train_and_apply_dates(all_dates: list[pd.Timestamp], settings: Settings) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """Split all snapshot dates into (train_dates, apply_dates) for the train-until-cutoff-then-freeze
    scheme: train_dates are the subset at or before `train_cutoff_date`, downsampled to
    `walk_forward_train_frequency` (quarterly by default) and starting no earlier than
    `train_cutoff_date - walk_forward_train_years`; apply_dates is every date (all of them receive a
    prediction, from either a training-phase model or the frozen model after the cutoff).
    """
    cutoff = pd.Timestamp(settings.train_cutoff_date)
    train_start = cutoff - pd.DateOffset(years=settings.walk_forward_train_years)
    freq = _TRAIN_FREQUENCY_ALIASES.get(settings.walk_forward_train_frequency, settings.walk_forward_train_frequency)
    quarter_ends = set(pd.date_range(train_start, cutoff, freq=freq))
    train_dates = sorted(d for d in all_dates if train_start <= d <= cutoff and d in quarter_ends)
    if not train_dates or train_dates[-1] != cutoff:
        # Ensure the cutoff itself is always a training/freeze point, even if it doesn't fall
        # exactly on a quarter-end in the dataset's monthly snapshot calendar.
        eligible = [d for d in all_dates if train_start <= d <= cutoff]
        if eligible:
            train_dates = sorted(set(train_dates) | {eligible[-1]})
    apply_dates = sorted(all_dates)
    return train_dates, apply_dates


def train_and_score(settings: Settings) -> pd.DataFrame:
    features = read_parquet(PROCESSED_DIR / "features.parquet")
    prices = read_parquet(RAW_DIR / "prices.parquet")
    log.info(
        "Training/scoring universe rows=%s snapshots=%s tickers=%s features=%s",
        len(features),
        features["snapshot_date"].nunique(),
        features["ticker"].nunique(),
        len(MODEL_FEATURES),
    )
    labeled = _add_component_targets(features, prices, settings.benchmark_ticker, settings.walk_forward_label_horizon_months)
    if settings.walk_forward_scoring:
        labeled, importance, diagnostics = _walk_forward_component_scores(labeled, settings)
    else:
        models, importance = _fit_component_models(labeled)
        for probability, model in models.items():
            labeled[probability] = _predict(model, labeled[_agent_features(probability)])
        diagnostics = pd.DataFrame([{"mode": "full_sample", "training_rows": len(labeled)}])
    # Meta-agent (the global decision-maker): instead of hard-coded prior weights, it LEARNS,
    # walk-forward, how much to trust each specialist agent from each agent's marginal ranking
    # contribution to realized forward alpha. Falls back to the fixed GARP prior when a snapshot has
    # too little observable history. Weights + partial ICs are persisted for the "learning" view.
    labeled, meta_weights = _meta_agent_scores(labeled, settings)
    # The master signal is now the meta-agent output `final_score` (there is no alpha agent). Its
    # per-snapshot OOS rank-IC vs. realized forward alpha — plus a rolling and per-year trend — is
    # appended to the diagnostics here, once final_score exists.
    diagnostics = _master_signal_diagnostics(diagnostics, labeled)
    settings.run_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(settings.run_dir / "model_walk_forward_diagnostics.csv", index=False)
    write_parquet(meta_weights, PROCESSED_DIR / "meta_weights_by_snapshot.parquet")
    meta_weights.to_csv(settings.run_dir / "meta_weights_by_snapshot.csv", index=False)
    horizon_comparison = _label_horizon_comparison(labeled, prices, settings)
    horizon_comparison.to_csv(settings.run_dir / "label_horizon_comparison.csv", index=False)
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
                "current snapshot. All four component targets (quality, improvement, mispricing, "
                "timing) are genuine forward-looking labels observed at current_date + that agent's "
                "own horizon; any label not yet observable at the training cutoff is masked and "
                "replaced by the deterministic GARP fallback for that component."
            ),
            "component_target_definitions": {
                "quality_probability": "Forward change in reported ROIC over the label horizon (fundamental quality improving).",
                "improvement_probability": "Forward observed fundamental growth vs. today's market-implied expectation.",
                "mispricing_probability": "Whether today's real valuation discount resolved into forward alpha (cheap-and-won / expensive-and-lost).",
                "timing_probability": "Short-horizon forward excess return — an entry/momentum timing signal.",
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
                    "alpha left unexplained by the other three agents (measured out-of-sample on a "
                    "30% chronological hold-out inside the training window), clipped at 0 and "
                    "normalized to sum to 1. Falls back to the prior when history is thin or no agent "
                    "adds marginal ranking power."
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
    """Build the master evaluation label plus all four agent targets, each from information
    observable ONLY `its own horizon` ahead.

    `target_future_alpha` (realized forward excess return over the default horizon) is the master
    evaluation label — the meta-agent learns against it and OOS rank-IC is measured on it. The four
    agent targets each use their own horizon (AGENT_HORIZON_MONTHS_OVERRIDE): quality/improvement at
    the default 12m, mispricing at 6m, timing at 3m. Each is masked separately in the walk-forward
    loop under the same no-lookahead rule, just a different offset.
    """
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    price_cache = _price_cache(prices)
    df = df.copy()
    df["snapshot_date_dt"] = pd.to_datetime(df["snapshot_date"])
    # roic is a hard reported fundamental; realized_growth is now the observed fundamental-growth
    # percentile (see features/transforms.py) — neither is a re-projection of the input scores.
    forward_columns = [col for col in ("roic", "realized_growth") if col in df.columns]
    feature_cache = _feature_cache(df, forward_columns)

    mispricing_horizon = AGENT_HORIZON_MONTHS_OVERRIDE.get("mispricing_probability", horizon_months)
    timing_horizon = AGENT_HORIZON_MONTHS_OVERRIDE.get("timing_probability", horizon_months)

    def forward_alpha(ticker: str, start: pd.Timestamp, months: int) -> float | None:
        stock_ret = _forward_return(price_cache, ticker, start, months=months)
        bench_ret = _forward_return(price_cache, benchmark_ticker, start, months=months)
        return None if stock_ret is None or bench_ret is None else stock_ret - bench_ret

    alphas = []
    quality_targets = []
    improvement_targets = []
    mispricing_targets = []
    timing_targets = []
    log.info(
        "Building forward-looking labels rows=%s horizon_months=%s mispricing_horizon=%s timing_horizon=%s",
        len(df), horizon_months, mispricing_horizon, timing_horizon,
    )
    for index, row in enumerate(df.itertuples(index=False), start=1):
        alpha = forward_alpha(row.ticker, row.snapshot_date_dt, horizon_months)
        alphas.append(alpha)

        # Each agent that overrides the horizon needs its own forward alpha at that horizon.
        mispricing_alpha = alpha if mispricing_horizon == horizon_months else forward_alpha(row.ticker, row.snapshot_date_dt, mispricing_horizon)
        timing_alpha = alpha if timing_horizon == horizon_months else forward_alpha(row.ticker, row.snapshot_date_dt, timing_horizon)

        # target_quality: forward improvement in the ACTUALLY REPORTED ROIC (fundamental quality),
        # not the derived quality_score. NA when roic unavailable at either end.
        current_roic = getattr(row, "roic", None) if "roic" in forward_columns else None
        fwd_roic = _forward_feature_value(feature_cache, row.ticker, row.snapshot_date_dt, "roic", horizon_months) if "roic" in forward_columns else None
        if fwd_roic is None or current_roic is None or pd.isna(current_roic):
            quality_targets.append(None)
        else:
            quality_targets.append(float(min(1.0, max(0.0, (fwd_roic - float(current_roic) + 1) / 2))))

        # target_improvement: forward observed fundamental growth vs. today's market expectation.
        fwd_realized_growth = (
            _forward_feature_value(feature_cache, row.ticker, row.snapshot_date_dt, "realized_growth", horizon_months)
            if "realized_growth" in forward_columns else None
        )
        improvement_targets.append(
            None if fwd_realized_growth is None else float(min(1.0, max(0.0, (fwd_realized_growth - row.expected_growth + 1) / 2)))
        )

        # target_mispricing: does today's real valuation discount resolve into forward alpha over
        # its own (shorter) horizon? Cheap-and-outperformed => 1; expensive-and-underperformed => 1;
        # conflicting => ~0.5. Uses the actual valuation_score, not the derived expectation_gap.
        if mispricing_alpha is None:
            mispricing_targets.append(None)
        else:
            discount = float(row.valuation_score)  # high = cheap
            outperformed = 1.0 if mispricing_alpha > 0 else 0.0
            mispricing_targets.append(discount * outperformed + (1 - discount) * (1 - outperformed))

        # target_timing: short-horizon (3m) forward excess return, squashed to [0,1]. High = the
        # stock outran the benchmark shortly after this snapshot — the entry was well timed.
        timing_targets.append(None if timing_alpha is None else float(min(1.0, max(0.0, (timing_alpha + 1) / 2))))

        if index % 10000 == 0:
            log.info("Forward-looking labels progress rows=%s/%s", index, len(df))

    df["target_future_alpha"] = alphas
    df["target_quality"] = quality_targets
    df["target_improvement"] = improvement_targets
    df["target_mispricing"] = mispricing_targets
    df["target_timing"] = timing_targets
    trainable = df["target_future_alpha"].notna()
    if trainable.sum() < 5:
        log.warning("Few future labels available; model will lean on deterministic GARP score.")
        df["target_future_alpha"] = df["garp_score"]
    return df.drop(columns=["snapshot_date_dt"])


def _walk_forward_component_scores(df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Train-until-cutoff-then-freeze: relearn the 4 agents only on train_dates (quarterly, up to
    and including `train_cutoff_date`); every apply_date > cutoff is scored with the LAST trained
    model (no further learning), exactly like a model trained once and deployed. Dates within the
    training window that fall between training snapshots (e.g. non-quarter months before the
    cutoff) are also scored with the latest trained-so-far model rather than retrained individually
    — this is the same "apply without retraining" behavior used after the cutoff, just still
    advancing quarter to quarter until the cutoff is reached.
    """
    scored = df.copy()
    scored["snapshot_date_dt"] = pd.to_datetime(scored["snapshot_date"])
    for probability, target in COMPONENT_TARGETS.items():
        scored[probability] = _fallback_probability(scored, target)

    all_dates = sorted(scored["snapshot_date_dt"].dropna().unique())
    train_dates, apply_dates = _train_and_apply_dates(all_dates, settings)
    train_dates_set = set(train_dates)
    cutoff = pd.Timestamp(settings.train_cutoff_date)
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
        "Walk-forward train-until-cutoff snapshots=%s train_dates=%s (quarterly, <= %s) components=%s",
        len(apply_dates), len(train_dates), cutoff.date().isoformat(), len(COMPONENT_TARGETS),
    )
    models: dict = {}
    last_training_snapshot: pd.Timestamp | None = None
    last_training_stats: dict = {}
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
                models, latest_importance = _fit_component_models(train)
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

        phase = "training" if pd.Timestamp(date) <= cutoff else "frozen"
        if models:
            for probability, model in models.items():
                features = _agent_features(probability)
                scored.loc[current_mask, probability] = _predict(model, scored.loc[current_mask, features]).values
            oos_metrics = _oos_metrics(scored.loc[current_mask])
            diagnostics.append({
                "snapshot_date": pd.Timestamp(date).date().isoformat(),
                "mode": "walk_forward_model",
                "phase": phase,
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
                "phase": phase,
                "is_train_date": bool(is_train_date),
                "training_snapshot_date": "",
                "fallback_reason": "not_enough_years" if is_train_date else "before_first_training_snapshot",
                "uses_rows_through_current_date": True,
                "min_training_years": settings.min_walk_forward_training_years,
                "max_training_years": settings.max_walk_forward_training_years,
                "label_horizon_months": settings.walk_forward_label_horizon_months,
            })
    return scored.drop(columns=["snapshot_date_dt"]), latest_importance, pd.DataFrame(diagnostics)


def _master_signal_diagnostics(diagnostics: pd.DataFrame, labeled: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Append the OOS quality of the MASTER signal (`final_score`, the meta-agent output) to the
    per-snapshot diagnostics: its Spearman rank-IC vs. realized forward alpha, a rolling mean, and a
    per-calendar-year mean + approximate t-stat (mean / (std / sqrt(n))).

    This replaces the old per-agent "alpha" IC column: with no generalist alpha agent, the signal
    whose quality-over-time actually matters is the combined meta-agent score. With ~90 walk-forward
    snapshots the rolling/year trend is descriptive, not a formal significance test — reported as-is,
    without forcing a "the system is improving" narrative if the data doesn't show one.
    """
    diagnostics = diagnostics.copy()
    if diagnostics.empty or "snapshot_date" not in diagnostics.columns or "final_score" not in labeled.columns:
        diagnostics["rank_ic_final"] = pd.NA
        diagnostics["rank_ic_final_rolling"] = pd.NA
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

    diagnostics["rank_ic_final"] = diagnostics["snapshot_date"].astype(str).map(ic_by_snapshot)
    ic = pd.to_numeric(diagnostics["rank_ic_final"], errors="coerce")
    diagnostics["rank_ic_final_rolling"] = ic.rolling(window=window, min_periods=max(3, window // 2)).mean()
    years = pd.to_datetime(diagnostics["snapshot_date"], errors="coerce").dt.year
    year_stats = ic.groupby(years).agg(["mean", "std", "count"])
    year_stats["t_stat"] = year_stats["mean"] / (year_stats["std"] / year_stats["count"].pow(0.5))
    diagnostics["rank_ic_final_year_mean"] = years.map(year_stats["mean"].to_dict())
    diagnostics["rank_ic_final_year_tstat"] = years.map(year_stats["t_stat"].to_dict())
    return diagnostics


# Prior (the fixed GARP combination) — used as the meta-agent fallback and as the baseline the
# learned weights are compared against. No alpha agent: the four experts are quality/improvement/
# mispricing/timing.
AGENT_PRIOR_WEIGHTS = {
    "quality_probability": 0.30,
    "improvement_probability": 0.25,
    "mispricing_probability": 0.25,
    "timing_probability": 0.20,
}
AGENT_KEYS = list(AGENT_PRIOR_WEIGHTS.keys())


def _meta_agent_scores(labeled: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Learn, per training snapshot (quarterly, up to `train_cutoff_date`), how to weight the four
    agents against the realized forward alpha — then FREEZE those weights for every snapshot after
    the cutoff (train-until-cutoff-then-freeze, same scheme as _walk_forward_component_scores).

    Each training snapshot fits ONLY on rows whose forward alpha was already observable at that
    date, using non-negative least squares so weights are >=0 and sum to 1 — interpretable and hard
    to overfit with 4 params. Snapshots without enough observable history fall back to the fixed
    prior weights. After the cutoff, apply_dates reuse the last learned (or prior) weights without
    refitting — this is where "learning" stops and "frozen simulation" begins.
    """
    labeled = labeled.copy()
    labeled["snapshot_date_dt"] = pd.to_datetime(labeled["snapshot_date"])
    labeled["final_score"] = _prior_combination(labeled)
    labeled["meta_weight_source"] = "prior"

    if not settings.walk_forward_scoring:
        labeled["ml_score"] = labeled["final_score"]
        rows = [{
            "snapshot_date": "all", "source": "prior", "phase": "training", "n_train": 0,
            **AGENT_PRIOR_WEIGHTS, **{f"partial_ic_{key}": float("nan") for key in AGENT_KEYS},
        }]
        return labeled.drop(columns=["snapshot_date_dt"]), pd.DataFrame(rows)

    all_dates = sorted(labeled["snapshot_date_dt"].dropna().unique())
    train_dates, apply_dates = _train_and_apply_dates(all_dates, settings)
    train_dates_set = set(train_dates)
    cutoff = pd.Timestamp(settings.train_cutoff_date)
    label_offset = pd.DateOffset(months=settings.walk_forward_label_horizon_months)
    max_history = pd.DateOffset(years=settings.max_walk_forward_training_years)
    weight_rows = []
    weights = dict(AGENT_PRIOR_WEIGHTS)
    partial_ics = {key: float("nan") for key in AGENT_KEYS}
    source = "prior"
    last_training_snapshot: pd.Timestamp | None = None
    for date in apply_dates:
        if date in train_dates_set:
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
            "phase": "training" if pd.Timestamp(date) <= cutoff else "frozen",
            "training_snapshot_date": last_training_snapshot.date().isoformat() if last_training_snapshot is not None else "",
            "n_train": int(len(train)),
            **{key: float(weights[key]) for key in AGENT_KEYS},
            **{f"partial_ic_{key}": float(partial_ics[key]) for key in AGENT_KEYS},
        })
    labeled["ml_score"] = labeled["final_score"]
    return labeled.drop(columns=["snapshot_date_dt"]), pd.DataFrame(weight_rows)


def _prior_combination(df: pd.DataFrame) -> pd.Series:
    return sum(AGENT_PRIOR_WEIGHTS[key] * df[key] for key in AGENT_KEYS).clip(0, 1)


META_FIT_MIN_ROWS = 80
META_FIT_MIN_VALIDATION_ROWS = 30
META_FIT_VALIDATION_FRACTION = 0.30


def _fit_meta_weights(train: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]] | None:
    """Weight each agent by its MARGINAL ranking contribution, not its raw rank-IC.

    Two design choices, both to reward complementarity over redundancy:

    1. A chronological 70/30 split INSIDE the training window (fit vs. validation), same walk-forward
       discipline as the outer train/apply split — no lookahead, and an agent must generalize within
       the window to earn weight, not just fit it.

    2. For each agent, we regress the realized forward alpha on the OTHER three agents (OLS on the
       fit part), take the residual on the validation part — the alpha the other three left
       unexplained — and score the agent by its Spearman rank-IC against THAT residual (partial
       rank-IC). An agent that merely echoes the others' ranking scores ~0; an agent that ranks the
       leftover alpha earns weight. Weights are the clipped-at-0 partial ICs, normalized to sum 1.

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

    partial_ics: dict[str, float] = {}
    for position, key in enumerate(AGENT_KEYS):
        others = [i for i in range(len(AGENT_KEYS)) if i != position]
        design_fit = np.column_stack([scores[fit_slice][:, others], np.ones(split)])
        coefficients, *_ = np.linalg.lstsq(design_fit, y[fit_slice], rcond=None)
        design_val = np.column_stack([scores[val_slice][:, others], np.ones(len(t) - split)])
        residual = y[val_slice] - design_val @ coefficients
        agent_val = scores[val_slice][:, position]
        pair = pd.DataFrame({"agent": agent_val, "residual": residual}).dropna()
        if len(pair) < META_FIT_MIN_VALIDATION_ROWS or pair["agent"].nunique() < 2:
            partial_ics[key] = 0.0
            continue
        ic = pair["agent"].corr(pair["residual"], method="spearman")
        partial_ics[key] = max(0.0, float(ic)) if np.isfinite(ic) else 0.0

    total = sum(partial_ics.values())
    if total <= 0:
        return None
    weights = {key: partial_ics[key] / total for key in AGENT_KEYS}
    return weights, partial_ics


LABEL_HORIZON_CANDIDATES_MONTHS = [3, 6, 12]


def _label_horizon_comparison(scored: pd.DataFrame, prices: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Does the master signal predict better at 3, 6, or 12 months out?

    Reuses the ALREADY-COMBINED meta-agent score `final_score` and checks its Spearman rank-IC
    against the realized forward alpha computed at each candidate horizon, per training-phase
    snapshot (the frozen phase always uses the same model, so it adds no new information about which
    horizon is best). This answers "does the strategy's edge live at a shorter or longer horizon?"
    without retraining 3x — an evidence-based check on the default 12m label horizon.
    """
    if "final_score" not in scored.columns:
        return pd.DataFrame()
    cutoff = pd.Timestamp(settings.train_cutoff_date)
    scored = scored.copy()
    scored["snapshot_date_dt"] = pd.to_datetime(scored["snapshot_date"])
    training_rows = scored[scored["snapshot_date_dt"] <= cutoff].copy()
    if training_rows.empty:
        training_rows = scored.copy()
    price_cache = _price_cache(prices.assign(date=pd.to_datetime(prices["date"])))

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


def _price_cache(prices: pd.DataFrame) -> dict[str, tuple[list[pd.Timestamp], list[float]]]:
    cache = {}
    for ticker, group in prices.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        cache[ticker] = (
            group["date"].tolist(),
            group["adj_close"].astype(float).tolist(),
        )
    return cache


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
    """Fit the four specialist agents, each on its own feature subset."""
    models = {}
    importances = {}
    for probability, target in COMPONENT_TARGETS.items():
        features = _agent_features(probability)
        model, importance = _fit_model(df, target, features)
        models[probability] = model
        importances[probability] = importance
    return models, importances


def _fit_model(df: pd.DataFrame, target: str, features: list[str]):
    x = df[features].fillna(0.5)
    y = df[target].fillna(df["garp_score"])
    try:
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(n_estimators=80, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=1)
        model.fit(x, y)
        importance = dict(zip(features, model.feature_importances_.tolist()))
        return model, importance
    except Exception as exc:
        log.warning("LightGBM unavailable or failed (%s). Falling back to sklearn.", exc)
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=80, random_state=42, min_samples_leaf=2, n_jobs=1)
        model.fit(x, y)
        importance = dict(zip(features, model.feature_importances_.tolist()))
        return model, importance


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
