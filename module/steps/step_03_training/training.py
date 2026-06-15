"""Training workflows for base agents and the meta learner."""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from environment import (
    RISK_BEAR_HARD_THRESHOLD,
    SCORE_DISPERSION_MIN_SCALE,
    OOF_N_SPLITS,
    SCORE_DISPERSION_MIN_STD,
    SECTOR_CONFIDENCE_PEERS,
    SECTOR_SCORE_PRIOR_BASE,
    SECTOR_SCORE_PRIOR_WEIGHT,
    EXPORT_FEATURE_USAGE_REPORT,
    QUALITY_FEATURE_COLUMNS, QUALITY_FEATURE_EXCLUDE,
    GROWTH_FEATURE_COLUMNS, GROWTH_FEATURE_EXCLUDE,
    GARP_VALUATION_FEATURE_COLUMNS, GARP_VALUATION_FEATURE_EXCLUDE,
    FUNDAMENTAL_TREND_FEATURE_COLUMNS, FUNDAMENTAL_TREND_FEATURE_EXCLUDE,
    CATALYST_FEATURE_COLUMNS, CATALYST_FEATURE_EXCLUDE,
    RISK_BEAR_FEATURE_COLUMNS, RISK_BEAR_FEATURE_EXCLUDE,
    TECHNICAL_GUARDRAIL_FEATURE_COLUMNS, TECHNICAL_GUARDRAIL_FEATURE_EXCLUDE,
    SENTIMENT_FEATURE_COLUMNS, SENTIMENT_FEATURE_EXCLUDE,
    SECTOR_ROTATION_FEATURE_COLUMNS, SECTOR_ROTATION_FEATURE_EXCLUDE,
    META_FEATURE_COLUMNS, META_FEATURE_EXCLUDE,
    ENABLE_RECENCY_WEIGHTING, TRAINING_RECENCY_HALFLIFE_YEARS,
    ENABLE_TEMPORAL_AGENT_VALIDATION,
    TEMPORAL_VALIDATION_TOP_PCT,
    TEMPORAL_VALIDATION_MIN_TOP_K,
    TEMPORAL_VALIDATION_HALFLIFE_QUARTERS,
    TEMPORAL_VALIDATION_TREND_WEIGHT,
    TEMPORAL_VALIDATION_WEIGHT_CLIP_MIN,
    TEMPORAL_VALIDATION_WEIGHT_CLIP_MAX,
    RULE_QUALITY_GATE_ENABLED,
    RULE_QUALITY_MIN_IC,
    RULE_QUALITY_MIN_SPREAD,
    RULE_QUALITY_MIN_STABILITY,
    RULE_QUALITY_REF_IC,
    RULE_QUALITY_REF_SPREAD,
    NEUTRAL_SCORE_PENALTY_ENABLED,
    NEUTRAL_SCORE_VALUE,
    NEUTRAL_SCORE_PENALIZED_VALUE,
    NEUTRAL_SCORE_EPS,
)
from module.agents.alpha_meta_learner import AlphaMetaLearner
from module.common.recency_weights import compute_recency_weights
from module.common.garp_validation import validate_no_forward_features, validate_critical_garp_features
from module.common.regime import MarketRegimeModel, apply_regime_weighting
from module.steps.step_03_training.agent_diagnostics import (
    apply_diversity_multipliers,
    build_research_actions,
    compute_agent_behavior_features,
    compute_agent_redundancy,
    compute_rule_quality,
    export_fold_diagnostics,
)
from module.steps.step_03_training.agent_config import build_agents_config, build_sector_rotation_agent
from module.steps.step_03_training.oof import generate_oof_scores

log = logging.getLogger(__name__)


_OBSOLETE_UNUSED_COLUMNS = {
    "sector_specialized_score",
    "meta_score",
    "obsolete_meta_score",
    "obsolete_fundamental_score",
    "obsolete_valuation_score",
    "obsolete_momentum_score",
    "obsolete_bear_score",
    "obsolete_sentiment_score",
}


def _requested_feature_map() -> Dict[str, Dict[str, list[str]]]:
    return {
        "quality": {"include": list(QUALITY_FEATURE_COLUMNS), "exclude": list(QUALITY_FEATURE_EXCLUDE)},
        "growth": {"include": list(GROWTH_FEATURE_COLUMNS), "exclude": list(GROWTH_FEATURE_EXCLUDE)},
        "valuation": {"include": list(GARP_VALUATION_FEATURE_COLUMNS), "exclude": list(GARP_VALUATION_FEATURE_EXCLUDE)},
        "fundamental_trend": {"include": list(FUNDAMENTAL_TREND_FEATURE_COLUMNS), "exclude": list(FUNDAMENTAL_TREND_FEATURE_EXCLUDE)},
        "catalyst": {"include": list(CATALYST_FEATURE_COLUMNS), "exclude": list(CATALYST_FEATURE_EXCLUDE)},
        "risk_bear": {"include": list(RISK_BEAR_FEATURE_COLUMNS), "exclude": list(RISK_BEAR_FEATURE_EXCLUDE)},
        "technical_guardrail": {"include": list(TECHNICAL_GUARDRAIL_FEATURE_COLUMNS), "exclude": list(TECHNICAL_GUARDRAIL_FEATURE_EXCLUDE)},
        "sentiment": {"include": list(SENTIMENT_FEATURE_COLUMNS), "exclude": list(SENTIMENT_FEATURE_EXCLUDE)},
        "sector_rotation": {"include": list(SECTOR_ROTATION_FEATURE_COLUMNS), "exclude": list(SECTOR_ROTATION_FEATURE_EXCLUDE)},
        "meta_learner": {"include": list(META_FEATURE_COLUMNS), "exclude": list(META_FEATURE_EXCLUDE)},
    }


def _drop_obsolete_unused_columns(df: pd.DataFrame, *, owner: str) -> pd.DataFrame:
    """Drop obsolete columns if present before model training."""
    if df is None or df.empty:
        return df
    drop_cols = [c for c in _OBSOLETE_UNUSED_COLUMNS if c in df.columns]
    if not drop_cols:
        return df
    out = df.drop(columns=drop_cols, errors="ignore")
    log.info("[%s] Dropped obsolete unused columns: %s", owner, ", ".join(sorted(drop_cols)))
    return out


def _export_feature_usage_report(
    *,
    agents: Dict[str, Any],
    df_train: pd.DataFrame,
    fold_id: int | str,
    agents_results_dir: str,
) -> None:
    if not EXPORT_FEATURE_USAGE_REPORT:
        return

    requested = _requested_feature_map()
    records: list[Dict[str, Any]] = []

    for agent_name, cfg in requested.items():
        include = list(cfg.get("include", []))
        exclude = list(cfg.get("exclude", []))
        include_set = set(include)
        available = [c for c in include if c in df_train.columns]
        missing = [c for c in include if c not in df_train.columns]

        # Columnas presentes pero no calculables en este fold (todo NaN)
        not_calculated_or_no_data: list[str] = []
        low_data_coverage: list[str] = []
        for col in available:
            s = pd.to_numeric(df_train[col], errors="coerce")
            non_null = int(s.notna().sum())
            if non_null == 0:
                not_calculated_or_no_data.append(col)
            elif non_null < max(5, int(0.05 * len(s))):
                low_data_coverage.append(col)

        agent_obj = agents.get(agent_name)
        used = list(getattr(agent_obj, "_feature_cols", []) or []) if agent_obj is not None else []
        used_not_requested = [c for c in used if c not in include_set]

        skipped_due_to_data = [c for c in include if c not in used and c in set(not_calculated_or_no_data + low_data_coverage)]

        records.append(
            {
                "fold": str(fold_id),
                "agent": agent_name,
                "requested_n": len(include),
                "available_n": len(available),
                "used_n": len(used),
                "missing_n": len(missing),
                "not_calculated_or_no_data_n": len(not_calculated_or_no_data),
                "low_data_coverage_n": len(low_data_coverage),
                "skipped_due_to_data_n": len(skipped_due_to_data),
                "used_not_requested_n": len(used_not_requested),
                "requested": "|".join(include),
                "exclude": "|".join(exclude),
                "available": "|".join(available),
                "missing": "|".join(missing),
                "not_calculated_or_no_data": "|".join(not_calculated_or_no_data),
                "low_data_coverage": "|".join(low_data_coverage),
                "skipped_due_to_data": "|".join(skipped_due_to_data),
                "used": "|".join(used),
                "used_not_requested": "|".join(used_not_requested),
            }
        )

    out_dir = Path(agents_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "feature_usage_report.csv"
    json_path = out_dir / "feature_usage_report.json"

    pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("[FeatureUsage] Reporte fold %s -> %s", fold_id, csv_path.name)


def _series_stats(s: pd.Series) -> Dict[str, float]:
    v = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return {
            "n": 0,
            "min": float("nan"),
            "q10": float("nan"),
            "q25": float("nan"),
            "mean": float("nan"),
            "q50": float("nan"),
            "q75": float("nan"),
            "q90": float("nan"),
            "max": float("nan"),
            "ge_050_ratio": float("nan"),
            "ge_055_ratio": float("nan"),
            "ge_060_ratio": float("nan"),
        }
    return {
        "n": int(v.shape[0]),
        "min": float(v.min()),
        "q10": float(v.quantile(0.10)),
        "q25": float(v.quantile(0.25)),
        "mean": float(v.mean()),
        "q50": float(v.quantile(0.50)),
        "q75": float(v.quantile(0.75)),
        "q90": float(v.quantile(0.90)),
        "max": float(v.max()),
        "ge_050_ratio": float((v >= 0.50).mean()),
        "ge_055_ratio": float((v >= 0.55).mean()),
        "ge_060_ratio": float((v >= 0.60).mean()),
    }


def _log_score_stats(tag: str, s: pd.Series) -> None:
    st = _series_stats(s)
    if st["n"] == 0:
        log.info(f"[{tag}] No data para resumen de score")
        return
    log.info(
        f"[{tag}] n={st['n']} | min={st['min']:.4f} q10={st['q10']:.4f} q25={st['q25']:.4f} "
        f"mean={st['mean']:.4f} q50={st['q50']:.4f} q75={st['q75']:.4f} q90={st['q90']:.4f} max={st['max']:.4f}"
    )
    log.info(
        f"[{tag}] cobertura umbral: >=0.50={st['ge_050_ratio']:.1%} | >=0.55={st['ge_055_ratio']:.1%} | >=0.60={st['ge_060_ratio']:.1%}"
    )


def _build_rule_quality_multipliers(rule_quality: pd.DataFrame) -> Dict[str, float]:
    """Build per-agent rule multipliers from OOF rule quality diagnostics.

    Multiplier range is [0, 1]. Values close to 0 mute noisy/inverted rule
    telemetry; values near 1 keep high-quality rule signals.
    """
    if not RULE_QUALITY_GATE_ENABLED or rule_quality is None or rule_quality.empty:
        return {}

    out: Dict[str, float] = {}
    for row in rule_quality.itertuples(index=False):
        rule_col = str(getattr(row, "rule_col", ""))
        if not rule_col.endswith("_rule_signal"):
            continue
        agent_name = rule_col[: -len("_rule_signal")]
        if not agent_name:
            continue

        n = float(pd.to_numeric(getattr(row, "n", 0.0), errors="coerce"))
        ic = float(pd.to_numeric(getattr(row, "spearman_ic", np.nan), errors="coerce"))
        spread = float(pd.to_numeric(getattr(row, "top_bottom_spread", np.nan), errors="coerce"))
        stability = float(pd.to_numeric(getattr(row, "stability", np.nan), errors="coerce"))

        if (not np.isfinite(ic)) or (not np.isfinite(spread)) or n < 30:
            out[agent_name] = 0.0
            continue

        # Hard gate for clearly weak/inverted rule quality.
        if ic <= float(RULE_QUALITY_MIN_IC) or spread <= float(RULE_QUALITY_MIN_SPREAD):
            out[agent_name] = 0.0
            continue

        ic_denom = max(float(RULE_QUALITY_REF_IC) - float(RULE_QUALITY_MIN_IC), 1e-6)
        ic_score = float(np.clip((ic - float(RULE_QUALITY_MIN_IC)) / ic_denom, 0.0, 1.0))

        spread_denom = max(float(RULE_QUALITY_REF_SPREAD) - float(RULE_QUALITY_MIN_SPREAD), 1e-6)
        spread_score = float(np.clip((spread - float(RULE_QUALITY_MIN_SPREAD)) / spread_denom, 0.0, 1.0))

        if np.isfinite(stability):
            stab_denom = max(1.0 - float(RULE_QUALITY_MIN_STABILITY), 1e-6)
            stability_score = float(np.clip((stability - float(RULE_QUALITY_MIN_STABILITY)) / stab_denom, 0.0, 1.0))
        else:
            stability_score = 0.5

        # Mild sample-size attenuation for low-support diagnostics.
        sample_score = float(np.clip(np.sqrt(min(n, 400.0) / 400.0), 0.25, 1.0))

        quality = (0.55 * ic_score) + (0.30 * spread_score) + (0.15 * stability_score)
        out[agent_name] = float(np.clip(quality * sample_score, 0.0, 1.0))

    return out


def _apply_rule_quality_multipliers(df: pd.DataFrame, multipliers: Dict[str, float]) -> pd.DataFrame:
    """Apply per-agent OOF quality multipliers to rule telemetry columns."""
    if df.empty or not multipliers:
        return df

    out = df.copy()
    for agent_name, raw_mult in multipliers.items():
        m = float(np.clip(raw_mult, 0.0, 1.0))
        sig_col = f"{agent_name}_rule_signal"
        conf_col = f"{agent_name}_rule_confidence"

        if sig_col in out.columns:
            sig = pd.to_numeric(out[sig_col], errors="coerce").fillna(0.0)
            out[sig_col] = (sig * m).clip(-1.0, 1.0)
        if conf_col in out.columns:
            conf = pd.to_numeric(out[conf_col], errors="coerce").fillna(0.0)
            out[conf_col] = (conf * m).clip(0.0, 1.0)

        out[f"{agent_name}_rule_quality_multiplier"] = m

    rule_sig_cols = [c for c in out.columns if c.endswith("_rule_signal")]
    if rule_sig_cols:
        out["rules_consensus_signal"] = out[rule_sig_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).mean(axis=1)
    rule_conf_cols = [c for c in out.columns if c.endswith("_rule_confidence")]
    if rule_conf_cols:
        out["rules_consensus_confidence"] = out[rule_conf_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).mean(axis=1)

    return out


def _compute_dispersion_scales(df: pd.DataFrame, score_cols: list[str]) -> Dict[str, float]:
    scales: Dict[str, float] = {}
    log.info(f"[Dispersion] SCORE_DISPERSION_MIN_STD={SCORE_DISPERSION_MIN_STD:.4f}")
    log.info(f"[Dispersion] SCORE_DISPERSION_MIN_SCALE={SCORE_DISPERSION_MIN_SCALE:.4f}")
    for col in score_cols:
        if col not in df.columns:
            continue
        std = float(df[col].std()) if SCORE_DISPERSION_MIN_STD > 0 else 0.0
        scale = 1.0 if std >= SCORE_DISPERSION_MIN_STD else (std / SCORE_DISPERSION_MIN_STD)
        bounded = max(0.0, min(1.0, scale))
        # Guardrail: prevents a useful score from becoming completely neutral during inference.
        if bounded < 1.0:
            bounded = max(bounded, float(SCORE_DISPERSION_MIN_SCALE))
        scales[col] = bounded
        log.info(
            f"[Dispersion] {col}: std={std:.6f} -> scale={scales[col]:.4f} "
            f"({'sin shrink' if scales[col] >= 1.0 else 'shrink activo'})"
        )
    return scales


def _apply_dispersion_shrink(df: pd.DataFrame, scales: Dict[str, float]) -> pd.DataFrame:
    for col, scale in scales.items():
        if col not in df.columns or scale >= 1.0:
            continue
        before = df[col].copy()
        df[col] = 0.5 + (df[col] - 0.5) * scale
        _log_score_stats(f"Dispersion/{col}/before", before)
        _log_score_stats(f"Dispersion/{col}/after", df[col])
    return df


def _apply_neutral_penalty(
    df: pd.DataFrame,
    score_cols: list[str],
    *,
    neutral_value: float,
    penalized_value: float,
    eps: float,
    exclude_cols: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Penalize exact neutral scores so unknown/missing evidence does not rank up."""
    if df.empty or not score_cols:
        return df

    out = df.copy()
    excluded = set(exclude_cols or set())
    neutral = float(neutral_value)
    penalized = float(np.clip(penalized_value, 0.0, 1.0))
    tol = max(float(eps), 0.0)

    for col in score_cols:
        if col in excluded or col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        neutral_mask = s.notna() & (np.abs(s - neutral) <= tol)
        if bool(neutral_mask.any()):
            out.loc[neutral_mask, col] = penalized
            log.info(
                "[NeutralPenalty] %s: penalized %d neutral scores %.3f -> %.3f",
                col,
                int(neutral_mask.sum()),
                neutral,
                penalized,
            )
    return out


def _apply_sector_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    if "sector" not in df.columns or "final_score" not in df.columns:
        return df
    sector_counts = (
        df.reset_index()[["ticker", "sector"]]
        .drop_duplicates(subset="ticker")
        .groupby("sector")["ticker"]
        .nunique()
    )
    df["sector_peer_count"] = df["sector"].map(sector_counts).fillna(0).astype(int)
    if SECTOR_CONFIDENCE_PEERS > 0:
        df["sector_confidence"] = np.sqrt(df["sector_peer_count"] / SECTOR_CONFIDENCE_PEERS).clip(upper=1.0)
    else:
        df["sector_confidence"] = 1.0
    if "sector_score" in df.columns:
        sector_score = df["sector_score"].fillna(0.5)
    else:
        sector_score = 0.5
    # Additive sector tilt: shifts scores up/down based on sector outlook.
    # sector_score > 0.5 → positive tilt (strong sector), < 0.5 → negative tilt.
    # The tilt magnitude is controlled by SECTOR_SCORE_PRIOR_WEIGHT and
    # dampened by sector_confidence (fewer peers → weaker tilt).
    df["final_score_raw"] = df["final_score"]
    sector_tilt = (sector_score - 0.5) * SECTOR_SCORE_PRIOR_WEIGHT * df["sector_confidence"]
    df["final_score"] = (df["final_score"] + sector_tilt).clip(0.0, 1.0)

    log.info(
        f"[SectorAdjust] params: SECTOR_CONFIDENCE_PEERS={SECTOR_CONFIDENCE_PEERS}, "
        f"SECTOR_SCORE_PRIOR_BASE={SECTOR_SCORE_PRIOR_BASE:.3f}, "
        f"SECTOR_SCORE_PRIOR_WEIGHT={SECTOR_SCORE_PRIOR_WEIGHT:.3f}"
    )
    _log_score_stats("SectorAdjust/final_score_raw", df["final_score_raw"])
    _log_score_stats("SectorAdjust/sector_score", sector_score)
    _log_score_stats("SectorAdjust/sector_confidence", df["sector_confidence"])
    _log_score_stats("SectorAdjust/sector_tilt", sector_tilt)
    _log_score_stats("SectorAdjust/final_score_adjusted", df["final_score"])

    if "bear_risk_score" in df.columns:
        bear_risk = pd.to_numeric(df["bear_risk_score"], errors="coerce").fillna(0.5)
        log.info(
            f"[RiskDiag] bear_risk>=hard_threshold({RISK_BEAR_HARD_THRESHOLD:.2f}): "
            f"{int((bear_risk >= RISK_BEAR_HARD_THRESHOLD).sum())}/{len(bear_risk)} ({(bear_risk >= RISK_BEAR_HARD_THRESHOLD).mean():.1%})"
        )

    return df


def _compute_recency_weights(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Delegate to the shared recency weight utility.

    Returns per-observation exponential recency weights or ``None`` when
    disabled.  See :func:`module.common.recency_weights.compute_recency_weights`.
    """
    return compute_recency_weights(
        df,
        enabled=ENABLE_RECENCY_WEIGHTING,
        halflife_years=float(TRAINING_RECENCY_HALFLIFE_YEARS),
    )


def _instantiate_base_agents(agents_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        ag_name: cfg["cls"](**cfg["kwargs"])
        for ag_name, cfg in agents_config.items()
    }


def _temporal_quarter_labels(df: pd.DataFrame) -> pd.Series:
    if "year_quarter" in df.columns:
        q = pd.PeriodIndex(df["year_quarter"].astype(str), freq="Q")
        return pd.Series(q.astype(str), index=df.index)
    if isinstance(df.index, pd.MultiIndex) and "date" in df.index.names:
        dt = pd.to_datetime(df.index.get_level_values("date"), errors="coerce")
    else:
        dt = pd.to_datetime(df.index, errors="coerce")
    q = pd.PeriodIndex(dt, freq="Q")
    return pd.Series(q.astype(str), index=df.index)


def _compute_temporal_agent_validation(
    *,
    df_train: pd.DataFrame,
    df_scores: pd.DataFrame,
    y_train: pd.Series,
    score_cols: list[str],
) -> pd.DataFrame:
    if not bool(ENABLE_TEMPORAL_AGENT_VALIDATION):
        return pd.DataFrame()

    valid_cols = [c for c in score_cols if c in df_scores.columns]
    if not valid_cols:
        return pd.DataFrame()

    y = pd.to_numeric(y_train.reindex(df_scores.index), errors="coerce")
    up_global = float(y.mean()) if y.notna().any() else 0.5
    if "tp_sl_outcome" in df_train.columns:
        tp_base = (
            df_train["tp_sl_outcome"]
            .astype(str)
            .str.upper()
            .eq("TP")
            .astype(float)
            .reindex(df_scores.index)
        )
    else:
        tp_base = y.copy()
    tp_global = float(tp_base.mean()) if tp_base.notna().any() else up_global

    q_labels = _temporal_quarter_labels(df_train).reindex(df_scores.index)
    q_sorted = sorted(q_labels.dropna().unique().tolist())
    if not q_sorted:
        return pd.DataFrame()
    q_rank = {q: i for i, q in enumerate(q_sorted)}
    max_rank = max(q_rank.values())
    halflife_q = max(float(TEMPORAL_VALIDATION_HALFLIFE_QUARTERS), 1.0)

    rows: list[Dict[str, Any]] = []
    top_pct = float(np.clip(TEMPORAL_VALIDATION_TOP_PCT, 0.01, 0.50))
    min_top_k = max(int(TEMPORAL_VALIDATION_MIN_TOP_K), 3)

    for score_col in valid_cols:
        s = pd.to_numeric(df_scores[score_col], errors="coerce")
        quarter_stats: list[Dict[str, Any]] = []
        for q in q_sorted:
            q_idx = q_labels.index[q_labels == q]
            s_q = s.reindex(q_idx).dropna()
            if s_q.empty:
                continue
            n_q = int(len(s_q))
            k_q = max(min_top_k, int(np.ceil(top_pct * n_q)))
            k_q = min(k_q, n_q)
            top_idx = s_q.nlargest(k_q).index
            up_q = pd.to_numeric(y.reindex(top_idx), errors="coerce").dropna()
            tp_q = pd.to_numeric(tp_base.reindex(top_idx), errors="coerce").dropna()
            up_hit = float(up_q.mean()) if not up_q.empty else 0.5
            tp_hit = float(tp_q.mean()) if not tp_q.empty else up_hit
            combo = 0.55 * up_hit + 0.45 * tp_hit

            age_q = float(max_rank - q_rank[q])
            rec_w = float(0.5 ** (age_q / halflife_q))
            quarter_stats.append(
                {
                    "quarter": q,
                    "n_total": n_q,
                    "k_top": k_q,
                    "up_hit": up_hit,
                    "tp_hit": tp_hit,
                    "combo": combo,
                    "recency_w": rec_w,
                }
            )

        if not quarter_stats:
            rows.append(
                {
                    "score_col": score_col,
                    "quarters_evaluated": 0,
                    "weighted_up_hit_rate": 0.5,
                    "weighted_tp_hit_rate": 0.5,
                    "weighted_combo": 0.5,
                    "recent_combo": 0.5,
                    "trend_slope": 0.0,
                    "reliability_score": 0.5,
                    "reliability_multiplier": 1.0,
                    "global_up_hit_rate": up_global,
                    "global_tp_hit_rate": tp_global,
                }
            )
            continue

        qs = pd.DataFrame(quarter_stats)
        w = pd.to_numeric(qs["recency_w"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if not np.isfinite(w).any() or float(np.sum(w)) <= 0.0:
            w = np.ones(len(qs), dtype=float)
        w = w / float(np.sum(w))

        up_w = float(np.average(qs["up_hit"].to_numpy(dtype=float), weights=w))
        tp_w = float(np.average(qs["tp_hit"].to_numpy(dtype=float), weights=w))
        combo_w = float(np.average(qs["combo"].to_numpy(dtype=float), weights=w))
        recent_combo = float(qs.iloc[-1]["combo"])

        trend = 0.0
        if len(qs) >= 3:
            x = np.arange(len(qs), dtype=float)
            yq = qs["combo"].to_numpy(dtype=float)
            try:
                trend = float(np.polyfit(x, yq, 1)[0])
            except Exception:
                trend = 0.0

        rel_base = 0.40 * combo_w + 0.30 * up_w + 0.30 * tp_w
        rel_delta = rel_base - 0.5
        trend_adj = float(TEMPORAL_VALIDATION_TREND_WEIGHT) * trend
        reliability_score = float(np.clip(0.5 + rel_delta + trend_adj, 0.0, 1.0))

        coverage = float(min(1.0, len(qs) / 6.0))
        mult_raw = 1.0 + 0.9 * (reliability_score - 0.5)
        mult_raw += 0.6 * trend_adj
        multiplier = 1.0 + (mult_raw - 1.0) * coverage
        multiplier = float(
            np.clip(
                multiplier,
                float(TEMPORAL_VALIDATION_WEIGHT_CLIP_MIN),
                float(TEMPORAL_VALIDATION_WEIGHT_CLIP_MAX),
            )
        )

        rows.append(
            {
                "score_col": score_col,
                "quarters_evaluated": int(len(qs)),
                "weighted_up_hit_rate": up_w,
                "weighted_tp_hit_rate": tp_w,
                "weighted_combo": combo_w,
                "recent_combo": recent_combo,
                "trend_slope": trend,
                "reliability_score": reliability_score,
                "reliability_multiplier": multiplier,
                "global_up_hit_rate": up_global,
                "global_tp_hit_rate": tp_global,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("score_col").reset_index(drop=True)


def _apply_temporal_agent_reliability(df: pd.DataFrame, validation_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or validation_df is None or validation_df.empty:
        return df
    out = df.copy()
    for _, row in validation_df.iterrows():
        score_col = str(row.get("score_col", ""))
        if score_col not in out.columns:
            continue
        mult = float(pd.to_numeric(row.get("reliability_multiplier"), errors="coerce"))
        if not np.isfinite(mult):
            continue
        out[score_col] = (0.5 + (pd.to_numeric(out[score_col], errors="coerce").fillna(0.5) - 0.5) * mult).clip(0.0, 1.0)
        agent_name = score_col.replace("_score", "")
        out[f"{agent_name}_temporal_multiplier"] = mult
        out[f"{agent_name}_temporal_reliability"] = float(pd.to_numeric(row.get("reliability_score"), errors="coerce"))
        out[f"{agent_name}_temporal_trend"] = float(pd.to_numeric(row.get("trend_slope"), errors="coerce"))
    return out


def _export_temporal_agent_validation(validation_df: pd.DataFrame, output_dir: str, fold_id: int | str) -> None:
    if validation_df is None or validation_df.empty:
        return
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "agent_temporal_validation.csv"
    json_path = out_dir / "agent_temporal_validation.json"
    validation_df.to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(validation_df.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8")
    log.info("[AgentTemporal] Fold %s validation -> %s", fold_id, csv_path.name)


def _get_agent_fallback_score(agent: Any, cfg: Dict[str, Any]) -> float:
    try:
        return float(getattr(agent, "_neutral_score", (cfg.get("kwargs") or {}).get("neutral_score", 0.5)))
    except Exception:
        return 0.5


def _fit_base_agents(
    agents: Dict[str, Any],
    agents_config: Dict[str, Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    fold: int,
    sample_weight: Optional[np.ndarray] = None,
) -> None:
    for ag_name, agent in agents.items():
        cfg = agents_config[ag_name]
        y_fit = (1 - y) if cfg.get("invert_y") else y
        sector_col = cfg.get("sector_col")
        fit_kwargs: Dict[str, Any] = {"fold": fold}
        if sector_col:
            fit_kwargs["sector_col"] = sector_col
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        try:
            agent.fit(X, y_fit, **fit_kwargs)
        except TypeError:
            # Fallback for agents whose fit() does not accept sample_weight
            fit_kwargs_basic = {k: v for k, v in fit_kwargs.items() if k != "sample_weight"}
            agent.fit(X, y_fit, **fit_kwargs_basic)


def _predict_base_scores(
    agents: Dict[str, Any],
    agents_config: Dict[str, Dict[str, Any]],
    X: pd.DataFrame,
) -> pd.DataFrame:
    out = X.copy()
    for ag_name, agent in agents.items():
        sector_col = agents_config[ag_name].get("sector_col")
        fallback_score = _get_agent_fallback_score(agent, agents_config[ag_name])
        if not getattr(agent, "is_trained", False):
            scores = pd.Series(fallback_score, index=out.index)
        elif sector_col:
            # Always score from the original feature frame to avoid
            # cross-agent contamination and keep base models independent.
            scores = agent.predict_score(X, sector_col)
        else:
            scores = agent.predict_score(X)
        scores = pd.to_numeric(pd.Series(scores, index=out.index), errors="coerce").fillna(fallback_score)
        # Align score direction for investment: high = better to invest.
        # BearAgent devuelve riesgo [0,1], por eso guardamos ambas vistas:
        #   - bear_risk_score: riesgo (alto = peor)
        #   - bear_score: safety (alto = mejor)
        if ag_name in {"bear", "risk_bear"}:
            risk = scores.astype(float).clip(0.0, 1.0)
            risk_col = "risk_bear_risk_score" if ag_name == "risk_bear" else "bear_risk_score"
            safety_col = "risk_bear_score" if ag_name == "risk_bear" else "bear_score"
            out[risk_col] = risk.values
            out[safety_col] = (1.0 - risk).values
            _log_score_stats(f"AgentScore/{ag_name}/risk", out[risk_col])
            _log_score_stats(f"AgentScore/{ag_name}/safety", out[safety_col])
        else:
            out[f"{ag_name}_score"] = scores.values
            _log_score_stats(f"AgentScore/{ag_name}", out[f"{ag_name}_score"])

        # Optional explainability channel: if the agent exposes a rule engine,
        # persist its latest rule-derived diagnostics as fold features.
        if hasattr(agent, "get_last_rule_details"):
            try:
                details = agent.get_last_rule_details()
                rs = pd.to_numeric(pd.Series(details.get("rule_signal"), index=out.index), errors="coerce").fillna(0.0)
                rh = pd.to_numeric(pd.Series(details.get("rule_hits"), index=out.index), errors="coerce").fillna(0.0)
                rc = pd.to_numeric(pd.Series(details.get("rule_confidence"), index=out.index), errors="coerce").fillna(0.0)
                out[f"{ag_name}_rule_signal"] = rs.values
                out[f"{ag_name}_rule_hits"] = rh.values
                out[f"{ag_name}_rule_confidence"] = rc.values
            except Exception as exc:
                log.debug("[RuleDiag/%s] Could not export rule details: %s", ag_name, exc)

    score_cols = [c for c in [f"{name}_score" for name in agents_config.keys()] + ["sector_score"] if c in out.columns]
    if score_cols:
        ensemble_mean = out[score_cols].mean(axis=1)
        _log_score_stats("AgentScore/ensemble_mean_pre_meta", ensemble_mean)

    rule_sig_cols = [c for c in out.columns if c.endswith("_rule_signal")]
    if rule_sig_cols:
        out["rules_consensus_signal"] = out[rule_sig_cols].mean(axis=1)
    rule_conf_cols = [c for c in out.columns if c.endswith("_rule_confidence")]
    if rule_conf_cols:
        out["rules_consensus_confidence"] = out[rule_conf_cols].mean(axis=1)
    return out


def train_fold(
    df_train_norm: pd.DataFrame,
    df_test_norm: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_alpha_train: Optional[pd.Series],
    target_alpha_test: Optional[pd.Series],
    fold_id: int,
    agent_models_results_dir: str,
    agents_results_dir: str,
    random_seed: int = 42,
    sector_map: Optional[Dict[str, str]] = None,
    spy_prices: Optional[pd.Series] = None,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    df_train_norm = _drop_obsolete_unused_columns(df_train_norm, owner=f"Fold {fold_id}/train")
    df_test_norm = _drop_obsolete_unused_columns(df_test_norm, owner=f"Fold {fold_id}/test")

    agents_config = build_agents_config(agent_models_results_dir=agent_models_results_dir, random_seed=random_seed)
    requested_model_features = []
    for _cfg in agents_config.values():
        requested_model_features.extend((_cfg.get("kwargs") or {}).get("include_features", []))
    validate_no_forward_features(requested_model_features, context=f"fold {fold_id}/agent_config")
    validate_critical_garp_features(df_train_norm, context=f"fold {fold_id}/train")
    validate_critical_garp_features(df_test_norm, context=f"fold {fold_id}/test")
    # ── Step 1: Train base agents ────────────────────────────────────────────
    log.info(f"[Fold {fold_id}] 1/3 — Entrenando agentes base con datos de entrenamiento del fold...")
    base_agents = _instantiate_base_agents(agents_config)
    recency_weights = _compute_recency_weights(df_train_norm)
    if recency_weights is not None:
        log.info(
            f"[Fold {fold_id}] Recency weighting enabled "
            f"(halflife={TRAINING_RECENCY_HALFLIFE_YEARS}Y, "
            f"weight_range=[{recency_weights.min():.3f}, {recency_weights.max():.3f}])"
        )
    _fit_base_agents(base_agents, agents_config, df_train_norm, y_train, fold=fold_id,
                     sample_weight=recency_weights)

    # Entrenar SectorRotationAgent (opera a nivel sector, no ticker)
    sector_agent = build_sector_rotation_agent(agent_models_results_dir=agent_models_results_dir, random_seed=random_seed)
    if sector_map is not None and "forward_return" in df_train_norm.columns:
        log.info(f"[Fold {fold_id}] 1/3 — Entrenando SectorRotationAgent (top-down, nivel sector)...")
        sector_agent.fit(df_train_norm, sector_map=sector_map, spy_prices=spy_prices, fold=fold_id)
    else:
        log.info(f"[Fold {fold_id}] SectorRotationAgent sin datos suficientes — score neutro 0.5")

    # ── Step 2: OOF scores ───────────────────────────────────────────────────
    log.info(f"[Fold {fold_id}] 2/3 — Generando scores OOF ({OOF_N_SPLITS} splits temporales) como input del MetaLearner...")
    oof_scores = generate_oof_scores(
        df_train_norm,
        y_train,
        agents_config=agents_config,
        n_splits=OOF_N_SPLITS,
        random_seed=random_seed,
        sector_map=sector_map,
        spy_prices=spy_prices,
    )
    df_train_with_oof = df_train_norm.copy()
    for col_name, scores_series in oof_scores.items():
        df_train_with_oof[col_name] = scores_series

    if "risk_bear_score" in df_train_with_oof.columns:
        bear_risk = df_train_with_oof["risk_bear_score"].astype(float).clip(0.0, 1.0)
        df_train_with_oof["risk_bear_risk_score"] = bear_risk
        df_train_with_oof["risk_bear_score"] = 1.0 - bear_risk
        _log_score_stats("OOF/risk_bear_risk", df_train_with_oof["risk_bear_risk_score"])
        _log_score_stats("OOF/risk_bear_safety", df_train_with_oof["risk_bear_score"])

    if "sector_score" not in df_train_with_oof.columns:
        df_train_with_oof["sector_score"] = 0.5
        log.info("[Fold %s] sector_score OOF not available — using neutral 0.5 for meta training.", fold_id)
    else:
        _log_score_stats("OOF/sector_score", df_train_with_oof["sector_score"])

    if bool(NEUTRAL_SCORE_PENALTY_ENABLED):
        neutral_penalty_cols = [
            c for c in [f"{ag_name}_score" for ag_name in agents_config.keys()] if c in df_train_with_oof.columns
        ]
        if "risk_bear_score" in df_train_with_oof.columns and "risk_bear_score" not in neutral_penalty_cols:
            neutral_penalty_cols.append("risk_bear_score")
        df_train_with_oof = _apply_neutral_penalty(
            df_train_with_oof,
            neutral_penalty_cols,
            neutral_value=float(NEUTRAL_SCORE_VALUE),
            penalized_value=float(NEUTRAL_SCORE_PENALIZED_VALUE),
            eps=float(NEUTRAL_SCORE_EPS),
            exclude_cols={"sector_score"},
        )

    temporal_score_cols = [f"{ag_name}_score" for ag_name in agents_config.keys() if f"{ag_name}_score" in df_train_with_oof.columns]
    temporal_validation = _compute_temporal_agent_validation(
        df_train=df_train_norm,
        df_scores=df_train_with_oof,
        y_train=y_train,
        score_cols=temporal_score_cols,
    )
    if not temporal_validation.empty:
        _export_temporal_agent_validation(temporal_validation, agents_results_dir, fold_id)
        for _, row in temporal_validation.iterrows():
            log.info(
                "[AgentTemporal] %s mult=%.3f rel=%.3f up=%.2f%% tp=%.2f%% trend=%+.4f q=%d",
                str(row.get("score_col", "")).replace("_score", ""),
                float(row.get("reliability_multiplier", 1.0)),
                float(row.get("reliability_score", 0.5)),
                100.0 * float(row.get("weighted_up_hit_rate", 0.5)),
                100.0 * float(row.get("weighted_tp_hit_rate", 0.5)),
                float(row.get("trend_slope", 0.0)),
                int(row.get("quarters_evaluated", 0)),
            )
        df_train_with_oof = _apply_temporal_agent_reliability(df_train_with_oof, temporal_validation)

    diag_score_cols = [
        c for c in [f"{ag_name}_score" for ag_name in agents_config.keys()] if c in df_train_with_oof.columns
    ]
    if "sector_score" in df_train_with_oof.columns:
        diag_score_cols.append("sector_score")

    redundancy_diag = compute_agent_redundancy(df_train_with_oof, diag_score_cols)
    diversity_multipliers = dict(redundancy_diag.get("multipliers", {}) or {})
    if diversity_multipliers:
        for col_name, mult in sorted(diversity_multipliers.items()):
            log.info("[Diversity] %s multiplier=%.3f", col_name, float(mult))
    df_train_with_oof = apply_diversity_multipliers(df_train_with_oof, diversity_multipliers)

    behavior_train = compute_agent_behavior_features(df_train_with_oof, diag_score_cols)
    for col in behavior_train.columns:
        df_train_with_oof[col] = behavior_train[col].reindex(df_train_with_oof.index).values

    rule_quality = compute_rule_quality(df_train_with_oof, y_train)
    rule_quality_multipliers = _build_rule_quality_multipliers(rule_quality)
    if rule_quality_multipliers:
        for ag_name, mult in sorted(rule_quality_multipliers.items()):
            log.info("[RuleQuality] %s multiplier=%.3f", ag_name, float(mult))
        df_train_with_oof = _apply_rule_quality_multipliers(df_train_with_oof, rule_quality_multipliers)

    research_actions = build_research_actions(
        redundancy=redundancy_diag,
        rule_quality=rule_quality,
        behavior_features=behavior_train,
    )
    export_fold_diagnostics(
        output_dir=agents_results_dir,
        fold_id=fold_id,
        redundancy=redundancy_diag,
        rule_quality=rule_quality,
        behavior_features=behavior_train,
        actions=research_actions,
    )

    score_cols = [c for c in diag_score_cols if c in df_train_with_oof.columns]
    dispersion_scales = _compute_dispersion_scales(df_train_with_oof, score_cols)
    df_train_with_oof = _apply_dispersion_shrink(df_train_with_oof, dispersion_scales)

    # ── Step 3: MetaLearner ──────────────────────────────────────────────────
    meta = AlphaMetaLearner(results_dir=agent_models_results_dir, random_seed=random_seed)
    log.info(f"[Fold {fold_id}] 2/3 — Entrenando MetaLearner sobre scores OOF (anti-leakage)...")
    regime_model = MarketRegimeModel()
    regime_model.fit(df_train_with_oof)
    df_train_with_oof["regime_state"] = regime_model.predict(df_train_with_oof)
    df_train_with_oof, _ = apply_regime_weighting(df_train_with_oof, regime_col="regime_state")

    meta.fit(
        df_train_with_oof,
        y_train,
        fold=fold_id,
        sector_col="sector",
        target_alpha=target_alpha_train,
    )

    # ── Step 4: Test predictions ─────────────────────────────────────────────
    log.info(f"[Fold {fold_id}] 3/3 — Generando predicciones sobre el quarter de test ({len(df_test_norm)} tickers)...")
    df_test = _predict_base_scores(base_agents, agents_config, df_test_norm)

    if sector_agent.is_trained and sector_map is not None:
        sector_scores_test = sector_agent.predict_sector_scores(df_test, sector_map)
        df_test["sector_score"] = sector_agent.map_to_tickers(
            df_test, sector_map, sector_scores_test
        ).values
    else:
        df_test["sector_score"] = 0.5

    if bool(NEUTRAL_SCORE_PENALTY_ENABLED):
        neutral_penalty_cols_test = [
            c for c in [f"{ag_name}_score" for ag_name in agents_config.keys()] if c in df_test.columns
        ]
        if "risk_bear_score" in df_test.columns and "risk_bear_score" not in neutral_penalty_cols_test:
            neutral_penalty_cols_test.append("risk_bear_score")
        df_test = _apply_neutral_penalty(
            df_test,
            neutral_penalty_cols_test,
            neutral_value=float(NEUTRAL_SCORE_VALUE),
            penalized_value=float(NEUTRAL_SCORE_PENALIZED_VALUE),
            eps=float(NEUTRAL_SCORE_EPS),
            exclude_cols={"sector_score"},
        )

    if rule_quality_multipliers:
        df_test = _apply_rule_quality_multipliers(df_test, rule_quality_multipliers)

    df_test = apply_diversity_multipliers(df_test, diversity_multipliers)
    behavior_test = compute_agent_behavior_features(df_test, diag_score_cols)
    for col in behavior_test.columns:
        df_test[col] = behavior_test[col].reindex(df_test.index).values

    df_test["regime_state"] = regime_model.predict(df_test)
    df_test, _ = apply_regime_weighting(df_test, regime_col="regime_state")

    if not temporal_validation.empty:
        df_test = _apply_temporal_agent_reliability(df_test, temporal_validation)

    df_test = _apply_dispersion_shrink(df_test, dispersion_scales)
    components = meta.predict_components(df_test, "sector")
    # Avoid duplicate column names (e.g., regime_adjusted_score) that would
    # make df_test[col] return a 2D frame and break downstream assignments.
    for col in components.columns:
        df_test[col] = components[col].reindex(df_test.index).values
    df_test["final_score"] = pd.to_numeric(df_test["regime_adjusted_score"], errors="coerce").fillna(0.5).values
    _log_score_stats("Meta/final_score_pre_sector_adjust", df_test["final_score"])
    df_test = _apply_sector_adjustments(df_test)
    df_test["label"] = y_test.values
    if target_alpha_test is not None:
        df_test["target_alpha"] = pd.to_numeric(target_alpha_test.reindex(df_test.index), errors="coerce")
    log.info(f"[Fold {fold_id}] 3/3 — Predicciones listas. Scores en rango [{df_test['final_score'].min():.3f}, {df_test['final_score'].max():.3f}]")

    agents_dict = {**base_agents, "sector_rotation": sector_agent, "meta_learner": meta}
    _export_feature_usage_report(
        agents=agents_dict,
        df_train=df_train_norm,
        fold_id=fold_id,
        agents_results_dir=agents_results_dir,
    )
    return agents_dict, df_test, df_train_with_oof


def train_full_history(
    df_norm: pd.DataFrame,
    y: pd.Series,
    target_alpha: Optional[pd.Series],
    agent_models_results_dir: str,
    random_seed: int = 42,
    sector_map: Optional[Dict[str, str]] = None,
    spy_prices: Optional[pd.Series] = None,
) -> Tuple[Dict, pd.DataFrame, Dict[str, float]]:
    df_norm = _drop_obsolete_unused_columns(df_norm, owner="FullHistory")

    agents_config = build_agents_config(agent_models_results_dir=agent_models_results_dir, random_seed=random_seed)
    base_agents = _instantiate_base_agents(agents_config)
    meta = AlphaMetaLearner(results_dir=agent_models_results_dir, random_seed=random_seed)

    recency_weights = _compute_recency_weights(df_norm)
    _fit_base_agents(base_agents, agents_config, df_norm, y, fold=0,
                     sample_weight=recency_weights)

    sector_agent = build_sector_rotation_agent(agent_models_results_dir=agent_models_results_dir, random_seed=random_seed)
    if sector_map is not None and "forward_return" in df_norm.columns:
        sector_agent.fit(df_norm, sector_map=sector_map, spy_prices=spy_prices, fold=0)

    df_with_scores = _predict_base_scores(base_agents, agents_config, df_norm)

    if sector_agent.is_trained and sector_map is not None:
        sector_scores = sector_agent.predict_sector_scores(df_with_scores, sector_map)
        df_with_scores["sector_score"] = sector_agent.map_to_tickers(
            df_with_scores, sector_map, sector_scores
        ).values
    else:
        df_with_scores["sector_score"] = 0.5

    if bool(NEUTRAL_SCORE_PENALTY_ENABLED):
        neutral_penalty_cols_full = [
            c for c in [f"{ag_name}_score" for ag_name in agents_config.keys()] if c in df_with_scores.columns
        ]
        if "risk_bear_score" in df_with_scores.columns and "risk_bear_score" not in neutral_penalty_cols_full:
            neutral_penalty_cols_full.append("risk_bear_score")
        df_with_scores = _apply_neutral_penalty(
            df_with_scores,
            neutral_penalty_cols_full,
            neutral_value=float(NEUTRAL_SCORE_VALUE),
            penalized_value=float(NEUTRAL_SCORE_PENALIZED_VALUE),
            eps=float(NEUTRAL_SCORE_EPS),
            exclude_cols={"sector_score"},
        )

    diag_score_cols = [
        c for c in [f"{ag_name}_score" for ag_name in agents_config.keys()] if c in df_with_scores.columns
    ]
    if "sector_score" in df_with_scores.columns:
        diag_score_cols.append("sector_score")

    redundancy_diag = compute_agent_redundancy(df_with_scores, diag_score_cols)
    diversity_multipliers = dict(redundancy_diag.get("multipliers", {}) or {})
    df_with_scores = apply_diversity_multipliers(df_with_scores, diversity_multipliers)

    behavior_full = compute_agent_behavior_features(df_with_scores, diag_score_cols)
    for col in behavior_full.columns:
        df_with_scores[col] = behavior_full[col].reindex(df_with_scores.index).values

    rule_quality_full = compute_rule_quality(df_with_scores, y)
    rule_quality_multipliers = _build_rule_quality_multipliers(rule_quality_full)
    if rule_quality_multipliers:
        for ag_name, mult in sorted(rule_quality_multipliers.items()):
            log.info("[RuleQuality/full_history] %s multiplier=%.3f", ag_name, float(mult))
        df_with_scores = _apply_rule_quality_multipliers(df_with_scores, rule_quality_multipliers)

    full_actions = build_research_actions(
        redundancy=redundancy_diag,
        rule_quality=rule_quality_full,
        behavior_features=behavior_full,
    )
    export_fold_diagnostics(
        output_dir=agent_models_results_dir,
        fold_id="full_history",
        redundancy=redundancy_diag,
        rule_quality=rule_quality_full,
        behavior_features=behavior_full,
        actions=full_actions,
    )

    score_cols = [c for c in diag_score_cols if c in df_with_scores.columns]
    dispersion_scales = _compute_dispersion_scales(df_with_scores, score_cols)
    df_with_scores = _apply_dispersion_shrink(df_with_scores, dispersion_scales)

    regime_model = MarketRegimeModel()
    regime_model.fit(df_with_scores)
    df_with_scores["regime_state"] = regime_model.predict(df_with_scores)
    df_with_scores, _ = apply_regime_weighting(df_with_scores, regime_col="regime_state")

    meta.fit(df_with_scores, y, fold=0, sector_col="sector", target_alpha=target_alpha)

    agents_dict = {**base_agents, "sector_rotation": sector_agent, "meta_learner": meta}
    return agents_dict, df_with_scores, dispersion_scales
