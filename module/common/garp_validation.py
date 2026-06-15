"""Fail-fast validation and interpretability helpers for the GARP/value-growth pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd

from environment import (
    PRIMARY_STRATEGY_PROFILE,
    REQUIRED_GARP_AGENTS,
    GARP_FORBIDDEN_FEATURE_PATTERNS,
    GARP_CRITICAL_FEATURES,
    GARP_SCORE_WEIGHTS,
)


FORWARD_LABEL_COLUMNS = {
    "forward_return",
    "spy_alpha",
    "sector_alpha",
    "garp_composite_target",
    "fundamental_improvement_proxy",
    "initial_valuation_reasonableness",
    "downside_penalty",
    "label",
    "tp_level",
    "sl_level",
    "tp_sl_outcome",
    "tp_sl_strategy",
    "tp_sl_strategy_variant",
}


def validate_garp_runtime_config(agent_config: Mapping[str, Mapping]) -> None:
    """Fail fast when the runtime does not represent the single supported architecture."""
    if PRIMARY_STRATEGY_PROFILE != "garp_value_growth":
        raise RuntimeError(f"Unsupported strategy profile {PRIMARY_STRATEGY_PROFILE!r}; only garp_value_growth is supported.")

    agents = set(agent_config.keys())
    required = set(REQUIRED_GARP_AGENTS)
    missing = sorted(required - agents)
    if missing:
        raise RuntimeError(f"Missing required GARP agents: {missing}")

    forbidden_agents = sorted(agents.intersection({"fundamental", "momentum", "bear"}))
    if forbidden_agents:
        raise RuntimeError(f"Old agent names are not supported in the GARP architecture: {forbidden_agents}")

    risk_cfg = agent_config.get("risk_bear", {})
    if not bool(risk_cfg.get("invert_y")):
        raise RuntimeError("risk_bear must be configured as an inverted negative/risk learner.")


def _matches_forbidden(col: str) -> bool:
    c = str(col).lower()
    return any(pattern in c for pattern in GARP_FORBIDDEN_FEATURE_PATTERNS)


def validate_no_forward_features(feature_columns: Iterable[str], *, context: str) -> None:
    """Ensure training/prediction features do not contain labels or forward-looking variables."""
    cols = [str(c) for c in feature_columns]
    bad = sorted(c for c in cols if c in FORWARD_LABEL_COLUMNS or _matches_forbidden(c))
    if bad:
        raise RuntimeError(f"Forward/label columns are forbidden as model features in {context}: {bad}")


def validate_critical_garp_features(df: pd.DataFrame, *, context: str, min_present_ratio: float = 0.35) -> None:
    """Fail if the dataset is missing too much of the GARP feature backbone."""
    present = [c for c in GARP_CRITICAL_FEATURES if c in df.columns]
    ratio = len(present) / max(len(GARP_CRITICAL_FEATURES), 1)
    if ratio < float(min_present_ratio):
        missing = sorted(set(GARP_CRITICAL_FEATURES) - set(present))
        raise RuntimeError(
            f"Insufficient GARP feature coverage in {context}: present={len(present)}/"
            f"{len(GARP_CRITICAL_FEATURES)} ({ratio:.1%}); missing={missing}"
        )


def classify_opportunities(df: pd.DataFrame, score_col: str = "final_score") -> pd.Series:
    """Classify each ticker into an objective GARP opportunity bucket."""
    idx = df.index
    q = pd.to_numeric(df.get("quality_score", 0.5), errors="coerce").fillna(0.5)
    g = pd.to_numeric(df.get("growth_score", 0.5), errors="coerce").fillna(0.5)
    v = pd.to_numeric(df.get("valuation_score", 0.5), errors="coerce").fillna(0.5)
    t = pd.to_numeric(df.get("fundamental_trend_score", 0.5), errors="coerce").fillna(0.5)
    c = pd.to_numeric(df.get("catalyst_score", 0.5), errors="coerce").fillna(0.5)
    r = pd.to_numeric(df.get("risk_bear_score", 0.5), errors="coerce").fillna(0.5)
    tech = pd.to_numeric(df.get("technical_guardrail_score", 0.5), errors="coerce").fillna(0.5)
    final = pd.to_numeric(df.get(score_col, df.get("regime_adjusted_score", 0.5)), errors="coerce").fillna(0.5)

    labels = pd.Series("Descartar", index=idx, dtype=object)
    labels[(v >= 0.68) & (g >= 0.62) & (q >= 0.55) & (r >= 0.50)] = "Growth infravalorado"
    labels[(q >= 0.70) & (g >= 0.62) & (v >= 0.45) & (r >= 0.55)] = "Quality Growth razonable"
    labels[(v >= 0.68) & (c >= 0.60) & (t >= 0.52) & (r >= 0.50)] = "Value con catalizador"
    labels[(q >= 0.72) & (t >= 0.62) & (v >= 0.50) & (r >= 0.58)] = "Compounder a precio razonable"
    labels[(v >= 0.60) & (t >= 0.62) & (c >= 0.58) & (q.between(0.40, 0.65)) & (r >= 0.48)] = "Turnaround"
    labels[(v >= 0.70) & (g.between(0.40, 0.58)) & (tech >= 0.45) & (r >= 0.50)] = "Cíclica barata"
    labels[(v >= 0.62) & ((q < 0.42) | (t < 0.40) | (r < 0.42))] = "Value trap"
    labels[(g >= 0.65) & (v < 0.38)] = "Growth caro"
    labels[(final < 0.50) | (r < 0.35) | (tech < 0.30)] = "Descartar"
    return labels


def add_ticker_explainability(df: pd.DataFrame, score_col: str = "final_score") -> pd.DataFrame:
    """Add opportunity class, driver/risk summaries and selection/discard rationales."""
    out = df.copy()
    out["opportunity_class"] = classify_opportunities(out, score_col=score_col)
    out["opportunity_type"] = out["opportunity_class"]

    score_cols = [
        "quality_score", "growth_score", "valuation_score", "fundamental_trend_score",
        "catalyst_score", "risk_bear_score", "technical_guardrail_score", "sector_score",
    ]
    feature_driver_cols = [
        "roic", "fcf_margin", "gross_margin", "revenue_yoy_growth", "fcf_yoy_growth",
        "eps_growth_trend_3y", "fcf_yield", "earnings_yield", "ev_to_ebitda", "pe_ratio",
        "roic_trend_2y", "net_margin_trend_2y", "eps_revision", "debt_to_ebitda",
        "volatility_60d", "price_vs_52w_high", "momentum_6m", "momentum_12m",
    ]

    def _top(row: pd.Series, positive: bool) -> str:
        items = []
        for col in score_cols:
            if col in row.index and pd.notna(row[col]):
                val = float(row[col])
                dist = val - 0.5
                items.append((col, dist, val))
        for col in feature_driver_cols:
            if col in row.index and pd.notna(row[col]):
                val = float(pd.to_numeric(row[col], errors="coerce"))
                if np.isfinite(val):
                    # raw features are listed descriptively without pretending to be SHAP.
                    items.append((col, val, val))
        items = sorted(items, key=lambda x: x[1], reverse=positive)
        return "; ".join(f"{name}={val:.3f}" for name, _, val in items[:5])

    out["top_5_positive_drivers"] = out.apply(lambda r: _top(r, True), axis=1)
    out["top_5_risks"] = out.apply(lambda r: _top(r, False), axis=1)
    final = pd.to_numeric(out.get(score_col, 0.5), errors="coerce").fillna(0.5)
    risk = pd.to_numeric(out.get("risk_bear_score", 0.5), errors="coerce").fillna(0.5)
    val = pd.to_numeric(out.get("valuation_score", 0.5), errors="coerce").fillna(0.5)
    growth = pd.to_numeric(out.get("growth_score", 0.5), errors="coerce").fillna(0.5)
    quality = pd.to_numeric(out.get("quality_score", 0.5), errors="coerce").fillna(0.5)
    expectation_gap = pd.to_numeric(out.get("expectation_gap_score", pd.Series(0.5, index=out.index)), errors="coerce").fillna(0.5)
    overexp = pd.to_numeric(out.get("overexpectation_penalty", pd.Series(0.5, index=out.index)), errors="coerce").fillna(0.5)
    out["value_trap_flag"] = ((val >= 0.62) & ((quality < 0.42) | (risk < 0.42))).astype(bool)
    out["expensive_growth_flag"] = ((growth >= 0.65) & ((val < 0.38) | (overexp >= 0.70))).astype(bool)
    if "moat_proxy_score" not in out.columns:
        out["moat_proxy_score"] = quality
    if "expectation_gap_score" not in out.columns:
        out["expectation_gap_score"] = expectation_gap
    out["selection_reason"] = np.where(
        final >= 0.57,
        "Selected: high GARP score combining quality/growth/valuation with acceptable risk.",
        "Not selected: final GARP score below portfolio threshold.",
    )
    out["discard_reason"] = np.where(
        risk < 0.40, "Discard: excessive risk/value-trap profile.",
        np.where(val < 0.35, "Discard: growth appears too expensive versus fundamentals.",
                 np.where(final < 0.50, "Discard: weak composite GARP conviction.", "")),
    )
    out["why_not_value_trap"] = np.where(
        (val >= 0.50) & (quality >= 0.50) & (risk >= 0.50),
        "Valuation is supported by quality and acceptable balance/risk signals.",
        "Value-trap risk remains: valuation not confirmed by quality/risk.",
    )
    out["why_not_expensive_growth"] = np.where(
        (growth >= 0.55) & (val >= 0.45) & (overexp < 0.70),
        "Growth score is accompanied by reasonable valuation and no extreme overexpectation penalty.",
        np.where(growth >= 0.65, "Growth is strong but valuation/expectations may be stretched.", "Growth is not the sole thesis."),
    )
    out["reason_for_classification"] = (
        "type=" + out["opportunity_type"].astype(str)
        + "; quality=" + quality.round(2).astype(str)
        + "; growth=" + growth.round(2).astype(str)
        + "; valuation=" + val.round(2).astype(str)
        + "; expectation_gap=" + expectation_gap.round(2).astype(str)
        + "; overexpectation=" + overexp.round(2).astype(str)
    )
    return out


def build_validation_audit(df: pd.DataFrame, feature_columns: Iterable[str], output_dir: str | Path, fold_id: str) -> None:
    """Export anti-leakage, anti-momentum and score-contribution diagnostics."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = sorted(str(c) for c in feature_columns)
    feature_rows = []
    for col in features:
        domain = "technical_guardrail" if "momentum" in col or "rsi" in col or "volatility" in col else "fundamental_or_context"
        feature_rows.append({
            "fold_id": fold_id,
            "feature": col,
            "temporal_origin": "point_in_time_snapshot_or_train_only_cross_section",
            "domain": domain,
            "forbidden_forward_feature": col in FORWARD_LABEL_COLUMNS or _matches_forbidden(col),
        })
    pd.DataFrame(feature_rows).to_csv(out_dir / "garp_feature_leakage_audit.csv", index=False)

    score = pd.to_numeric(df.get("final_score", df.get("regime_adjusted_score", 0.5)), errors="coerce")
    momentum_corr = {}
    for col in ["momentum_6m", "momentum_12m"]:
        if col in df.columns:
            momentum_corr[col] = float(score.corr(pd.to_numeric(df[col], errors="coerce")))
    score_cols = [c for c in [
        "quality_score", "growth_score", "valuation_score", "fundamental_trend_score",
        "catalyst_score", "risk_bear_score", "technical_guardrail_score", "sector_score",
    ] if c in df.columns]
    contributions = []
    for col in score_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        contributions.append({
            "fold_id": fold_id,
            "score_column": col,
            "mean_score": float(s.mean()) if s.notna().any() else np.nan,
            "std_score": float(s.std()) if s.notna().any() else np.nan,
            "corr_with_final_score": float(s.corr(score)) if s.notna().sum() > 1 else np.nan,
            "configured_weight": float(GARP_SCORE_WEIGHTS.get(col.replace("_score", ""), np.nan)),
        })
    pd.DataFrame(contributions).to_csv(out_dir / "garp_agent_score_contribution.csv", index=False)

    selected = df[score >= score.quantile(0.80)] if score.notna().sum() else df.head(0)
    mediocre = selected[pd.to_numeric(selected.get("momentum_6m", 0), errors="coerce").fillna(0.0) <= 0.03]
    strong_mom = df[(pd.to_numeric(df.get("momentum_6m", 0), errors="coerce").fillna(0.0) >= 0.20) & (score < score.quantile(0.50))]
    examples = {
        "fold_id": fold_id,
        "score_momentum_correlations": momentum_corr,
        "selected_with_mediocre_momentum": _ticker_sample(mediocre),
        "discarded_despite_strong_momentum": _ticker_sample(strong_mom),
    }
    (out_dir / "garp_anti_momentum_audit.json").write_text(json.dumps(examples, indent=2, default=str), encoding="utf-8")


def _ticker_sample(df: pd.DataFrame, n: int = 10) -> list[str]:
    if df is None or df.empty:
        return []
    if isinstance(df.index, pd.MultiIndex) and "ticker" in df.index.names:
        return [str(x) for x in df.index.get_level_values("ticker")[:n].tolist()]
    if "ticker" in df.columns:
        return [str(x) for x in df["ticker"].head(n).tolist()]
    return [str(x) for x in df.index[:n].tolist()]
