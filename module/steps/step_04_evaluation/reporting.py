"""Consolidated evaluation reporting utilities."""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from module.steps.step_04_evaluation.explainability import (
    AgentExplainer,
    FEATURE_DESCRIPTIONS,
    _format_value,
)
log = logging.getLogger(__name__)

_COLUMN_ORDER = [
    "ticker",
    "fold_id",
    "entry_date",
    "actual_entry_date",
    "sector",
    "score",
    "confidence",
    "ev",
    "tp_pct",
    "sl_pct",
    "entry_price",
    "tp_price",
    "sl_price",
    "selected",
    "outcome",
    "days_to_outcome",
    "outcome_date",
]

# ---------------------------------------------------------------------------
# Score interpretation thresholds per agent
# ---------------------------------------------------------------------------

AGENT_LABELS = {
    "fundamental": {
        "high":   "Strong financial health",
        "medium": "Acceptable financial health",
        "low":    "Financial weaknesses detected",
    },
    "valuation": {
        "high":   "Attractive valuation",
        "medium": "Reasonable valuation",
        "low":    "Possibly overvalued",
    },
    "momentum": {
        "high":   "Positive technical trend",
        "medium": "Neutral technical trend",
        "low":    "Negative technical trend",
    },
    "bear": {
        "high":   "Low risk (defensive profile)",
        "medium": "Moderate risk",
        "low":    "High risk detected",
    },
    "sentiment": {
        "high":   "Positive external sentiment",
        "medium": "Neutral external sentiment",
        "low":    "Negative external sentiment",
    },
    "meta_learner": {
        "high":   "High probability of Outperform",
        "medium": "Mixed signal",
        "low":    "High probability of Underperform",
    },
}


def _is_ratio_or_normalized_feature(feature: str) -> bool:
    f = str(feature).lower().strip()
    if not f:
        return False
    allowed_tokens = [
        "ratio", "margin", "yield", "growth", "trend", "momentum", "volatility",
        "rsi", "macd", "beta", "zscore", "pct", "coverage",
        "score", "prior", "dispersion", "consensus", "confidence", "quality",
        "fscore", "accrual", "atr", "bb_", "vs_5y", "vs_52w", "debt_to_", "_to_",
        "revision", "surprise", "beater", "overbought", "oversold", "bullish",
        "above_sma", "cross_sma", "expansion", "decline", "losses", "risk",
    ]
    if any(tok in f for tok in allowed_tokens):
        return True
    blocked_prefixes = [
        "revenue", "net_income", "operating_income", "gross_profit", "fcf", "ebitda",
        "total_assets", "total_liabilities", "total_equity", "total_debt", "cash",
        "shares", "eps_est", "eps_reported", "market_cap", "capex", "income_tax",
        "depreciation", "operating_cash_flow",
    ]
    if any(f.startswith(p) for p in blocked_prefixes):
        return False
    return False


def _agent_text_label(agent: str, score: float) -> str:
    """Converts an agent score to a human-readable text label.

    Args:
        agent (str): Agent name (must match a key in AGENT_LABELS).
        score (float): Agent score in [0, 1].

    Returns:
        str: Formatted string like "0.72 â€” Strong financial health".
    """
    labels = AGENT_LABELS.get(agent, {
        "high": "High score", "medium": "Medium score", "low": "Low score"
    })
    if score >= 0.65:
        tier = "high"
    elif score >= 0.40:
        tier = "medium"
    else:
        tier = "low"
    return f"{score:.2f} â€” {labels[tier]}"


# ---------------------------------------------------------------------------
# Key features per agent (for text explanations)
# These provide the most interpretable context without requiring SHAP
# ---------------------------------------------------------------------------

AGENT_KEY_FEATURES: Dict[str, List[str]] = {
    "fundamental": [
        "roe", "net_margin", "revenue_yoy_growth", "debt_to_ebitda",
        "interest_coverage", "fcf_margin", "piotroski_fscore",
        "earnings_quality", "consecutive_losses",
    ],
    "valuation": [
        "pe_ratio", "pb_ratio", "ev_to_ebitda", "fcf_yield",
        "pe_vs_5y_median", "ev_ebitda_vs_5y_median",
    ],
    "momentum": [
        "momentum_3m", "momentum_6m", "momentum_12m",
        "rsi_14", "sma_200", "volatility_60d",
    ],
    "bear": [
        "debt_to_ebitda", "consecutive_losses", "insider_sell_ratio",
        "current_ratio", "revenue_yoy_growth",
    ],
    "sentiment": [
        "analyst_buy_ratio", "analyst_consensus", "eps_surprise_pct",
        "beat_rate_4q", "mspr_3m", "insider_net_ratio_90d",
    ],
}


def _describe_feature_value(feature: str, value: float) -> str:
    """Generates a short natural-language phrase for a featureâ€“value pair.

    Example: roe=0.28 â†’ "Return on Equity = 28.0%"

    Args:
        feature (str): Feature column name.
        value (float): Feature value.

    Returns:
        str: Human-readable description, or empty string if value is NaN.
    """
    if pd.isna(value):
        return ""
    desc = FEATURE_DESCRIPTIONS.get(feature, feature.replace("_", " ").title())
    formatted = _format_value(feature, value)
    return f"{desc} = {formatted}"


def _build_agent_explanation(
    row: pd.Series,
    agent: str,
    agent_score: float,
    shap_drivers: Optional[List[Dict]] = None,
    top_n: int = 4,
) -> str:
    """Builds a natural-language explanation for a single agent's prediction.

    Priority:
    1. If SHAP drivers are available (from explainability.py), uses the top
       positive and negative SHAP contributors.
    2. Otherwise, uses the agent's key features with their actual values.

    Args:
        row (pd.Series): Full feature row for the ticker/fold.
        agent (str): Agent name.
        agent_score (float): Agent score in [0, 1].
        shap_drivers (Optional[List[Dict]]): Pre-computed SHAP driver list.
        top_n (int): Maximum number of positive/negative factors to include.

    Returns:
        str: Multi-line explanation string.
    """
    label = "Outperform" if agent_score >= 0.5 else "Underperform"
    score_txt = _agent_text_label(agent, agent_score)

    # -- Path 1: SHAP available --
    if shap_drivers:
        positives = [d for d in shap_drivers if d.get("shap_value", 0) > 0][:top_n]
        negatives = [d for d in shap_drivers if d.get("shap_value", 0) < 0][:top_n]

        parts = [f"[{label}] {score_txt}"]

        if positives:
            favor = "; ".join(
                f"{d.get('description', d['feature'])} = "
                f"{_format_value(d['feature'], d.get('raw_value', np.nan))}"
                f" (+{d['shap_value']:.3f})"
                for d in positives
            )
            parts.append(f"In favor: {favor}")

        if negatives:
            contra = "; ".join(
                f"{d.get('description', d['feature'])} = "
                f"{_format_value(d['feature'], d.get('raw_value', np.nan))}"
                f" ({d['shap_value']:.3f})"
                for d in negatives
            )
            parts.append(f"Against: {contra}")

        return " | ".join(parts)

    # -- Path 2: fallback using agent key features --
    key_features = AGENT_KEY_FEATURES.get(agent, [])
    observations: List[str] = []

    for feat in key_features:
        val = row.get(feat, np.nan)
        if pd.isna(val):
            continue
        phrase = _describe_feature_value(feat, float(val))
        if phrase:
            observations.append(phrase)

    if not observations:
        return f"[{label}] {score_txt}"

    obs_text = "; ".join(observations[:top_n])
    return f"[{label}] {score_txt} | Observed factors: {obs_text}"


# ---------------------------------------------------------------------------
# Fold score DataFrame construction
# ---------------------------------------------------------------------------

def build_fold_scores_df(
    df_test_scored: pd.DataFrame,
    y_test: pd.Series,
    fold_id: int,
    year_quarter: str,
    agents: Dict,
    audit_df: Optional[pd.DataFrame] = None,
    actual_returns: Optional[Dict[str, float]] = None,
    benchmark_return: Optional[float] = None,
    ticker_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Builds the score DataFrame for a single walk-forward fold.

    Produces one row per ticker with:
      - Per-agent scores plus human-readable interpretation labels.
      - Per-agent explanations based on SHAP drivers or key features.
      - Realized result if available.

    Args:
        df_test_scored (pd.DataFrame): Test DataFrame with agent scores and
            final_score column. Indexed by (ticker, date).
        y_test (pd.Series): Ground-truth labels for the fold.
        fold_id (int): Walk-forward fold number.
        year_quarter (str): Quarter identifier, e.g. "2025Q3".
        agents (Dict): Dictionary of trained agents ``{name: agent}``.
        audit_df (Optional[pd.DataFrame]): Selection audit DataFrame (optional).
        actual_returns (Optional[Dict[str, float]]): ``{ticker: actual_return}``
            if the test period has elapsed.
        benchmark_return (Optional[float]): S&P 500 return for the test period.
        ticker_weights (Optional[Dict[str, float]]): Portfolio weight per ticker.

    Returns:
        pd.DataFrame: One row per ticker with all score, explanation, and
            result columns.
    """
    tickers_index = df_test_scored.index.get_level_values("ticker")
    # One row per ticker: take the last available observation in the fold
    ticker_rows: Dict[str, pd.Series] = {}
    for ticker in tickers_index.unique():
        mask = tickers_index == ticker
        ticker_rows[ticker] = df_test_scored.loc[mask].iloc[-1]

    # Build audit mapping from the audit DataFrame
    audit_map: Dict[str, Dict] = {}
    if audit_df is not None and not audit_df.empty and "ticker" in audit_df.columns:
        audit_map = audit_df.drop_duplicates(subset="ticker").set_index("ticker").to_dict(orient="index")

    records: List[Dict] = []

    for ticker, row in ticker_rows.items():
        final_score = float(row.get("final_score", 0.5))
        confidence_up = float(pd.to_numeric(row.get("confidence_up", row.get("confidence", final_score)), errors="coerce"))
        if not np.isfinite(confidence_up):
            confidence_up = final_score
        confidence_up = float(np.clip(confidence_up, 0.0, 1.0))
        confidence_tp_vs_sl = float(pd.to_numeric(row.get("confidence_tp_vs_sl", row.get("historical_tp_prob", confidence_up)), errors="coerce"))
        if not np.isfinite(confidence_tp_vs_sl):
            confidence_tp_vs_sl = confidence_up
        confidence_tp_vs_sl = float(np.clip(confidence_tp_vs_sl, 0.0, 1.0))
        audit = audit_map.get(ticker, {})

        rec: Dict = {
            # --- Identification ---
            "year_quarter":     year_quarter,
            "fold":             fold_id,
            "ticker":           ticker,
            "sector":           row.get("sector", "Unknown"),
            "industry":         row.get("industry", "Unknown"),

            # --- Prediction ---
            "final_score":      round(final_score, 4),
            "final_score_raw":  round(float(row.get("final_score_raw", final_score)), 4),
            "prediction":       "Outperform" if final_score >= 0.5 else "Underperform",
            "confidence":       "High" if abs(confidence_up - 0.5) > 0.25 else "Moderate",
            "confidence_up":    round(confidence_up, 4),
            "confidence_tp_vs_sl": round(confidence_tp_vs_sl, 4),
            "common_score":     round(final_score, 4),
            "common_label":     "Outperform" if final_score >= 0.5 else "Underperform",

            # --- Selection and portfolio weight ---
            "selected":         bool(audit.get("selected", False)),
            "rank":             audit.get("rank", None),
            "selection_reason": audit.get("selection_reason", "unknown"),
            "portfolio_weight": round(ticker_weights[ticker], 4) if ticker_weights and ticker in ticker_weights else None,
            "sector_peer_count": int(row.get("sector_peer_count", 0)) if pd.notna(row.get("sector_peer_count", None)) else None,
            "sector_confidence": round(float(row.get("sector_confidence", 1.0)), 4),
        }

        # --- Per-agent scores + interpretation text ---
        for ag_name in ["fundamental", "valuation", "momentum", "bear", "sentiment"]:
            score_col = f"{ag_name}_score"
            ag_score = float(row.get(score_col, 0.5))
            rec[score_col] = round(ag_score, 4)
            rec[f"{ag_name}_interpretation"] = _agent_text_label(ag_name, ag_score)

        # --- Per-agent explanation with concrete feature values ---
        for ag_name, agent in agents.items():
            if ag_name == "meta_learner":
                continue
            if ag_name == "sector_rotation":
                ag_score = float(row.get("sector_score", 0.5))
            else:
                ag_score = float(row.get(f"{ag_name}_score", 0.5))

            # Attempt to get SHAP drivers from the agent's explainer
            shap_drivers = None
            explainer = getattr(agent, "_explainer", None)
            if explainer is not None:
                try:
                    exp = explainer.explain_prediction(row, ticker, ag_score, top_n=5)
                    shap_drivers = exp.get("top_drivers", [])
                except Exception:
                    log.debug("SHAP explain failed for %s/%s", ag_name, ticker, exc_info=True)

            rec[f"{ag_name}_explanation"] = _build_agent_explanation(
                row=row,
                agent=ag_name,
                agent_score=ag_score,
                shap_drivers=shap_drivers,
                top_n=4,
            )

        # --- Realized result (if the test period has elapsed) ---
        if actual_returns is not None:
            actual = actual_returns.get(ticker)
            rec["actual_return"] = round(actual, 4) if actual is not None else None
            if actual is not None and benchmark_return is not None:
                rec["alpha_real"] = round(actual - benchmark_return, 4)
                rec["beat_benchmark"] = actual > benchmark_return
            else:
                rec["alpha_real"] = None
                rec["beat_benchmark"] = None
        else:
            rec["actual_return"] = None
            rec["alpha_real"] = None
            rec["beat_benchmark"] = None

        # --- Ground-truth label from the model ---
        label_idx = (ticker, row.name[1]) if isinstance(row.name, tuple) else None
        if label_idx is not None and label_idx in y_test.index:
            rec["true_label"] = int(y_test.loc[label_idx])
        else:
            rec["true_label"] = None

        records.append(rec)

    df = pd.DataFrame(records)

    # Sort: selected tickers first, then by descending final score
    df = df.sort_values(
        ["selected", "final_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Export: individual fold + cumulative global
# ---------------------------------------------------------------------------

def export_fold_scores(
    df: pd.DataFrame,
    agents_results_dir: str,
    fold_id: int | str,
) -> Path:
    """Saves the fold CSV and returns the output path.

    Args:
        df (pd.DataFrame): Fold score DataFrame produced by
            :func:`build_fold_scores_df`.
        agents_results_dir (str): Root output directory.
        fold_id (int | str): Fold identifier used in the filename.

    Returns:
        Path: Path to the written CSV file.
    """
    out_dir = Path(agents_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scores.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    log.info(f"[FoldReport] Fold {fold_id}: scores for {len(df)} tickers saved â†’ {path.name}")
    return path


def export_quarter_snapshot_audit(
    df_test_scored: pd.DataFrame,
    year_quarter: str,
    agents_results_dir: str,
) -> Path:
    """Exports a complete per-ticker snapshot for the quarter.

    Includes:
      - One row per ticker (last available observation in the fold)
      - All available features
      - Snapshot/reporting metadata (carry-forward flags, dates used)
      - Agent scores and final score

    Args:
        df_test_scored (pd.DataFrame): Scored test DataFrame indexed by
            (ticker, date).
        year_quarter (str): Quarter identifier string (e.g. "2025Q3").
        agents_results_dir (str): Root output directory.

    Returns:
        Path: Path to the written CSV audit file.
    """
    out_dir = Path(agents_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ticker_snapshot_audit.csv"

    if df_test_scored.empty:
        pd.DataFrame().to_csv(path, index=False, encoding="utf-8")
        return path

    tickers = df_test_scored.index.get_level_values("ticker")
    rows: List[pd.Series] = []
    for ticker in tickers.unique():
        mask = tickers == ticker
        rows.append(df_test_scored.loc[mask].iloc[-1])

    out_df = pd.DataFrame(rows).reset_index(drop=True)
    if "ticker" in out_df.columns:
        out_df["ticker"] = out_df["ticker"].astype(str)
    else:
        out_df.insert(0, "ticker", list(tickers.unique()))

    if "year_quarter" in out_df.columns:
        out_df["year_quarter"] = year_quarter
    else:
        out_df.insert(0, "year_quarter", year_quarter)
    out_df.to_csv(path, index=False, encoding="utf-8")
    log.info(f"[FoldReport] Snapshot audit {year_quarter}: {len(out_df)} tickers â†’ {path.name}")
    return path


def export_quarter_agent_feature_audit(
    df_test_scored: pd.DataFrame,
    agents: Dict,
    year_quarter: str,
    agents_results_dir: str,
) -> Path:
    """Exports granular agentâ€“feature traceability for a quarter.

    Produces one row per (ticker, agent, feature used by that agent), allowing
    full inspection of which features drove each agent's score for each ticker.

    Args:
        df_test_scored (pd.DataFrame): Scored test DataFrame indexed by
            (ticker, date).
        agents (Dict): Dictionary of trained agents ``{name: agent}``.
        year_quarter (str): Quarter identifier string (e.g. "2025Q3").
        agents_results_dir (str): Root output directory.

    Returns:
        Path: Path to the written CSV audit file.
    """
    out_dir = Path(agents_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ticker_agent_feature_audit.csv"

    if df_test_scored.empty:
        pd.DataFrame().to_csv(path, index=False, encoding="utf-8")
        return path

    tickers = df_test_scored.index.get_level_values("ticker")
    rows: List[Dict] = []

    agent_order = ["fundamental", "valuation", "momentum", "bear", "sentiment", "sector_rotation", "meta_learner"]
    for ticker in tickers.unique():
        mask = tickers == ticker
        row = df_test_scored.loc[mask].iloc[-1]
        final_score = float(row.get("final_score", np.nan)) if pd.notna(row.get("final_score", np.nan)) else np.nan

        for agent_name in agent_order:
            if agent_name not in agents:
                continue
            if agent_name == "sector_rotation":
                score_col = "sector_score"
            elif agent_name == "meta_learner":
                score_col = "final_score"
            else:
                score_col = f"{agent_name}_score"

            agent_score = row.get(score_col, np.nan)
            try:
                agent_score = float(agent_score)
            except Exception:
                agent_score = np.nan

            agent_obj = agents.get(agent_name)
            feat_cols = getattr(agent_obj, "_feature_cols", None)
            if not feat_cols:
                feat_cols = []

            if not feat_cols:
                rows.append({
                    "year_quarter": year_quarter,
                    "ticker": ticker,
                    "agent": agent_name,
                    "agent_score": agent_score,
                    "final_score": final_score,
                    "feature": "__no_feature_list__",
                    "feature_value": np.nan,
                    "feature_present": False,
                })
                continue

            for feat in feat_cols:
                if not _is_ratio_or_normalized_feature(feat):
                    continue
                val = row.get(feat, np.nan)
                rows.append({
                    "year_quarter": year_quarter,
                    "ticker": ticker,
                    "agent": agent_name,
                    "agent_score": agent_score,
                    "final_score": final_score,
                    "feature": feat,
                    "feature_value": val,
                    "feature_present": bool(pd.notna(val)),
                })

    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    log.info(f"[FoldReport] Agent-feature audit {year_quarter}: {len(rows)} rows â†’ {path.name}")
    return path


# NOTE: `export_all_folds_scores` removed — not referenced elsewhere in the codebase.

log = logging.getLogger(__name__)


def _score_label(score: float) -> str:
    return "Outperform" if score >= 0.5 else "Underperform"


def _normalise_ticker_list(tickers: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(tk) for tk in tickers if pd.notna(tk)))


def _rule_based_drivers(row: pd.Series, score: float) -> List[Dict]:
    checks = [
        ("roe", lambda v: v > 0.15, "ROE solido", "positive"),
        ("net_margin", lambda v: v > 0.10, "Margen neto positivo", "positive"),
        ("debt_to_ebitda", lambda v: v > 6.0, "Deuda elevada vs EBITDA", "negative"),
        ("current_ratio", lambda v: v < 1.0, "Riesgo de liquidez", "negative"),
        ("revenue_yoy_growth", lambda v: v > 0, "Crecimiento de ingresos", "positive"),
        ("fcf", lambda v: v < 0, "FCF negativo", "negative"),
        ("momentum_12m", lambda v: v > 0, "Momentum anual positivo", "positive"),
        ("rsi_14", lambda v: v > 70, "RSI sobrecomprado", "negative"),
        ("eps_surprise_pct", lambda v: v > 0, "Sorpresa de EPS positiva", "positive"),
        ("beat_rate_4q", lambda v: v >= 0.75, "Beat rate consistente", "positive"),
        ("mspr_3m", lambda v: v > 0, "MSPR positivo", "positive"),
        ("insider_net_ratio_90d", lambda v: v > 0, "Balance neto insider positivo", "positive"),
    ]

    drivers: List[Dict] = []
    for feat, fn, desc, direction in checks:
        val = row.get(feat, pd.NA)
        if pd.isna(val):
            continue
        try:
            if fn(float(val)):
                drivers.append({
                    "feature": feat,
                    "description": desc,
                    "shap_value": float(0.0),
                    "raw_value": float(val),
                    "direction": direction,
                })
        except Exception:
            log.debug("Rule-based driver check failed for %s", feat, exc_info=True)
            continue

    if not drivers:
        numeric = row.select_dtypes(include="number") if hasattr(row, "select_dtypes") else pd.Series(dtype=float)
        for feat, val in numeric.abs().sort_values(ascending=False).head(3).items():
            drivers.append({
                "feature": feat,
                "description": feat.replace("_", " ").title(),
                "shap_value": float(0.0),
                "raw_value": float(row.get(feat, 0.0)),
                "direction": "positive" if score >= 0.5 else "negative",
            })

    return drivers


def _fallback_explanation(agent_name: str, row: pd.Series, ticker: str, score: float) -> tuple[str, List[Dict]]:
    text = AgentExplainer._rule_based_text(row, score, ticker)
    drivers = _rule_based_drivers(row, score)
    if not drivers:
        drivers = [{
            "feature": f"{agent_name}_score",
            "description": f"Score del agente {agent_name}",
            "shap_value": 0.0,
            "raw_value": float(score),
            "direction": "positive" if score >= 0.5 else "negative",
        }]
    return text, drivers


def _format_driver_list(drivers: List[Dict]) -> str:
    parts: List[str] = []
    for driver in drivers:
        label = driver.get("description") or driver.get("feature") or ""
        raw_value = driver.get("raw_value")
        if pd.isna(raw_value):
            value_text = "N/A"
        else:
            try:
                value_text = f"{float(raw_value):.3f}"
            except Exception:
                value_text = str(raw_value)
        parts.append(f"{label}={value_text}")
    return ", ".join(parts)


def _flatten_text(text: str) -> str:
    return " ".join(str(text).split())


def _split_driver_groups(drivers: List[Dict]) -> tuple[str, str]:
    positives: List[Dict] = []
    negatives: List[Dict] = []
    for d in drivers:
        shap_value = float(d.get("shap_value", 0.0))
        direction = str(d.get("direction", "")).strip().lower()
        if abs(shap_value) > 1e-12:
            if shap_value > 0:
                positives.append(d)
            else:
                negatives.append(d)
            continue

        if direction == "negative":
            negatives.append(d)
        else:
            positives.append(d)
    return _format_driver_list(positives), _format_driver_list(negatives)


def build_selection_audit_df(
    df_scored: pd.DataFrame,
    selected_tickers: Sequence[str],
    score_col: str = "final_score",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Create a flat audit table with the reason each ticker was or was not selected."""
    if df_scored.empty:
        return pd.DataFrame()

    df = df_scored.copy()
    if "ticker" not in df.columns:
        df = df.reset_index()

    if score_col not in df.columns:
        raise KeyError(f"Score column '{score_col}' not found in scored dataframe")

    if "ticker" in df.columns and df.duplicated(subset="ticker").any():
        # Keep the latest occurrence per ticker to avoid ambiguous audit rows.
        df = df.drop_duplicates(subset="ticker", keep="last")

    selected_list = _normalise_ticker_list(selected_tickers)
    selected_set = set(selected_list)
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["selected"] = df["ticker"].isin(selected_set)
    df["common_score"] = df[score_col].astype(float)
    df["common_label"] = df["common_score"].map(_score_label)
    df["score_gap_vs_threshold"] = df["common_score"] - threshold

    selected_scores = df.loc[df["selected"], ["ticker", "common_score"]].copy()
    selected_cutoff = float(selected_scores["common_score"].min()) if not selected_scores.empty else float("nan")
    df["score_gap_vs_selected_cutoff"] = df["common_score"] - selected_cutoff
    df["selected_cutoff_score"] = selected_cutoff

    selected_rank_map = {tk: idx + 1 for idx, tk in enumerate(selected_list)}
    df["selected_rank"] = df["ticker"].map(selected_rank_map)

    def _reason(row: pd.Series) -> str:
        if row["selected"]:
            if row["common_score"] >= threshold:
                return "selected_above_threshold"
            return "selected_by_fallback"
        if row["common_score"] >= threshold:
            return "qualified_but_not_selected"
        return "below_threshold"

    df["selection_reason"] = df.apply(_reason, axis=1)
    df["above_threshold"] = df["common_score"] >= threshold
    df["distance_to_threshold"] = df["common_score"] - threshold
    df["selected_position"] = df["rank"].where(df["selected"])

    return df


def build_explanation_candidate_tickers(
    audit_df: pd.DataFrame,
    threshold: float = 0.5,
    top_extra: int = 10,
    near_margin: float = 0.05,
    max_candidates: int = 30,
) -> List[str]:
    """Choose a compact but informative set of tickers to explain."""
    if audit_df.empty:
        return []

    ordered = audit_df.sort_values("common_score", ascending=False)
    candidates: List[str] = []

    if "selected" in ordered.columns:
        candidates.extend(ordered.loc[ordered["selected"], "ticker"].tolist())

    candidates.extend(ordered.head(top_extra)["ticker"].tolist())

    near_mask = ordered["common_score"].sub(threshold).abs() <= near_margin
    candidates.extend(ordered.loc[near_mask, "ticker"].tolist())

    # Complete with highest scores to keep explanations aligned with ranking.
    candidates.extend(ordered["ticker"].tolist())

    return _normalise_ticker_list(candidates)[:max_candidates]


def export_selection_audit(
    audit_df: pd.DataFrame,
    results_dir: str,
    fold_id: Optional[int | str] = None,
    prefix: str = "fold",
) -> tuple[Path, Path]:
    """Persist the selection audit as CSV and JSON."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "selection_audit.csv"
    json_path = out_dir / "selection_audit.json"

    if audit_df.empty:
        csv_path.write_text("", encoding="utf-8")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"rows": 0, "selected": 0}, f, indent=2, ensure_ascii=False)
        return csv_path, json_path

    audit_df.to_csv(csv_path, index=False, encoding="utf-8")
    summary = {
        "rows": int(len(audit_df)),
        "selected": int(audit_df["selected"].sum()) if "selected" in audit_df.columns else 0,
        "qualified_but_not_selected": int((audit_df["selection_reason"] == "qualified_but_not_selected").sum())
        if "selection_reason" in audit_df.columns else 0,
        "below_threshold": int((audit_df["selection_reason"] == "below_threshold").sum())
        if "selection_reason" in audit_df.columns else 0,
        "selected_by_fallback": int((audit_df["selection_reason"] == "selected_by_fallback").sum())
        if "selection_reason" in audit_df.columns else 0,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    return csv_path, json_path


def export_ticker_explanations(
    agents: Dict,
    df_test: pd.DataFrame,
    scores: pd.Series,
    fold_id: int | str,
    agents_results_dir: str,
    candidate_tickers: Sequence[str],
    audit_df: Optional[pd.DataFrame] = None,
    explanation_top_n: int = 6,
    prefix: str = "fold",
) -> tuple[Path, Path]:
    """Generate flat per-agent explanation rows for a compact candidate universe."""
    if scores.empty:
        out_dir = Path(agents_results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "ticker_explanations.csv"
        json_path = out_dir / "ticker_explanations.json"
        csv_path.write_text("", encoding="utf-8")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"fold": fold_id, "tickers": {}}, f, indent=2, ensure_ascii=False)
        return csv_path, json_path

    out_dir = Path(agents_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ticker_explanations.csv"
    json_path = out_dir / "ticker_explanations.json"

    tickers_col = df_test.index.get_level_values("ticker")
    ticker_scores = pd.Series(scores.values, index=tickers_col).groupby(level=0).last()
    selected = _normalise_ticker_list(candidate_tickers)
    audit_lookup = {}
    if audit_df is not None and not audit_df.empty:
        audit_unique = audit_df
        if audit_unique.duplicated(subset="ticker").any():
            audit_unique = audit_unique.drop_duplicates(subset="ticker", keep="last")
        audit_lookup = audit_unique.set_index("ticker").to_dict(orient="index")

    all_explanations = {"fold": fold_id, "tickers": {}}
    flat_rows: List[Dict] = []

    for ticker in selected:
        mask = tickers_col == ticker
        if not mask.any():
            continue
        row = df_test.loc[mask].iloc[-1]
        common_score = float(ticker_scores.get(ticker, 0.5))
        common_label = _score_label(common_score)
        audit_row = audit_lookup.get(ticker, {})

        ticker_exp = {
            "common_score": round(common_score, 4),
            "common_label": common_label,
            "selection_reason": audit_row.get("selection_reason", "unknown"),
            "selected": bool(audit_row.get("selected", False)),
            "agents": {},
        }

        for ag_name, ag in agents.items():
            explainer = getattr(ag, "_explainer", None)
            agent_score = common_score
            # sector_rotation stores its score in "sector_score", not "sector_rotation_score"
            if ag_name == "sector_rotation":
                agent_score_col = "sector_score"
            else:
                agent_score_col = f"{ag_name}_score"
            if agent_score_col in df_test.columns:
                try:
                    agent_score = float(row[agent_score_col])
                except Exception:
                    agent_score = common_score
            if ag_name == "sector_rotation" and (pd.isna(agent_score) or agent_score_col not in df_test.columns):
                audit_sector_score = audit_row.get("sector_score")
                if audit_sector_score is not None and not pd.isna(audit_sector_score):
                    agent_score = float(audit_sector_score)
            agent_label = _score_label(agent_score)

            flat_row = {
                "fold": fold_id,
                "ticker": ticker,
                "common_score": round(common_score, 4),
                "common_label": common_label,
                "selection_reason": audit_row.get("selection_reason", "unknown"),
                "selected": bool(audit_row.get("selected", False)),
                "rank": audit_row.get("rank"),
                "agent": ag_name,
                "agent_score": round(agent_score, 4),
                "agent_label": agent_label,
                "has_explainer": explainer is not None,
                "explanation_text": "",
                "favor_factors": "",
                "contra_factors": "",
                "top_drivers_json": "[]",
            }

            if explainer is None:
                fallback_text, fallback_drivers = _fallback_explanation(ag_name, row, ticker, agent_score)
                favor_text, contra_text = _split_driver_groups(fallback_drivers[:explanation_top_n])
                flat_row["explanation_text"] = _flatten_text(fallback_text)
                flat_row["favor_factors"] = favor_text
                flat_row["contra_factors"] = contra_text
                flat_row["top_drivers_json"] = json.dumps(fallback_drivers[:explanation_top_n], ensure_ascii=False, default=str)
                ticker_exp["agents"][ag_name] = {
                    "text": fallback_text,
                    "top_drivers": fallback_drivers[:explanation_top_n],
                }
                flat_rows.append(flat_row)
                continue

            try:
                exp = explainer.explain_prediction(row, ticker, agent_score, top_n=explanation_top_n, fold=fold_id)
                top_drivers = exp.get("top_drivers", [])[:explanation_top_n]
                favor_text, contra_text = _split_driver_groups(top_drivers)
                ticker_exp["agents"][ag_name] = {
                    "text": exp.get("text", ""),
                    "top_drivers": top_drivers,
                }
                flat_row["explanation_text"] = _flatten_text(exp.get("text", ""))
                flat_row["favor_factors"] = favor_text
                flat_row["contra_factors"] = contra_text
                flat_row["top_drivers_json"] = json.dumps(top_drivers, ensure_ascii=False, default=str)
                flat_rows.append(flat_row)
            except Exception as ex:
                log.debug(f"Explain {ag_name}/{ticker}: {ex}")
                fallback_text, fallback_drivers = _fallback_explanation(ag_name, row, ticker, agent_score)
                favor_text, contra_text = _split_driver_groups(fallback_drivers[:explanation_top_n])
                flat_row["explanation_text"] = _flatten_text(f"Error generating explanation: {ex}. {fallback_text}")
                flat_row["favor_factors"] = favor_text
                flat_row["contra_factors"] = contra_text
                flat_row["top_drivers_json"] = json.dumps(fallback_drivers[:explanation_top_n], ensure_ascii=False, default=str)
                ticker_exp["agents"][ag_name] = {
                    "text": fallback_text,
                    "top_drivers": fallback_drivers[:explanation_top_n],
                }
                flat_rows.append(flat_row)

        all_explanations["tickers"][ticker] = ticker_exp

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_explanations, f, indent=2, ensure_ascii=False, default=str)

    flat_df = pd.DataFrame(flat_rows)
    if not flat_df.empty:
        flat_df.to_csv(csv_path, index=False, encoding="utf-8")
    else:
        csv_path.write_text("", encoding="utf-8")

    log.info(
        f"[Explainer] Fold {fold_id}: explicaciones de {len(selected)} tickers "
        f"-> {csv_path.name} y {json_path.name}"
    )

    return csv_path, json_path

def build_strategy_csv(
    signals: pd.DataFrame,
    *,
    fold_id: str = "",
    agent_weights: Optional[Dict[str, float]] = None,
    agent_hit_rates: Optional[Dict[str, float]] = None,
    extra_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Assemble the full strategy output DataFrame (all stocks).

    Parameters
    ----------
    signals:
        Output of the full strategy pipeline (signal_generation â†’
        confidence_model â†’ portfolio_selection â†’ backtesting_engine).
        Must contain ``ticker``.
    fold_id:
        Identifier for the current fold (e.g. ``"2023Q4"``).
    agent_weights:
        Mapping agent_col â†’ dynamic weight.  Added as ``weight_<agent>``
        columns for the debug dataset.
    agent_hit_rates:
        Mapping agent_col â†’ EWMA hit rate.  Added as ``hit_rate_<agent>``
        columns.
    extra_cols:
        Additional column names from *signals* to include verbatim.

    Returns
    -------
    pd.DataFrame ready to be written with :meth:`pandas.DataFrame.to_csv`.
    """
    df = signals.copy()

    # Ensure fold_id column
    df["fold_id"] = str(fold_id)

    # Agent weight & hit-rate columns (debug layer) â€” skip if already present
    if agent_weights:
        for col, w in agent_weights.items():
            dest = f"weight_{col}"
            if dest not in df.columns:
                df[dest] = round(float(w), 6)
    if agent_hit_rates:
        for col, hr in agent_hit_rates.items():
            dest = f"hit_rate_{col}"
            if dest not in df.columns:
                df[dest] = round(float(hr), 6)

    # Determine column order
    base_cols = [c for c in _COLUMN_ORDER if c in df.columns]

    # Agent score columns (e.g. fundamental_score, momentum_score, â€¦)
    score_cols = sorted([c for c in df.columns if c.endswith("_score")])

    # Agent weight / hit-rate columns
    metric_cols = sorted(
        [c for c in df.columns if c.startswith("weight_") or c.startswith("hit_rate_")]
    )

    # Any user-supplied extra columns that aren't already included
    extra = [c for c in (extra_cols or []) if c in df.columns and c not in base_cols + score_cols + metric_cols]

    ordered_cols = base_cols + score_cols + metric_cols + extra
    # Add any remaining columns not explicitly ordered
    ordered_set = set(ordered_cols)
    remaining = [c for c in df.columns if c not in ordered_set]
    final_cols = ordered_cols + remaining

    # Deduplicate while preserving order (in case df.columns has dupes)
    seen: set = set()
    deduped = []
    for c in final_cols:
        if c not in seen:
            seen.add(c)
            deduped.append(c)

    return df[[c for c in deduped if c in df.columns]].reset_index(drop=True)


def export_strategy_csv(
    signals: pd.DataFrame,
    output_path: str | Path,
    *,
    fold_id: str = "",
    agent_weights: Optional[Dict[str, float]] = None,
    agent_hit_rates: Optional[Dict[str, float]] = None,
    extra_cols: Optional[List[str]] = None,
) -> Path:
    """Write the strategy output to a CSV file and return the path.

    Parameters
    ----------
    signals:
        Strategy pipeline output (all stocks).
    output_path:
        Destination file path (e.g. ``results/strategy/2023Q4_output.csv``).
    fold_id, agent_weights, agent_hit_rates, extra_cols:
        Forwarded to :func:`build_strategy_csv`.

    Returns
    -------
    Resolved :class:`pathlib.Path` of the written file.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_df = build_strategy_csv(
        signals,
        fold_id=fold_id,
        agent_weights=agent_weights,
        agent_hit_rates=agent_hit_rates,
        extra_cols=extra_cols,
    )
    csv_df.to_csv(out_path, index=False)
    return out_path

def generate_text_report(
	summary: Dict,
	fold_results: List[Dict],
	agent_diag_history: Dict[str, List],
	backtest_results_dir: str,
) -> None:
	lines = []
	sep = "=" * 65
	sep_s = "-" * 65

	lines.append(sep)
	lines.append("  RESULTS REPORT â€” Walk-Forward Backtest")
	lines.append(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	lines.append(sep)

	lines.append("\n  GLOBAL METRICS (all folds concatenated)")
	lines.append(sep_s)
	lines.append(f"  Folds completados:         {summary.get('n_folds', 0)}")
	lines.append(f"  Alpha medio por fold:      {summary.get('mean_alpha', 0):+.2%}")
	lines.append(f"  Folds con alpha positivo:  {summary.get('pct_folds_positive_alpha', 0):.0%}")

	gs = summary.get("global_strategy_sharpe", 0)
	gb = summary.get("global_benchmark_sharpe", 0)
	lines.append(f"  Sharpe estrategia:         {gs:.3f}")
	lines.append(f"  Sharpe benchmark (S&P500): {gb:.3f}")
	lines.append(f"  Sortino estrategia:        {summary.get('global_strategy_sortino', 0):.3f}")
	lines.append(f"  Max Drawdown estrategia:   {summary.get('global_strategy_max_drawdown', 0):.2%}")
	lines.append(f"  Max Drawdown benchmark:    {summary.get('global_benchmark_max_drawdown', 0):.2%}")
	lines.append(f"  Calmar ratio:              {summary.get('global_strategy_calmar', 0):.3f}")
	lines.append(f"  Volatilidad anualizada:    {summary.get('global_strategy_volatility', 0):.2%}")

	if fold_results:
		lines.append("\n  DETALLE POR FOLD")
		lines.append(sep_s)
		header = (
			f"  {'Fold':>4}  {'Train':>4}Y  "
			f"{'Periodo Test':<24}  "
			f"{'Ret Strat':>9}  {'Ret Bench':>9}  "
			f"{'Alpha':>7}  {'Sharpe':>6}  {'AUC':>6}"
		)
		lines.append(header)
		lines.append("  " + "-" * 63)
		for fr in fold_results:
			test_period = f"{fr.get('test_start','')} -> {fr.get('test_end','')}"
			strat_ret = fr.get("strategy_cumulative_return", 0)
			bench_ret = fr.get("benchmark_cumulative_return", 0)
			alpha_v = fr.get("alpha", 0)
			sharpe_v = fr.get("strategy_sharpe", 0)
			auc_v = fr.get("roc_auc", fr.get("auc", float("nan")))
			train_y = fr.get("train_years", "?")
			fold_id = fr.get("fold", "?")
			auc_str = f"{auc_v:.3f}" if isinstance(auc_v, float) and not pd.isna(auc_v) else "  N/A"
			lines.append(
				f"  {fold_id:>4}  {train_y:>4}Y  "
				f"{test_period:<24}  "
				f"{strat_ret:>+9.2%}  {bench_ret:>+9.2%}  "
				f"{alpha_v:>+7.2%}  {sharpe_v:>6.3f}  {auc_str:>6}"
			)

	by_train = summary.get("by_train_years", {})
	if by_train:
		lines.append("\n  DESGLOSE POR LONGITUD DE TRAIN")
		lines.append(sep_s)
		lines.append(f"  {'Train':>5}  {'N folds':>7}  {'Ret medio':>9}  {'Alpha medio':>11}  {'a>0':>5}  {'Sharpe':>6}")
		lines.append("  " + "-" * 50)
		for ny, stats in sorted(by_train.items()):
			lines.append(
				f"  {ny:>4}Y  {stats['n_folds']:>7}  "
				f"{stats['mean_strategy_return']:>+9.2%}  "
				f"{stats['mean_alpha']:>+11.2%}  "
				f"{stats['pct_positive_alpha']:>5.0%}  "
				f"{stats['mean_strategy_sharpe']:>6.3f}"
			)

	lines.append("\n  AGENT AUC (last trained fold)")
	lines.append(sep_s)
	for ag_name, history in agent_diag_history.items():
		if not history:
			continue
		last = history[-1]
		cv = last.get("cv_metrics") or last.get("cv_lr") or {}
		auc = cv.get("mean_auc", None)
		std = cv.get("std_auc", None)
		if auc is not None:
			std_str = f" Â± {std:.4f}" if std is not None else ""
			lines.append(f"  {ag_name:<15}  AUC = {auc:.4f}{std_str}")

	lines.append(f"\n{sep}")
	lines.append(f"  Results saved to: {backtest_results_dir}/")
	lines.append(sep)

	report_text = "\n".join(lines)
	report_path = Path(backtest_results_dir) / "report.txt"
	report_path.parent.mkdir(parents=True, exist_ok=True)
	with open(report_path, "w", encoding="utf-8") as f:
		f.write(report_text)

