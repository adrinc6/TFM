"""Fold report generation — consolidated CSV of per-fold results.

Produces two files per completed fold and a global consolidated file at the end:

1. results/agents/fold_{N}_scores.csv
   One row per ticker. Columns:
     - Identification: year_quarter, fold, ticker, sector, industry
     - Selection: selected, rank, selection_reason
     - Scores from each agent (0–1) with human-readable interpretation
     - Final score and prediction
     - Realised result (if available): actual_return, beat_benchmark
     - Per-agent explanation: which factors were in favour / against,
       with actual metric values in natural language

2. results/agents/all_folds_scores.csv
   Concatenation of all fold_{N}_scores.csv. Enables filtering by quarter,
   comparing tickers over time, and auditing per-agent decisions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from module.steps.step_04_evaluation.explainability import (
    FEATURE_DESCRIPTIONS,
    _format_value,
)
log = logging.getLogger(__name__)

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
        "rsi", "macd", "beta", "zscore", "zsector", "pct", "coverage",
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
        str: Formatted string like "0.72 — Strong financial health".
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
    return f"{score:.2f} — {labels[tier]}"


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
    """Generates a short natural-language phrase for a feature–value pair.

    Example: roe=0.28 → "Return on Equity = 28.0%"

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
      - Realised result if available.

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
            "confidence":       "High" if abs(final_score - 0.5) > 0.25 else "Moderate",

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

        # --- Realised result (if the test period has elapsed) ---
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
    path = out_dir / f"quarter_{fold_id}_scores.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    log.info(f"[FoldReport] Fold {fold_id}: scores for {len(df)} tickers saved → {path.name}")
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
    path = out_dir / f"quarter_{year_quarter}_ticker_snapshot_audit.csv"

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
    log.info(f"[FoldReport] Snapshot audit {year_quarter}: {len(out_df)} tickers → {path.name}")
    return path


def export_quarter_agent_feature_audit(
    df_test_scored: pd.DataFrame,
    agents: Dict,
    year_quarter: str,
    agents_results_dir: str,
) -> Path:
    """Exports granular agent–feature traceability for a quarter.

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
    path = out_dir / f"quarter_{year_quarter}_ticker_agent_feature_audit.csv"

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
    log.info(f"[FoldReport] Agent-feature audit {year_quarter}: {len(rows)} rows → {path.name}")
    return path


def export_all_folds_scores(
    agents_results_dir: str,
) -> Optional[Path]:
    """Concatenates all quarter_*_scores.csv files into all_folds_scores.csv.

    Should be called at the end of the pipeline after all folds complete.

    Args:
        agents_results_dir (str): Root output directory containing per-fold CSV files.

    Returns:
        Optional[Path]: Path to the consolidated CSV, or None if no fold files
            were found.
    """
    out_dir = Path(agents_results_dir)
    fold_files = sorted(out_dir.glob("quarter_*_scores.csv"))
    if not fold_files:
        log.warning("[FoldReport] No quarter_*_scores.csv files found to consolidate.")
        return None

    dfs = []
    for f in fold_files:
        try:
            dfs.append(pd.read_csv(f, encoding="utf-8"))
        except Exception as e:
            log.warning(f"[FoldReport] Error reading {f.name}: {e}")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    path = out_dir / "all_folds_scores.csv"
    combined.to_csv(path, index=False, encoding="utf-8")
    log.info(
        f"[FoldReport] Consolidated CSV for all folds: "
        f"{len(combined)} rows | "
        f"{combined['year_quarter'].nunique()} quarters | "
        f"{combined['ticker'].nunique()} unique tickers → {path.name}"
    )
    return path
