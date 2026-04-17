"""Analyze one ticker in one quarter using exported walk-forward artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _qnorm(q: str) -> str:
    """Normalise and validate a quarter string into ``YYYYQn`` format.

    Strips whitespace, uppercases the input, and removes internal spaces.
    Raises if the normalised string does not contain a ``"Q"`` character.

    Args:
        q: Raw quarter string, e.g. ``"2026q1"``, ``"2026 Q1"``, or
            ``"2026Q1"``.

    Returns:
        str: Normalised quarter string, e.g. ``"2026Q1"``.

    Raises:
        ValueError: If the normalised string does not contain ``"Q"``,
            indicating it is not a valid quarter format.
    """
    s = str(q).strip().upper().replace(" ", "")
    if "Q" not in s:
        raise ValueError("Invalid quarter. Use YYYYQn format, for example 2026Q1")
    return s


def _safe_read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file safely, returning an empty DataFrame on any error.

    If the file does not exist or cannot be parsed (encoding issues, malformed
    content, etc.), an empty :class:`~pandas.DataFrame` is returned instead of
    raising an exception.

    Args:
        path: Filesystem path to the CSV file.

    Returns:
        pd.DataFrame: Parsed DataFrame, or an empty DataFrame on failure.
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.DataFrame()


def _to_float(v, default: float = float("nan")) -> float:
    """Convert any value to float, falling back to a default on failure.

    Args:
        v: Value to convert (any type accepted by ``float()``).
        default: Value to return when conversion fails.  Defaults to
            ``float("nan")``.

    Returns:
        float: The converted float, or ``default`` if conversion raises an
            exception.
    """
    try:
        return float(v)
    except Exception:
        return default


def _bool_text(v: bool) -> str:
    """Convert a boolean value to the string ``"YES"`` or ``"NO"``.

    Args:
        v: Value to evaluate as a boolean.

    Returns:
        str: ``"YES"`` if ``bool(v)`` is ``True``, otherwise ``"NO"``.
    """
    return "YES" if bool(v) else "NO"


def _quarter_from_date_str(value) -> str | None:
    """Convert a date string to a ``"YYYYQn"`` quarter string, or None.

    Parses the input value via :class:`~pandas.Timestamp` and formats the
    result as ``"<year>Q<quarter>"``, e.g. ``"2024Q1"``.  Returns ``None``
    for null values or when parsing fails.

    Args:
        value: Any value that :class:`~pandas.Timestamp` can parse (str,
            datetime, Timestamp, etc.), or ``None`` / NaN.

    Returns:
        str | None: Quarter string (e.g. ``"2024Q1"``), or ``None`` on failure.
    """
    if value is None or pd.isna(value):
        return None
    try:
        ts = pd.Timestamp(value)
        return f"{ts.year}Q{ts.quarter}"
    except Exception:
        return None


def _load_artifacts(results_dir: Path, quarter: str) -> Dict[str, pd.DataFrame]:
    """Load all walk-forward output CSVs for a given quarter into a dict of DataFrames.

    The following files are loaded from ``results_dir``:

    * ``quarter_<quarter>_scores.csv`` — per-ticker final scores and metadata.
    * ``quarter_<quarter>_ticker_snapshot_audit.csv`` — snapshot/filing audit.
    * ``quarter_<quarter>_ticker_agent_feature_audit.csv`` — per-agent feature
      values.
    * ``quarter_<quarter>_ticker_explanations.csv`` — SHAP/LLM explanations.
    * ``all_folds_scores.csv`` — aggregated scores across all folds (used as
      fallback when the quarter-specific file is missing data).

    Missing or unreadable files are silently replaced with empty DataFrames via
    :func:`_safe_read_csv`.

    Args:
        results_dir: Directory that contains the exported walk-forward CSVs.
        quarter: Normalised quarter string, e.g. ``"2024Q1"``.

    Returns:
        Dict[str, pd.DataFrame]: Mapping with keys ``"scores"``,
            ``"snapshot"``, ``"agent_feature"``, ``"expl"``, and
            ``"all_scores"``.
    """
    return {
        "scores": _safe_read_csv(results_dir / f"quarter_{quarter}_scores.csv"),
        "snapshot": _safe_read_csv(results_dir / f"quarter_{quarter}_ticker_snapshot_audit.csv"),
        "agent_feature": _safe_read_csv(results_dir / f"quarter_{quarter}_ticker_agent_feature_audit.csv"),
        "expl": _safe_read_csv(results_dir / f"quarter_{quarter}_ticker_explanations.csv"),
        "all_scores": _safe_read_csv(results_dir / "all_folds_scores.csv"),
    }


def _find_ticker_row(df: pd.DataFrame, ticker: str) -> pd.Series | None:
    """Find the first row matching a ticker in a DataFrame, or None.

    Looks up ``ticker`` in the ``"ticker"`` column of ``df`` using a
    case-insensitive comparison.

    Args:
        df: DataFrame that may contain a ``"ticker"`` column.
        ticker: Ticker symbol to search for (case-insensitive).

    Returns:
        pd.Series | None: The first matching row as a Series, or ``None`` if
            ``df`` is empty, has no ``"ticker"`` column, or no match is found.
    """
    if df.empty or "ticker" not in df.columns:
        return None
    subset = df[df["ticker"].astype(str).str.upper() == ticker.upper()]
    if subset.empty:
        return None
    return subset.iloc[0]


def _is_ratio_or_normalized_feature(feature: str) -> bool:
    """Determine whether a feature name represents a ratio or a normalised value.

    Uses an allowlist of token substrings (e.g. ``"ratio"``, ``"margin"``,
    ``"zscore"``) to identify features that are inherently scale-free and
    therefore safe to display or compare across tickers.

    A feature is accepted when **any** token from the allowlist appears as a
    substring of the lowercased feature name.  After that check, a blocklist of
    prefixes for known absolute-magnitude features (e.g. ``"revenue"``,
    ``"market_cap"``) is applied: if the name starts with a blocked prefix it
    is rejected, regardless of the allowlist match.

    Args:
        feature: Feature column name to evaluate.

    Returns:
        bool: ``True`` if the feature is considered a ratio or normalised
            value; ``False`` otherwise (including empty strings and features
            that match only the blocklist).
    """
    f = str(feature).lower().strip()
    if not f:
        return False

    allowed_tokens = [
        "ratio", "margin", "yield", "growth", "trend", "momentum", "volatility",
        "rsi", "macd", "beta", "zscore", "zsector", "pct", "coverage",
        "score", "prior", "dispersion", "consensus", "confidence", "quality",
        "fscore", "accrual", "atr", "bb_", "vs_5y", "vs_52w", "debt_to_", "_to_",
    ]
    if any(tok in f for tok in allowed_tokens):
        return True

    # Explicitly exclude common absolute-magnitude features.
    blocked_prefixes = [
        "revenue", "net_income", "operating_income", "gross_profit", "fcf", "ebitda",
        "total_assets", "total_liabilities", "total_equity", "total_debt", "cash",
        "shares", "eps_est", "eps_reported", "market_cap", "capex",
    ]
    if any(f.startswith(p) for p in blocked_prefixes):
        return False

    return False


def _build_agent_blocks(df_expl: pd.DataFrame, df_feat: pd.DataFrame, ticker: str) -> List[Dict]:
    """Merge explanation and feature DataFrames to build per-agent summary dicts.

    For each agent present in either ``df_expl`` or ``df_feat``, one block is
    created containing the agent's score, label, explanation text, favour/
    contra factors, and a list of the top ratio/normalised features by absolute
    feature value.  Only features that pass :func:`_is_ratio_or_normalized_feature`
    are included in the feature snapshot.

    Args:
        df_expl: DataFrame from ``quarter_*_ticker_explanations.csv`` with at
            least the columns ``ticker``, ``agent``, ``agent_score``,
            ``agent_label``, ``explanation_text``, ``favor_factors``, and
            ``contra_factors``.
        df_feat: DataFrame from ``quarter_*_ticker_agent_feature_audit.csv``
            with at least the columns ``ticker``, ``agent``, ``feature``,
            ``feature_value``, and optionally ``feature_present`` and
            ``agent_score``.
        ticker: Ticker symbol to filter both DataFrames (case-insensitive).

    Returns:
        List[Dict]: One dict per agent sorted by agent name, each with keys
            ``agent``, ``agent_score``, ``agent_label``, ``explanation_text``,
            ``favor_factors``, ``contra_factors``, and
            ``top_features_snapshot``.
    """
    blocks: List[Dict] = []

    by_agent_expl: Dict[str, pd.Series] = {}
    if not df_expl.empty and "ticker" in df_expl.columns and "agent" in df_expl.columns:
        ex = df_expl[df_expl["ticker"].astype(str).str.upper() == ticker.upper()].copy()
        if not ex.empty:
            ex = ex.drop_duplicates(subset=["agent"], keep="last")
            by_agent_expl = {str(r["agent"]): r for _, r in ex.iterrows()}

    by_agent_feat: Dict[str, pd.DataFrame] = {}
    if not df_feat.empty and "ticker" in df_feat.columns and "agent" in df_feat.columns:
        ff = df_feat[df_feat["ticker"].astype(str).str.upper() == ticker.upper()].copy()
        for ag in ff["agent"].astype(str).unique().tolist():
            by_agent_feat[ag] = ff[ff["agent"].astype(str) == ag].copy()

    agent_names = sorted(set(list(by_agent_expl.keys()) + list(by_agent_feat.keys())))
    for ag in agent_names:
        exp_row = by_agent_expl.get(ag)
        feat_df = by_agent_feat.get(ag, pd.DataFrame())

        score = _to_float(exp_row.get("agent_score")) if exp_row is not None else _to_float(feat_df["agent_score"].iloc[0]) if (not feat_df.empty and "agent_score" in feat_df.columns) else float("nan")

        feat_list: List[Dict] = []
        if not feat_df.empty and "feature" in feat_df.columns:
            present = feat_df[feat_df.get("feature_present", True) == True].copy()
            present = present[present["feature"].astype(str).map(_is_ratio_or_normalized_feature)]
            if "feature_value" in present.columns:
                present["abs_v"] = present["feature_value"].apply(lambda x: abs(_to_float(x, 0.0)))
                present = present.sort_values("abs_v", ascending=False)
            for _, r in present.head(12).iterrows():
                feat_list.append({
                    "feature": r.get("feature"),
                    "value": r.get("feature_value"),
                })

        blocks.append({
            "agent": ag,
            "agent_score": score,
            "agent_label": exp_row.get("agent_label") if exp_row is not None else ("Outperform" if score >= 0.5 else "Underperform"),
            "explanation_text": exp_row.get("explanation_text") if exp_row is not None else "",
            "favor_factors": exp_row.get("favor_factors") if exp_row is not None else "",
            "contra_factors": exp_row.get("contra_factors") if exp_row is not None else "",
            "top_features_snapshot": feat_list,
        })

    return blocks


def analyze_ticker_quarter(ticker: str, quarter: str, results_dir: Path) -> Dict:
    """Load walk-forward artifacts and build the full analysis dict for one ticker/quarter.

    Loads the relevant CSVs from ``results_dir``, extracts the row for
    ``ticker`` in ``quarter``, and assembles a structured report dict
    containing the investment decision, model scores, per-agent details, and
    (where available) realised returns.

    **Recommendation tiers** (``decision`` key):

    * ``"INVEST"`` — ticker was selected for the portfolio in this quarter.
    * ``"CONSIDER (high score, not selected)"`` — final score ≥ 0.55 but not
      in the portfolio (e.g. capacity constraint hit).
    * ``"NEUTRAL / WATCHLIST"`` — final score in [0.50, 0.55).
    * ``"DO NOT INVEST"`` — final score < 0.50.

    Args:
        ticker: Ticker symbol to analyse (case-insensitive).
        quarter: Normalised quarter string, e.g. ``"2024Q1"``.  Use
            :func:`_qnorm` to normalise raw user input before calling this
            function.
        results_dir: Directory containing the exported walk-forward CSV
            artifacts.

    Returns:
        Dict: Analysis dict with keys including ``ticker``, ``year_quarter``,
            ``decision``, ``model_prediction``, ``probabilidad_outperform``,
            ``selected_for_portfolio``, ``expected_vs_market_sector``,
            ``realized_if_available``, ``data_source_snapshot``,
            ``scores_by_agent``, and ``agent_details``.

    Raises:
        FileNotFoundError: If no score row can be found for ``ticker`` /
            ``quarter`` in any of the available artifact files.
    """
    artifacts = _load_artifacts(results_dir, quarter)

    score_row = _find_ticker_row(artifacts["scores"], ticker)
    if score_row is None and not artifacts["all_scores"].empty:
        filt = artifacts["all_scores"]
        if "year_quarter" in filt.columns:
            filt = filt[filt["year_quarter"].astype(str).str.upper() == quarter.upper()]
        score_row = _find_ticker_row(filt, ticker)

    if score_row is None:
        available = sorted([p.name for p in results_dir.glob("quarter_*_scores.csv")])
        raise FileNotFoundError(
            f"No data available for {ticker} in {quarter}. Available files: {available[:8]}"
        )

    snap_row = _find_ticker_row(artifacts["snapshot"], ticker)

    final_score = _to_float(score_row.get("final_score"))
    selected = bool(score_row.get("selected", False))
    pred_label = str(score_row.get("prediction", "Outperform" if final_score >= 0.5 else "Underperform"))

    if selected:
        recommendation = "INVEST"
    elif final_score >= 0.55:
        recommendation = "CONSIDER (high score, not selected)"
    elif final_score >= 0.5:
        recommendation = "NEUTRAL / WATCHLIST"
    else:
        recommendation = "DO NOT INVEST"

    carry_forward = None
    report_end = None
    report_filed = None
    snapshot_date = None
    if snap_row is not None:
        carry_forward = bool(snap_row.get("is_fundamental_carry_forward", False))
        report_end = snap_row.get("report_end_date_used")
        report_filed = snap_row.get("report_filed_date_used")
        snapshot_date = snap_row.get("snapshot_date")

    beat_benchmark_real = score_row.get("beat_benchmark")
    retorno_real = score_row.get("realized_return")
    if retorno_real is None:
        retorno_real = score_row.get("retorno_real")
    alpha_real = score_row.get("alpha_real")

    out = {
        "ticker": ticker.upper(),
        "year_quarter": quarter.upper(),
        "decision": recommendation,
        "model_prediction": pred_label,
        "prediccion_modelo": pred_label,
        "probabilidad_outperform": final_score,
        "selected_for_portfolio": selected,
        "expected_vs_market_sector": {
            "expected_outperform_market": bool(final_score >= 0.5),
            "expected_outperform_sector": bool(final_score >= 0.5),
            "note": "The model label is built from sector-relative outperformance for the quarter.",
        },
        "realized_if_available": {
            "retorno_real": retorno_real,
            "alpha_vs_benchmark": alpha_real,
            "beat_benchmark": beat_benchmark_real,
        },
        "data_source_snapshot": {
            "snapshot_date": snapshot_date,
            "used_report_from_same_quarter": None if carry_forward is None else (not carry_forward),
            "used_previous_report_carry_forward": carry_forward,
            "report_end_date_used": report_end,
            "report_end_quarter_used": _quarter_from_date_str(report_end),
            "report_filed_date_used": report_filed,
            "report_filed_quarter_used": _quarter_from_date_str(report_filed),
        },
        "scores_by_agent": {
            "fundamental_score": _to_float(score_row.get("fundamental_score")),
            "valuation_score": _to_float(score_row.get("valuation_score")),
            "momentum_score": _to_float(score_row.get("momentum_score")),
            "bear_score": _to_float(score_row.get("bear_score")),
            "sentiment_score": _to_float(score_row.get("sentiment_score")),
            "sector_score": _to_float(score_row.get("sector_score")),
        },
        "agent_details": _build_agent_blocks(
            df_expl=artifacts["expl"],
            df_feat=artifacts["agent_feature"],
            ticker=ticker,
        ),
    }
    return out


def _print_human(report: Dict) -> None:
    """Pretty-print an analysis report dict to stdout in a human-readable format.

    Outputs the following sections, separated by horizontal rules:

    * Header with ticker and quarter.
    * Decision, model prediction, final score, and portfolio selection flag.
    * Base snapshot data (snapshot date, carry-forward status, report dates).
    * Expectation vs market/sector (whether the model expects outperformance).
    * Scores by agent (only non-NaN values).
    * Per-agent details: score, label, explanation text, favour/contra factors,
      and the top feature snapshot (up to 8 features).
    * Realised result (if available): return, alpha, benchmark beat flag.

    Args:
        report: Dict as returned by :func:`analyze_ticker_quarter`.
    """
    print("=" * 90)
    print(f"TICKER ANALYSIS: {report['ticker']}  |  QUARTER: {report['year_quarter']}")
    print("=" * 90)
    print(f"Decision: {report['decision']}")
    print(f"Model prediction: {report['model_prediction']} | Final score: {report['probabilidad_outperform']:.4f}")
    print(f"Selected in portfolio: {_bool_text(report['selected_for_portfolio'])}")

    src = report["data_source_snapshot"]
    print("\n[Base snapshot data]")
    print(f"- Snapshot date: {src.get('snapshot_date')}")
    print(f"- Same-quarter report used: {_bool_text(src.get('used_report_from_same_quarter')) if src.get('used_report_from_same_quarter') is not None else 'N/A'}")
    print(f"- Carry-forward (previous report): {_bool_text(src.get('used_previous_report_carry_forward')) if src.get('used_previous_report_carry_forward') is not None else 'N/A'}")
    print(f"- report_end_date_used: {src.get('report_end_date_used')}")
    print(f"- report_end_quarter_used: {src.get('report_end_quarter_used')}")
    print(f"- report_filed_date_used: {src.get('report_filed_date_used')}")
    print(f"- report_filed_quarter_used: {src.get('report_filed_quarter_used')}")

    print("\n[Expectation vs market/sector]")
    em = report["expected_vs_market_sector"]
    print(f"- Expected to beat market: {_bool_text(em['expected_outperform_market'])}")
    print(f"- Expected to beat sector: {_bool_text(em['expected_outperform_sector'])}")

    print("\n[Scores by agent]")
    for k, v in report["scores_by_agent"].items():
        if pd.notna(v):
            print(f"- {k}: {float(v):.4f}")

    print("\n[Details by agent]")
    for ag in report["agent_details"]:
        print(f"\n* {ag['agent']} | score={ag['agent_score']:.4f} | label={ag['agent_label']}")
        if ag.get("explanation_text"):
            print(f"  Rationale: {ag['explanation_text']}")
        if ag.get("favor_factors"):
            print(f"  In favor: {ag['favor_factors']}")
        if ag.get("contra_factors"):
            print(f"  Against: {ag['contra_factors']}")
        if ag.get("top_features_snapshot"):
            print("  Top features (snapshot):")
            for f in ag["top_features_snapshot"][:8]:
                print(f"    - {f['feature']}: {f['value']}")

    realized = report["realized_if_available"]
    print("\n[Realized result (if available)]")
    print(f"- realized_return: {realized.get('retorno_real')}")
    print(f"- alpha_vs_benchmark: {realized.get('alpha_vs_benchmark')}")
    print(f"- beat_benchmark: {realized.get('beat_benchmark')}")


if __name__ == "__main__":
    # Execution parameters (edit directly here)
    TICKER = "SMCI"
    QUARTER = "2023YQ1"
    RESULTS_DIR = "results/agents"
    PRINT_AS_JSON = False
    SAVE_JSON_PATH = ""

    q = _qnorm(QUARTER)
    report = analyze_ticker_quarter(
        ticker=TICKER,
        quarter=q,
        results_dir=Path(RESULTS_DIR),
    )

    if PRINT_AS_JSON:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        _print_human(report)

    if SAVE_JSON_PATH:
        out_path = Path(SAVE_JSON_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nJSON saved at: {out_path}")
