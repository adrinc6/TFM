"""Analyze one ticker in one quarter using exported walk-forward artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _qnorm(q: str) -> str:
    s = str(q).strip().upper().replace(" ", "")
    if "Q" not in s:
        raise ValueError("Invalid quarter. Use YYYYQn format, for example 2026Q1")
    return s


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.DataFrame()


def _to_float(v, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _bool_text(v: bool) -> str:
    return "YES" if bool(v) else "NO"


def _quarter_from_date_str(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    try:
        ts = pd.Timestamp(value)
        return f"{ts.year}Q{ts.quarter}"
    except Exception:
        return None


def _load_artifacts(results_dir: Path, quarter: str) -> Dict[str, pd.DataFrame]:
    return {
        "scores": _safe_read_csv(results_dir / f"quarter_{quarter}_scores.csv"),
        "snapshot": _safe_read_csv(results_dir / f"quarter_{quarter}_ticker_snapshot_audit.csv"),
        "agent_feature": _safe_read_csv(results_dir / f"quarter_{quarter}_ticker_agent_feature_audit.csv"),
        "expl": _safe_read_csv(results_dir / f"quarter_{quarter}_ticker_explanations.csv"),
        "all_scores": _safe_read_csv(results_dir / "all_folds_scores.csv"),
    }


def _find_ticker_row(df: pd.DataFrame, ticker: str) -> pd.Series | None:
    if df.empty or "ticker" not in df.columns:
        return None
    subset = df[df["ticker"].astype(str).str.upper() == ticker.upper()]
    if subset.empty:
        return None
    return subset.iloc[0]


def _is_ratio_or_normalized_feature(feature: str) -> bool:
    """Strict filter: only ratio/normalized-like features are allowed in analysis output."""
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
