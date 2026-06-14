"""Portfolio intelligence and thesis-review utilities for GARP / Value-Growth.

This module is intentionally independent from the training/ranking core.  It
reuses point-in-time snapshots and GARP score fields when they are available,
and falls back to transparent percentile heuristics when reviewing a master
snapshot that has raw fundamentals but no trained agent scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


THESIS_COMPONENTS = [
    "quality_score",
    "growth_score",
    "valuation_score",
    "fundamental_trend_score",
    "catalyst_score",
    "risk_score",
    "moat_proxy_score",
    "expectation_gap_score",
]

POSITION_OUTPUT_COLUMNS = [
    "ticker",
    "weight",
    "purchase_date",
    "original_snapshot_date",
    "current_snapshot_date",
    "position_health_score",
    "buy_hold_sell_rating",
    "exit_score",
    "conviction_score",
    "thesis_score",
    "thesis_status",
    "thesis_history_trend",
    "valuation_status",
    "opportunity_type",
    "moat_proxy_score",
    "expectation_gap_score",
    "overexpectation_penalty",
    "top_positive_drivers",
    "top_risks",
    "action_recommended",
    "review_priority",
    "exit_reason",
    "latest_thesis_events",
    "thesis_changes",
    "original_buy_reason",
    "original_opportunity_type",
    "original_valuation_status",
    "original_thesis_score",
    "original_scores_json",
    "current_vs_original_summary",
    "buy_today_flag",
    "hold_today_flag",
    "sell_today_flag",
    "best_alternative_ticker",
    "best_alternative_score",
    "opportunity_cost_flag",
]

THESIS_HISTORY_COLUMNS = [
    "ticker",
    "date",
    "thesis_score",
    "position_health_score",
    "conviction_score",
    "valuation_status",
    "quality_score",
    "growth_score",
    "valuation_score",
    "fundamental_trend_score",
    "catalyst_score",
    "risk_score",
    "moat_proxy_score",
    "expectation_gap_score",
    "overexpectation_penalty",
    "thesis_events",
]


def _as_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.MultiIndex) and {"date", "ticker"}.issubset(set(out.index.names)):
        idx_df = out.index.to_frame(index=False)
        out = out.reset_index(drop=True)
        out["date"] = pd.to_datetime(idx_df["date"], errors="coerce")
        out["ticker"] = idx_df["ticker"].astype(str)
    elif "date" not in out.columns or "ticker" not in out.columns:
        raise ValueError("Snapshot DataFrame must contain date and ticker, either as columns or MultiIndex levels.")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    out = out.dropna(subset=["date", "ticker"]).copy()
    return out


def _positions_frame(
    *,
    positions: pd.DataFrame | None = None,
    tickers: Iterable[str] | None = None,
) -> pd.DataFrame:
    if positions is not None and not positions.empty:
        out = positions.copy()
    elif tickers:
        tickers_list = [str(t).strip().upper().replace(".", "-") for t in tickers if str(t).strip()]
        if not tickers_list:
            raise ValueError("No valid tickers were provided for portfolio review.")
        out = pd.DataFrame({"ticker": sorted(set(tickers_list))})
    else:
        raise ValueError("Provide either positions or tickers for portfolio review.")

    if "ticker" not in out.columns:
        raise ValueError("Portfolio review positions must contain a ticker column.")
    out["ticker"] = out["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    if "weight" not in out.columns:
        out["weight"] = 1.0 / max(len(out), 1)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    total_weight = float(out["weight"].sum())
    if total_weight > 0:
        out["weight"] = out["weight"] / total_weight
    if "purchase_date" in out.columns:
        out["purchase_date"] = pd.to_datetime(out["purchase_date"], errors="coerce").dt.normalize()
    else:
        out["purchase_date"] = pd.NaT
    if "snapshot_date" in out.columns:
        out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce").dt.normalize()
    else:
        out["snapshot_date"] = out["purchase_date"]
    return out


def _rank_score(frame: pd.DataFrame, candidates: list[str], *, higher_is_better: bool = True) -> pd.Series:
    values = []
    for col in candidates:
        if col in frame.columns:
            s = pd.to_numeric(frame[col], errors="coerce")
            if s.notna().any():
                rank = s.rank(pct=True, ascending=higher_is_better)
                values.append(rank)
    if not values:
        return pd.Series(0.5, index=frame.index, dtype=float)
    return pd.concat(values, axis=1).mean(axis=1).fillna(0.5).clip(0.0, 1.0)


def _coalesce_score(frame: pd.DataFrame, target_col: str, fallback: pd.Series) -> pd.Series:
    if target_col in frame.columns:
        s = pd.to_numeric(frame[target_col], errors="coerce")
        if s.notna().any():
            return s.fillna(fallback).clip(0.0, 1.0)
    return fallback.fillna(0.5).clip(0.0, 1.0)


def add_portfolio_review_scores(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """Add transparent thesis-review scores to a point-in-time universe snapshot."""
    out = _as_multiindex(snapshot_df)

    quality_fb = _rank_score(out, [
        "gross_margin", "operating_margin", "net_margin", "fcf_margin", "roic", "roe", "roa",
        "cash_conversion", "asset_turnover", "profitability_consistency", "moat_proxy_score",
    ])
    growth_fb = _rank_score(out, [
        "revenue_yoy_growth", "revenue_growth_3y", "revenue_growth_5y", "eps_growth",
        "eps_growth_trend_3y", "fcf_yoy_growth", "fcf_growth_3y", "gross_profit_growth",
        "operating_income_growth", "growth_acceleration",
    ])
    valuation_high = _rank_score(out, ["fcf_yield", "earnings_yield", "shareholder_yield"])
    valuation_low = _rank_score(out, [
        "pe_ratio", "forward_pe", "peg_ratio", "ev_to_ebitda", "ev_to_sales",
        "price_to_fcf", "price_to_sales", "valuation_percentile_sector", "valuation_percentile_universe",
    ], higher_is_better=False)
    valuation_fb = pd.concat([valuation_high, valuation_low], axis=1).mean(axis=1).fillna(0.5).clip(0.0, 1.0)
    trend_fb = _rank_score(out, [
        "roic_trend_2y", "net_margin_trend_2y", "gross_margin_trend_2y", "fcf_margin_trend_2y",
        "revenue_growth_acceleration", "delta_roic", "delta_fcf_margin", "delta_leverage",
        "eps_revision", "analyst_revision_trend",
    ])
    catalyst_fb = _rank_score(out, [
        "catalyst_score", "earnings_surprise", "eps_revision", "analyst_revision_trend",
        "buyback_yield", "insider_sentiment", "sector_momentum", "revenue_growth_acceleration",
        "margin_expansion", "fcf_margin_trend_2y",
    ])
    risk_fb = _rank_score(out, [
        "debt_to_ebitda", "debt_to_equity", "net_debt_to_ebitda", "volatility_60d",
        "max_drawdown_12m", "earnings_volatility", "fcf_volatility", "dilution_rate",
        "overexpectation_penalty",
    ], higher_is_better=False)
    moat_fb = pd.concat([
        _rank_score(out, ["gross_margin", "roic", "fcf_margin"]),
        _rank_score(out, ["gross_margin_stability", "margin_stability", "roic_persistence", "fcf_consistency"]),
    ], axis=1).mean(axis=1).fillna(quality_fb).clip(0.0, 1.0)

    out["quality_score"] = _coalesce_score(out, "quality_score", quality_fb)
    out["growth_score"] = _coalesce_score(out, "growth_score", growth_fb)
    out["valuation_score"] = _coalesce_score(out, "valuation_score", valuation_fb)
    out["fundamental_trend_score"] = _coalesce_score(out, "fundamental_trend_score", trend_fb)
    out["catalyst_score"] = _coalesce_score(out, "catalyst_score", catalyst_fb)
    out["risk_score"] = _coalesce_score(out, "risk_score", _coalesce_score(out, "risk_bear_score", risk_fb))
    out["moat_proxy_score"] = _coalesce_score(out, "moat_proxy_score", moat_fb)

    expectation_fb = (
        0.30 * out["quality_score"]
        + 0.25 * out["growth_score"]
        + 0.20 * out["moat_proxy_score"]
        + 0.25 * out["valuation_score"]
    ).clip(0.0, 1.0)
    out["expectation_gap_score"] = _coalesce_score(out, "expectation_gap_score", expectation_fb)
    out["overexpectation_penalty"] = _coalesce_score(
        out,
        "overexpectation_penalty",
        (1.0 - out["valuation_score"]).where(out["growth_score"] >= 0.60, (1.0 - out["valuation_score"]) * 0.7),
    )
    out["thesis_score"] = (
        0.20 * out["quality_score"]
        + 0.18 * out["growth_score"]
        + 0.15 * out["fundamental_trend_score"]
        + 0.12 * out["valuation_score"]
        + 0.12 * out["moat_proxy_score"]
        + 0.10 * out["expectation_gap_score"]
        + 0.08 * out["catalyst_score"]
        + 0.05 * out["risk_score"]
    ).clip(0.0, 1.0)
    return out


def _latest_universe_snapshot(df: pd.DataFrame, review_date: pd.Timestamp | None) -> pd.DataFrame:
    if df.empty:
        return df
    dates = sorted(pd.to_datetime(df["date"].dropna().unique()))
    if not dates:
        return df.head(0)
    anchor = pd.Timestamp(review_date).normalize() if review_date is not None and pd.notna(review_date) else pd.Timestamp(dates[-1]).normalize()
    eligible_dates = [d for d in dates if pd.Timestamp(d) <= anchor]
    snap_date = pd.Timestamp(eligible_dates[-1] if eligible_dates else dates[0]).normalize()
    return df[df["date"] == snap_date].copy()


def _ticker_snapshot(df: pd.DataFrame, ticker: str, anchor_date: pd.Timestamp | None) -> pd.Series | None:
    subset = df[df["ticker"] == ticker].sort_values("date")
    if subset.empty:
        return None
    if anchor_date is not None and pd.notna(anchor_date):
        eligible = subset[subset["date"] <= pd.Timestamp(anchor_date).normalize()]
        if not eligible.empty:
            return eligible.iloc[-1]
    return subset.iloc[-1]


def _valuation_status(row: pd.Series) -> str:
    valuation = float(row.get("valuation_score", 0.5))
    gap = float(row.get("expectation_gap_score", 0.5))
    overexp = float(row.get("overexpectation_penalty", 0.5))
    composite = 0.55 * valuation + 0.30 * gap + 0.15 * (1.0 - overexp)
    if composite >= 0.72:
        return "Undervalued"
    if composite >= 0.55:
        return "Fairly Valued"
    if composite >= 0.42:
        return "Fully Valued"
    if composite >= 0.28:
        return "Overvalued"
    return "Extremely Overvalued"


def _position_health_from_row(row: pd.Series, original: pd.Series | None = None) -> int:
    health_raw = float(row.get("thesis_score", 0.5))
    if original is not None:
        health_raw += 0.15 * (float(row.get("thesis_score", 0.5)) - float(original.get("thesis_score", 0.5)))
    return int(round(max(0.0, min(1.0, health_raw)) * 100))


def _conviction_score(row: pd.Series | Mapping[str, object], thesis_status: str, better_opportunity: bool = False) -> int:
    """Confidence to keep owning the position today; separate from buy ranking."""
    status_bonus = {
        "Improving": 0.12,
        "Intact": 0.06,
        "Maturing": -0.03,
        "Weakening": -0.16,
        "Broken": -0.35,
    }.get(str(thesis_status), 0.0)
    valuation_state = str(row.get("valuation_status", _valuation_status(pd.Series(row))))
    valuation_adj = {
        "Undervalued": 0.08,
        "Fairly Valued": 0.03,
        "Fully Valued": -0.06,
        "Overvalued": -0.14,
        "Extremely Overvalued": -0.25,
        "Unknown": -0.10,
    }.get(valuation_state, 0.0)
    base = (
        0.22 * float(row.get("quality_score", 0.5))
        + 0.18 * float(row.get("moat_proxy_score", 0.5))
        + 0.16 * float(row.get("growth_score", 0.5))
        + 0.13 * float(row.get("fundamental_trend_score", 0.5))
        + 0.10 * float(row.get("valuation_score", 0.5))
        + 0.08 * float(row.get("expectation_gap_score", 0.5))
        + 0.06 * float(row.get("catalyst_score", 0.5))
        + 0.07 * float(row.get("risk_score", 0.5))
    )
    penalty = 0.08 if better_opportunity else 0.0
    score = base + status_bonus + valuation_adj - penalty
    return int(round(max(0.0, min(1.0, score)) * 100))


def _detect_thesis_events(row: pd.Series, prev: pd.Series | None = None) -> list[str]:
    """Detect material quarter/month changes without introducing new models."""
    events: list[str] = []
    if prev is None:
        return events

    def delta(col: str) -> float:
        return float(row.get(col, 0.5)) - float(prev.get(col, 0.5))

    checks = [
        ("quality_score", "Quality Upgrade", "Quality Deterioration"),
        ("growth_score", "Growth Acceleration", "Growth Slowdown"),
        ("fundamental_trend_score", "Thesis Improvement", "Thesis Deterioration"),
        ("catalyst_score", "Catalyst Strengthening", "Catalyst Exhausted"),
        ("moat_proxy_score", "Moat Strengthening", "Moat Deterioration"),
        ("expectation_gap_score", "Mispricing Widening", "Expectation Gap Closing"),
    ]
    for col, up_event, down_event in checks:
        d = delta(col)
        if d >= 0.08:
            events.append(up_event)
        elif d <= -0.08:
            events.append(down_event)

    thesis_d = delta("thesis_score")
    valuation_d = delta("valuation_score")
    overexp_d = delta("overexpectation_penalty")
    if thesis_d >= 0.08:
        events.append("Thesis Improvement")
    elif thesis_d <= -0.08:
        events.append("Thesis Deterioration")
    if valuation_d <= -0.12 and overexp_d >= 0.08:
        events.append("Overvaluation Risk")
    elif valuation_d >= 0.12:
        events.append("Re-rating Opportunity")

    for margin_col in ["fcf_margin", "operating_margin", "net_margin", "gross_margin"]:
        if margin_col in row.index and margin_col in prev.index:
            md = float(pd.to_numeric(row.get(margin_col), errors="coerce")) - float(pd.to_numeric(prev.get(margin_col), errors="coerce"))
            if np.isfinite(md) and md >= 0.03:
                events.append("Margin Expansion")
                break
            if np.isfinite(md) and md <= -0.03:
                events.append("Margin Compression")
                break

    return list(dict.fromkeys(events))


def build_thesis_history(
    scored_snapshots: pd.DataFrame,
    ticker: str,
    *,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return the thesis time-series for a ticker between purchase and review dates."""
    subset = scored_snapshots[scored_snapshots["ticker"] == str(ticker).upper().replace(".", "-")].sort_values("date").copy()
    if start_date is not None and pd.notna(start_date):
        subset = subset[subset["date"] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None and pd.notna(end_date):
        subset = subset[subset["date"] <= pd.Timestamp(end_date).normalize()]
    if subset.empty:
        return pd.DataFrame(columns=THESIS_HISTORY_COLUMNS)

    rows: list[dict[str, object]] = []
    original = subset.iloc[0]
    prev = None
    for _, row in subset.iterrows():
        valuation_state = _valuation_status(row)
        events = _detect_thesis_events(row, prev)
        status, _ = _thesis_status(row, original if prev is not None else None)
        temp = dict(row)
        temp["valuation_status"] = valuation_state
        rows.append({
            "ticker": str(row["ticker"]),
            "date": str(pd.Timestamp(row["date"]).date()),
            "thesis_score": round(float(row.get("thesis_score", 0.5)), 4),
            "position_health_score": _position_health_from_row(row, original),
            "conviction_score": _conviction_score(temp, status),
            "valuation_status": valuation_state,
            "quality_score": round(float(row.get("quality_score", 0.5)), 4),
            "growth_score": round(float(row.get("growth_score", 0.5)), 4),
            "valuation_score": round(float(row.get("valuation_score", 0.5)), 4),
            "fundamental_trend_score": round(float(row.get("fundamental_trend_score", 0.5)), 4),
            "catalyst_score": round(float(row.get("catalyst_score", 0.5)), 4),
            "risk_score": round(float(row.get("risk_score", 0.5)), 4),
            "moat_proxy_score": round(float(row.get("moat_proxy_score", 0.5)), 4),
            "expectation_gap_score": round(float(row.get("expectation_gap_score", 0.5)), 4),
            "overexpectation_penalty": round(float(row.get("overexpectation_penalty", 0.5)), 4),
            "thesis_events": "; ".join(events),
        })
        prev = row
    return pd.DataFrame(rows, columns=THESIS_HISTORY_COLUMNS)


def _history_trend(history: pd.DataFrame) -> str:
    if history is None or len(history) < 2:
        return "Insufficient History"
    thesis = pd.to_numeric(history["thesis_score"], errors="coerce")
    if thesis.notna().sum() < 2:
        return "Insufficient History"
    last_delta = float(thesis.iloc[-1] - thesis.iloc[-2])
    total_delta = float(thesis.iloc[-1] - thesis.iloc[0])
    recent_events = "; ".join(history["thesis_events"].tail(3).fillna("").tolist())
    if "Thesis Deterioration" in recent_events and last_delta <= -0.05:
        return "Deteriorating"
    if "Thesis Improvement" in recent_events and last_delta >= 0.05:
        return "Improving"
    if total_delta >= 0.08:
        return "Improving"
    if total_delta <= -0.08:
        return "Deteriorating"
    return "Stable"


def _opportunity_type(row: pd.Series) -> str:
    q = float(row.get("quality_score", 0.5)); g = float(row.get("growth_score", 0.5)); v = float(row.get("valuation_score", 0.5))
    t = float(row.get("fundamental_trend_score", 0.5)); c = float(row.get("catalyst_score", 0.5)); r = float(row.get("risk_score", 0.5))
    if v >= 0.68 and g >= 0.62 and q >= 0.55 and r >= 0.50:
        return "Growth infravalorado"
    if q >= 0.70 and g >= 0.62 and v >= 0.45 and r >= 0.55:
        return "Quality Growth razonable"
    if v >= 0.68 and c >= 0.60 and t >= 0.52 and r >= 0.50:
        return "Value con catalizador"
    if q >= 0.72 and t >= 0.62 and v >= 0.50 and r >= 0.58:
        return "Compounder a precio razonable"
    if v >= 0.62 and (q < 0.42 or t < 0.40 or r < 0.42):
        return "Value trap"
    if g >= 0.65 and v < 0.38:
        return "Growth caro"
    return "Descartar" if float(row.get("thesis_score", 0.5)) < 0.50 else "Quality Growth razonable"


def _driver_strings(row: pd.Series) -> tuple[str, str]:
    scores = {col: float(row.get(col, 0.5)) for col in THESIS_COMPONENTS if col in row.index}
    positives = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
    risks = sorted(scores.items(), key=lambda kv: kv[1])[:5]
    return (
        "; ".join(f"{k}={v:.2f}" for k, v in positives),
        "; ".join(f"{k}={v:.2f}" for k, v in risks),
    )


def _thesis_status(current: pd.Series, original: pd.Series | None) -> tuple[str, str]:
    if original is None:
        q = float(current.get("quality_score", 0.5)); g = float(current.get("growth_score", 0.5)); t = float(current.get("fundamental_trend_score", 0.5)); r = float(current.get("risk_score", 0.5))
        if min(q, g, t, r) < 0.35:
            return "Broken", "No original snapshot; current thesis has severe quality/growth/trend/risk weakness."
        if float(current.get("thesis_score", 0.5)) >= 0.62:
            return "Intact", "No original snapshot; current GARP thesis remains acceptable."
        return "Weakening", "No original snapshot; current thesis score is mediocre."

    thesis_delta = float(current.get("thesis_score", 0.5)) - float(original.get("thesis_score", 0.5))
    fundamental_delta = np.mean([
        float(current.get(col, 0.5)) - float(original.get(col, 0.5))
        for col in ["quality_score", "growth_score", "fundamental_trend_score", "moat_proxy_score", "catalyst_score"]
    ])
    risk_current = float(current.get("risk_score", 0.5))
    valuation_state = _valuation_status(current)
    if risk_current < 0.32 or fundamental_delta <= -0.25:
        return "Broken", f"Thesis deterioration is severe: thesis_delta={thesis_delta:+.2f}, fundamental_delta={fundamental_delta:+.2f}."
    if fundamental_delta <= -0.10 or thesis_delta <= -0.12:
        return "Weakening", f"Thesis is deteriorating: thesis_delta={thesis_delta:+.2f}, fundamental_delta={fundamental_delta:+.2f}."
    if valuation_state in {"Fully Valued", "Overvalued", "Extremely Overvalued"} and fundamental_delta >= -0.03:
        return "Maturing", f"Business remains acceptable but valuation/expectation gap has largely closed ({valuation_state})."
    if fundamental_delta >= 0.08 and thesis_delta >= 0.05:
        return "Improving", f"Thesis improved: thesis_delta={thesis_delta:+.2f}, fundamental_delta={fundamental_delta:+.2f}."
    return "Intact", f"Thesis broadly intact: thesis_delta={thesis_delta:+.2f}, fundamental_delta={fundamental_delta:+.2f}."


def _action_and_exit(row: Mapping[str, object], better_opportunity: bool) -> tuple[str, str, int, str, str, bool, bool, bool]:
    status = str(row["thesis_status"])
    valuation_status = str(row["valuation_status"])
    health = float(row["position_health_score"])
    risk = float(row["risk_score"])
    overexp = float(row["overexpectation_penalty"])

    exit_reasons: list[str] = []
    if status == "Broken":
        exit_reasons.append("Thesis Broken")
    elif status == "Weakening":
        exit_reasons.append("Thesis Weakening")
    if valuation_status == "Fully Valued":
        exit_reasons.append("Fully Valued")
    elif valuation_status in {"Overvalued", "Extremely Overvalued"}:
        exit_reasons.append("Overvalued")
    if risk < 0.40:
        exit_reasons.append("Risk Increase")
    if float(row.get("quality_score", 0.5)) < 0.40:
        exit_reasons.append("Quality Deterioration")
    if float(row.get("growth_score", 0.5)) < 0.40:
        exit_reasons.append("Growth Deterioration")
    if float(row.get("catalyst_score", 0.5)) < 0.35 and status in {"Maturing", "Weakening"}:
        exit_reasons.append("Catalyst Exhausted")
    if better_opportunity:
        exit_reasons.append("Better Opportunity Available")

    exit_score = 0
    exit_score += {"Broken": 50, "Weakening": 30, "Maturing": 15, "Intact": 5, "Improving": 0}.get(status, 10)
    exit_score += {"Extremely Overvalued": 30, "Overvalued": 22, "Fully Valued": 14, "Fairly Valued": 4, "Undervalued": 0}.get(valuation_status, 5)
    exit_score += max(0, int(round((0.50 - risk) * 60)))
    exit_score += max(0, int(round((overexp - 0.65) * 40)))
    if better_opportunity:
        exit_score += 12
    exit_score = int(max(0, min(100, exit_score)))

    if status == "Improving" and health >= 72 and valuation_status in {"Undervalued", "Fairly Valued"}:
        rating = "Strong Buy"; action = "Add"
    elif health >= 66 and valuation_status in {"Undervalued", "Fairly Valued"} and status in {"Improving", "Intact"}:
        rating = "Buy"; action = "Add"
    elif exit_score >= 70 or status == "Broken":
        rating = "Sell"; action = "Sell"
    elif exit_score >= 50 or status == "Weakening":
        rating = "Reduce"; action = "Reduce"
    elif exit_score >= 35 or status == "Maturing" or better_opportunity:
        rating = "Review"; action = "Review"
    else:
        rating = "Hold"; action = "Hold"

    if rating == "Sell" or (status == "Broken" and exit_score >= 70):
        review_priority = "Critical"
    elif rating == "Reduce" or exit_score >= 50:
        review_priority = "High"
    elif rating == "Review" or better_opportunity:
        review_priority = "Medium"
    else:
        review_priority = "Low"
    buy_today = rating in {"Strong Buy", "Buy"}
    sell_today = rating == "Sell"
    hold_today = rating in {"Strong Buy", "Buy", "Hold", "Review"}
    return rating, action, exit_score, review_priority, "; ".join(dict.fromkeys(exit_reasons)) or "No thesis-based exit trigger", buy_today, hold_today, sell_today


def review_portfolio(
    snapshots: pd.DataFrame,
    *,
    positions: pd.DataFrame | None = None,
    tickers: Iterable[str] | None = None,
    review_date: str | pd.Timestamp | None = None,
    output_dir: str | Path | None = None,
    opportunity_score_gap: float = 0.15,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Review existing positions against original and current point-in-time snapshots."""
    snapshots_norm = _as_multiindex(snapshots)
    scored_all = add_portfolio_review_scores(snapshots_norm)
    review_ts = pd.Timestamp(review_date).normalize() if review_date is not None else None
    current_universe = _latest_universe_snapshot(scored_all, review_ts)
    pos = _positions_frame(positions=positions, tickers=tickers)
    held = set(pos["ticker"].astype(str))

    opportunity_pool = current_universe[~current_universe["ticker"].isin(held)].copy()
    if not opportunity_pool.empty:
        opportunity_pool = opportunity_pool.sort_values("thesis_score", ascending=False)
        best_alt_row = opportunity_pool.iloc[0]
        best_alt_ticker = str(best_alt_row["ticker"])
        best_alt_score = float(best_alt_row["thesis_score"])
    else:
        best_alt_ticker = ""
        best_alt_score = float("nan")

    rows: list[dict[str, object]] = []
    history_frames: list[pd.DataFrame] = []
    for _, position in pos.iterrows():
        ticker = str(position["ticker"])
        original_anchor = position.get("snapshot_date") if pd.notna(position.get("snapshot_date")) else position.get("purchase_date")
        current_row = _ticker_snapshot(scored_all, ticker, review_ts)
        if current_row is None:
            rows.append({
                "ticker": ticker,
                "weight": float(position.get("weight", 0.0)),
                "purchase_date": str(position.get("purchase_date", "")),
                "original_snapshot_date": "",
                "current_snapshot_date": "",
                "position_health_score": 0,
                "buy_hold_sell_rating": "Sell",
                "exit_score": 100,
                "conviction_score": 0,
                "thesis_score": 0.0,
                "thesis_status": "Broken",
                "thesis_history_trend": "Missing Current Snapshot",
                "valuation_status": "Unknown",
                "opportunity_type": "Descartar",
                "moat_proxy_score": 0.0,
                "expectation_gap_score": 0.0,
                "overexpectation_penalty": 1.0,
                "top_positive_drivers": "",
                "top_risks": "missing_current_snapshot=1.00",
                "action_recommended": "Sell",
                "review_priority": "Critical",
                "exit_reason": "Thesis Broken",
                "latest_thesis_events": "Missing Current Snapshot",
                "thesis_changes": "No current snapshot is available for this ticker.",
                "original_buy_reason": "",
                "original_opportunity_type": "",
                "original_valuation_status": "",
                "original_thesis_score": np.nan,
                "original_scores_json": "{}",
                "current_vs_original_summary": "No current snapshot is available for comparison.",
                "buy_today_flag": False,
                "hold_today_flag": False,
                "sell_today_flag": True,
                "best_alternative_ticker": best_alt_ticker,
                "best_alternative_score": best_alt_score,
                "opportunity_cost_flag": bool(np.isfinite(best_alt_score)),
            })
            continue

        original_row = _ticker_snapshot(scored_all, ticker, original_anchor) if pd.notna(original_anchor) else None
        history = build_thesis_history(
            scored_all,
            ticker,
            start_date=pd.Timestamp(original_row["date"]) if original_row is not None else original_anchor,
            end_date=pd.Timestamp(current_row["date"]),
        )
        if not history.empty:
            history_frames.append(history)
        status, changes = _thesis_status(current_row, original_row)
        valuation_status = _valuation_status(current_row)
        opportunity_type = _opportunity_type(current_row)
        positives, risks = _driver_strings(current_row)

        health = _position_health_from_row(current_row, original_row)
        better_opportunity = bool(np.isfinite(best_alt_score) and (best_alt_score - float(current_row.get("thesis_score", 0.5)) >= float(opportunity_score_gap)))
        temp = dict(current_row)
        temp.update({"thesis_status": status, "valuation_status": valuation_status, "position_health_score": health})
        rating, action, exit_score, review_priority, exit_reason, buy_today, hold_today, sell_today = _action_and_exit(temp, better_opportunity)
        conviction = _conviction_score(temp, status, better_opportunity=better_opportunity)
        thesis_history_trend = _history_trend(history)
        latest_events = ""
        if not history.empty:
            latest_events = str(history["thesis_events"].replace("", np.nan).dropna().tail(1).iloc[0]) if history["thesis_events"].replace("", np.nan).dropna().any() else ""

        original_opportunity_type = _opportunity_type(original_row) if original_row is not None else ""
        original_valuation_status = _valuation_status(original_row) if original_row is not None else ""
        original_thesis_score = float(original_row.get("thesis_score", np.nan)) if original_row is not None else np.nan
        original_scores = {
            col: round(float(original_row.get(col, np.nan)), 4)
            for col in THESIS_COMPONENTS + ["thesis_score", "overexpectation_penalty"]
            if original_row is not None and col in original_row.index and pd.notna(original_row.get(col))
        }
        original_buy_reason = (
            f"Bought as {original_opportunity_type}: thesis={original_thesis_score:.2f}, "
            f"valuation={original_valuation_status}, drivers={_driver_strings(original_row)[0]}"
            if original_row is not None else "No original snapshot supplied."
        )
        current_vs_original = (
            f"{status}: {changes} Current valuation={valuation_status}; "
            f"current thesis={float(current_row.get('thesis_score', 0.5)):.2f} vs original={original_thesis_score:.2f}."
            if original_row is not None else changes
        )

        rows.append({
            "ticker": ticker,
            "weight": float(position.get("weight", 0.0)),
            "purchase_date": "" if pd.isna(position.get("purchase_date")) else str(pd.Timestamp(position.get("purchase_date")).date()),
            "original_snapshot_date": "" if original_row is None else str(pd.Timestamp(original_row["date"]).date()),
            "current_snapshot_date": str(pd.Timestamp(current_row["date"]).date()),
            "position_health_score": health,
            "buy_hold_sell_rating": rating,
            "exit_score": exit_score,
            "conviction_score": conviction,
            "thesis_score": round(float(current_row.get("thesis_score", 0.5)), 4),
            "thesis_status": status,
            "thesis_history_trend": thesis_history_trend,
            "valuation_status": valuation_status,
            "opportunity_type": opportunity_type,
            "moat_proxy_score": round(float(current_row.get("moat_proxy_score", 0.5)), 4),
            "expectation_gap_score": round(float(current_row.get("expectation_gap_score", 0.5)), 4),
            "overexpectation_penalty": round(float(current_row.get("overexpectation_penalty", 0.5)), 4),
            "top_positive_drivers": positives,
            "top_risks": risks,
            "action_recommended": action,
            "review_priority": review_priority,
            "exit_reason": exit_reason,
            "latest_thesis_events": latest_events,
            "thesis_changes": changes,
            "original_buy_reason": original_buy_reason,
            "original_opportunity_type": original_opportunity_type,
            "original_valuation_status": original_valuation_status,
            "original_thesis_score": round(original_thesis_score, 4) if np.isfinite(original_thesis_score) else np.nan,
            "original_scores_json": json.dumps(original_scores, sort_keys=True),
            "current_vs_original_summary": current_vs_original,
            "buy_today_flag": buy_today,
            "hold_today_flag": hold_today,
            "sell_today_flag": sell_today,
            "best_alternative_ticker": best_alt_ticker,
            "best_alternative_score": round(best_alt_score, 4) if np.isfinite(best_alt_score) else np.nan,
            "opportunity_cost_flag": better_opportunity,
        })

    review = pd.DataFrame(rows)
    for col in POSITION_OUTPUT_COLUMNS:
        if col not in review.columns:
            review[col] = np.nan
    review = review[POSITION_OUTPUT_COLUMNS]

    summary = {
        "review_date": str(review_ts.date()) if review_ts is not None else str(pd.Timestamp(current_universe["date"].max()).date()) if not current_universe.empty else "",
        "positions_reviewed": int(len(review)),
        "average_position_health_score": float(pd.to_numeric(review["position_health_score"], errors="coerce").mean()) if not review.empty else np.nan,
        "ratings_count": review["buy_hold_sell_rating"].value_counts().to_dict() if not review.empty else {},
        "thesis_status_count": review["thesis_status"].value_counts().to_dict() if not review.empty else {},
        "best_positions": review.sort_values("position_health_score", ascending=False)["ticker"].head(5).tolist() if not review.empty else [],
        "weakest_positions": review.sort_values("position_health_score", ascending=True)["ticker"].head(5).tolist() if not review.empty else [],
        "positions_to_review": review.loc[review["review_priority"].isin(["Critical", "High", "Medium"]), "ticker"].tolist() if not review.empty else [],
        "critical_positions": review.loc[review["review_priority"].eq("Critical"), "ticker"].tolist() if not review.empty else [],
        "possible_sales": review.loc[review["sell_today_flag"].astype(bool), "ticker"].tolist() if not review.empty else [],
        "possible_adds": review.loc[review["buy_hold_sell_rating"].isin(["Strong Buy", "Buy"]), "ticker"].tolist() if not review.empty else [],
        "best_new_opportunity": best_alt_ticker,
        "best_new_opportunity_score": round(best_alt_score, 4) if np.isfinite(best_alt_score) else None,
    }
    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame(columns=THESIS_HISTORY_COLUMNS)
    event_df = (
        history_df.loc[history_df["thesis_events"].astype(str).str.len() > 0, ["ticker", "date", "thesis_events"]].copy()
        if not history_df.empty else pd.DataFrame(columns=["ticker", "date", "thesis_events"])
    )

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        review.to_csv(out_dir / "portfolio_review_positions.csv", index=False)
        history_df.to_csv(out_dir / "portfolio_thesis_history.csv", index=False)
        event_df.to_csv(out_dir / "portfolio_thesis_events.csv", index=False)
        (out_dir / "portfolio_review_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        _write_review_report(out_dir / "portfolio_review_report.md", review, summary, event_df)
        if not opportunity_pool.empty:
            cols = ["ticker", "date", "thesis_score", "quality_score", "growth_score", "valuation_score", "expectation_gap_score", "risk_score"]
            opportunity_pool[[c for c in cols if c in opportunity_pool.columns]].head(25).to_csv(out_dir / "portfolio_review_opportunity_cost.csv", index=False)

    return review, summary


def _write_review_report(path: Path, review: pd.DataFrame, summary: Mapping[str, object], events: pd.DataFrame) -> None:
    """Write a concise monthly/quarterly thesis review report in Markdown."""
    lines = [
        "# Portfolio Review Report",
        "",
        f"- Review date: {summary.get('review_date', '')}",
        f"- Positions reviewed: {summary.get('positions_reviewed', 0)}",
        f"- Average position health: {summary.get('average_position_health_score', np.nan):.1f}",
        f"- Best new opportunity: {summary.get('best_new_opportunity') or '-'}",
        "",
        "## Positions requiring attention",
    ]
    attention = review[review["review_priority"].isin(["Critical", "High", "Medium"])] if not review.empty else pd.DataFrame()
    if attention.empty:
        lines.append("- No positions require elevated review priority.")
    else:
        for _, row in attention.sort_values(["review_priority", "exit_score"], ascending=[True, False]).iterrows():
            lines.append(
                f"- {row['ticker']}: {row['review_priority']} | {row['buy_hold_sell_rating']} | "
                f"{row['thesis_status']} | exit_score={row['exit_score']} | {row['exit_reason']}"
            )

    lines.extend(["", "## Recent thesis events"])
    if events.empty:
        lines.append("- No material thesis events detected.")
    else:
        for _, row in events.tail(25).iterrows():
            lines.append(f"- {row['date']} {row['ticker']}: {row['thesis_events']}")

    lines.extend(["", "## Best / weakest positions"])
    lines.append("- Best positions: " + (", ".join(summary.get("best_positions", [])) or "-"))
    lines.append("- Weakest positions: " + (", ".join(summary.get("weakest_positions", [])) or "-"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_positions_csv(path: str | Path) -> pd.DataFrame:
    """Load a portfolio-review CSV with ticker, optional weight/purchase_date/snapshot_date."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Portfolio positions file not found: {p}")
    return pd.read_csv(p)
