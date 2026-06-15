from __future__ import annotations

import json

import pandas as pd

from module.common.portfolio_intelligence import add_portfolio_review_scores, build_thesis_history, review_portfolio


def _snapshots() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2023-03-31",
                "ticker": "AAA",
                "quality_score": 0.62,
                "growth_score": 0.60,
                "valuation_score": 0.64,
                "fundamental_trend_score": 0.58,
                "catalyst_score": 0.55,
                "risk_score": 0.62,
                "moat_proxy_score": 0.60,
                "expectation_gap_score": 0.63,
                "overexpectation_penalty": 0.30,
            },
            {
                "date": "2024-03-31",
                "ticker": "AAA",
                "quality_score": 0.78,
                "growth_score": 0.72,
                "valuation_score": 0.58,
                "fundamental_trend_score": 0.75,
                "catalyst_score": 0.66,
                "risk_score": 0.70,
                "moat_proxy_score": 0.76,
                "expectation_gap_score": 0.62,
                "overexpectation_penalty": 0.38,
            },
            {
                "date": "2024-03-31",
                "ticker": "BBB",
                "quality_score": 0.30,
                "growth_score": 0.28,
                "valuation_score": 0.72,
                "fundamental_trend_score": 0.25,
                "catalyst_score": 0.20,
                "risk_score": 0.25,
                "moat_proxy_score": 0.22,
                "expectation_gap_score": 0.40,
                "overexpectation_penalty": 0.45,
            },
            {
                "date": "2024-03-31",
                "ticker": "CCC",
                "quality_score": 0.90,
                "growth_score": 0.86,
                "valuation_score": 0.74,
                "fundamental_trend_score": 0.82,
                "catalyst_score": 0.80,
                "risk_score": 0.82,
                "moat_proxy_score": 0.88,
                "expectation_gap_score": 0.78,
                "overexpectation_penalty": 0.28,
            },
        ]
    )


def test_portfolio_review_compares_purchase_snapshot_with_current_snapshot(tmp_path):
    positions = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "weight": [1.0],
            "purchase_date": ["2023-04-15"],
            "snapshot_date": ["2023-03-31"],
        }
    )

    review, summary = review_portfolio(
        _snapshots(),
        positions=positions,
        review_date="2024-04-30",
        output_dir=tmp_path,
    )

    row = review.iloc[0]
    assert row["original_snapshot_date"] == "2023-03-31"
    assert row["current_snapshot_date"] == "2024-03-31"
    assert row["thesis_status"] == "Improving"
    assert row["conviction_score"] >= 60
    assert row["thesis_history_trend"] in {"Improving", "Stable"}
    assert "Growth Acceleration" in row["latest_thesis_events"] or "Thesis Improvement" in row["latest_thesis_events"]
    assert "Bought as" in row["original_buy_reason"]
    assert row["buy_hold_sell_rating"] in {"Strong Buy", "Buy", "Hold"}
    assert bool(row["hold_today_flag"]) is True
    assert summary["positions_reviewed"] == 1
    assert (tmp_path / "portfolio_review_positions.csv").exists()
    assert (tmp_path / "portfolio_thesis_history.csv").exists()
    assert (tmp_path / "portfolio_thesis_events.csv").exists()
    assert (tmp_path / "portfolio_review_report.md").exists()
    payload = json.loads((tmp_path / "portfolio_review_summary.json").read_text())
    assert payload["best_new_opportunity"] == "CCC"


def test_portfolio_review_flags_broken_thesis_and_exit_reason():
    positions = pd.DataFrame({"ticker": ["BBB"], "weight": [1.0]})

    review, _ = review_portfolio(_snapshots(), positions=positions, review_date="2024-04-30")

    row = review.iloc[0]
    assert row["thesis_status"] == "Broken"
    assert row["buy_hold_sell_rating"] == "Sell"
    assert bool(row["sell_today_flag"]) is True
    assert "Thesis Broken" in row["exit_reason"]


def test_portfolio_review_scores_can_be_derived_from_raw_snapshot_features():
    raw = pd.DataFrame(
        {
            "date": ["2024-03-31", "2024-03-31"],
            "ticker": ["AAA", "BBB"],
            "gross_margin": [0.65, 0.30],
            "roic": [0.18, 0.04],
            "revenue_yoy_growth": [0.20, -0.03],
            "fcf_yield": [0.05, 0.01],
            "pe_ratio": [18, 45],
            "debt_to_ebitda": [1.0, 5.0],
        }
    )

    scored = add_portfolio_review_scores(raw)

    assert scored.loc[scored["ticker"] == "AAA", "quality_score"].iloc[0] > scored.loc[scored["ticker"] == "BBB", "quality_score"].iloc[0]
    assert scored.loc[scored["ticker"] == "AAA", "thesis_score"].iloc[0] > scored.loc[scored["ticker"] == "BBB", "thesis_score"].iloc[0]


def test_thesis_history_detects_material_events():
    scored = add_portfolio_review_scores(_snapshots())

    history = build_thesis_history(
        scored,
        "AAA",
        start_date=pd.Timestamp("2023-03-31"),
        end_date=pd.Timestamp("2024-03-31"),
    )

    assert list(history["date"]) == ["2023-03-31", "2024-03-31"]
    assert history["position_health_score"].iloc[-1] >= history["position_health_score"].iloc[0]
    assert "Quality Upgrade" in history["thesis_events"].iloc[-1]
    assert "Thesis Improvement" in history["thesis_events"].iloc[-1]
