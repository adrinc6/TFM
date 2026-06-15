from __future__ import annotations

import json

import pandas as pd

from module.common.static_viewer import generate_static_viewer


def test_static_viewer_generates_dashboard_pages_and_position_detail(tmp_path):
    run_dir = tmp_path / "run"
    review_dir = run_dir / "general" / "portfolio_review"
    review_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "ticker": ["AAA"],
            "conviction_score": [82],
            "thesis_score": [0.76],
            "position_health_score": [79],
            "valuation_status": ["Fairly Valued"],
            "buy_hold_sell_rating": ["Hold"],
            "review_priority": ["Low"],
            "exit_score": [8],
            "exit_reason": ["No thesis-based exit trigger"],
            "opportunity_type": ["Quality Growth razonable"],
            "thesis_status": ["Improving"],
            "opportunity_cost_flag": [False],
            "best_alternative_ticker": ["BBB"],
            "best_alternative_score": [0.91],
        }
    ).to_csv(review_dir / "portfolio_review_positions.csv", index=False)
    pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "date": ["2023-03-31", "2024-03-31"],
            "thesis_score": [0.61, 0.76],
            "conviction_score": [65, 82],
            "position_health_score": [61, 79],
            "valuation_score": [0.65, 0.58],
            "moat_proxy_score": [0.60, 0.76],
            "catalyst_score": [0.55, 0.66],
            "expectation_gap_score": [0.63, 0.62],
            "thesis_events": ["", "Quality Upgrade; Thesis Improvement"],
        }
    ).to_csv(review_dir / "portfolio_thesis_history.csv", index=False)
    pd.DataFrame(
        {
            "ticker": ["AAA"],
            "date": ["2024-03-31"],
            "thesis_events": ["Quality Upgrade; Thesis Improvement"],
        }
    ).to_csv(review_dir / "portfolio_thesis_events.csv", index=False)
    pd.DataFrame({"ticker": ["BBB"], "thesis_score": [0.91], "quality_score": [0.88]}).to_csv(
        review_dir / "portfolio_review_opportunity_cost.csv", index=False
    )
    (review_dir / "portfolio_review_summary.json").write_text(json.dumps({"positions_reviewed": 1, "average_position_health_score": 79, "best_new_opportunity": "BBB"}))

    viewer_dir = generate_static_viewer(run_dir)

    expected = [
        "index.html",
        "run_summary.html",
        "portfolio_review.html",
        "portfolio_health.html",
        "thesis_history.html",
        "thesis_events.html",
        "opportunity_cost.html",
        "watchlist.html",
        "position_AAA.html",
    ]
    for name in expected:
        assert (viewer_dir / name).exists(), name
    assert "Quality Upgrade" in (viewer_dir / "thesis_events.html").read_text()
    assert "Position AAA" in (viewer_dir / "position_AAA.html").read_text()
