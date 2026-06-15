from __future__ import annotations

import pandas as pd

from module.common.portfolio_evolution import run_portfolio_evolution


def _snapshots() -> pd.DataFrame:
    rows = []
    for date, shift in [("2024-01-31", 0.0), ("2024-02-29", 0.04), ("2024-03-31", 0.08)]:
        for i, ticker in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]):
            base = 0.72 - i * 0.04 + shift
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "quality_score": min(base, 0.95),
                    "growth_score": min(base - 0.02, 0.95),
                    "valuation_score": max(0.45, 0.70 - i * 0.03),
                    "fundamental_trend_score": min(base - 0.01, 0.95),
                    "catalyst_score": min(base - 0.08, 0.95),
                    "risk_score": min(base, 0.95),
                    "moat_proxy_score": min(base - 0.03, 0.95),
                    "expectation_gap_score": max(0.45, 0.68 - i * 0.02),
                    "overexpectation_penalty": 0.30,
                }
            )
    return pd.DataFrame(rows)


def test_portfolio_evolution_exports_live_portfolio_artifacts(tmp_path):
    evolution, transactions, holdings, summary = run_portfolio_evolution(
        _snapshots(),
        review_frequency="M",
        min_positions=5,
        max_positions=5,
        output_dir=tmp_path,
    )

    assert not evolution.empty
    assert not transactions.empty
    assert not holdings.empty
    assert evolution["n_positions"].min() >= 5
    assert {"ADD", "HOLD"}.intersection(set(transactions.get("action", [])))
    assert summary["ending_positions"] == 5
    for name in [
        "portfolio_evolution.csv",
        "portfolio_transactions.csv",
        "portfolio_monthly_holdings.csv",
        "portfolio_decision_log.csv",
        "portfolio_turnover.csv",
        "portfolio_monthly_summary.json",
    ]:
        assert (tmp_path / name).exists(), name


def test_portfolio_evolution_holds_intact_positions_and_logs_memory(tmp_path):
    evolution, transactions, holdings, _ = run_portfolio_evolution(
        _snapshots(),
        review_frequency="M",
        min_positions=5,
        max_positions=5,
        output_dir=tmp_path,
    )
    decision_log = pd.read_csv(tmp_path / "portfolio_decision_log.csv")
    turnover = pd.read_csv(tmp_path / "portfolio_turnover.csv")

    assert evolution["n_positions"].min() >= 5
    assert "HOLD" in set(decision_log["action"])
    assert holdings["months_since_entry"].max() >= 1
    assert {"thesis_persistence_score", "months_thesis_intact", "original_buy_reason"}.issubset(holdings.columns)
    assert {"monthly_turnover", "annual_turnover", "average_holding_period", "median_holding_period"}.issubset(turnover.columns)


def test_portfolio_evolution_sells_broken_thesis_with_explanation(tmp_path):
    data = _snapshots()
    mask = (data["date"].eq("2024-02-29")) & (data["ticker"].eq("AAA"))
    data.loc[mask, ["quality_score", "growth_score", "fundamental_trend_score", "risk_score", "moat_proxy_score"]] = 0.20

    run_portfolio_evolution(data, review_frequency="M", min_positions=5, max_positions=5, output_dir=tmp_path)
    decision_log = pd.read_csv(tmp_path / "portfolio_decision_log.csv")

    aaa_actions = decision_log[decision_log["ticker"].eq("AAA")]
    assert "SELL" in set(aaa_actions["action"])
    sell = aaa_actions[aaa_actions["action"].eq("SELL")].iloc[0]
    assert sell["thesis_status"] == "Broken"
    assert isinstance(sell["reason"], str) and sell["reason"]


def test_portfolio_evolution_respects_configured_start_and_end_dates(tmp_path):
    evolution, transactions, holdings, summary = run_portfolio_evolution(
        _snapshots(),
        review_frequency="M",
        start_date="2024-02-01",
        end_date="2024-02-29",
        min_positions=5,
        max_positions=5,
        output_dir=tmp_path,
    )

    assert evolution["date"].tolist() == ["2024-02-29"]
    assert summary["start_date"] == "2024-02-29"
    assert summary["end_date"] == "2024-02-29"
    assert transactions["date"].nunique() == 1
    assert holdings["date"].nunique() == 1
