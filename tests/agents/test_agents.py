from __future__ import annotations

import json

import pandas as pd
import pytest

from environment import Settings
from module.modeling.agents import _recency_weights, build_agent_scores


def _train_four_years() -> pd.DataFrame:
    dates = pd.date_range("2011-01-15", "2015-01-15", freq="3MS").astype(str)
    return pd.DataFrame({"snapshot_date": dates})


def test_recency_off_returns_none() -> None:
    assert _recency_weights(_train_four_years(), Settings(recency_weighting="off")) is None


def test_recency_linear_weights_increase_towards_recent() -> None:
    weights = _recency_weights(_train_four_years(), Settings(recency_weighting="linear"))
    assert weights.iloc[-1] > weights.iloc[0]          # lo reciente pesa más
    assert (weights >= 1.0 - 1e-6).all()               # el más antiguo pesa ~1


def test_recency_exponential_respects_halflife() -> None:
    train = _train_four_years()
    weights = _recency_weights(train, Settings(recency_weighting="exponential"))
    dates = pd.to_datetime(train["snapshot_date"])
    age_years = (dates.max() - dates) / pd.Timedelta(days=365.25)
    # A una vida media (3 años) de antigüedad, el peso debe ser ~0.5 del más reciente (=1.0).
    idx = (age_years - 3.0).abs().idxmin()
    assert weights.iloc[-1] == pytest.approx(1.0)
    assert weights.loc[idx] == pytest.approx(0.5, abs=0.03)


def test_agents_start_at_anchor_with_expanding_history_and_score_monthly(agent_settings) -> None:
    scores = build_agent_scores(agent_settings)

    assert scores["snapshot_date"].min() == "2000-02-15"
    feb_to_apr = scores.loc[scores["snapshot_date"].isin(["2000-02-15", "2000-03-15", "2000-04-15"])]
    assert set(feb_to_apr["model_retrain_date"]) == {"2000-02-15"}
    assert {"quality", "momentum", "value", "meta_score"} <= set(scores.columns)
    assert scores["training_start_date"].min() == "1996-02-15"


def test_agents_write_only_oos_diagnostics_and_manifest(agent_settings) -> None:
    build_agent_scores(agent_settings)
    agents_dir = next((agent_settings.processed_output_dir / "agents").iterdir())
    diagnostics = pd.read_parquet(agents_dir / "rank_ic_diagnostics.parquet")
    manifest = json.loads((agents_dir / "manifest.json").read_text(encoding="utf-8"))
    weights = pd.read_parquet(agents_dir / "meta_weights.parquet")

    assert (pd.to_datetime(diagnostics["label_end_date"]) >= pd.to_datetime(diagnostics["prediction_date"])).all()
    assert manifest["config"]["missing_policy"] == "median_train_only_with_indicator"
    assert set(weights["weight_status"]) <= {"learned", "fallback_equal"}


def test_future_label_change_does_not_rewrite_past_predictions(agent_settings) -> None:
    first = build_agent_scores(agent_settings)
    target_path = agent_settings.processed_output_dir / "targets_forward_3m.parquet"
    targets = pd.read_parquet(target_path)
    future = targets["snapshot_date"] == "2001-06-15"
    targets.loc[future, "forward_excess_return_3m"] = (
        targets.loc[future, "forward_excess_return_3m"] * 1.5 + 0.1
    )
    targets.to_parquet(target_path, index=False)

    second = build_agent_scores(agent_settings)
    columns = ["ticker", "snapshot_date", "quality", "momentum", "value", "meta_score"]
    before_cutoff = first.loc[first["snapshot_date"] <= "2000-12-15", columns]
    rebuilt_before_cutoff = second.loc[second["snapshot_date"] <= "2000-12-15", columns]

    pd.testing.assert_frame_equal(
        before_cutoff.reset_index(drop=True), rebuilt_before_cutoff.reset_index(drop=True)
    )
