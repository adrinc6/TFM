from __future__ import annotations

import pandas as pd

from module.features import build_features


def test_features_are_point_in_time_and_targets_are_separate(feature_settings) -> None:
    features = build_features(feature_settings)
    targets = pd.read_parquet(feature_settings.processed_output_dir / "targets_forward_3m.parquet")
    baselines = pd.read_parquet(feature_settings.processed_output_dir / "baseline_scores.parquet")

    assert "forward_excess_return_3m" not in features.columns
    assert {"factor_roe", "factor_pe", "factor_relative_return_3m"} <= set(features.columns)
    assert targets["target_available"].any()
    assert not targets.loc[targets["snapshot_date"] == "2000-08-15", "target_available"].any()
    assert {"garp_score", "momentum_score"} <= set(baselines.columns)


def test_stale_prices_do_not_receive_factors_or_baseline_scores(feature_settings) -> None:
    features = build_features(feature_settings)
    baselines = pd.read_parquet(feature_settings.processed_output_dir / "baseline_scores.parquet")
    stale = features.loc[(features["ticker"] == "CCC") & (features["snapshot_date"] == "2000-03-15")].iloc[0]
    stale_baseline = baselines.loc[
        (baselines["ticker"] == "CCC") & (baselines["snapshot_date"] == "2000-03-15")
    ].iloc[0]

    assert not stale["is_price_fresh"]
    assert pd.isna(stale["factor_roe"])
    assert pd.isna(stale_baseline["garp_score"])
