from __future__ import annotations

import pandas as pd
import pytest

from module.common.target_engineering import build_tp_sl_targets
from module.steps.step_04_evaluation.evaluator import _prepare_fold_labels


def _make_fold_df() -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples(
        [("AAA", pd.Timestamp("2024-03-31"))],
        names=["ticker", "date"],
    )
    return pd.DataFrame(
        {
            "snapshot_date": [pd.Timestamp("2024-04-01")],
            "volatility_60d": [0.02],
            "sector": ["Tech"],
        },
        index=idx,
    )


def test_build_tp_sl_targets_generates_tp_hit_label():
    df = _make_fold_df()
    prices = pd.Series(
        [100.0, 106.0, 108.0, 110.0],
        index=pd.to_datetime(["2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04"]),
        name="Close",
    )
    bundle = build_tp_sl_targets(
        df,
        prices_dict={"AAA": prices},
        lag_days=0,
        max_holding_days=10,
        tp_default=0.05,
        sl_default=0.05,
    )
    assert float(bundle.hit_label.iloc[0]) == 1.0
    assert str(bundle.outcome.iloc[0]) == "TP"


def test_prepare_fold_labels_is_garp_and_does_not_require_tp_sl_prices():
    df_train = _make_fold_df()
    df_test = _make_fold_df()
    df_train, df_test, y_train, y_test, alpha_train, alpha_test = _prepare_fold_labels(
        df_train=df_train,
        df_test=df_test,
        prices_dict=None,
    )
    assert "garp_composite_target" in df_train.columns
    assert len(y_train) == len(df_train)
    assert len(y_test) == len(df_test)
