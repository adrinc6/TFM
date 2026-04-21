"""Target engineering utilities for TP/SL training targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

_TP_MIN = 0.02
_TP_MAX = 0.25
_SL_MIN = 0.01
_SL_MAX = 0.15
_VOL_SCALE_MIN = 0.5
_VOL_SCALE_MAX = 2.0


@dataclass
class TpSlTargetBundle:
    """Container with TP/SL-native training targets."""

    hit_label: pd.Series
    outcome: pd.Series
    tp_level: pd.Series
    sl_level: pd.Series


def _extract_close(price_obj) -> pd.Series:
    if price_obj is None:
        return pd.Series(dtype=float)
    if isinstance(price_obj, pd.Series):
        return pd.to_numeric(price_obj, errors="coerce").dropna().sort_index()
    if isinstance(price_obj, pd.DataFrame) and not price_obj.empty:
        # Upstream price frames are expected to expose "Close"; fallback keeps
        # compatibility with single-column custom frames used in tests.
        col = "Close" if "Close" in price_obj.columns else price_obj.columns[-1]
        return pd.to_numeric(price_obj[col], errors="coerce").dropna().sort_index()
    return pd.Series(dtype=float)


def infer_tp_sl_levels(
    df: pd.DataFrame,
    *,
    tp_default: float = 0.08,
    sl_default: float = 0.05,
    volatility_col: str = "volatility_60d",
) -> tuple[pd.Series, pd.Series]:
    """Infer per-row TP/SL levels from market volatility (no linear score mapping)."""
    idx = df.index
    if volatility_col not in df.columns:
        return (
            pd.Series(float(tp_default), index=idx, dtype=float),
            pd.Series(float(sl_default), index=idx, dtype=float),
        )

    vol = pd.to_numeric(df[volatility_col], errors="coerce")
    vol_ref = float(vol.dropna().median()) if vol.notna().any() else np.nan
    if not np.isfinite(vol_ref) or vol_ref <= 0:
        scale = pd.Series(1.0, index=idx, dtype=float)
    else:
        scale = (vol / vol_ref).clip(_VOL_SCALE_MIN, _VOL_SCALE_MAX).fillna(1.0).astype(float)

    tp = (float(tp_default) * scale).clip(_TP_MIN, _TP_MAX)
    sl = (float(sl_default) * scale).clip(_SL_MIN, _SL_MAX)
    return tp, sl


def build_tp_sl_targets(
    df: pd.DataFrame,
    *,
    prices_dict: Dict[str, object],
    lag_days: int = 45,
    max_holding_days: int = 90,
    tp_default: float = 0.08,
    sl_default: float = 0.05,
) -> TpSlTargetBundle:
    """Build TP/SL-first labels: 1 if TP is hit before SL within the horizon."""
    if df is None or df.empty or not isinstance(df.index, pd.MultiIndex):
        raise ValueError("TP/SL target generation requires a non-empty MultiIndex DataFrame.")
    if not prices_dict:
        raise ValueError("TP/SL target generation requires non-empty prices_dict.")

    from module.strategy.backtesting_engine import simulate_tp_sl
    tp_level, sl_level = infer_tp_sl_levels(df, tp_default=tp_default, sl_default=sl_default)
    hit_label = pd.Series(np.nan, index=df.index, dtype=float)
    outcome = pd.Series(index=df.index, dtype="object")

    for (ticker, dt), row in df.iterrows():
        prices = _extract_close(prices_dict.get(str(ticker)))
        if prices.empty:
            continue
        has_snapshot = "snapshot_date" in row and pd.notna(row.get("snapshot_date"))
        snapshot_dt = pd.Timestamp(row.get("snapshot_date")) if has_snapshot else pd.Timestamp(dt)
        entry_date = snapshot_dt + pd.Timedelta(days=max(int(lag_days), 0))
        sim = simulate_tp_sl(
            ticker=str(ticker),
            prices=prices,
            entry_date=entry_date,
            tp_pct=float(tp_level.loc[(ticker, dt)]),
            sl_pct=float(sl_level.loc[(ticker, dt)]),
            max_holding_days=max_holding_days,
        )
        out = str(sim.get("outcome", "NONE")).upper()
        outcome.loc[(ticker, dt)] = out
        hit_label.loc[(ticker, dt)] = 1.0 if out == "TP" else 0.0

    return TpSlTargetBundle(
        hit_label=hit_label,
        outcome=outcome,
        tp_level=tp_level,
        sl_level=sl_level,
    )
