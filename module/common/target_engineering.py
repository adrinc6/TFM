"""Target engineering utilities for alpha, quintiles, and triple barrier labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class TargetBundle:
    """Container with all engineered targets for a fold."""

    alpha: pd.Series
    quintile: pd.Series
    triple_barrier: pd.Series
    direction: pd.Series
    benchmark_return: pd.Series


def _spy_quarterly_returns(spy_prices: Optional[pd.Series]) -> Dict[str, float]:
    if spy_prices is None or len(spy_prices) == 0:
        return {}
    spy = pd.to_numeric(spy_prices, errors="coerce").dropna().sort_index()
    if spy.empty:
        return {}
    quarterly = spy.resample("QE").last().dropna()
    out: Dict[str, float] = {}
    for i in range(1, len(quarterly)):
        p0 = float(quarterly.iloc[i - 1])
        p1 = float(quarterly.iloc[i])
        if p0 > 0:
            out[str(quarterly.index[i].to_period("Q"))] = float(p1 / p0 - 1.0)
    return out


def build_alpha_target(
    df: pd.DataFrame,
    *,
    spy_prices: Optional[pd.Series] = None,
    sector_map: Optional[Dict[str, str]] = None,
    mode: str = "sector",
    min_sector_peers: int = 10,
) -> pd.Series:
    """Compute forward excess return (alpha) with sector/SPY benchmark."""
    if df is None or df.empty or "forward_return" not in df.columns:
        return pd.Series(dtype=float)

    fwd = pd.to_numeric(df["forward_return"], errors="coerce")
    idx = df.index
    dates = idx.get_level_values("date") if isinstance(idx, pd.MultiIndex) else pd.Index(df["date"])
    q = dates.to_period("Q")

    universe_benchmark = fwd.groupby(q).transform("median")

    spy_lookup = _spy_quarterly_returns(spy_prices)
    spy_benchmark = pd.Series([spy_lookup.get(str(x), np.nan) for x in q], index=idx, dtype=float)

    sector_benchmark = pd.Series(np.nan, index=idx, dtype=float)
    if sector_map is not None and len(sector_map) > 0 and isinstance(idx, pd.MultiIndex):
        tickers = idx.get_level_values("ticker").astype(str)
        sector_series = pd.Series(tickers, index=idx).map(sector_map).fillna("Unknown")
        temp = pd.DataFrame(
            {
                "fwd": fwd.values,
                "sector": sector_series.values,
                "quarter": q.astype(str),
            },
            index=idx,
        )
        grp = ["sector", "quarter"]
        med = temp.groupby(grp)["fwd"].transform("median")
        cnt = temp.groupby(grp)["fwd"].transform("count")
        enough = (temp["sector"] != "Unknown") & (cnt >= int(min_sector_peers))
        sector_benchmark = pd.Series(np.where(enough, med, np.nan), index=idx, dtype=float)

    mode_norm = str(mode).strip().lower()
    if mode_norm == "spy":
        benchmark = spy_benchmark.copy()
        benchmark = benchmark.fillna(universe_benchmark)
    elif mode_norm == "sector":
        benchmark = sector_benchmark.copy()
        benchmark = benchmark.fillna(spy_benchmark)
        benchmark = benchmark.fillna(universe_benchmark)
    else:
        benchmark = universe_benchmark.copy()

    alpha = (fwd - benchmark).replace([np.inf, -np.inf], np.nan)
    return alpha


def alpha_to_quintiles(alpha: pd.Series, dates: pd.Index) -> pd.Series:
    """Discretize alpha into 5 ordinal buckets by timestamp."""
    if alpha is None or alpha.empty:
        return pd.Series(dtype=float)

    periods = pd.PeriodIndex(dates.to_period("Q"), freq="Q")
    out = pd.Series(np.nan, index=alpha.index, dtype=float)

    for period in periods.unique():
        mask = periods == period
        vals = pd.to_numeric(alpha.loc[mask], errors="coerce").dropna()
        if len(vals) < 5:
            continue
        rank = vals.rank(method="first")
        q = np.ceil(rank / len(vals) * 5.0).clip(1, 5)
        out.loc[vals.index] = q.astype(float)
    return out


def _close_series(prices_obj) -> pd.Series:
    if prices_obj is None:
        return pd.Series(dtype=float)
    if isinstance(prices_obj, pd.Series):
        s = pd.to_numeric(prices_obj, errors="coerce").dropna().sort_index()
        return s
    if isinstance(prices_obj, pd.DataFrame) and not prices_obj.empty:
        col = "Close" if "Close" in prices_obj.columns else prices_obj.columns[-1]
        s = pd.to_numeric(prices_obj[col], errors="coerce").dropna().sort_index()
        return s
    return pd.Series(dtype=float)


def _first_on_or_after(series: pd.Series, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    cand = series.index[series.index >= ts]
    return None if len(cand) == 0 else pd.Timestamp(cand[0])


def triple_barrier_labels(
    df: pd.DataFrame,
    *,
    prices_dict: Dict[str, object],
    pt_sl: float = 0.08,
    max_holding_months: int = 3,
    lag_days: int = 45,
) -> pd.Series:
    """Compute Lopez de Prado triple barrier labels (+1, 0, -1)."""
    if df is None or df.empty or not isinstance(df.index, pd.MultiIndex):
        return pd.Series(dtype=float)

    labels = pd.Series(np.nan, index=df.index, dtype=float)

    for (ticker, dt), row in df.iterrows():
        prices = _close_series(prices_dict.get(str(ticker)))
        if prices.empty:
            continue

        if "snapshot_date" in row and pd.notna(row.get("snapshot_date")):
            start = pd.Timestamp(row.get("snapshot_date"))
        else:
            start = pd.Timestamp(dt)
        entry_req = start + pd.Timedelta(days=max(int(lag_days), 0))
        entry_date = _first_on_or_after(prices, entry_req)
        if entry_date is None:
            continue

        entry_px = float(prices.loc[entry_date])
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue

        up = entry_px * (1.0 + float(pt_sl))
        dn = entry_px * (1.0 - float(pt_sl))
        expiry = entry_date + pd.DateOffset(months=max(int(max_holding_months), 1))

        window = prices.loc[(prices.index >= entry_date) & (prices.index <= expiry)]
        if len(window) < 2:
            continue

        label = 0.0
        for px in window.iloc[1:]:
            p = float(px)
            if p >= up:
                label = 1.0
                break
            if p <= dn:
                label = -1.0
                break
        labels.loc[(ticker, dt)] = label

    return labels


def build_targets(
    df: pd.DataFrame,
    *,
    prices_dict: Optional[Dict[str, object]] = None,
    spy_prices: Optional[pd.Series] = None,
    sector_map: Optional[Dict[str, str]] = None,
    benchmark_mode: str = "sector",
    min_sector_peers: int = 10,
    triple_barrier_pt_sl: float = 0.08,
    lag_days: int = 45,
    holding_period_months: int = 3,
) -> TargetBundle:
    """Build full target set for ranking/optimization-aware training."""
    alpha = build_alpha_target(
        df,
        spy_prices=spy_prices,
        sector_map=sector_map,
        mode=benchmark_mode,
        min_sector_peers=min_sector_peers,
    )
    dates = df.index.get_level_values("date") if isinstance(df.index, pd.MultiIndex) else pd.to_datetime(df["date"])
    quint = alpha_to_quintiles(alpha, dates)

    if prices_dict:
        tb = triple_barrier_labels(
            df,
            prices_dict=prices_dict,
            pt_sl=triple_barrier_pt_sl,
            max_holding_months=holding_period_months,
            lag_days=lag_days,
        )
    else:
        tb = pd.Series(np.nan, index=df.index, dtype=float)

    direction = (alpha > 0).astype(float)
    benchmark = pd.to_numeric(df["forward_return"], errors="coerce") - alpha
    return TargetBundle(
        alpha=alpha,
        quintile=quint,
        triple_barrier=tb,
        direction=direction,
        benchmark_return=benchmark,
    )
