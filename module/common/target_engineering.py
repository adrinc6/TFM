"""Target engineering utilities for TP/SL training targets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

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


class VolatilityRegimeTpSlLearner:
    """Learns TP/SL levels per volatility-regime cluster from historical price paths.

    Implements Option C (empirical quantiles) and Option A (clustering) from the
    problem specification: stocks are grouped into low/medium/high volatility
    regimes and empirical return distributions are computed per cluster to derive
    adaptive TP/SL levels.

    The fitted learner can then predict TP/SL for new observations based on
    their volatility regime assignment.

    Example::

        learner = VolatilityRegimeTpSlLearner(n_clusters=3)
        learner.fit(historical_df)
        tp, sl = learner.predict(live_df)
    """

    _REGIME_LOW = "low_vol"
    _REGIME_MED = "medium_vol"
    _REGIME_HIGH = "high_vol"

    def __init__(
        self,
        n_clusters: int = 3,
        tp_quantile: float = 0.75,
        sl_quantile: float = 0.25,
        min_samples_per_cluster: int = 20,
        volatility_col: str = "volatility_60d",
        return_col: str = "forward_return",
        random_seed: int = 42,
    ) -> None:
        """Initialise the learner.

        Args:
            n_clusters (int): Number of volatility regimes. 3 maps to
                low/medium/high. Values 1–10 are supported.
            tp_quantile (float): Upper quantile of positive returns used as the
                TP level (e.g. 0.75 = 75th percentile of wins).
            sl_quantile (float): Lower quantile of negative returns (absolute
                value) used as the SL level (e.g. 0.25 = 25th percentile of
                losses by magnitude).
            min_samples_per_cluster (int): Minimum observations required in a
                cluster to compute empirical quantiles. Clusters below this
                threshold fall back to the global default.
            volatility_col (str): Column containing annualised daily return
                standard deviation (used for regime assignment).
            return_col (str): Column containing the realised holding-period
                return for each observation (used to compute quantiles).
            random_seed (int): Random seed for reproducibility.
        """
        self.n_clusters = int(max(1, min(10, n_clusters)))
        self.tp_quantile = float(np.clip(tp_quantile, 0.5, 0.99))
        self.sl_quantile = float(np.clip(sl_quantile, 0.01, 0.5))
        self.min_samples = int(min_samples_per_cluster)
        self.volatility_col = str(volatility_col)
        self.return_col = str(return_col)
        self.random_seed = int(random_seed)

        self._vol_breakpoints: List[float] = []
        self._cluster_tp: Dict[str, float] = {}
        self._cluster_sl: Dict[str, float] = {}
        self._global_tp: float = _TP_MIN
        self._global_sl: float = _SL_MIN
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "VolatilityRegimeTpSlLearner":
        """Learn TP/SL distributions from historical observations.

        Args:
            df (pd.DataFrame): Historical DataFrame containing at least
                ``volatility_col`` and ``return_col`` columns.  Extra columns
                are ignored.

        Returns:
            VolatilityRegimeTpSlLearner: Fitted instance (self).
        """
        if df is None or df.empty:
            log.warning("[VolatilityRegimeTpSlLearner] Empty dataframe; learner not fitted.")
            return self

        vol = pd.to_numeric(df.get(self.volatility_col, pd.Series(dtype=float)), errors="coerce")
        ret = pd.to_numeric(df.get(self.return_col, pd.Series(dtype=float)), errors="coerce")
        valid = vol.notna() & ret.notna()
        vol, ret = vol[valid], ret[valid]

        if len(vol) < self.min_samples:
            log.warning(
                "[VolatilityRegimeTpSlLearner] Only %d valid rows; "
                "using global defaults.", len(vol),
            )
            self._fit_global(ret)
            self.is_fitted = True
            return self

        # Compute breakpoints that divide the vol distribution into n_clusters
        # equal-frequency bins (quantile-based clustering, Option A).
        # Deduplicate breakpoints that may collapse when volatility values are
        # concentrated (e.g. many identical readings), then warn the caller.
        raw_breakpoints = [
            float(np.nanpercentile(vol, 100.0 * i / self.n_clusters))
            for i in range(1, self.n_clusters)
        ]
        breakpoints = sorted(set(raw_breakpoints))
        if len(breakpoints) < len(raw_breakpoints):
            log.warning(
                "[VolatilityRegimeTpSlLearner] %d duplicate breakpoints collapsed "
                "to %d unique values — consider a larger dataset or fewer clusters.",
                len(raw_breakpoints) - len(breakpoints), len(breakpoints),
            )
        self._vol_breakpoints = breakpoints

        # Assign each row to a cluster label
        cluster_labels = self._assign_clusters(vol, breakpoints)

        self._fit_global(ret)

        for label in cluster_labels.unique():
            mask = cluster_labels == label
            cluster_ret = ret[mask]
            if len(cluster_ret) < self.min_samples:
                self._cluster_tp[label] = self._global_tp
                self._cluster_sl[label] = self._global_sl
                continue
            self._cluster_tp[label] = self._compute_tp(cluster_ret)
            self._cluster_sl[label] = self._compute_sl(cluster_ret)
            log.debug(
                "[VolatilityRegimeTpSlLearner] Cluster %s: n=%d tp=%.4f sl=%.4f",
                label, len(cluster_ret),
                self._cluster_tp[label], self._cluster_sl[label],
            )

        self.is_fitted = True
        log.info(
            "[VolatilityRegimeTpSlLearner] Fitted %d clusters. "
            "Global defaults: tp=%.4f sl=%.4f",
            len(self._cluster_tp), self._global_tp, self._global_sl,
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Return per-row TP/SL levels based on cluster assignment.

        Args:
            df (pd.DataFrame): Feature DataFrame with at least
                ``volatility_col``.  Must share the same index you want
                the returned Series to carry.

        Returns:
            Tuple[pd.Series, pd.Series]: ``(tp_series, sl_series)`` with the
            same index as ``df``, clipped to ``[_TP_MIN, _TP_MAX]`` and
            ``[_SL_MIN, _SL_MAX]`` respectively.
        """
        idx = df.index
        if not self.is_fitted:
            log.warning("[VolatilityRegimeTpSlLearner] Not fitted; returning global defaults.")
            return (
                pd.Series(self._global_tp or float(_TP_MIN), index=idx, dtype=float),
                pd.Series(self._global_sl or float(_SL_MIN), index=idx, dtype=float),
            )

        vol = pd.to_numeric(df.get(self.volatility_col, pd.Series(dtype=float)), errors="coerce")
        vol = vol.reindex(idx).fillna(vol.median() if vol.notna().any() else 0.0)

        cluster_labels = self._assign_clusters(vol, self._vol_breakpoints)

        tp_vals = cluster_labels.map(
            lambda l: self._cluster_tp.get(l, self._global_tp)
        ).astype(float).clip(_TP_MIN, _TP_MAX)

        sl_vals = cluster_labels.map(
            lambda l: self._cluster_sl.get(l, self._global_sl)
        ).astype(float).clip(_SL_MIN, _SL_MAX)

        return (
            pd.Series(tp_vals.values, index=idx, dtype=float),
            pd.Series(sl_vals.values, index=idx, dtype=float),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fit_global(self, ret: pd.Series) -> None:
        self._global_tp = float(np.clip(self._compute_tp(ret), _TP_MIN, _TP_MAX))
        self._global_sl = float(np.clip(self._compute_sl(ret), _SL_MIN, _SL_MAX))

    def _compute_tp(self, ret: pd.Series) -> float:
        """Upper quantile of positive returns (win side)."""
        wins = ret[ret > 0]
        if len(wins) == 0:
            return float(_TP_MIN)
        return float(np.clip(np.nanpercentile(wins, 100.0 * self.tp_quantile), _TP_MIN, _TP_MAX))

    def _compute_sl(self, ret: pd.Series) -> float:
        """Lower quantile of negative returns, expressed as positive magnitude (loss side)."""
        losses = ret[ret < 0].abs()
        if len(losses) == 0:
            return float(_SL_MIN)
        return float(np.clip(np.nanpercentile(losses, 100.0 * self.sl_quantile), _SL_MIN, _SL_MAX))

    @staticmethod
    def _assign_clusters(vol: pd.Series, breakpoints: List[float]) -> pd.Series:
        """Assign each observation to a volatility cluster label."""
        labels = pd.Series("cluster_0", index=vol.index, dtype=object)
        for i, bp in enumerate(breakpoints):
            labels[vol > bp] = f"cluster_{i + 1}"
        return labels

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return a summary of fitted cluster TP/SL levels.

        Returns:
            Dict: Mapping of cluster label → {tp, sl}.
        """
        out: Dict[str, Dict[str, float]] = {
            "_global": {"tp": self._global_tp, "sl": self._global_sl},
        }
        for label in sorted(self._cluster_tp):
            out[label] = {
                "tp": self._cluster_tp[label],
                "sl": self._cluster_sl.get(label, self._global_sl),
            }
        return out


def build_tp_sl_targets(
    df: pd.DataFrame,
    *,
    prices_dict: Dict[str, object],
    lag_days: int = 45,
    max_holding_days: int = 90,
    tp_default: float = 0.08,
    sl_default: float = 0.05,
    tp_sl_learner: Optional[VolatilityRegimeTpSlLearner] = None,
) -> TpSlTargetBundle:
    """Build TP/SL-first labels: 1 if TP is hit before SL within the horizon.

    When a fitted :class:`VolatilityRegimeTpSlLearner` is supplied, per-row
    TP/SL levels are derived from learned cluster distributions instead of the
    simple volatility-scaling fallback.

    Args:
        df (pd.DataFrame): Feature DataFrame with a (ticker, date) MultiIndex.
        prices_dict (Dict[str, object]): Mapping of ticker → price series or
            DataFrame.
        lag_days (int): Calendar days after the snapshot date before entry.
            Enforces the T1 = T0 + 45 days temporal structure.
        max_holding_days (int): Maximum holding period in calendar days (T2 −
            T1).
        tp_default (float): Default TP percentage used when no learner is
            provided and volatility data is unavailable.
        sl_default (float): Default SL percentage (same conditions as above).
        tp_sl_learner (Optional[VolatilityRegimeTpSlLearner]): Pre-fitted
            cluster learner.  If supplied, its ``predict()`` output is used
            instead of ``infer_tp_sl_levels()``.

    Returns:
        TpSlTargetBundle: Dataclass containing hit_label, outcome, tp_level,
            and sl_level series all aligned to ``df.index``.
    """
    if df is None or df.empty or not isinstance(df.index, pd.MultiIndex):
        raise ValueError("TP/SL target generation requires a non-empty MultiIndex DataFrame.")
    if not prices_dict:
        raise ValueError("TP/SL target generation requires non-empty prices_dict.")

    from module.steps.step_04_evaluation.strategy import simulate_tp_sl

    if tp_sl_learner is not None and tp_sl_learner.is_fitted:
        tp_level, sl_level = tp_sl_learner.predict(df)
    else:
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

