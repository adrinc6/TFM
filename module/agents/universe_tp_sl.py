"""Universal TP/SL agent with dynamic feature selection and model search.

This agent is designed for the target:
  P(TP hit before SL within horizon)

Key behavior:
  - Trains on the full universe (no per-sector model split).
  - Selects between 6 and 10 base metrics from agent-specific candidates.
  - Builds percentile transforms for each selected metric:
      * within sector (same snapshot date)
      * within universe (same snapshot date)
  - Tries multiple model families and keeps the best by temporal CV AUC.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from module.agents.base import BaseAgent
from environment import (
    ENABLE_AGENT_RULE_ENGINE,
    AGENT_RULE_N_BINS,
    AGENT_RULE_MIN_SAMPLES,
    AGENT_RULE_MIN_EDGE,
    AGENT_RULE_MAX_RULES,
    AGENT_RULE_MAX_PAIR_FEATURES,
    AGENT_RULE_MIN_STABILITY,
    AGENT_RULE_BLEND,
    AGENT_RULE_SECTOR_MIN_SAMPLES,
)

log = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    _SK_OK = True
except Exception:
    _SK_OK = False

try:
    import xgboost as xgb

    _XGB_OK = True
except Exception:
    _XGB_OK = False

# Suffixes that mark a feature as derived from temporal history (deltas,
# momentum, trend).  Rules referencing these are skipped at inference when
# the frame is a single-period snapshot with no historical context.
_TEMPORAL_FEATURE_SUFFIXES = (
    "_delta_1q",
    "_delta_2q",
    "_delta_4q",
    "_delta_8q",
    "_momentum_1q",
    "_momentum_4q",
    "_trend_4q",
)


class UniversalTpSlAgent(BaseAgent):
    """Single-universe TP/SL classifier with percentile-aware feature engineering."""

    def __init__(
        self,
        name: str,
        results_dir: str,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        random_seed: int = 42,
        min_features: int = 6,
        max_features: int = 10,
        neutral_score: float = 0.5,
        save_artifacts: bool = True,
    ):
        super().__init__(name=name, results_dir=results_dir, random_seed=random_seed, save_artifacts=save_artifacts)
        if not _SK_OK:
            raise ImportError("scikit-learn is required for UniversalTpSlAgent")

        self.include_features = list(include_features or [])
        self.exclude_features = set(exclude_features or [])
        self.min_features = int(max(3, min_features))
        self.max_features = int(max(self.min_features, max_features))
        self._neutral_score = float(neutral_score)

        self._selected_base_features: List[str] = []
        self._feature_cols: List[str] = []
        self._model = None
        self._best_model_name: str = ""
        self._cv_summary: Dict[str, Dict[str, float]] = {}
        self._rule_engine_enabled = bool(ENABLE_AGENT_RULE_ENGINE)
        self._rule_blend = float(np.clip(AGENT_RULE_BLEND, 0.0, 1.0))
        self._rule_min_stability = float(np.clip(AGENT_RULE_MIN_STABILITY, 0.0, 1.0))
        self._global_tp_rate: float = 0.5
        self._feature_bin_thresholds: Dict[str, List[float]] = {}
        self._rules: List[Dict[str, Any]] = []
        # Per-sector rule map: {sector: {"rules": [...], "bin_thresholds": {...}, "tp_rate": float}}
        self._sector_rules_map: Dict[str, Any] = {}
        self._last_rule_signal: Optional[pd.Series] = None
        self._last_rule_hits: Optional[pd.Series] = None
        self._last_rule_confidence: Optional[pd.Series] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fold: Optional[int] = None,
        sector_col: str = "sector",
    ) -> "UniversalTpSlAgent":
        if X is None or X.empty:
            return self

        X_raw = self._prepare_base_candidate_frame(X)
        X_raw, y_aligned = self.clean_features(X_raw, y)
        if X_raw.empty or y_aligned is None or len(X_raw) < 30:
            log.warning("[%s] Not enough clean samples to train.", self.name)
            return self
        if not self.has_multiple_classes(y_aligned):
            log.warning("[%s] Target has a single class in this fold.", self.name)
            return self

        self._selected_base_features = self._select_base_features(X_raw, y_aligned)
        if not self._selected_base_features:
            log.warning("[%s] No base features selected.", self.name)
            return self

        X_base_selected = X_raw[self._selected_base_features].copy()
        sector_series = X.loc[X_raw.index, sector_col] if sector_col in X.columns else None
        if self._rule_engine_enabled:
            self._fit_rule_engine(X_base_selected, y_aligned, sector_series=sector_series)
        X_model = self._build_model_frame(X_raw[self._selected_base_features], sector_series)
        if self._rule_engine_enabled and self._rules:
            rule_frame = self._build_rule_frame(X_base_selected, sector_series=sector_series)
            rs, rh, rc = self._compute_rule_signal(rule_frame, sector_series=sector_series)
            X_model["rule_signal"] = rs.reindex(X_model.index).fillna(0.0)
            X_model["rule_hits"] = rh.reindex(X_model.index).fillna(0.0)
            X_model["rule_confidence"] = rc.reindex(X_model.index).fillna(0.0)
        X_model, y_aligned = self.clean_features(X_model, y_aligned)

        if X_model.empty or len(X_model) < 30 or not self.has_multiple_classes(y_aligned):
            log.warning("[%s] Feature frame invalid after percentile engineering.", self.name)
            return self

        self._feature_cols = list(X_model.columns)
        best_name, best_model, cv_summary = self._choose_best_model(X_model, y_aligned)
        self._cv_summary = cv_summary
        self._best_model_name = best_name

        if best_model is None:
            log.warning("[%s] Model search failed.", self.name)
            return self

        best_model.fit(X_model, y_aligned)
        self._model = best_model
        self.is_trained = True

        importances = self._model_importance_series(self._model, self._feature_cols)
        self.save_feature_importances(importances, fold)

        self._diagnostics = {
            "selected_base_features": self._selected_base_features,
            "n_base_features": int(len(self._selected_base_features)),
            "n_model_features": int(len(self._feature_cols)),
            "best_model": self._best_model_name,
            "cv_summary": self._cv_summary,
            "top_features": importances.nlargest(15).to_dict(),
            "rule_engine_enabled": bool(self._rule_engine_enabled),
            "rule_count": int(len(self._rules)),
            "global_tp_rate": float(self._global_tp_rate),
            "rule_blend": float(self._rule_blend),
            "sector_rule_counts": {
                sec: len(data["rules"]) for sec, data in self._sector_rules_map.items()
            },
            "top_rules": self._rules[:10],
        }
        self.record_train_metrics(
            {
                "best_cv_auc": float(self._cv_summary.get(self._best_model_name, {}).get("mean_auc", 0.0)),
                "n_features": float(len(self._feature_cols)),
            },
            fold=fold,
        )
        self.save_diagnostics(fold)
        self._save_rules_artifact(fold)
        return self

    def predict_score(self, X: pd.DataFrame, sector_col: str = "sector") -> pd.Series:
        if not self.is_trained or self._model is None:
            raise RuntimeError(f"[{self.name}] Not trained.")

        X_raw = self._prepare_base_candidate_frame(X)
        for col in self._selected_base_features:
            if col not in X_raw.columns:
                X_raw[col] = 0.0
        X_raw = X_raw[self._selected_base_features]
        X_raw = self.clean_features_predict(X_raw)

        sector_series = X.loc[X_raw.index, sector_col] if sector_col in X.columns else None
        rule_signal = pd.Series(0.0, index=X_raw.index, dtype=float)
        rule_hits = pd.Series(0.0, index=X_raw.index, dtype=float)
        rule_conf = pd.Series(0.0, index=X_raw.index, dtype=float)
        if self._rule_engine_enabled and self._rules:
            rule_frame = self._build_rule_frame(X_raw, sector_series=sector_series)
            rule_signal, rule_hits, rule_conf = self._compute_rule_signal(rule_frame, sector_series=sector_series)
        X_model = self._build_model_frame(X_raw, sector_series)
        if self._rule_engine_enabled and self._rules:
            X_model["rule_signal"] = rule_signal.reindex(X_model.index).fillna(0.0)
            X_model["rule_hits"] = rule_hits.reindex(X_model.index).fillna(0.0)
            X_model["rule_confidence"] = rule_conf.reindex(X_model.index).fillna(0.0)
        X_model = self.clean_features_predict(X_model)
        X_model = self._align_to_feature_cols(X_model, fill_value=0.0)

        if hasattr(self._model, "predict_proba"):
            score = self._model.predict_proba(X_model)[:, 1]
        else:
            raw = self._model.decision_function(X_model)
            score = 1.0 / (1.0 + np.exp(-raw))

        score = self._apply_calibration(score)
        score_series = pd.Series(np.clip(score, 0.0, 1.0), index=X_model.index, name=f"{self.name}_score")
        if self._rule_engine_enabled and self._rules and self._rule_blend > 0:
            signal = rule_signal.reindex(score_series.index).fillna(0.0)
            confidence = rule_conf.reindex(score_series.index).fillna(0.0).clip(0.0, 1.0)
            score_series = (score_series + self._rule_blend * confidence * signal).clip(0.0, 1.0)

        self._last_rule_signal = rule_signal.reindex(score_series.index).fillna(0.0)
        self._last_rule_hits = rule_hits.reindex(score_series.index).fillna(0.0)
        self._last_rule_confidence = rule_conf.reindex(score_series.index).fillna(0.0)
        return score_series

    def get_last_rule_details(self) -> Dict[str, pd.Series]:
        idx = None
        if self._last_rule_signal is not None:
            idx = self._last_rule_signal.index
        elif self._last_rule_hits is not None:
            idx = self._last_rule_hits.index
        elif self._last_rule_confidence is not None:
            idx = self._last_rule_confidence.index
        else:
            idx = pd.Index([])
        return {
            "rule_signal": self._last_rule_signal if self._last_rule_signal is not None else pd.Series(0.0, index=idx, dtype=float),
            "rule_hits": self._last_rule_hits if self._last_rule_hits is not None else pd.Series(0.0, index=idx, dtype=float),
            "rule_confidence": self._last_rule_confidence if self._last_rule_confidence is not None else pd.Series(0.0, index=idx, dtype=float),
        }

    def _fit_rule_engine(
        self, X_base: pd.DataFrame, y: pd.Series, sector_series: Optional[pd.Series] = None
    ) -> None:
        """Mine sector-specific bin rules correlated with P(TP hit).

        Rules are mined independently per sector.  Sectors with fewer than
        ``AGENT_RULE_SECTOR_MIN_SAMPLES`` observations are pooled together
        under ``_other`` so their signal is not wasted.  When no sector
        information is available, a single ``_global`` group is mined.
        """
        self._rules = []
        self._feature_bin_thresholds = {}
        self._sector_rules_map = {}

        y_al = pd.to_numeric(y.reindex(X_base.index), errors="coerce").dropna()
        if y_al.empty:
            self._global_tp_rate = 0.5
            return
        Xb = X_base.reindex(y_al.index).copy()
        Xb = self._build_rule_frame(Xb, sector_series=sector_series)
        self._global_tp_rate = float(y_al.mean())

        sector_min = max(int(AGENT_RULE_SECTOR_MIN_SAMPLES), int(AGENT_RULE_MIN_SAMPLES) * 2)
        all_rules: List[Dict[str, Any]] = []

        if sector_series is not None:
            sec_al = sector_series.reindex(y_al.index).fillna("_other").astype(str)
            sector_counts = sec_al.value_counts()
            valid_sectors = sector_counts[sector_counts >= sector_min].index.tolist()

            # Pool sectors without enough data into "_other"
            sec_effective = sec_al.copy()
            small_mask = ~sec_al.isin(valid_sectors)
            if small_mask.any():
                sec_effective[small_mask] = "_other"

            sectors_to_mine = list(valid_sectors)
            if small_mask.any() and int(small_mask.sum()) >= sector_min:
                sectors_to_mine.append("_other")
        else:
            sec_effective = pd.Series("_global", index=y_al.index)
            sectors_to_mine = ["_global"]

        for sector in sectors_to_mine:
            mask = sec_effective == sector
            n = int(mask.sum())
            if n < sector_min:
                continue
            X_sec = Xb[mask]
            y_sec = y_al[mask]
            if y_sec.nunique() < 2:
                log.debug("[%s] Sector %s: single class — skip rule mining.", self.name, sector)
                continue
            rules_sec, thresh_sec, tp_rate_sec = self._mine_rules_for_group(
                X_sec, y_sec, sector_name=sector
            )
            if rules_sec:
                self._sector_rules_map[sector] = {
                    "rules": rules_sec,
                    "bin_thresholds": thresh_sec,
                    "tp_rate": tp_rate_sec,
                }
                all_rules.extend(rules_sec)
                log.debug(
                    "[%s] Sector %-24s | n=%4d  tp_rate=%.2f  rules=%d",
                    self.name, sector, n, tp_rate_sec, len(rules_sec),
                )

        self._rules = sorted(all_rules, key=lambda rr: float(rr.get("weight", 0.0)), reverse=True)
        log.debug(
            "[%s] Rule engine: %d sector(s) mined, %d total rules.",
            self.name, len(self._sector_rules_map), len(self._rules),
        )

    def _mine_rules_for_group(
        self,
        Xb: pd.DataFrame,
        y_al: pd.Series,
        sector_name: str = "_global",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]], float]:
        """Mine single-feature and pairwise bin rules for one sector group."""
        rules: List[Dict[str, Any]] = []
        bin_thresholds: Dict[str, List[float]] = {}
        tp_rate = float(y_al.mean()) if not y_al.empty else 0.5

        min_samples = max(int(AGENT_RULE_MIN_SAMPLES), 20)
        min_edge = float(max(AGENT_RULE_MIN_EDGE, 0.0))
        n_bins = max(3, int(AGENT_RULE_N_BINS))
        min_stability = float(np.clip(self._rule_min_stability, 0.0, 1.0))

        # Rank features by absolute correlation with TP target
        feature_rank: List[Tuple[str, float]] = []
        for col in Xb.columns:
            s = pd.to_numeric(Xb[col], errors="coerce")
            valid = s.notna() & y_al.notna()
            if int(valid.sum()) < min_samples:
                continue
            corr = s[valid].corr(y_al[valid])
            if pd.notna(corr):
                feature_rank.append((str(col), float(abs(corr))))
        feature_rank = sorted(feature_rank, key=lambda kv: kv[1], reverse=True)
        ordered_features = [c for c, _ in feature_rank] if feature_rank else list(Xb.columns)

        binned_cache: Dict[str, pd.Series] = {}
        for col in ordered_features:
            s_col = pd.to_numeric(Xb[col], errors="coerce")
            binned = self._build_quantile_bins(s_col, n_bins=n_bins)
            binned_cache[col] = binned
            thresholds = self._infer_bin_thresholds(s_col, binned)
            if thresholds:
                bin_thresholds[col] = thresholds

            grp = pd.DataFrame({"bin": binned, "y": y_al}).dropna()
            if grp.empty:
                continue
            grouped = grp.groupby("bin", dropna=True)["y"].agg(["mean", "count"]).reset_index()
            for _, r in grouped.iterrows():
                cnt = int(r.get("count", 0))
                if cnt < min_samples:
                    continue
                win = float(r.get("mean", 0.0))
                edge = win - tp_rate
                if abs(edge) < min_edge:
                    continue
                mask = pd.to_numeric(binned, errors="coerce") == int(r.get("bin"))
                stability = self._rule_stability(mask, y_al)
                if stability < min_stability:
                    continue
                is_temporal = any(col.endswith(sfx) for sfx in _TEMPORAL_FEATURE_SUFFIXES)
                rules.append(
                    {
                        "type": "single",
                        "sector": sector_name,
                        "features": [str(col)],
                        "bins": [int(r.get("bin"))],
                        "support": cnt,
                        "win_rate": win,
                        "edge": edge,
                        "stability": stability,
                        "weight": float(abs(edge) * np.log1p(cnt) * (0.5 + 0.8 * stability)),
                        "requires_temporal": bool(is_temporal),
                    }
                )

        # Pairwise rules restricted to top features to cap combinatorial cost
        top_pair = ordered_features[: max(2, int(AGENT_RULE_MAX_PAIR_FEATURES))]
        for i in range(len(top_pair)):
            f1 = top_pair[i]
            b1 = binned_cache.get(f1)
            if b1 is None:
                continue
            for j in range(i + 1, len(top_pair)):
                f2 = top_pair[j]
                b2 = binned_cache.get(f2)
                if b2 is None:
                    continue
                tmp = pd.DataFrame({"b1": b1, "b2": b2, "y": y_al}).dropna()
                if tmp.empty:
                    continue
                grouped = tmp.groupby(["b1", "b2"], dropna=True)["y"].agg(["mean", "count"]).reset_index()
                for _, r in grouped.iterrows():
                    cnt = int(r.get("count", 0))
                    if cnt < min_samples:
                        continue
                    win = float(r.get("mean", 0.0))
                    edge = win - tp_rate
                    if abs(edge) < min_edge:
                        continue
                    mask = (pd.to_numeric(b1, errors="coerce") == int(r.get("b1"))) & (
                        pd.to_numeric(b2, errors="coerce") == int(r.get("b2"))
                    )
                    stability = self._rule_stability(mask, y_al)
                    if stability < min_stability:
                        continue
                    is_temporal = any(
                        f1.endswith(sfx) or f2.endswith(sfx) for sfx in _TEMPORAL_FEATURE_SUFFIXES
                    )
                    rules.append(
                        {
                            "type": "pair",
                            "sector": sector_name,
                            "features": [str(f1), str(f2)],
                            "bins": [int(r.get("b1")), int(r.get("b2"))],
                            "support": cnt,
                            "win_rate": win,
                            "edge": edge,
                            "stability": stability,
                            "weight": float(abs(edge) * np.log1p(cnt) * 1.15 * (0.5 + 0.8 * stability)),
                            "requires_temporal": bool(is_temporal),
                        }
                    )

        rules = sorted(rules, key=lambda rr: float(rr.get("weight", 0.0)), reverse=True)
        rules = rules[: max(1, int(AGENT_RULE_MAX_RULES))]
        return rules, bin_thresholds, tp_rate

    def _compute_signal_for_group(
        self,
        X_group: pd.DataFrame,
        rules: List[Dict[str, Any]],
        bin_thresholds: Dict[str, List[float]],
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Compute (signal, confidence, hit_count) for a single sector rule set.

        signal     = weighted average edge, bounded [-1, 1]
        confidence = fraction of sector rules matched, bounded [0, 1]
        hit_count  = raw count of matched rules (diagnostic)
        """
        idx = X_group.index
        num = pd.Series(0.0, index=idx, dtype=float)
        den = pd.Series(0.0, index=idx, dtype=float)
        hits = pd.Series(0.0, index=idx, dtype=float)

        avail = set(X_group.columns)
        binned_cache: Dict[str, pd.Series] = {}
        for col in {f for r in rules for f in r.get("features", [])}:
            if col not in avail:
                continue
            binned_cache[col] = self._assign_bins_with_thresholds(
                pd.to_numeric(X_group[col], errors="coerce"),
                bin_thresholds.get(col, []),
            )

        for rule in rules:
            feats = list(rule.get("features", []))
            bins_r = list(rule.get("bins", []))
            if not feats or not bins_r or len(feats) != len(bins_r):
                continue
            # Skip temporal rules when the frame lacks those columns
            if rule.get("requires_temporal") and not all(f in avail for f in feats):
                continue
            mask = pd.Series(True, index=idx)
            for feat, bb in zip(feats, bins_r):
                bser = binned_cache.get(feat)
                if bser is None:
                    mask &= False
                    continue
                mask &= (pd.to_numeric(bser, errors="coerce") == int(bb))
            if not bool(mask.any()):
                continue
            w = float(rule.get("weight", 0.0))
            edge = float(rule.get("edge", 0.0))
            num.loc[mask] += edge * w
            den.loc[mask] += w
            hits.loc[mask] += 1.0

        signal = (num / den.replace(0.0, np.nan)).fillna(0.0).clip(-1.0, 1.0)
        confidence = (hits / max(1.0, float(len(rules)))).clip(0.0, 1.0)
        return signal, confidence, hits

    def _build_rule_frame(
        self,
        X: pd.DataFrame,
        sector_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Build the feature frame used for rule mining and rule-signal inference.

        Columns produced (in order):
          1. Raw base metrics (from X)
          2. Multi-timeframe temporal features (_delta_*q, _momentum_*q, _trend_4q)
          3. Sector percentile of each base metric ({col}__pct_sector)
             — only when sector_series is provided, computed cross-sectionally
               per (snapshot_date, sector) group so that rules capture where a
               stock sits relative to its sector peers.
        """
        if X is None or X.empty:
            return pd.DataFrame(index=(X.index if X is not None else None))
        out = X.copy()
        # Step 1: temporal deltas computed on raw metrics only (not on derived cols)
        temporal_frame = self._build_multiframe_deltas(X)
        if not temporal_frame.empty:
            out = pd.concat([out, temporal_frame], axis=1)
        # Step 2: sector percentile of each base metric (cross-sectional per snapshot)
        if sector_series is not None:
            date_key = self._resolve_date_key(X.index)
            sec = sector_series.reindex(X.index).fillna("Unknown").astype(str)
            sec_key = (
                pd.Series(date_key.astype(str), index=X.index).astype(str)
                + "|"
                + sec
            )
            for col in X.columns:
                out[f"{col}__pct_sector"] = self._percentile_by_group(X[col], sec_key)
        return out

    @staticmethod
    def _rule_stability(mask: pd.Series, y: pd.Series) -> float:
        m = pd.Series(mask, index=y.index).fillna(False).astype(bool)
        y_num = pd.to_numeric(y, errors="coerce")
        valid = m & y_num.notna()
        n = int(valid.sum())
        if n < 25:
            return 0.0

        idx = y_num.index
        if isinstance(idx, pd.MultiIndex) and "date" in idx.names:
            date_key = pd.PeriodIndex(pd.to_datetime(idx.get_level_values("date"), errors="coerce"), freq="Q").astype(str)
            tmp = pd.DataFrame({"q": date_key, "y": y_num, "m": m})
            per_q = (
                tmp.loc[tmp["m"] & tmp["y"].notna()]
                .groupby("q", dropna=True)["y"]
                .agg(["mean", "count"])
            )
            per_q = per_q[per_q["count"] >= 6]
            if len(per_q) >= 3:
                std = float(np.nanstd(per_q["mean"].to_numpy(dtype=float)))
                stability = float(np.clip(1.0 - (std / 0.25), 0.0, 1.0))
                coverage = float(np.clip(len(per_q) / 8.0, 0.0, 1.0))
                return float(0.7 * stability + 0.3 * coverage)

        # Fallback for sparse or non-temporal indices.
        return 0.55

    @staticmethod
    def _build_multiframe_deltas(X: pd.DataFrame) -> pd.DataFrame:
        """Compute multi-timeframe delta, momentum, and trend features per ticker.

        For each numeric column the following temporal views are produced when a
        (ticker, date) MultiIndex is available and enough historical periods exist:

          _delta_1q    — quarter-over-quarter change (1 period)
          _delta_2q    — 6-month change (2 periods)
          _delta_4q    — year-over-year change (4 periods)
          _delta_8q    — 2-year change (8 periods)
          _momentum_1q — acceleration of QoQ change (2nd derivative)
          _momentum_4q — acceleration of YoY change (2nd derivative)
          _trend_4q    — 4-period rolling mean (level trend)

        Each column is only emitted when >= 30 valid non-null values exist.
        """
        if not isinstance(X.index, pd.MultiIndex):
            return pd.DataFrame()
        idx_names = list(X.index.names)
        if "ticker" not in idx_names or "date" not in idx_names:
            return pd.DataFrame()

        df = X.copy()
        df["_ticker"] = df.index.get_level_values("ticker")
        df["_date"] = df.index.get_level_values("date")
        df = df.sort_values(["_ticker", "_date"])

        result: Dict[str, pd.Series] = {}
        min_valid = 30

        for col in X.columns:  # iterate X.columns to skip _ticker/_date helpers
            s = pd.to_numeric(df[col], errors="coerce")
            g = df["_ticker"]

            # ── Deltas ────────────────────────────────────────────────────────────
            d1 = s - s.groupby(g).shift(1)
            d2 = s - s.groupby(g).shift(2)
            d4 = s - s.groupby(g).shift(4)
            d8 = s - s.groupby(g).shift(8)

            if int(d1.notna().sum()) >= min_valid:
                result[f"{col}_delta_1q"] = d1
            if int(d2.notna().sum()) >= min_valid:
                result[f"{col}_delta_2q"] = d2
            if int(d4.notna().sum()) >= min_valid:
                result[f"{col}_delta_4q"] = d4
            if int(d8.notna().sum()) >= min_valid:
                result[f"{col}_delta_8q"] = d8

            # ── Momentum (acceleration = change-in-change) ────────────────────────
            if int(d1.notna().sum()) >= min_valid:
                mom1 = d1 - d1.groupby(g).shift(1)
                if int(mom1.notna().sum()) >= min_valid:
                    result[f"{col}_momentum_1q"] = mom1
            if int(d4.notna().sum()) >= min_valid:
                mom4 = d4 - d4.groupby(g).shift(4)
                if int(mom4.notna().sum()) >= min_valid:
                    result[f"{col}_momentum_4q"] = mom4

            # ── Trend: 4-period rolling mean per ticker ───────────────────────────
            trend4 = s.groupby(g).transform(lambda x: x.rolling(4, min_periods=2).mean())
            if int(trend4.notna().sum()) >= min_valid:
                result[f"{col}_trend_4q"] = trend4

        if not result:
            return pd.DataFrame()
        return pd.DataFrame(result, index=X.index)

    @staticmethod
    def _build_quantile_bins(s: pd.Series, n_bins: int) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce")
        valid = x.dropna()
        out = pd.Series(np.nan, index=x.index, dtype=float)
        if valid.nunique() <= 1:
            out.loc[valid.index] = 0
            return out
        q = min(max(int(n_bins), 3), int(valid.nunique()))
        try:
            out.loc[valid.index] = pd.qcut(valid, q=q, labels=False, duplicates="drop")
        except Exception:
            out.loc[valid.index] = pd.cut(valid, bins=q, labels=False, duplicates="drop")
        return pd.to_numeric(out, errors="coerce")

    @staticmethod
    def _infer_bin_thresholds(values: pd.Series, bins: pd.Series) -> List[float]:
        x = pd.to_numeric(values, errors="coerce")
        b = pd.to_numeric(bins, errors="coerce")
        tmp = pd.DataFrame({"x": x, "b": b}).dropna()
        if tmp.empty:
            return []
        thresholds: List[float] = []
        for bb in sorted(tmp["b"].astype(int).unique().tolist()):
            v = tmp.loc[tmp["b"].astype(int) == bb, "x"]
            if v.empty:
                continue
            thresholds.append(float(v.max()))
        return thresholds

    def _compute_rule_signal(
        self,
        X_base: pd.DataFrame,
        sector_series: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Return (signal, hit_count, confidence) using sector-specific rules.

        Each ticker is dispatched to its sector's rule set.  Sectors without a
        dedicated rule set fall back to ``_other`` then ``_global`` (if present).
        Confidence is normalised per-sector so it reflects the fraction of that
        sector's rules that fired, not the global rule pool size.
        """
        _empty = pd.Series(0.0, index=X_base.index if X_base is not None else pd.Index([]), dtype=float)
        if X_base is None or X_base.empty or not self._rules:
            return _empty, _empty.copy(), _empty.copy()

        signal = pd.Series(0.0, index=X_base.index, dtype=float)
        confidence = pd.Series(0.0, index=X_base.index, dtype=float)
        hits = pd.Series(0.0, index=X_base.index, dtype=float)

        if self._sector_rules_map and sector_series is not None:
            sec_al = sector_series.reindex(X_base.index).fillna("_other").astype(str)
            for sector in sec_al.unique():
                sector_mask = sec_al == sector
                if not bool(sector_mask.any()):
                    continue
                sector_data = (
                    self._sector_rules_map.get(sector)
                    or self._sector_rules_map.get("_other")
                    or self._sector_rules_map.get("_global")
                    or (next(iter(self._sector_rules_map.values())) if self._sector_rules_map else None)
                )
                if sector_data is None:
                    continue
                sig_sec, conf_sec, h_sec = self._compute_signal_for_group(
                    X_base[sector_mask],
                    sector_data["rules"],
                    sector_data["bin_thresholds"],
                )
                signal.update(sig_sec)
                confidence.update(conf_sec)
                hits.update(h_sec)
        else:
            # No sector info: use best available rule set
            fallback = (
                self._sector_rules_map.get("_global")
                or self._sector_rules_map.get("_other")
                or (next(iter(self._sector_rules_map.values())) if self._sector_rules_map else None)
            )
            if fallback is None:
                return _empty, _empty.copy(), _empty.copy()
            signal, confidence, hits = self._compute_signal_for_group(
                X_base, fallback["rules"], fallback["bin_thresholds"]
            )

        return signal, hits, confidence

    @staticmethod
    def _assign_bins_with_thresholds(values: pd.Series, thresholds: List[float]) -> pd.Series:
        x = pd.to_numeric(values, errors="coerce")
        if not thresholds:
            return pd.Series(np.nan, index=x.index, dtype=float)
        thr = np.asarray([float(t) for t in thresholds if np.isfinite(t)], dtype=float)
        if thr.size == 0:
            return pd.Series(np.nan, index=x.index, dtype=float)
        bins = np.searchsorted(thr, x.to_numpy(dtype=float), side="left")
        out = pd.Series(bins, index=x.index, dtype=float)
        out.loc[x.isna()] = np.nan
        return out

    def _save_rules_artifact(self, fold: Optional[int] = None) -> None:
        if not self.save_artifacts or not self._rules:
            return
        suffix = f"_{fold}" if fold is not None else ""
        sector_summary = {
            sec: {
                "tp_rate": float(data.get("tp_rate", 0.5)),
                "rule_count": int(len(data.get("rules", []))),
                "top_rules": data.get("rules", [])[:5],
            }
            for sec, data in self._sector_rules_map.items()
        }
        payload = {
            "agent": self.name,
            "fold": fold,
            "global_tp_rate": float(self._global_tp_rate),
            "rule_blend": float(self._rule_blend),
            "rule_count": int(len(self._rules)),
            "sector_count": int(len(self._sector_rules_map)),
            "sector_summary": sector_summary,
            "all_rules": self._rules,
        }
        path = Path(self.results_dir) / f"rules{suffix}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _prepare_base_candidate_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.include_features if c in X.columns and c not in self.exclude_features]
        if not available:
            numeric_cols = [
                c for c in X.columns
                if c not in self.exclude_features and pd.api.types.is_numeric_dtype(X[c])
            ]
            available = numeric_cols
        return X[available].copy()

    def _select_base_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        if X.empty:
            return []

        # Rank by absolute point-biserial proxy (Pearson on binary y).
        scores: Dict[str, float] = {}
        y_num = pd.to_numeric(y, errors="coerce")
        for col in X.columns:
            s = pd.to_numeric(X[col], errors="coerce")
            valid = s.notna() & y_num.notna()
            if valid.sum() < 20:
                continue
            corr = s[valid].corr(y_num[valid])
            if pd.notna(corr):
                scores[col] = float(abs(corr))

        ranked = [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
        if not ranked:
            ranked = list(X.columns)

        keep = int(min(self.max_features, max(self.min_features, min(len(ranked), self.max_features))))
        selected = ranked[:keep]

        if len(selected) < self.min_features:
            for col in X.columns:
                if col not in selected:
                    selected.append(col)
                if len(selected) >= self.min_features:
                    break
        return selected[: self.max_features]

    def _build_model_frame(
        self,
        X_base: pd.DataFrame,
        sector_series: Optional[pd.Series],
    ) -> pd.DataFrame:
        out = X_base.copy()
        date_key = self._resolve_date_key(X_base.index)

        for col in list(X_base.columns):
            out[f"{col}__pct_universe"] = self._percentile_by_group(X_base[col], date_key)

            if sector_series is not None:
                sec = sector_series.reindex(X_base.index).fillna("Unknown").astype(str)
                sec_key = pd.Series(date_key.astype(str), index=X_base.index).astype(str) + "|" + sec
                out[f"{col}__pct_sector"] = self._percentile_by_group(X_base[col], sec_key)

        return out

    @staticmethod
    def _resolve_date_key(index: pd.Index) -> pd.Series:
        if isinstance(index, pd.MultiIndex) and "date" in index.names:
            return pd.Series(index.get_level_values("date"), index=index)
        return pd.Series(index, index=index)

    @staticmethod
    def _percentile_by_group(values: pd.Series, group_key: pd.Series) -> pd.Series:
        df = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"), "g": group_key})
        ranked = df.groupby("g")["v"].rank(pct=True, method="average")
        return ranked.fillna(0.5).astype(float)

    def _choose_best_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[str, Optional[object], Dict[str, Dict[str, float]]]:
        candidates = self._model_candidates()
        summary: Dict[str, Dict[str, float]] = {}

        best_name = ""
        best_model = None
        best_auc = -1.0

        for name, model in candidates.items():
            mean_auc, std_auc = self._temporal_cv_auc(model, X, y)
            summary[name] = {"mean_auc": float(mean_auc), "std_auc": float(std_auc)}
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_name = name
                best_model = model

        return best_name, best_model, summary

    def _model_candidates(self) -> Dict[str, object]:
        models: Dict[str, object] = {
            "logistic": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=400,
                            class_weight="balanced",
                            random_state=self.random_seed,
                        ),
                    ),
                ]
            ),
            # min_samples_leaf=10 consistently applied across all tree-based
            # models to enforce the same regularization floor: at least 10
            # training samples must be in every leaf.  This prevents individual
            # tickers/quarters from dominating splits and is the key guard
            # against overfitting on the ~5Y quarterly dataset.
            "random_forest": RandomForestClassifier(
                n_estimators=320,
                max_depth=6,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=self.random_seed,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=220,
                max_depth=2,
                learning_rate=0.04,
                subsample=0.80,
                min_samples_leaf=10,
                random_state=self.random_seed,
            ),
        }

        if _XGB_OK:
            models["xgboost"] = xgb.XGBClassifier(
                n_estimators=260,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.80,
                colsample_bytree=0.75,
                reg_lambda=2.0,
                reg_alpha=0.5,
                min_child_weight=10,
                eval_metric="auc",
                random_state=self.random_seed,
                n_jobs=-1,
                tree_method="hist",
            )

        return models

    def _temporal_cv_auc(self, model: object, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float]:
        if len(X) < 50:
            return 0.0, 0.0

        X_sorted, y_sorted = self._sort_by_time(X, y)
        n_splits = max(2, min(5, len(X_sorted) // 30))
        if len(X_sorted) <= n_splits:
            return 0.0, 0.0

        tscv = TimeSeriesSplit(n_splits=n_splits)
        aucs: List[float] = []

        for tr, val in tscv.split(X_sorted):
            y_tr = y_sorted.iloc[tr]
            y_val = y_sorted.iloc[val]
            if y_tr.nunique() < 2 or y_val.nunique() < 2:
                continue
            model.fit(X_sorted.iloc[tr], y_tr)
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X_sorted.iloc[val])[:, 1]
            else:
                raw = model.decision_function(X_sorted.iloc[val])
                p = 1.0 / (1.0 + np.exp(-raw))
            aucs.append(float(roc_auc_score(y_val, p)))

        if not aucs:
            return 0.0, 0.0
        return float(np.mean(aucs)), float(np.std(aucs))

    @staticmethod
    def _sort_by_time(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        if isinstance(X.index, pd.MultiIndex) and "date" in X.index.names:
            order = pd.Series(X.index.get_level_values("date")).argsort()
            Xs = X.iloc[order]
            ys = y.reindex(Xs.index)
            return Xs, ys
        return X, y.reindex(X.index)

    @staticmethod
    def _model_importance_series(model: object, feature_cols: List[str]) -> pd.Series:
        if isinstance(model, Pipeline):
            inner = model.named_steps.get("clf")
        else:
            inner = model

        if inner is not None and hasattr(inner, "feature_importances_"):
            arr = np.asarray(getattr(inner, "feature_importances_"), dtype=float)
            return pd.Series(arr, index=feature_cols).fillna(0.0)

        if inner is not None and hasattr(inner, "coef_"):
            coef = np.asarray(getattr(inner, "coef_"), dtype=float)
            if coef.ndim == 2:
                coef = np.abs(coef[0])
            else:
                coef = np.abs(coef)
            return pd.Series(coef, index=feature_cols).fillna(0.0)

        return pd.Series(np.zeros(len(feature_cols), dtype=float), index=feature_cols)
