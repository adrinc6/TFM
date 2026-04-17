"""Abstract base agent and feature selector for the multi-agent stock picker system."""
import json
import logging
import numpy as np
import pandas as pd

from module.common.feature_policy import is_ratio_or_normalized_feature
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from environment import (
    FEATURE_SELECTOR_RELEVANCE_WEIGHT,
    FEATURE_SELECTOR_RF_N_ESTIMATORS,
    FEATURE_SELECTOR_RF_MAX_DEPTH,
    FEATURE_IMPORTANCE_CUTOFF_FRACTION,
    FEATURE_IMPORTANCE_MIN_KEEP,
    FEATURE_IMPORTANCE_MAX_KEEP,
)

log = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class defining the common contract for all agents.

    Each agent:
      - Receives domain-specific features (fundamentals, price data, etc.)
      - Trains without seeing future data (the pipeline enforces temporal order)
      - Returns a score in [0.0, 1.0] where 1 = bullish / Outperform signal
      - Saves complete diagnostics under results/agents/<name>/
    """

    def __init__(self, name: str, results_dir: str, random_seed: int = 42,
                 save_artifacts: bool = True):
        """Initialises the base agent.

        Args:
            name (str): Agent identifier used for file naming and logging.
            results_dir (str): Root directory where artefacts are saved.
            random_seed (int): Random seed for reproducibility.
            save_artifacts (bool): Whether to persist diagnostics and models.
        """
        self.name          = name
        self.results_dir   = Path(results_dir) / name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed   = random_seed
        self.is_trained    = False
        self.save_artifacts = save_artifacts
        self._diagnostics:   Dict[str, Any]  = {}
        self._train_history: List[Dict]       = []

    # ── Public interface ──────────────────────────────────────────────────────

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "BaseAgent":
        """Trains the agent on data from a single walk-forward fold.

        Args:
            X (pd.DataFrame): Feature matrix for the training fold.
            y (pd.Series): Binary target labels (1 = Outperform).
            **kwargs: Additional arguments (e.g. fold index, sector column).

        Returns:
            BaseAgent: The trained agent instance (self).
        """
        ...

    @abstractmethod
    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        """Returns bullish scores in [0, 1] for each observation.

        Args:
            X (pd.DataFrame): Feature matrix for the prediction set.

        Returns:
            pd.Series: Scores indexed like X. Score of 1 = Outperform.
        """
        ...

    # ── Diagnostics persistence ───────────────────────────────────────────────

    def save_diagnostics(self, fold: Optional[int | str] = None, extra: Optional[Dict] = None):
        """Serialises agent diagnostics to a JSON file.

        Args:
            fold (Optional[int | str]): Fold identifier appended to the filename.
            extra (Optional[Dict]): Additional data to merge into the output.
        """
        if not self.save_artifacts:
            return
        data = {
            "agent":     self.name,
            "timestamp": datetime.now().isoformat(),
            "fold":      fold,
            **self._diagnostics,
            **(extra or {}),
        }
        suffix = f"_{fold}" if fold is not None else ""
        path   = self.results_dir / f"diagnostics{suffix}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info(f"[{self.name}] Diagnostics → {path.name}")

    def save_feature_importances(self, importances: pd.Series, fold: Optional[int | str] = None):
        """Saves feature importances sorted in descending order to CSV.

        Args:
            importances (pd.Series): Feature importance values indexed by
                feature name.
            fold (Optional[int | str]): Fold identifier appended to the filename.
        """
        if not self.save_artifacts:
            return
        suffix = f"_{fold}" if fold is not None else ""
        path   = self.results_dir / f"feature_importances{suffix}.csv"
        importances.sort_values(ascending=False).to_csv(path, header=["importance"])
        log.info(f"[{self.name}] Top-5 features: "
                 + " | ".join(f"{k}={v:.3f}" for k, v in importances.nlargest(5).items()))

    def save_predictions(self, preds_df: pd.DataFrame, fold: Optional[int | str] = None):
        """Saves a predictions DataFrame to CSV.

        Args:
            preds_df (pd.DataFrame): DataFrame of predictions to persist.
            fold (Optional[int | str]): Fold identifier appended to the filename.
        """
        if not self.save_artifacts:
            return
        suffix = f"_{fold}" if fold is not None else ""
        path   = self.results_dir / f"predictions{suffix}.csv"
        preds_df.to_csv(path)
        log.info(f"[{self.name}] Predictions ({len(preds_df)} obs) → {path.name}")

    def record_train_metrics(self, metrics: Dict[str, float], fold: Optional[int] = None):
        """Records training metrics to the in-memory history and persists to disk.

        Args:
            metrics (Dict[str, float]): Metric name-value pairs from training.
            fold (Optional[int]): Fold index for record-keeping.
        """
        self._diagnostics["last_train_metrics"] = metrics
        if not self.save_artifacts:
            return
        entry = {"fold": fold, "ts": datetime.now().isoformat(), **metrics}
        self._train_history.append(entry)
        path = self.results_dir / "train_history.json"
        with open(path, "w") as f:
            json.dump(self._train_history, f, indent=2, default=str)

    # ── Preprocessing helpers ─────────────────────────────────────────────────

    @staticmethod
    def _is_ratio_or_normalized_feature(col_name: str, series: Optional[pd.Series] = None) -> bool:
        """Applies the strict feature policy check.

        Args:
            col_name (str): Column name to check.
            series (Optional[pd.Series]): Optional value series for binary-flag detection.

        Returns:
            bool: True if the feature passes the ratio/normalized policy.
        """
        return is_ratio_or_normalized_feature(col_name, series)

    @staticmethod
    def _filter_ratio_normalized_columns(X: pd.DataFrame) -> pd.DataFrame:
        """Removes non-ratio/normalized magnitude features under strict policy.

        Args:
            X (pd.DataFrame): Input feature matrix.

        Returns:
            pd.DataFrame: Copy with only policy-compliant columns retained.
        """
        if X is None or X.empty:
            return X
        keep_cols = [
            col for col in X.columns
            if BaseAgent._is_ratio_or_normalized_feature(col, X[col])
        ]
        # If no columns survive, return an empty frame so the caller fails fast.
        return X[keep_cols].copy()

    @staticmethod
    def _clean_numeric(X: pd.DataFrame) -> pd.DataFrame:
        """Applies shared numeric cleaning used by both training and inference.

        Steps:
          1. Replace inf/-inf with NaN.
          2. Drop fully-empty columns.
          3. Impute NaN with column median; residual NaN → 0.
          4. Absolute clip at ±1e15 (prevents overflow when computing std).
          5. Clip ±10σ per column (removes extreme outliers).
          6. Final guarantee: no inf or NaN (required by XGBoost/sklearn).

        Args:
            X (pd.DataFrame): Feature matrix to clean.

        Returns:
            pd.DataFrame: Cleaned feature matrix with the same columns.
        """
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna(axis=1, how="all")
        medians = X.median(numeric_only=True)
        X = X.fillna(medians).fillna(0)
        X = X.clip(-1e15, 1e15)
        means = X.mean(numeric_only=True)
        stds  = X.std(numeric_only=True).replace(0, 1)
        lower = (means - 10 * stds).reindex(X.columns)
        upper = (means + 10 * stds).reindex(X.columns)
        X = X.clip(lower=lower, upper=upper, axis=1)
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        return X

    @staticmethod
    def clean_features(
        X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> tuple:
        """Cleans a feature matrix for training, aligning it with labels.

        Unlike :meth:`clean_features_predict`, this method also removes rows
        with more than 50% missing values (preserves training data quality).

        Args:
            X (pd.DataFrame): Feature matrix for training.
            y (Optional[pd.Series]): Target labels to align with X after
                row filtering.

        Returns:
            tuple: ``(X_clean, y_aligned)`` if y is provided, else
                ``(X_clean, None)``.

        Raises:
            ValueError: If no feature columns survive the strict
                ratio/normalized policy filter.
        """
        X = X.replace([np.inf, -np.inf], np.nan)
        # Remove rows with more than 50% NaN (training only; not in predict)
        X = X.dropna(thresh=max(1, int(len(X.columns) * 0.5)), axis=0)

        if y is not None:
            common = X.index.intersection(y.index)
            if len(common) == 0:
                # Positional fallback when indices share no elements
                min_len = min(len(X), len(y))
                X = X.iloc[:min_len].reset_index(drop=True)
                y = y.iloc[:min_len].reset_index(drop=True)
            else:
                y = y.loc[common].dropna()
                X = X.loc[y.index]

        X = BaseAgent._filter_ratio_normalized_columns(X)
        if X.shape[1] == 0:
            raise ValueError(
                "No features remained after applying the strict ratio/normalized filter. "
                "Update agent features to use ratios, z-scores, or binary flags."
            )
        X = BaseAgent._clean_numeric(X)
        return (X, y) if y is not None else (X, None)

    @staticmethod
    def clean_features_predict(X: pd.DataFrame) -> pd.DataFrame:
        """Cleans a feature matrix for inference without dropping rows.

        Guarantees that :meth:`predict_score` returns exactly ``len(X)`` rows.

        Args:
            X (pd.DataFrame): Feature matrix for test or live prediction.

        Returns:
            pd.DataFrame: Cleaned feature matrix with the same row count.
        """
        X = BaseAgent._filter_ratio_normalized_columns(X)
        return BaseAgent._clean_numeric(X)

    @staticmethod
    def class_balance(y: pd.Series) -> Dict[str, float]:
        """Computes class balance statistics for the target label.

        Args:
            y (pd.Series): Binary target series (0 or 1).

        Returns:
            Dict[str, float]: Dictionary with keys n_samples, n_positive,
                n_negative, and positive_ratio.
        """
        counts = y.value_counts()
        total  = len(y)
        return {
            "n_samples":      total,
            "n_positive":     int(counts.get(1, 0)),
            "n_negative":     int(counts.get(0, 0)),
            "positive_ratio": float(counts.get(1, 0) / total) if total > 0 else 0.0,
        }

    # ── Feature structural helpers ────────────────────────────────────────────

    @staticmethod
    def _unique_existing_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
        """Returns a deduplicated, order-preserving list of columns that exist in df.

        Args:
            df (pd.DataFrame): DataFrame whose columns are the reference set.
            columns (List[str]): Candidate column names.

        Returns:
            List[str]: Subset of columns that are present in df, with duplicates
                removed and insertion order preserved.
        """
        return [c for c in dict.fromkeys(columns) if c in df.columns]

    def _prepare_base_features(self, X: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Selects base feature columns for the agent, preserving order and avoiding duplicates.

        Args:
            X (pd.DataFrame): Full feature matrix.
            feature_cols (List[str]): Desired feature column names.

        Returns:
            pd.DataFrame: Subset of X with only the available feature columns.
        """
        df = X.copy()
        selected = self._unique_existing_columns(df, feature_cols)
        return df[selected].copy()

    def _prepare_with_sector_dummies(
        self,
        X: pd.DataFrame,
        feature_cols: List[str],
        sector_col: str,
        fit_mode: bool,
        include_zsector: bool = True,
        dummies_attr: str = "_sector_dummies",
    ) -> pd.DataFrame:
        """Builds a feature matrix with base features, optional z-sector columns,
        and one-hot sector dummies.

        During ``fit_mode=True``, the set of dummy columns is saved to
        ``self.<dummies_attr>`` so that the same schema can be reproduced at
        inference time.

        Args:
            X (pd.DataFrame): Full feature matrix.
            feature_cols (List[str]): Base feature column names.
            sector_col (str): Column containing the sector label.
            fit_mode (bool): If True, fits the dummy column schema from data.
                If False, aligns to the previously fitted schema.
            include_zsector (bool): Whether to append ``*_zsector`` columns.
            dummies_attr (str): Attribute name used to store/retrieve the dummy
                column list.

        Returns:
            pd.DataFrame: Feature matrix including base, zsector, and dummy columns.
        """
        df = X.copy()
        selected = self._unique_existing_columns(df, feature_cols)

        if include_zsector:
            zsec_cols = [c for c in df.columns if c.endswith("_zsector")]
            selected = self._unique_existing_columns(df, selected + zsec_cols)

        if sector_col in df.columns:
            dummies = pd.get_dummies(df[sector_col], prefix="sector", dtype=float)
            dummies.index = df.index
            if fit_mode:
                setattr(self, dummies_attr, list(dummies.columns))
            else:
                stored = getattr(self, dummies_attr, [])
                for col in stored:
                    if col not in dummies.columns:
                        dummies[col] = 0.0
                dummies = dummies[[c for c in stored if c in dummies.columns]]
            df = pd.concat([df, dummies], axis=1)
            selected = self._unique_existing_columns(df, selected + list(dummies.columns))

        return df[selected].copy()

    def _align_to_feature_cols(self, X: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
        """Aligns a feature matrix to the schema stored in ``self._feature_cols``.

        Adds missing columns filled with ``fill_value`` and reorders columns to
        match the training schema.

        Args:
            X (pd.DataFrame): Feature matrix to align.
            fill_value (float): Value used to fill missing columns.

        Returns:
            pd.DataFrame: Aligned feature matrix with exactly the columns in
                ``self._feature_cols``.
        """
        if not hasattr(self, "_feature_cols"):
            return X
        for col in self._feature_cols:
            if col not in X.columns:
                X[col] = fill_value
        return X[self._feature_cols]


class FeatureSelector:
    """Two-step feature selector that must be fitted only on training data.

    Step 0 — Mandatory base-vs-zsector exclusion:
        For each pair (base_col, base_col_zsector), retains only the version
        with the higher point-biserial correlation with the target y. In a
        tie, the _zsector version is preferred for sectoral comparability.

    Step 1 — Greedy redundancy removal by correlation:
        Features sorted by mean correlation with all others (most redundant
        first) are evaluated against already-kept features. When
        ``|corr| > corr_threshold``, the feature with lower point-biserial
        correlation with y is discarded.

    Step 2 — Combined score ranking (relevance with y + RF importance):
        For each remaining feature a combined score is computed:
            combined = w_relevance * norm(|pb_y|) + (1-w_relevance) * norm(RF_importance)
        where pb_y is the point-biserial correlation with y and RF_importance
        is the Gini importance from an auxiliary Random Forest.

    Step 3 — Importance-cutoff rule + [min_keep, max_keep] bounds:
        Features with ``RF_importance >= top_importance * cutoff_fraction``
        are retained, then clamped to the [min_keep, max_keep] range.
        The selected features are rescaled by normalized importance weights.

    Example:
        >>> selector = FeatureSelector()
        >>> X_train_sel = selector.fit_transform(X_train, y_train)
        >>> X_test_sel  = selector.transform(X_test)
    """

    def __init__(self, corr_threshold: float = 0.95, top_n: int = 10,
                 min_features: int = 5, random_seed: int = 42,
                 relevance_weight: float = FEATURE_SELECTOR_RELEVANCE_WEIGHT,
                 rf_n_estimators: int = FEATURE_SELECTOR_RF_N_ESTIMATORS,
                 rf_max_depth: int = FEATURE_SELECTOR_RF_MAX_DEPTH,
                 zsector_pair_policy: str = "auto"):
        """Initialises the FeatureSelector.

        Args:
            corr_threshold (float): Absolute correlation threshold above which
                two features are considered redundant.
            top_n (int): Legacy parameter; effectively replaced by the
                cutoff-fraction rule in Step 3.
            min_features (int): Minimum number of features to keep after
                correlation filtering.
            random_seed (int): Random seed for the auxiliary Random Forest.
            relevance_weight (float): Weight of point-biserial relevance in the
                combined score (range [0, 1]).
            rf_n_estimators (int): Number of trees in the auxiliary RF.
            rf_max_depth (int): Maximum tree depth in the auxiliary RF.
            zsector_pair_policy (str): Policy for resolving base/zsector pairs.
                ``"auto"`` uses pb_y to decide; ``"force_zsector"`` always
                keeps the zsector version.
        """
        self.corr_threshold  = corr_threshold
        self.top_n           = top_n
        self.min_features    = min_features
        self.random_seed     = random_seed
        self.relevance_weight = float(max(0.0, min(1.0, relevance_weight)))
        self.rf_n_estimators = max(int(rf_n_estimators), 20)
        self.rf_max_depth = max(int(rf_max_depth), 2)
        self.cutoff_fraction = float(max(0.0, min(1.0, FEATURE_IMPORTANCE_CUTOFF_FRACTION)))
        self.min_keep = max(int(FEATURE_IMPORTANCE_MIN_KEEP), 1)
        self.max_keep = max(int(FEATURE_IMPORTANCE_MAX_KEEP), self.min_keep)
        policy = str(zsector_pair_policy).strip().lower()
        self.zsector_pair_policy = policy if policy in {"auto", "force_zsector"} else "auto"
        self._selected_cols: List[str] = []
        self._selected_weights_pct: Dict[str, float] = {}
        self._dropped_pair:  List[str] = []
        self._dropped_corr:  List[str] = []
        self._dropped_imp:   List[str] = []

    # ── fit_transform ──────────────────────────────────────────────────────────

    def fit_transform(self, X: pd.DataFrame, y: pd.Series,
                      agent_name: str = "") -> pd.DataFrame:
        """Fits the selector and returns X with the selected features.

        Args:
            X (pd.DataFrame): Feature matrix (already cleaned).
            y (pd.Series): Binary target aligned with X.
            agent_name (str): Agent name for log messages (optional).

        Returns:
            pd.DataFrame: X filtered to selected features (rescaled by
                importance weights).
        """
        cols    = list(X.columns)
        prefix  = f"[FeatureSelector/{agent_name}]" if agent_name else "[FeatureSelector]"

        # ── Step 1: point-biserial correlation with y ─────────────────────────
        # Used to decide which of a highly-correlated pair to discard.
        try:
            from scipy.stats import pointbiserialr
            pb_corr: Dict[str, float] = {}
            for c in cols:
                try:
                    r, _ = pointbiserialr(y, X[c])
                    pb_corr[c] = abs(float(r))
                except Exception:
                    pb_corr[c] = 0.0
        except ImportError:
            # Fallback: Pearson correlation with y
            pb_corr = {c: abs(float(X[c].corr(y.astype(float)))) for c in cols}

        # ── Step 0: mandatory base vs _zsector exclusion ─────────────────────
        # Hard rule: never allow both the raw and the sector-normalized version
        # of the same indicator simultaneously.
        # For each (base, base_zsector) pair, keep the one with higher |pb_y|.
        # In a tie, prefer _zsector for sectoral comparability.
        to_drop_pair: List[str] = []
        pair_drop_detail: List[str] = []
        for col in cols:
            if not col.endswith("_zsector"):
                continue
            base_col = col[:-8]
            if base_col not in cols:
                continue

            pb_base = float(pb_corr.get(base_col, 0.0))
            pb_zsec = float(pb_corr.get(col, 0.0))
            if self.zsector_pair_policy == "force_zsector":
                loser = base_col
                winner = col
                decision_txt = "policy=force_zsector"
            elif pb_zsec >= pb_base:
                loser = base_col
                winner = col
                decision_txt = "policy=auto"
            else:
                loser = col
                winner = base_col
                decision_txt = "policy=auto"

            if loser not in to_drop_pair:
                to_drop_pair.append(loser)
                pair_drop_detail.append(
                    f"{loser} (exclusive pair with {winner}; "
                    f"pb_y({loser})={pb_corr.get(loser, 0.0):.3f} <= "
                    f"pb_y({winner})={pb_corr.get(winner, 0.0):.3f}; {decision_txt})"
                )

        cols_after_pair = [c for c in cols if c not in to_drop_pair]
        self._dropped_pair = to_drop_pair

        if to_drop_pair:
            log.info(f"{prefix} Step 0 — base vs _zsector exclusion: "
                     f"removed {len(to_drop_pair)} / {len(cols)} features "
                     f"(policy={self.zsector_pair_policy})")
            for detail in pair_drop_detail:
                log.info(f"{prefix}   REMOVED by exclusive pair: {detail}")

        # ── Step 1: greedy removal by inter-feature correlation ───────────────
        # Algorithm:
        #   1. Compute mean correlation of each feature with all others
        #      (most "redundant" features have higher mean correlation).
        #   2. Sort from HIGHEST to LOWEST mean correlation (most redundant first).
        #   3. For each feature not yet discarded, check whether |corr| > threshold
        #      with ANY already-kept feature.
        #      If yes: compare pb_y of both → discard the one with lower pb_y.
        #   This ensures all pairs are evaluated correctly.
        corr_matrix = X[cols_after_pair].corr().abs()

        # Sort from most to least correlated on average (most redundant first)
        mean_corr = corr_matrix.mean()
        cols_sorted = list(mean_corr.sort_values(ascending=False).index)

        to_drop_corr: List[str]  = []
        kept_so_far:  List[str]  = []
        corr_drop_detail: List[str] = []

        for feat in cols_sorted:
            if feat in to_drop_corr:
                continue
            # Check whether this feature is highly correlated with any kept feature
            rival = None
            rival_corr = 0.0
            for k in kept_so_far:
                c_val = float(corr_matrix.at[feat, k])
                if c_val > self.corr_threshold and c_val > rival_corr:
                    rival      = k
                    rival_corr = c_val
            if rival is not None:
                # Discard the one less aligned with y
                if pb_corr.get(feat, 0) >= pb_corr.get(rival, 0):
                    # 'feat' is better → discard rival, keep feat
                    to_drop_corr.append(rival)
                    kept_so_far.remove(rival)
                    kept_so_far.append(feat)
                    corr_drop_detail.append(
                        f"{rival} (|corr|={rival_corr:.3f} with {feat}; "
                        f"pb_y({rival})={pb_corr.get(rival, 0):.3f} < "
                        f"pb_y({feat})={pb_corr.get(feat, 0):.3f})"
                    )
                else:
                    # rival is better → discard feat
                    to_drop_corr.append(feat)
                    corr_drop_detail.append(
                        f"{feat} (|corr|={rival_corr:.3f} with {rival}; "
                        f"pb_y({feat})={pb_corr.get(feat, 0):.3f} < "
                        f"pb_y({rival})={pb_corr.get(rival, 0):.3f})"
                    )
            else:
                kept_so_far.append(feat)

        remaining = [c for c in cols_after_pair if c not in to_drop_corr]
        if len(remaining) < self.min_features:
            # Revert: not enough features remain after filtering
            remaining        = cols_after_pair
            to_drop_corr     = []
            corr_drop_detail = []
        self._dropped_corr = to_drop_corr

        log.info(f"{prefix} Step 1 — correlation >{self.corr_threshold:.0%}: "
                 f"removed {len(to_drop_corr)} / {len(cols)} features")
        for detail in corr_drop_detail:
            log.info(f"{prefix}   REMOVED by corr: {detail}")

        # ── Step 2: combined score (relevance with y + RF importance) ─────────
        n_remaining = len(remaining)
        imp_all = pd.Series(1.0 / n_remaining if n_remaining else 0.0, index=remaining)
        pb_abs_all = pd.Series({c: float(pb_corr.get(c, 0.0)) for c in remaining})

        def _minmax_norm(s: pd.Series) -> pd.Series:
            if s is None or s.empty:
                return pd.Series(dtype=float)
            s = s.astype(float)
            s_min = float(s.min())
            s_max = float(s.max())
            if s_max - s_min <= 1e-12:
                return pd.Series(1.0, index=s.index)
            return (s - s_min) / (s_max - s_min)

        try:
            from sklearn.ensemble import RandomForestClassifier as _RFC
            rf = _RFC(n_estimators=self.rf_n_estimators, max_depth=self.rf_max_depth, n_jobs=-1,
                      random_state=self.random_seed, class_weight="balanced")
            rf.fit(X[remaining], y)
            imp_all = pd.Series(rf.feature_importances_, index=remaining).sort_values(ascending=False)
        except Exception:
            log.warning("[FeatureSelector] RF importance fallback failed", exc_info=True)

        pb_norm = _minmax_norm(pb_abs_all)
        imp_norm = _minmax_norm(imp_all.reindex(remaining).fillna(0.0))
        combined = (
            self.relevance_weight * pb_norm.reindex(remaining).fillna(0.0)
            + (1.0 - self.relevance_weight) * imp_norm.reindex(remaining).fillna(0.0)
        ).sort_values(ascending=False)

        # ── Step 3: importance-cutoff rule + [min_keep, max_keep] bounds ──────
        imp_rank = imp_all.reindex(remaining).fillna(0.0).sort_values(ascending=False)
        top_importance = float(imp_rank.iloc[0]) if len(imp_rank) else 0.0
        cutoff_value = top_importance * self.cutoff_fraction

        if len(imp_rank):
            selected_by_cut = imp_rank[imp_rank >= cutoff_value].index.tolist()
        else:
            selected_by_cut = []

        if len(selected_by_cut) < self.min_keep:
            selected_by_cut = imp_rank.head(self.min_keep).index.tolist()

        if len(selected_by_cut) > self.max_keep:
            selected_by_cut = imp_rank.head(self.max_keep).index.tolist()

        self._selected_cols = list(selected_by_cut)
        self._dropped_imp = [c for c in remaining if c not in self._selected_cols]

        # Normalise importance weights to sum to 100 for the selected features.
        imp_selected = imp_rank.reindex(self._selected_cols).fillna(0.0)
        imp_sum = float(imp_selected.sum())
        if imp_sum > 0:
            weights_pct = (imp_selected / imp_sum * 100.0)
        else:
            eq = 100.0 / max(len(self._selected_cols), 1)
            weights_pct = pd.Series(eq, index=self._selected_cols)
        self._selected_weights_pct = {c: float(weights_pct[c]) for c in self._selected_cols}

        # ── Detailed logs ──────────────────────────────────────────────────────
        log.info(
            f"{prefix} Step 2 — combined score "
            f"(w_relevance={self.relevance_weight:.2f}, w_importance={1.0 - self.relevance_weight:.2f})"
        )
        log.info(
            f"{prefix} Step 3 — importance cutoff: top={top_importance:.4f}, "
            f"cutoff={self.cutoff_fraction:.2f} (value={cutoff_value:.4f}), "
            f"min_keep={self.min_keep}, max_keep={self.max_keep}"
        )
        log.info(f"{prefix}   {'Feature':<35} {'Combined':>10} {'Imp_RF':>10} {'pb_y':>8}")
        log.info(f"{prefix}   {'-'*35} {'-'*10} {'-'*10} {'-'*8}")
        for rank, feat in enumerate(self._selected_cols, 1):
            imp_abs = float(imp_all[feat]) if feat in imp_all.index else 0.0
            pb_val  = pb_corr.get(feat, 0.0)
            cmb_val = float(combined.get(feat, 0.0))
            w_pct = self._selected_weights_pct.get(feat, 0.0)
            log.info(
                f"{prefix}   {rank:2}. {feat:<33} {cmb_val:>10.4f} {imp_abs:>10.4f} {pb_val:>8.3f}"
                f"  weight={w_pct:>6.2f}%"
            )

        if self._dropped_imp:
            log.info(f"{prefix}   Discarded by low combined score ({len(self._dropped_imp)}): "
                     + ", ".join(self._dropped_imp))

        log.info(
            f"{prefix} Summary: {len(cols)} features → "
            f"pair_drop={len(self._dropped_pair)} → "
            f"corr_drop={len(self._dropped_corr)} → "
            f"imp_drop={len(self._dropped_imp)} → "
            f"{len(self._selected_cols)} selected (cutoff rule + [{self.min_keep},{self.max_keep}])"
        )

        return self.transform(X)

    # ── transform ─────────────────────────────────────────────────────────────

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies the fitted feature selection and importance-weight rescaling.

        Args:
            X (pd.DataFrame): Feature matrix to transform.

        Returns:
            pd.DataFrame: DataFrame restricted to the selected columns and
                rescaled by their normalized importance weights.
        """
        result = X.reindex(columns=self._selected_cols, fill_value=0.0).copy()
        # Rescale by normalized weights: sum of weights = 100.
        for col, weight_pct in self._selected_weights_pct.items():
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0) * (weight_pct / 100.0)
        return result.copy()

    def report(self) -> Dict:
        """Returns a summary dictionary of the selector's configuration and results.

        Returns:
            Dict: Configuration parameters and lists of selected / dropped features.
        """
        return {
            "corr_threshold": self.corr_threshold,
            "top_n": self.top_n,
            "min_features": self.min_features,
            "relevance_weight": self.relevance_weight,
            "importance_weight": 1.0 - self.relevance_weight,
            "rf_n_estimators": self.rf_n_estimators,
            "rf_max_depth": self.rf_max_depth,
            "zsector_pair_policy": self.zsector_pair_policy,
            "cutoff_fraction": self.cutoff_fraction,
            "min_keep": self.min_keep,
            "max_keep": self.max_keep,
            "n_selected":   len(self._selected_cols),
            "selected":     self._selected_cols,
            "selected_weights_pct": self._selected_weights_pct,
            "dropped_pair": self._dropped_pair,
            "dropped_corr": self._dropped_corr,
            "dropped_imp":  self._dropped_imp,
        }
