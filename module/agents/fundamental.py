"""Fundamental agent (XGBoost) for the multi-agent stock picker.

Evaluates a company's financial health.

Consumes (from consolidated + feature_engineering):
  Profitability: roe, roa, roi, roic, net_margin, gross_margin,
                 fcf_margin, ebitda_margin, operating_margin
  Liquidity:     current_ratio, quick_ratio
  Solvency:      debt_equity, debt_to_ebitda, interest_coverage
  Growth:        revenue_yoy_growth, net_income_yoy_growth, eps_yoy_growth,
                 fcf_yoy_growth, operating_income_yoy_growth
  Quality:       accruals_ratio, capex_to_revenue, consecutive_losses

  Sector dummy:  sector_*   (one-hot from companies.csv)
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from module.agents.base import BaseAgent, FeatureSelector
from module.common.feature_controls import resolve_feature_columns
from module.steps.step_04_evaluation.explainability import build_explainer_for_agent, AgentExplainer
from environment import (
    FUNDAMENTAL_N_ESTIMATORS, FUNDAMENTAL_MAX_DEPTH, FUNDAMENTAL_LEARNING_RATE,
    FUNDAMENTAL_SUBSAMPLE, FUNDAMENTAL_COLSAMPLE, FUNDAMENTAL_MIN_CHILD_WEIGHT,
    FEATURE_CORR_THRESHOLD, FEATURE_TOP_N,
    FUNDAMENTAL_FEATURE_COLUMNS, FUNDAMENTAL_FEATURE_EXCLUDE,
    DEGENERATE_MODEL_FALLBACK_SCORE, DEGENERATE_MODEL_IMPORTANCE_EPS,
)

log = logging.getLogger(__name__)

try:
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    from sklearn.model_selection import TimeSeriesSplit
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False
    log.error("[FundamentalAgent] Install: pip install xgboost scikit-learn")

class FundamentalAgent(BaseAgent):
    """XGBoost model that learns which combination of financial ratios best
    predicts 1-year Outperformance.

    The sector from companies.csv is fed in two complementary ways:
      1. One-hot dummies (sector_Technology, sector_Healthcare, …)

    """

    def __init__(self, results_dir: str, random_seed: int = 42,
                 n_estimators: int = FUNDAMENTAL_N_ESTIMATORS,
                 max_depth: int = FUNDAMENTAL_MAX_DEPTH,
                 learning_rate: float = FUNDAMENTAL_LEARNING_RATE,
                 subsample: float = FUNDAMENTAL_SUBSAMPLE,
                 colsample_bytree: float = FUNDAMENTAL_COLSAMPLE,
                 min_child_weight: int = FUNDAMENTAL_MIN_CHILD_WEIGHT,
                 save_artifacts: bool = True):
        """Initialises the FundamentalAgent.

        Args:
            results_dir (str): Directory where training artefacts are saved.
            random_seed (int): Random seed for reproducibility.
            n_estimators (int): Number of boosting rounds.
            max_depth (int): Maximum tree depth.
            learning_rate (float): Step-shrinkage (eta) parameter.
            subsample (float): Row sub-sampling ratio per tree.
            colsample_bytree (float): Feature sub-sampling ratio per tree.
            min_child_weight (int): Minimum sum of instance weight per leaf.
            save_artifacts (bool): Whether to save diagnostics and models.

        Raises:
            ImportError: If xgboost or scikit-learn is not installed.
        """
        super().__init__("fundamental", results_dir, random_seed, save_artifacts)
        if not _DEPS_OK:
            raise ImportError("xgboost and scikit-learn are required.")
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.learning_rate    = learning_rate
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self._model = None
        self._feature_cols: List[str] = []
        self._selector:     Optional[FeatureSelector] = None
        self._explainer:    Optional[AgentExplainer] = None
        self._is_degenerate_model: bool = False
        self._degenerate_reason: str = ""

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series,
            fold: Optional[int] = None, sector_col: str = "sector",
            sample_weight: Optional[np.ndarray] = None) -> "FundamentalAgent":
        """Trains XGBoost on financial ratio features.

        Args:
            X (pd.DataFrame): Feature matrix for the training fold.
            y (pd.Series): Binary target labels (1 = Outperform).
            fold (Optional[int]): Walk-forward fold index for artefact naming.
            sector_col (str): Column name for sector labels (used for dummies).
            sample_weight (Optional[np.ndarray]): Per-sample weights (e.g.,
                exponential recency weights).  If None, uniform weights are used.

        Returns:
            FundamentalAgent: The fitted agent instance (self).
        """
        log.info(f"[FundamentalAgent] Training XGBoost — {len(X)} obs, {len(X.columns)} features")
        # Align X and y by position before any processing
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len].copy()
        y = y.iloc[:min_len].copy()
        X_prep       = self._prepare(X, sector_col, fit_mode=True)
        X_prep, y_cl = self.clean_features(X_prep, y.reset_index(drop=True))
        X_prep       = X_prep.reset_index(drop=True)
        y_cl         = y_cl.reset_index(drop=True)

        # Align sample weights to the cleaned index if provided.
        # Track which original positions survive clean_features before reset_index.
        sw: Optional[np.ndarray] = None
        if sample_weight is not None:
            sw_full = np.asarray(sample_weight, dtype=float)[:min_len]
            # y_cl was built from y.reset_index(drop=True) and then passed through
            # clean_features which may drop rows.  The survived positions are
            # recorded in y_cl.index (integer positions in the pre-reset array).
            survived_positions = y_cl.index.values  # integer positions before reset
            if survived_positions.max() < len(sw_full):
                sw = sw_full[survived_positions]
            elif len(y_cl) == len(sw_full):
                # No rows dropped; use weights directly
                sw = sw_full

        if not self.has_multiple_classes(y_cl):
            log.warning("[FundamentalAgent] Label without enough class variance after cleaning - training skipped.")
            return self

        # Feature selection: only on training data (no leakage)
        self._selector = FeatureSelector(corr_threshold=FEATURE_CORR_THRESHOLD, top_n=FEATURE_TOP_N,
                                         min_features=3, random_seed=self.random_seed,
                                         )
        X_prep = self._selector.fit_transform(X_prep, y_cl, agent_name="fundamental")

        self._feature_cols = list(X_prep.columns)
        bal          = self.class_balance(y_cl)
        spw          = bal["n_negative"] / bal["n_positive"] if bal["n_positive"] > 0 else 1.0

        base = xgb.XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree, min_child_weight=self.min_child_weight,
            scale_pos_weight=spw, eval_metric="auc",
            random_state=self.random_seed, n_jobs=-1, tree_method="hist",
        )
        self._model = base

        cv = self._cv(X_prep, y_cl, spw)
        log.info(f"[FundamentalAgent] CV AUC={cv['mean_auc']:.4f} ± {cv['std_auc']:.4f}  ({len(self._feature_cols)} selected features)")

        self._model.fit(X_prep, y_cl, sample_weight=sw)
        self.is_trained = True

        # Feature importances from the XGBoost estimator
        imp = pd.Series(self._model.feature_importances_, index=self._feature_cols)
        imp_abs_sum = float(np.abs(imp).sum())
        self._is_degenerate_model = bool(imp_abs_sum <= float(DEGENERATE_MODEL_IMPORTANCE_EPS))
        self._degenerate_reason = "all_feature_importances_zero" if self._is_degenerate_model else ""
        if self._is_degenerate_model:
            log.warning(
                "[FundamentalAgent] Degenerate model detected (importance_sum=%.3e). "
                "Predictions will use conservative fallback score %.2f.",
                imp_abs_sum,
                float(DEGENERATE_MODEL_FALLBACK_SCORE),
            )
        self.save_feature_importances(imp, fold)

        self._diagnostics = {
            "class_balance": bal, "cv_metrics": cv,
            "n_features": len(self._feature_cols),
            "top_features": imp.nlargest(15).to_dict(),
            "feature_selection": self._selector.report(),
            "model_quality": {
                "degenerate": bool(self._is_degenerate_model),
                "degenerate_reason": self._degenerate_reason,
                "importance_abs_sum": imp_abs_sum,
                "fallback_score": float(DEGENERATE_MODEL_FALLBACK_SCORE),
            },
        }
        self.record_train_metrics(cv, fold)
        self.save_diagnostics(fold)
        if self.save_artifacts:
            self._explainer = build_explainer_for_agent(
                self.name, self._model, self._feature_cols,
                X_prep, self.results_dir.as_posix(), fold, model_type="tree"
            )
        else:
            self._explainer = None
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_score(self, X: pd.DataFrame, sector_col: str = "sector") -> pd.Series:
        """Returns fundamental quality scores in [0, 1].

        Args:
            X (pd.DataFrame): Feature matrix.
            sector_col (str): Column name for sector labels.

        Returns:
            pd.Series: Probability of Outperform indexed like X.

        Raises:
            RuntimeError: If the agent has not been trained yet.
        """
        if not self.is_trained:
            raise RuntimeError("[FundamentalAgent] Not trained.")
        if self._is_degenerate_model:
            return pd.Series(
                float(DEGENERATE_MODEL_FALLBACK_SCORE),
                index=X.index,
                name="fundamental_score",
            )
        X_prep = self.clean_features_predict(self._prepare(X, sector_col, fit_mode=False))
        if self._selector is not None:
            X_prep = self._selector.transform(X_prep)
        X_al = self._align(X_prep)
        proba = self._model.predict_proba(X_al)[:, 1]
        proba = self._apply_calibration(proba)
        return pd.Series(proba, index=X.index, name="fundamental_score")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _prepare(self, X: pd.DataFrame, sector_col: str, fit_mode: bool) -> pd.DataFrame:
        """Selects base features and computes derived financial health ratios.

        Sector one-hot dummies are intentionally excluded: the sector-level
        signal is already captured by the cross-sectional rank features built
        by SectorFeatureBuilder (e.g., pe_rank_sector, roe_pct_sector).

        Args:
            X (pd.DataFrame): Raw feature matrix.
            sector_col (str): Column name for the sector label (not used
                internally; retained for API consistency with other agents).
            fit_mode (bool): If True, stores the column list for alignment.

        Returns:
            pd.DataFrame: Prepared feature matrix.
        """
        selected = resolve_feature_columns(
            default_cols=[],
            available_cols=list(X.columns),
            include_cols=FUNDAMENTAL_FEATURE_COLUMNS,
            exclude_cols=FUNDAMENTAL_FEATURE_EXCLUDE,
            logger=log,
            owner="FundamentalAgent",
        )
        df = X[selected].copy()

        # Derived: profitability quality — combines margins with earnings quality.
        # High margin + high FCF/NI coherence = sustainable profitability.
        if "net_margin" in df.columns and "earnings_quality" in df.columns:
            margin_rank = df["net_margin"].rank(pct=True)
            quality_rank = df["earnings_quality"].rank(pct=True)
            df["profitability_quality"] = (margin_rank + quality_rank) / 2.0
            selected = list(df.columns)

        # Derived: fundamental momentum — are key ratios improving?
        # Positive when ROE, margins, and current ratio are all trending up.
        trend_cols = [c for c in ["roe_trend_2y", "net_margin_trend_2y", "gross_margin_trend_3y"] if c in df.columns]
        if len(trend_cols) >= 2:
            df["fundamental_momentum"] = df[trend_cols].mean(axis=1)
            selected = list(df.columns)

        result = df.copy()
        if fit_mode:
            self._feature_cols = list(result.columns)
        return result

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aligns X to the training feature schema.

        Args:
            X (pd.DataFrame): Feature matrix to align.

        Returns:
            pd.DataFrame: Aligned feature matrix.
        """
        return self._align_to_feature_cols(X, fill_value=0.0)

    def _cv(self, X: pd.DataFrame, y: pd.Series, spw: float) -> Dict:
        """Runs time-series cross-validation for the XGBoost model.

        Args:
            X (pd.DataFrame): Cleaned, selected feature matrix.
            y (pd.Series): Binary target series.
            spw (float): ``scale_pos_weight`` for class imbalance correction.

        Returns:
            Dict: Dictionary with mean_auc, std_auc, mean_acc, and mean_f1.
        """
        date_order = X.index.get_level_values("date") if "date" in X.index.names else X.index
        sort_idx = date_order.argsort()
        X = X.iloc[sort_idx]
        y = y.reindex(X.index)
        if len(X) < 3:
            log.warning("[FundamentalAgent] CV skipped: insufficient samples (%s)", len(X))
            return {"mean_auc": 0.0, "std_auc": 0.0, "mean_acc": 0.0, "mean_f1": 0.0}
        n_splits = min(5, len(X) - 1)
        if n_splits < 2:
            log.warning("[FundamentalAgent] CV skipped: invalid n_splits=%s for n=%s", n_splits, len(X))
            return {"mean_auc": 0.0, "std_auc": 0.0, "mean_acc": 0.0, "mean_f1": 0.0}
        tss = TimeSeriesSplit(n_splits=n_splits)
        aucs, accs, f1s = [], [], []
        for tr, val in tss.split(X):
            y_tr = y.iloc[tr]
            y_val = y.iloc[val]
            if not self.has_multiple_classes(y_tr):
                continue
            clf = xgb.XGBClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate, subsample=self.subsample,
                colsample_bytree=self.colsample_bytree, min_child_weight=self.min_child_weight,
                scale_pos_weight=spw, eval_metric="auc",
                random_state=self.random_seed, n_jobs=-1, tree_method="hist",
            )
            clf.fit(X.iloc[tr], y_tr, eval_set=[(X.iloc[val], y_val)], verbose=False)
            p = clf.predict_proba(X.iloc[val])[:, 1]
            if self.has_multiple_classes(y_val):
                aucs.append(roc_auc_score(y_val, p))
            accs.append(accuracy_score(y_val, (p >= 0.5).astype(int)))
            f1s.append(f1_score(y_val, (p >= 0.5).astype(int), zero_division=0))
        return {
            "mean_auc": float(np.mean(aucs)) if aucs else 0.0,
            "std_auc":  float(np.std(aucs))  if aucs else 0.0,
            "mean_acc": float(np.mean(accs)) if accs else 0.0,
            "mean_f1":  float(np.mean(f1s)) if f1s else 0.0,
        }
