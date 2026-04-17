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
  Sector Z:      *_zsector  (columns added by SectorNormalizer)
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
      2. Sector-normalised ratios (*_zsector)
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

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series,
            fold: Optional[int] = None, sector_col: str = "sector") -> "FundamentalAgent":
        """Trains XGBoost on financial ratio features.

        Args:
            X (pd.DataFrame): Feature matrix for the training fold.
            y (pd.Series): Binary target labels (1 = Outperform).
            fold (Optional[int]): Walk-forward fold index for artefact naming.
            sector_col (str): Column name for sector labels (used for dummies).

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

        # Feature selection: only on training data (no leakage)
        self._selector = FeatureSelector(corr_threshold=FEATURE_CORR_THRESHOLD, top_n=FEATURE_TOP_N,
                                         min_features=3, random_seed=self.random_seed,
                                         zsector_pair_policy="force_zsector")
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

        self._model.fit(X_prep, y_cl)
        self.is_trained = True

        # Feature importances from the XGBoost estimator
        imp = pd.Series(self._model.feature_importances_, index=self._feature_cols)
        self.save_feature_importances(imp, fold)

        self._diagnostics = {
            "class_balance": bal, "cv_metrics": cv,
            "n_features": len(self._feature_cols),
            "top_features": imp.nlargest(15).to_dict(),
            "feature_selection": self._selector.report(),
        }
        self.record_train_metrics(cv, fold)
        self.save_diagnostics(fold)
        self._explainer = build_explainer_for_agent(
            self.name, self._model, self._feature_cols,
            X_prep, self.results_dir.parent.as_posix(), fold, model_type="tree"
        )
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
        X_prep = self.clean_features_predict(self._prepare(X, sector_col, fit_mode=False))
        if self._selector is not None:
            X_prep = self._selector.transform(X_prep)
        X_al = self._align(X_prep)
        proba = self._model.predict_proba(X_al)[:, 1]
        return pd.Series(proba, index=X.index, name="fundamental_score")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _prepare(self, X: pd.DataFrame, sector_col: str, fit_mode: bool) -> pd.DataFrame:
        """Selects base features and sector-normalised columns.

        Only base columns and _zsector variants are included; sector one-hot
        dummies are excluded because they capture sector-level means rather
        than the relative position of the ticker, which is already encoded by
        the _zsector columns.

        Args:
            X (pd.DataFrame): Raw feature matrix.
            sector_col (str): Column name for the sector label (unused here).
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
        zsector_cols = [c for c in X.columns if c.endswith("_zsector")]
        result = X[list(dict.fromkeys(selected + zsector_cols))].copy()
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
        tss = TimeSeriesSplit(n_splits=5)
        aucs, accs, f1s = [], [], []
        for tr, val in tss.split(X):
            clf = xgb.XGBClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate, subsample=self.subsample,
                colsample_bytree=self.colsample_bytree, min_child_weight=self.min_child_weight,
                scale_pos_weight=spw, eval_metric="auc",
                random_state=self.random_seed, n_jobs=-1, tree_method="hist",
            )
            clf.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[val], y.iloc[val])], verbose=False)
            p = clf.predict_proba(X.iloc[val])[:, 1]
            if y.iloc[val].nunique() > 1:
                aucs.append(roc_auc_score(y.iloc[val], p))
            accs.append(accuracy_score(y.iloc[val], (p >= 0.5).astype(int)))
            f1s.append(f1_score(y.iloc[val], (p >= 0.5).astype(int), zero_division=0))
        return {
            "mean_auc": float(np.mean(aucs)) if aucs else 0.0,
            "std_auc":  float(np.std(aucs))  if aucs else 0.0,
            "mean_acc": float(np.mean(accs)),
            "mean_f1":  float(np.mean(f1s)),
        }
