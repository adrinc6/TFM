"""Valuation agent (GBM) for the multi-agent stock picker.

Detects relative undervaluation.

Consumes:
  Own multiples:   pe_ratio, pb_ratio, ps_ratio, ev_to_ebitda,
                   fcf_yield, earnings_yield
  vs. history:     pe_vs_5y_median, pb_vs_5y_median, ev_ebitda_vs_5y_median
  vs. sector:      pe_sector_pct, pb_sector_pct, evebitda_sector_pct
  Analysts:        eps_surprise_pct, eps_revision, eps_est, eps_reported
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from module.agents.base import BaseAgent, FeatureSelector
from module.common.feature_controls import resolve_feature_columns
from module.steps.step_04_evaluation.explainability import build_explainer_for_agent, AgentExplainer
from environment import (
    VALUATION_N_ESTIMATORS, VALUATION_MAX_DEPTH, VALUATION_LEARNING_RATE,
    VALUATION_SUBSAMPLE, FEATURE_CORR_THRESHOLD, FEATURE_TOP_N,
    VALUATION_FEATURE_COLUMNS, VALUATION_FEATURE_EXCLUDE,
)

log = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    from sklearn.model_selection import TimeSeriesSplit
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

class ValuationAgent(BaseAgent):
    """Gradient Boosting model that detects undervaluation by combining:
      1. Multiples vs. own historical median (mean-reversion signal)
      2. Percentile rank of the multiple within the sector (from companies.csv)
      3. Analyst signals (EPS surprises / upward revisions)
    """

    def __init__(self, results_dir: str, random_seed: int = 42,
                 n_estimators: int = VALUATION_N_ESTIMATORS,
                 max_depth: int = VALUATION_MAX_DEPTH,
                 learning_rate: float = VALUATION_LEARNING_RATE,
                 subsample: float = VALUATION_SUBSAMPLE,
                 save_artifacts: bool = True):
        """Initialises the ValuationAgent.

        Args:
            results_dir (str): Directory where training artefacts are saved.
            random_seed (int): Random seed for reproducibility.
            n_estimators (int): Number of boosting rounds.
            max_depth (int): Maximum tree depth.
            learning_rate (float): Step-shrinkage parameter.
            subsample (float): Row sub-sampling ratio per tree.
            save_artifacts (bool): Whether to save diagnostics and models.

        Raises:
            ImportError: If scikit-learn is not installed.
        """
        super().__init__("valuation", results_dir, random_seed, save_artifacts)
        if not _DEPS_OK:
            raise ImportError("scikit-learn is required.")
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.subsample     = subsample
        self._model:              Optional[Pipeline] = None
        self._feature_cols:       List[str]          = []
        self._selector:           Optional[FeatureSelector] = None
        self._explainer:          Optional[AgentExplainer] = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series,
            fold: Optional[int] = None, sector_col: str = "sector") -> "ValuationAgent":
        """Trains the GBM on valuation multiples and analyst signals.

        Args:
            X (pd.DataFrame): Feature matrix for the training fold.
            y (pd.Series): Binary target labels (1 = Outperform).
            fold (Optional[int]): Walk-forward fold index for artefact naming.
            sector_col (str): Column name for sector labels.

        Returns:
            ValuationAgent: The fitted agent instance (self).
        """
        log.info(f"[ValuationAgent] Training GBM — {len(X)} obs, {len(X.columns)} features")
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len].copy()
        y = y.iloc[:min_len].copy()
        X_prep       = self._prepare(X, sector_col, fit_mode=True)
        X_prep, y_cl = self.clean_features(X_prep, y.reset_index(drop=True))
        X_prep       = X_prep.reset_index(drop=True)
        y_cl         = y_cl.reset_index(drop=True)

        if not self.has_multiple_classes(y_cl):
            log.warning("[ValuationAgent] Label without enough class variance after cleaning - training skipped.")
            return self

        # Feature selection: training data only (no leakage)
        self._selector = FeatureSelector(corr_threshold=FEATURE_CORR_THRESHOLD, top_n=FEATURE_TOP_N,
                                         min_features=3, random_seed=self.random_seed,
                                         )
        X_prep = self._selector.fit_transform(X_prep, y_cl, agent_name="valuation")

        self._feature_cols = list(X_prep.columns)
        bal          = self.class_balance(y_cl)

        gbc = GradientBoostingClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            random_state=self.random_seed,
        )
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    gbc),
        ])

        cv = self._cv(X_prep, y_cl)
        log.info(f"[ValuationAgent] CV AUC={cv['mean_auc']:.4f} ± {cv['std_auc']:.4f}  ({len(self._feature_cols)} selected features)")
        self._model.fit(X_prep, y_cl)
        self.is_trained = True

        imp = pd.Series(self._model.named_steps["clf"].feature_importances_, index=self._feature_cols)
        self.save_feature_importances(imp, fold)

        # Save sector multiples distribution for analysis
        sector_summary = self._sector_summary(X, sector_col)
        self._diagnostics = {
            "class_balance": bal, "cv_metrics": cv,
            "sector_multiples_distribution": sector_summary,
            "top_features": imp.nlargest(10).to_dict(),
            "feature_selection": self._selector.report(),
        }
        self.record_train_metrics(cv, fold)
        self.save_diagnostics(fold)
        gbm_model = self._model.named_steps["clf"]
        if self.save_artifacts:
            self._explainer = build_explainer_for_agent(
                self.name, gbm_model, self._feature_cols,
                X_prep, self.results_dir.as_posix(), fold, model_type="tree"
            )
        else:
            self._explainer = None
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_score(self, X: pd.DataFrame, sector_col: str = "sector") -> pd.Series:
        """Returns valuation scores in [0, 1].

        Args:
            X (pd.DataFrame): Feature matrix.
            sector_col (str): Column name for sector labels.

        Returns:
            pd.Series: Probability of Outperform indexed like X.

        Raises:
            RuntimeError: If the agent has not been trained yet.
        """
        if not self.is_trained:
            raise RuntimeError("[ValuationAgent] Not trained.")
        X_prep = self.clean_features_predict(self._prepare(X, sector_col, fit_mode=False))
        if self._selector is not None:
            X_prep = self._selector.transform(X_prep)
        X_al = self._align(X_prep)
        return pd.Series(self._apply_calibration(self._model.predict_proba(X_al)[:, 1]),
                         index=X.index, name="valuation_score")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _prepare(self, X: pd.DataFrame, sector_col: str, fit_mode: bool) -> pd.DataFrame:
        """Selects valuation feature columns for training or inference.

        Args:
            X (pd.DataFrame): Raw feature matrix.
            sector_col (str): Column name for sector labels (informational).
            fit_mode (bool): If True, stores the column list for alignment.

        Returns:
            pd.DataFrame: Prepared feature matrix.
        """
        df       = X.copy()
        selected = resolve_feature_columns(
            default_cols=[],
            available_cols=list(df.columns),
            include_cols=VALUATION_FEATURE_COLUMNS,
            exclude_cols=VALUATION_FEATURE_EXCLUDE,
            logger=log,
            owner="ValuationAgent",
        )

        selected = list(dict.fromkeys(selected))
        result   = df[[c for c in selected if c in df.columns]].copy()
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
        return self._align_to_feature_cols(X, fill_value=np.nan)

    def _cv(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Runs time-series cross-validation and returns aggregated metrics.

        Args:
            X (pd.DataFrame): Cleaned, selected feature matrix.
            y (pd.Series): Binary target series.

        Returns:
            Dict: Dictionary with mean_auc, std_auc, mean_acc, and mean_f1.
        """
        date_order = X.index.get_level_values("date") if "date" in X.index.names else X.index
        sort_idx = date_order.argsort()
        X = X.iloc[sort_idx]
        y = y.reindex(X.index)
        if len(X) < 3:
            log.warning("[ValuationAgent] CV skipped: insufficient samples (%s)", len(X))
            return {"mean_auc": 0.0, "std_auc": 0.0, "mean_acc": 0.0, "mean_f1": 0.0}
        n_splits = min(5, len(X) - 1)
        if n_splits < 2:
            log.warning("[ValuationAgent] CV skipped: invalid n_splits=%s for n=%s", n_splits, len(X))
            return {"mean_auc": 0.0, "std_auc": 0.0, "mean_acc": 0.0, "mean_f1": 0.0}
        tss = TimeSeriesSplit(n_splits=n_splits)
        aucs, accs, f1s = [], [], []
        for tr, val in tss.split(X):
            y_tr = y.iloc[tr]
            y_val = y.iloc[val]
            if not self.has_multiple_classes(y_tr):
                continue
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(
                    n_estimators=self.n_estimators, max_depth=self.max_depth,
                    learning_rate=self.learning_rate, subsample=self.subsample,
                    random_state=self.random_seed)),
            ])
            pipe.fit(X.iloc[tr], y_tr)
            p = pipe.predict_proba(X.iloc[val])[:, 1]
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

    def _sector_summary(self, X: pd.DataFrame, sector_col: str) -> Dict:
        """Computes descriptive statistics for valuation multiples by sector.

        Args:
            X (pd.DataFrame): Feature matrix containing sector and multiple columns.
            sector_col (str): Column name for sector labels.

        Returns:
            Dict: Nested dictionary keyed by multiple → sector → statistics.
        """
        if sector_col not in X.columns:
            return {}
        out = {}
        for mult in VALUATION_FEATURE_COLUMNS:
            if mult not in X.columns:
                continue
            out[mult] = (X.groupby(sector_col)[mult]
                          .agg(["mean","median","std","count"])
                          .to_dict(orient="index"))
        return out
