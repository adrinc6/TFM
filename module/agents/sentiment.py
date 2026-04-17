"""Sentiment agent (Random Forest) for the multi-agent stock picker.

Consumes features computed by SentimentFeatureBuilder:
  Analysts:  analyst_buy_ratio, analyst_bearish_score, analyst_consensus,
             analyst_dispersion, analyst_strong_buy_pct,
             analyst_consensus_change
  Insiders:  mspr_3m, mspr_trend, insider_net_ratio_90d, insider_sell_ratio
  EPS:       beat_rate_4q, eps_surprise_avg_4q, eps_surprise_pct

Captures the "wisdom of the crowd" from institutional analysts plus insider
signals as a predictive factor for relative performance over 1 quarter.
Complements the ValuationAgent (which uses EPS surprises as a secondary feature).
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from module.agents.base import BaseAgent, FeatureSelector
from module.common.feature_controls import resolve_feature_columns
from module.steps.step_04_evaluation.explainability import AgentExplainer, build_explainer_for_agent
from environment import (
    SENTIMENT_N_ESTIMATORS, SENTIMENT_MAX_DEPTH, SENTIMENT_MIN_SAMPLES_LEAF,
    FEATURE_CORR_THRESHOLD, FEATURE_TOP_N,
    SENTIMENT_FEATURE_COLUMNS, SENTIMENT_FEATURE_EXCLUDE,
    DEBUG_EXPORT_AGENT_INPUTS,
)

log = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    from sklearn.model_selection import TimeSeriesSplit
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

class SentimentAgent(BaseAgent):
    """Random Forest trained on institutional sentiment signals.

    Captures:
      - Analyst consensus and trend (recommendation_trends)
      - Monthly insider MSPR (insider_sentiment) — predictive signal over 30-90d
      - Net insider buying (insider_transactions)
      - EPS beat rate and surprises (eps_surprises)

    Score [0, 1]: 1 = very positive sentiment → bullish signal
    """

    def __init__(
        self,
        results_dir: str,
        random_seed: int = 42,
        n_estimators: int = SENTIMENT_N_ESTIMATORS,
        max_depth: int = SENTIMENT_MAX_DEPTH,
        min_samples_leaf: int = SENTIMENT_MIN_SAMPLES_LEAF,
        save_artifacts: bool = True,
    ):
        """Initialises the SentimentAgent.

        Args:
            results_dir (str): Directory where training artefacts are saved.
            random_seed (int): Random seed for reproducibility.
            n_estimators (int): Number of trees in the Random Forest.
            max_depth (int): Maximum tree depth.
            min_samples_leaf (int): Minimum samples per leaf node.
            save_artifacts (bool): Whether to save diagnostics and models.

        Raises:
            ImportError: If scikit-learn is not installed.
        """
        super().__init__("sentiment", results_dir, random_seed, save_artifacts)
        if not _DEPS_OK:
            raise ImportError("scikit-learn is required.")
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._model:        Optional[Pipeline]         = None
        self._feature_cols: List[str]                  = []
        self._selector:     Optional[FeatureSelector]  = None
        self._explainer:    Optional[AgentExplainer]   = None

    def _export_aapl_debug_input(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fold: Optional[int] = None,
        stage: str = "raw",
    ) -> None:
        """Exports AAPL rows to CSV for debugging when DEBUG_EXPORT_AGENT_INPUTS is enabled.

        Args:
            X (pd.DataFrame): Feature matrix to inspect.
            y (pd.Series): Target series aligned with X.
            fold (Optional[int]): Fold index for file naming.
            stage (str): Processing stage label (e.g., 'raw', 'cleaned').
        """
        if not DEBUG_EXPORT_AGENT_INPUTS:
            return
        ticker_series = None
        if isinstance(X.index, pd.MultiIndex) and "ticker" in X.index.names:
            ticker_series = pd.Series(X.index.get_level_values("ticker"), index=X.index)
        elif "ticker" in X.columns:
            ticker_series = pd.Series(X["ticker"].values, index=X.index)

        if ticker_series is None:
            return

        mask = ticker_series.astype(str).str.upper().eq("AAPL")
        if not mask.any():
            return

        debug_df = X.loc[mask].copy()
        if y is not None:
            if debug_df.index.isin(y.index).all():
                debug_df["target"] = y.loc[debug_df.index].values
            else:
                debug_df["target"] = y.iloc[mask.to_numpy()].values
        if isinstance(debug_df.index, pd.MultiIndex):
            debug_df = debug_df.reset_index()
        else:
            debug_df = debug_df.reset_index(drop=False)

        suffix = f"_{fold}" if fold is not None else ""
        stage_suffix = "" if stage == "raw" else f"_{stage}"
        out_path = self.results_dir / f"aapl_sentiment_input{stage_suffix}{suffix}.csv"
        debug_df.to_csv(out_path, index=False)
        log.info(f"[SentimentAgent] Debug AAPL {stage} input -> {out_path.name} ({len(debug_df)} rows)")

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series,
            fold: Optional[int] = None) -> "SentimentAgent":
        """Trains the Random Forest on analyst and insider sentiment features.

        Args:
            X (pd.DataFrame): Feature matrix for the training fold.
            y (pd.Series): Binary target labels (1 = Outperform).
            fold (Optional[int]): Walk-forward fold index for artefact naming.

        Returns:
            SentimentAgent: The fitted agent instance (self).
        """
        log.info(f"[SentimentAgent] Training RandomForest (analysts + insiders + EPS) — {len(X)} obs")

        min_len = min(len(X), len(y))
        X = X.iloc[:min_len].copy()
        y = y.iloc[:min_len].copy()

        self._export_aapl_debug_input(X, y, fold=fold, stage="raw")

        X_prep       = self._prepare(X)
        X_prep, y_cl = self.clean_features(X_prep, y.reset_index(drop=True))

        self._export_aapl_debug_input(X_prep, y_cl, fold=fold, stage="cleaned")

        X_prep       = X_prep.reset_index(drop=True)
        y_cl         = y_cl.reset_index(drop=True)

        if len(X_prep) < 20:
            log.warning(
                f"[SentimentAgent] Insufficient samples to train: "
                f"{len(X_prep)} < 20 after cleaning."
            )
            return self

        # Selección de features
        self._selector = FeatureSelector(
            corr_threshold=FEATURE_CORR_THRESHOLD,
            top_n=FEATURE_TOP_N,
            min_features=5,
            random_seed=self.random_seed,
        )
        X_prep = self._selector.fit_transform(X_prep, y_cl, agent_name="sentiment")
        self._export_aapl_debug_input(X_prep, y_cl, fold=fold, stage="selected")
        self._feature_cols = list(X_prep.columns)

        bal = self.class_balance(y_cl)

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced",
            random_state=self.random_seed,
            n_jobs=-1,
        )
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    rf),
        ])

        cv = self._cv(X_prep, y_cl)
        log.info(f"[SentimentAgent] CV AUC={cv['mean_auc']:.4f} ± {cv['std_auc']:.4f}  ({len(self._feature_cols)} selected features)")
        self._model.fit(X_prep, y_cl)
        self.is_trained = True

        rf_fitted = self._model.named_steps["clf"]
        imp = pd.Series(rf_fitted.feature_importances_, index=self._feature_cols)
        self.save_feature_importances(imp, fold)

        self._explainer = build_explainer_for_agent(
            agent_name="sentiment",
            model=rf_fitted,
            feature_cols=self._feature_cols,
            X_train=X_prep,
            results_dir=self.results_dir.parent.as_posix(),
            fold=fold,
            model_type="tree",
        )

        self._diagnostics = {
            "class_balance":     bal,
            "cv_metrics":        cv,
            "top_features":      imp.nlargest(10).to_dict(),
            "feature_selection": self._selector.report(),
            "coverage": {
                "pct_has_analyst": float((X_prep.get("analyst_buy_ratio", pd.Series(np.nan)).notna()).mean()),
                "pct_has_mspr":    float((X_prep.get("mspr_3m", pd.Series(np.nan)).notna()).mean()),
                "pct_has_beat":    float((X_prep.get("beat_rate_4q", pd.Series(np.nan)).notna()).mean()),
            },
        }
        self.record_train_metrics(cv, fold)
        self.save_diagnostics(fold)
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        """Returns sentiment scores in [0, 1].

        Args:
            X (pd.DataFrame): Feature matrix.

        Returns:
            pd.Series: Probability of Outperform indexed like X.

        Raises:
            RuntimeError: If the agent has not been trained yet.
        """
        if not self.is_trained:
            raise RuntimeError("[SentimentAgent] Not trained.")
        X_prep = self.clean_features_predict(self._prepare(X))
        if self._selector is not None:
            X_prep = self._selector.transform(X_prep)
        X_al = self._align(X_prep)
        return pd.Series(
            self._model.predict_proba(X_al)[:, 1],
            index=X.index,
            name="sentiment_score",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Selects feature columns and derives composite sentiment signals.

        Args:
            X (pd.DataFrame): Raw feature matrix.

        Returns:
            pd.DataFrame: Prepared feature matrix with derived signals added.
        """
        df = X.copy()
        selected = resolve_feature_columns(
            default_cols=[],
            available_cols=list(df.columns),
            include_cols=SENTIMENT_FEATURE_COLUMNS,
            exclude_cols=SENTIMENT_FEATURE_EXCLUDE,
            logger=log,
            owner="SentimentAgent",
        )

        # Derived feature: net bullish signal (buy_ratio - bearish_score)
        if "analyst_buy_ratio" in df.columns and "analyst_bearish_score" in df.columns:
            df["analyst_net_bullish"] = df["analyst_buy_ratio"] - df["analyst_bearish_score"]
            selected.append("analyst_net_bullish")

        # Derived feature: z-score of net insider balance (ratio form)
        net_col = None
        if "insider_net_ratio_90d" in df.columns:
            net_col = "insider_net_ratio_90d"
        elif "insider_net_shares_90d" in df.columns:
            # Backward compatibility with historical datasets.
            net_col = "insider_net_shares_90d"

        if net_col is not None:
            net = df[net_col]
            std = net.std()
            if std > 0:
                df["insider_net_zscore"] = (net - net.mean()) / std
            else:
                df["insider_net_zscore"] = 0.0
            selected.append("insider_net_zscore")

        # Derived feature: positive MSPR flag (insiders actively buying)
        if "mspr_3m" in df.columns:
            df["mspr_positive"] = (df["mspr_3m"] > 20).astype(float)
            df["mspr_negative"] = (df["mspr_3m"] < -20).astype(float)
            selected += ["mspr_positive", "mspr_negative"]

        # Derived feature: beat streak (beat_rate > 75% = consistent outperformer)
        if "beat_rate_4q" in df.columns:
            df["consistent_beater"] = (df["beat_rate_4q"] >= 0.75).astype(float)
            selected.append("consistent_beater")

        selected = self._unique_existing_columns(df, selected)
        return df[selected].copy()

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aligns X to the training feature schema.

        Args:
            X (pd.DataFrame): Feature matrix to align.

        Returns:
            pd.DataFrame: Aligned feature matrix.
        """
        return self._align_to_feature_cols(X, fill_value=0.0)

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
        tss = TimeSeriesSplit(n_splits=5)
        aucs, accs, f1s = [], [], []
        for tr, val in tss.split(X):
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight="balanced",
                random_state=self.random_seed,
                n_jobs=-1,
            )
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", rf)])
            pipe.fit(X.iloc[tr], y.iloc[tr])
            p = pipe.predict_proba(X.iloc[val])[:, 1]
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
