"""Momentum agent (Random Forest) for the multi-agent stock picker.

Consumes technical features computed by TechnicalFeatureBuilder:
  Oscillators:  rsi_14, rsi_28
  Trend:        macd, macd_signal, macd_hist, sma_20/50/200 (distance %)
  Bands:        bb_pct
  52-week:      price_vs_52w_high, price_vs_52w_low
  Momentum:     momentum_1m/3m/6m/12m
  Volatility:   volatility_20d, volatility_60d, atr_14
  Volume:       vol_ratio_20_50
  Macro:        vix, yield_curve, sp500_momentum_3m, sp500_momentum_12m

Derived features built internally:
  rsi_overbought, rsi_oversold, above_sma200, macd_bullish,
  momentum_quality, vol_expansion, high_vix_regime,
  inverted_yield_curve, cross_sma_20_50
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import re

from module.agents.base import BaseAgent, FeatureSelector
from module.common.feature_controls import resolve_feature_columns
from module.steps.step_04_evaluation.explainability import build_explainer_for_agent, AgentExplainer
from environment import (
    MOMENTUM_N_ESTIMATORS, MOMENTUM_MAX_DEPTH, MOMENTUM_MIN_SAMPLES_LEAF,
    FEATURE_CORR_THRESHOLD, FEATURE_TOP_N,
    MOMENTUM_FEATURE_COLUMNS, MOMENTUM_FEATURE_EXCLUDE,
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

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


class _TFTLite(nn.Module):
    """Lightweight transformer encoder for sequence momentum classification."""

    def __init__(self, n_features: int, d_model: int = 32, n_heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.proj(x)
        z = self.encoder(h)
        pooled = z.mean(dim=1)
        gated = pooled * self.gate(pooled)
        return self.out(gated).squeeze(-1)

class MomentumAgent(BaseAgent):
    """Random Forest trained on technical indicators and earnings momentum.

    Uses TimeSeriesSplit for internal cross-validation (respects temporal
    order of price data, preventing look-ahead in validation).
    """

    def __init__(self, results_dir: str, random_seed: int = 42,
                 n_estimators: int = MOMENTUM_N_ESTIMATORS,
                 max_depth: int = MOMENTUM_MAX_DEPTH,
                 min_samples_leaf: int = MOMENTUM_MIN_SAMPLES_LEAF,
                 save_artifacts: bool = True):
        """Initialises the MomentumAgent.

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
        super().__init__("momentum", results_dir, random_seed, save_artifacts)
        if not _DEPS_OK:
            raise ImportError("scikit-learn is required.")
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._model:        Optional[Pipeline] = None
        self._feature_cols: List[str]          = []
        self._selector:     Optional[FeatureSelector] = None
        self._explainer:    Optional[AgentExplainer] = None
        self._deep_model: Optional[object] = None
        self._deep_seq_cols: List[str] = []
        self._deep_seq_features: List[str] = []
        self._deep_seq_len: int = 0
        self._deep_blend_weight: float = 0.35

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series,
            fold: Optional[int] = None) -> "MomentumAgent":
        """Trains the Random Forest on technical and earnings momentum features.

        Args:
            X (pd.DataFrame): Feature matrix for the training fold.
            y (pd.Series): Binary target labels (1 = Outperform).
            fold (Optional[int]): Walk-forward fold index for artefact naming.

        Returns:
            MomentumAgent: The fitted agent instance (self).
        """
        log.info(f"[MomentumAgent] Training RandomForest — {len(X)} obs, {len(X.columns)} features")
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len].copy()
        y = y.iloc[:min_len].copy()
        X_prep       = self._prepare(X, fit_mode=True)
        X_prep, y_cl = self.clean_features(X_prep, y.reset_index(drop=True))
        X_prep       = X_prep.reset_index(drop=True)
        y_cl         = y_cl.reset_index(drop=True)

        if not self.has_multiple_classes(y_cl):
            log.warning("[MomentumAgent] Label without enough class variance after cleaning - training skipped.")
            return self

        # Feature selection: only on training data (no leakage)
        self._selector = FeatureSelector(corr_threshold=FEATURE_CORR_THRESHOLD, top_n=FEATURE_TOP_N,
                                         min_features=3, random_seed=self.random_seed)
        X_prep = self._selector.fit_transform(X_prep, y_cl, agent_name="momentum")

        self._feature_cols = list(X_prep.columns)
        bal          = self.class_balance(y_cl)

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf, class_weight="balanced",
            random_state=self.random_seed, n_jobs=-1,
        )
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    rf),
        ])

        cv = self._cv(X_prep, y_cl)
        log.info(f"[MomentumAgent] CV AUC={cv['mean_auc']:.4f} ± {cv['std_auc']:.4f}  ({len(self._feature_cols)} selected features)")
        self._model.fit(X_prep, y_cl)
        self.is_trained = True

        imp = pd.Series(self._model.named_steps["clf"].feature_importances_, index=self._feature_cols)
        self.save_feature_importances(imp, fold)

        # Save macro regime distribution for diagnostics
        regime_summary = self._regime_summary(X_prep)
        self._diagnostics = {
            "class_balance": bal, "cv_metrics": cv,
            "regime_summary": regime_summary,
            "top_features": imp.nlargest(10).to_dict(),
            "feature_selection": self._selector.report(),
        }
        self.record_train_metrics(cv, fold)
        self.save_diagnostics(fold)
        rf_model = self._model.named_steps["clf"]

        # Deep sequence augmentation: uses seq_* columns when available.
        self._fit_deep_sequence_model(X.reset_index(drop=True), y.reset_index(drop=True))

        if self.save_artifacts:
            self._explainer = build_explainer_for_agent(
                self.name, rf_model, self._feature_cols,
                X_prep, self.results_dir.as_posix(), fold, model_type="tree"
            )
        else:
            self._explainer = None
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        """Returns bullish momentum scores in [0, 1].

        Args:
            X (pd.DataFrame): Feature matrix.

        Returns:
            pd.Series: Probability of Outperform indexed like X.

        Raises:
            RuntimeError: If the agent has not been trained yet.
        """
        if not self.is_trained:
            raise RuntimeError("[MomentumAgent] Not trained.")
        X_prep = self.clean_features_predict(self._prepare(X, fit_mode=False))
        if self._selector is not None:
            X_prep = self._selector.transform(X_prep)
        X_al = self._align(X_prep)
        rf_score = pd.Series(self._model.predict_proba(X_al)[:, 1], index=X.index, name="momentum_score")

        deep_score = self._predict_deep_sequence(X)
        if deep_score is None:
            return rf_score

        w = float(np.clip(self._deep_blend_weight, 0.0, 1.0))
        blended = ((1.0 - w) * rf_score + w * deep_score.reindex(rf_score.index).fillna(0.5)).clip(0.0, 1.0)
        return pd.Series(blended, index=X.index, name="momentum_score")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _prepare(self, X: pd.DataFrame, fit_mode: bool) -> pd.DataFrame:
        """Selects and engineers features for training or inference.

        Applies the feature column policy, then derives binary signals and
        composite features (e.g., rsi_overbought, macd_bullish).

        Args:
            X (pd.DataFrame): Raw feature matrix.
            fit_mode (bool): If True, stores the final column list for later
                alignment at inference time.

        Returns:
            pd.DataFrame: Prepared feature matrix.
        """
        df = X.copy()
        selected = resolve_feature_columns(
            default_cols=[],
            available_cols=list(df.columns),
            include_cols=MOMENTUM_FEATURE_COLUMNS,
            exclude_cols=MOMENTUM_FEATURE_EXCLUDE,
            logger=log,
            owner="MomentumAgent",
        )

        seq_cols = [c for c in df.columns if str(c).startswith("seq_")]
        if seq_cols:
            selected += seq_cols

        # Derived features: binary signals and composite indicators
        if "rsi_14" in df.columns:
            df["rsi_overbought"] = (df["rsi_14"] > 70).astype(float)
            df["rsi_oversold"]   = (df["rsi_14"] < 30).astype(float)
            selected += ["rsi_overbought", "rsi_oversold"]
        if "sma_200" in df.columns:
            df["above_sma200"] = (df["sma_200"] > 0).astype(float)
            selected.append("above_sma200")
        if "macd" in df.columns and "macd_signal" in df.columns:
            df["macd_bullish"] = (df["macd"] > df["macd_signal"]).astype(float)
            selected.append("macd_bullish")
        if "sma_20" in df.columns and "sma_50" in df.columns:
            df["cross_sma_20_50"] = (df["sma_20"] > df["sma_50"]).astype(float)
            selected.append("cross_sma_20_50")
        if "momentum_12m" in df.columns and "momentum_1m" in df.columns:
            df["momentum_quality"] = df["momentum_12m"] - df["momentum_1m"]
            selected.append("momentum_quality")
        if "vol_ratio_20_50" in df.columns:
            df["vol_expansion"] = (df["vol_ratio_20_50"] > 1.5).astype(float)
            selected.append("vol_expansion")

        selected = self._unique_existing_columns(df, selected)
        result = df[selected].copy()
        if fit_mode:
            self._feature_cols = list(result.columns)
        return result

    def _fit_deep_sequence_model(self, X_raw: pd.DataFrame, y: pd.Series) -> None:
        """Train a deterministic TFT-lite model on seq_* features when available."""
        if not _TORCH_OK:
            return

        seq_cols = [c for c in X_raw.columns if str(c).startswith("seq_")]
        if len(seq_cols) < 80:
            return

        pat = re.compile(r"^seq_(.+)_t(\d+)$")
        parsed = []
        for c in seq_cols:
            m = pat.match(str(c))
            if m:
                parsed.append((c, m.group(1), int(m.group(2))))
        if not parsed:
            return

        feats = sorted({p[1] for p in parsed})
        steps = sorted({p[2] for p in parsed})
        if len(feats) < 2 or len(steps) < 20:
            return

        self._deep_seq_cols = [p[0] for p in parsed]
        self._deep_seq_features = feats
        self._deep_seq_len = len(steps)

        col_map = {(feat, t): col for col, feat, t in parsed}
        X_arr = np.zeros((len(X_raw), len(steps), len(feats)), dtype=np.float32)
        for fi, feat in enumerate(feats):
            for ti, step in enumerate(steps):
                col = col_map.get((feat, step))
                if col is None:
                    continue
                X_arr[:, ti, fi] = pd.to_numeric(X_raw[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        y_vec = pd.to_numeric(y.reindex(X_raw.index), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if len(np.unique(y_vec)) < 2:
            return

        torch.manual_seed(int(self.random_seed))
        np.random.seed(int(self.random_seed))

        model = _TFTLite(n_features=len(feats), d_model=32, n_heads=4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()

        X_t = torch.tensor(X_arr, dtype=torch.float32)
        y_t = torch.tensor(y_vec, dtype=torch.float32)

        model.train()
        for _ in range(15):
            opt.zero_grad(set_to_none=True)
            logits = model(X_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            opt.step()

        self._deep_model = model.eval()
        log.info("[MomentumAgent] Deep sequence model trained (TFT-lite) with %d seq features x %d steps", len(feats), len(steps))

    def _predict_deep_sequence(self, X_raw: pd.DataFrame) -> Optional[pd.Series]:
        if not _TORCH_OK or self._deep_model is None or not self._deep_seq_features or self._deep_seq_len <= 0:
            return None

        pat = re.compile(r"^seq_(.+)_t(\d+)$")
        parsed = {}
        for c in X_raw.columns:
            m = pat.match(str(c))
            if m:
                parsed[(m.group(1), int(m.group(2)))] = c

        feats = self._deep_seq_features
        steps = list(range(self._deep_seq_len))
        X_arr = np.zeros((len(X_raw), len(steps), len(feats)), dtype=np.float32)
        for fi, feat in enumerate(feats):
            for ti, step in enumerate(steps):
                col = parsed.get((feat, step))
                if col is None:
                    continue
                X_arr[:, ti, fi] = pd.to_numeric(X_raw[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        with torch.no_grad():
            p = torch.sigmoid(self._deep_model(torch.tensor(X_arr, dtype=torch.float32))).detach().cpu().numpy()
        return pd.Series(p.astype(float), index=X_raw.index)

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
        if len(X) < 3:
            log.warning("[MomentumAgent] CV skipped: insufficient samples (%s)", len(X))
            return {"mean_auc": 0.0, "std_auc": 0.0, "mean_acc": 0.0, "mean_f1": 0.0}
        n_splits = min(5, len(X) - 1)
        if n_splits < 2:
            log.warning("[MomentumAgent] CV skipped: invalid n_splits=%s for n=%s", n_splits, len(X))
            return {"mean_auc": 0.0, "std_auc": 0.0, "mean_acc": 0.0, "mean_f1": 0.0}
        tss = TimeSeriesSplit(n_splits=n_splits)
        aucs, accs, f1s = [], [], []
        for tr, val in tss.split(X):
            y_tr = y.iloc[tr]
            y_val = y.iloc[val]
            if not self.has_multiple_classes(y_tr):
                continue
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf, class_weight="balanced",
                random_state=self.random_seed, n_jobs=-1,
            )
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", rf)])
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

    @staticmethod
    def _regime_summary(X: pd.DataFrame) -> Dict:
        """Computes market-regime proportions from binary indicator columns.

        Args:
            X (pd.DataFrame): Feature matrix containing binary regime flags.

        Returns:
            Dict: Proportion of observations where each regime flag is active.
        """
        out = {}
        if "above_sma200" in X.columns:
            out["pct_above_sma200"]   = float(X["above_sma200"].mean())
        if "rsi_overbought" in X.columns:
            out["pct_rsi_overbought"] = float(X["rsi_overbought"].mean())
        if "rsi_oversold" in X.columns:
            out["pct_rsi_oversold"]   = float(X["rsi_oversold"].mean())
        if "macd_bullish" in X.columns:
            out["pct_macd_bullish"]   = float(X["macd_bullish"].mean())
        if "consistent_beater" in X.columns:
            out["pct_consistent_beater"] = float(X["consistent_beater"].mean())
        return out
