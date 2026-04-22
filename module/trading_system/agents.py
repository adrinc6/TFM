from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from environment import (
    FUNDAMENTAL_FEATURE_COLUMNS,
    MOMENTUM_FEATURE_COLUMNS,
    RANDOM_SEED,
    SECTOR_ROTATION_FEATURE_COLUMNS,
    SENTIMENT_FEATURE_COLUMNS,
    VALUATION_FEATURE_COLUMNS,
)
from module.trading_system.feature_selection import FeatureSelector


@dataclass
class ModelPerformance:
    model_name: str
    cv_auc: float


@dataclass
class AgentOutput:
    scores: pd.Series
    model_name: str
    selected_features: list[str]
    performance: Dict[str, float] = field(default_factory=dict)


class BaseAgent:
    def __init__(self, name: str, feature_pool: list[str], min_features: int = 8, max_features: int = 12) -> None:
        self.name = name
        self.feature_pool = feature_pool
        self.min_features = min_features
        self.max_features = max_features
        self.selector = FeatureSelector()
        self.selected_features: list[str] = []
        self.model = None
        self.model_name: str = ""
        self.model_scores: Dict[str, float] = {}
        self.feature_importances: Dict[str, float] = {}

    @staticmethod
    def _safe_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        x = df.loc[:, list(columns)].copy()
        for c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.fillna(x.median(numeric_only=True))
        x = x.fillna(0.0)
        return x

    def _available_feature_pool(self, df: pd.DataFrame) -> list[str]:
        candidates = [c for c in self.feature_pool if c in df.columns]
        if len(candidates) >= self.min_features:
            return candidates
        auto_ratios = [
            c
            for c in df.columns
            if c not in {"forward_return", "year_quarter", "sector", "industry", "snapshot_date"}
            and pd.api.types.is_numeric_dtype(df[c])
            and ("_ratio" in c or "_margin" in c or "yield" in c or "pct" in c or "momentum" in c)
        ]
        merged = list(dict.fromkeys(candidates + auto_ratios))
        return merged[: max(self.max_features * 3, len(merged))]

    @staticmethod
    def _append_percentile_context(df: pd.DataFrame, selected_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
        out = df.copy()
        added: list[str] = []
        snapshot_col = "snapshot_date" if "snapshot_date" in out.columns else None
        sector_col = "sector" if "sector" in out.columns else None

        if snapshot_col is None:
            return out, selected_features

        for feat in selected_features:
            if feat not in out.columns:
                continue

            global_col = f"{feat}_pct_global"
            out[global_col] = out.groupby(snapshot_col)[feat].rank(method="average", pct=True)
            added.append(global_col)

            if sector_col is not None:
                sector_col_name = f"{feat}_pct_sector"
                out[sector_col_name] = out.groupby([snapshot_col, sector_col])[feat].rank(method="average", pct=True)
                added.append(sector_col_name)

        out[added] = out[added].fillna(0.5)
        return out, selected_features + added

    def _candidate_models(self) -> Dict[str, object]:
        return {
            "logistic": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
            "rf": RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            "gbm": GradientBoostingClassifier(random_state=RANDOM_SEED),
        }

    def _score_model(self, model, x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series) -> float:
        if y_train.nunique() < 2 or y_val.nunique() < 2:
            return 0.5
        mdl = clone(model)
        mdl.fit(x_train, y_train)
        if hasattr(mdl, "predict_proba"):
            p = mdl.predict_proba(x_val)[:, 1]
        else:
            raw = mdl.decision_function(x_val)
            p = 1.0 / (1.0 + np.exp(-raw))
        return float(roc_auc_score(y_val, p))

    def fit(self, train_df: pd.DataFrame, y_train: pd.Series, val_df: pd.DataFrame, y_val: pd.Series) -> None:
        pool = self._available_feature_pool(train_df)
        selection = self.selector.select(train_df, pool, y_train)
        selected = selection.selected
        if not selected:
            selected = pool[: min(len(pool), self.max_features)]
        selected = selected[: self.max_features]
        if len(selected) < self.min_features and len(pool) >= self.min_features:
            missing = [f for f in pool if f not in selected]
            selected = (selected + missing)[: self.min_features]

        train_aug, train_cols = self._append_percentile_context(train_df, selected)
        val_aug, val_cols = self._append_percentile_context(val_df, selected)
        model_features = [c for c in train_cols if c in train_aug.columns and c in val_aug.columns]

        x_train = self._safe_numeric(train_aug, model_features)
        x_val = self._safe_numeric(val_aug, model_features)
        y_train = pd.to_numeric(y_train, errors="coerce").fillna(0).astype(int)
        y_val = pd.to_numeric(y_val, errors="coerce").fillna(0).astype(int)

        best_score = -np.inf
        best_name = ""
        best_model = None
        model_scores: Dict[str, float] = {}

        for model_name, model in self._candidate_models().items():
            try:
                score = self._score_model(model, x_train, y_train, x_val, y_val)
            except Exception:
                score = -np.inf
            model_scores[model_name] = score
            if score > best_score:
                best_score = score
                best_name = model_name
                best_model = clone(model)

        if best_model is None:
            best_name = "logistic"
            best_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)

        all_train = pd.concat([train_aug, val_aug], axis=0)
        all_y = pd.concat([y_train, y_val], axis=0)
        x_all = self._safe_numeric(all_train, model_features)
        best_model.fit(x_all, all_y)

        self.model = best_model
        self.model_name = best_name
        self.model_scores = {k: float(v) for k, v in model_scores.items()}
        self.selected_features = model_features
        self.feature_importances = selection.importances

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None or not self.selected_features:
            return pd.Series(0.5, index=df.index, name=f"{self.name}_score")

        augmented, _ = self._append_percentile_context(df, [f for f in self.selected_features if not f.endswith("_pct_global") and not f.endswith("_pct_sector")])
        use_cols = [c for c in self.selected_features if c in augmented.columns]
        x = self._safe_numeric(augmented, use_cols)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x)[:, 1]
        else:
            raw = self.model.decision_function(x)
            probs = 1.0 / (1.0 + np.exp(-raw))

        s = pd.Series(np.clip(probs, 0.0, 1.0), index=df.index, name=f"{self.name}_score")
        return s


class FundamentalAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="fundamental", feature_pool=list(FUNDAMENTAL_FEATURE_COLUMNS))


class ValuationAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="valuation", feature_pool=list(VALUATION_FEATURE_COLUMNS))


class TechnicalAgent(BaseAgent):
    def __init__(self) -> None:
        technical_features = [
            "rsi_14",
            "rsi_28",
            "macd",
            "macd_signal",
            "macd_hist",
            "sma_20",
            "sma_50",
            "sma_200",
            "bb_pct",
            "atr_14",
            "volatility_20d",
            "volatility_60d",
            "vol_ratio_20_50",
            "price_vs_52w_high",
            "price_vs_52w_low",
        ]
        super().__init__(name="technical", feature_pool=technical_features)


class MomentumAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="momentum", feature_pool=list(MOMENTUM_FEATURE_COLUMNS))


class SectorRotationAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="sector_rotation", feature_pool=list(SECTOR_ROTATION_FEATURE_COLUMNS))


class SentimentAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="sentiment", feature_pool=list(SENTIMENT_FEATURE_COLUMNS))


def build_agents() -> list[BaseAgent]:
    return [
        FundamentalAgent(),
        ValuationAgent(),
        TechnicalAgent(),
        MomentumAgent(),
        SectorRotationAgent(),
        SentimentAgent(),
    ]
