from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from environment import (
    ANALYSIS_FREQUENCY,
    META_LR_C,
    RANDOM_SEED,
    TP_SL_MAX_HOLDING_DAYS,
    TP_SL_PRIMARY_STRATEGY,
)
from module.trading_system.agents import BaseAgent, build_agents
from module.trading_system.data import DataLayer
from module.trading_system.strategies import BaseStrategy, StrategyResult, build_strategies


VALIDATION_SPLIT_RATIO = 0.20
MIN_TRAIN_PERIODS = 8


@dataclass
class TrainingArtifacts:
    diagnostics: pd.DataFrame
    strategy_diagnostics: pd.DataFrame
    model_performance: pd.DataFrame
    strategy_performance: pd.DataFrame


class MetaModel:
    def __init__(self) -> None:
        self.model = LogisticRegression(
            C=META_LR_C,
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
        self.feature_names: list[str] = []
        self.validation_auc: float = np.nan

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series) -> None:
        self.feature_names = list(x_train.columns)
        self.model.fit(x_train, y_train)
        if y_val.nunique() >= 2:
            p = self.model.predict_proba(x_val[self.feature_names])[:, 1]
            self.validation_auc = float(roc_auc_score(y_val, p))
        else:
            self.validation_auc = 0.5

    def predict(self, x: pd.DataFrame) -> pd.Series:
        p = self.model.predict_proba(x[self.feature_names])[:, 1]
        return pd.Series(np.clip(p, 0.0, 1.0), index=x.index, name="meta_score")


def _time_key(series: pd.Series, freq: str) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    if freq == "annual":
        return dt.dt.to_period("Y").astype(str)
    return dt.dt.to_period("Q").astype(str)


def _split_train_val(df: pd.DataFrame, freq: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = _time_key(df["snapshot_date"], freq)
    periods = sorted(keys.dropna().unique().tolist())
    if len(periods) <= 2:
        split_point = max(int(len(df) * (1.0 - VALIDATION_SPLIT_RATIO)), 1)
        return df.iloc[:split_point].copy(), df.iloc[split_point:].copy()
    val_periods = set(periods[-max(1, int(len(periods) * VALIDATION_SPLIT_RATIO)):])
    train = df[keys.isin([p for p in periods if p not in val_periods])].copy()
    val = df[keys.isin(val_periods)].copy()
    if train.empty or val.empty:
        split_point = max(int(len(df) * (1.0 - VALIDATION_SPLIT_RATIO)), 1)
        train, val = df.iloc[:split_point].copy(), df.iloc[split_point:].copy()
    return train, val


def _walk_forward_folds(df: pd.DataFrame, freq: str, min_train_periods: int = MIN_TRAIN_PERIODS) -> list[tuple[pd.Index, pd.Index]]:
    keys = _time_key(df["snapshot_date"], freq)
    unique_periods = sorted(keys.dropna().unique().tolist())
    if len(unique_periods) <= min_train_periods + 1:
        if len(unique_periods) < 2:
            return []
        train_periods = unique_periods[:-1]
        test_period = unique_periods[-1]
        return [(df[keys.isin(train_periods)].index, df[keys == test_period].index)]

    folds: list[tuple[pd.Index, pd.Index]] = []
    for i in range(min_train_periods, len(unique_periods)):
        train_periods = unique_periods[:i]
        test_period = unique_periods[i]
        train_idx = df[keys.isin(train_periods)].index
        test_idx = df[keys == test_period].index
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        folds.append((train_idx, test_idx))
    return folds


def _compute_strategy_result(
    strategies: Dict[str, BaseStrategy],
    prices_cache: Dict[str, pd.DataFrame],
    ticker: str,
    snapshot_date: pd.Timestamp,
) -> Dict[str, StrategyResult]:
    price_df = prices_cache.get(str(ticker), pd.DataFrame())
    out: Dict[str, StrategyResult] = {}
    for strategy_name, strategy in strategies.items():
        out[strategy_name] = strategy.evaluate_outcome(
            prices=price_df,
            entry_date=snapshot_date,
            max_holding_days=TP_SL_MAX_HOLDING_DAYS,
        )
    return out


def _add_strategy_targets(master_df: pd.DataFrame, prices_cache: Dict[str, pd.DataFrame], strategies: Dict[str, BaseStrategy]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = master_df.reset_index().copy()
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="coerce")
    data = data.dropna(subset=["snapshot_date"])

    records: List[dict] = []
    strategy_rows: List[dict] = []

    for row in data.itertuples(index=False):
        ticker = str(getattr(row, "ticker"))
        snapshot_date = pd.Timestamp(getattr(row, "snapshot_date"))
        strategy_map = _compute_strategy_result(strategies, prices_cache, ticker=ticker, snapshot_date=snapshot_date)

        rec = row._asdict()
        rec["ticker"] = ticker
        rec["snapshot_date"] = snapshot_date

        for strategy_name, result in strategy_map.items():
            rec[f"label_{strategy_name}"] = int(result.label)
            rec[f"outcome_{strategy_name}"] = result.outcome
            rec[f"days_to_event_{strategy_name}"] = int(result.days_to_event)
            rec[f"tp_level_{strategy_name}"] = float(result.tp_level) if np.isfinite(result.tp_level) else np.nan
            rec[f"sl_level_{strategy_name}"] = float(result.sl_level) if np.isfinite(result.sl_level) else np.nan
            rec[f"entry_price_{strategy_name}"] = float(result.entry_price) if np.isfinite(result.entry_price) else np.nan

            strategy_rows.append(
                {
                    "ticker": ticker,
                    "date": pd.Timestamp(getattr(row, "date")),
                    "snapshot_date": snapshot_date,
                    "strategy": strategy_name,
                    "tp_pct": result.tp_pct,
                    "sl_pct": result.sl_pct,
                    "tp_level": rec[f"tp_level_{strategy_name}"],
                    "sl_level": rec[f"sl_level_{strategy_name}"],
                    "entry_price": rec[f"entry_price_{strategy_name}"],
                    "actual_outcome": result.outcome,
                    "label": result.label,
                    "days_to_event": result.days_to_event,
                    "sector": rec.get("sector", "Unknown"),
                    "year_quarter": rec.get("year_quarter"),
                }
            )

        records.append(rec)

    target_df = pd.DataFrame(records).set_index(["ticker", "date"]).sort_index()
    strategy_df = pd.DataFrame(strategy_rows).set_index(["ticker", "date"]).sort_index()
    return target_df, strategy_df


def _agent_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    drop_prefixes = (
        "label_",
        "outcome_",
        "days_to_event_",
        "tp_level_",
        "sl_level_",
        "entry_price_",
    )
    drop_cols = [c for c in df.columns if any(c.startswith(prefix) for prefix in drop_prefixes)]
    feat = df.drop(columns=drop_cols, errors="ignore")
    return feat


def train_multi_agent_system(data_layer: DataLayer | None = None) -> TrainingArtifacts:
    data_layer = data_layer or DataLayer()
    strategies = build_strategies()
    master_df = data_layer.load_master_dataset()
    prices_cache = data_layer.load_price_cache(master_df.index.get_level_values("ticker"))

    training_df, strategy_truth = _add_strategy_targets(master_df, prices_cache, strategies)
    label_col = f"label_{TP_SL_PRIMARY_STRATEGY}"
    if label_col not in training_df.columns:
        raise ValueError(f"Missing primary strategy label column: {label_col}")

    features_df = _agent_feature_frame(training_df)
    y_full = pd.to_numeric(training_df[label_col], errors="coerce").fillna(0).astype(int)

    folds = _walk_forward_folds(training_df, ANALYSIS_FREQUENCY, min_train_periods=MIN_TRAIN_PERIODS)
    if not folds:
        raise RuntimeError("Not enough temporal periods for walk-forward training")

    diagnostics_parts: list[pd.DataFrame] = []
    strategy_diag_parts: list[pd.DataFrame] = []
    model_perf_rows: list[dict] = []

    for fold_no, (train_idx, test_idx) in enumerate(folds, start=1):
        fold_train_df = training_df.loc[train_idx].copy()
        fold_test_df = training_df.loc[test_idx].copy()

        if fold_train_df.empty or fold_test_df.empty:
            continue

        train_sub, val_sub = _split_train_val(fold_train_df.reset_index(), ANALYSIS_FREQUENCY)
        train_sub = train_sub.set_index(["ticker", "date"]).sort_index()
        val_sub = val_sub.set_index(["ticker", "date"]).sort_index()
        if train_sub.empty or val_sub.empty:
            continue

        x_train = _agent_feature_frame(train_sub)
        x_val = _agent_feature_frame(val_sub)
        x_test = _agent_feature_frame(fold_test_df)

        y_train = pd.to_numeric(train_sub[label_col], errors="coerce").fillna(0).astype(int)
        y_val = pd.to_numeric(val_sub[label_col], errors="coerce").fillna(0).astype(int)
        y_test = pd.to_numeric(fold_test_df[label_col], errors="coerce").fillna(0).astype(int)

        agents: list[BaseAgent] = build_agents()

        test_pred = pd.DataFrame(index=fold_test_df.index)
        val_pred = pd.DataFrame(index=val_sub.index)

        for agent in agents:
            agent.fit(x_train, y_train, x_val, y_val)
            test_pred[f"{agent.name}_score"] = agent.predict_proba(x_test)
            val_pred[f"{agent.name}_score"] = agent.predict_proba(x_val)

            model_perf_rows.append(
                {
                    "fold": fold_no,
                    "model": agent.name,
                    "selected_feature_count": len(agent.selected_features),
                    "selected_features": ",".join(agent.selected_features),
                    "best_model": agent.model_name,
                    **{f"cv_auc_{k}": float(v) for k, v in agent.model_scores.items()},
                }
            )

        meta_train_idx, meta_val_idx = _split_train_val(
            val_sub.reset_index(),
            ANALYSIS_FREQUENCY,
        )
        val_reset = val_pred.reset_index()
        val_reset["date"] = pd.to_datetime(val_reset["date"], errors="coerce")
        val_keys = pd.MultiIndex.from_frame(val_reset[["ticker", "date"]])
        train_keys = pd.MultiIndex.from_frame(meta_train_idx[["ticker", "date"]]) if not meta_train_idx.empty else pd.MultiIndex(levels=[[], []], codes=[[], []])
        valid_keys = pd.MultiIndex.from_frame(meta_val_idx[["ticker", "date"]]) if not meta_val_idx.empty else pd.MultiIndex(levels=[[], []], codes=[[], []])
        train_mask = val_keys.isin(train_keys)
        valid_mask = val_keys.isin(valid_keys)

        meta_x_train = val_pred.loc[train_mask] if bool(train_mask.any()) else val_pred
        meta_y_train = y_val.loc[meta_x_train.index]
        meta_x_val = val_pred.loc[valid_mask] if bool(valid_mask.any()) else val_pred
        meta_y_val = y_val.loc[meta_x_val.index]

        meta = MetaModel()
        meta.fit(
            x_train=meta_x_train,
            y_train=meta_y_train,
            x_val=meta_x_val,
            y_val=meta_y_val,
        )
        meta_test = meta.predict(test_pred)
        test_pred["meta_score"] = meta_test

        fold_diag = fold_test_df[["snapshot_date", "year_quarter", "sector"]].copy()
        fold_diag = fold_diag.join(test_pred)
        fold_diag["label"] = y_test
        fold_diag["fold"] = fold_no

        diagnostics_parts.append(fold_diag)

        fold_strategy_truth = strategy_truth.loc[fold_test_df.index].copy()
        fold_strategy_truth = fold_strategy_truth.join(test_pred, how="left")
        fold_strategy_truth["fold"] = fold_no
        strategy_diag_parts.append(fold_strategy_truth)

        model_perf_rows.append(
            {
                "fold": fold_no,
                "model": "meta_model",
                "selected_feature_count": len(meta.feature_names),
                "selected_features": ",".join(meta.feature_names),
                "best_model": "logistic",
                "cv_auc_logistic": float(meta.validation_auc),
            }
        )

    diagnostics = pd.concat(diagnostics_parts, axis=0).sort_index() if diagnostics_parts else pd.DataFrame()
    strategy_diagnostics = pd.concat(strategy_diag_parts, axis=0).sort_index() if strategy_diag_parts else pd.DataFrame()
    model_performance = pd.DataFrame(model_perf_rows)
    strategy_performance = _summarize_strategy_performance(strategy_diagnostics)

    return TrainingArtifacts(
        diagnostics=diagnostics,
        strategy_diagnostics=strategy_diagnostics,
        model_performance=model_performance,
        strategy_performance=strategy_performance,
    )


def _summarize_strategy_performance(strategy_diag: pd.DataFrame) -> pd.DataFrame:
    if strategy_diag is None or strategy_diag.empty:
        return pd.DataFrame(
            columns=[
                "strategy",
                "n_obs",
                "hit_rate",
                "expected_value",
                "avg_days_to_event",
            ]
        )

    rows = []
    for strategy, g in strategy_diag.groupby("strategy"):
        labels = pd.to_numeric(g["label"], errors="coerce").fillna(0.0)
        hit_rate = float(labels.mean())
        tp_pct = float(pd.to_numeric(g["tp_pct"], errors="coerce").mean())
        sl_pct = float(pd.to_numeric(g["sl_pct"], errors="coerce").mean())
        ev = float((labels * tp_pct - (1 - labels) * sl_pct).mean())
        avg_days = float(pd.to_numeric(g["days_to_event"], errors="coerce").mean())
        rows.append(
            {
                "strategy": strategy,
                "n_obs": int(len(g)),
                "hit_rate": hit_rate,
                "expected_value": ev,
                "avg_days_to_event": avg_days,
            }
        )

    return pd.DataFrame(rows).sort_values("expected_value", ascending=False).reset_index(drop=True)
