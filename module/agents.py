"""Agentes Ridge especializados con entrenamiento walk-forward trimestral."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from environment import Settings
from module.meta import AGENT_NAMES, combine_agent_scores
from module.utils import read_parquet, sha256_file, write_json, write_parquet

log = logging.getLogger(__name__)

AGENT_FEATURES = {
    "quality": [
        "factor_roe",
        "factor_roic",
        "factor_net_margin",
        "factor_operating_margin",
        "factor_gross_margin",
        "factor_fcf_margin",
        "factor_debt_equity",
        "factor_current_ratio",
    ],
    "momentum": [
        "factor_relative_return_3m",
        "factor_relative_return_6m",
        "factor_relative_return_12m",
    ],
    "value": ["factor_pe", "factor_pb", "factor_ps", "factor_ev_ebitda"],
}


def build_agent_scores(settings: Settings) -> pd.DataFrame:
    """Entrena agentes sin futuro y combina sus scores mediante meta rank-IC."""
    output_dir = settings.processed_output_dir
    feature_path = output_dir / "features_point_in_time.parquet"
    target_path = output_dir / "targets_forward_3m.parquet"
    features = read_parquet(feature_path, "RUN_MODE='features'")
    targets = read_parquet(target_path, "RUN_MODE='features'")
    _validate_inputs(features, targets)
    run_dir = output_dir / "agents" / _run_id(settings, feature_path, target_path)
    frame = features.merge(targets, on=["ticker", "snapshot_date"], how="left", validate="one_to_one")
    frame["snapshot_ts"] = pd.to_datetime(frame["snapshot_date"])
    frame["label_end_ts"] = pd.to_datetime(frame["label_end_date"])
    frame["is_quarterly"] = frame["review_type"].eq("fundamental_quarterly")

    predictions, coefficients = _walk_forward_scores(frame, settings)
    if predictions.empty:
        raise RuntimeError(
            "No se pudieron entrenar agentes con la historia disponible. "
            "Revisa la cobertura de features, etiquetas y el mínimo de filas de entrenamiento."
        )
    meta_scores, weights, diagnostics = combine_agent_scores(predictions, targets, settings)
    wide = _wide_scores(predictions, meta_scores)

    write_parquet(wide, run_dir / "agent_scores.parquet")
    write_parquet(weights, run_dir / "meta_weights.parquet")
    write_parquet(diagnostics, run_dir / "rank_ic_diagnostics.parquet")
    write_parquet(coefficients, run_dir / "model_coefficients.parquet")
    write_json(_manifest(settings, feature_path, target_path, predictions, run_dir), run_dir / "manifest.json")
    log.info("Agentes: rows=%s runs=%s output=%s", len(wide), wide["snapshot_date"].nunique(), run_dir)
    return wide


def _walk_forward_scores(frame: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchor = _execution_anchor(settings)
    retrain_dates = sorted(
        frame.loc[
            frame["is_quarterly"] & frame["snapshot_ts"].ge(anchor), "snapshot_ts"
        ].drop_duplicates()
    )
    rows: list[dict] = []
    coefficients: list[dict] = []
    for index, retrain_date in enumerate(retrain_dates):
        next_date = retrain_dates[index + 1] if index + 1 < len(retrain_dates) else None
        training_start = max(
            frame["snapshot_ts"].min(), retrain_date - pd.DateOffset(years=settings.train_lookback_years)
        )
        training = frame.loc[
            frame["is_quarterly"]
            & frame["snapshot_ts"].ge(training_start)
            & frame["snapshot_ts"].lt(retrain_date)
            & frame["target_available"].fillna(False)
            & frame["label_end_ts"].le(retrain_date)
            & frame["is_price_fresh"].fillna(False)
        ]
        scoring = frame.loc[
            frame["snapshot_ts"].ge(retrain_date)
            & (frame["snapshot_ts"].lt(next_date) if next_date is not None else True)
            & frame["is_price_fresh"].fillna(False)
        ]
        for agent, columns in AGENT_FEATURES.items():
            train = training.dropna(subset=["forward_excess_return_3m"])
            if len(train) < settings.min_training_rows:
                log.warning("[%s %s] filas insuficientes: %s", agent, retrain_date.date(), len(train))
                continue
            model = _ridge_pipeline(settings)
            model.fit(train[columns], train["forward_excess_return_3m"])
            if scoring.empty:
                continue
            values = model.predict(scoring[columns])
            for row, score in zip(scoring.itertuples(index=False), values, strict=True):
                rows.append(
                    {
                        "ticker": row.ticker,
                        "snapshot_date": row.snapshot_date,
                        "model_retrain_date": retrain_date.date().isoformat(),
                        "agent": agent,
                        "score": float(score),
                        "is_quarterly": bool(row.is_quarterly),
                        "training_start_date": pd.Timestamp(training_start).date().isoformat(),
                        "training_end_date": retrain_date.date().isoformat(),
                        "training_rows": len(train),
                    }
                )
            coefficients.extend(_coefficient_rows(model, agent, retrain_date, len(train)))
    return pd.DataFrame(rows), pd.DataFrame(coefficients)


def _ridge_pipeline(settings: Settings) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=settings.ridge_alpha)),
        ]
    )


def _coefficient_rows(model: Pipeline, agent: str, retrain_date: pd.Timestamp, rows: int) -> list[dict]:
    names = model[:-1].get_feature_names_out()
    coefficients = model.named_steps["ridge"].coef_
    return [
        {
            "agent": agent,
            "model_retrain_date": retrain_date.date().isoformat(),
            "feature": str(name),
            "coefficient": float(value),
            "training_rows": rows,
        }
        for name, value in zip(names, coefficients, strict=True)
    ]


def _wide_scores(predictions: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        "ticker",
        "snapshot_date",
        "model_retrain_date",
        "is_quarterly",
        "training_start_date",
        "training_end_date",
        "training_rows",
    ]
    wide = predictions.pivot(index=identifiers, columns="agent", values="score").reset_index()
    wide.columns.name = None
    for agent in AGENT_NAMES:
        if agent not in wide:
            wide[agent] = float("nan")
        wide[f"{agent}_rank"] = wide.groupby("snapshot_date")[agent].rank(method="average", pct=True)
    wide = wide.merge(meta, on=["ticker", "snapshot_date"], how="left", validate="one_to_one")
    return wide.sort_values(["snapshot_date", "ticker"], ignore_index=True)


def _execution_anchor(settings: Settings) -> pd.Timestamp:
    month = (settings.execution_quarter - 1) * 3 + 1
    return pd.Timestamp(year=settings.execution_year, month=month, day=1) + pd.Timedelta(
        days=settings.execution_lag_days
    )


def _run_id(settings: Settings, feature_path: Path, target_path: Path) -> str:
    config = {
        "anchor": [settings.execution_year, settings.execution_quarter, settings.execution_lag_days],
        "lookback_years": settings.train_lookback_years,
        "target_horizon_months": settings.target_horizon_months,
        "ridge_alpha": settings.ridge_alpha,
        "meta_ic_lookback_quarters": settings.meta_ic_lookback_quarters,
        "features": sha256_file(feature_path),
        "targets": sha256_file(target_path),
    }
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return f"ridge-{hashlib.sha256(encoded).hexdigest()[:12]}"


def _manifest(
    settings: Settings, feature_path: Path, target_path: Path, predictions: pd.DataFrame, run_dir: Path
) -> dict:
    import sklearn

    return {
        "run_scope": settings.run_scope,
        "run_dir": str(run_dir),
        "inputs": {"features": sha256_file(feature_path), "targets": sha256_file(target_path)},
        "config": {
            "execution_year": settings.execution_year,
            "execution_quarter": settings.execution_quarter,
            "execution_lag_days": settings.execution_lag_days,
            "train_lookback_years": settings.train_lookback_years,
            "target_horizon_months": settings.target_horizon_months,
            "ridge_alpha": settings.ridge_alpha,
            "meta_ic_lookback_quarters": settings.meta_ic_lookback_quarters,
            "missing_policy": "median_train_only_with_indicator",
        },
        "versions": {"scikit_learn": sklearn.__version__},
        "predictions": len(predictions),
    }


def _validate_inputs(features: pd.DataFrame, targets: pd.DataFrame) -> None:
    required_features = {"ticker", "snapshot_date", "review_type", "is_price_fresh", *sum(AGENT_FEATURES.values(), [])}
    required_targets = {"ticker", "snapshot_date", "label_end_date", "target_available", "forward_excess_return_3m"}
    if missing := required_features - set(features.columns):
        raise ValueError(f"features_point_in_time.parquet no contiene: {sorted(missing)}")
    if missing := required_targets - set(targets.columns):
        raise ValueError(f"targets_forward_3m.parquet no contiene: {sorted(missing)}")
