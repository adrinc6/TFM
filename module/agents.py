"""Agentes especializados (Ridge o LightGBM) con entrenamiento walk-forward trimestral.

Cada agente aprende, en cada retrain, a ordenar las acciones por su retorno futuro. El motor
(`model_type`) y el objetivo (`objective`) son configurables:

- `model_type="ridge"`: regresion lineal (el enfoque original).
- `model_type="lightgbm"`: arboles con gradient boosting, captura interacciones no lineales.
- `objective="regression"`: predice el exceso de retorno (tratado segun `label_transform`).
- `objective="quartile"`: clasifica el cuartil superior vs el inferior del retorno futuro (los
  del medio se excluyen del entrenamiento). El score es la probabilidad del cuartil superior,
  un ranking valido que la cartera y el rank-IC consumen igual que un score de regresion.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
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
    write_parquet(coefficients, run_dir / "model_feature_attribution.parquet")
    write_json(_manifest(settings, feature_path, target_path, predictions, run_dir), run_dir / "manifest.json")
    log.info("Agentes: rows=%s runs=%s output=%s", len(wide), wide["snapshot_date"].nunique(), run_dir)
    return wide


def _agent_features(frame: pd.DataFrame, settings: Settings) -> dict[str, list[str]]:
    """Features por agente. Con B3 activo, quality recibe la tendencia de fundamentales y value
    la descomposicion precio/fundamental — pero solo las columnas que existan en el frame.
    """
    features = {agent: list(columns) for agent, columns in AGENT_FEATURES.items()}
    if settings.fundamental_momentum:
        from module.features import MOMENTUM_FACTORS_QUALITY, MOMENTUM_FACTORS_VALUE
        features["quality"] += [c for c in MOMENTUM_FACTORS_QUALITY if c in frame.columns]
        features["value"] += [c for c in MOMENTUM_FACTORS_VALUE if c in frame.columns]
    if settings.market_regime_feature:
        from module.features import REGIME_INTERACTION_FACTORS
        # El regimen afecta sobre todo a momentum; sus interacciones van a ese agente.
        features["momentum"] += [c for c in REGIME_INTERACTION_FACTORS if c in frame.columns]
    return features


def _walk_forward_scores(frame: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    agent_features = _agent_features(frame, settings)
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
        for agent, columns in agent_features.items():
            train = training.dropna(subset=["forward_excess_return_3m"])
            train, target = _prepare_training(train, settings)
            if len(train) < settings.min_training_rows:
                log.warning("[%s %s] filas insuficientes: %s", agent, retrain_date.date(), len(train))
                continue
            model = _build_model(settings)
            _fit_model(model, train, columns, target, settings)
            if scoring.empty:
                continue
            values = _score(model, scoring[columns], settings)
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
            coefficients.extend(_importance_rows(model, agent, retrain_date, len(train), settings))
    return pd.DataFrame(rows), pd.DataFrame(coefficients)


def _prepare_training(train: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve (filas de entrenamiento, etiqueta) segun el objetivo.

    - "rank_regression" (principal): etiqueta = percentil transversal del retorno (0..1) dentro de
      cada snapshot. Regresion normal, pero alineada con el rank-IC. Todas las filas.
    - "ranking": etiqueta = grados enteros de relevancia por snapshot (para LGBMRanker). Todas.
    - "regression": exceso de retorno crudo tratado por `label_transform`. Todas.
    - "quartile" (ablacion): clasificacion binaria del cuartil superior (1) vs inferior (0) del
      retorno DENTRO de cada snapshot; el centro se EXCLUYE del entrenamiento pero se puntua entero.
    """
    returns = train["forward_excess_return_3m"].astype(float)
    if settings.objective == "rank_regression":
        return train, returns.groupby(train["snapshot_date"]).rank(method="average", pct=True)
    if settings.objective == "ranking":
        # LGBMRanker necesita etiquetas de relevancia enteras (0..K). Usamos deciles por snapshot.
        deciles = returns.groupby(train["snapshot_date"]).rank(method="average", pct=True)
        return train, (deciles * 9).round().astype(int)
    if settings.objective == "quartile":
        quartile = returns.groupby(train["snapshot_date"]).rank(method="average", pct=True)
        top = quartile >= 0.75
        bottom = quartile <= 0.25
        mask = top | bottom
        return train.loc[mask], top.loc[mask].astype(int)
    return train, _transform_label(train, "forward_excess_return_3m", settings)


def _transform_label(train: pd.DataFrame, column: str, settings: Settings) -> pd.Series:
    """Tratamiento de la etiqueta de REGRESION (B2). Solo sobre `train`, nunca scoring.

    - "none": exceso de retorno crudo.
    - "winsor": recorta las colas al percentil `label_winsor_pct` (los outliers no dominan).
    - "rank": percentil transversal del retorno futuro. Entrena contra el orden, que es lo que
      mide el rank-IC.
    """
    values = train[column].astype(float)
    transform = settings.label_transform
    if transform == "winsor":
        lower = values.quantile(settings.label_winsor_pct)
        upper = values.quantile(1 - settings.label_winsor_pct)
        return values.clip(lower, upper)
    if transform == "rank":
        return values.groupby(train["snapshot_date"]).rank(method="average", pct=True)
    return values


def _build_model(settings: Settings) -> Pipeline:
    """Pipeline del agente segun motor y objetivo.

    LightGBM no necesita escalado (los arboles son invariantes a transformaciones monotonas) y
    maneja los NA de forma nativa, asi que va sin imputer ni scaler. Ridge si los necesita.
    """
    if settings.model_type == "lightgbm":
        common = dict(
            n_estimators=settings.lgbm_n_estimators,
            max_depth=settings.lgbm_max_depth,
            learning_rate=settings.lgbm_learning_rate,
            min_child_samples=settings.lgbm_min_child_samples,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=settings.random_seed,
            n_jobs=1,
            verbose=-1,
        )
        if settings.objective == "quartile":
            estimator = LGBMClassifier(**common)
        elif settings.objective == "ranking":
            estimator = LGBMRanker(objective="lambdarank", **common)
        else:
            estimator = LGBMRegressor(**common)
        return Pipeline([("model", estimator)])
    # Ridge (lineal): imputa, escala y regresa. La clasificacion con Ridge no aplica.
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=settings.ridge_alpha)),
        ]
    )


def _fit_model(
    model: Pipeline, train: pd.DataFrame, columns: list[str], target: pd.Series, settings: Settings
) -> None:
    """Ajusta el modelo. El LGBMRanker necesita `group` = nº de filas por snapshot, y que las
    filas esten agrupadas (contiguas) por snapshot; el resto de modelos se ajustan directo.
    """
    if settings.objective == "ranking":
        order = train.sort_values("snapshot_date").index
        ordered = train.loc[order]
        group = ordered.groupby("snapshot_date", sort=True).size().to_numpy()
        model.named_steps["model"].fit(ordered[columns], target.loc[order], group=group)
    else:
        model.fit(train[columns], target)


def _score(model: Pipeline, features: pd.DataFrame, settings: Settings) -> np.ndarray:
    """Score por fila. Probabilidad del cuartil superior en clasificacion; el valor en el resto."""
    if settings.objective == "quartile" and settings.model_type == "lightgbm":
        return model.predict_proba(features)[:, 1]
    return model.predict(features)


def _importance_rows(
    model: Pipeline, agent: str, retrain_date: pd.Timestamp, rows: int, settings: Settings
) -> list[dict]:
    """Trazabilidad del modelo: coeficientes (Ridge) o importancias de features (LightGBM)."""
    if settings.model_type == "lightgbm":
        estimator = model.named_steps["model"]
        names = estimator.feature_name_
        values = estimator.feature_importances_
    else:
        names = model[:-1].get_feature_names_out()
        values = model.named_steps["ridge"].coef_
    return [
        {
            "agent": agent,
            "model_retrain_date": retrain_date.date().isoformat(),
            "feature": str(name),
            "coefficient": float(value),
            "training_rows": rows,
        }
        for name, value in zip(names, values, strict=True)
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
        "min_training_rows": settings.min_training_rows,
        "min_rank_ic_cross_section": settings.min_rank_ic_cross_section,
        "label_transform": settings.label_transform,
        "label_winsor_pct": settings.label_winsor_pct,
        "fundamental_momentum": settings.fundamental_momentum,
        "market_regime_feature": settings.market_regime_feature,
        "model_type": settings.model_type,
        "objective": settings.objective,
        "lgbm": [settings.lgbm_n_estimators, settings.lgbm_max_depth,
                 settings.lgbm_learning_rate, settings.lgbm_min_child_samples],
        "random_seed": settings.random_seed,
        "meta_type": settings.meta_type,
        "features": sha256_file(feature_path),
        "targets": sha256_file(target_path),
    }
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return f"{settings.model_type}-{hashlib.sha256(encoded).hexdigest()[:12]}"


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
            "model_type": settings.model_type,
            "objective": settings.objective,
            "lgbm_n_estimators": settings.lgbm_n_estimators,
            "lgbm_max_depth": settings.lgbm_max_depth,
            "lgbm_learning_rate": settings.lgbm_learning_rate,
            "lgbm_min_child_samples": settings.lgbm_min_child_samples,
            "random_seed": settings.random_seed,
            "meta_type": settings.meta_type,
            "ridge_alpha": settings.ridge_alpha,
            "meta_ic_lookback_quarters": settings.meta_ic_lookback_quarters,
            "label_transform": settings.label_transform,
            "label_winsor_pct": settings.label_winsor_pct,
            "fundamental_momentum": settings.fundamental_momentum,
            "market_regime_feature": settings.market_regime_feature,
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
