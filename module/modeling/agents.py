"""Agentes especializados LightGBM con entrenamiento walk-forward.

Cada agente (calidad, momentum, valor) aprende, en cada retrain, a ordenar las acciones por su
retorno futuro. LightGBM (arboles con gradient boosting) captura interacciones no lineales entre
factores y maneja los NA de forma nativa. El objetivo (`objective`) es configurable:

- `rank_regression` (principal): regresion sobre el percentil transversal del retorno, alineada
  con el rank-IC.
- `ranking`: LGBMRanker (lambdarank) agrupado por snapshot; optimiza el orden directamente.
- `quartile`: clasifica cuartil superior vs inferior; el score es la probabilidad del superior.

El score de cada agente se combina en un meta-score (ver `module.meta`), que es lo que la cartera
opera y sobre lo que se mide el rank-IC final.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor

from environment import MIN_TRAINING_ROWS, RECENCY_HALFLIFE_YEARS, Settings
from module.modeling.meta import combine_agent_scores
from module.modeling.catalog import AGENT_NAMES, catalog_by_agent
from module.common.utils import read_parquet, sha256_file, write_json, write_parquet

log = logging.getLogger(__name__)

AGENT_FEATURES = {
    "quality": ["factor_roe", "factor_roic", "factor_net_margin", "factor_operating_margin",
                "factor_gross_margin", "factor_fcf_margin", "factor_debt_equity", "factor_current_ratio"],
    "momentum": ["factor_relative_return_3m", "factor_relative_return_6m", "factor_relative_return_12m"],
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

    predictions, coefficients, local_attribution = _walk_forward_scores(frame, settings)
    if predictions.empty:
        raise RuntimeError(
            "No se pudieron entrenar agentes con la historia disponible. "
            "Revisa la cobertura de features, etiquetas y el mínimo de filas de entrenamiento."
        )
    regime_bull_by_date = _regime_bull_by_date(output_dir) if settings.meta_type == "regime" else None
    meta_scores, weights, diagnostics = combine_agent_scores(
        predictions, targets, settings, regime_bull_by_date=regime_bull_by_date)
    wide = _wide_scores(predictions, meta_scores)

    write_parquet(wide, run_dir / "agent_scores.parquet")
    write_parquet(weights, run_dir / "meta_weights.parquet")
    write_parquet(diagnostics, run_dir / "rank_ic_diagnostics.parquet")
    write_parquet(coefficients, run_dir / "model_feature_attribution.parquet")
    write_parquet(local_attribution, run_dir / "agent_local_attribution.parquet")
    feature_diagnostics = _feature_diagnostics(frame, targets, coefficients, settings)
    write_parquet(feature_diagnostics, run_dir / "feature_diagnostics.parquet")
    write_json(_feature_catalog_payload(settings, feature_diagnostics), run_dir / "feature_catalog.json")
    write_json(_manifest(settings, feature_path, target_path, predictions, run_dir), run_dir / "manifest.json")
    log.info("Agentes: rows=%s runs=%s output=%s", len(wide), wide["snapshot_date"].nunique(), run_dir)
    return wide


def _feature_diagnostics(frame: pd.DataFrame, targets: pd.DataFrame, importance: pd.DataFrame,
                         settings: Settings) -> pd.DataFrame:
    """Diagnóstico causal disponible para cada factor.

    El rank-IC univariante se reporta como exploración; las importancias de modelo proceden de
    entrenamientos walk-forward y por tanto no mezclan una etiqueta futura en la predicción.
    """
    merged = frame.merge(targets[["ticker", "snapshot_date", "forward_excess_return_3m", "target_available"]],
                         on=["ticker", "snapshot_date"], how="left", suffixes=("", "_target"))
    rows: list[dict] = []
    for column in sorted(c for c in frame.columns if c.startswith("factor_")):
        usable = merged.loc[merged["target_available"].fillna(False), [column, "forward_excess_return_3m"]].dropna()
        rank_ic = _safe_spearman(usable[column], usable["forward_excess_return_3m"]) if len(usable) >= settings.min_rank_ic_cross_section else np.nan
        relevant = importance.loc[importance["feature"] == column] if not importance.empty else pd.DataFrame()
        rows.append({
            "feature": column,
            "coverage": float(frame[column].notna().mean()),
            "univariate_rank_ic": None if pd.isna(rank_ic) else float(rank_ic),
            "model_importance_mean": float(relevant["coefficient"].mean()) if not relevant.empty else 0.0,
            "model_importance_positive_fraction": float((relevant["coefficient"] > 0).mean()) if not relevant.empty else 0.0,
            "selection_status": "diagnostic_only" if settings.feature_weighting_mode != "oos_stability_prune" else "candidate",
        })
    return pd.DataFrame(rows)


def _feature_catalog_payload(settings: Settings, diagnostics: pd.DataFrame) -> dict:
    from module.modeling.catalog import FEATURE_CATALOG
    diagnostics_by_feature = diagnostics.set_index("feature").to_dict("index") if not diagnostics.empty else {}
    return {
        "enabled_blocks": list(settings.enabled_feature_blocks),
        "enabled_agents": list(settings.enabled_agents),
        "features": [{
            "name": spec.name, "block": spec.block, "agents": list(spec.agents),
            "direction": spec.direction, "source": spec.source, "diagnostics": diagnostics_by_feature.get(spec.name, {}),
        } for spec in FEATURE_CATALOG],
    }


def _regime_bull_by_date(output_dir: Path) -> dict[str, bool]:
    """Régimen bull/bear por snapshot: True si el benchmark subió a 12m hasta esa fecha.

    Point-in-time: usa `price_return_12m` del benchmark, que solo mira el pasado. Lo consume el
    meta_type="regime" para inclinar los pesos de los agentes. Si falta el benchmark, devuelve
    vacío y el modo regime recae en rank_ic.
    """
    path = output_dir / "benchmark_point_in_time.parquet"
    if not path.exists():
        return {}
    bench = pd.read_parquet(path)
    ret12 = pd.to_numeric(bench["price_return_12m"], errors="coerce")
    return {str(date): bool(value > 0) for date, value in zip(bench["snapshot_date"], ret12) if pd.notna(value)}


def _agent_features(frame: pd.DataFrame, settings: Settings) -> dict[str, list[str]]:
    """Features por agente. Con B3 activo, quality recibe la tendencia de fundamentales y value
    la descomposicion precio/fundamental — pero solo las columnas que existan en el frame.
    """
    configured = catalog_by_agent(tuple(settings.enabled_feature_blocks), tuple(settings.enabled_agents))
    # Conserva las tres familias históricas aunque un catálogo personalizado no las enumere.
    # El catálogo es autoritativo: conservar siempre la lista histórica haría que
    # una ablación de quality_core mantuviese ROE/ROIC y no midiese su aporte real.
    # El catálogo completo por defecto ya incluye todos los factores históricos.
    features = {agent: list(dict.fromkeys(configured.get(agent, []))) for agent in settings.enabled_agents}

    def add(agent: str, factor_columns) -> None:
        if agent in features:
            features[agent] += [c for c in factor_columns if c in frame.columns]

    if settings.fundamental_momentum:
        from module.modeling.features import MOMENTUM_FACTORS_QUALITY, MOMENTUM_FACTORS_VALUE
        add("quality", MOMENTUM_FACTORS_QUALITY)
        add("value", MOMENTUM_FACTORS_VALUE)
    if settings.market_regime_feature:
        from module.modeling.features import REGIME_INTERACTION_FACTORS
        add("momentum", REGIME_INTERACTION_FACTORS)
    # Artefactos nuevos: cada bloque alimenta a su agente natural (por su factor_<source>).
    if settings.price_momentum_multi:
        from module.modeling.artifacts import PRICE_MOMENTUM_SOURCES
        add("momentum", [f"factor_{s}" for s in PRICE_MOMENTUM_SOURCES])
    if settings.moving_averages:
        from module.modeling.artifacts import MOVING_AVERAGE_SOURCES
        add("momentum", [f"factor_{s}" for s in MOVING_AVERAGE_SOURCES])
    if settings.regime_extended:
        from module.modeling.artifacts import REGIME_EXTENDED_SOURCES
        add("momentum", [f"factor_{s}" for s in REGIME_EXTENDED_SOURCES])
    if settings.quality_growth_derived:
        from module.modeling.artifacts import QUALITY_GROWTH_SOURCES
        add("quality", [f"factor_{s}" for s in QUALITY_GROWTH_SOURCES])
    # LightGBM no acepta una matriz sin columnas; el filtro además permite fixtures antiguos.
    return {agent: [column for column in columns if column in frame.columns]
            for agent, columns in features.items() if any(column in frame.columns for column in columns)}


def _walk_forward_scores(
    frame: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    agent_features = _agent_features(frame, settings)
    anchor = _execution_anchor(settings)
    retrain_dates = sorted(
        frame.loc[
            frame["is_quarterly"] & frame["snapshot_ts"].ge(anchor), "snapshot_ts"
        ].drop_duplicates()
    )
    rows: list[dict] = []
    model_history: list[dict] = []
    coefficients: list[dict] = []
    local_attribution: list[dict] = []
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
        # `train`/`target`/`weights` no dependen del agente (solo de retrain_date): se preparan una
        # vez por fecha, no una vez por (fecha × agente × familia). Resultado idéntico, menos trabajo.
        train = training.dropna(subset=["forward_excess_return_3m"])
        train, target = _prepare_training(train, settings)
        recency = _recency_weights(train, settings)
        for agent, columns in agent_features.items():
            columns = _selected_feature_columns(train, columns, settings, agent=agent)
            if len(train) < MIN_TRAINING_ROWS:
                log.warning("[%s %s] filas insuficientes: %s", agent, retrain_date.date(), len(train))
                continue
            if scoring.empty:
                continue
            model_scores: list[np.ndarray] = []
            families: list[str] = []
            fitted: list[tuple[str, object]] = []
            for family, model in _build_models(settings):
                _fit_model(model, train, columns, target, settings, family, weights=recency)
                model_scores.append(_score(model, scoring[columns], settings, family))
                families.append(family)
                fitted.append((family, model))
            if not model_scores:
                continue
            score_frame = pd.DataFrame(np.column_stack(model_scores))
            if settings.intra_agent_ensemble_mode == "single":
                values = score_frame.iloc[:, 0].to_numpy()
            else:
                # Promediar rangos conserva la semántica transversal del meta-agente y hace
                # comparables scores de árboles y modelos lineales.
                ranks = score_frame.rank(method="average", pct=True)
                if settings.intra_agent_ensemble_mode == "rank_ic_weighted":
                    family_weights = _model_rank_ic_weights(model_history, frame, agent, families, retrain_date, settings)
                    values = ranks.mul([family_weights[family] for family in families], axis=1).sum(axis=1).to_numpy()
                else:
                    values = ranks.mean(axis=1).to_numpy()
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
            for family, model in fitted:
                coefficients.extend(_importance_rows(model, agent, retrain_date, len(train), settings, family))
                if family == "lightgbm":
                    local_attribution.extend(_local_contribution_rows(model, scoring, columns, agent, retrain_date))
            for family, family_scores in zip(families, model_scores, strict=True):
                for row, score in zip(scoring.itertuples(index=False), family_scores, strict=True):
                    model_history.append({"ticker": row.ticker, "snapshot_date": row.snapshot_date,
                                          "agent": agent, "family": family, "score": float(score)})
    return pd.DataFrame(rows), pd.DataFrame(coefficients), pd.DataFrame(local_attribution)


def _model_rank_ic_weights(
    history_rows: list[dict], frame: pd.DataFrame, agent: str, families: list[str],
    retrain_date: pd.Timestamp, settings: Settings,
) -> dict[str, float]:
    """Causal model-family weights from prior, already labelled OOS predictions."""
    equal = {family: 1 / len(families) for family in families}
    if len(families) < 2 or not history_rows:
        return equal
    history = pd.DataFrame(history_rows)
    history = history.loc[history["agent"].eq(agent)]
    if history.empty:
        return equal
    labels = frame[["ticker", "snapshot_date", "label_end_ts", "target_available", "forward_excess_return_3m"]].drop_duplicates()
    history = history.merge(labels, on=["ticker", "snapshot_date"], how="left")
    history = history.loc[history["target_available"].fillna(False) & history["label_end_ts"].le(retrain_date)]
    positive: dict[str, float] = {}
    for family in families:
        values = []
        family_history = history.loc[history["family"].eq(family)]
        for _, cohort in family_history.groupby("snapshot_date", sort=True):
            clean = cohort[["score", "forward_excess_return_3m"]].dropna()
            if len(clean) >= settings.min_rank_ic_cross_section:
                ic = _safe_spearman(clean["score"], clean["forward_excess_return_3m"])
                if pd.notna(ic):
                    values.append(float(ic))
        recent = values[-settings.meta_ic_lookback_quarters:]
        positive[family] = max(float(np.mean(recent)), 0.0) if recent else 0.0
    total = sum(positive.values())
    return {family: positive[family] / total for family in families} if total > 0 else equal


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    """Spearman without calling scipy for a degenerate cross-section."""
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))


def _selected_feature_columns(train: pd.DataFrame, columns: list[str], settings: Settings,
                              agent: str = "") -> list[str]:
    """Poda causal: usa solo etiquetas ya cerradas presentes en ``train``.

    No sustituye el peso no lineal del modelo; elimina factores sin cobertura o sin evidencia
    univariante estable cuando el usuario activa explícitamente ``oos_stability_prune``.
    """
    usable = [column for column in columns if column in train.columns]
    if settings.feature_weighting_mode not in ("oos_stability_prune", "block_gated"):
        return usable
    candidates: list[tuple[str, float]] = []
    for column in usable:
        coverage = float(train[column].notna().mean())
        if coverage < settings.feature_selection_min_coverage:
            continue
        cohort_ics = []
        for _, cohort in train[["snapshot_date", column, "forward_excess_return_3m"]].groupby("snapshot_date"):
            clean = cohort[[column, "forward_excess_return_3m"]].dropna()
            if len(clean) >= settings.min_rank_ic_cross_section:
                value = _safe_spearman(clean[column], clean["forward_excess_return_3m"])
                if pd.notna(value):
                    cohort_ics.append(float(value))
        recent = cohort_ics[-settings.feature_selection_lookback_quarters:]
        if not recent:
            continue
        positive = float(np.mean(np.asarray(recent) > 0))
        mean = float(np.mean(recent))
        if positive >= settings.feature_selection_min_positive_fraction and mean > 0:
            candidates.append((column, mean))
    if settings.feature_selection_min_permutation_importance > 0 and candidates:
        permutation = _temporal_permutation_importance(train, [name for name, _ in candidates], settings)
        candidates = [(name, score) for name, score in candidates
                      if permutation.get(name, float("-inf")) >= settings.feature_selection_min_permutation_importance]
    if settings.feature_weighting_mode == "block_gated":
        from module.modeling.catalog import FEATURE_CATALOG
        by_feature = {spec.name: spec.block for spec in FEATURE_CATALOG}
        block_scores: dict[str, list[float]] = {}
        for name, score in candidates:
            block_scores.setdefault(by_feature.get(name, "uncatalogued"), []).append(score)
        enabled_blocks = {block for block, scores in block_scores.items() if float(np.mean(scores)) > 0}
        candidates = [(name, score) for name, score in candidates if by_feature.get(name, "uncatalogued") in enabled_blocks]
    candidates.sort(key=lambda item: item[1], reverse=True)
    maximum = settings.feature_selection_max_features_per_agent
    chosen = [name for name, _ in candidates[:maximum or None]]
    # No se permite una matriz vacía: fallback transparente al catálogo completo.
    return chosen or usable


def _temporal_permutation_importance(train: pd.DataFrame, columns: list[str], settings: Settings) -> dict[str, float]:
    """Permutation importance on rolling, strictly earlier-training validation cohorts.

    For every closed validation snapshot a Ridge model is fitted on earlier snapshots only;
    one feature is then shuffled inside that snapshot and the Rank-IC degradation is measured.
    This is deliberately invoked only when the threshold is positive, because it is expensive.
    """
    if not columns:
        return {}
    dates = sorted(pd.to_datetime(train["snapshot_date"]).dropna().unique())[-settings.feature_selection_lookback_quarters:]
    drops: dict[str, list[float]] = {column: [] for column in columns}
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {}
    rng = np.random.default_rng(settings.random_seed)
    for date in dates:
        validation = train.loc[pd.to_datetime(train["snapshot_date"]).eq(date)]
        fitting = train.loc[pd.to_datetime(train["snapshot_date"]).lt(date)]
        if len(fitting) < MIN_TRAINING_ROWS or len(validation) < settings.min_rank_ic_cross_section:
            continue
        target = fitting["forward_excess_return_3m"].groupby(fitting["snapshot_date"]).rank(method="average", pct=True)
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=8.0))
        model.fit(fitting[columns], target)
        base = _safe_spearman(pd.Series(model.predict(validation[columns]), index=validation.index),
                              validation["forward_excess_return_3m"])
        if pd.isna(base):
            continue
        for column in columns:
            shuffled = validation[columns].copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            altered = _safe_spearman(pd.Series(model.predict(shuffled), index=validation.index),
                                     validation["forward_excess_return_3m"])
            if pd.notna(altered):
                drops[column].append(float(base - altered))
    return {column: float(np.mean(values)) for column, values in drops.items() if values}


def _prepare_training(train: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve (filas de entrenamiento, etiqueta) segun el objetivo.

    - "rank_regression" (principal): etiqueta = percentil transversal del retorno (0..1) dentro de
      cada snapshot. Regresion alineada con el rank-IC. Todas las filas.
    - "ranking": etiqueta = grados enteros de relevancia (deciles) por snapshot, para LGBMRanker.
    - "quartile" (ablacion): clasificacion binaria del cuartil superior (1) vs inferior (0) del
      retorno DENTRO de cada snapshot; el centro se EXCLUYE del entrenamiento pero se puntua entero.
    """
    returns = train["forward_excess_return_3m"].astype(float)
    percentile = returns.groupby(train["snapshot_date"]).rank(method="average", pct=True)
    if settings.objective == "rank_regression":
        return train, percentile
    if settings.objective == "ranking":
        # LGBMRanker necesita etiquetas de relevancia enteras (0..9): deciles por snapshot.
        return train, (percentile * 9).round().astype(int)
    if settings.objective == "quartile":
        top = percentile >= 0.75
        mask = top | (percentile <= 0.25)
        return train.loc[mask], top.loc[mask].astype(int)
    raise ValueError(f"OBJECTIVE desconocido: {settings.objective!r}")


def _build_models(settings: Settings) -> list[tuple[str, object]]:
    """Estimador LightGBM segun el objetivo. Los arboles manejan NA de forma nativa y son
    invariantes a transformaciones monotonas, asi que no necesitan imputer ni scaler."""
    common = dict(
        n_estimators=settings.lgbm_n_estimators,
        max_depth=settings.lgbm_max_depth,
        learning_rate=settings.lgbm_learning_rate,
        min_child_samples=settings.lgbm_min_child_samples,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=settings.random_seed,
        n_jobs=settings.lgbm_n_jobs,
        verbose=-1,
    )
    models: list[tuple[str, object]] = []
    families = list(settings.enabled_model_families)
    # This mode is meaningful even when the user selected only tree families:
    # inject one regularised linear reference into every active agent ensemble.
    if settings.feature_weighting_mode == "regularized_linear_ensemble" and "elastic_net" not in families:
        families.append("elastic_net")
    for family in families:
        if family == "lightgbm":
            if settings.objective == "quartile":
                models.append((family, LGBMClassifier(**common)))
            elif settings.objective == "ranking":
                models.append((family, LGBMRanker(objective="lambdarank", **common)))
            else:
                models.append((family, LGBMRegressor(**common)))
        elif family == "elastic_net":
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import ElasticNet
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            models.append((family, make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                 ElasticNet(alpha=0.03, l1_ratio=0.25,
                                                            random_state=settings.random_seed, max_iter=5000))))
        elif family == "catboost":
            try:
                from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor
            except ImportError:
                log.warning("catboost no está instalado; se omite del ensemble.")
                continue
            # thread_count comparte semántica con lgbm_n_jobs (-1 = todos los núcleos): así el
            # multihilo es coherente entre familias y controlable desde un único setting. Determinista.
            common_catboost = dict(iterations=settings.lgbm_n_estimators, depth=settings.lgbm_max_depth,
                                   learning_rate=settings.lgbm_learning_rate, random_seed=settings.random_seed,
                                   thread_count=settings.lgbm_n_jobs,
                                   verbose=False, allow_writing_files=False)
            if settings.objective == "quartile":
                models.append((family, CatBoostClassifier(loss_function="Logloss", **common_catboost)))
            elif settings.objective == "ranking":
                models.append((family, CatBoostRanker(loss_function="YetiRank", **common_catboost)))
            else:
                models.append((family, CatBoostRegressor(loss_function="RMSE", **common_catboost)))
        else:
            raise ValueError(f"Familia de modelo desconocida: {family}")
    return models


def _recency_weights(train: pd.DataFrame, settings: Settings) -> pd.Series | None:
    """Peso por fila según la antigüedad de su snapshot dentro de la ventana de entrenamiento.

    Da más peso a los años recientes. Devuelve None cuando `recency_weighting == "off"` (todas las
    filas pesan igual, comportamiento por defecto). La antigüedad se mide en años respecto al
    snapshot más reciente de la ventana:
      - "linear":      peso = (span_años + 1) - antigüedad_años  (reciente pesa más, mínimo ~1).
      - "exponential": peso = 0.5 ** (antigüedad_años / half_life).
    """
    if settings.recency_weighting == "off":
        return None
    dates = pd.to_datetime(train["snapshot_date"])
    age_years = (dates.max() - dates) / pd.Timedelta(days=365.25)
    if settings.recency_weighting == "linear":
        span = float(age_years.max())
        weights = (span + 1.0) - age_years
    else:  # exponential
        weights = 0.5 ** (age_years / RECENCY_HALFLIFE_YEARS)
    return weights.clip(lower=1e-6)


def _fit_model(model, train: pd.DataFrame, columns: list[str], target: pd.Series, settings: Settings,
               family: str = "lightgbm", weights: pd.Series | None = None) -> None:
    """Ajusta el modelo. El LGBMRanker necesita `group` = nº de filas por snapshot, y que las
    filas esten agrupadas (contiguas) por snapshot; el resto de objetivos se ajustan directo.
    Con `recency_weighting` activo se pasa `sample_weight` (mayor peso a lo reciente).

    `weights` (peso por recencia) se calcula una vez por retrain_date y se pasa aquí; solo depende
    de `train["snapshot_date"]`, no del agente ni de la familia. Si no se pasa, se calcula aquí
    (mismo resultado) para no romper llamadas externas.
    """
    if weights is None:
        weights = _recency_weights(train, settings)
    if family == "lightgbm" and settings.objective == "ranking":
        order = train.sort_values("snapshot_date").index
        ordered = train.loc[order]
        group = ordered.groupby("snapshot_date", sort=True).size().to_numpy()
        sample_weight = weights.loc[order].to_numpy() if weights is not None else None
        model.fit(ordered[columns], target.loc[order], group=group, sample_weight=sample_weight)
    else:
        sample_weight = weights.to_numpy() if weights is not None else None
        if family == "elastic_net":
            kwargs = {"elasticnet__sample_weight": sample_weight} if sample_weight is not None else {}
            model.fit(train[columns], target, **kwargs)
        elif family == "catboost":
            kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
            if settings.objective == "ranking":
                kwargs["group_id"] = pd.factorize(train["snapshot_date"], sort=True)[0]
            model.fit(train[columns], target, **kwargs)
        else:
            model.fit(train[columns], target, sample_weight=sample_weight)


def _score(model, features: pd.DataFrame, settings: Settings, family: str = "lightgbm") -> np.ndarray:
    """Score por fila. Probabilidad del cuartil superior en clasificacion; el valor en el resto."""
    if family in ("lightgbm", "catboost") and settings.objective == "quartile":
        return model.predict_proba(features)[:, 1]
    return model.predict(features)


def _importance_rows(model, agent: str, retrain_date: pd.Timestamp, rows: int, settings: Settings,
                     family: str = "lightgbm") -> list[dict]:
    """Importancias de features del agente LightGBM (trazabilidad y explicabilidad)."""
    if family == "lightgbm":
        names, values = model.feature_name_, model.feature_importances_
    elif family == "catboost":
        names, values = model.feature_names_, model.feature_importances_
    else:
        # Pipeline SimpleImputer -> StandardScaler -> ElasticNet.
        estimator = model.steps[-1][1]
        names, values = model.feature_names_in_, np.abs(estimator.coef_)
    return [
        {
            "agent": agent,
            "model_family": family,
            "model_retrain_date": retrain_date.date().isoformat(),
            "feature": str(name),
            "coefficient": float(value),
            "training_rows": rows,
        }
        for name, value in zip(names, values, strict=True)
    ]


def _local_contribution_rows(
    model, scoring: pd.DataFrame, columns: list[str], agent: str, retrain_date: pd.Timestamp
) -> list[dict]:
    """Devuelve las cinco contribuciones locales más relevantes por predicción.

    LightGBM entrega contribuciones aditivas mediante ``pred_contrib``. Para clasificadores
    están expresadas en margen log-odds; para regresión/ranking, en unidades del score. Se
    guarda la base por separado para que la UI no confunda una contribución con un score final.
    """
    if scoring.empty:
        return []
    contributions = np.asarray(model.booster_.predict(scoring[columns], pred_contrib=True))
    expected = len(columns) + 1
    if contributions.ndim != 2 or contributions.shape != (len(scoring), expected):
        log.warning("Contribuciones locales no disponibles para %s (%s).", agent, contributions.shape)
        return []
    rows: list[dict] = []
    for observation, vector in zip(scoring.itertuples(index=False), contributions, strict=True):
        base_value = float(vector[-1])
        strongest = sorted(range(len(columns)), key=lambda index: abs(float(vector[index])), reverse=True)[:5]
        for position, index in enumerate(strongest, start=1):
            contribution = float(vector[index])
            feature = columns[index]
            value = getattr(observation, feature)
            rows.append(
                {
                    "ticker": observation.ticker,
                    "snapshot_date": observation.snapshot_date,
                    "model_retrain_date": retrain_date.date().isoformat(),
                    "agent": agent,
                    "feature": feature,
                    "factor_value": None if pd.isna(value) else float(value),
                    "local_contribution": contribution,
                    "direction": "positive" if contribution > 0 else "negative" if contribution < 0 else "neutral",
                    "importance_rank": position,
                    "base_value": base_value,
                }
            )
    return rows


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
        "meta_ic_lookback_quarters": settings.meta_ic_lookback_quarters,
        "min_training_rows": MIN_TRAINING_ROWS,
        "min_rank_ic_cross_section": settings.min_rank_ic_cross_section,
        "fundamental_momentum": settings.fundamental_momentum,
        "market_regime_feature": settings.market_regime_feature,
        "objective": settings.objective,
        "lgbm": [settings.lgbm_n_estimators, settings.lgbm_max_depth,
                 settings.lgbm_learning_rate, settings.lgbm_min_child_samples],
        "random_seed": settings.random_seed,
        "meta_type": settings.meta_type,
        "laboratory": {
            "enabled_feature_blocks": settings.enabled_feature_blocks,
            "enabled_agents": settings.enabled_agents,
            "enabled_model_families": settings.enabled_model_families,
            "intra_agent_ensemble_mode": settings.intra_agent_ensemble_mode,
            "feature_weighting_mode": settings.feature_weighting_mode,
            "selection": [settings.feature_selection_min_coverage, settings.feature_selection_lookback_quarters,
                          settings.feature_selection_min_permutation_importance,
                          settings.feature_selection_min_positive_fraction,
                          settings.feature_selection_max_features_per_agent],
        },
        "features": sha256_file(feature_path),
        "targets": sha256_file(target_path),
    }
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return f"lgbm-{hashlib.sha256(encoded).hexdigest()[:12]}"


def _manifest(
    settings: Settings, feature_path: Path, target_path: Path, predictions: pd.DataFrame, run_dir: Path
) -> dict:
    import lightgbm

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
            "objective": settings.objective,
            "lgbm_n_estimators": settings.lgbm_n_estimators,
            "lgbm_max_depth": settings.lgbm_max_depth,
            "lgbm_learning_rate": settings.lgbm_learning_rate,
            "lgbm_min_child_samples": settings.lgbm_min_child_samples,
            "random_seed": settings.random_seed,
            "meta_type": settings.meta_type,
            "meta_ic_lookback_quarters": settings.meta_ic_lookback_quarters,
            "fundamental_momentum": settings.fundamental_momentum,
            "market_regime_feature": settings.market_regime_feature,
            "missing_policy": "median_train_only_with_indicator",
        },
        "versions": {"lightgbm": lightgbm.__version__},
        "predictions": len(predictions),
    }


def _validate_inputs(features: pd.DataFrame, targets: pd.DataFrame) -> None:
    required_features = {"ticker", "snapshot_date", "review_type", "is_price_fresh", *sum(AGENT_FEATURES.values(), [])}
    required_targets = {"ticker", "snapshot_date", "label_end_date", "target_available", "forward_excess_return_3m"}
    if missing := required_features - set(features.columns):
        raise ValueError(f"features_point_in_time.parquet no contiene: {sorted(missing)}")
    if missing := required_targets - set(targets.columns):
        raise ValueError(f"targets_forward_3m.parquet no contiene: {sorted(missing)}")
