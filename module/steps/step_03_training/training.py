"""Training workflows for base agents and the meta learner."""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from environment import (
    BEAR_HARD_THRESHOLD,
    SCORE_DISPERSION_MIN_SCALE,
    OOF_N_SPLITS,
    SCORE_DISPERSION_MIN_STD,
    SECTOR_CONFIDENCE_PEERS,
    SECTOR_SCORE_PRIOR_BASE,
    SECTOR_SCORE_PRIOR_WEIGHT,
    EXPORT_FEATURE_USAGE_REPORT,
    FUNDAMENTAL_FEATURE_COLUMNS, FUNDAMENTAL_FEATURE_EXCLUDE,
    VALUATION_FEATURE_COLUMNS, VALUATION_FEATURE_EXCLUDE,
    MOMENTUM_FEATURE_COLUMNS, MOMENTUM_FEATURE_EXCLUDE,
    BEAR_FEATURE_COLUMNS, BEAR_FEATURE_EXCLUDE,
    SENTIMENT_FEATURE_COLUMNS, SENTIMENT_FEATURE_EXCLUDE,
    SECTOR_ROTATION_FEATURE_COLUMNS, SECTOR_ROTATION_FEATURE_EXCLUDE,
    META_FEATURE_COLUMNS, META_FEATURE_EXCLUDE,
    FUNDAMENTAL_N_ESTIMATORS, FUNDAMENTAL_MAX_DEPTH, FUNDAMENTAL_LEARNING_RATE,
    FUNDAMENTAL_SUBSAMPLE, FUNDAMENTAL_COLSAMPLE, FUNDAMENTAL_MIN_CHILD_WEIGHT,
    VALUATION_N_ESTIMATORS, VALUATION_MAX_DEPTH, VALUATION_LEARNING_RATE, VALUATION_SUBSAMPLE,
    MOMENTUM_N_ESTIMATORS, MOMENTUM_MAX_DEPTH, MOMENTUM_MIN_SAMPLES_LEAF,
    BEAR_N_ESTIMATORS, BEAR_MAX_DEPTH, BEAR_RULE_WEIGHT, BEAR_ML_WEIGHT,
    META_GBM_N_ESTIMATORS, META_GBM_MAX_DEPTH, META_GBM_LEARNING_RATE,
    SECTOR_SPECIALIST_MIN_SAMPLES,
    FEATURE_IMPORTANCE_CUTOFF_FRACTION, FEATURE_IMPORTANCE_MIN_KEEP, FEATURE_IMPORTANCE_MAX_KEEP,
    FEATURE_SELECTOR_RELEVANCE_WEIGHT, FEATURE_SELECTOR_RF_N_ESTIMATORS, FEATURE_SELECTOR_RF_MAX_DEPTH,
    CACHE_SCHEMA_VERSION,
)
from module.common.cache import CacheManager
from module.agents.meta_learner import MetaLearner
from module.steps.step_03_training.agent_config import build_agents_config, build_sector_rotation_agent
from module.steps.step_03_training.oof import generate_oof_scores

log = logging.getLogger(__name__)


def _requested_feature_map() -> Dict[str, Dict[str, list[str]]]:
    return {
        "fundamental": {"include": list(FUNDAMENTAL_FEATURE_COLUMNS), "exclude": list(FUNDAMENTAL_FEATURE_EXCLUDE)},
        "valuation": {"include": list(VALUATION_FEATURE_COLUMNS), "exclude": list(VALUATION_FEATURE_EXCLUDE)},
        "momentum": {"include": list(MOMENTUM_FEATURE_COLUMNS), "exclude": list(MOMENTUM_FEATURE_EXCLUDE)},
        "bear": {"include": list(BEAR_FEATURE_COLUMNS), "exclude": list(BEAR_FEATURE_EXCLUDE)},
        "sentiment": {"include": list(SENTIMENT_FEATURE_COLUMNS), "exclude": list(SENTIMENT_FEATURE_EXCLUDE)},
        "sector_rotation": {"include": list(SECTOR_ROTATION_FEATURE_COLUMNS), "exclude": list(SECTOR_ROTATION_FEATURE_EXCLUDE)},
        "meta_learner": {"include": list(META_FEATURE_COLUMNS), "exclude": list(META_FEATURE_EXCLUDE)},
    }


def _agent_hyperparams_ctx(agent_name: str) -> Dict[str, Any]:
    """Returns a dict of hyperparameters for an agent, used as part of a cache key.

    Captures all environment constants that affect an agent's training outcome.
    Changing any of these values will invalidate that agent's cache.
    """
    _shared_selector = {
        "cutoff": FEATURE_IMPORTANCE_CUTOFF_FRACTION,
        "min_keep": FEATURE_IMPORTANCE_MIN_KEEP,
        "max_keep": FEATURE_IMPORTANCE_MAX_KEEP,
        "rf_n": FEATURE_SELECTOR_RF_N_ESTIMATORS,
        "rf_depth": FEATURE_SELECTOR_RF_MAX_DEPTH,
        "rel_weight": FEATURE_SELECTOR_RELEVANCE_WEIGHT,
        "sector_min_samples": SECTOR_SPECIALIST_MIN_SAMPLES,
    }
    _hp: Dict[str, Dict[str, Any]] = {
        "fundamental": {
            "n_estimators": FUNDAMENTAL_N_ESTIMATORS,
            "max_depth": FUNDAMENTAL_MAX_DEPTH,
            "lr": FUNDAMENTAL_LEARNING_RATE,
            "subsample": FUNDAMENTAL_SUBSAMPLE,
            "colsample": FUNDAMENTAL_COLSAMPLE,
            "min_child_weight": FUNDAMENTAL_MIN_CHILD_WEIGHT,
            **_shared_selector,
        },
        "valuation": {
            "n_estimators": VALUATION_N_ESTIMATORS,
            "max_depth": VALUATION_MAX_DEPTH,
            "lr": VALUATION_LEARNING_RATE,
            "subsample": VALUATION_SUBSAMPLE,
            **_shared_selector,
        },
        "momentum": {
            "n_estimators": MOMENTUM_N_ESTIMATORS,
            "max_depth": MOMENTUM_MAX_DEPTH,
            "min_samples_leaf": MOMENTUM_MIN_SAMPLES_LEAF,
            **_shared_selector,
        },
        "bear": {
            "n_estimators": BEAR_N_ESTIMATORS,
            "max_depth": BEAR_MAX_DEPTH,
            "rule_weight": BEAR_RULE_WEIGHT,
            "ml_weight": BEAR_ML_WEIGHT,
            **_shared_selector,
        },
        # SectorRotationAgent hyperparams are module-private in sector_rotation.py
        # (200 estimators, depth 3, lr 0.05) — hardcoded here to match.
        "sector_rotation": {
            "n_estimators": 200,
            "max_depth": 3,
            "lr": 0.05,
        },
        "meta_learner": {
            "n_estimators": META_GBM_N_ESTIMATORS,
            "max_depth": META_GBM_MAX_DEPTH,
            "lr": META_GBM_LEARNING_RATE,
        },
    }
    return _hp.get(agent_name, {})


def _make_fold_caches(
    fold_cache_root: Path,
    train_start_ts: pd.Timestamp,
    train_end_ts: pd.Timestamp,
    random_seed: int,
) -> Dict[str, Any]:
    """Create all CacheManager instances needed for one fold.

    Returns a dict with keys:
      - "agents": dict[agent_name → CacheManager]  (independent per agent)
      - "oof":    CacheManager  (depends on all base agents)
      - "meta":   CacheManager  (depends on OOF + meta config)
      - "test":   CacheManager  (full fold; used to skip prediction entirely)

    Cache granularity means that changing one agent's features or hyperparams
    only invalidates that agent's cache, the OOF cache, the meta cache, and
    the test-scored cache — not the other agent caches.
    """
    feature_map = _requested_feature_map()
    fold_dates = f"{train_start_ts.date()}__{train_end_ts.date()}"
    base_agents = ["fundamental", "valuation", "momentum", "bear", "sector_rotation"]

    agent_caches: Dict[str, CacheManager] = {}
    for ag in base_agents:
        ctx: Dict[str, Any] = {
            "v": CACHE_SCHEMA_VERSION,
            "fold": fold_dates,
            "seed": random_seed,
            "features": feature_map.get(ag, {}),
            "hp": _agent_hyperparams_ctx(ag),
        }
        agent_caches[ag] = CacheManager(fold_cache_root, ctx, namespace=f"agent_{ag}")

    oof_ctx: Dict[str, Any] = {
        "v": CACHE_SCHEMA_VERSION,
        "fold": fold_dates,
        "seed": random_seed,
        "n_splits": OOF_N_SPLITS,
        "agents": {
            ag: {"features": feature_map.get(ag, {}), "hp": _agent_hyperparams_ctx(ag)}
            for ag in base_agents
        },
    }
    oof_cache = CacheManager(fold_cache_root, oof_ctx, namespace="fold_oof")

    meta_ctx: Dict[str, Any] = {
        **oof_ctx,
        "meta_features": feature_map.get("meta_learner", {}),
        "meta_hp": _agent_hyperparams_ctx("meta_learner"),
    }
    meta_cache = CacheManager(fold_cache_root, meta_ctx, namespace="fold_meta")

    test_ctx: Dict[str, Any] = {**meta_ctx}
    test_cache = CacheManager(fold_cache_root, test_ctx, namespace="fold_test")

    return {
        "agents": agent_caches,
        "oof": oof_cache,
        "meta": meta_cache,
        "test": test_cache,
    }


def _export_feature_usage_report(
    *,
    agents: Dict[str, Any],
    df_train: pd.DataFrame,
    fold_id: int | str,
    agents_results_dir: str,
) -> None:
    if not EXPORT_FEATURE_USAGE_REPORT:
        return

    requested = _requested_feature_map()
    records: list[Dict[str, Any]] = []

    for agent_name, cfg in requested.items():
        include = list(cfg.get("include", []))
        exclude = list(cfg.get("exclude", []))
        include_set = set(include)
        available = [c for c in include if c in df_train.columns]
        missing = [c for c in include if c not in df_train.columns]

        # Columnas presentes pero no calculables en este fold (todo NaN)
        not_calculated_or_no_data: list[str] = []
        low_data_coverage: list[str] = []
        for col in available:
            s = pd.to_numeric(df_train[col], errors="coerce")
            non_null = int(s.notna().sum())
            if non_null == 0:
                not_calculated_or_no_data.append(col)
            elif non_null < max(5, int(0.05 * len(s))):
                low_data_coverage.append(col)

        agent_obj = agents.get(agent_name)
        used = list(getattr(agent_obj, "_feature_cols", []) or []) if agent_obj is not None else []
        used_not_requested = [c for c in used if c not in include_set]

        skipped_due_to_data = [c for c in include if c not in used and c in set(not_calculated_or_no_data + low_data_coverage)]

        records.append(
            {
                "fold": str(fold_id),
                "agent": agent_name,
                "requested_n": len(include),
                "available_n": len(available),
                "used_n": len(used),
                "missing_n": len(missing),
                "not_calculated_or_no_data_n": len(not_calculated_or_no_data),
                "low_data_coverage_n": len(low_data_coverage),
                "skipped_due_to_data_n": len(skipped_due_to_data),
                "used_not_requested_n": len(used_not_requested),
                "requested": "|".join(include),
                "exclude": "|".join(exclude),
                "available": "|".join(available),
                "missing": "|".join(missing),
                "not_calculated_or_no_data": "|".join(not_calculated_or_no_data),
                "low_data_coverage": "|".join(low_data_coverage),
                "skipped_due_to_data": "|".join(skipped_due_to_data),
                "used": "|".join(used),
                "used_not_requested": "|".join(used_not_requested),
            }
        )

    out_dir = Path(agents_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "feature_usage_report.csv"
    json_path = out_dir / "feature_usage_report.json"

    pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("[FeatureUsage] Reporte fold %s -> %s", fold_id, csv_path.name)


def _series_stats(s: pd.Series) -> Dict[str, float]:
    v = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return {
            "n": 0,
            "min": float("nan"),
            "q10": float("nan"),
            "q25": float("nan"),
            "mean": float("nan"),
            "q50": float("nan"),
            "q75": float("nan"),
            "q90": float("nan"),
            "max": float("nan"),
            "ge_050_ratio": float("nan"),
            "ge_055_ratio": float("nan"),
            "ge_060_ratio": float("nan"),
        }
    return {
        "n": int(v.shape[0]),
        "min": float(v.min()),
        "q10": float(v.quantile(0.10)),
        "q25": float(v.quantile(0.25)),
        "mean": float(v.mean()),
        "q50": float(v.quantile(0.50)),
        "q75": float(v.quantile(0.75)),
        "q90": float(v.quantile(0.90)),
        "max": float(v.max()),
        "ge_050_ratio": float((v >= 0.50).mean()),
        "ge_055_ratio": float((v >= 0.55).mean()),
        "ge_060_ratio": float((v >= 0.60).mean()),
    }


def _log_score_stats(tag: str, s: pd.Series) -> None:
    st = _series_stats(s)
    if st["n"] == 0:
        log.info(f"[{tag}] No data para resumen de score")
        return
    log.info(
        f"[{tag}] n={st['n']} | min={st['min']:.4f} q10={st['q10']:.4f} q25={st['q25']:.4f} "
        f"mean={st['mean']:.4f} q50={st['q50']:.4f} q75={st['q75']:.4f} q90={st['q90']:.4f} max={st['max']:.4f}"
    )
    log.info(
        f"[{tag}] cobertura umbral: >=0.50={st['ge_050_ratio']:.1%} | >=0.55={st['ge_055_ratio']:.1%} | >=0.60={st['ge_060_ratio']:.1%}"
    )


def _compute_dispersion_scales(df: pd.DataFrame, score_cols: list[str]) -> Dict[str, float]:
    scales: Dict[str, float] = {}
    log.info(f"[Dispersion] SCORE_DISPERSION_MIN_STD={SCORE_DISPERSION_MIN_STD:.4f}")
    log.info(f"[Dispersion] SCORE_DISPERSION_MIN_SCALE={SCORE_DISPERSION_MIN_SCALE:.4f}")
    for col in score_cols:
        if col not in df.columns:
            continue
        std = float(df[col].std()) if SCORE_DISPERSION_MIN_STD > 0 else 0.0
        scale = 1.0 if std >= SCORE_DISPERSION_MIN_STD else (std / SCORE_DISPERSION_MIN_STD)
        bounded = max(0.0, min(1.0, scale))
        # Guardrail: prevents a useful score from becoming completely neutral during inference.
        if bounded < 1.0:
            bounded = max(bounded, float(SCORE_DISPERSION_MIN_SCALE))
        scales[col] = bounded
        log.info(
            f"[Dispersion] {col}: std={std:.6f} -> scale={scales[col]:.4f} "
            f"({'sin shrink' if scales[col] >= 1.0 else 'shrink activo'})"
        )
    return scales


def _apply_dispersion_shrink(df: pd.DataFrame, scales: Dict[str, float]) -> pd.DataFrame:
    for col, scale in scales.items():
        if col not in df.columns or scale >= 1.0:
            continue
        before = df[col].copy()
        df[col] = 0.5 + (df[col] - 0.5) * scale
        _log_score_stats(f"Dispersion/{col}/before", before)
        _log_score_stats(f"Dispersion/{col}/after", df[col])
    return df


def _apply_sector_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    if "sector" not in df.columns or "final_score" not in df.columns:
        return df
    sector_counts = (
        df.reset_index()[["ticker", "sector"]]
        .drop_duplicates(subset="ticker")
        .groupby("sector")["ticker"]
        .nunique()
    )
    df["sector_peer_count"] = df["sector"].map(sector_counts).fillna(0).astype(int)
    if SECTOR_CONFIDENCE_PEERS > 0:
        df["sector_confidence"] = np.sqrt(df["sector_peer_count"] / SECTOR_CONFIDENCE_PEERS).clip(upper=1.0)
    else:
        df["sector_confidence"] = 1.0
    if "sector_score" in df.columns:
        sector_score = df["sector_score"].fillna(0.5)
    else:
        sector_score = 0.5
    # Additive sector tilt: shifts scores up/down based on sector outlook.
    # sector_score > 0.5 → positive tilt (strong sector), < 0.5 → negative tilt.
    # The tilt magnitude is controlled by SECTOR_SCORE_PRIOR_WEIGHT and
    # dampened by sector_confidence (fewer peers → weaker tilt).
    df["final_score_raw"] = df["final_score"]
    sector_tilt = (sector_score - 0.5) * SECTOR_SCORE_PRIOR_WEIGHT * df["sector_confidence"]
    df["final_score"] = (df["final_score"] + sector_tilt).clip(0.0, 1.0)

    log.info(
        f"[SectorAdjust] params: SECTOR_CONFIDENCE_PEERS={SECTOR_CONFIDENCE_PEERS}, "
        f"SECTOR_SCORE_PRIOR_BASE={SECTOR_SCORE_PRIOR_BASE:.3f}, "
        f"SECTOR_SCORE_PRIOR_WEIGHT={SECTOR_SCORE_PRIOR_WEIGHT:.3f}"
    )
    _log_score_stats("SectorAdjust/final_score_raw", df["final_score_raw"])
    _log_score_stats("SectorAdjust/sector_score", sector_score)
    _log_score_stats("SectorAdjust/sector_confidence", df["sector_confidence"])
    _log_score_stats("SectorAdjust/sector_tilt", sector_tilt)
    _log_score_stats("SectorAdjust/final_score_adjusted", df["final_score"])

    if "bear_risk_score" in df.columns:
        bear_risk = pd.to_numeric(df["bear_risk_score"], errors="coerce").fillna(0.5)
        log.info(
            f"[RiskDiag] bear_risk>=hard_threshold({BEAR_HARD_THRESHOLD:.2f}): "
            f"{int((bear_risk >= BEAR_HARD_THRESHOLD).sum())}/{len(bear_risk)} ({(bear_risk >= BEAR_HARD_THRESHOLD).mean():.1%})"
        )

    return df


def _instantiate_base_agents(agents_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        ag_name: cfg["cls"](**cfg["kwargs"])
        for ag_name, cfg in agents_config.items()
    }


def _fit_base_agents(
    agents: Dict[str, Any],
    agents_config: Dict[str, Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    fold: int,
) -> None:
    for ag_name, agent in agents.items():
        cfg = agents_config[ag_name]
        y_fit = (1 - y) if cfg.get("invert_y") else y
        sector_col = cfg.get("sector_col")
        if sector_col:
            agent.fit(X, y_fit, fold=fold, sector_col=sector_col)
        else:
            agent.fit(X, y_fit, fold=fold)


def _predict_base_scores(
    agents: Dict[str, Any],
    agents_config: Dict[str, Dict[str, Any]],
    X: pd.DataFrame,
) -> pd.DataFrame:
    out = X.copy()
    for ag_name, agent in agents.items():
        sector_col = agents_config[ag_name].get("sector_col")
        if not getattr(agent, "is_trained", False):
            scores = pd.Series(0.5, index=out.index)
        elif sector_col:
            scores = agent.predict_score(out, sector_col)
        else:
            scores = agent.predict_score(out)
        # Align score direction for investment: high = better to invest.
        # BearAgent devuelve riesgo [0,1], por eso guardamos ambas vistas:
        #   - bear_risk_score: riesgo (alto = peor)
        #   - bear_score: safety (alto = mejor)
        if ag_name == "bear":
            risk = scores.astype(float).clip(0.0, 1.0)
            out["bear_risk_score"] = risk.values
            out["bear_score"] = (1.0 - risk).values
            _log_score_stats(f"AgentScore/{ag_name}/risk", out["bear_risk_score"])
            _log_score_stats(f"AgentScore/{ag_name}/safety", out["bear_score"])
        else:
            out[f"{ag_name}_score"] = scores.values
            _log_score_stats(f"AgentScore/{ag_name}", out[f"{ag_name}_score"])

    score_cols = [
        c for c in (
            "fundamental_score",
            "valuation_score",
            "momentum_score",
            "bear_score",
            "sentiment_score",
            "sector_score",
        ) if c in out.columns
    ]
    if score_cols:
        ensemble_mean = out[score_cols].mean(axis=1)
        _log_score_stats("AgentScore/ensemble_mean_pre_meta", ensemble_mean)
    return out


def train_fold(
    df_train_norm: pd.DataFrame,
    df_test_norm: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    fold_id: int,
    agent_models_results_dir: str,
    agents_results_dir: str,
    random_seed: int = 42,
    sector_map: Optional[Dict[str, str]] = None,
    spy_prices: Optional[pd.Series] = None,
    fold_cache_root: Optional[Path] = None,
    train_start_ts: Optional[pd.Timestamp] = None,
    train_end_ts: Optional[pd.Timestamp] = None,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    agents_config = build_agents_config(agent_models_results_dir=agent_models_results_dir, random_seed=random_seed)

    # ── Build per-fold cache managers (None if caching is disabled) ──────────
    fc: Optional[Dict[str, Any]] = None
    if fold_cache_root is not None and train_start_ts is not None and train_end_ts is not None:
        try:
            fc = _make_fold_caches(
                fold_cache_root=Path(fold_cache_root),
                train_start_ts=train_start_ts,
                train_end_ts=train_end_ts,
                random_seed=random_seed,
            )
        except Exception as exc:
            log.warning("[Fold %s] Could not create fold caches: %s", fold_id, exc)
            fc = None

    # ── Fast path: entire fold result cached ────────────────────────────────
    if fc is not None:
        cached_test = fc["test"].load_pickle("df_test_scored")
        cached_train_oof = fc["test"].load_pickle("df_train_with_oof")
        if cached_test is not None and cached_train_oof is not None:
            log.info("[Fold %s] Cache hit: fold_test — skipping all training and prediction.", fold_id)
            # Reconstruct trained agents from per-sector caches (cheap: file I/O only).
            base_agents = _instantiate_base_agents(agents_config)
            for ag_name, agent in base_agents.items():
                cfg = agents_config[ag_name]
                y_fit = (1 - y_train) if cfg.get("invert_y") else y_train
                sector_col = cfg.get("sector_col")
                sector_cache_dir = fc["agents"][ag_name].run_dir
                if sector_col:
                    agent.fit(df_train_norm, y_fit, fold=fold_id, sector_col=sector_col,
                              sector_cache_dir=sector_cache_dir)
                else:
                    agent.fit(df_train_norm, y_fit, fold=fold_id,
                              sector_cache_dir=sector_cache_dir)
                base_agents[ag_name] = agent
            sector_agent = build_sector_rotation_agent(
                agent_models_results_dir=agent_models_results_dir, random_seed=random_seed
            )
            sr_cache = fc["agents"].get("sector_rotation")
            if sr_cache is not None:
                cached_sr = sr_cache.load_pickle("model")
                if cached_sr is not None:
                    sector_agent = cached_sr
            meta = MetaLearner(results_dir=agent_models_results_dir, random_seed=random_seed)
            meta_cached = fc["meta"].load_pickle("model")
            if meta_cached is not None:
                meta = meta_cached
            cached_test["label"] = y_test.values
            agents_dict = {**base_agents, "sector_rotation": sector_agent, "meta_learner": meta}
            period_dir = Path(agents_results_dir) / str(fold_id)
            _export_feature_usage_report(
                agents=agents_dict,
                df_train=df_train_norm,
                fold_id=fold_id,
                agents_results_dir=period_dir.as_posix(),
            )
            return agents_dict, cached_test, cached_train_oof

    # ── Step 1: Train base agents (per-sector cache inside fit()) ────────────
    log.info(f"[Fold {fold_id}] 1/3 — Entrenando agentes base con datos de entrenamiento del fold...")
    base_agents = _instantiate_base_agents(agents_config)

    for ag_name, agent in base_agents.items():
        cfg = agents_config[ag_name]
        y_fit = (1 - y_train) if cfg.get("invert_y") else y_train
        sector_col = cfg.get("sector_col")
        # Per-sector cache dir: encodes all agent-level context via the CacheManager key path.
        # Each sector child model is then keyed only by sector name inside that directory.
        sector_cache_dir = fc["agents"][ag_name].run_dir if fc is not None else None
        if sector_col:
            agent.fit(df_train_norm, y_fit, fold=fold_id, sector_col=sector_col,
                      sector_cache_dir=sector_cache_dir)
        else:
            agent.fit(df_train_norm, y_fit, fold=fold_id, sector_cache_dir=sector_cache_dir)
        base_agents[ag_name] = agent
        n_trained = len(getattr(agent, "_sector_agents", {}))
        log.debug("[Fold %s] agent=%s trained_sectors=%d", fold_id, ag_name, n_trained)

    # Entrenar SectorRotationAgent (opera a nivel sector, no ticker)
    sector_agent = build_sector_rotation_agent(agent_models_results_dir=agent_models_results_dir, random_seed=random_seed)
    sr_cache = fc["agents"].get("sector_rotation") if fc is not None else None
    cached_sr = sr_cache.load_pickle("model") if sr_cache is not None else None
    if cached_sr is not None:
        log.info("[Fold %s] Cache hit: agent=sector_rotation — skipping fit.", fold_id)
        sector_agent = cached_sr
    elif sector_map is not None and "forward_return" in df_train_norm.columns:
        log.info(f"[Fold {fold_id}] 1/3 — Entrenando SectorRotationAgent (top-down, nivel sector)...")
        sector_agent.fit(df_train_norm, sector_map=sector_map, spy_prices=spy_prices, fold=fold_id)
        if sr_cache is not None:
            try:
                sr_cache.save_pickle("model", sector_agent)
                log.info("[Fold %s] Cache save: agent=sector_rotation", fold_id)
            except Exception as exc:
                log.warning("[Fold %s] Could not cache sector_rotation: %s", fold_id, exc)
    else:
        log.warning(f"[Fold {fold_id}] SectorRotationAgent: sin sector_map o forward_return — score neutro 0.5")

    # ── Step 2: OOF scores (fold-level cache) ────────────────────────────────
    oof_cache = fc["oof"] if fc is not None else None
    cached_oof_bundle = oof_cache.load_pickle("oof_bundle") if oof_cache is not None else None

    if cached_oof_bundle is not None:
        log.info("[Fold %s] Cache hit: fold_oof — skipping OOF generation.", fold_id)
        df_train_with_oof = cached_oof_bundle["df_train_with_oof"]
        dispersion_scales = cached_oof_bundle["dispersion_scales"]
    else:
        log.info(f"[Fold {fold_id}] 2/3 — Generando scores OOF ({OOF_N_SPLITS} splits temporales) como input del MetaLearner...")
        oof_scores = generate_oof_scores(
            df_train_norm,
            y_train,
            agents_config=agents_config,
            n_splits=OOF_N_SPLITS,
            random_seed=random_seed,
            sector_map=sector_map,
            spy_prices=spy_prices,
        )
        df_train_with_oof = df_train_norm.copy()
        for col_name, scores_series in oof_scores.items():
            df_train_with_oof[col_name] = scores_series

        if "bear_score" in df_train_with_oof.columns:
            bear_risk = df_train_with_oof["bear_score"].astype(float).clip(0.0, 1.0)
            df_train_with_oof["bear_risk_score"] = bear_risk
            df_train_with_oof["bear_score"] = 1.0 - bear_risk
            _log_score_stats("OOF/bear_risk", df_train_with_oof["bear_risk_score"])
            _log_score_stats("OOF/bear_safety", df_train_with_oof["bear_score"])

        if "sector_score" not in df_train_with_oof.columns:
            df_train_with_oof["sector_score"] = 0.5
            log.info("[Fold %s] sector_score OOF not available — using neutral 0.5 for meta training.", fold_id)
        else:
            _log_score_stats("OOF/sector_score", df_train_with_oof["sector_score"])

        score_cols = [f"{ag_name}_score" for ag_name in agents_config.keys()]
        score_cols.append("sector_score")
        dispersion_scales = _compute_dispersion_scales(df_train_with_oof, score_cols)
        df_train_with_oof = _apply_dispersion_shrink(df_train_with_oof, dispersion_scales)

        if oof_cache is not None:
            try:
                oof_cache.save_pickle("oof_bundle", {
                    "df_train_with_oof": df_train_with_oof,
                    "dispersion_scales": dispersion_scales,
                })
                log.info("[Fold %s] Cache save: fold_oof", fold_id)
            except Exception as exc:
                log.warning("[Fold %s] Could not cache OOF bundle: %s", fold_id, exc)

    # ── Step 3: MetaLearner (meta-level cache) ────────────────────────────────
    meta_cache = fc["meta"] if fc is not None else None
    cached_meta = meta_cache.load_pickle("model") if meta_cache is not None else None
    meta = MetaLearner(results_dir=agent_models_results_dir, random_seed=random_seed)

    if cached_meta is not None:
        log.info("[Fold %s] Cache hit: fold_meta — skipping MetaLearner training.", fold_id)
        meta = cached_meta
    else:
        log.info(f"[Fold {fold_id}] 2/3 — Entrenando MetaLearner sobre scores OOF (anti-leakage)...")
        meta.fit(df_train_with_oof, y_train, fold=fold_id, sector_col="sector")
        if meta_cache is not None:
            try:
                meta_cache.save_pickle("model", meta)
                log.info("[Fold %s] Cache save: fold_meta", fold_id)
            except Exception as exc:
                log.warning("[Fold %s] Could not cache MetaLearner: %s", fold_id, exc)

    # ── Step 4: Test predictions ─────────────────────────────────────────────
    log.info(f"[Fold {fold_id}] 3/3 — Generando predicciones sobre el quarter de test ({len(df_test_norm)} tickers)...")
    df_test = _predict_base_scores(base_agents, agents_config, df_test_norm)

    if sector_agent.is_trained and sector_map is not None:
        sector_scores_test = sector_agent.predict_sector_scores(df_test, sector_map)
        df_test["sector_score"] = sector_agent.map_to_tickers(
            df_test, sector_map, sector_scores_test
        ).values
    else:
        df_test["sector_score"] = 0.5

    df_test = _apply_dispersion_shrink(df_test, dispersion_scales)
    df_test["final_score"] = meta.predict_score(df_test, "sector").values
    _log_score_stats("Meta/final_score_pre_sector_adjust", df_test["final_score"])
    df_test = _apply_sector_adjustments(df_test)
    df_test["label"] = y_test.values
    log.info(f"[Fold {fold_id}] 3/3 — Predicciones listas. Scores en rango [{df_test['final_score'].min():.3f}, {df_test['final_score'].max():.3f}]")

    # Save full fold test result to cache for the fast-path on future runs
    if fc is not None:
        try:
            fc["test"].save_pickle("df_test_scored", df_test.drop(columns=["label"], errors="ignore"))
            fc["test"].save_pickle("df_train_with_oof", df_train_with_oof)
            log.info("[Fold %s] Cache save: fold_test", fold_id)
        except Exception as exc:
            log.warning("[Fold %s] Could not cache fold_test: %s", fold_id, exc)

    agents_dict = {**base_agents, "sector_rotation": sector_agent, "meta_learner": meta}
    period_dir = Path(agents_results_dir) / str(fold_id)
    _export_feature_usage_report(
        agents=agents_dict,
        df_train=df_train_norm,
        fold_id=fold_id,
        agents_results_dir=period_dir.as_posix(),
    )
    return agents_dict, df_test, df_train_with_oof


def train_full_history(
    df_norm: pd.DataFrame,
    y: pd.Series,
    agent_models_results_dir: str,
    random_seed: int = 42,
    sector_map: Optional[Dict[str, str]] = None,
    spy_prices: Optional[pd.Series] = None,
) -> Tuple[Dict, pd.DataFrame, Dict[str, float]]:
    agents_config = build_agents_config(agent_models_results_dir=agent_models_results_dir, random_seed=random_seed)
    base_agents = _instantiate_base_agents(agents_config)
    meta = MetaLearner(results_dir=agent_models_results_dir, random_seed=random_seed)

    _fit_base_agents(base_agents, agents_config, df_norm, y, fold=0)

    sector_agent = build_sector_rotation_agent(agent_models_results_dir=agent_models_results_dir, random_seed=random_seed)
    if sector_map is not None and "forward_return" in df_norm.columns:
        sector_agent.fit(df_norm, sector_map=sector_map, spy_prices=spy_prices, fold=0)

    df_with_scores = _predict_base_scores(base_agents, agents_config, df_norm)

    if sector_agent.is_trained and sector_map is not None:
        sector_scores = sector_agent.predict_sector_scores(df_with_scores, sector_map)
        df_with_scores["sector_score"] = sector_agent.map_to_tickers(
            df_with_scores, sector_map, sector_scores
        ).values
    else:
        df_with_scores["sector_score"] = 0.5

    score_cols = [f"{ag_name}_score" for ag_name in agents_config.keys()]
    if "sector_score" in df_with_scores.columns:
        score_cols.append("sector_score")
    dispersion_scales = _compute_dispersion_scales(df_with_scores, score_cols)
    df_with_scores = _apply_dispersion_shrink(df_with_scores, dispersion_scales)

    meta.fit(df_with_scores, y, fold=0, sector_col="sector")

    agents_dict = {**base_agents, "sector_rotation": sector_agent, "meta_learner": meta}
    return agents_dict, df_with_scores, dispersion_scales
