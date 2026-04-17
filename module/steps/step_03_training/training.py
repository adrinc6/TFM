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
)
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

    csv_path = out_dir / f"quarter_{fold_id}_feature_usage_report.csv"
    json_path = out_dir / f"quarter_{fold_id}_feature_usage_report.json"

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
    sector_prior = SECTOR_SCORE_PRIOR_BASE + SECTOR_SCORE_PRIOR_WEIGHT * sector_score
    df["final_score_raw"] = df["final_score"]
    adjustment = (sector_prior * df["sector_confidence"]).clip(0.0, 1.0)
    df["final_score"] = (0.5 + (df["final_score"] - 0.5) * adjustment).clip(0.0, 1.0)

    log.info(
        f"[SectorAdjust] params: SECTOR_CONFIDENCE_PEERS={SECTOR_CONFIDENCE_PEERS}, "
        f"SECTOR_SCORE_PRIOR_BASE={SECTOR_SCORE_PRIOR_BASE:.3f}, "
        f"SECTOR_SCORE_PRIOR_WEIGHT={SECTOR_SCORE_PRIOR_WEIGHT:.3f}"
    )
    _log_score_stats("SectorAdjust/final_score_raw", df["final_score_raw"])
    _log_score_stats("SectorAdjust/sector_score", sector_score)
    _log_score_stats("SectorAdjust/sector_confidence", df["sector_confidence"])
    _log_score_stats("SectorAdjust/adjustment", adjustment)
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

    score_cols = [c for c in ["fundamental_score", "valuation_score", "momentum_score", "bear_score", "sentiment_score"] if c in out.columns]
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
    agents_results_dir: str,
    random_seed: int = 42,
    sector_map: Optional[Dict[str, str]] = None,
    spy_prices: Optional[pd.Series] = None,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    agents_config = build_agents_config(agents_results_dir=agents_results_dir, random_seed=random_seed)
    base_agents = _instantiate_base_agents(agents_config)
    meta = MetaLearner(results_dir=agents_results_dir, random_seed=random_seed)

    log.info(f"[Fold {fold_id}] 1/3 — Entrenando 6 agentes base con datos de entrenamiento del fold...")
    _fit_base_agents(base_agents, agents_config, df_train_norm, y_train, fold=fold_id)

    # Entrenar SectorRotationAgent (opera a nivel sector, no ticker)
    sector_agent = build_sector_rotation_agent(agents_results_dir=agents_results_dir, random_seed=random_seed)
    if sector_map is not None and "forward_return" in df_train_norm.columns:
        log.info(f"[Fold {fold_id}] 1/3 — Entrenando SectorRotationAgent (top-down, nivel sector)...")
        sector_agent.fit(df_train_norm, sector_map=sector_map, spy_prices=spy_prices, fold=fold_id)
    else:
        log.warning(f"[Fold {fold_id}] SectorRotationAgent: sin sector_map o forward_return — score neutro 0.5")

    log.info(f"[Fold {fold_id}] 2/3 — Generando scores OOF ({OOF_N_SPLITS} splits temporales) como input del MetaLearner...")
    oof_scores = generate_oof_scores(
        df_train_norm,
        y_train,
        agents_config=agents_config,
        n_splits=OOF_N_SPLITS,
        random_seed=random_seed,
    )
    df_train_with_oof = df_train_norm.copy()
    for col_name, scores_series in oof_scores.items():
        df_train_with_oof[col_name] = scores_series

    # BearAgent OOF arrives as risk; convert to safety to align direction.
    if "bear_score" in df_train_with_oof.columns:
        bear_risk = df_train_with_oof["bear_score"].astype(float).clip(0.0, 1.0)
        df_train_with_oof["bear_risk_score"] = bear_risk
        df_train_with_oof["bear_score"] = 1.0 - bear_risk
        _log_score_stats("OOF/bear_risk", df_train_with_oof["bear_risk_score"])
        _log_score_stats("OOF/bear_safety", df_train_with_oof["bear_score"])

    # Add sector_score to train OOF using the already-trained agent
    if sector_agent.is_trained and sector_map is not None:
        sector_scores_train = sector_agent.predict_sector_scores(df_train_with_oof, sector_map)
        df_train_with_oof["sector_score"] = sector_agent.map_to_tickers(
            df_train_with_oof, sector_map, sector_scores_train
        ).values

    score_cols = [f"{ag_name}_score" for ag_name in agents_config.keys()]
    if "sector_score" in df_train_with_oof.columns:
        score_cols.append("sector_score")
        _log_score_stats("OOF/sector_score", df_train_with_oof["sector_score"])
    dispersion_scales = _compute_dispersion_scales(df_train_with_oof, score_cols)
    df_train_with_oof = _apply_dispersion_shrink(df_train_with_oof, dispersion_scales)

    log.info(f"[Fold {fold_id}] 2/3 — Entrenando MetaLearner sobre scores OOF (anti-leakage)...")
    meta.fit(df_train_with_oof, y_train, fold=fold_id, sector_col="sector")

    log.info(f"[Fold {fold_id}] 3/3 — Generando predicciones sobre el quarter de test ({len(df_test_norm)} tickers)...")
    df_test = _predict_base_scores(base_agents, agents_config, df_test_norm)

    # Add sector_score to test
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

    agents_dict = {**base_agents, "sector_rotation": sector_agent, "meta_learner": meta}
    _export_feature_usage_report(
        agents=agents_dict,
        df_train=df_train_norm,
        fold_id=fold_id,
        agents_results_dir=agents_results_dir,
    )
    return agents_dict, df_test, df_train_with_oof


def train_full_history(
    df_norm: pd.DataFrame,
    y: pd.Series,
    agents_results_dir: str,
    random_seed: int = 42,
    sector_map: Optional[Dict[str, str]] = None,
    spy_prices: Optional[pd.Series] = None,
) -> Tuple[Dict, pd.DataFrame, Dict[str, float]]:
    agents_config = build_agents_config(agents_results_dir=agents_results_dir, random_seed=random_seed)
    base_agents = _instantiate_base_agents(agents_config)
    meta = MetaLearner(results_dir=agents_results_dir, random_seed=random_seed)

    _fit_base_agents(base_agents, agents_config, df_norm, y, fold=0)

    sector_agent = build_sector_rotation_agent(agents_results_dir=agents_results_dir, random_seed=random_seed)
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
