"""Live fold orchestration and scoring."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from environment import PORTFOLIO_MIN_SCORE, SECTOR_ZSCORE_MIN_PEERS
from module.steps.step_02_dataset.builders.sector import SectorNormalizer
from module.steps.step_02_dataset.dataset import build_live_features
from module.steps.step_02_dataset.normalization import apply_sector_normalization
from module.steps.step_03_training.training import (
    _apply_dispersion_shrink,
    _apply_sector_adjustments,
    train_full_history,
)

from module.steps.step_04_evaluation.selection_reports import (
    build_explanation_candidate_tickers,
    build_selection_audit_df,
    export_selection_audit,
    export_ticker_explanations,
)
from module.steps.step_05_live.live_prices import download_live_prices
from module.steps.step_05_live.returns import qtd_return

log = logging.getLogger(__name__)


def run_live_fold(
    df: pd.DataFrame,
    sector_map: Dict[str, str],
    tickers_ok: List[str],
    as_of_date: str,
    router,
    fundamental_builder,
    technical_builder,
    valuation_builder,
    insider_builder,
    sentiment_builder,
    results_dir: str,
    agents_results_dir: str,
    top_n: int = 10,
    min_history_quarters: int = 4,
    random_seed: int = 42,
) -> None:
    """
    Fold live out-of-sample:
      1. Construye features a as_of_date usando datos guardados en disco.
      2. Entrena agentes sobre todo el historico disponible en df.
      3. Genera predicciones (scores) para cada ticker.
      4. Descarga precios actuales en memoria para calcular retornos reales.
      5. Calcula y registra el alpha de la estrategia vs benchmark.
      6. Guarda predicciones y metricas en CSV y JSON.
    """
    as_of = pd.Timestamp(as_of_date)
    today = pd.Timestamp.today().normalize()
    today_str = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    log.info("\n" + "=" * 60)
    log.info("  FOLD LIVE — predicción out-of-sample con precios reales")
    log.info(f"  Entrenado hasta : {as_of.date()}")
    log.info(f"  Evaluación real : {as_of.date()} → {today.date()}")
    log.info("=" * 60)

    # 1. Construir features live
    df_live = build_live_features(
        tickers=tickers_ok,
        as_of=as_of,
        router=router,
        fundamental_builder=fundamental_builder,
        technical_builder=technical_builder,
        valuation_builder=valuation_builder,
        insider_builder=insider_builder,
        sentiment_builder=sentiment_builder,
        min_history_quarters=min_history_quarters,
    )
    if df_live.empty:
        log.error("No se generaron features para el fold live - abortando.")
        return

    # 2. Entrenar sobre historico completo
    log.info("Entrenando agentes sobre el histórico completo (sin OOF — entreno final para producción)...")
    normalizer = SectorNormalizer(min_peers=SECTOR_ZSCORE_MIN_PEERS)
    df_train = df[~df.index.duplicated(keep="last")].copy()
    df_train_norm = apply_sector_normalization(df_train, sector_map, normalizer, fit=True)
    df_train_norm = df_train_norm[~df_train_norm.index.duplicated(keep="last")]

    # Label consistente con el walk-forward: outperformance relativa por quarter.
    from module.steps.step_04_evaluation.evaluator import _excess_return_label
    forward_train = _excess_return_label(df_train, sector_map=sector_map).reindex(df_train_norm.index)
    y_train = forward_train.dropna().astype(int)
    df_train_norm = df_train_norm.loc[y_train.index]

    agents, _, dispersion_scales = train_full_history(
        df_norm=df_train_norm,
        y=y_train,
        agents_results_dir=agents_results_dir,
        random_seed=random_seed,
        sector_map=sector_map,
    )

    # 3. Predecir en live
    df_live_norm = apply_sector_normalization(df_live, sector_map, normalizer, fit=False)
    df_live_norm = df_live_norm[~df_live_norm.index.duplicated(keep="last")]

    fundamental = agents["fundamental"]
    valuation = agents["valuation"]
    momentum = agents["momentum"]
    bear = agents["bear"]
    sentiment = agents["sentiment"]
    sector_rotation = agents["sector_rotation"]
    meta = agents["meta_learner"]

    df_live_norm["fundamental_score"] = fundamental.predict_score(df_live_norm, "sector").values
    df_live_norm["valuation_score"] = valuation.predict_score(df_live_norm, "sector").values
    df_live_norm["momentum_score"] = momentum.predict_score(df_live_norm).values
    df_live_norm["bear_score"] = bear.predict_score(df_live_norm).values
    if getattr(sentiment, "is_trained", False):
        df_live_norm["sentiment_score"] = sentiment.predict_score(df_live_norm).values
    else:
        log.warning("[SentimentAgent] No entrenado en live fold; usando score neutro 0.5.")
        df_live_norm["sentiment_score"] = 0.5
    if getattr(sector_rotation, "is_trained", False):
        sector_scores_live = sector_rotation.predict_sector_scores(df_live_norm, sector_map)
        df_live_norm["sector_score"] = sector_rotation.map_to_tickers(
            df_live_norm, sector_map, sector_scores_live
        ).values
    else:
        log.warning("[SectorRotationAgent] No entrenado en live fold; usando score neutro 0.5.")
        df_live_norm["sector_score"] = 0.5
    df_live_norm = _apply_dispersion_shrink(df_live_norm, dispersion_scales)
    df_live_norm["final_score"] = meta.predict_score(df_live_norm, "sector").values
    df_live_norm = _apply_sector_adjustments(df_live_norm)

    ticker_scores = (
        df_live_norm["final_score"]
        .reset_index(level="date", drop=True)
        .sort_values(ascending=False)
    )

    min_stocks = max(1, top_n // 2)
    qualified = ticker_scores[ticker_scores >= PORTFOLIO_MIN_SCORE]
    if not qualified.empty:
        n_take = max(min(len(qualified), top_n), min_stocks)
        top_bulls = list(ticker_scores.head(n_take).index)
        log.info(f"{len(qualified)} tickers superaron umbral {PORTFOLIO_MIN_SCORE:.2f} → seleccionando top {n_take} (mín={min_stocks})")
    else:
        top_bulls = list(ticker_scores.head(top_n).index)
        log.warning(f"Ningún ticker superó el umbral {PORTFOLIO_MIN_SCORE:.2f} — usando top-{top_n} por ranking")
    top_bears = list(ticker_scores.tail(top_n).index)

    log.info(f"\n{'='*60}")
    log.info(f"  TOP {top_n} OUTPERFORM  (score >= {PORTFOLIO_MIN_SCORE:.2f})")
    log.info(f"{'='*60}")
    for tk in top_bulls:
        log.info(f"    {tk:<8}  score={ticker_scores[tk]:.3f}  [{sector_map.get(tk,'?')}]")

    # 4. Descargar precios actuales (solo top picks + SPY como benchmark)
    tickers_to_fetch = list(dict.fromkeys(top_bulls + ["SPY"]))
    log.info(
        f"\nDescargando precios reales ({as_of.date()} → {today.date()}) "
        f"para {len(tickers_to_fetch)} tickers (en memoria, sin persistir)..."
    )
    live_prices = download_live_prices(
        tickers=tickers_to_fetch,
        start=as_of.strftime("%Y-%m-%d"),
        end=today_str,
    )

    # 5. Calcular retornos reales y alpha
    bull_returns = {
        t: r for t in top_bulls
        if (r := qtd_return(live_prices.get(t, pd.Series()))) is not None
    }
    benchmark_return = qtd_return(live_prices.get("SPY", pd.Series()))
    portfolio_return = float(np.mean(list(bull_returns.values()))) if bull_returns else None
    alpha = (
        portfolio_return - benchmark_return
        if portfolio_return is not None and benchmark_return is not None
        else None
    )

    log.info(f"\n{'='*60}")
    log.info(f"  RESULTADO REAL  ({as_of.date()} → {today.date()})")
    log.info(f"{'='*60}")
    if benchmark_return is not None:
        log.info(f"  Benchmark SPY          : {benchmark_return:+.2%}")
    if portfolio_return is not None:
        log.info(f"  Portfolio top-{top_n} (long): {portfolio_return:+.2%}")
    if alpha is not None:
        log.info(f"  Alpha vs benchmark     : {alpha:+.2%}")

    log.info(f"\n  Retornos individuales top-{top_n}:")
    for tk in top_bulls:
        ret = bull_returns.get(tk)
        bm = benchmark_return if benchmark_return else 0.0
        alp = (ret - bm) if ret is not None else None
        log.info(
            f"    {tk:<8}  ret={( ret or 0):+.2%}  "
            f"alpha={( alp or 0):+.2%}  [{sector_map.get(tk,'?')}]"
        )

    # 6. Guardar resultados
    tickers_idx = df_live_norm.index.get_level_values("ticker")
    all_preds = []
    for tk in ticker_scores.index:
        mask = tickers_idx == tk
        if not mask.any():
            continue
        row = df_live_norm.loc[mask].iloc[0]
        actual_ret = bull_returns.get(tk)
        all_preds.append({
            "ticker": tk,
            "sector": sector_map.get(tk, "?"),
            "final_score": round(float(row["final_score"]), 4),
            "fundamental_score": round(float(row["fundamental_score"]), 4),
            "valuation_score": round(float(row["valuation_score"]), 4),
            "momentum_score": round(float(row["momentum_score"]), 4),
            "bear_score": round(float(row["bear_score"]), 4),
            "sentiment_score": round(float(row.get("sentiment_score", 0.5)), 4),
            "sector_score": round(float(row.get("sector_score", 0.5)), 4),
            "label": "Outperform" if row["final_score"] >= 0.5 else "Underperform",
            "actual_return": round(actual_ret, 4) if actual_ret is not None else None,
        })

    quarter_label = f"Q{(as_of.month - 1) // 3 + 1} {as_of.year}"
    pred_df = pd.DataFrame(all_preds).sort_values("final_score", ascending=False)
    csv_path = Path(results_dir) / "predictions_LIVE.csv"
    json_path = Path(results_dir) / "predictions_LIVE.json"

    live_tag = "LIVE"
    score_cols = ["ticker", "final_score", "fundamental_score", "valuation_score",
                  "momentum_score", "bear_score", "sentiment_score", "sector_score"]
    audit_source = df_live_norm.reset_index()[
        [c for c in score_cols if c in df_live_norm.reset_index().columns]
    ].copy()
    audit_source["label"] = audit_source["final_score"].map(
        lambda v: "Outperform" if float(v) >= 0.5 else "Underperform"
    )
    live_audit = build_selection_audit_df(
        df_scored=audit_source,
        selected_tickers=top_bulls,
        score_col="final_score",
        threshold=PORTFOLIO_MIN_SCORE,
    )
    export_selection_audit(live_audit, results_dir, fold_id=live_tag, prefix="live")

    live_candidates = build_explanation_candidate_tickers(
        audit_df=live_audit,
        threshold=PORTFOLIO_MIN_SCORE,
        top_extra=max(top_n * 2, 10),
        near_margin=0.05,
        max_candidates=60,
    )
    export_ticker_explanations(
        agents=agents,
        df_test=df_live_norm,
        scores=df_live_norm["final_score"],
        fold_id=live_tag,
        agents_results_dir=agents_results_dir,
        candidate_tickers=live_candidates,
        audit_df=live_audit,
        explanation_top_n=6,
        prefix="live",
    )

    pred_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "quarter": quarter_label,
                "train_end": as_of.strftime("%Y-%m-%d"),
                "eval_date": today.strftime("%Y-%m-%d"),
                "n_tickers": len(all_preds),
                "portfolio_return": round(portfolio_return, 4) if portfolio_return is not None else None,
                "benchmark_return": round(benchmark_return, 4) if benchmark_return is not None else None,
                "alpha": round(alpha, 4) if alpha is not None else None,
                "top_picks": [p for p in all_preds if p["ticker"] in top_bulls],
                "bottom_picks": [p for p in all_preds if p["ticker"] in top_bears],
                "all": all_preds,
            },
            f, indent=2, ensure_ascii=False,
        )

    log.info(f"\nResultados live guardados:")
    log.info(f"  CSV  → {csv_path}")
    log.info(f"  JSON → {json_path}")
