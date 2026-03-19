# =============================================================================
# module/pipeline/live_fold.py
# Fold live out-of-sample: predicción Q1 2026 + evaluación con precios reales.
# =============================================================================
"""
Responsabilidades:
  - run_live_fold: construye features a una fecha de inicio de quarter,
    entrena agentes sobre todo el histórico, genera predicciones, descarga
    precios actuales en memoria y calcula el alpha real de la estrategia.

Anti-leakage:
  - Los features se construyen con datos hasta as_of (inclusive).
  - Los precios live se descargan solo para calcular retornos reales,
    nunca como feature del modelo.
  - La descarga de precios empieza exactamente en as_of para no usar
    precios de períodos anteriores.
"""
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from module.feature_engineering import SectorNormalizer
from module.pipeline.dataset_builder import build_live_features, apply_sector_normalization
from module.pipeline.trainer import train_full_history

log = logging.getLogger(__name__)


def _qtd_return(close_series: pd.Series) -> Optional[float]:
    """Retorno desde el primer día disponible hasta el último."""
    s = close_series.dropna()
    if len(s) < 2:
        return None
    return float((s.iloc[-1] - s.iloc[0]) / s.iloc[0])


def _download_live_prices(
    tickers: List[str],
    start: str,
    end: str,
) -> Dict[str, pd.Series]:
    """
    Descarga precios de cierre en memoria (sin guardar en disco) para calcular
    retornos reales del período live. Usa Yahoo Finance HTTP directo (sin yfinance).

    Args:
        tickers: Lista de tickers a descargar.
        start: Fecha de inicio (inclusive) en formato 'YYYY-MM-DD'.
        end: Fecha de fin (exclusive) en formato 'YYYY-MM-DD'.

    Returns:
        Diccionario {ticker: Series de cierre}.
    """
    from module.fetcher_finnhub import YahooClient

    yahoo = YahooClient()
    live_prices: Dict[str, pd.Series] = {}
    failed = []

    for ticker in tickers:
        try:
            df_live = yahoo.ohlcv(ticker, start=start, end=end)
            if df_live is None or df_live.empty or len(df_live) < 2:
                failed.append(ticker)
                continue
            close_col = "AdjClose" if "AdjClose" in df_live.columns else "Close"
            live_prices[ticker] = df_live[close_col].rename(ticker)
        except Exception as e:
            log.debug(f"[{ticker}] Error precio live: {e}")
            failed.append(ticker)

    if failed:
        log.info(
            f"[LiveFold] {len(failed)}/{len(tickers)} tickers sin precio live: "
            f"{', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}"
        )
    return live_prices


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
      2. Entrena agentes sobre todo el histórico disponible en df.
      3. Genera predicciones (scores) para cada ticker.
      4. Descarga precios actuales en memoria para calcular retornos reales.
      5. Calcula y registra el alpha de la estrategia vs benchmark.
      6. Guarda predicciones y métricas en CSV y JSON.

    Args:
        df: Dataset histórico completo (multi-índice ticker, date).
        sector_map: Diccionario {ticker: sector}.
        tickers_ok: Lista de tickers con datos disponibles.
        as_of_date: Fecha inicio del quarter live (ej: '2026-01-01').
        router: DataRouter configurado.
        fundamental_builder / technical_builder / valuation_builder /
        insider_builder / sentiment_builder: Builders de features (ya inicializados).
        results_dir: Directorio donde guardar CSV y JSON de predicciones.
        agents_results_dir: Directorio de diagnósticos de agentes.
        top_n: Número de tickers long (bulls) y short (bears) a seleccionar.
        min_history_quarters: Mínimo trimestres para incluir un ticker.
        random_seed: Semilla de reproducibilidad.
    """
    as_of = pd.Timestamp(as_of_date)
    today = pd.Timestamp.today().normalize()
    today_str = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    log.info("\n" + "=" * 60)
    log.info("  FOLD LIVE — out-of-sample")
    log.info(f"  Train hasta: {as_of.date()}  |  Eval: {as_of.date()} → {today.date()}")
    log.info("=" * 60)

    # ── 1. Construir features live
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
        log.error("No se generaron features para el fold live — abortando.")
        return

    # ── 2. Entrenar sobre histórico completo
    log.info("Entrenando agentes finales sobre histórico completo...")
    normalizer = SectorNormalizer()
    df_train = df[~df.index.duplicated(keep="last")].copy()
    df_train_norm = apply_sector_normalization(df_train, sector_map, normalizer, fit=True)
    df_train_norm = df_train_norm[~df_train_norm.index.duplicated(keep="last")]

    median_ret = df_train["forward_return"].median()
    forward_train = df_train["forward_return"].reindex(df_train_norm.index)
    y_train = (forward_train > median_ret).astype(int).dropna()
    df_train_norm = df_train_norm.loc[y_train.index]

    agents, _ = train_full_history(
        df_norm=df_train_norm,
        y=y_train,
        agents_results_dir=agents_results_dir,
        random_seed=random_seed,
    )

    # ── 3. Predecir en live
    df_live_norm = apply_sector_normalization(df_live, sector_map, normalizer, fit=False)
    df_live_norm = df_live_norm[~df_live_norm.index.duplicated(keep="last")]

    fundamental = agents["fundamental"]
    valuation = agents["valuation"]
    momentum = agents["momentum"]
    bear = agents["bear"]
    sentiment = agents["sentiment"]
    meta = agents["meta_learner"]

    df_live_norm["fundamental_score"] = fundamental.predict_score(df_live_norm, "sector").values
    df_live_norm["valuation_score"] = valuation.predict_score(df_live_norm, "sector").values
    df_live_norm["momentum_score"] = momentum.predict_score(df_live_norm).values
    df_live_norm["bear_score"] = bear.predict_score(df_live_norm).values
    df_live_norm["sentiment_score"] = sentiment.predict_score(df_live_norm).values
    df_live_norm["final_score"] = meta.predict_score(df_live_norm, "sector").values

    ticker_scores = (
        df_live_norm["final_score"]
        .reset_index(level="date", drop=True)
        .sort_values(ascending=False)
    )
    top_bulls = list(ticker_scores.head(top_n).index)
    top_bears = list(ticker_scores.tail(top_n).index)

    log.info(f"\n{'='*60}")
    log.info(f"  TOP {top_n} predichos OUTPERFORM")
    log.info(f"{'='*60}")
    for tk in top_bulls:
        log.info(f"  {tk:<8} score={ticker_scores[tk]:.3f}  [{sector_map.get(tk,'?')}]")

    log.info(f"\n  BOTTOM {top_n} predichos UNDERPERFORM")
    log.info(f"{'='*60}")
    for tk in top_bears:
        log.info(f"  {tk:<8} score={ticker_scores[tk]:.3f}  [{sector_map.get(tk,'?')}]")

    # ── 4. Descargar precios actuales (solo top/bottom + SPY)
    tickers_to_fetch = list(dict.fromkeys(top_bulls + top_bears + ["SPY"]))
    log.info(
        f"\nDescargando precios actuales ({as_of.date()} → {today.date()}) "
        f"en memoria para {len(tickers_to_fetch)} tickers (sin guardar)..."
    )
    live_prices = _download_live_prices(
        tickers=tickers_to_fetch,
        start=as_of.strftime("%Y-%m-%d"),
        end=today_str,
    )

    # ── 5. Calcular retornos reales y alpha
    bull_returns = {
        t: r for t in top_bulls
        if (r := _qtd_return(live_prices.get(t, pd.Series()))) is not None
    }
    bear_returns = {
        t: r for t in top_bears
        if (r := _qtd_return(live_prices.get(t, pd.Series()))) is not None
    }
    benchmark_return = _qtd_return(live_prices.get("SPY", pd.Series()))
    portfolio_return = float(np.mean(list(bull_returns.values()))) if bull_returns else None
    alpha = (
        portfolio_return - benchmark_return
        if portfolio_return is not None and benchmark_return is not None
        else None
    )

    log.info(f"\n{'='*60}")
    log.info(f"  RESULTADO REAL ({as_of.date()} → {today.date()})")
    log.info(f"{'='*60}")
    if benchmark_return is not None:
        log.info(f"  Benchmark (SPY):         {benchmark_return:+.2%}")
    if portfolio_return is not None:
        log.info(f"  Portfolio top-{top_n} (long): {portfolio_return:+.2%}")
    if alpha is not None:
        log.info(f"  Alpha:                   {alpha:+.2%}")

    log.info(f"\n  Retornos individuales top-{top_n}:")
    for tk in top_bulls:
        ret = bull_returns.get(tk)
        bm = benchmark_return if benchmark_return else 0.0
        alp = (ret - bm) if ret is not None else None
        log.info(
            f"    {tk:<8} {(ret or 0):+.2%}  "
            f"(alpha {(alp or 0):+.2%})  [{sector_map.get(tk,'?')}]"
        )

    # ── 6. Guardar resultados
    tickers_idx = df_live_norm.index.get_level_values("ticker")
    all_preds = []
    for tk in ticker_scores.index:
        mask = tickers_idx == tk
        if not mask.any():
            continue
        row = df_live_norm.loc[mask].iloc[0]
        actual_ret = bull_returns.get(tk) or bear_returns.get(tk)
        all_preds.append({
            "ticker": tk,
            "sector": sector_map.get(tk, "?"),
            "final_score": round(float(row["final_score"]), 4),
            "fundamental_score": round(float(row["fundamental_score"]), 4),
            "valuation_score": round(float(row["valuation_score"]), 4),
            "momentum_score": round(float(row["momentum_score"]), 4),
            "bear_score": round(float(row["bear_score"]), 4),
            "sentiment_score": round(float(row.get("sentiment_score", 0.5)), 4),
            "label": "Outperform" if row["final_score"] >= 0.5 else "Underperform",
            "actual_return": round(actual_ret, 4) if actual_ret is not None else None,
        })

    quarter_label = f"Q{(as_of.month - 1) // 3 + 1} {as_of.year}"
    pred_df = pd.DataFrame(all_preds).sort_values("final_score", ascending=False)
    csv_path = Path(results_dir) / f"predictions_{quarter_label.replace(' ', '_')}.csv"
    json_path = Path(results_dir) / f"predictions_{quarter_label.replace(' ', '_')}.json"

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

    log.info(f"\nResultados guardados en:")
    log.info(f"  {csv_path}")
    log.info(f"  {json_path}")
