# =============================================================================
# module/pipeline/dataset_builder.py
# Construcción del dataset maestro con features de Finnhub.
# =============================================================================
"""
Responsabilidades:
  - build_master_dataset : genera observaciones trimestrales por ticker con
    features fundamentales, técnicos, de valoración, insider/MSPR,
    sentiment (analistas + MSPR + EPS) y macro.
  - build_live_features  : construye features a una fecha fija para predicción.
  - apply_sector_normalization: Z-score sectorial sin leakage train/test.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from module.data_router import DataRouter
from module.feature_engineering import (
    FundamentalFeatureBuilder,
    TechnicalFeatureBuilder,
    ValuationFeatureBuilder,
    InsiderFeatureBuilder,
    SentimentFeatureBuilder,
    SectorNormalizer,
)

log = logging.getLogger(__name__)


def build_master_dataset(
    tickers: List[str],
    router: DataRouter,
    fundamental_builder: FundamentalFeatureBuilder,
    technical_builder: TechnicalFeatureBuilder,
    valuation_builder: ValuationFeatureBuilder,
    insider_builder: InsiderFeatureBuilder,
    sentiment_builder: SentimentFeatureBuilder,
    min_history_quarters: int = 4,
    forward_return_days: int = 63,
) -> pd.DataFrame:
    """
    Para cada ticker genera observaciones trimestrales con:
      - Features fundamentales (FinnhubConsolidator → FundamentalFeatureBuilder)
      - Features técnicos de precio (Yahoo OHLCV → TechnicalFeatureBuilder)
      - Múltiplos de valoración + EPS surprises + consenso analistas
      - Señales de insiders (transacciones 90d + MSPR mensual)
      - Sentiment completo (analistas + MSPR + EPS)
      - Macro context (snapshot antes de la fecha)
      - Forward return (label)

    Args:
        tickers:              Lista de tickers a procesar.
        router:               DataRouter configurado con data_finnhub/.
        fundamental_builder:  Builder de features fundamentales.
        technical_builder:    Builder de features técnicos.
        valuation_builder:    Builder de valoración + analistas.
        insider_builder:      Builder de insider transactions + MSPR.
        sentiment_builder:    Builder de sentiment (analistas + insiders + EPS).
        min_history_quarters: Mínimo de períodos requeridos por ticker.
        forward_return_days:  Días de trading para el label.

    Returns:
        DataFrame multi-índice (ticker, date) con todas las features y el label.
    """
    log.info("Construyendo dataset maestro (Finnhub)...")
    macro = router.load_macro()
    records = []

    for i, ticker in enumerate(tickers, 1):
        try:
            prices       = router.load_prices(ticker)
            consolidated = router.load_consolidated(ticker)

            if prices is None or consolidated is None:
                log.debug(f"[{ticker}] Sin precios o consolidado — skip")
                continue
            if len(consolidated) < min_history_quarters:
                log.debug(f"[{ticker}] Insuficientes períodos ({len(consolidated)}) — skip")
                continue

            # Datos de analistas e insiders (Finnhub)
            eps_df    = router.load_eps_surprises(ticker)
            rec_df    = router.load_recommendation_trends(ticker)
            ins_df    = router.load_insider_transactions(ticker)
            mspr_df   = router.load_insider_sentiment(ticker)
            info      = router.get_ticker_info(ticker)

            fund_enriched = fundamental_builder.build(consolidated)
            eval_dates    = consolidated.index.tolist()

            for as_of in eval_dates:
                as_of = pd.Timestamp(as_of)

                fund_snap = router.get_fundamental_snapshot(fund_enriched, as_of)
                if fund_snap is None:
                    continue

                price_window = router.get_price_window(prices, as_of, lookback_days=400)
                if len(price_window) < 20:
                    continue

                # Features técnicos
                tech_feats = technical_builder.build(price_window, as_of)

                # Features de valoración (incluye analistas EPS + consenso)
                val_feats = valuation_builder.build(
                    prices_df=prices,
                    fund_snapshot=fund_snap,
                    hist_fund=fund_enriched[fund_enriched.index <= as_of],
                    as_of=as_of,
                    analyst_df=eps_df,
                    recommendation_df=rec_df,
                )

                # Features de insiders (transacciones 90d + MSPR)
                insider_window = (
                    router.get_insider_window(ins_df, as_of, lookback_days=90)
                    if ins_df is not None else None
                )
                mspr_window = (
                    router.get_sentiment_series(mspr_df, as_of, lookback_months=6)
                    if mspr_df is not None else None
                )
                insider_feats = insider_builder.build(
                    insider_df=insider_window,
                    mspr_df=mspr_window,
                )

                # Features de sentiment (para el SentimentAgent)
                sentiment_feats = sentiment_builder.build(
                    recommendation_df=rec_df,
                    mspr_df=mspr_df,
                    insider_df=insider_window,
                    eps_df=eps_df,
                    as_of=as_of,
                )

                # Macro snapshot
                macro_snap = (
                    router.get_macro_snapshot(macro, as_of)
                    if macro is not None else None
                )

                # Label: forward return
                fwd_return = router.compute_forward_return(prices, as_of, forward_return_days)
                if fwd_return is None:
                    continue

                record = {
                    "ticker":          ticker,
                    "date":            as_of,
                    "sector":          info["sector"],
                    "industry":        info["industry"],
                    "forward_return":  fwd_return,
                }
                record.update(fund_snap.to_dict())
                record.update(tech_feats.to_dict())
                record.update(val_feats.to_dict())
                record.update(insider_feats.to_dict())
                record.update(sentiment_feats.to_dict())
                if macro_snap is not None:
                    record.update(macro_snap.to_dict())

                records.append(record)

        except Exception as e:
            log.warning(f"[{ticker}] Error construyendo features: {e}")
            continue

        if i % 50 == 0:
            log.info(f"  Procesados {i}/{len(tickers)} tickers — {len(records)} observaciones")

    if not records:
        raise RuntimeError("Dataset maestro vacío. Revisa los datos en data_finnhub/")

    df = pd.DataFrame(records).set_index(["ticker", "date"]).sort_index()
    log.info(
        f"Dataset maestro: {len(df)} observaciones — "
        f"{df.index.get_level_values('ticker').nunique()} tickers — "
        f"{len(df.columns)} features"
    )
    return df


def build_live_features(
    tickers: List[str],
    as_of: pd.Timestamp,
    router: DataRouter,
    fundamental_builder: FundamentalFeatureBuilder,
    technical_builder: TechnicalFeatureBuilder,
    valuation_builder: ValuationFeatureBuilder,
    insider_builder: InsiderFeatureBuilder,
    sentiment_builder: SentimentFeatureBuilder,
    min_history_quarters: int = 4,
) -> pd.DataFrame:
    """
    Construye features a una fecha fija (as_of) para predicción out-of-sample.
    Idéntico a build_master_dataset pero sin label y para una única fecha.
    """
    log.info(f"Construyendo features live (as_of={as_of.date()})...")
    macro   = router.load_macro()
    records = []

    for ticker in tickers:
        try:
            prices       = router.load_prices(ticker)
            consolidated = router.load_consolidated(ticker)

            if prices is None or consolidated is None:
                continue
            if len(consolidated) < min_history_quarters:
                continue

            eps_df   = router.load_eps_surprises(ticker)
            rec_df   = router.load_recommendation_trends(ticker)
            ins_df   = router.load_insider_transactions(ticker)
            mspr_df  = router.load_insider_sentiment(ticker)
            info     = router.get_ticker_info(ticker)

            fund_enriched = fundamental_builder.build(consolidated)
            fund_snap = router.get_fundamental_snapshot(fund_enriched, as_of)
            if fund_snap is None:
                continue

            price_window = router.get_price_window(prices, as_of, lookback_days=400)
            if len(price_window) < 20:
                continue

            tech_feats = technical_builder.build(price_window, as_of)

            val_feats = valuation_builder.build(
                prices_df=prices,
                fund_snapshot=fund_snap,
                hist_fund=fund_enriched[fund_enriched.index <= as_of],
                as_of=as_of,
                analyst_df=eps_df,
                recommendation_df=rec_df,
            )

            insider_window = (
                router.get_insider_window(ins_df, as_of, lookback_days=90)
                if ins_df is not None else None
            )
            mspr_window = (
                router.get_sentiment_series(mspr_df, as_of, lookback_months=6)
                if mspr_df is not None else None
            )
            insider_feats = insider_builder.build(
                insider_df=insider_window,
                mspr_df=mspr_window,
            )

            sentiment_feats = sentiment_builder.build(
                recommendation_df=rec_df,
                mspr_df=mspr_df,
                insider_df=insider_window,
                eps_df=eps_df,
                as_of=as_of,
            )

            macro_snap = (
                router.get_macro_snapshot(macro, as_of)
                if macro is not None else None
            )

            record = {
                "ticker":   ticker,
                "date":     as_of,
                "sector":   info["sector"],
                "industry": info["industry"],
            }
            record.update(fund_snap.to_dict())
            record.update(tech_feats.to_dict())
            record.update(val_feats.to_dict())
            record.update(insider_feats.to_dict())
            record.update(sentiment_feats.to_dict())
            if macro_snap is not None:
                record.update(macro_snap.to_dict())
            records.append(record)

        except Exception as e:
            log.debug(f"[{ticker}] Error features live: {e}")

    if not records:
        log.error(f"No se generaron features live para as_of={as_of.date()}")
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index(["ticker", "date"]).sort_index()
    log.info(f"Features live: {len(df)} tickers")
    return df


def apply_sector_normalization(
    df: pd.DataFrame,
    sector_map: Dict[str, str],
    normalizer: SectorNormalizer,
    fit: bool = True,
) -> pd.DataFrame:
    """
    Aplica SectorNormalizer al dataset añadiendo columnas '_zsector'.

    Cuando fit=True (solo en train), calcula media/std por sector usando TODAS
    las observaciones del DataFrame para estadísticas más robustas.
    Cuando fit=False (test/live), reutiliza estadísticas del fit anterior.

    Args:
        df:         DataFrame multi-índice (ticker, date).
        sector_map: Diccionario {ticker: sector} desde profiles Finnhub.
        normalizer: Instancia de SectorNormalizer.
        fit:        True solo en train para evitar leakage.

    Returns:
        DataFrame con columnas '_zsector' añadidas.
    """
    if fit:
        features_dict: Dict[str, pd.Series] = {}
        for i, ((ticker, _date), row) in enumerate(df.iterrows()):
            features_dict[f"{ticker}_{i}"] = row
        expanded_sector_map = {
            f"{ticker}_{i}": sector_map.get(ticker, "Unknown")
            for i, (ticker, _date) in enumerate(df.index)
        }
        normalizer.fit(features_dict, expanded_sector_map)

    tickers_col = df.index.get_level_values("ticker")
    sectors     = tickers_col.map(lambda t: sector_map.get(t, "Unknown"))

    df_norm = df.copy()
    for col in normalizer.COLS:
        if col not in df_norm.columns:
            df_norm[f"{col}_zsector"] = np.nan
            continue
        zcol = f"{col}_zsector"
        df_norm[zcol] = np.nan
        for sector in sectors.unique():
            stats = normalizer._stats.get(sector, {}).get(col)
            if stats is None:
                continue
            mean, std = stats
            mask = sectors == sector
            if std > 0:
                df_norm.loc[mask, zcol] = (df_norm.loc[mask, col] - mean) / std
            else:
                df_norm.loc[mask, zcol] = 0.0

    return df_norm
