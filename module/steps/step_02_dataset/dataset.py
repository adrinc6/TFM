"""Dataset construction for training and live inference."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from module.common.data_router import DataRouter
from module.steps.step_02_dataset.builders.fundamental import FundamentalFeatureBuilder
from module.steps.step_02_dataset.builders.technical import TechnicalFeatureBuilder
from module.steps.step_02_dataset.builders.valuation import ValuationFeatureBuilder
from module.steps.step_02_dataset.builders.insider import InsiderFeatureBuilder
from module.steps.step_02_dataset.builders.sentiment import SentimentFeatureBuilder

log = logging.getLogger(__name__)


def _load_ticker_sources(router: DataRouter, ticker: str) -> Dict[str, Optional[pd.DataFrame]]:
    return {
        "prices": router.load_prices(ticker),
        "consolidated": router.load_consolidated(ticker),
        "eps_df": router.load_eps_surprises(ticker),
        "rec_df": router.load_recommendation_trends(ticker),
        "ins_df": router.load_insider_transactions(ticker),
        "mspr_df": router.load_insider_sentiment(ticker),
        "info": router.get_ticker_info(ticker),
    }


def _build_feature_record(
    ticker: str,
    as_of: pd.Timestamp,
    sources: Dict[str, Optional[pd.DataFrame]],
    fund_enriched: pd.DataFrame,
    macro: Optional[pd.DataFrame],
    router: DataRouter,
    technical_builder: TechnicalFeatureBuilder,
    valuation_builder: ValuationFeatureBuilder,
    insider_builder: InsiderFeatureBuilder,
    sentiment_builder: SentimentFeatureBuilder,
    include_label: bool,
    forward_return_days: int,
) -> Optional[Dict]:
    prices = sources["prices"]
    eps_df = sources["eps_df"]
    rec_df = sources["rec_df"]
    ins_df = sources["ins_df"]
    mspr_df = sources["mspr_df"]
    info = sources["info"] or {}

    fund_snap = router.get_fundamental_snapshot(fund_enriched, as_of)
    if fund_snap is None:
        return None

    price_window = router.get_price_window(prices, as_of, lookback_days=400)
    if len(price_window) < 20:
        return None

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
        "ticker": ticker,
        "date": as_of,
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
    }

    if include_label:
        fwd_return = router.compute_forward_return(prices, as_of, forward_return_days)
        if fwd_return is None:
            return None
        record["forward_return"] = fwd_return

    record.update(fund_snap.to_dict())
    record.update(tech_feats.to_dict())
    record.update(val_feats.to_dict())
    record.update(insider_feats.to_dict())
    record.update(sentiment_feats.to_dict())
    if macro_snap is not None:
        record.update(macro_snap.to_dict())
    return record


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
    log.info("Construyendo dataset maestro (Finnhub)...")
    macro = router.load_macro()
    records = []

    for i, ticker in enumerate(tickers, 1):
        try:
            sources = _load_ticker_sources(router, ticker)
            prices = sources["prices"]
            consolidated = sources["consolidated"]

            if prices is None or consolidated is None:
                log.debug(f"[{ticker}] Sin precios o consolidado — skip")
                continue
            if len(consolidated) < min_history_quarters:
                log.debug(f"[{ticker}] Insuficientes periodos ({len(consolidated)}) — skip")
                continue

            fund_enriched = fundamental_builder.build(consolidated)
            eval_dates = consolidated.index.tolist()

            for as_of in eval_dates:
                as_of = pd.Timestamp(as_of)

                record = _build_feature_record(
                    ticker=ticker,
                    as_of=as_of,
                    sources=sources,
                    fund_enriched=fund_enriched,
                    macro=macro,
                    router=router,
                    technical_builder=technical_builder,
                    valuation_builder=valuation_builder,
                    insider_builder=insider_builder,
                    sentiment_builder=sentiment_builder,
                    include_label=True,
                    forward_return_days=forward_return_days,
                )
                if record is not None:
                    records.append(record)

        except (ValueError, TypeError, KeyError) as e:
            log.warning(f"[{ticker}] Error de datos al construir features: {type(e).__name__}: {e}")
            continue
        except Exception as e:
            log.error(
                f"[{ticker}] Error inesperado construyendo features: {type(e).__name__}: {e}",
                exc_info=True,
            )
            continue

        if i % 50 == 0:
            log.info(f"  Procesados {i}/{len(tickers)} tickers — {len(records)} observaciones")

    if not records:
        raise RuntimeError("Dataset maestro vacio. Revisa los datos en data_finnhub/")

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
    log.info(f"Construyendo features live (as_of={as_of.date()})...")
    macro = router.load_macro()
    records = []

    for ticker in tickers:
        try:
            sources = _load_ticker_sources(router, ticker)
            prices = sources["prices"]
            consolidated = sources["consolidated"]

            if prices is None or consolidated is None:
                continue
            if len(consolidated) < min_history_quarters:
                continue

            fund_enriched = fundamental_builder.build(consolidated)

            record = _build_feature_record(
                ticker=ticker,
                as_of=as_of,
                sources=sources,
                fund_enriched=fund_enriched,
                macro=macro,
                router=router,
                technical_builder=technical_builder,
                valuation_builder=valuation_builder,
                insider_builder=insider_builder,
                sentiment_builder=sentiment_builder,
                include_label=False,
                forward_return_days=63,
            )
            if record is not None:
                records.append(record)

        except (ValueError, TypeError, KeyError) as e:
            log.debug(f"[{ticker}] Error de datos en features live: {type(e).__name__}: {e}")
        except Exception as e:
            log.error(f"[{ticker}] Error inesperado en features live: {type(e).__name__}: {e}", exc_info=True)

    if not records:
        log.error(f"No se generaron features live para as_of={as_of.date()}")
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index(["ticker", "date"]).sort_index()
    log.info(f"Features live: {len(df)} tickers")
    return df
