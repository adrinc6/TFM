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
    router: DataRouter,
    fundamental_builder: FundamentalFeatureBuilder,
    technical_builder: TechnicalFeatureBuilder,
    valuation_builder: ValuationFeatureBuilder,
    insider_builder: InsiderFeatureBuilder,
    sentiment_builder: SentimentFeatureBuilder,
    include_label: bool,
    days_before: int = 0,
) -> Optional[Dict]:
    prices = sources["prices"]
    eps_df = sources["eps_df"]
    rec_df = sources["rec_df"]
    ins_df = sources["ins_df"]
    mspr_df = sources["mspr_df"]
    info_source = sources.get("info")
    info = info_source if info_source is not None else {}

    # ── Feature date: el día en que realmente operamos ───────────────────────
    # Con days_before > 0, TODOS los features (técnicos Y fundamentales) se
    # calculan como si fuéramos en entry_date (el día real de compra).
    #
    # Ejemplo con days_before=30, as_of=Mar 31 (fin de Q1):
    #   feature_date = Apr 1 - 30 = Mar 2
    #   - Fundamentales: último snapshot disponible ≤ Mar 2 = Q4 (Dic 31) ✓
    #   - Features técnicas: precios hasta Mar 2                          ✓
    #   - Label: retorno Mar 2 → Jun 1                                    ✓
    #
    # Esto replica exactamente la realidad: el 2 de Marzo tienes los datos
    # de Q4 (publicados en Enero-Febrero) y los precios hasta ese día.
    # Los fundamentales de Q1 (Mar 31) NO están disponibles aún.
    #
    # Con days_before=0: feature_date = as_of, sin cambio de comportamiento.
    if days_before > 0:
        q_end = DataRouter.quarter_end(as_of)
        feature_date = q_end + pd.Timedelta(days=1) - pd.Timedelta(days=days_before)
        # Salvaguarda: feature_date nunca puede superar as_of
        feature_date = min(feature_date, as_of)
    else:
        feature_date = as_of

    # Snapshot fundamental: último trimestre disponible ANTES de feature_date.
    # Con days_before=30: feature_date=Mar 2 → retorna Q4 (Dic 31). Realista.
    fund_snap = router.get_fundamental_snapshot(fund_enriched, feature_date)
    if fund_snap is None:
        return None

    # Trend features con datos hasta feature_date (sin look-ahead en fundamentales).
    fund_hist_feature = fund_enriched[fund_enriched.index <= feature_date]
    trend_feats = fundamental_builder.snapshot_trends(fund_hist_feature)
    for k, v in trend_feats.items():
        if k not in fund_snap.index:
            fund_snap[k] = v

    price_window = router.get_price_window(prices, feature_date, lookback_days=400)
    if len(price_window) < 20:
        return None

    tech_feats = technical_builder.build(price_window, feature_date)
    val_feats = valuation_builder.build(
        prices_df=prices,
        fund_snapshot=fund_snap,
        hist_fund=fund_enriched[fund_enriched.index <= feature_date],
        as_of=feature_date,
    )

    insider_window = (
        router.get_insider_window(ins_df, feature_date, lookback_days=90)
        if ins_df is not None else None
    )
    mspr_window = (
        router.get_sentiment_series(mspr_df, feature_date, lookback_months=6)
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
        as_of=feature_date,
    )

    # Identificador trimestral: "2024Q1", "2024Q2", etc.
    year_quarter = f"{as_of.year}Q{as_of.quarter}"

    record = {
        "ticker": ticker,
        "date": as_of,
        "year_quarter": year_quarter,
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
    }

    if include_label:
        # Label trimestral: retorno del holding period real.
        # Con days_before=30: mide desde ~30 días antes del inicio del Q siguiente
        # hasta ~30 días antes del inicio del Q+2 — alineado con el backtester.
        fwd_return = router.compute_quarterly_forward_return(prices, as_of, days_before=days_before)
        if fwd_return is None:
            return None
        record["forward_return"] = fwd_return

    record.update(fund_snap.to_dict())
    record.update(tech_feats.to_dict())
    record.update(val_feats.to_dict())
    record.update(insider_feats.to_dict())
    record.update(sentiment_feats.to_dict())
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
    days_before: int = 0,
) -> pd.DataFrame:
    log.info(f"Construyendo dataset maestro para {len(tickers)} tickers...")
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
                    router=router,
                    fundamental_builder=fundamental_builder,
                    technical_builder=technical_builder,
                    valuation_builder=valuation_builder,
                    insider_builder=insider_builder,
                    sentiment_builder=sentiment_builder,
                    include_label=True,
                    days_before=days_before,
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
            log.info(f"  [{i}/{len(tickers)}] tickers procesados — {len(records)} observaciones acumuladas")

    if not records:
        raise RuntimeError("Dataset maestro vacio. Revisa los datos en data_finnhub/")

    df = pd.DataFrame(records)
    # year_quarter se conserva como columna (no como nivel de índice) para análisis posterior
    df = df.set_index(["ticker", "date"]).sort_index()
    log.info(
        f"Dataset maestro listo: {len(df)} observaciones | "
        f"{df.index.get_level_values('ticker').nunique()} tickers | "
        f"{df['year_quarter'].nunique()} quarters | "
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
                router=router,
                fundamental_builder=fundamental_builder,
                technical_builder=technical_builder,
                valuation_builder=valuation_builder,
                insider_builder=insider_builder,
                sentiment_builder=sentiment_builder,
                include_label=False,
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
