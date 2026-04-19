"""Dataset construction for training and live inference."""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from module.common.data_router import DataRouter
from module.steps.step_02_dataset.builders.fundamental import FundamentalFeatureBuilder
from module.steps.step_02_dataset.builders.technical import TechnicalFeatureBuilder
from module.steps.step_02_dataset.builders.valuation import ValuationFeatureBuilder
from module.steps.step_02_dataset.builders.insider import InsiderFeatureBuilder
from module.steps.step_02_dataset.builders.sentiment import SentimentFeatureBuilder
from environment import (
    FUNDAMENTAL_FEATURE_COLUMNS,
    FUNDAMENTAL_FEATURE_EXCLUDE,
    VALUATION_FEATURE_COLUMNS,
    VALUATION_FEATURE_EXCLUDE,
    MOMENTUM_FEATURE_COLUMNS,
    MOMENTUM_FEATURE_EXCLUDE,
    BEAR_FEATURE_COLUMNS,
    BEAR_FEATURE_EXCLUDE,
    SENTIMENT_FEATURE_COLUMNS,
    SENTIMENT_FEATURE_EXCLUDE,
    SECTOR_ROTATION_FEATURE_COLUMNS,
    SECTOR_ROTATION_FEATURE_EXCLUDE,
)

log = logging.getLogger(__name__)


_MASTER_METADATA_COLS = {
    "year_quarter",
    "snapshot_date",
    "sector",
    "industry",
    "forward_return",
    "report_end_date_used",
    "report_filed_date_used",
    "is_fundamental_carry_forward",
}


def _required_master_feature_columns() -> list[str]:
    cols = []
    for group in [
        FUNDAMENTAL_FEATURE_COLUMNS,
        FUNDAMENTAL_FEATURE_EXCLUDE,
        VALUATION_FEATURE_COLUMNS,
        VALUATION_FEATURE_EXCLUDE,
        MOMENTUM_FEATURE_COLUMNS,
        MOMENTUM_FEATURE_EXCLUDE,
        BEAR_FEATURE_COLUMNS,
        BEAR_FEATURE_EXCLUDE,
        SENTIMENT_FEATURE_COLUMNS,
        SENTIMENT_FEATURE_EXCLUDE,
        SECTOR_ROTATION_FEATURE_COLUMNS,
        SECTOR_ROTATION_FEATURE_EXCLUDE,
    ]:
        for c in group:
            if c not in cols:
                cols.append(c)
    return cols


def _enforce_master_feature_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Keep metadata + explicit configured features; add missing ones as NaN."""
    if df is None or df.empty:
        return df

    required = _required_master_feature_columns()
    missing_now = [c for c in required if c not in df.columns]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    keep_cols = [c for c in _MASTER_METADATA_COLS if c in df.columns] + required
    keep_cols = list(dict.fromkeys(keep_cols))

    extra_cols = [c for c in df.columns if c not in keep_cols]
    if extra_cols:
        preview = ", ".join(extra_cols[:12])
        more = " ..." if len(extra_cols) > 12 else ""
        log.info(
            "[Dataset] Columnas fuera del esquema explicito descartadas: %s%s (total=%d)",
            preview,
            more,
            len(extra_cols),
        )

    if missing_now:
        log.info("[Dataset] Columnas requeridas faltantes rellenadas como NaN: %d", len(missing_now))

    return df[keep_cols].copy()


def _build_filing_date_map_for_ticker(data_dir: str, ticker: str) -> Dict[pd.Timestamp, pd.Timestamp]:
    """Build mapping report_end_date -> filed_date for a ticker from raw Finnhub files."""
    per_ticker: Dict[pd.Timestamp, pd.Timestamp] = {}
    for file_name in ["financials_reported_quarterly.json", "financials_reported_annual.json"]:
        file_path = Path(data_dir) / str(ticker) / file_name
        if not file_path.exists():
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        for item in payload.get("data", []):
            end_date = item.get("endDate")
            filed_date = item.get("filedDate") or item.get("acceptedDate")
            if not end_date or not filed_date:
                continue
            try:
                end_ts = pd.Timestamp(end_date).normalize()
                filed_ts = pd.Timestamp(filed_date).normalize()
            except Exception:
                continue
            old = per_ticker.get(end_ts)
            if old is None or filed_ts > old:
                per_ticker[end_ts] = filed_ts
    return per_ticker


def _fund_snapshot_as_of_filing(
    fund_enriched: pd.DataFrame,
    filing_date_map: Dict[pd.Timestamp, pd.Timestamp],
    snapshot_date: pd.Timestamp,
) -> Optional[pd.Series]:
    """
    Select latest fundamental report that is already published at snapshot_date.
    If no filing metadata is available, fallback to latest report_end_date <= snapshot_date.
    """
    if fund_enriched is None or fund_enriched.empty:
        return None

    idx = pd.to_datetime(fund_enriched.index).normalize()
    report_dates = pd.Index(idx)

    def _filed_date_exact(report_dt: pd.Timestamp) -> pd.Timestamp:
        dt = pd.Timestamp(report_dt).normalize()
        filed = filing_date_map.get(dt)
        if filed is None:
            return pd.NaT
        return pd.Timestamp(filed).normalize()

    if filing_date_map:
        filed_series = pd.to_datetime(report_dates.map(_filed_date_exact), errors="coerce")
        links = pd.DataFrame(
            {
                "report_end_date": pd.to_datetime(report_dates).normalize(),
                "report_filed_date": filed_series,
            }
        )
        links = links.dropna(subset=["report_filed_date"])
        links = links[links["report_filed_date"] <= snapshot_date]

        if not links.empty:
            # Regla principal: usar el ultimo reporte publicado (filedDate mas reciente).
            # Si hay empate por filedDate, tomar el endDate mas reciente.
            links = links.sort_values(["report_filed_date", "report_end_date"]).reset_index(drop=True)
            chosen = links.iloc[-1]
            chosen_report = pd.Timestamp(chosen["report_end_date"]).normalize()
            chosen_filed = pd.Timestamp(chosen["report_filed_date"]).normalize()

            selected = fund_enriched.loc[fund_enriched.index.normalize() == chosen_report]
            if not selected.empty:
                out = selected.iloc[-1].copy()
                out["report_end_date_used"] = chosen_report
                out["report_filed_date_used"] = chosen_filed
                out["is_fundamental_carry_forward"] = bool(chosen_filed.to_period("Q") != snapshot_date.to_period("Q"))
                return out

    av = fund_enriched[fund_enriched.index <= snapshot_date]
    if av.empty:
        return None
    out = av.iloc[-1].copy()
    chosen_report = pd.Timestamp(av.index[-1]).normalize()
    out["report_end_date_used"] = chosen_report
    out["report_filed_date_used"] = _filed_date_exact(chosen_report) if filing_date_map else pd.NaT
    chosen_filed = pd.Timestamp(out["report_filed_date_used"]).normalize() if pd.notna(out["report_filed_date_used"]) else pd.NaT
    if pd.notna(chosen_filed):
        out["is_fundamental_carry_forward"] = bool(chosen_filed.to_period("Q") != snapshot_date.to_period("Q"))
    else:
        out["is_fundamental_carry_forward"] = bool(chosen_report.to_period("Q") != snapshot_date.to_period("Q"))
    return out


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
    filing_date_map: Dict[pd.Timestamp, pd.Timestamp],
    include_label: bool,
    snapshot_lag_days: int = 45,
    holding_period_months: int = 3,
    technical_lookback_days: int = 300,
) -> Optional[Dict]:
    prices = sources["prices"]
    eps_df = sources["eps_df"]
    rec_df = sources["rec_df"]
    ins_df = sources["ins_df"]
    mspr_df = sources["mspr_df"]
    info_source = sources.get("info")
    info = info_source if info_source is not None else {}

    # Snapshot date: primer dia del quarter + snapshot_lag_days.
    # Esto asegura un punto temporal consistente por quarter.
    q_start = as_of.to_period("Q").start_time.normalize()
    feature_date = q_start + pd.Timedelta(days=max(int(snapshot_lag_days), 0))

    # Snapshot fundamental por fecha de publicacion real (filedDate).
    # Si no hay reporte del quarter actual, usa el ultimo previamente publicado.
    fund_snap = _fund_snapshot_as_of_filing(
        fund_enriched=fund_enriched,
        filing_date_map=filing_date_map,
        snapshot_date=feature_date,
    )
    if fund_snap is None:
        return None

    # Trend features with data up to feature_date (no look-ahead on fundamentals).
    fund_hist_feature = fund_enriched[fund_enriched.index <= feature_date]
    trend_feats = fundamental_builder.snapshot_trends(fund_hist_feature)
    for k, v in trend_feats.items():
        if k not in fund_snap.index:
            fund_snap[k] = v

    price_window = router.get_price_window(prices, feature_date, lookback_days=technical_lookback_days)
    if len(price_window) < 20:
        return None

    tech_feats = technical_builder.build(
        price_window,
        feature_date,
        lookback_days=technical_lookback_days,
    )
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
        as_of=feature_date,
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
        "snapshot_date": feature_date,
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
    }

    if include_label:
        # Label del snapshot: retorno desde snapshot_date hasta snapshot_date + holding.
        fwd_return = router.compute_forward_return_from_snapshot(
            prices=prices,
            snapshot_date=feature_date,
            holding_period_months=holding_period_months,
        )
        record["forward_return"] = np.nan if fwd_return is None else fwd_return

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
    snapshot_lag_days: int = 45,
    holding_period_months: int = 3,
    technical_lookback_days: int = 300,
) -> pd.DataFrame:
    log.info(f"Building master dataset para {len(tickers)} tickers...")
    records = []

    try:
        from tqdm import tqdm
        ticker_iter = tqdm(tickers, desc="Master dataset", unit="ticker")
    except ImportError:
        ticker_iter = tickers

    for i, ticker in enumerate(ticker_iter, 1):
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
            filing_date_map = _build_filing_date_map_for_ticker(str(router.data_dir), ticker)

            # Panel trimestral continuo por ticker: una fila por quarter.
            min_q = pd.Timestamp(consolidated.index.min()).to_period("Q")
            max_q = pd.Timestamp(consolidated.index.max()).to_period("Q") + 1
            eval_dates = [p.end_time.normalize() for p in pd.period_range(min_q, max_q, freq="Q")]

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
                    filing_date_map=filing_date_map,
                    include_label=True,
                    snapshot_lag_days=snapshot_lag_days,
                    holding_period_months=holding_period_months,
                    technical_lookback_days=technical_lookback_days,
                )
                if record is not None:
                    records.append(record)

        except (ValueError, TypeError, KeyError) as e:
            log.warning(f"[{ticker}] Data error al construir features: {type(e).__name__}: {e}")
            continue
        except Exception as e:
            log.error(
                f"[{ticker}] Error inesperado construyendo features: {type(e).__name__}: {e}",
                exc_info=True,
            )
            continue

    if not records:
        raise RuntimeError("Master dataset is empty. Check the data in data_finnhub/")

    df = pd.DataFrame(records)
    # year_quarter is kept as a column (not as an index level) for downstream analysis
    df = df.set_index(["ticker", "date"]).sort_index()
    df = _enforce_master_feature_schema(df)
    log.info(
        f"Master dataset listo: {len(df)} observaciones | "
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
                filing_date_map=_build_filing_date_map_for_ticker(str(router.data_dir), ticker),
                include_label=False,
            )
            if record is not None:
                records.append(record)

        except (ValueError, TypeError, KeyError) as e:
            log.debug(f"[{ticker}] Data error en features live: {type(e).__name__}: {e}")
        except Exception as e:
            log.error(f"[{ticker}] Error inesperado en features live: {type(e).__name__}: {e}", exc_info=True)

    if not records:
        log.error(f"No live features were generated for as_of={as_of.date()}")
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index(["ticker", "date"]).sort_index()
    df = _enforce_master_feature_schema(df)
    log.info(f"Features live: {len(df)} tickers")
    return df
