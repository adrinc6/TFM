"""Step 05 live price download helpers."""

from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from module.steps.step_01_data.clients import YahooClient

log = logging.getLogger(__name__)


def download_live_prices(
    tickers: List[str],
    start: str,
    end: str,
) -> Dict[str, pd.Series]:
    """
    Download closing prices in memory (without writing to disk) to compute
    realized live-period returns.
    """
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
            log.debug(f"[{ticker}] Live price error: {e}")
            failed.append(ticker)

    if failed:
        log.info(
            f"[LiveFold] {len(failed)}/{len(tickers)} tickers without live prices: "
            f"{', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}"
        )
    return live_prices
