"""Data routing module for the multi-agent stock picker pipeline.

Loads and routes all data from data_finnhub/. Centralises temporal alignment
and sector integration from Finnhub profiles. Serves data to agents without
introducing look-ahead bias.
"""
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


class DataRouter:
    """Loads data from data_finnhub/ and serves it to agents respecting
    temporal order (no look-ahead).

    Data source: Finnhub + direct Yahoo Finance HTTP (data_finnhub/).
    """

    # Valid ticker pattern: letters, digits, hyphen and dot (e.g. BRK-B, BF.B)
    _VALID_TICKER_RE = __import__("re").compile(r"^[A-Za-z0-9.\-]{1,10}$")

    def __init__(self, data_dir: str):
        """Initialises the DataRouter with the path to the data directory.

        Args:
            data_dir (str): Path to the root data directory (data_finnhub/).
        """
        self.data_dir           = Path(data_dir).resolve()
        self._companies_cache:  Optional[pd.DataFrame] = None

    def _validate_ticker(self, ticker: str) -> str:
        """Validates and normalises a ticker to prevent path-traversal attacks.

        Args:
            ticker (str): Raw ticker string to validate.

        Returns:
            str: The sanitised ticker string.

        Raises:
            ValueError: If the ticker is invalid or would escape the data directory.
        """
        t = str(ticker).strip()
        if not self._VALID_TICKER_RE.match(t):
            raise ValueError(f"Invalid or potentially dangerous ticker: {ticker!r}")
        # Verify the resulting path does not escape data_dir
        resolved = (self.data_dir / t).resolve()
        if not str(resolved).startswith(str(self.data_dir)):
            raise ValueError(f"Ticker produces path outside data_dir: {ticker!r}")
        return t

    # ── Companies / Sector ────────────────────────────────────────────────────

    def load_companies(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Builds the sector/industry DataFrame from Finnhub profile.json files.

        Result is cached after the first call.

        Args:
            tickers (Optional[List[str]]): List of tickers to include. If None,
                all tickers that have a profile.json in data_dir are used.

        Returns:
            pd.DataFrame: DataFrame indexed by ticker with sector, industry and
                market_cap_mil columns.
        """
        if self._companies_cache is not None:
            return self._companies_cache

        from module.steps.step_01_data.consolidation import build_companies_df

        if tickers is None:
            # Auto-discover tickers from subdirectories
            tickers = [
                d.name for d in self.data_dir.iterdir()
                if d.is_dir() and not d.name.startswith("_")
            ]

        df = build_companies_df(str(self.data_dir), tickers)
        self._companies_cache = df
        return self._companies_cache

    def get_ticker_info(self, ticker: str) -> Dict:
        """Returns sector, industry and market cap info for a single ticker.

        Args:
            ticker (str): The ticker symbol to look up.

        Returns:
            Dict: Dictionary with keys 'sector', 'industry', and
                'market_cap_mil'. Unknown values default to 'Unknown' or NaN.
        """
        c = self.load_companies()
        if c.empty or ticker not in c.index:
            return {"sector": "Unknown", "industry": "Unknown", "market_cap_mil": np.nan}
        row = c.loc[ticker]
        return {
            "sector":         row.get("sector",         "Unknown"),
            "industry":       row.get("industry",       "Unknown"),
            "market_cap_mil": row.get("market_cap_mil", np.nan),
        }

    def get_sector_map(self, tickers: Optional[List[str]] = None) -> Dict[str, str]:
        """Returns a mapping of ticker to sector for all available tickers.

        Args:
            tickers (Optional[List[str]]): Restrict the mapping to these tickers.
                If None, all available tickers are included.

        Returns:
            Dict[str, str]: Mapping ``{ticker: sector}``.
        """
        c = self.load_companies(tickers)
        if c.empty or "sector" not in c.columns:
            return {}
        return c["sector"].fillna("Unknown").to_dict()

    # ── Price loaders ─────────────────────────────────────────────────────────

    def load_prices(self, ticker: str) -> Optional[pd.DataFrame]:
        """Loads daily OHLCV data from data_finnhub/{ticker}/prices.json.

        Columns returned: Open, High, Low, Close, AdjClose, Volume.
        Close is preserved as the canonical execution/return price reference.

        Args:
            ticker (str): The ticker symbol to load prices for.

        Returns:
            Optional[pd.DataFrame]: DataFrame indexed by date, or None if
                the file does not exist or cannot be parsed.
        """
        ticker = self._validate_ticker(ticker)
        path = self.data_dir / ticker / "prices.json"
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.warning(f"[DataRouter] Error reading prices for {ticker}: {e}")
            return None

        records = data.get("data", [])
        if not records:
            return None

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # Standardise column names to the format expected by feature builders
        rename = {
            "open":      "Open",
            "high":      "High",
            "low":       "Low",
            "close":     "Close",
            "adj_close": "AdjClose",
            "volume":    "Volume",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        return df

    def load_consolidated(self, ticker: str) -> Optional[pd.DataFrame]:
        """Loads consolidated fundamentals from data_finnhub/consolidated/{ticker}.csv.

        This file is generated by FinnhubConsolidator and contains time-series
        fundamental metrics aligned by quarter-end date.

        Args:
            ticker (str): The ticker symbol to load fundamentals for.

        Returns:
            Optional[pd.DataFrame]: DataFrame indexed by date, or None if the
                file does not exist.
        """
        ticker = self._validate_ticker(ticker)
        path = self.data_dir / "consolidated" / f"{ticker}.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df[~df.index.duplicated(keep="last")]

    # ── Analyst data loaders ──────────────────────────────────────────────────

    def load_eps_surprises(self, ticker: str) -> Optional[pd.DataFrame]:
        """Loads EPS surprise series from data_finnhub/{ticker}/eps_surprises.json.

        Columns: eps_actual, eps_estimate, eps_surprise_pct, eps_beat.

        Args:
            ticker (str): The ticker symbol.

        Returns:
            Optional[pd.DataFrame]: Parsed DataFrame, or None if unavailable.
        """
        from module.steps.step_01_data.parsers import EPSSurprisesParser
        ticker = self._validate_ticker(ticker)
        path = self.data_dir / ticker / "eps_surprises.json"
        df = EPSSurprisesParser().parse(path)
        return df if df is not None and not df.empty else None

    def load_recommendation_trends(self, ticker: str) -> Optional[pd.DataFrame]:
        """Loads analyst consensus series from recommendation_trends.json.

        Columns: analyst_buy_ratio, analyst_bearish_score, analyst_consensus, etc.

        Args:
            ticker (str): The ticker symbol.

        Returns:
            Optional[pd.DataFrame]: Parsed DataFrame, or None if unavailable.
        """
        from module.steps.step_01_data.parsers import RecommendationParser
        ticker = self._validate_ticker(ticker)
        path = self.data_dir / ticker / "recommendation_trends.json"
        df = RecommendationParser().parse(path)
        return df if not df.empty else None

    def load_insider_transactions(self, ticker: str) -> Optional[pd.DataFrame]:
        """Loads insider transaction records from insider_transactions.json.

        Columns: date, name, transaction_code, shares, is_buy, is_sell.

        Args:
            ticker (str): The ticker symbol.

        Returns:
            Optional[pd.DataFrame]: Parsed DataFrame, or None if unavailable.
        """
        from module.steps.step_01_data.parsers import InsiderTransactionsParser
        ticker = self._validate_ticker(ticker)
        path = self.data_dir / ticker / "insider_transactions.json"
        df = InsiderTransactionsParser().parse(path)
        return df if not df.empty else None

    def load_insider_sentiment(self, ticker: str) -> Optional[pd.DataFrame]:
        """Loads monthly MSPR data from data_finnhub/{ticker}/insider_sentiment.json.

        Columns: mspr, insider_net_buy.

        Args:
            ticker (str): The ticker symbol.

        Returns:
            Optional[pd.DataFrame]: Parsed DataFrame, or None if unavailable.
        """
        from module.steps.step_01_data.parsers import InsiderSentimentParser
        ticker = self._validate_ticker(ticker)
        path = self.data_dir / ticker / "insider_sentiment.json"
        df = InsiderSentimentParser().parse(path)
        return df if not df.empty else None

    def load_sp500_prices(self) -> Optional[pd.Series]:
        """Loads S&P 500 price series to be used as a benchmark.

        Returns:
            Optional[pd.Series]: Daily close prices indexed by date, or None
                if the file does not exist or cannot be parsed.
        """
        macro_dir = self.data_dir / "_macro"
        path = macro_dir / "sp500.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data.get("data", [])
            if not records:
                return None
            s = pd.DataFrame(records)
            s["date"] = pd.to_datetime(s["date"])
            s = s.set_index("date")["close"].sort_index()
            return s[~s.index.duplicated(keep="last")]
        except Exception as e:
            log.warning(f"[DataRouter] Error loading sp500: {e}")
            return None

    # ── Temporal alignment helpers (no look-ahead) ────────────────────────────

    def get_fundamental_snapshot(
        self, consolidated: pd.DataFrame, as_of: pd.Timestamp
    ) -> Optional[pd.Series]:
        """Returns the last available fundamental row strictly before as_of.

        Args:
            consolidated (pd.DataFrame): Time-indexed fundamentals DataFrame.
            as_of (pd.Timestamp): The point-in-time cut-off date.

        Returns:
            Optional[pd.Series]: The most recent row at or before as_of, or
                None if no such row exists.
        """
        av = consolidated[consolidated.index <= as_of]
        return av.iloc[-1] if not av.empty else None

    def get_price_window(
        self, prices: pd.DataFrame, as_of: pd.Timestamp, lookback_days: int = 400
    ) -> pd.DataFrame:
        """Returns a window of historical prices up to as_of without look-ahead.

        Args:
            prices (pd.DataFrame): Full price history.
            as_of (pd.Timestamp): End date (inclusive) of the window.
            lookback_days (int): Number of calendar days to look back.

        Returns:
            pd.DataFrame: Slice of prices within the lookback window.
        """
        return prices.loc[as_of - pd.DateOffset(days=lookback_days): as_of]

    def get_insider_window(
        self, insider: pd.DataFrame, as_of: pd.Timestamp, lookback_days: int = 90
    ) -> pd.DataFrame:
        """Returns insider transactions within a lookback window ending at as_of.

        Args:
            insider (pd.DataFrame): Full insider transactions table with a 'date' column.
            as_of (pd.Timestamp): End date (inclusive) of the window.
            lookback_days (int): Number of calendar days to look back.

        Returns:
            pd.DataFrame: Filtered insider transactions within the window.
        """
        start = as_of - pd.DateOffset(days=lookback_days)
        return insider[(insider["date"] >= start) & (insider["date"] <= as_of)]

    def get_sentiment_series(
        self, df: pd.DataFrame, as_of: pd.Timestamp, lookback_months: int = 6
    ) -> pd.DataFrame:
        """Returns the sentiment time series up to as_of for computing trends.

        Args:
            df (pd.DataFrame): Full sentiment DataFrame indexed by date.
            as_of (pd.Timestamp): End date (inclusive) of the window.
            lookback_months (int): Number of months to look back.

        Returns:
            pd.DataFrame: Slice of df within the lookback window, or an empty
                DataFrame if df is None or empty.
        """
        if df is None or df.empty:
            return pd.DataFrame()
        start = as_of - pd.DateOffset(months=lookback_months)
        return df[(df.index >= start) & (df.index <= as_of)]

    @staticmethod
    def quarter_end(ts: pd.Timestamp) -> pd.Timestamp:
        """Returns the last day of the quarter that contains ts.

        Args:
            ts (pd.Timestamp): Any date within the quarter.

        Returns:
            pd.Timestamp: The quarter-end date.
        """
        return ts + pd.offsets.QuarterEnd(0)

    @staticmethod
    def next_quarter_end(ts: pd.Timestamp) -> pd.Timestamp:
        """Returns the last day of the quarter following the one that contains ts.

        Args:
            ts (pd.Timestamp): Any date within the current quarter.

        Returns:
            pd.Timestamp: The next quarter-end date.
        """
        return DataRouter.quarter_end(ts) + pd.offsets.QuarterEnd(1)

    def compute_quarterly_forward_return(
        self, prices: pd.DataFrame, as_of: pd.Timestamp,
                lag_days: int = 45,
                holding_period_months: int = 3,
                days_before: Optional[int] = None,
    ) -> Optional[float]:
        """Computes the forward return for a quarterly snapshot.

        The entry/exit dates are derived from the quarter end of ``as_of``:

        - **Default mode** (``days_before=None``):
          Entry = quarter_end(as_of) + lag_days
          Exit  = entry + holding_period_months

          Example with lag_days=45, holding_period_months=3, as_of=Mar 31 (Q1):
            - Entry: ~May 15 (mid Q2)
            - Exit:  ~Aug 15 (mid Q3)

        - **Legacy mode** (``days_before`` > 0): entry/exit are anchored to
          ``days_before`` days before the start of the next quarter.

        This method is for label construction only and must never be used as
        a feature (it would introduce look-ahead bias).

        Args:
            prices (pd.DataFrame): Full daily price history.
            as_of (pd.Timestamp): Snapshot date (typically quarter end).
            lag_days (int): Calendar days to wait after quarter end before entry.
            holding_period_months (int): Length of the holding period in months.
            days_before (Optional[int]): If set, uses the legacy entry/exit logic.

        Returns:
            Optional[float]: Decimal forward return (e.g. 0.05 = 5 %), or None
                if insufficient price data is available.
        """
        cc = "Close" if "Close" in prices.columns else prices.columns[0]

        q_end_current = self.quarter_end(as_of)
        q_end_next = self.next_quarter_end(as_of)

        if days_before is not None:
            if days_before > 0:
                # Entry = first day of next quarter minus days_before
                entry_date = q_end_current + pd.Timedelta(days=1) - pd.Timedelta(days=days_before)
                # Exit = first day of Q+2 minus days_before
                exit_date  = q_end_next    + pd.Timedelta(days=1) - pd.Timedelta(days=days_before)

                entry_window = prices[prices.index <= entry_date]
                exit_window  = prices[prices.index <= exit_date]

                if entry_window.empty or exit_window.empty:
                    return None

                p0 = float(entry_window[cc].iloc[-1])
                p1 = float(exit_window[cc].iloc[-1])
            else:
                past_window   = prices[prices.index <= q_end_current]
                future_window = prices[(prices.index > q_end_current) & (prices.index <= q_end_next)]

                if past_window.empty or future_window.empty:
                    return None

                p0 = float(past_window[cc].iloc[-1])
                p1 = float(future_window[cc].iloc[-1])
        else:
            lag_days = max(int(lag_days), 0)
            holding_period_months = max(int(holding_period_months), 1)
            entry_date = q_end_current + pd.Timedelta(days=lag_days)
            exit_date = entry_date + pd.DateOffset(months=holding_period_months)
            entry_window = prices[prices.index <= entry_date]
            exit_window = prices[prices.index <= exit_date]
            if entry_window.empty or exit_window.empty:
                return None
            p0 = float(entry_window[cc].iloc[-1])
            p1 = float(exit_window[cc].iloc[-1])

        if p0 <= 0 or pd.isna(p0) or pd.isna(p1):
            return None
        return (p1 - p0) / p0

    def compute_forward_return_from_snapshot(
        self,
        prices: pd.DataFrame,
        snapshot_date: pd.Timestamp,
        holding_period_months: int = 3,
    ) -> Optional[float]:
        """Computes the forward return from a concrete snapshot date.

        Entry price is the last available price at or before ``snapshot_date``.
        Exit price is the last available price at or before
        ``snapshot_date + holding_period_months``.

        Args:
            prices (pd.DataFrame): Full daily price history.
            snapshot_date (pd.Timestamp): The entry date for the return calculation.
            holding_period_months (int): Length of the holding period in months.

        Returns:
            Optional[float]: Decimal forward return, or None if insufficient
                price data is available.
        """
        if prices is None or prices.empty:
            return None
        cc = "Close" if "Close" in prices.columns else prices.columns[0]
        holding_period_months = max(int(holding_period_months), 1)

        entry_window = prices[prices.index <= snapshot_date]
        exit_date = snapshot_date + pd.DateOffset(months=holding_period_months)
        exit_window = prices[prices.index <= exit_date]
        if entry_window.empty or exit_window.empty:
            return None

        p0 = float(entry_window[cc].iloc[-1])
        p1 = float(exit_window[cc].iloc[-1])
        if p0 <= 0 or pd.isna(p0) or pd.isna(p1):
            return None
        return (p1 - p0) / p0
