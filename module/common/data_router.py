# =============================================================================
# module/data_router.py
# Carga y enruta todos los datos desde data_finnhub/. Centraliza el
# alineamiento temporal y la integración del sector desde profiles Finnhub.
# =============================================================================
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


class DataRouter:
    """
    Carga datos de data_finnhub/ y los sirve a los agentes respetando
    el orden temporal (sin look-ahead).

    Fuente de datos: Finnhub + Yahoo Finance HTTP directo (data_finnhub/).
    """

    # Patrón válido para tickers: letras, dígitos, guión y punto (e.g. BRK-B, BF.B)
    _VALID_TICKER_RE = __import__("re").compile(r"^[A-Za-z0-9.\-]{1,10}$")

    def __init__(self, data_dir: str):
        self.data_dir           = Path(data_dir).resolve()
        self._companies_cache:  Optional[pd.DataFrame] = None

    def _validate_ticker(self, ticker: str) -> str:
        """Valida y normaliza un ticker para prevenir path-traversal."""
        t = str(ticker).strip()
        if not self._VALID_TICKER_RE.match(t):
            raise ValueError(f"Ticker inválido o potencialmente peligroso: {ticker!r}")
        # Verificar que la ruta resultante no escape del data_dir
        resolved = (self.data_dir / t).resolve()
        if not str(resolved).startswith(str(self.data_dir)):
            raise ValueError(f"Ticker produce ruta fuera del data_dir: {ticker!r}")
        return t

    # ── Companies / Sector ────────────────────────────────────────────────────

    def load_companies(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Construye el DataFrame de sector/industry desde los profile.json
        descargados por Finnhub. Resultado cacheado.

        Args:
            tickers: Lista de tickers a incluir. Si None, usa todos los que
                     tienen profile.json en data_dir.
        """
        if self._companies_cache is not None:
            return self._companies_cache

        from module.steps.step_01_data.consolidation import build_companies_df

        if tickers is None:
            # Auto-descubrir tickers desde directorios
            tickers = [
                d.name for d in self.data_dir.iterdir()
                if d.is_dir() and not d.name.startswith("_")
            ]

        df = build_companies_df(str(self.data_dir), tickers)
        self._companies_cache = df
        return self._companies_cache

    def get_ticker_info(self, ticker: str) -> Dict:
        """Devuelve {'sector', 'industry', 'market_cap_mil'} para un ticker."""
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
        """Devuelve {ticker: sector} para todos los tickers."""
        c = self.load_companies(tickers)
        if c.empty or "sector" not in c.columns:
            return {}
        return c["sector"].fillna("Unknown").to_dict()

    # ── Loaders de precios ─────────────────────────────────────────────────────

    def load_prices(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        OHLCV diario desde data_finnhub/{ticker}/prices.json.
        Columnas: Open, High, Low, Close, AdjClose, Volume.
        """
        ticker = self._validate_ticker(ticker)
        path = self.data_dir / ticker / "prices.json"
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.warning(f"[DataRouter] Error leyendo prices de {ticker}: {e}")
            return None

        records = data.get("data", [])
        if not records:
            return None

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # Estandarizar nombres de columnas al formato que esperan los builders
        rename = {
            "open":      "Open",
            "high":      "High",
            "low":       "Low",
            "close":     "Close",
            "adj_close": "AdjClose",
            "volume":    "Volume",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        # Usar adj_close como Close si está disponible (ajustado por splits/dividendos)
        if "AdjClose" in df.columns and not df["AdjClose"].isna().all():
            df["Close"] = df["AdjClose"]

        return df

    def load_consolidated(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fundamentales consolidados desde data_finnhub/consolidated/{ticker}.csv.
        Generado por FinnhubConsolidator.
        """
        ticker = self._validate_ticker(ticker)
        path = self.data_dir / "consolidated" / f"{ticker}.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df[~df.index.duplicated(keep="last")]

    # ── Loaders de datos de analistas ─────────────────────────────────────────

    def load_eps_surprises(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Series de EPS surprises desde data_finnhub/{ticker}/eps_surprises.json.
        Columnas: eps_actual, eps_estimate, eps_surprise_pct, eps_beat
        """
        from module.steps.step_01_data.parsers import EPSSurprisesParser
        ticker = self._validate_ticker(ticker)
        path = self.data_dir / ticker / "eps_surprises.json"
        df = EPSSurprisesParser().parse(path)
        return df if df is not None and not df.empty else None

    def load_recommendation_trends(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Series de consenso de analistas desde recommendation_trends.json.
        Columnas: analyst_buy_ratio, analyst_bearish_score, analyst_consensus, etc.
        """
        from module.steps.step_01_data.parsers import RecommendationParser
        ticker = self._validate_ticker(ticker)
        path = self.data_dir / ticker / "recommendation_trends.json"
        df = RecommendationParser().parse(path)
        return df if not df.empty else None

    def load_insider_transactions(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Transacciones insider desde data_finnhub/{ticker}/insider_transactions.json.
        Columnas: date, name, transaction_code, shares, is_buy, is_sell
        """
        from module.steps.step_01_data.parsers import InsiderTransactionsParser
        ticker = self._validate_ticker(ticker)
        path = self.data_dir / ticker / "insider_transactions.json"
        df = InsiderTransactionsParser().parse(path)
        return df if not df.empty else None

    def load_insider_sentiment(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        MSPR mensual desde data_finnhub/{ticker}/insider_sentiment.json.
        Columnas: mspr, insider_net_buy
        """
        from module.steps.step_01_data.parsers import InsiderSentimentParser
        ticker = self._validate_ticker(ticker)
        path = self.data_dir / ticker / "insider_sentiment.json"
        df = InsiderSentimentParser().parse(path)
        return df if not df.empty else None

    def load_sp500_prices(self) -> Optional[pd.Series]:
        """Serie de precios del S&P 500 como benchmark."""
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
            log.warning(f"[DataRouter] Error cargando sp500: {e}")
            return None

    # ── Helpers de alineamiento temporal (sin look-ahead) ─────────────────────

    def get_fundamental_snapshot(
        self, consolidated: pd.DataFrame, as_of: pd.Timestamp
    ) -> Optional[pd.Series]:
        """Última fila de fundamentales disponible ANTES de as_of."""
        av = consolidated[consolidated.index <= as_of]
        return av.iloc[-1] if not av.empty else None

    def get_price_window(
        self, prices: pd.DataFrame, as_of: pd.Timestamp, lookback_days: int = 400
    ) -> pd.DataFrame:
        """Ventana de precios históricos hasta as_of sin look-ahead."""
        return prices.loc[as_of - pd.DateOffset(days=lookback_days): as_of]

    def get_insider_window(
        self, insider: pd.DataFrame, as_of: pd.Timestamp, lookback_days: int = 90
    ) -> pd.DataFrame:
        """Transacciones insider en ventana de lookback_days días hasta as_of."""
        start = as_of - pd.DateOffset(days=lookback_days)
        return insider[(insider["date"] >= start) & (insider["date"] <= as_of)]

    def get_sentiment_series(
        self, df: pd.DataFrame, as_of: pd.Timestamp, lookback_months: int = 6
    ) -> pd.DataFrame:
        """Serie de sentiment hasta as_of para calcular tendencias."""
        if df is None or df.empty:
            return pd.DataFrame()
        start = as_of - pd.DateOffset(months=lookback_months)
        return df[(df.index >= start) & (df.index <= as_of)]

    @staticmethod
    def quarter_end(ts: pd.Timestamp) -> pd.Timestamp:
        """Devuelve el último día del trimestre al que pertenece ts."""
        return ts + pd.offsets.QuarterEnd(0)

    @staticmethod
    def next_quarter_end(ts: pd.Timestamp) -> pd.Timestamp:
        """Devuelve el último día del trimestre siguiente al de ts."""
        return DataRouter.quarter_end(ts) + pd.offsets.QuarterEnd(1)

    def compute_quarterly_forward_return(
        self, prices: pd.DataFrame, as_of: pd.Timestamp,
                lag_days: int = 45,
                holding_period_months: int = 3,
                days_before: Optional[int] = None,
    ) -> Optional[float]:
        """
                Retorno del holding period real del snapshot trimestral:
                    - Entrada: fin de quarter + lag_days
                    - Salida : entrada + holding_period_months

                Ejemplo con lag_days=45, holding_period_months=3 y as_of=Mar 31 (Q1):
                    - Entrada : ~May 15 (mitad de Q2)
                    - Salida  : ~Aug 15 (mitad de Q3)

                Compatibilidad: si se pasa days_before, se conserva el comportamiento
                anterior (entrada/salida antes del inicio de quarter).

        Solo para construir el label — nunca como feature.
        """
        cc = "Close" if "Close" in prices.columns else prices.columns[0]

        q_end_current = self.quarter_end(as_of)
        q_end_next = self.next_quarter_end(as_of)

        if days_before is not None:
            if days_before > 0:
                # entry = primer día de Q siguiente - days_before
                entry_date = q_end_current + pd.Timedelta(days=1) - pd.Timedelta(days=days_before)
                # exit  = primer día de Q+2 - days_before
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
        """Forward return from a concrete snapshot_date to snapshot_date + holding window."""
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
