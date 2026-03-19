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

    def __init__(self, data_dir: str):
        self.data_dir           = Path(data_dir)
        self._macro_cache:      Optional[pd.DataFrame] = None
        self._companies_cache:  Optional[pd.DataFrame] = None

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

    def enrich_with_sector(self, df: pd.DataFrame, ticker_col: str = "ticker") -> pd.DataFrame:
        """Añade columnas 'sector' e 'industry' a un DataFrame con columna ticker."""
        c = self.load_companies()
        if c.empty or ticker_col not in df.columns:
            df["sector"]   = "Unknown"
            df["industry"] = "Unknown"
            return df
        mapping = c[["sector", "industry"]].rename_axis("ticker")
        df = df.join(mapping, on=ticker_col, how="left")
        df["sector"]   = df["sector"].fillna("Unknown")
        df["industry"] = df["industry"].fillna("Unknown")
        return df

    # ── Loaders de precios ─────────────────────────────────────────────────────

    def load_prices(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        OHLCV diario desde data_finnhub/{ticker}/prices.json.
        Columnas: Open, High, Low, Close, AdjClose, Volume.
        """
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
        path = self.data_dir / "consolidated" / f"{ticker}.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    # ── Loaders de datos de analistas ─────────────────────────────────────────

    def load_eps_surprises(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Series de EPS surprises desde data_finnhub/{ticker}/eps_surprises.json.
        Columnas: eps_actual, eps_estimate, eps_surprise_pct, eps_beat
        """
        from module.steps.step_01_data.parsers import EPSSurprisesParser
        path = self.data_dir / ticker / "eps_surprises.json"
        return EPSSurprisesParser().parse(path) or None

    def load_recommendation_trends(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Series de consenso de analistas desde recommendation_trends.json.
        Columnas: analyst_buy_ratio, analyst_bearish_score, analyst_consensus, etc.
        """
        from module.steps.step_01_data.parsers import RecommendationParser
        path = self.data_dir / ticker / "recommendation_trends.json"
        df = RecommendationParser().parse(path)
        return df if not df.empty else None

    def load_insider_transactions(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Transacciones insider desde data_finnhub/{ticker}/insider_transactions.json.
        Columnas: date, name, transaction_code, shares, is_buy, is_sell
        """
        from module.steps.step_01_data.parsers import InsiderTransactionsParser
        path = self.data_dir / ticker / "insider_transactions.json"
        df = InsiderTransactionsParser().parse(path)
        return df if not df.empty else None

    def load_insider_sentiment(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        MSPR mensual desde data_finnhub/{ticker}/insider_sentiment.json.
        Columnas: mspr, insider_net_buy
        """
        from module.steps.step_01_data.parsers import InsiderSentimentParser
        path = self.data_dir / ticker / "insider_sentiment.json"
        df = InsiderSentimentParser().parse(path)
        return df if not df.empty else None

    # ── Loader macro ───────────────────────────────────────────────────────────

    def load_macro(self) -> Optional[pd.DataFrame]:
        """
        Carga datos macro desde data_finnhub/_macro/*.json.
        Series: vix, sp500, us10y, us2y + derivados calculados.
        Cacheado en memoria.
        """
        if self._macro_cache is not None:
            return self._macro_cache

        macro_dir = self.data_dir / "_macro"
        if not macro_dir.exists():
            log.warning("[DataRouter] _macro/ no encontrado.")
            return None

        series_map = {"vix": None, "sp500": None, "us10y": None, "us2y": None}

        for name in series_map:
            path = macro_dir / f"{name}.json"
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                records = data.get("data", [])
                if records:
                    s = pd.DataFrame(records)
                    s["date"] = pd.to_datetime(s["date"])
                    s = s.set_index("date")["close"].sort_index()
                    s.name = name
                    series_map[name] = s
            except Exception as e:
                log.warning(f"[DataRouter] Error cargando macro {name}: {e}")

        available = {k: v for k, v in series_map.items() if v is not None}
        if not available:
            log.warning("[DataRouter] Sin datos macro disponibles.")
            return None

        macro = pd.concat(list(available.values()), axis=1)

        # Derivados
        if "us10y" in macro.columns and "us2y" in macro.columns:
            macro["yield_curve"] = macro["us10y"] - macro["us2y"]

        if "sp500" in macro.columns:
            macro["sp500_momentum_3m"]  = macro["sp500"].pct_change(63)
            macro["sp500_momentum_12m"] = macro["sp500"].pct_change(252)

        self._macro_cache = macro.sort_index()
        log.info(f"[DataRouter] Macro cargado: {len(macro)} filas, columnas: {list(macro.columns)}")
        return self._macro_cache

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
            return s.set_index("date")["close"].sort_index()
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

    def get_sentiment_window(
        self, df: pd.DataFrame, as_of: pd.Timestamp, lookback_months: int = 6
    ) -> Optional[pd.Series]:
        """
        Último snapshot de datos de sentiment (MSPR, recomendaciones) antes de as_of.
        Usado por InsiderSentimentParser y RecommendationParser.
        """
        if df is None or df.empty:
            return None
        av = df[df.index <= as_of]
        return av.iloc[-1] if not av.empty else None

    def get_sentiment_series(
        self, df: pd.DataFrame, as_of: pd.Timestamp, lookback_months: int = 6
    ) -> pd.DataFrame:
        """Serie de sentiment hasta as_of para calcular tendencias."""
        if df is None or df.empty:
            return pd.DataFrame()
        start = as_of - pd.DateOffset(months=lookback_months)
        return df[(df.index >= start) & (df.index <= as_of)]

    def get_macro_snapshot(
        self, macro: pd.DataFrame, as_of: pd.Timestamp
    ) -> Optional[pd.Series]:
        """Último snapshot macro disponible antes de as_of."""
        av = macro[macro.index <= as_of]
        return av.iloc[-1] if not av.empty else None

    def compute_forward_return(
        self, prices: pd.DataFrame, as_of: pd.Timestamp, forward_days: int = 63
    ) -> Optional[float]:
        """
        Retorno forward desde as_of hasta +forward_days días de trading.
        Solo para construir el label — nunca como feature de los agentes.
        """
        cc     = "Close" if "Close" in prices.columns else prices.columns[0]
        past   = prices[prices.index <= as_of]
        future = prices[prices.index >  as_of]
        if past.empty or future.empty:
            return None
        p0 = float(past[cc].iloc[-1])
        p1 = float(future[cc].iloc[min(forward_days - 1, len(future) - 1)])
        if p0 <= 0 or pd.isna(p0) or pd.isna(p1):
            return None
        return (p1 - p0) / p0
