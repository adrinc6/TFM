# =============================================================================
# module/data_router.py
# Carga y enruta todos los datos crudos. Centraliza el alineamiento temporal
# y la integración del sector desde companies.csv.
# =============================================================================
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)


class DataRouter:
    """
    Carga datos de disco y los sirve a los agentes respetando el orden temporal.

    Punto clave anti-leakage:
        load_consolidated() desplaza el índice +FILING_LAG_DAYS días,
        simulando que el informe trimestral no es público hasta ~45 días
        después del cierre del período.
    """

    def __init__(self, data_dir: str, filing_lag_days: int = 45):
        self.data_dir        = Path(data_dir)
        self.filing_lag_days = filing_lag_days
        self._macro_cache:     Optional[pd.DataFrame] = None
        self._companies_cache: Optional[pd.DataFrame] = None

    # ── Companies / Sector ────────────────────────────────────────────────────

    def load_companies(self) -> pd.DataFrame:
        """
        Carga companies.csv: Company Name | Ticker | Market Cap (mil) | Sector | Industry
        Devuelve DataFrame indexado por Ticker (uppercase). Resultado cacheado.
        """
        if self._companies_cache is not None:
            return self._companies_cache

        path = self.data_dir / "companies.csv"
        if not path.exists():
            log.warning("[DataRouter] companies.csv no encontrado — sin info de sector.")
            self._companies_cache = pd.DataFrame()
            return self._companies_cache

        df = pd.read_csv(path)

        # Normalizar nombres de columnas
        df.columns = (
            df.columns.str.strip()
                      .str.lower()
                      .str.replace(r"[\s]+", "_", regex=True)
                      .str.replace(r"[()]", "", regex=True)
        )
        # Alias flexibles
        for old, new in [("name","company_name"), ("symbol","ticker"),
                         ("market_cap","market_cap_mil")]:
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})

        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].str.strip().str.upper()
            df = df.drop_duplicates("ticker").set_index("ticker")

        for col in ["sector", "industry"]:
            if col in df.columns:
                df[col] = df[col].str.strip().str.title().fillna("Unknown")

        self._companies_cache = df
        nsectors = df["sector"].nunique() if "sector" in df.columns else 0
        log.info(f"[DataRouter] companies.csv cargado: {len(df)} empresas — {nsectors} sectores")
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

    def get_sector_map(self) -> Dict[str, str]:
        """Devuelve {ticker: sector} para todos los tickers de companies.csv."""
        c = self.load_companies()
        if c.empty or "sector" not in c.columns:
            return {}
        return c["sector"].fillna("Unknown").to_dict()

    def enrich_with_sector(self, df: pd.DataFrame, ticker_col: str = "ticker") -> pd.DataFrame:
        """Añade columnas 'sector' e 'industry' a un DataFrame que tenga columna ticker."""
        c = self.load_companies()
        if c.empty or ticker_col not in df.columns:
            df["sector"] = "Unknown"
            df["industry"] = "Unknown"
            return df
        mapping = c[["sector", "industry"]].rename_axis("ticker")
        df = df.join(mapping, on=ticker_col, how="left")
        df["sector"]   = df["sector"].fillna("Unknown")
        df["industry"] = df["industry"].fillna("Unknown")
        return df

    # ── Loaders de datos financieros ──────────────────────────────────────────

    def load_prices(self, ticker: str) -> Optional[pd.DataFrame]:
        """OHLCV diario desde data/prices/TICKER.csv"""
        path = self.data_dir / "prices" / f"{ticker}.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df.sort_index()

    def load_consolidated(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fundamentales consolidados desde data/consolidated/TICKER.csv.
        Aplica +FILING_LAG_DAYS al índice para evitar data leakage:
            Q1 cierra 31-Mar → disponible aprox. 15-May.
        """
        path = self.data_dir / "consolidated" / f"{ticker}.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.index = df.index + pd.DateOffset(days=self.filing_lag_days)
        return df

    def load_insider(self, ticker: str) -> Optional[pd.DataFrame]:
        """Transacciones insider desde data/insider/TICKER.csv"""
        path = self.data_dir / "insider" / f"{ticker}.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path, parse_dates=["date"])
        return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    def load_analyst(self, ticker: str) -> Optional[pd.DataFrame]:
        """Estimaciones de analistas / EPS dates desde data/analyst/TICKER.csv"""
        path = self.data_dir / "analyst" / f"{ticker}.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df.sort_index()

    def load_macro(self) -> Optional[pd.DataFrame]:
        """VIX, SP500, yield curve — macro_consolidated.csv. Cacheado."""
        if self._macro_cache is not None:
            return self._macro_cache
        path = self.data_dir / "macro" / "macro_consolidated.csv"
        if not path.exists():
            log.warning("[DataRouter] macro_consolidated.csv no encontrado.")
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        self._macro_cache = df.sort_index()
        return self._macro_cache

    def load_sp500_prices(self) -> Optional[pd.Series]:
        """Precios del S&P 500 como benchmark (data/macro/sp500.csv)."""
        path = self.data_dir / "macro" / "sp500.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df.iloc[:, 0].sort_index()

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
        start = as_of - pd.DateOffset(days=lookback_days)
        return insider[(insider["date"] >= start) & (insider["date"] <= as_of)]

    def get_macro_snapshot(
        self, macro: pd.DataFrame, as_of: pd.Timestamp
    ) -> Optional[pd.Series]:
        av = macro[macro.index <= as_of]
        return av.iloc[-1] if not av.empty else None

    def compute_forward_return(
        self, prices: pd.DataFrame, as_of: pd.Timestamp, forward_days: int = 252
    ) -> Optional[float]:
        """
        Retorno forward desde as_of hasta +forward_days días de trading.
        ¡Solo para construir el label y nunca como feature de los agentes!
        """
        cc     = "Close" if "Close" in prices.columns else prices.columns[3]
        past   = prices[prices.index <= as_of]
        future = prices[prices.index >  as_of]
        if past.empty or future.empty:
            return None
        p0 = float(past[cc].iloc[-1])
        p1 = float(future[cc].iloc[min(forward_days - 1, len(future) - 1)])
        if p0 <= 0 or pd.isna(p0) or pd.isna(p1):
            return None
        return (p1 - p0) / p0
