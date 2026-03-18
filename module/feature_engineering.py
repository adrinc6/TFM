# =============================================================================
# module/feature_engineering.py
# Builders de features para cada agente + normalización sectorial Z-score.
# El sector viene de companies.csv via DataRouter.
# =============================================================================
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ── 1. Features Fundamentales ─────────────────────────────────────────────────
class FundamentalFeatureBuilder:
    """
    Enriquece el DataFrame consolidado (output de FinancialDataConsolidator)
    con ratios adicionales calculados de los datos base.

    Datos que consume (columnas del consolidado):
        revenue, net_income, operating_income, gross_profit,
        total_assets, total_equity, total_debt, fcf,
        operating_cash_flow, interest_expense, capex,
        ebitda, eps, current_ratio, debt_equity, debt_to_ebitda

    Ratios que produce:
        Rentabilidad:  roe, roa, roi, roic, net_margin, gross_margin,
                       fcf_margin, ebitda_margin, operating_margin
        Liquidez:      current_ratio, quick_ratio (ya en consolidado)
        Solvencia:     debt_equity, debt_to_ebitda, interest_coverage
        Crecimiento:   revenue_yoy_growth, net_income_yoy_growth,
                       eps_yoy_growth, fcf_yoy_growth,
                       operating_income_yoy_growth, total_debt_yoy_growth
        Calidad:       accruals_ratio, capex_to_revenue
        Riesgo:        consecutive_losses, revenue_decline
    """

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._yoy_growth(df)
        df = self._quality_metrics(df)
        df = self._coverage_ratios(df)
        df = self._risk_flags(df)
        return df

    def _yoy_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        pairs = {
            "revenue_yoy_growth":          "revenue",
            "net_income_yoy_growth":       "net_income",
            "operating_income_yoy_growth": "operating_income",
            "fcf_yoy_growth":              "fcf",
            "eps_yoy_growth":              "eps",
            "total_debt_yoy_growth":       "total_debt",
        }
        for feat, col in pairs.items():
            if col in df.columns:
                df[feat] = df[col].pct_change(periods=4)   # 4 trimestres = YoY
        return df

    def _quality_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        # Accruals ratio: alto → beneficios de baja calidad contable
        if {"net_income", "operating_cash_flow", "total_assets"}.issubset(df.columns):
            df["accruals_ratio"] = (
                (df["net_income"] - df["operating_cash_flow"])
                / df["total_assets"].replace(0, np.nan)
            )
        if "capex" in df.columns and "revenue" in df.columns:
            df["capex_to_revenue"] = df["capex"].abs() / df["revenue"].replace(0, np.nan)
        if "ebitda" in df.columns and "revenue" in df.columns:
            df["ebitda_margin"] = df["ebitda"] / df["revenue"].replace(0, np.nan)
        if "operating_income" in df.columns and "revenue" in df.columns:
            df["operating_margin"] = df["operating_income"] / df["revenue"].replace(0, np.nan)
        return df

    def _coverage_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        if "operating_income" in df.columns and "interest_expense" in df.columns:
            ie = df["interest_expense"].replace(0, np.nan).abs()
            df["interest_coverage"] = df["operating_income"] / ie
        return df

    def _risk_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        if "net_income" in df.columns:
            losses = (df["net_income"] < 0).astype(int)
            groups = (losses != losses.shift()).cumsum()
            df["consecutive_losses"] = losses * (losses.groupby(groups).cumcount() + 1)
        if "revenue_yoy_growth" in df.columns:
            df["revenue_decline"] = (df["revenue_yoy_growth"] < 0).astype(int)
        return df


# ── 2. Features Técnicos de Precio ───────────────────────────────────────────
class TechnicalFeatureBuilder:
    """
    Calcula indicadores técnicos sobre ventana OHLCV diaria hasta as_of_date.

    Datos que consume:  prices_df con columnas Open/High/Low/Close/Volume
    Features que produce:
        Osciladores:    rsi_14, rsi_28
        Tendencia:      macd, macd_signal, macd_hist
        SMAs:           sma_20/50/200 (como distancia % al precio)
        Bandas:         bb_pct (posición dentro de Bollinger 20d)
        52 semanas:     price_vs_52w_high, price_vs_52w_low
        Momentum:       momentum_1m/3m/6m/12m
        Volatilidad:    volatility_20d, volatility_60d, atr_14
        Volumen:        vol_ratio_20_50
    """

    def build(self, prices_df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        df = prices_df[prices_df.index <= as_of].copy()
        if len(df) < 20:
            return pd.Series(dtype=float)

        close = df["Close"]  if "Close"  in df.columns else df.iloc[:, 3]
        high  = df["High"]   if "High"   in df.columns else df.iloc[:, 1]
        low   = df["Low"]    if "Low"    in df.columns else df.iloc[:, 2]
        vol   = df["Volume"] if "Volume" in df.columns else df.iloc[:, 4]
        f: Dict = {}

        # RSI
        f["rsi_14"] = self._rsi(close, 14)
        f["rsi_28"] = self._rsi(close, 28)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9, adjust=False).mean()
        f["macd"]        = float(macd.iloc[-1])
        f["macd_signal"] = float(sig.iloc[-1])
        f["macd_hist"]   = float((macd - sig).iloc[-1])

        # SMAs — distancia % relativa al precio actual
        p = float(close.iloc[-1])
        for w in [20, 50, 200]:
            if len(close) >= w:
                sma = float(close.rolling(w).mean().iloc[-1])
                f[f"sma_{w}"] = (p / sma - 1) if sma != 0 else 0.0

        # Bollinger Bands (20d, 2σ) → posición [0=lower, 1=upper]
        if len(close) >= 20:
            s20 = close.rolling(20).mean()
            d20 = close.rolling(20).std()
            band_width = (4 * d20).replace(0, np.nan)
            f["bb_pct"] = float(((close - (s20 - 2 * d20)) / band_width).iloc[-1])

        # 52 semanas
        lk252 = close.tail(252)
        if len(lk252) > 10:
            f["price_vs_52w_high"] = float(p / lk252.max() - 1)
            f["price_vs_52w_low"]  = float(p / lk252.min() - 1)

        # Momentum puro
        for days, name in [(21,"momentum_1m"),(63,"momentum_3m"),
                           (126,"momentum_6m"),(252,"momentum_12m")]:
            if len(close) > days:
                f[name] = float(p / close.iloc[-(days + 1)] - 1)

        # Volatilidad realizada anualizada
        ret = close.pct_change().dropna()
        for w in [20, 60]:
            if len(ret) >= w:
                f[f"volatility_{w}d"] = float(ret.tail(w).std() * np.sqrt(252))

        # ATR 14
        if len(df) >= 14:
            f["atr_14"] = self._atr(high, low, close, 14)

        # Ratio volumen 20d vs 50d
        if len(vol) >= 50:
            f["vol_ratio_20_50"] = float(vol.tail(20).mean() / vol.tail(50).mean())

        return pd.Series(f)

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> float:
        delta = close.diff().dropna()
        gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - 100 / (1 + rs)
        return float(rsi.iloc[-1]) if not rsi.empty else np.nan

    @staticmethod
    def _atr(high, low, close, period) -> float:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return float(tr.ewm(com=period - 1, adjust=False).mean().iloc[-1])


# ── 3. Features de Valoración ─────────────────────────────────────────────────
class ValuationFeatureBuilder:
    """
    Calcula múltiplos de valoración cruzando precio con fundamentales.

    Datos que consume:
        prices_df        → precio de cierre en as_of
        fund_snapshot    → última fila de fundamentales disponible
        hist_fund        → histórico de fundamentales (para comparativa 5Y)

    Features que produce:
        Múltiplos:       pe_ratio, pb_ratio, ps_ratio, ev_to_ebitda,
                         fcf_yield, earnings_yield
        Vs historial:    pe_vs_5y_median, pb_vs_5y_median, ev_ebitda_vs_5y_median
        Analistas:       eps_surprise_pct, eps_revision
    """

    def build(
        self,
        prices_df:    pd.DataFrame,
        fund_snapshot: pd.Series,
        hist_fund:    pd.DataFrame,
        as_of:        pd.Timestamp,
        analyst_df:   Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        f: Dict = {}

        # Precio actual
        recent = prices_df[prices_df.index <= as_of]
        if recent.empty:
            return pd.Series(dtype=float)
        cc    = "Close" if "Close" in recent.columns else recent.columns[3]
        price = float(recent[cc].iloc[-1])

        shares = fund_snapshot.get("shares_diluted", np.nan)

        # P/E y Earnings Yield
        eps = fund_snapshot.get("eps", np.nan)
        if pd.notna(eps) and eps > 0:
            f["pe_ratio"]      = price / eps
            f["earnings_yield"] = eps / price

        # P/B
        equity = fund_snapshot.get("total_equity", np.nan)
        if pd.notna(equity) and pd.notna(shares) and shares > 0 and equity > 0:
            f["pb_ratio"] = price / (equity / shares)

        # P/S  (TTM ≈ trimestral × 4)
        rev = fund_snapshot.get("revenue", np.nan)
        if pd.notna(rev) and pd.notna(shares) and shares > 0 and rev > 0:
            f["ps_ratio"] = price / ((rev * 4) / shares)

        # Market Cap
        mktcap = price * shares if pd.notna(shares) and shares > 0 else np.nan

        # EV/EBITDA
        ebitda = fund_snapshot.get("ebitda", np.nan)
        debt   = fund_snapshot.get("total_debt", np.nan)
        cash   = fund_snapshot.get("cash", 0.0)
        if all(pd.notna(x) for x in [mktcap, ebitda, debt]) and ebitda > 0:
            ev = mktcap + debt - (cash or 0.0)
            f["ev_to_ebitda"] = ev / (ebitda * 4)

        # FCF Yield
        fcf = fund_snapshot.get("fcf", np.nan)
        if pd.notna(fcf) and pd.notna(mktcap) and mktcap > 0:
            f["fcf_yield"] = (fcf * 4) / mktcap

        # Comparativa vs historial propio (5 años)
        f.update(self._vs_history(f, hist_fund, prices_df, shares))

        # Señales de analistas
        if analyst_df is not None and not analyst_df.empty:
            f.update(self._analyst_features(analyst_df, as_of))

        return pd.Series(f)

    def _vs_history(self, current: Dict, hist: pd.DataFrame,
                    prices: pd.DataFrame, shares: float) -> Dict:
        out = {}
        cc  = "Close" if "Close" in prices.columns else prices.columns[3]
        close_series = prices[cc].sort_index()
        try:
            # P/E histórico (usa asof para encontrar el precio del día de trading más cercano)
            if "eps" in hist.columns and "pe_ratio" in current:
                hist_pe = []
                for dt, e in hist["eps"].dropna().items():
                    if e > 0:
                        p = close_series.asof(dt)
                        if pd.notna(p) and p > 0:
                            hist_pe.append(p / e)
                if len(hist_pe) >= 4:
                    med = float(np.nanmedian(hist_pe))
                    out["pe_vs_5y_median"] = current["pe_ratio"] / med - 1 if med else np.nan

            # P/B histórico
            if "total_equity" in hist.columns and "pb_ratio" in current:
                shr_hist = hist.get("shares_diluted", pd.Series(dtype=float))
                hist_pb  = []
                for dt, eq in hist["total_equity"].dropna().items():
                    sh = shr_hist.get(dt, np.nan) if isinstance(shr_hist, pd.Series) else np.nan
                    if pd.notna(sh) and sh > 0 and eq > 0:
                        p = close_series.asof(dt)
                        if pd.notna(p) and p > 0:
                            hist_pb.append(p / (eq / sh))
                if len(hist_pb) >= 4:
                    med = float(np.nanmedian(hist_pb))
                    out["pb_vs_5y_median"] = current.get("pb_ratio", np.nan) / med - 1 if med else np.nan

            # EV/EBITDA histórico
            if "ebitda" in hist.columns and "ev_to_ebitda" in current:
                hist_ev = []
                for dt, eb in hist["ebitda"].dropna().items():
                    if eb > 0:
                        p = close_series.asof(dt)
                        sh = hist["shares_diluted"].get(dt, np.nan) if "shares_diluted" in hist.columns else np.nan
                        d  = hist["total_debt"].get(dt, 0)          if "total_debt"     in hist.columns else 0
                        c  = hist["cash"].get(dt, 0)                if "cash"           in hist.columns else 0
                        if pd.notna(p) and p > 0 and pd.notna(sh) and sh > 0:
                            ev = p * sh + d - c
                            hist_ev.append(ev / (eb * 4))
                if len(hist_ev) >= 4:
                    med = float(np.nanmedian(hist_ev))
                    out["ev_ebitda_vs_5y_median"] = current.get("ev_to_ebitda", np.nan) / med - 1 if med else np.nan
        except Exception as e:
            log.debug(f"[ValuationFeat] comparativa histórica: {e}")
        return out

    @staticmethod
    def _analyst_features(analyst_df: pd.DataFrame, as_of: pd.Timestamp) -> Dict:
        available = analyst_df[analyst_df.index <= as_of]
        if available.empty:
            return {}
        f: Dict = {}
        last = available.iloc[-1]
        for col in ["eps_surprise_pct", "eps_revision", "eps_est", "eps_reported"]:
            if col in last.index and pd.notna(last[col]):
                f[col] = float(last[col])
        return f


# ── 4. Features de Insiders ───────────────────────────────────────────────────
class InsiderFeatureBuilder:
    """
    Procesa transacciones insider en una ventana temporal.

    Datos que consume:  data/insider/TICKER.csv
    Features que produce:
        insider_net_shares_90d   → compras netas (positivo = insider comprando)
        insider_sell_ratio       → proporción de ventas sobre total [0,1]
    """
    BUY_KW  = ["purchase","buy","acquisition","acquired","award"]
    SELL_KW = ["sale","sell","sold","disposition"]

    def build(self, insider_df: Optional[pd.DataFrame]) -> pd.Series:
        default = pd.Series({"insider_net_shares_90d": 0.0, "insider_sell_ratio": 0.5})
        if insider_df is None or insider_df.empty:
            return default

        txt = insider_df.get("transactionType", pd.Series(dtype=str)).str.lower().fillna("")
        is_buy  = txt.apply(lambda x: any(k in x for k in self.BUY_KW))
        is_sell = txt.apply(lambda x: any(k in x for k in self.SELL_KW))

        if "shares" in insider_df.columns:
            buys  = float(insider_df.loc[is_buy,  "shares"].sum())
            sells = float(insider_df.loc[is_sell, "shares"].sum())
        else:
            buys  = float(is_buy.sum())
            sells = float(is_sell.sum())

        total = buys + sells
        return pd.Series({
            "insider_net_shares_90d": buys - sells,
            "insider_sell_ratio":     sells / total if total > 0 else 0.5,
        })


# ── 5. Normalización Sectorial Z-score ───────────────────────────────────────
class SectorNormalizer:
    """
    Z-score relativo al sector para eliminar sesgos sectoriales en los ratios.
    Usa el sector de companies.csv.

    Ejemplo: ROE=0.25 en Utilities es excelente; en Tech es mediocre.
             La normalización sectorial lo captura correctamente.

    Uso:
        norm = SectorNormalizer()
        norm.fit(features_dict, sector_map)         # solo con datos de train
        X_normalized = norm.transform(features, sector)

    Añade columnas con sufijo '_zsector' (ej: roe_zsector) al Series original.
    """

    COLS = [
        "roe","roa","roi","roic","net_margin","gross_margin","fcf_margin",
        "ebitda_margin","operating_margin","current_ratio","quick_ratio",
        "debt_equity","debt_to_ebitda","pe_ratio","pb_ratio","ev_to_ebitda",
        "fcf_yield","revenue_yoy_growth","net_income_yoy_growth","eps_yoy_growth",
        "momentum_12m","volatility_20d","interest_coverage",
    ]

    def __init__(self, min_peers: int = 3):
        self.min_peers = min_peers
        self._stats: Dict[str, Dict[str, Tuple[float, float]]] = {}  # sector→col→(mean,std)

    def fit(
        self,
        features_dict: Dict[str, pd.Series],  # {ticker: features_series}
        sector_map:    Dict[str, str],         # {ticker: sector}  ← viene de companies.csv
    ) -> "SectorNormalizer":
        from collections import defaultdict
        buckets: Dict[str, List[pd.Series]] = defaultdict(list)
        for ticker, feat in features_dict.items():
            sector = sector_map.get(ticker, "Unknown")
            buckets[sector].append(feat)

        self._stats = {}
        for sector, feats in buckets.items():
            if len(feats) < self.min_peers:
                continue
            combined = pd.DataFrame(feats)
            self._stats[sector] = {}
            for col in self.COLS:
                if col not in combined.columns:
                    continue
                vals = combined[col].dropna()
                if len(vals) >= self.min_peers:
                    self._stats[sector][col] = (float(vals.mean()), float(vals.std()))
        return self

    def transform(self, features: pd.Series, sector: str) -> pd.Series:
        result = features.copy()
        stats  = self._stats.get(sector, {})
        for col in self.COLS:
            val = features.get(col, np.nan)
            if col not in stats or pd.isna(val):
                result[f"{col}_zsector"] = np.nan
                continue
            mean, std = stats[col]
            result[f"{col}_zsector"] = (val - mean) / std if std > 0 else 0.0
        return result

    def fit_transform(
        self,
        features_dict: Dict[str, pd.Series],
        sector_map:    Dict[str, str],
    ) -> Dict[str, pd.Series]:
        self.fit(features_dict, sector_map)
        return {
            t: self.transform(f, sector_map.get(t, "Unknown"))
            for t, f in features_dict.items()
        }
