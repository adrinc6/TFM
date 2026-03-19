# =============================================================================
# module/feature_engineering.py
# Builders de features para cada agente + normalización sectorial Z-score.
# Fuente de datos: Finnhub (data_finnhub/) via DataRouter.
# =============================================================================
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ── 1. Features Fundamentales ─────────────────────────────────────────────────
class FundamentalFeatureBuilder:
    """
    Enriquece el DataFrame consolidado (output de FinnhubConsolidator)
    con ratios y señales adicionales.

    Datos que consume (columnas del consolidado):
        revenue, net_income, operating_income, gross_profit,
        total_assets, total_equity, total_debt, fcf,
        operating_cash_flow, interest_expense, capex,
        ebitda, eps / eps_diluted, current_ratio, debt_equity, debt_to_ebitda,
        bf_roe, bf_roa, bf_gross_margin, bf_net_margin (series basic_financials)

    Ratios que produce:
        Rentabilidad:  roe, roa, roi, roic, net_margin, gross_margin,
                       fcf_margin, ebitda_margin, operating_margin
        Solvencia:     debt_equity, debt_to_ebitda, interest_coverage
        Crecimiento:   revenue_yoy_growth, net_income_yoy_growth,
                       eps_yoy_growth, fcf_yoy_growth,
                       operating_income_yoy_growth, total_debt_yoy_growth
        Calidad:       accruals_ratio, capex_to_revenue, earnings_quality
        Tendencias:    roe_trend_3y, gross_margin_trend_3y (pendiente bf_series)
        Riesgo:        consecutive_losses, revenue_decline
    """

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._yoy_growth(df)
        df = self._quality_metrics(df)
        df = self._coverage_ratios(df)
        df = self._risk_flags(df)
        df = self._trend_features(df)
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
        if "accruals_ratio" not in df.columns:
            if {"net_income", "operating_cash_flow", "total_assets"}.issubset(df.columns):
                df["accruals_ratio"] = (
                    (df["net_income"] - df["operating_cash_flow"])
                    / df["total_assets"].replace(0, np.nan)
                )

        if "capex" in df.columns and "revenue" in df.columns:
            df["capex_to_revenue"] = df["capex"].abs() / df["revenue"].replace(0, np.nan)

        # Earnings quality: FCF / Net Income (>1 = muy buena calidad, <0 = preocupante)
        if "fcf" in df.columns and "net_income" in df.columns:
            df["earnings_quality"] = df["fcf"] / df["net_income"].replace(0, np.nan)

        if "ebitda_margin" not in df.columns:
            if "ebitda" in df.columns and "revenue" in df.columns:
                df["ebitda_margin"] = df["ebitda"] / df["revenue"].replace(0, np.nan)

        if "operating_margin" not in df.columns:
            if "operating_income" in df.columns and "revenue" in df.columns:
                df["operating_margin"] = df["operating_income"] / df["revenue"].replace(0, np.nan)

        return df

    def _coverage_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        if "interest_coverage" not in df.columns:
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

    def _trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula tendencia (pendiente de regresión lineal normalizada) para
        métricas clave usando series de basic_financials si están disponibles.
        Una pendiente positiva indica mejora sostenida.
        """
        def _slope(series: pd.Series, n: int = 8) -> float:
            """Pendiente normalizada de los últimos n períodos."""
            vals = series.dropna().tail(n)
            if len(vals) < 3:
                return np.nan
            x = np.arange(len(vals), dtype=float)
            y = vals.values.astype(float)
            if y.std() == 0:
                return 0.0
            coeffs = np.polyfit(x, y / (abs(y.mean()) + 1e-10), 1)
            return float(coeffs[0])

        for col, feat in [
            ("bf_roe",          "roe_trend_3y"),
            ("bf_gross_margin", "gross_margin_trend_3y"),
            ("bf_net_margin",   "net_margin_trend_3y"),
            ("roe",             "roe_trend_2y"),
            ("net_margin",      "net_margin_trend_2y"),
        ]:
            if col in df.columns:
                df[feat] = df[col].expanding(3).apply(
                    lambda s: _slope(s, n=8), raw=False
                )

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
        for days, name in [(21, "momentum_1m"), (63, "momentum_3m"),
                           (126, "momentum_6m"), (252, "momentum_12m")]:
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
            v20 = vol.tail(20).mean()
            v50 = vol.tail(50).mean()
            if v50 > 0:
                f["vol_ratio_20_50"] = float(v20 / v50)

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
    Enriquecido con datos de Finnhub: revenue surprises, beat rate,
    series de consenso de analistas.

    Datos que consume:
        prices_df        → precio de cierre en as_of
        fund_snapshot    → última fila de fundamentales disponible
        hist_fund        → histórico de fundamentales (para comparativa 5Y)
        analyst_df       → EPS surprises (eps_surprises.json)
        recommendation_df → consenso analistas (recommendation_trends.json)

    Features que produce:
        Múltiplos:       pe_ratio, pb_ratio, ps_ratio, ev_to_ebitda,
                         fcf_yield, earnings_yield
        Vs historial:    pe_vs_5y_median, pb_vs_5y_median, ev_ebitda_vs_5y_median
        Analistas EPS:   eps_surprise_pct, eps_revision, beat_rate_4q,
                         eps_surprise_avg_4q
        Analistas rec:   analyst_buy_ratio, analyst_bearish_score,
                         analyst_consensus, analyst_dispersion,
                         analyst_consensus_change
        Finnhub ratios:  bf_ev_ebitda, bf_fcf_yield, bf_pe (si disponibles)
    """

    def build(
        self,
        prices_df:         pd.DataFrame,
        fund_snapshot:     pd.Series,
        hist_fund:         pd.DataFrame,
        as_of:             pd.Timestamp,
        analyst_df:        Optional[pd.DataFrame] = None,
        recommendation_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        f: Dict = {}

        # Precio actual
        recent = prices_df[prices_df.index <= as_of]
        if recent.empty:
            return pd.Series(dtype=float)
        cc    = "Close" if "Close" in recent.columns else recent.columns[0]
        price = float(recent[cc].iloc[-1])

        shares = fund_snapshot.get("shares_diluted", np.nan)

        # P/E y Earnings Yield
        eps = fund_snapshot.get("eps", fund_snapshot.get("eps_diluted", np.nan))
        if pd.notna(eps) and eps > 0:
            f["pe_ratio"]      = price / eps
            f["earnings_yield"] = eps / price

        # P/B
        equity = fund_snapshot.get("total_equity", np.nan)
        if pd.notna(equity) and pd.notna(shares) and shares > 0 and equity > 0:
            f["pb_ratio"] = price / (equity / shares)

        # P/S (TTM ≈ trimestral × 4)
        rev = fund_snapshot.get("revenue", np.nan)
        if pd.notna(rev) and pd.notna(shares) and shares > 0 and rev > 0:
            f["ps_ratio"] = price / ((rev * 4) / shares)

        # Market Cap
        mktcap = price * shares if pd.notna(shares) and shares > 0 else np.nan

        # EV/EBITDA
        ebitda = fund_snapshot.get("ebitda", np.nan)
        debt   = fund_snapshot.get("total_debt", np.nan)
        cash   = fund_snapshot.get("cash", 0.0) or 0.0
        if all(pd.notna(x) for x in [mktcap, ebitda, debt]) and ebitda > 0:
            ev = mktcap + debt - cash
            f["ev_to_ebitda"] = ev / (ebitda * 4)

        # FCF Yield
        fcf = fund_snapshot.get("fcf", np.nan)
        if pd.notna(fcf) and pd.notna(mktcap) and mktcap > 0:
            f["fcf_yield"] = (fcf * 4) / mktcap

        # Ratios de Finnhub basic_financials (más fiables cuando disponibles)
        for bf_col, feat_col in [
            ("bf_ev_ebitda", "bf_ev_ebitda"),
            ("bf_fcf_yield", "bf_fcf_yield"),
            ("bf_pe",        "bf_pe"),
            ("bf_pb",        "bf_pb_annual"),
            ("bf_ps",        "bf_ps_ttm"),
        ]:
            val = fund_snapshot.get(bf_col, np.nan)
            if pd.notna(val):
                f[feat_col] = float(val)

        # Comparativa vs historial propio (5 años)
        f.update(self._vs_history(f, hist_fund, prices_df, shares))

        # Señales de EPS surprises (Finnhub eps_surprises.json)
        if analyst_df is not None and not analyst_df.empty:
            f.update(self._eps_surprise_features(analyst_df, as_of))

        # Señales de consenso de analistas (recommendation_trends.json)
        if recommendation_df is not None and not recommendation_df.empty:
            f.update(self._recommendation_features(recommendation_df, as_of))

        return pd.Series(f)

    def _vs_history(self, current: Dict, hist: pd.DataFrame,
                    prices: pd.DataFrame, shares: float) -> Dict:
        out = {}
        cc  = "Close" if "Close" in prices.columns else prices.columns[0]
        close_series = prices[cc].sort_index()
        try:
            eps_col = "eps" if "eps" in hist.columns else "eps_diluted"
            if eps_col in hist.columns and "pe_ratio" in current:
                hist_pe = []
                for dt, e in hist[eps_col].dropna().items():
                    if e > 0:
                        p = close_series.asof(dt)
                        if pd.notna(p) and p > 0:
                            hist_pe.append(p / e)
                if len(hist_pe) >= 4:
                    med = float(np.nanmedian(hist_pe))
                    out["pe_vs_5y_median"] = current["pe_ratio"] / med - 1 if med else np.nan

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

            if "ebitda" in hist.columns and "ev_to_ebitda" in current:
                hist_ev = []
                for dt, eb in hist["ebitda"].dropna().items():
                    if eb > 0:
                        p  = close_series.asof(dt)
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
    def _eps_surprise_features(analyst_df: pd.DataFrame, as_of: pd.Timestamp) -> Dict:
        """
        Features derivadas de eps_surprises.json (Finnhub).
        Columnas esperadas: eps_actual, eps_estimate, eps_surprise_pct, eps_beat
        """
        available = analyst_df[analyst_df.index <= as_of]
        if available.empty:
            return {}
        f: Dict = {}

        # Último trimestre
        last = available.iloc[-1]
        for col in ["eps_surprise_pct", "eps_actual", "eps_estimate"]:
            if col in last.index and pd.notna(last[col]):
                f[col] = float(last[col])

        # Últimos 4 trimestres: beat rate y sorpresa media
        last4 = available.tail(4)
        if "eps_beat" in last4.columns:
            f["beat_rate_4q"] = float(last4["eps_beat"].mean())
        if "eps_surprise_pct" in last4.columns:
            f["eps_surprise_avg_4q"] = float(last4["eps_surprise_pct"].mean())

        # Variación de estimación (revisión): último vs penúltimo estimate
        if "eps_estimate" in available.columns and len(available) >= 2:
            e_last = available["eps_estimate"].dropna()
            if len(e_last) >= 2:
                f["eps_revision"] = float(e_last.iloc[-1] / e_last.iloc[-2] - 1) if e_last.iloc[-2] != 0 else np.nan

        return f

    @staticmethod
    def _recommendation_features(rec_df: pd.DataFrame, as_of: pd.Timestamp) -> Dict:
        """
        Features derivadas de recommendation_trends.json (Finnhub).
        Columnas esperadas: analyst_buy_ratio, analyst_bearish_score,
                            analyst_consensus, analyst_dispersion, analyst_total
        """
        available = rec_df[rec_df.index <= as_of]
        if available.empty:
            return {}
        f: Dict = {}

        last = available.iloc[-1]
        for col in ["analyst_buy_ratio", "analyst_bearish_score",
                    "analyst_consensus", "analyst_dispersion", "analyst_strong_buy_pct"]:
            if col in last.index and pd.notna(last[col]):
                f[col] = float(last[col])

        # Cambio de consenso (momentum del consenso)
        if "analyst_consensus" in available.columns and len(available) >= 2:
            c_now  = available["analyst_consensus"].dropna()
            if len(c_now) >= 2:
                f["analyst_consensus_change"] = float(c_now.iloc[-1] - c_now.iloc[-2])

        return f


# ── 4. Features de Insiders ───────────────────────────────────────────────────
class InsiderFeatureBuilder:
    """
    Procesa transacciones insider y MSPR mensual (Finnhub).

    Datos que consume:
        insider_df      → insider_transactions.json (columnas: date, shares, is_buy, is_sell)
        mspr_df         → insider_sentiment.json (columnas: mspr, insider_net_buy)

    Features que produce:
        insider_net_shares_90d  → compras netas (positivo = insider comprando)
        insider_sell_ratio      → proporción de ventas sobre total [0,1]
        mspr_3m                 → MSPR promedio últimos 3 meses (-100 a 100)
        mspr_trend              → cambio de MSPR (último - 3 meses atrás)
    """

    def build(
        self,
        insider_df: Optional[pd.DataFrame] = None,
        mspr_df:    Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        f: Dict = {
            "insider_net_shares_90d": 0.0,
            "insider_sell_ratio":     0.5,
            "mspr_3m":                np.nan,
            "mspr_trend":             np.nan,
        }

        # Transacciones insider
        if insider_df is not None and not insider_df.empty:
            buys  = float(insider_df.loc[insider_df["is_buy"]  == 1, "shares"].sum())
            sells = float(insider_df.loc[insider_df["is_sell"] == 1, "shares"].sum())
            total = buys + sells
            f["insider_net_shares_90d"] = buys - sells
            f["insider_sell_ratio"]     = sells / total if total > 0 else 0.5

        # MSPR mensual (insider sentiment)
        if mspr_df is not None and not mspr_df.empty and "mspr" in mspr_df.columns:
            mspr_vals = mspr_df["mspr"].dropna()
            if len(mspr_vals) >= 1:
                f["mspr_3m"] = float(mspr_vals.tail(3).mean())
            if len(mspr_vals) >= 3:
                f["mspr_trend"] = float(mspr_vals.iloc[-1] - mspr_vals.iloc[-3])

        return pd.Series(f)


# ── 5. Features de Sentiment (analistas + insiders) ───────────────────────────
class SentimentFeatureBuilder:
    """
    Construye el vector de features para el SentimentAgent.
    Combina señales de analistas (recomendaciones) y sentimiento de insiders (MSPR).

    Features que produce:
        analyst_buy_ratio, analyst_bearish_score, analyst_consensus,
        analyst_dispersion, analyst_strong_buy_pct, analyst_consensus_change,
        mspr_3m, mspr_trend, insider_net_shares_90d, insider_sell_ratio,
        beat_rate_4q, eps_surprise_avg_4q, eps_surprise_pct
    """

    def build(
        self,
        recommendation_df: Optional[pd.DataFrame],
        mspr_df:           Optional[pd.DataFrame],
        insider_df:        Optional[pd.DataFrame],
        eps_df:            Optional[pd.DataFrame],
        as_of:             pd.Timestamp,
    ) -> pd.Series:
        f: Dict = {}

        # Señales de analistas (recomendaciones)
        if recommendation_df is not None and not recommendation_df.empty:
            available = recommendation_df[recommendation_df.index <= as_of]
            if not available.empty:
                last = available.iloc[-1]
                for col in ["analyst_buy_ratio", "analyst_bearish_score",
                            "analyst_consensus", "analyst_dispersion",
                            "analyst_strong_buy_pct"]:
                    if col in last.index and pd.notna(last[col]):
                        f[col] = float(last[col])
                if "analyst_consensus" in available.columns and len(available) >= 2:
                    c = available["analyst_consensus"].dropna()
                    if len(c) >= 2:
                        f["analyst_consensus_change"] = float(c.iloc[-1] - c.iloc[-2])

        # MSPR (insider sentiment)
        if mspr_df is not None and not mspr_df.empty and "mspr" in mspr_df.columns:
            available_mspr = mspr_df[mspr_df.index <= as_of]
            if not available_mspr.empty:
                mspr_vals = available_mspr["mspr"].dropna()
                if len(mspr_vals) >= 1:
                    f["mspr_3m"] = float(mspr_vals.tail(3).mean())
                if len(mspr_vals) >= 3:
                    f["mspr_trend"] = float(mspr_vals.iloc[-1] - mspr_vals.iloc[-3])

        # Transacciones insider
        if insider_df is not None and not insider_df.empty:
            buys  = float(insider_df.loc[insider_df["is_buy"]  == 1, "shares"].sum())
            sells = float(insider_df.loc[insider_df["is_sell"] == 1, "shares"].sum())
            total = buys + sells
            f["insider_net_shares_90d"] = buys - sells
            f["insider_sell_ratio"]     = sells / total if total > 0 else 0.5

        # EPS surprises
        if eps_df is not None and not eps_df.empty:
            available_eps = eps_df[eps_df.index <= as_of]
            if not available_eps.empty:
                last = available_eps.iloc[-1]
                if "eps_surprise_pct" in last.index and pd.notna(last["eps_surprise_pct"]):
                    f["eps_surprise_pct"] = float(last["eps_surprise_pct"])
                last4 = available_eps.tail(4)
                if "eps_beat" in last4.columns:
                    f["beat_rate_4q"] = float(last4["eps_beat"].mean())
                if "eps_surprise_pct" in last4.columns:
                    f["eps_surprise_avg_4q"] = float(last4["eps_surprise_pct"].mean())

        return pd.Series(f)


# ── 6. Normalización Sectorial Z-score ───────────────────────────────────────
class SectorNormalizer:
    """
    Z-score relativo al sector para eliminar sesgos sectoriales en los ratios.
    El sector proviene de los profiles Finnhub via DataRouter.

    Ejemplo: ROE=0.25 en Utilities es excelente; en Tech es mediocre.
             La normalización sectorial lo captura correctamente.

    Uso:
        norm = SectorNormalizer()
        norm.fit(features_dict, sector_map)
        X_normalized = norm.transform(features, sector)
    """

    COLS = [
        "roe", "roa", "roi", "roic", "net_margin", "gross_margin", "fcf_margin",
        "ebitda_margin", "operating_margin", "current_ratio",
        "debt_equity", "debt_to_ebitda", "pe_ratio", "pb_ratio", "ev_to_ebitda",
        "fcf_yield", "revenue_yoy_growth", "net_income_yoy_growth", "eps_yoy_growth",
        "momentum_12m", "volatility_20d", "interest_coverage",
        "analyst_buy_ratio", "analyst_consensus", "mspr_3m", "beat_rate_4q",
    ]

    def __init__(self, min_peers: int = 3):
        self.min_peers = min_peers
        self._stats: Dict[str, Dict[str, Tuple[float, float]]] = {}

    def fit(
        self,
        features_dict: Dict[str, pd.Series],
        sector_map:    Dict[str, str],
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
