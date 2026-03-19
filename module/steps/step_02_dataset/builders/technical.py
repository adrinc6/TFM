"""Technical indicator feature builder."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class TechnicalFeatureBuilder:
    """Calcula indicadores tecnicos sobre ventana OHLCV diaria."""

    def build(self, prices_df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        df = prices_df[prices_df.index <= as_of].copy()
        if len(df) < 20:
            return pd.Series(dtype=float)

        close = df["Close"] if "Close" in df.columns else df.iloc[:, 3]
        high = df["High"] if "High" in df.columns else df.iloc[:, 1]
        low = df["Low"] if "Low" in df.columns else df.iloc[:, 2]
        vol = df["Volume"] if "Volume" in df.columns else df.iloc[:, 4]
        f: Dict = {}

        f["rsi_14"] = self._rsi(close, 14)
        f["rsi_28"] = self._rsi(close, 28)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        sig = macd.ewm(span=9, adjust=False).mean()
        f["macd"] = float(macd.iloc[-1])
        f["macd_signal"] = float(sig.iloc[-1])
        f["macd_hist"] = float((macd - sig).iloc[-1])

        p = float(close.iloc[-1])
        for w in [20, 50, 200]:
            if len(close) >= w:
                sma = float(close.rolling(w).mean().iloc[-1])
                f[f"sma_{w}"] = (p / sma - 1) if sma != 0 else 0.0

        if len(close) >= 20:
            s20 = close.rolling(20).mean()
            d20 = close.rolling(20).std()
            band_width = (4 * d20).replace(0, np.nan)
            f["bb_pct"] = float(((close - (s20 - 2 * d20)) / band_width).iloc[-1])

        lk252 = close.tail(252)
        if len(lk252) > 10:
            f["price_vs_52w_high"] = float(p / lk252.max() - 1)
            f["price_vs_52w_low"] = float(p / lk252.min() - 1)

        for days, name in [
            (21, "momentum_1m"),
            (63, "momentum_3m"),
            (126, "momentum_6m"),
            (252, "momentum_12m"),
        ]:
            if len(close) > days:
                f[name] = float(p / close.iloc[-(days + 1)] - 1)

        ret = close.pct_change().dropna()
        for w in [20, 60]:
            if len(ret) >= w:
                f[f"volatility_{w}d"] = float(ret.tail(w).std() * np.sqrt(252))

        if len(df) >= 14:
            f["atr_14"] = self._atr(high, low, close, 14)

        if len(vol) >= 50:
            v20 = vol.tail(20).mean()
            v50 = vol.tail(50).mean()
            if v50 > 0:
                f["vol_ratio_20_50"] = float(v20 / v50)

        return pd.Series(f)

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> float:
        delta = close.diff().dropna()
        gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        return float(rsi.iloc[-1]) if not rsi.empty else np.nan

    @staticmethod
    def _atr(high, low, close, period) -> float:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return float(tr.ewm(com=period - 1, adjust=False).mean().iloc[-1])
