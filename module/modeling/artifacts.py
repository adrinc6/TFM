"""Catálogo de artefactos activables del sistema.

Un "artefacto" es un bloque de features/contexto que se puede activar o desactivar por un flag
de `Settings`. El barrido de ablations los activa uno a uno para medir cuáles suben el rank-IC
del meta_final. Todos son **point-in-time**: cada feature en la fecha t usa solo datos
observables en t (sin lookahead), verificado con tests.

Cada artefacto expone una función `add_<nombre>(frame, ...)` que anade sus columnas crudas al
frame; el ranking a factores y el enganche a los agentes ocurre en `features.py`/`agents.py`.

Artefactos existentes (definidos en features.py por historia): momentum de fundamentales,
regimen bull/bear, neutralizacion por sector. Artefactos nuevos (aqui):
- momentum de precio multi-horizonte
- medias moviles / tendencia del activo
- regimen de mercado ampliado
- calidad/crecimiento derivados
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---- Momentum de precio multi-horizonte ----------------------------------------------------
# Columnas crudas que anade; cada una se rankea a un factor en features.py y va al agente momentum.
PRICE_MOMENTUM_SOURCES = ("mom_acceleration", "mom_reversal_1m", "mom_volatility")


def add_price_momentum_multi(frame: pd.DataFrame) -> None:
    """Aceleracion del momentum, reversion a corto y volatilidad reciente del activo.

    Todo se deriva de los retornos ya presentes en el panel (`price_return_*m`), que son
    point-in-time. No mira al futuro.

    - aceleracion = retorno 3m menos retorno 12m: el momentum reciente frente al de largo plazo.
    - reversion 1m = menos el retorno del ultimo mes: los que mas cayeron a 1 mes tienden a rebotar.
    - volatilidad = dispersion entre los retornos 1/3/6/12m como proxy de riesgo reciente.
    """
    r1 = pd.to_numeric(frame["price_return_1m"], errors="coerce")
    r3 = pd.to_numeric(frame["price_return_3m"], errors="coerce")
    r6 = pd.to_numeric(frame["price_return_6m"], errors="coerce")
    r12 = pd.to_numeric(frame["price_return_12m"], errors="coerce")
    frame["mom_acceleration"] = r3 - r12
    frame["mom_reversal_1m"] = -r1
    frame["mom_volatility"] = pd.concat([r1, r3, r6, r12], axis=1).std(axis=1)


# ---- Medias moviles / tendencia del activo -------------------------------------------------
MOVING_AVERAGE_SOURCES = ("ma_price_vs_sma6", "ma_price_vs_sma12", "ma_distance_to_high12")


def add_moving_averages(frame: pd.DataFrame, price_series: dict[str, tuple[list, list]]) -> None:
    """Precio vs sus medias moviles y distancia al maximo reciente (tendencia individual).

    `price_series` es {ticker: (fechas ordenadas, precios)} de la serie mensual PIT. Se computan
    las medias moviles sobre esa serie de forma vectorizada (rolling por ticker, incluyendo la
    fecha actual — la SMA es del pasado + hoy, sin lookahead) y se mapean a cada (ticker, fecha):
    - precio / SMA de los ultimos 6 y 12 meses - 1 (por encima de su media = tendencia alcista).
    - precio / maximo de los ultimos 12 meses - 1 (cerca de maximos = fuerza).
    """
    rows = []
    for ticker, (dates, prices) in price_series.items():
        s = pd.DataFrame({"ticker": ticker, "snapshot_date": list(dates),
                          "price": pd.to_numeric(pd.Series(prices), errors="coerce")})
        p = s["price"]
        s["ma_price_vs_sma6"] = p / p.rolling(6, min_periods=1).mean() - 1
        s["ma_price_vs_sma12"] = p / p.rolling(12, min_periods=1).mean() - 1
        s["ma_distance_to_high12"] = p / p.rolling(12, min_periods=1).max() - 1
        rows.append(s.drop(columns="price"))
    if not rows:
        for col in MOVING_AVERAGE_SOURCES:
            frame[col] = np.nan
        return
    ma = pd.concat(rows, ignore_index=True)
    merged = frame.merge(ma, on=["ticker", "snapshot_date"], how="left")
    for col in MOVING_AVERAGE_SOURCES:
        frame[col] = merged[col].to_numpy()


# ---- Regimen de mercado ampliado -----------------------------------------------------------
# El regimen es el mismo para todos los tickers de un snapshot; las columnas se anaden por fila.
REGIME_EXTENDED_SOURCES = ("regime_sp500_vol", "regime_sp500_drawdown")


def add_regime_extended(frame: pd.DataFrame, benchmark: pd.DataFrame) -> None:
    """Volatilidad y drawdown del SP500 (contexto macro), sin lookahead.

    Se calcula de la serie del benchmark hasta cada snapshot: la volatilidad de los retornos
    recientes y el drawdown del indice desde su maximo. Se mapea a cada fila por fecha.
    """
    bench = benchmark.sort_values("snapshot_date").copy()
    ret = pd.to_numeric(bench["price"], errors="coerce").pct_change()
    bench["regime_sp500_vol"] = ret.rolling(6, min_periods=2).std()
    running_max = pd.to_numeric(bench["price"], errors="coerce").cummax()
    bench["regime_sp500_drawdown"] = pd.to_numeric(bench["price"], errors="coerce") / running_max - 1
    lookup_vol = dict(zip(bench["snapshot_date"], bench["regime_sp500_vol"]))
    lookup_dd = dict(zip(bench["snapshot_date"], bench["regime_sp500_drawdown"]))
    frame["regime_sp500_vol"] = frame["snapshot_date"].map(lookup_vol)
    frame["regime_sp500_drawdown"] = frame["snapshot_date"].map(lookup_dd)


# ---- Calidad / crecimiento derivados -------------------------------------------------------
QUALITY_GROWTH_SOURCES = ("qg_roe_trend", "qg_margin_stability", "qg_growth_surprise")

# Técnicos diarios calculados de OHLCV histórico, siempre hasta la fecha de snapshot.
MARKET_RISK_SOURCES = (
    "momentum_12_1", "realized_vol_63d", "realized_vol_126d", "downside_vol_63d", "beta_252d",
    "drawdown_252d", "max_drawdown_252d", "range_21d", "range_63d", "volume_relative",
    "volume_volatility", "amihud_illiquidity", "gap_21d",
)


def add_market_risk_liquidity(
    frame: pd.DataFrame, prices: pd.DataFrame | None, benchmark_ticker: str,
    risk_windows: tuple[int, ...] = (63, 126, 252), technical_windows: tuple[int, ...] = (21, 63, 252),
) -> None:
    """Añade factores técnicos diarios con ``merge_asof`` por ticker, sin mirar al futuro."""
    if prices is None or prices.empty or not {"ticker", "date", "adj_close"} <= set(prices):
        for name in MARKET_RISK_SOURCES:
            frame[name] = np.nan
        return
    raw = prices.copy()
    risk_short, risk_long, beta_window = (list(risk_windows) + [63, 126, 252])[:3]
    tech_short, tech_medium, tech_long = (list(technical_windows) + [21, 63, 252])[:3]
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    for column in ("adj_close", "open", "high", "low", "volume"):
        raw[column] = pd.to_numeric(raw.get(column), errors="coerce")
    raw = raw.dropna(subset=["ticker", "date", "adj_close"]).sort_values(["ticker", "date"])
    benchmark = raw.loc[raw["ticker"] == benchmark_ticker, ["date", "adj_close"]].rename(
        columns={"adj_close": "benchmark_close"}).sort_values("date")
    daily: list[pd.DataFrame] = []
    for ticker, group in raw.groupby("ticker", sort=False):
        if ticker == benchmark_ticker:
            continue
        g = group.copy().sort_values("date")
        g["ret"] = g["adj_close"].pct_change()
        g["momentum_12_1"] = g["adj_close"].shift(tech_short) / g["adj_close"].shift(tech_long) - 1
        g["realized_vol_63d"] = g["ret"].rolling(risk_short, min_periods=max(15, risk_short // 3)).std() * np.sqrt(252)
        g["realized_vol_126d"] = g["ret"].rolling(risk_long, min_periods=max(15, risk_long // 3)).std() * np.sqrt(252)
        g["downside_vol_63d"] = g["ret"].where(g["ret"] < 0).rolling(risk_short, min_periods=15).std() * np.sqrt(252)
        peak = g["adj_close"].rolling(beta_window, min_periods=tech_short).max()
        g["drawdown_252d"] = g["adj_close"] / peak - 1
        g["max_drawdown_252d"] = g["drawdown_252d"].rolling(beta_window, min_periods=tech_short).min()
        intraday = (g["high"] - g["low"]) / g["adj_close"]
        g["range_21d"] = intraday.rolling(tech_short, min_periods=max(5, tech_short // 3)).mean()
        g["range_63d"] = intraday.rolling(tech_medium, min_periods=max(5, tech_medium // 3)).mean()
        g["volume_relative"] = g["volume"].rolling(tech_short, min_periods=5).mean() / g["volume"].rolling(tech_long, min_periods=tech_short).mean() - 1
        g["volume_volatility"] = np.log(g["volume"].where(g["volume"] > 0)).rolling(tech_medium, min_periods=15).std()
        g["amihud_illiquidity"] = (g["ret"].abs() / (g["adj_close"] * g["volume"])).rolling(tech_short, min_periods=5).mean()
        g["gap_21d"] = (g["open"] / g["adj_close"].shift() - 1).abs().rolling(tech_short, min_periods=5).mean()
        g["beta_252d"] = np.nan  # se rellena tras sincronizar la serie del benchmark.
        daily.append(g[["ticker", "date", *MARKET_RISK_SOURCES]])
    if not daily:
        for name in MARKET_RISK_SOURCES:
            frame[name] = np.nan
        return
    tech = pd.concat(daily, ignore_index=True).sort_values(["date", "ticker"])
    snapshots = frame[["ticker", "snapshot_date"]].copy()
    snapshots["_row"] = np.arange(len(snapshots))
    snapshots["snapshot_date"] = pd.to_datetime(snapshots["snapshot_date"])
    merged = pd.merge_asof(snapshots.sort_values(["snapshot_date", "ticker"]), tech,
                           left_on="snapshot_date", right_on="date", by="ticker", direction="backward")
    merged = merged.sort_values("_row")
    for name in MARKET_RISK_SOURCES:
        frame[name] = merged[name].to_numpy()
    # Beta necesita retornos de acción y benchmark sincronizados; se calcula tras el merge diario.
    beta_rows = []
    for ticker, group in raw.loc[raw["ticker"] != benchmark_ticker].groupby("ticker", sort=False):
        g = pd.merge_asof(group[["date", "adj_close"]].sort_values("date"), benchmark,
                          on="date", direction="backward").dropna()
        returns = pd.DataFrame({"asset": g["adj_close"].pct_change(), "benchmark": g["benchmark_close"].pct_change()})
        beta = returns["asset"].rolling(beta_window, min_periods=risk_short).cov(returns["benchmark"]) / returns["benchmark"].rolling(beta_window, min_periods=risk_short).var()
        beta_rows.append(pd.DataFrame({"ticker": ticker, "date": g["date"], "beta_252d": beta}))
    if beta_rows:
        beta = pd.concat(beta_rows).sort_values(["date", "ticker"])
        beta_join = pd.merge_asof(snapshots.sort_values(["snapshot_date", "ticker"]), beta,
                                  left_on="snapshot_date", right_on="date", by="ticker", direction="backward")
        frame["beta_252d"] = beta_join.sort_values("_row")["beta_252d"].to_numpy()


def add_quality_growth_derived(frame: pd.DataFrame) -> None:
    """Tendencia y estabilidad de los fundamentales, point-in-time por ticker.

    Se calcula sobre la secuencia de valores que la empresa fue publicando (el panel ya trae el
    fundamental observable en cada snapshot), comparando cada fila con el pasado del MISMO ticker:
    - roe_trend: cambio del ROE frente a su media de los ultimos 4 snapshots (mejora sostenida).
    - margin_stability: -desviacion del net_margin en los ultimos 4 (menos volatil = mas calidad).
    - growth_surprise: eps_growth_yoy actual frente a su media reciente (acelera o desacelera).

    Se asume que `frame` viene ordenado por (ticker, snapshot_date), como sale del panel PIT.
    Los `rolling` usan `shift()` para no incluir el valor actual (solo pasado del mismo ticker).
    """
    ticker = frame["ticker"]
    for col, src in (("qg_roe_trend", "roe"), ("qg_growth_surprise", "eps_growth_yoy")):
        values = pd.to_numeric(frame[src], errors="coerce")
        past_mean = values.groupby(ticker).transform(
            lambda s: s.shift().rolling(4, min_periods=2).mean()
        )
        frame[col] = values - past_mean
    margin = pd.to_numeric(frame["net_margin"], errors="coerce")
    frame["qg_margin_stability"] = -margin.groupby(ticker).transform(
        lambda s: s.shift().rolling(4, min_periods=2).std()
    )
