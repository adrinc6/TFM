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
