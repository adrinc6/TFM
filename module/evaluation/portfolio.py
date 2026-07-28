"""Reglas causales de la cartera dinámica, gobernadas por alfa esperado en puntos básicos.

Toda decisión de compra, venta y rotación se toma comparando **magnitudes económicas** —el alfa
esperado calibrado de cada acción, en puntos básicos— contra umbrales también económicos. La versión
anterior comparaba percentiles del meta-rank contra percentiles fijos, lo que tiene dos defectos que
esta no tiene: un percentil no dice cuánto se espera ganar, y una rotación decidida por percentiles
puede destruir valor aunque el candidato sea mejor, porque el coste de operar no entra en la cuenta.

Aquí una rotación solo se emite si la ventaja de alfa esperado **supera el coste de ida y vuelta de
la propia operación** más un margen. El ranking del meta sigue decidiendo el orden de preferencia; lo
que cambia es que el umbral tiene unidades de rentabilidad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from environment import Settings


BPS = 10_000.0


@dataclass
class PortfolioState:
    holdings: dict[str, float] = field(default_factory=dict)
    entry_dates: dict[str, str] = field(default_factory=dict)
    entry_prices: dict[str, float] = field(default_factory=dict)
    units: dict[str, float] = field(default_factory=dict)
    entry_costs: dict[str, float] = field(default_factory=dict)
    costs_paid: float = 0.0

    @classmethod
    def empty(cls) -> PortfolioState:
        return cls()


def decide_orders(
    state: PortfolioState, scores_at_date: pd.DataFrame, settings: Settings,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Aplica expulsión por alfa, sustitución que paga su coste y rebalanceo con tolerancia.

    Devuelve las órdenes y los pesos objetivo. **Los pesos pueden sumar menos de 1**: el residuo es
    efectivo, y solo aparece bajo ``cash_policy="opportunity_cash"`` cuando no hay candidatas por
    encima del umbral de alfa esperado. Con ``fully_invested`` el residuo es cero por construcción.

    El alfa esperado procede de la calibración isotónica causal (``expected_excess_return``) y es
    ``NaN`` mientras no haya suficientes cohortes cerradas para calibrar. Un ``NaN`` nunca activa una
    venta ni bloquea una compra: la regla es "actuar solo ante evidencia económica", de modo que
    durante el arranque manda la ordenación y, en cuanto hay calibración, mandan los umbrales.
    """
    if scores_at_date.empty:
        return [], dict(state.holdings)
    date = str(scores_at_date["snapshot_date"].iloc[0])
    strictness = _strictness(scores_at_date, settings)
    ranked = scores_at_date.dropna(subset=["meta_rank"]).copy()
    ranked["percentile"] = pd.to_numeric(ranked["meta_rank"], errors="coerce") * 100
    ranked = ranked.dropna(subset=["percentile"]).sort_values("percentile", ascending=False)
    percentile = dict(zip(ranked["ticker"].astype(str), ranked["percentile"]))
    alpha_bps = _expected_alpha_bps(ranked)

    # Endurecer en snapshots sin fundamentales nuevos significa vender menos y rotar menos: el
    # umbral de salida baja (cuesta más gatillar una venta) y la ventaja exigida para rotar sube.
    exit_threshold = settings.exit_expected_alpha_bps / strictness
    round_trip = 2.0 * (settings.commission_bps + settings.slippage_bps)
    rotation_threshold = round_trip + settings.rotation_edge_bps * strictness

    holders = set(state.holdings)
    orders: list[dict[str, Any]] = []
    for ticker in sorted(holders):
        if _below(alpha_bps.get(ticker), exit_threshold):
            holders.remove(ticker)
            orders.append(_order(date, ticker, "sell", "expected_alpha_below_exit", state.holdings.get(ticker, 0.0), 0.0))

    admissible = [
        ticker for ticker in ranked["ticker"].astype(str)
        if not _below(alpha_bps.get(ticker), exit_threshold)
    ]
    for ticker in admissible:
        if len(holders) >= settings.target_size:
            break
        if ticker not in holders:
            holders.add(ticker)
            orders.append(_order(date, ticker, "buy", "initial_fill", 0.0, None))

    # Con la política de oportunidad, las plazas que no se han podido cubrir con candidatas por
    # encima del umbral quedan vacías y se traducen en efectivo. Con `fully_invested` se rellenan
    # con las mejores disponibles aunque no superen el umbral.
    if settings.cash_policy == "fully_invested":
        for ticker in ranked["ticker"].astype(str):
            if len(holders) >= settings.target_size:
                break
            holders.add(ticker)
            orders.append(_order(date, ticker, "buy", "fully_invested_fill", 0.0, None))

    while len(holders) >= settings.target_size:
        outsider = next((ticker for ticker in admissible if ticker not in holders), None)
        if outsider is None:
            break
        worst = min(holders, key=lambda ticker: _alpha_or(alpha_bps, ticker, percentile))
        advantage = _advantage(alpha_bps, outsider, worst)
        if advantage is None or advantage < rotation_threshold:
            break
        holders.remove(worst)
        holders.add(outsider)
        orders.extend((_order(date, worst, "sell", "displaced_by_net_edge", state.holdings.get(worst, 0.0), 0.0),
                       _order(date, outsider, "buy", "net_edge_over_worst", 0.0, None)))

    invested = _invested_fraction(len(holders), settings)
    target = _weights(
        sorted(holders, key=lambda ticker: -percentile.get(ticker, -1.0)),
        alpha_bps, invested, settings,
    )
    target = _apply_rebalance_tolerance(state.holdings, target, settings.rebalance_drift_tolerance * strictness)
    planned = {order["ticker"] for order in orders}
    for ticker, weight in target.items():
        current = state.holdings.get(ticker, 0.0)
        if abs(weight - current) > 1e-12 and ticker not in planned:
            orders.append(_order(date, ticker, "buy" if weight > current else "sell", "rebalance", current, weight))
    for order in orders:
        if order["weight_after"] is None:
            order["weight_after"] = target.get(order["ticker"], 0.0)
    return orders, target


def _expected_alpha_bps(ranked: pd.DataFrame) -> dict[str, float]:
    """Alfa esperado por ticker, en puntos básicos. Ausente cuando aún no hay calibración."""
    if "expected_excess_return" not in ranked:
        return {}
    values = pd.to_numeric(ranked["expected_excess_return"], errors="coerce") * BPS
    return {
        str(ticker): float(value)
        for ticker, value in zip(ranked["ticker"].astype(str), values)
        if pd.notna(value)
    }


def _below(alpha: float | None, threshold: float) -> bool:
    """Solo hay evidencia de incumplimiento si el alfa esperado existe y queda por debajo."""
    return alpha is not None and alpha < threshold


def _advantage(alpha_bps: dict[str, float], outsider: str, worst: str) -> float | None:
    """Ventaja de alfa esperado del candidato sobre la peor posición, o None si falta calibración.

    Sin calibración no se puede saber si la rotación cubre su coste, y una rotación que no se puede
    justificar económicamente no se hace: es la única forma de garantizar que la rotación nunca
    destruye valor por construcción.
    """
    if outsider not in alpha_bps or worst not in alpha_bps:
        return None
    return alpha_bps[outsider] - alpha_bps[worst]


def _alpha_or(alpha_bps: dict[str, float], ticker: str, percentile: dict[str, float]) -> float:
    """Ordena por alfa esperado; sin calibración, por percentil escalado a una magnitud comparable."""
    if ticker in alpha_bps:
        return alpha_bps[ticker]
    return percentile.get(ticker, -1.0) - 1e6


def _invested_fraction(holders: int, settings: Settings) -> float:
    """Fracción invertida. El hueco entre plazas cubiertas y objetivo se traduce en efectivo."""
    if settings.cash_policy == "fully_invested" or holders >= settings.target_size:
        return 1.0
    if holders == 0:
        return 1.0 - settings.max_cash_weight
    empty = (settings.target_size - holders) / settings.target_size
    return 1.0 - min(empty, settings.max_cash_weight)


def _strictness(scores: pd.DataFrame, settings: Settings) -> float:
    return 1.0 if "is_quarterly" not in scores or bool(scores["is_quarterly"].iloc[0]) else settings.price_only_strictness_multiplier


def _weights(
    tickers: list[str], alpha_bps: dict[str, float], invested: float, settings: Settings,
) -> dict[str, float]:
    """Reparte ``invested`` entre las posiciones. El resto de la cartera queda en efectivo."""
    if not tickers:
        return {}
    if settings.sizing_mode == "equal" or len(tickers) == 1:
        return {ticker: invested / len(tickers) for ticker in tickers}
    known = [alpha_bps[ticker] for ticker in tickers if ticker in alpha_bps]
    if not known:
        return {ticker: invested / len(tickers) for ticker in tickers}
    # Sin calibración, la posición recibe la mediana de las conocidas: es neutral, no penaliza ni
    # premia, y mantiene un único camino de cálculo para toda la cartera.
    neutral = float(pd.Series(known).median())
    low, high = min(known), max(known)
    span = high - low
    raw = {
        ticker: 1.0 + ((alpha_bps.get(ticker, neutral) - low) / span if span > 0 else 0.0)
        for ticker in tickers
    }
    total = sum(raw.values())
    return {ticker: invested * weight / total for ticker, weight in raw.items()}


def _apply_rebalance_tolerance(current: dict[str, float], target: dict[str, float], tolerance: float) -> dict[str, float]:
    frozen = {ticker: current[ticker] for ticker in target if current.get(ticker, 0.0) > 0 and abs(target[ticker] - current[ticker]) / current[ticker] < tolerance}
    movable = [ticker for ticker in target if ticker not in frozen]
    available = sum(target.values()) - sum(frozen.values())
    total = sum(target[ticker] for ticker in movable)
    if available < 0 or (movable and total <= 0):
        return target
    return {**frozen, **{ticker: available * target[ticker] / total for ticker in movable}} if movable else frozen


def _order(date: str, ticker: str, side: str, reason: str, before: float, after: float | None) -> dict[str, Any]:
    return {"snapshot_date": date, "ticker": ticker, "side": side, "reason": reason, "weight_before": before, "weight_after": after}
