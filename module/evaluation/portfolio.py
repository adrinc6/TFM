"""Reglas causales de la cartera dinámica, gobernadas por alfa esperado en puntos básicos.

Toda decisión de compra, venta y rotación se toma comparando **magnitudes económicas** —el alfa
esperado calibrado de cada acción, en puntos básicos— contra umbrales también económicos, con un
principio único: **una venta solo se emite si el destino del dinero es mejor que la posición después
de costes**. Hay exactamente dos destinos posibles y cada uno tiene su regla:

- **Otra acción (rotación)**: la ventaja de alfa esperado debe superar el coste de ida y vuelta de
  la operación más un margen. Es la única vía de venta bajo ``fully_invested``: vender por umbral
  con la obligación de recomprar inmediatamente pagaría una ida y vuelta para quedar igual.
- **Efectivo (solo ``opportunity_cash``)**: el alfa esperado debe caer bajo el umbral de salida y la
  plaza debe poder quedar vacía sin violar el suelo de diversificación ni el tope de efectivo.

Las compras nuevas tienen **histéresis**: entrar exige el umbral de salida más el coste de ida y
vuelta de la propia operación. Sin esa banda, una acción oscilando alrededor del umbral se compraría
y vendería en snapshots consecutivos pagando costes con ventaja esperada nula.

El ranking del meta decide el orden de preferencia y los desempates (la confianza de los agentes);
el alfa calibrado decide si cada operación se paga a sí misma (la magnitud económica); los costes
son el listón que toda operación debe superar.
"""

from __future__ import annotations

import math
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
    """Aplica venta a efectivo con suelo, compra con histéresis y rotación que paga su coste.

    Devuelve las órdenes y los pesos objetivo. **Los pesos pueden sumar menos de 1**: el residuo es
    efectivo, y solo aparece bajo ``cash_policy="opportunity_cash"``; nunca supera
    ``max_cash_weight`` porque el número de posiciones nunca baja del suelo
    ``ceil((1 − max_cash_weight) · target_size)``. Con ``fully_invested`` el residuo es cero por
    construcción. Si hay puntuaciones en la fecha, la cartera objetivo nunca queda vacía.

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
    candidates = list(dict.fromkeys(ranked["ticker"].astype(str)))

    # Endurecer en snapshots sin fundamentales nuevos significa operar menos en ambos sentidos: el
    # umbral de salida baja (cuesta más gatillar una venta), y tanto la entrada como la rotación
    # exigen más ventaja. La banda de histéresis se ensancha.
    exit_threshold = settings.exit_expected_alpha_bps / strictness
    round_trip = 2.0 * (settings.commission_bps + settings.slippage_bps)
    entry_threshold = (settings.exit_expected_alpha_bps + round_trip) * strictness
    rotation_threshold = round_trip + settings.rotation_edge_bps * strictness
    floor = _position_floor(settings)

    holders = set(state.holdings)
    removed: set[str] = set()
    orders: list[dict[str, Any]] = []

    # 1. Ventas a efectivo, solo bajo la política de oportunidad: el destino es efectivo al 0 %, así
    # que exige evidencia calibrada bajo el umbral y respeta el suelo de posiciones (que garantiza a
    # la vez el tope de efectivo y un mínimo de diversificación). Bajo `fully_invested` no existe
    # este destino: vender por umbral obligaría a recomprar en el mismo snapshot.
    if settings.cash_policy == "opportunity_cash":
        below = sorted(
            (ticker for ticker in holders if _below(alpha_bps.get(ticker), exit_threshold)),
            key=lambda ticker: _alpha_or(alpha_bps, ticker, percentile),
        )
        for ticker in below:
            if len(holders) <= floor:
                break
            holders.remove(ticker)
            removed.add(ticker)
            orders.append(_order(date, ticker, "sell", "expected_alpha_below_exit", state.holdings.get(ticker, 0.0), 0.0))

    # 2. Compras con histéresis: una entrada nueva debe esperar cubrir su propio coste de ida y
    # vuelta por encima del umbral de salida. Sin calibración (NaN) manda el ranking.
    for ticker in candidates:
        if len(holders) >= settings.target_size:
            break
        if ticker in holders or ticker in removed:
            continue
        if _below(alpha_bps.get(ticker), entry_threshold):
            continue
        holders.add(ticker)
        orders.append(_order(date, ticker, "buy", "initial_fill", 0.0, None))

    # 3. Rellenos obligatorios por política: `fully_invested` completa hasta el objetivo con las
    # mejores por ranking aunque no superen el umbral; `opportunity_cash` solo hasta el suelo de
    # diversificación. Nunca se recompra lo vendido en este mismo snapshot.
    mandatory = settings.target_size if settings.cash_policy == "fully_invested" else floor
    reason = "fully_invested_fill" if settings.cash_policy == "fully_invested" else "cash_floor_fill"
    for ticker in candidates:
        if len(holders) >= mandatory:
            break
        if ticker in holders or ticker in removed:
            continue
        holders.add(ticker)
        orders.append(_order(date, ticker, "buy", reason, 0.0, None))

    # 4. Rotación: un outsider desplaza a la peor posición solo si su ventaja de alfa esperado paga
    # el coste de ida y vuelta más el margen. Bajo `opportunity_cash` el outsider debe además superar
    # el umbral de entrada: si no, su plaza natural sería efectivo, no la cartera.
    while len(holders) >= settings.target_size:
        outsider = next(
            (
                ticker for ticker in candidates
                if ticker not in holders and ticker not in removed
                and (
                    settings.cash_policy == "fully_invested"
                    or not _below(alpha_bps.get(ticker), entry_threshold)
                )
            ),
            None,
        )
        if outsider is None:
            break
        worst = min(holders, key=lambda ticker: _alpha_or(alpha_bps, ticker, percentile))
        advantage = _advantage(alpha_bps, outsider, worst)
        if advantage is None or advantage < rotation_threshold:
            break
        holders.remove(worst)
        removed.add(worst)
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


def _position_floor(settings: Settings) -> int:
    """Número mínimo de posiciones. Garantiza el tope de efectivo y acota la concentración.

    Con `target_size` plazas y tope de efectivo `max_cash_weight`, al menos
    `ceil((1 − max_cash_weight) · target_size)` plazas deben estar ocupadas: así el efectivo nunca
    supera el tope y ninguna posición puede acercarse a concentrar la cartera (con 12 plazas y tope
    del 25 %, el suelo son 9 posiciones y el peso máximo implícito queda en torno al 15 %).
    """
    if settings.cash_policy == "fully_invested":
        return settings.target_size
    floor = math.ceil((1.0 - settings.max_cash_weight) * settings.target_size)
    return max(1, min(settings.target_size, floor))


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
