"""Motor de decisiones de cartera: dado el estado actual y los scores de una fecha,
produce las ordenes que llevan al estado siguiente.

La logica sigue las reglas acordadas en `docs/plan_fases.md` (Fase 4):

  1. Expulsion: un tenente cuyo percentil baja por debajo de `min_hold_percentile` sale.
  2. Ventaja: un candidato solo desplaza a un tenente si le supera por al menos
     `rotation_edge_percentiles` percentiles.
  3. Tamano flexible: la cartera intenta llegar a `target_max`, sin bajar de `target_min`,
     rellenando con candidatos que superen `entry_min_percentile`.
  4. Sizing: peso proporcional al percentil con tope `max_weight_per_position`; el excedente
     se reparte proporcionalmente entre los que no tocan el tope.

No hay tenencia minima: cada revision (mensual o trimestral) decide desde cero. Los scores
cambian entre revisiones mensuales porque incluyen precio (P/E, momentum, etc.), aunque los
fundamentales no se hayan movido.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from environment import Settings


@dataclass
class PortfolioState:
    """Composicion actual de la cartera. Solo pesos; los precios los gestiona el backtest."""

    holdings: dict[str, float] = field(default_factory=dict)   # ticker -> peso (0..1)
    entry_dates: dict[str, str] = field(default_factory=dict)  # ticker -> snapshot_date de entrada
    months_held: dict[str, int] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "PortfolioState":
        return cls()

    @classmethod
    def from_holdings(
        cls,
        holdings: dict[str, float],
        entry_dates: dict[str, str] | None = None,
    ) -> "PortfolioState":
        entry_dates = entry_dates or {ticker: "1900-01-01" for ticker in holdings}
        return cls(
            holdings=dict(holdings),
            entry_dates=dict(entry_dates),
            months_held={ticker: 0 for ticker in holdings},
        )

    def apply(self, orders: list[dict[str, Any]], prices: dict[str, float]) -> "PortfolioState":
        """Devuelve un nuevo estado tras ejecutar las ordenes. `prices` es informacional
        (para trazar `entry_price` fuera del state), no se usa aqui — el peso ya viene fijado.
        """
        new_holdings = dict(self.holdings)
        new_entry_dates = dict(self.entry_dates)
        new_months_held = {ticker: months + 1 for ticker, months in self.months_held.items()}

        for order in orders:
            ticker = order["ticker"]
            if order["side"] == "sell":
                new_holdings.pop(ticker, None)
                new_entry_dates.pop(ticker, None)
                new_months_held.pop(ticker, None)
            elif order["side"] == "buy":
                new_holdings[ticker] = order["weight_after"]
                if ticker not in new_entry_dates:
                    new_entry_dates[ticker] = order.get("snapshot_date", "")
                    new_months_held[ticker] = 0
        return PortfolioState(
            holdings=new_holdings,
            entry_dates=new_entry_dates,
            months_held=new_months_held,
        )


def decide_orders(
    state: PortfolioState,
    scores_at_date: pd.DataFrame,
    settings: Settings,
) -> list[dict[str, Any]]:
    """Ordenes que llevan `state` al estado siguiente segun los scores.

    `scores_at_date` es el corte transversal de una unica fecha. Debe traer al menos
    `ticker`, `meta_rank` (percentil 0..1, tal como lo produce Fase 3) y `snapshot_date`.
    """
    if scores_at_date.empty:
        return []

    snapshot_date = scores_at_date["snapshot_date"].iloc[0]
    frame = scores_at_date.copy()
    frame["percentile_100"] = pd.to_numeric(frame["meta_rank"], errors="coerce") * 100
    frame = frame.dropna(subset=["percentile_100"]).sort_values("percentile_100", ascending=False)

    percentile_by_ticker = dict(zip(frame["ticker"], frame["percentile_100"]))

    survivors, drop_orders = _apply_expulsion(state, percentile_by_ticker, settings, snapshot_date)
    survivors_after_fill, fill_orders = _fill_slots(
        survivors, percentile_by_ticker, frame, settings, snapshot_date
    )
    rotation_orders = _apply_rotation(
        survivors_after_fill, percentile_by_ticker, frame, settings, snapshot_date
    )

    target = _target_state(state, drop_orders + fill_orders + rotation_orders, percentile_by_ticker)
    resize_orders = _resize_to_target(state, target, settings, snapshot_date)

    return drop_orders + fill_orders + rotation_orders + resize_orders


def _apply_expulsion(
    state: PortfolioState,
    percentile_by_ticker: dict[str, float],
    settings: Settings,
    snapshot_date: str,
) -> tuple[set[str], list[dict[str, Any]]]:
    survivors = set(state.holdings)
    orders: list[dict[str, Any]] = []
    for ticker in list(state.holdings):
        percentile = percentile_by_ticker.get(ticker)
        if percentile is None or percentile < settings.min_hold_percentile:
            orders.append(
                {
                    "snapshot_date": snapshot_date,
                    "ticker": ticker,
                    "side": "sell",
                    "reason": "dropped_below_min",
                    "weight_before": state.holdings[ticker],
                    "weight_after": 0.0,
                }
            )
            survivors.discard(ticker)
    return survivors, orders


def _fill_slots(
    survivors: set[str],
    percentile_by_ticker: dict[str, float],
    ranked: pd.DataFrame,
    settings: Settings,
    snapshot_date: str,
) -> tuple[set[str], list[dict[str, Any]]]:
    """Rellena huecos hasta `target_max` con candidatos por encima de `entry_min_percentile`."""
    orders: list[dict[str, Any]] = []
    result = set(survivors)
    candidates = [
        ticker
        for ticker in ranked["ticker"]
        if ticker not in result
        and percentile_by_ticker[ticker] >= settings.entry_min_percentile
    ]
    slots_available = settings.target_max - len(result)
    for ticker in candidates[:slots_available]:
        reason = "initial_fill" if not survivors else "hole_filled_after_drop"
        orders.append(
            {
                "snapshot_date": snapshot_date,
                "ticker": ticker,
                "side": "buy",
                "reason": reason,
                "weight_before": 0.0,
                "weight_after": None,           # lo fija el sizing final
            }
        )
        result.add(ticker)
    return result, orders


def _apply_rotation(
    current: set[str],
    percentile_by_ticker: dict[str, float],
    ranked: pd.DataFrame,
    settings: Settings,
    snapshot_date: str,
) -> list[dict[str, Any]]:
    """Un candidato fuera desplaza al peor tenente si le supera por rotation_edge_percentiles."""
    if len(current) < settings.target_max:
        return []
    orders: list[dict[str, Any]] = []
    working = set(current)
    while True:
        outsiders = [
            ticker
            for ticker in ranked["ticker"]
            if ticker not in working
            and percentile_by_ticker[ticker] >= settings.entry_min_percentile
        ]
        if not outsiders:
            break
        best_outsider = outsiders[0]
        worst_holder = min(working, key=lambda ticker: percentile_by_ticker.get(ticker, 0))
        gap = percentile_by_ticker[best_outsider] - percentile_by_ticker[worst_holder]
        if gap < settings.rotation_edge_percentiles:
            break
        orders.append(
            {
                "snapshot_date": snapshot_date,
                "ticker": worst_holder,
                "side": "sell",
                "reason": "displaced_by_edge",
                "weight_before": None,
                "weight_after": 0.0,
            }
        )
        orders.append(
            {
                "snapshot_date": snapshot_date,
                "ticker": best_outsider,
                "side": "buy",
                "reason": "edge_over_worst",
                "weight_before": 0.0,
                "weight_after": None,
            }
        )
        working.discard(worst_holder)
        working.add(best_outsider)
    return orders


def _target_state(
    state: PortfolioState,
    pending_orders: list[dict[str, Any]],
    percentile_by_ticker: dict[str, float],
) -> list[str]:
    tickers = set(state.holdings)
    for order in pending_orders:
        if order["side"] == "buy":
            tickers.add(order["ticker"])
        else:
            tickers.discard(order["ticker"])
    return sorted(tickers, key=lambda ticker: -percentile_by_ticker.get(ticker, 0))


def _resize_to_target(
    state: PortfolioState,
    target_tickers: list[str],
    settings: Settings,
    snapshot_date: str,
) -> list[dict[str, Any]]:
    """Fija el peso final de cada ticker en la cartera objetivo.

    Peso proporcional al ranking dentro de la cartera (mejor -> mas peso) con tope
    `max_weight_per_position`. El excedente se reparte entre los que no tocan tope.
    """
    if not target_tickers:
        return []

    weights = _compute_weights(target_tickers, settings.max_weight_per_position)

    orders: list[dict[str, Any]] = []
    for ticker, weight in weights.items():
        current_weight = state.holdings.get(ticker, 0.0)
        if abs(current_weight - weight) < 1e-9:
            continue
        orders.append(
            {
                "snapshot_date": snapshot_date,
                "ticker": ticker,
                "side": "buy",
                "reason": "rebalance",
                "weight_before": current_weight,
                "weight_after": weight,
            }
        )
    return orders


def _compute_weights(target_tickers: list[str], max_weight: float) -> dict[str, float]:
    """Pesos proporcionales al ranking (1er, 2do, 3er, ...) con tope y reparto del excedente.

    Como base uso una rampa lineal decreciente: el mejor tiene peso relativo N, el segundo
    N-1, ..., el ultimo 1. Es determinista, no depende de la escala del score.
    """
    n = len(target_tickers)
    raw = {ticker: (n - index) for index, ticker in enumerate(target_tickers)}
    total = sum(raw.values())
    weights = {ticker: value / total for ticker, value in raw.items()}

    # Iterativo: recorta al tope y redistribuye el excedente hasta que nadie exceda.
    while True:
        capped = {ticker: min(weight, max_weight) for ticker, weight in weights.items()}
        excess = 1.0 - sum(capped.values())
        if excess <= 1e-9:
            weights = capped
            break
        room_holders = [ticker for ticker, weight in capped.items() if weight < max_weight - 1e-12]
        if not room_holders:
            # Todos al tope y aun sobra: no se puede llegar al 100 %. Queda como cash.
            weights = capped
            break
        distributable = sum(weights[ticker] for ticker in room_holders)
        if distributable <= 0:
            weights = capped
            break
        for ticker in room_holders:
            capped[ticker] += excess * (weights[ticker] / distributable)
        weights = capped
        # Puede que la redistribucion haya empujado a alguien por encima del tope: se vuelve a iterar.
    return weights
