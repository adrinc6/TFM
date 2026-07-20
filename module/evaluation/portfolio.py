"""Motor de decisiones de cartera: dado el estado actual y los scores de una fecha,
produce las ordenes que llevan al estado siguiente.

La logica sigue las reglas de cartera descritas en `docs/doc.md`:

  1. Expulsion: un tenente cuyo percentil queda en o por debajo de `min_hold_percentile` sale.
  2. Ventaja: un candidato solo desplaza a un tenente si le supera por al menos
     `rotation_edge_percentiles` percentiles.
  3. Tamano fijo: la cartera siempre completa `target_size` con el top-N del meta-rank.
  4. Sizing: peso proporcional al meta-rank, con una relacion maxima 2:1 entre el mayor y menor.

No hay tenencia minima: cada revision (mensual o trimestral) decide desde cero. Los scores
cambian entre revisiones mensuales porque incluyen precio (P/E, momentum, etc.), aunque los
fundamentales no se hayan movido.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from environment import Settings


@dataclass
class PortfolioState:
    """Composicion actual de la cartera. Solo pesos; los precios los gestiona el backtest."""

    holdings: dict[str, float] = field(default_factory=dict)   # ticker -> peso (0..1)
    entry_dates: dict[str, str] = field(default_factory=dict)  # ticker -> snapshot_date de entrada
    entry_prices: dict[str, float] = field(default_factory=dict)  # ticker -> precio PIT de entrada
    months_held: dict[str, int] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "PortfolioState":
        return cls()

    @classmethod
    def from_holdings(
        cls,
        holdings: dict[str, float],
        entry_dates: dict[str, str] | None = None,
        entry_prices: dict[str, float] | None = None,
    ) -> "PortfolioState":
        entry_dates = entry_dates or {ticker: "1900-01-01" for ticker in holdings}
        return cls(
            holdings=dict(holdings),
            entry_dates=dict(entry_dates),
            entry_prices=dict(entry_prices or {}),
            months_held={ticker: 0 for ticker in holdings},
        )

    def apply(self, orders: list[dict[str, Any]], prices: dict[str, float],
              target_weights: dict[str, float] | None = None) -> "PortfolioState":
        """Devuelve un nuevo estado tras ejecutar las ordenes. `prices` es informacional
        (para trazar `entry_price` fuera del state), no se usa aqui — el peso ya viene fijado.

        `target_weights` es el peso objetivo NORMALIZADO (suma 1) de cada tenente superviviente. Se
        aplica a TODOS los tenentes, no solo a los que generaron orden: un tenente cuya deriva es
        pequena no se opera (ahorra coste) pero su peso vigente se refresca al objetivo para que la
        cartera siga sumando 1. Sin esto los pesos derivan por encima del 100 % (apalancamiento
        ficticio en el mark-to-market). Ver `_resize_to_target`.
        """
        new_holdings = dict(self.holdings)
        new_entry_dates = dict(self.entry_dates)
        new_months_held = {ticker: months + 1 for ticker, months in self.months_held.items()}
        new_entry_prices = dict(self.entry_prices)

        for order in orders:
            ticker = order["ticker"]
            if order["side"] == "sell":
                new_holdings.pop(ticker, None)
                new_entry_dates.pop(ticker, None)
                new_entry_prices.pop(ticker, None)
                new_months_held.pop(ticker, None)
            elif order["side"] == "buy":
                new_holdings[ticker] = order["weight_after"]
                if ticker not in new_entry_dates:
                    new_entry_dates[ticker] = order.get("snapshot_date", "")
                    new_months_held[ticker] = 0
                    price = order.get("price", prices.get(ticker))
                    if price is not None:
                        new_entry_prices[ticker] = float(price)
        # Refresca el peso vigente de cada tenente al objetivo normalizado (bookkeeping, sin coste):
        # asi la cartera suma 1 aunque una posicion no se haya rebalanceado por micro-deriva.
        if target_weights is not None:
            for ticker in list(new_holdings):
                if ticker in target_weights:
                    new_holdings[ticker] = target_weights[ticker]
        return PortfolioState(
            holdings=new_holdings,
            entry_dates=new_entry_dates,
            entry_prices=new_entry_prices,
            months_held=new_months_held,
        )


def decide_orders(
    state: PortfolioState,
    scores_at_date: pd.DataFrame,
    settings: Settings,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Ordenes que llevan `state` al estado siguiente segun los scores.

    `scores_at_date` es el corte transversal de una unica fecha. Debe traer al menos
    `ticker`, `meta_rank` (percentil 0..1, tal como lo produce Fase 3) y `snapshot_date`.

    Devuelve `(ordenes, pesos_objetivo)`: `pesos_objetivo` (suma 1) permite al backtest refrescar el
    peso vigente de todos los tenentes, no solo de los que generan orden.
    """
    if scores_at_date.empty:
        return [], {}

    snapshot_date = scores_at_date["snapshot_date"].iloc[0]
    frame = scores_at_date.copy()
    frame["percentile_100"] = pd.to_numeric(frame["meta_rank"], errors="coerce") * 100
    frame = frame.dropna(subset=["percentile_100"]).sort_values("percentile_100", ascending=False)

    percentile_by_ticker = dict(zip(frame["ticker"], frame["percentile_100"]))

    survivors, drop_orders = _apply_expulsion(state, percentile_by_ticker, settings, snapshot_date)
    survivors_after_fill, fill_orders = _fill_slots(survivors, frame, settings, snapshot_date)
    rotation_orders = _apply_rotation(
        survivors_after_fill, percentile_by_ticker, frame, settings, snapshot_date
    )

    target = _target_state(state, drop_orders + fill_orders + rotation_orders, percentile_by_ticker)
    resize_orders, target_weights = _resize_to_target(
        state, target, percentile_by_ticker, settings, snapshot_date
    )

    merged = _merge_intents_with_sizing(
        drop_orders + fill_orders + rotation_orders, resize_orders
    )
    return merged, target_weights


# Motivos de "intención de entrada": marcan que un ticker debe entrar, pero su peso real lo fija
# el sizing (`rebalance`). Se escriben con weight_after=None y no deben quedar como órdenes sueltas.
_ENTRY_INTENT_REASONS = frozenset({"initial_fill", "hole_filled_after_drop", "edge_over_worst"})


def _merge_intents_with_sizing(
    decision_orders: list[dict[str, Any]], resize_orders: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Colapsa cada intención de ENTRADA con su orden de sizing del mismo ticker en UNA sola orden.

    Antes se escribían dos filas el mismo día por cada ticker que entraba: la intención
    (`hole_filled_after_drop`/`initial_fill`/`edge_over_worst`, con peso 0→0 al no resolverse su
    weight_after) y el `rebalance` real (0→peso). La primera no tiene efecto en equity ni costes
    (notional 0), pero ensuciaba el log de operaciones y lo duplicaba. Aquí se funden en una única
    orden real que conserva el MOTIVO de entrada (por qué entró) y el peso final del sizing. Las
    ventas y los rebalances de tenentes ya presentes se mantienen tal cual.
    """
    resize_by_ticker = {order["ticker"]: order for order in resize_orders}
    merged: list[dict[str, Any]] = []
    consumed_resizes: set[str] = set()
    for order in decision_orders:
        if order.get("reason") in _ENTRY_INTENT_REASONS:
            sizing = resize_by_ticker.get(order["ticker"])
            if sizing is not None:
                # Orden real: peso del sizing, motivo de la intención (más informativo).
                merged.append({**sizing, "reason": order["reason"]})
                consumed_resizes.add(order["ticker"])
            # Si no hay sizing asociado, la intención era un placeholder 0→0: se descarta.
            continue
        merged.append(order)
    # Rebalances que no correspondían a una entrada (tenentes ya presentes) se conservan.
    for order in resize_orders:
        if order["ticker"] not in consumed_resizes:
            merged.append(order)
    return merged


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
        if percentile is None or percentile <= settings.min_hold_percentile:
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
    ranked: pd.DataFrame,
    settings: Settings,
    snapshot_date: str,
) -> tuple[set[str], list[dict[str, Any]]]:
    """Rellena huecos hasta el tamaño fijo con las mejores candidatas disponibles."""
    orders: list[dict[str, Any]] = []
    result = set(survivors)
    candidates = [
        ticker
        for ticker in ranked["ticker"]
        if ticker not in result
    ]
    slots_available = settings.target_size - len(result)
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
    if len(current) < settings.target_size:
        return []
    orders: list[dict[str, Any]] = []
    working = set(current)
    while True:
        outsiders = [
            ticker
            for ticker in ranked["ticker"]
            if ticker not in working
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
    percentile_by_ticker: dict[str, float],
    settings: Settings,
    snapshot_date: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Fija el peso final de cada ticker en la cartera objetivo.

    Peso proporcional al meta-rank con una relación máxima 2:1 entre el mayor y menor. Solo se
    rebalancea a un tenente ya presente si se desvía del objetivo más que la tolerancia en puntos
    porcentuales; las entradas y salidas siempre generan orden.

    Devuelve `(ordenes, pesos_objetivo)`. `pesos_objetivo` es el mapa NORMALIZADO (suma 1) de todos
    los tenentes objetivo; el estado lo aplica a todos (tambien a los no rebalanceados por
    micro-deriva) para que la cartera vigente siga sumando 1.
    """
    if not target_tickers:
        return [], {}

    weights = _compute_weights(target_tickers, percentile_by_ticker)
    drift_threshold = settings.rebalance_drift_tolerance / 100

    orders: list[dict[str, Any]] = []
    for ticker, weight in weights.items():
        current_weight = state.holdings.get(ticker, 0.0)
        if abs(current_weight - weight) < 1e-9:
            continue
        # Tenente ya presente cuya deriva es pequena: no se rebalancea (ahorra coste).
        if current_weight > 0 and abs(current_weight - weight) < drift_threshold:
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
    return orders, weights


def _compute_weights(
    target_tickers: list[str], percentile_by_ticker: dict[str, float],
) -> dict[str, float]:
    """Pesos proporcionales al meta-rank y con relación máxima 2:1.

    El ranking efectivo (incluido el perfil) es la única señal de tamaño. Al recortar los scores
    superiores a dos veces el menor score elegible, la normalización conserva el orden y garantiza
    que ninguna posición pese más del doble que la menor.
    """
    raw = {ticker: max(0.0, float(percentile_by_ticker.get(ticker, 0.0)))
           for ticker in target_tickers}
    positive = [value for value in raw.values() if value > 0]
    if not positive:
        return {ticker: 1 / len(target_tickers) for ticker in target_tickers}
    floor = min(positive)
    capped = {ticker: min(value if value > 0 else floor, 2 * floor)
              for ticker, value in raw.items()}
    total = sum(capped.values())
    return {ticker: value / total for ticker, value in capped.items()}
