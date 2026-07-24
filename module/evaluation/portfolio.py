"""Motor de decisiones de cartera: dado el estado actual y los scores de una fecha,
produce las ordenes que llevan al estado siguiente.

La logica sigue las reglas de cartera descritas en `docs/doc.md`:

  1. Expulsion: un tenente cuyo percentil queda en o por debajo de `min_hold_percentile` sale.
  2. Ventaja: un candidato solo desplaza a un tenente si le supera por al menos
     `rotation_edge_percentiles` percentiles.
  3. Tamano fijo: la cartera siempre completa `target_size` con el top-N del meta-rank.
  4. Sizing: escala lineal min-max DENTRO de la cartera sobre el `meta_score` crudo; el mejor del
     basket pesa el doble que el peor, lineal en medio. Se usa el meta_score (no el percentil global,
     que se apiña en el top y aplana los pesos a ~1/N). Ver `_compute_weights`.
  5. Rebalanceo real: un tenente solo se reajusta si su peso objetivo difiere del actual en al
     menos `rebalance_drift_tolerance` (fraccion RELATIVA a la posicion, p. ej. 0.25 = 25 %); por
     debajo se "congela" y el presupuesto se reparte entre los que si se mueven, conservando las
     relaciones del target global (ver `_resize_to_target`).

No hay tenencia minima: cada revision (mensual o trimestral) decide desde cero. Los scores
cambian entre revisiones mensuales porque incluyen precio (P/E, momentum, etc.), aunque los
fundamentales no se hayan movido.

Por eso las revisiones que NO traen fundamentales nuevos (`is_quarterly=False`, es decir el
`review_type=price_monthly` del panel) aplican `price_only_strictness_multiplier` sobre los tres
umbrales de las reglas 1, 2 y 5: sin informacion fundamental nueva, mover la cartera equivale a
rotar por ruido de precio. Con el multiplicador en 1.0 el comportamiento es el historico.
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

    @classmethod
    def empty(cls) -> "PortfolioState":
        return cls()

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
        new_entry_prices = dict(self.entry_prices)

        for order in orders:
            ticker = order["ticker"]
            if order["side"] == "sell":
                new_holdings.pop(ticker, None)
                new_entry_dates.pop(ticker, None)
                new_entry_prices.pop(ticker, None)
            elif order["side"] == "buy":
                new_holdings[ticker] = order["weight_after"]
                if ticker not in new_entry_dates:
                    new_entry_dates[ticker] = order.get("snapshot_date", "")
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
        )


def decide_orders(
    state: PortfolioState,
    scores_at_date: pd.DataFrame,
    settings: Settings,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Ordenes que llevan `state` al estado siguiente segun los scores.

    `scores_at_date` es el corte transversal de una unica fecha. Debe traer al menos
    `ticker`, `meta_rank` (percentil 0..1, tal como lo produce Fase 3) y `snapshot_date`.
    Si trae `is_quarterly` (lo produce el panel), las revisiones sin fundamentales nuevos
    endurecen sus umbrales con `price_only_strictness_multiplier`.

    Devuelve `(ordenes, pesos_objetivo)`: `pesos_objetivo` (suma 1) permite al backtest refrescar el
    peso vigente de todos los tenentes, no solo de los que generan orden.
    """
    if scores_at_date.empty:
        return [], {}

    snapshot_date = scores_at_date["snapshot_date"].iloc[0]
    strictness = _strictness_multiplier(scores_at_date, settings)
    frame = scores_at_date.copy()
    frame["percentile_100"] = pd.to_numeric(frame["meta_rank"], errors="coerce") * 100
    frame = frame.dropna(subset=["percentile_100"]).sort_values("percentile_100", ascending=False)

    percentile_by_ticker = dict(zip(frame["ticker"], frame["percentile_100"]))
    # Señal de SIZING: el `meta_score` crudo (no el percentil global, que se apiña en el top y aplana
    # los pesos). El sizing se hace in-basket sobre él (ver `_compute_weights`). Selección, expulsión
    # y rotación siguen usando el percentil global (`percentile_by_ticker`), que es lo correcto para
    # decisiones cross-universo. Fallback al percentil si faltara `meta_score` (datos/tests antiguos).
    if "meta_score" in frame.columns:
        meta_score_by_ticker = dict(zip(frame["ticker"], pd.to_numeric(frame["meta_score"], errors="coerce")))
        meta_score_by_ticker = {t: v for t, v in meta_score_by_ticker.items() if pd.notna(v)}
    else:
        meta_score_by_ticker = {}
    if not meta_score_by_ticker:
        meta_score_by_ticker = percentile_by_ticker

    survivors, drop_orders = _apply_expulsion(
        state, percentile_by_ticker, settings, snapshot_date, strictness
    )
    survivors_after_fill, fill_orders = _fill_slots(survivors, frame, settings, snapshot_date)
    rotation_orders = _apply_rotation(
        survivors_after_fill, percentile_by_ticker, frame, settings, snapshot_date, strictness
    )

    target = _target_state(state, drop_orders + fill_orders + rotation_orders, percentile_by_ticker)
    resize_orders, target_weights = _resize_to_target(
        state, target, meta_score_by_ticker, settings, snapshot_date, strictness
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


def _strictness_multiplier(scores_at_date: pd.DataFrame, settings: Settings) -> float:
    """Factor de endurecimiento de la revision: >1 solo si NO trae fundamentales nuevos.

    `is_quarterly` viene del panel (`review_type`). Si falta (datos o tests antiguos) se asume
    revision con fundamentales -> factor 1.0, es decir el comportamiento historico.
    """
    if "is_quarterly" not in scores_at_date.columns:
        return 1.0
    is_quarterly = scores_at_date["is_quarterly"].iloc[0]
    if pd.isna(is_quarterly) or bool(is_quarterly):
        return 1.0
    return float(settings.price_only_strictness_multiplier)


def _apply_expulsion(
    state: PortfolioState,
    percentile_by_ticker: dict[str, float],
    settings: Settings,
    snapshot_date: str,
    strictness: float = 1.0,
) -> tuple[set[str], list[dict[str, Any]]]:
    # Sin fundamentales nuevos se exige menos percentil para conservar: expulsar por ruido de
    # precio rotaria la cartera sin informacion que lo justifique.
    min_hold = settings.min_hold_percentile / strictness
    survivors = set(state.holdings)
    orders: list[dict[str, Any]] = []
    for ticker in list(state.holdings):
        percentile = percentile_by_ticker.get(ticker)
        if percentile is None or percentile <= min_hold:
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
    strictness: float = 1.0,
) -> list[dict[str, Any]]:
    """Un candidato fuera desplaza al peor tenente si le supera por rotation_edge_percentiles."""
    if len(current) < settings.target_size:
        return []
    # Sin fundamentales nuevos se exige mas ventaja para desplazar a un tenente.
    rotation_edge = settings.rotation_edge_percentiles * strictness
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
        if gap < rotation_edge:
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
    meta_score_by_ticker: dict[str, float],
    settings: Settings,
    snapshot_date: str,
    strictness: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Fija el peso final de cada ticker en la cartera objetivo, gestionando el rebalanceo real.

    En una sola pasada (sin iterar):

    1. Calcula el TARGET IDEAL sobre la cartera COMPLETA con la escala in-basket de `_compute_weights`
       (el mejor `meta_score` del basket pesa el doble que el peor). Esas son las relaciones "base".
    2. Clasifica cada tenente ya presente comparando su target con su peso actual:
       - si el cambio RELATIVO a su posición es menor que `rebalance_drift_tolerance` (fracción,
         p. ej. 0.25 = 25 %) -> CONGELADO: conserva su peso actual y NO genera orden (ahorra coste
         y evita micro-rotación por ruido);
       - si es mayor o igual -> MÓVIL: se rebalancea. Las entradas nuevas (peso actual 0) son
         siempre móviles.
    3. Reparte el presupuesto que dejan los congelados (1 - suma de sus pesos) entre los móviles
       MANTENIENDO las relaciones del target global (proporcional a su target ideal). Así dos
       móviles con percentiles parecidos reciben pesos parecidos, y no se re-separan al recalcular
       sobre el subconjunto.

    Devuelve `(ordenes, pesos_objetivo)`. `pesos_objetivo` es el mapa final ENCAJADO y NORMALIZADO
    (suma 1: congelados en su peso actual + móviles repartidos); el estado lo aplica a todos para
    que la cartera vigente siga sumando 1.
    """
    if not target_tickers:
        return [], {}

    target = _compute_weights(target_tickers, meta_score_by_ticker)
    # Sin fundamentales nuevos se exige mas deriva para mover un peso: mas posiciones congeladas.
    min_fraction = settings.rebalance_drift_tolerance * strictness

    # Clasificación: contra el target global fijo (una sola pasada, independiente del orden).
    frozen: dict[str, float] = {}
    movers: list[str] = []
    for ticker in target_tickers:
        current = state.holdings.get(ticker, 0.0)
        if current > 0 and abs(target[ticker] - current) / current < min_fraction:
            frozen[ticker] = current
        else:
            movers.append(ticker)

    budget = 1.0 - sum(frozen.values())
    movers_target_sum = sum(target[ticker] for ticker in movers)
    if budget < -1e-9 or (movers and movers_target_sum <= 0):
        # Fallback seguro: los congelados ya suman > 1 (apalancamiento) o los móviles no tienen
        # target positivo. Se renormaliza TODO al target global (evita pesos negativos) y se
        # acepta operar. Caso raro: solo por deriva acumulada extrema.
        weights = dict(target)
    elif not movers:
        # Todos congelados: sus pesos actuales ya suman ~1 (budget ~ 0). Se conservan tal cual.
        weights = dict(frozen)
    else:
        # Reparto normal: el presupuesto liberado por los congelados va a los móviles conservando
        # las relaciones del target global.
        weights = dict(frozen)
        for ticker in movers:
            weights[ticker] = budget * target[ticker] / movers_target_sum

    orders: list[dict[str, Any]] = []
    for ticker, weight in weights.items():
        current_weight = state.holdings.get(ticker, 0.0)
        if abs(current_weight - weight) < 1e-9:
            continue  # congelado o sin cambio: no opera
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
    target_tickers: list[str],
    signal_by_ticker: dict[str, float],
) -> dict[str, float]:
    """Pesos según una escala LINEAL min-max DENTRO de la cartera (in-basket) sobre `meta_score`.

    Motivo del cambio: los tenentes son siempre el top-N del universo, así que su percentil GLOBAL
    (`meta_rank`) se apiña en 0.97-1.00 (con ~500 tickers hay muchos por percentil entero). Cualquier
    escala anclada al percentil global degenera ahí en ~1/N: todos ~12,5 % con 8 posiciones. El
    `meta_score` crudo (la salida del meta antes de rankear) NO está comprimido y sí ordena a los
    seleccionados, así que la escala se calcula sobre él re-medido entre los tickers del basket:

        lo, hi = min(meta_score), max(meta_score)   # sobre los tickers del basket
        r      = (meta_score - lo) / (hi - lo)       # 0 el peor del basket, 1 el mejor
        peso   = (1 + r) / Σ (1 + r)                 # el mejor pesa EXACTAMENTE el doble que el peor

    El ancla es el PEOR del basket (no `min_hold_percentile`, que solo tenía sentido sobre el
    percentil global). Suave y siempre 2:1. Si todos tienen el mismo `meta_score` (hi == lo) ->
    equiponderación. Un solo ticker -> peso 1. Sin pesos mín/máx (el ratio 2:1 basta). Sin lookahead:
    `meta_score` es el corte transversal de la fecha.
    """
    if not target_tickers:
        return {}
    if len(target_tickers) == 1:
        return {target_tickers[0]: 1.0}
    values = [float(signal_by_ticker.get(ticker, 0.0)) for ticker in target_tickers]
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 0:  # todos iguales: equiponderación
        return {ticker: 1 / len(target_tickers) for ticker in target_tickers}
    scores = {
        ticker: 1.0 + (float(signal_by_ticker.get(ticker, 0.0)) - lo) / span
        for ticker in target_tickers
    }
    total = sum(scores.values())
    return {ticker: score / total for ticker, score in scores.items()}
