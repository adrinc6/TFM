"""Reglas causales de la cartera dinámica, gobernadas por alfa esperado en puntos básicos.

Toda decisión de compra, venta y rotación se toma comparando **magnitudes económicas** —el alfa
esperado calibrado de cada acción, en puntos básicos— contra umbrales también económicos, con un
principio único: **una venta solo se emite si el destino del dinero es mejor que la posición después
de costes**. Hay exactamente dos destinos posibles y cada uno tiene su regla:

- **Otra acción (rotación)**: la ventaja de alfa esperado debe superar el coste de ida y vuelta de
  la operación más un margen. Es la única vía de venta con ``max_cash_weight = 0``: vender por
  umbral con la obligación de recomprar inmediatamente pagaría una ida y vuelta para quedar igual.
- **Efectivo (solo con ``max_cash_weight > 0``)**: el alfa esperado debe caer bajo el umbral de
  salida y la plaza debe poder quedar vacía sin violar el suelo de diversificación ni el tope.

``max_cash_weight`` es la **única** variable que gobierna la exposición: 0 significa "siempre
invertido" y hace que el suelo de diversificación coincida con ``target_size``, de modo que el
relleno obligatorio deja la cartera llena. No hay una política de efectivo aparte que diga lo mismo
por otra vía; existió y se eliminó porque los dos grados de libertad eran redundantes.

Hay dos excepciones al principio económico, y ninguna se compara contra coste alguno porque ninguna
decide si una operación es rentable, sino si la acción pertenece al universo invertible:

- **Pérdida de cobertura**: una posición sin ``meta_rank`` en el snapshot actual ya no pertenece al
  universo scoreable. Se vende con el motivo ``missing_current_score``, **ignora** el mínimo de
  tenencia y nunca se recompra en ese mismo snapshot.
- **Suelo de cobertura** (``coverage_percentile_floor``, 0 lo desactiva): una posición que cae por
  debajo de ese percentil se vende **siempre**, con el motivo ``below_coverage_percentile``. Es la
  generalización de la anterior —percentil ausente sustituido por percentil demasiado bajo— pero
  **sí respeta** el mínimo de tenencia, porque la posición sigue siendo scoreable y la exclusión es
  una preferencia declarada, no una pérdida de cobertura. La venta no se frena por el suelo de
  diversificación: qué pasa con la plaza liberada (recompra o efectivo) lo decide después,
  igual que siempre, el relleno obligatorio del paso 4. Con ``max_cash_weight = 0`` esa plaza se
  rellena en el mismo snapshot, así que la regla actúa como una rotación que no exige cubrir su
  coste y aumenta la rotación; con tope de efectivo puede quedarse en efectivo hasta el suelo de
  diversificación.

Las compras nuevas tienen **histéresis**: entrar exige el umbral de salida más el coste de ida y
vuelta de la propia operación. Sin esa banda, una acción oscilando alrededor del umbral se compraría
y vendería en snapshots consecutivos pagando costes con ventaja esperada nula.

``exit_expected_alpha_bps`` y ``rotation_edge_bps`` se definen en el catálogo en puntos básicos
**anuales** y se convierten geométricamente (compuesto, no lineal) al horizonte real del modelo
antes de compararse: así el mismo valor de catálogo significa lo mismo sin importar qué
``target_horizon_months`` se elija, en vez de que 250 pb signifiquen un 10 %/año con horizonte de 3
meses y un 2,5 %/año con horizonte de 12.

``minimum_holding_period`` añade un suelo de tiempo por encima de las decisiones económicas: una
posición que no ha cumplido ese mínimo de meses en cartera no puede venderse por caída del alfa
esperado ni por rotación. No cubre la pérdida de ``meta_rank``, que invalida la posición. No busca
más alfa: busca estabilidad frente a la rotación de alta frecuencia sobre el mismo modelo ya
congelado.

``price_only_sell_only`` restringe los snapshots que solo traen precio nuevo (sin fundamentales
frescos): se sigue permitiendo vender una posición que ya no cumple, pero se prohíbe comprar
cualquier reemplazo (compra nueva, relleno obligatorio o rotación), porque no hay información nueva
que justifique elegir una acción distinta a la ya elegida con datos reales. La plaza vendida queda en
efectivo, transitoriamente y sin el tope de ``max_cash_weight``, hasta el siguiente snapshot con
fundamentales frescos.

El ranking del meta decide el orden de preferencia y los desempates (la confianza de los agentes);
el alfa calibrado decide si cada operación se paga a sí misma (la magnitud económica); los costes
son el listón que toda operación debe superar; y el mínimo de tenencia es el único freno que no
depende de ninguna magnitud económica, sino solo del tiempo.
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


def months_held(entry: str, date: str) -> int:
    """Meses completos transcurridos entre la fecha de entrada y la fecha actual."""
    left, right = pd.Timestamp(entry), pd.Timestamp(date)
    return max(0, (right.year - left.year) * 12 + right.month - left.month)


def decide_orders(
    state: PortfolioState, scores_at_date: pd.DataFrame, settings: Settings,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Aplica venta a efectivo con suelo, compra con histéresis y rotación que paga su coste.

    Devuelve las órdenes y los pesos objetivo. **Los pesos pueden sumar menos de 1**: el residuo es
    efectivo, que aparece con ``max_cash_weight > 0`` (acotado por ese tope y por el suelo de
    diversificación) o, transitoriamente y sin tope propio, bajo ``price_only_sell_only`` en un
    snapshot sin fundamentales nuevos. Con tope 0 y sin ese bloqueo el residuo es cero por
    construcción. Si hay puntuaciones en la fecha, la cartera objetivo nunca queda vacía.

    El alfa esperado (``expected_excess_return``) procede de la curva causal decil -> retorno real
    anualizado, y es ``NaN`` mientras no haya cohortes cerradas suficientes. Un ``NaN`` de alfa no
    activa una venta ni bloquea una compra: durante el arranque manda la ordenación y, en cuanto hay
    evidencia, mandan los umbrales. Una ausencia de ``meta_rank`` es distinta: invalida la posición
    actual y provoca una salida forzada por cobertura. Todas las comparaciones son en base anual.
    """
    if scores_at_date.empty:
        return [], dict(state.holdings)
    date = str(scores_at_date["snapshot_date"].iloc[0])
    is_quarterly = _is_quarterly(scores_at_date)
    strictness = _strictness(is_quarterly, settings)
    ranked = scores_at_date.dropna(subset=["meta_rank"]).copy()
    ranked["percentile"] = pd.to_numeric(ranked["meta_rank"], errors="coerce") * 100
    ranked = ranked.dropna(subset=["percentile"]).sort_values("percentile", ascending=False)
    percentile = dict(zip(ranked["ticker"].astype(str), ranked["percentile"]))
    alpha_bps = _expected_alpha_bps(ranked)
    candidates = list(dict.fromkeys(ranked["ticker"].astype(str)))

    # Todo se compara en base ANUAL: el alfa esperado ya viene anualizado de la curva por deciles y
    # los umbrales del catálogo son anuales, así que ninguno de los dos necesita conversión. El que
    # sí la necesita es el coste: comisión y slippage se pagan una vez por operación, no cada año, y
    # compararlos con un alfa anual los infravaloraría en cuanto el horizonte baja de 12 meses. Se
    # anualiza el coste de ida y vuelta repartiéndolo sobre el horizonte que dura la posición.
    # Endurecer en snapshots sin fundamentales nuevos significa operar menos en ambos sentidos: el
    # umbral de salida baja (cuesta más gatillar una venta), y tanto la entrada como la rotación
    # exigen más ventaja. La banda de histéresis se ensancha.
    horizon = settings.target_horizon_months
    exit_annual_bps = settings.exit_expected_alpha_bps
    rotation_annual_bps = settings.rotation_edge_bps
    exit_threshold = exit_annual_bps / strictness
    round_trip = _horizon_cost_to_annual_bps(
        2.0 * (settings.commission_bps + settings.slippage_bps), horizon,
    )
    entry_threshold = (exit_annual_bps + round_trip) * strictness
    rotation_threshold = round_trip + rotation_annual_bps * strictness
    floor = _position_floor(settings)
    minimum_months = _minimum_holding_months(settings)
    # Sin fundamentales nuevos y con la variable activa, se puede eliminar una posición mala pero no
    # sustituirla: no hay información nueva que justifique una elección distinta a la ya hecha con
    # datos reales.
    price_only_gate = settings.price_only_sell_only and not is_quarterly

    holders = set(state.holdings)
    removed: set[str] = set()
    orders: list[dict[str, Any]] = []
    # Una posición que no ha cumplido el mínimo de tenencia no puede venderse por ningún motivo, ni
    # por caída de alfa ni por rotación.
    protected = {
        ticker for ticker in holders
        if months_held(state.entry_dates.get(ticker, date), date) < minimum_months
    }

    # 1. Cobertura: una acción sin rank actual no pertenece al universo scoreable. No puede ser la
    # peor "desplazable" y bloquear el bucle de rotación: se vende, aun bajo mínimo de tenencia, y
    # `removed` impide recomprarla en este snapshot. Si no hay reemplazo admisible, queda efectivo.
    scoreable = set(candidates)
    for ticker in sorted(holders - scoreable):
        holders.remove(ticker)
        removed.add(ticker)
        protected.discard(ticker)
        orders.append(_order(
            date, ticker, "sell", "missing_current_score", state.holdings.get(ticker, 0.0), 0.0,
        ))

    # 1-bis. Cobertura por ranking: una posición que cae bajo el suelo deja de pertenecer a una
    # cartera construida sobre este ranking. Es el mismo mandato que `missing_current_score` —el
    # percentil ausente sustituido por un percentil demasiado bajo— y por eso no se compara contra
    # ningún coste: no decide si una operación es rentable, sino si la acción sigue perteneciendo al
    # universo invertible. A diferencia de aquella, sí respeta el mínimo de tenencia, porque la
    # posición sigue siendo scoreable y la exclusión es una preferencia declarada, no una pérdida de
    # cobertura. Se vende siempre: qué pasa con la plaza liberada (recompra o efectivo) lo decide el
    # relleno obligatorio del paso 4, no esta regla.
    if settings.coverage_percentile_floor > 0:
        below_floor = sorted(
            (
                ticker for ticker in holders
                if ticker not in protected
                and percentile.get(ticker, 0.0) < settings.coverage_percentile_floor
            ),
            key=lambda ticker: percentile.get(ticker, 0.0),
        )
        for ticker in below_floor:
            holders.remove(ticker)
            removed.add(ticker)
            orders.append(_order(
                date, ticker, "sell", "below_coverage_percentile",
                state.holdings.get(ticker, 0.0), 0.0,
            ))

    # 2. Ventas a efectivo: con tope de efectivo el destino es efectivo al 0 % con el suelo de
    # diversificación de siempre; bajo `price_only_sell_only` en un snapshot sin fundamentales
    # nuevos, el destino es efectivo transitorio sin suelo propio, porque no hay con qué reemplazar
    # la posición vendida. Con tope 0 y sin ese bloqueo no existe esta vía: vender por umbral
    # obligaría a recomprar en el mismo snapshot.
    if settings.max_cash_weight > 0.0 or price_only_gate:
        cash_exit_floor = floor if settings.max_cash_weight > 0.0 else 0
        below = sorted(
            (
                ticker for ticker in holders
                if ticker not in protected and _below(alpha_bps.get(ticker), exit_threshold)
            ),
            key=lambda ticker: _alpha_or(alpha_bps, ticker, percentile),
        )
        for ticker in below:
            if len(holders) <= cash_exit_floor:
                break
            holders.remove(ticker)
            removed.add(ticker)
            orders.append(_order(date, ticker, "sell", "expected_alpha_below_exit", state.holdings.get(ticker, 0.0), 0.0))

    if not price_only_gate:
        # 3. Compras con histéresis: una entrada nueva debe esperar cubrir su propio coste de ida y
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

        # 4. Rellenos obligatorios: se completa hasta el suelo de diversificación con las mejores
        # por ranking aunque no superen el umbral. Con tope de efectivo 0 ese suelo es el objetivo
        # entero, así que la cartera queda siempre llena; el motivo distingue ambos casos para que
        # la auditoría de órdenes siga diciendo por qué se compró. Nunca se recompra lo vendido en
        # este mismo snapshot.
        mandatory = floor
        reason = "fully_invested_fill" if settings.max_cash_weight <= 0.0 else "cash_floor_fill"
        for ticker in candidates:
            if len(holders) >= mandatory:
                break
            if ticker in holders or ticker in removed:
                continue
            holders.add(ticker)
            orders.append(_order(date, ticker, "buy", reason, 0.0, None))

        # 5. Rotación: un outsider desplaza a la peor posición desplazable solo si su ventaja de
        # alfa esperado paga el coste de ida y vuelta más el margen. Las posiciones protegidas por
        # el mínimo de tenencia nunca son la "peor" desplazable. Con tope de efectivo el outsider
        # debe además superar el umbral de entrada: si no, su plaza natural sería efectivo.
        while len(holders) >= settings.target_size:
            movable = [ticker for ticker in holders if ticker not in protected]
            if not movable:
                break
            outsider = next(
                (
                    ticker for ticker in candidates
                    if ticker not in holders and ticker not in removed
                    and (
                        settings.max_cash_weight <= 0.0
                        or not _below(alpha_bps.get(ticker), entry_threshold)
                    )
                ),
                None,
            )
            if outsider is None:
                break
            worst = min(movable, key=lambda ticker: _alpha_or(alpha_bps, ticker, percentile))
            advantage = _advantage(alpha_bps, outsider, worst)
            if advantage is None or advantage < rotation_threshold:
                break
            holders.remove(worst)
            removed.add(worst)
            holders.add(outsider)
            orders.extend((_order(date, worst, "sell", "displaced_by_net_edge", state.holdings.get(worst, 0.0), 0.0),
                           _order(date, outsider, "buy", "net_edge_over_worst", 0.0, None)))

    invested = _invested_fraction(len(holders), settings, blocked=price_only_gate)
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


def _horizon_cost_to_annual_bps(horizon_bps: float, horizon_months: int) -> float:
    """Anualiza un coste que se paga una sola vez por operación, sobre el horizonte de la posición.

    Compuesto, no lineal: multiplicar linealmente (0,3 % en 3 meses → 1,2 %/año) es el mismo atajo
    aritmético que el proyecto ya corrigió en CAGR e IR. Con horizonte de 12 meses la conversión es
    la identidad; con horizontes más cortos el coste anualizado es mayor, porque la misma comisión
    se repite más veces al año.
    """
    rate = (1.0 + horizon_bps / BPS) ** (12.0 / horizon_months) - 1.0
    return rate * BPS


def _minimum_holding_months(settings: Settings) -> int:
    """Meses mínimos de tenencia, como fracción del horizonte del modelo."""
    horizon = settings.target_horizon_months
    if settings.minimum_holding_period == "none":
        return 0
    if settings.minimum_holding_period == "quarter_horizon":
        return max(1, math.ceil(horizon / 4))
    if settings.minimum_holding_period == "half_horizon":
        return max(1, math.ceil(horizon / 2))
    return horizon  # full_horizon


def _is_quarterly(scores: pd.DataFrame) -> bool:
    return bool(scores["is_quarterly"].iloc[0]) if "is_quarterly" in scores else True


def _position_floor(settings: Settings) -> int:
    """Número mínimo de posiciones. Garantiza el tope de efectivo y acota la concentración.

    Con `target_size` plazas y tope de efectivo `max_cash_weight`, al menos
    `ceil((1 − max_cash_weight) · target_size)` plazas deben estar ocupadas: así el efectivo nunca
    supera el tope y ninguna posición puede acercarse a concentrar la cartera (con 12 plazas y tope
    del 25 %, el suelo son 9 posiciones y el peso máximo implícito queda en torno al 15 %).

    Con tope 0 la fórmula devuelve `target_size` por sí sola: "siempre invertido" no necesita un
    caso especial, es el límite natural de la misma expresión.
    """
    floor = math.ceil((1.0 - settings.max_cash_weight) * settings.target_size)
    return max(1, min(settings.target_size, floor))


def _invested_fraction(holders: int, settings: Settings, blocked: bool = False) -> float:
    """Fracción invertida. El hueco entre plazas cubiertas y objetivo se traduce en efectivo,
    acotado por `max_cash_weight` salvo cuando la compra de reemplazo está bloqueada.

    Hay dos motivos distintos para no llenar la cartera y no se comportan igual:

    - **No hay con qué llenarla** (universo escaso): las plazas ausentes no existen, así que el
      capital se reparte entre las que sí hay y el tope de efectivo decide cuánto se retiene. Aquí
      vivía un error: una rama por política devolvía `holders / target_size` y con 9 de 12 plazas
      dejaba un 25 % en efectivo bajo `fully_invested`, mientras que el tope 0 de `opportunity_cash`
      invertía el 100 %. La política llamada «siempre invertida» retenía más efectivo que la de
      efectivo con tope cero, y ese efectivo no respetaba tope alguno porque no se declaraba en
      ninguna variable.
    - **Hay con qué llenarla pero está prohibido comprar** (`price_only_sell_only` en un snapshot
      sin fundamentales nuevos): el hueco es efectivo transitorio y deliberado. Repartirlo entre las
      supervivientes concentraría la cartera justo cuando se ha decidido no actuar sin información
      nueva, que es lo contrario de lo que persigue la variable. Por eso no se reparte ni se acota.
    """
    if holders >= settings.target_size:
        return 1.0
    if blocked:
        return holders / settings.target_size
    if settings.max_cash_weight <= 0.0:
        return 1.0
    if holders == 0:
        return 1.0 - settings.max_cash_weight
    empty = (settings.target_size - holders) / settings.target_size
    return 1.0 - min(empty, settings.max_cash_weight)


def _strictness(is_quarterly: bool, settings: Settings) -> float:
    return 1.0 if is_quarterly else settings.price_only_strictness_multiplier


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
