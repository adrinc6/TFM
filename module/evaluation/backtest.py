"""Backtest contable de la cartera dinámica; SPY es únicamente el benchmark.

Este módulo es **contabilidad, no ciencia**: no calcula ni reporta Rank-IC. La métrica predictiva se
mide en el runner sobre los diagnósticos del meta, y tenerla también aquí producía dos poblaciones
distintas bajo la misma clave (una con recorte temporal y otra sin él).

Las métricas económicas se devuelven **segmentadas**: la ventana de selección (hasta
``SELECTION_UNTIL_YEAR``) es la única que alimentó decisiones, y 2025-26 se reporta aparte como
confirmación fuera de muestra. Mezclarlas en una sola cifra de portada infla el resultado con un
periodo que ninguna decisión pudo ver.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from environment import Settings
from module.evaluation.portfolio import PortfolioState, decide_orders
from module.studies.catalog import KNOWN_STRESS_YEARS, SELECTION_UNTIL_YEAR


# Convención tipo CRSP para una posición que desaparece del panel (exclusión de cotización): el
# inversor no recupera el 100 %, pero tampoco lo pierde todo. Marcar la posición plana, como hacía
# la versión anterior, es el supuesto más favorable posible y sesga el resultado al alza.
DELISTING_RETURN = -0.30


@dataclass
class BacktestResult:
    positions: pd.DataFrame
    orders: pd.DataFrame
    equity: pd.DataFrame
    annual_metrics: pd.DataFrame
    summary: dict = field(default_factory=dict)


def run_backtest(
    scores: pd.DataFrame, prices: pd.DataFrame, benchmark: pd.DataFrame, settings: Settings,
    targets: pd.DataFrame | None = None,
) -> BacktestResult:
    from module.evaluation.profiles import apply_profile
    scores = apply_profile(scores, settings.profile, targets).sort_values("snapshot_date")
    prices_by_date = _prices(prices)
    benchmark_by_date = {row.snapshot_date: float(row.price) for row in benchmark.itertuples(index=False)}
    grouped = {date: frame for date, frame in scores.groupby("snapshot_date", sort=False)}
    snapshots = sorted(grouped)
    if not snapshots:
        raise ValueError("No hay snapshots para el backtest.")
    state, value, benchmark_value = PortfolioState.empty(), 100.0, 100.0
    previous_prices: dict[str, float] = {}
    previous_benchmark = benchmark_by_date.get(snapshots[0])
    if not previous_benchmark:
        raise ValueError("Sin benchmark PIT inicial.")
    positions_rows: list[dict] = []
    orders_rows: list[dict] = []
    equity_rows: list[dict] = []
    corrupt: list[dict] = []
    delisted: list[dict] = []
    cash_weight = 0.0
    for index, date in enumerate(snapshots):
        stock_return, drifted, cash_weight = _mark_to_market(
            state.holdings, cash_weight, previous_prices, prices_by_date, date,
            settings.max_monthly_position_return, corrupt, delisted,
        )
        state.holdings = drifted
        price = benchmark_by_date.get(date, previous_benchmark)
        benchmark_return = price / previous_benchmark - 1 if index else 0.0
        benchmark_value *= 1 + benchmark_return
        before_orders = value * (1 + stock_return)
        current_prices = prices_by_date.get(date, {})
        frame = grouped[date]
        tradable = frame.loc[frame["ticker"].map(lambda ticker: bool(current_prices.get(ticker, 0) > 0))]
        orders, target = decide_orders(state, tradable, settings)
        if not target:
            target = {ticker: weight for ticker, weight in state.holdings.items() if current_prices.get(ticker, 0) > 0}
        if not target:
            raise ValueError(f"No hay acciones negociables para invertir en {date}.")
        target = _capped(target)
        cash_weight = max(0.0, 1.0 - sum(target.values()))
        priced, drag = _price_orders(orders, current_prices, state.holdings, before_orders, settings, previous_prices)
        value_after = before_orders * (1 - drag)
        prior_entries = dict(state.entry_dates)
        prior_prices = dict(state.entry_prices)
        buy_costs = {row["ticker"]: row["commission_amount"] + row["slippage_amount"] for row in priced if row["side"] == "buy"}
        position_values = {ticker: value_after * weight for ticker, weight in target.items()}
        units = {ticker: position_values[ticker] / current_prices[ticker] for ticker in target}
        state = PortfolioState(
            holdings=target,
            entry_dates={ticker: prior_entries.get(ticker, date) for ticker in target},
            entry_prices={ticker: prior_prices.get(ticker, current_prices[ticker]) for ticker in target},
            units=units,
            entry_costs={ticker: buy_costs.get(ticker, 0.0) for ticker in target},
            costs_paid=state.costs_paid + before_orders * drag,
        )
        previous_prices = {ticker: current_prices[ticker] for ticker in target}
        turnover = sum(abs(row["weight_after"] - row["weight_before"]) for row in priced)
        for ticker, weight in target.items():
            positions_rows.append({"snapshot_date": date, "ticker": ticker, "weight": weight, "entry_date": state.entry_dates[ticker], "entry_price": state.entry_prices[ticker], "valuation_price": current_prices[ticker], "units": units[ticker], "market_value": position_values[ticker], "entry_cost": state.entry_costs[ticker], "months_held": _months(state.entry_dates[ticker], date), "current_percentile": _percentile(ticker, frame)})
        orders_rows.extend(priced)
        equity_rows.append({"snapshot_date": date, "period_start_portfolio_value": value, "period_start_benchmark_value": benchmark_value / (1 + benchmark_return) if 1 + benchmark_return else benchmark_value, "portfolio_value": value_after, "benchmark_value": benchmark_value, "portfolio_return": value_after / value - 1, "benchmark_return": benchmark_return, "excess_return": value_after / value - 1 - benchmark_return, "turnover_pct": turnover, "gross_return": stock_return, "cost_drag": drag, "positions_value": sum(position_values.values()), "cash_weight": cash_weight, "invested_weight": sum(target.values()), "cumulative_costs": state.costs_paid})
        value, previous_benchmark = value_after, price
    equity = pd.DataFrame(equity_rows)
    annual = _annual_metrics(equity, settings)
    summary = _summary(equity, annual, settings)
    summary.update({
        "corrupt_returns_neutralized": len(corrupt),
        "delisted_positions": len(delisted),
        "portfolio_policy": "expected_alpha_bps",
        "sizing_mode": settings.sizing_mode,
        "cash_policy": settings.cash_policy,
    })
    return BacktestResult(pd.DataFrame(positions_rows), pd.DataFrame(orders_rows), equity, annual, summary)


def _prices(prices: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {str(date): dict(zip(group.ticker, group.price)) for date, group in prices.groupby("snapshot_date")}


def _capped(weights: dict[str, float]) -> dict[str, float]:
    """Normaliza solo si los pesos suman más de 1. Un residuo por debajo de 1 es efectivo legítimo."""
    total = sum(weights.values())
    if total <= 0:
        return {}
    if total <= 1.0:
        return dict(weights)
    return {ticker: weight / total for ticker, weight in weights.items()}


def _periods_per_year(settings: Settings) -> float:
    return 12.0 / max(int(settings.snapshot_step_months), 1)


def _mark_to_market(
    holdings: dict[str, float], cash_weight: float, previous: dict[str, float],
    prices: dict[str, dict[str, float]], date: str, maximum: float,
    corrupt: list[dict], delisted: list[dict],
) -> tuple[float, dict[str, float], float]:
    """Revaloriza la cartera y devuelve (retorno, pesos nuevos, efectivo nuevo).

    El efectivo rinde 0 %, de modo que no aporta retorno pero **sí participa en la renormalización**:
    los pesos entrantes suman uno contando el efectivo, así que los salientes deben hacerlo también.
    Renormalizar solo la parte invertida haría que el efectivo desapareciera de la contabilidad en
    cuanto los precios se movieran.

    Una posición cuyo precio desaparece del panel se trata como exclusión de cotización y sufre
    ``DELISTING_RETURN``. La alternativa —marcarla plana— regala al backtest un rescate del 100 % en
    exactamente los casos en que una acción suele estar colapsando. Esa posición ya no es negociable,
    así que se liquida contra efectivo: no se puede seguir valorando algo sin precio.
    """
    if not holdings:
        return 0.0, {}, min(max(cash_weight, 0.0), 1.0)
    grown: dict[str, float] = {}
    liquidated = 0.0
    for ticker, weight in holdings.items():
        old, new = previous.get(ticker), prices.get(date, {}).get(ticker)
        tradable = bool(new and new > 0)
        if old and old > 0 and tradable:
            change = new / old - 1
        elif old and old > 0:
            change = DELISTING_RETURN
            delisted.append({"snapshot_date": date, "ticker": ticker, "assumed_return": change})
        else:
            change = 0.0
        if abs(change) > maximum:
            corrupt.append({"snapshot_date": date, "ticker": ticker, "position_return": change})
            change = 0.0
        value = weight * (1 + change)
        if tradable:
            grown[ticker] = value
        else:
            liquidated += value
    total = sum(grown.values()) + liquidated + cash_weight
    if total <= 0:
        return -1.0, {}, 1.0
    return (
        total - 1.0,
        {ticker: value / total for ticker, value in grown.items()},
        (liquidated + cash_weight) / total,
    )


def _price_orders(orders: list[dict], prices: dict[str, float], holdings: dict[str, float], value: float, settings: Settings, fallback: dict[str, float]) -> tuple[list[dict], float]:
    rows, total = [], 0.0
    rate = (settings.commission_bps + settings.slippage_bps) / 10_000
    for order in orders:
        before, after = float(order.get("weight_before") or holdings.get(order["ticker"], 0.0)), float(order.get("weight_after") or 0.0)
        notional = abs(after - before) * value
        commission, slippage = notional * settings.commission_bps / 10_000, notional * settings.slippage_bps / 10_000
        rows.append({**order, "weight_before": before, "weight_after": after, "price": prices.get(order["ticker"], fallback.get(order["ticker"])), "notional": notional, "commission_amount": commission, "slippage_amount": slippage})
        total += notional * rate
    return rows, total / value if value else 0.0


def _months(start: str, end: str) -> int:
    left, right = pd.Timestamp(start), pd.Timestamp(end)
    return max(0, (right.year - left.year) * 12 + right.month - left.month)


def _percentile(ticker: str, frame: pd.DataFrame) -> float:
    row = frame.loc[frame.ticker.eq(ticker), "meta_rank"]
    return float(row.iloc[0] * 100) if not row.empty else float("nan")


def _information_ratio(excess: np.ndarray, periods_per_year: float) -> float:
    """Ratio de información **anualizado**: única definición del proyecto.

    Media y desviación típica de los excesos por periodo, escaladas por la raíz del número de
    periodos al año. Antes convivían dos fórmulas incompatibles con este mismo nombre —una sin
    anualizar y otra como mediana de ratios anuales—, persistidas en artefactos distintos, lo que
    hacía que dos cifras del mismo backtest no fueran comparables.
    """
    if len(excess) < 2:
        return 0.0
    tracking = float(np.std(excess, ddof=1))
    if tracking <= 0:
        return 0.0
    return float(np.mean(excess) / tracking * np.sqrt(periods_per_year))


def _annual_metrics(equity: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    periods = _periods_per_year(settings)
    rows = []
    for year, group in equity.assign(year=pd.to_datetime(equity.snapshot_date).dt.year).groupby("year"):
        portfolio_return = group.portfolio_value.iloc[-1] / group.period_start_portfolio_value.iloc[0] - 1
        benchmark_return = group.benchmark_value.iloc[-1] / group.period_start_benchmark_value.iloc[0] - 1
        drawdown = 1 - group.portfolio_value.to_numpy() / np.maximum.accumulate(group.portfolio_value.to_numpy())
        rows.append({
            "year": int(year),
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return,
            "alpha": (1 + portfolio_return) / (1 + benchmark_return) - 1,
            "beats_benchmark": portfolio_return > benchmark_return,
            "max_drawdown_year": float(drawdown.max()),
            "information_ratio_year": _information_ratio(group.excess_return.to_numpy(), periods),
            "mean_cash_weight": float(group.cash_weight.mean()),
            "turnover": float(group.turnover_pct.sum()),
        })
    return pd.DataFrame(rows)


def _window_metrics(equity: pd.DataFrame, annual: pd.DataFrame, periods_per_year: float) -> dict:
    """Métricas económicas de una ventana temporal. Una sola definición, aplicada a cada segmento."""
    if equity.empty or annual.empty:
        return {}
    dates = pd.to_datetime(equity.snapshot_date)
    years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, 1 / 12)
    cagr = (equity.portfolio_value.iloc[-1] / equity.period_start_portfolio_value.iloc[0]) ** (1 / years) - 1
    benchmark_cagr = (equity.benchmark_value.iloc[-1] / equity.period_start_benchmark_value.iloc[0]) ** (1 / years) - 1
    excess = equity.excess_return.to_numpy()
    drawdown = 1 - equity.portfolio_value.to_numpy() / np.maximum.accumulate(equity.portfolio_value.to_numpy())
    return {
        "years": float(years),
        "cagr_portfolio": float(cagr),
        "cagr_benchmark": float(benchmark_cagr),
        # Exceso geométrico real: el cociente de acumulados, no la resta de dos CAGR.
        "geometric_excess_return": float((1 + cagr) / (1 + benchmark_cagr) - 1),
        "information_ratio": _information_ratio(excess, periods_per_year),
        "max_drawdown": float(drawdown.max()),
        "beat_rate": float((annual.alpha > 0).mean()),
        "mean_annual_alpha": float(annual.alpha.mean()),
        "median_annual_alpha": float(annual.alpha.median()),
        "worst_year_alpha": float(annual.alpha.min()),
        "annualized_turnover": float(equity["turnover_pct"].mean() * periods_per_year),
        "mean_cash_weight": float(equity["cash_weight"].mean()),
        "total_cost_drag": float(equity["cost_drag"].sum()),
        "n_periods": len(equity),
    }


def _summary(equity: pd.DataFrame, annual: pd.DataFrame, settings: Settings) -> dict:
    """Métricas de la ventana de selección en la raíz; confirmación y curva completa aparte."""
    periods = _periods_per_year(settings)
    years = pd.to_datetime(equity.snapshot_date).dt.year
    selection = equity.loc[years.le(SELECTION_UNTIL_YEAR)]
    stress = equity.loc[years.isin(KNOWN_STRESS_YEARS)]
    annual_selection = annual.loc[annual["year"].le(SELECTION_UNTIL_YEAR)]
    annual_stress = annual.loc[annual["year"].isin(KNOWN_STRESS_YEARS)]
    return {
        **_window_metrics(selection, annual_selection, periods),
        "selection_until_year": SELECTION_UNTIL_YEAR,
        "confirmation": _window_metrics(stress, annual_stress, periods),
        "full_curve": _window_metrics(equity, annual, periods),
        "commission_bps": settings.commission_bps,
        "slippage_bps": settings.slippage_bps,
        "target_size": settings.target_size,
        "exit_expected_alpha_bps": settings.exit_expected_alpha_bps,
        "rotation_edge_bps": settings.rotation_edge_bps,
        "max_cash_weight": settings.max_cash_weight,
    }
