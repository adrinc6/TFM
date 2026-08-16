"""Narrativa de la cartera: qué tuvo, cuánto tiempo, cuánto aportó y qué decidió mal.

El TFM reporta hoy la cartera como una curva y un puñado de métricas agregadas. Eso responde a «cuánto
ganó» pero no a «qué hizo», que es lo primero que pregunta cualquiera que mire un sistema de
selección de acciones: qué nombres tuvo, cuáles repitió, cuáles le dieron el resultado y en cuáles
se equivocó. Este módulo construye ese material desde los artefactos que el backtest ya persiste.

**Es descriptivo y no decide nada.** Ninguna cifra de aquí participa en ninguna selección: se
calcula sobre el ganador ya congelado, después de que todas las decisiones estén tomadas. Mirar las
peores decisiones para cambiar la estrategia sería ajustar sobre el resultado conocido.

**De dónde sale cada cosa.** La contribución por acción viene de ``contributions.parquet``, que el
propio backtest emite posición a posición: no se reconstruye aquí, porque recalcular la convención
de exclusión de cotización y la neutralización de retornos corruptos crearía una segunda verdad. Los
resultados realizados vienen del ``realized_pnl_pct`` que ``_price_orders`` ya calcula neto de
costes. La permanencia sale de ``positions.parquet``.

**El mapa de posiciones no genera artefacto nuevo**: ``positions.parquet`` ya es esa tabla en formato
largo —snapshot, ticker, peso—, y duplicarla solo crearía una copia que se desincroniza.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from module.common.utils import write_json
from module.studies.catalog import KNOWN_STRESS_YEARS, SELECTION_UNTIL_YEAR

log = logging.getLogger(__name__)

WINDOWS = ("selection", "confirmation")

# Cuántos nombres se publican en cada extremo de un ranking. Suficiente para que se vea el patrón y
# no tanto como para que la tabla del manuscrito deje de caber en una página.
TOP_N = 15

# Meses posteriores a una venta sobre los que se mide el coste de oportunidad de haber salido.
POST_EXIT_MONTHS = 12


def _year(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["snapshot_date"]).dt.year


def _window_slice(frame: pd.DataFrame, window: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    years = _year(frame)
    if window == "selection":
        return frame.loc[years.le(SELECTION_UNTIL_YEAR)]
    return frame.loc[years.isin(KNOWN_STRESS_YEARS)]


# --- Presencia y permanencia -------------------------------------------------------------------

def holdings_table(
    positions: pd.DataFrame, contributions: pd.DataFrame, orders: pd.DataFrame,
    step_months: int,
) -> pd.DataFrame:
    """Una fila por acción: cuánto estuvo, cuánto pesó y cuánto aportó.

    Es la tabla que sostiene «las que más han estado en cartera» y «las que más aportaron», y se
    construye una sola vez para que ambas lecturas salgan de la misma fuente.
    """
    if positions.empty:
        return pd.DataFrame()
    frame = positions.copy()
    frame["snapshot_date"] = frame["snapshot_date"].astype(str)
    snapshots = sorted(frame["snapshot_date"].unique())
    total_snapshots = len(snapshots)
    rows = []
    for ticker, group in frame.groupby("ticker"):
        dates = sorted(group["snapshot_date"].unique())
        rows.append({
            "ticker": str(ticker),
            "snapshots_held": len(dates),
            "months_held_total": len(dates) * step_months,
            "share_of_period": len(dates) / total_snapshots if total_snapshots else np.nan,
            "episodes": _episodes(dates, snapshots),
            "mean_weight": float(group["weight"].mean()),
            "max_weight": float(group["weight"].max()),
            "first_snapshot": dates[0],
            "last_snapshot": dates[-1],
            "longest_streak_snapshots": _longest_streak(dates, snapshots),
        })
    table = pd.DataFrame(rows)

    gross = (
        contributions.groupby("ticker")["contribution"].sum().rename("gross_contribution")
        if not contributions.empty else pd.Series(dtype=float, name="gross_contribution")
    )
    costs = _costs_by_ticker(orders)
    table = table.merge(gross, on="ticker", how="left").merge(costs, on="ticker", how="left")
    table["gross_contribution"] = table["gross_contribution"].fillna(0.0)
    table["cost_contribution"] = table["cost_contribution"].fillna(0.0)
    # El coste resta: la contribución neta es lo que la acción dejó en la cartera después de pagar
    # por entrar y salir. Es la cifra honesta para ordenar aciertos y errores.
    table["net_contribution"] = table["gross_contribution"] - table["cost_contribution"]
    return table.sort_values("snapshots_held", ascending=False, ignore_index=True)


def _costs_by_ticker(orders: pd.DataFrame) -> pd.DataFrame:
    """Comisión y slippage pagados por acción, como fracción del valor de la cartera.

    El drag del motor es una cifra de cartera, no de posición; repartirlo por el nocional de cada
    orden es la única imputación que suma exactamente el total y no inventa ninguna clave de
    reparto.
    """
    empty = pd.DataFrame({"ticker": pd.Series(dtype=str), "cost_contribution": pd.Series(dtype=float)})
    if orders.empty or "commission_amount" not in orders.columns:
        return empty
    frame = orders.copy()
    frame["paid"] = frame["commission_amount"].fillna(0.0) + frame["slippage_amount"].fillna(0.0)
    notional = frame["notional"].replace(0, np.nan)
    # `notional` está en unidades de valor de cartera igual que `paid`, así que el cociente entre el
    # coste pagado y el valor operado, multiplicado por el peso operado, da la fracción perdida.
    frame["cost_contribution"] = frame["paid"] / notional * (
        frame["weight_after"].fillna(0.0) - frame["weight_before"].fillna(0.0)
    ).abs()
    grouped = frame.groupby("ticker")["cost_contribution"].sum().reset_index()
    return grouped if not grouped.empty else empty


def _episodes(dates: list[str], snapshots: list[str]) -> int:
    """Veces distintas que la acción entró en cartera, no snapshots en los que estuvo."""
    index = {date: position for position, date in enumerate(snapshots)}
    positions = sorted(index[date] for date in dates)
    return 1 + sum(1 for previous, current in pairwise(positions) if current != previous + 1)


def _longest_streak(dates: list[str], snapshots: list[str]) -> int:
    index = {date: position for position, date in enumerate(snapshots)}
    positions = sorted(index[date] for date in dates)
    best = streak = 1
    for previous, current in pairwise(positions):
        streak = streak + 1 if current == previous + 1 else 1
        best = max(best, streak)
    return best


# --- Decisiones --------------------------------------------------------------------------------

def _full_exits(orders: pd.DataFrame) -> pd.DataFrame:
    """Ventas que **cierran** la posición, no los recortes de rebalanceo.

    Un rebalanceo que baja el peso también emite una orden de venta con su resultado realizado, pero
    la posición sigue abierta: contarla como operación cerrada mezclaría dos cosas distintas y
    ensuciaría tanto los aciertos como los errores. Una salida es `weight_after == 0`.
    """
    if orders.empty or "side" not in orders.columns:
        return pd.DataFrame()
    sales = orders.loc[orders["side"].eq("sell")].copy()
    if sales.empty:
        return sales
    return sales.loc[sales["weight_after"].fillna(0.0).abs().le(1e-12)]


def round_trips(orders: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """Operaciones cerradas, con lo que se ganó o se perdió en cada una.

    ``realized_pnl_pct`` ya viene neto de comisión y slippage de ida y vuelta desde
    ``_price_orders``: no se recalcula aquí. La permanencia se toma del último snapshot en que la
    acción **seguía** en cartera, que es el anterior a la venta: el día que se vende ya no figura.
    """
    sales = _full_exits(orders)
    if sales.empty or "realized_pnl_pct" not in sales.columns:
        return pd.DataFrame()
    sales = sales.loc[sales["realized_pnl_pct"].notna()].copy()
    if sales.empty:
        return pd.DataFrame()
    sales["stamp"] = pd.to_datetime(sales["snapshot_date"])
    if not positions.empty:
        held = positions[["ticker", "snapshot_date", "entry_date", "months_held"]].copy()
        held["stamp"] = pd.to_datetime(held["snapshot_date"])
        sales = pd.merge_asof(
            sales.sort_values("stamp"),
            held.sort_values("stamp").drop(columns="snapshot_date"),
            on="stamp", by="ticker", direction="backward", allow_exact_matches=False,
        )
    columns = [
        column for column in
        ["ticker", "snapshot_date", "entry_date", "months_held", "reason", "realized_pnl_pct",
         "buy_price", "sell_price", "notional"]
        if column in sales.columns
    ]
    return sales[columns].sort_values("realized_pnl_pct", ascending=False, ignore_index=True)


def sold_and_recovered(
    orders: pd.DataFrame, prices: pd.DataFrame, benchmark: pd.DataFrame, months: int = POST_EXIT_MONTHS,
) -> pd.DataFrame:
    """Qué hizo cada acción vendida en los meses siguientes, frente al índice.

    Es el coste de oportunidad de salir, y la crítica honesta a una doctrina de umbrales: vender
    barato para no pagar costes solo compensa si lo vendido no sigue subiendo. Se mide contra el
    benchmark porque el dinero liberado se podría haber puesto en el índice.
    """
    if orders.empty or prices.empty:
        return pd.DataFrame()
    sales = _full_exits(orders).copy()
    if sales.empty:
        return pd.DataFrame()
    price = prices[["ticker", "snapshot_date", "price"]].copy()
    price["snapshot_date"] = price["snapshot_date"].astype(str)
    price["stamp"] = pd.to_datetime(price["snapshot_date"])
    index = benchmark[["snapshot_date", "price"]].copy()
    index["snapshot_date"] = index["snapshot_date"].astype(str)
    index["stamp"] = pd.to_datetime(index["snapshot_date"])

    rows = []
    for row in sales.itertuples(index=False):
        exit_stamp = pd.Timestamp(str(row.snapshot_date))
        horizon = exit_stamp + pd.DateOffset(months=months)
        stock = _return_between(price.loc[price["ticker"].eq(row.ticker)], exit_stamp, horizon)
        market = _return_between(index, exit_stamp, horizon)
        if stock is None or market is None:
            continue
        rows.append({
            "ticker": row.ticker,
            "exit_snapshot": str(row.snapshot_date),
            "reason": getattr(row, "reason", None),
            "post_exit_return": stock,
            "post_exit_benchmark_return": market,
            "post_exit_excess": (1 + stock) / (1 + market) - 1,
            "months": months,
        })
    frame = pd.DataFrame(rows)
    return frame.sort_values("post_exit_excess", ascending=False, ignore_index=True) if not frame.empty else frame


def _return_between(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    """Retorno entre dos fechas usando el último precio observado en cada una, o None si falta."""
    at_start = frame.loc[frame["stamp"].le(start)].tail(1)
    at_end = frame.loc[frame["stamp"].le(end)].tail(1)
    if at_start.empty or at_end.empty:
        return None
    first, last = float(at_start["price"].iloc[0]), float(at_end["price"].iloc[0])
    if first <= 0 or at_end["stamp"].iloc[0] <= at_start["stamp"].iloc[0]:
        return None
    return last / first - 1.0


# --- Estructura de la cartera ------------------------------------------------------------------

def concentration(positions: pd.DataFrame, equity: pd.DataFrame) -> dict[str, Any]:
    """Cuán concentrada estuvo la cartera y cuánto tiempo pasó en efectivo."""
    if positions.empty:
        return {"available": False}
    by_snapshot = positions.groupby("snapshot_date")["weight"]
    hhi = by_snapshot.apply(lambda weights: float((weights ** 2).sum()))
    sizes = by_snapshot.size()
    cash = equity["cash_weight"] if "cash_weight" in equity.columns else pd.Series(dtype=float)
    return {
        "available": True,
        "distinct_tickers_ever_held": int(positions["ticker"].nunique()),
        "mean_positions": float(sizes.mean()),
        "min_positions": int(sizes.min()),
        "max_positions": int(sizes.max()),
        "mean_hhi": float(hhi.mean()),
        "max_hhi": float(hhi.max()),
        "mean_cash_weight": float(cash.mean()) if not cash.empty else None,
        "snapshots_fully_invested": int((cash <= 1e-9).sum()) if not cash.empty else None,
    }


def holding_duration(positions: pd.DataFrame, step_months: int) -> dict[str, Any]:
    """Distribución de la permanencia: cuánto dura una posición en esta cartera."""
    if positions.empty or "months_held" not in positions.columns:
        return {"available": False}
    # La permanencia de una posición es la que tenía en su último snapshot, no la media de todos.
    final = positions.sort_values("snapshot_date").groupby(["ticker", "entry_date"]).tail(1)
    months = pd.to_numeric(final["months_held"], errors="coerce").dropna()
    if months.empty:
        return {"available": False}
    return {
        "available": True,
        "episodes": len(months),
        "mean_months": float(months.mean()),
        "median_months": float(months.median()),
        "p90_months": float(np.percentile(months, 90)),
        "max_months": float(months.max()),
        "share_single_snapshot": float((months <= step_months).mean()),
    }


def sector_exposure(positions: pd.DataFrame, sector_by_ticker: Mapping[str, str]) -> dict[str, Any]:
    """Peso medio por sector, con la salvedad de procedencia **dentro** del bloque.

    El sector viene de `profiles.parquet`, que es una foto **actual** de Finnhub y no una serie
    point-in-time: una empresa reclasificada aparece hoy en su sector de hoy también en 2015. Es el
    mismo uso descriptivo que ya hace la neutralización por sector, nunca una señal, y por eso la
    salvedad viaja pegada al dato en vez de quedarse en la metodología.
    """
    if positions.empty or not sector_by_ticker:
        return {"available": False, "sector_is_point_in_time": False}
    frame = positions.copy()
    frame["sector"] = frame["ticker"].map(sector_by_ticker).fillna("N/A")
    snapshots = frame["snapshot_date"].nunique()
    by_sector = frame.groupby("sector")["weight"].agg(["sum", "size", "max"])
    by_sector["mean_portfolio_weight"] = by_sector["sum"] / snapshots
    rows = (
        by_sector.reset_index()
        .rename(columns={"size": "position_snapshots", "max": "max_single_weight"})
        .drop(columns="sum")
        .sort_values("mean_portfolio_weight", ascending=False)
    )
    return {
        "available": True,
        "sector_is_point_in_time": False,
        "sector_source": "raw/profiles.parquet::finnhubIndustry (foto actual, no PIT)",
        "sectors": rows.to_dict("records"),
    }


# --- Ensamblado --------------------------------------------------------------------------------

def _records(frame: pd.DataFrame, column: str, ascending: bool, top: int = TOP_N) -> list[dict[str, Any]]:
    if frame.empty or column not in frame.columns:
        return []
    ordered = frame.sort_values(column, ascending=ascending).head(top)
    return ordered.astype(object).where(pd.notna(ordered), None).to_dict("records")


def build_portfolio_narrative(
    evidence_dir: Path, prepared: Path, configuration: Mapping[str, Any],
    sector_by_ticker: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Bloque narrativo y tabla por acción, ambos sobre el ganador ya congelado."""
    step_months = int(configuration.get("snapshot_step_months") or 1)
    positions = _read(evidence_dir / "positions.parquet")
    orders = _read(evidence_dir / "orders.parquet")
    contributions = _read(evidence_dir / "contributions.parquet")
    equity = _read(evidence_dir / "equity.parquet")
    prices = _read(prepared / "asset_price_point_in_time.parquet")
    benchmark = _read(prepared / "benchmark_point_in_time.parquet")

    full_table = holdings_table(positions, contributions, orders, step_months)
    payload: dict[str, Any] = {
        "snapshot_step_months": step_months,
        "position_map_artifact": "positions.parquet",
        "windows": {},
        "caveats": [
            (
                "Todo este bloque es descriptivo y posterior a la congelación del ganador: ninguna "
                "cifra participó en ninguna selección."
            ),
            (
                "El sector no es point-in-time: procede de una foto actual de Finnhub y solo se usa "
                "para agrupar."
            ),
            (
                "Las peores decisiones se leen con el resultado ya conocido. Señalan dónde falló la "
                "doctrina, no una regla que se pueda añadir sin volver a ajustar sobre el resultado."
            ),
        ],
    }
    for window in WINDOWS:
        window_positions = _window_slice(positions, window)
        window_orders = _window_slice(orders, window)
        window_contributions = _window_slice(contributions, window)
        window_equity = _window_slice(equity, window)
        table = holdings_table(window_positions, window_contributions, window_orders, step_months)
        trips = round_trips(window_orders, window_positions)
        payload["windows"][window] = {
            "available": not window_positions.empty,
            "most_held": _records(table, "snapshots_held", ascending=False),
            "best_contributors": _records(table, "net_contribution", ascending=False),
            "worst_contributors": _records(table, "net_contribution", ascending=True),
            "best_round_trips": _records(trips, "realized_pnl_pct", ascending=False),
            "worst_round_trips": _records(trips, "realized_pnl_pct", ascending=True),
            "sold_and_recovered": _records(
                sold_and_recovered(window_orders, prices, benchmark), "post_exit_excess",
                ascending=False,
            ),
            "concentration": concentration(window_positions, window_equity),
            "holding_duration": holding_duration(window_positions, step_months),
            "sector_exposure": sector_exposure(window_positions, sector_by_ticker or {}),
        }
    return payload, full_table


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def write_portfolio_narrative(
    directory: Path, payload: Mapping[str, Any], holdings: pd.DataFrame,
) -> None:
    from module.common.utils import write_parquet

    write_json(dict(payload), directory / "portfolio_narrative.json")
    write_parquet(holdings, directory / "portfolio_narrative_holdings.parquet")
