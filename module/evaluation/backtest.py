"""Simulador puro de backtest: aplica ordenes a precios PIT y produce curva de equity,
costes, y metricas anuales.

No decide nada. Toda la logica de decision vive en `module/portfolio.py`. Este modulo:

  1. Itera por cada snapshot.
  2. Actualiza valor de mercado de las posiciones existentes con los precios PIT.
  3. Pide a portfolio las ordenes de esa fecha.
   4. Aplica costes de comision y slippage.
  5. Registra equity, positions y orders.
  6. Calcula metricas anuales y summary compatible con el ranking de Fase 6.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from environment import Settings
from module.evaluation.portfolio import PortfolioState, decide_orders
from module.common.utils import read_parquet, write_json, write_parquet

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    positions: pd.DataFrame
    orders: pd.DataFrame
    equity: pd.DataFrame
    annual_metrics: pd.DataFrame
    summary: dict = field(default_factory=dict)


def run_backtest_from_run_dir(settings: Settings, agent_run_dir: Path | None = None) -> BacktestResult:
    """Lanza el backtest para el directorio exacto de agentes de un escenario.

    El fallback por fecha de modificación sólo sirve al modo CLI ``backtest`` aislado. La
    orquestación de runs debe transportar ``agent_run_dir`` desde su propia etapa de agentes.
    """
    processed = settings.processed_output_dir
    if agent_run_dir is None:
        agents_root = processed / "agents"
        run_dirs = sorted(
            [path for path in agents_root.iterdir() if path.is_dir()], key=lambda path: path.stat().st_mtime
        ) if agents_root.exists() else []
        if not run_dirs:
            raise RuntimeError(f"No hay run_dir de agentes en {agents_root}. Ejecuta primero RUN_MODE='agents'.")
        run_dir = run_dirs[-1]
        log.warning("Backtest aislado: se usa el agente más reciente %s.", run_dir.name)
    else:
        run_dir = Path(agent_run_dir)
        if not (run_dir / "agent_scores.parquet").exists():
            raise FileNotFoundError(f"El agente del escenario no contiene scores: {run_dir}")

    scores = read_parquet(run_dir / "agent_scores.parquet", "RUN_MODE='agents'")
    asset_prices = read_parquet(
        processed / "asset_price_point_in_time.parquet", "RUN_MODE='dataset'"
    )
    benchmark = read_parquet(
        processed / "benchmark_point_in_time.parquet", "RUN_MODE='dataset'"
    )
    diagnostics_path = run_dir / "rank_ic_diagnostics.parquet"
    diagnostics = pd.read_parquet(diagnostics_path) if diagnostics_path.exists() else None

    result = run_backtest(scores, asset_prices, benchmark, settings, diagnostics)
    _write_outputs(result, run_dir, settings)
    log.info("Backtest completado en %s", run_dir)
    return result


def run_backtest(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    settings: Settings,
    diagnostics: pd.DataFrame | None = None,
) -> BacktestResult:
    """Simula la cartera fecha a fecha. Pura funcion: no escribe nada al disco.

    `diagnostics` son los rank-IC OOS por agente y cohorte. Si se pasan, el summary incluye las
    senales de aprendizaje que usa la seleccion. El `settings.profile` reordena la seleccion por
    estilo de inversor (ver module.profiles) sin reentrenar: mismas señales, distinta cartera.
    """
    from module.evaluation.profiles import apply_profile
    scores = apply_profile(scores, settings.profile)
    scores = scores.sort_values("snapshot_date").copy()
    price_index = _price_lookup(prices)
    benchmark_index = _benchmark_lookup(benchmark)

    snapshots = sorted(scores["snapshot_date"].unique())
    if not snapshots:
        raise ValueError("No hay snapshots en los scores del backtest.")

    state = PortfolioState.empty()
    initial_value = 100.0
    portfolio_value = initial_value
    benchmark_value = initial_value

    previous_benchmark_price = benchmark_index.get(snapshots[0])
    if previous_benchmark_price is None:
        raise ValueError(f"Sin precio de benchmark en {snapshots[0]}.")

    positions_rows: list[dict] = []
    orders_rows: list[dict] = []
    equity_rows: list[dict] = []
    corrupt_log: list[dict] = []
    previous_position_prices: dict[str, float] = {}

    for index, snapshot_date in enumerate(snapshots):
        gross_return = _mark_to_market(
            state.holdings, previous_position_prices, price_index, snapshot_date,
            settings.max_monthly_position_return, corrupt_log,
        )
        portfolio_value_pre_orders = portfolio_value * (1 + gross_return)

        current_benchmark_price = benchmark_index.get(snapshot_date, previous_benchmark_price)
        benchmark_return = current_benchmark_price / previous_benchmark_price - 1 if index > 0 else 0.0
        benchmark_value *= 1 + benchmark_return

        scores_at_date = scores.loc[scores["snapshot_date"] == snapshot_date]
        raw_orders = decide_orders(state, scores_at_date, settings)
        priced_orders, cost_drag = _price_orders(raw_orders, price_index, snapshot_date,
                                                  state.holdings, settings)

        portfolio_value_post_orders = portfolio_value_pre_orders * (1 - cost_drag)
        portfolio_return = portfolio_value_post_orders / portfolio_value - 1 if index > 0 else 0.0

        state = state.apply(priced_orders, price_index.get(snapshot_date, {}))
        turnover = sum(abs(order["weight_after"] - order["weight_before"])
                       for order in priced_orders if order["weight_before"] is not None
                       and order["weight_after"] is not None)

        previous_position_prices = {
            ticker: price_index[snapshot_date].get(ticker, previous_position_prices.get(ticker))
            for ticker in state.holdings
        }

        for ticker, weight in state.holdings.items():
            entry_date = state.entry_dates.get(ticker, snapshot_date)
            positions_rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "ticker": ticker,
                    "weight": weight,
                    "entry_date": entry_date,
                    "entry_price": state.entry_prices.get(ticker),
                    "valuation_price": price_index.get(snapshot_date, {}).get(ticker),
                    "months_held": state.months_held.get(ticker, 0),
                    "current_percentile": float(_percentile_of(ticker, scores_at_date)),
                }
            )

        for order in priced_orders:
            orders_rows.append(order)

        equity_rows.append(
            {
                "snapshot_date": snapshot_date,
                "portfolio_value": portfolio_value_post_orders,
                "benchmark_value": benchmark_value,
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
                "excess_return": portfolio_return - benchmark_return,
                "turnover_pct": turnover,
            }
        )

        portfolio_value = portfolio_value_post_orders
        previous_benchmark_price = current_benchmark_price

    positions = pd.DataFrame(positions_rows)
    orders = pd.DataFrame(orders_rows)
    equity = pd.DataFrame(equity_rows)
    annual_metrics = _annual_metrics(equity)
    summary = _summary(equity, annual_metrics, settings, diagnostics)
    summary["corrupt_returns_neutralized"] = len(corrupt_log)
    if corrupt_log:
        log.warning("Guarda anti-artefactos: %s retornos corruptos neutralizados", len(corrupt_log))

    return BacktestResult(
        positions=positions, orders=orders, equity=equity,
        annual_metrics=annual_metrics, summary=summary,
    )


def _price_lookup(prices: pd.DataFrame) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    for row in prices.itertuples(index=False):
        lookup.setdefault(row.snapshot_date, {})[row.ticker] = float(row.price)
    return lookup


def _benchmark_lookup(benchmark: pd.DataFrame) -> dict[str, float]:
    return {row.snapshot_date: float(row.price) for row in benchmark.itertuples(index=False)}


def _mark_to_market(
    holdings: dict[str, float],
    previous_prices: dict[str, float],
    price_index: dict[str, dict[str, float]],
    snapshot_date: str,
    max_return: float,
    corrupt_log: list[dict] | None = None,
) -> float:
    """Retorno bruto (sin costes) de las posiciones desde el snapshot anterior.

    Guarda anti-artefactos: un retorno mensual de una posicion mayor que `max_return` (p.ej. +200 %)
    es imposible para una accion normal y se trata como dato corrupto (split mal ajustado, ticker
    reciclado, precio erroneo). Se NEUTRALIZA (esa posicion aporta 0 ese mes) y se registra, para
    que un artefacto de datos no infle el rendimiento. Ver bitacora: el caso de julio 2010 (+953 %).
    """
    if not holdings or not previous_prices:
        return 0.0
    current = price_index.get(snapshot_date, {})
    total_return = 0.0
    for ticker, weight in holdings.items():
        old_price = previous_prices.get(ticker)
        new_price = current.get(ticker)
        if old_price is None or new_price is None or old_price <= 0:
            continue
        position_return = new_price / old_price - 1
        if abs(position_return) > max_return:
            if corrupt_log is not None:
                corrupt_log.append({
                    "snapshot_date": snapshot_date, "ticker": ticker,
                    "position_return": position_return, "old_price": old_price, "new_price": new_price,
                })
            continue   # neutralizado: dato corrupto, no cuenta
        total_return += weight * position_return
    return total_return


def _price_orders(
    raw_orders: list[dict],
    price_index: dict[str, dict[str, float]],
    snapshot_date: str,
    current_holdings: dict[str, float],
    settings: Settings,
) -> tuple[list[dict], float]:
    """Anade precio y costes a cada orden y calcula el drag de cash sobre el portfolio."""
    commission_rate = settings.commission_bps / 10_000
    slippage_rate = settings.slippage_bps / 10_000
    current_prices = price_index.get(snapshot_date, {})
    priced: list[dict] = []
    total_notional_cost = 0.0

    for order in raw_orders:
        ticker = order["ticker"]
        price = current_prices.get(ticker)
        weight_before = order.get("weight_before")
        weight_after = order.get("weight_after")
        if weight_before is None:
            weight_before = current_holdings.get(ticker, 0.0)
        if weight_after is None:
            weight_after = 0.0 if order["side"] == "sell" else current_holdings.get(ticker, 0.0)

        notional_change = abs((weight_after or 0.0) - (weight_before or 0.0))
        commission = notional_change * commission_rate
        slippage = notional_change * slippage_rate

        priced.append(
            {
                **order,
                "price": price,
                "weight_before": weight_before,
                "weight_after": weight_after,
                "commission": commission,
                "slippage": slippage,
            }
        )
        total_notional_cost += commission + slippage
    return priced, total_notional_cost


def _percentile_of(ticker: str, scores_at_date: pd.DataFrame) -> float:
    row = scores_at_date.loc[scores_at_date["ticker"] == ticker]
    if row.empty:
        return float("nan")
    return float(row["meta_rank"].iloc[0]) * 100


def _annual_metrics(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame(columns=[
            "year", "portfolio_return", "benchmark_return", "alpha",
            "beats_benchmark", "max_drawdown_year", "information_ratio_year",
        ])
    frame = equity.copy()
    frame["date"] = pd.to_datetime(frame["snapshot_date"])
    frame["year"] = frame["date"].dt.year
    rows: list[dict] = []
    for year, group in frame.groupby("year"):
        first_pv = group["portfolio_value"].iloc[0]
        last_pv = group["portfolio_value"].iloc[-1]
        first_bv = group["benchmark_value"].iloc[0]
        last_bv = group["benchmark_value"].iloc[-1]
        portfolio_return = last_pv / first_pv - 1 if first_pv > 0 else 0.0
        benchmark_return = last_bv / first_bv - 1 if first_bv > 0 else 0.0
        alpha = portfolio_return - benchmark_return

        equity_series = group["portfolio_value"].to_numpy()
        drawdown = 1 - equity_series / np.maximum.accumulate(equity_series)
        max_drawdown = float(drawdown.max()) if len(drawdown) else 0.0

        excess = group["excess_return"].to_numpy()
        te = float(np.std(excess, ddof=1)) if len(excess) > 1 else 0.0
        ir = float(np.mean(excess)) / te if te > 0 else 0.0

        rows.append(
            {
                "year": int(year),
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
                "alpha": alpha,
                "beats_benchmark": alpha > 0,
                "max_drawdown_year": max_drawdown,
                "information_ratio_year": ir,
            }
        )
    return pd.DataFrame(rows)


def _summary(
    equity: pd.DataFrame,
    annual: pd.DataFrame,
    settings: Settings,
    diagnostics: pd.DataFrame | None = None,
) -> dict:
    """Resumen del run.

    IMPORTANTE sobre el alfa: NO se compone a lo largo de los anios. Multiplicar el alfa de
    25 anios produce cifras absurdas (cientos de miles de %) que ademas premian
    desproporcionadamente un unico anio excepcional. Se reporta la **alfa anualizada
    geometrica** (media compuesta del alfa anual) y la **alfa mediana anual**, ambas
    interpretables como "X % al anio". El alfa NO decide que escenario gana: la seleccion de
    la Fase 6 usa senales de aprendizaje (rank-IC), no de rentabilidad. Ver docs/doc.md.
    """
    if equity.empty:
        return {}

    excess = equity["excess_return"].to_numpy()
    te = float(np.std(excess, ddof=1)) if len(excess) > 1 else 0.0
    ir_total = float(np.mean(excess)) / te if te > 0 else 0.0

    equity_series = equity["portfolio_value"].to_numpy()
    drawdown_series = 1 - equity_series / np.maximum.accumulate(equity_series)
    max_drawdown_total = float(drawdown_series.max()) if len(drawdown_series) else 0.0

    beat_rate = float((annual["alpha"] > 0).mean()) if not annual.empty else 0.0
    median_alpha = float(annual["alpha"].median()) if not annual.empty else 0.0
    worst_year_alpha = float(annual["alpha"].min()) if not annual.empty else 0.0
    mean_annual_alpha = float(annual["alpha"].mean()) if not annual.empty else 0.0

    cagr_portfolio, cagr_benchmark, geometric_excess = _economic_metrics(equity)

    mean_rank_ic, rank_ic_positive_fraction, rank_ic_std = _rank_ic_signals(diagnostics)

    return {
        # --- senales de APRENDIZAJE (lo que decide la seleccion) ---
        "mean_rank_ic": mean_rank_ic,
        "rank_ic_positive_fraction": rank_ic_positive_fraction,
        "rank_ic_std": rank_ic_std,
        # --- consistencia y riesgo ---
        "beat_rate": beat_rate,
        "max_drawdown": max_drawdown_total,
        # --- rendimiento economico (CAGR estandar, no acumulados inflados) ---
        "cagr_portfolio": cagr_portfolio,
        "cagr_benchmark": cagr_benchmark,
        "cagr_difference": cagr_portfolio - cagr_benchmark,
        "geometric_excess_return": geometric_excess,
        # --- alfa aritmetica media anual (estadistico aparte, bien nombrado) ---
        "mean_annual_alpha": mean_annual_alpha,
        "median_annual_alpha": median_alpha,
        "worst_year_alpha": worst_year_alpha,
        "information_ratio": ir_total,
        # --- parametros del run ---
        "commission_bps": settings.commission_bps,
        "slippage_bps": settings.slippage_bps,
        "target_min": settings.target_min,
        "target_max": settings.target_max,
        "entry_min_percentile": settings.entry_min_percentile,
        "min_hold_percentile": settings.min_hold_percentile,
        "rotation_edge_percentiles": settings.rotation_edge_percentiles,
        "max_weight_per_position": settings.max_weight_per_position,
    }


def _economic_metrics(equity: pd.DataFrame) -> tuple[float, float, float]:
    """(CAGR cartera, CAGR benchmark, exceso geometrico relativo anualizado).

    CAGR = (valor_final / valor_inicial) ^ (1/anios) - 1, con `anios` = tiempo real transcurrido
    entre el primer y el ultimo snapshot. Es la metrica estandar de rendimiento, interpretable y
    no inflada por el numero de periodos (a diferencia del acumulado bruto).

    Exceso geometrico relativo = geomean((1+r_cart)/(1+r_bench)) - 1, anualizado: cuanto bate la
    cartera al indice por anio, de forma multiplicativa (no la resta aritmetica de retornos).
    """
    dates = pd.to_datetime(equity["snapshot_date"])
    years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, 1e-9)

    pv = equity["portfolio_value"].to_numpy()
    bv = equity["benchmark_value"].to_numpy()
    cagr_portfolio = float((pv[-1] / pv[0]) ** (1 / years) - 1) if pv[0] > 0 else 0.0
    cagr_benchmark = float((bv[-1] / bv[0]) ** (1 / years) - 1) if bv[0] > 0 else 0.0

    pr = pd.to_numeric(equity["portfolio_return"], errors="coerce").fillna(0).to_numpy()
    br = pd.to_numeric(equity["benchmark_return"], errors="coerce").fillna(0).to_numpy()
    relative = (1 + pr) / (1 + br)
    relative = relative[np.isfinite(relative) & (relative > 0)]
    if len(relative) > 1:
        periods_per_year = len(relative) / years
        geometric_excess = float(np.prod(relative) ** (periods_per_year / len(relative)) - 1)
    else:
        geometric_excess = 0.0
    return cagr_portfolio, cagr_benchmark, geometric_excess


def _rank_ic_signals(diagnostics: pd.DataFrame | None) -> tuple[float, float, float]:
    """(rank-IC medio, fraccion de cohortes con rank-IC>0, std del rank-IC) del META_FINAL.

    Es la evidencia de aprendizaje del sistema REALMENTE operado: la cartera consume el
    meta_score, asi que la metrica principal es el rank-IC del `meta_final`, NO el promedio de
    los agentes individuales (que no es lo que se opera). Si no hay filas de meta_final (p. ej.
    diagnostics antiguos o tests puros de cartera), cae al conjunto completo por compatibilidad.
    """
    if diagnostics is None or diagnostics.empty or "rank_ic" not in diagnostics.columns:
        return 0.0, 0.0, 0.0
    subset = diagnostics
    if "agent" in diagnostics.columns and (diagnostics["agent"] == "meta_final").any():
        subset = diagnostics.loc[diagnostics["agent"] == "meta_final"]
    values = subset["rank_ic"].dropna()
    if values.empty:
        return 0.0, 0.0, 0.0
    mean_ic = float(values.mean())
    positive_fraction = float((values > 0).mean())
    std_ic = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return mean_ic, positive_fraction, std_ic


def _write_outputs(result: BacktestResult, run_dir: Path, settings: Settings) -> None:
    write_parquet(result.positions, run_dir / "positions.parquet")
    write_parquet(result.orders, run_dir / "orders.parquet")
    write_parquet(result.equity, run_dir / "equity.parquet")
    write_parquet(result.annual_metrics, run_dir / "annual_metrics.parquet")
    write_json(result.summary, run_dir / "backtest_summary.json")

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["backtest"] = result.summary
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
