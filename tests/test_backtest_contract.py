"""Contabilidad del backtest: pesos, efectivo, costes y segmentación temporal.

El backtest es contabilidad. Si no cuadra, todo lo económico que se reporte encima es ruido, y era
el módulo con más superficie y ninguna prueba.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from environment import Settings
from module.evaluation.backtest import DELISTING_RETURN, run_backtest
from module.studies.catalog import KNOWN_STRESS_YEARS, SELECTION_UNTIL_YEAR


def _panel(months: int = 36, tickers: tuple[str, ...] = ("A", "B", "C", "D")) -> tuple[pd.DataFrame, ...]:
    dates = pd.date_range("2015-01-30", periods=months, freq=pd.DateOffset(months=1))
    rng = np.random.default_rng(0)
    scores, prices, benchmark = [], [], []
    level = {ticker: 100.0 for ticker in tickers}
    benchmark_level = 100.0
    for index, date in enumerate(dates):
        stamp = str(date.date())
        benchmark_level *= 1 + rng.normal(0.005, 0.02)
        benchmark.append({"snapshot_date": stamp, "price": benchmark_level})
        for position, ticker in enumerate(tickers):
            level[ticker] *= 1 + rng.normal(0.006, 0.03)
            prices.append({"snapshot_date": stamp, "ticker": ticker, "price": level[ticker]})
            scores.append({
                "snapshot_date": stamp, "ticker": ticker,
                "meta_rank": 1.0 - position / len(tickers),
                "meta_score": 1.0 - position / len(tickers),
                "expected_excess_return": 0.05 - 0.02 * position,
                "is_quarterly": index % 3 == 0,
            })
    return pd.DataFrame(scores), pd.DataFrame(prices), pd.DataFrame(benchmark)


def _settings(**overrides) -> Settings:
    base = {"target_size": 2, "sizing_mode": "equal", "exit_expected_alpha_bps": 0.0,
            "snapshot_step_months": 1}
    return Settings(**{**base, **overrides})


def test_weights_plus_cash_always_sum_to_one() -> None:
    scores, prices, benchmark = _panel()
    result = run_backtest(scores, prices, benchmark, _settings(
        cash_policy="opportunity_cash", max_cash_weight=0.25, exit_expected_alpha_bps=250.0,
        target_size=4,
    ))
    total = result.equity["invested_weight"] + result.equity["cash_weight"]
    assert np.allclose(total, 1.0)
    assert (result.equity["cash_weight"] >= -1e-12).all()


def test_cash_cap_binds_the_target_while_drift_is_left_to_the_tolerance() -> None:
    """El tope gobierna la decisión, no la deriva posterior de precios.

    Entre rebalanceos los pesos se mueven con el mercado y el peso del efectivo flota: eso es
    precisamente lo que hace la tolerancia de rebalanceo, evitar operar por desviaciones pequeñas.
    Con tolerancia nula cada snapshot vuelve al objetivo y el tope se cumple exactamente.
    """
    scores, prices, benchmark = _panel()
    result = run_backtest(scores, prices, benchmark, _settings(
        cash_policy="opportunity_cash", max_cash_weight=0.25, exit_expected_alpha_bps=250.0,
        target_size=4, rebalance_drift_tolerance=0.0,
    ))
    assert (result.equity["cash_weight"] <= 0.25 + 1e-9).all()


def test_fully_invested_never_holds_cash() -> None:
    scores, prices, benchmark = _panel()
    result = run_backtest(scores, prices, benchmark, _settings(cash_policy="fully_invested"))
    assert np.allclose(result.equity["cash_weight"], 0.0)
    assert np.allclose(result.equity["invested_weight"], 1.0)


def test_cost_drag_equals_traded_notional_times_the_rate() -> None:
    scores, prices, benchmark = _panel()
    settings = _settings(commission_bps=5, slippage_bps=10)
    result = run_backtest(scores, prices, benchmark, settings)
    rate = (settings.commission_bps + settings.slippage_bps) / 10_000
    orders = result.orders
    expected = orders.groupby("snapshot_date")["notional"].sum() * rate
    paid = orders.groupby("snapshot_date").apply(
        lambda group: (group["commission_amount"] + group["slippage_amount"]).sum(),
        include_groups=False,
    )
    assert np.allclose(expected.to_numpy(), paid.to_numpy())


def test_zero_cost_configuration_pays_nothing() -> None:
    scores, prices, benchmark = _panel()
    result = run_backtest(scores, prices, benchmark, _settings(commission_bps=0, slippage_bps=0))
    assert np.allclose(result.equity["cost_drag"], 0.0)


def test_information_ratio_is_annualised_once_and_only_once() -> None:
    """Una sola definición: media/desviación de excesos por periodo escalada por √periodos."""
    scores, prices, benchmark = _panel()
    result = run_backtest(scores, prices, benchmark, _settings(snapshot_step_months=1))
    excess = result.equity.loc[
        pd.to_datetime(result.equity.snapshot_date).dt.year.le(SELECTION_UNTIL_YEAR), "excess_return"
    ].to_numpy()
    expected = excess.mean() / excess.std(ddof=1) * np.sqrt(12)
    assert result.summary["information_ratio"] == pytest_approx(expected)


def pytest_approx(value: float, tolerance: float = 1e-9):
    class _Approx:
        def __eq__(self, other: float) -> bool:
            return abs(other - value) <= tolerance
        def __repr__(self) -> str:
            return f"~{value}"
    return _Approx()


def test_geometric_excess_is_a_ratio_not_a_subtraction() -> None:
    scores, prices, benchmark = _panel()
    summary = run_backtest(scores, prices, benchmark, _settings()).summary
    portfolio, index = summary["cagr_portfolio"], summary["cagr_benchmark"]
    assert summary["geometric_excess_return"] == pytest_approx((1 + portfolio) / (1 + index) - 1)


def test_metrics_are_segmented_and_selection_window_excludes_the_reserved_era() -> None:
    scores, prices, benchmark = _panel(months=140)
    result = run_backtest(scores, prices, benchmark, _settings())
    years = pd.to_datetime(result.equity.snapshot_date).dt.year
    assert years.max() >= KNOWN_STRESS_YEARS[0]
    assert result.summary["n_periods"] == int(years.le(SELECTION_UNTIL_YEAR).sum())
    assert result.summary["confirmation"]["n_periods"] == int(years.isin(KNOWN_STRESS_YEARS).sum())
    assert result.summary["full_curve"]["n_periods"] == len(result.equity)
    # La cifra de portada no puede coincidir con la curva completa cuando la era reservada existe.
    assert result.summary["cagr_portfolio"] != result.summary["full_curve"]["cagr_portfolio"]


def test_price_only_sell_only_creates_transitional_cash_under_fully_invested() -> None:
    """El efectivo transitorio de `price_only_sell_only` cuadra y se cierra al volver a haber
    fundamentales nuevos, aunque `cash_policy` sea `fully_invested`."""
    scores, prices, benchmark = _panel(months=12, tickers=("A", "B", "C", "D"))
    dates = sorted(scores["snapshot_date"].unique())
    quarterly_dates = [dates[index] for index in range(0, len(dates), 3)]
    result = run_backtest(scores, prices, benchmark, _settings(
        cash_policy="fully_invested", price_only_sell_only=True,
        exit_expected_alpha_bps=1_000_000.0, target_size=2, target_horizon_months=12,
    ))
    total = result.equity["invested_weight"] + result.equity["cash_weight"]
    assert np.allclose(total, 1.0)
    equity = result.equity.set_index("snapshot_date")
    # A partir del segundo snapshot trimestral, el relleno obligatorio de `fully_invested` vuelve a
    # completar la cartera sin efectivo residual (el primero no tiene nada que vender todavía).
    later_quarterly = [date for date in quarterly_dates if date in equity.index][1:]
    assert (equity.loc[later_quarterly, "cash_weight"] < 1e-9).all()


def test_a_delisted_position_is_not_rescued_at_par() -> None:
    """Marcar plana una posición cuyo precio desaparece regala un rescate del 100 %."""
    scores, prices, benchmark = _panel(months=6, tickers=("A", "B"))
    prices = prices.loc[~(prices["ticker"].eq("B") & prices["snapshot_date"].ge("2015-04"))]
    scores = scores.loc[~(scores["ticker"].eq("B") & scores["snapshot_date"].ge("2015-04"))]
    result = run_backtest(scores, prices, benchmark, _settings(target_size=2))
    assert DELISTING_RETURN < 0
    assert result.summary["delisted_positions"] >= 1
