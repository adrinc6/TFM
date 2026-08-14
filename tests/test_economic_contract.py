"""Contrato económico de la cartera: umbrales en puntos básicos, efectivo y contabilidad."""

from __future__ import annotations

import pandas as pd
import pytest

from environment import Settings
from module.evaluation.portfolio import (
    PortfolioState, _horizon_cost_to_annual_bps, decide_orders,
)
from module.research.robustness import score_permutation


def _scores(**columns: object) -> pd.DataFrame:
    frame = pd.DataFrame(columns)
    frame["snapshot_date"] = ["2020-03-31"] * len(frame)
    frame["is_quarterly"] = [True] * len(frame)
    return frame


def test_add_one_placebo_p_value_never_zero() -> None:
    tickers = [f"T{i}" for i in range(8)]
    scores = pd.DataFrame({
        "ticker": tickers, "snapshot_date": ["2020-03-31"] * 8,
        "meta_rank": [i / 7 for i in range(8)],
    })
    targets = pd.DataFrame({
        "ticker": tickers, "snapshot_date": ["2020-03-31"] * 8,
        "forward_excess_return": [i / 7 for i in range(8)], "target_available": [True] * 8,
    })
    result = score_permutation(scores, targets, iterations=5, minimum_cross_section=8)
    assert result["observed_mean_rank_ic"] == 1.0
    assert result["p_value"] == 1 / 6


def test_spy_is_never_a_stock_position_and_zero_cash_cap_holds_no_cash() -> None:
    scores = _scores(ticker=["AAPL", "MSFT"], meta_rank=[0.99, 0.90],
                     expected_excess_return=[0.05, 0.04])
    _, weights = decide_orders(
        PortfolioState.empty(), scores,
        Settings(target_size=2, max_cash_weight=0.0, sizing_mode="equal"),
    )
    assert "SPY" not in weights
    assert sum(weights.values()) == 1.0


def test_rotation_requires_clearing_the_round_trip_cost() -> None:
    """Una rotación solo se autoriza si la ventaja supera el coste de ida y vuelta más el margen."""
    state = PortfolioState(holdings={"A": 0.5, "B": 0.5})
    settings = Settings(
        target_size=2, commission_bps=5, slippage_bps=10, rotation_edge_bps=50,
        exit_expected_alpha_bps=0.0, sizing_mode="equal", target_horizon_months=12,
    )
    # Con horizonte de 12 meses la conversión anual->horizonte es la identidad.
    # Umbral = 2*(5+10) + 50 = 80 pb. Ventaja de C sobre B = 70 pb: insuficiente.
    insufficient = _scores(ticker=["A", "B", "C"], meta_rank=[0.91, 0.84, 0.92],
                           expected_excess_return=[0.0200, 0.0100, 0.0170])
    _, weights = decide_orders(state, insufficient, settings)
    assert set(weights) == {"A", "B"}

    # Ventaja de C sobre B = 90 pb: supera el umbral y la rotación se autoriza.
    sufficient = _scores(ticker=["A", "B", "C"], meta_rank=[0.91, 0.84, 0.92],
                         expected_excess_return=[0.0200, 0.0100, 0.0190])
    orders, weights = decide_orders(state, sufficient, settings)
    assert set(weights) == {"A", "C"}
    assert {order["reason"] for order in orders} >= {"displaced_by_net_edge", "net_edge_over_worst"}


def test_missing_score_exits_even_under_minimum_holding_and_never_rebuys() -> None:
    state = PortfolioState(
        holdings={"A": 0.5, "MISSING": 0.5},
        entry_dates={"A": "2020-03-31", "MISSING": "2020-03-31"},
    )
    scores = _scores(
        ticker=["A", "C"], meta_rank=[0.90, 0.99], expected_excess_return=[0.05, 0.06],
    )
    orders, weights = decide_orders(
        state, scores,
        Settings(target_size=2, sizing_mode="equal", minimum_holding_period="full_horizon"),
    )
    assert set(weights) == {"A", "C"}
    assert "MISSING" not in weights
    assert [(order["ticker"], order["reason"]) for order in orders if order["side"] == "sell"] == [
        ("MISSING", "missing_current_score"),
    ]


def test_missing_score_cannot_block_rotation_of_current_p4_position() -> None:
    state = PortfolioState(
        holdings={"LOW": 1 / 3, "MID": 1 / 3, "MISSING": 1 / 3},
        entry_dates={"LOW": "2010-01-31", "MID": "2010-01-31", "MISSING": "2020-03-31"},
    )
    scores = _scores(
        ticker=["LOW", "MID", "FILL", "CHALLENGER"],
        meta_rank=[0.04, 0.80, 0.99, 0.98],
        expected_excess_return=[-0.092, 0.06, 0.10, 0.095],
    )
    orders, weights = decide_orders(
        state, scores,
        Settings(
            target_size=3, sizing_mode="equal", commission_bps=5, slippage_bps=10,
            rotation_edge_bps=50, target_horizon_months=12,
        ),
    )
    assert set(weights) == {"MID", "FILL", "CHALLENGER"}
    assert {order["reason"] for order in orders} >= {
        "missing_current_score", "displaced_by_net_edge", "net_edge_over_worst",
    }


def test_missing_score_keeps_price_only_sell_only_precedence() -> None:
    state = PortfolioState(holdings={"A": 0.5, "MISSING": 0.5})
    scores = pd.DataFrame({
        "ticker": ["A", "C"], "snapshot_date": ["2020-04-30"] * 2,
        "is_quarterly": [False] * 2, "meta_rank": [0.90, 0.99],
        "expected_excess_return": [0.05, 0.06],
    })
    orders, weights = decide_orders(
        state, scores,
        Settings(target_size=2, sizing_mode="equal", price_only_sell_only=True),
    )
    assert set(weights) == {"A"}
    assert [(order["ticker"], order["reason"]) for order in orders] == [
        ("MISSING", "missing_current_score"),
    ]


def test_missing_score_respects_the_diversification_floor() -> None:
    state = PortfolioState(holdings={"A": 0.25, "B": 0.25, "C": 0.25, "MISSING": 0.25})
    scores = _scores(
        ticker=["A", "B", "C", "D"], meta_rank=[0.99, 0.90, 0.80, 0.70],
        expected_excess_return=[0.05, 0.04, 0.03, 0.0],
    )
    orders, weights = decide_orders(
        state, scores,
        Settings(
            target_size=4, sizing_mode="equal", max_cash_weight=0.25,
            exit_expected_alpha_bps=100.0,
        ),
    )
    assert set(weights) == {"A", "B", "C"}
    assert sum(weights.values()) == pytest.approx(0.75)
    assert [(order["ticker"], order["reason"]) for order in orders] == [
        ("MISSING", "missing_current_score"),
    ]


def test_position_below_expected_alpha_threshold_is_sold_to_cash() -> None:
    """La venta a efectivo solo existe bajo la política de oportunidad y respeta el suelo."""
    state = PortfolioState(holdings={"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    scores = _scores(ticker=["A", "B", "C", "D"], meta_rank=[0.95, 0.30, 0.90, 0.85],
                     expected_excess_return=[0.0300, 0.0050, 0.0250, 0.0200])
    orders, weights = decide_orders(
        state, scores,
        Settings(target_size=4, max_cash_weight=0.25,
                 exit_expected_alpha_bps=100.0, sizing_mode="equal", target_horizon_months=12),
    )
    assert {order["reason"] for order in orders} >= {"expected_alpha_below_exit"}
    assert set(weights) == {"A", "C", "D"}
    assert abs(1.0 - sum(weights.values()) - 0.25) < 1e-12


def test_zero_cash_cap_never_sells_just_to_rebuy_the_same_names() -> None:
    """Con todas las posiciones bajo el umbral y sin retador mejor, no se emite ninguna orden.

    La versión anterior vendía la cartera entera por umbral y la recompraba en el mismo snapshot
    (las vendidas seguían siendo las mejores por ranking): una ida y vuelta completa para quedar
    igual. Una venta sin destino mejor después de costes no es una decisión, es un peaje.
    """
    state = PortfolioState(holdings={"A": 0.5, "B": 0.5})
    scores = _scores(ticker=["A", "B"], meta_rank=[0.91, 0.70],
                     expected_excess_return=[0.0050, 0.0030])
    orders, weights = decide_orders(
        state, scores,
        Settings(target_size=2, max_cash_weight=0.0, exit_expected_alpha_bps=250.0,
                 sizing_mode="equal"),
    )
    assert orders == []
    assert weights == {"A": 0.5, "B": 0.5}


def test_uncalibrated_expected_alpha_never_triggers_a_sale() -> None:
    """Sin calibración no hay evidencia económica, y sin evidencia no se actúa."""
    state = PortfolioState(holdings={"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    scores = _scores(ticker=["A", "B", "C", "D"], meta_rank=[0.91, 0.70, 0.60, 0.50],
                     expected_excess_return=[float("nan")] * 4)
    orders, weights = decide_orders(
        state, scores,
        Settings(target_size=4, max_cash_weight=0.25,
                 exit_expected_alpha_bps=250.0, sizing_mode="equal"),
    )
    assert not [order for order in orders if order["reason"] == "expected_alpha_below_exit"]
    assert set(weights) == {"A", "B", "C", "D"}


def test_cash_cap_respects_the_cap_and_the_diversification_floor() -> None:
    """Sin candidatas sobre el umbral la plaza queda en efectivo, pero nunca bajo el suelo.

    Con tope del 25 % y 4 plazas, el suelo son 3 posiciones: el 75 % invertido se reparte entre las
    3 mejores por ranking aunque la tercera no supere el umbral. Sin ese suelo, una única admisible
    concentraría el 75 % de la cartera en una acción.
    """
    scores = _scores(ticker=["A", "B", "C", "D"], meta_rank=[0.99, 0.95, 0.90, 0.85],
                     expected_excess_return=[0.0400, 0.0300, 0.0005, 0.0002])
    settings = Settings(
        target_size=4, max_cash_weight=0.25,
        exit_expected_alpha_bps=100.0, sizing_mode="equal",
    )
    orders, weights = decide_orders(PortfolioState.empty(), scores, settings)
    cash = 1.0 - sum(weights.values())
    assert set(weights) == {"A", "B", "C"}
    assert {order["reason"] for order in orders} >= {"cash_floor_fill"}
    assert abs(cash - settings.max_cash_weight) < 1e-12


def test_entry_needs_the_exit_threshold_plus_its_own_round_trip() -> None:
    """Histéresis: comprar exige umbral de salida + coste de ida y vuelta; mantener, solo el umbral.

    Sin la banda, una acción oscilando alrededor del umbral se compra y se vende en snapshots
    consecutivos pagando costes con ventaja esperada nula.
    """
    settings = Settings(
        target_size=2, max_cash_weight=0.25,
        exit_expected_alpha_bps=100.0, commission_bps=5, slippage_bps=10, sizing_mode="equal",
        target_horizon_months=12,
    )
    # Con horizonte de 12 meses la conversión anual->horizonte es la identidad.
    # Banda: mantener exige 100 pb; entrar exige 100 + 2*(5+10) = 130 pb. B está en 115 pb.
    in_band = _scores(ticker=["A", "B"], meta_rank=[0.99, 0.95],
                      expected_excess_return=[0.0400, 0.0115])
    orders, weights = decide_orders(PortfolioState.empty(), in_band, settings)
    buys = {order["ticker"] for order in orders if order["reason"] == "initial_fill"}
    assert "B" not in buys

    state = PortfolioState(holdings={"A": 0.5, "B": 0.5})
    orders, weights = decide_orders(state, in_band, settings)
    assert not [order for order in orders if order["reason"] == "expected_alpha_below_exit"]
    assert set(weights) == {"A", "B"}


def test_zero_cash_cap_fills_the_slot_even_below_threshold() -> None:
    scores = _scores(ticker=["A", "B", "C"], meta_rank=[0.99, 0.95, 0.10],
                     expected_excess_return=[0.0400, 0.0300, -0.0500])
    _, weights = decide_orders(
        PortfolioState.empty(), scores,
        Settings(target_size=3, max_cash_weight=0.0, exit_expected_alpha_bps=100.0,
                 sizing_mode="equal"),
    )
    assert set(weights) == {"A", "B", "C"}
    assert sum(weights.values()) == 1.0


def test_alpha_proportional_sizing_caps_at_two_to_one() -> None:
    scores = _scores(ticker=["A", "B"], meta_rank=[1.0, 0.80],
                     expected_excess_return=[0.0600, 0.0200])
    _, weights = decide_orders(
        PortfolioState.empty(), scores,
        Settings(target_size=2, sizing_mode="alpha_proportional", exit_expected_alpha_bps=0.0),
    )
    assert weights["A"] == 2 / 3
    assert weights["B"] == 1 / 3


def test_operating_cost_annualizes_geometrically_not_linearly() -> None:
    """El coste de operar se anualiza compuesto, no multiplicando por las vueltas que caben al año.

    Todo se compara en base anual: el alfa esperado ya viene anualizado de la curva por deciles y los
    umbrales del catálogo son anuales. El que se convierte es el coste, porque se paga una vez por
    operación: con horizonte de 6 meses, 30 pb de ida y vuelta equivalen a (1,003)^2 - 1 ≈ 60,09 pb
    anuales, no a los 60 pb exactos de un prorrateo lineal.
    """
    horizon_bps = 2.0 * (5.0 + 10.0)
    annual = _horizon_cost_to_annual_bps(horizon_bps, 6)
    assert annual == pytest.approx(60.09, abs=0.01)
    assert annual > horizon_bps * 2.0  # compuesto, nunca por debajo del prorrateo lineal
    # Con horizonte de 12 meses la conversión es la identidad: el coste ya es anual.
    assert _horizon_cost_to_annual_bps(horizon_bps, 12) == pytest.approx(horizon_bps)


def test_exit_threshold_compares_annual_alpha_against_annual_catalog_threshold() -> None:
    """Un alfa anual por debajo del umbral anual del catálogo gatilla la venta, sin reescalados."""
    state = PortfolioState(holdings={"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    settings = Settings(
        target_size=4, max_cash_weight=0.25,
        exit_expected_alpha_bps=100.0, sizing_mode="equal", target_horizon_months=6,
    )
    # B espera 49,9 pb ANUALES, por debajo del umbral de 100 pb anuales: se vende. El horizonte de 6
    # meses ya no comprime el umbral, porque el alfa con el que se compara también es anual.
    scores = _scores(ticker=["A", "B", "C", "D"], meta_rank=[0.95, 0.90, 0.85, 0.80],
                     expected_excess_return=[0.0300, 0.00499, 0.0250, 0.0200])
    orders, _ = decide_orders(state, scores, settings)
    sold = [order for order in orders if order["reason"] == "expected_alpha_below_exit"]
    assert [order["ticker"] for order in sold] == ["B"]


def test_minimum_holding_period_quarter_horizon_protects_for_one_quarter() -> None:
    """Con horizonte de 12 meses, quarter_horizon exige 3 meses antes de vender o rotar."""
    settings = Settings(
        target_size=2, max_cash_weight=0.25,
        exit_expected_alpha_bps=0.0, rotation_edge_bps=0.0, commission_bps=0, slippage_bps=0,
        sizing_mode="equal", target_horizon_months=12, minimum_holding_period="quarter_horizon",
    )
    state = PortfolioState(
        holdings={"A": 0.5, "B": 0.5}, entry_dates={"A": "2020-01-31", "B": "2020-01-31"},
    )
    outsider = pd.DataFrame({
        "ticker": ["A", "B", "C"], "is_quarterly": [True] * 3,
        "meta_rank": [0.60, 0.30, 0.95], "expected_excess_return": [0.0050, 0.0001, 0.0400],
    })
    # A los 2 meses (2020-03-31): por debajo del mínimo de un trimestre, nada se mueve.
    early = outsider.assign(snapshot_date="2020-03-31")
    orders, weights = decide_orders(state, early, settings)
    assert not orders
    assert set(weights) == {"A", "B"}

    # A los 3 meses (2020-04-30): mínimo cumplido, la rotación ya puede desplazar a B.
    ready = outsider.assign(snapshot_date="2020-04-30")
    orders, weights = decide_orders(state, ready, settings)
    assert orders
    assert "C" in weights


def test_minimum_holding_period_half_and_full_horizon_extend_protection() -> None:
    """half_horizon exige 6 meses y full_horizon 12, con horizonte de 12 meses."""
    base = {
        "target_size": 2, "max_cash_weight": 0.25,
        "exit_expected_alpha_bps": 0.0, "rotation_edge_bps": 0.0, "commission_bps": 0,
        "slippage_bps": 0, "sizing_mode": "equal", "target_horizon_months": 12,
    }
    state = PortfolioState(
        holdings={"A": 0.5, "B": 0.5}, entry_dates={"A": "2020-01-31", "B": "2020-01-31"},
    )
    scores_at_6_months = pd.DataFrame({
        "ticker": ["A", "B", "C"], "snapshot_date": ["2020-07-31"] * 3, "is_quarterly": [True] * 3,
        "meta_rank": [0.60, 0.30, 0.95], "expected_excess_return": [0.0050, 0.0001, 0.0400],
    })
    _, weights_half = decide_orders(
        state, scores_at_6_months, Settings(**{**base, "minimum_holding_period": "half_horizon"}),
    )
    assert "C" in weights_half

    orders_full, weights_full = decide_orders(
        state, scores_at_6_months, Settings(**{**base, "minimum_holding_period": "full_horizon"}),
    )
    assert not orders_full
    assert set(weights_full) == {"A", "B"}


def test_minimum_holding_period_none_allows_immediate_sale() -> None:
    """Sin mínimo, una posición recién comprada puede venderse o rotarse en el mismo snapshot."""
    settings = Settings(
        target_size=2, max_cash_weight=0.25,
        exit_expected_alpha_bps=0.0, rotation_edge_bps=0.0, commission_bps=0, slippage_bps=0,
        sizing_mode="equal", target_horizon_months=12, minimum_holding_period="none",
    )
    state = PortfolioState(
        holdings={"A": 0.5, "B": 0.5}, entry_dates={"A": "2020-03-31", "B": "2020-03-31"},
    )
    scores = pd.DataFrame({
        "ticker": ["A", "B", "C"], "snapshot_date": ["2020-03-31"] * 3, "is_quarterly": [True] * 3,
        "meta_rank": [0.60, 0.30, 0.95], "expected_excess_return": [0.0050, 0.0001, 0.0400],
    })
    _, weights = decide_orders(state, scores, settings)
    assert "C" in weights


def test_price_only_sell_only_allows_sale_but_blocks_replacement() -> None:
    """En un snapshot de solo precio se puede vender una posición mala pero no comprar reemplazo."""
    state = PortfolioState(holdings={"A": 0.5, "B": 0.5})
    settings = Settings(
        target_size=2, max_cash_weight=0.0, exit_expected_alpha_bps=100.0,
        sizing_mode="equal", target_horizon_months=12, price_only_sell_only=True,
    )
    scores = pd.DataFrame({
        "ticker": ["A", "B", "C"], "snapshot_date": ["2020-04-30"] * 3, "is_quarterly": [False] * 3,
        "meta_rank": [0.95, 0.30, 0.99], "expected_excess_return": [0.0300, 0.0005, 0.0500],
    })
    orders, weights = decide_orders(state, scores, settings)
    assert {order["reason"] for order in orders} == {"expected_alpha_below_exit"}
    assert set(weights) == {"A"}
    assert 1.0 - sum(weights.values()) > 0


def test_price_only_sell_only_has_no_effect_on_quarterly_snapshots() -> None:
    """Con fundamentales nuevos, la variable no cambia nada: compra y rotación normales."""
    state = PortfolioState(holdings={"A": 0.5, "B": 0.5})
    settings = Settings(
        target_size=2, max_cash_weight=0.0, exit_expected_alpha_bps=100.0,
        sizing_mode="equal", target_horizon_months=12, price_only_sell_only=True,
    )
    scores = pd.DataFrame({
        "ticker": ["A", "B", "C"], "snapshot_date": ["2020-04-30"] * 3, "is_quarterly": [True] * 3,
        "meta_rank": [0.95, 0.30, 0.99], "expected_excess_return": [0.0300, 0.0005, 0.0500],
    })
    _, weights = decide_orders(state, scores, settings)
    assert set(weights) == {"A", "C"}
    assert sum(weights.values()) == 1.0


def test_scarce_universe_invests_fully_when_the_cash_cap_is_zero() -> None:
    """Con tope 0 y menos candidatas que plazas, el capital se reparte entre las que hay.

    Antes existía una rama por política que repartía sobre `target_size` en vez de sobre las plazas
    ocupadas: con 2 candidatas y 4 plazas retenía un 50 % en efectivo bajo la política llamada
    «siempre invertida», más que el tope 0 de la política de efectivo. Ese efectivo no lo declaraba
    ninguna variable, así que no respetaba tope alguno.
    """
    scores = _scores(ticker=["A", "B"], meta_rank=[0.99, 0.95],
                     expected_excess_return=[0.0400, 0.0300])
    _, weights = decide_orders(
        PortfolioState.empty(), scores,
        Settings(target_size=4, max_cash_weight=0.0, sizing_mode="equal"),
    )
    assert set(weights) == {"A", "B"}
    assert sum(weights.values()) == pytest.approx(1.0)


def test_blocked_purchases_leave_cash_instead_of_concentrating() -> None:
    """Un hueco por compra prohibida no se reparte: concentrar sería actuar sin información nueva.

    Es la diferencia con el universo escaso. Aquí sí hay candidatas y se ha decidido no comprarlas,
    así que subir del 50 % al 100 % una superviviente contradiría el propósito de la variable.
    """
    state = PortfolioState(holdings={"A": 0.5, "B": 0.5})
    scores = pd.DataFrame({
        "ticker": ["A", "B", "C"], "snapshot_date": ["2020-04-30"] * 3, "is_quarterly": [False] * 3,
        "meta_rank": [0.95, 0.30, 0.99], "expected_excess_return": [0.0300, 0.0005, 0.0500],
    })
    _, weights = decide_orders(state, scores, Settings(
        target_size=2, max_cash_weight=0.0, exit_expected_alpha_bps=100.0,
        sizing_mode="equal", target_horizon_months=12, price_only_sell_only=True,
    ))
    assert set(weights) == {"A"}
    assert weights["A"] == pytest.approx(0.5)


def test_coverage_floor_sells_a_position_that_fell_below_the_percentile() -> None:
    """Una posición bajo el suelo se vende aunque su alfa no active ninguna otra regla."""
    state = PortfolioState(
        holdings={"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
        entry_dates={ticker: "2015-01-31" for ticker in "ABCD"},
    )
    # D cae al percentil 50: por encima del umbral de alfa, pero bajo el suelo de cobertura.
    scores = _scores(ticker=["A", "B", "C", "D"], meta_rank=[0.99, 0.95, 0.90, 0.50],
                     expected_excess_return=[0.0400, 0.0350, 0.0300, 0.0250])
    orders, weights = decide_orders(state, scores, Settings(
        target_size=4, max_cash_weight=0.25, sizing_mode="equal",
        exit_expected_alpha_bps=0.0, coverage_percentile_floor=60.0,
    ))
    assert ("D", "below_coverage_percentile") in [
        (order["ticker"], order["reason"]) for order in orders
    ]
    assert "D" not in weights


def test_coverage_floor_respects_the_minimum_holding_period() -> None:
    """A diferencia de `missing_current_score`, el suelo de cobertura sí espera al mínimo.

    La posición sigue siendo scoreable: no ha perdido cobertura, solo ha caído en el ranking. El
    mínimo de tenencia existe precisamente para no deshacer una elección antes de darle tiempo.
    """
    state = PortfolioState(
        holdings={"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
        entry_dates={"A": "2015-01-31", "B": "2015-01-31", "C": "2015-01-31",
                     "D": "2020-03-31"},
    )
    scores = _scores(ticker=["A", "B", "C", "D"], meta_rank=[0.99, 0.95, 0.90, 0.50],
                     expected_excess_return=[0.0400, 0.0350, 0.0300, 0.0250])
    orders, weights = decide_orders(state, scores, Settings(
        target_size=4, max_cash_weight=0.25, sizing_mode="equal",
        exit_expected_alpha_bps=0.0, coverage_percentile_floor=60.0,
        minimum_holding_period="full_horizon", target_horizon_months=12,
    ))
    assert "below_coverage_percentile" not in {order["reason"] for order in orders}
    assert "D" in weights


def test_coverage_floor_disabled_by_default_changes_nothing() -> None:
    """Con suelo 0 el catálogo se comporta exactamente como antes de existir la variable."""
    state = PortfolioState(
        holdings={"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
        entry_dates={ticker: "2015-01-31" for ticker in "ABCD"},
    )
    scores = _scores(ticker=["A", "B", "C", "D"], meta_rank=[0.99, 0.95, 0.90, 0.04],
                     expected_excess_return=[0.0400, 0.0350, 0.0300, 0.0250])
    base = dict(target_size=4, max_cash_weight=0.25, sizing_mode="equal",
                exit_expected_alpha_bps=0.0)
    orders, weights = decide_orders(state, scores, Settings(**base))
    assert "below_coverage_percentile" not in {order["reason"] for order in orders}
    assert set(weights) == {"A", "B", "C", "D"}


def test_coverage_floor_sells_every_position_below_it_unconditionally() -> None:
    """La venta por suelo de cobertura es incondicional: no la frena el suelo de diversificación.

    Qué pasa con las plazas liberadas (recompra o efectivo) lo decide después el relleno
    obligatorio, no esta regla.
    """
    state = PortfolioState(
        holdings={"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
        entry_dates={ticker: "2015-01-31" for ticker in "ABCD"},
    )
    # Las cuatro posiciones caen bajo el suelo; con tope 25 % y 4 plazas el suelo sería 3, pero
    # eso ya no frena la venta.
    scores = _scores(ticker=["A", "B", "C", "D"], meta_rank=[0.50, 0.40, 0.30, 0.20],
                     expected_excess_return=[0.0400, 0.0350, 0.0300, 0.0250])
    orders, weights = decide_orders(state, scores, Settings(
        target_size=4, max_cash_weight=0.25, sizing_mode="equal",
        exit_expected_alpha_bps=0.0, coverage_percentile_floor=60.0,
    ))
    sold = {order["ticker"] for order in orders if order["reason"] == "below_coverage_percentile"}
    assert sold == {"A", "B", "C", "D"}


def test_coverage_floor_with_zero_cash_cap_refills_in_the_same_snapshot() -> None:
    """Con tope 0 la plaza liberada se recompra ya: la regla actúa como rotación sin test de coste.

    Es una consecuencia declarada, no un descuido: el relleno obligatorio no aplica umbrales, así
    que la venta por cobertura fuerza una rotación que el bucle económico habría rechazado y
    aumenta la rotación de la cartera. Con tope de efectivo la plaza puede quedarse vacía.
    """
    state = PortfolioState(
        holdings={"A": 0.5, "B": 0.5}, entry_dates={"A": "2015-01-31", "B": "2015-01-31"},
    )
    scores = _scores(ticker=["A", "B", "C"], meta_rank=[0.99, 0.50, 0.70],
                     expected_excess_return=[0.0400, 0.0300, 0.0305])
    orders, weights = decide_orders(state, scores, Settings(
        target_size=2, max_cash_weight=0.0, sizing_mode="equal",
        exit_expected_alpha_bps=0.0, coverage_percentile_floor=60.0,
    ))
    reasons = {(order["ticker"], order["reason"]) for order in orders}
    assert ("B", "below_coverage_percentile") in reasons
    assert set(weights) == {"A", "C"}
    assert sum(weights.values()) == pytest.approx(1.0)
