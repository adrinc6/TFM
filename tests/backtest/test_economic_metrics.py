"""CAGR y exceso geométrico: cuadran con casos manuales pequeños."""

from __future__ import annotations

import pandas as pd

from module.backtest import _economic_metrics


def test_cagr_matches_manual_case() -> None:
    """Cartera que duplica en 2 años exactos -> CAGR = sqrt(2)-1 ~ 41.4%."""
    equity = pd.DataFrame({
        "snapshot_date": ["2000-01-15", "2001-01-15", "2002-01-15"],
        "portfolio_value": [100.0, 141.4213562, 200.0],
        "benchmark_value": [100.0, 110.0, 121.0],   # +10%/ano exacto
        "portfolio_return": [0.0, 0.414213562, 0.414213562],
        "benchmark_return": [0.0, 0.10, 0.10],
    })
    cagr_pf, cagr_bench, geo_excess = _economic_metrics(equity)
    # 2.0025 anios reales por el calendario; toleramos por 365.25
    assert abs(cagr_pf - 0.414) < 0.01, cagr_pf
    assert abs(cagr_bench - 0.10) < 0.01, cagr_bench
    # exceso geometrico: la cartera bate al bench ~28.5%/ano ((1.414)/(1.10)-1 compuesto)
    assert geo_excess > 0.25


def test_cagr_not_inflated_by_horizon() -> None:
    """El CAGR NO crece con el numero de periodos (a diferencia del acumulado bruto)."""
    # 10 anios al 10% compuesto: valor final 100*1.1^10 = 259.4, CAGR debe seguir siendo 10%.
    dates = pd.date_range("2000-01-15", periods=11, freq="YS")
    values = [100.0 * 1.10 ** i for i in range(11)]
    equity = pd.DataFrame({
        "snapshot_date": [d.date().isoformat() for d in dates],
        "portfolio_value": values,
        "benchmark_value": values,
        "portfolio_return": [0.0] + [0.10] * 10,
        "benchmark_return": [0.0] + [0.10] * 10,
    })
    cagr_pf, _, _ = _economic_metrics(equity)
    assert abs(cagr_pf - 0.10) < 0.005, f"CAGR deberia ser ~10%, es {cagr_pf}"


def test_price_guard_neutralizes_impossible_returns() -> None:
    """Guarda anti-artefactos: un salto de precio imposible no infla el equity."""
    from module.backtest import _mark_to_market
    log = []
    # +1000 % en un mes (dato corrupto) -> neutralizado, la posicion aporta 0.
    corrupt = _mark_to_market({"AAA": 1.0}, {"AAA": 100.0}, {"d": {"AAA": 1100.0}},
                              "d", max_return=2.0, corrupt_log=log)
    assert corrupt == 0.0
    assert len(log) == 1 and log[0]["ticker"] == "AAA"
    # Un retorno grande pero posible (+80 %) SI cuenta.
    ok = _mark_to_market({"AAA": 1.0}, {"AAA": 100.0}, {"d": {"AAA": 180.0}},
                         "d", max_return=2.0, corrupt_log=[])
    assert abs(ok - 0.80) < 1e-9
