"""Correcciones del plan revisado: meta_final diagnosticado, motores/objetivos, huellas."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from module.modeling.agents import _prepare_training, build_agent_scores
from module.modeling.meta import AGENT_NAMES, _weights_as_of
from module.runs.experiments import stage_fingerprint


def _train_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for date in ("2000-01-15", "2000-02-15"):
        for i in range(20):
            rows.append({"snapshot_date": date, "forward_excess_return_3m": rng.normal()})
    return pd.DataFrame(rows)


def test_rank_regression_label_is_cross_sectional_percentile() -> None:
    """rank_regression: la etiqueta es el percentil del retorno dentro de cada snapshot."""
    from environment import Settings
    settings = Settings(run_scope="dev", objective="rank_regression")
    train, label = _prepare_training(_train_frame(), settings)
    assert len(train) == 40                       # no descarta filas
    assert label.between(0, 1).all()              # percentiles en [0,1]
    assert abs(label.groupby(train["snapshot_date"]).mean().mean() - 0.525) < 0.1


def test_quartile_excludes_middle_from_training() -> None:
    """quartile (ablacion): entrena solo con extremos; el centro se excluye del train."""
    from environment import Settings
    settings = Settings(run_scope="dev", objective="quartile")
    train, label = _prepare_training(_train_frame(), settings)
    assert len(train) < 40                         # el centro se cae
    assert set(label.unique()) <= {0, 1}           # binaria
    # aproximadamente la mitad de cada snapshot (top 25 % + bottom 25 %)
    assert 0.4 < len(train) / 40 < 0.6


def test_catboost_ranker_importances_do_not_crash() -> None:
    """CatBoostRanker (YetiRank) devuelve feature_importances_ = None (0-d array); _importance_rows
    debe omitirlas sin lanzar 'iteration over a 0-d array' en vez de abortar el escenario ranking."""
    from environment import Settings
    from module.modeling.agents import (_build_family_model, _fit_model, _importance_rows,
                                         _prepare_training)

    settings = Settings(run_scope="dev", objective="ranking")
    frame = _train_frame().assign(ticker=[f"T{i}" for i in range(len(_train_frame()))],
                                  f0=np.random.default_rng(1).normal(size=40),
                                  f1=np.random.default_rng(2).normal(size=40))
    model = _build_family_model(settings, "catboost")
    if model is None:
        pytest.skip("catboost no instalado")
    columns = ["f0", "f1"]
    train, target = _prepare_training(frame, settings)
    _fit_model(model, train, columns, target, settings, "catboost")
    rows = _importance_rows(model, "momentum", pd.Timestamp("2000-01-15"), len(train), settings,
                            "catboost")
    assert rows == []  # importancias no disponibles: se omiten, no se aborta


def test_meta_final_is_diagnosed(agent_settings) -> None:
    """El diagnostico incluye el meta_final (lo que opera la cartera), no solo los agentes."""
    build_agent_scores(agent_settings)
    run_dir = next((agent_settings.processed_output_dir / "agents").iterdir())
    diag = pd.read_parquet(run_dir / "rank_ic_diagnostics.parquet")
    agents_present = set(diag["agent"].unique())
    assert "meta_final" in agents_present, f"falta meta_final; hay {agents_present}"
    assert "meta_equal_weight" in agents_present


def test_seed_and_objective_change_fingerprint() -> None:
    """Cambiar semilla, objetivo o hiperparametros cambia la huella del run (reproducibilidad)."""
    from environment import Settings
    base = Settings(run_scope="dev")
    for field, value in (("random_seed", 7), ("objective", "quartile"), ("lgbm_max_depth", 6)):
        changed = replace(base, **{field: value})
        assert stage_fingerprint("agents", base) != stage_fingerprint("agents", changed), (
            f"{field} deberia cambiar la huella de agents"
        )


def test_meta_type_changes_fingerprint() -> None:
    """Cada meta_type debe dar una huella de agents distinta: antes colisionaban (flag muerto)."""
    from environment import Settings
    base = Settings(run_scope="dev", meta_type="rank_ic")
    for value in ("equal", "regime"):
        changed = replace(base, meta_type=value)
        assert stage_fingerprint("agents", base) != stage_fingerprint("agents", changed), (
            f"meta_type={value} deberia cambiar la huella de agents"
        )


def _labelled_with_agent_signal() -> pd.DataFrame:
    """Historia sintética donde 'momentum' predice bien y 'quality'/'value' predicen al azar.

    Cada cohorte pasada (quarterly, ya etiquetada) tiene score de momentum correlacionado con el
    retorno futuro y los otros dos sin relación, para que el rank-IC reciente ordene los pesos.
    """
    rng = np.random.default_rng(0)
    rows = []
    dates = pd.date_range("2010-03-15", periods=8, freq="3MS")
    for date in dates:
        for i in range(30):
            future = rng.normal()
            scores = {"momentum": future + rng.normal(scale=0.3),  # buena señal
                      "quality": rng.normal(), "value": rng.normal()}  # ruido
            for agent, score in scores.items():
                rows.append({
                    "ticker": f"T{i}", "snapshot_date": date.date().isoformat(),
                    "snapshot_ts": date, "label_end_ts": date + pd.DateOffset(months=3),
                    "is_quarterly": True, "target_available": True,
                    "agent": agent, "score": score, "forward_excess_return_3m": future,
                })
    return pd.DataFrame(rows)


def test_meta_type_equal_gives_fixed_thirds() -> None:
    """equal: peso 1/3 fijo por agente, ignora el rank-IC reciente."""
    from environment import Settings
    labelled = _labelled_with_agent_signal()
    date = pd.Timestamp("2011-06-15")  # posterior a toda la historia sintética
    weights, _ = _weights_as_of(labelled, date, set(AGENT_NAMES),
                                Settings(run_scope="dev", meta_type="equal"))
    for agent in AGENT_NAMES:
        assert weights[agent] == pytest.approx(1 / 3)


def test_meta_type_rank_ic_favours_the_predictive_agent() -> None:
    """rank_ic: el agente con mejor rank-IC reciente (momentum) recibe más peso que el ruido."""
    from environment import Settings
    labelled = _labelled_with_agent_signal()
    date = pd.Timestamp("2011-06-15")
    weights, _ = _weights_as_of(labelled, date, set(AGENT_NAMES),
                                Settings(run_scope="dev", meta_type="rank_ic"))
    assert weights["momentum"] > weights["quality"]
    assert weights["momentum"] > weights["value"]


def test_meta_type_regime_tilts_over_rank_ic_without_lookahead() -> None:
    """regime: en bull sube momentum y baja quality respecto a rank_ic; en bear, al revés.

    El régimen entra como argumento `bull` (derivado del pasado del benchmark), así que el test
    fija el régimen explícitamente: no hay forma de que la señal mire al futuro.
    """
    from environment import Settings
    # Señal donde los tres agentes predicen algo (momentum el que más, pero sin saturar el tope del
    # clamp), para que el tilt de régimen sea observable y no quede enmascarado por el cap del 50 %.
    rng = np.random.default_rng(1)
    rows = []
    for date_ in pd.date_range("2010-03-15", periods=8, freq="3MS"):
        for i in range(30):
            future = rng.normal()
            scores = {"momentum": future * 0.5 + rng.normal(scale=0.8),
                      "quality": future * 0.4 + rng.normal(scale=0.9),
                      "value": future * 0.35 + rng.normal(scale=0.95)}
            for agent, score in scores.items():
                rows.append({"ticker": f"T{i}", "snapshot_date": date_.date().isoformat(),
                             "snapshot_ts": date_, "label_end_ts": date_ + pd.DateOffset(months=3),
                             "is_quarterly": True, "target_available": True,
                             "agent": agent, "score": score, "forward_excess_return_3m": future})
    labelled = pd.DataFrame(rows)
    date = pd.Timestamp("2011-06-15")
    settings = Settings(run_scope="dev", meta_type="regime")
    base, _ = _weights_as_of(labelled, date, set(AGENT_NAMES),
                             Settings(run_scope="dev", meta_type="rank_ic"))
    bull, _ = _weights_as_of(labelled, date, set(AGENT_NAMES), settings, bull=True)
    bear, _ = _weights_as_of(labelled, date, set(AGENT_NAMES), settings, bull=False)

    assert bull["momentum"] > base["momentum"]     # bull realza momentum
    assert bear["quality"] > base["quality"]       # bear realza quality
    assert sum(bull.values()) == pytest.approx(1.0)  # renormalizado
    assert sum(bear.values()) == pytest.approx(1.0)


def test_meta_type_regime_without_signal_falls_back_to_rank_ic() -> None:
    """Sin régimen disponible (bull=None), regime se comporta igual que rank_ic."""
    from environment import Settings
    labelled = _labelled_with_agent_signal()
    date = pd.Timestamp("2011-06-15")
    regime, _ = _weights_as_of(labelled, date, set(AGENT_NAMES),
                               Settings(run_scope="dev", meta_type="regime"), bull=None)
    rank_ic, _ = _weights_as_of(labelled, date, set(AGENT_NAMES),
                                Settings(run_scope="dev", meta_type="rank_ic"))
    for agent in AGENT_NAMES:
        assert regime[agent] == pytest.approx(rank_ic[agent])


def test_rank_ic_clamps_every_agent_between_min_and_max() -> None:
    """rank_ic: aunque un agente domine el rank-IC, ningún peso sale de [0.10, 0.50] y suma 1.

    La señal fuerte (momentum) satura al tope; los de ruido no bajan del mínimo universal, así que el
    meta nunca ignora del todo a nadie ni deja que uno acapare.
    """
    from environment import Settings
    from module.modeling.meta import AGENT_WEIGHT_MAX, AGENT_WEIGHT_MIN
    labelled = _labelled_with_agent_signal()   # momentum predice; quality/value ruido
    date = pd.Timestamp("2011-06-15")
    weights, _ = _weights_as_of(labelled, date, set(AGENT_NAMES),
                                Settings(run_scope="dev", meta_type="rank_ic"))
    assert sum(weights.values()) == pytest.approx(1.0)
    for agent, weight in weights.items():
        assert AGENT_WEIGHT_MIN - 1e-9 <= weight <= AGENT_WEIGHT_MAX + 1e-9
    assert weights["momentum"] == pytest.approx(AGENT_WEIGHT_MAX)  # el dominante topa en 50 %


def test_equal_mode_is_not_clamped() -> None:
    """El clamp solo afecta a rank_ic/regime: equal sigue dando 1/N exacto (1/3 con 3 agentes)."""
    from environment import Settings
    labelled = _labelled_with_agent_signal()
    date = pd.Timestamp("2011-06-15")
    weights, _ = _weights_as_of(labelled, date, set(AGENT_NAMES),
                                Settings(run_scope="dev", meta_type="equal"))
    for agent in AGENT_NAMES:
        assert weights[agent] == pytest.approx(1 / 3)
