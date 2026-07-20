"""Contrato del ciclo unificado study/full_study: clasificación de variables, Fase 2 greedy
(sin producto cartesiano) y fase de cartera (criterio económico, sin reentrenar).
"""

from __future__ import annotations

import pandas as pd

from module.runs import execution
from module.runs.experiments import (
    MODEL_FIELDS, PORTFOLIO_FIELDS, PORTFOLIO_STRESS_FIELDS, split_variables, stress_variables,
)


# --- Clasificación de variables -------------------------------------------------------------

def test_split_variables_separates_model_and_portfolio() -> None:
    variables = {
        "train_lookback_years": [5, 8, 12],   # modelo
        "execution_lag_days": [15, 45],        # modelo (afecta observabilidad -> dataset)
        "commission_bps": [0, 10],             # cartera
        "target_size": [5, 10],                # cartera
        "profile": ["balanced", "quality"],    # cartera
        "desconocida": [1, 2],                 # se descarta
    }
    model_vars, portfolio_vars = split_variables(variables)
    assert set(model_vars) == {"train_lookback_years", "execution_lag_days"}
    assert set(portfolio_vars) == {"commission_bps", "target_size", "profile"}
    assert "desconocida" not in model_vars and "desconocida" not in portfolio_vars


def test_mechanical_portfolio_rules_are_stressed_not_optimized() -> None:
    """drift/expulsión/rotación son reglas de fricción: se estresan, no se optimizan por IR."""
    variables = {
        "target_size": [5, 10],                    # cartera optimizable
        "rebalance_drift_tolerance": [1.0, 2.5],   # cartera mecánica -> estrés
        "min_hold_percentile": [60, 80],           # cartera mecánica -> estrés
        "rotation_edge_percentiles": [5, 15],      # cartera mecánica -> estrés
    }
    _, portfolio_vars = split_variables(variables)
    stress = stress_variables(variables)
    # Los ejes mecánicos salen de la optimización de cartera...
    assert set(portfolio_vars) == {"target_size"}
    # ...y aparecen exactamente en el bloque de estrés.
    assert set(stress) == {"rebalance_drift_tolerance", "min_hold_percentile",
                           "rotation_edge_percentiles"}
    assert PORTFOLIO_STRESS_FIELDS <= PORTFOLIO_FIELDS  # siguen siendo campos de cartera
    assert "target_size" not in PORTFOLIO_STRESS_FIELDS


def test_lag_days_is_model_not_portfolio() -> None:
    """execution_lag_days cambia qué fundamentales son observables -> es de modelo."""
    assert "execution_lag_days" in MODEL_FIELDS
    assert "execution_lag_days" not in PORTFOLIO_FIELDS


def test_portfolio_fields_do_not_overlap_model() -> None:
    assert MODEL_FIELDS.isdisjoint(PORTFOLIO_FIELDS)
    assert {"target_size", "commission_bps", "slippage_bps", "profile"} <= PORTFOLIO_FIELDS


# --- Barrido inteligente del full_study -----------------------------------------------------

def test_study_and_full_study_share_the_same_catalogue_by_phase() -> None:
    """Manual permite subconjuntos; full usa todo, pero valores y fases son idénticos."""
    from module.scenarios.variables import FULL_STUDY_OPTIONS, FULL_STUDY_PHASE3_OPTIONS, STUDY_OPTIONS

    assert set(STUDY_OPTIONS) == set(FULL_STUDY_OPTIONS) | set(FULL_STUDY_PHASE3_OPTIONS)
    assert FULL_STUDY_OPTIONS["target_size"] == [5, 8, 10, 12, 15]
    assert FULL_STUDY_OPTIONS["min_hold_percentile"] == [60, 70, 80, 85]
    # Cada fase utiliza exactamente los valores del catálogo común.
    for axis, values in FULL_STUDY_OPTIONS.items():
        assert values == STUDY_OPTIONS[axis]
    for axis, values in FULL_STUDY_PHASE3_OPTIONS.items():
        assert values == STUDY_OPTIONS[axis]
    assert {"lgbm_max_depth", "lgbm_n_estimators", "lgbm_learning_rate",
            "lgbm_min_child_samples"}.isdisjoint(FULL_STUDY_OPTIONS)
    assert {"commission_bps", "slippage_bps", "profile"}.isdisjoint(STUDY_OPTIONS)


def test_fixed_portfolio_size_is_a_simple_portfolio_axis() -> None:
    from module.scenarios.variables import FULL_STUDY_OPTIONS

    _, portfolio_vars = split_variables(FULL_STUDY_OPTIONS)
    assert portfolio_vars["target_size"] == [5, 8, 10, 12, 15]


def test_full_study_sweep_has_no_snapshot_day() -> None:
    """snapshot_day se eliminó: la rejilla la define execution_lag_days (fin_de_periodo + lag)."""
    from module.scenarios.variables import FULL_STUDY_OPTIONS, STUDY_OPTIONS

    assert "snapshot_day" not in STUDY_OPTIONS
    assert "snapshot_day" not in FULL_STUDY_OPTIONS
    assert "execution_lag_days" in FULL_STUDY_OPTIONS


def test_experimental_keeps_controls_that_study_does_not_optimize() -> None:
    from module.scenarios.variables import EXPERIMENT_OPTIONS, STUDY_OPTIONS

    assert {"commission_bps", "slippage_bps", "profile"} <= set(EXPERIMENT_OPTIONS)
    assert {"commission_bps", "slippage_bps", "profile"}.isdisjoint(STUDY_OPTIONS)
    assert "target_size" in EXPERIMENT_OPTIONS
    assert "target_size" in STUDY_OPTIONS


def test_manual_study_routes_hyperparameters_to_phase3(monkeypatch) -> None:
    from environment import Settings

    captured = {}

    def fake_run(_settings, **kwargs):
        captured.update(kwargs)
        return "study-ok"

    monkeypatch.setattr(execution, "run_optimization", fake_run)
    result = execution.execute_study(
        Settings(), study_payload={"name": "manual"},
        variables={"train_lookback_years": [4, 8], "lgbm_max_depth": [3, 5],
                   "target_size": [8, 10]},
    )

    assert result == "study-ok"
    assert "lgbm_max_depth" not in captured["model_vars"]
    assert captured["hyperparameter_options"] == {"lgbm_max_depth": [3, 5]}
    assert "target_size" in captured["portfolio_vars"]


def test_study_data_window_uses_the_longest_selected_training_lookback() -> None:
    from environment import Settings

    effective = execution._with_study_data_start(
        Settings(execution_year=2015, execution_quarter=1, execution_lag_days=45),
        {"train_lookback_years": [4, 8, 12]},
    )

    # El lag desplaza el ancla a febrero, pero el dataset se abre en enero de 2003 para cubrir
    # toda la ventana de 12 años y conservar calentamiento de variables técnicas.
    assert effective.data_start_date == "2003-01-01"


# --- Fase 2 greedy: sin producto cartesiano -------------------------------------------------

class _FakeStore:
    """Store mínimo que solo registra add_to_study; run_dir se ignora en estos tests."""
    def add_to_study(self, *_args, **_kwargs) -> None:
        pass


def _run_counter(monkeypatch, summaries_by_overrides):
    """Sustituye execute_run y _summary_for_run por versiones que no entrenan y cuentan llamadas."""
    calls = {"n": 0, "overrides": []}

    def fake_execute_run(settings, *, mode, run_kind, study_id, label, description, tags,
                         grid_definition, store, agent_dir=None, **_kw):
        calls["n"] += 1
        overrides = tuple(sorted(grid_definition.get("overrides", {}).items()))
        calls["overrides"].append(overrides)
        return f"run-{calls['n']}"

    def fake_summary(_store, run_id):
        idx = int(run_id.split("-")[1]) - 1
        return summaries_by_overrides[idx] if idx < len(summaries_by_overrides) else {"mean_rank_ic": 0.0}

    monkeypatch.setattr(execution, "execute_run", fake_execute_run)
    monkeypatch.setattr(execution, "_summary_for_run", fake_summary)
    return calls


def test_greedy_phase2_is_not_cartesian(monkeypatch) -> None:
    """Con N ejes de 2 niveles, Fase 2 hace ~1+N runs, nunca 2^N."""
    from environment import Settings
    settings = Settings()
    # Fase 1: 3 ejes, cada uno con baseline + 2 niveles (mejor y segundo).
    phase1 = pd.DataFrame([
        {"phase": 1, "scenario": "baseline", "axis": None, "overrides": {}, "mean_rank_ic": 0.010,
         "rank_ic_positive_fraction": 0.5, "rank_ic_std": 0.05},
        {"phase": 1, "scenario": "train_8", "axis": "train_lookback_years", "overrides": {"train_lookback_years": 8},
         "mean_rank_ic": 0.020, "rank_ic_positive_fraction": 0.6, "rank_ic_std": 0.05},
        {"phase": 1, "scenario": "train_12", "axis": "train_lookback_years", "overrides": {"train_lookback_years": 12},
         "mean_rank_ic": 0.018, "rank_ic_positive_fraction": 0.58, "rank_ic_std": 0.05},
        {"phase": 1, "scenario": "depth_6", "axis": "lgbm_max_depth", "overrides": {"lgbm_max_depth": 6},
         "mean_rank_ic": 0.015, "rank_ic_positive_fraction": 0.55, "rank_ic_std": 0.05},
        {"phase": 1, "scenario": "depth_5", "axis": "lgbm_max_depth", "overrides": {"lgbm_max_depth": 5},
         "mean_rank_ic": 0.014, "rank_ic_positive_fraction": 0.54, "rank_ic_std": 0.05},
    ])
    model_vars = {"train_lookback_years": [8, 12], "lgbm_max_depth": [6, 5]}
    calls = _run_counter(monkeypatch, [{"mean_rank_ic": 0.02}] * 20)

    rows, selected, decision = execution._greedy_phase2(
        phase1, model_vars, settings, _FakeStore(), "sid", name="t", description="", tags=[])

    # 1 combined_best + 1 "second" por eje con 2 opciones = 1 + 2 = 3 runs. Nunca 2^2=4 cartesiano
    # ni más. (La cadencia baseline es trimestral, así que no se dispara el caso quarter.)
    assert calls["n"] == 3, f"esperaba 3 runs greedy, hubo {calls['n']}"
    assert isinstance(selected, dict)
    assert "axes" in decision


# --- Fase de cartera: criterio económico ----------------------------------------------------

def test_economic_score_prefers_higher_ir_and_lower_drawdown() -> None:
    assert execution._economic_score({"information_ratio": 0.5}, "information_ratio") == 0.5
    # drawdown se convierte a negativo: menor drawdown => mayor score
    assert execution._economic_score({"max_drawdown": 0.2}, "max_drawdown") == -0.2
    assert execution._economic_score({}, "information_ratio") == float("-inf")


def test_portfolio_phase_selects_best_by_information_ratio(monkeypatch) -> None:
    """La fase de cartera elige por criterio económico y re-backtestea (mode=backtest)."""
    from environment import Settings
    settings = Settings()
    modes_seen = []

    def fake_execute_run(settings, *, mode, run_kind, study_id, label, description, tags,
                         grid_definition, store, agent_dir=None, **_kw):
        modes_seen.append(mode)
        key = grid_definition.get("axis"), grid_definition.get("value")
        return f"{key[0]}={key[1]}"

    def fake_summary(_store, run_id):
        # information_ratio por valor: para target_size gana 10; para commission_bps gana 0.
        if run_id.startswith("target_size="):
            value = int(run_id.split("=")[1])
            return {"information_ratio": 0.4 if value == 10 else 0.1}
        if run_id.startswith("commission_bps="):
            value = int(run_id.split("=")[1])
            return {"information_ratio": 0.6 if value == 0 else 0.3}
        return {"information_ratio": 0.0}

    monkeypatch.setattr(execution, "execute_run", fake_execute_run)
    monkeypatch.setattr(execution, "_summary_for_run", fake_summary)

    portfolio_vars = {"target_size": [8, 10], "commission_bps": [0, 10]}
    selected, trace, _rows = execution._portfolio_phase(
        settings, portfolio_vars, _FakeStore(), "sid", agent_dir=None, name="t", description="", tags=[])

    assert selected == {"target_size": 10, "commission_bps": 0}
    assert all(m == "backtest" for m in modes_seen), "la fase de cartera no debe reentrenar"
    assert trace["axes"]["target_size"]["chosen"] == 10


# --- Regresión: `phase` debe ser string en todas las fases (evita crash de pyarrow) ----------

def test_all_comparison_phases_are_strings_and_concat_writes_parquet(tmp_path) -> None:
    """`comparison_data.parquet` mezclaba `phase` int (fases 1/2/3) con string (cartera/perfiles):
    pandas infiere un dtype numerico para la columna y pyarrow revienta al escribir los valores
    string ("Could not convert 'cartera' ... to int64"). Fases 4 y 5 (antes "cartera"/"perfiles")
    deben quedar como string, igual que las fases 1/2/3, para que el concat + write_parquet no
    falle nunca, sea cual sea el orden de filas."""
    from module.common.utils import write_parquet

    phase1 = pd.DataFrame([{"phase": "1", "scenario": "baseline", "mean_rank_ic": 0.01, "run_id": "r1"}])
    phase2 = pd.DataFrame([{"phase": "2", "scenario": "combined_best", "mean_rank_ic": 0.02, "run_id": "r2"}])
    hyper = pd.DataFrame([{"phase": "3", "scenario": "hyper_a", "mean_rank_ic": 0.02, "run_id": "r3"}])
    cartera = pd.DataFrame([{"phase": "4_cartera", "scenario": "target_size=10", "mean_rank_ic": 0.02, "run_id": "r4"}])
    perfiles = pd.DataFrame([{"phase": "5_perfiles", "scenario": "balanced", "mean_rank_ic": 0.02, "run_id": "r5"}])

    comparison = pd.concat([phase1, phase2, hyper, cartera, perfiles], ignore_index=True, sort=False)
    assert comparison["phase"].apply(type).eq(str).all()

    out = tmp_path / "comparison_data.parquet"
    write_parquet(comparison, out)  # no debe lanzar ArrowInvalid
    assert set(pd.read_parquet(out)["phase"]) == {"1", "2", "3", "4_cartera", "5_perfiles"}


def test_safe_scenario_run_skips_on_failure(monkeypatch) -> None:
    """Un escenario que falla al entrenar se salta y se registra, sin propagar la excepción."""
    from environment import Settings
    settings = Settings()
    skipped = []

    def boom(*_a, **_k):
        raise RuntimeError("No se pudieron entrenar agentes con la historia disponible.")

    monkeypatch.setattr(execution, "execute_run", boom)
    run_id = execution._safe_scenario_run(
        _FakeStore(), settings, study_id="sid", label="x", description="", tags=[],
        grid_definition={"overrides": {"train_lookback_years": 2}}, skipped=skipped, scenario="train_2")

    assert run_id is None
    assert len(skipped) == 1 and skipped[0]["scenario"] == "train_2"
    assert "entrenar" in skipped[0]["error"]


def test_safe_scenario_run_returns_id_on_success(monkeypatch) -> None:
    from environment import Settings
    settings = Settings()
    skipped = []
    monkeypatch.setattr(execution, "execute_run", lambda *_a, **_k: "run-ok")
    run_id = execution._safe_scenario_run(
        _FakeStore(), settings, study_id="sid", label="x", description="", tags=[],
        grid_definition={"overrides": {}}, skipped=skipped, scenario="baseline")
    assert run_id == "run-ok"
    assert skipped == []


def test_portfolio_phase_skips_profile_axis(monkeypatch) -> None:
    """El perfil no se barre en la fase de cartera (se cubre con los 8 runs de perfil)."""
    from environment import Settings
    settings = Settings()

    def fake_execute_run(*_a, **_k):
        raise AssertionError("no debería ejecutarse ningún run para el eje profile")

    monkeypatch.setattr(execution, "execute_run", fake_execute_run)
    selected, trace, _rows = execution._portfolio_phase(
        settings, {"profile": ["balanced", "quality"]}, _FakeStore(), "sid",
        agent_dir=None, name="t", description="", tags=[])
    assert selected == {}
    assert trace["axes"] == {}


def test_portfolio_selection_summary_excludes_reserved_years(tmp_path) -> None:
    """Un retorno extremo en 2025 no puede cambiar la métrica usada para elegir cartera."""
    from environment import Settings
    from module.runs.results_store import ResultsStore

    store = ResultsStore(tmp_path / "results")
    run_id = "candidate"
    artifacts = store.runs_root / run_id / "artifacts"
    artifacts.mkdir(parents=True)
    pd.DataFrame([
        {"snapshot_date": "2024-01-31", "portfolio_value": 100.0, "benchmark_value": 100.0,
         "portfolio_return": 0.0, "benchmark_return": 0.0, "excess_return": 0.0, "turnover_pct": 0.0},
        {"snapshot_date": "2024-12-31", "portfolio_value": 110.0, "benchmark_value": 105.0,
         "portfolio_return": 0.10, "benchmark_return": 0.05, "excess_return": 0.05, "turnover_pct": 0.1},
        {"snapshot_date": "2025-12-31", "portfolio_value": 1000.0, "benchmark_value": 106.0,
         "portfolio_return": 8.0, "benchmark_return": 0.01, "excess_return": 7.99, "turnover_pct": 0.1},
    ]).to_parquet(artifacts / "equity.parquet", index=False)

    summary = execution._portfolio_selection_summary(store, run_id, Settings())

    assert summary["portfolio_selection_until_year"] == 2024
    assert summary["portfolio_selection_periods"] == 2
    assert summary["cagr_portfolio"] < 0.2
