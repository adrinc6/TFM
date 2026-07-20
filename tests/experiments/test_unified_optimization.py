"""Contrato del ciclo unificado study/full_study: clasificación de variables, Fase 2 greedy
(sin producto cartesiano) y fase de cartera (criterio económico, sin reentrenar).
"""

from __future__ import annotations

import pandas as pd

from module.runs import execution
from module.runs.experiments import MODEL_FIELDS, PORTFOLIO_FIELDS, split_variables


# --- Clasificación de variables -------------------------------------------------------------

def test_split_variables_separates_model_and_portfolio() -> None:
    variables = {
        "train_lookback_years": [5, 8, 12],   # modelo
        "execution_lag_days": [15, 45],        # modelo (afecta observabilidad -> dataset)
        "commission_bps": [0, 10],             # cartera
        "target_min": [6, 10],                 # cartera
        "profile": ["balanced", "quality"],    # cartera
        "desconocida": [1, 2],                 # se descarta
    }
    model_vars, portfolio_vars = split_variables(variables)
    assert set(model_vars) == {"train_lookback_years", "execution_lag_days"}
    assert set(portfolio_vars) == {"commission_bps", "target_min", "profile"}
    assert "desconocida" not in model_vars and "desconocida" not in portfolio_vars


def test_lag_days_is_model_not_portfolio() -> None:
    """execution_lag_days cambia qué fundamentales son observables -> es de modelo."""
    assert "execution_lag_days" in MODEL_FIELDS
    assert "execution_lag_days" not in PORTFOLIO_FIELDS


def test_portfolio_fields_do_not_overlap_model() -> None:
    assert MODEL_FIELDS.isdisjoint(PORTFOLIO_FIELDS)
    assert {"target_min", "target_max", "commission_bps", "slippage_bps", "profile"} <= PORTFOLIO_FIELDS


# --- Barrido inteligente del full_study -----------------------------------------------------

def test_full_study_sweep_keeps_all_axes_but_trims_density() -> None:
    """El full_study conserva todos los ejes de modelo (no elimina ninguno 'por si acaso') y recorta
    la densidad de niveles donde el solapamiento es evidente. En cartera, target_min/target_max/
    max_weight se fusionan en el eje compuesto target_band."""
    from module.scenarios.variables import FULL_STUDY_OPTIONS, STUDY_OPTIONS

    fused = {"target_min", "target_max", "max_weight_per_position"}
    # Todos los ejes del study manual siguen presentes salvo los tres fusionados en target_band.
    assert set(STUDY_OPTIONS) - fused <= set(FULL_STUDY_OPTIONS)
    assert "target_band" in FULL_STUDY_OPTIONS
    assert not (fused & set(FULL_STUDY_OPTIONS)), "los ejes fusionados no deben barrerse sueltos"
    # Los ejes simples conservados son subconjunto (o igual) de los valores del study manual.
    for axis, values in FULL_STUDY_OPTIONS.items():
        if axis == "target_band":
            continue
        assert set(values) <= set(STUDY_OPTIONS[axis]), f"{axis} introduce valores nuevos no admitidos"
    # Al menos los ejes densos y suaves están recortados.
    assert len(FULL_STUDY_OPTIONS["train_lookback_years"]) < len(STUDY_OPTIONS["train_lookback_years"])
    assert len(FULL_STUDY_OPTIONS["lgbm_max_depth"]) < len(STUDY_OPTIONS["lgbm_max_depth"])


def test_target_band_is_a_valid_composite_portfolio_axis() -> None:
    """target_band expande a (target_min, target_max, max_weight) coherentes y válidos."""
    from dataclasses import replace

    from environment import Settings
    from module.runs.experiments import PORTFOLIO_FIELDS, split_variables
    from module.scenarios.variables import FULL_STUDY_OPTIONS, TARGET_BANDS

    assert "target_band" in PORTFOLIO_FIELDS
    _, portfolio_vars = split_variables(FULL_STUDY_OPTIONS)
    assert "target_band" in portfolio_vars
    assert [(b["target_min"], b["target_max"]) for b in TARGET_BANDS] == [(5, 8), (8, 12), (12, 15)]
    # Cada banda cumple las restricciones de Settings (incluida max_weight * target_min >= 1).
    for band in TARGET_BANDS:
        replace(Settings(), **band)


def test_full_study_sweep_has_no_snapshot_day() -> None:
    """snapshot_day se eliminó: la rejilla la define execution_lag_days (fin_de_periodo + lag)."""
    from module.scenarios.variables import FULL_STUDY_OPTIONS, STUDY_OPTIONS

    assert "snapshot_day" not in STUDY_OPTIONS
    assert "snapshot_day" not in FULL_STUDY_OPTIONS
    assert "execution_lag_days" in FULL_STUDY_OPTIONS


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

    # Valores válidos para la constraint max_weight_per_position(0.15)*target_min>=1 (target_min>=7).
    def fake_execute_run(settings, *, mode, run_kind, study_id, label, description, tags,
                         grid_definition, store, agent_dir=None, **_kw):
        modes_seen.append(mode)
        key = grid_definition.get("axis"), grid_definition.get("value")
        return f"{key[0]}={key[1]}"

    def fake_summary(_store, run_id):
        # information_ratio por valor: para target_min gana 10; para commission_bps gana 0.
        if run_id.startswith("target_min="):
            v = int(run_id.split("=")[1]); return {"information_ratio": 0.4 if v == 10 else 0.1}
        if run_id.startswith("commission_bps="):
            v = int(run_id.split("=")[1]); return {"information_ratio": 0.6 if v == 0 else 0.3}
        return {"information_ratio": 0.0}

    monkeypatch.setattr(execution, "execute_run", fake_execute_run)
    monkeypatch.setattr(execution, "_summary_for_run", fake_summary)

    portfolio_vars = {"target_min": [8, 10], "commission_bps": [0, 10]}
    selected, trace, _rows = execution._portfolio_phase(
        settings, portfolio_vars, _FakeStore(), "sid", agent_dir=None, name="t", description="", tags=[])

    assert selected == {"target_min": 10, "commission_bps": 0}
    assert all(m == "backtest" for m in modes_seen), "la fase de cartera no debe reentrenar"
    assert trace["axes"]["target_min"]["chosen"] == 10


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
    cartera = pd.DataFrame([{"phase": "4_cartera", "scenario": "target_min=10", "mean_rank_ic": 0.02, "run_id": "r4"}])
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
