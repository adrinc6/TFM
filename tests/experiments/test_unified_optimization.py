"""Contrato del ciclo unificado study/full_study: clasificación de variables, Fase 2 greedy
(sin producto cartesiano) y fase de cartera (criterio económico, sin reentrenar).
"""

from __future__ import annotations

import pandas as pd
import pytest

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


def test_greedy_phase2_reexplores_quarter_when_cadence_not_quarterly(monkeypatch) -> None:
    """Si la cadencia ganadora es anual, se re-explora execution_quarter sobre la combinación."""
    from environment import Settings
    settings = Settings()  # baseline fundamental_step_months=3, execution_quarter=1
    phase1 = pd.DataFrame([
        {"phase": 1, "scenario": "baseline", "axis": None, "overrides": {}, "mean_rank_ic": 0.010,
         "rank_ic_positive_fraction": 0.5, "rank_ic_std": 0.05},
        {"phase": 1, "scenario": "cadence_anual", "axis": "fundamental_step_months",
         "overrides": {"fundamental_step_months": 12}, "mean_rank_ic": 0.030,
         "rank_ic_positive_fraction": 0.7, "rank_ic_std": 0.04},
    ])
    model_vars = {"fundamental_step_months": [12]}
    calls = _run_counter(monkeypatch, [{"mean_rank_ic": 0.03}] * 20)

    rows, selected, decision = execution._greedy_phase2(
        phase1, model_vars, settings, _FakeStore(), "sid", name="t", description="", tags=[])

    # combined_best (cadence=12) + re-exploración de los trimestres != 1 (Q2,Q3,Q4) = 1 + 3 = 4
    assert decision.get("quarter_reexplored_for_cadence") == 12
    quarter_runs = [r for r in rows if "quarter" in r["scenario"]]
    assert len(quarter_runs) == 3, f"esperaba 3 runs de quarter, hubo {len(quarter_runs)}"


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
    selected, trace = execution._portfolio_phase(
        settings, portfolio_vars, _FakeStore(), "sid", agent_dir=None, name="t", description="", tags=[])

    assert selected == {"target_min": 10, "commission_bps": 0}
    assert all(m == "backtest" for m in modes_seen), "la fase de cartera no debe reentrenar"
    assert trace["axes"]["target_min"]["chosen"] == 10


def test_portfolio_phase_skips_profile_axis(monkeypatch) -> None:
    """El perfil no se barre en la fase de cartera (se cubre con los 8 runs de perfil)."""
    from environment import Settings
    settings = Settings()

    def fake_execute_run(*_a, **_k):
        raise AssertionError("no debería ejecutarse ningún run para el eje profile")

    monkeypatch.setattr(execution, "execute_run", fake_execute_run)
    selected, trace = execution._portfolio_phase(
        settings, {"profile": ["balanced", "quality"]}, _FakeStore(), "sid",
        agent_dir=None, name="t", description="", tags=[])
    assert selected == {}
    assert trace["axes"] == {}
