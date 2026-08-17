from __future__ import annotations

import json
from pathlib import Path

import pytest

from module.common import utils
from module.common.utils import write_json
from module.storage import studies
from module.studies.runner import _profile_comparison_row
from module.studies.selection import choose_candidate
from module.web import queries
from module.web.api import study_preflight


def test_atomic_write_survives_a_transient_file_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un bloqueo momentáneo del sistema de ficheros no puede tumbar horas de cómputo.

    En Windows un antivirus o el indexador pueden retener el destino unas decenas de milisegundos
    y hacer fallar `os.replace` con WinError 5 pese a tener permisos. Ocurrió de verdad: abortó un
    Portfolio Study a las 7 de 1.728 combinaciones.
    """
    target = tmp_path / "estado.json"
    attempts = {"count": 0}
    original = utils.os.replace

    def flaky_replace(source, destination):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PermissionError(5, "Acceso denegado")
        return original(source, destination)

    monkeypatch.setattr(utils.os, "replace", flaky_replace)
    monkeypatch.setattr(utils, "_REPLACE_BACKOFF_SECONDS", 0.0)
    write_json({"ok": True}, target)

    assert attempts["count"] == 3, "no reintentó tras el bloqueo transitorio"
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}
    # El temporal no queda huérfano en el directorio.
    assert [item.name for item in tmp_path.iterdir()] == ["estado.json"]


def test_atomic_write_gives_up_on_a_permanent_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si el bloqueo no es transitorio, el error debe propagarse y no silenciarse."""
    def always_locked(source, destination):
        raise PermissionError(5, "Acceso denegado")

    monkeypatch.setattr(utils.os, "replace", always_locked)
    monkeypatch.setattr(utils, "_REPLACE_BACKOFF_SECONDS", 0.0)
    with pytest.raises(PermissionError):
        write_json({"ok": True}, tmp_path / "estado.json")


def _result(rank_ic: float) -> dict:
    """Cohortes mensuales de 2015 a 2024, la rejilla real del ganador.

    El número de cohortes importa: la puerta pareada exige al menos un bloque completo de fechas
    comunes (un año) antes de considerar que hay evidencia comparable.
    """
    dates = [f"20{year:02d}-{month:02d}-28" for year in range(15, 25) for month in range(1, 13)]
    return {
        "evaluation_key": str(rank_ic),
        "summary": {"mean_rank_ic": rank_ic, "rank_ic_positive_fraction": 0.75, "rank_ic_std": 0.02},
        "eras": [
            {"era": "2015-2018", "rank_ic": rank_ic},
            {"era": "2019-2021", "rank_ic": rank_ic},
            {"era": "2022-2024", "rank_ic": rank_ic},
        ],
        "rank_ic_by_cohort": [{"date": date, "rank_ic": rank_ic} for date in dates],
        "known_stress_not_selection": [{"year": 2025, "alpha": 99}],
    }


def test_selection_uses_rank_ic_and_excludes_known_stress() -> None:
    incumbent = {"candidate_id": "a", "value": 60, "result": _result(0.03)}
    challenger = {"candidate_id": "b", "value": 45, "result": _result(0.05)}
    decision = choose_candidate(incumbent, [incumbent, challenger], "execution_lag_days")
    assert decision["winner_candidate_id"] == "b"
    assert decision["selection_metric"] == "rank_ic_only"
    assert decision["known_stress_excluded"] is True


def test_run_exists_before_result_and_resume_keeps_identity(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(studies, "STUDIES_ROOT", tmp_path / "studies")
    study_id, _ = studies.create_study({"name": "test", "configuration": {}, "catalog": {}})
    first = studies.create_run(
        study_id, logical_key="baseline", phase="temporal",
        variable_id="baseline", value="baseline", configuration={},
    )
    second = studies.create_run(
        study_id, logical_key="baseline", phase="temporal",
        variable_id="baseline", value="baseline", configuration={}, attempt=2,
    )
    assert first["status"] == "queued"
    assert second["run_id"] == first["run_id"]
    assert second["attempt"] == 2


def test_events_are_persistent_incremental_and_print_madrid_time(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(studies, "STUDIES_ROOT", tmp_path / "studies")
    monkeypatch.setattr(studies, "madrid_time_of_day", lambda: "11:58:44")
    study_id, _ = studies.create_study({"name": "test", "configuration": {}, "catalog": {}})
    studies.append_event(study_id, "info", "one", "Primero.")
    studies.append_event(study_id, "info", "two", "Segundo.")
    assert [row["sequence"] for row in studies.read_events(study_id, after=1)] == [2]
    assert "[11:58:44] [INFO]" in capsys.readouterr().out


def test_profile_comparison_row_contains_only_parquet_scalars() -> None:
    row = _profile_comparison_row("balanced", {
        "summary": {
            "geometric_excess_return": 0.03,
            "information_ratio": 0.4,
            "annualized_turnover": 1.2,
            "mean_cash_weight": 0.0,
            "confirmation": {},
        },
        "rank_ic": {"mean_rank_ic": 0.05},
        "eras": [],
    })
    assert row == {
        "profile": "balanced", "mean_rank_ic": 0.05, "geometric_excess_return": 0.03,
        "information_ratio": 0.4, "annualized_turnover": 1.2, "mean_cash_weight": 0.0,
    }
    assert all(not isinstance(value, (dict, list)) for value in row.values())


def test_study_contains_only_model_study_identity(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(studies, "STUDIES_ROOT", tmp_path / "studies")
    _, directory = studies.create_study({"name": "test", "configuration": {}, "catalog": {}})
    study_id = json.loads((directory / "study.json").read_text(encoding="utf-8"))["study_id"]
    studies.create_run(
        study_id, logical_key="baseline", phase="temporal",
        variable_id="baseline", value="baseline", configuration={},
    )
    payload = json.loads((directory / "study.json").read_text(encoding="utf-8"))
    assert payload["study_type"] == "model_study"
    assert set(payload) >= {"study_id", "study_type", "status", "configuration"}
    assert queries.studies()[0]["max_rank_ic"] is None
    studies.update_study(study_id, status="succeeded", completed_runs=35)
    assert queries.studies()[0]["runs_remaining"] == 0


def test_full_scientific_route_only_changes_storage_not_selection() -> None:
    """La ruta científica completa levanta la regla 5 sin tocar qué se ejecuta ni cómo se elige."""
    off = study_preflight({"definition": None})
    on = study_preflight({"definition": None, "retain_all_runs": True})
    assert off["retain_all_runs"] is False and on["retain_all_runs"] is True
    # Mismo plan experimental: ni un run más, ni un candidato distinto.
    assert on["definition"] == off["definition"]
    assert on["budget"]["total_runs"] == off["budget"]["total_runs"]
    assert on["budget"]["predictive_evaluations"] == off["budget"]["predictive_evaluations"]
    # El único efecto es de disco, y debe declararse antes de lanzar.
    assert on["budget"]["retained_run_evidence"] == off["budget"]["total_runs"] - 2
    assert on["budget"]["estimated_incremental_bytes"] > off["budget"]["estimated_incremental_bytes"]


def test_post_winner_diagnostics_are_optional_and_on_by_default() -> None:
    """Apagar los diagnósticos posteriores recorta el presupuesto sin tocar la selección.

    Un Study cuyo único fin es elegir configuración termina al congelar el ganador: así no gasta la
    única evaluación de la era reservada 2025-26 sobre una configuración que se va a descartar.
    """
    on = study_preflight({"definition": None})
    off = study_preflight({"definition": None, "post_winner_diagnostics": False})
    # El defecto es ejecutarlos: apagarlos es la excepción y hay que pedirlo.
    assert on["post_winner_diagnostics"] is True
    assert off["post_winner_diagnostics"] is False
    # La ciencia de la selección no cambia: mismos candidatos y mismas evaluaciones predictivas.
    assert off["definition"] == on["definition"]
    assert off["budget"]["predictive_evaluations"] == on["budget"]["predictive_evaluations"]
    # Lo que desaparece del presupuesto es exactamente lo que ya no se ejecuta.
    assert off["budget"]["profiles"] == 0
    assert off["budget"]["robustness_groups"] == 0
    assert off["budget"]["portfolio_diagnostics"] == 0
    assert off["budget"]["total_runs"] < on["budget"]["total_runs"]
    assert off["budget"]["estimated_minutes"] < on["budget"]["estimated_minutes"]


def test_retention_counts_only_the_runs_that_will_actually_execute() -> None:
    """Sin diagnósticos posteriores, la evidencia completa se cuenta sobre el total ya recortado.

    El orden importa: `retention_budget` deriva cuántos runs retienen evidencia de `total_runs`, así
    que debe verlo después del recorte y no prometer disco para runs que nunca se ejecutan.
    """
    from module.studies.catalog import recommended_definition

    definition = recommended_definition()
    with_diagnostics = study_preflight({"definition": definition, "retain_all_runs": True})
    without = study_preflight({
        "definition": definition, "retain_all_runs": True, "post_winner_diagnostics": False,
    })
    for budget in (with_diagnostics["budget"], without["budget"]):
        assert budget["retained_run_evidence"] == budget["total_runs"] - 2
    assert without["budget"]["retained_run_evidence"] < with_diagnostics["budget"]["retained_run_evidence"]
    assert (
        without["budget"]["estimated_incremental_bytes"]
        < with_diagnostics["budget"]["estimated_incremental_bytes"]
    )


def test_retained_run_evidence_is_confined_to_its_study(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(studies, "STUDIES_ROOT", tmp_path / "studies")
    study_id, directory = studies.create_study({
        "name": "test", "configuration": {}, "catalog": {}, "retain_all_runs": True,
    })
    run = studies.create_run(
        study_id, logical_key="predictive:foo:ab", phase="temporal",
        variable_id="foo", value="ab", configuration={},
    )
    assert run["evidence_path"] is None
    source = f"run:{run['run_id']}"
    # Sin evidencia retenida la vista lo dice, no devuelve la del ganador por descuido.
    with pytest.raises(FileNotFoundError):
        queries._evidence_dir(directory, study_id, source)
    (directory / "runs_evidence" / "predictive__foo__ab").mkdir(parents=True)
    studies.update_run(study_id, run["run_id"], evidence_path="runs_evidence/predictive__foo__ab")
    assert queries._evidence_dir(directory, study_id, source).name == "predictive__foo__ab"
    # La ruta viene del artefacto, pero nunca puede escapar del Study.
    studies.update_run(study_id, run["run_id"], evidence_path="../../../etc")
    with pytest.raises(ValueError):
        queries._evidence_dir(directory, study_id, source)
    assert queries._evidence_dir(directory, study_id, None).name == "evidence"
    assert queries._evidence_dir(directory, study_id, "baseline").name == "evidence_baseline"
