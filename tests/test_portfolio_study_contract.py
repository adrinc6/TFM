"""Contrato del Portfolio Study: rejilla, criterio de selección y retención del mejor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from module.studies.catalog import BY_ID, SELECTION_UNTIL_YEAR
from module.studies.portfolio_study import (
    PORTFOLIO_STUDY_VARIABLES, SELECTION_METRIC, _metric, _profiles_with_winner,
    combination_key, improvement, portfolio_grid, run_portfolio_study, selection_evidence,
)


def test_grid_is_the_full_cartesian_product_of_the_six_variables() -> None:
    """El cartesiano existe para ver interacciones; si faltara una combinación, no las vería."""
    expected = 1
    for variable_id in PORTFOLIO_STUDY_VARIABLES:
        expected *= len(BY_ID[variable_id].values)
    grid = portfolio_grid()
    assert len(grid) == expected
    assert len({tuple(sorted(item.items(), key=str)) for item in grid}) == expected
    assert all(set(item) == set(PORTFOLIO_STUDY_VARIABLES) for item in grid)


def test_grid_respects_a_restricted_definition() -> None:
    """Acotar la rejilla no debe exigir tocar el catálogo."""
    definition = {
        "target_size": {"values": [8, 12]},
        "max_cash_weight": {"values": [0.0]},
    }
    grid = portfolio_grid(definition)
    assert {item["target_size"] for item in grid} == {8, 12}
    assert {item["max_cash_weight"] for item in grid} == {0.0}
    # Las variables no acotadas conservan todo el catálogo.
    assert {item["sizing_mode"] for item in grid} == set(BY_ID["sizing_mode"].values)


def test_cost_assumptions_are_never_optimized() -> None:
    """Optimizar comisión o slippage sería elegir el mundo en el que la estrategia luce mejor."""
    assert "commission_bps" not in PORTFOLIO_STUDY_VARIABLES
    assert "slippage_bps" not in PORTFOLIO_STUDY_VARIABLES


def test_non_finite_metric_never_wins() -> None:
    """Una combinación degenerada (IR NaN o infinito) no puede ganar por accidente."""
    assert _metric({SELECTION_METRIC: 0.4}) == pytest.approx(0.4)
    assert _metric({SELECTION_METRIC: float("nan")}) == float("-inf")
    assert _metric({SELECTION_METRIC: float("inf")}) == float("-inf")
    assert _metric({SELECTION_METRIC: None}) == float("-inf")
    assert _metric({}) == float("-inf")


def test_only_the_winner_keeps_evidence_and_it_is_the_best_by_ir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regla 5: las descartadas guardan solo resumen; la evidencia es la del mejor vigente.

    Se simula la evaluación para no depender de un dataset real: cada combinación recibe un IR
    determinista y la mejor NO es la primera ni la última, para que el reemplazo de evidencia se
    ejercite de verdad.
    """
    scores = {8: 0.10, 12: 0.90, 16: 0.50}

    def fake_evaluation(
        values: dict[str, Any], profile: str, evidence_dir: Path, retain_dir: Path | None = None,
        *, include_model_artifacts: bool = False,
    ) -> dict[str, Any]:
        ratio = scores[int(values["target_size"])]
        if retain_dir is not None:
            retain_dir.mkdir(parents=True, exist_ok=True)
            (retain_dir / "marca.txt").write_text(str(values["target_size"]), encoding="utf-8")
        return {
            "summary": {
                "information_ratio": ratio,
                "geometric_excess_return": ratio / 10,
                "annualized_turnover": 2.0,
                "confirmation": {"information_ratio": -1.0},
            },
            "rank_ic": {"mean_rank_ic": 0.1},
        }

    monkeypatch.setattr("module.studies.runner.run_profile_evaluation", fake_evaluation)
    # La rejilla recorta la evidencia; aquí se simula la evaluación, así que basta con que el
    # recorte no falle sobre un fichero mínimo.
    monkeypatch.setattr(
        "module.studies.portfolio_study.selection_evidence",
        lambda evidence_dir, workspace: workspace,
    )
    monkeypatch.setattr(
        "module.studies.portfolio_study._profiles_with_winner",
        lambda values, evidence_dir, selection_dir, output_dir, profiles=None: [],
    )

    definition = {
        "target_size": {"values": [8, 12, 16]},
        "max_cash_weight": {"values": [0.0]},
        "sizing_mode": {"values": ["equal"]},
        "minimum_holding_period": {"values": ["none"]},
        "coverage_percentile_floor": {"values": [0.0]},
        "rebalance_drift_tolerance": {"values": [0.0]},
    }
    output = tmp_path / "portfolio_study"
    # `workers=1` en todas las pruebas de este fichero: la evaluación real se sustituye por un doble
    # con `monkeypatch`, que solo existe en este proceso. Con la rejilla repartida entre procesos
    # hijos, cada uno importaría el módulo de verdad y ejecutaría el backtest completo.
    winner = run_portfolio_study(
        {"target_size": 8, "commission_bps": 5.0}, tmp_path / "evidence", output,
        definition=definition, workers=1,
    )

    assert winner["winner_combination"]["target_size"] == 12
    assert winner["selection_metric"] == "information_ratio"
    # La configuración devuelta mantiene lo que no es de cartera y sobrescribe lo que sí.
    assert winner["configuration"]["commission_bps"] == 5.0
    assert winner["configuration"]["target_size"] == 12

    # Queda exactamente una carpeta de evidencia y es la del ganador, no la de la última evaluada.
    assert (output / "evidence_best" / "marca.txt").read_text(encoding="utf-8") == "12"
    assert not (output / "_staging").exists()

    # Las descartadas sobreviven solo como filas de resumen.
    grid = pd.read_parquet(output / "portfolio_grid.parquet")
    assert len(grid) == 3
    assert set(grid["information_ratio"]) == {0.10, 0.90, 0.50}

    payload = json.loads((output / "portfolio_winner.json").read_text(encoding="utf-8"))
    assert payload["combinations"] == 3
    assert payload["winner_summary"]["information_ratio"] == pytest.approx(0.90)
    # La era reservada se persiste aparte y nunca entra en el resumen que decide.
    assert "confirmation" not in payload["winner_summary"]
    assert payload["winner_confirmation"]["information_ratio"] == pytest.approx(-1.0)


def test_grid_backtest_cannot_see_the_reserved_era(tmp_path: Path) -> None:
    """La rejilla simula hasta 2024: el resultado de 2025-2026 no llega a existir.

    Es una garantía más fuerte que filtrar el resumen al elegir. Si la simulación entrara en la era
    reservada, su resultado estaría calculado y bastaría con mirarlo para elegir por él; recortando
    los scores, no hay nada que mirar.
    """
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    frame = pd.DataFrame({
        "ticker": ["A"] * 4,
        "snapshot_date": ["2023-06-30", "2024-06-28", "2025-06-30", "2026-06-30"],
        "meta_rank": [0.9, 0.8, 0.7, 0.6],
    })
    frame.to_parquet(evidence / "agent_scores.parquet", index=False)
    (evidence / "dataset_reference.json").write_text("{}", encoding="utf-8")

    workspace = selection_evidence(evidence, tmp_path / "recortada")
    truncated = pd.read_parquet(workspace / "agent_scores.parquet")
    years = pd.to_datetime(truncated["snapshot_date"]).dt.year

    assert int(years.max()) <= SELECTION_UNTIL_YEAR
    assert not (years >= 2025).any()
    # El original no se toca: la era reservada sigue disponible para confirmar al ganador.
    assert len(pd.read_parquet(evidence / "agent_scores.parquet")) == 4
    # Los ficheros auxiliares que el backtest necesita viajan con la copia.
    assert (workspace / "dataset_reference.json").exists()


def test_resume_skips_already_evaluated_combinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reanudar no repite trabajo: la rejilla no tiene runs individuales que reutilizar.

    Sin esto, pausar el estudio tiraría horas de cómputo, porque `run_portfolio_study` recorrería
    la rejilla entera desde el principio.
    """
    evaluated: list[int] = []

    def fake_evaluation(
        values: dict[str, Any], profile: str, evidence_dir: Path, retain_dir: Path | None = None,
        *, include_model_artifacts: bool = False,
    ) -> dict[str, Any]:
        size = int(values["target_size"])
        # Solo cuentan las evaluaciones de la rejilla, que van contra la evidencia recortada; la
        # reevaluación final del ganador usa la completa y siempre ocurre, se reanude o no.
        if evidence_dir.name == "_selection_evidence":
            evaluated.append(size)
        if retain_dir is not None:
            retain_dir.mkdir(parents=True, exist_ok=True)
            (retain_dir / "marca.txt").write_text(str(size), encoding="utf-8")
        return {
            "summary": {"information_ratio": {8: 0.10, 12: 0.90, 16: 0.50}[size], "confirmation": {}},
            "rank_ic": {},
        }

    monkeypatch.setattr("module.studies.runner.run_profile_evaluation", fake_evaluation)
    # Identidad, pero conservando el nombre real de la carpeta: el test distingue la evidencia
    # recortada de la completa por él.
    monkeypatch.setattr(
        "module.studies.portfolio_study.selection_evidence",
        lambda evidence_dir, workspace: (workspace.mkdir(parents=True, exist_ok=True), workspace)[1],
    )
    monkeypatch.setattr(
        "module.studies.portfolio_study._profiles_with_winner",
        lambda values, evidence_dir, selection_dir, output_dir, profiles=None: [],
    )

    definition = {
        "target_size": {"values": [8, 12, 16]},
        "max_cash_weight": {"values": [0.0]},
        "sizing_mode": {"values": ["equal"]},
        "minimum_holding_period": {"values": ["none"]},
        "coverage_percentile_floor": {"values": [0.0]},
        "rebalance_drift_tolerance": {"values": [0.0]},
    }
    output = tmp_path / "estudio"
    run_portfolio_study(
        {"target_size": 8}, tmp_path / "evidence", output, definition=definition, workers=1,
    )
    first_pass = list(evaluated)
    assert sorted(first_pass) == [8, 12, 16]

    # Segunda pasada sobre el mismo directorio: no debe reevaluar ninguna combinación de la rejilla.
    evaluated.clear()
    winner = run_portfolio_study(
        {"target_size": 8}, tmp_path / "evidence", output, definition=definition, workers=1,
    )
    assert evaluated == [], "la reanudación repitió combinaciones ya evaluadas"
    # Y el ganador reconstruido desde el artefacto sigue siendo el correcto.
    assert winner["winner_combination"]["target_size"] == 12
    assert len(pd.read_parquet(output / "portfolio_grid.parquet")) == 3


def test_combination_key_is_order_independent() -> None:
    """La identidad de una combinación no puede depender del orden de inserción del dict."""
    variables = list(PORTFOLIO_STUDY_VARIABLES)
    forward = {variable: index for index, variable in enumerate(variables)}
    backward = {variable: forward[variable] for variable in reversed(variables)}
    assert combination_key(forward) == combination_key(backward)


def test_profiles_are_evaluated_with_the_winning_portfolio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Los ocho perfiles se miden con la cartera ganadora, no con la del modelo.

    Y se miden dos veces: sobre la serie recortada para la ventana de selección y sobre la completa
    para la era reservada, igual que el ganador.
    """
    from module.evaluation.profiles import PROFILE_NAMES

    seen: list[tuple[str, str]] = []

    def fake_evaluation(
        values: dict[str, Any], profile: str, evidence_dir: Path, retain_dir: Path | None = None,
        *, include_model_artifacts: bool = False,
    ) -> dict[str, Any]:
        seen.append((profile, evidence_dir.name))
        assert int(values["target_size"]) == 25, "el perfil no usó la cartera ganadora"
        return {
            "summary": {"information_ratio": 0.3, "confirmation": {"information_ratio": -0.2}},
            "rank_ic": {},
        }

    monkeypatch.setattr("module.studies.runner.run_profile_evaluation", fake_evaluation)
    output = tmp_path / "salida"
    output.mkdir()
    rows = _profiles_with_winner(
        {"target_size": 25}, tmp_path / "completa", tmp_path / "recortada", output,
    )

    assert [row["profile"] for row in rows] == list(PROFILE_NAMES)
    # Cada perfil se evalúa contra las dos series.
    for profile in PROFILE_NAMES:
        assert (profile, "recortada") in seen
        assert (profile, "completa") in seen
    assert all(row["confirmation_information_ratio"] == pytest.approx(-0.2) for row in rows)
    assert (output / "portfolio_profiles.parquet").exists()


def test_only_the_selected_profiles_are_evaluated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Elegir un subconjunto de perfiles debe evaluar ese subconjunto y ninguno más."""
    seen: list[str] = []

    def fake_evaluation(
        values: dict[str, Any], profile: str, evidence_dir: Path, retain_dir: Path | None = None,
        *, include_model_artifacts: bool = False,
    ) -> dict[str, Any]:
        seen.append(profile)
        return {"summary": {"information_ratio": 0.2, "confirmation": {}}, "rank_ic": {}}

    monkeypatch.setattr("module.studies.runner.run_profile_evaluation", fake_evaluation)
    output = tmp_path / "salida"
    output.mkdir()
    rows = _profiles_with_winner(
        {"target_size": 8}, tmp_path / "completa", tmp_path / "recortada", output,
        ["value", "defensive"],
    )
    assert [row["profile"] for row in rows] == ["value", "defensive"]
    assert set(seen) == {"value", "defensive"}

    # Una lista vacía es una renuncia deliberada al diagnóstico, no "todos".
    seen.clear()
    assert _profiles_with_winner(
        {"target_size": 8}, tmp_path / "completa", tmp_path / "recortada", output, [],
    ) == []
    assert seen == []


def test_improvement_reports_deltas_against_the_starting_portfolio() -> None:
    before = {"information_ratio": 0.20, "annualized_turnover": 4.0}
    after = {"information_ratio": 0.35, "annualized_turnover": 3.0}
    delta = improvement(before, after)
    assert delta["information_ratio"]["delta"] == pytest.approx(0.15)
    assert delta["annualized_turnover"]["delta"] == pytest.approx(-1.0)


def test_portfolio_evidence_sources_resolve_to_their_directories(tmp_path: Path) -> None:
    """El panel abre la cartera ganadora y cada perfil como si fueran runs normales.

    Sin estas dos fuentes el Portfolio Study solo enseñaría cifras agregadas, y la evidencia de
    cartera que sí conserva —posiciones, órdenes, efectivo— quedaría inaccesible desde el panel.
    """
    from module.web.queries import _evidence_dir

    study = tmp_path / "study-x"
    (study / "evidence_best_full").mkdir(parents=True)
    (study / "profiles" / "value").mkdir(parents=True)

    assert _evidence_dir(study, "study-x", "portfolio-winner") == study / "evidence_best_full"
    assert _evidence_dir(study, "study-x", "portfolio-profile:value") == (study / "profiles" / "value").resolve()

    # Un perfil sin evidencia se declara ausente en vez de devolver una ruta que no existe.
    with pytest.raises(FileNotFoundError):
        _evidence_dir(study, "study-x", "portfolio-profile:growth")

    # La ruta se confina bajo `profiles/`: un nombre con salto de directorio no puede escapar.
    with pytest.raises(ValueError):
        _evidence_dir(study, "study-x", "portfolio-profile:../../etc")


def test_the_winner_keeps_a_complete_evidence_directory_but_the_grid_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """El ganador de cartera se explora como un run del Model Study; la rejilla no paga por ello.

    El Portfolio Study no reentrena nada, así que los artefactos de modelo de su ganador **son** los
    del Model Study de origen y se enlazan en vez de recalcularse. Pero enlazarlos en cada una de las
    cientos de combinaciones desechables sería ruido de E/S sin beneficio, así que la rejilla debe
    seguir pidiendo solo evidencia de cartera.
    """
    linked: list[bool] = []

    def fake_evaluation(
        values: dict[str, Any], profile: str, evidence_dir: Path, retain_dir: Path | None = None,
        *, include_model_artifacts: bool = False,
    ) -> dict[str, Any]:
        if evidence_dir.name == "_selection_evidence":
            linked.append(include_model_artifacts)
        if retain_dir is not None:
            retain_dir.mkdir(parents=True, exist_ok=True)
            (retain_dir / "modelo.txt").write_text(str(include_model_artifacts), encoding="utf-8")
        return {"summary": {"information_ratio": float(values["target_size"]), "confirmation": {}},
                "rank_ic": {}}

    monkeypatch.setattr("module.studies.runner.run_profile_evaluation", fake_evaluation)
    monkeypatch.setattr(
        "module.studies.portfolio_study.selection_evidence",
        lambda evidence_dir, workspace: (workspace.mkdir(parents=True, exist_ok=True), workspace)[1],
    )
    monkeypatch.setattr(
        "module.studies.portfolio_study._profiles_with_winner",
        lambda values, evidence_dir, selection_dir, output_dir, profiles=None: [],
    )

    output = tmp_path / "portfolio_study"
    run_portfolio_study(
        {"target_size": 8}, tmp_path / "evidence", output,
        definition=_definition([8, 12]), workers=1,
    )

    assert linked and not any(linked), "la rejilla no debe enlazar artefactos de modelo"
    assert (output / "evidence_best_full" / "modelo.txt").read_text(encoding="utf-8") == "True"


def test_the_three_diagnostics_are_written_without_touching_any_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Los diagnósticos se calculan solos al elegir ganador y no tocan nada que decida.

    Aquí fallan a propósito —la evidencia sintética no tiene con qué calcularlos—, y ese es
    justamente el contrato que importa: un diagnóstico roto deja constancia en su artefacto y **no**
    tumba un estudio cuya rejilla ya ha costado horas.
    """
    def fake_evaluation(
        values: dict[str, Any], profile: str, evidence_dir: Path, retain_dir: Path | None = None,
        *, include_model_artifacts: bool = False,
    ) -> dict[str, Any]:
        if retain_dir is not None:
            retain_dir.mkdir(parents=True, exist_ok=True)
        return {"summary": {"information_ratio": float(values["target_size"]), "confirmation": {}},
                "rank_ic": {}}

    monkeypatch.setattr("module.studies.runner.run_profile_evaluation", fake_evaluation)
    monkeypatch.setattr(
        "module.studies.portfolio_study.selection_evidence",
        lambda evidence_dir, workspace: (workspace.mkdir(parents=True, exist_ok=True), workspace)[1],
    )
    monkeypatch.setattr(
        "module.studies.portfolio_study._profiles_with_winner",
        lambda values, evidence_dir, selection_dir, output_dir, profiles=None: [],
    )

    output = tmp_path / "portfolio_study"
    output.mkdir()
    for name in ("winner.json", "decisions.json"):
        (output / name).write_text('{"intacto": true}', encoding="utf-8")

    winner = run_portfolio_study(
        {"target_size": 8}, tmp_path / "evidence", output, definition=_definition([8, 12]),
        workers=1,
    )

    assert set(winner["diagnostics"]) == {"cost_sensitivity", "capacity", "portfolio_narrative"}
    for name, state in winner["diagnostics"].items():
        assert state["available"] is False, "sin panel real no hay diagnóstico que calcular"
        payload = json.loads((output / f"{name}.json").read_text(encoding="utf-8"))
        assert payload["available"] is False and payload["error"]
    # Y nada de lo que decide ha sido tocado.
    for name in ("winner.json", "decisions.json"):
        assert json.loads((output / name).read_text(encoding="utf-8")) == {"intacto": True}


def _definition(sizes: list[int]) -> dict[str, dict[str, Any]]:
    return {
        "target_size": {"values": sizes},
        "max_cash_weight": {"values": [0.0]},
        "sizing_mode": {"values": ["equal"]},
        "minimum_holding_period": {"values": ["none"]},
        "coverage_percentile_floor": {"values": [0.0]},
        "rebalance_drift_tolerance": {"values": [0.0]},
    }


def test_robustness_and_attribution_are_inherited_from_the_source_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un Portfolio Study se explora igual que un Model Study, sin inventarse evidencia propia.

    Robustez y atribución son las del modelo cuyos scores reutiliza —no reentrena nada—, así que
    servirlas desde el study de origen es honesto. Lo que no sería honesto es servirlas sin decir de
    dónde vienen, ni pisar el artefacto propio de un estudio que sí lo tenga.

    El origen se resuelve con `safe_study_path`, así que la herencia queda confinada a la raíz de
    studies: un identificador que apunte fuera no puede sacar ficheros de ningún otro sitio.
    """
    import module.storage.studies as storage
    from module.web.queries import _inherited

    monkeypatch.setattr(storage, "STUDIES_ROOT", tmp_path)
    portfolio, origin = tmp_path / "study-cartera", tmp_path / "study-origen"
    portfolio.mkdir()
    origin.mkdir()
    (portfolio / "portfolio_winner.json").write_text(
        json.dumps({"source_study_id": origin.name}), encoding="utf-8",
    )
    (origin / "robustness.json").write_text(json.dumps({"permutation": {"p_value": 0.0001}}), encoding="utf-8")

    payload = _inherited(portfolio, "robustness")
    assert payload["inherited_from_study_id"] == origin.name
    assert payload["permutation"]["p_value"] == pytest.approx(0.0001)

    # Con artefacto propio no se hereda nada: el suyo manda.
    (portfolio / "robustness.json").write_text(json.dumps({"propio": True}), encoding="utf-8")
    assert _inherited(portfolio, "robustness") is None
    # Y una vista que no se hereda nunca entra por este camino.
    assert _inherited(portfolio, "learning") is None


def test_a_study_without_an_origin_inherits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un Model Study no puede recibir la robustez de otro estudio por esta vía."""
    import module.storage.studies as storage
    from module.web.queries import _inherited

    monkeypatch.setattr(storage, "STUDIES_ROOT", tmp_path)
    study = tmp_path / "study-modelo"
    study.mkdir()
    assert _inherited(study, "robustness") is None
