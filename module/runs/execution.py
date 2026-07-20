"""Ejecucion controlada de runs y studies para CLI y consola local."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from environment import Settings, ensure_directories
from module.data.dataset import build_point_in_time_dataset
from module.data.ingest.pipeline import download_raw_data
from module.modeling.agents import build_agent_scores
from module.modeling.features import build_features
from module.evaluation.backtest import run_backtest_from_run_dir
from module.evaluation.profiles import PROFILE_NAMES
from module.evaluation.robustness import leave_one_year_out
from module.evaluation.stats import block_bootstrap_ci
from module.runs.experiments import split_variables
from module.runs.results_store import ResultsStore, execution_hash
from escenarios.variables import STUDY_OPTIONS
from module.runs.recycle import cache_dir, publish as publish_recycle, restore as restore_recycle, stage_key
from module.common.utils import setup_logging, write_json, write_parquet

log = logging.getLogger(__name__)

STAGE_HANDLERS = {
    "download": download_raw_data,
    "dataset": build_point_in_time_dataset,
    "features": build_features,
    "agents": build_agent_scores,
    "backtest": run_backtest_from_run_dir,
}
# Las descargas son explícitas: una optimización reutiliza datos crudos ya obtenidos y no debe
# disparar llamadas de red por cada escenario.
STAGE_ORDER = ("dataset", "features", "agents", "backtest")


def stages_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "full":
        return STAGE_ORDER
    if mode not in STAGE_HANDLERS:
        raise ValueError(f"Modo no disponible en la consola: {mode!r}.")
    return (mode,)


def execute_run(
    settings: Settings,
    *,
    mode: str,
    run_kind: str = "experimental",
    label: str = "Run experimental",
    description: str = "",
    tags: Iterable[str] = (),
    study_id: str | None = None,
    grid_definition: Mapping[str, Any] | None = None,
    store: ResultsStore | None = None,
    force_rerun: bool = False,
    agent_dir: Path | None = None,
) -> str:
    """Ejecuta etapas existentes y publica una copia inmutable de sus artefactos.

    ``agent_dir`` fija el run de agentes exacto sobre el que backtestear (modo ``backtest``); sin
    él se usa el más reciente. Lo usan la fase de perfiles y la de cartera para re-backtestear
    sobre el finalista concreto sin reentrenar.
    """
    store = store or ResultsStore()
    stages = stages_for_mode(mode)
    effective = replace(settings, run_mode=mode)
    ensure_directories(effective)
    input_paths = _run_input_paths(effective)
    from module.common.utils import sha256_file
    fingerprints = {name: sha256_file(path) for name, path in input_paths.items() if path.exists()}
    reusable_digest = execution_hash(effective, run_kind=run_kind, mode=mode, stages=stages, inputs=fingerprints)
    if not force_rerun:
        existing = store.find_completed_execution(reusable_digest)
        if existing:
            if study_id:
                store.add_to_study(study_id, existing, reused=True)
            log.info("Run reutilizado: %s", existing)
            return existing
    run_id, run_dir, _ = store.create_run(
        effective, run_kind=run_kind, mode=mode, stages=stages, label=label,
        description=description, tags=tags, study_id=study_id, grid_definition=grid_definition,
        input_paths=input_paths,
    )
    setup_logging(run_dir / "logs" / "execution.log")
    store.set_status(run_dir, "running")
    # El modo backtest aislado conserva el comportamiento CLI; en un pipeline completo esta
    # referencia se reemplaza exactamente por la salida de la etapa agents del propio run. Si el
    # llamante fija un agent_dir explícito (fase de cartera/perfiles sobre el finalista), se usa ese.
    if mode == "backtest":
        agent_dir = agent_dir or _latest_agent_run(effective.processed_output_dir)
    try:
        for stage in stages:
            log.info("Run %s: iniciando %s", run_id, stage)
            output = _run_cached_stage(stage, effective, agent_dir=agent_dir)
            if stage == "agents":
                agent_dir = output
        source = agent_dir or (_latest_agent_run(effective.processed_output_dir) if mode == "backtest" else None)
        summary: dict[str, Any] = {}
        if source is not None:
            summary = store.publish_artifacts(run_dir, source)
            _write_csv_exports(run_dir / "artifacts")
        store.complete(run_dir, summary)
        return run_id
    except Exception as exc:
        log.exception("Run %s fallo", run_id)
        store.fail(run_dir, exc)
        raise


def execute_study(
    settings: Settings,
    *,
    study_payload: Mapping[str, Any],
    variables: Mapping[str, list[Any]],
    mode: str = "full",
    store: ResultsStore | None = None,
) -> str:
    """Study completo: mismo ciclo que la optimización oficial, limitado a las variables marcadas.

    Las variables del usuario se reparten en ejes de modelo (barridos en Fase 1/2 por rank-IC) y
    ejes de cartera (optimizados al final por re-backtest). ``mode`` se ignora: el ciclo siempre
    entrena y backtestea. La robustez completa (placebo por permutación + carteras aleatorias) solo
    se ejecuta si el usuario la marca (``study.include_robustness``); si no, solo bootstrap + LOYO.
    """
    model_vars, portfolio_vars = split_variables(variables)
    include_robustness = bool(study_payload.get("include_robustness", False))
    payload = {
        "name": str(study_payload.get("name", "study")), "kind": str(study_payload.get("kind", "exploratory")),
        "description": str(study_payload.get("description", "")), "tags": list(study_payload.get("tags", [])),
        "variables": dict(variables), "strategy": "unified_full_cycle",
        "selection_metric": "rank_ic_oos", "selection_until_year": 2024, "reserved_years": [2025, 2026],
    }
    return run_optimization(settings, model_vars=model_vars, portfolio_vars=portfolio_vars,
                            payload=payload, store=store, include_full_robustness=include_robustness)


def execute_official_optimization(settings: Settings, store: ResultsStore | None = None) -> str:
    """Optimización oficial: ciclo completo barriendo TODAS las variables barribles.

    Barrido derivado de ``STUDY_OPTIONS`` (todos los ejes de modelo en Fase 1/2 y todos los de
    cartera al final). Reemplaza al catálogo fijo reducido de ``escenarios/fase1_ejes.py``.
    """
    variables = {axis: list(values) for axis, values in STUDY_OPTIONS.items()}
    model_vars, portfolio_vars = split_variables(variables)
    payload = {
        "name": "optimization-official", "kind": "optimization",
        "description": "Optimización oficial: barrido completo de modelo (Fase 1/2), afinado, cartera y validación reservada.",
        "tags": ["official", "optimization"], "strategy": "unified_full_cycle",
        "selection_metric": "rank_ic_oos", "selection_until_year": 2024, "reserved_years": [2025, 2026],
    }
    # La optimización oficial SIEMPRE ejecuta la robustez completa (placebo + carteras aleatorias):
    # es el estudio que sostiene la credibilidad del TFM.
    return run_optimization(settings, model_vars=model_vars, portfolio_vars=portfolio_vars,
                            payload=payload, store=store, include_full_robustness=True)


def execute_official_phase1(settings: Settings, store: ResultsStore | None = None) -> str:
    """Alias de compatibilidad: la acción oficial ejecuta el ciclo completo unificado."""
    return execute_official_optimization(settings, store)


def run_optimization(
    settings: Settings,
    *,
    model_vars: Mapping[str, list[Any]],
    portfolio_vars: Mapping[str, list[Any]],
    payload: Mapping[str, Any],
    store: ResultsStore | None = None,
    include_full_robustness: bool = False,
) -> str:
    """Ciclo completo unificado que comparten study y optimización oficial.

    Fase 1 (ejes de modelo aislados) → Fase 2 (greedy top-2, sin cartesiano) → afinado de
    hiperparámetros → run final → fase de cartera (re-backtest por criterio económico) → 8
    perfiles → robustez → validación reservada. La selección de modelo es siempre por rank-IC OOS;
    los ejes de cartera se eligen por métrica económica porque no alteran el aprendizaje.
    """
    store = store or ResultsStore()
    study_id, study_dir = store.create_study(dict(payload))
    name = str(payload.get("name", "study"))
    description = str(payload.get("description", ""))
    tags = list(payload.get("tags", []))

    # --- Fase 1: cada eje de modelo aislado sobre el baseline ---
    # Un escenario que no entrena (p.ej. ventana muy corta o cadencia que deja pocas filas) se
    # salta y se registra, sin abortar el estudio: con un barrido amplio es esperable que algunas
    # combinaciones extremas no sean viables.
    phase1_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for spec in _isolated_specs(settings, model_vars):
        run_id = _safe_scenario_run(
            store, replace(settings, **spec["overrides"]), study_id=study_id,
            label=f"{name} · Fase 1 · {spec['name']}", description=description, tags=tags,
            grid_definition={"phase": "1", **spec}, skipped=skipped, scenario=spec["name"])
        if run_id is None:
            continue
        phase1_rows.append({"phase": "1", "scenario": spec["name"], "axis": spec.get("axis"),
                            "overrides": spec["overrides"], "run_id": run_id,
                            **spec["overrides"], **_summary_for_run(store, run_id)})
    phase1 = pd.DataFrame(phase1_rows)
    if phase1.empty:
        raise RuntimeError(
            "Ningún escenario de Fase 1 fue viable (todos fallaron al entrenar). "
            f"Escenarios omitidos: {len(skipped)}. Revisa los valores barridos en STUDY_OPTIONS.")

    # --- Fase 2: greedy incremental con top-2 por eje (sin producto cartesiano) ---
    phase2_rows, model_overrides, phase1_decision = _greedy_phase2(
        phase1, model_vars, settings, store, study_id, name=name, description=description, tags=tags)
    phase2 = pd.DataFrame(phase2_rows)

    # --- Fase 3: afinado de hiperparámetros sobre el ganador de modelo ---
    hyper_specs = _hyperparameter_specs(model_overrides)
    hyper_rows = _execute_official_specs(hyper_specs, settings, store, study_id, phase="3",
                                         label_prefix=name, description=description, tags=tags)
    hyper = pd.DataFrame(hyper_rows)
    chosen_hyper = _stable_best(hyper)
    final_model_overrides = dict(chosen_hyper.get("overrides", {})) if chosen_hyper else dict(model_overrides)

    # --- Run final del modelo ganador ---
    final_settings = replace(settings, **final_model_overrides)
    final_id = execute_run(final_settings, mode="full", run_kind="optimization_final", study_id=study_id,
                           label=f"{name} · finalista", description=description,
                           tags=[*tags, "final"], grid_definition={"phase": "final", "overrides": final_model_overrides}, store=store)
    store.add_to_study(study_id, final_id)
    final_agent_dir = _latest_agent_run(final_settings.processed_output_dir)

    # --- Fase de cartera: greedy por todos los ejes de cartera, por Information Ratio ---
    portfolio_overrides, portfolio_trace, portfolio_rows = _portfolio_phase(
        final_settings, portfolio_vars, store, study_id, agent_dir=final_agent_dir,
        name=name, description=description, tags=tags)
    portfolio_settings = replace(final_settings, **portfolio_overrides)
    if portfolio_overrides:
        portfolio_final_id = execute_run(
            portfolio_settings, mode="backtest", run_kind="optimization_final", study_id=study_id,
            label=f"{name} · finalista + cartera", description=description, tags=[*tags, "final", "portfolio"],
            grid_definition={"phase": "4_cartera", "overrides": portfolio_overrides},
            store=store, agent_dir=final_agent_dir)
        store.add_to_study(study_id, portfolio_final_id)
    else:
        portfolio_final_id = final_id

    # --- Fase final = 8 perfiles de inversor sobre la cartera óptima. Es la SALIDA del study:
    #     un run por perfil, todos sobre el modelo y la cartera ya optimizados. ---
    profile_run_ids: dict[str, str] = {}
    profile_rows: list[dict[str, Any]] = []
    for profile in PROFILE_NAMES:
        profile_id = execute_run(
            replace(portfolio_settings, profile=profile), mode="backtest", run_kind="scenario", study_id=study_id,
            label=f"{name} · perfil {profile}",
            description="Perfil de inversor sobre el modelo y la cartera optimizados (resultado final).",
            tags=[*tags, "final", "profile"],
            grid_definition={"phase": "5_perfiles", "profile": profile, "parent_run_id": portfolio_final_id},
            store=store, agent_dir=final_agent_dir)
        store.add_to_study(study_id, profile_id)
        profile_run_ids[profile] = profile_id
        profile_rows.append({"phase": "5_perfiles", "scenario": profile, "axis": "profile",
                             "overrides": {"profile": profile}, "run_id": profile_id,
                             **_summary_for_run(store, profile_id)})

    # --- Robustez, validación reservada y decisión final ---
    final_summary = _summary_for_run(store, portfolio_final_id)
    reserved = _reserved_validation(store, final_id)
    robustness = _final_robustness(
        store, final_id, portfolio_final_id=portfolio_final_id, final_settings=final_settings,
        include_full=include_full_robustness)
    best_config = _best_config_summary(final_model_overrides, portfolio_overrides,
                                       profile_run_ids, store)
    # Perfil recomendado por Information Ratio (rentabilidad ajustada al riesgo), no por CAGR puro.
    recommended = _recommended_profile(profile_run_ids, store)
    # Comparativa completa: todas las fases (modelo 1/2/3 + cartera + perfiles) para la app.
    comparison = pd.concat(
        [phase1, phase2, hyper, pd.DataFrame(portfolio_rows), pd.DataFrame(profile_rows)],
        ignore_index=True, sort=False)
    write_parquet(comparison, study_dir / "comparison_data.parquet")
    write_json({
        "strategy": "unified_full_cycle", "selection_metric": "rank_ic_oos",
        "phase1": phase1_decision, "model_winner": {"overrides": final_model_overrides,
                                                    "summary": _summary_for_run(store, final_id)},
        "phase_portfolio": portfolio_trace, "portfolio_overrides": portfolio_overrides,
        "final_run_id": portfolio_final_id, "model_final_run_id": final_id,
        "final_settings_overrides": {**final_model_overrides, **portfolio_overrides},
        "final_summary": final_summary,
        # Salida final del study: un run por perfil sobre el modelo+cartera óptimos.
        "final_profile_run_ids": profile_run_ids, "profile_run_ids": profile_run_ids,
        "recommended_profile": recommended,
        "reserved_validation": reserved, "robustness": robustness, "best_config": best_config,
        "skipped_scenarios": skipped,
    }, study_dir / "decision.json")
    manifest = json.loads((study_dir / "study_manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "succeeded"
    write_json(manifest, study_dir / "study_manifest.json")
    return study_id


def _isolated_specs(settings: Settings, variables: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    """Baseline + una modificación cada vez para atribuir su efecto a un eje."""
    specs = [{"name": "baseline", "axis": None, "overrides": {}}]
    for axis, values in variables.items():
        baseline = getattr(settings, axis)
        for value in values:
            if value == baseline:
                continue
            safe_name = str(value).replace(" ", "_").replace(".", "_")
            specs.append({"name": f"{axis}_{safe_name}", "axis": axis, "overrides": {axis: value}})
    return specs


def _greedy_phase2(
    phase1: pd.DataFrame, model_vars: Mapping[str, list[Any]], settings: Settings,
    store: ResultsStore, study_id: str, *, name: str, description: str, tags: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Fase 2 greedy con top-2 por eje, sin producto cartesiano.

    Parte del mejor nivel de cada eje (combinado). Recorre los ejes ordenados por su impacto en
    Fase 1 y en cada uno prueba su 1º y 2º mejor sobre la combinación en curso, fijando el que sube
    el rank-IC. Explora el 2º mejor de cada eje con ~2·N runs, no 2^N.
    """
    baseline_row = phase1.loc[phase1["scenario"] == "baseline"]
    baseline_ic = float(baseline_row["mean_rank_ic"].iloc[0]) if not baseline_row.empty and "mean_rank_ic" in baseline_row else float("-inf")

    # Mejores dos niveles de cada eje y ranking de ejes por impacto (mejora sobre baseline).
    axis_options: dict[str, list[dict[str, Any]]] = {}
    axis_impact: dict[str, float] = {}
    decision: dict[str, Any] = {"axes": {}}
    for axis in model_vars:
        candidates = phase1.loc[phase1["axis"] == axis]
        best = _stable_best(candidates)
        if not best:
            continue
        second = _second_stable(candidates.to_dict("records"), str(best["scenario"]))
        axis_options[axis] = [best] + ([second] if second else [])
        axis_impact[axis] = float(best.get("mean_rank_ic", float("-inf"))) - baseline_ic
        decision["axes"][axis] = {"chosen": best.get("scenario"), "overrides": dict(best.get("overrides") or {}),
                                   "second": second}

    # Combinación inicial: el mejor nivel de cada eje.
    selected: dict[str, Any] = {}
    for options in axis_options.values():
        selected.update(dict(options[0].get("overrides") or {}))

    rows: list[dict[str, Any]] = []
    best_ic = float("-inf")

    skipped: list[dict[str, Any]] = []

    def evaluate(overrides: dict[str, Any], label: str) -> float:
        nonlocal rows
        run_id = _safe_scenario_run(
            store, replace(settings, **overrides), study_id=study_id,
            label=f"{name} · Fase 2 · {label}", description=description, tags=[*tags, "phase2"],
            grid_definition={"phase": "2", "overrides": overrides}, skipped=skipped, scenario=label)
        if run_id is None:
            return float("-inf")
        summary = _summary_for_run(store, run_id)
        rows.append({"phase": "2", "scenario": label, "overrides": dict(overrides), **overrides, **summary, "run_id": run_id})
        return float(summary.get("mean_rank_ic", float("-inf")))

    best_ic = evaluate(selected, "combined_best")

    # Greedy: para cada eje (más impactante primero), probar su 2º mejor sobre la combinación.
    for axis in sorted(axis_impact, key=axis_impact.get, reverse=True):
        options = axis_options[axis]
        if len(options) < 2:
            continue
        trial = dict(selected)
        for key in dict(options[0].get("overrides") or {}):
            trial.pop(key, None)
        trial.update(dict(options[1].get("overrides") or {}))
        ic = evaluate(trial, f"{axis}_second")
        if ic > best_ic:
            best_ic, selected = ic, trial

    return rows, selected, decision


def _portfolio_phase(
    final_settings: Settings, portfolio_vars: Mapping[str, list[Any]], store: ResultsStore,
    study_id: str, *, agent_dir: Path | None, name: str, description: str, tags: list[str],
    criterion: str = "information_ratio",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Optimiza los ejes de cartera sobre el finalista, re-backtesteando sin reentrenar.

    Greedy por eje: para cada variable de cartera prueba sus valores (re-backtest sobre el mismo
    ``agent_dir``) y fija el que maximiza ``criterion`` (métrica económica del backtest_summary,
    p.ej. information_ratio). El rank-IC no discrimina aquí porque estos ejes no cambian el
    aprendizaje. El perfil de inversor no se barre: se reporta con los 8 perfiles al final.
    """
    selected: dict[str, Any] = {}
    trace: dict[str, Any] = {"criterion": criterion, "axes": {}}
    comparison_rows: list[dict[str, Any]] = []
    for axis, values in portfolio_vars.items():
        if axis == "profile":  # el perfil se cubre con los 8 runs de perfil, no se barre aquí
            continue
        current = selected.get(axis, getattr(final_settings, axis))
        best_value, best_score, best_run = current, float("-inf"), None
        axis_results: list[dict[str, Any]] = []
        for value in values:
            trial = {**selected, axis: value}
            # Algunas combinaciones violan las restricciones de Settings (p.ej.
            # max_weight_per_position * target_min < 1); se omiten sin abortar la fase.
            try:
                candidate_settings = replace(final_settings, **trial)
            except ValueError as exc:
                axis_results.append({"value": value, "skipped": str(exc)})
                continue
            run_id = execute_run(
                candidate_settings, mode="backtest", run_kind="scenario", study_id=study_id,
                label=f"{name} · Cartera · {axis}={value}", description=description,
                tags=[*tags, "portfolio"], grid_definition={"phase": "4_cartera", "axis": axis, "value": value},
                store=store, agent_dir=agent_dir)
            store.add_to_study(study_id, run_id)
            summary = _summary_for_run(store, run_id)
            score = _economic_score(summary, criterion)
            axis_results.append({"value": value, criterion: score, "run_id": run_id})
            comparison_rows.append({"phase": "4_cartera", "scenario": f"{axis}={value}", "axis": axis,
                                    "overrides": {axis: value}, "run_id": run_id, **summary})
            if score > best_score:
                best_value, best_score, best_run = value, score, run_id
        selected[axis] = best_value
        trace["axes"][axis] = {"chosen": best_value, "score": best_score, "run_id": best_run,
                                "candidates": axis_results}
    return selected, trace, comparison_rows


def _economic_score(summary: Mapping[str, Any], criterion: str) -> float:
    """Métrica económica para elegir un eje de cartera; drawdown se maximiza como negativo."""
    value = summary.get(criterion)
    if value is None:
        return float("-inf")
    return -float(value) if criterion == "max_drawdown" else float(value)


def _safe_scenario_run(
    store: ResultsStore, settings: Settings, *, study_id: str, label: str, description: str,
    tags: list[str], grid_definition: Mapping[str, Any], skipped: list[dict[str, Any]],
    scenario: str, mode: str = "full", agent_dir: Path | None = None,
) -> str | None:
    """Ejecuta un escenario del barrido tolerando fallos: si no entrena, se salta y se registra.

    Devuelve el run_id si tuvo éxito, o None si el escenario no fue viable (p.ej. sin filas de
    entrenamiento suficientes). Así un escenario extremo no aborta el estudio completo.
    """
    try:
        run_id = execute_run(settings, mode=mode, run_kind="scenario", study_id=study_id,
                             label=label, description=description, tags=tags,
                             grid_definition=grid_definition, store=store, agent_dir=agent_dir)
        store.add_to_study(study_id, run_id)
        return run_id
    except Exception as exc:  # noqa: BLE001 — el barrido debe continuar aunque un escenario falle
        log.warning("Escenario omitido (%s): %s", scenario, exc)
        skipped.append({"scenario": scenario, "error": str(exc), "overrides": dict(grid_definition.get("overrides", {}))})
        return None


def _recommended_profile(profile_run_ids: Mapping[str, str], store: ResultsStore) -> dict[str, Any]:
    """Perfil recomendado entre los 8 finales, por Information Ratio (rentabilidad ajustada al
    riesgo). Se prefiere al CAGR puro porque no premia asumir más riesgo por más rentabilidad."""
    best_profile, best_ir, best_run = None, float("-inf"), None
    for profile, run_id in profile_run_ids.items():
        ir = _summary_for_run(store, run_id).get("information_ratio")
        if ir is not None and float(ir) > best_ir:
            best_profile, best_ir, best_run = profile, float(ir), run_id
    return {"profile": best_profile, "information_ratio": None if best_profile is None else best_ir,
            "run_id": best_run}


def _best_config_summary(
    model_overrides: Mapping[str, Any], portfolio_overrides: Mapping[str, Any],
    profile_run_ids: Mapping[str, str], store: ResultsStore,
) -> dict[str, Any]:
    """Resumen legible: mejor modelo, mejor gestión de cartera y perfil recomendado por Info Ratio."""
    best_profile, best_ir = None, float("-inf")
    for profile, run_id in profile_run_ids.items():
        ir = _summary_for_run(store, run_id).get("information_ratio")
        if ir is not None and float(ir) > best_ir:
            best_profile, best_ir = profile, float(ir)
    return {"model": dict(model_overrides), "portfolio": dict(portfolio_overrides),
            "best_profile": best_profile,
            "best_profile_information_ratio": None if best_profile is None else best_ir}


def _execute_official_specs(
    specs: list[dict[str, Any]], settings: Settings, store: ResultsStore, study_id: str, *, phase: str,
    label_prefix: str = "Optimization", description: str = "Escenario dirigido del catálogo oficial.",
    tags: Iterable[str] = ("official",),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for spec in specs:
        overrides = dict(spec["overrides"])
        run_id = _safe_scenario_run(
            store, replace(settings, **overrides), study_id=study_id,
            label=f"{label_prefix} · Fase {phase} · {spec['name']}", description=description,
            tags=[*tags, f"phase{phase}"], grid_definition={"phase": phase, "overrides": overrides},
            skipped=skipped, scenario=spec["name"])
        if run_id is None:
            continue
        rows.append({"phase": phase, "scenario": spec["name"], "overrides": overrides,
                     **overrides, **_summary_for_run(store, run_id), "run_id": run_id})
    return rows


def _stable_best(frame: pd.DataFrame) -> dict[str, Any] | None:
    if frame.empty or "mean_rank_ic" not in frame:
        return None
    ranked = frame.copy()
    if "rank_ic_positive_fraction" not in ranked:
        ranked["rank_ic_positive_fraction"] = 0.0
    if "rank_ic_std" not in ranked:
        ranked["rank_ic_std"] = float("inf")
    ranked["rank_ic_positive_fraction"] = ranked["rank_ic_positive_fraction"].fillna(0.0)
    ranked["rank_ic_std"] = ranked["rank_ic_std"].fillna(float("inf"))
    ranked = ranked.sort_values(["mean_rank_ic", "rank_ic_positive_fraction", "rank_ic_std"],
                                ascending=[False, False, True])
    return ranked.iloc[0].to_dict()


def _second_stable(candidates: list[dict[str, Any]], winner: str) -> dict[str, Any] | None:
    alternatives = [candidate for candidate in candidates if str(candidate.get("name", candidate.get("scenario"))) != winner]
    best = _stable_best(pd.DataFrame(alternatives))
    if not best:
        return None
    return {"name": best.get("name", best.get("scenario")), "overrides": best.get("overrides", {})}


def _hyperparameter_specs(base: Mapping[str, Any]) -> list[dict[str, Any]]:
    variants = {
        "base": {}, "lr_003": {"lgbm_learning_rate": 0.03}, "lr_010": {"lgbm_learning_rate": 0.10},
        "trees_400": {"lgbm_n_estimators": 400}, "leaf_20": {"lgbm_min_child_samples": 20},
        "leaf_100": {"lgbm_min_child_samples": 100},
    }
    return [{"name": f"hp_{name}", "overrides": {**dict(base), **overrides}}
            for name, overrides in variants.items()]


def _reserved_validation(store: ResultsStore, run_id: str) -> dict[str, Any]:
    path = store.runs_root / run_id / "artifacts" / "rank_ic_diagnostics.parquet"
    if not path.exists():
        return {"reserved_years": [2025, 2026], "n_cohorts": 0, "rank_ic_mean": None}
    frame = pd.read_parquet(path)
    if "agent" in frame:
        frame = frame.loc[frame["agent"] == "meta_final"].copy()
    frame["year"] = pd.to_datetime(frame["prediction_date"]).dt.year
    reserved = frame.loc[frame["year"].isin([2025, 2026])]
    return {"reserved_years": [2025, 2026], "n_cohorts": int(len(reserved)),
            "rank_ic_mean": float(reserved["rank_ic"].mean()) if not reserved.empty else None}


def _final_robustness(
    store: ResultsStore, run_id: str, *, portfolio_final_id: str | None = None,
    final_settings: Settings | None = None, include_full: bool = False,
) -> dict[str, Any]:
    """Diagnósticos sobre el finalista publicado.

    Siempre calcula bootstrap por bloques + leave-one-year-out (baratos, no reentrenan). Con
    ``include_full`` añade la robustez cara que reentrena/simula: el placebo por permutación de
    etiquetas (¿colapsa el rank-IC con retornos barajados? = la señal no es artefacto ni fuga) y
    el test de carteras aleatorias (¿la cartera bate al azar de escoger acciones?).
    """
    path = store.runs_root / run_id / "artifacts" / "rank_ic_diagnostics.parquet"
    if not path.exists():
        return {}
    diagnostics = pd.read_parquet(path)
    meta = diagnostics.loc[diagnostics["agent"] == "meta_final"].copy()
    if meta.empty:
        return {}
    bootstrap = block_bootstrap_ci(meta.sort_values("prediction_date").set_index("prediction_date")["rank_ic"])
    loyo = leave_one_year_out(diagnostics).to_dict("records")

    if include_full and final_settings is not None:
        label_permutation = _label_permutation(final_settings, diagnostics)
        random_portfolio = _random_portfolio(store, portfolio_final_id, final_settings)
    else:
        label_permutation = {"status": "no solicitada (robustez completa desactivada)", "n_permutations": 0}
        random_portfolio = {"status": "no solicitada (robustez completa desactivada)"}

    payload = {"block_bootstrap": bootstrap, "leave_one_year_out": loyo,
               "label_permutation": label_permutation, "random_portfolio": random_portfolio}
    write_json(payload, store.runs_root / run_id / "artifacts" / "robustness.json")
    return payload


def _label_permutation(final_settings: Settings, diagnostics: pd.DataFrame, n_permutations: int = 5) -> dict[str, Any]:
    """Placebo: reentrena el finalista con los retornos futuros BARAJADOS y mide su rank-IC.

    Si el sistema aprende de verdad, el rank-IC del meta_final debe COLAPSAR a ~0 con etiquetas
    aleatorias; si no colapsa, hay fuga o el resultado real es artefacto. Se permutan los targets,
    se reentrena (semilla distinta cada vez), se recoge el rank-IC del meta_final, y se restauran
    los targets originales. Los runs placebo se eliminan para no ensuciar el processed dir.
    """
    from module.evaluation.robustness import label_permutation_test
    import numpy as np

    processed = final_settings.processed_output_dir
    targets_path = processed / "targets_forward_3m.parquet"
    if not targets_path.exists():
        return {"status": "sin targets para permutar", "n_permutations": 0}

    base_targets = pd.read_parquet(targets_path)
    permuted_ic: list[float] = []
    rng = np.random.default_rng(0)
    try:
        for i in range(n_permutations):
            shuffled = base_targets.copy()
            shuffled["forward_excess_return_3m"] = rng.permutation(
                shuffled["forward_excess_return_3m"].to_numpy())
            shuffled.to_parquet(targets_path, index=False)
            try:
                build_agent_scores(replace(final_settings, random_seed=1000 + i))
                perm_run = _latest_agent_run(processed)
                perm_diag = pd.read_parquet(perm_run / "rank_ic_diagnostics.parquet")
                permuted_ic.append(float(
                    perm_diag.loc[perm_diag["agent"] == "meta_final", "rank_ic"].mean()))
                import shutil
                shutil.rmtree(perm_run, ignore_errors=True)  # no dejar runs placebo
            except Exception as exc:  # noqa: BLE001 — una permutación fallida no aborta el placebo
                log.warning("Permutación %s falló: %s", i, exc)
    finally:
        base_targets.to_parquet(targets_path, index=False)  # restaurar siempre los targets reales

    if not permuted_ic:
        return {"status": "ninguna permutación entrenó", "n_permutations": 0}
    return label_permutation_test(diagnostics, permuted_ic)


def _random_portfolio(store: ResultsStore, portfolio_final_id: str | None, final_settings: Settings) -> dict[str, Any]:
    """Compara el CAGR anual de la cartera del finalista contra carteras aleatorias del mismo tamaño.

    Si el finalista está en la cola alta de la distribución aleatoria (percentil > 0.95), su
    rendimiento no se explica por el azar de escoger acciones. Usa los retornos anuales realizados
    de la cartera (`annual_metrics.parquet`) y el pool de retornos anuales de los activos del panel
    de precios PIT del mismo año.
    """
    from module.evaluation.robustness import random_portfolio_test
    import numpy as np

    if portfolio_final_id is None:
        return {"status": "sin cartera final"}
    annual_path = store.runs_root / portfolio_final_id / "artifacts" / "annual_metrics.parquet"
    prices_path = final_settings.processed_output_dir / "asset_price_point_in_time.parquet"
    if not annual_path.exists() or not prices_path.exists():
        return {"status": "faltan annual_metrics o precios PIT"}

    annual = pd.read_parquet(annual_path)
    model_annual = pd.Series(annual["portfolio_return"].to_numpy(),
                             index=pd.to_numeric(annual["year"]).astype(int))

    # Pool de retornos anuales por año desde la serie mensual PIT: cambio dic→dic por ticker.
    prices = pd.read_parquet(prices_path).copy()
    prices["year"] = pd.to_datetime(prices["snapshot_date"]).dt.year
    yearly_last = prices.sort_values("snapshot_date").groupby(["ticker", "year"])["price"].last()
    returns_by_year: dict[int, np.ndarray] = {}
    for ticker, per_year in yearly_last.groupby(level=0):
        series = per_year.droplevel(0).sort_index()
        annual_return = series.pct_change()
        for year, value in annual_return.dropna().items():
            returns_by_year.setdefault(int(year), []).append(float(value))
    returns_by_year = {year: np.asarray(values) for year, values in returns_by_year.items()
                       if len(values) > 0}
    if not returns_by_year:
        return {"status": "sin pool de retornos por año"}

    portfolio_size = int(getattr(final_settings, "target_max", 10))
    return random_portfolio_test(model_annual, returns_by_year, portfolio_size, n_simulations=1000)


def _latest_agent_run(processed: Path) -> Path | None:
    root = Path(processed) / "agents"
    candidates = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.stat().st_mtime) if root.exists() else []
    return candidates[-1] if candidates else None


def _run_input_paths(settings: Settings) -> dict[str, Path]:
    """Inputs crudos que determinan un cálculo completo y permiten deduplicarlo con seguridad."""
    from environment import DEV_RAW_DIR, RAW_DIR
    raw = DEV_RAW_DIR if settings.dev_mode else RAW_DIR
    return {name: raw / name for name in ("finnhub_metrics.parquet", "prices.parquet", "profiles.parquet", "report_dates.parquet")}


def _run_cached_stage(stage: str, settings: Settings, *, agent_dir: Path | None = None) -> Path | None:
    """Ejecuta una etapa o restaura la misma transformación desde ``data/recycle``."""
    processed = settings.processed_output_dir
    if stage == "download":
        STAGE_HANDLERS[stage](settings)
        return None
    inputs, outputs, destination = _cache_contract(stage, processed, agent_dir=agent_dir)
    key = stage_key(stage, settings, inputs)
    if restore_recycle(stage, key, destination):
        log.info("Reciclaje %s: restaurada clave %s", stage, key[:12])
        return _cached_agent_dir(processed, key) if stage == "agents" else agent_dir
    if stage == "backtest":
        if agent_dir is None:
            raise RuntimeError("No se puede ejecutar backtest sin el agente exacto del escenario.")
        run_backtest_from_run_dir(settings, agent_dir)
    else:
        STAGE_HANDLERS[stage](settings)
    publish_recycle(stage, key, outputs(), settings)
    log.info("Reciclaje %s: publicada clave %s", stage, key[:12])
    return _latest_agent_run(processed) if stage == "agents" else agent_dir


def _cached_agent_dir(processed: Path, key: str) -> Path:
    """Obtiene el único agente identificado por una entrada de caché restaurada."""
    candidates = [path for path in cache_dir("agents", key).iterdir() if path.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"La caché de agentes {key[:12]} no identifica un único agente.")
    agent_dir = processed / "agents" / candidates[0].name
    if not (agent_dir / "agent_scores.parquet").exists():
        raise FileNotFoundError(f"Restauración incompleta del agente {candidates[0].name}.")
    return agent_dir


def _cache_contract(stage: str, processed: Path, *, agent_dir: Path | None = None):
    """Define inputs, outputs y destino por etapa; es la frontera de reutilización."""
    if stage == "dataset":
        raw = processed.parent / "raw" if processed.name == "dev" else processed.parent / "raw"
        # La fuente real se obtiene del Settings en el handler; los nombres son estables en data/raw.
        from environment import RAW_DIR, DEV_RAW_DIR
        source = DEV_RAW_DIR if processed.name == "dev" else RAW_DIR
        inputs = [source / name for name in ("finnhub_metrics.parquet", "prices.parquet", "profiles.parquet", "report_dates.parquet")]
        names = ("panel_point_in_time.parquet", "asset_price_point_in_time.parquet", "benchmark_point_in_time.parquet")
        return inputs, lambda: [processed / name for name in names], processed
    if stage == "features":
        inputs = [processed / name for name in ("panel_point_in_time.parquet", "asset_price_point_in_time.parquet", "benchmark_point_in_time.parquet")]
        names = ("features_point_in_time.parquet", "targets_forward_3m.parquet", "baseline_scores.parquet", "features_coverage.json")
        return inputs, lambda: [processed / name for name in names], processed
    if stage == "agents":
        inputs = [processed / "features_point_in_time.parquet", processed / "targets_forward_3m.parquet"]
        root = processed / "agents"
        return inputs, lambda: [_latest_agent_run(processed)] if _latest_agent_run(processed) else [], root
    if stage == "backtest":
        agent = agent_dir
        if agent is None:
            raise FileNotFoundError("No hay run de agentes para ejecutar el backtest.")
        inputs = [agent / "agent_scores.parquet", processed / "asset_price_point_in_time.parquet", processed / "benchmark_point_in_time.parquet"]
        names = ("positions.parquet", "orders.parquet", "equity.parquet", "annual_metrics.parquet", "backtest_summary.json")
        return inputs, lambda: [agent / name for name in names], agent
    raise ValueError(f"Etapa no cacheable: {stage}")


def _summary_for_run(store: ResultsStore, run_id: str) -> dict[str, Any]:
    path = store.runs_root / run_id / "artifacts" / "backtest_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_csv_exports(artifacts: Path) -> None:
    csv_dir = artifacts / "csv"
    csv_dir.mkdir(exist_ok=True)
    mapping = {
        "agent_scores.parquet": "ranking_by_snapshot.csv",
        "positions.parquet": "positions_history.csv",
        "orders.parquet": "orders_history.csv",
        "annual_metrics.parquet": "annual_metrics.csv",
    }
    for parquet_name, csv_name in mapping.items():
        source = artifacts / parquet_name
        if source.exists():
            pd.read_parquet(source).to_csv(csv_dir / csv_name, index=False)
