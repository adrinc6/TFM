"""Ejecucion controlada de runs y studies para CLI y consola local."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
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
from module.runs.experiments import split_variables, stress_variables
from module.runs.results_store import ResultsStore, canonical_json, execution_hash
from module.scenarios.variables import COST_STRESS_CASES, FULL_STUDY_OPTIONS, FULL_STUDY_PHASE3_OPTIONS
from module.runs.recycle import (cache_dir, cache_lock, publish as publish_recycle,
                                 restore as restore_recycle, stage_key, STAGE_FIELDS)
from module.common.utils import setup_logging, write_json, write_parquet

log = logging.getLogger(__name__)

PARALLEL_SCENARIO_WORKERS = 4
MODEL_THREADS_PER_WORKER = 3

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


def _record_run_telemetry(store: ResultsStore, run_dir: Path, wall_seconds: float,
                          cpu_seconds: float, model_threads: int) -> None:
    rss_bytes = None
    try:
        import psutil
        rss_bytes = psutil.Process().memory_info().rss
    except (ImportError, OSError):
        pass
    store.record_telemetry(run_dir, {
        "wall_seconds": round(wall_seconds, 3), "process_cpu_seconds": round(cpu_seconds, 3),
        "effective_logical_cores": round(cpu_seconds / wall_seconds, 3) if wall_seconds else 0.0,
        "rss_bytes_at_finish": rss_bytes, "model_threads": model_threads,
    })


def _phase1_worker(store_root: str, settings: Settings, study_id: str, name: str,
                   description: str, tags: list[str], spec: dict[str, Any]) -> tuple[dict[str, Any], str | None, list[dict[str, Any]]]:
    """Proceso aislado: no comparte memoria, logging ni workspace mutable con otro escenario."""
    skipped: list[dict[str, Any]] = []
    run_id = _safe_scenario_run(
        ResultsStore(Path(store_root)), replace(settings, lgbm_n_jobs=MODEL_THREADS_PER_WORKER, **spec["overrides"]),
        study_id=study_id, label=f"{name} · Fase 1 · {spec['name']}", description=description,
        tags=tags, grid_definition={"phase": "1", **spec}, skipped=skipped, scenario=spec["name"],
        update_progress=False,
    )
    return spec, run_id, skipped


def _parallel_phase1_specs(specs: list[dict[str, Any]], settings: Settings, store: ResultsStore,
                           study_id: str, name: str, description: str, tags: list[str]) -> tuple[list[tuple[dict[str, Any], str | None]], list[dict[str, Any]]]:
    """Materializa dos escenarios como máximo; devuelve resultados en el orden científico original."""
    log.info("Fase 1: %d escenarios; %d workers de %d hilos de modelo", len(specs),
             PARALLEL_SCENARIO_WORKERS, MODEL_THREADS_PER_WORKER)
    # La primera tanda puede incluir baseline y escenarios realmente independientes. Después se
    # forman tandas de hasta cuatro, sin dos claves de dataset/features todavía no materializadas
    # en la misma tanda.
    baseline = [spec for spec in specs if spec.get("axis") is None]
    remaining = [spec for spec in specs if spec.get("axis") is not None]
    first_wave = list(baseline)
    first_signatures = [_materialization_signature(replace(settings, **spec["overrides"])) for spec in baseline]
    materialized: dict[str, set[str]] = {"dataset": set(), "features": set(), "agents": set()}
    first_used = {stage: {signature[index] for signature in first_signatures}
                  for index, stage in enumerate(("dataset", "features", "agents"))}
    for spec in list(remaining):
        signature = _materialization_signature(replace(settings, **spec["overrides"]))
        if len(first_wave) >= PARALLEL_SCENARIO_WORKERS or any(
            signature[index] in first_used[stage] and signature[index] not in materialized[stage]
            for index, stage in enumerate(("dataset", "features", "agents"))
        ):
            continue
        first_wave.append(spec)
        first_signatures.append(signature)
        for index, stage in enumerate(("dataset", "features", "agents")):
            first_used[stage].add(signature[index])
        remaining.remove(spec)
    waves = [first_wave] if first_wave else []
    for signature in first_signatures:
        for index, stage in enumerate(("dataset", "features", "agents")):
            materialized[stage].add(signature[index])
    while remaining:
        wave: list[dict[str, Any]] = []
        used = {"dataset": set(), "features": set(), "agents": set()}
        for spec in list(remaining):
            signature = _materialization_signature(replace(settings, **spec["overrides"]))
            if len(wave) >= PARALLEL_SCENARIO_WORKERS or any(
                signature[index] in used[stage] and signature[index] not in materialized[stage]
                for index, stage in enumerate(("dataset", "features", "agents"))
            ):
                continue
            wave.append(spec)
            for index, stage in enumerate(("dataset", "features", "agents")):
                used[stage].add(signature[index])
            remaining.remove(spec)
        if not wave:  # defensa ante una especificación inesperada no hasheable
            wave.append(remaining.pop(0))
        waves.append(wave)
        for spec in wave:
            signature = _materialization_signature(replace(settings, **spec["overrides"]))
            for index, stage in enumerate(("dataset", "features", "agents")):
                materialized[stage].add(signature[index])
    completed: dict[str, tuple[dict[str, Any], str | None]] = {}
    skipped: list[dict[str, Any]] = []
    for number, wave in enumerate(waves, start=1):
        log.info("Fase 1: tanda %d/%d con %d escenarios independientes", number, len(waves), len(wave))
        with ProcessPoolExecutor(max_workers=len(wave)) as pool:
            futures = [pool.submit(_phase1_worker, str(store.root), settings, study_id, name, description, tags, spec)
                       for spec in wave]
            for future in as_completed(futures):
                spec, run_id, task_skipped = future.result()
                log.info("Fase 1: escenario terminado %s (%s)", spec["name"], run_id or "omitido")
                completed[spec["name"]] = (spec, run_id)
                skipped.extend(task_skipped)
    return [completed[spec["name"]] for spec in specs], skipped


def _materialization_signature(settings: Settings) -> tuple[str, str, str]:
    """Firma previa a ejecutar: detecta recursos compartidos sin leer/escribir la caché."""
    dataset = canonical_json({field: getattr(settings, field) for field in STAGE_FIELDS["dataset"]})
    features = canonical_json({"dataset": dataset,
                               "settings": {field: getattr(settings, field) for field in STAGE_FIELDS["features"]}})
    agents = canonical_json({"features": features,
                             "settings": {field: getattr(settings, field) for field in STAGE_FIELDS["agents"]}})
    return dataset, features, agents


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
    if mode == "backtest":
        agent_dir = agent_dir or _latest_agent_run(effective.processed_output_dir)
        if agent_dir is None:
            raise RuntimeError("No hay un artefacto de agentes para el backtest.")
        agent_dir = Path(agent_dir)
    input_paths = _run_input_paths(effective)
    if mode == "backtest":
        input_paths["agent_scores"] = agent_dir / "agent_scores.parquet"
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
    run_started = time.perf_counter()
    cpu_started = time.process_time()
    runtime = (replace(effective, workspace_dir=run_dir / "workspace" / "processed")
               if mode == "full" else effective)
    ensure_directories(runtime)
    # El modo backtest aislado conserva el comportamiento CLI; en un pipeline completo esta
    # referencia se reemplaza exactamente por la salida de la etapa agents del propio run. Si el
    # llamante fija un agent_dir explícito (fase de cartera/perfiles sobre el finalista), se usa ese.
    if mode == "backtest":
        agent_dir = _prepare_agent_workspace(agent_dir, run_dir)
    try:
        for stage in stages:
            log.info("Run %s: iniciando %s", run_id, stage)
            started = time.perf_counter()
            output, recycled = _run_cached_stage(stage, runtime, agent_dir=agent_dir)
            store.record_stage_timing(run_dir, stage, time.perf_counter() - started,
                                      source="recycled" if recycled else "computed")
            if stage == "agents":
                agent_dir = output
        source = agent_dir or (_latest_agent_run(runtime.processed_output_dir) if mode == "backtest" else None)
        summary: dict[str, Any] = {}
        if source is not None:
            summary = store.publish_artifacts(run_dir, source)
            _write_csv_exports(run_dir / "artifacts")
        wall_seconds = time.perf_counter() - run_started
        cpu_seconds = time.process_time() - cpu_started
        _record_run_telemetry(store, run_dir, wall_seconds, cpu_seconds, effective.lgbm_n_jobs)
        store.complete(run_dir, summary)
        return run_id
    except Exception as exc:
        _record_run_telemetry(store, run_dir, time.perf_counter() - run_started,
                              time.process_time() - cpu_started, effective.lgbm_n_jobs)
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
    phase3_vars = {axis: list(values) for axis, values in variables.items()
                   if axis in FULL_STUDY_PHASE3_OPTIONS}
    phase12_and_portfolio = {axis: values for axis, values in variables.items()
                             if axis not in phase3_vars}
    model_vars, portfolio_vars = split_variables(phase12_and_portfolio)
    stress_vars = stress_variables(phase12_and_portfolio)
    include_robustness = bool(study_payload.get("include_robustness", False))
    payload = {
        "name": str(study_payload.get("name", "study")), "kind": str(study_payload.get("kind", "exploratory")),
        "description": str(study_payload.get("description", "")),
        "hypothesis": str(study_payload.get("hypothesis", "")).strip(),
        "tags": list(study_payload.get("tags", [])),
        "variables": dict(variables), "strategy": "unified_full_cycle",
        "selection_metric": "rank_ic_oos", "selection_until_year": 2024, "reserved_years": [2025, 2026],
    }
    return run_optimization(settings, model_vars=model_vars, portfolio_vars=portfolio_vars,
                            stress_vars=stress_vars, payload=payload, store=store,
                            include_full_robustness=include_robustness,
                            hyperparameter_options=phase3_vars)


def execute_official_optimization(
    settings: Settings, store: ResultsStore | None = None, *, name: str = "optimization-official",
    hypothesis: str = "", resume_study_id: str | None = None,
) -> str:
    """Optimización oficial: ciclo completo barriendo TODAS las variables barribles.

    Barrido derivado del mismo contrato que usa ``study``: el manual permite subconjuntos y el
    oficial ejecuta todos los ejes/valores, respetando su asignación a Fases 1–2, 3 y 4.
    """
    actual_store = store or ResultsStore()
    variables = {axis: list(values) for axis, values in FULL_STUDY_OPTIONS.items()}
    model_vars, portfolio_vars = split_variables(variables)
    stress_vars = stress_variables(variables)
    payload = {
        "name": name.strip() or "optimization-official", "kind": "optimization",
        "description": "Optimización oficial: barrido completo de modelo (Fase 1/2), afinado, cartera y validación reservada.",
        "tags": ["official", "optimization"], "strategy": "unified_full_cycle",
        "selection_metric": "rank_ic_oos", "selection_until_year": 2024, "reserved_years": [2025, 2026],
    }
    payload["hypothesis"] = hypothesis.strip()
    if hypothesis.strip():
        payload["description"] = hypothesis.strip()
    # La optimización oficial SIEMPRE ejecuta la robustez completa (placebo + carteras aleatorias):
    # es el estudio que sostiene la credibilidad del TFM.
    with _full_study_lock(actual_store):
        return run_optimization(settings, model_vars=model_vars, portfolio_vars=portfolio_vars,
                                stress_vars=stress_vars, payload=payload, store=actual_store,
                                include_full_robustness=True, resume_study_id=resume_study_id)


@contextmanager
def _full_study_lock(store: ResultsStore):
    """Bloqueo entre procesos; elimina automáticamente locks de procesos ya muertos."""
    lock_path = store.root / "full_study.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            owner = int(lock_path.read_text(encoding="utf-8").strip())
            os.kill(owner, 0)
        except (OSError, ValueError):
            lock_path.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"Ya hay un full study activo (PID {owner}).")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
    finally:
        os.close(descriptor)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def run_optimization(
    settings: Settings,
    *,
    model_vars: Mapping[str, list[Any]],
    portfolio_vars: Mapping[str, list[Any]],
    stress_vars: Mapping[str, list[Any]] | None = None,
    payload: Mapping[str, Any],
    store: ResultsStore | None = None,
    include_full_robustness: bool = False,
    resume_study_id: str | None = None,
    hyperparameter_options: Mapping[str, list[Any]] | None = None,
) -> str:
    """Ciclo completo unificado que comparten study y optimización oficial.

    Fase 1 (ejes de modelo aislados) → Fase 2 (greedy top-2, sin cartesiano) → afinado de
    hiperparámetros → run final → fase de cartera (re-backtest por criterio económico) → 8
    perfiles → robustez → validación reservada. La selección de modelo es siempre por rank-IC OOS;
    los ejes de cartera se eligen por métrica económica porque no alteran el aprendizaje.
    """
    store = store or ResultsStore()
    # Todas las alternativas del study deben usar la misma ventana temporal. Se toma el máximo
    # lookback solicitado antes de materializar el dataset: evita construir desde 1990 cuando el
    # primer entrenamiento posible es 2003 (2015 menos 12 años), sin recortar ninguna cohorte.
    settings = _with_study_data_start(settings, model_vars)
    payload = {**payload, "effective_data_start_date": settings.data_start_date}
    if resume_study_id:
        study_id, study_dir = store.resume_study(resume_study_id)
    else:
        study_id, study_dir = store.create_study(dict(payload))
        store.update_study_status(study_id, "running", current_phase="1", completed_scenarios=0)
    name = str(payload.get("name", "study"))
    description = str(payload.get("description", ""))
    tags = list(payload.get("tags", []))

    # --- Fase 1: cada eje de modelo aislado sobre el baseline ---
    # Un escenario que no entrena (p.ej. ventana muy corta o cadencia que deja pocas filas) se
    # salta y se registra, sin abortar el estudio: con un barrido amplio es esperable que algunas
    # combinaciones extremas no sean viables.
    phase1_rows: list[dict[str, Any]] = []
    phase1_specs = _isolated_specs(settings, model_vars)
    phase1_results, skipped = _parallel_phase1_specs(
        phase1_specs, settings, store, study_id, name, description, tags,
    )
    for spec, run_id in phase1_results:
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
    store.update_study_status(study_id, "running", current_phase="2", completed_scenarios=len(phase1_rows))
    log.info("Fase 1 completada: %d válidos, %d omitidos. Iniciando Fase 2 greedy.", len(phase1_rows), len(skipped))

    # --- Fase 2: greedy incremental con top-2 por eje (sin producto cartesiano) ---
    phase2_rows, model_overrides, phase1_decision = _greedy_phase2(
        phase1, model_vars, settings, store, study_id, name=name, description=description, tags=tags)
    phase2 = pd.DataFrame(phase2_rows)
    store.update_study_status(study_id, "running", current_phase="3")
    log.info("Fase 2 completada: %d evaluaciones. Iniciando Fase 3.", len(phase2_rows))

    # --- Fase 3: afinado de hiperparámetros sobre el ganador de modelo ---
    hyper_rows, final_model_overrides = _hyperparameter_phase(
        model_overrides, settings, store, study_id, name=name, description=description, tags=tags,
        options=hyperparameter_options)
    hyper = pd.DataFrame(hyper_rows)
    log.info("Fase 3 de hiperparámetros completada: %d evaluaciones. Validando semillas.", len(hyper_rows))
    seed_rows = _seed_stability_phase(
        final_model_overrides, settings, store, study_id, name=name,
        description=description, tags=tags,
    )
    seed_stability = pd.DataFrame(seed_rows)
    store.update_study_status(study_id, "running", current_phase="final_model")

    # --- Run final del modelo ganador ---
    final_settings = replace(settings, **final_model_overrides)
    final_id = execute_run(final_settings, mode="full", run_kind="optimization_final", study_id=study_id,
                           label=f"{name} · finalista", description=description,
                           tags=[*tags, "final"], grid_definition={"phase": "final", "overrides": final_model_overrides}, store=store)
    store.add_to_study(study_id, final_id)
    final_agent_dir = store.runs_root / final_id / "artifacts"
    store.update_study_status(study_id, "running", current_phase="4_cartera")
    log.info("Modelo final %s publicado. Iniciando Fase 4 de cartera.", final_id)

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

    # Costs are stress assumptions, never a criterion that can be optimised away.
    # Every predefined pair is reported on the selected model/portfolio, but none
    # can alter ``portfolio_overrides`` or the recommended configuration.
    cost_stress_rows = _cost_stress_phase(
        portfolio_settings, store, study_id, agent_dir=final_agent_dir,
        name=name, description=description, tags=tags,
    )
    # Reglas de cartera mecánicas (drift/expulsión/rotación): se estresan igual que los costes —
    # se reporta cada valor sobre el finalista, pero NINGUNO altera portfolio_overrides.
    portfolio_stress_rows = _portfolio_stress_phase(
        portfolio_settings, dict(stress_vars or {}), store, study_id, agent_dir=final_agent_dir,
        name=name, description=description, tags=tags,
    )
    store.update_study_status(study_id, "running", current_phase="5_perfiles")
    log.info("Fase de cartera finalizada. Iniciando perfiles.")

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
    store.update_study_status(study_id, "running", current_phase="robustness_and_reserved")
    best_config = _best_config_summary(final_model_overrides, portfolio_overrides,
                                       profile_run_ids, store)
    # Comparativa completa: todas las fases (modelo 1/2/3 + cartera + perfiles) para la app.
    comparison = pd.concat(
        [phase1, phase2, hyper, seed_stability, pd.DataFrame(portfolio_rows),
         pd.DataFrame(cost_stress_rows), pd.DataFrame(portfolio_stress_rows),
         pd.DataFrame(profile_rows)],
        ignore_index=True, sort=False)
    write_parquet(comparison, study_dir / "comparison_data.parquet")
    write_json({
        "strategy": "unified_full_cycle", "selection_metric": "rank_ic_oos",
        "hypothesis": str(payload.get("hypothesis", "")),
        "phase1": phase1_decision, "model_winner": {"overrides": final_model_overrides,
                                                    "summary": _summary_for_run(store, final_id)},
        "seed_stability": seed_rows,
        "phase_portfolio": portfolio_trace, "portfolio_overrides": portfolio_overrides,
        "final_run_id": portfolio_final_id, "model_final_run_id": final_id,
        "cost_stress": cost_stress_rows,
        "portfolio_stress": portfolio_stress_rows,
        "final_settings_overrides": {**final_model_overrides, **portfolio_overrides},
        "final_summary": final_summary,
        # Salida final del study: un run por perfil sobre el modelo+cartera óptimos.
        "final_profile_run_ids": profile_run_ids, "profile_run_ids": profile_run_ids,
        "profile_reference": "balanced",
        "profile_selection_policy": "reported_not_optimized",
        "reserved_validation": reserved, "robustness": robustness, "best_config": best_config,
        "skipped_scenarios": skipped,
    }, study_dir / "decision.json")
    store.update_study_status(study_id, "succeeded", current_phase="complete")
    return study_id


def _with_study_data_start(settings: Settings, model_vars: Mapping[str, list[Any]]) -> Settings:
    """Calcula el primer año necesario para todo el study, no para un escenario aislado.

    Se redondea al 1 de enero del año de inicio del entrenamiento. Fase 1 compara todos los ejes
    contra exactamente el mismo panel PIT; las ventanas técnicas conservan su tratamiento normal
    de observaciones iniciales incompletas (sin inventar historia anterior al corte).
    """
    values = model_vars.get("train_lookback_years", [settings.train_lookback_years])
    try:
        maximum_lookback = max(int(value) for value in values)
    except (TypeError, ValueError):
        maximum_lookback = settings.train_lookback_years
    anchor_month = (settings.execution_quarter - 1) * 3 + 1
    anchor = pd.Timestamp(year=settings.execution_year, month=anchor_month, day=1)
    anchor += pd.Timedelta(days=settings.execution_lag_days)
    first_training_date = anchor - pd.DateOffset(years=maximum_lookback)
    return replace(settings, data_start_date=f"{first_training_date.year}-01-01")


def _isolated_specs(settings: Settings, variables: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    """Baseline + una modificación cada vez para atribuir su efecto a un eje."""
    specs = [{"name": "baseline", "axis": None, "overrides": {}}]
    selection_axes = {
        "feature_selection_min_coverage", "feature_selection_lookback_quarters",
        "feature_selection_min_permutation_importance", "feature_selection_min_positive_fraction",
        "feature_selection_max_features_per_agent",
    }
    for axis, values in variables.items():
        if axis in selection_axes:
            continue
        baseline = getattr(settings, axis)
        for value in values:
            if value == baseline:
                continue
            safe_name = str(value).replace(" ", "_").replace(".", "_")
            specs.append({"name": f"{axis}_{safe_name}", "axis": axis, "overrides": {axis: value}})
    for mode in ("oos_stability_prune", "block_gated"):
        for axis in selection_axes:
            for value in variables.get(axis, []):
                if value == getattr(settings, axis):
                    continue
                safe_name = str(value).replace(" ", "_").replace(".", "_")
                specs.append({"name": f"{mode}_{axis}_{safe_name}", "axis": axis,
                              "overrides": {"feature_weighting_mode": mode, axis: value}})
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
    baseline_positive = (float(baseline_row["rank_ic_positive_fraction"].iloc[0])
                         if not baseline_row.empty and "rank_ic_positive_fraction" in baseline_row else 0.0)

    # Mejores dos niveles de cada eje y ranking de ejes por impacto (mejora sobre baseline).
    axis_options: dict[str, list[dict[str, Any]]] = {}
    axis_impact: dict[str, float] = {}
    decision: dict[str, Any] = {"axes": {}}
    for axis in model_vars:
        candidates = phase1.loc[phase1["axis"] == axis]
        best = _stable_best(candidates)
        if not best:
            continue
        best_ic = float(best.get("mean_rank_ic", float("-inf")))
        best_positive = float(best.get("rank_ic_positive_fraction", 0.0))
        if best_ic <= baseline_ic or best_positive < baseline_positive:
            decision["axes"][axis] = {"chosen": "baseline", "accepted": False,
                                       "reason": "no mejora estable frente al baseline"}
            continue
        second = _second_stable(candidates.to_dict("records"), str(best["scenario"]))
        axis_options[axis] = [best] + ([second] if second else [])
        axis_impact[axis] = best_ic - baseline_ic
        decision["axes"][axis] = {"chosen": best.get("scenario"), "accepted": True,
                                   "overrides": dict(best.get("overrides") or {}),
                                   "second": second}

    # Combinación inicial: el mejor nivel de cada eje.
    selected: dict[str, Any] = {}
    for options in axis_options.values():
        selected.update(dict(options[0].get("overrides") or {}))

    rows: list[dict[str, Any]] = []
    best_score = (float("-inf"), float("-inf"), float("-inf"), 0)

    skipped: list[dict[str, Any]] = []

    def evaluate(overrides: dict[str, Any], label: str) -> tuple[float, float, float, int]:
        nonlocal rows
        run_id = _safe_scenario_run(
            store, replace(settings, **overrides), study_id=study_id,
            label=f"{name} · Fase 2 · {label}", description=description, tags=[*tags, "phase2"],
            grid_definition={"phase": "2", "overrides": overrides}, skipped=skipped, scenario=label)
        if run_id is None:
            return (float("-inf"), float("-inf"), float("-inf"), 0)
        summary = _summary_for_run(store, run_id)
        rows.append({"phase": "2", "scenario": label, "overrides": dict(overrides), **overrides, **summary, "run_id": run_id})
        return _stability_key(summary)

    best_score = evaluate(selected, "combined_best")

    # Greedy: para cada eje (más impactante primero), probar su 2º mejor sobre la combinación.
    for axis in sorted(axis_impact, key=axis_impact.get, reverse=True):
        options = axis_options[axis]
        if len(options) < 2:
            continue
        trial = dict(selected)
        for key in dict(options[0].get("overrides") or {}):
            trial.pop(key, None)
        trial.update(dict(options[1].get("overrides") or {}))
        score = evaluate(trial, f"{axis}_second")
        if score > best_score:
            best_score, selected = score, trial

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
        best_overrides, best_value, best_score, best_run = None, None, (float("-inf"),) * 3, None
        axis_results: list[dict[str, Any]] = []
        for value in values:
            overrides = {axis: value}
            trial = {**selected, **overrides}
            # Settings valida el escenario antes de re-backtestearlo.
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
            summary = _portfolio_selection_summary(store, run_id, candidate_settings)
            score = _economic_score(summary, criterion)
            axis_results.append({"value": value, criterion: score, "run_id": run_id})
            comparison_rows.append({"phase": "4_cartera", "scenario": f"{axis}={value}", "axis": axis,
                                    "overrides": dict(overrides), "run_id": run_id, **summary})
            order_key = _portfolio_order_key(summary, criterion)
            if order_key > best_score:
                best_overrides, best_value, best_score, best_run = overrides, value, order_key, run_id
        if best_overrides is not None:
            selected.update(best_overrides)  # expande el eje (simple o compuesto) sobre la selección
        trace["axes"][axis] = {"chosen": best_value, "score": best_score, "run_id": best_run,
                                "candidates": axis_results}
    return selected, trace, comparison_rows


def _economic_score(summary: Mapping[str, Any], criterion: str) -> float:
    """Métrica económica para elegir un eje de cartera; drawdown se maximiza como negativo."""
    value = summary.get(criterion)
    if value is None:
        return float("-inf")
    return -float(value) if criterion == "max_drawdown" else float(value)


def _portfolio_order_key(summary: Mapping[str, Any], criterion: str) -> tuple[float, float, float]:
    """IR principal; desempates por menor drawdown y mayor estabilidad anual."""
    return (
        _economic_score(summary, criterion),
        -float(summary.get("max_drawdown", float("inf"))),
        float(summary.get("beat_rate", 0.0)),
    )


def _portfolio_selection_summary(
    store: ResultsStore, run_id: str, settings: Settings, *, selection_until_year: int = 2024,
) -> dict[str, Any]:
    """Recalcula métricas económicas sin permitir que la reserva elija la cartera."""
    from module.evaluation.backtest import _annual_metrics, _summary

    if not hasattr(store, "runs_root"):
        return _summary_for_run(store, run_id)
    artifacts = store.runs_root / run_id / "artifacts"
    equity_path = artifacts / "equity.parquet"
    if not equity_path.exists():
        return _summary_for_run(store, run_id)
    equity = pd.read_parquet(equity_path)
    dates = pd.to_datetime(equity["snapshot_date"], errors="coerce")
    selection_equity = equity.loc[dates.dt.year <= selection_until_year].copy()
    diagnostics_path = artifacts / "rank_ic_diagnostics.parquet"
    diagnostics = pd.read_parquet(diagnostics_path) if diagnostics_path.exists() else None
    if diagnostics is not None and "prediction_date" in diagnostics:
        prediction_dates = pd.to_datetime(diagnostics["prediction_date"], errors="coerce")
        diagnostics = diagnostics.loc[prediction_dates.dt.year <= selection_until_year].copy()
    summary = _summary(selection_equity, _annual_metrics(selection_equity), settings, diagnostics)
    summary["portfolio_selection_until_year"] = selection_until_year
    summary["portfolio_selection_periods"] = int(len(selection_equity))
    return summary


def _cost_stress_phase(
    settings: Settings, store: ResultsStore, study_id: str, *, agent_dir: Path | None,
    name: str, description: str, tags: list[str],
) -> list[dict[str, Any]]:
    """Report all cost assumptions without selecting the cheapest one."""
    rows: list[dict[str, Any]] = []
    for case in COST_STRESS_CASES:
        overrides = dict(case["overrides"])
        run_id = execute_run(
            replace(settings, **overrides), mode="backtest", run_kind="stress", study_id=study_id,
            label=f"{name} · estrés de costes · {case['name']}", description=description,
            tags=[*tags, "cost_stress"],
            grid_definition={"phase": "4b_cost_stress", "overrides": overrides}, store=store,
            agent_dir=agent_dir,
        )
        store.add_to_study(study_id, run_id)
        rows.append({"phase": "4b_cost_stress", "scenario": case["name"], "axis": "cost_stress",
                     "overrides": overrides, "run_id": run_id, **overrides, **_summary_for_run(store, run_id)})
    return rows


def _portfolio_stress_phase(
    settings: Settings, stress_vars: Mapping[str, list[Any]], store: ResultsStore, study_id: str, *,
    agent_dir: Path | None, name: str, description: str, tags: list[str],
) -> list[dict[str, Any]]:
    """Estresa las reglas de cartera mecánicas sin seleccionar ninguna.

    Para cada eje mecánico (drift/expulsión/rotación) barre sus valores aislados sobre el finalista
    (re-backtest sobre el mismo ``agent_dir``, no reentrena) y reporta las métricas económicas. Igual
    que el estrés de costes: NO devuelve overrides, así el barrido no puede mover la configuración
    recomendada. El objetivo es demostrar que el resultado es estable ante estas reglas de fricción,
    no elegir la más favorable.
    """
    rows: list[dict[str, Any]] = []
    for axis, values in stress_vars.items():
        for value in values:
            overrides = {axis: value}
            try:
                candidate_settings = replace(settings, **overrides)
            except ValueError as exc:
                log.warning("Estrés de cartera omitido (%s=%s): %s", axis, value, exc)
                continue
            run_id = execute_run(
                candidate_settings, mode="backtest", run_kind="stress", study_id=study_id,
                label=f"{name} · estrés de cartera · {axis}={value}", description=description,
                tags=[*tags, "portfolio_stress"],
                grid_definition={"phase": "4c_portfolio_stress", "axis": axis, "value": value,
                                 "overrides": overrides},
                store=store, agent_dir=agent_dir)
            store.add_to_study(study_id, run_id)
            rows.append({"phase": "4c_portfolio_stress", "scenario": f"{axis}={value}", "axis": axis,
                         "overrides": overrides, "run_id": run_id, **overrides,
                         **_summary_for_run(store, run_id)})
    return rows


def _safe_scenario_run(
    store: ResultsStore, settings: Settings, *, study_id: str, label: str, description: str,
    tags: list[str], grid_definition: Mapping[str, Any], skipped: list[dict[str, Any]],
    scenario: str, mode: str = "full", agent_dir: Path | None = None, update_progress: bool = True,
) -> str | None:
    """Ejecuta un escenario del barrido tolerando fallos: si no entrena, se salta y se registra.

    Devuelve el run_id si tuvo éxito, o None si el escenario no fue viable (p.ej. sin filas de
    entrenamiento suficientes). Así un escenario extremo no aborta el estudio completo.
    """
    try:
        if update_progress and hasattr(store, "update_study_status"):
            store.update_study_status(study_id, "running", current_scenario=scenario,
                                      current_phase=str(grid_definition.get("phase", "unknown")))
        log.info("Escenario %s: preparando modo %s", scenario, mode)
        run_id = execute_run(settings, mode=mode, run_kind="scenario", study_id=study_id,
                             label=label, description=description, tags=tags,
                             grid_definition=grid_definition, store=store, agent_dir=agent_dir)
        store.add_to_study(study_id, run_id)
        log.info("Escenario %s: publicado como run %s", scenario, run_id)
        return run_id
    except Exception as exc:  # noqa: BLE001 — el barrido debe continuar aunque un escenario falle
        log.warning("Escenario omitido (%s): %s", scenario, exc)
        skipped.append({"scenario": scenario, "error": str(exc), "overrides": dict(grid_definition.get("overrides", {}))})
        return None


def _best_config_summary(
    model_overrides: Mapping[str, Any], portfolio_overrides: Mapping[str, Any],
    profile_run_ids: Mapping[str, str], store: ResultsStore,
) -> dict[str, Any]:
    """Resumen legible del modelo/cartera; los perfiles se reportan sin optimizarlos."""
    return {"model": dict(model_overrides), "portfolio": dict(portfolio_overrides),
            "profile_reference": "balanced", "profiles_reported": list(profile_run_ids)}


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
    if "rank_ic_selection_cohorts" not in ranked:
        ranked["rank_ic_selection_cohorts"] = 0
    ranked["rank_ic_positive_fraction"] = ranked["rank_ic_positive_fraction"].fillna(0.0)
    ranked["rank_ic_std"] = ranked["rank_ic_std"].fillna(float("inf"))
    ranked = ranked.sort_values(
        ["mean_rank_ic", "rank_ic_positive_fraction", "rank_ic_std", "rank_ic_selection_cohorts"],
        ascending=[False, False, True, False],
    )
    return ranked.iloc[0].to_dict()


def _stability_key(summary: Mapping[str, Any]) -> tuple[float, float, float, int]:
    """Orden estable: señal, frecuencia positiva, menor dispersión y más cohortes."""
    return (
        float(summary.get("mean_rank_ic", float("-inf"))),
        float(summary.get("rank_ic_positive_fraction", 0.0)),
        -float(summary.get("rank_ic_std", float("inf"))),
        int(summary.get("rank_ic_selection_cohorts", 0)),
    )


def _second_stable(candidates: list[dict[str, Any]], winner: str) -> dict[str, Any] | None:
    alternatives = [candidate for candidate in candidates if str(candidate.get("name", candidate.get("scenario"))) != winner]
    best = _stable_best(pd.DataFrame(alternatives))
    if not best:
        return None
    return {"name": best.get("name", best.get("scenario")), "overrides": best.get("overrides", {})}


def _hyperparameter_phase(
    base: Mapping[str, Any], settings: Settings, store: ResultsStore, study_id: str, *,
    name: str, description: str, tags: list[str],
    options: Mapping[str, list[Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Afinado greedy por eje: combina ganadores sin ejecutar el producto cartesiano."""
    selected = dict(base)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    search = FULL_STUDY_PHASE3_OPTIONS if options is None else options
    for axis, values in search.items():
        axis_rows: list[dict[str, Any]] = []
        for value in values:
            overrides = {**selected, axis: value}
            scenario = f"{axis}_{str(value).replace('.', '_')}"
            run_id = _safe_scenario_run(
                store, replace(settings, **overrides), study_id=study_id,
                label=f"{name} · Fase 3 · {scenario}", description=description,
                tags=[*tags, "phase3"],
                grid_definition={"phase": "3", "axis": axis, "value": value,
                                 "overrides": overrides},
                skipped=skipped, scenario=scenario,
            )
            if run_id is None:
                continue
            row = {"phase": "3", "scenario": scenario, "axis": axis, "value": value,
                   "overrides": dict(overrides), **overrides,
                   **_summary_for_run(store, run_id), "run_id": run_id}
            rows.append(row)
            axis_rows.append(row)
        winner = _stable_best(pd.DataFrame(axis_rows))
        if winner is not None:
            selected[axis] = winner["value"]
    return rows, selected


def _seed_stability_phase(
    overrides: Mapping[str, Any], settings: Settings, store: ResultsStore, study_id: str, *,
    name: str, description: str, tags: list[str], seeds: tuple[int, ...] = (7, 42, 2026),
) -> list[dict[str, Any]]:
    """Valida el finalista con semillas prefijadas; nunca elige la semilla más afortunada."""
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for seed in seeds:
        candidate = {**dict(overrides), "random_seed": seed}
        scenario = f"seed_{seed}"
        run_id = _safe_scenario_run(
            store, replace(settings, **candidate), study_id=study_id,
            label=f"{name} · estabilidad · {scenario}", description=description,
            tags=[*tags, "seed_stability"],
            grid_definition={"phase": "3b_seed_stability", "axis": "random_seed",
                             "value": seed, "overrides": candidate},
            skipped=skipped, scenario=scenario,
        )
        if run_id is not None:
            rows.append({"phase": "3b_seed_stability", "scenario": scenario,
                         "axis": "random_seed", "value": seed, "overrides": candidate,
                         "run_id": run_id, **_summary_for_run(store, run_id)})
    return rows


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
        label_permutation = _label_permutation(
            final_settings, diagnostics,
            targets_path=store.runs_root / run_id / "artifacts" / "targets_forward_3m.parquet",
            input_dir=store.runs_root / run_id / "workspace" / "processed",
        )
        random_portfolio = _random_portfolio(store, portfolio_final_id, final_settings)
    else:
        label_permutation = {"status": "no solicitada (robustez completa desactivada)", "n_permutations": 0}
        random_portfolio = {"status": "no solicitada (robustez completa desactivada)"}

    payload = {"block_bootstrap": bootstrap, "leave_one_year_out": loyo,
               "label_permutation": label_permutation, "random_portfolio": random_portfolio}
    write_json(payload, store.runs_root / run_id / "artifacts" / "robustness.json")
    return payload


def _label_permutation(
    final_settings: Settings, diagnostics: pd.DataFrame, n_permutations: int = 5, *,
    targets_path: Path | None = None, input_dir: Path | None = None,
) -> dict[str, Any]:
    """Placebo: reentrena el finalista con los retornos futuros BARAJADOS y mide su rank-IC.

    Si el sistema aprende de verdad, el rank-IC del meta_final debe COLAPSAR a ~0 con etiquetas
    aleatorias; si no colapsa, hay fuga o el resultado real es artefacto. Se permutan los targets,
    se reentrena (semilla distinta cada vez) en un workspace privado y se recoge el rank-IC del
    meta_final. Los targets y el directorio de agentes reales nunca se modifican.
    """
    from module.evaluation.robustness import label_permutation_test
    import numpy as np

    processed = final_settings.processed_output_dir
    targets_path = targets_path or (processed / "targets_forward_3m.parquet")
    if not targets_path.exists():
        return {"status": "sin targets para permutar", "n_permutations": 0}

    base_targets = pd.read_parquet(targets_path)
    permuted_ic: list[float] = []
    rng = np.random.default_rng(0)
    import tempfile
    with tempfile.TemporaryDirectory(prefix="tfm-placebo-") as temporary_name:
        workspace = Path(temporary_name)
        for i in range(n_permutations):
            shuffled = base_targets.copy()
            shuffled["forward_excess_return_3m"] = rng.permutation(
                shuffled["forward_excess_return_3m"].to_numpy())
            shuffled_path = workspace / f"targets-permutation-{i}.parquet"
            write_parquet(shuffled, shuffled_path)
            try:
                run_root = workspace / f"agents-{i}"
                build_agent_scores(replace(final_settings, random_seed=1000 + i),
                                   target_path_override=shuffled_path, run_root=run_root,
                                   input_dir_override=input_dir)
                candidates = [path for path in run_root.iterdir() if path.is_dir()]
                if len(candidates) != 1:
                    raise RuntimeError("El placebo no produjo un único run de agentes.")
                perm_run = candidates[0]
                perm_diag = pd.read_parquet(perm_run / "rank_ic_diagnostics.parquet")
                permuted_ic.append(float(
                    perm_diag.loc[perm_diag["agent"] == "meta_final", "rank_ic"].mean()))
            except Exception as exc:  # noqa: BLE001 — una permutación fallida no aborta el placebo
                log.warning("Permutación %s falló: %s", i, exc)

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
    prices_path = store.runs_root / portfolio_final_id / "artifacts" / "asset_price_point_in_time.parquet"
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

    portfolio_size = int(getattr(final_settings, "target_size", 8))
    return random_portfolio_test(model_annual, returns_by_year, portfolio_size, n_simulations=1000)


def _latest_agent_run(processed: Path) -> Path | None:
    root = Path(processed) / "agents"
    candidates = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.stat().st_mtime) if root.exists() else []
    return candidates[-1] if candidates else None


def _prepare_agent_workspace(source: Path, run_dir: Path) -> Path:
    """Materializa entradas de agentes en un workspace privado para un único backtest."""
    workspace = run_dir / "workspace" / "agent"
    workspace.mkdir(parents=True, exist_ok=False)
    required = ("agent_scores.parquet",)
    optional = (
        "rank_ic_diagnostics.parquet", "meta_weights.parquet", "model_feature_attribution.parquet",
        "agent_local_attribution.parquet", "feature_diagnostics.parquet", "feature_catalog.json",
        "manifest.json", "asset_price_point_in_time.parquet", "stock_panel.parquet",
    )
    for name in (*required, *optional):
        item = source / name
        if item.exists():
            shutil.copy2(item, workspace / name)
        elif name in required:
            raise FileNotFoundError(f"Falta {name} en el agente exacto {source}.")
    write_json({"source_agent_dir": str(source.resolve())}, workspace / "parent_agent.json")
    return workspace


def _run_input_paths(settings: Settings) -> dict[str, Path]:
    """Inputs crudos que determinan un cálculo completo y permiten deduplicarlo con seguridad."""
    from environment import DEV_RAW_DIR, RAW_DIR
    raw = DEV_RAW_DIR if settings.dev_mode else RAW_DIR
    return {name: raw / name for name in ("finnhub_metrics.parquet", "prices.parquet", "profiles.parquet", "report_dates.parquet")}


def _run_cached_stage(stage: str, settings: Settings, *, agent_dir: Path | None = None) -> tuple[Path | None, bool]:
    """Ejecuta una etapa o restaura la misma transformación desde ``data/recycle``.

    Devuelve ``(salida, reciclada)``: ``reciclada`` es True si la etapa se restauró de la caché en
    lugar de recalcularse, para que la app pueda distinguir el coste real del reciclado.
    """
    processed = settings.processed_output_dir
    if stage == "download":
        STAGE_HANDLERS[stage](settings)
        return None, False
    inputs, outputs, destination = _cache_contract(stage, processed, agent_dir=agent_dir, settings=settings)
    key = stage_key(stage, settings, inputs)
    if restore_recycle(stage, key, destination):
        log.info("Reciclaje %s: restaurada clave %s", stage, key[:12])
        return (_cached_agent_dir(processed, key) if stage == "agents" else agent_dir), True
    with cache_lock(stage, key):
        if restore_recycle(stage, key, destination):
            log.info("Reciclaje %s: restaurada tras espera %s", stage, key[:12])
            return (_cached_agent_dir(processed, key) if stage == "agents" else agent_dir), True
        if stage == "backtest":
            if agent_dir is None:
                raise RuntimeError("No se puede ejecutar backtest sin el agente exacto del escenario.")
            run_backtest_from_run_dir(settings, agent_dir)
        else:
            STAGE_HANDLERS[stage](settings)
        publish_recycle(stage, key, outputs(), settings)
    log.info("Reciclaje %s: publicada clave %s", stage, key[:12])
    return (_latest_agent_run(processed) if stage == "agents" else agent_dir), False


def _cached_agent_dir(processed: Path, key: str) -> Path:
    """Obtiene el único agente identificado por una entrada de caché restaurada."""
    candidates = [path for path in cache_dir("agents", key).iterdir() if path.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"La caché de agentes {key[:12]} no identifica un único agente.")
    agent_dir = processed / "agents" / candidates[0].name
    if not (agent_dir / "agent_scores.parquet").exists():
        raise FileNotFoundError(f"Restauración incompleta del agente {candidates[0].name}.")
    return agent_dir


def _cache_contract(
    stage: str, processed: Path, *, agent_dir: Path | None = None, settings: Settings | None = None,
):
    """Define inputs, outputs y destino por etapa; es la frontera de reutilización."""
    if stage == "dataset":
        # La fuente real se obtiene del Settings en el handler; los nombres son estables en data/raw.
        from environment import RAW_DIR, DEV_RAW_DIR
        source = (DEV_RAW_DIR if settings.dev_mode else RAW_DIR) if settings else (
            DEV_RAW_DIR if processed.name == "dev" else RAW_DIR)
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


# Los artefactos de un run son inmutables una vez escritos, así que su resumen se calcula una
# sola vez por (runs_root, run_id) y se reutiliza. Sin esto se releen backtest_summary.json y
# rank_ic_diagnostics.parquet varias veces por run (Fase 1/2, perfiles, recomendado, best_config).
_SUMMARY_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _summary_for_run(store: ResultsStore, run_id: str) -> dict[str, Any]:
    cache_key = (str(store.runs_root), run_id)
    cached = _SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    result = _compute_summary_for_run(store, run_id)
    _SUMMARY_CACHE[cache_key] = result
    return dict(result)


def _compute_summary_for_run(store: ResultsStore, run_id: str) -> dict[str, Any]:
    path = store.runs_root / run_id / "artifacts" / "backtest_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    # Model selection is strictly frozen at the end of 2024.  The 2025-26
    # diagnostics remain available for the final reserved validation only.
    diagnostics_path = store.runs_root / run_id / "artifacts" / "rank_ic_diagnostics.parquet"
    if not diagnostics_path.exists():
        return summary
    diagnostics = pd.read_parquet(diagnostics_path)
    if "agent" in diagnostics:
        diagnostics = diagnostics.loc[diagnostics["agent"] == "meta_final"]
    if diagnostics.empty:
        return summary
    dates = pd.to_datetime(diagnostics["prediction_date"])
    values = pd.to_numeric(diagnostics.loc[dates.dt.year <= 2024, "rank_ic"], errors="coerce").dropna()
    if values.empty:
        return summary
    summary.update({
        "mean_rank_ic": float(values.mean()),
        "rank_ic_positive_fraction": float((values > 0).mean()),
        "rank_ic_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "rank_ic_selection_until_year": 2024,
        "rank_ic_selection_cohorts": int(len(values)),
    })
    return summary


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
