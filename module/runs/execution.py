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
from module.runs.experiments import ScenarioSpec
from module.runs.results_store import ResultsStore, execution_hash
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
) -> str:
    """Ejecuta etapas existentes y publica una copia inmutable de sus artefactos."""
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
    # referencia se reemplaza exactamente por la salida de la etapa agents del propio run.
    agent_dir: Path | None = _latest_agent_run(effective.processed_output_dir) if mode == "backtest" else None
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
    """Ejecuta un study dirigido: Fase 1 aislada y Fase 2 de combinaciones controladas.

    No usa producto cartesiano. Esto evita miles de ejecuciones y mantiene interpretable el
    efecto de cada eje antes de combinar únicamente los ganadores.
    """
    store = store or ResultsStore()
    study_id, study_dir = store.create_study({**dict(study_payload), "variables": dict(variables),
                                               "strategy": "two_phase_directed"})
    phase1_specs = _isolated_specs(settings, variables)
    phase1_rows: list[dict[str, Any]] = []
    for spec in phase1_specs:
        candidate = replace(settings, **spec["overrides"])
        run_id = execute_run(
            candidate, mode=mode, run_kind="scenario", study_id=study_id,
            label=f"{study_payload.get('name', 'study')} · Fase 1 · {spec['name']}",
            description=str(study_payload.get("description", "")),
            tags=study_payload.get("tags", ()), grid_definition={"phase": 1, **spec}, store=store,
        )
        store.add_to_study(study_id, run_id)
        phase1_rows.append({"phase": 1, "scenario": spec["name"], "axis": spec.get("axis"),
                            "overrides": spec["overrides"], "run_id": run_id,
                            **spec["overrides"], **_summary_for_run(store, run_id)})
    phase1 = pd.DataFrame(phase1_rows)
    phase2_specs, decision = _directed_phase2(phase1, settings, variables)
    phase2_rows = _execute_official_specs(phase2_specs, settings, store, study_id, phase=2,
                                          label_prefix=str(study_payload.get("name", "Study")),
                                          description=str(study_payload.get("description", "")),
                                          tags=study_payload.get("tags", ()))
    comparison = pd.concat([phase1, pd.DataFrame(phase2_rows)], ignore_index=True, sort=False)
    write_parquet(comparison, study_dir / "comparison_data.parquet")
    phase2_best = _stable_best(pd.DataFrame(phase2_rows))
    write_json({"strategy": "two_phase_directed", "criterion": "mean_rank_ic", "phase1": decision,
                "phase2_winner": phase2_best}, study_dir / "decision.json")
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


def _directed_phase2(
    phase1: pd.DataFrame, settings: Settings, variables: Mapping[str, list[Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Combina mejores niveles y prueba segundos niveles uno a uno, sin cartesiano."""
    selected: dict[str, Any] = {}
    decision: dict[str, Any] = {"axes": {}}
    for axis in variables:
        candidates = phase1.loc[(phase1["axis"] == axis) | (phase1["scenario"] == "baseline")].copy()
        best = _stable_best(candidates)
        if not best:
            continue
        selected.update(dict(best.get("overrides") or {}))
        second = _second_stable(candidates.to_dict("records"), str(best["scenario"]))
        decision["axes"][axis] = {"chosen": best["scenario"], "overrides": best.get("overrides", {}),
                                   "second": second}
    specs = [{"name": "combined_best", "overrides": dict(selected)}]
    for axis, data in decision["axes"].items():
        if not data.get("second"):
            continue
        variant = dict(selected)
        for key in data["overrides"]:
            variant.pop(key, None)
        variant.update(data["second"]["overrides"])
        specs.append({"name": f"{axis}_second", "overrides": variant})
    return specs, decision


def official_phase1_specs() -> list[ScenarioSpec]:
    """Catalogo oficial conservado: Fase 1 aislada, no una rejilla libre."""
    from escenarios.fase1_ejes import SCENARIOS
    return list(SCENARIOS)


def execute_official_optimization(settings: Settings, store: ResultsStore | None = None) -> str:
    """Ejecuta las fases oficiales sin usar carpetas de resultados heredadas.

    Fase 1 aísla ejes; Fase 2 combina únicamente ganadores estables y sus alternativas;
    después afina un conjunto pequeño de hiperparámetros y valida el finalista en 2025-2026.
    """
    specs = official_phase1_specs()
    store = store or ResultsStore()
    payload = {
        "name": "optimization-official", "kind": "optimization",
        "description": "Optimización oficial: Fase 1 aislada, Fase 2 dirigida, afinado y validación reservada.",
        "tags": ["official", "optimization"], "selection_metric": "rank_ic_oos",
        "selection_until_year": 2024, "reserved_years": [2025, 2026],
    }
    study_id, study_dir = store.create_study(payload)
    phase1_rows: list[dict[str, Any]] = []
    for spec in specs:
        candidate = spec.apply_to(settings)
        run_id = execute_run(candidate, mode="full", run_kind="scenario", study_id=study_id,
                             label=f"Fase 1 · {spec.name}", description=payload["description"],
                             tags=payload["tags"], grid_definition={"official_scenario": spec.name,
                                                                       "overrides": dict(spec.overrides)}, store=store)
        store.add_to_study(study_id, run_id)
        phase1_rows.append({"phase": 1, "scenario": spec.name, **dict(spec.overrides),
                            **_summary_for_run(store, run_id), "run_id": run_id})
    phase1 = pd.DataFrame(phase1_rows)
    phase2_specs, phase1_decision = _phase2_from_phase1(phase1)
    phase2_rows = _execute_official_specs(phase2_specs, settings, store, study_id, phase=2)
    phase2 = pd.DataFrame(phase2_rows)
    chosen_phase2 = _stable_best(phase2)
    chosen_overrides = dict(chosen_phase2.get("overrides", {})) if chosen_phase2 else {}
    hyper_specs = _hyperparameter_specs(chosen_overrides)
    hyper_rows = _execute_official_specs(hyper_specs, settings, store, study_id, phase=3)
    hyper = pd.DataFrame(hyper_rows)
    chosen_hyper = _stable_best(hyper)
    final_overrides = dict(chosen_hyper.get("overrides", {})) if chosen_hyper else chosen_overrides
    final_settings = replace(settings, **final_overrides)
    final_id = execute_run(final_settings, mode="full", run_kind="optimization_final", study_id=study_id,
                           label="Optimization · finalista", description=payload["description"],
                           tags=["official", "final"], grid_definition={"phase": "final", "overrides": final_overrides}, store=store)
    store.add_to_study(study_id, final_id)
    # Cada perfil es un backtest trazable sobre las predicciones del finalista. Se guarda como
    # run hijo y nunca se reconstruye ad hoc desde la interfaz.
    profile_run_ids: dict[str, str] = {}
    for profile in PROFILE_NAMES:
        profile_settings = replace(final_settings, profile=profile)
        profile_id = execute_run(profile_settings, mode="backtest", run_kind="scenario", study_id=study_id,
                                 label=f"Optimization · finalista · perfil {profile}",
                                 description="Backtest del perfil sobre el finalista seleccionado.",
                                 tags=["official", "final", "profile"],
                                 grid_definition={"phase": "profiles", "profile": profile,
                                                  "parent_run_id": final_id}, store=store)
        store.add_to_study(study_id, profile_id)
        profile_run_ids[profile] = profile_id
    final_summary = _summary_for_run(store, final_id)
    reserved = _reserved_validation(store, final_id)
    robustness = _final_robustness(store, final_id)
    comparison = pd.concat([phase1, phase2, hyper], ignore_index=True, sort=False)
    write_parquet(comparison, study_dir / "comparison_data.parquet")
    write_json({
        "selection_metric": "rank_ic_oos", "phase1": phase1_decision,
        "phase2_winner": chosen_phase2, "hyperparameter_winner": chosen_hyper,
        "final_run_id": final_id, "final_settings_overrides": final_overrides,
        "final_summary": final_summary, "profile_run_ids": profile_run_ids,
        "reserved_validation": reserved, "robustness": robustness,
    }, study_dir / "decision.json")
    manifest = json.loads((study_dir / "study_manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "succeeded"
    write_json(manifest, study_dir / "study_manifest.json")
    return study_id


def execute_official_phase1(settings: Settings, store: ResultsStore | None = None) -> str:
    """Alias de compatibilidad: la acción oficial ejecuta ahora el flujo completo."""
    return execute_official_optimization(settings, store)


def _execute_official_specs(
    specs: list[dict[str, Any]], settings: Settings, store: ResultsStore, study_id: str, *, phase: int,
    label_prefix: str = "Optimization", description: str = "Escenario dirigido del catálogo oficial.",
    tags: Iterable[str] = ("official",),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        overrides = dict(spec["overrides"])
        run_id = execute_run(replace(settings, **overrides), mode="full", run_kind="scenario", study_id=study_id,
                             label=f"{label_prefix} · Fase {phase} · {spec['name']}",
                             description=description,
                             tags=[*tags, f"phase{phase}"],
                             grid_definition={"phase": phase, "overrides": overrides}, store=store)
        store.add_to_study(study_id, run_id)
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


def _phase2_from_phase1(phase1: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compone Fase 2 sin producto cartesiano: ganador conjunto y segundos controlados."""
    from escenarios.fase1_ejes import ARTIFACTS, AXES
    by_name = {str(row["scenario"]): row for row in phase1.to_dict("records")}
    baseline = by_name.get("baseline", {})
    selected: dict[str, Any] = {}
    decision: dict[str, Any] = {"axes": {}, "artifacts": []}
    for axis, levels in AXES.items():
        candidates = []
        for level, overrides in levels.items():
            row = by_name.get(level)
            if row:
                candidates.append({"name": level, "overrides": overrides, **row})
        best = _stable_best(pd.DataFrame(candidates))
        if best:
            selected.update(dict(best["overrides"]))
            decision["axes"][axis] = {"chosen": best["name"], "overrides": best["overrides"],
                                        "second": _second_stable(candidates, best["name"])}
    for name, overrides in ARTIFACTS.items():
        row = by_name.get(f"artifact_{name}")
        if row and row.get("mean_rank_ic", float("-inf")) > baseline.get("mean_rank_ic", float("inf")):
            selected.update(overrides)
            decision["artifacts"].append(name)
    specs = [{"name": "combined_best", "overrides": dict(selected)}]
    for axis, data in decision["axes"].items():
        second = data.get("second")
        if second:
            variant = dict(selected)
            for key in data["overrides"]:
                variant.pop(key, None)
            variant.update(second["overrides"])
            specs.append({"name": f"{axis}_second", "overrides": variant})
    return specs, decision


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


def _final_robustness(store: ResultsStore, run_id: str) -> dict[str, Any]:
    """Diagnósticos no destructivos sobre el finalista publicado."""

    path = store.runs_root / run_id / "artifacts" / "rank_ic_diagnostics.parquet"
    if not path.exists():
        return {}
    diagnostics = pd.read_parquet(path)
    meta = diagnostics.loc[diagnostics["agent"] == "meta_final"].copy()
    if meta.empty:
        return {}
    bootstrap = block_bootstrap_ci(meta.sort_values("prediction_date").set_index("prediction_date")["rank_ic"])
    loyo = leave_one_year_out(diagnostics).to_dict("records")
    payload = {"block_bootstrap": bootstrap, "leave_one_year_out": loyo,
               "label_permutation": {"status": "pendiente de ejecución aislada", "n_permutations": 0}}
    write_json(payload, store.runs_root / run_id / "artifacts" / "robustness.json")
    return payload


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
