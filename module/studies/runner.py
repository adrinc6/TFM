"""Único ejecutor científico: configuración cerrada → métricas y evidencia."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from environment import DATA_DIR
from module.common.utils import link_or_copy, write_json, write_parquet
from module.evaluation.backtest import BacktestResult, run_backtest
from module.modeling.agents import build_agent_scores
from module.studies.catalog import KNOWN_STRESS_YEARS, SELECTION_ERAS, SELECTION_UNTIL_YEAR
from module.studies.config import settings_from_values
from module.storage.cache import canonical_json, enforce_cache_limit
from module.storage.datasets import dataset_identity, ensure_prepared


SUMMARY_CACHE = DATA_DIR / "cache" / "evaluations"
log = logging.getLogger(__name__)

# Artefactos que describen el **modelo** de un ganador, frente a los que describen su cartera
# (equity, posiciones, órdenes, contribuciones, métricas anuales). Una sola lista, porque la usan
# dos caminos: quien retiene la evidencia de un run del Model Study y quien la enlaza a la evidencia
# del ganador del Portfolio Study, que reutiliza esos mismos ficheros sin reentrenar nada.
MODEL_EVIDENCE_FILES = (
    "agent_scores.parquet", "meta_weights.parquet", "rank_ic_diagnostics.parquet",
    "rank_tail_diagnostics.parquet", "model_feature_attribution.parquet",
    "agent_local_attribution.parquet", "feature_diagnostics.parquet",
    "signal_health.parquet", "signal_calibration.parquet", "feature_catalog.json",
    "manifest.json", "dataset_reference.json",
)


def evaluation_key(
    values: Mapping[str, Any],
    *,
    dataset_hash: str,
    random_seed: int,
    profile: str,
    overrides: Mapping[str, Any] | None = None,
    target_identity: str = "real",
) -> str:
    """Identidad de una evaluación. **Incluye el hash del dataset y esto no es opcional.**

    Sin él, dos configuraciones idénticas evaluadas sobre datos distintos —porque se reingirieron
    precios, porque cambió el universo o porque cambió el código de features— compartían clave y la
    segunda leía de caché el resultado de la primera. En los artefactos existentes se puede observar
    la misma `evaluation_key` con dos CAGR distintos: es un fallo de corrección, no de reporte.
    """
    payload = {
        "values": dict(values),
        "dataset_hash": dataset_hash,
        "random_seed": random_seed,
        "profile": profile,
        "overrides": dict(overrides or {}),
        "target_identity": target_identity,
        "runner_schema": 2,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def run_evaluation(
    values: Mapping[str, Any],
    *,
    random_seed: int = 42,
    profile: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    target_override: Path | None = None,
    target_identity: str = "real",
    retain_dir: Path | None = None,
) -> dict[str, Any]:
    effective_profile = profile or "balanced"
    started = time.perf_counter()
    base_settings = settings_from_values(
        values, random_seed=random_seed, profile=effective_profile, overrides=overrides,
    )
    # La identidad del dataset se resuelve antes que la clave de evaluación: es barata (hashea
    # settings y las huellas de los ficheros raw, sin construir nada) y sin ella la clave no puede
    # distinguir dos evaluaciones que solo difieren en los datos.
    expected_hash, _ = dataset_identity(base_settings)
    key = evaluation_key(
        values, dataset_hash=expected_hash, random_seed=random_seed, profile=effective_profile,
        overrides=overrides, target_identity=target_identity,
    )
    cached = SUMMARY_CACHE / key / "summary.json"
    if cached.exists() and retain_dir is None:
        log.info("Evaluación %s: resumen reutilizado de caché", key[:12])
        payload = json.loads(cached.read_text(encoding="utf-8"))
        payload["source"] = "cached"
        return payload

    log.info("Evaluación %s: inicio (perfil=%s, semilla=%s)", key[:12], effective_profile, random_seed)
    dataset_hash, prepared, dataset_reused = ensure_prepared(base_settings)
    if dataset_hash != expected_hash:
        raise RuntimeError(
            f"El dataset materializado ({dataset_hash[:12]}) no coincide con el previsto "
            f"({expected_hash[:12]}); la clave de evaluación sería inconsistente."
        )
    log.info("Evaluación %s: dataset %s (%s)", key[:12], dataset_hash[:12], "reutilizado" if dataset_reused else "creado")
    work = Path(tempfile.mkdtemp(prefix=f"evaluation-{key[:10]}-"))
    try:
        runtime = settings_from_values(
            values, workspace_dir=prepared, random_seed=random_seed, profile=effective_profile,
            overrides=overrides,
        )
        agents_root = work / "agents"
        build_agent_scores(
            runtime,
            target_path_override=target_override,
            run_root=agents_root,
            input_dir_override=prepared,
        )
        agent_dirs = [path for path in agents_root.iterdir() if path.is_dir()]
        if len(agent_dirs) != 1:
            raise RuntimeError("El runner esperaba exactamente un directorio de agente.")
        agent_dir = agent_dirs[0]
        scores = pd.read_parquet(agent_dir / "agent_scores.parquet")
        diagnostics = pd.read_parquet(agent_dir / "rank_ic_diagnostics.parquet")
        prices = pd.read_parquet(prepared / "asset_price_point_in_time.parquet")
        benchmark = pd.read_parquet(prepared / "benchmark_point_in_time.parquet")
        targets = pd.read_parquet(target_override or (prepared / "targets_forward.parquet"))
        result = run_backtest(scores, prices, benchmark, runtime, targets)
        tail_path = agent_dir / "rank_tail_diagnostics.parquet"
        tail = pd.read_parquet(tail_path) if tail_path.exists() else pd.DataFrame()
        summary = _summary(
            result, diagnostics, tail, dataset_hash=dataset_hash, evaluation_key=key,
            elapsed_seconds=time.perf_counter() - started, dataset_reused=dataset_reused,
        )
        cache_dir = SUMMARY_CACHE / key
        cache_dir.mkdir(parents=True, exist_ok=True)
        write_json(summary, cached)
        enforce_cache_limit(protected={key})
        if retain_dir is not None:
            _retain_evidence(retain_dir, prepared, agent_dir, result, summary)
        log.info("Evaluación %s: terminada en %.1fs", key[:12], summary["elapsed_seconds"])
        return summary
    finally:
        shutil.rmtree(work, ignore_errors=True)


@functools.lru_cache(maxsize=8)
def _cached_parquet(resolved: str, stamp: tuple[int, int]) -> pd.DataFrame:
    """Lectura memoizada de un Parquet de solo lectura, con la ruta y su mtime/tamaño como clave.

    La rejilla del Portfolio Study evalúa cientos de combinaciones contra **los mismos cinco
    ficheros**: la evidencia del ganador se congela antes del bucle y el panel preparado no cambia.
    Releerlos en cada iteración era E/S pura repetida sin resultado distinto.

    ``stamp`` (mtime_ns, tamaño) entra en la clave para que reescribir un artefacto invalide la
    entrada en vez de servir un marco obsoleto; sin él, un estudio encadenado podría leer la
    evidencia del estudio anterior. Los consumidores del backtest no mutan lo que reciben, así que
    el marco se comparte sin copia; los dos que sí lo transforman —``apply_profile`` con perfil de
    estilo y ``_meta_diagnostics``— ya copian por su cuenta.
    """
    return pd.read_parquet(resolved)


def _read_frozen_parquet(path: Path) -> pd.DataFrame:
    """Envuelve `_cached_parquet` resolviendo ruta y sello; ante un `stat` fallido, lee directo."""
    try:
        info = path.stat()
    except OSError:
        return pd.read_parquet(path)
    return _cached_parquet(str(path.resolve()), (info.st_mtime_ns, info.st_size))


def run_profile_evaluation(
    values: Mapping[str, Any], profile: str, evidence_dir: Path, retain_dir: Path | None = None,
    *, include_model_artifacts: bool = False,
) -> dict[str, Any]:
    """Backtest de un perfil sobre los scores ya congelados del ganador, sin reentrenar.

    Con ``include_model_artifacts`` la evidencia retenida deja de ser solo de cartera: se enlazan
    también los artefactos de modelo que se acaban de leer de ``evidence_dir``. No se recalculan
    porque **son los mismos**: este camino no reentrena, así que reejecutar el ganador para
    reproducirlos gastaría un entrenamiento completo y podría no coincidir, ante cualquier
    no-determinismo, con la evidencia que sí alimentó la decisión. Se enlaza con `link_or_copy`, así
    que el coste de disco es cero.

    Queda desactivado por defecto porque la rejilla del Portfolio Study evalúa cientos de
    combinaciones desechables y enlazar y borrar en cada una sería ruido de E/S sin beneficio.
    """
    log.info("Perfil %s: backtest sobre scores congelados", profile)
    runtime = settings_from_values(values, profile=profile)
    reference = json.loads((evidence_dir / "dataset_reference.json").read_text(encoding="utf-8"))
    prepared = Path(reference["prepared_path"])
    read_started = time.perf_counter()
    scores = _read_frozen_parquet(evidence_dir / "agent_scores.parquet")
    diagnostics = _read_frozen_parquet(evidence_dir / "rank_ic_diagnostics.parquet")
    prices = _read_frozen_parquet(prepared / "asset_price_point_in_time.parquet")
    benchmark = _read_frozen_parquet(prepared / "benchmark_point_in_time.parquet")
    targets = _read_frozen_parquet(prepared / "targets_forward.parquet")
    read_elapsed = time.perf_counter() - read_started
    backtest_started = time.perf_counter()
    result = run_backtest(scores, prices, benchmark, runtime, targets)
    log.debug(
        "Perfil %s: lectura %.3fs, backtest %.3fs", profile, read_elapsed,
        time.perf_counter() - backtest_started,
    )
    summary = {
        "dataset_hash": reference["dataset_hash"], "profile": profile,
        "summary": result.summary,
        "eras": _profile_eras(result.annual_metrics),
        "rank_ic": _rank_ic_block(_meta_diagnostics(diagnostics).pipe(
            lambda frame: frame.loc[frame["year"].le(SELECTION_UNTIL_YEAR)]
        )),
    }
    if retain_dir is not None:
        retain_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(result.equity, retain_dir / "equity.parquet")
        write_parquet(result.annual_metrics, retain_dir / "annual_metrics.parquet")
        write_parquet(result.positions, retain_dir / "positions.parquet")
        write_parquet(result.orders, retain_dir / "orders.parquet")
        write_parquet(result.contributions, retain_dir / "contributions.parquet")
        write_json(summary, retain_dir / "summary.json")
        if include_model_artifacts:
            for name in MODEL_EVIDENCE_FILES:
                source = evidence_dir / name
                if source.exists():
                    link_or_copy(source, retain_dir / name)
    log.info("Perfil %s: terminado (IR=%s, alfa=%s)", profile, result.summary.get("information_ratio"), result.summary.get("mean_annual_alpha"))
    return summary


def _profile_comparison_row(profile: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Fila escalar para la comparación Parquet; el resumen completo vive por perfil."""
    summary = result["summary"]
    rank_ic = result["rank_ic"]
    return {
        "profile": profile,
        "mean_rank_ic": rank_ic.get("mean_rank_ic"),
        "geometric_excess_return": summary.get("geometric_excess_return"),
        "information_ratio": summary.get("information_ratio"),
        "annualized_turnover": summary.get("annualized_turnover"),
        "mean_cash_weight": summary.get("mean_cash_weight"),
    }


def _profile_eras(annual: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for start, end in SELECTION_ERAS:
        part = annual.loc[annual["year"].between(start, end)]
        rows.append({
            "era": f"{start}-{end}",
            "mean_alpha": _finite(part["alpha"].mean()),
            "information_ratio": _finite(part["information_ratio_year"].mean()),
        })
    return rows


def _meta_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    frame = diagnostics.copy()
    if "agent" in frame and frame["agent"].eq("meta_final").any():
        frame = frame.loc[frame["agent"].eq("meta_final")]
    frame["year"] = pd.to_datetime(frame["prediction_date"]).dt.year
    return frame


def _rank_ic_block(frame: pd.DataFrame) -> dict[str, Any]:
    """Bloque de métricas predictivas de una ventana. Una sola definición para todas las ventanas.

    ``ic_ir`` es el Rank-IC medio dividido por su desviación típica: la magnitud que la ley
    fundamental relaciona con el ratio de información alcanzable. Se guardaba `rank_ic_std` y nunca
    se dividía por él.
    """
    values = pd.to_numeric(frame.get("rank_ic"), errors="coerce").dropna()
    if values.empty:
        return {"mean_rank_ic": None, "rank_ic_positive_fraction": None, "rank_ic_std": None,
                "ic_ir": None, "n_cohorts": 0}
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "mean_rank_ic": _finite(values.mean()),
        "rank_ic_positive_fraction": _finite((values > 0).mean()),
        "rank_ic_std": _finite(std),
        "ic_ir": _finite(values.mean() / std) if std > 0 else None,
        "n_cohorts": int(len(values)),
    }


def _summary(
    result: BacktestResult,
    diagnostics: pd.DataFrame,
    tail: pd.DataFrame,
    *,
    dataset_hash: str,
    evaluation_key: str,
    elapsed_seconds: float,
    dataset_reused: bool,
) -> dict[str, Any]:
    """Resumen de una evaluación: ciencia (Rank-IC) aquí, economía ya segmentada por el backtest.

    Las métricas económicas llegan de ``run_backtest`` ya recortadas a la ventana de selección, con
    la confirmación 2025-26 en su propio bloque. Este resumen no las recalcula: antes existían dos
    poblaciones distintas bajo las mismas claves, una con recorte temporal y otra sin él.
    """
    meta_ic = _meta_diagnostics(diagnostics)
    selected_ic = meta_ic.loc[meta_ic["year"].le(SELECTION_UNTIL_YEAR)]
    confirmation_ic = meta_ic.loc[meta_ic["year"].isin(KNOWN_STRESS_YEARS)]
    annual = result.annual_metrics.loc[result.annual_metrics["year"].le(SELECTION_UNTIL_YEAR)].copy()
    tail_selection = tail.copy()
    if not tail_selection.empty:
        date_column = "prediction_date" if "prediction_date" in tail_selection else "snapshot_date"
        tail_selection["year"] = pd.to_datetime(tail_selection[date_column]).dt.year
        tail_selection = tail_selection.loc[tail_selection["year"].le(SELECTION_UNTIL_YEAR)]
    era_rows = []
    for start, end in SELECTION_ERAS:
        ic = selected_ic.loc[selected_ic["year"].between(start, end), "rank_ic"].dropna()
        years = annual.loc[annual["year"].between(start, end)]
        era_rows.append({
            "era": f"{start}-{end}",
            "rank_ic": _finite(ic.mean()),
            "information_ratio": _finite(years["information_ratio_year"].mean()),
            "mean_alpha": _finite(years["alpha"].mean()),
            "positive_alpha_years": int((years["alpha"] > 0).sum()),
        })
    known = result.annual_metrics.loc[result.annual_metrics["year"].isin(KNOWN_STRESS_YEARS)]
    selection_summary = dict(result.summary)
    selection_summary.update(_rank_ic_block(selected_ic))
    selection_summary["tail_spread"] = _finite(
        pd.to_numeric(tail_selection.get("top_decile_minus_universe"), errors="coerce").mean()
        if "top_decile_minus_universe" in tail_selection else None
    )
    selection_summary["transfer_coefficient"] = _transfer_coefficient(
        selection_summary, tail_selection,
    )
    selection_summary["positive_alpha_eras"] = int(
        sum((row["mean_alpha"] or 0) > 0 for row in era_rows)
    )
    return {
        "evaluation_key": evaluation_key,
        "dataset_hash": dataset_hash,
        "source": "computed",
        "dataset_reused": dataset_reused,
        "elapsed_seconds": elapsed_seconds,
        "selection_until_year": SELECTION_UNTIL_YEAR,
        "summary": selection_summary,
        "eras": era_rows,
        "rank_ic_by_cohort": [
            {"date": str(row.prediction_date), "rank_ic": _finite(row.rank_ic)}
            for row in selected_ic[["prediction_date", "rank_ic"]].itertuples(index=False)
        ],
        # La era reservada nunca participó en una decisión: es la única medida predictiva del
        # trabajo libre de sesgo de selección. Se calcula siempre y se publica salga lo que salga.
        "confirmation_2025_2026": {
            **_rank_ic_block(confirmation_ic),
            "rank_ic_by_cohort": [
                {"date": str(row.prediction_date), "rank_ic": _finite(row.rank_ic)}
                for row in confirmation_ic[["prediction_date", "rank_ic"]].itertuples(index=False)
            ],
            "annual_metrics": known.to_dict("records"),
            "role": "confirmacion_fuera_de_muestra_no_usada_en_seleccion",
        },
        "known_stress_not_selection": known.to_dict("records"),
    }


def _transfer_coefficient(summary: Mapping[str, Any], tail: pd.DataFrame) -> float | None:
    """Fracción del diferencial bruto de la señal que llega al alfa realizado de la cartera.

    Compara el alfa geométrico neto con el diferencial decil superior menos decil inferior que la
    señal ofrecía. Es la traducción empírica de la ``TC`` de la ley fundamental y responde
    directamente a la pregunta del tribunal: ¿cuánta señal destruye la implementación?
    """
    if "top_minus_bottom" not in tail or tail.empty:
        return None
    spread = pd.to_numeric(tail["top_minus_bottom"], errors="coerce").dropna()
    realized = summary.get("geometric_excess_return")
    if spread.empty or realized is None or abs(float(spread.mean())) < 1e-9:
        return None
    return _finite(float(realized) / float(spread.mean()))


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _retain_evidence(
    destination: Path,
    prepared: Path,
    agent_dir: Path,
    result: BacktestResult,
    summary: Mapping[str, Any],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in MODEL_EVIDENCE_FILES:
        source = agent_dir / name
        if source.exists():
            shutil.copy2(source, destination / name)
    write_parquet(result.equity, destination / "equity.parquet")
    write_parquet(result.annual_metrics, destination / "annual_metrics.parquet")
    write_parquet(result.orders, destination / "orders.parquet")
    write_parquet(result.positions, destination / "positions.parquet")
    write_parquet(result.contributions, destination / "contributions.parquet")
    write_json(dict(summary), destination / "summary.json")
    write_json(
        {"dataset_hash": summary["dataset_hash"], "prepared_path": str(prepared)},
        destination / "dataset_reference.json",
    )


def execute_model_study(study_id: str) -> dict[str, Any]:
    """Ejecuta de principio a fin la única ruta científica del proyecto."""
    from module.evaluation.profiles import PROFILE_NAMES
    from module.studies.catalog import BY_ID
    from module.studies.config import (
        diagnostic_portfolio_variables, initial_values, ordered_predictive_variables,
    )
    from module.studies.selection import choose_candidate
    from module.storage.datasets import validate_dataset_reference
    from module.storage.evidence import write_report, write_winner
    from module.storage.studies import (
        append_event, create_run, directory_size, find_run,
        read_study, safe_study_path, update_run, update_study, write_storage_manifest,
    )

    study = read_study(study_id)
    definition = study["configuration"]
    dev_scope = study.get("run_scope") == "dev"
    # Ruta científica completa: la regla 5 (descartados solo con resumen) se levanta a petición
    # explícita y cada run conserva su evidencia entera bajo runs_evidence/<run_id>.
    retain_all = bool(study.get("retain_all_runs"))
    # Los studies que solo sirven para elegir configuración paran en el ganador: lo posterior es
    # diagnóstico informativo y, en el caso de la atribución, gastaría la única evaluación de la era
    # reservada 2025-26 sobre una configuración que se va a descartar. El defecto es True para que
    # los studies ya existentes en disco, que no llevan la clave, conserven su recorrido al reanudar.
    post_winner = bool(study.get("post_winner_diagnostics", True))
    directory = safe_study_path(study_id)
    runs_evidence = directory / "runs_evidence"
    evidence = directory / "evidence"
    evidence.mkdir(exist_ok=True)
    values = initial_values(definition)
    if dev_scope:
        values["target_size"] = 3
    decisions: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    from module.storage.studies import list_runs
    completed = sum(run["status"] == "succeeded" for run in list_runs(study_id))

    def execute(
        logical_key: str,
        phase: str,
        variable_id: str,
        value: Any,
        configuration: Mapping[str, Any],
        *,
        seed: int = 42,
        profile: str = "balanced",
        overrides: Mapping[str, Any] | None = None,
        retain: Path | None = None,
        target_override: Path | None = None,
        target_identity: str = "real",
    ) -> dict[str, Any]:
        nonlocal completed
        # El destino explícito (baseline, ganador) manda; en modo completo el resto de runs
        # retiene bajo su propio directorio, nombrado por la clave lógica para que la evidencia
        # sea legible sin abrir el JSON del run.
        if retain is None and retain_all:
            retain = runs_evidence / logical_key.replace(":", "__")
        previous = find_run(study_id, logical_key)
        if previous and previous["status"] == "succeeded" and previous.get("result"):
            append_event(study_id, "info", "run_reused", f"Run reutilizado: {logical_key}", run_id=previous["run_id"])
            return previous["result"]
        attempt = int(previous.get("attempt", 0)) + 1 if previous else 1
        run = create_run(
            study_id, logical_key=logical_key, phase=phase,
            variable_id=variable_id, value=value, configuration=configuration,
            attempt=attempt,
        )
        started = time.perf_counter()
        update_run(study_id, run["run_id"], status="running", stage="evaluation", progress=0.05)
        update_study(study_id, phase=phase, current_run_id=run["run_id"], heartbeat=_utc_now())
        append_event(
            study_id, "info", "run_started",
            f"Run {run['run_id']} · {phase} · {variable_id}={value}",
            run_id=run["run_id"], phase=phase, variable_id=variable_id,
        )
        try:
            runtime_overrides = dict(overrides or {})
            if dev_scope:
                runtime_overrides.update({
                    "run_scope": "dev",
                    "data_start_date": "2016-01-01",
                    "end_date": "2024-12-31",
                    "min_rank_ic_cross_section": 3,
                    "train_lookback_years": min(
                        int(configuration["train_lookback_years"]), 4,
                    ),
                    "lgbm_n_estimators": min(
                        int(configuration["lgbm_n_estimators"]), 30,
                    ),
                    "lgbm_min_child_samples": min(
                        int(configuration["lgbm_min_child_samples"]), 5,
                    ),
                    "lgbm_n_jobs": 1,
                })
                runtime_overrides.setdefault("target_size", 3)
            result = run_evaluation(
                configuration, random_seed=seed, profile=profile,
                overrides=runtime_overrides, retain_dir=retain,
                target_override=target_override, target_identity=target_identity,
            )
            elapsed = time.perf_counter() - started
            bytes_persisted = directory_size(retain) if retain and retain.exists() else 0
            update_run(
                study_id, run["run_id"], status="succeeded", stage="completed",
                progress=1.0, elapsed_seconds=elapsed,
                bytes_persisted=bytes_persisted, result=result,
                evidence_path=_evidence_path(retain, directory),
            )
            completed += 1
            total = max(int(study.get("budget", {}).get("total_runs", 1)), 1)
            update_study(
                study_id, completed_runs=completed,
                progress=min(0.95, completed / total), heartbeat=_utc_now(),
            )
            append_event(
                study_id, "info", "run_succeeded",
                f"Run {run['run_id']} terminado en {elapsed:.1f}s",
                run_id=run["run_id"], source=result.get("source"),
            )
            return result
        except Exception as exc:
            update_run(
                study_id, run["run_id"], status="failed", stage="failed",
                elapsed_seconds=time.perf_counter() - started,
                error=str(exc), traceback=traceback.format_exc(),
            )
            update_study(study_id, status="failed", error=str(exc), traceback=traceback.format_exc())
            append_event(study_id, "error", "run_failed", f"Run {run['run_id']} falló: {exc}", run_id=run["run_id"])
            raise

    update_study(study_id, status="running", phase="predictive", progress=0.01)
    append_event(study_id, "info", "study_started", "Comienza el Model Study.")
    baseline = execute(
        "predictive:baseline", "temporal", "baseline", "baseline", values,
        retain=directory / "evidence_baseline",
    )
    incumbent = {
        "candidate_id": "predictive:baseline",
        "value": "baseline",
        "result": baseline,
        "configuration": dict(values),
    }
    ledger.append(_ledger_row(incumbent, "temporal", "baseline", selected=True))

    for variable_id in ordered_predictive_variables(definition):
        spec = BY_ID[variable_id]
        candidates = []
        for value in definition[variable_id]["values"]:
            configuration = dict(values)
            configuration[variable_id] = value
            logical_key = f"predictive:{variable_id}:{_key_value(value)}"
            if configuration == incumbent["configuration"]:
                result = incumbent["result"]
                candidate_id = incumbent["candidate_id"]
            else:
                result = execute(
                    logical_key, spec.stage, variable_id, value, configuration,
                )
                candidate_id = logical_key
            candidates.append({
                "candidate_id": candidate_id,
                "value": value,
                "result": result,
                "configuration": configuration,
            })
        comparison_incumbent = next(
            (candidate for candidate in candidates if candidate["configuration"] == incumbent["configuration"]),
            incumbent,
        )
        decision = choose_candidate(comparison_incumbent, candidates, variable_id)
        winner = next(
            candidate for candidate in candidates
            if candidate["candidate_id"] == decision["winner_candidate_id"]
        )
        values = dict(winner["configuration"])
        incumbent = winner
        decisions.append(decision)
        for candidate in candidates:
            ledger.append(_ledger_row(
                candidate, spec.stage, variable_id,
                selected=candidate["candidate_id"] == winner["candidate_id"],
                decision=decision,
            ))
        append_event(
            study_id, "info", "decision",
            f"{spec.label}: elegido {winner['value']} solo por Rank-IC.",
            variable_id=variable_id, winner=winner["value"],
        )

    write_parquet(pd.DataFrame(ledger), directory / "evaluation_ledger.parquet")
    write_json({"decisions": decisions}, directory / "decisions.json")
    winner_result = execute(
        "winner:evidence", "winner", "winner", "final",
        values, retain=evidence,
    )
    winner_run = find_run(study_id, "winner:evidence")
    winner_payload = {
        "study_id": study_id,
        "winner_run_id": winner_run["run_id"] if winner_run else incumbent["candidate_id"],
        "configuration": values,
        "selection_metric": "rank_ic_only",
        "selection_until_year": 2024,
        "known_stress_role": "known_stress_not_selection",
        "summary": winner_result,
    }
    write_winner(directory / "winner.json", winner_payload)

    # Los diagnósticos posteriores al ganador son opcionales: un study cuyo único fin es elegir
    # configuración termina aquí. Se mantienen las claves vacías para que el informe y el payload
    # final sigan teniendo la misma forma con y sin diagnósticos.
    robustness: dict[str, Any] = {}
    attribution: dict[str, Any] = {}
    if not post_winner:
        append_event(
            study_id, "info", "post_selection",
            "Ganador congelado; los diagnósticos posteriores quedan desactivados en este Study.",
        )
    else:
        append_event(study_id, "info", "post_selection", "Ganador congelado; comienzan diagnósticos informativos.")
        portfolio_rows = []
        for variable_id in diagnostic_portfolio_variables(definition):
            base = values[variable_id]
            for value in definition[variable_id]["values"]:
                if value == base:
                    continue
                override = {variable_id: value}
                result = execute(
                    f"portfolio:{variable_id}:{_key_value(value)}",
                    "portfolio", variable_id, value, values, overrides=override,
                )
                portfolio_rows.append({
                    "variable_id": variable_id,
                    "base_value": json.dumps(base, ensure_ascii=False),
                    "diagnostic_value": json.dumps(value, ensure_ascii=False),
                    **result["summary"],
                })
        write_parquet(pd.DataFrame(portfolio_rows), directory / "portfolio_comparison.parquet")

        profile_rows = []
        profiles_root = evidence / "profiles"
        for profile in PROFILE_NAMES:
            logical = f"profile:{profile}"
            run = create_run(
                study_id, logical_key=logical, phase="profiles",
                variable_id="profile", value=profile, configuration=values,
                attempt=1,
            ) if not find_run(study_id, logical) else find_run(study_id, logical)
            if run["status"] == "succeeded" and run.get("result"):
                result = run["result"]
            else:
                update_run(study_id, run["run_id"], status="running", stage="backtest", progress=0.1)
                append_event(study_id, "info", "profile_started", f"Perfil {profile}.", run_id=run["run_id"])
                retain_profile = profiles_root / profile
                result = run_profile_evaluation(values, profile, evidence, retain_profile)
                # El perfil siempre materializa su cartera; se registra la ruta para que las vistas
                # pesadas del run la encuentren igual que las de cualquier otro run con evidencia.
                update_run(
                    study_id, run["run_id"], status="succeeded", stage="completed", progress=1.0,
                    result=result, evidence_path=_evidence_path(retain_profile, directory),
                )
                completed += 1
            profile_rows.append(_profile_comparison_row(profile, result))
        write_parquet(pd.DataFrame(profile_rows), directory / "profile_comparison.parquet")

        dev = dev_scope
        diagnostics = pd.read_parquet(evidence / "rank_ic_diagnostics.parquet")
        scores = pd.read_parquet(evidence / "agent_scores.parquet")
        reference = json.loads((evidence / "dataset_reference.json").read_text(encoding="utf-8"))
        prepared = validate_dataset_reference(reference["dataset_hash"])
        targets = pd.read_parquet(prepared / "targets_forward.parquet")
        robustness = _robustness_base(
            diagnostics, scores, targets, prepared, evidence, values, winner_result, dev=dev,
        )
        for seed in (7, 2026):
            result = execute(f"robustness:seed:{seed}", "robustness", "seed", seed, values, seed=seed)
            robustness["seeds"].append({"seed": seed, "summary": result["summary"]})
        robustness["seed_dispersion"] = _seed_dispersion(winner_result, robustness["seeds"])
        placebo_count = 2 if dev else 5
        for seed in range(101, 101 + placebo_count):
            shuffled = _shuffle_targets(targets, seed)
            temporary = Path(tempfile.mkdtemp(prefix=f"placebo-{seed}-"))
            try:
                target_path = temporary / "targets_forward.parquet"
                write_parquet(shuffled, target_path)
                result = execute(
                    f"robustness:placebo:{seed}", "robustness", "label_placebo", seed,
                    values, seed=seed, target_override=target_path,
                    target_identity=f"placebo-{seed}",
                )
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
            robustness["label_placebos"].append({"seed": seed, "summary": result["summary"]})
        write_json(robustness, directory / "robustness.json")

        # Atribución: se ejecuta con el ganador ya congelado y fuera de todo bucle de decisión, de
        # modo que la era reservada 2025-26 se evalúa exactamente una vez y su resultado se publica
        # sea cual sea. El protocolo está pre-registrado en docs/bitacora.md antes de la ejecución.
        append_event(study_id, "info", "attribution_started", "Atribución de factores y confirmación fuera de muestra.")
        from module.research.attribution import build_attribution, write_attribution

        attribution = build_attribution(
            evidence, prepared, values,
            candidate_ic_series=[
                [row["rank_ic"] for row in run.get("result", {}).get("rank_ic_by_cohort", [])
                 if row.get("rank_ic") is not None]
                for run in list_runs(study_id)
                if run.get("status") == "succeeded" and run.get("result")
            ],
        )
        attribution["confirmation_2025_2026"] = winner_result.get("confirmation_2025_2026", {})
        write_attribution(directory / "attribution.json", attribution)

    update_study(
        study_id, status="succeeded", phase="completed", progress=1.0,
        winner_run_id=winner_payload["winner_run_id"], completed_runs=completed,
        worker_pid=None, heartbeat=_utc_now(), error=None, traceback=None,
    )
    final = {
        **winner_payload, "status": "succeeded",
        "robustness": robustness, "attribution": attribution,
    }
    write_report(directory / "report.md", final)
    write_storage_manifest(study_id)
    append_event(study_id, "info", "study_succeeded", "Model Study terminado correctamente.")
    return final


def _evidence_path(retain: Path | None, directory: Path) -> str | None:
    """Ruta de la evidencia de un run, relativa al Study, o None si no materializó ninguna."""
    if retain is None or not retain.exists():
        return None
    return str(retain.relative_to(directory))


def _ledger_row(
    candidate: Mapping[str, Any],
    phase: str,
    variable_id: str,
    *,
    selected: bool,
    decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = candidate["result"]
    return {
        "logical_key": candidate["candidate_id"],
        "phase": phase,
        "variable_id": variable_id,
        "candidate_value": json.dumps(candidate["value"], ensure_ascii=False),
        "selected": selected,
        "selection_metric": "rank_ic_only",
        "mean_rank_ic": result["summary"].get("mean_rank_ic"),
        "positive_fraction": result["summary"].get("rank_ic_positive_fraction"),
        "rank_ic_std": result["summary"].get("rank_ic_std"),
        "eras": json.dumps(result.get("eras", []), ensure_ascii=False),
        "reason": next(
            (
                row["reason"] for row in (decision or {}).get("candidates", [])
                if row["candidate_id"] == candidate["candidate_id"]
            ),
            "Baseline inicial.",
        ),
        "source": result.get("source"),
        "elapsed_seconds": result.get("elapsed_seconds"),
    }


def _shuffle_targets(targets: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = targets.copy()
    columns = ["forward_return", "forward_benchmark_return", "forward_excess_return"]
    for _, indices in frame.groupby("snapshot_date").groups.items():
        positions = np.asarray(list(indices))
        frame.loc[positions, columns] = frame.loc[positions, columns].to_numpy()[
            rng.permutation(len(positions))
        ]
    return frame


def _robustness_base(
    diagnostics: pd.DataFrame,
    scores: pd.DataFrame,
    targets: pd.DataFrame,
    prepared: Path,
    evidence: Path,
    values: Mapping[str, Any],
    winner_result: Mapping[str, Any],
    *,
    dev: bool,
) -> dict[str, Any]:
    """Construye el bloque de robustez sin seeds ni placebos (esos requieren reentrenar)."""
    from module.research.robustness import bootstrap_and_eras, random_portfolios, score_permutation

    iterations = 199 if dev else 9_999
    bootstrap_iterations = 200 if dev else 2_000
    random_iterations = 100 if dev else 1_000
    return {
        "scientific_selection_effect": "none",
        "dev_only_not_scientific": dev,
        "bootstrap_and_era_exclusion": bootstrap_and_eras(
            diagnostics, iterations=bootstrap_iterations,
        ),
        "permutation": score_permutation(
            scores, targets, iterations=iterations,
            minimum_cross_section=3 if dev else 8,
        ),
        "random_portfolios": random_portfolios(
            prepared, evidence, values, simulations=random_iterations,
        ),
        "seeds": [],
        "label_placebos": [],
        "agent_rank_ic": _agent_summary(diagnostics),
        "meta_weight_stability": _meta_stability(evidence / "meta_weights.parquet"),
        "known_stress_not_selection": winner_result.get("known_stress_not_selection", []),
    }


def _seed_dispersion(
    winner: Mapping[str, Any], seeds: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Rango de resultados entre semillas: la prueba directa de la palabra «estable».

    El Rank-IC apenas dependía de la semilla, pero el alfa de una cartera concentrada llegaba a
    cambiar de signo. Con el ensemble de semillas dentro de cada agente esta dispersión debería
    haberse reducido; si el rango de alfa sigue cruzando cero, hay que saberlo y decirlo, porque el
    tribunal preguntará por la reproducibilidad del resultado económico.
    """
    summaries = [dict(winner.get("summary", {}))] + [dict(row["summary"]) for row in seeds]
    out: dict[str, Any] = {"n_seeds": len(summaries)}
    for metric in ("mean_rank_ic", "geometric_excess_return", "information_ratio", "cagr_portfolio"):
        values = [row.get(metric) for row in summaries]
        finite = [float(value) for value in values if isinstance(value, (int, float)) and np.isfinite(value)]
        if not finite:
            continue
        out[metric] = {
            "min": min(finite), "median": float(np.median(finite)), "max": max(finite),
            "range": max(finite) - min(finite),
            "crosses_zero": bool(min(finite) < 0 < max(finite)),
        }
    alpha = out.get("geometric_excess_return", {})
    out["economic_conclusion_stable"] = bool(alpha and not alpha.get("crosses_zero", True))
    return out


def _agent_summary(diagnostics: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "agent": str(agent),
            "mean_rank_ic": _finite(group["rank_ic"].mean()),
            "positive_fraction": _finite((group["rank_ic"] > 0).mean()),
            "observations": int(group["rank_ic"].notna().sum()),
        }
        for agent, group in diagnostics.groupby("agent")
    ]


def _meta_stability(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    frame = pd.read_parquet(path)
    if frame.empty or not {"snapshot_date", "agent", "weight"}.issubset(frame.columns):
        return {"available": False}
    weights = (
        frame.pivot(index="snapshot_date", columns="agent", values="weight")
        .fillna(0.0)
        .sort_index()
    )
    turnover = weights.diff().abs().sum(axis=1).iloc[1:] / 2.0
    return {
        "available": True,
        "rows": int(len(frame)),
        "mean_concentration": _finite((weights ** 2).sum(axis=1).mean()),
        "mean_turnover": _finite(turnover.mean()) if not turnover.empty else 0.0,
    }


def _key_value(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]


def _utc_now() -> str:
    from module.storage.studies import utc_now
    return utc_now()
