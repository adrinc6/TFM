"""Registro inmutable de ejecuciones y estudios.

Esta capa separa la identidad de un experimento de los directorios temporales que usan
las etapas de datos. Todo resultado publicable se copia a ``results/runs`` con un
manifiesto versionado y se indexa en ``results/registry.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from environment import PROJECT_ROOT, Settings
from module.common.utils import pid_alive, sha256_file, write_json, write_parquet


RESULTS_ROOT = PROJECT_ROOT / "results"
SCHEMA_VERSION = 1

# El lock de fichero protege entre procesos, pero dentro de un mismo proceso todos los hilos
# comparten PID y el esquema PID-vivo no los distingue. Este cerrojo serializa los appends al
# registro entre los hilos del motor multihilo antes de tocar el lock de fichero.
_REGISTRY_THREAD_LOCK = threading.Lock()


def _release_lock(lock: Path, timeout: float = 5.0) -> None:
    """Borra el fichero de bloqueo tolerando la contención de Windows.

    En Windows un ``unlink`` puede fallar con WinError 32 si otro proceso aún tiene un handle
    abierto sobre el fichero (p. ej. lo está leyendo para comprobar el PID dueño). Es transitorio:
    se reintenta unos instantes en vez de propagar y perder el escenario.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            lock.unlink(missing_ok=True)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                return  # mejor dejar un lock huérfano (se auto-limpia) que abortar el escenario
            time.sleep(0.05)


@contextmanager
def _exclusive_file_lock(lock: Path, timeout: float = 30):
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            os.close(descriptor)
            break
        except FileExistsError:
            # Lock ocupado: si el dueño murió, lo reclamamos; si vive, esperamos hasta el timeout.
            try:
                owner = int(lock.read_text(encoding="utf-8").strip())
            except ValueError:
                _release_lock(lock)
                continue
            except PermissionError:
                owner = None  # WinError 32 al leer: otro proceso lo tiene abierto → sigue vivo
            except OSError:
                _release_lock(lock)
                continue
            if owner is not None and not pid_alive(owner):
                _release_lock(lock)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"No se pudo adquirir el bloqueo {lock.name}.")
            time.sleep(0.05)
        except PermissionError:
            # WinError 32 al crear: contención transitoria de Windows, no un lock lógico. Reintentar.
            if time.monotonic() >= deadline:
                raise TimeoutError(f"No se pudo adquirir el bloqueo {lock.name}.")
            time.sleep(0.05)
    try:
        yield
    finally:
        _release_lock(lock)


def canonical_json(value: Any) -> str:
    """Representacion estable para hashes, independiente del orden de diccionarios."""
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def config_hash(
    settings: Settings,
    *,
    run_kind: str,
    mode: str,
    stages: Iterable[str],
    intent: Mapping[str, Any] | None = None,
    grid_definition: Mapping[str, Any] | None = None,
    inputs: Mapping[str, str] | None = None,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "settings": asdict(settings),
        "run_kind": run_kind,
        "mode": mode,
        "stages": list(stages),
        "intent": dict(intent or {}),
        "grid_definition": dict(grid_definition or {}),
        "inputs": dict(inputs or {}),
        "python": platform.python_version(),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def execution_hash(
    settings: Settings, *, run_kind: str, mode: str, stages: Iterable[str], inputs: Mapping[str, str]
) -> str:
    """Identidad de cálculo reutilizable, sin etiqueta ni study de origen."""
    # ``run_kind`` describe el papel del resultado (scenario/final/stress), no el cálculo. Se
    # conserva en la firma pública por compatibilidad, pero se normaliza para poder promover un
    # ganador ya calculado sin reentrenarlo.
    _ = run_kind
    return config_hash(settings, run_kind="calculation", mode=mode, stages=stages,
                       intent={}, inputs=inputs)


def git_revision() -> str | None:
    """Devuelve el commit actual sin convertir un repositorio no confiable en error."""
    try:
        result = subprocess.run(
            ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=False, timeout=3,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _run_id(runs_root: Path, full_hash: str) -> str:
    day = datetime.now(UTC).strftime("%Y%m%d")
    base = f"{day}--{full_hash[:12]}"
    candidate = base
    repetition = 2
    while (runs_root / candidate).exists():
        candidate = f"{base}--r{repetition:02d}"
        repetition += 1
    return candidate


def _input_fingerprints(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items() if path.exists()}


class ResultsStore:
    """Crea, publica y consulta runs y studies sin sobrescribir resultados."""

    def __init__(self, root: Path = RESULTS_ROOT) -> None:
        self.root = Path(root)
        self.runs_root = self.root / "runs"
        self.studies_root = self.root / "studies"
        self.registry_path = self.root / "registry.jsonl"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.studies_root.mkdir(parents=True, exist_ok=True)

    def create_run(
        self,
        settings: Settings,
        *,
        run_kind: str,
        mode: str,
        stages: Iterable[str],
        label: str,
        description: str = "",
        tags: Iterable[str] = (),
        study_id: str | None = None,
        grid_definition: Mapping[str, Any] | None = None,
        input_paths: Mapping[str, Path] | None = None,
    ) -> tuple[str, Path, dict[str, Any]]:
        inputs = _input_fingerprints(input_paths or {})
        intent = {"label": label, "description": description, "tags": list(tags), "study_id": study_id}
        digest = config_hash(settings, run_kind=run_kind, mode=mode, stages=stages,
                             intent=intent, grid_definition=grid_definition, inputs=inputs)
        reusable_digest = execution_hash(settings, run_kind=run_kind, mode=mode, stages=stages, inputs=inputs)
        run_id = _run_id(self.runs_root, digest)
        run_dir = self.runs_root / run_id
        artifacts = run_dir / "artifacts"
        artifacts.mkdir(parents=True)
        now = datetime.now(UTC).isoformat()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "config_hash": digest,
            "execution_hash": reusable_digest,
            "status": "queued",
            "run_kind": run_kind,
            "created_at_utc": now,
            "started_at_utc": None,
            "finished_at_utc": None,
            "intent": {**intent, "parent_run_ids": []},
            "execution": {"mode": mode, "scope": settings.run_scope, "stages": list(stages)},
            "effective_settings": asdict(settings),
            "grid_definition": dict(grid_definition or {}),
            "inputs": inputs,
            "environment": {"git_revision": git_revision(), "python_version": sys.version,
                            "platform": platform.platform()},
            "outputs": {"artifacts": [], "summary": {}},
            "error": None,
        }
        write_json(manifest, run_dir / "run_manifest.json")
        write_json(asdict(settings), run_dir / "config.json")
        write_json({"status": "queued", "updated_at_utc": now}, run_dir / "status.json")
        return run_id, run_dir, manifest

    def set_status(self, run_dir: Path, status: str, error: str | None = None) -> None:
        path = run_dir / "run_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        now = datetime.now(UTC).isoformat()
        manifest["status"] = status
        if status == "running":
            manifest["started_at_utc"] = now
        if status in {"succeeded", "failed", "cancelled"}:
            manifest["finished_at_utc"] = now
        manifest["error"] = error
        write_json(manifest, path)
        write_json({"status": status, "updated_at_utc": now, "error": error}, run_dir / "status.json")

    def record_stage_timing(self, run_dir: Path, stage: str, seconds: float,
                            source: str | None = None, stage_key: str | None = None) -> None:
        path = run_dir / "run_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        execution = manifest.setdefault("execution", {})
        execution.setdefault("stage_timings_seconds", {})[stage] = round(seconds, 3)
        if source is not None:
            # "computed" (recalculada) vs "recycled" (restaurada de data/recycle): permite a la app
            # mostrar el coste real de cada etapa distinguiéndolo del reciclado.
            execution.setdefault("stage_source", {})[stage] = source
        if stage_key is not None:
            # Clave de reciclado de esta etapa (settings acotados + inputs + codigo de la propia
            # etapa): depende solo de lo que afecta a esta etapa, no del resto del pipeline. Permite
            # auditar por que una etapa se reciclo o no sin recalcular nada.
            execution.setdefault("stage_keys", {})[stage] = stage_key
        write_json(manifest, path)

    def record_telemetry(self, run_dir: Path, telemetry: Mapping[str, Any]) -> None:
        """Persiste telemetría observada sin formar parte de la identidad del cálculo."""
        path = run_dir / "run_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.setdefault("execution", {})["telemetry"] = dict(telemetry)
        write_json(manifest, path)

    def publish_artifacts(self, run_dir: Path, source_dir: Path) -> dict[str, Any]:
        """Copia los artefactos relevantes de un run de agentes a la carpeta inmutable."""
        source_dir = Path(source_dir)
        target = run_dir / "artifacts"
        wanted = (
            "backtest_summary.json", "agent_scores.parquet", "meta_weights.parquet",
            "rank_ic_diagnostics.parquet", "annual_metrics.parquet", "positions.parquet",
            "orders.parquet", "equity.parquet", "model_feature_attribution.parquet",
            "agent_local_attribution.parquet", "feature_diagnostics.parquet", "feature_catalog.json",
            "profile_results.json", "robustness.json", "manifest.json",
        )
        published: list[str] = []
        for name in wanted:
            source = source_dir / name
            if source.exists():
                shutil.copy2(source, target / name)
                published.append(name)
        # Los precios PIT son necesarios para reproducir en el dashboard la trayectoria de un
        # ticker y sus compras/ventas; se copian desde el processed asociado al run de agentes.
        price_source = source_dir.parent.parent / "asset_price_point_in_time.parquet"
        if not price_source.exists():
            price_source = source_dir / "asset_price_point_in_time.parquet"
        if price_source.exists():
            shutil.copy2(price_source, target / "asset_price_point_in_time.parquet")
            published.append("asset_price_point_in_time.parquet")
        panel_source = source_dir.parent.parent / "panel_point_in_time.parquet"
        if not panel_source.exists():
            panel_source = source_dir / "stock_panel.parquet"
        if panel_source.exists():
            # El panel completo conserva ratios, precios y fechas de disponibilidad sin depender
            # de la carpeta mutable data/processed al consultar resultados antiguos.
            shutil.copy2(panel_source, target / "stock_panel.parquet")
            published.append("stock_panel.parquet")
        targets_source = source_dir.parent.parent / "targets_forward_3m.parquet"
        if targets_source.exists():
            shutil.copy2(targets_source, target / "targets_forward_3m.parquet")
            published.append("targets_forward_3m.parquet")
        self._materialize_lifecycle(target)
        self._materialize_learning_summary(target)
        if (target / "backtest_summary.json").exists():
            summary = json.loads((target / "backtest_summary.json").read_text(encoding="utf-8"))
        else:
            summary = {}
        path = run_dir / "run_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["outputs"] = {"artifacts": published, "summary": summary}
        write_json(manifest, path)
        return summary

    @staticmethod
    def _materialize_lifecycle(artifacts: Path) -> None:
        """Materializa precio de entrada/valoracion sin inventar P&L monetario."""
        positions_path, orders_path = artifacts / "positions.parquet", artifacts / "orders.parquet"
        if not positions_path.exists() or not orders_path.exists():
            return
        positions = pd.read_parquet(positions_path)
        orders = pd.read_parquet(orders_path)
        if positions.empty:
            return
        positions["snapshot_date"] = pd.to_datetime(positions["snapshot_date"])
        orders["snapshot_date"] = pd.to_datetime(orders["snapshot_date"])
        buy_prices = (orders.loc[orders.get("side", pd.Series(dtype=str)) == "buy"]
                      [["ticker", "snapshot_date", "price"]].dropna().sort_values("snapshot_date"))
        entry_map: dict[tuple[str, str], float] = {}
        for row in buy_prices.itertuples(index=False):
            entry_map[(row.ticker, str(row.snapshot_date.date()))] = float(row.price)
        # `positions.parquet` ya trae entry_price/valuation_price por snapshot (precio PIT real,
        # incluidos los snapshots en que solo se mantiene la posicion sin orden ese dia). Solo se
        # rellena entry_price si faltase, sin pisar el valuation_price ya correcto por fecha.
        if "entry_price" not in positions or positions["entry_price"].isna().any():
            fallback_entry = [entry_map.get((row.ticker, str(pd.Timestamp(row.entry_date).date())))
                              for row in positions.itertuples(index=False)]
            positions["entry_price"] = positions.get("entry_price", pd.Series(fallback_entry)).fillna(
                pd.Series(fallback_entry, index=positions.index))
        positions["unrealized_return_pct"] = ((positions["valuation_price"] / positions["entry_price"] - 1) * 100)
        write_parquet(positions, artifacts / "position_lifecycle.parquet")

    @staticmethod
    def _materialize_learning_summary(artifacts: Path) -> None:
        """Precalcula el resumen de rank-IC por agente que el dashboard mostraba al vuelo.

        Es aditivo: si no existe, el endpoint /api/learning lo recompone desde el parquet.
        """
        diagnostics_path = artifacts / "rank_ic_diagnostics.parquet"
        if not diagnostics_path.exists():
            return
        frame = pd.read_parquet(diagnostics_path)
        if frame.empty or "agent" not in frame.columns:
            return
        summary = []
        for agent, group in frame.groupby("agent"):
            values = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
            summary.append({
                "agent": agent,
                "mean_rank_ic": float(values.mean()) if not values.empty else None,
                "positive_fraction": float((values > 0).mean()) if not values.empty else None,
                "rank_ic_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "n_cohorts": int(len(values)),
            })
        write_json({"summary": summary}, artifacts / "learning_summary.json")

    def complete(self, run_dir: Path, summary: Mapping[str, Any] | None = None) -> None:
        self.set_status(run_dir, "succeeded")
        self._append_registry(run_dir, dict(summary or {}))

    def fail(self, run_dir: Path, error: Exception | str) -> None:
        self.set_status(run_dir, "failed", str(error))
        self._append_registry(run_dir, {})

    def _append_registry(self, run_dir: Path, summary: Mapping[str, Any]) -> None:
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        entry = {
            "run_id": manifest["run_id"], "config_hash": manifest["config_hash"],
            "execution_hash": manifest.get("execution_hash"),
            "status": manifest["status"], "run_kind": manifest["run_kind"],
            "created_at_utc": manifest["created_at_utc"], "study_id": manifest["intent"].get("study_id"),
            "label": manifest["intent"].get("label"), "tags": manifest["intent"].get("tags", []),
            "path": str(run_dir.resolve().relative_to(self.root.resolve())).replace("\\", "/"),
            "summary": dict(summary),
        }
        lock = self.registry_path.with_suffix(".lock")
        with _REGISTRY_THREAD_LOCK, _exclusive_file_lock(lock):
            with self.registry_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(entry) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def create_study(self, payload: Mapping[str, Any]) -> tuple[str, Path]:
        label = str(payload.get("name") or "study").strip().lower().replace(" ", "-")
        digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:12]
        study_id = f"{datetime.now(UTC).strftime('%Y%m%d')}--{label}--{digest}"
        path = self.studies_root / study_id
        suffix = 2
        while path.exists():
            path = self.studies_root / f"{study_id}--r{suffix:02d}"
            suffix += 1
        path.mkdir(parents=True)
        manifest = {"schema_version": SCHEMA_VERSION, "study_id": path.name,
                    "created_at_utc": datetime.now(UTC).isoformat(), "status": "queued", **dict(payload)}
        write_json(manifest, path / "study_manifest.json")
        write_json({"run_ids": []}, path / "run_ids.json")
        return path.name, path

    def resume_study(self, study_id: str) -> tuple[str, Path]:
        """Abre un study incompleto; los runs ya terminados se deduplican por execution_hash."""
        path = (self.studies_root / study_id).resolve()
        try:
            path.relative_to(self.studies_root.resolve())
        except ValueError as exc:
            raise ValueError("Study fuera del registro de resultados.") from exc
        manifest_path = path / "study_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No existe el study {study_id}.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "succeeded":
            raise ValueError("Un study completado no se puede reanudar.")
        self.update_study_status(path.name, "running")
        return path.name, path

    def update_study_status(self, study_id: str, status: str, **extra: Any) -> None:
        path = self.studies_root / study_id / "study_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.update(extra)
        manifest["status"] = status
        manifest["updated_at_utc"] = datetime.now(UTC).isoformat()
        write_json(manifest, path)

    def stop_study(self, study_id: str) -> dict[str, Any]:
        """Parada dura de un study: borra los runs a medias, lo deja ``cancelled`` (reanudable) y
        mata su proceso — que, al correr como hilo dentro del servidor, ES el servidor.

        El botón "Parar" del dashboard llama aquí. Como el study comparte proceso con el servidor,
        matarlo tira también el dashboard: hay que relanzar ``python main.py`` después. Por eso el
        estado terminal, el borrado de runs incompletos y la liberación del lock se hacen ANTES de
        matar (si no, morirían con el proceso), y el kill se difiere a un hilo para que la respuesta
        HTTP del ``/stop`` llegue al navegador antes de que el servidor caiga. ``data/recycle`` nunca
        se toca; los runs en curso quedan incompletos, así que se eliminan y se regeneran al reanudar.
        """
        import socket
        path = (self.studies_root / study_id).resolve()
        try:
            path.relative_to(self.studies_root.resolve())
        except ValueError as exc:
            raise ValueError("Study fuera del registro de resultados.") from exc
        manifest_path = path / "study_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No existe el study {study_id}.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # 1) Borra los runs a medias del study (incompletos = sin valor, se rehacen al reanudar).
        removed_runs: list[str] = []
        if self.runs_root.exists():
            for run_dir in self.runs_root.iterdir():
                run_manifest = run_dir / "run_manifest.json"
                if not run_manifest.exists():
                    continue
                try:
                    payload = json.loads(run_manifest.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if payload.get("intent", {}).get("study_id") != study_id:
                    continue
                if payload.get("status") in {"queued", "running"}:
                    shutil.rmtree(run_dir, ignore_errors=True)
                    removed_runs.append(run_dir.name)

        # 2) Estado terminal + liberar el lock global, ANTES de matar (si no, mueren con el proceso).
        self.update_study_status(study_id, "cancelled")
        lock_path = self.root / "full_study.lock"
        lock_path.unlink(missing_ok=True)

        # 3) Mata el árbol del proceso (servidor + workers) en un hilo diferido, para que la
        #    respuesta HTTP salga antes. En daemon-thread, owner_pid es el propio servidor.
        killed_pid = None
        pid = manifest.get("owner_pid")
        host = manifest.get("owner_host")
        if isinstance(pid, int) and host == socket.gethostname() and pid_alive(pid):
            killed_pid = pid

            def _kill_tree() -> None:
                import psutil
                time.sleep(0.5)  # deja responder al handler HTTP antes de caer
                try:
                    parent = psutil.Process(pid)
                    for proc in parent.children(recursive=True) + [parent]:
                        try:
                            proc.kill()
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass

            threading.Thread(target=_kill_tree, name=f"stop-study-{study_id}", daemon=True).start()
        return {"killed_pid": killed_pid, "removed_runs": removed_runs}

    def add_to_study(self, study_id: str, run_id: str, *, reused: bool = False) -> None:
        path = self.studies_root / study_id / "run_ids.json"
        with _exclusive_file_lock(path.with_suffix(".lock")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if run_id not in data["run_ids"]:
                data["run_ids"].append(run_id)
            if reused and run_id not in data.setdefault("reused_run_ids", []):
                data["reused_run_ids"].append(run_id)
            write_json(data, path)

    def find_completed_execution(self, reusable_digest: str) -> str | None:
        """Devuelve un run terminado con mismos datos, código y configuración efectiva.

        El ``execution_hash`` ya está indexado en cada línea de ``registry.jsonl``, así que se
        resuelve por escaneo del índice sin reabrir cada ``run_manifest.json`` (O(runs) lecturas
        de disco → una sola lectura del índice). Solo se cae al manifiesto para entradas antiguas
        que no traigan el hash indexado (compatibilidad hacia atrás).
        """
        for entry in reversed(list_registry(self.root)):
            if entry.get("status") != "succeeded":
                continue
            digest = entry.get("execution_hash")
            if digest is None:
                manifest_path = self.runs_root / str(entry["run_id"]) / "run_manifest.json"
                if not manifest_path.exists():
                    continue
                digest = json.loads(manifest_path.read_text(encoding="utf-8")).get("execution_hash")
            if digest == reusable_digest:
                return str(entry["run_id"])
        return None


def list_registry(root: Path = RESULTS_ROOT) -> list[dict[str, Any]]:
    path = Path(root) / "registry.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
