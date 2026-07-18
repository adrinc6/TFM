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
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from environment import PROJECT_ROOT, Settings
from module.common.utils import sha256_file, write_json, write_parquet


RESULTS_ROOT = PROJECT_ROOT / "results"
RUNS_ROOT = RESULTS_ROOT / "runs"
STUDIES_ROOT = RESULTS_ROOT / "studies"
REGISTRY_PATH = RESULTS_ROOT / "registry.jsonl"
SCHEMA_VERSION = 1


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
        "git_revision": git_revision(),
        "python": platform.python_version(),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def execution_hash(
    settings: Settings, *, run_kind: str, mode: str, stages: Iterable[str], inputs: Mapping[str, str]
) -> str:
    """Identidad de cálculo reutilizable, sin etiqueta ni study de origen."""
    return config_hash(settings, run_kind=run_kind, mode=mode, stages=stages, intent={}, inputs=inputs)


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


def _safe_relative(path: Path) -> str:
    return str(path.resolve().relative_to(RESULTS_ROOT.resolve())).replace("\\", "/")


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

    def publish_artifacts(self, run_dir: Path, source_dir: Path) -> dict[str, Any]:
        """Copia los artefactos relevantes de un run de agentes a la carpeta inmutable."""
        source_dir = Path(source_dir)
        target = run_dir / "artifacts"
        wanted = (
            "backtest_summary.json", "agent_scores.parquet", "meta_weights.parquet",
            "rank_ic_diagnostics.parquet", "annual_metrics.parquet", "positions.parquet",
            "orders.parquet", "equity.parquet", "model_feature_attribution.parquet",
            "agent_local_attribution.parquet",
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
        if price_source.exists():
            shutil.copy2(price_source, target / "asset_price_point_in_time.parquet")
            published.append("asset_price_point_in_time.parquet")
        panel_source = source_dir.parent.parent / "panel_point_in_time.parquet"
        if panel_source.exists():
            # El panel completo conserva ratios, precios y fechas de disponibilidad sin depender
            # de la carpeta mutable data/processed al consultar resultados antiguos.
            shutil.copy2(panel_source, target / "stock_panel.parquet")
            published.append("stock_panel.parquet")
        for name in ("ranking_by_snapshot.csv", "ranking_by_agents.csv", "positions_history.csv",
                     "orders_history.csv", "annual_metrics.csv"):
            source = source_dir / name
            if source.exists():
                csv_dir = target / "csv"
                csv_dir.mkdir(exist_ok=True)
                shutil.copy2(source, csv_dir / name)
                published.append(f"csv/{name}")
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
        positions["entry_price"] = [entry_map.get((row.ticker, str(pd.Timestamp(row.entry_date).date())))
                                    for row in positions.itertuples(index=False)]
        latest_prices = (orders[["ticker", "snapshot_date", "price"]].dropna().sort_values("snapshot_date")
                         .drop_duplicates(["ticker", "snapshot_date"], keep="last"))
        price_by_snapshot = {(row.ticker, row.snapshot_date): float(row.price)
                             for row in latest_prices.itertuples(index=False)}
        positions["valuation_price"] = [price_by_snapshot.get((row.ticker, row.snapshot_date))
                                         for row in positions.itertuples(index=False)]
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
        with self.registry_path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(entry) + "\n")

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

    def add_to_study(self, study_id: str, run_id: str, *, reused: bool = False) -> None:
        path = self.studies_root / study_id / "run_ids.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if run_id not in data["run_ids"]:
            data["run_ids"].append(run_id)
        if reused:
            data.setdefault("reused_run_ids", []).append(run_id)
        write_json(data, path)

    def find_completed_execution(self, reusable_digest: str) -> str | None:
        """Devuelve un run terminado con mismos datos, código y configuración efectiva."""
        for entry in reversed(list_registry(self.root)):
            if entry.get("status") != "succeeded":
                continue
            manifest_path = self.runs_root / str(entry["run_id"]) / "run_manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("execution_hash") == reusable_digest:
                return str(entry["run_id"])
        return None


def list_registry(root: Path = RESULTS_ROOT) -> list[dict[str, Any]]:
    path = Path(root) / "registry.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
