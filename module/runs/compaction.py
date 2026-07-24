"""Compactación histórica segura y explícita.

El comando es siempre ``dry-run`` salvo que se pase ``--apply``. Solo propone eliminar un
artefacto pesado cuando existe otra copia byte-a-byte idéntica dentro de un run preservado del
mismo study; nunca limpia por antigüedad ni toca la caché de reciclado.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from module.common.utils import sha256_file, write_json
from module.runs.results_store import RESULTS_ROOT, ResultsStore

HEAVY_DUPLICATE_NAMES = frozenset({
    "agent_scores.parquet",
    "model_feature_attribution.parquet",
    "agent_local_attribution.parquet",
    "feature_diagnostics.parquet",
    "asset_price_point_in_time.parquet",
    "benchmark_point_in_time.parquet",
    "stock_panel.parquet",
    "panel_point_in_time.parquet",
    "features_point_in_time.parquet",
    "targets_forward.parquet",
    "targets_forward_3m.parquet",
})


def compact_historical_study(
    study_id: str, *, apply: bool = False, store: ResultsStore | None = None,
) -> dict[str, Any]:
    """Informa o elimina duplicados comprobados, preservando toda la evidencia decisional."""
    store = store or ResultsStore()
    study_dir = (store.studies_root / study_id).resolve()
    try:
        study_dir.relative_to(store.studies_root.resolve())
    except ValueError as exc:
        raise ValueError("Study fuera del registro de resultados.") from exc
    decision_path = study_dir / "decision.json"
    run_ids_path = study_dir / "run_ids.json"
    if not decision_path.exists() or not run_ids_path.exists():
        raise FileNotFoundError("El study debe tener decision.json y run_ids.json antes de compactar.")

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    members_payload = json.loads(run_ids_path.read_text(encoding="utf-8"))
    members = {
        str(run_id) for run_id in (
            list(members_payload.get("run_ids", []))
            + list(members_payload.get("reused_run_ids", []))
        )
    }
    preserved = _preserved_run_ids(decision)
    missing_preserved = sorted(run_id for run_id in preserved if run_id not in members)
    if missing_preserved:
        raise RuntimeError(
            "La decisión referencia runs que no figuran en el study: " + ", ".join(missing_preserved)
        )

    canonical: dict[tuple[str, str], Path] = {}
    for run_id in sorted(preserved):
        artifacts = store.runs_root / run_id / "artifacts"
        for name in HEAVY_DUPLICATE_NAMES:
            path = artifacts / name
            if path.is_file():
                canonical[(name, sha256_file(path))] = path

    candidates: list[dict[str, Any]] = []
    total_bytes = 0
    for run_id in sorted(members - preserved):
        artifacts = store.runs_root / run_id / "artifacts"
        if not artifacts.exists():
            continue
        for name in sorted(HEAVY_DUPLICATE_NAMES):
            path = artifacts / name
            if not path.is_file():
                continue
            digest = sha256_file(path)
            duplicate = canonical.get((name, digest))
            if duplicate is None:
                continue
            size = path.stat().st_size
            candidates.append({
                "run_id": run_id,
                "artifact": name,
                "bytes": size,
                "sha256": digest,
                "identical_copy": str(duplicate),
            })
            total_bytes += size

    removed: list[dict[str, Any]] = []
    if apply:
        # Se vuelve a comprobar cada hash inmediatamente antes de borrar.
        for item in candidates:
            path = store.runs_root / item["run_id"] / "artifacts" / item["artifact"]
            reference = Path(item["identical_copy"])
            if (
                path.is_file()
                and reference.is_file()
                and sha256_file(path) == item["sha256"] == sha256_file(reference)
            ):
                path.unlink()
                removed.append(item)

    report = {
        "study_id": study_id,
        "mode": "apply" if apply else "dry-run",
        "preserved_run_ids": sorted(preserved),
        "candidate_files": candidates,
        "candidate_bytes": total_bytes,
        "removed_files": removed,
        "removed_bytes": sum(int(item["bytes"]) for item in removed),
        "cache_modified": False,
        "selection_artifacts_modified": False,
    }
    report_name = "historical_compaction_applied.json" if apply else "historical_compaction_dry_run.json"
    write_json(report, study_dir / report_name)
    return report


def _preserved_run_ids(decision: dict[str, Any]) -> set[str]:
    keys = (
        "final_run_id", "model_final_run_id", "finalist_run_id", "portfolio_final_run_id",
        "recommended_run_id",
    )
    result = {str(decision[key]) for key in keys if decision.get(key)}
    result.update(
        str(run_id) for run_id in dict(decision.get("profile_run_ids") or {}).values() if run_id
    )
    for candidate in decision.get("candidates", []):
        if candidate.get("selected") and candidate.get("run_id"):
            result.add(str(candidate["run_id"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compacta duplicados históricos comprobados.")
    parser.add_argument("study_id")
    parser.add_argument(
        "--apply", action="store_true",
        help="Aplica el plan. Sin esta opción solo genera un dry-run.",
    )
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()
    report = compact_historical_study(
        args.study_id, apply=args.apply, store=ResultsStore(args.results_root),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
