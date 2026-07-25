"""Escritura de la evidencia final del único Model Study."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from module.common.utils import write_json


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Informe del Model Study",
        "",
        f"- Study: `{payload['study_id']}`",
        f"- Estado: `{payload.get('status', 'succeeded')}`",
        f"- Ganador final: `{payload.get('winner_run_id', 'no disponible')}`",
        "- Selección: exclusivamente Rank-IC robusto hasta 2024.",
        "- 2025–2026: estrés conocido, no utilizado para seleccionar.",
        "",
        "## Interpretación",
        "",
        "La robustez, los perfiles y las carteras son evidencia informativa posterior. "
        "No modifican la configuración predictiva ganadora.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_winner(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(dict(payload), path)
