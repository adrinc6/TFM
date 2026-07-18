"""Decision automatica de artefactos: acepta los que mejoran el rank-IC de forma estable."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from module.runs.experiments import decide_accepted_artifacts


def _write_scenario_diag(root: Path, name: str, rank_ics: list[float]) -> None:
    """Crea un escenario con un rank_ic_diagnostics de meta_final con los valores dados."""
    run_dir = root / name / "agents" / f"lgbm-{name}"
    run_dir.mkdir(parents=True)
    dates = pd.date_range("2016-02-15", periods=len(rank_ics), freq="QS")
    diag = pd.DataFrame({
        "agent": "meta_final",
        "prediction_date": [d.date().isoformat() for d in dates],
        "label_end_date": [d.date().isoformat() for d in dates],
        "observations": 100,
        "rank_ic": rank_ics,
        "is_quarterly": True,
    })
    diag.to_parquet(run_dir / "rank_ic_diagnostics.parquet")
    (root / name / "scenario_config.json").write_text("{}", encoding="utf-8")


def test_accepts_helpful_and_rejects_useless_artifact(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 20
    base = list(rng.normal(0.01, 0.05, n))
    # helpful: baseline + 0.04 sistematico en cada fecha -> mejor en el 100 % de las fechas.
    helpful = [b + 0.04 for b in base]
    # useless: mismo que baseline con ruido centrado en 0 -> ni mejor ni peor.
    useless = [b + rng.normal(0, 0.05) for b in base]

    _write_scenario_diag(tmp_path, "baseline", base)
    _write_scenario_diag(tmp_path, "artifact_helpful", helpful)
    _write_scenario_diag(tmp_path, "artifact_useless", useless)

    artifacts = {"helpful": {"moving_averages": True}, "useless": {"regime_extended": True}}
    decision = decide_accepted_artifacts(tmp_path, artifacts)

    assert "helpful" in decision["accepted"]
    assert "useless" not in decision["accepted"]
    # la config final activa el flag del aceptado
    assert decision["config_final"] == {"moving_averages": True}


def test_no_baseline_returns_error(tmp_path: Path) -> None:
    decision = decide_accepted_artifacts(tmp_path, {"x": {"moving_averages": True}})
    assert decision["accepted"] == []
    assert "error" in decision
