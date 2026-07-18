"""Orquestador multi-eje: elige el mejor nivel de cada eje + artefactos, y reserva anios."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from module.experiments import (
    RESERVED_ERA_YEARS,
    SELECTION_UNTIL_YEAR,
    _best_level,
    _meta_final_ic,
    _phase2_specs,
    _reserved_era_validation,
    decide_best_config,
)


def _write_scenario(root: Path, name: str, rank_ics: list[float], start: str = "2016-02-15") -> None:
    """Crea un escenario con diagnostics de meta_final (una cohorte trimestral por valor)."""
    run_dir = root / name / "agents" / f"lgbm-{name}"
    run_dir.mkdir(parents=True)
    dates = pd.date_range(start, periods=len(rank_ics), freq="QS")
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


def test_selects_best_level_of_each_axis(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 16
    base = list(rng.normal(0.005, 0.03, n))
    # baseline mediocre; en el eje ventana, el nivel "train_8y" es sistematicamente mejor.
    _write_scenario(tmp_path, "baseline", base)
    _write_scenario(tmp_path, "train_8y", [b + 0.03 for b in base])   # mejor nivel del eje
    _write_scenario(tmp_path, "train_12y", [b - 0.01 for b in base])  # peor
    # eje profundidad: depth_6 mejor que baseline
    _write_scenario(tmp_path, "depth_6", [b + 0.02 for b in base])
    _write_scenario(tmp_path, "depth_3", [b - 0.02 for b in base])
    # un artefacto util
    _write_scenario(tmp_path, "artifact_sector", [b + 0.04 for b in base])

    artifacts = {"sector": {"neutralize_by_sector": True}}
    axes = {
        "train_lookback_years": {
            "baseline": {},
            "train_8y": {"train_lookback_years": 8},
            "train_12y": {"train_lookback_years": 12},
        },
        "lgbm_max_depth": {
            "baseline": {},
            "depth_3": {"lgbm_max_depth": 3},
            "depth_6": {"lgbm_max_depth": 6},
        },
    }
    decision = decide_best_config(tmp_path, artifacts, axes, until_year=None)

    assert decision["axis_choices"]["train_lookback_years"]["chosen"] == "train_8y"
    assert decision["axis_choices"]["lgbm_max_depth"]["chosen"] == "depth_6"
    # la config final combina el mejor nivel de cada eje + el artefacto aceptado
    assert decision["config_final"] == {
        "train_lookback_years": 8,
        "lgbm_max_depth": 6,
        "neutralize_by_sector": True,
    }


def test_reserved_era_is_excluded_from_selection(tmp_path: Path) -> None:
    # 12 cohortes trimestrales desde 2023: 2023-2024 (seleccion) y 2025-2026 (reservada).
    # En la era reservada el nivel A es peor, en la de seleccion es mejor. La seleccion debe
    # ignorar la reservada y elegir A pese a su mal comportamiento en 2025-2026.
    n = 12
    # QS desde 2023-01-15: 7 cohortes en 2023-24 (seleccion) + 5 en 2025-26 (reservada).
    good_selection = [0.05] * 7 + [-0.20] * 5   # 2023-24 bien, 2025-26 fatal
    flat = [0.0] * n
    _write_scenario(tmp_path, "baseline", flat, start="2023-01-15")
    _write_scenario(tmp_path, "level_a", good_selection, start="2023-01-15")

    axes = {"axis": {"baseline": {}, "level_a": {"train_lookback_years": 8}}}
    decision = decide_best_config(tmp_path, {}, axes, until_year=2024)
    assert decision["axis_choices"]["axis"]["chosen"] == "level_a"

    # y sin reservar (mirando todo), level_a deja de ganar por el hundimiento de 2025-26.
    decision_all = decide_best_config(tmp_path, {}, axes, until_year=None)
    assert decision_all["axis_choices"]["axis"]["chosen"] == "baseline"


def test_until_year_filters_cohorts(tmp_path: Path) -> None:
    _write_scenario(tmp_path, "s", [0.1] * 12, start="2023-01-15")
    full = _meta_final_ic(tmp_path / "s", until_year=None)
    filtered = _meta_final_ic(tmp_path / "s", until_year=2024)
    assert len(full) == 12
    # QS desde 2023-01-15 arranca en 2023-04-01; 2023-2024 = 7 cohortes
    assert len(filtered) == 7
    assert filtered.index.year.max() == 2024


def test_reserved_era_validation_reads_reserved_years() -> None:
    dates = pd.date_range("2023-01-15", periods=12, freq="QS")   # arranca en 2023-04-01
    diag = pd.DataFrame({
        "agent": "meta_final",
        "prediction_date": [d.date().isoformat() for d in dates],
        "rank_ic": [0.02] * 7 + [0.05] * 5,                       # 7 en seleccion, 5 en reservada
    })
    result = _reserved_era_validation(diag)
    assert result["reserved_years"] == list(RESERVED_ERA_YEARS)
    assert result["n_cohorts"] == 5                               # 2025-2026 = 5 cohortes
    assert result["rank_ic_mean"] == 0.05
    assert abs(result["rank_ic_selection_period"] - 0.02) < 1e-9


def test_phase2_builds_directed_combinations() -> None:
    axis_choices = {
        "train_lookback_years": {
            "chosen": "train_8y",
            "levels": {
                "train_8y": {"mean_rank_ic": 0.03, "positive_fraction": 0.6, "variance": 0.01, "n_cohorts": 16},
                "train_12y": {"mean_rank_ic": 0.02, "positive_fraction": 0.55, "variance": 0.01, "n_cohorts": 16},
            },
            "overrides_by_level": {"train_8y": {"train_lookback_years": 8},
                                   "train_12y": {"train_lookback_years": 12}},
        },
    }
    specs = _phase2_specs(axis_choices, {"neutralize_by_sector": True})
    names = [s.name for s in specs]
    assert "phase2_best" in names
    best = next(s for s in specs if s.name == "phase2_best")
    assert best.overrides == {"train_lookback_years": 8, "neutralize_by_sector": True}
    # variante con el 2º mejor del eje
    second = next(s for s in specs if s.name == "phase2_train_lookback_years_2nd")
    assert second.overrides == {"train_lookback_years": 12, "neutralize_by_sector": True}


def test_selection_until_year_default_reserves_recent_years() -> None:
    assert SELECTION_UNTIL_YEAR == 2024
    assert RESERVED_ERA_YEARS == (2025, 2026)
