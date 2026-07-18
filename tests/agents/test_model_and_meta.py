"""Correcciones del plan revisado: meta_final diagnosticado, motores/objetivos, huellas."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from module.agents import _prepare_training, build_agent_scores
from module.experiments import stage_fingerprint


def _train_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for date in ("2000-01-15", "2000-02-15"):
        for i in range(20):
            rows.append({"snapshot_date": date, "forward_excess_return_3m": rng.normal()})
    return pd.DataFrame(rows)


def test_rank_regression_label_is_cross_sectional_percentile() -> None:
    """rank_regression: la etiqueta es el percentil del retorno dentro de cada snapshot."""
    from environment import Settings
    settings = Settings(run_scope="dev", objective="rank_regression")
    train, label = _prepare_training(_train_frame(), settings)
    assert len(train) == 40                       # no descarta filas
    assert label.between(0, 1).all()              # percentiles en [0,1]
    assert abs(label.groupby(train["snapshot_date"]).mean().mean() - 0.525) < 0.1


def test_quartile_excludes_middle_from_training() -> None:
    """quartile (ablacion): entrena solo con extremos; el centro se excluye del train."""
    from environment import Settings
    settings = Settings(run_scope="dev", objective="quartile")
    train, label = _prepare_training(_train_frame(), settings)
    assert len(train) < 40                         # el centro se cae
    assert set(label.unique()) <= {0, 1}           # binaria
    # aproximadamente la mitad de cada snapshot (top 25 % + bottom 25 %)
    assert 0.4 < len(train) / 40 < 0.6


def test_meta_final_is_diagnosed(agent_settings) -> None:
    """El diagnostico incluye el meta_final (lo que opera la cartera), no solo los agentes."""
    build_agent_scores(agent_settings)
    run_dir = next((agent_settings.processed_output_dir / "agents").iterdir())
    diag = pd.read_parquet(run_dir / "rank_ic_diagnostics.parquet")
    agents_present = set(diag["agent"].unique())
    assert "meta_final" in agents_present, f"falta meta_final; hay {agents_present}"
    assert "meta_equal_weight" in agents_present


def test_seed_and_objective_change_fingerprint() -> None:
    """Cambiar semilla, objetivo o hiperparametros cambia la huella del run (reproducibilidad)."""
    from environment import Settings
    base = Settings(run_scope="dev")
    for field, value in (("random_seed", 7), ("objective", "quartile"), ("lgbm_max_depth", 6)):
        changed = replace(base, **{field: value})
        assert stage_fingerprint("agents", base) != stage_fingerprint("agents", changed), (
            f"{field} deberia cambiar la huella de agents"
        )
