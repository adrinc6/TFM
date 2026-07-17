"""B2: tratamiento de la etiqueta de entrenamiento (none / winsor / rank)."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from environment import Settings
from module.agents import _transform_label


def _train() -> pd.DataFrame:
    # Dos snapshots, con un outlier gigante en el primero.
    return pd.DataFrame({
        "snapshot_date": ["d1", "d1", "d1", "d1", "d2", "d2", "d2", "d2"],
        "forward_excess_return_3m": [0.01, 0.02, 0.03, 5.00, -0.01, 0.00, 0.02, 0.04],
    })


def test_none_returns_raw_values() -> None:
    settings = Settings(run_scope="dev", label_transform="none")
    out = _transform_label(_train(), "forward_excess_return_3m", settings)
    assert list(out) == [0.01, 0.02, 0.03, 5.00, -0.01, 0.00, 0.02, 0.04]


def test_winsor_clips_the_outlier() -> None:
    settings = Settings(run_scope="dev", label_transform="winsor", label_winsor_pct=0.10)
    out = _transform_label(_train(), "forward_excess_return_3m", settings)
    # El 5.00 se recorta al percentil 90 global; deja de ser 5.
    assert out.max() < 5.00
    # Los valores centrales no cambian.
    assert out.iloc[0] == 0.01


def test_rank_is_cross_sectional_percentile() -> None:
    settings = Settings(run_scope="dev", label_transform="rank")
    out = _transform_label(_train(), "forward_excess_return_3m", settings)
    # Dentro de d1: 0.01<0.02<0.03<5.00 -> percentiles 0.25/0.5/0.75/1.0
    assert list(out.iloc[:4].round(3)) == [0.25, 0.5, 0.75, 1.0]
    # El outlier pasa a ser simplemente "el mayor" (1.0), su magnitud deja de importar.
    assert out.iloc[3] == 1.0
