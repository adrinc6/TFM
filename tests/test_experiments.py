"""Sistema de experimentos — overrides aislados, binding correcto, run_dir, cache key y métricas.

Tests puros: no ejecutan el pipeline ni requieren datos descargados ni modelo entrenado. Verifican
la maquinaria del runner sobre configuración viva y tablas sintéticas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from environment import Settings
from module import ml
from module.backtest import baselines
from module.experiments import Scenario, scoring_cache_key
from module.experiments import runner as exp_runner
from module.experiments.metrics import collect_metrics
from module.experiments.overrides import apply_overrides, split_overrides
from module.strategy import portfolio, sizing


def test_overrides_se_restauran_al_salir():
    """Tras el context manager, todos los globales tocados vuelven a su valor original."""
    base = Settings()
    prior_original = dict(ml.AGENT_PRIOR_WEIGHTS)
    maxw_original = sizing.MAX_POSITION_WEIGHT
    rot_original = portfolio.MIN_ROTATION_ADVANTAGE
    scenario = Scenario(
        name="s",
        why="w",
        overrides={
            "ml.AGENT_PRIOR_WEIGHTS": {"quality_probability": 0.6, "timing_probability": 0.25, "alpha_probability": 0.15},
            "strategy.MAX_POSITION_WEIGHT": 0.22,
            "strategy.MIN_ROTATION_ADVANTAGE": 0.15,
        },
    )
    with apply_overrides(scenario, base):
        assert ml.AGENT_PRIOR_WEIGHTS["quality_probability"] == 0.6
        assert sizing.MAX_POSITION_WEIGHT == 0.22
        assert portfolio.MIN_ROTATION_ADVANTAGE == 0.15
    assert ml.AGENT_PRIOR_WEIGHTS == prior_original
    assert sizing.MAX_POSITION_WEIGHT == maxw_original
    assert portfolio.MIN_ROTATION_ADVANTAGE == rot_original


def test_binding_constante_importada_toca_el_consumidor():
    """MIN_ROTATION_ADVANTAGE se importa en portfolio con `from environment import`; el override
    debe reescribir el nombre en portfolio (no basta con environment)."""
    base = Settings()
    scenario = Scenario(name="s", why="w", overrides={"strategy.MIN_ROTATION_ADVANTAGE": 0.99})
    with apply_overrides(scenario, base):
        assert portfolio.MIN_ROTATION_ADVANTAGE == 0.99


def test_max_portfolio_size_toca_ambos_modulos_y_review_top_n():
    base = Settings()
    scenario = Scenario(name="s", why="w", overrides={"strategy.MAX_PORTFOLIO_SIZE": 15})
    with apply_overrides(scenario, base):
        assert portfolio.MAX_PORTFOLIO_SIZE == 15
        assert baselines.MAX_PORTFOLIO_SIZE == 15
        assert portfolio.REVIEW_TOP_N == 45  # max(15*3, 20)


def test_settings_override_via_replace():
    base = Settings()
    scenario = Scenario(name="s", why="w", overrides={"settings.transaction_cost_bps": 10.0})
    with apply_overrides(scenario, base) as settings:
        assert settings.transaction_cost_bps == 10.0
    assert base.transaction_cost_bps == 5.0  # el base no se muta


def test_run_dir_override():
    base = Settings()
    override = Settings(run_dir_override=str(Path("results") / "experiments" / "x" / "esc"))
    assert override.run_dir == Path("results") / "experiments" / "x" / "esc"
    assert base.run_dir_override is None
    assert "results" in str(base.run_dir)  # sin override, deriva del run_name


def test_split_overrides_valida_nombres():
    with pytest.raises(KeyError):
        split_overrides({"ml.NO_EXISTE": 1})
    with pytest.raises(KeyError):
        split_overrides({"settings.no_existe": 1})
    settings_ovr, ml_ovr, strategy_ovr = split_overrides({
        "settings.transaction_cost_bps": 10.0,
        "ml.RANDOM_STATE": 7,
        "strategy.MAX_POSITION_WEIGHT": 0.2,
    })
    assert settings_ovr == {"transaction_cost_bps": 10.0}
    assert ml_ovr == {"RANDOM_STATE": 7}
    assert strategy_ovr == {"MAX_POSITION_WEIGHT": 0.2}


def test_scenario_rechaza_namespace_desconocido():
    with pytest.raises(ValueError):
        Scenario(name="s", why="w", overrides={"otro.X": 1})


def test_cache_key_estrategia_comparte_ml_no():
    base = Settings()
    baseline = Scenario(name="baseline", why="w")
    solo_estrategia = Scenario(name="e", why="w", overrides={"strategy.MAX_POSITION_WEIGHT": 0.22})
    cambia_ml = Scenario(name="m", why="w", overrides={"ml.RANDOM_STATE": 7})
    cambia_ventana = Scenario(name="v", why="w", overrides={"settings.max_walk_forward_training_years": 3})

    key_base = scoring_cache_key(baseline, base)
    assert scoring_cache_key(solo_estrategia, base) == key_base  # estrategia no re-scorea
    assert scoring_cache_key(cambia_ml, base) != key_base        # ml sí
    assert scoring_cache_key(cambia_ventana, base) != key_base   # ventana ML sí


def test_collect_metrics_tolera_tablas_faltantes(tmp_path):
    """Con outputs mínimos y sin diagnostics.csv, devuelve nan sin romper."""
    outputs = {"portfolio_monthly_summary": {}}
    metrics = collect_metrics(outputs, tmp_path)
    assert metrics["rank_ic_final_mean"] != metrics["rank_ic_final_mean"]  # nan
    assert "cumulative_alpha" in metrics


def test_collect_metrics_lee_rank_ic_y_placebo(tmp_path):
    diag = pd.DataFrame({
        "snapshot_date": ["2019-01-31", "2020-01-31"],
        "rank_ic_final": [0.05, 0.03],
        "rank_ic_final_year_mean": [0.05, 0.03],
        "rank_ic_final_year_tstat": [1.2, 0.8],
    })
    diag.to_csv(tmp_path / "model_walk_forward_diagnostics.csv", index=False)
    outputs = {
        "placebo_summary": pd.DataFrame([{"percentil_real": 0.92}]),
        "baseline_comparison": pd.DataFrame([
            {"estrategia": "Sistema completo (x)", "alpha_acumulada": 0.30},
            {"estrategia": "momentum_only", "alpha_acumulada": 0.20},
            {"estrategia": "equal_weight_universe", "alpha_acumulada": 0.10},
        ]),
        "portfolio_monthly_summary": {"cumulative_alpha": 0.30, "information_ratio": 0.5, "excess_return_t_stat": 1.1},
    }
    metrics = collect_metrics(outputs, tmp_path)
    assert metrics["rank_ic_final_mean"] == pytest.approx(0.04)
    assert metrics["rank_ic_positive_years"] == 2
    assert metrics["placebo_percentile"] == pytest.approx(0.92)
    assert metrics["alpha_vs_best_baseline"] == pytest.approx(0.10)  # 0.30 - 0.20
    assert metrics["cumulative_alpha"] == pytest.approx(0.30)


def test_rank_ic_tstat_promedia_por_ano_no_por_snapshot(tmp_path):
    """Un año con más snapshots no debe pesar más en el t-stat medio: se deduplica a un valor/año.
    Año 2019: 3 snapshots, tstat=3.0; año 2020: 1 snapshot, tstat=1.0. Media por año = 2.0."""
    diag = pd.DataFrame({
        "snapshot_date": ["2019-03-31", "2019-06-30", "2019-09-30", "2020-03-31"],
        "rank_ic_final": [0.05, 0.05, 0.05, 0.03],
        "rank_ic_final_year_mean": [0.05, 0.05, 0.05, 0.03],
        "rank_ic_final_year_tstat": [3.0, 3.0, 3.0, 1.0],
    })
    diag.to_csv(tmp_path / "model_walk_forward_diagnostics.csv", index=False)
    metrics = collect_metrics({"portfolio_monthly_summary": {}}, tmp_path)
    assert metrics["rank_ic_final_tstat"] == pytest.approx(2.0)  # (3.0 + 1.0) / 2, no 10/4=2.5
    assert metrics["rank_ic_positive_years"] == 2


def test_ensure_inputs_corre_solo_las_etapas_que_faltan(tmp_path, monkeypatch):
    """Con prices y master presentes pero sin features, corre solo 'features' (no download ni dataset)."""
    raw, master, processed = tmp_path / "raw", tmp_path / "master", tmp_path / "processed"
    for d in (raw, master, processed):
        d.mkdir()
    (raw / "prices.parquet").write_text("x")
    (master / "master_point_in_time.parquet").write_text("x")
    monkeypatch.setattr(exp_runner, "RAW_DIR", raw)
    monkeypatch.setattr(exp_runner, "MASTER_DIR", master)
    monkeypatch.setattr(exp_runner, "PROCESSED_DIR", processed)

    calls = {"download": 0, "dataset": 0, "features": 0}
    monkeypatch.setattr("module.ingest.pipeline.download_raw_data", lambda s: calls.__setitem__("download", 1))
    monkeypatch.setattr("module.dataset.build_master_dataset", lambda s: calls.__setitem__("dataset", 1))
    monkeypatch.setattr("module.features.pipeline.build_features", lambda: calls.__setitem__("features", 1))
    exp_runner._ensure_experiment_inputs(Settings())
    assert calls == {"download": 0, "dataset": 0, "features": 1}


def test_ensure_inputs_encadena_todo_si_no_hay_nada(tmp_path, monkeypatch):
    """Sin ningún artefacto, corre download → dataset → features en cadena (como el pipeline)."""
    raw, master, processed = tmp_path / "raw", tmp_path / "master", tmp_path / "processed"
    for d in (raw, master, processed):
        d.mkdir()
    monkeypatch.setattr(exp_runner, "RAW_DIR", raw)
    monkeypatch.setattr(exp_runner, "MASTER_DIR", master)
    monkeypatch.setattr(exp_runner, "PROCESSED_DIR", processed)

    order = []
    monkeypatch.setattr("module.ingest.pipeline.download_raw_data", lambda s: order.append("download"))
    monkeypatch.setattr("module.dataset.build_master_dataset", lambda s: order.append("dataset"))
    monkeypatch.setattr("module.features.pipeline.build_features", lambda: order.append("features"))
    exp_runner._ensure_experiment_inputs(Settings())
    assert order == ["download", "dataset", "features"]
