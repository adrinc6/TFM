"""Fixture para tests del HTML: un `run_dir` con los artefactos minimos de Fase 3 y 4."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def minimal_run_dir(tmp_path) -> Path:
    """Crea un `run_dir` con los cinco parquets del backtest + JSONs esperados."""
    run_dir = tmp_path / "agents" / "ridge-abcdef123456"
    run_dir.mkdir(parents=True)

    # Fase 3: agent_scores y meta_weights y diagnostics
    scores = pd.DataFrame([
        {"ticker": "AAA", "snapshot_date": "2000-01-15", "is_quarterly": True,
         "quality": 0.9, "momentum": 0.85, "value": 0.8, "meta_score": 0.85, "meta_rank": 0.95},
        {"ticker": "BBB", "snapshot_date": "2000-01-15", "is_quarterly": True,
         "quality": 0.7, "momentum": 0.6, "value": 0.5, "meta_score": 0.60, "meta_rank": 0.60},
        {"ticker": "CCC", "snapshot_date": "2000-01-15", "is_quarterly": True,
         "quality": 0.5, "momentum": 0.3, "value": 0.4, "meta_score": 0.40, "meta_rank": 0.30},
    ])
    scores.to_parquet(run_dir / "agent_scores.parquet")

    diagnostics = pd.DataFrame([
        {"agent": "quality", "prediction_date": "2000-01-15",
         "label_end_date": "2000-04-15", "observations": 3, "rank_ic": 0.12, "is_quarterly": True},
        {"agent": "momentum", "prediction_date": "2000-01-15",
         "label_end_date": "2000-04-15", "observations": 3, "rank_ic": 0.08, "is_quarterly": True},
        {"agent": "value", "prediction_date": "2000-01-15",
         "label_end_date": "2000-04-15", "observations": 3, "rank_ic": 0.03, "is_quarterly": True},
    ])
    diagnostics.to_parquet(run_dir / "rank_ic_diagnostics.parquet")

    weights = pd.DataFrame([
        {"snapshot_date": "2000-01-15", "agent": "quality", "weight": 0.5,
         "mean_rank_ic": 0.12, "realized_cohorts": 3, "weight_status": "learned"},
        {"snapshot_date": "2000-01-15", "agent": "momentum", "weight": 0.3,
         "mean_rank_ic": 0.08, "realized_cohorts": 3, "weight_status": "learned"},
        {"snapshot_date": "2000-01-15", "agent": "value", "weight": 0.2,
         "mean_rank_ic": 0.03, "realized_cohorts": 3, "weight_status": "learned"},
    ])
    weights.to_parquet(run_dir / "meta_weights.parquet")

    coefficients = pd.DataFrame([
        {"agent": "quality", "model_retrain_date": "2000-01-15",
         "feature": "factor_roe", "coefficient": 0.5, "training_rows": 50},
    ])
    coefficients.to_parquet(run_dir / "model_coefficients.parquet")

    # Fase 4: los cinco parquets del backtest
    positions = pd.DataFrame([
        {"snapshot_date": "2000-01-15", "ticker": "AAA", "weight": 0.4,
         "entry_date": "2000-01-15", "months_held": 0, "current_percentile": 95.0},
        {"snapshot_date": "2000-01-15", "ticker": "BBB", "weight": 0.3,
         "entry_date": "2000-01-15", "months_held": 0, "current_percentile": 60.0},
        {"snapshot_date": "2000-01-15", "ticker": "CCC", "weight": 0.3,
         "entry_date": "2000-01-15", "months_held": 0, "current_percentile": 30.0},
    ])
    positions.to_parquet(run_dir / "positions.parquet")

    orders = pd.DataFrame([
        {"snapshot_date": "2000-01-15", "ticker": "AAA", "side": "buy",
         "reason": "initial_fill", "weight_before": 0.0, "weight_after": 0.4,
         "price": 100.0, "commission": 0.0002, "slippage": 0.0004},
    ])
    orders.to_parquet(run_dir / "orders.parquet")

    equity = pd.DataFrame([
        {"snapshot_date": "2000-01-15", "portfolio_value": 100.0, "benchmark_value": 100.0,
         "portfolio_return": 0.0, "benchmark_return": 0.0, "excess_return": 0.0, "turnover_pct": 1.0},
        {"snapshot_date": "2000-04-15", "portfolio_value": 108.0, "benchmark_value": 105.0,
         "portfolio_return": 0.08, "benchmark_return": 0.05, "excess_return": 0.03, "turnover_pct": 0.1},
        {"snapshot_date": "2001-01-15", "portfolio_value": 115.0, "benchmark_value": 108.0,
         "portfolio_return": 0.065, "benchmark_return": 0.029, "excess_return": 0.036, "turnover_pct": 0.15},
    ])
    equity.to_parquet(run_dir / "equity.parquet")

    annual = pd.DataFrame([
        {"year": 2000, "portfolio_return": 0.08, "benchmark_return": 0.05, "alpha": 0.03,
         "beats_benchmark": True, "max_drawdown_year": 0.05, "information_ratio_year": 0.6},
        {"year": 2001, "portfolio_return": 0.065, "benchmark_return": 0.029, "alpha": 0.036,
         "beats_benchmark": True, "max_drawdown_year": 0.03, "information_ratio_year": 0.4},
    ])
    annual.to_parquet(run_dir / "annual_metrics.parquet")

    summary = {
        # senales de aprendizaje (lo que selecciona)
        "mean_rank_ic": 0.05,
        "rank_ic_positive_fraction": 0.62,
        "rank_ic_std": 0.12,
        # consistencia y riesgo
        "beat_rate": 1.0,
        "max_drawdown": 0.05,
        # alfa: solo reportada, anual, no compuesta
        "annualized_alpha": 0.033,
        "median_alpha": 0.033,
        "worst_year_alpha": 0.03,
        "information_ratio": 0.5,
        # parametros del run
        "commission_bps": 5, "slippage_bps": 10,
        "target_size": 8, "min_hold_percentile": 80,
        "rotation_edge_percentiles": 10,
    }
    (run_dir / "backtest_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    manifest = {"run_scope": "dev", "config": {}, "backtest": summary}
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    return run_dir
