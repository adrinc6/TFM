from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from environment import RESULTS_DIR
from module.steps.step_03_trainer.pipeline import run_training_pipeline
from module.steps.step_04_evaluation.evaluation_engine import run_evaluation_engine


def _ensure_results_dir() -> Path:
    out = Path(RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _diagnostics_payload(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []

    score_cols = [c for c in df.columns if c.endswith("_score")]
    base_cols = [
        "ticker",
        "date",
        "strategy",
        "tp_level",
        "sl_level",
        "actual_outcome",
        "days_to_event",
        "meta_score",
        "selected_in_portfolio",
    ]
    keep = [c for c in base_cols + score_cols if c in df.columns]
    out = df[keep].copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    return json.loads(out.to_json(orient="records", date_format="iso"))


def run() -> dict[str, pd.DataFrame]:
    training_artifacts = run_training_pipeline()
    evaluation_output = run_evaluation_engine(training_artifacts)

    results_dir = _ensure_results_dir()

    evaluation_output.diagnostics_all_strategies.reset_index().to_parquet(results_dir / "diagnostics_all_strategies.parquet", index=False)
    evaluation_output.diagnostics_per_stock.to_parquet(results_dir / "diagnostics_per_stock.parquet", index=False)
    evaluation_output.model_strategy_metrics.to_csv(results_dir / "model_strategy_metrics.csv", index=False)
    evaluation_output.model_performance.to_csv(results_dir / "model_performance.csv", index=False)
    evaluation_output.strategy_performance.to_csv(results_dir / "strategy_performance.csv", index=False)
    evaluation_output.portfolio.to_csv(results_dir / "portfolio.csv", index=False)

    payload = _diagnostics_payload(evaluation_output.diagnostics_per_stock)
    print(json.dumps(payload, ensure_ascii=False))

    return {
        "diagnostics_all_strategies": evaluation_output.diagnostics_all_strategies,
        "diagnostics_per_stock": evaluation_output.diagnostics_per_stock,
        "model_strategy_metrics": evaluation_output.model_strategy_metrics,
        "model_performance": evaluation_output.model_performance,
        "strategy_performance": evaluation_output.strategy_performance,
        "portfolio": evaluation_output.portfolio,
    }


if __name__ == "__main__":
    run()
